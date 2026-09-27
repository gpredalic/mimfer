/*
 * The engine: one runtime, two execution paths.
 *
 * Owns the config, the two arenas, the tensor registry, the KV + linear/conv
 * state pools, the prefill/decode plans and the round scheduler.
 *
 * - Host build (no MM_WITH_CUDA): the CPU reference path. Arenas are plain
 *   calloc, plans dispatch straight into src/kernels/cpu/cx.c, no device
 *   state exists. The tiny fake model (host_tiny_shape) stands in for
 *   artifact parsing so the whole lifecycle -- load, prefill, decode,
 *   sampling, teardown -- runs end-to-end and is bit-deterministic.
 *
 * - CUDA build (MM_WITH_CUDA): the same lifecycle owns real device memory.
 *   Arenas are device allocations carrying a host mirror (alloc.h): weights
 *   are filled into the mirror and H2D'd once at load, so every plan
 *   operand the builder bakes (weights, activations, pool bases) is a
 *   device pointer. Per-round scalars go out with one H2D of the control
 *   buffer (e->ctrl -> e->ctrl_dev, xfer stream, event-ordered before the
 *   compute stream); freshly allocated KV block-table slices are pushed the
 *   same way; the sampled tokens come back with one D2H (ab.toks_dev ->
 *   e->toks_host) at the round boundary. Plans are captured into CUDA
 *   graphs at load when !no_graph AND sampling is greedy (the non-greedy
 *   SAMPLE launcher stages logits through host memory, which graphs cannot
 *   express; such builds run direct dispatch, see cx.cu).
 *
 * Load order -- the pools MUST exist before the plans are built, because
 * mm_plan_build bakes the pools' per-layer bases into the op table:
 *   1. finalize the model shape (n_full/n_lin layers, per-token KV bytes)
 *      (+ in the CUDA build: probe the device, open the handle set)
 *   2. size the weight arena from weights + pool headroom
 *   3. reserve the activation map for M = chunk (pointers stable for life)
 *   4. create the KV pool + the linear/conv state pool
 *      (+ CUDA: one H2D of the used weight-arena region: filled weights and
 *       the calloc-zeroed pools + block table land on device in one copy)
 *   5. build one prefill plan (M = chunk) + one decode plan per batch 1..n
 *      (+ CUDA: capture the built plans when the policy allows it)
 *   6. reset slot 0's recurrence/conv state and zero the scheduler
 */
#include "mimfer/engine.h"
#include "mimfer/cuda_mem.h"
#include "tensor_registry.h"

#include <stdlib.h>
#include <string.h>

/* Tiny hybrid shape for the host reference -- identical to the plan tests so
 * load + step exercise the exact layout the plan builder was validated on:
 * [lin,lin,lin,full] x2  ->  full layers at li 3 and 7  ->  n_full=2, n_lin=6. */
static void host_tiny_shape(mm_model_cfg *m)
{
    memset(m, 0, sizeof *m);
    m->hidden = 64;
    m->inter  = 128;
    m->layers = 8;
    m->q_heads = 8;
    m->kv_heads = 2;
    m->head_dim = 32;
    m->rope_dim = 16;
    m->lin_k_heads = 4;
    m->lin_k_dim = 16;
    m->lin_v_heads = 8;
    m->lin_v_dim = 16;
    m->conv_k = 4;
    m->vocab = 512;
    m->max_pos = 1024;
    m->rope_theta = 1e7f;
    m->norm_eps = 1e-6f;
    m->full_layer_step = 4;
    m->full_layer_offset = 0;
    m->mtp_layers = 0;
}

/* Upper bound on the bytes the KV + linear/conv state pools consume from the
 * weight arena. Uses the finalized per-token / per-seq bytes plus the 2 MiB
 * (state pool) + 4 KiB (KV/conv) alignment headroom the pool allocator
 * reserves on top of the raw size. */
static size_t host_pool_bytes(const mm_model_cfg *m, uint32_t cap_tokens,
                              uint32_t concurrency)
{
    size_t kv = (size_t)m->kv_bytes_per_token * cap_tokens;
    size_t st = (size_t)m->state_bytes_per_seq * concurrency;
    /* bf16 conv tail per slot (see mm_linstate_create conv_bytes). */
    size_t cv = (size_t)m->n_lin_layers * (m->conv_k - 1) *
                (2u * m->lin_k_heads * m->lin_k_dim +
                 (size_t)m->lin_v_heads * m->lin_v_dim) * 2u * concurrency;
    return kv + st + cv + (size_t)(MM_PAGE_2M + MM_PAGE_4K + (1u * 1024 * 1024));
}

/* Upper bound on the activation arena for a batch of M tokens, mirroring the
 * column counts mm_plan_reserve_acts allocates (see plan.c): the widest layer
 * (full vs linear) wins for the shared qkv buffer, and the S score tile is M
 * columns per row. A 1 MiB margin covers per-alloc alignment slack. */
static size_t host_act_bytes(const mm_model_cfg *m, uint32_t M)
{
    uint32_t qdim = m->q_heads * m->head_dim;
    uint32_t kvdim = m->kv_heads * m->head_dim;
    uint32_t vdim = m->lin_v_heads * m->lin_v_dim;
    uint32_t attn_out = qdim > vdim ? qdim : vdim;
    uint32_t qkv = qdim + 2 * kvdim;
    uint32_t qkv_lin = 2 * m->lin_k_heads * m->lin_k_dim + vdim +
                       2 * m->lin_v_heads;
    if (qkv_lin > qkv)
        qkv = qkv_lin;
    size_t cols = 3 * m->hidden + qkv + qdim + 2 * kvdim + attn_out +
                 3 * m->inter + m->hidden + m->inter + m->vocab + M;
    return (size_t)M * cols * 2 + (size_t)M * 4 + (1u * 1024 * 1024);
}

/* Allocate KV blocks [lo,hi) for a slot so the kernels' reads/writes hit a
 * registered block. Idempotent: an already-allocated block just returns
 * MM_OK, so this is safe to call on every round. */
static mm_status ensure_kv_blocks(mm_engine *e, int slot, uint32_t lo,
                                  uint32_t hi)
{
    uint32_t b, first, last;
    if (!e->kv || lo >= hi)
        return MM_OK;
    first = lo / MM_KV_BLOCK_TOK;
    last  = (hi - 1) / MM_KV_BLOCK_TOK;
    for (b = first; b <= last; b++) {
        uint32_t blk;
        MM_CHECK(mm_kvpool_alloc_block(e->kv, slot, b, &blk));
    }
    return MM_OK;
}

/* Fill the control buffer's sampling fields for this round. Each round
 * consumes exactly one 64-bit draw, mapped to the fixed-point 1.31 uniform
 * the SAMPLE kernel reads (greedy ignores it, but the stream still advances
 * so non-greedy runs stay deterministic). */
static void set_sampling(mm_engine *e)
{
    e->ctrl.temperature = e->cfg.temperature;
    e->ctrl.top_k = (uint32_t)e->cfg.top_k;
    e->ctrl.top_p = e->cfg.top_p;
    e->ctrl.u = mm_rng_fixed32(mm_rng_next(&e->rng));
}

/* Look up (or build into a free cache slot) the prefill plan for M tokens.
 * At most 8 distinct chunk sizes are cached; the decode plans (M = 1..n) are
 * built at load. */
static mm_plan *prefill_plan(mm_engine *e, uint32_t M)
{
    int i;
    for (i = 0; i < 8; i++)
        if (e->pre_cM[i] == M)
            return &e->pre_c[i];
    for (i = 0; i < 8; i++) {
        if (e->pre_cM[i] == 0) {
            memset(&e->pre_c[i], 0, sizeof e->pre_c[i]);
            if (mm_plan_build(e, &e->pre_c[i], PH_PREFILL, M) != MM_OK)
                return NULL;
            e->pre_cM[i] = M;
            return &e->pre_c[i];
        }
    }
    return NULL;   /* all 8 cache slots in use */
}

/* ------------------------------------------------- round boundary I/O */
/*
 * MM_WITH_CUDA only: the host build never compiles these helpers -- its
 * arenas are host memory, so the plan operands the CPU kernels read
 * (e->ab.*, pool bases, block tables) are host pointers already, and
 * there is no device state to move.
 */
#ifdef MM_WITH_CUDA
/* Two fixed events pair the xfer and compute streams across a round:
 * MM_EV_CTRL orders the round's H2D (control buffer + block-table
 * slices) before the first compute-stream work; MM_EV_TOK orders the
 * token D2H after the last. */
#define MM_EV_CTRL 0
#define MM_EV_TOK  1

/* Push this round's control buffer (host mirror e->ctrl -> the pinned
 * device buffer e->ctrl_dev) on the xfer stream and order the compute
 * stream after it. KV block-table slices pushed before this call ride
 * the same stream and get the same ordering for free. */
static mm_status push_ctrl(mm_engine *e)
{
    MM_CHECK(mm_h2d_async(e->ctrl_dev, &e->ctrl, sizeof e->ctrl,
                          MM_ST_XFER));
    MM_CHECK(mm_event_record(MM_EV_CTRL, MM_ST_XFER));
    MM_CHECK(mm_event_wait(MM_EV_CTRL, MM_ST_COMPUTE));
    return MM_OK;
}

/* One host sync per round (the cuda_rt.h contract: the only host syncs
 * are the round boundaries, where the tokens are consumed): order the
 * token D2H (ab.toks_dev -> pinned e->toks_host, xfer stream) after the
 * compute work that wrote the tokens, then wait for the copy. Callers
 * read e->toks_host afterwards. */
static mm_status fetch_tokens(mm_engine *e, uint32_t n)
{
    MM_CHECK(mm_event_record(MM_EV_TOK, MM_ST_COMPUTE));
    MM_CHECK(mm_event_wait(MM_EV_TOK, MM_ST_XFER));
    MM_CHECK(mm_d2h_async(e->toks_host, e->ab.toks_dev, (size_t)n * 4,
                          MM_ST_XFER));
    MM_CHECK(mm_stream_sync(MM_ST_XFER));
    return MM_OK;
}
#endif /* MM_WITH_CUDA */

/* One prefill round: the waiting queue head (a free slot) is assigned, its
 * prompt (chunked to exactly plen tokens) is embedded + forwarded, and the
 * continuation token is sampled at the last position. KVSTORE stores [0,plen)
 * before ATT_PREFILL reads it (store-before-read), so the freshly-zeroed pool
 * is never sampled. */
static mm_status step_prefill(mm_engine *e, int slot)
{
    mm_seq *seq;
    mm_plan *plan;
    uint32_t plen;
    mm_status s;

    MM_CHECK(mm_sched_assign(&e->sched, slot));   /* queue head -> this slot */
    seq = &e->sched.slots[slot];
    plen = seq->plen;
    MM_REQUIRE(plen > 0 && plen <= e->cfg.chunk, MM_ERR_STATE);

    plan = prefill_plan(e, plen);
    if (!plan)
        return MM_ERR_NOMEM;

    memset(&e->ctrl, 0, sizeof e->ctrl);
    memcpy(e->ctrl.toks, seq->ptoks, plen * sizeof(uint32_t));
    e->ctrl.pos0 = 0;
    e->ctrl.kv_len = plen;
    e->ctrl.n_slots = 0;                 /* prefill: tokens come from toks[] */
    set_sampling(e);
    mm_rope_fill_ctrl(&e->rope, &e->ctrl);

    MM_CHECK(ensure_kv_blocks(e, slot, 0, plen));
#ifdef MM_WITH_CUDA
    /* Round boundary (device path): the allocator wrote the new block ids
     * into the host mirror only -- push the block-table slice this round
     * allocated ([0 .. last], the slot's first prefill), then the control
     * buffer; both ride the xfer stream and the compute stream is
     * event-ordered after them by push_ctrl. */
    MM_CHECK(mm_kvpool_push_tab(e->kv, slot, 0,
                                (plen - 1) / MM_KV_BLOCK_TOK + 1));
    MM_CHECK(push_ctrl(e));
#endif
    s = mm_plan_launch(e, plan);
    if (s != MM_OK)
        return s;

#ifdef MM_WITH_CUDA
    MM_CHECK(fetch_tokens(e, plen));
    e->last_tok[slot] = e->toks_host[plen - 1];
#else
    e->last_tok[slot] = ((const uint32_t *)e->ab.toks_dev)[plen - 1];
#endif
    seq->len = plen;
    seq->prefilled = plen;
    seq->gen = 0;
    seq->state = RS_DECODE;
    e->n_tokens_out += 1;
    return MM_OK;
}

/* One decode round: every active slot advances one position. Slot s (in
 * ascending slot order) maps to decode row s; each row stores its token at
 * slot_pos, attends over [0, slot_len), and samples the next token. A slot
 * that hits its budget (max_out) or the context cap is finished and freed. */
static mm_status step_decode(mm_engine *e, uint32_t n_dec)
{
    const uint32_t *out;
    uint32_t m = 0;
    int slot;
    mm_status s;

    MM_REQUIRE(n_dec >= 1 && n_dec <= e->cfg.concurrency, MM_ERR_STATE);

    memset(&e->ctrl, 0, sizeof e->ctrl);
    e->ctrl.n_slots = n_dec;             /* decode: tokens come from slot_*[] */
    set_sampling(e);
    mm_rope_fill_ctrl(&e->rope, &e->ctrl);

    for (slot = 0; slot < (int)e->sched.n; slot++) {
        mm_seq *seq = &e->sched.slots[slot];
        uint32_t pos;
        if (seq->state != RS_DECODE)
            continue;
        pos = seq->len;                   /* position of the new token */
        e->ctrl.slot_tok[m] = e->last_tok[slot];
        e->ctrl.slot_pos[m] = pos;
        e->ctrl.slot_len[m] = pos + 1;    /* KV length incl. the stored tok */
        MM_CHECK(ensure_kv_blocks(e, slot, pos, pos + 1));
#ifdef MM_WITH_CUDA
        /* The block the new token lands in, host mirror -> device (xfer
         * stream; event-ordered by push_ctrl below). A no-op re-push when
         * the block was already allocated (idempotent, same bytes). */
        MM_CHECK(mm_kvpool_push_tab(e->kv, slot,
                                    pos / MM_KV_BLOCK_TOK,
                                    pos / MM_KV_BLOCK_TOK + 1));
#endif
        m++;
    }
    MM_REQUIRE(m == n_dec, MM_ERR_STATE);
#ifdef MM_WITH_CUDA
    MM_CHECK(push_ctrl(e));
#endif
    s = mm_plan_launch(e, &e->dec[n_dec]);
    if (s != MM_OK)
        return s;

#ifdef MM_WITH_CUDA
    MM_CHECK(fetch_tokens(e, n_dec));
    out = e->toks_host;
#else
    out = (const uint32_t *)e->ab.toks_dev;
#endif
    m = 0;
    for (slot = 0; slot < (int)e->sched.n; slot++) {
        mm_seq *seq = &e->sched.slots[slot];
        if (seq->state != RS_DECODE)
            continue;
        e->last_tok[slot] = out[m];
        seq->gen++;
        seq->len++;
        e->n_tokens_out++;
        if (seq->gen >= seq->max_out || seq->len >= e->cfg.max_ctx) {
            seq->finished = 1;
            MM_CHECK(mm_sched_done(&e->sched, slot));
        }
        m++;
    }
    return MM_OK;
}

/* ------------------------------------------------------------ lifecycle */

mm_status mm_engine_create(const mm_engine_cfg *cfg, mm_engine **out)
{
    mm_engine *e;
    size_t w, cap;
    mm_status s;
    /* Names the stage that reaches the 'fail:' label so the cleanup log
     * can say what failed (see the fail: label below). */
    const char *stage = "entry";

    /* Every create stage is traced at WARN and every failure path prints
     * the stage, the numeric code, mm_status_str() and the relevant
     * parameters before it returns. */

    if (!(cfg && out)) {
        MM_LOGE("engine_create: cfg or out is NULL: code=%d (%s)",
                (int)MM_ERR_STATE, mm_status_str(MM_ERR_STATE));
        return MM_ERR_STATE;
    }
    *out = NULL;

    stage = "cfg_validate";
    s = mm_engine_cfg_validate(cfg);
    if (s != MM_OK) {
        MM_LOGE("engine_create: mm_engine_cfg_validate failed: "
                "code=%d (%s)", (int)s, mm_status_str(s));
        return s;
    }
    MM_LOGW("engine_create: cfg_validate ok (max_ctx=%u kv_capacity=%u "
            "concurrency=%u chunk=%u rope_yarn=%d spec=%d profile_id=%u "
            "no_graph=%d)",
            cfg->max_ctx, cfg->kv_capacity, cfg->concurrency, cfg->chunk,
            cfg->rope_yarn, cfg->spec, cfg->profile_id, cfg->no_graph);

    e = calloc(1, sizeof *e);
    if (!e) {
        MM_LOGE("engine_create: calloc(%zu) failed: code=%d (%s)",
                sizeof *e, (int)MM_ERR_NOMEM, mm_status_str(MM_ERR_NOMEM));
        return MM_ERR_NOMEM;
    }
    e->cfg = *cfg;

#ifdef MM_WITH_CUDA
    /* Device gate (appliance contract): a visible device, the process-wide
     * handle set, and the probed device info. The target-profile check is
     * a hard gate in production builds; MIMFER_SOFT_DEVICE_GATE downgrades
     * it to a warning so the CUDA execution path can be exercised on
     * non-target silicon (test rigs). */
    {
        int ndev = 0;

        stage = "cuda_device_count";
        s = mm_cuda_device_count(&ndev);
        if (s != MM_OK) {
            MM_LOGE("engine_create: mm_cuda_device_count failed: "
                    "code=%d (%s)", (int)s, mm_status_str(s));
            free(e);
            return s;
        }
        MM_LOGW("engine_create: mm_cuda_device_count ok (ndev=%d)", ndev);

        if (ndev <= 0) {
            MM_LOGE("engine_create: no CUDA devices visible (ndev=%d): "
                    "code=%d (%s)", ndev, (int)MM_ERR_DEVICE,
                    mm_status_str(MM_ERR_DEVICE));
            free(e);
            return MM_ERR_DEVICE;
        }

        stage = "cuda_init";
        s = mm_cuda_init(0);
        if (s != MM_OK) {
            MM_LOGE("engine_create: mm_cuda_init(0) failed: code=%d (%s)",
                    (int)s, mm_status_str(s));
            free(e);
            return s;
        }
        MM_LOGW("engine_create: mm_cuda_init(0) ok");

        stage = "device_probe";
        s = mm_device_probe(0, &e->dev);
        if (s != MM_OK) {
            MM_LOGE("engine_create: mm_device_probe(0) failed: "
                    "code=%d (%s)", (int)s, mm_status_str(s));
            free(e);
            return s;
        }
        MM_LOGW("engine_create: mm_device_probe ok: %s (sm_%d%d, %d SMs, "
                "%zu GiB, %zu GB/s derived, l2 %zu MiB, smem/SM %zu KiB)",
                e->dev.name, e->dev.cc_major, e->dev.cc_minor,
                e->dev.sm_count, e->dev.vram_bytes >> 30, e->dev.bw_gbps,
                e->dev.l2_bytes >> 20, e->dev.smem_per_sm >> 10);

        stage = "device_check";
        {
            mm_status ds = mm_device_check(&e->dev, &MM_TARGET_PRO4000);
            if (ds != MM_OK) {
                MM_LOGE("engine_create: mm_device_check failed: "
                        "code=%d (%s)", (int)ds, mm_status_str(ds));
                if (!getenv("MIMFER_SOFT_DEVICE_GATE")) {
                    mm_cuda_shutdown();
                    free(e);
                    return ds;
                }
                MM_LOGW("engine_create: device profile gate failed "
                        "(code=%d, %s); continuing "
                        "(MIMFER_SOFT_DEVICE_GATE set)",
                        (int)ds, mm_status_str(ds));
            } else {
                MM_LOGW("engine_create: mm_device_check ok "
                        "(target profile matched)");
            }
        }
    }
    e->on_host = 0;
#else
    e->on_host = 1;
#endif

    /* No artifact in this build: stand in the fixed tiny hybrid shape in
     * place of parsing the model section. */
    host_tiny_shape(&e->mc);

    /* kv_capacity 0 = auto: one full sequence, 64-token aligned. */
    if (e->cfg.kv_capacity == 0)
        e->cfg.kv_capacity = ((e->cfg.max_ctx + 63) / 64) * 64;
    if (e->cfg.kv_capacity < e->cfg.max_ctx)
        e->cfg.kv_capacity = e->cfg.max_ctx;
    if (e->cfg.concurrency > MM_MAX_CONCURRENCY)
        e->cfg.concurrency = MM_MAX_CONCURRENCY;

    stage = "model_cfg_finalize";
    s = mm_model_cfg_finalize(&e->mc);
    if (s != MM_OK) {
        MM_LOGE("engine_create: mm_model_cfg_finalize failed: code=%d (%s) "
                "(max_ctx=%u kv_capacity=%u concurrency=%u)",
                (int)s, mm_status_str(s), e->cfg.max_ctx, e->cfg.kv_capacity,
                e->cfg.concurrency);
#ifdef MM_WITH_CUDA
        mm_cuda_shutdown();
#endif
        free(e);
        return s;
    }
    MM_LOGW("engine_create: mm_model_cfg_finalize ok (layers=%u hidden=%u "
            "kv_capacity=%u)",
            e->mc.layers, e->mc.hidden, e->cfg.kv_capacity);

    /* RoPE effective table (plain RoPE, or the YaRN blend when the
     * config enables it): built once here, then re-copied into the
     * control buffer after every round's reset (step functions). */
    stage = "rope_init";
    s = mm_rope_init(&e->rope, &e->mc, &e->cfg);
    if (s != MM_OK) {
        MM_LOGE("engine_create: mm_rope_init failed: code=%d (%s) "
                "(rope_yarn=%d rope_factor=%f rope_orig_ctx=%u)",
                (int)s, mm_status_str(s), e->cfg.rope_yarn,
                (double)e->cfg.rope_factor, e->cfg.rope_orig_ctx);
#ifdef MM_WITH_CUDA
        mm_cuda_shutdown();
#endif
        free(e);
        return s;
    }
    {
        char rdesc[96];
        mm_rope_desc(&e->rope, &e->mc, rdesc, sizeof rdesc);
        MM_LOGW("engine_create: mm_rope_init ok -- rope: %s", rdesc);
    }

    /* Weights profile: validate the request against the checkpoint's
     * native ceiling (the artifact's RoPE context, config.c) and pin
     * the active checkpoint identity for the load log. */
    stage = "profile_get";
    if (e->cfg.profile_id != 0) {
        const mm_profile *p = mm_profile_get(e->cfg.profile_id);
        if (!p) {
            MM_LOGE("engine_create: mm_profile_get(%u) returned NULL: "
                    "code=%d (%s)", e->cfg.profile_id, (int)MM_ERR_RANGE,
                    mm_status_str(MM_ERR_RANGE));
#ifdef MM_WITH_CUDA
            mm_cuda_shutdown();
#endif
            free(e);
            return MM_ERR_RANGE;
        }
        if (e->cfg.max_ctx > p->max_ctx) {
            MM_LOGE("engine_create: profile %s: max_ctx %u exceeds the "
                    "checkpoint ceiling %u (extend with YaRN, not by "
                    "raising --max-ctx past it); code=%d (%s)",
                    p->name, e->cfg.max_ctx, p->max_ctx, (int)MM_ERR_RANGE,
                    mm_status_str(MM_ERR_RANGE));
#ifdef MM_WITH_CUDA
            mm_cuda_shutdown();
#endif
            free(e);
            return MM_ERR_RANGE;
        }
        e->profile = p;
        MM_LOGI("engine: profile: %s (%s; %s; mtp_layers=%u "
                "dflash2_max=%u vision=%d)",
                p->name, p->repo, p->quant, p->mtp_layers,
                p->dflash2_max, p->has_vision);
    }

    /* Speculative decoding: the flag surface is real (parsed, range-
     * checked, cross-validated against --draft-tokens), but the
     * draft/verify execution loop is planned work (docs/dflash2.md).
     * Refuse to start with a speculative backend instead of silently
     * running non-speculative decode. */
    if (e->cfg.spec != 0) {
        const char *backend = e->cfg.spec == 1 ? "mtp" : "dflash2";
        MM_LOGE("engine_create: --spec %s is not implemented in this "
                "build (draft/verify loop: planned work, docs/dflash2.md); "
                "start with --spec off; code=%d (%s)",
                backend, (int)MM_ERR_UNSUPPORTED,
                mm_status_str(MM_ERR_UNSUPPORTED));
#ifdef MM_WITH_CUDA
        mm_cuda_shutdown();
#endif
        free(e);
        return MM_ERR_UNSUPPORTED;
    }

    if (e->cfg.vision)
        MM_LOGW("engine: vision hook enabled -- image submission "
                "returns MM_ERR_UNSUPPORTED until the pipeline lands "
                "(planned work)");

    /* Weight arena: the fake weights + the pools (bump-allocated from it).
     * CUDA build: device arena -- canonical device pointers for the plans,
     * plus the host mirror the fake fill writes through (alloc.h). */
    w = mm_model_weights_bytes(&e->mc);
    cap = w + host_pool_bytes(&e->mc, e->cfg.kv_capacity, e->cfg.concurrency);
    MM_LOGW("engine_create: arena sizing (weights=%zu, w_arena cap=%zu, "
            "a_arena cap=%zu)",
            w, cap, host_act_bytes(&e->mc, e->cfg.chunk));
#ifdef MM_WITH_CUDA
    stage = "w_arena (device)";
    s = mm_arena_init_device(&e->w_arena, cap, "w_arena");
    if (s != MM_OK)
        goto fail;
    MM_LOGW("engine_create: mm_arena_init_device w_arena ok "
            "(cap=%zu base=%p hbase=%p)",
            cap, e->w_arena.base, e->w_arena.hbase);
    /* Activation arena: sized for M = chunk, the largest batch we reserve. */
    stage = "a_arena (device)";
    s = mm_arena_init_device(&e->a_arena,
                             host_act_bytes(&e->mc, e->cfg.chunk), "a_arena");
    if (s != MM_OK)
        goto fail;
    MM_LOGW("engine_create: mm_arena_init_device a_arena ok "
            "(cap=%zu base=%p)", e->a_arena.cap, e->a_arena.base);
    /* Pinned host I/O block: device control buffer + token readback mirror
     * (layout in cuda_mem.h). */
    stage = "pin_alloc";
    s = mm_pin_alloc(MM_PIN_BYTES, &e->pin);
    if (s != MM_OK)
        goto fail;
    e->ctrl_dev = (mm_ctrl *)e->pin;
    e->toks_host = (uint32_t *)((char *)e->pin +
                                mm_align_up(sizeof(mm_ctrl), 16));
    MM_LOGW("engine_create: mm_pin_alloc ok (%zu bytes, ptr=%p)",
            (size_t)MM_PIN_BYTES, e->pin);
#else
    stage = "w_arena (host)";
    s = mm_arena_init_host(&e->w_arena, cap, "w_arena");
    if (s != MM_OK)
        goto fail;
    MM_LOGW("engine_create: mm_arena_init_host w_arena ok (cap=%zu base=%p)",
            cap, e->w_arena.base);
    /* Activation arena: sized for M = chunk, the largest batch we reserve. */
    stage = "a_arena (host)";
    s = mm_arena_init_host(&e->a_arena, host_act_bytes(&e->mc, e->cfg.chunk),
                           "a_arena");
    if (s != MM_OK)
        goto fail;
    MM_LOGW("engine_create: mm_arena_init_host a_arena ok (cap=%zu base=%p)",
            e->a_arena.cap, e->a_arena.base);
#endif

    mm_rng_seed(&e->rng, e->cfg.seed ? e->cfg.seed : 1);
    *out = e;
    MM_LOGW("engine_create: ok");
    return MM_OK;

fail:
    MM_LOGE("engine_create: stage '%s' failed: code=%d (%s); cleaning up",
            stage, (int)s, mm_status_str(s));
    mm_arena_free(&e->a_arena);
    mm_arena_free(&e->w_arena);
#ifdef MM_WITH_CUDA
    if (e->pin)
        mm_pin_free(e->pin);
    mm_cuda_shutdown();
#endif
    free(e);
    return s;
}

mm_status mm_engine_load(mm_engine *e)
{
    uint32_t m;
    mm_status s;

    MM_REQUIRE(e && !e->loaded && e->w_arena.base, MM_ERR_STATE);

    /* Weights: register every tensor + fill the arena deterministically. */
    s = mm_model_fill_fake(e, e->cfg.seed ? e->cfg.seed : 1);
    if (s != MM_OK)
        return s;

    /* Activation map first (the plans reference these pointers). */
    s = mm_plan_reserve_acts(e, e->cfg.chunk);
    if (s != MM_OK)
        return s;

    /* Pools -- MUST precede the plans: mm_plan_build bakes their bases. */
    s = mm_kvpool_create(&e->mc, e->cfg.kv_capacity, MM_KV_BF16,
                         e->cfg.max_ctx, e->cfg.concurrency, &e->w_arena,
                         &e->kv);
    if (s != MM_OK)
        return s;
    s = mm_linstate_create(&e->mc, e->cfg.concurrency, &e->w_arena, &e->st);
    if (s != MM_OK)
        return s;

#ifdef MM_WITH_CUDA
    /* Stage everything allocated so far onto the device in one H2D: the
     * host mirror holds the filled weights plus the pools allocated after
     * the fill (KV, linear/conv state, block tables), which carry the
     * calloc-zeroed bytes the kernels expect at start (a zero block table
     * is the "no block" state). */
    {
        size_t used = mm_arena_used(&e->w_arena);
        MM_CHECK(mm_h2d_async(e->w_arena.base,
                              mm_arena_at_host(&e->w_arena, 0), used,
                              MM_ST_XFER));
        MM_CHECK(mm_stream_sync(MM_ST_XFER));
    }
#endif

    /* Plans: one prefill (M = chunk) + one decode per batch size 1..n. */
    memset(&e->pre_c[0], 0, sizeof e->pre_c[0]);
    s = mm_plan_build(e, &e->pre_c[0], PH_PREFILL, e->cfg.chunk);
    if (s != MM_OK)
        return s;
    e->pre_cM[0] = e->cfg.chunk;

    for (m = 1; m <= e->cfg.concurrency; m++) {
        memset(&e->dec[m], 0, sizeof e->dec[m]);
        s = mm_plan_build(e, &e->dec[m], PH_DECODE, m);
        if (s != MM_OK)
            return s;
    }

    /* Deterministic, zeroed recurrence/conv state for the first slot. */
    s = mm_linstate_reset(e->st, 0);
    if (s != MM_OK)
        return s;

#ifdef MM_WITH_CUDA
    /* Graph capture (the design's fast path: running a plan is then one
     * cudaGraphLaunch). Greedy sampling only: the non-greedy SAMPLE
     * launcher stages logits through host memory, which a graph cannot
     * express (see the launcher note in src/kernels/cuda/cx.cu). */
    if (!e->cfg.no_graph &&
        (e->cfg.temperature <= 0.0f || e->cfg.top_k == 0))
        MM_CHECK(mm_plan_capture_all(e));
#endif

    mm_sched_init(&e->sched, e->cfg.concurrency);
    e->loaded = 1;
    return MM_OK;
}

mm_status mm_engine_step(mm_engine *e)
{
    int slot = -1;
    uint32_t n_dec = 0;
    int rtype;

    MM_REQUIRE(e && e->loaded, MM_ERR_STATE);
    rtype = mm_sched_pick_round(&e->sched, &slot, &n_dec);
    if (rtype == 0)
        return MM_OK;                    /* idle: nothing to do this round */
    if (rtype == 1)
        return step_prefill(e, slot);
    return step_decode(e, n_dec);
}

void mm_engine_destroy(mm_engine *e)
{
    int m;
    if (!e)
        return;
    if (e->kv) {
        mm_kvpool_destroy(e->kv);
        e->kv = NULL;
    }
    if (e->st) {
        mm_linstate_destroy(e->st);
        e->st = NULL;
    }
    for (m = 1; m <= (int)e->cfg.concurrency; m++)
        mm_plan_destroy(&e->dec[m]);
    for (m = 0; m < 8; m++)
        if (e->pre_cM[m])
            mm_plan_destroy(&e->pre_c[m]);
    mm_arena_free(&e->a_arena);
    mm_arena_free(&e->w_arena);
#ifdef MM_WITH_CUDA
    /* The pinned host I/O block (control buffer + token readback) and the
     * process-wide handle set: both were opened in mm_engine_create. */
    if (e->pin)
        mm_pin_free(e->pin);
    mm_cuda_shutdown();
#endif
    free(e);
}

/* Accessors for callers that reach the buffers through the engine handle
 * (the CPU kernels use e->ab / e->ctrl directly; the GPU build and external
 * tooling go through these). */
const mm_abufs *mm_engine_ab(const mm_engine *e)
{
    return e ? &e->ab : NULL;
}

const mm_ctrl *mm_engine_ctrl_dev(const mm_engine *e)
{
    return e ? &e->ctrl : NULL;
}

/* Vision hook (planned pipeline): real flag surface, explicit error
 * until the vision tower lands. The request is validated (non-empty
 * payload, loaded engine, hook enabled) before the unsupported return,
 * so callers see the same failure for a disabled hook as for a bad
 * request -- and neither can silently succeed. */
mm_status mm_engine_submit_image(mm_engine *e, const void *px, size_t n)
{
    MM_REQUIRE(e && px && n > 0, MM_ERR_STATE);
    MM_REQUIRE(e->loaded, MM_ERR_STATE);
    if (!e->cfg.vision) {
        MM_LOGE("engine: vision hook disabled (start with --vision)");
        return MM_ERR_STATE;
    }
    MM_LOGW("engine: vision image submission (n=%zu) is planned work; "
            "the pipeline is not implemented in this build", n);
    return MM_ERR_UNSUPPORTED;
}

