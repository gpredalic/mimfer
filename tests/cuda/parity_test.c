/*
 * CPU<->CUDA parity — CUDA-side comparator (nvcc build, requires a GPU).
 *
 * Reads the golden file (magic MMPAR001) written by tests/host/par_golden.c
 * and replays the exact same op-by-op walk on the CUDA build: the same tiny
 * hybrid shape, the same control-buffer fills (mirroring engine.c's step
 * functions byte for byte), the same KV block-table pushes, and one
 * mm_kx_invoke() per plan op on the compute stream. After every op — and
 * after each round's pool/state regions — every observable is read back to
 * the host and compared against the golden under the shared tolerance
 * classes declared in the golden writer's header:
 *
 *   class 0  bit-exact      EMBED / COPY / SAMPLE outputs, the KV pool
 *   class 1  1 bf16 ulp     every computed bf16 output
 *   class 2  4 f32 ulp      the f32 recurrence state; 1 bf16 ulp where the
 *                           stored state is bf16 (the conv tail)
 *
 * A pass means the GPU kernels reproduce the CPU reference (the
 * correctness oracle) within those classes across a full prefill plus 16
 * decode rounds. Graph capture/launch and the engine's round stepping are
 * covered by the GPU smoke test (tests/cuda/gpu_smoke.c); this test is the
 * numeric gate.
 *
 * Build (GPU machine; the .c sources compile as C, cx.cu as CUDA):
 *   nvcc -O2 -fmad=false -DMM_WITH_CUDA \
 *        -Iinclude -Isrc/model -Isrc/kernels \
 *        -x c tests/cuda/parity_test.c \
 *        -x c src/engine/engine.c -x c src/plan/plan.c \
 *        -x c src/alloc/alloc.c -x c src/config/config.c \
 *        -x c src/core/mimfer.c -x c src/cuda/cuda_rt.c \
 *        -x c src/cuda/cuda_mem.c -x c src/sampling/sampling.c \
 *        -x c src/kv/kv.c -x c src/kernels/kx.c \
 *        src/kernels/cuda/cx.cu \
 *        -x c src/model/tensor_registry.c -x c src/sched/sched.c \
 *        -o /tmp/parity_test
 * Run (golden file produced by the host build of tests/host/par_golden.c):
 *   /tmp/parity_test /tmp/par_golden.bin
 */
#include "mimfer/engine.h"
#include "mimfer/cuda_rt.h"
#include "kx.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Host self-check (no GPU): the same file builds against the CPU
 * reference, where d2h() is a memcpy and the device I/O helpers are
 * no-ops (the host kernels read the host control mirror directly). Run
 * it against a fresh golden file to verify the reader, the observable
 * sizing and the walk end to end:
 *   gcc -std=c11 -Wall -Wextra -Werror -Iinclude -Isrc/model -Isrc/kernels \
 *       tests/cuda/parity_test.c src/engine/engine.c src/plan/plan.c \
 *       src/alloc/alloc.c src/config/config.c src/core/mimfer.c \
 *       src/cuda/cuda_rt.c src/cuda/cuda_mem.c src/sampling/sampling.c \
 *       src/kv/kv.c src/kernels/cpu/cx.c src/kernels/kx.c \
 *       src/model/tensor_registry.c src/sched/sched.c -lm \
 *       -o /tmp/par_selfcheck
 *   /tmp/par_selfcheck /tmp/par_golden.bin
 */

/* Same constants as the golden writer (tests/host/par_golden.c): the two
 * walks must consume the same shape, prompt and seed. */
#define PAR_MAGIC  "MMPAR001"
#define PAR_SEED   12345u
#define PAR_MAXCTX 128u
#define PAR_MAXOUT 16u
#define PAR_CHUNK  8u
static const uint32_t PAR_PROMPT[4] = { 5, 9, 17, 42 };
#define PAR_PLEN ((uint32_t)(sizeof PAR_PROMPT / sizeof PAR_PROMPT[0]))

/* Parity-test events. The engine's own round-boundary events are 0/1 (its
 * step functions are not used here — the walk drives mm_kx_invoke directly),
 * so the remaining pool slots are free for the test's own ordering. */
#define PV_EV_PUSH 2   /* ctrl H2D done  -> compute may start the round   */
#define PV_EV_READ 3   /* compute done   -> xfer may D2H the observable   */

static int g_fail = 0;
#define CHECK(cond, msg)                                                     \
    do {                                                                     \
        if (!(cond)) {                                                       \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg);    \
            g_fail = 1;                                                      \
        }                                                                    \
    } while (0)

/* Report counters (printed at the end). */
static uint64_t g_ops, g_obs, g_elems, g_exact;
static long long g_max_ulp[3];        /* f32-ulp distance, per class       */

#define BF16_ULP_F32 65536LL          /* 1 bf16 ulp == 2^16 f32 ulp        */
#define F32_TOL      4LL

/* One observable: which operand (0..3) or pool region (4..6) + size.
 * Identical to the golden writer's op_obs / pool_obs, so the two walks
 * agree on what "the observable of this op" means. */
typedef struct obs {
    int    tag;
    size_t bytes;
} obs;

#define MAX_OBS 4

/* --------------------------------------------------- golden file reader */

static FILE *g_f;

static int fr_u32(uint32_t *v)
{
    return fread(v, 4, 1, g_f) == 1;
}

static int fr_bytes(void *p, size_t n)
{
    return n == 0 || fread(p, 1, n, g_f) == n;
}

/* ------------------------------------------------------- tolerance math */

/* Little-endian bf16 -> f32 (bf16 is the high half of an f32). */
static float b16_rd(const uint8_t *p)
{
    uint32_t u = ((uint32_t)p[0]) | ((uint32_t)p[1] << 8);

    u <<= 16;
    return *(float *)&u;
}

/* Signed-magnitude -> integer where |map(a)-map(b)| equals the f32 ulp
 * distance for same-sign values. (One-ulp slip at the zero crossing; the
 * tolerances used here are >= 4 f32 ulp and same-sign values that far
 * apart never cross zero.) */
static long long f32_map(float f)
{
    uint32_t b = *(uint32_t *)&f;

    return (long long)(b & 0x80000000u ? ~b : b);
}

/* 1 if a and b agree within `tol` f32 ulps (0 = bitwise equal). */
static int ulp_ok(float a, float b, long long tol)
{
    long long d;

    if (a != a || b != b)                  /* NaN: never a parity pass    */
        return 0;
    if (a == b || a == 0.0f || b == 0.0f)  /* incl. +/-0: equal           */
        return 1;
    if ((a < 0.0f) != (b < 0.0f))          /* sign flip: beyond any tol   */
        return 0;
    d = f32_map(a) - f32_map(b);
    return d <= tol && d >= -tol;
}

/* Tolerance class per (op, tag) — the rules the golden writer's header
 * documents. Tags 4..6 are the per-round pool/state observables (op type
 * is irrelevant for them). */
static int obs_class(uint32_t op, int tag)
{
    if (tag == 4)                          /* KV pool: bit-exact          */
        return 0;
    if (tag == 5 || tag == 6)              /* lin f32 / conv bf16 state   */
        return 2;
    switch (op) {
    case OP_EMBED: case OP_COPY: case OP_SAMPLE:
        return 0;
    case OP_LIN_PRE: case OP_LIN_DEC:
        return tag == 0 ? 2 : 1;           /* f32 state / bf16 out        */
    case OP_CONV4:
        return tag == 3 ? 2 : 1;           /* bf16 conv tail / bf16 out   */
    default:
        return 1;                          /* computed bf16               */
    }
}

/* Element size (bytes) for the ulp comparisons; class 0 memcmps whole. */
static int elem_bytes(uint32_t op, int tag)
{
    if (tag == 5)
        return 4;                          /* lin state: f32              */
    if (tag == 6)
        return 2;                          /* conv tail: bf16             */
    if ((op == OP_LIN_PRE || op == OP_LIN_DEC) && tag == 0)
        return 4;
    return 2;                              /* bf16                        */
}

/* -------------------------------------------------- observable sizing */
/* Sizing mirror of the golden writer's op_obs: the observable output of
 * one op, from the finalized model shape (the same numbers the plan
 * builder bakes). KVSTORE has none: its effect lives in the pool, which
 * is dumped after the round. */
static int op_obs(const mm_engine *e, const mm_op_desc *op, obs *out)
{
    const mm_model_cfg *mc = &e->mc;
    uint32_t M = op->n0;
    uint32_t qdim = mc->q_heads * mc->head_dim;
    uint32_t kvdim = mc->kv_heads * mc->head_dim;
    uint32_t vdim = mc->lin_v_heads * mc->lin_v_dim;
    uint32_t attn_out = qdim > vdim ? qdim : vdim;
    uint32_t ch = 2u * mc->lin_k_heads * mc->lin_k_dim + vdim;
    int n = 0;

#define O(t, b) do { out[n].tag = (t); out[n].bytes = (size_t)(b); n++; } \
                while (0)
    switch (op->op) {
    case OP_EMBED:
        O(2, M * op->n1 * 2u); break;
    case OP_RMSNORM:
        O(0, M * op->n1 * 2u); O(2, M * op->n1 * 2u); break;
    case OP_ADD_RMSNORM:
        O(0, M * op->n1 * 2u); O(2, M * op->n1 * 2u); break;
    case OP_GEMM_F4:
    case OP_GEMM_BF16:
    case OP_QKV:
    case OP_LMHEAD:
        O(2, M * op->n2 * 2u); break;
    case OP_SILU_MUL:
        O(2, M * op->n1 * 2u); break;
    case OP_ROPE:
        O(0, M * qdim * 2u); O(1, M * kvdim * 2u); break;
    case OP_ATT_PREFILL:
    case OP_ATT_DECODE:
        O(2, M * attn_out * 2u); break;
    case OP_CONV4:
        O(2, M * op->n1 * 2u);
        O(3, (mc->conv_k - 1u) * ch * 2u); break;
    case OP_LIN_PRE:
    case OP_LIN_DEC:
        O(0, mc->lin_v_heads * mc->lin_k_dim * mc->lin_v_dim * 4u);
        O(2, M * op->n1 * mc->lin_v_dim * 2u); break;
    case OP_COPY:
        O(1, op->n0); break;
    case OP_SAMPLE:
        O(1, M * 4u); break;
    case OP_KVSTORE:
        n = 0; break;
    default:
        return -1;
    }
#undef O
    return n;
}

static const void *obs_ptr(const mm_op_desc *op, int tag)
{
    switch (tag) {
    case 0:  return op->a;
    case 1:  return op->b;
    case 2:  return op->c;
    default: return op->d;
    }
}

/* Per-round pool/state observables (tags 4..6): the slot-0 regions the
 * single-slot kernels read/write, sized exactly as the golden writer. */
static void pool_obs(const mm_engine *e, obs *out)
{
    const mm_model_cfg *mc = &e->mc;
    size_t kv_layer = (size_t)mc->kv_heads * e->kv->n_blocks
                      * MM_KV_BLOCK_TOK * mc->head_dim * 2u;
    size_t cv_layer = e->st->conv_bytes
                      / (size_t)(mc->n_lin_layers ? mc->n_lin_layers : 1);

    out[0].tag = 4;
    out[0].bytes = (size_t)mc->n_full_layers * 2u * kv_layer;
    out[1].tag = 5;
    out[1].bytes = (size_t)mc->n_lin_layers * e->st->layer_bytes;
    out[2].tag = 6;
    out[2].bytes = (size_t)mc->n_lin_layers * cv_layer;
}

static const void *pool_ptr(const mm_engine *e, int tag)
{
    const mm_model_cfg *mc = &e->mc;
    uint32_t i;

    switch (tag) {
    case 4:
        for (i = 0; i < mc->layers; i++)
            if (mm_model_cfg_layer_is_full(mc, i + 1))
                return e->kv->k_base[i];
        return NULL;
    case 5:
        return e->st->base;
    case 6:
        return e->st->conv;
    default:
        return NULL;
    }
}

/* -------------------------------------------------------- device I/O -- */

static uint8_t *g_dev;               /* D2H readback scratch              */
static size_t   g_dev_cap;
static uint8_t *g_gol;               /* golden bytes scratch              */
static size_t   g_gol_cap;

static int buf_grow(uint8_t **p, size_t *cap, size_t n)
{
    if (n <= *cap)
        return 1;
    *p = realloc(*p, n);
    if (!*p) {
        fprintf(stderr, "FAIL: parity scratch realloc (%zu)\n", n);
        return 0;
    }
    *cap = n;
    return 1;
}

#ifdef MM_WITH_CUDA
/* One observable: D2H on the xfer stream, event-ordered after the
 * compute work and synced (the parity-test mirror of the engine's
 * fetch_tokens round-boundary read). */
static mm_status d2h(const void *dev, size_t n)
{
    if (n > g_dev_cap) {
        if (!buf_grow(&g_dev, &g_dev_cap, n))
            return MM_ERR_NOMEM;
    }
    MM_CHECK(mm_event_record(PV_EV_READ, MM_ST_COMPUTE));
    MM_CHECK(mm_event_wait(PV_EV_READ, MM_ST_XFER));
    MM_CHECK(mm_d2h_async(g_dev, dev, n, MM_ST_XFER));
    MM_CHECK(mm_stream_sync(MM_ST_XFER));
    return MM_OK;
}

/* The round's sampled tokens (ab.toks_dev -> the pinned e->toks_host). */
static mm_status fetch_tok(mm_engine *e, uint32_t n, const uint32_t **out)
{
    MM_CHECK(mm_event_record(PV_EV_READ, MM_ST_COMPUTE));
    MM_CHECK(mm_event_wait(PV_EV_READ, MM_ST_XFER));
    MM_CHECK(mm_d2h_async(e->toks_host, e->ab.toks_dev, (size_t)n * 4,
                          MM_ST_XFER));
    MM_CHECK(mm_stream_sync(MM_ST_XFER));
    *out = e->toks_host;
    return MM_OK;
}

/* This round's control buffer -> device (xfer stream), compute ordered
 * after it — the engine's push_ctrl replicated for the direct walk. The
 * block-table slices pushed earlier ride the same stream. */
static mm_status push_ctrl(mm_engine *e)
{
    MM_CHECK(mm_h2d_async(e->ctrl_dev, &e->ctrl, sizeof e->ctrl,
                          MM_ST_XFER));
    MM_CHECK(mm_event_record(PV_EV_PUSH, MM_ST_XFER));
    MM_CHECK(mm_event_wait(PV_EV_PUSH, MM_ST_COMPUTE));
    return MM_OK;
}
#else
/* Host self-check variants: the arenas are host memory, so the "device"
 * pointers are the data itself (the CPU kernels read the host control
 * mirror e->ctrl directly; the block tables are host arrays). */
static mm_status d2h(const void *dev, size_t n)
{
    if (n > g_dev_cap) {
        if (!buf_grow(&g_dev, &g_dev_cap, n))
            return MM_ERR_NOMEM;
    }
    memcpy(g_dev, dev, n);
    return MM_OK;
}

static mm_status fetch_tok(mm_engine *e, uint32_t n, const uint32_t **out)
{
    (void)n;
    *out = (const uint32_t *)e->ab.toks_dev;
    return MM_OK;
}

static mm_status push_ctrl(mm_engine *e)
{
    (void)e;
    return MM_OK;
}
#endif

/* Control-buffer fill, mirroring engine.c's set_sampling() (one 64-bit rng
 * draw per round; the same stream the golden walk consumed). */
static void set_sampling(mm_engine *e)
{
    e->ctrl.temperature = e->cfg.temperature;
    e->ctrl.top_k = (uint32_t)e->cfg.top_k;
    e->ctrl.top_p = e->cfg.top_p;
    e->ctrl.u = mm_rng_fixed32(mm_rng_next(&e->rng));
}

/* Allocate KV blocks [lo,hi) for slot 0: mirrors engine.c's
 * ensure_kv_blocks (idempotent). */
static mm_status ensure_blocks(mm_engine *e, uint32_t lo, uint32_t hi)
{
    uint32_t b, first, last;

    if (!e->kv || lo >= hi)
        return MM_OK;
    first = lo / MM_KV_BLOCK_TOK;
    last = (hi - 1) / MM_KV_BLOCK_TOK;
    for (b = first; b <= last; b++) {
        uint32_t blk;
        if (mm_kvpool_alloc_block(e->kv, 0, b, &blk) != MM_OK)
            return MM_ERR_NOMEM;
    }
    return MM_OK;
}

/* ------------------------------------------------------ comparison ---- */

/* Compare one observable (golden bytes `g` vs device readback `got`)
 * under its tolerance class. Class 0 memcmps; classes 1/2 compare
 * element-wise in f32 ulps. Reports the first violation; tracks the max
 * deviation per class for the end-of-run summary. */
static void cmp_obs(const char *ctx, uint32_t op, int tag,
                    const uint8_t *g, const uint8_t *got, size_t n)
{
    int cls = obs_class(op, tag);

    if (cls == 0) {
        size_t i;
        for (i = 0; i < n; i++) {
            g_elems++;
            if (g[i] != got[i]) {
                fprintf(stderr,
                        "FAIL %s: bit-exact mismatch at byte %zu/%zu "
                        "(golden %02x, got %02x)\n",
                        ctx, i, n, g[i], got[i]);
                g_fail = 1;
                return;
            }
            g_exact++;
        }
        return;
    }
    {
        int es = elem_bytes(op, tag);
        long long tol = (es == 4) ? F32_TOL : BF16_ULP_F32;
        size_t ne = n / (size_t)es;
        size_t i;

        for (i = 0; i < ne; i++) {
            float fg, ff;
            const uint8_t *pg = g + i * (size_t)es;
            const uint8_t *pf = got + i * (size_t)es;

            if (es == 4) {
                memcpy(&fg, pg, 4);
                memcpy(&ff, pf, 4);
            } else {
                fg = b16_rd(pg);
                ff = b16_rd(pf);
            }
            g_elems++;
            if (fg == ff) {
                g_exact++;
                continue;
            }
            {
                long long d = f32_map(fg) - f32_map(ff);
                if (d < 0)
                    d = -d;
                if (d > g_max_ulp[cls])
                    g_max_ulp[cls] = d;
            }
            if (!ulp_ok(fg, ff, tol)) {
                long long d = f32_map(fg) - f32_map(ff);
                if (d < 0)
                    d = -d;
                fprintf(stderr,
                        "FAIL %s: element %zu/%zu deviates %lld f32 ulp "
                        "(tol %lld): golden %.9g got %.9g\n",
                        ctx, i, ne, d, tol, (double)fg, (double)ff);
                g_fail = 1;
                return;
            }
        }
    }
}

/* ------------------------------------------------------- round walk --- */

/* Run one plan op by op (mm_kx_invoke on the compute stream), cross-check
 * the plan structure against the golden's op descriptors, and compare
 * every observable. After the last op, compare the three pool/state
 * regions. Mirrors the golden writer's run_plan. */
static mm_status run_round(mm_engine *e, const mm_plan *plan,
                           uint32_t round, const char *what)
{
    uint32_t g_round, M, n_ops, i;

    if (!fr_u32(&g_round) || !fr_u32(&M) || !fr_u32(&n_ops))
        return MM_ERR_STATE;
    CHECK(g_round == round, "golden round id matches");
    CHECK(M == plan->M, "golden M matches the plan");
    CHECK(n_ops == plan->n_ops, "golden n_ops matches the plan");
    for (i = 0; i < n_ops; i++) {
        uint32_t gop, gl, gn0, gn1, gn2, n_obs, j;
        const mm_op_desc *op = &plan->ops[i];
        mm_kcall kc;
        obs o[MAX_OBS];
        int n;

        if (!fr_u32(&gop) || !fr_u32(&gl) || !fr_u32(&gn0) ||
            !fr_u32(&gn1) || !fr_u32(&gn2) || !fr_u32(&n_obs))
            return MM_ERR_STATE;
        CHECK(gop == (uint32_t)op->op && gl == (uint32_t)op->layer &&
              gn0 == op->n0 && gn1 == op->n1 && gn2 == op->n2,
              "plan structure matches the golden walk");

        memset(&kc, 0, sizeof kc);
        kc.op = op;
        kc.e = e;
        kc.stream = NULL;            /* launchers use the compute stream */
        MM_CHECK(mm_kx_invoke(&kc));
        g_ops++;

        n = op_obs(e, op, o);
        CHECK(n >= 0 && (uint32_t)n == n_obs, "observable set matches");
        for (j = 0; j < (uint32_t)n; j++) {
            uint32_t gtag, gnb;
            char ctx[128];

            if (!fr_u32(&gtag) || !fr_u32(&gnb))
                return MM_ERR_STATE;
            CHECK(gtag == (uint32_t)o[j].tag && gnb == o[j].bytes,
                  "observable tag/size matches the golden");
            if (!buf_grow(&g_gol, &g_gol_cap, o[j].bytes) ||
                !fr_bytes(g_gol, o[j].bytes))
                return MM_ERR_STATE;
            MM_CHECK(d2h(obs_ptr(op, o[j].tag), o[j].bytes));
            g_obs++;
            snprintf(ctx, sizeof ctx, "%s op%u tag%d", what, i, o[j].tag);
            cmp_obs(ctx, (uint32_t)op->op, o[j].tag, g_gol, g_dev,
                    o[j].bytes);
        }
    }

    {
        obs p[3];
        int j;

        pool_obs(e, p);
        for (j = 0; j < 3; j++) {
            uint32_t gtag, gnb;
            char ctx[128];

            if (!fr_u32(&gtag) || !fr_u32(&gnb))
                return MM_ERR_STATE;
            CHECK(gtag == (uint32_t)p[j].tag && gnb == p[j].bytes,
                  "pool observable tag/size matches the golden");
            if (!buf_grow(&g_gol, &g_gol_cap, p[j].bytes) ||
                !fr_bytes(g_gol, p[j].bytes))
                return MM_ERR_STATE;
            MM_CHECK(d2h(pool_ptr(e, p[j].tag), p[j].bytes));
            g_obs++;
            snprintf(ctx, sizeof ctx, "%s pool tag%d", what, p[j].tag);
            cmp_obs(ctx, (uint32_t)OP_N, p[j].tag, g_gol, g_dev,
                    p[j].bytes);
        }
    }
    return MM_OK;
}

/* One prefill round for the PAR_PROMPT: mirrors the golden walk (ctrl
 * fill, block alloc + table push, push_ctrl, op-by-op run), then reads the
 * sampled continuation token back from the device. */
static mm_status par_prefill(mm_engine *e, const mm_plan *plan,
                             uint32_t *tok)
{
    const uint32_t plen = PAR_PLEN;
    const uint32_t *out;
    mm_status s;

    memset(&e->ctrl, 0, sizeof e->ctrl);
    memcpy(e->ctrl.toks, PAR_PROMPT, plen * sizeof(uint32_t));
    e->ctrl.pos0 = 0;
    e->ctrl.kv_len = plen;
    e->ctrl.n_slots = 0;
    set_sampling(e);
    mm_rope_fill_ctrl(&e->rope, &e->ctrl);

    MM_CHECK(ensure_blocks(e, 0, plen));
#ifdef MM_WITH_CUDA
    MM_CHECK(mm_kvpool_push_tab(e->kv, 0, 0,
                                (plen - 1) / MM_KV_BLOCK_TOK + 1));
#endif
    MM_CHECK(push_ctrl(e));
    s = run_round(e, plan, 0, "prefill");
    if (s != MM_OK)
        return s;

    MM_CHECK(fetch_tok(e, plen, &out));
    *tok = out[plen - 1];
    CHECK(*tok < e->mc.vocab, "prefill token in vocab");
    return MM_OK;
}

/* One decode round for slot 0 at position `pos` (round_id 1..n): mirrors
 * the golden walk, then reads the sampled next token back from the
 * device. */
static mm_status par_decode(mm_engine *e, const mm_plan *plan,
                            uint32_t round_id, uint32_t pos,
                            uint32_t tok_in, uint32_t *tok)
{
    const uint32_t *out;
    mm_status s;

    memset(&e->ctrl, 0, sizeof e->ctrl);
    e->ctrl.n_slots = 1;
    e->ctrl.slot_tok[0] = tok_in;
    e->ctrl.slot_pos[0] = pos;
    e->ctrl.slot_len[0] = pos + 1;
    set_sampling(e);
    mm_rope_fill_ctrl(&e->rope, &e->ctrl);

    MM_CHECK(ensure_blocks(e, pos, pos + 1));
#ifdef MM_WITH_CUDA
    MM_CHECK(mm_kvpool_push_tab(e->kv, 0, pos / MM_KV_BLOCK_TOK,
                                pos / MM_KV_BLOCK_TOK + 1));
#endif
    MM_CHECK(push_ctrl(e));
    s = run_round(e, plan, round_id, "decode");
    if (s != MM_OK)
        return s;

    MM_CHECK(fetch_tok(e, 1, &out));
    *tok = out[0];
    CHECK(*tok < e->mc.vocab, "decode token in vocab");
    return MM_OK;
}

/* CPU-reference coverage vector for the GPU build.
 *
 * check_coverage()'s mirror assertion compares kx_cuda_oppref() against
 * kx_cpu_oppref(). The GPU link set cannot provide the latter from
 * src/kernels/cpu/cx.c: that TU also defines the 18 kx_op_* launchers
 * that cx.cu defines, so linking both would multiply-define them. This
 * copy is compiled only in the -DMM_WITH_CUDA build (the host self-check
 * links the real definition from cpu/cx.c, where the guard is off) and
 * MUST stay identical to kx_cpu_oppref() in src/kernels/cpu/cx.c — the
 * same 18 executable opcodes, default 0. */
#ifdef MM_WITH_CUDA
int kx_cpu_oppref(int op)
{
    switch (op) {
    case OP_EMBED:
    case OP_LMHEAD:
    case OP_RMSNORM:
    case OP_ADD_RMSNORM:
    case OP_GEMM_F4:
    case OP_GEMM_BF16:
    case OP_QKV:
    case OP_ROPE:
    case OP_KVSTORE:
    case OP_ATT_DECODE:
    case OP_ATT_PREFILL:
    case OP_SOFTMAX_CAUSAL:
    case OP_SILU_MUL:
    case OP_CONV4:
    case OP_LIN_DEC:
    case OP_LIN_PRE:
    case OP_COPY:
    case OP_SAMPLE:
        return 1;
    default:
        return 0;   /* OP_NOP / OP_N / unknown */
    }
}
#endif

/* Coverage: the GPU launcher set must mirror the CPU reference set, so no
 * executable opcode is left without a GPU path (same assertion the golden
 * writer makes for kx_cpu_oppref). Host self-check: the CPU reference set
 * is the oracle and must itself cover every executable opcode. */
static void check_coverage(void)
{
    int op, n = 0;

    for (op = 0; op < OP_N; op++) {
        if (op == OP_NOP)
            continue;
#ifdef MM_WITH_CUDA
        CHECK(kx_cuda_oppref(op) == kx_cpu_oppref(op),
              "GPU coverage mirrors the CPU reference");
        if (kx_cuda_oppref(op))
            n++;
#else
        CHECK(kx_cpu_oppref(op) == 1,
              "CPU reference covers every executable opcode");
        n++;
#endif
    }
#ifdef MM_WITH_CUDA
    printf("  coverage: GPU launchers cover %d/%d executable opcodes\n",
           n, OP_N - 1);
#else
    printf("  coverage: CPU reference covers %d/%d executable opcodes\n",
           n, OP_N - 1);
#endif
}

int main(int argc, char **argv)
{
    const char *path = argc > 1 ? argv[1] : "/tmp/par_golden.bin";
    mm_engine_cfg cfg;
    mm_engine *e = NULL;
    mm_plan pre;
    char magic[8] = {0};
    uint32_t h_seed, h_maxctx, h_maxout, h_chunk, h_rounds, term;
    uint32_t tok = 0, pos, i;
    int dec_done = 0;
    mm_status s;
    FILE *f;

#ifdef MM_WITH_CUDA
    CHECK(mm_cuda_enabled() == 1, "build must define MM_WITH_CUDA");
#endif

    f = fopen(path, "rb");
    CHECK(f != NULL, "open golden file");
    if (g_fail)
        return 1;
    g_f = f;
    CHECK(fread(magic, 1, sizeof magic, f) == sizeof magic &&
          memcmp(magic, PAR_MAGIC, sizeof PAR_MAGIC - 1) == 0,
          "golden magic MMPAR001");
    CHECK(fr_u32(&h_seed) && fr_u32(&h_maxctx) && fr_u32(&h_maxout) &&
          fr_u32(&h_chunk) && fr_u32(&h_rounds), "golden header");
    CHECK(h_seed == PAR_SEED && h_maxctx == PAR_MAXCTX &&
          h_maxout == PAR_MAXOUT && h_chunk == PAR_CHUNK &&
          h_rounds == 1 + PAR_MAXOUT,
          "golden config matches this test's constants");

    check_coverage();
    if (g_fail)
        return 1;

    mm_engine_cfg_default(&cfg);   /* canonical defaults, then overrides */
    cfg.max_ctx = PAR_MAXCTX;
    cfg.kv_capacity = 0;
    cfg.kv_dtype = MM_KV_BF16;
    cfg.concurrency = 1;
    cfg.chunk = PAR_CHUNK;
    cfg.draft = 0;
    cfg.temperature = 0.0f;      /* greedy: tokens are the argmax rows */
    cfg.top_k = 0;
    cfg.top_p = 1.0f;
    cfg.seed = PAR_SEED;
    cfg.verbose = 0;
    cfg.no_graph = 1;            /* direct per-op dispatch; graphs: gpu_smoke */

    s = mm_engine_create(&cfg, &e);
    CHECK(s == MM_OK, "engine create");
    if (s != MM_OK)
        fprintf(stderr, "        engine create: code=%d (%s)\n",
                (int)s, mm_status_str(s));
    if (g_fail)
        return 1;
    CHECK(e->mc.n_full_layers == 2 && e->mc.n_lin_layers == 6,
          "tiny hybrid shape (2 full, 6 lin)");
    s = mm_engine_load(e);
    CHECK(s == MM_OK, "engine load");
    if (g_fail) {
        mm_engine_destroy(e);
        return 1;
    }

    memset(&pre, 0, sizeof pre);
    CHECK(mm_plan_build(e, &pre, PH_PREFILL, PAR_PLEN) == MM_OK,
          "prefill plan build");
    if (g_fail) {
        mm_plan_destroy(&pre);
        mm_engine_destroy(e);
        return 1;
    }

    s = par_prefill(e, &pre, &tok);
    if (s == MM_OK) {
        for (i = 0; i < PAR_MAXOUT; i++) {
            pos = PAR_PLEN + i;
            s = par_decode(e, &e->dec[1], i + 1, pos, tok, &tok);
            if (s != MM_OK)
                break;
            dec_done++;
        }
    }
    CHECK(s == MM_OK && dec_done == (int)PAR_MAXOUT, "round walk compared");
    CHECK(fr_u32(&term) && term == 0, "golden terminator");
    CHECK(fgetc(g_f) == EOF, "golden EOF");

    printf("  compared %llu ops, %llu observables, %llu elements "
           "(%llu bit-exact)\n",
           (unsigned long long)g_ops, (unsigned long long)g_obs,
           (unsigned long long)g_elems, (unsigned long long)g_exact);
    printf("  max deviation class 1 (computed bf16): %lld f32 ulp "
           "= %.3f bf16 ulp (tol 1 bf16 ulp)\n",
           g_max_ulp[1], (double)g_max_ulp[1] / (double)BF16_ULP_F32);
    printf("  max deviation class 2 (f32 state / bf16 conv tail): "
           "%lld f32 ulp (tol 4 f32 / 1 bf16 ulp)\n", g_max_ulp[2]);

    if (!g_fail)
        printf("PARITY PASSED (%s: 1 prefill + %u decode rounds)\n",
               path, PAR_MAXOUT);
    else
        printf("PARITY FAILED (%s)\n", path);

    mm_plan_destroy(&pre);
    mm_engine_destroy(e);
    free(g_dev);
    free(g_gol);
    return g_fail ? 1 : 0;
}