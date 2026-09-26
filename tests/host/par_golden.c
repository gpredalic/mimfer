/*
 * CPU<->CUDA parity — golden writer (HOST build, no GPU needed).
 *
 * Runs the validated CPU reference path op by op and records the
 * observable output of every plan op into a golden file that the CUDA
 * parity test (tests/cuda/parity_test.c, nvcc build) compares its GPU
 * outputs against. The CPU path is the correctness oracle
 * (READ_MEMORY.md): everything this file writes is produced by the
 * bit-deterministic reference kernels.
 *
 * The walk drives the engine's REAL plans (the prefill plan for the
 * prompt length, the engine's decode plan for 1 slot) through
 * mm_kx_invoke one op at a time, with the control buffer filled
 * exactly as engine.c's step functions fill it, so the operands,
 * scalars and pool state the kernels see are the engine's own.
 *
 * Golden file format (little-endian):
 *   u8   magic[8] = "MMPAR001"
 *   u32  seed, max_ctx, max_out, chunk, n_rounds
 *   per round:
 *     u32  round_id (0 = prefill, 1.. = decode), M, n_ops
 *     per op:
 *       u32  op_type, layer, n0, n1, n2
 *       u32  n_obs
 *       per observable:
 *         u32  tag (0..3 = operand a..d; 4 = kv pool; 5 = lin state;
 *                  6 = conv state)
 *         u32  nbytes
 *         u8   bytes[nbytes]
 *   Pool/state observables (tags 4..6) are appended after the last op
 *   of every round: they capture the in-place KVSTORE / CONV4 / LIN_*
 *   pool effects that have no single output operand.
 *
 * Tolerance classes (consumed by the parity test; declared here so the
 * two files share one definition of "what must match"):
 *   class 0  bit-exact      EMBED, COPY, SAMPLE tokens, KVSTORE pool
 *   class 1  1 bf16 ulp     every computed bf16 output
 *   class 2  4 f32 ulp      fp32 recurrence / conv state
 *
 * Build (host, no GPU):
 *   gcc -std=c11 -Wall -Wextra -Werror -Iinclude -Isrc/model -Isrc/kernels \
 *       tests/host/par_golden.c src/engine/engine.c src/plan/plan.c \
 *       src/alloc/alloc.c src/config/config.c src/core/mimfer.c \
 *       src/cuda/cuda_rt.c src/cuda/cuda_mem.c src/sampling/sampling.c \
 *       src/kv/kv.c src/kernels/cpu/cx.c src/kernels/kx.c \
 *       src/model/tensor_registry.c src/sched/sched.c -lm \
 *       -o /tmp/par_golden
 * Run:
 *   /tmp/par_golden /tmp/par_golden.bin
 */
#include "mimfer/engine.h"
#include "kx.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PAR_MAGIC "MMPAR001"
/* Fixed shape for the parity walk: same seed/prompt as the host smoke
 * test's short case, so the golden stream is reproducible and comparable. */
#define PAR_SEED    12345u
#define PAR_MAXCTX  128u
#define PAR_MAXOUT  16u
#define PAR_CHUNK   8u
static const uint32_t PAR_PROMPT[4] = { 5, 9, 17, 42 };

static int g_fail = 0;
#define CHECK(cond, msg)                                                     \
    do {                                                                     \
        if (!(cond)) {                                                       \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg);    \
            g_fail = 1;                                                      \
        }                                                                    \
    } while (0)

/* One observable: which operand (0..3) or pool region (4..6) + size. */
typedef struct obs {
    int    tag;
    size_t bytes;
} obs;

#define MAX_OBS 4

/* Observable outputs of one op, sized from the finalized model shape
 * (the same numbers the plan builder bakes). KVSTORE has none: its
 * effect lives in the pool, dumped after the round. */
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

/* Note: the tolerance CLASS per observable is not stored in the golden
 * file -- the parity test derives it from (op type, tag) with the same
 * rules documented in this header (classes 0/1/2). */

static const void *obs_ptr(const mm_op_desc *op, int tag)
{
    switch (tag) {
    case 0:  return op->a;
    case 1:  return op->b;
    case 2:  return op->c;
    default: return op->d;
    }
}

/* Per-round pool/state observables: tags 4 (kv pool, bit-exact),
 * 5 (linear recurrence state, f32), 6 (conv state, bf16). All are the
 * slot-0 regions the single-slot reference kernels read/write. */
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

/* ---------------------------------------------------- file machinery */

static int fw_u32(FILE *f, uint32_t v)
{
    return fwrite(&v, 4, 1, f) == 1;
}

static int fw_bytes(FILE *f, const void *p, size_t n)
{
    return n == 0 || fwrite(p, 1, n, f) == n;
}

static int dump_obs(FILE *f, const char *ctx, int tag, const void *p,
                    size_t n)
{
    if (!fw_u32(f, (uint32_t)tag) || !fw_u32(f, (uint32_t)n) ||
        !fw_bytes(f, p, n)) {
        fprintf(stderr, "FAIL: golden write (%s)\n", ctx);
        return 0;
    }
    return 1;
}

/* ------------------------------------------------- round driving ---- */

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

/* Control-buffer fill, mirroring engine.c's set_sampling() + the two
 * step functions' ctrl layout (the parity test mirrors the same way, so
 * the two walks consume identical u draws). */
static void set_sampling(mm_engine *e)
{
    e->ctrl.temperature = e->cfg.temperature;
    e->ctrl.top_k = (uint32_t)e->cfg.top_k;
    e->ctrl.top_p = e->cfg.top_p;
    e->ctrl.u = mm_rng_fixed32(mm_rng_next(&e->rng));
}

/* Run one plan op by op, dumping every observable after each op and the
 * pool/state regions after the round. */
static mm_status run_plan(FILE *f, mm_engine *e, const mm_plan *plan,
                          uint32_t round, const char *what)
{
    uint32_t i;

    if (!fw_u32(f, round) || !fw_u32(f, plan->M) ||
        !fw_u32(f, plan->n_ops)) {
        fprintf(stderr, "FAIL: golden write (round header %s)\n", what);
        return MM_ERR_STATE;
    }
    for (i = 0; i < plan->n_ops; i++) {
        mm_kcall kc;
        obs o[MAX_OBS];
        int n, j;

        kc.op = &plan->ops[i];
        kc.e = e;
        kc.stream = NULL;
        if (mm_kx_invoke(&kc) != MM_OK)
            return MM_ERR_CUDA;

        n = op_obs(e, &plan->ops[i], o);
        if (n < 0) {
            fprintf(stderr, "FAIL: no observables for op %u (%s)\n", i,
                    what);
            return MM_ERR_STATE;
        }
        if (!fw_u32(f, (uint32_t)plan->ops[i].op) ||
            !fw_u32(f, (uint32_t)plan->ops[i].layer) ||
            !fw_u32(f, plan->ops[i].n0) || !fw_u32(f, plan->ops[i].n1) ||
            !fw_u32(f, plan->ops[i].n2) || !fw_u32(f, (uint32_t)n))
            return MM_ERR_STATE;
        for (j = 0; j < n; j++)
            if (!dump_obs(f, what, o[j].tag, obs_ptr(&plan->ops[i], o[j].tag),
                          o[j].bytes))
                return MM_ERR_STATE;
    }

    {
        obs p[3];
        int j;

        pool_obs(e, p);
        for (j = 0; j < 3; j++)
            if (!dump_obs(f, what, p[j].tag, pool_ptr(e, p[j].tag),
                          p[j].bytes))
                return MM_ERR_STATE;
    }
    return MM_OK;
}

/* One prefill round for the PAR_PROMPT: mirrors engine.c's step_prefill
 * (blocks allocated, ctrl filled, plan run), then records the sampled
 * continuation token in *tok. */
static mm_status par_prefill(FILE *f, mm_engine *e, const mm_plan *plan,
                             uint32_t *tok)
{
    const uint32_t plen = (uint32_t)(sizeof PAR_PROMPT / sizeof PAR_PROMPT[0]);
    mm_status s;

    memset(&e->ctrl, 0, sizeof e->ctrl);
    memcpy(e->ctrl.toks, PAR_PROMPT, plen * sizeof(uint32_t));
    e->ctrl.pos0 = 0;
    e->ctrl.kv_len = plen;
    e->ctrl.n_slots = 0;
    set_sampling(e);
    mm_rope_fill_ctrl(&e->rope, &e->ctrl);

    MM_CHECK(ensure_blocks(e, 0, plen));
    s = run_plan(f, e, plan, 0, "prefill");
    if (s != MM_OK)
        return s;

    *tok = ((const uint32_t *)e->ab.toks_dev)[plen - 1];
    CHECK(*tok < e->mc.vocab, "prefill token in vocab");
    return MM_OK;
}

/* One decode round for slot 0 at position `pos` (round_id 1..n):
 * mirrors engine.c's step_decode ctrl layout, then records the sampled
 * token in *tok. */
static mm_status par_decode(FILE *f, mm_engine *e, const mm_plan *plan,
                            uint32_t round_id, uint32_t pos,
                            uint32_t tok_in, uint32_t *tok)
{
    mm_status s;

    memset(&e->ctrl, 0, sizeof e->ctrl);
    e->ctrl.n_slots = 1;
    e->ctrl.slot_tok[0] = tok_in;
    e->ctrl.slot_pos[0] = pos;
    e->ctrl.slot_len[0] = pos + 1;
    set_sampling(e);
    mm_rope_fill_ctrl(&e->rope, &e->ctrl);

    MM_CHECK(ensure_blocks(e, pos, pos + 1));
    s = run_plan(f, e, plan, round_id, "decode");
    if (s != MM_OK)
        return s;

    *tok = ((const uint32_t *)e->ab.toks_dev)[0];
    CHECK(*tok < e->mc.vocab, "decode token in vocab");
    return MM_OK;
}

/* Coverage assertion: every executable opcode the dispatcher can run
 * must be covered by the CPU reference (the parity test asserts the same
 * for kx_cuda_oppref). */
static void check_coverage(void)
{
    int op;

    for (op = 0; op < OP_N; op++) {
        if (op == OP_NOP)
            continue;
        CHECK(kx_cpu_oppref(op) == 1,
              "CPU reference must cover every executable opcode");
    }
    printf("  coverage: CPU reference covers %d/%d executable opcodes\n",
           OP_N - 1, OP_N - 1);
}

int main(int argc, char **argv)
{
    const char *path = argc > 1 ? argv[1] : "/tmp/par_golden.bin";
    mm_engine_cfg cfg;
    mm_engine *e = NULL;
    mm_plan pre;
    uint32_t plen = (uint32_t)(sizeof PAR_PROMPT / sizeof PAR_PROMPT[0]);
    uint32_t tok, pos, i;
    mm_status s;
    FILE *f;

    mm_engine_cfg_default(&cfg);   /* canonical defaults, then overrides */
    cfg.max_ctx = PAR_MAXCTX;
    cfg.kv_capacity = 0;
    cfg.kv_dtype = MM_KV_BF16;
    cfg.concurrency = 1;
    cfg.chunk = PAR_CHUNK;
    cfg.draft = 0;
    cfg.temperature = 0.0f;       /* greedy: tokens are the argmax rows  */
    cfg.top_k = 0;
    cfg.top_p = 1.0f;
    cfg.seed = PAR_SEED;
    cfg.verbose = 0;
    cfg.no_graph = 1;

    CHECK(mm_engine_create(&cfg, &e) == MM_OK, "engine create");
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
    CHECK(mm_plan_build(e, &pre, PH_PREFILL, plen) == MM_OK,
          "prefill plan build");

    f = fopen(path, "wb");
    CHECK(f != NULL, "open golden file");
    if (g_fail) {
        mm_plan_destroy(&pre);
        mm_engine_destroy(e);
        return 1;
    }
    CHECK(fwrite(PAR_MAGIC, 1, sizeof PAR_MAGIC - 1, f) ==
          sizeof PAR_MAGIC - 1, "golden magic");
    CHECK(fw_u32(f, PAR_SEED) && fw_u32(f, PAR_MAXCTX) &&
          fw_u32(f, PAR_MAXOUT) && fw_u32(f, PAR_CHUNK) &&
          fw_u32(f, 1 + PAR_MAXOUT), "golden header");

    /* Round 0: prefill, then PAR_MAXOUT decode rounds for slot 0. The
     * position after prefill is plen (the first decode token lands at
     * pos = plen). */
    tok = 0;
    s = par_prefill(f, e, &pre, &tok);
    if (s == MM_OK) {
        for (i = 0; i < PAR_MAXOUT; i++) {
            pos = plen + i;
            s = par_decode(f, e, &e->dec[1], i + 1, pos, tok, &tok);
            if (s != MM_OK)
                break;
        }
    }
    CHECK(s == MM_OK, "round walk");
    CHECK(fw_u32(f, 0), "golden terminator");   /* sentinel round id */
    CHECK(fflush(f) == 0, "golden flush");
    CHECK(fclose(f) == 0, "golden close");

    check_coverage();

    printf("PAR GOLDEN WRITTEN %s (1 prefill + %u decode rounds, seed %u)\n",
           path, PAR_MAXOUT, PAR_SEED);
    mm_plan_destroy(&pre);
    mm_engine_destroy(e);
    return g_fail ? 1 : 0;
}
