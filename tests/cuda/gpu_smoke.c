/*
 * End-to-end CUDA smoke test (nvcc build, requires a GPU).
 *
 * Drives the engine's REAL round path — mm_engine_step with the
 * scheduler, the prefill/decode step functions, graph launch or direct
 * per-op dispatch, and the round-boundary device I/O (control-buffer
 * H2D, token D2H, block-table pushes) — for the same two cases as the
 * host engine smoke test (tests/host/engine_smoke.c), and verifies:
 *
 *   1. device lifecycle: create (device gate + probe), load (device
 *      arenas, pools, plan build + capture), step to idle, destroy;
 *   2. no CUDA errors: every mm_status is MM_OK and a final
 *      mm_device_sync_all() is clean (stream syncs surface invalid
 *      memory accesses as errors; for a deeper sweep run the binary
 *      under cuda-memcheck or compute-sanitizer);
 *   3. reproducible tokens: two runs of a case with a fixed seed give
 *      byte-identical token streams;
 *   4. graph vs direct: the captured CUDA-graph runs (no_graph = 0)
 *      produce the same token stream as the per-op direct-dispatch
 *      runs (no_graph = 1), and capture is verified to have happened;
 *   5. CPU-oracle agreement: the token heads match the known host CPU
 *      reference heads (the engine_smoke outputs), i.e. end-to-end
 *      agreement on top of the per-op parity gate.
 *
 * The numeric per-op gate is tests/cuda/parity_test.c; this test is the
 * plumbing/behavior gate.
 *
 * Build (GPU machine; the .c sources compile as C, cx.cu as CUDA):
 *   nvcc -O2 -fmad=false -DMM_WITH_CUDA \
 *        -Iinclude -Isrc/model -Isrc/kernels \
 *        -x c tests/cuda/gpu_smoke.c \
 *        -x c src/engine/engine.c -x c src/plan/plan.c \
 *        -x c src/alloc/alloc.c -x c src/config/config.c \
 *        -x c src/core/mimfer.c -x c src/cuda/cuda_rt.c \
 *        -x c src/cuda/cuda_mem.c -x c src/sampling/sampling.c \
 *        -x c src/kv/kv.c -x c src/kernels/kx.c \
 *        src/kernels/cuda/cx.cu \
 *        -x c src/model/tensor_registry.c -x c src/sched/sched.c \
 *        -o /tmp/gpu_smoke
 * Run:
 *   /tmp/gpu_smoke
 */
#include "mimfer/engine.h"
#include "mimfer/cuda_rt.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if !defined(MM_WITH_CUDA)
#error "gpu_smoke is a CUDA build test: build with -DMM_WITH_CUDA"
#endif

static int g_fail = 0;
#define CHECK(cond, msg)                                                     \
    do {                                                                     \
        if (!(cond)) {                                                       \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg);    \
            g_fail = 1;                                                      \
        }                                                                    \
    } while (0)

/* The host CPU reference (tests/host/engine_smoke.c, the correctness
 * oracle) produces these token heads for the two cases below; the GPU
 * end-to-end path must match them exactly (greedy sampling). */
static const uint32_t SHORT_HEAD[8] = { 451, 20, 430, 317, 0, 152, 24, 414 };
static const uint32_t LONG_HEAD[8]  = { 127, 230, 247, 387, 485, 327, 294,
                                        387 };

typedef enum smoke_mode {
    MODE_GRAPH = 0,   /* plans captured into CUDA graphs (fast path)   */
    MODE_DIRECT = 1   /* per-op direct dispatch (no_graph)             */
} smoke_mode;

/* One full engine lifecycle for a configured case: create, load, submit a
 * 4-token prompt, step to idle, capture the sampled token stream (one
 * entry per round), final device sync, destroy. `mode` selects the plan
 * execution mode. Mirrors tests/host/engine_smoke.c's run_case. */
static mm_status run_case(smoke_mode mode, uint32_t max_ctx, uint32_t max_out,
                          uint64_t seed, uint32_t *out, uint32_t cap,
                          uint32_t *n_out, uint64_t *tokens_out)
{
    mm_engine_cfg cfg;
    mm_engine *e = NULL;
    const uint32_t prompt[4] = { 5, 9, 17, 42 };
    uint32_t n = 0, rounds = 0;
    mm_status s;

    mm_engine_cfg_default(&cfg);   /* canonical defaults, then overrides */
    cfg.max_ctx = max_ctx;
    cfg.kv_capacity = 0;               /* auto: one sequence, 64-aligned */
    cfg.kv_dtype = MM_KV_BF16;
    cfg.concurrency = 1;
    cfg.chunk = 8;                     /* prompt (4) fits one prefill    */
    cfg.draft = 0;
    cfg.temperature = 0.0f;            /* greedy (graph-capturable)      */
    cfg.top_k = 0;
    cfg.top_p = 1.0f;
    cfg.seed = seed;
    cfg.verbose = 0;
    cfg.no_graph = (mode == MODE_DIRECT);

    s = mm_engine_create(&cfg, &e);
    if (s != MM_OK)
        return s;
    if (e->mc.n_full_layers != 2 || e->mc.n_lin_layers != 6) {
        mm_engine_destroy(e);
        return MM_ERR_STATE;
    }

    s = mm_engine_load(e);
    if (s != MM_OK) {
        mm_engine_destroy(e);
        return s;
    }
    if (!e->kv || !e->st || !e->ab.toks_dev || !e->ctrl_dev) {
        mm_engine_destroy(e);
        return MM_ERR_STATE;
    }
    /* Verify the execution mode actually took effect. */
    CHECK(e->pre_c[0].captured == (mode == MODE_GRAPH),
          "prefill capture state matches the requested mode");
    CHECK((e->pre_c[0].graph != NULL) == (mode == MODE_GRAPH),
          "prefill graph handle matches the requested mode");
    CHECK(e->dec[1].captured == (mode == MODE_GRAPH),
          "decode capture state matches the requested mode");
    if (g_fail) {
        mm_engine_destroy(e);
        return MM_ERR_STATE;
    }

    if (mm_sched_submit(&e->sched, prompt, 4, max_out) != MM_OK) {
        mm_engine_destroy(e);
        return MM_ERR_STATE;
    }

    /* Step until idle: one prefill round, then decode to the budget. */
    while (mm_sched_active(&e->sched) || mm_sched_queued(&e->sched)) {
        s = mm_engine_step(e);
        if (s != MM_OK) {
            mm_engine_destroy(e);
            return s;
        }
        if (n < cap)
            out[n] = e->last_tok[0];
        n++;
        if (++rounds > 1024) {
            mm_engine_destroy(e);
            return MM_ERR_STATE;
        }
    }

    /* Final error sweep: any invalid access or launch error the round
     * syncs missed surfaces here as an mm_status. */
    s = mm_device_sync_all();
    *n_out = n;
    *tokens_out = e->n_tokens_out;
    mm_engine_destroy(e);
    return s;
}

/* Static helper: two streams of `n` tokens byte-identical? */
static int streams_eq(const uint32_t *a, const uint32_t *b, uint32_t n)
{
    uint32_t i;
    for (i = 0; i < n; i++)
        if (a[i] != b[i])
            return 0;
    return 1;
}

/* Run a case in both execution modes (twice each) and verify:
 * determinism within a mode, graph == direct cross-mode agreement,
 * round count, and the CPU-oracle token head. */
static void check_case(const char *name, uint32_t max_ctx, uint32_t max_out,
                       uint64_t seed, const uint32_t *oracle_head)
{
    uint32_t ga[1024], gb[1024], da[1024], db[1024];
    uint32_t nga = 0, ngb = 0, nda = 0, ndb = 0;
    uint64_t tga = 0, tgb = 0, tda = 0, tdb = 0;
    int i;
    uint32_t n;
    mm_status s;

    s = run_case(MODE_GRAPH, max_ctx, max_out, seed, ga, 1024, &nga, &tga);
    CHECK(s == MM_OK, "graph run A");
    s = run_case(MODE_GRAPH, max_ctx, max_out, seed, gb, 1024, &ngb, &tgb);
    CHECK(s == MM_OK, "graph run B");
    s = run_case(MODE_DIRECT, max_ctx, max_out, seed, da, 1024, &nda, &tda);
    CHECK(s == MM_OK, "direct run A");
    s = run_case(MODE_DIRECT, max_ctx, max_out, seed, db, 1024, &ndb, &tdb);
    CHECK(s == MM_OK, "direct run B");

    n = (uint32_t)(1 + max_out);
    CHECK(nga == n && ngb == n && nda == n && ndb == n,
          "round count = 1 prefill + max_out decode (all modes)");
    CHECK(tga == tgb && tga == tda && tga == tdb,
          "same token counter across all runs");
    CHECK(streams_eq(ga, gb, nga), "deterministic graph-mode token stream");
    CHECK(streams_eq(da, db, nda), "deterministic direct-mode token stream");
    CHECK(streams_eq(ga, da, n), "graph and direct dispatch agree");
    for (i = 0; i < 8 && i < (int)n; i++)
        CHECK(ga[i] == oracle_head[i], "token stream matches the CPU oracle");

    printf("  %-22s mode=graph+direct rounds=%u tokens_out=%llu head=[",
           name, (unsigned)n, (unsigned long long)tga);
    for (i = 0; i < 8 && i < (int)n; i++)
        printf("%s%u", i ? " " : "", ga[i]);
    printf("]\n");
}

int main(void)
{
    CHECK(mm_cuda_enabled() == 1, "build must define MM_WITH_CUDA");

    /* Same cases as the host engine smoke test (the CPU oracle). Short
     * run: stays within KV block 0. Long run: plen=4 + 70 decode crosses
     * the 64-token boundary into block 1 (block-table path beyond 0). */
    check_case("short (block 0)", 128, 16, 12345, SHORT_HEAD);
    check_case("long (crosses block)", 128, 70, 99, LONG_HEAD);

    if (g_fail) {
        fprintf(stderr, "GPU SMOKE FAILED\n");
        return 1;
    }
    printf("GPU SMOKE PASSED\n");
    return 0;
}