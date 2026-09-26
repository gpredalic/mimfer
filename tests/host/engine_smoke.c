/*
 * End-to-end host smoke test for the CPU-reference engine path (src/engine/
 * engine.c). Drives mm_engine_create -> mm_engine_load -> (submit + repeated
 * mm_engine_step) -> mm_engine_destroy against a tiny fake hybrid model with
 * no GPU. Exercises the real plan builder, the real kernel dispatch (kx.c ->
 * cx.c CPU references), the real KV + linear/conv pools, the real scheduler
 * and sampling -- and checks that the output token stream is deterministic
 * (byte-identical) across two independent runs with the same seed.
 *
 * Build (host, no GPU):
 *   gcc -std=c11 -Wall -Wextra -Werror -Iinclude -Isrc/model -Isrc/kernels \
 *       tests/host/engine_smoke.c \
 *       src/engine/engine.c src/plan/plan.c src/alloc/alloc.c \
 *       src/config/config.c src/core/mimfer.c src/cuda/cuda_rt.c \
 *       src/sampling/sampling.c src/kv/kv.c src/kernels/cpu/cx.c \
 *       src/kernels/kx.c src/model/tensor_registry.c src/sched/sched.c \
 *       -lm -o /tmp/engine_smoke
 */
#include "mimfer/engine.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail = 0;
#define CHECK(cond, msg)                                                      \
    do {                                                                      \
        if (!(cond)) {                                                        \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg);     \
            g_fail = 1;                                                       \
        }                                                                     \
    } while (0)

/* One full run of a configured shape: create, load, submit a 4-token prompt,
 * step to idle, capture the sampled token stream (one entry per round),
 * destroy. max_out large enough that the sequence crosses the 64-token KV
 * block boundary exercises the block-table path beyond block 0. */
static mm_status run_case(uint32_t max_ctx, uint32_t max_out, uint64_t seed,
                          uint32_t *out, uint32_t cap, uint32_t *n_out,
                          uint64_t *tokens_out)
{
    mm_engine_cfg cfg;
    mm_engine *e = NULL;
    const uint32_t prompt[4] = { 5, 9, 17, 42 };
    uint32_t n = 0, rounds = 0;
    mm_status s;

    memset(&cfg, 0, sizeof cfg);
    cfg.max_ctx = max_ctx;
    cfg.kv_capacity = 0;               /* auto: one sequence, 64-aligned     */
    cfg.kv_dtype = MM_KV_BF16;
    cfg.concurrency = 1;
    cfg.chunk = 8;                     /* prompt (4) fits one prefill round   */
    cfg.draft = 0;
    cfg.temperature = 0.0f;            /* greedy                             */
    cfg.top_k = 0;
    cfg.top_p = 1.0f;
    cfg.seed = seed;
    cfg.verbose = 0;
    cfg.no_graph = 1;

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
    if (!e->kv || !e->st || !e->ab.toks_dev) {
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

    *n_out = n;
    *tokens_out = e->n_tokens_out;
    mm_engine_destroy(e);
    return MM_OK;
}

/* Run a case twice and require the two token streams to be byte-identical.
 * A 1 (prefill continuation) + max_out (decode) rounds is expected. */
static void check_case(const char *name, uint32_t max_ctx, uint32_t max_out,
                       uint64_t seed)
{
    uint32_t a[1024], b[1024], na = 0, nb = 0;
    uint64_t ta = 0, tb = 0;
    int i;
    int same;

    mm_status sa = run_case(max_ctx, max_out, seed, a, 1024, &na, &ta);
    CHECK(sa == MM_OK, name);
    mm_status sb = run_case(max_ctx, max_out, seed, b, 1024, &nb, &tb);
    CHECK(sb == MM_OK, name);

    CHECK(na == (uint32_t)(1 + max_out) && nb == na,
          "round count = 1 prefill + max_out decode");
    CHECK(ta == tb, "same token counter both runs");
    same = (na == nb);
    for (i = 0; same && i < (int)na; i++)
        if (a[i] != b[i])
            same = 0;
    CHECK(same, "deterministic (byte-identical) token stream");

    printf("  %-22s rounds=%u tokens_out=%llu head=[", name, (unsigned)na,
           (unsigned long long)ta);
    for (i = 0; i < (int)na && i < 8; i++)
        printf("%s%u", i ? " " : "", a[i]);
    printf("]\n");
}

int main(void)
{
    /* Short run: stays within KV block 0. */
    check_case("short (block 0)", 128, 16, 12345);
    /* Long run: plen=4 + 70 decode crosses the 64-token boundary into
     * block 1, exercising the per-slot block table beyond index 0. */
    check_case("long (crosses block)", 128, 70, 99);

    if (g_fail) {
        fprintf(stderr, "ENGINE SMOKE FAILED\n");
        return 1;
    }
    printf("ENGINE SMOKE PASSED\n");
    return 0;
}
