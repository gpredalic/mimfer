/*
 * Slot lifecycle regression (multi-request teardown), host + GPU.
 *
 * The round scheduler recycles slots: when a sequence finishes, its slot
 * must be torn down so the next sequence starts from exactly the state a
 * fresh engine provides:
 *
 *   S1  the slot's KV blocks are released back to the pool
 *       (mm_kvpool_release_slot) -- without the release the blocks leak on
 *       every sequence and the pool exhausts under multi-request load;
 *   S2  the slot's linear recurrence/conv state is reset
 *       (mm_linstate_reset) -- without the reset the finished sequence's
 *       state contaminates the next sequence's linear layers;
 *   S3  the slot's block-table row is back to the "no block" state (the
 *       host mirror is zeroed by the release; the device row is pushed the
 *       same zeroed bytes by the teardown).
 *
 * The proof: sequence A (40 prompt + 40 generated = 80 context tokens,
 * spanning two 64-token KV blocks) runs to completion; sequence B then runs
 * on the slot A freed (the scheduler reuses the lowest free slot) and its
 * full token stream must be byte-identical to B's stream on a fresh engine
 * with the same seed. The pool's free-block count must return to its
 * post-load value after A and after B, and A's block-table row must be all
 * zero after teardown.
 *
 * One source, two builds: the host suite (gcc, MM_WITH_CUDA undefined --
 * the CPU reference is the golden path; the CPU kernels read the block
 * table from the very mirror this test checks) and the GPU suite (nvcc,
 * MM_WITH_CUDA -- the same engine code against device memory). No external
 * oracle is needed: the fresh-engine comparison is in-process (two engines
 * run sequentially in one process; the CUDA handle set is shut down and
 * re-opened between them, the same pattern gpu_smoke uses for its cases).
 *
 * Build (host):
 *   make slot-lifecycle
 * Build (GPU):
 *   make gpu-slot    (run on the GPU machine; GPU_VALIDATION.md §8)
 */
#include "mimfer/engine.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SEED        12345
#define A_PLEN      40      /* A: 40 prompt + 40 generated = 80 context   */
#define A_MAX_OUT   40      /*        -> spans two 64-token KV blocks     */
#define B_PLEN      5
#define B_MAX_OUT   12      /* B: 5 prompt + 12 generated = 17 context    */
#define MAX_ROUNDS  128

static int g_fail = 0;
#define CHECK(cond, msg)                                                      \
    do {                                                                      \
        if (!(cond)) {                                                        \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg);     \
            g_fail = 1;                                                       \
        }                                                                     \
    } while (0)

/* Drive one submitted sequence to completion, one scheduler round at a
 * time. Records the sampled token per round (1 prefill continuation + 1
 * per decode round) and the slot the sequence ran on. The per-round token
 * attribution uses the pre-step active-slot snapshot: a slot that finishes
 * in the round is RS_FREE by the time the step returns, but its token is
 * still in e->last_tok. If free_at_65 is non-NULL it is set to the number
 * of free KV blocks observed after the round that brought the sequence's
 * committed length to 65 (the round that allocated KV block 1). */
static mm_status drive_seq(mm_engine *e, const uint32_t *prompt, uint32_t plen,
                           uint32_t max_out, uint32_t *toks, uint32_t *n_tok,
                           int *slot_used, uint32_t *free_at_65)
{
    int r, s, n_act;
    int active[MM_MAX_CONCURRENCY];

    MM_CHECK(mm_sched_submit(&e->sched, prompt, plen, max_out));
    *n_tok = 0;
    *slot_used = -1;
    for (;;) {
        int slot = -1;
        uint32_t n_dec = 0;
        r = mm_sched_pick_round(&e->sched, &slot, &n_dec);
        if (r == 0)
            break;                       /* idle: the sequence is done */
        n_act = 0;
        for (s = 0; s < (int)e->sched.n; s++)
            if (e->sched.slots[s].state == RS_DECODE)
                active[n_act++] = s;
        MM_CHECK(mm_engine_step(e));
        if (r == 1) {                     /* prefill round: one token */
            if (*n_tok < MAX_ROUNDS)
                toks[(*n_tok)++] = e->last_tok[slot];
            if (*slot_used < 0)
                *slot_used = slot;
        } else {                          /* decode round: one token per
                                             active slot, ascending order */
            for (s = 0; s < n_act; s++) {
                if (*n_tok < MAX_ROUNDS)
                    toks[(*n_tok)++] = e->last_tok[active[s]];
                if (*slot_used < 0)
                    *slot_used = active[s];
            }
        }
        if (free_at_65)
            for (s = 0; s < (int)e->sched.n; s++)
                if (e->sched.slots[s].len == 65)
                    *free_at_65 = mm_kvpool_free_blocks(e->kv);
    }
    return MM_OK;
}

/* Create + load an engine with the test shape; verify the tiny hybrid. */
static mm_status open_engine(uint64_t seed, mm_engine **out)
{
    mm_engine_cfg cfg;
    mm_engine *e = NULL;
    mm_status s;

    mm_engine_cfg_default(&cfg);
    cfg.max_ctx = 128;          /* auto KV pool: 128 tok = 2 blocks of 64 */
    cfg.kv_capacity = 0;        /* auto                                    */
    cfg.kv_dtype = MM_KV_BF16;
    cfg.concurrency = 2;        /* two slots: reuse is a choice, not a
                                   single-slot triviality                  */
    cfg.chunk = 64;             /* both prompts (40, 5) fit one prefill
                                   round (single-chunk prefill)            */
    cfg.draft = 0;
    cfg.temperature = 0.0f;     /* greedy: deterministic + graph-capturable */
    cfg.top_k = 0;
    cfg.top_p = 1.0f;
    cfg.seed = seed;            /* same seed => identical fake weights in
                                   both engines                            */
    cfg.verbose = 0;
    cfg.no_graph = 0;           /* host build ignores; CUDA: graph decode  */

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
    *out = e;
    return MM_OK;
}

int main(void)
{
    mm_engine *fresh = NULL, *reuse = NULL;
    const uint32_t b_prompt[B_PLEN] = { 9, 21, 44, 6, 101 };
    uint32_t a_prompt[A_PLEN];
    uint32_t b_ref[MAX_ROUNDS], a_tok[MAX_ROUNDS], b_tok[MAX_ROUNDS];
    uint32_t n_ref = 0, n_a = 0, n_b = 0;
    uint32_t free0, free65 = 0, row, i;
    int s_ref = -1, s_a = -1, s_b = -1;
    int same;
    mm_status s;

    for (i = 0; i < A_PLEN; i++)
        a_prompt[i] = (uint32_t)((7 * i + 3) % 500);

    /* Case 1 -- fresh engine, B only: the reference stream. */
    s = open_engine(SEED, &fresh);
    CHECK(s == MM_OK, "fresh engine create+load");
    if (s == MM_OK) {
        s = drive_seq(fresh, b_prompt, B_PLEN, B_MAX_OUT, b_ref, &n_ref,
                      &s_ref, NULL);
        CHECK(s == MM_OK, "fresh: drive B");
        CHECK(n_ref == 1 + B_MAX_OUT,
              "fresh: rounds = 1 prefill + B_MAX_OUT decode");
        CHECK(s_ref == 0, "fresh: B runs on slot 0");
        mm_engine_destroy(fresh);
        fresh = NULL;
    }

    /* Case 2 -- the engine under test: A to completion (its slot is torn
     * down), then B on the slot A freed. */
    s = open_engine(SEED, &reuse);
    CHECK(s == MM_OK, "reuse engine create+load");
    if (s == MM_OK) {
        free0 = mm_kvpool_free_blocks(reuse->kv);
        CHECK(free0 >= 2, "pool holds at least two KV blocks");

        s = drive_seq(reuse, a_prompt, A_PLEN, A_MAX_OUT, a_tok, &n_a, &s_a,
                      &free65);
        CHECK(s == MM_OK, "reuse: drive A to completion");
        CHECK(n_a == 1 + A_MAX_OUT,
              "reuse: A rounds = 1 prefill + A_MAX_OUT decode");
        CHECK(s_a == 0, "reuse: A runs on slot 0");
        CHECK(free65 == free0 - 2,
              "A spans two KV blocks (both allocated by len 65)");
        /* S1: A's blocks are back in the pool -- no leak. */
        CHECK(mm_kvpool_free_blocks(reuse->kv) == free0,
              "S1: A's KV blocks released (pool free count back to post-load)");
        /* S3: A's block-table row is the "no block" state (the exact bytes
         * the CPU kernels read on the host build; the push source on the
         * device build). */
        row = (uint32_t)s_a * reuse->kv->max_blocks_seq;
        for (i = 0; i < reuse->kv->max_blocks_seq; i++)
            CHECK(reuse->kv->blk_tab_host[row + i] == 0,
                  "S3: A's block-table row zeroed after teardown");
        CHECK(reuse->n_tokens_out == (uint64_t)(1 + A_MAX_OUT),
              "token counter after A");

        s = drive_seq(reuse, b_prompt, B_PLEN, B_MAX_OUT, b_tok, &n_b, &s_b,
                      NULL);
        CHECK(s == MM_OK, "reuse: drive B on A's freed slot");
        CHECK(n_b == 1 + B_MAX_OUT,
              "reuse: B rounds = 1 prefill + B_MAX_OUT decode");
        CHECK(s_b == s_a, "B runs on the same slot A finished on");
        CHECK(mm_kvpool_free_blocks(reuse->kv) == free0,
              "S1: B's KV blocks released (pool free count back to post-load)");
        /* S2 + correctness: B's stream on the reused slot must equal the
         * fresh engine's B stream element for element. A leaked linear
         * recurrence/conv state on A's slot changes B's logits from the
         * first linear layer onward and breaks this comparison. */
        same = (n_b == n_ref);
        for (i = 0; same && i < n_b; i++)
            if (b_tok[i] != b_ref[i])
                same = 0;
        CHECK(same,
              "S2: B on the reused slot == fresh-engine B (state reset)");
        CHECK(reuse->n_tokens_out ==
              (uint64_t)(1 + A_MAX_OUT + 1 + B_MAX_OUT),
              "token counter after A+B");
        mm_engine_destroy(reuse);
    }

    printf("  fresh B              rounds=%u slot=%d head=[",
           (unsigned)n_ref, s_ref);
    for (i = 0; i < n_ref && i < 8; i++)
        printf("%s%u", i ? " " : "", b_ref[i]);
    printf("]\n");
    printf("  reuse A (teardown)   rounds=%u slot=%d head=[",
           (unsigned)n_a, s_a);
    for (i = 0; i < n_a && i < 8; i++)
        printf("%s%u", i ? " " : "", a_tok[i]);
    printf("]\n");
    printf("  reuse B (same slot)  rounds=%u slot=%d head=[",
           (unsigned)n_b, s_b);
    for (i = 0; i < n_b && i < 8; i++)
        printf("%s%u", i ? " " : "", b_tok[i]);
    printf("]\n");

    if (g_fail) {
        fprintf(stderr, "SLOT LIFECYCLE FAILED\n");
        return 1;
    }
    printf("SLOT LIFECYCLE PASSED\n");
    return 0;
}
