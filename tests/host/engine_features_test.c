/*
 * End-to-end host test for the new engine feature surface
 * (src/engine/engine.c gates + the new cfg fields):
 *
 *   - YaRN on vs plain: same engine, same seed, same prompt -> the
 *     token streams MUST differ (different RoPE table changes every
 *     full-attention layer's logits); plain vs plain stays
 *     byte-identical (determinism, unchanged golden path).
 *   - speculative backends: --spec mtp / --spec dflash2 are validated
 *     by the flag surface but refused at engine start with
 *     MM_ERR_UNSUPPORTED (the draft/verify loop is planned work,
 *     docs/dflash2.md).
 *   - vision hook: enabled -> image submission returns
 *     MM_ERR_UNSUPPORTED; disabled or empty payload -> MM_ERR_STATE.
 *   - weights profiles: the selected profile is pinned on the engine
 *     and logged; a max_ctx above the checkpoint ceiling is refused.
 */
#include "mimfer/engine.h"

#include <math.h>
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

/* Boot a loaded engine (tiny host shape) with the given rope/vision
 * settings; NULL on any failure. */
static mm_engine *boot(int rope_yarn, uint32_t vision)
{
    mm_engine_cfg cfg;
    mm_engine *e = NULL;

    mm_engine_cfg_default(&cfg);
    cfg.max_ctx = 128;
    cfg.kv_capacity = 0;
    cfg.concurrency = 1;
    cfg.chunk = 8;
    cfg.temperature = 0.0f;
    cfg.verbose = 0;
    cfg.no_graph = 1;
    cfg.rope_yarn = rope_yarn;
    cfg.rope_factor = 4.0f;
    cfg.vision = vision;
    if (mm_engine_create(&cfg, &e) != MM_OK)
        return NULL;
    if (mm_engine_load(e) != MM_OK) {
        mm_engine_destroy(e);
        return NULL;
    }
    return e;
}

/* One prefill round only (submit + one step); the continuation token and
 * the post-round q buffer (the last full layer's rotated query, in-place)
 * are the observables. */
static uint32_t prefill_step(mm_engine *e, uint16_t **q_out)
{
    const uint32_t prompt[4] = { 5, 9, 17, 42 };
    mm_status s;

    if (mm_sched_submit(&e->sched, prompt, 4, 4) != MM_OK)
        return 0;
    s = mm_engine_step(e);
    if (s != MM_OK)
        return 0;
    *q_out = (uint16_t *)(uintptr_t)e->ab.q;
    return e->last_tok[0];
}

static int buf_differs(const uint16_t *a, const uint16_t *b, size_t n)
{
    size_t i;
    for (i = 0; i < n; i++)
        if (a[i] != b[i])
            return 1;
    return 0;
}

static void test_yarn_e2e(void)
{
    mm_engine *ep, *ey, *ep2;
    uint16_t *qp, *qy, *qp2;
    size_t qn;
    uint32_t tokp = 0, tokp2 = 0;

    /* The property under test: the engine's RoPE table reaches the
     * forward pass. After one prefill round the last full layer's q
     * buffer holds the rotated query, in place: with the YaRN table it
     * MUST differ from the plain table's, and plain vs plain must stay
     * byte-identical (determinism, unchanged golden path). (The fake
     * model's 512-vocab greedy argmax need not flip in 17 rounds, so the
     * stream itself is asserted only for the plain path.) */
    ep = boot(0, 0);
    CHECK(ep != NULL, "plain engine boots");
    ey = boot(1, 0);
    CHECK(ey != NULL, "yarn engine boots");
    if (ep && ey)
        qn = (size_t)4 * ep->mc.q_heads * ep->mc.head_dim;

    if (ep) {
        CHECK(ep->rope.kind == MM_ROPE_PLAIN, "plain table on engine");
        CHECK(ep->rope.mscale == 1.0f, "plain mscale 1.0");
        tokp = prefill_step(ep, &qp);
    }
    if (ey) {
        CHECK(ey->rope.kind == MM_ROPE_YARN, "yarn table on engine");
        CHECK(ey->rope.mscale == (float)(0.1 * log(4.0) + 1.0),
              "yarn mscale (engine)");
        prefill_step(ey, &qy);
        CHECK(buf_differs(qp, qy, qn),
              "yarn q differs from plain (the table reaches the kernels)");
        mm_engine_destroy(ey);
    }
    /* Determinism: a second plain round is byte-identical to the first. */
    ep2 = boot(0, 0);
    CHECK(ep2 != NULL, "second plain engine boots");
    if (ep2) {
        tokp2 = prefill_step(ep2, &qp2);
        CHECK(tokp2 == tokp, "plain runs same continuation token");
        CHECK(!buf_differs(qp, qp2, qn),
              "plain q buffers byte-identical (golden path unchanged)");
        mm_engine_destroy(ep2);
    }
    if (ep)
        mm_engine_destroy(ep);
}

static void test_spec_refused(void)
{
    mm_engine_cfg cfg;
    mm_engine *e = NULL;
    mm_status s;

    /* mtp backend. */
    mm_engine_cfg_default(&cfg);
    cfg.max_ctx = 128;
    cfg.verbose = 0;
    cfg.no_graph = 1;
    cfg.spec = 1;
    cfg.draft = 4;
    s = mm_engine_create(&cfg, &e);
    CHECK(s == MM_ERR_UNSUPPORTED, "--spec mtp refused at start");

    /* dflash2 backend. */
    cfg.spec = 2;
    s = mm_engine_create(&cfg, &e);
    CHECK(s == MM_ERR_UNSUPPORTED, "--spec dflash2 refused at start");
    CHECK(e == NULL, "no engine handle on refusal");
}

static void test_vision_hook(void)
{
    mm_engine *e;
    char px[64];
    mm_status s;

    memset(px, 0xAB, sizeof px);
    e = boot(0, 0);
    CHECK(e != NULL, "engine (vision off) boots");
    if (e) {
        s = mm_engine_submit_image(e, px, sizeof px);
        CHECK(s == MM_ERR_STATE, "vision off: STATE, not silent success");
        s = mm_engine_submit_image(e, px, 0);
        CHECK(s == MM_ERR_STATE, "empty payload refused");
        s = mm_engine_submit_image(NULL, px, sizeof px);
        CHECK(s == MM_ERR_STATE, "NULL engine refused");
        mm_engine_destroy(e);
    }
    e = boot(0, 1);
    CHECK(e != NULL, "engine (vision on) boots");
    if (e) {
        s = mm_engine_submit_image(e, px, sizeof px);
        CHECK(s == MM_ERR_UNSUPPORTED,
              "vision on: explicit unsupported (planned work)");
        mm_engine_destroy(e);
    }
}

static void test_profiles(void)
{
    mm_engine_cfg cfg;
    mm_engine *e = NULL;
    mm_status s;

    /* The selected profile is pinned on the engine. */
    mm_engine_cfg_default(&cfg);
    cfg.max_ctx = 32768;
    cfg.verbose = 0;
    cfg.no_graph = 1;
    cfg.profile_id = 2;              /* neroued */
    s = mm_engine_create(&cfg, &e);
    CHECK(s == MM_OK, "profile neroued accepted at ceiling");
    if (e) {
        CHECK(e->profile != NULL, "profile pinned");
        CHECK(e->profile && !strcmp(e->profile->name, "neroued"),
              "neroued identity");
        CHECK(e->profile && !strcmp(e->profile->repo,
                                    "Neroued/Qwen3.8-27B-nvfp4-NInfer"),
              "neroued checkpoint identity");
        mm_engine_destroy(e);
    }

    /* A context above the checkpoint ceiling is a hard error. */
    cfg.profile_id = 1;              /* quasar: ceiling 262144 */
    cfg.max_ctx = 300000;            /* > 262144, <= MM_MAX_SEQ */
    s = mm_engine_create(&cfg, &e);
    CHECK(s == MM_ERR_RANGE, "max_ctx above the profile ceiling refused");
    CHECK(e == NULL, "no engine handle on refusal");
}

int main(void)
{
    test_yarn_e2e();
    test_spec_refused();
    test_vision_hook();
    test_profiles();

    if (g_fail) {
        fprintf(stderr, "ENGINE FEATURES FAILED\n");
        return 1;
    }
    printf("ENGINE FEATURES PASSED\n");
    return 0;
}
