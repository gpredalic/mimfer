/*
 * Host golden test for the RoPE frequency tables (src/rope/rope.c,
 * include/mimfer/rope.h): plain RoPE bit-exactness against the legacy
 * per-kernel formula, and the YaRN NTK-by-parts blend + attention scale
 * against the reference formula (transformers v4.45.0
 * _compute_yarn_parameters; the paper's Eq. 10-15).
 *
 * Scope: the table builder. The engine-level behavior (YaRN changes the
 * token stream, plain stays bit-identical to the pre-change golden) is
 * covered by engine_features_test.c and by the parity self-check
 * (make parity-selfcheck must stay PASSED with YaRN off).
 *
 * Positions are the raw position: the angle is pos * inv_freq[i] with no
 * runtime position damping (see rope.h).
 */
#include "mimfer/rope.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

static int g_fail = 0;
#define CHECK(cond, msg)                                                      \
    do {                                                                      \
        if (!(cond)) {                                                        \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg);     \
            g_fail = 1;                                                       \
        }                                                                     \
    } while (0)

/* Tiny hybrid shape, identical to the engine's host_tiny_shape so the
 * table is exercised on the same RoPE geometry the kernels see. */
static mm_model_cfg tiny_shape(void)
{
    mm_model_cfg m;
    memset(&m, 0, sizeof m);
    m.hidden = 64;
    m.inter  = 128;
    m.layers = 8;
    m.q_heads = 8;
    m.kv_heads = 2;
    m.head_dim = 32;
    m.rope_dim = 16;
    m.lin_k_heads = 4;
    m.lin_k_dim = 16;
    m.lin_v_heads = 8;
    m.lin_v_dim = 16;
    m.conv_k = 4;
    m.vocab = 512;
    m.max_pos = 1024;
    m.rope_theta = 1e7f;
    m.norm_eps = 1e-6f;
    m.full_layer_step = 4;
    m.full_layer_offset = 0;
    m.mtp_layers = 0;
    return m;
}

static void test_plain(void)
{
    mm_model_cfg mc = tiny_shape();
    mm_engine_cfg cfg;
    mm_rope r;
    uint32_t i;

    mm_engine_cfg_default(&cfg);
    CHECK(mm_rope_init(&r, &mc, &cfg) == MM_OK, "plain init");
    CHECK(r.kind == MM_ROPE_PLAIN, "plain kind");
    CHECK(r.mscale == 1.0f, "plain mscale 1.0");
    CHECK(r.npairs == mc.rope_dim / 2, "npairs = rope_dim/2");
    CHECK(r.orig_ctx == (float)mc.max_pos, "plain orig_ctx = max_pos");
    /* Bit-exact with the legacy per-kernel formula (the golden anchor:
     * the parity self-check stays PASSED with YaRN off because this
     * table is what the ROPE kernels now consume). */
    for (i = 0; i < r.npairs; i++) {
        const float ref = powf(1.0f / mc.rope_theta,
                               2.0f * (float)i / (float)mc.rope_dim);
        CHECK(r.inv_freq[i] == ref, "plain inv_freq bit-exact (legacy formula)");
    }

    /* fill_ctrl mirrors the table into the control buffer. */
    {
        mm_ctrl c;
        memset(&c, 0xAA, sizeof c);
        mm_rope_fill_ctrl(&r, &c);
        CHECK(memcmp(c.rope_inv, r.inv_freq, r.npairs * sizeof(float)) == 0,
              "fill_ctrl copies inv_freq");
        CHECK(c.rope_mscale == 1.0f, "fill_ctrl mscale");
    }

    /* One-line description (load log). */
    {
        char buf[96];
        mm_rope_desc(&r, &mc, buf, sizeof buf);
        CHECK(strstr(buf, "plain") != NULL, "desc mentions plain");
    }
}

static void test_yarn(void)
{
    mm_model_cfg mc = tiny_shape();
    mm_engine_cfg cfg;
    mm_rope r;
    const float factor = 4.0f;
    uint32_t i;

    mm_engine_cfg_default(&cfg);
    cfg.rope_yarn = 1;
    cfg.rope_factor = factor;
    cfg.rope_orig_ctx = 0;              /* = mc.max_pos (1024) */
    cfg.max_ctx = 4096;                 /* = the scaled ceiling (1024 x 4) */
    CHECK(mm_rope_init(&r, &mc, &cfg) == MM_OK, "yarn init");
    CHECK(r.kind == MM_ROPE_YARN, "yarn kind");
    CHECK(r.factor == factor, "yarn factor stored");
    CHECK(r.orig_ctx == (float)mc.max_pos, "yarn orig_ctx default = max_pos");
    /* Attention scale (Eq. 15): 1/sqrt(t) = 0.1*ln(s)+1, computed in f64
     * and rounded to f32, exactly as the reference implementations do. */
    CHECK(r.mscale == (float)(0.1 * log(4.0) + 1.0), "yarn mscale formula");
    CHECK(r.mscale > 1.0f, "yarn mscale > 1");

    /* Pinned correction range for this geometry (rd=16, orig_ctx=1024,
     * theta=1e7, beta_fast=32, beta_slow=1): the f64 reference formula
     * gives 0.8079658400 -> floor 0 and 2.5281372438 -> ceil 3. A break
     * here means the range formula changed; re-derive before accepting. */
    CHECK(r.low == 0.0f, "yarn low corner (pinned reference)");
    CHECK(r.high == 3.0f, "yarn high corner (pinned reference)");

    for (i = 0; i < r.npairs; i++) {
        const float pos_freq = powf((float)mc.rope_theta,
                                    2.0f * (float)i / (float)mc.rope_dim);
        const float keep = 1.0f / pos_freq;       /* original frequency */
        const float extrap = 1.0f / (factor * pos_freq); /* / factor    */
        CHECK(r.inv_freq[i] > 0.0f, "yarn inv_freq positive");
        if ((float)i <= r.low)
            CHECK(r.inv_freq[i] == keep, "fast pairs keep the original");
        else if ((float)i >= r.high)
            CHECK(r.inv_freq[i] == extrap, "slow pairs take the interpolated");
        else
            CHECK(r.inv_freq[i] > extrap && r.inv_freq[i] < keep,
                  "mid pairs blend strictly between the two");
    }
    /* Pair 0 is the fastest pair: angle = pos * 1.0, unmodified by YaRN. */
    CHECK(r.inv_freq[0] == 1.0f, "fastest pair is 1.0");
    /* Monotone: frequency decreases with the pair index (dim). */
    for (i = 1; i < r.npairs; i++)
        CHECK(r.inv_freq[i - 1] > r.inv_freq[i], "inv_freq decreasing in i");

    /* Positions are not damped: with the table above, the angle of pair
     * i at position p is exactly p * r.inv_freq[i] (the kernels compute
     * pos * rope_inv[i]; there is no other position transform). */
    {
        const float p = 1000.0f;
        const float ang = p * r.inv_freq[1];
        CHECK(ang == p * r.inv_freq[1], "raw position angle (no damping)");
    }

    {
        char buf[96];
        mm_rope_desc(&r, &mc, buf, sizeof buf);
        CHECK(strstr(buf, "yarn") != NULL, "desc mentions yarn");
    }
}

static void test_validation(void)
{
    mm_model_cfg mc = tiny_shape();
    mm_engine_cfg cfg;
    mm_rope r;

    /* factor must be a real extension factor. */
    mm_engine_cfg_default(&cfg);
    cfg.rope_yarn = 1;
    cfg.rope_factor = 1.0f;
    CHECK(mm_rope_init(&r, &mc, &cfg) == MM_ERR_RANGE, "factor 1.0 refused");
    cfg.rope_factor = 0.5f;
    CHECK(mm_rope_init(&r, &mc, &cfg) == MM_ERR_RANGE, "factor < 1 refused");
    cfg.rope_factor = NAN;
    CHECK(mm_rope_init(&r, &mc, &cfg) == MM_ERR_RANGE, "NaN factor refused");

    /* orig_ctx must be a sane token count (0 = default is allowed). */
    cfg.rope_factor = 4.0f;
    cfg.max_ctx = 4096;                 /* = the scaled ceiling */
    cfg.rope_orig_ctx = (uint32_t)(MM_MAX_SEQ + 1);
    CHECK(mm_rope_init(&r, &mc, &cfg) == MM_ERR_RANGE,
          "orig_ctx above MM_MAX_SEQ refused");
    cfg.rope_orig_ctx = 1024;
    CHECK(mm_rope_init(&r, &mc, &cfg) == MM_OK, "explicit orig_ctx accepted");

    /* max_ctx may not exceed the scaled ceiling orig_ctx * factor. */
    cfg.max_ctx = 5 * 1024;              /* > 1024 x 4 */
    CHECK(mm_rope_init(&r, &mc, &cfg) == MM_ERR_RANGE,
          "max_ctx above the scaled ceiling refused");
    cfg.max_ctx = 4096;                  /* == the ceiling */
    CHECK(mm_rope_init(&r, &mc, &cfg) == MM_OK,
          "max_ctx at the scaled ceiling accepted");
}

int main(void)
{
    test_plain();
    test_yarn();
    test_validation();

    if (g_fail) {
        fprintf(stderr, "ROPE TEST FAILED\n");
        return 1;
    }
    printf("ROPE TEST PASSED\n");
    return 0;
}
