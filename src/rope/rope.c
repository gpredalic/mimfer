/*
 * RoPE frequency tables: plain and YaRN (see include/mimfer/rope.h).
 *
 * Plain: the legacy per-kernel formula, bit-exact (same f32 powf
 * expression the CPU/CUDA kernels used to run per call).
 *
 * YaRN: the correction range is computed in f64 exactly as the reference
 * implementations do (transformers v4.45.0 _yarn_find_correction_range /
 * _yarn_find_correction_dim; the paper's Eq. 11-12 with alpha=1, beta=32);
 * the per-pair blend and the attention scale are then applied in f32.
 * (Host f32 powf/sinf/cosf vs device libm differ by <= 2 f32 ulp — the
 * existing parity gate (1 bf16 ulp) covers it; the blend itself is input,
 * not math, so it is bit-identical across CPU and CUDA.)
 */
#include "mimfer/rope.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

/* Two full rotations (beta_fast) and one (beta_slow) of the base wave,
 * the reference defaults (paper: alpha = 1, beta = 32). */
#define YARN_BETA_FAST 32.0
#define YARN_BETA_SLOW 1.0
#define MM_PI_D 3.14159265358979323846

/* dim * log(max_pos / (2*pi*num_rotations)) / (2*log(theta)) — the
 * reference's correction-dim formula, in f64. */
static double yarn_corr_dim(double dim, double max_pos, double base,
                            double num_rotations)
{
    return dim * log(max_pos / (2.0 * MM_PI_D * num_rotations)) /
           (2.0 * log(base));
}

mm_status mm_rope_init(mm_rope *r, const mm_model_cfg *mc,
                       const mm_engine_cfg *cfg)
{
    const uint32_t rd = mc->rope_dim;
    const uint32_t np = rd / 2;
    const float inv_theta = 1.0f / mc->rope_theta;

    MM_REQUIRE(mc && r && cfg, MM_ERR_STATE);
    MM_REQUIRE(rd >= 2 && rd % 2 == 0 && np <= MM_MAX_ROPE_PAIRS,
               MM_ERR_SHAPE);
    memset(r, 0, sizeof *r);
    r->npairs = np;

    if (!cfg->rope_yarn) {
        uint32_t i;
        r->kind = MM_ROPE_PLAIN;
        r->factor = 1.0f;
        r->orig_ctx = (float)mc->max_pos;
        r->mscale = 1.0f;
        for (i = 0; i < np; i++)
            r->inv_freq[i] = powf(inv_theta, 2.0f * (float)i / (float)rd);
        return MM_OK;
    }

    /* ------------------------------------------------------- YaRN */
    {
        const float factor = cfg->rope_factor;
        const uint32_t orig = cfg->rope_orig_ctx ? cfg->rope_orig_ctx
                                                 : mc->max_pos;
        double dlow, dhigh;
        uint32_t i;

        MM_REQUIRE(factor > 1.0f && factor < 1e6f && factor == factor,
                   MM_ERR_RANGE);   /* NaN == NaN is false */
        MM_REQUIRE(orig >= 1 && orig <= MM_MAX_SEQ, MM_ERR_RANGE);
        if ((double)cfg->max_ctx > (double)orig * (double)factor) {
            MM_LOGE("rope: yarn: max_ctx %u exceeds the scaled ceiling "
                    "(orig_ctx %u x factor %g)",
                    cfg->max_ctx, orig, (double)factor);
            return MM_ERR_RANGE;
        }

        /* Correction range (f64, reference semantics). */
        dlow = yarn_corr_dim((double)rd, (double)orig,
                             (double)mc->rope_theta, YARN_BETA_FAST);
        dhigh = yarn_corr_dim((double)rd, (double)orig,
                              (double)mc->rope_theta, YARN_BETA_SLOW);
        {
            int64_t low = (int64_t)floor(dlow), high = (int64_t)ceil(dhigh);
            if (low < 0) low = 0;
            if (high > (int64_t)rd - 1) high = rd - 1;
            if (low > high) low = high;
            r->low = (float)low;
            r->high = (float)high;
        }

        r->kind = MM_ROPE_YARN;
        r->factor = factor;
        r->orig_ctx = (float)orig;
        /* Eq. 15: 1/sqrt(t) = 0.1*ln(s) + 1. */
        r->mscale = (float)(0.1 * log((double)factor) + 1.0);

        for (i = 0; i < np; i++) {
            /* Base wave of pair i and its two variants, f32. */
            const float pos_freq =
                powf((float)mc->rope_theta, 2.0f * (float)i / (float)rd);
            const float inv_extrap = 1.0f / pos_freq;          /* keep */
            const float inv_interp = 1.0f / (factor * pos_freq); /* / s */
            float ramp = ((float)i - r->low) / (r->high - r->low);
            if (ramp < 0.0f)
                ramp = 0.0f;
            if (ramp > 1.0f)
                ramp = 1.0f;
            if (r->high - r->low < 0.001f)
                ramp = ((float)i - r->low) > 0.0f ? 1.0f : 0.0f;
            /* Low (fast) pairs keep the original frequency; high (slow)
             * pairs take the interpolated one; the ramp blends. */
            r->inv_freq[i] = inv_interp * ramp + inv_extrap * (1.0f - ramp);
        }
    }
    return MM_OK;
}

void mm_rope_fill_ctrl(const mm_rope *r, mm_ctrl *c)
{
    memcpy(c->rope_inv, r->inv_freq, (size_t)r->npairs * sizeof(float));
    c->rope_mscale = r->mscale;
}

void mm_rope_desc(const mm_rope *r, const mm_model_cfg *mc, char *buf,
                  size_t n)
{
    if (r->kind == MM_ROPE_PLAIN)
        snprintf(buf, n, "plain (theta=%g rope_dim=%u)",
                 (double)mc->rope_theta, mc->rope_dim);
    else
        snprintf(buf, n, "yarn (s=%g orig_ctx=%d low=%d high=%d mscale=%.4f)",
                 (double)r->factor, (int)r->orig_ctx, (int)r->low,
                 (int)r->high, (double)r->mscale);
}
