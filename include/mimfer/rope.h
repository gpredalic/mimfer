/*
 * RoPE frequency table: plain RoPE and YaRN.
 *
 * YaRN (Peng et al., "YaRN: Efficient Context Window Extension of Large
 * Language Models", ICLR 2024, Eq. 10-15). Reference implementations
 * (Hugging Face transformers v4.45.0 modeling_rope_utils
 * _compute_yarn_parameters, jquesnelle/yarn LlamaYaRNScaledRotaryEmbedding,
 * vLLM YaRNScalingRotaryEmbedding) all agree on the form this module
 * implements:
 *
 *   - NTK-by-parts: each inverse frequency is a blend between the original
 *     (fast / low pair dims keep it) and the interpolated one (slow / high
 *     pair dims use inv_freq / s); the middle dims ramp linearly between
 *     the two over the correction range [low, high] derived from the
 *     original context, theta and the number of full rotations.
 *   - attention scale: the rotated q/k components are multiplied by
 *     mscale = 0.1*ln(s) + 1 (the paper's 1/sqrt(t), t the attention
 *     temperature). It is folded into the ROPE kernel, so the attention
 *     kernels and the KV layout are unchanged.
 *   - positions are NOT damped: the angle is the raw position times the
 *     blended inverse frequency. No runtime position rescaling exists in
 *     the paper or any reference implementation.
 *
 * The table (effective inverse frequencies + mscale) is computed once at
 * engine load and passed to the ROPE kernels through the control buffer
 * (kernels.h mm_ctrl.rope_inv / rope_mscale): the kernels never recompute
 * it, so the CPU and CUDA paths consume bit-identical inputs. The plain
 * path is bit-exact with the legacy per-kernel formula
 * powf(1/theta, 2i/rope_dim) and mscale == 1.0, so the golden reference
 * is unchanged when YaRN is off.
 */
#ifndef MIMFER_ROPE_H
#define MIMFER_ROPE_H

#include "mimfer/config.h"
#include "mimfer/kernels.h"

typedef enum mm_rope_kind {
    MM_ROPE_PLAIN = 0,
    MM_ROPE_YARN  = 1
} mm_rope_kind;

typedef struct mm_rope {
    mm_rope_kind kind;
    float    factor;    /* 1.0 plain; scale factor s (> 1.0) yarn          */
    float    orig_ctx;  /* original max context (yarn)                     */
    float    low;       /* correction range, low corner (yarn)            */
    float    high;      /* correction range, high corner (yarn)           */
    float    mscale;    /* 1.0 plain; 0.1*ln(s)+1 yarn                     */
    float    inv_freq[MM_MAX_ROPE_PAIRS]; /* npairs effective inv freqs    */
    uint32_t npairs;    /* rope_dim / 2                                    */
} mm_rope;

/* Build the effective table for the model's RoPE shape under the engine
 * config (plain, or YaRN when cfg->rope_yarn). Validates the YaRN
 * parameters: factor > 1.0 and finite, orig_ctx sane (0 = model max_pos),
 * and max_ctx within the scaled ceiling orig_ctx * factor. */
mm_status mm_rope_init(mm_rope *r, const mm_model_cfg *mc,
                       const mm_engine_cfg *cfg);

/* Copy the table into the control buffer (after every round's ctrl
 * reset; the engine does this in its step functions). */
void mm_rope_fill_ctrl(const mm_rope *r, mm_ctrl *c);

/* One-line description for the load log; n must be >= 96. */
void mm_rope_desc(const mm_rope *r, const mm_model_cfg *mc, char *buf,
                  size_t n);

#endif /* MIMFER_ROPE_H */
