/*
 * Tensor registry (implementation of the mm_tensreg API declared in
 * engine.h) plus the host-side model generator used by the CPU reference
 * build and its tests.
 *
 * mm_tens_add / mm_tens_find are the ONLY registry primitives the engine
 * and the plan builder use: a bounded linear table of (name, view). No
 * hashmap, no tree, no allocation per tensor — the table is fixed
 * (MM_MAX_TENSORS) and lookup is O(n), which is the whole "Qwen-only
 * scale" argument: a few hundred names at most, resolved once at load.
 *
 * Real artifact loading: mm_art_model_parse() reads the ART_MODEL section
 * (fixed layout, documented below) into mm_model_cfg; mm_model_weights_load()
 * validates the ART_WEIGHTS section's internal tensor index against the
 * model shape and stages every tensor's bytes into the weight arena.
 * mm_model_fill_fake() remains the stand-in for the golden path (no
 * --artifact): deterministic bf16 values so parity tests run without an
 * artifact and repeated runs are byte-identical.
 */

#ifdef __cplusplus
extern "C" {
#endif

#ifndef MM_MODEL_TENSOR_REGISTRY_H
#define MM_MODEL_TENSOR_REGISTRY_H

#include "mimfer/engine.h"

/* Total bytes of the standard model tensor set at the given shape:
 * embed + lm_head + final norm, plus per-layer (input/post norms, MLP,
 * and either the full-attention or the linear-attention parameter set).
 * Used to size the weight arena before the model is filled. Returns 0 for
 * a cfg that would not finalize. */
size_t mm_model_weights_bytes(const mm_model_cfg *mc);

/* Populate e->w_arena + e->reg with the full standard tensor set (bf16).
 * Weights are small-deterministic-random, norm gains are 1.0, conv biases
 * 0.0 and A_log a fixed small positive (decay). Deterministic in `seed`:
 * the same seed fills the same weights. Requires e->mc finalized and
 * e->w_arena initialized. Golden path only (the engine uses
 * mm_model_weights_load() when --artifact is given). */
mm_status mm_model_fill_fake(mm_engine *e, uint64_t seed);

/* -------------------------------------------------- canonical tensor set */
/*
 * The standard tensor set enumerated by (name, rows, cols) in load order:
 * embed + lm_head + final norm, then per layer the input/post norms, MLP,
 * and either the full-attention or the linear-attention parameter set.
 * Both the arena sizing and the loaders agree on this order. The cfg must
 * be finalized (the full/lin split comes from the stride pattern).
 */

/* Number of tensors in the set (0 for a cfg that would not finalize). */
size_t mm_model_spec_count(const mm_model_cfg *mc);

/* The i-th tensor (0-based, load order). name_sz must be >= 48 (the
 * registry name size). MM_ERR_RANGE if i is out of range or the cfg
 * would not finalize; names are copied NUL-terminated. */
mm_status mm_model_spec_at(const mm_model_cfg *mc, size_t i, char *name,
                           size_t name_sz, uint32_t *rows, uint32_t *cols);

/* -------------------------------------------------- ART_MODEL section */
/*
 * The ART_MODEL payload (little-endian, artifact.h "fixed layout"):
 *
 *   u32  version            (1)
 *   u32  flags              (0)
 *   u32  hidden, inter, layers
 *   u32  q_heads, kv_heads, head_dim, rope_dim
 *   u32  lin_k_heads, lin_k_dim, lin_v_heads, lin_v_dim
 *   u32  conv_k, vocab, max_pos
 *   f32  rope_theta, norm_eps
 *   u32  full_layer_step, full_layer_offset, mtp_layers
 *   [reserved extras: version 1 carries none]
 *
 * = 84 bytes; a longer payload carries reserved extra bytes, ignored by
 * v1 readers. Parses into `out` (input fields only; the caller runs
 * mm_model_cfg_finalize for the derived fields + range gates).
 */
#define MM_ART_MODEL_VERSION  1
#define MM_ART_MODEL_FIXED_SZ 84u
mm_status mm_art_model_parse(const uint8_t *data, size_t len,
                             mm_model_cfg *out);

/* -------------------------------------------------- ART_WEIGHTS section */
/*
 * The ART_WEIGHTS payload (little-endian) starts with the sub-checksum
 * block when the section carries one (n_subck u64 FNV values, see
 * mm_art_verify_section) -- the engine strips it before calling
 * mm_model_weights_load. The internal tensor index:
 *
 *   u32  version            (1)
 *   u32  flags              (0)
 *   u32  n_tensors
 *   entries (n_tensors):
 *     u16  name_len         (1..48, NUL-terminated)
 *     u8   name[name_len]
 *     u32  rows, cols
 *     u8   dtype            (MM_DT_BF16; fp4 -> MM_ERR_UNSUPPORTED)
 *     u8   pad[3]
 *     u64  offset           (payload-relative, after the index)
 *     u64  nbytes           (== rows * cols * 2 for bf16)
 *   packed tensor bytes
 *
 * Validation gates (hard fail, no partial state):
 *   header version/flags              -> MM_ERR_CORRUPT
 *   n_tensors != canonical count      -> MM_ERR_SHAPE  (count gate)
 *   truncated entry / bad name /      -> MM_ERR_CORRUPT
 *   nbytes != rows*cols*2 / offsets
 *   outside the payload               -> MM_ERR_CORRUPT
 *   dtype not bf16 (fp4)              -> MM_ERR_UNSUPPORTED
 *   entry i != canonical spec i       -> MM_ERR_SHAPE
 *   (name + rows + cols, in order -- the metadata/runtime consistency
 *    gate: the weights must describe exactly the model section's shape)
 * Then every tensor is staged into e->w_arena (256-aligned, via the host
 * mirror) and registered as a bf16 view. Requires e->mc finalized and
 * e->w_arena initialized with capacity >= weights + pools. Section
 * checksums (payload ck + sub-ck) are gated by mm_art_verify at create.
 */
#define MM_ART_WEIGHTS_VERSION 1
mm_status mm_model_weights_load(mm_engine *e, const uint8_t *data, size_t len);

#endif /* MM_MODEL_TENSOR_REGISTRY_H */
#ifdef __cplusplus
}
#endif
