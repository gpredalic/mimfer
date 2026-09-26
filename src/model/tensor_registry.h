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
 * mm_model_fill_fake() is a stand-in for artifact weight loading. In the
 * CPU reference build there is no .mimfer artifact; the engine needs a
 * populated weight arena + registry so the plan builder and kernels can
 * run end-to-end. It registers every tensor of the standard model with
 * deterministic bf16 values so repeated runs are byte-identical.
 */
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
 * e->w_arena initialized. Host/test support only. */
mm_status mm_model_fill_fake(mm_engine *e, uint64_t seed);

#endif /* MM_MODEL_TENSOR_REGISTRY_H */
