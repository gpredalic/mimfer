/*
 * Sampling policy and the deterministic RNG.
 *
 * Policy: greedy, or temperature + top-k + top-p (standard chain, applied
 * in the order temperature -> top-k -> top-p). The host fills the control
 * buffer's sampling fields each round; the SAMPLE kernel does the rest on
 * device so the logits never round-trip to the host.
 *
 * Randomness: one splitmix64 stream seeded from cfg.seed. Each token step
 * consumes exactly one 64-bit draw U; the kernel maps it to a uniform in
 * [0,1) as U_hi / 2^32. Determinism contract: same seed + same prompt
 * sequence => byte-identical token stream, on the GPU build AND the CPU
 * reference build (the test suite enforces agreement).
 *
 * Speculative verification (phase 2) reuses the same stream: the verify
 * pass samples from the RESIDUAL distribution p_t - p_d using the SAME U,
 * which is what keeps accepted/rejected sequences identical to
 * non-speculative sampling.
 */
#ifndef MIMFER_SAMPLING_H
#define MIMFER_SAMPLING_H

#include "mimfer.h"

typedef struct mm_rng {
    uint64_t state;
    uint64_t draws;      /* count, for telemetry                          */
} mm_rng;

void     mm_rng_seed(mm_rng *r, uint64_t seed);
uint64_t mm_rng_next(mm_rng *r);
/* Map a draw to a fixed-point uniform in [0,1): hi 32 bits / 2^32. */
static inline uint64_t mm_rng_fixed32(uint64_t u)
{
    return u >> 32;
}

/* Validate a policy: temperature >= 0, top_k > 0, 0 < top_p <= 1.
 * Greedy is temperature == 0 and top_k == 0 and top_p == 1. */
mm_status mm_policy_check(float temperature, int top_k, float top_p);

#endif /* MIMFER_SAMPLING_H */
