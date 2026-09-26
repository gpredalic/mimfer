/*
 * Sampling policy + deterministic RNG.
 *
 * One splitmix64 stream, one 64-bit draw per token. The kernel maps the
 * draw to a fixed-point uniform via the top 32 bits (see sampling.h).
 * Host and GPU builds use this exact stream, which is what makes the
 * determinism contract testable without a GPU.
 */
#include "mimfer/sampling.h"

void mm_rng_seed(mm_rng *r, uint64_t seed)
{
    r->state = seed;
    r->draws = 0;
    /* Mix once so that seed 0 does not start the stream at a fixed point. */
    r->state = mm_splitmix64(&r->state);
}

uint64_t mm_rng_next(mm_rng *r)
{
    r->draws++;
    return mm_splitmix64(&r->state);
}

mm_status mm_policy_check(float temperature, int top_k, float top_p)
{
    if (temperature < 0.0f)
        return MM_ERR_RANGE;
    if (top_k < 0)
        return MM_ERR_RANGE;
    if (!(top_p > 0.0f) || top_p > 1.0f)
        return MM_ERR_RANGE;
    /* Greedy is a closed policy: no partial sampling knobs. */
    if (temperature == 0.0f && (top_k != 0 || top_p != 1.0f))
        return MM_ERR_RANGE;
    /* Non-greedy with top_k == 0 and top_p == 1.0 is pure temperature —
     * valid, just unusual. */
    return MM_OK;
}
