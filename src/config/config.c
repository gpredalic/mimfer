/*
 * Configuration: target device profile, Qwen3.8-27B shape, engine knobs.
 *
 * Device validation policy: the fields that break the model if wrong
 * (name, compute capability, VRAM, bandwidth) are hard checks; the fields
 * that only change tuning (SM count, L2, shared memory) are probed and
 * logged, never fatal. See docs/architecture.md "Device gate".
 */
#include "mimfer/config.h"
#include "mimfer/cuda_rt.h"
#include "mimfer/sampling.h"
#include <stdio.h>
#include <string.h>
#include <strings.h>

#if defined(MM_WITH_CUDA)
#include <cuda_runtime.h>
#include <cuda.h>   /* driver API: cuDeviceGetAttribute (memory clock rate) */
#endif

/* ------------------------------------------------------ device profile */
/*
 * Nominal profile for the RTX PRO 4000 Blackwell. Datasheet values that
 * were not independently confirmed are marked ESTIMATE; the runtime check
 * (mm_device_check) treats them as soft bounds and the probed values are
 * logged at startup so a real run documents the actual silicon.
 */
const mm_dev_profile MM_TARGET_PRO4000 = {
    .name              = "RTX PRO 4000",
    .cc_major          = 12,      /* Blackwell, sm_120x */
    .cc_minor          = 0,       /* accept 12.0 or 12.1 */
    .sm_count          = 0,       /* ESTIMATE: take the probed value */
    .vram_bytes        = 24ull * 1024 * 1024 * 1024,
    .l2_bytes          = 96ull * 1024 * 1024,  /* ESTIMATE */
    .smem_per_sm       = 100ull * 1024,        /* ESTIMATE */
    .bw_gbps           = 672,      /* ESTIMATE: 192-bit GDDR7 @ 28 Gbps */
    .max_threads_per_sm = 1536,
    .regs_per_sm       = 65536,
};

mm_status mm_device_probe(int device_ordinal, mm_dev_info *out)
{
#if defined(MM_WITH_CUDA)
    /* 'struct' tag, not the bare name: CUDA 13 removed the plain
     * `cudaDeviceProp` typedef. The struct tag is stable and identical
     * across 12.8 and 13.x, in C and C++, so this compiles on both. */
    struct cudaDeviceProp p;
    cudaError_t e = cudaGetDeviceProperties(&p, device_ordinal);
    if (e != cudaSuccess) {
        MM_LOGE("cudaGetDeviceProperties(%d): %s",
                device_ordinal, cudaGetErrorString(e));
        return MM_ERR_CUDA;
    }
    memset(out, 0, sizeof *out);
    /* p.name is 256 bytes, out->name 128: bound the source explicitly
     * (real device names are far shorter; this keeps the copy warning-
     * free at -O2 where -Wformat-truncation fires). */
    snprintf(out->name, sizeof out->name, "%.*s",
             (int)(sizeof out->name - 1), p.name);
    out->cc_major = p.major;
    out->cc_minor = p.minor;
    out->sm_count = p.multiProcessorCount;
    out->vram_bytes = (size_t)p.totalGlobalMem;
    out->l2_bytes = (size_t)p.l2CacheSize;
    out->smem_per_sm = (size_t)p.sharedMemPerMultiprocessor;
    out->max_threads_per_sm = p.maxThreadsPerMultiProcessor;
    out->regs_per_sm = p.regsPerMultiprocessor;
    {
        /* Derived bandwidth: SDR clock x bus width x 2 (DDR), GB/s.
         * The clock rate is no longer a cudaDeviceProp member (removed in
         * CUDA 13); query the driver attribute instead. Same value as the
         * old p.memoryClockRate field (kHz), on 12.8 and 13.x alike.
         * Link: -lcuda (NVCCFLAGS). */
        CUdevice dev;
        int clock_khz;
        if (cuInit(0) != CUDA_SUCCESS ||
            cuDeviceGet(&dev, device_ordinal) != CUDA_SUCCESS ||
            cuDeviceGetAttribute(&clock_khz,
                                 CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
                                 dev) != CUDA_SUCCESS) {
            MM_LOGE("cuDeviceGetAttribute(MEMORY_CLOCK_RATE, %d) failed",
                    device_ordinal);
            return MM_ERR_CUDA;
        }
        double hz = (double)clock_khz * 1000.0;
        double bw = hz * (double)p.memoryBusWidth / 8.0 * 2.0;
        out->bw_gbps = (size_t)(bw / 1e9);
    }
    return MM_OK;
#else
    (void)device_ordinal;
    (void)out;
    return MM_ERR_UNSUPPORTED;
#endif
}

mm_status mm_device_check(const mm_dev_info *d, const mm_dev_profile *p)
{
    MM_LOGI("device: %s (sm_%d%d, %d SMs, %zu GiB, %zu GB/s derived)",
            d->name, d->cc_major, d->cc_minor, d->sm_count,
            d->vram_bytes >> 30, d->bw_gbps);

    if (p->name[0] && !strstr(d->name, p->name)) {
        MM_LOGE("device gate: name \"%s\" does not contain \"%s\"",
                d->name, p->name);
        return MM_ERR_DEVICE;
    }
    if (d->cc_major != p->cc_major ||
        (p->cc_minor > 0 && d->cc_minor != p->cc_minor)) {
        MM_LOGE("device gate: compute capability %d.%d, need %d.%d",
                d->cc_major, d->cc_minor, p->cc_major, p->cc_minor);
        return MM_ERR_DEVICE;
    }
    /* VRAM: the driver-reported totalGlobalMem sits below the datasheet
     * nominal — the driver/firmware reserves part of the card (RTX PRO
     * 4000 24 GB: the driver reports 25,153,044,480 bytes = 23.4256 GiB,
     * 2.39% below nominal; measured 2026-09-27, driver 595.71.05).
     * Accept up to MM_VRAM_TOLERANCE_PCT below nominal (floor inclusive). */
    {
        size_t vram_floor =
            p->vram_bytes * (100 - MM_VRAM_TOLERANCE_PCT) / 100;
        if (d->vram_bytes < vram_floor) {
            MM_LOGE("device gate: %zu GiB < required %zu GiB "
                    "(%d%% tolerance, floor %zu GiB)",
                    d->vram_bytes >> 30, p->vram_bytes >> 30,
                    MM_VRAM_TOLERANCE_PCT, vram_floor >> 30);
            return MM_ERR_DEVICE;
        }
    }
    /* Bandwidth within 10% of nominal (derived value is a product of the
     * two driver-reported fields, so this mainly guards a wrong card). */
    if (p->bw_gbps > 0) {
        size_t lo = p->bw_gbps * 9 / 10, hi = p->bw_gbps * 11 / 10;
        if (d->bw_gbps < lo || d->bw_gbps > hi) {
            MM_LOGE("device gate: derived bandwidth %zu GB/s outside [%zu..%zu]",
                    d->bw_gbps, lo, hi);
            return MM_ERR_DEVICE;
        }
    }
    MM_LOGI("device: l2 %zu MiB, smem/SM %zu KiB (soft, logged only)",
            d->l2_bytes >> 20, d->smem_per_sm >> 10);
    return MM_OK;
}

/* Startup VRAM workload-fit gate: weights+pools (wcap) + activations
 * (acap) + pinned I/O (pin) must fit the reported total minus headroom.
 * Called in the CUDA create path before any device allocation; host
 * builds compile it too (pure arithmetic, unit-tested there). */
mm_status mm_vram_workload_fit(const mm_dev_info *info, size_t wcap,
                               size_t acap, size_t pin)
{
    size_t need, budget;

    if (!info || info->vram_bytes == 0) {
        MM_LOGE("vram fit: no device budget to check against");
        return MM_ERR_DEVICE;
    }
    need = wcap + acap + pin;
    budget = info->vram_bytes -
             info->vram_bytes * MM_VRAM_FIT_HEADROOM_PCT / 100;
    if (need > budget) {
        MM_LOGE("vram fit: workload %zu MiB (weights+pools %zu, activations "
                "%zu, pinned %zu) > budget %zu MiB (of %zu MiB total, %d%% "
                "headroom): the model does not fit this device",
                need >> 20, wcap >> 20, acap >> 20, pin >> 20,
                budget >> 20, info->vram_bytes >> 20,
                MM_VRAM_FIT_HEADROOM_PCT);
        return MM_ERR_NOMEM;
    }
    MM_LOGW("vram fit: ok (workload %zu MiB <= budget %zu MiB, of %zu MiB "
            "total)", need >> 20, budget >> 20, info->vram_bytes >> 20);
    return MM_OK;
}

/* ------------------------------------------------------------ model */

const mm_model_cfg MM_Q38_DEFAULT = {
    .hidden          = MM_Q38_HIDDEN,
    .inter           = MM_Q38_INTER,
    .layers          = MM_Q38_LAYERS,
    .q_heads         = MM_Q38_Q_HEADS,
    .kv_heads        = MM_Q38_KV_HEADS,
    .head_dim        = MM_Q38_HEAD_DIM,
    .rope_dim        = MM_Q38_ROPE_DIM,
    .lin_k_heads     = MM_Q38_LIN_K_HEADS,
    .lin_k_dim       = MM_Q38_LIN_K_DIM,
    .lin_v_heads     = MM_Q38_LIN_V_HEADS,
    .lin_v_dim       = MM_Q38_LIN_V_DIM,
    .conv_k          = MM_Q38_CONV_K,
    .vocab           = MM_Q38_VOCAB,
    .max_pos         = MM_Q38_MAX_POS,
    .rope_theta      = MM_Q38_ROPE_THETA,
    .norm_eps        = MM_Q38_NORM_EPS,
    .full_layer_step = 4,
    .full_layer_offset = 0,
    .mtp_layers      = MM_Q38_MTP_LAYERS,
};

/* Layer i (1-indexed) is full-attention when
 * (i - 1 - offset) % step == step - 1.
 * With step 4, offset 0 that is 4, 8, ..., 64: [lin,lin,lin,full] x 16. */
int mm_model_cfg_layer_is_full(const mm_model_cfg *c, uint32_t i)
{
    return ((i - 1 - c->full_layer_offset) % c->full_layer_step) ==
           c->full_layer_step - 1;
}

mm_status mm_model_cfg_finalize(mm_model_cfg *c)
{
    uint32_t i, n_full = 0;

    MM_REQUIRE(c->hidden > 0 && c->inter > 0 && c->layers > 0, MM_ERR_SHAPE);
    MM_REQUIRE(c->q_heads % c->kv_heads == 0, MM_ERR_SHAPE);
    MM_REQUIRE(c->head_dim % 2 == 0, MM_ERR_SHAPE);
    MM_REQUIRE(c->rope_dim <= c->head_dim && c->rope_dim % 2 == 0,
               MM_ERR_SHAPE);
    MM_REQUIRE(c->hidden % 16 == 0 && c->inter % 16 == 0, MM_ERR_SHAPE);
    MM_REQUIRE(c->full_layer_step >= 1 && c->full_layer_step <= 8,
               MM_ERR_SHAPE);
    MM_REQUIRE(c->full_layer_offset < c->full_layer_step, MM_ERR_SHAPE);
    MM_REQUIRE(c->vocab > 0, MM_ERR_SHAPE);
    MM_REQUIRE(c->max_pos <= MM_MAX_SEQ, MM_ERR_RANGE);
    MM_REQUIRE(c->lin_k_heads > 0 && c->lin_v_heads > 0, MM_ERR_SHAPE);
    MM_REQUIRE(c->lin_k_dim % 16 == 0 && c->lin_v_dim % 16 == 0,
               MM_ERR_SHAPE);

    for (i = 1; i <= c->layers; i++)
        if (mm_model_cfg_layer_is_full(c, i))
            n_full++;
    MM_REQUIRE(n_full > 0, MM_ERR_SHAPE);

    c->n_full_layers = n_full;
    c->n_lin_layers = c->layers - n_full;
    c->state_bytes_per_seq =
        (size_t)c->n_lin_layers * c->lin_v_heads * c->lin_k_dim *
        c->lin_v_dim * 4u;
    /* BF16 K and V, all full layers (the FP8 KV variant halves this at
     * pool-creation time). */
    c->kv_bytes_per_token =
        (size_t)c->n_full_layers * c->kv_heads * 2u * c->head_dim * 2u;
    return MM_OK;
}

/* ------------------------------------------------ weights profiles */
/*
 * The two validated reference checkpoints (README "Supported artifacts").
 * Same model shape (Qwen3.8-27B, NVFP4); the NInfer artifacts differ in
 * the quantization metadata, the MTP draft head and the DFlash2 draft
 * window they carry. max_ctx is the native (unextended) context the
 * artifact's RoPE was trained for: engine startup hard-fails above it,
 * which is what makes YaRN (rope.h) the only sanctioned extension path.
 */
const mm_profile MM_PROFILES[MM_N_PROFILES] = {
    [0] = {
        .id          = 1,
        .name        = "quasar",
        .repo        = "MirkoCovizzi/Qwen3.8-27B-QUASAR-NVFP4-NInfer",
        .quant       = "NVFP4 (NInfer Artifact V2/V3)",
        .max_ctx     = 262144,
        .def_ctx     = 32768,
        .def_chunk   = 2048,
        .mtp_layers  = 1,
        .dflash2_max = 8,
        .has_vision  = 1,
    },
    [1] = {
        .id          = 2,
        .name        = "neroued",
        .repo        = "Neroued/Qwen3.8-27B-nvfp4-NInfer",
        .quant       = "NVFP4 (NInfer Artifact V2/V3)",
        .max_ctx     = 262144,
        .def_ctx     = 32768,
        .def_chunk   = 2048,
        .mtp_layers  = 1,
        .dflash2_max = 4,
        .has_vision  = 1,
    },
};

const mm_profile *mm_profile_by_name(const char *name)
{
    int i;
    if (!name)
        return NULL;
    for (i = 0; i < MM_N_PROFILES; i++) {
        if (!strcasecmp(name, MM_PROFILES[i].name))
            return &MM_PROFILES[i];
    }
    return NULL;
}

const mm_profile *mm_profile_get(uint32_t id)
{
    int i;
    for (i = 0; i < MM_N_PROFILES; i++)
        if (MM_PROFILES[i].id == id)
            return &MM_PROFILES[i];
    return NULL;
}

/* ------------------------------------------------------ engine config */

void mm_engine_cfg_default(mm_engine_cfg *c)
{
    memset(c, 0, sizeof *c);
    c->artifact    = NULL;
    c->max_ctx     = 32768;
    c->kv_capacity = 0;        /* auto */
    c->kv_dtype    = MM_KV_BF16;
    c->concurrency = 1;
    c->chunk       = 2048;
    c->draft       = 0;
    c->temperature = 0.0f;
    c->top_k       = 0;
    c->top_p       = 1.0f;
    c->seed        = 0x4d1bfe57ull;   /* "mimfer" */
    c->verbose     = MM_LOG_WARN;
    c->no_graph    = 0;
    c->rope_factor = 4.0f;            /* YaRN default scale factor s      */
}

mm_status mm_engine_cfg_validate(const mm_engine_cfg *c)
{
    MM_REQUIRE(c->max_ctx > 0 && c->max_ctx <= MM_MAX_SEQ, MM_ERR_RANGE);
    MM_REQUIRE(c->kv_capacity == 0 ||
               c->kv_capacity >= (size_t)c->max_ctx, MM_ERR_RANGE);
    MM_REQUIRE(c->kv_capacity % MM_KV_BLOCK_TOK == 0, MM_ERR_RANGE);
    MM_REQUIRE(c->concurrency >= 1 &&
               c->concurrency <= MM_MAX_CONCURRENCY, MM_ERR_RANGE);
    MM_REQUIRE(c->chunk >= 1 && c->chunk <= MM_MAX_PREFILL_CH,
               MM_ERR_RANGE);
    MM_REQUIRE(c->draft <= MM_DRAFT_MAX, MM_ERR_RANGE);
    MM_REQUIRE((int)c->verbose >= MM_LOG_ERR &&
               (int)c->verbose <= MM_LOG_DBG, MM_ERR_RANGE);
    /* YaRN (rope.h): the scaling params are range-checked whether or
     * not the scaling is enabled -- the flags are real, so a nonsense
     * value is refused even while dormant (factor must be a real
     * extension factor; orig_ctx 0 = model max_pos, else sane tokens). */
    MM_REQUIRE(c->rope_yarn == 0 || c->rope_yarn == 1, MM_ERR_RANGE);
    MM_REQUIRE(c->rope_factor > 1.0f && c->rope_factor < 1e6f &&
               c->rope_factor == c->rope_factor, MM_ERR_RANGE);
    MM_REQUIRE(c->rope_orig_ctx == 0 ||
               c->rope_orig_ctx <= MM_MAX_SEQ, MM_ERR_RANGE);
    /* Speculative decoding: validated scaffolding (docs/dflash2.md).
     * The flags are real; the draft/verify loop is not built yet, so a
     * non-off backend must always carry a window, the draft LM head
     * flag needs a backend, and a window without a backend is refused. */
    MM_REQUIRE(c->spec >= 0 && c->spec <= 2, MM_ERR_RANGE);
    MM_REQUIRE(c->lm_head_draft == 0 || c->lm_head_draft == 1,
               MM_ERR_RANGE);
    if (c->lm_head_draft)
        MM_REQUIRE(c->spec != 0, MM_ERR_RANGE);
    if (c->spec != 0)
        MM_REQUIRE(c->draft >= 1, MM_ERR_RANGE);
    else
        MM_REQUIRE(c->draft == 0, MM_ERR_RANGE);
    /* Weights profile + vision hook. */
    MM_REQUIRE(c->profile_id == 0 || c->profile_id <= (uint32_t)MM_N_PROFILES,
               MM_ERR_RANGE);
    MM_REQUIRE(c->vision == 0 || c->vision == 1, MM_ERR_RANGE);
    return mm_policy_check(c->temperature, c->top_k, c->top_p);
}
