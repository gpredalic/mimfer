/*
 * Static configuration: target device profile and model shape.
 *
 * The engine is an appliance: these numbers are not user config. The CLI
 * exposes only the knobs in mm_engine_cfg; everything else is checked at
 * startup and a mismatch is a hard error (MM_ERR_DEVICE / MM_ERR_SHAPE).
 * Device profile: nominal datasheet numbers for the RTX PRO 4000 Blackwell.
 * At startup mm_device_probe() reads cudaDeviceProp and compares field by
 * field; bandwidth is derived (memClock x busWidth x 2), must be within 10%
 * of nominal. See docs/architecture.md.
 */

#ifdef __cplusplus
extern "C" {
#endif

#ifndef MIMFER_CONFIG_H
#define MIMFER_CONFIG_H

#include "mimfer.h"

/* ------------------------------------------------------ device profile */

typedef struct mm_dev_profile {
    const char *name;        /* device name substring to match             */
    int         cc_major;    /* compute capability major                   */
    int         cc_minor;    /* compute capability minor (must be 12.x)    */
    int         sm_count;    /* SMs; persistent grids are sized from this  */
    size_t      vram_bytes;  /* total device memory                        */
    size_t      l2_bytes;    /* L2 size (access-policy windows)            */
    size_t      smem_per_sm; /* shared memory per SM (bytes)               */
    size_t      bw_gbps;     /* nominal memory bandwidth, GB/s             */
    int         max_threads_per_sm;
    int         regs_per_sm;
} mm_dev_profile;

/* The one and only supported target. */
extern const mm_dev_profile MM_TARGET_PRO4000;

/* Probed device info as read from the driver at startup. */
typedef struct mm_dev_info {
    char        name[128];
    int         cc_major, cc_minor;
    int         sm_count;
    size_t      vram_bytes;
    size_t      l2_bytes;
    size_t      smem_per_sm;
    size_t      bw_gbps;     /* derived from clock x bus width             */
    int         max_threads_per_sm;
    int         regs_per_sm;
} mm_dev_info;

/* Fill info from the driver. GPU build only. */
mm_status mm_device_probe(int device_ordinal, mm_dev_info *out);

/* Compare probed info against the target profile; log every mismatch. */
mm_status mm_device_check(const mm_dev_info *info, const mm_dev_profile *p);

/* ------------------------------------------------------- model shape */
/*
 * Qwen3.8-27B (model_type qwen3_5, text core): hybrid of 64 layers in the
 * pattern [lin, lin, lin, full] x 16.
 *
 *   "lin"  : Gated DeltaNet linear attention — stateful recurrence (fp32
 *            state per sequence), causal conv k=4, swish output gate.
 *            O(1) work per token; no KV growth.
 *   "full" : GQA attention — 24 q / 4 kv heads, head_dim 256, partial RoPE
 *            (first 64 dims), swish output gate. Standard paged KV cache.
 *
 * RMSNorm pre-norm (no bias) everywhere; SiLU-GLU MLP, intermediate 17408.
 * The artifact is the source of truth: mm_model_cfg is parsed from the
 * artifact's model section and validated against these invariants. The
 * constants below are the expected shape used for arena sizing BEFORE the
 * artifact is read (memory is committed before any section is parsed).
 */

#define MM_Q38_HIDDEN        5120
#define MM_Q38_INTER         17408
#define MM_Q38_LAYERS        64
#define MM_Q38_LIN_LAYERS    48
#define MM_Q38_FULL_LAYERS   16
#define MM_Q38_Q_HEADS       24
#define MM_Q38_KV_HEADS      4
#define MM_Q38_HEAD_DIM      256
#define MM_Q38_ROPE_DIM      64      /* partial_rotary_factor 0.25 x 256 */
#define MM_Q38_ROPE_THETA    1e7f
#define MM_Q38_LIN_K_HEADS   16
#define MM_Q38_LIN_K_DIM     128
#define MM_Q38_LIN_V_HEADS   48
#define MM_Q38_LIN_V_DIM     128
#define MM_Q38_CONV_K        4
#define MM_Q38_VOCAB         248320
#define MM_Q38_MAX_POS       262144
#define MM_Q38_NORM_EPS      1e-6f
#define MM_Q38_MTP_LAYERS    1

typedef struct mm_model_cfg {
    uint32_t hidden, inter, layers;
    uint32_t q_heads, kv_heads, head_dim, rope_dim;
    uint32_t lin_k_heads, lin_k_dim, lin_v_heads, lin_v_dim;
    uint32_t conv_k;
    uint32_t vocab;
    uint32_t max_pos;
    float    rope_theta, norm_eps;
    uint32_t full_layer_step;      /* every Nth layer (1-indexed) is full */
    uint32_t full_layer_offset;    /* offset into the stride pattern     */
    uint32_t mtp_layers;
    /* Derived, filled by mm_model_cfg_finalize(): */
    uint32_t n_lin_layers, n_full_layers;
    size_t   state_bytes_per_seq;  /* all linear layers, fp32 states      */
    size_t   kv_bytes_per_token;   /* full-attention KV, per KV dtype     */
} mm_model_cfg;

/* Expected default; the artifact overrides/validates. */
extern const mm_model_cfg MM_Q38_DEFAULT;

/* Derive layer counts, per-seq state size, per-token KV bytes; validate. */
mm_status mm_model_cfg_finalize(mm_model_cfg *c);

/* Layer i (1-indexed) is full-attention under the stride pattern. */
int mm_model_cfg_layer_is_full(const mm_model_cfg *c, uint32_t i);

/* KV storage dtype for the full-attention layers. */
typedef enum mm_kv_dtype {
    MM_KV_BF16 = 0,
    MM_KV_FP8  = 1,   /* e4m3 K and V, fp32 scale (phase 2)               */
    MM_KV_N
} mm_kv_dtype;

/* ------------------------------------------------ weights profiles */
/*
 * A named operating point for one reference Qwen3.8-27B NVFP4 checkpoint:
 * the checkpoint identity, the capabilities its NInfer artifact carries,
 * and the context defaults the CLI applies when the profile is selected
 * (explicit flags always win). The engine validates the request against
 * the profile ceiling and logs the active checkpoint identity at load.
 * These are the two validated reference checkpoints of this project
 * (README "Supported artifacts"); the profiles describe the same model
 * shape -- the NInfer artifacts differ in quantization metadata, the
 * MTP draft head and the DFlash2 draft window they carry.
 */
#define MM_N_PROFILES 2

typedef struct mm_profile {
    uint32_t    id;          /* 1..MM_N_PROFILES (0 = no profile)        */
    const char *name;        /* CLI name: --weights-profile NAME         */
    const char *repo;        /* Hugging Face checkpoint identity         */
    const char *quant;       /* weight quantization                      */
    uint32_t    max_ctx;     /* native context ceiling (RoPE)            */
    uint32_t    def_ctx;     /* default --max-ctx when selected          */
    uint32_t    def_chunk;   /* default --chunk when selected            */
    uint32_t    mtp_layers;  /* MTP draft layers the artifact carries    */
    uint32_t    dflash2_max; /* DFlash2 draft window ceiling the artifact
                                allows (never above MM_DRAFT_MAX)        */
    int         has_vision;  /* the artifact carries vision weights      */
} mm_profile;

extern const mm_profile MM_PROFILES[MM_N_PROFILES];

/* Resolve a profile by its CLI name (case-insensitive); NULL if absent. */
const mm_profile *mm_profile_by_name(const char *name);
/* Resolve a profile by id (1..MM_N_PROFILES); NULL if absent. */
const mm_profile *mm_profile_get(uint32_t id);

/* ------------------------------------------------------ engine config */
/* The only user-visible knobs. Everything else is derived. */
typedef struct mm_engine_cfg {
    const char *artifact;      /* path to .mimfer                          */
    uint32_t    max_ctx;       /* per-sequence limit (<= MM_MAX_SEQ)       */
    uint32_t    kv_capacity;   /* tokens in shared KV pool; 0 = auto       */
    mm_kv_dtype kv_dtype;
    uint32_t    concurrency;   /* 1..MM_MAX_CONCURRENCY, fixed at start    */
    uint32_t    chunk;         /* prefill chunk tokens                     */
    uint32_t    draft;         /* 0 = off, else draft window 1..MM_DRAFT_MAX */
    float       temperature;   /* 0 = greedy                               */
    int         top_k;         /* 0 = off                                  */
    float       top_p;         /* 1.0 = off                                */
    uint64_t    seed;
    int         verbose;       /* log level                                */
    int         no_graph;      /* disable CUDA graph capture (debug)       */
    /* RoPE long-context scaling (mimfer/rope.h): 0 = plain RoPE (the
     * golden-reference default). */
    int         rope_yarn;     /* 1 = YaRN (NTK-by-parts + attention scale) */
    float       rope_factor;   /* YaRN scale factor s (required > 1.0)     */
    uint32_t    rope_orig_ctx; /* original context; 0 = model max_pos      */
    /* Speculative decoding backend: 0 = off, 1 = MTP draft head,
     * 2 = DFlash2 (docs/dflash2.md). Validated scaffolding: the
     * draft/verify execution loop is planned work and the engine
     * refuses to start with a non-off backend (MM_ERR_UNSUPPORTED). */
    int         spec;
    int         lm_head_draft; /* score the draft with the draft LM head   */
    uint32_t    profile_id;    /* weights profile (0 = none)               */
    int         vision;        /* vision pipeline hook (flags + validation;
                                   image submission is unsupported yet)    */
} mm_engine_cfg;

void mm_engine_cfg_default(mm_engine_cfg *c);
/* Validate ranges; log and fail on anything outside the appliance bounds. */
mm_status mm_engine_cfg_validate(const mm_engine_cfg *c);

#endif /* MIMFER_CONFIG_H */
#ifdef __cplusplus
}
#endif
