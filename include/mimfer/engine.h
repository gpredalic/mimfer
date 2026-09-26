/*
 * The engine: the appliance's single owner. Owns config, device, arenas,
 * pools, tensor registry, plans, scheduler, tokenizer, sampling, telemetry.
 *
 * Lifecycle (one process, one engine):
 *
 *   mm_engine_create(cfg)
 *     1. probe device, check target profile           (hard fail)
 *     2. open artifact, verify superblock + TOC       (hard fail)
 *     3. parse model section -> mm_model_cfg          (validated)
 *     4. compute the memory map (design.md) and commit
 *        the four device allocations
 *     5. load weights: mmap the artifact, verify per-
 *        section checksums, async H2D in 256 MiB slices
 *     6. build tokenizer from artifact
 *     7. build + capture plans (prefill/decode/verify)
 *   run loop: mm_engine_step() driven by the CLI
 *   mm_engine_destroy()
 *
 * Weights are not copied into a private layout: the tensor registry maps
 * artifact tensor names to arena slots with their (name, shape, dtype).
 * The plan builder is a pure function of (model_cfg, registry, buffers),
 * which is what makes the whole thing testable without a GPU.
 */
#ifndef MIMFER_ENGINE_H
#define MIMFER_ENGINE_H

#include "mimfer.h"
#include "config.h"
#include "alloc.h"
#include "tensor.h"
#include "kernels.h"
#include "kv.h"
#include "plan.h"
#include "sched.h"
#include "artifact.h"
#include "tokenizer.h"
#include "sampling.h"
#include "telemetry.h"
#include "rope.h"

/* ------------------------------------------------ tensor registry */
/*
 * One entry per loaded tensor. Names follow the HF-style convention of
 * the source checkpoints (the packer validates the exact set at pack time
 * and the plan builder fails loudly on any missing name):
 *
 *   model.layers.{i}.input_layernorm.weight
 *   model.layers.{i}.self_attn.{q,k,v,o}_proj.weight   (full layers)
 *   model.layers.{i}.linear_attn.in_proj.weight        (lin layers)
 *   model.layers.{i}.linear_attn.conv1d.weight
 *   model.layers.{i}.linear_attn.dt_bias
 *   model.layers.{i}.linear_attn.A_log
 *   model.layers.{i}.linear_attn.norm.weight
 *   model.layers.{i}.linear_attn.out_proj.weight
 *   model.layers.{i}.mlp.{gate,up,down}_proj.weight
 *   model.layers.{i}.post_attention_layernorm.weight
 *   model.norm.weight   model.embed_tokens.weight   lm_head.weight
 *   mtp.layers.{i}.*    mtp.norm.weight             (draft head)
 *
 * fp4 entries carry a weight view (payload + scale); everything else a
 * plain bf16/f32 view.
 */
#define MM_MAX_TENSORS 4096

typedef struct mm_tens {
    char      name[48];
    mm_tensor v;      /* bf16/f32 view (valid when !is_fp4)               */
    mm_wt     w;      /* fp4 view (valid when is_fp4)                      */
    int       is_fp4;
} mm_tens;

typedef struct mm_tensreg {
    mm_tens   t[MM_MAX_TENSORS];
    uint32_t  n;
} mm_tensreg;

const mm_tens *mm_tens_find(const mm_tensreg *r, const char *name);
mm_status mm_tens_add(mm_tensreg *r, const char *name, int is_fp4,
                      const mm_tensor *v, const mm_wt *w);

/* ------------------------------------------------------ engine struct */

/* Fixed activation buffer map (reserved once at load; pointers stable). */
typedef struct mm_abufs {
    void  *x;        /* [M][hidden] bf16, M = chunk (or batch)            */
    void  *res;      /* residual stream [M][hidden]                       */
    void  *normed;   /* normalized [M][hidden]                            */
    void  *qkv;      /* [M][qkv_dim] (full layers)                        */
    void  *q; void  *k; void  *v; void  *o;   /* split buffers            */
    void  *gate; void *up; void *mlp;         /* [M][inter]               */
    void  *S;        /* attention S tile [q_tile][kv_max] bf16            */
    void  *part;     /* split-K partial sums (gemv)                       */
    void  *scratch;  /* dequant scratch for prefill gemm (bf16 W tile)    */
    void  *logits;   /* [M][vocab] bf16                                   */
    void  *toks_dev; /* token ids [chunk] (control buffer mirror)         */
} mm_abufs;

struct mm_engine {
    mm_engine_cfg cfg;
    mm_dev_info   dev;
    mm_model_cfg  mc;
    mm_rope       rope;           /* effective RoPE table (plain or YaRN)  */
    const mm_profile *profile;    /* selected weights profile (or NULL)     */
    mm_artifact  *art;          /* open handle (lazy mmap sections)       */
    mm_tensreg    reg;          /* loaded tensors                         */
    mm_arena      w_arena;      /* weights (bump, never reset)            */
    mm_arena      a_arena;      /* activations (resettable)               */
    mm_kvpool    *kv;
    mm_linstate  *st;
    mm_prefix    *pre;
    mm_abufs      ab;           /* fixed activation pointers              */
    mm_ctrl      *ctrl_dev;     /* device control buffer                  */
    mm_ctrl       ctrl;         /* host mirror                            */
    void         *pin;          /* pinned host I/O block                  */

    mm_plan       dec[MM_MAX_CONCURRENCY + 1]; /* by n_slots (1..n)       */
    mm_plan       ver[MM_DRAFT_MAX + 1];        /* by draft window        */
    mm_plan       pre_c[8];                    /* prefill cache by M     */
    uint32_t      pre_cM[8];
    mm_sched      sched;                       /* round scheduler        */

    /* Device-side I/O (MM_WITH_CUDA build; both NULL on the host build):
     * ctrl_dev is the device control buffer (the pinned block's first
     * region; the host mirror is `ctrl`); toks_host is the token readback
     * mirror of ab.toks_dev (the pinned block's second region). */
    uint32_t     *toks_host;

    mm_tok        tok;
    mm_rng        rng;
    mm_tel        tel;
    uint64_t      tick;         /* global round counter                   */
    uint64_t      n_tokens_out;
    uint32_t      last_tok[MM_MAX_CONCURRENCY]; /* last sampled token/slot */
    int           loaded;
    int           on_host;      /* 1 = CPU reference build (tests)        */
};

mm_status mm_engine_create(const mm_engine_cfg *cfg, mm_engine **out);
mm_status mm_engine_load(mm_engine *e);
/* One scheduler round: prefill the waiting head if any slot is free and
 * a prompt waits; otherwise decode all active slots. */
mm_status mm_engine_step(mm_engine *e);
void      mm_engine_destroy(mm_engine *e);

/* Slot access for the scheduler (slots live in mm_sched, defined there). */
const mm_abufs *mm_engine_ab(const mm_engine *e);
const mm_ctrl  *mm_engine_ctrl_dev(const mm_engine *e);

/* Vision hook (planned pipeline): submit an image for the vision tower.
 * The surface is real -- the flag parses, the request is validated
 * (non-empty, engine loaded, cfg.vision on) -- but image execution is
 * planned work: until the vision pipeline lands the call returns
 * MM_ERR_UNSUPPORTED. */
mm_status mm_engine_submit_image(mm_engine *e, const void *px, size_t n);

#endif /* MIMFER_ENGINE_H */
