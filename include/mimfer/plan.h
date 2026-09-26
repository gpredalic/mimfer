/*
 * Plan IR: precompiled, address-fixed operation lists.
 *
 * At load time the model builder (src/model/qwen38.c) walks the layer
 * config and emits one plan per (phase, shape):
 *
 *   PH_PREFILL  M = tokens in the chunk actually being run (plans are
 *               built on demand for the distinct chunk sizes seen; the
 *               common case is exactly one, cfg.chunk)
 *   PH_DECODE   M = number of active slots (1..MM_MAX_CONCURRENCY);
 *               batched decode of all active slots in one traversal
 *   PH_VERIFY   M = 1 + draft, batch 1 (speculative verification pass)
 *
 * A plan is a flat array of mm_ops. Every tensor operand is a device
 * pointer that is FIXED at load time (weights from the weight arena,
 * activations from the activation arena's reserved region, KV from the
 * pool bases). After capture, running a plan is one cudaGraphLaunch; the
 * runtime path is graph launch + host bookkeeping only.
 *
 * Scalar shape parameters that change per step (positions, slot set,
 * sequence lengths) are NOT baked into the graph: they ride in a small
 * device "control" buffer that the scheduler writes each round via the
 * xfer stream before launching the graph (one 256 B H2D, async).
 */
#ifndef MIMFER_PLAN_H
#define MIMFER_PLAN_H

#include "mimfer.h"
#include "config.h"
#include "cuda_rt.h"

typedef enum mm_phase {
    PH_PREFILL = 0,
    PH_DECODE  = 1,
    PH_VERIFY  = 2,
    PH_N
} mm_phase;

/* Opcodes. Stable and versioned: the plan builder and the dispatcher
 * (kernels.h) must agree; tests enforce coverage of every opcode. */
typedef enum mm_op {
    OP_EMBED = 0,     /* x = embed[toks]                                   */
    OP_LMHEAD,        /* logits = x @ W_lm^T (fp4)                         */
    OP_RMSNORM,       /* y = rmsnorm(x, w)                                 */
    OP_ADD_RMSNORM,   /* res = res + x; y = rmsnorm(res, w)                */
    OP_GEMM_F4,       /* y = x @ W^T, W nvfp4 (M<=8: split-K gemv, else    */
                      /*   dequant-to-scratch + bf16 mma gemm)             */
    OP_GEMM_BF16,     /* y = x @ W^T, both bf16 (attention matmuls, MTP)   */
    OP_QKV,           /* qkv = x @ W_qkv^T (full-attention layers)         */
    OP_ROPE,          /* in-place partial rope on q and k (positions from  */
                      /*   the control buffer)                             */
    OP_KVSTORE,       /* copy k,v into the paged pool for toks [a..b)      */
    OP_ATT_DECODE,    /* paged GQA decode, one CTA per (slot, q_head)      */
    OP_ATT_PREFILL,   /* materialized causal attention over S tiles        */
    OP_SOFTMAX_CAUSAL,/* S tile -> P (scale + causal mask), in place       */
    OP_SILU_MUL,      /* y = silu(gate) * up                                */
    OP_CONV4,         /* causal conv over qkv channels + conv-state update */
    OP_LIN_DEC,       /* DeltaNet decode: state update + output            */
    OP_LIN_PRE,       /* DeltaNet prefill: sequential recurrence over M    */
    OP_COPY,          /* device->device copy (residual aliasing)           */
    OP_SAMPLE,        /* logits -> next token (policy from control buf)    */
    OP_NOP,
    OP_N
} mm_op;

#define MM_MAX_OPS 1200   /* 64 layers x ~14 ops + head/tail ops           */

typedef struct mm_op_desc {
    int      op;
    int      layer;       /* -1 for global ops                              */
    const void *a;        /* operand A (pointer, meaning per opcode)       */
    const void *b;        /* operand B                                      */
    void     *c;          /* operand C (output)                             */
    const void *d;        /* operand D (output 2 / aux)                     */
    uint32_t  n0, n1, n2; /* small shape scalars (M, cols, rows...)         */
    char      note[24];   /* for --plan dump readability                    */
} mm_op_desc;

typedef struct mm_plan {
    mm_phase   phase;
    uint32_t   M;
    uint32_t   n_ops;
    mm_op_desc ops[MM_MAX_OPS];
    mm_graph *graph;       /* captured; NULL in --no-graph / host builds    */
    int        captured;
} mm_plan;

typedef struct mm_engine mm_engine;   /* opaque; defined in engine.h       */

/* Reserve the fixed activation buffer map (e->ab) from the activation
 * arena, sized for the worst-case M. Call once at load with the prefill
 * chunk; every plan references these stable pointers. Idempotent: a
 * second call with a smaller M is ignored. */
mm_status mm_plan_reserve_acts(mm_engine *e, uint32_t M);

/* Build (fill ops + reserve activation slots) but do not capture. */
mm_status mm_plan_build(mm_engine *e, mm_plan *p, mm_phase ph, uint32_t M);
/* Capture every built plan into CUDA graphs. */
mm_status mm_plan_capture_all(mm_engine *e);
/* Launch on the compute stream (after the control buffer H2D). */
mm_status mm_plan_launch(mm_engine *e, const mm_plan *p);
void      mm_plan_destroy(mm_plan *p);
/* Human-readable dump (CLI --plan). */
int       mm_plan_dump(const mm_plan *p, char *buf, size_t n);

#endif /* MIMFER_PLAN_H */
