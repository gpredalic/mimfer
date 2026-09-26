/*
 * Kernel dispatch.
 *
 * One switch (src/kernels/kx.c) maps an opcode to a launch function. In
 * the GPU build the launch functions live in the .cu files under
 * src/kernels/; in the host test build the SAME symbols are provided by tests/cpu/cx.c as
 * CPU references, so plans, scheduler and artifact logic run end-to-end
 * without a GPU. This is the entire "custom operator dispatch": a switch
 * over a stable opcode enum, no table, no registry, no plugins.
 *
 * Every launch is stateless w.r.t. the engine: per-op pointers arrive in
 * the mm_op_desc (fixed at plan build time); per-round scalars (positions,
 * slot set, sampling state) arrive in the device control buffer (one small
 * async H2D per round, xfer stream). Determinism: no atomics in math-
 * critical paths; split-K reductions are order-faithful.
 *
 * mm_ctrl is POD: the host mirror and the device buffer are the same
 * struct. 16 KiB dominated by the prefill token array; a decode round
 * writes only the first 64 B.
 */
#ifndef MIMFER_KERNELS_H
#define MIMFER_KERNELS_H

#include "mimfer.h"
#include "plan.h"

/* The dispatcher ABI is C: the switch (src/kernels/kx.c), the engine and
 * the C test binaries link against these symbols with C linkage, while the
 * GPU launchers are DEFINED in a C++ translation unit (src/kernels/cuda/
 * cx.cu, compiled by nvcc). The guard pins C linkage on every declaration
 * below, so the .cu definitions inherit it from their prototypes. */
#ifdef __cplusplus
extern "C" {
#endif

/* mm_engine is forward-declared by plan.h; kernels only need the pointer. */

/* Opaque stream handle: cudaStream_t in GPU builds, ignored in host. */
typedef void *mm_stream_h;

/* Effective RoPE inverse-frequency table capacity: rope_dim/2 pairs,
 * bounded well above the model's 32 (Qwen3.8-27B partial RoPE, rope_dim
 * 64) for any future model in this appliance line. */
#define MM_MAX_ROPE_PAIRS 128

typedef struct mm_ctrl {
    /* prefill / verify: the token ids of this chunk (M = n used) */
    uint32_t toks[MM_MAX_PREFILL_CH];
    /* decode: one row per active slot */
    uint32_t slot_tok[MM_MAX_CONCURRENCY];
    uint32_t slot_pos[MM_MAX_CONCURRENCY];  /* position of the new token */
    uint32_t slot_len[MM_MAX_CONCURRENCY];  /* KV length BEFORE this tok */
    uint32_t n_slots;
    uint32_t pos0;      /* prefill: position of toks[0]                  */
    uint32_t kv_len;    /* prefill: KV length after this chunk           */
    uint32_t draft;     /* verify: draft window length                   */
    /* sampling state (written by the sampler each step) */
    float    temperature;
    uint32_t top_k;
    float    top_p;
    uint64_t u;         /* one uniform [0,1) as fixed-point 1.31        */
    /* RoPE effective table (rope.h): filled at load and re-copied after
     * every round's ctrl reset. rope_inv[p] is the effective inverse
     * frequency of pair p (plain RoPE: powf(1/theta, 2p/rd), bit-exact
     * with the legacy per-kernel formula; YaRN: the NTK-by-parts blend);
     * rope_mscale scales the rotated q/k components (plain 1.0; YaRN
     * 0.1*ln(factor)+1, the paper's attention temperature folded into
     * the rotary embedding). */
    float    rope_inv[MM_MAX_ROPE_PAIRS];
    float    rope_mscale;
} mm_ctrl;

typedef struct mm_kcall {
    const mm_op_desc *op;   /* fixed pointers + shape scalars            */
    const mm_engine  *e;    /* pools, control buffer, sampling, cfg      */
    mm_stream_h       stream;
} mm_kcall;

/* The dispatcher (one switch). */
mm_status mm_kx_invoke(const mm_kcall *kc);

/* Per-op launchers. op_desc operand meanings per opcode:
 *
 *  EMBED        a=embed(bf16|fp4 wt), c=x;  n2: 0=bf16 1=fp4; toks from ctrl
 *  LMHEAD       a=x, b=W_lm (n2: 0=bf16 1=fp4), c=logits
 *  RMSNORM      a=x, b=gain, c=y
 *  ADD_RMSNORM  a=residual (in-place: a += b; kernel writes a), b=delta,
 *               c=normalized output, d=gain
 *  GEMM_F4      a=x (bf16 [M,K]), b=W fp4 payload, d=W scale, c=y;
 *               n0=M, n1=K, n2=out cols; M<=8 -> split-K gemv, else
 *               dequant->scratch (op->c scratch base in note? NO: scratch
 *               pointer in e->act + op->n2<<... ) — see design.md.
 *  GEMM_BF16    a=x bf16, b=W bf16, c=y;  n0=M n1=K n2=cols
 *  QKV          like GEMM_F4 (fused qkv weight)
 *  ROPE         a=q buffer, b=k buffer, n0=n vecs, n1=rope_dim (ctrl: pos)
 *  KVSTORE      a=k, b=v, c=layer kv base (n0=layer), n1=n toks (ctrl)
 *  ATT_DECODE   a=layer kv base, b=q buffer, c=o buffer;
 *               n0=layer, n1=batch (n_slots from ctrl)
 *  ATT_PREFILL  a=layer kv base, b=q buffer, c=o buffer; n0=M
 *               (k,v are read from the pool the preceding KVSTORE wrote)
 *  SOFTMAX_CAUSAL a=S tile, n0=rows n1=cols n2=kv_start (ctrl: pos0)
 *  SILU_MUL     a=gate, b=up, c=y
 *  CONV4        a=x [M,C], b=conv_w [C][4], c=out, d=conv state (ctrl)
 *  LIN_DEC      a=layer state base, b=x, c=out, d=qkv buffer (ctrl: slot)
 *  LIN_PRE      a=layer state base, b=x, c=out, d=qkv buffer (ctrl: pos0)
 *  COPY         a=src, b=dst, n0=bytes
 *  SAMPLE       a=logits, b=out token(s) (ctrl: policy + u)
 */
mm_status kx_op_embed(const mm_kcall *);
mm_status kx_op_lmhead(const mm_kcall *);
mm_status kx_op_rmsnorm(const mm_kcall *);
mm_status kx_op_add_rmsnorm(const mm_kcall *);
mm_status kx_op_gemm_f4(const mm_kcall *);
mm_status kx_op_gemm_bf16(const mm_kcall *);
mm_status kx_op_qkv(const mm_kcall *);
mm_status kx_op_rope(const mm_kcall *);
mm_status kx_op_kvstore(const mm_kcall *);
mm_status kx_op_att_decode(const mm_kcall *);
mm_status kx_op_att_prefill(const mm_kcall *);
mm_status kx_op_softmax_causal(const mm_kcall *);
mm_status kx_op_silu_mul(const mm_kcall *);
mm_status kx_op_conv4(const mm_kcall *);
mm_status kx_op_lin_dec(const mm_kcall *);
mm_status kx_op_lin_pre(const mm_kcall *);
mm_status kx_op_copy(const mm_kcall *);
mm_status kx_op_sample(const mm_kcall *);

#ifdef __cplusplus
}   /* extern "C" */
#endif

#endif /* MIMFER_KERNELS_H */
