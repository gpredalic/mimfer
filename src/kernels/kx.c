/*
 * Kernel dispatcher (see src/kernels/kx.h and mimfer/kernels.h).
 *
 * mm_kx_invoke() is the entire "custom operator dispatch": one switch over
 * the stable opcode enum that routes a plan op to its launcher. In the GPU
 * build the kx_op_* launchers are CUDA launches; in the CPU reference build
 * they are the implementations in src/kernels/cpu/cx.c. This file is
 * build-independent — it only switches, so the same dispatcher works for
 * both. It is intentionally stateless: everything a launcher needs arrives
 * in the mm_kcall (the fixed op descriptor + the engine pointer).
 */
#include "mimfer/kernels.h"
#include "mimfer/engine.h"

mm_status mm_kx_invoke(const mm_kcall *kc)
{
    const mm_op_desc *op;

    if (!kc || !kc->op || !kc->e)
        return MM_ERR_STATE;
    op = kc->op;
    switch (op->op) {
    case OP_EMBED:          return kx_op_embed(kc);
    case OP_LMHEAD:         return kx_op_lmhead(kc);
    case OP_RMSNORM:        return kx_op_rmsnorm(kc);
    case OP_ADD_RMSNORM:    return kx_op_add_rmsnorm(kc);
    case OP_GEMM_F4:        return kx_op_gemm_f4(kc);
    case OP_GEMM_BF16:      return kx_op_gemm_bf16(kc);
    case OP_QKV:            return kx_op_qkv(kc);
    case OP_ROPE:           return kx_op_rope(kc);
    case OP_KVSTORE:        return kx_op_kvstore(kc);
    case OP_ATT_DECODE:     return kx_op_att_decode(kc);
    case OP_ATT_PREFILL:    return kx_op_att_prefill(kc);
    case OP_SOFTMAX_CAUSAL: return kx_op_softmax_causal(kc);
    case OP_SILU_MUL:       return kx_op_silu_mul(kc);
    case OP_CONV4:          return kx_op_conv4(kc);
    case OP_LIN_DEC:        return kx_op_lin_dec(kc);
    case OP_LIN_PRE:        return kx_op_lin_pre(kc);
    case OP_COPY:           return kx_op_copy(kc);
    case OP_SAMPLE:         return kx_op_sample(kc);
    case OP_NOP:            return MM_OK;
    case OP_N:
    default:
        MM_LOGE("kx: unhandled opcode %d", op->op);
        return MM_ERR_UNSUPPORTED;
    }
}
