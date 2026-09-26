/*
 * CPU kernel dispatch internals (see src/kernels/kx.c and src/kernels/cpu/).
 *
 * The public contract — mm_kx_invoke(), the per-op kx_op_* launchers,
 * mm_kcall, mm_ctrl — lives in mimfer/kernels.h. This header exists so the
 * CPU reference implementation and its unit tests share one documented
 * entry point for what the CPU reference covers.
 *
 * CPU reference scope (implemented in src/kernels/cpu/cx.c):
 *   - single-slot: reads/writes slot 0 of the KV pool and of the linear and
 *     conv state pools. The appliance prefill is one prompt at a time and the
 *     engine skeleton runs a single decode slot, so slot 0 is the whole
 *     working set; a multi-slot CPU reference would add the slot offset.
 *   - bf16 in/out with f32 accumulation; fixed loop order, no atomics, so
 *     outputs are bit-deterministic for a fixed input.
 *   - DeltaNet (LIN_PRE/LIN_DEC) uses a fixed per-head decay as its
 *     reference recurrence; the GPU kernels are the source of truth for the
 *     exact data-dependent gating. CONV4 runs the true causal convolution.
 */
#ifndef MM_KERNELS_KX_H
#define MM_KERNELS_KX_H

#include "mimfer/kernels.h"
#include "mimfer/plan.h"

/* C linkage: the CPU reference (C) and the C test binaries call these;
 * kx_cuda_oppref is DEFINED in the C++ translation unit cx.cu and inherits
 * the C linkage from this prototype (see the guard in mimfer/kernels.h). */
#ifdef __cplusplus
extern "C" {
#endif

/* 1 if the CPU reference implements opcode `op` (i.e. it is a real op the
 * dispatcher will run on the CPU path). Used by coverage tests to assert the
 * reference covers every executable opcode. OP_N / out-of-range -> 0. */
int kx_cpu_oppref(int op);

/* Same coverage query for the CUDA launch path (src/kernels/cuda/cx.cu),
 * provided by the MM_WITH_CUDA build. The parity test asserts the two
 * coverage sets agree so no executable opcode is left without a GPU
 * launcher. Unreferenced in host builds (the symbol exists only in the
 * .cu), so the prototype is safe in both. */
int kx_cuda_oppref(int op);

#ifdef __cplusplus
}   /* extern "C" */
#endif

#endif /* MM_KERNELS_KX_H */
