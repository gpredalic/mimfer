/*
 * CUDA memory management: the pinned host I/O block.
 *
 * The four-allocation device model (alloc.h) owns all cudaMalloc'd memory
 * through the arenas. This module owns the ONE remaining host-side
 * allocation the device path needs: a pinned host block for async token /
 * control-buffer I/O (alloc.h: "plus one small cudaHostAlloc for token
 * I/O"). Pinned host memory is what makes mm_h2d_async / mm_d2h_async
 * actually asynchronous on the xfer stream.
 *
 * One block per process, allocated once at engine load (engine's `pin`
 * field), freed at destroy. Layout (see the engine):
 *     [0 .. sizeof(mm_ctrl))            device control buffer (H2D each
 *                                      round from the host mirror e->ctrl)
 *     [sizeof .. + chunk*4)             token readback mirror of
 *                                      e->ab.toks_dev (D2H each round)
 * The size is a compile-time constant: one full control buffer + one max
 * prefill chunk of u32 token ids (see mm_pin_size below).
 *
 * Host build: the stubs are plain calloc/free — honest, but NOT pinned
 * (there is no device to stage to; the host build's "async" copies are
 * synchronous memcpys in cuda_rt.c).
 */
#ifndef MIMFER_CUDA_MEM_H
#define MIMFER_CUDA_MEM_H

#include "mimfer.h"
#include "kernels.h"   /* mm_ctrl: the pinned block's first region */

/* One full control buffer (16-aligned) + one max prefill chunk of
 * u32 token ids. */
#define MM_PIN_BYTES (mm_align_up(sizeof(mm_ctrl), 16) + MM_MAX_PREFILL_CH * 4u)

/* Allocate the pinned host I/O block (once). NULL out *out on failure. */
mm_status mm_pin_alloc(size_t bytes, void **out);
void      mm_pin_free(void *p);

#endif /* MIMFER_CUDA_MEM_H */