/*
 * Thin CUDA runtime layer. One source file (src/cuda/cuda.c) implements all
 * of it. The public header is CUDA-free (portable C), so the scheduler and
 * tests never see <cuda_runtime.h>:
 *
 *   - two streams:
 *       MM_ST_COMPUTE  every kernel launch, every graph launch
 *       MM_ST_XFER     every H2D / D2H (token I/O, telemetry D2H)
 *     Ordering between them is event-based; the token path never calls
 *     cudaStreamSynchronize (the only host syncs are round boundaries in
 *     the scheduler, which is where the next token is consumed anyway).
 *
 *   - a fixed event pool (16 events, no allocation).
 *
 *   - graph primitives used by the plan layer (plan.h): capture on
 *     MM_ST_COMPUTE, instantiate, launch. Capture records an optional
 *     L2 access-policy window per kernel node so the instantiated graph
 *     can pin the current layer's weights in L2 (see docs/design.md
 *     "Cache-aware scheduling").
 *
 * No cudaMalloc lives here: the allocator (alloc.h) owns device memory.
 */
#ifndef MIMFER_CUDA_RT_H
#define MIMFER_CUDA_RT_H

#include "mimfer.h"

typedef enum mm_stream_id {
    MM_ST_COMPUTE = 0,
    MM_ST_XFER    = 1,
    MM_ST_N
} mm_stream_id;

/* Build probe (portable: no <cuda_runtime.h> needed). The engine's device
 * probe and the parity test branch on these without linking the driver. */
/* 1 when this build was compiled with -DMM_WITH_CUDA (kernels run on GPU). */
int         mm_cuda_enabled(void);
/* Number of visible CUDA devices (0 on the host build). */
mm_status   mm_cuda_device_count(int *n_out);

/* Process-wide CUDA handle set. Init exactly once, before anything else. */
mm_status mm_cuda_init(int device_ordinal);
mm_status mm_cuda_shutdown(void);

/* Streams. Opaque on purpose. */
typedef struct mm_stream { int id; } mm_stream;
mm_stream mm_cuda_stream(mm_stream_id id);

/* Events: index into the fixed pool. */
#define MM_N_EVENTS 16
mm_status mm_event_record(int ev, mm_stream_id st);
mm_status mm_event_wait(int ev, mm_stream_id st);
mm_status mm_stream_sync(mm_stream_id st);

/* Async copies (pinned host memory only). */
mm_status mm_h2d_async(void *dst, const void *src, size_t n, mm_stream_id st);
mm_status mm_d2h_async(void *dst, const void *src, size_t n, mm_stream_id st);
mm_status mm_mset_async(void *dst, int value, size_t n, mm_stream_id st);

/* Opaque graph handle. */
typedef struct mm_graph mm_graph;
mm_status mm_graph_create(mm_graph **out);
/* Begin capture on the compute stream. */
mm_status mm_graph_capture_begin(mm_graph *g);
/* End capture and instantiate. `window` (may be NULL) sets the L2
 * access-policy window on all kernel nodes of this graph. */
mm_status mm_graph_commit(mm_graph *g, const void *window_base,
                          size_t window_bytes, int hit_ratio_pct);
mm_status mm_graph_launch(mm_graph *g, mm_stream_id st);
void      mm_graph_destroy(mm_graph *g);

/* Per-node window support during capture: record the window that applies
 * to the NEXT launched kernel (used by the plan capture loop). */
mm_status mm_graph_set_node_window(mm_graph *g, const void *base,
                                   size_t bytes, int hit_ratio_pct);

/* Device bookkeeping (used by allocator and telemetry). */
mm_status mm_device_vram_free(size_t *bytes_out);
mm_status mm_device_sync_all(void);

/* Capture state: 1 between a successful mm_graph_capture_begin and the
 * matching mm_graph_commit, 0 otherwise (and always 0 on the host build).
 * The .cu kernel launchers query this to stay capture-legal:
 * cudaStreamSynchronize is forbidden on a stream in capture mode, and a
 * device->host readback cannot be captured, so the per-op stream sync and
 * the EMBED error-flag round trip are suspended for the duration of a
 * capture. The kernel launches and all device writes are captured as
 * usual; the per-op error contract resumes for direct dispatch. */
int         mm_cuda_capturing(void);

/*
 * Internal add-on for CUDA-only translation units (the .cu kernel launch
 * path under src/kernels/cuda/): raw stream handles. Kept out of this
 * header on purpose — cuda_rt.h stays CUDA-free so the scheduler, engine
 * and tests never see <cuda_runtime.h>.
 */
#ifdef MM_WITH_CUDA
#include <cuda_runtime.h>
/* Raw compute/xfer stream handle for kernel launches (valid after
 * mm_cuda_init). Opaque mm_stream stays the portable ABI. */
cudaStream_t mm_cuda_stream_raw(mm_stream_id id);
#endif /* MM_WITH_CUDA */

#endif /* MIMFER_CUDA_RT_H */
