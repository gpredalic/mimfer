/*
 * Pinned host I/O memory (see cuda_mem.h). The only cudaHostAlloc in the
 * process; everything else device-side lives in the arenas (alloc.c) and
 * the host side uses plain calloc/free.
 */
#include "mimfer/cuda_mem.h"
#include <stdlib.h>

#if defined(MM_WITH_CUDA)
#include <cuda_runtime.h>
#endif

mm_status mm_pin_alloc(size_t bytes, void **out)
{
    void *p = NULL;

    if (!out)
        return MM_ERR_STATE;
    *out = NULL;
#if defined(MM_WITH_CUDA)
    cudaError_t e = cudaHostAlloc(&p, bytes, cudaHostAllocDefault);
    if (e != cudaSuccess) {
        MM_LOGE("cudaHostAlloc(%zu B): %s", bytes, cudaGetErrorString(e));
        return MM_ERR_NOMEM;
    }
#else
    p = calloc(1, bytes ? bytes : 1);
    if (!p)
        return MM_ERR_NOMEM;
#endif
    *out = p;
    return MM_OK;
}

void mm_pin_free(void *p)
{
    if (!p)
        return;
#if defined(MM_WITH_CUDA)
    cudaFreeHost(p);
#else
    free(p);
#endif
}