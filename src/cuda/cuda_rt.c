/*
 * CUDA runtime layer (see cuda_rt.h). Everything CUDA-specific lives in
 * this file; the rest of the tree compiles clean without nvcc when
 * MM_WITH_CUDA is not defined and gets honest no-op stubs.
 */
#include "mimfer/cuda_rt.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef MM_WITH_CUDA

#include <cuda_runtime.h>

static cudaStream_t g_st[MM_ST_N];
static cudaEvent_t g_ev[MM_N_EVENTS];
static int         g_inited;
static int         g_capturing;   /* compute stream in capture mode */

int mm_cuda_enabled(void)
{
    return 1;
}

mm_status mm_cuda_device_count(int *n_out)
{
    int n;
    CK(cudaGetDeviceCount(&n));
    *n_out = n;
    return MM_OK;
}

cudaStream_t mm_cuda_stream_raw(mm_stream_id id)
{
    return g_st[id];
}

#define CK(call)                                                        \
    do {                                                                \
        cudaError_t _e = (call);                                        \
        if (_e != cudaSuccess) {                                        \
            MM_LOGE("cuda: %s (line %d): %s", #call, __LINE__,          \
                    cudaGetErrorString(_e));                            \
            return MM_ERR_CUDA;                                         \
        }                                                               \
    } while (0)

mm_status mm_cuda_init(int device_ordinal)
{
    int dev = device_ordinal < 0 ? 0 : device_ordinal;

    if (g_inited)
        return MM_OK;
    CK(cudaSetDevice(dev));
    CK(cudaStreamCreateWithFlags(&g_st[MM_ST_COMPUTE],
                                 cudaStreamNonBlocking));
    CK(cudaStreamCreateWithFlags(&g_st[MM_ST_XFER], cudaStreamNonBlocking));
    for (int i = 0; i < MM_N_EVENTS; i++)
        CK(cudaEventCreateWithFlags(&g_ev[i], cudaEventDisableTiming));
    g_inited = 1;
    return MM_OK;
}

mm_status mm_cuda_shutdown(void)
{
    if (!g_inited)
        return MM_OK;
    for (int i = 0; i < MM_N_EVENTS; i++)
        cudaEventDestroy(g_ev[i]);
    cudaStreamDestroy(g_st[MM_ST_COMPUTE]);
    cudaStreamDestroy(g_st[MM_ST_XFER]);
    g_inited = 0;
    return MM_OK;
}

mm_stream mm_cuda_stream(mm_stream_id id)
{
    mm_stream s;
    s.id = (int)id;
    return s;
}

mm_status mm_event_record(int ev, mm_stream_id st)
{
    CK(cudaEventRecord(g_ev[ev], g_st[st]));
    return MM_OK;
}

mm_status mm_event_wait(int ev, mm_stream_id st)
{
    CK(cudaStreamWaitEvent(g_st[st], g_ev[ev], 0));
    return MM_OK;
}

mm_status mm_stream_sync(mm_stream_id st)
{
    CK(cudaStreamSynchronize(g_st[st]));
    return MM_OK;
}

mm_status mm_h2d_async(void *dst, const void *src, size_t n, mm_stream_id st)
{
    CK(cudaMemcpyAsync(dst, src, n, cudaMemcpyHostToDevice, g_st[st]));
    return MM_OK;
}

mm_status mm_d2h_async(void *dst, const void *src, size_t n, mm_stream_id st)
{
    CK(cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToHost, g_st[st]));
    return MM_OK;
}

mm_status mm_mset_async(void *dst, int value, size_t n, mm_stream_id st)
{
    CK(cudaMemsetAsync(dst, value, n, g_st[st]));
    return MM_OK;
}

typedef struct mm_graph {
    cudaGraphExec_t exec;
    int             have_exec;
    /* Pending L2 window applied to kernel nodes at commit. v1 applies
     * the same window to all kernel nodes of a graph; per-layer graphs
     * with their own windows come with the per-layer plan split. */
    int             win_valid;
    void           *win_base;
    size_t          win_bytes;
    int             win_pct;
} mm_graph;

mm_status mm_graph_create(mm_graph **out)
{
    mm_graph *g = calloc(1, sizeof *g);
    *out = g;
    return g ? MM_OK : MM_ERR_NOMEM;
}

mm_status mm_graph_capture_begin(mm_graph *g)
{
    cudaError_t e;

    (void)g;
    if (g_capturing)
        return MM_ERR_STATE;   /* one capture at a time (one plan at a time) */
    e = cudaStreamBeginCapture(g_st[MM_ST_COMPUTE],
                               cudaStreamCaptureModeThreadLocal);
    if (e != cudaSuccess) {
        MM_LOGE("cuda: StreamBeginCapture: %s", cudaGetErrorString(e));
        return MM_ERR_CUDA;
    }
    /* The flag flips only after a successful begin: the kernel launchers
     * read it (mm_cuda_capturing) to skip the per-op stream syncs and
     * host readbacks that a capturing stream cannot legally do. */
    g_capturing = 1;
    return MM_OK;
}

/* Apply an L2 access-policy window to one kernel node. */
static cudaError_t set_node_window(cudaGraph_t graph, cudaGraphNode_t node,
                                   const void *base, size_t bytes,
                                   int hit_pct)
{
    cudaKernelNodeParams p;
    cudaLaunchAttribute attrs[1];
    cudaError_t e;

    e = cudaGraphGetKernelNodeParams(graph, node, &p);
    if (e != cudaSuccess)
        return e;
    attrs[0].id = cudaLaunchAttributeAccessPolicyWindow;
    attrs[0].val.accessPolicyWindow.base_ptr = (void *)base;
    attrs[0].val.accessPolicyWindow.num_bytes = bytes;
    attrs[0].val.accessPolicyWindow.hitRatio = (float)hit_pct / 100.0f;
    attrs[0].val.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    attrs[0].val.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
    p.extra = attrs;
    p.numAttrs = 1;
    return cudaGraphSetKernelNodeParams(graph, node, &p);
}

mm_status mm_graph_set_node_window(mm_graph *g, const void *base,
                                   size_t bytes, int hit_pct)
{
    if (bytes == 0)
        return MM_OK;
    g->win_valid = 1;
    g->win_base = (void *)base;
    g->win_bytes = bytes;
    g->win_pct = hit_pct;
    return MM_OK;
}

mm_status mm_graph_commit(mm_graph *g, const void *window_base,
                          size_t window_bytes, int hit_pct)
{
    cudaGraph_t graph;
    cudaGraphNode_t *nodes;
    size_t n = 0;
    int use_window = 0;

    /* EndCapture first, and drop the capture flag before anything else:
     * the launchers must go back to their normal per-op sync contract the
     * moment capture ends, even if the subsequent instantiation fails. */
    {
        cudaError_t e = cudaStreamEndCapture(g_st[MM_ST_COMPUTE], &graph);
        g_capturing = 0;
        if (e != cudaSuccess) {
            MM_LOGE("cuda: StreamEndCapture: %s", cudaGetErrorString(e));
            return MM_ERR_CUDA;
        }
    }
    if (window_base && window_bytes > 0) {
        g->win_base = (void *)window_base;
        g->win_bytes = window_bytes;
        g->win_pct = hit_pct;
        g->win_valid = 1;
    }
    use_window = g->win_valid;

    CK(cudaGraphGetNodes(graph, NULL, &n));
    if (n > 0) {
        nodes = malloc(n * sizeof *nodes);
        if (!nodes) {
            cudaGraphDestroy(graph);
            return MM_ERR_NOMEM;
        }
        CK(cudaGraphGetNodes(graph, nodes, &n));
        if (use_window) {
            for (size_t i = 0; i < n; i++) {
                cudaGraphNodeType t;
                if (cudaGraphNodeType(nodes[i], &t) != cudaSuccess)
                    continue;
                if (t == cudaGraphNodeTypeKernel) {
                    if (set_node_window(graph, nodes[i], g->win_base,
                                        g->win_bytes, g->win_pct)
                        != cudaSuccess)
                        MM_LOGW("cuda: node window rejected on node %zu "
                                "(continuing without it)", i);
                }
            }
        }
        free(nodes);
    }
    CK(cudaGraphInstantiate(&g->exec, graph, 0));
    CK(cudaGraphDestroy(graph));
    g->have_exec = 1;
    return MM_OK;
}

mm_status mm_graph_launch(mm_graph *g, mm_stream_id st)
{
    if (!g->have_exec)
        return MM_ERR_STATE;
    CK(cudaGraphLaunch(g->exec, g_st[st]));
    return MM_OK;
}

void mm_graph_destroy(mm_graph *g)
{
    if (!g)
        return;
    if (g->have_exec)
        cudaGraphExecDestroy(g->exec);
    free(g);
}

mm_status mm_device_vram_free(size_t *bytes_out)
{
    size_t free_b, total;
    CK(cudaMemGetInfo(&free_b, &total));
    (void)total;
    *bytes_out = free_b;
    return MM_OK;
}

mm_status mm_device_sync_all(void)
{
    CK(cudaDeviceSynchronize());
    return MM_OK;
}

int mm_cuda_capturing(void)
{
    return g_capturing;
}

#else /* !MM_WITH_CUDA ------------------------------------------------- */

/* Host build: every CUDA entry point is an honest error stub so the whole
 * tree links and the host test path runs. The engine refuses to start a
 * GPU round on this build. */
typedef struct mm_graph { int unused; } mm_graph;

static int g_warned;

int mm_cuda_enabled(void)
{
    return 0;    /* host reference build: kernels run on the CPU */
}

mm_status mm_cuda_device_count(int *n_out)
{
    *n_out = 0;  /* honest: no CUDA device on this build */
    return MM_OK;
}

static mm_status nocuda(const char *op)
{
    if (!g_warned) {
        MM_LOGE("cuda: %s requested but this build has no CUDA "
                "(rebuild with -DMM_WITH_CUDA=1)", op);
        g_warned = 1;
    }
    return MM_ERR_UNSUPPORTED;
}

mm_status mm_cuda_init(int device_ordinal)
{
    (void)device_ordinal;
    return nocuda("mm_cuda_init");
}
mm_status mm_cuda_shutdown(void) { return MM_OK; }
mm_stream mm_cuda_stream(mm_stream_id id)
{
    mm_stream s;
    s.id = (int)id;
    return s;
}
mm_status mm_event_record(int ev, mm_stream_id st)
{
    (void)ev; (void)st;
    return nocuda("event_record");
}
mm_status mm_event_wait(int ev, mm_stream_id st)
{
    (void)ev; (void)st;
    return nocuda("event_wait");
}
mm_status mm_stream_sync(mm_stream_id st)
{
    (void)st;
    return nocuda("stream_sync");
}
/* CPU reference: both endpoints are host memory, so this is a synchronous
 * host copy. The "async" suffix is a no-op on this build (there is no device
 * stream to enqueue on); the bytes actually move, which is what the loader
 * and any host->host staging path expect. */
mm_status mm_h2d_async(void *dst, const void *src, size_t n, mm_stream_id st)
{
    (void)st;
    if (!dst || !src || n == 0)
        return MM_OK;
    memcpy(dst, src, n);
    return MM_OK;
}
mm_status mm_d2h_async(void *dst, const void *src, size_t n, mm_stream_id st)
{
    (void)st;
    if (!dst || !src || n == 0)
        return MM_OK;
    memcpy(dst, src, n);   /* host reference: the source is already host RAM */
    return MM_OK;
}
/* CPU reference: a synchronous fill. This is what the engine's
 * mm_linstate_reset() (and any pool pre-zero) relies on so that the CPU path
 * starts every run from a deterministic, fully-zeroed recurrence/conv state
 * instead of failing with MM_ERR_UNSUPPORTED. */
mm_status mm_mset_async(void *dst, int value, size_t n, mm_stream_id st)
{
    (void)st;
    if (!dst || n == 0)
        return MM_OK;
    memset(dst, value, n);
    return MM_OK;
}
mm_status mm_graph_create(mm_graph **out)
{
    *out = calloc(1, sizeof **out);
    return *out ? MM_OK : MM_ERR_NOMEM;
}
mm_status mm_graph_capture_begin(mm_graph *g)
{
    (void)g;
    return nocuda("graph_capture_begin");
}
mm_status mm_graph_set_node_window(mm_graph *g, const void *base,
                                   size_t bytes, int hit_pct)
{
    (void)g; (void)base; (void)bytes; (void)hit_pct;
    return MM_OK;   /* harmless no-op on the host build */
}
mm_status mm_graph_commit(mm_graph *g, const void *base, size_t bytes,
                          int hit_pct)
{
    (void)g; (void)base; (void)bytes; (void)hit_pct;
    return nocuda("graph_commit");
}
mm_status mm_graph_launch(mm_graph *g, mm_stream_id st)
{
    (void)g; (void)st;
    return nocuda("graph_launch");
}
void mm_graph_destroy(mm_graph *g) { free(g); }
mm_status mm_device_vram_free(size_t *bytes_out)
{
    (void)bytes_out;
    return nocuda("device_vram_free");
}
mm_status mm_device_sync_all(void)
{
    return nocuda("device_sync_all");
}
int mm_cuda_capturing(void)
{
    return 0;   /* no capture state exists on the host build */
}

#endif /* MM_WITH_CUDA */