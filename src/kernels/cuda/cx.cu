/*
 * CUDA kernel launch path — GPU build only (nvcc, -DMM_WITH_CUDA).
 *
 * This file is the GPU half of the kernel dispatcher contract (mimfer/
 * kernels.h): the same kx_op_* symbols the CPU reference provides in
 * src/kernels/cpu/cx.c, implemented as CUDA kernel launches on the
 * runtime layer's compute stream (src/cuda/cuda_rt.c). The dispatcher
 * (src/kernels/kx.c), the plans and the engine are untouched: they only
 * ever saw the dispatcher ABI, which is exactly the point of the
 * stateless mm_kcall design.
 *
 * Parity-first (binding, see READ_MEMORY.md): every kernel mirrors the
 * CPU reference's math 1:1 —
 *   - bf16 in / bf16 out, f32 accumulation,
 *   - the SAME loop order as cx.c (thread-per-row/column where the
 *     reference is row-sequential; sequential inner loops for the
 *     recurrence ops CONV4 / LIN_PRE / LIN_DEC),
 *   - no atomics anywhere (deterministic by construction),
 * so the parity test (tests/cuda/parity_test.c) gates the GPU output at
 * 1 bf16 ulp per element for computed ops and bit-exact for pure data
 * movement (EMBED / KVSTORE / COPY) and sampled tokens (SAMPLE).
 *
 * Build flag: compile with -fmad=false. The CPU reference (plain gcc,
 * baseline x86-64) can only emit mul+add; allowing FMA contraction here
 * would change the f32 accumulation rounding and blow the 1-ulp bf16
 * gate. FMA / tensor-core paths are the Blackwell optimization step and
 * must each re-pass the parity test.
 *
 * Latency model: the reference launchers are per-op synchronous
 * (cudaStreamSynchronize after every launch) so per-op mm_status errors
 * propagate exactly like the CPU reference's returns. The
 * per-round-boundary sync fast path (cuda_rt.h) is the optimization step.
 *
 * Control buffer: per-round scalars ride in e->ctrl_dev (device); the
 * caller (scheduler / parity test) orders the ctrl H2D on the xfer
 * stream before the compute-stream launches (event wait). The sampling
 * POLICY and the slot count are read from the host mirror e->ctrl —
 * the scheduler fills both in the same round.
 *
 * KV rows: single-slot reference — slot 0's block table, exactly like
 * the CPU reference (kx.h scope note).
 *
 * EMBED range check: out-of-range tokens are clamped to row 0 on device
 * and a flag is set in the first 16 bytes of the activation `part`
 * buffer (split-K scratch, unused by the reference kernels) so the
 * launcher can return MM_ERR_RANGE — no extra cudaMalloc, the
 * four-allocation model (alloc.h) stays intact. (Graph-captured plans
 * skip the host readback of that flag — a D2H into stack memory is not
 * a legal graph node — so graph mode does not surface MM_ERR_RANGE from
 * EMBED; the clamp still happens, and the check is live in direct
 * dispatch and the parity test.)
 *
 * Graph capture: launchers stay capture-legal through mm_cuda_capturing
 * (cuda_rt.h) — the per-op stream sync and the EMBED flag readback are
 * suspended while the compute stream is in capture mode, so plans can
 * be captured into CUDA graphs at load (plan.c capture_one).
 */
#include <cuda_runtime.h>
#include <stdlib.h>
#include <math.h>

#include "mimfer/kernels.h"
#include "mimfer/engine.h"
#include "mimfer/tensor.h"
#include "mimfer/cuda_rt.h"
#include "kx.h"   /* kx_cuda_oppref prototype (C linkage for the .cu def) */

/* ------------------------------------------------------------- helpers */

/* Compute stream of the runtime layer: every kernel launch, per the
 * cuda_rt.h contract. */
static cudaStream_t kx_st(void)
{
    return mm_cuda_stream_raw(MM_ST_COMPUTE);
}

/* Wait for the compute stream and surface any launch / device error as
 * the launcher's mm_status (see the header note on the per-op sync
 * model). During a graph capture the stream sync is skipped: it is
 * illegal on a stream in capture mode, and the per-op error contract is
 * suspended for the capture's duration (cuda_rt.h mm_cuda_capturing).
 * Launch errors still surface via cudaGetLastError. */
static mm_status kx_sync(void)
{
    cudaError_t er;

    if (!mm_cuda_capturing()) {
        er = cudaStreamSynchronize(kx_st());
        if (er != cudaSuccess) {
            MM_LOGE("cuda: stream sync: %s", cudaGetErrorString(er));
            return MM_ERR_CUDA;
        }
    }
    er = cudaGetLastError();
    if (er != cudaSuccess) {
        MM_LOGE("cuda: last error: %s", cudaGetErrorString(er));
        return MM_ERR_CUDA;
    }
    return MM_OK;
}

/* Propagate a raw CUDA API error (host-side calls: memset/copy). */
#define CU_CK(call)                                                        \
    do {                                                                   \
        cudaError_t _er = (call);                                          \
        if (_er != cudaSuccess) {                                          \
            MM_LOGE("cuda: %s (line %d): %s", #call, __LINE__,             \
                    cudaGetErrorString(_er));                              \
            return MM_ERR_CUDA;                                            \
        }                                                                  \
    } while (0)

/* bf16 accessors (device side); the host side uses tensor.h directly. */
static __device__ inline float b16_rd(const uint16_t *p)
{
    return mm_f32_from_bf16(*p);
}
static __device__ inline void b16_wr(uint16_t *p, float f)
{
    *p = mm_bf16_from_f32(f);
}

/* -------------------------------------------------------------- EMBED */

/* x[m] = embed[tok[m]]: one block per token row, 8 bf16 (uint4) per
 * thread. Tokens: prefill toks[] / decode slot_tok[] (the CPU reference's
 * rule). Out-of-range token: clamped to row 0 + err flag (see header). */
__global__ void k_embed(const mm_ctrl *c, const uint16_t *w, uint16_t *x,
                        uint32_t M, uint32_t H, uint32_t V, uint32_t *err)
{
    uint32_t m = blockIdx.x;
    if (m >= M)
        return;
    const uint32_t *toks = c->n_slots ? c->slot_tok : c->toks;
    uint32_t t = toks[m];
    if (t >= V) {
        t = 0;
        *err = 1;
    }
    const uint4 *src = (const uint4 *)(w + (size_t)t * H);
    uint4 *dst = (uint4 *)(x + (size_t)m * H);
    for (uint32_t i = threadIdx.x; i < H / 8; i += blockDim.x)
        dst[i] = src[i];
}

mm_status kx_op_embed(const mm_kcall *kc)
{
    const mm_op_desc *op = kc->op;
    const mm_engine *e = kc->e;
    const uint32_t M = op->n0, H = op->n1, V = e->mc.vocab;
    uint32_t *flag = (uint32_t *)e->ab.part;   /* error-flag area (header) */
    uint32_t host_flag = 0;

    if (!flag)
        return MM_ERR_STATE;
    CU_CK(cudaMemsetAsync(flag, 0, 4, kx_st()));
    {
        dim3 grid(M);
        dim3 block(H / 8 < 256 ? H / 8 : 256);
        k_embed<<<grid, block, 0, kx_st()>>>(e->ctrl_dev,
                                             (const uint16_t *)op->a,
                                             (uint16_t *)op->c, M, H, V,
                                             flag);
    }
    MM_CHECK(kx_sync());
    /* The flag readback is a host round trip: it cannot be captured
     * (device->host memcpy into stack memory is not a legal graph node),
     * so a plan captured into a graph does not surface MM_ERR_RANGE from
     * EMBED -- the clamp + device flag write still happen, and the check
     * is fully live in direct dispatch (--no-graph) and the parity test.
     */
    if (!mm_cuda_capturing()) {
        CU_CK(cudaMemcpyAsync(&host_flag, flag, 4,
                              cudaMemcpyDeviceToHost, kx_st()));
        MM_CHECK(kx_sync());
        if (host_flag)
            return MM_ERR_RANGE;    /* CPU reference: MM_ERR_RANGE too */
    }
    return MM_OK;
}

/* ------------------------------------------------------------- RMSNORM */

/* y = rmsnorm(x, gain), one thread per row, f32 sum in the CPU
 * reference's exact order (i = 0..H-1), the reference's
 * 1/sqrt (not rsqrtf) and the same left-associative scale chain. */
__global__ void k_rmsnorm(const uint16_t *x, const uint16_t *g, uint16_t *y,
                          uint32_t M, uint32_t H, float eps)
{
    uint32_t m = blockIdx.x;
    if (m >= M)
        return;
    const uint16_t *xr = x + (size_t)m * H;
    uint16_t *yr = y + (size_t)m * H;
    float ss = 0.0f;
    for (uint32_t i = 0; i < H; i++) {
        float v = b16_rd(&xr[i]);
        ss += v * v;
    }
    float inv = 1.0f / sqrtf(ss / (float)H + eps);
    for (uint32_t i = 0; i < H; i++)
        b16_wr(&yr[i], b16_rd(&xr[i]) * inv * b16_rd(&g[i]));
}

mm_status kx_op_rmsnorm(const mm_kcall *kc)
{
    const mm_op_desc *op = kc->op;
    const mm_engine *e = kc->e;
    dim3 grid(op->n0);
    dim3 block(1);
    k_rmsnorm<<<grid, block, 0, kx_st()>>>((const uint16_t *)op->a,
                                           (const uint16_t *)op->b,
                                           (uint16_t *)op->c,
                                           op->n0, op->n1, e->mc.norm_eps);
    return kx_sync();
}

/* --------------------------------------------------------- ADD_RMSNORM */

/* res += delta (in place), then y = rmsnorm(res, gain) — the reference's
 * two steps per row, same order. a is the in-place residual (const
 * stripped, as in the CPU reference). */
__global__ void k_add_rmsnorm(uint16_t *res, const uint16_t *delta,
                              uint16_t *y, const uint16_t *g,
                              uint32_t M, uint32_t H, float eps)
{
    uint32_t m = blockIdx.x;
    if (m >= M)
        return;
    uint16_t *rr = res + (size_t)m * H;
    const uint16_t *dr = delta + (size_t)m * H;
    uint16_t *yr = y + (size_t)m * H;
    for (uint32_t i = 0; i < H; i++)
        b16_wr(&rr[i], b16_rd(&rr[i]) + b16_rd(&dr[i]));
    float ss = 0.0f;
    for (uint32_t i = 0; i < H; i++) {
        float v = b16_rd(&rr[i]);
        ss += v * v;
    }
    float inv = 1.0f / sqrtf(ss / (float)H + eps);
    for (uint32_t i = 0; i < H; i++)
        b16_wr(&yr[i], b16_rd(&rr[i]) * inv * b16_rd(&g[i]));
}

mm_status kx_op_add_rmsnorm(const mm_kcall *kc)
{
    const mm_op_desc *op = kc->op;
    const mm_engine *e = kc->e;
    dim3 grid(op->n0);
    dim3 block(1);
    k_add_rmsnorm<<<grid, block, 0, kx_st()>>>((uint16_t *)(uintptr_t)op->a,
                                               (const uint16_t *)op->b,
                                               (uint16_t *)op->c,
                                               (const uint16_t *)op->d,
                                               op->n0, op->n1,
                                               e->mc.norm_eps);
    return kx_sync();
}

/* ------------------------------------------------------------- SILU_MUL */

/* y[i] = silu(gate[i]) * up[i], one thread per element, the reference's
 * exact silu formula. */
__global__ void k_silu_mul(const uint16_t *gate, const uint16_t *up,
                           uint16_t *y, uint32_t n)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    float g = b16_rd(&gate[i]);
    float s = g / (1.0f + expf(-g));      /* silu(gate) */
    y[i] = mm_bf16_from_f32(s * b16_rd(&up[i]));
}

mm_status kx_op_silu_mul(const mm_kcall *kc)
{
    const mm_op_desc *op = kc->op;
    uint32_t n = op->n0 * op->n1, block = 256;
    uint32_t grid = (n + block - 1) / block;
    k_silu_mul<<<grid, block, 0, kx_st()>>>((const uint16_t *)op->a,
                                            (const uint16_t *)op->b,
                                            (uint16_t *)op->c, n);
    return kx_sync();
}

/* ------------------------------------------------ GEMM_BF16 / QKV / LMHEAD
 * y[m][r] = sum_k x[m][k] * W[r][k]: row-major x [M][K], W [cols][K]
 * (out features x in features), row-major y [M][cols]. One thread per
 * output element; the k loop in the CPU reference's order (k = 0..K-1,
 * f32 accumulate, no atomics). QKV and LMHEAD are the same math with the
 * plan's shape (the reference has one GEMM workhorse, as the CPU does).
 * Tensor-core / split-K variants are the Blackwell step. */
__global__ void k_gemm_bf16(const uint16_t *x, const uint16_t *w,
                            uint16_t *y, uint32_t M, uint32_t K,
                            uint32_t cols)
{
    uint32_t r = blockIdx.x, m = blockIdx.y;
    if (m >= M)
        return;
    const uint16_t *xr = x + (size_t)m * K;
    const uint16_t *wr = w + (size_t)r * K;
    float acc = 0.0f;
    for (uint32_t k = 0; k < K; k++)
        acc += b16_rd(&xr[k]) * b16_rd(&wr[k]);
    y[(size_t)m * cols + r] = mm_bf16_from_f32(acc);
}

static mm_status kx_gemm_bf16_launch(const mm_kcall *kc)
{
    const mm_op_desc *op = kc->op;
    dim3 grid(op->n2);
    dim3 block(1);

    grid.y = op->n0;
    k_gemm_bf16<<<grid, block, 0, kx_st()>>>((const uint16_t *)op->a,
                                             (const uint16_t *)op->b,
                                             (uint16_t *)op->c,
                                             op->n0, op->n1, op->n2);
    return kx_sync();
}

mm_status kx_op_gemm_bf16(const mm_kcall *kc)
{
    return kx_gemm_bf16_launch(kc);
}

mm_status kx_op_qkv(const mm_kcall *kc)
{
    return kx_gemm_bf16_launch(kc);
}

mm_status kx_op_lmhead(const mm_kcall *kc)
{
    return kx_gemm_bf16_launch(kc);
}

/* ------------------------------------------------------------- GEMM_F4 */

/* NVFP4 weight GEMM: y = x @ W^T with W dequantized in-loop (payload +
 * e4m3 group scales) through the shared mm_fp4_wget decoder (tensor.h),
 * so the format math is byte-for-byte the CPU reference's. One thread per
 * output element, k = 0..K-1 in order. (The M<=8 split-K gemv fast path
 * is the Blackwell step; this reference shape is what the parity test
 * validates.) */
__global__ void k_gemm_f4(const uint16_t *x, const mm_wt W, uint16_t *y,
                          uint32_t M, uint32_t K, uint32_t cols)
{
    uint32_t r = blockIdx.x, m = blockIdx.y;
    if (m >= M)
        return;
    const uint16_t *xr = x + (size_t)m * K;
    float acc = 0.0f;
    for (uint32_t k = 0; k < K; k++)
        acc += b16_rd(&xr[k]) * mm_fp4_wget(&W, (int)r, (int)k);
    y[(size_t)m * cols + r] = mm_bf16_from_f32(acc);
}

mm_status kx_op_gemm_f4(const mm_kcall *kc)
{
    const mm_op_desc *op = kc->op;
    mm_wt W;
    dim3 grid(op->n2);
    dim3 block(1);

    if (op->n1 % 16 != 0)
        return MM_ERR_SHAPE;      /* e4m3 group scale spans 16 columns */
    W.data = op->b;
    W.scale = op->d;
    W.rows = op->n2;
    W.cols = op->n1;
    grid.y = op->n0;
    k_gemm_f4<<<grid, block, 0, kx_st()>>>((const uint16_t *)op->a, W,
                                           (uint16_t *)op->c,
                                           op->n0, op->n1, op->n2);
    return kx_sync();
}

/* ----------------------------------------------------------------- ROPE */

/* Partial in-place RoPE over the first rope_dim dims of every q and k
 * row. One thread per (token, head, pair): element i is paired with
 * i + rd/2 at angle pos * rope_inv[i] — the effective table (plain
 * RoPE or the YaRN NTK-by-parts blend) arrives in the control buffer,
 * so CPU and CUDA consume bit-identical frequencies; the rotated
 * components carry the attention scale rope_mscale (plain 1.0). Host
 * vs device libm sinf/cosf differ by <= 2 f32 ulp; the parity gate is
 * 1 bf16 ulp. Rows: q = [M][q_heads][hd], k = [M][kv_heads][hd]. */
__global__ void k_rope(const mm_ctrl *c, uint16_t *q, uint16_t *k,
                       uint32_t M, uint32_t rd, uint32_t hd, uint32_t qdim,
                       uint32_t kvdim, uint32_t qh, uint32_t kh)
{
    uint32_t i = blockIdx.x, h = blockIdx.y, m = blockIdx.z;
    if (m >= M)
        return;
    uint16_t *row;
    if (h < qh)
        row = q + (size_t)m * qdim + (size_t)h * hd;
    else
        row = k + (size_t)m * kvdim + (size_t)(h - qh) * hd;
    uint32_t pos = c->n_slots ? c->slot_pos[m] : c->pos0 + m;
    float ang = (float)pos * c->rope_inv[i];
    float s = sinf(ang), co = cosf(ang);
    float x0 = b16_rd(&row[i]);
    float x1 = b16_rd(&row[i + rd / 2]);
    float ms = c->rope_mscale;
    b16_wr(&row[i], (x0 * co - x1 * s) * ms);
    b16_wr(&row[i + rd / 2], (x1 * co + x0 * s) * ms);
}

mm_status kx_op_rope(const mm_kcall *kc)
{
    const mm_op_desc *op = kc->op;
    const mm_engine *e = kc->e;
    const mm_model_cfg *mc = &e->mc;
    const uint32_t M = op->n0, rd = op->n1;
    const uint32_t hd = mc->head_dim, qdim = mc->q_heads * hd;
    const uint32_t kvdim = mc->kv_heads * hd;
    dim3 grid(rd / 2, mc->q_heads + mc->kv_heads, M);
    dim3 block(1);

    if (!op->a || !op->b || rd < 2 || rd > hd)
        return MM_ERR_STATE;
    k_rope<<<grid, block, 0, kx_st()>>>(e->ctrl_dev,
                                        (uint16_t *)(uintptr_t)op->a,
                                        (uint16_t *)(uintptr_t)op->b,
                                        M, rd, hd, qdim, kvdim,
                                        mc->q_heads, mc->kv_heads);
    return kx_sync();
}

/* -------------------------------------------------------------- KVSTORE */

/* Store k/v rows for toks [a..a+M) into the paged pool. Single-slot
 * reference: slot 0's block table (the CPU reference's scope), block id
 * for position pos is tab[pos/64] (ids run 1..n_blocks, id 0 = "no
 * block"); the pool is indexed 0..n_blocks-1, so the id is converted
 * before the offset math. Layout [kv_head][block][tok][hd]. One thread
 * per (token, kv head), 8 bf16 per iteration. */
__global__ void k_kvstore(const mm_ctrl *c, const uint16_t *k,
                          const uint16_t *v, uint16_t *kb, uint16_t *vb,
                          const uint32_t *tab, uint32_t M, uint32_t hd,
                          uint32_t kvdim, size_t head_stride,
                          size_t block_stride, size_t tok_stride)
{
    uint32_t m = blockIdx.x, h = blockIdx.y;
    if (m >= M)
        return;
    uint32_t pos = c->n_slots ? c->slot_pos[m] : c->pos0 + m;
    uint32_t bid = tab[pos / MM_KV_BLOCK_TOK];
    uint32_t blk = bid ? bid - 1 : 0;   /* 1-based id -> 0-based index */
    size_t off = (size_t)h * (head_stride / 2)
               + (size_t)blk * (block_stride / 2)
               + (pos % MM_KV_BLOCK_TOK) * (tok_stride / 2);
    const uint4 *ks = (const uint4 *)(k + (size_t)m * kvdim + (size_t)h * hd);
    uint4 *kd = (uint4 *)(kb + off);
    const uint4 *vs = (const uint4 *)(v + (size_t)m * kvdim + (size_t)h * hd);
    uint4 *vd = (uint4 *)(vb + off);
    for (uint32_t i = 0; i < hd / 8; i++) {
        kd[i] = ks[i];
        vd[i] = vs[i];
    }
}

mm_status kx_op_kvstore(const mm_kcall *kc)
{
    const mm_op_desc *op = kc->op;
    const mm_engine *e = kc->e;
    const mm_kvpool *kv = e->kv;
    const mm_model_cfg *mc = &e->mc;
    const uint32_t M = op->n1, hd = mc->head_dim;
    const uint32_t kvdim = mc->kv_heads * hd;
    const uint32_t li = op->n0;      /* layer, per the CPU reference */
    dim3 grid(M, mc->kv_heads);
    dim3 block(1);

    if (!kv || !op->c || !kv->v_base[li])
        return MM_ERR_STATE;
    if (hd % 8 != 0)
        return MM_ERR_SHAPE;
    k_kvstore<<<grid, block, 0, kx_st()>>>(e->ctrl_dev,
                                           (const uint16_t *)op->a,
                                           (const uint16_t *)op->b,
                                           (uint16_t *)op->c,
                                           (uint16_t *)(uintptr_t)kv->v_base[li],
                                           kv->blk_tab_dev, M, hd, kvdim,
                                           kv->head_stride,
                                           kv->block_stride,
                                           kv->tok_stride);
    return kx_sync();
}

/* ------------------------------------------------------------- ATT ops */

/* One paged KV row (head_dim bf16): slot 0's block table, the CPU
 * reference's kv_row() offset math in bf16 units. Block ids run
 * 1..n_blocks (id 0 == "no block"); the pool is indexed 0..n_blocks-1. */
static __device__ inline const uint16_t *kv_row_dev(const uint16_t *base,
                                                    const uint32_t *tab,
                                                    uint32_t kh, uint32_t t,
                                                    size_t head_stride,
                                                    size_t block_stride,
                                                    size_t tok_stride)
{
    uint32_t bid = tab[t / MM_KV_BLOCK_TOK];
    uint32_t blk = bid ? bid - 1 : 0;   /* 1-based id -> 0-based index */
    size_t off = (size_t)kh * (head_stride / 2)
               + (size_t)blk * (block_stride / 2)
               + (t % MM_KV_BLOCK_TOK) * (tok_stride / 2);
    return base + off;
}

/* Paged GQA decode. One thread per (slot row, q head): the CPU
 * reference's three passes verbatim — max, sum, P@V — with the q.k dot
 * product recomputed on every pass (no caching, no atomics), so the f32
 * sum order is the reference's order. len = slot_len[m]; kv head = h/g
 * (GQA). The optimized CTA-per-(slot,head) version is the Blackwell step
 * and must re-pass parity. */
__global__ void k_att_decode(const mm_ctrl *c, const uint16_t *q,
                             uint16_t *o, const uint16_t *kb,
                             const uint16_t *vb, const uint32_t *tab,
                             uint32_t M, uint32_t qh, uint32_t hd,
                             uint32_t qdim, float sc, uint32_t g,
                             size_t head_stride, size_t block_stride,
                             size_t tok_stride)
{
    uint32_t m = blockIdx.x, h = blockIdx.y;
    if (m >= M)
        return;
    uint32_t len = c->slot_len[m];
    uint32_t kh = h / g;
    const uint16_t *qr = q + (size_t)m * qdim + (size_t)h * hd;
    float mx = -INFINITY, sum = 0.0f;
    for (uint32_t t = 0; t < len; t++) {          /* pass 1: max */
        const uint16_t *kr = kv_row_dev(kb, tab, kh, t, head_stride,
                                        block_stride, tok_stride);
        float d = 0.0f;
        for (uint32_t i = 0; i < hd; i++)
            d += b16_rd(&qr[i]) * b16_rd(&kr[i]);
        if ((d *= sc) > mx)
            mx = d;
    }
    for (uint32_t t = 0; t < len; t++) {          /* pass 2: sum */
        const uint16_t *kr = kv_row_dev(kb, tab, kh, t, head_stride,
                                        block_stride, tok_stride);
        float d = 0.0f;
        for (uint32_t i = 0; i < hd; i++)
            d += b16_rd(&qr[i]) * b16_rd(&kr[i]);
        sum += expf(d * sc - mx);
    }
    {
        float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
        uint16_t *orow = o + (size_t)m * qdim + (size_t)h * hd;
        for (uint32_t i = 0; i < hd; i++) {       /* pass 3: P @ V */
            float acc = 0.0f;
            for (uint32_t t = 0; t < len; t++) {
                const uint16_t *kr = kv_row_dev(kb, tab, kh, t, head_stride,
                                                block_stride, tok_stride);
                float d = 0.0f;
                for (uint32_t j = 0; j < hd; j++)
                    d += b16_rd(&qr[j]) * b16_rd(&kr[j]);
                const uint16_t *vr = kv_row_dev(vb, tab, kh, t, head_stride,
                                                block_stride, tok_stride);
                acc += expf(d * sc - mx) * inv * b16_rd(&vr[i]);
            }
            b16_wr(&orow[i], acc);
        }
    }
}

mm_status kx_op_att_decode(const mm_kcall *kc)
{
    const mm_op_desc *op = kc->op;
    const mm_engine *e = kc->e;
    const mm_model_cfg *mc = &e->mc;
    const uint32_t M = op->n1, hd = mc->head_dim, qdim = mc->q_heads * hd;
    const void *kb = op->a, *vb = e->kv ? e->kv->v_base[op->layer] : 0;
    const float sc = 1.0f / sqrtf((float)hd);
    const uint32_t g = mc->q_heads / mc->kv_heads;
    dim3 grid(M, mc->q_heads);
    dim3 block(1);

    if (!e->kv || !vb)
        return MM_ERR_STATE;
    k_att_decode<<<grid, block, 0, kx_st()>>>(e->ctrl_dev,
                                              (const uint16_t *)op->b,
                                              (uint16_t *)op->c,
                                              (const uint16_t *)kb,
                                              (const uint16_t *)vb,
                                              e->kv->blk_tab_dev,
                                              M, mc->q_heads, hd, qdim, sc, g,
                                              e->kv->head_stride,
                                              e->kv->block_stride,
                                              e->kv->tok_stride);
    return kx_sync();
}

/* Causal prefill attention. Same structure as decode; row m attends to
 * positions 0..pos (inclusive), pos = ctrl_pos(m) from the device
 * control buffer (prefill: pos0 + m; decode: slot_pos[m]). */
__global__ void k_att_prefill(const mm_ctrl *c, const uint16_t *q,
                              uint16_t *o, const uint16_t *kb,
                              const uint16_t *vb, const uint32_t *tab,
                              uint32_t M, uint32_t qh, uint32_t hd,
                              uint32_t qdim, float sc, uint32_t g,
                              size_t head_stride, size_t block_stride,
                              size_t tok_stride)
{
    uint32_t m = blockIdx.x, h = blockIdx.y;
    if (m >= M)
        return;
    uint32_t pos = c->n_slots ? c->slot_pos[m] : c->pos0 + m;
    uint32_t kh = h / g;
    const uint16_t *qr = q + (size_t)m * qdim + (size_t)h * hd;
    float mx = -INFINITY, sum = 0.0f;
    for (uint32_t t = 0; t <= pos; t++) {         /* pass 1: max */
        const uint16_t *kr = kv_row_dev(kb, tab, kh, t, head_stride,
                                        block_stride, tok_stride);
        float d = 0.0f;
        for (uint32_t i = 0; i < hd; i++)
            d += b16_rd(&qr[i]) * b16_rd(&kr[i]);
        if ((d *= sc) > mx)
            mx = d;
    }
    for (uint32_t t = 0; t <= pos; t++) {         /* pass 2: sum */
        const uint16_t *kr = kv_row_dev(kb, tab, kh, t, head_stride,
                                        block_stride, tok_stride);
        float d = 0.0f;
        for (uint32_t i = 0; i < hd; i++)
            d += b16_rd(&qr[i]) * b16_rd(&kr[i]);
        sum += expf(d * sc - mx);
    }
    {
        float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
        uint16_t *orow = o + (size_t)m * qdim + (size_t)h * hd;
        for (uint32_t i = 0; i < hd; i++) {       /* pass 3: P @ V */
            float acc = 0.0f;
            for (uint32_t t = 0; t <= pos; t++) {
                const uint16_t *kr = kv_row_dev(kb, tab, kh, t, head_stride,
                                                block_stride, tok_stride);
                float d = 0.0f;
                for (uint32_t j = 0; j < hd; j++)
                    d += b16_rd(&qr[j]) * b16_rd(&kr[j]);
                const uint16_t *vr = kv_row_dev(vb, tab, kh, t, head_stride,
                                                block_stride, tok_stride);
                acc += expf(d * sc - mx) * inv * b16_rd(&vr[i]);
            }
            b16_wr(&orow[i], acc);
        }
    }
}

mm_status kx_op_att_prefill(const mm_kcall *kc)
{
    const mm_op_desc *op = kc->op;
    const mm_engine *e = kc->e;
    const mm_model_cfg *mc = &e->mc;
    const uint32_t M = op->n0, hd = mc->head_dim, qdim = mc->q_heads * hd;
    const void *kb = op->a, *vb = e->kv ? e->kv->v_base[op->layer] : 0;
    const float sc = 1.0f / sqrtf((float)hd);
    const uint32_t g = mc->q_heads / mc->kv_heads;
    dim3 grid(M, mc->q_heads);
    dim3 block(1);

    if (!e->kv || !vb)
        return MM_ERR_STATE;
    k_att_prefill<<<grid, block, 0, kx_st()>>>(e->ctrl_dev,
                                               (const uint16_t *)op->b,
                                               (uint16_t *)op->c,
                                               (const uint16_t *)kb,
                                               (const uint16_t *)vb,
                                               e->kv->blk_tab_dev,
                                               M, mc->q_heads, hd, qdim, sc, g,
                                               e->kv->head_stride,
                                               e->kv->block_stride,
                                               e->kv->tok_stride);
    return kx_sync();
}

/* -------------------------------------------------------- SOFTMAX_CAUSAL */

/* In-place causal row-softmax of a [rows][cols] bf16 score tile. Column j
 * holds absolute position kv_start + j; row i is masked where that
 * position exceeds kv_start + i (kv_start cancels in the comparison —
 * the reference's condition, same math). One thread per row, the
 * reference's two-pass max/sum, no scratch buffer (the reference's tmp[]
 * is recomputed in place here, identical values). */
__global__ void k_softmax_causal(uint16_t *S, uint32_t rows, uint32_t cols,
                                 uint32_t kv_start)
{
    uint32_t i = blockIdx.x;
    if (i >= rows)
        return;
    uint16_t *row = S + (size_t)i * cols;
    float mx = -INFINITY;
    for (uint32_t j = 0; j < cols; j++) {
        float v = (kv_start + j <= kv_start + i) ? b16_rd(&row[j])
                                                 : -INFINITY;
        if (v > mx)
            mx = v;
    }
    float sum = 0.0f;
    for (uint32_t j = 0; j < cols; j++) {
        float v = (kv_start + j <= kv_start + i) ? b16_rd(&row[j])
                                                 : -INFINITY;
        sum += (v == -INFINITY) ? 0.0f : expf(v - mx);
    }
    float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (uint32_t j = 0; j < cols; j++) {
        float v = (kv_start + j <= kv_start + i) ? b16_rd(&row[j]) : 0.0f;
        b16_wr(&row[j], v * inv);
    }
}

mm_status kx_op_softmax_causal(const mm_kcall *kc)
{
    const mm_op_desc *op = kc->op;
    uint32_t rows = op->n0, cols = op->n1, kv_start = op->n2;
    dim3 grid(rows);
    dim3 block(1);

    if (!op->a || cols == 0)
        return MM_ERR_STATE;
    k_softmax_causal<<<grid, block, 0, kx_st()>>>((uint16_t *)(uintptr_t)op->a,
                                                  rows, cols, kv_start);
    return kx_sync();
}

/* ---------------------------------------------------------------- CONV4 */

/* Causal depthwise conv over the qkv channels + conv-state update. One
 * thread per channel c, tokens m = 0..M-1 processed SEQUENTIALLY (the
 * state update for token m+1 depends on token m — the reference's exact
 * order). The trailing g/beta channels (ch..in_dim) pass through, as in
 * the reference (they are the DeltaNet gates, not mixer input). */
__global__ void k_conv4(const uint16_t *x, const uint16_t *w, uint16_t *out,
                        uint16_t *st, uint32_t M, uint32_t in_dim,
                        uint32_t ch, uint32_t K)
{
    uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ch)
        return;
    for (uint32_t m = 0; m < M; m++) {
        uint32_t base = (size_t)m * in_dim + c;
        float cur = b16_rd(&x[base]);
        float acc = 0.0f;
        for (uint32_t i = 0; i < K - 1; i++)
            acc += b16_rd(&w[(size_t)c * K + i]) *
                   b16_rd(&st[(size_t)i * ch + c]);
        acc += b16_rd(&w[(size_t)c * K + (K - 1)]) * cur;
        out[base] = mm_bf16_from_f32(acc);
        for (uint32_t i = 0; i < K - 2; i++)
            st[(size_t)i * ch + c] =
                mm_bf16_from_f32(b16_rd(&st[(size_t)(i + 1) * ch + c]));
        st[(size_t)(K - 2) * ch + c] = mm_bf16_from_f32(cur);
    }
}

mm_status kx_op_conv4(const mm_kcall *kc)
{
    const mm_op_desc *op = kc->op;
    const mm_engine *e = kc->e;
    const mm_model_cfg *mc = &e->mc;
    const uint32_t M = op->n0, in_dim = op->n1, K = mc->conv_k;
    const uint32_t ch = 2u * mc->lin_k_heads * mc->lin_k_dim
                      + mc->lin_v_heads * mc->lin_v_dim;
    uint32_t block = 256;
    uint32_t grid = (ch + block - 1) / block;

    if (!op->a || !op->b || !op->c || !op->d || K < 2 || ch > in_dim)
        return MM_ERR_STATE;
    k_conv4<<<grid, block, 0, kx_st()>>>((const uint16_t *)op->a,
                                         (const uint16_t *)op->b,
                                         (uint16_t *)op->c,
                                         (uint16_t *)(uintptr_t)op->d,
                                         M, in_dim, ch, K);
    return kx_sync();
}

/* ------------------------------------------------------- LIN_PRE / LIN_DEC */

/* Gated-DeltaNet reference recurrence (the CPU reference's fixed-decay
 * form; see src/kernels/kx.h — the GPU kernels are the source of truth
 * for the exact data-dependent gating once the real model is loaded).
 * Per v-head h the fp32 state S[h] is [k_dim][v_dim] and the reference
 * update is
 *
 *   S[h] = decay * S[h] + beta * k_h outer v_h;   o_h = S[h] @ q_h
 *
 * processed over the M new tokens of the qkv row buffer. One thread per
 * v-head; the m loop is SEQUENTIAL per head and each head's operation
 * order (state update then output, per m) is the reference's per-head
 * order exactly — heads are independent, so nothing is reordered.
 * q/k are shared across the v heads that map to one k head (GQA). */
__global__ void k_lin(float *st, const uint16_t *qkv, uint16_t *o,
                      uint32_t M, uint32_t kdim, uint32_t vd,
                      uint32_t vheads, uint32_t kq, uint32_t vdim,
                      uint32_t ch, uint32_t in_dim, uint32_t kvh)
{
    uint32_t h = blockIdx.x;
    if (h >= vheads)
        return;
    uint32_t kh = h / kvh;
    float decay = expf(-0.1f);
    float *S = st + (size_t)h * kdim * vd;
    for (uint32_t m = 0; m < M; m++) {
        const uint16_t *row = qkv + (size_t)m * in_dim;
        const uint16_t *qk = row + kh * kdim;
        const uint16_t *vv = row + 2u * kq + h * vd;
        float beta = b16_rd(&row[ch + vheads + h]);
        for (uint32_t i = 0; i < kdim; i++) {
            float kv = b16_rd(&qk[i]);
            float *Srow = S + (size_t)i * vd;
            for (uint32_t j = 0; j < vd; j++)
                Srow[j] = decay * Srow[j] + beta * kv * b16_rd(&vv[j]);
        }
        {
            uint16_t *orow = o + (size_t)m * vdim + h * vd;
            for (uint32_t j = 0; j < vd; j++) {
                float acc = 0.0f;
                for (uint32_t i = 0; i < kdim; i++)
                    acc += S[(size_t)i * vd + j] * b16_rd(&qk[i]);
                orow[j] = mm_bf16_from_f32(acc);
            }
        }
    }
}

static mm_status kx_lin_launch(const mm_kcall *kc)
{
    const mm_op_desc *op = kc->op;
    const mm_engine *e = kc->e;
    const mm_model_cfg *mc = &e->mc;
    const uint32_t M = op->n0;
    const uint32_t kdim = mc->lin_k_dim, vd = mc->lin_v_dim;
    const uint32_t kq = mc->lin_k_heads * kdim;
    const uint32_t vdim = mc->lin_v_heads * vd;
    const uint32_t ch = 2u * kq + vdim;
    const uint32_t in_dim = ch + 2u * mc->lin_v_heads;
    const uint32_t kvh = mc->lin_v_heads / mc->lin_k_heads;
    dim3 grid(mc->lin_v_heads);
    dim3 block(1);

    if (!op->a || !op->b || !op->c)
        return MM_ERR_STATE;
    k_lin<<<grid, block, 0, kx_st()>>>((float *)(uintptr_t)op->a,
                                       (const uint16_t *)op->b,
                                       (uint16_t *)op->c,
                                       M, kdim, vd, mc->lin_v_heads, kq,
                                       vdim, ch, in_dim, kvh);
    return kx_sync();
}

mm_status kx_op_lin_dec(const mm_kcall *kc)
{
    return kx_lin_launch(kc);
}

mm_status kx_op_lin_pre(const mm_kcall *kc)
{
    return kx_lin_launch(kc);
}

/* ----------------------------------------------------------------- COPY */

/* Device->device copy (residual aliasing): one async D2D memcpy on the
 * compute stream; the CPU reference is a synchronous host memcpy. */
mm_status kx_op_copy(const mm_kcall *kc)
{
    const mm_op_desc *op = kc->op;

    if (!op->a || !op->b)
        return MM_ERR_STATE;
    CU_CK(cudaMemcpyAsync((void *)op->b, op->a, op->n0,
                          cudaMemcpyDeviceToDevice, kx_st()));
    return kx_sync();
}

/* ---------------------------------------------------------------- SAMPLE */

/* Greedy (temperature <= 0 or top_k == 0): one device argmax per row,
 * first-max-wins tie-break — the CPU reference's exact comparison.
 * Non-greedy: the CPU reference's algorithm (temperature -> top-k ->
 * top-p -> the one fixed-point uniform u) runs on the host over a D2H
 * of the logit rows — bit-exact with the reference by construction. An
 * in-kernel top-k/top-p sampler is a future optimization that must
 * re-pass the parity test. Policy + slot count come from the host
 * mirror (e->ctrl); the logits live on device. */
__global__ void k_sample_greedy(const uint16_t *logits, uint32_t *toks,
                                uint32_t M, uint32_t V)
{
    uint32_t m = blockIdx.x;
    if (m >= M)
        return;
    const uint16_t *row = logits + (size_t)m * V;
    uint32_t best = 0;
    float bv = mm_f32_from_bf16(row[0]);
    for (uint32_t i = 1; i < V; i++) {
        const float v = mm_f32_from_bf16(row[i]);
        if (v > bv) {
            bv = v;
            best = i;
        }
    }
    toks[m] = best;
}

/* Host copy of the CPU reference's non-greedy row sampler (cx.c
 * kx_sample_row), kept verbatim: temperature scaling, O(V^2) selection
 * sort (reference-only), top-k truncation, top-p cutoff, then the
 * fixed-point uniform draw. Deterministic in (logits, policy, u). */
static uint32_t kx_sample_row_host(const uint16_t *logits, uint32_t V,
                                   const mm_ctrl *c)
{
    if (V == 0)
        return 0;
    if (c->temperature <= 0.0f || c->top_k == 0) {
        uint32_t best = 0;
        float bv = mm_f32_from_bf16(logits[0]);
        for (uint32_t i = 1; i < V; i++) {
            const float v = mm_f32_from_bf16(logits[i]);
            if (v > bv) {
                bv = v;
                best = i;
            }
        }
        return best;
    }
    float *tmp = (float *)malloc((size_t)V * sizeof *tmp);
    uint32_t *order = (uint32_t *)malloc((size_t)V * sizeof *order);
    if (!tmp || !order) {
        free(tmp);
        free(order);
        return 0;
    }
    const float temp = c->temperature;
    float mx = -INFINITY;
    for (uint32_t i = 0; i < V; i++) {
        tmp[i] = mm_f32_from_bf16(logits[i]) / temp;
        if (tmp[i] > mx)
            mx = tmp[i];
    }
    float sum = 0.0f;
    for (uint32_t i = 0; i < V; i++) {
        tmp[i] = expf(tmp[i] - mx);
        sum += tmp[i];
    }
    const float inv = 1.0f / (sum > 0.0f ? sum : 1.0f);
    for (uint32_t i = 0; i < V; i++) {
        tmp[i] *= inv;
        order[i] = i;
    }
    /* descending order by probability (simple O(V^2); reference only) */
    for (uint32_t i = 0; i < V; i++)
        for (uint32_t j = i + 1; j < V; j++)
            if (tmp[order[j]] > tmp[order[i]]) {
                const uint32_t t = order[i];
                order[i] = order[j];
                order[j] = t;
            }
    uint32_t n = (c->top_k > 0 && c->top_k < V) ? c->top_k : V;
    if (c->top_p < 1.0f) {
        float acc = 0.0f;
        for (uint32_t i = 0; i < n; i++) {
            acc += tmp[order[i]];
            if (acc >= c->top_p) {
                n = i + 1;
                break;
            }
        }
    }
    for (uint32_t i = n; i < V; i++)
        tmp[order[i]] = 0.0f;
    const float u = (float)(c->u & 0x7ffffffful) / (float)(1ull << 31);
    float run = 0.0f;
    uint32_t pick = order[n - 1];
    for (uint32_t i = 0; i < n; i++) {
        run += tmp[order[i]];
        if (u < run) {
            pick = order[i];
            break;
        }
    }
    free(tmp);
    free(order);
    return pick;
}

mm_status kx_op_sample(const mm_kcall *kc)
{
    const mm_op_desc *op = kc->op;
    const mm_engine *e = kc->e;
    const mm_ctrl *h = &e->ctrl;
    const uint32_t V = e->mc.vocab;
    const uint32_t M = h->n_slots ? h->n_slots : op->n0;
    const uint16_t *logits = (const uint16_t *)op->a;
    uint32_t *toks = (uint32_t *)(uintptr_t)op->b;

    if (!logits || !toks || V == 0)
        return MM_ERR_STATE;

    if (h->temperature <= 0.0f || h->top_k == 0) {
        dim3 grid(M);
        dim3 block(1);
        k_sample_greedy<<<grid, block, 0, kx_st()>>>(logits, toks, M, V);
        return kx_sync();
    }

    /* Non-greedy: stage the logit rows to the host, run the reference
     * algorithm there (bit-exact with the CPU path), stage the tokens
     * back. See the function header above. */
    uint16_t *hl = (uint16_t *)malloc((size_t)M * V * 2);
    uint32_t *ht = (uint32_t *)malloc((size_t)M * 4);
    uint32_t m;
    if (!hl || !ht) {
        free(hl);
        free(ht);
        return MM_ERR_NOMEM;
    }
    CU_CK(cudaMemcpyAsync(hl, logits, (size_t)M * V * 2,
                          cudaMemcpyDeviceToHost, kx_st()));
    CU_CK(cudaStreamSynchronize(kx_st()));
    for (m = 0; m < M; m++)
        ht[m] = kx_sample_row_host(hl + (size_t)m * V, V, h);
    CU_CK(cudaMemcpyAsync(toks, ht, (size_t)M * 4,
                          cudaMemcpyHostToDevice, kx_st()));
    free(hl);
    free(ht);
    return kx_sync();
}

/* ------------------------------------------------------------ oppref --- */

/* CUDA coverage query (kx.h): 1 if this translation unit provides a GPU
 * launcher for opcode `op`. Must mirror kx_cpu_oppref() exactly so the
 * parity test can assert the two coverage sets agree. */
int kx_cuda_oppref(int op)
{
    switch (op) {
    case OP_EMBED:
    case OP_LMHEAD:
    case OP_RMSNORM:
    case OP_ADD_RMSNORM:
    case OP_GEMM_F4:
    case OP_GEMM_BF16:
    case OP_QKV:
    case OP_ROPE:
    case OP_KVSTORE:
    case OP_ATT_DECODE:
    case OP_ATT_PREFILL:
    case OP_SOFTMAX_CAUSAL:
    case OP_SILU_MUL:
    case OP_CONV4:
    case OP_LIN_DEC:
    case OP_LIN_PRE:
    case OP_COPY:
    case OP_SAMPLE:
        return 1;
    default:
        return 0;
    }
}