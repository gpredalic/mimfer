/*
 * CPU reference kernels (see src/kernels/kx.h for the scope notes).
 *
 * These are the CPU stand-ins for the kx_op_* launchers declared in
 * mimfer/kernels.h. In a GPU build the .cu files provide the same symbols;
 * here the host reference build links this file so a plan can run end-to-end
 * on the CPU with no device. Correctness and determinism come first: bf16
 * in, f32 accumulate, fixed loop order, no atomics, no hidden randomness.
 *
 * A single source of truth for every operand convention is the op table in
 * mimfer/kernels.h; each function below reads exactly the fields documented
 * there. Per-round scalars (token positions, slot set, sampling policy) come
 * from the host control buffer e->ctrl; the batch is prefill when
 * n_slots == 0 and decode otherwise (see the ctrl_* helpers below).
 */
#include "kx.h"
#include "mimfer/engine.h"
#include "mimfer/tensor.h"
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------- helpers */

static inline float bf16_rd(const uint16_t *p) { return mm_f32_from_bf16(*p); }
static inline void  bf16_wr(uint16_t *p, float f) { *p = mm_bf16_from_f32(f); }

static int is_decode(const mm_ctrl *c) { return c->n_slots > 0; }
/* EMBED input tokens: toks[] for a prefill chunk, slot_tok[] for decode. */
static const uint32_t *ctrl_toks(const mm_ctrl *c)
{ return c->n_slots ? c->slot_tok : c->toks; }
/* Absolute KV position of batch row i. */
static uint32_t ctrl_pos(const mm_ctrl *c, uint32_t i)
{ return c->n_slots ? c->slot_pos[i] : c->pos0 + i; }

/* One KV row (head_dim bf16) for slot 0, kv head `head`, absolute position
 * `pos`, in the [kv_head][block][tok_in_block][head_dim] layout. */
static const uint16_t *kv_row(const mm_kvpool *kv, uint32_t head,
                              const void *base, uint32_t pos)
{
    /* Block ids run 1..n_blocks (id 0 == "no block"); the pool is indexed
     * 0..n_blocks-1, so convert the id to a block index. */
    uint32_t bid = kv->blk_tab_host[pos / MM_KV_BLOCK_TOK];   /* slot 0 */
    uint32_t blk = bid ? bid - 1 : 0;
    size_t off = head * (kv->head_stride / 2)
               + (size_t)blk * (kv->block_stride / 2)
               + (pos % MM_KV_BLOCK_TOK) * (kv->tok_stride / 2);
    return (const uint16_t *)base + off;
}

/* Row-major bf16 GEMM: y[M][cols] = x[M][K] @ W^T, W stored [cols][K].
 * The shared workhorse for LMHEAD / GEMM_BF16 / QKV. */
static void kx_gemm_bf16(const uint16_t *x, const uint16_t *w, uint16_t *y,
                         uint32_t M, uint32_t K, uint32_t cols)
{
    for (uint32_t m = 0; m < M; m++) {
        const uint16_t *xr = x + (size_t)m * K;
        uint16_t *yr = y + (size_t)m * cols;
        for (uint32_t r = 0; r < cols; r++) {
            const uint16_t *wr = w + (size_t)r * K;
            float acc = 0.0f;
            for (uint32_t k = 0; k < K; k++)
                acc += bf16_rd(&xr[k]) * bf16_rd(&wr[k]);
            bf16_wr(&yr[r], acc);
        }
    }
}

/* One RMSNorm row: y = x / rms(x) * gain, rms = sqrt(mean(x^2) + eps). */
static void kx_rms_row(const uint16_t *x, const uint16_t *g, uint16_t *y,
                       uint32_t H, float eps)
{
    float ss = 0.0f;
    for (uint32_t i = 0; i < H; i++) {
        float v = bf16_rd(&x[i]);
        ss += v * v;
    }
    /* 1/sqrt instead of rsqrtf: the latter is a glibc extension, not ISO C99,
     * so it is undeclared under -std=c11. */
    float inv = 1.0f / sqrtf(ss / (float)H + eps);
    for (uint32_t i = 0; i < H; i++)
        bf16_wr(&y[i], bf16_rd(&x[i]) * inv * bf16_rd(&g[i]));
}

/* ---------------------------------------------------------------- COPY */

mm_status kx_op_copy(const mm_kcall *kc)
{
    const mm_op_desc *op = kc->op;
    if (!op->a || !op->b)
        return MM_ERR_STATE;
    memcpy((void *)op->b, op->a, op->n0);
    return MM_OK;
}

/* -------------------------------------------------------------- EMBED */

mm_status kx_op_embed(const mm_kcall *kc)
{
    const mm_op_desc *op = kc->op;
    const mm_ctrl *c = &kc->e->ctrl;
    const uint32_t *toks = ctrl_toks(c);
    const uint16_t *w = op->a;
    uint16_t *x = op->c;
    uint32_t M = op->n0, H = op->n1, V = kc->e->mc.vocab;
    for (uint32_t m = 0; m < M; m++) {
        uint32_t t = toks[m];
        if (t >= V)
            return MM_ERR_RANGE;
        memcpy(x + (size_t)m * H, w + (size_t)t * H, (size_t)H * 2);
    }
    return MM_OK;
}

/* ----------------------------------------------------------- RMSNORM */

mm_status kx_op_rmsnorm(const mm_kcall *kc)
{
    const mm_op_desc *op = kc->op;
    const uint16_t *x = op->a, *g = op->b;
    uint16_t *y = op->c;
    uint32_t M = op->n0, H = op->n1;
    float eps = kc->e->mc.norm_eps;
    for (uint32_t m = 0; m < M; m++)
        kx_rms_row(x + (size_t)m * H, g, y + (size_t)m * H, H, eps);
    return MM_OK;
}

/* ------------------------------------------------------- ADD_RMSNORM */

mm_status kx_op_add_rmsnorm(const mm_kcall *kc)
{
    const mm_op_desc *op = kc->op;
    /* op->c is the non-const output; a/b/d are const in mm_op_desc, so the
     * in-place residual (a) needs the const stripped to be written back. */
    uint16_t *res = (uint16_t *)(uintptr_t)op->a;
    const uint16_t *delta = op->b;
    uint16_t *y = op->c;
    const uint16_t *g = op->d;
    uint32_t M = op->n0, H = op->n1;
    float eps = kc->e->mc.norm_eps;
    for (uint32_t m = 0; m < M; m++) {
        uint16_t *rr = res + (size_t)m * H;
        const uint16_t *dr = delta + (size_t)m * H;
        for (uint32_t i = 0; i < H; i++)
            bf16_wr(&rr[i], bf16_rd(&rr[i]) + bf16_rd(&dr[i]));
        kx_rms_row(rr, g, y + (size_t)m * H, H, eps);
    }
    return MM_OK;
}

/* ----------------------------------------------------------- SILU_MUL */

mm_status kx_op_silu_mul(const mm_kcall *kc)
{
    const mm_op_desc *op = kc->op;
    const uint16_t *gate = op->a, *up = op->b;
    uint16_t *y = op->c;
    uint32_t n = op->n0 * op->n1;
    for (uint32_t i = 0; i < n; i++) {
        float g = bf16_rd(&gate[i]);
        float s = g / (1.0f + expf(-g));   /* silu(gate) */
        bf16_wr(&y[i], s * bf16_rd(&up[i]));
    }
    return MM_OK;
}

/* ------------------------------------------------------ GEMM_BF16 / QKV */

mm_status kx_op_gemm_bf16(const mm_kcall *kc)
{
    const mm_op_desc *op = kc->op;
    kx_gemm_bf16(op->a, op->b, op->c, op->n0, op->n1, op->n2);
    return MM_OK;
}

/* QKV is the same GEMM (fused q/k/v or the linear in-proj); the plan picks
 * the shape via n2. Identical math to GEMM_BF16 for the bf16 reference. */
mm_status kx_op_qkv(const mm_kcall *kc)
{
    return kx_op_gemm_bf16(kc);
}

/* ------------------------------------------------------------- LMHEAD */

mm_status kx_op_lmhead(const mm_kcall *kc)
{
    return kx_op_gemm_bf16(kc);
}

/* ----------------------------------------------------------- GEMM_F4 */

/* NVFP4 weight GEMM: y = x @ W^T with W dequantized in-loop (payload +
 * e4m3 group scales). The fake host model uses bf16, so this path is a
 * correctness reference, not the hot path. */
mm_status kx_op_gemm_f4(const mm_kcall *kc)
{
    const mm_op_desc *op = kc->op;
    const uint16_t *x = op->a;
    uint16_t *y = op->c;            /* op->c is the non-const output */
    uint32_t M = op->n0, K = op->n1, cols = op->n2;
    mm_wt W = { op->b, op->d, cols, K };   /* rows=out, cols=in */
    MM_REQUIRE(K % 16 == 0, MM_ERR_SHAPE);
    for (uint32_t m = 0; m < M; m++) {
        const uint16_t *xr = x + (size_t)m * K;
        uint16_t *yr = y + (size_t)m * cols;
        for (uint32_t r = 0; r < cols; r++) {
            float acc = 0.0f;
            for (uint32_t k = 0; k < K; k++)
                acc += bf16_rd(&xr[k]) * mm_fp4_wget(&W, r, k);
            bf16_wr(&yr[r], acc);
        }
    }
    return MM_OK;
}

/* ------------------------------------------------------- KV row (write) */

/* Writable counterpart of kv_row() for KVSTORE: one KV row (head_dim bf16)
 * for slot 0, kv head `head`, absolute position `pos`, in the
 * [kv_head][block][tok_in_block][head_dim] layout. */
static uint16_t *kv_row_w(mm_kvpool *kv, uint32_t head, void *base, uint32_t pos)
{
    uint32_t bid = kv->blk_tab_host[pos / MM_KV_BLOCK_TOK];   /* slot 0 */
    uint32_t blk = bid ? bid - 1 : 0;   /* 1-based id -> 0-based index */
    size_t off = head * (kv->head_stride / 2)
             + (size_t)blk * (kv->block_stride / 2)
             + (pos % MM_KV_BLOCK_TOK) * (kv->tok_stride / 2);
    return (uint16_t *)base + off;
}

/* --------------------------------------------------------------- ROPE */

mm_status kx_op_rope(const mm_kcall *kc)
{
    const mm_op_desc *op = kc->op;
    const mm_engine *e = kc->e;
    const mm_ctrl *c = &e->ctrl;
    const mm_model_cfg *mc = &e->mc;
    const uint32_t M = op->n0, rd = op->n1;
    const uint32_t hd = mc->head_dim, qdim = mc->q_heads * hd;
    const uint32_t kvdim = mc->kv_heads * hd;
    /* a and b are in-place outputs; strip const to write q and k back. */
    uint16_t *q = (uint16_t *)(uintptr_t)op->a;
    uint16_t *k = (uint16_t *)(uintptr_t)op->b;
    /* Effective inverse-frequency table + attention scale: arrive in the
     * control buffer (the engine fills them after every round's reset).
     * The plain table is bit-exact with the legacy per-kernel powf
     * formula and mscale == 1.0, so the golden reference is unchanged. */
    const float *inv_freq = c->rope_inv;
    const float ms = c->rope_mscale;
    if (!q || !k || rd < 2 || rd > hd)
        return MM_ERR_STATE;
    for (uint32_t m = 0; m < M; m++) {
        const float posf = (float)ctrl_pos(c, m);
        for (uint32_t h = 0; h < mc->q_heads; h++) {
            uint16_t *row = q + (size_t)m * qdim + (size_t)h * hd;
            for (uint32_t i = 0; i < rd / 2; i++) {
                const float ang = posf * inv_freq[i];
                const float s = sinf(ang), co = cosf(ang);
                const float x0 = bf16_rd(&row[i]);
                const float x1 = bf16_rd(&row[i + rd / 2]);
                bf16_wr(&row[i], (x0 * co - x1 * s) * ms);
                bf16_wr(&row[i + rd / 2], (x1 * co + x0 * s) * ms);
            }
        }
        for (uint32_t h = 0; h < mc->kv_heads; h++) {
            uint16_t *row = k + (size_t)m * kvdim + (size_t)h * hd;
            for (uint32_t i = 0; i < rd / 2; i++) {
                const float ang = posf * inv_freq[i];
                const float s = sinf(ang), co = cosf(ang);
                const float x0 = bf16_rd(&row[i]);
                const float x1 = bf16_rd(&row[i + rd / 2]);
                bf16_wr(&row[i], (x0 * co - x1 * s) * ms);
                bf16_wr(&row[i + rd / 2], (x1 * co + x0 * s) * ms);
            }
        }
    }
    return MM_OK;
}

/* ------------------------------------------------------------- KVSTORE */

mm_status kx_op_kvstore(const mm_kcall *kc)
{
    const mm_op_desc *op = kc->op;
    const mm_engine *e = kc->e;
    const mm_ctrl *c = &e->ctrl;
    const mm_model_cfg *mc = &e->mc;
    const uint32_t li = op->n0, M = op->n1, hd = mc->head_dim;
    const uint32_t kvdim = mc->kv_heads * hd;
    const uint16_t *k = op->a, *v = op->b;
    uint16_t *kb = (uint16_t *)op->c;                 /* == k_base[li] */
    uint16_t *vb = (uint16_t *)(uintptr_t)e->kv->v_base[li];
    if (!e->kv || !kb || !vb)
        return MM_ERR_STATE;
    for (uint32_t m = 0; m < M; m++) {
        const uint32_t pos = ctrl_pos(c, m);
        const uint16_t *kr = k + (size_t)m * kvdim;
        const uint16_t *vr = v + (size_t)m * kvdim;
        for (uint32_t h = 0; h < mc->kv_heads; h++) {
            memcpy(kv_row_w(e->kv, h, kb, pos), kr + h * hd, hd * 2);
            memcpy(kv_row_w(e->kv, h, vb, pos), vr + h * hd, hd * 2);
        }
    }
    return MM_OK;
}

/* ---------------------------------------------------------- ATT_DECODE */

mm_status kx_op_att_decode(const mm_kcall *kc)
{
    const mm_op_desc *op = kc->op;
    const mm_engine *e = kc->e;
    const mm_ctrl *c = &e->ctrl;
    const mm_model_cfg *mc = &e->mc;
    const uint32_t M = op->n1, hd = mc->head_dim, qdim = mc->q_heads * hd;
    const uint16_t *q = op->b;
    uint16_t *o = op->c;
    const void *kb = op->a, *vb = e->kv->v_base[op->layer];
    if (!e->kv || !vb)
        return MM_ERR_STATE;
    const float sc = 1.0f / sqrtf((float)hd);
    const uint32_t g = mc->q_heads / mc->kv_heads;    /* q heads per kv head */
    for (uint32_t m = 0; m < M; m++) {
        const uint32_t len = c->slot_len[m];          /* KV length this slot */
        for (uint32_t h = 0; h < mc->q_heads; h++) {
            const uint32_t kh = h / g;
            const uint16_t *qr = q + (size_t)m * qdim + (size_t)h * hd;
            float mx = -INFINITY, sum = 0.0f;
            for (uint32_t t = 0; t < len; t++) {      /* softmax: max */
                const uint16_t *kr = kv_row(e->kv, kh, kb, t);
                float d = 0.0f;
                for (uint32_t i = 0; i < hd; i++)
                    d += bf16_rd(&qr[i]) * bf16_rd(&kr[i]);
                if ((d *= sc) > mx)
                    mx = d;
            }
            for (uint32_t t = 0; t < len; t++) {      /* softmax: sum */
                const uint16_t *kr = kv_row(e->kv, kh, kb, t);
                float d = 0.0f;
                for (uint32_t i = 0; i < hd; i++)
                    d += bf16_rd(&qr[i]) * bf16_rd(&kr[i]);
                sum += expf(d * sc - mx);
            }
            const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
            uint16_t *orow = o + (size_t)m * qdim + (size_t)h * hd;
            for (uint32_t i = 0; i < hd; i++) {       /* P @ V */
                float acc = 0.0f;
                for (uint32_t t = 0; t < len; t++) {
                    const uint16_t *kr = kv_row(e->kv, kh, kb, t);
                    float d = 0.0f;
                    for (uint32_t j = 0; j < hd; j++)
                        d += bf16_rd(&qr[j]) * bf16_rd(&kr[j]);
                    const uint16_t *vr = kv_row(e->kv, kh, vb, t);
                    acc += expf(d * sc - mx) * inv * bf16_rd(&vr[i]);
                }
                bf16_wr(&orow[i], acc);
            }
        }
    }
    return MM_OK;
}

/* ---------------------------------------------------------- ATT_PREFILL */

mm_status kx_op_att_prefill(const mm_kcall *kc)
{
    const mm_op_desc *op = kc->op;
    const mm_engine *e = kc->e;
    const mm_ctrl *c = &e->ctrl;
    const mm_model_cfg *mc = &e->mc;
    const uint32_t M = op->n0, hd = mc->head_dim, qdim = mc->q_heads * hd;
    const uint16_t *q = op->b;
    uint16_t *o = op->c;
    const void *kb = op->a, *vb = e->kv->v_base[op->layer];
    if (!e->kv || !vb)
        return MM_ERR_STATE;
    const float sc = 1.0f / sqrtf((float)hd);
    const uint32_t g = mc->q_heads / mc->kv_heads;
    for (uint32_t m = 0; m < M; m++) {
        const uint32_t pos = ctrl_pos(c, m);          /* attend to 0..pos */
        for (uint32_t h = 0; h < mc->q_heads; h++) {
            const uint32_t kh = h / g;
            const uint16_t *qr = q + (size_t)m * qdim + (size_t)h * hd;
            float mx = -INFINITY, sum = 0.0f;
            for (uint32_t t = 0; t <= pos; t++) {     /* causal softmax max */
                const uint16_t *kr = kv_row(e->kv, kh, kb, t);
                float d = 0.0f;
                for (uint32_t i = 0; i < hd; i++)
                    d += bf16_rd(&qr[i]) * bf16_rd(&kr[i]);
                if ((d *= sc) > mx)
                    mx = d;
            }
            for (uint32_t t = 0; t <= pos; t++) {     /* causal softmax sum */
                const uint16_t *kr = kv_row(e->kv, kh, kb, t);
                float d = 0.0f;
                for (uint32_t i = 0; i < hd; i++)
                    d += bf16_rd(&qr[i]) * bf16_rd(&kr[i]);
                sum += expf(d * sc - mx);
            }
            const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
            uint16_t *orow = o + (size_t)m * qdim + (size_t)h * hd;
            for (uint32_t i = 0; i < hd; i++) {
                float acc = 0.0f;
                for (uint32_t t = 0; t <= pos; t++) {
                    const uint16_t *kr = kv_row(e->kv, kh, kb, t);
                    float d = 0.0f;
                    for (uint32_t j = 0; j < hd; j++)
                        d += bf16_rd(&qr[j]) * bf16_rd(&kr[j]);
                    const uint16_t *vr = kv_row(e->kv, kh, vb, t);
                    acc += expf(d * sc - mx) * inv * bf16_rd(&vr[i]);
                }
                bf16_wr(&orow[i], acc);
            }
        }
    }
    return MM_OK;
}

/* -------------------------------------------------------- SOFTMAX_CAUSAL */

/* In-place causal row-softmax of a [rows][cols] bf16 score tile. Column j
 * holds absolute KV position kv_start + j; row i is causally masked where
 * that position exceeds kv_start + i. */
mm_status kx_op_softmax_causal(const mm_kcall *kc)
{
    const mm_op_desc *op = kc->op;
    const uint32_t rows = op->n0, cols = op->n1, kv_start = op->n2;
    uint16_t *S = (uint16_t *)(uintptr_t)op->a;
    float *tmp;
    if (!S || cols == 0)
        return MM_ERR_STATE;
    tmp = malloc((size_t)cols * sizeof *tmp);
    if (!tmp)
        return MM_ERR_NOMEM;
    for (uint32_t i = 0; i < rows; i++) {
        uint16_t *row = S + (size_t)i * cols;
        float mx = -INFINITY;
        for (uint32_t j = 0; j < cols; j++) {
            tmp[j] = (kv_start + j <= kv_start + i) ? bf16_rd(&row[j])
                                                    : -INFINITY;
            if (tmp[j] > mx)
                mx = tmp[j];
        }
        float sum = 0.0f;
        for (uint32_t j = 0; j < cols; j++) {
            tmp[j] = (tmp[j] == -INFINITY) ? 0.0f : expf(tmp[j] - mx);
            sum += tmp[j];
        }
        const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
        for (uint32_t j = 0; j < cols; j++)
            bf16_wr(&row[j], tmp[j] * inv);
    }
    free(tmp);
    return MM_OK;
}

/* ---------------------------------------------------------------- CONV4 */

/* Causal depthwise conv over the qkv channels + conv-state update. The plan
 * wires a = x [M][in_dim], b = conv_w [ch][K], c = out (in-place, == a),
 * d = conv state [K-1][ch] bf16 (slot-0 layer base). The first `ch` channels
 * are convolved; the trailing g/beta channels (ch..in_dim) pass through
 * untouched (they are the DeltaNet gates, not mixer input). */
mm_status kx_op_conv4(const mm_kcall *kc)
{
    const mm_op_desc *op = kc->op;
    const mm_engine *e = kc->e;
    const mm_model_cfg *mc = &e->mc;
    const uint32_t M = op->n0, in_dim = op->n1, K = mc->conv_k;
    const uint32_t ch = 2u * mc->lin_k_heads * mc->lin_k_dim
                      + mc->lin_v_heads * mc->lin_v_dim;
    const uint16_t *x = op->a, *w = op->b;
    uint16_t *out = op->c;
    uint16_t *st = (uint16_t *)(uintptr_t)op->d;   /* [K-1][ch] bf16 */
    if (!x || !w || !out || !st || K < 2 || ch > in_dim)
        return MM_ERR_STATE;
    for (uint32_t m = 0; m < M; m++) {
        for (uint32_t c = 0; c < ch; c++) {
            const uint32_t base = (size_t)m * in_dim + c;
            const float cur = bf16_rd(&x[base]);
            float acc = 0.0f;
            for (uint32_t i = 0; i < K - 1; i++)
                acc += bf16_rd(&w[(size_t)c * K + i]) *
                       bf16_rd(&st[(size_t)i * ch + c]);
            acc += bf16_rd(&w[(size_t)c * K + (K - 1)]) * cur;
            bf16_wr(&out[base], acc);
            for (uint32_t i = 0; i < K - 2; i++)
                bf16_wr(&st[(size_t)i * ch + c],
                        bf16_rd(&st[(size_t)(i + 1) * ch + c]));
            bf16_wr(&st[(size_t)(K - 2) * ch + c], cur);
        }
    }
    return MM_OK;
}

/* ------------------------------------------------------------ LIN state */

/* Gated-DeltaNet reference recurrence (LIN_PRE / LIN_DEC share it). Per v_head
 * h the fp32 state S[h] is [k_dim][v_dim]; with a FIXED decay (the reference
 * ignores the data-dependent gate for boundedness; see kx.h) it updates
 *
 *   S[h] = decay * S[h] + beta_h * (k_h outer v_h);   o_h = S[h] @ q_h
 *
 * and processes the M new tokens of the qkv row buffer sequentially. The plan
 * wires a = state base, b = qkv [M][in_dim], c = out [M][v_dim]. q and k are
 * shared across the v heads that map to one k head (GQA over the k heads). */
static mm_status kx_lin_run(const mm_op_desc *op, const mm_engine *e,
                            uint32_t M)
{
    const mm_model_cfg *mc = &e->mc;
    const uint32_t kdim = mc->lin_k_dim, vd = mc->lin_v_dim;
    const uint32_t kq = mc->lin_k_heads * kdim;
    const uint32_t vdim = mc->lin_v_heads * vd;
    const uint32_t ch = 2u * kq + vdim;
    const uint32_t in_dim = ch + 2u * mc->lin_v_heads;
    float *st = (float *)(uintptr_t)op->a;
    const uint16_t *qkv = op->b;
    uint16_t *o = op->c;
    if (!st || !qkv || !o)
        return MM_ERR_STATE;
    const float decay = expf(-0.1f);
    const uint32_t kvh = mc->lin_v_heads / mc->lin_k_heads;
    for (uint32_t m = 0; m < M; m++) {
        const uint16_t *row = qkv + (size_t)m * in_dim;
        for (uint32_t h = 0; h < mc->lin_v_heads; h++) {
            const uint32_t kh = h / kvh;
            const uint16_t *qk = row + kh * kdim;
            const uint16_t *vv = row + 2u * kq + h * vd;
            const float beta = bf16_rd(&row[ch + mc->lin_v_heads + h]);
            float *S = st + (size_t)h * kdim * vd;
            for (uint32_t i = 0; i < kdim; i++) {
                const float kv = bf16_rd(&qk[i]);
                for (uint32_t j = 0; j < vd; j++) {
                    float *s = &S[i * vd + j];
                    *s = decay * *s + beta * kv * bf16_rd(&vv[j]);
                }
            }
        }
        uint16_t *orow = o + (size_t)m * vdim;
        for (uint32_t h = 0; h < mc->lin_v_heads; h++) {
            const uint32_t kh = h / kvh;
            const uint16_t *qk = row + kh * kdim;
            const float *S = st + (size_t)h * kdim * vd;
            for (uint32_t j = 0; j < vd; j++) {
                float acc = 0.0f;
                for (uint32_t i = 0; i < kdim; i++)
                    acc += S[i * vd + j] * bf16_rd(&qk[i]);
                bf16_wr(&orow[h * vd + j], acc);
            }
        }
    }
    return MM_OK;
}

mm_status kx_op_lin_dec(const mm_kcall *kc)
{
    const mm_op_desc *op = kc->op;
    return kx_lin_run(op, kc->e, op->n0);
}

mm_status kx_op_lin_pre(const mm_kcall *kc)
{
    const mm_op_desc *op = kc->op;
    return kx_lin_run(op, kc->e, op->n0);
}

/* --------------------------------------------------------------- SAMPLE */

/* Pick one token from a [vocab] logit row per the control-buffer policy.
 * Greedy (temperature <= 0 or top_k == 0) is argmax. Otherwise softmax over
 * 1/temperature, truncate to top-k then top-p, and draw with the one uniform
 * (fixed-point 1.31) the engine seeded into c->u. Deterministic in
 * (logits, policy, u). */
static uint32_t kx_sample_row(const uint16_t *logits, uint32_t V,
                              const mm_ctrl *c)
{
    if (V == 0)
        return 0;
    if (c->temperature <= 0.0f || c->top_k == 0) {
        uint32_t best = 0;
        float bv = bf16_rd(&logits[0]);
        for (uint32_t i = 1; i < V; i++) {
            const float v = bf16_rd(&logits[i]);
            if (v > bv) {
                bv = v;
                best = i;
            }
        }
        return best;
    }
    float *tmp = malloc((size_t)V * sizeof *tmp);
    uint32_t *order = malloc((size_t)V * sizeof *order);
    if (!tmp || !order) {
        free(tmp);
        free(order);
        return 0;
    }
    const float temp = c->temperature;
    float mx = -INFINITY;
    for (uint32_t i = 0; i < V; i++) {
        tmp[i] = bf16_rd(&logits[i]) / temp;
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
    const mm_ctrl *c = &e->ctrl;
    const uint32_t V = e->mc.vocab;
    const uint32_t M = is_decode(c) ? c->n_slots : op->n0;
    const uint16_t *logits = op->a;
    uint32_t *toks = (uint32_t *)(uintptr_t)op->b;
    if (!logits || !toks || V == 0)
        return MM_ERR_STATE;
    for (uint32_t m = 0; m < M; m++)
        toks[m] = kx_sample_row(logits + (size_t)m * V, V, c);
    return MM_OK;
}

/* ------------------------------------------------------------- coverage */

int kx_cpu_oppref(int op)
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
        return 0;   /* OP_NOP / OP_N / unknown */
    }
}

