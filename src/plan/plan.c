/*
 * Plan builder + runtime (see plan.h).
 *
 * mm_plan_build() is a pure function of (model_cfg, tensor registry,
 * fixed activation buffers): it walks the 64 Qwen3.8 layers and emits a
 * flat, address-fixed op list. No allocation, no capture. That purity is
 * what lets the host test path build and "run" a plan with CPU reference
 * kernels and no GPU.
 *
 * Op semantics per opcode are documented in kernels.h; the operand
 * letters (a/b/c/d) and n0/n1/n2 below follow that contract.
 */
#include "mimfer/plan.h"
#include "mimfer/engine.h"
#include "mimfer/kernels.h"
#include <stdio.h>
#include <string.h>

/* ------------------------------------------------------------- helpers */

/* Append one op descriptor. Fails (MM_ERR_SHAPE) past MM_MAX_OPS. */
static mm_status emit(mm_plan *p, int op, int layer, const void *a,
                      const void *b, void *c, const void *d, uint32_t n0,
                      uint32_t n1, uint32_t n2, const char *note)
{
    MM_REQUIRE(p && p->n_ops < MM_MAX_OPS, MM_ERR_SHAPE);
    {
        mm_op_desc *o = &p->ops[p->n_ops++];
        o->op = op;
        o->layer = layer;
        o->a = a;
        o->b = b;
        o->c = c;
        o->d = d;
        o->n0 = n0;
        o->n1 = n1;
        o->n2 = n2;
        snprintf(o->note, sizeof o->note, "%.23s", note ? note : "");
    }
    return MM_OK;
}

/* Byte width of the fused QKV / in-proj buffer rows. */
static uint32_t qkv_row_dim(const mm_model_cfg *mc)
{
    uint32_t full = (mc->q_heads + 2u * mc->kv_heads) * mc->head_dim;
    uint32_t lin = 2u * mc->lin_k_heads * mc->lin_k_dim
               + mc->lin_v_heads * mc->lin_v_dim
               + 2u * mc->lin_v_heads;   /* g + beta gates, 1 elem/head */
    return full > lin ? full : lin;
}

/* Position of a linear-attention layer among the linear layers (0-based),
 * skipping the full layers. The recurrent-state and conv pools (see kv.h)
 * are sized for exactly n_lin_layers compact slots, so kernels must index
 * them by this, not the raw layer index — the raw index skips full layers
 * and can reach n_lin_layers (one past the pool). */
static uint32_t lin_index(const mm_model_cfg *mc, uint32_t li)
{
    uint32_t n = 0;
    for (uint32_t i = 0; i < li; i++)
        if (!mm_model_cfg_layer_is_full(mc, i + 1))
            n++;
    return n;
}

/* Allocate the fixed activation map from the activation arena, sized for
 * M tokens. Pointers stay stable for the engine's lifetime. */
mm_status mm_plan_reserve_acts(mm_engine *e, uint32_t M)
{
    const mm_model_cfg *mc = &e->mc;
    mm_arena *a = &e->a_arena;
    uint32_t qdim = mc->q_heads * mc->head_dim;
    uint32_t kvdim = mc->kv_heads * mc->head_dim;
    uint32_t vdim = mc->lin_v_heads * mc->lin_v_dim;
    uint32_t attn_out = qdim > vdim ? qdim : vdim;  /* raw attn width */
    uint32_t qkv = qkv_row_dim(mc);
    uint32_t h = mc->hidden, inter = mc->inter, V = mc->vocab;
    const size_t bf16 = 2;
    mm_status s;

    if (e->ab.x)              /* already reserved */
        return MM_OK;
    MM_REQUIRE(M > 0, MM_ERR_STATE);

#define MM_STR(x) #x
#define ALLOC(field, cols, elem)                                              \
    s = mm_arena_alloc_2d(a, (size_t)M, (size_t)(cols), (elem), 256,          \
                          MM_STR(field), &e->ab.field);                       \
    if (s != MM_OK) return s;

    ALLOC(x, h, bf16);
    ALLOC(res, h, bf16);
    ALLOC(normed, h, bf16);
    ALLOC(qkv, qkv, bf16);
    ALLOC(q, qdim, bf16);
    ALLOC(k, kvdim, bf16);
    ALLOC(v, kvdim, bf16);
    ALLOC(o, attn_out, bf16);
    ALLOC(gate, inter, bf16);
    ALLOC(up, inter, bf16);
    ALLOC(mlp, inter, bf16);
    ALLOC(S, M, bf16);
    ALLOC(part, h, bf16);
    ALLOC(scratch, inter, bf16);
    ALLOC(logits, V, bf16);
    s = mm_arena_alloc(a, (size_t)M * 4, 16, "a.toks_dev", &e->ab.toks_dev);
    if (s != MM_OK)
        return s;
#undef ALLOC
#undef MM_STR
    return MM_OK;
}

/* ---------------------------------------------------------------- launch */

/* Direct dispatch: run every op through the kernel dispatcher. Used by the
 * host (CPU reference) build and by the GPU build when --no-graph. */
static mm_status plan_run_ops(mm_engine *e, const mm_plan *p)
{
    mm_stream_h st = (mm_stream_h)0;   /* GPU dispatch uses its own stream */
    for (uint32_t i = 0; i < p->n_ops; i++) {
        mm_kcall kc;
        kc.op = &p->ops[i];
        kc.e = e;
        kc.stream = st;
        MM_CHECK(mm_kx_invoke(&kc));
    }
    return MM_OK;
}

mm_status mm_plan_launch(mm_engine *e, const mm_plan *p)
{
    MM_REQUIRE(e && p && p->n_ops > 0, MM_ERR_STATE);
    if (p->graph && p->captured)
        return mm_graph_launch(p->graph, MM_ST_COMPUTE);
    return plan_run_ops(e, p);
}

/* --------------------------------------------------------------- destroy */

void mm_plan_destroy(mm_plan *p)
{
    if (!p)
        return;
    if (p->graph)
        mm_graph_destroy(p->graph);
    p->graph = NULL;
    p->captured = 0;
    p->n_ops = 0;
}

/* ----------------------------------------------------------- human dump */

static const char *op_name(int op)
{
    switch (op) {
    case OP_EMBED: return "EMBED";
    case OP_LMHEAD: return "LMHEAD";
    case OP_RMSNORM: return "RMSNORM";
    case OP_ADD_RMSNORM: return "ADD_RMSNORM";
    case OP_GEMM_F4: return "GEMM_F4";
    case OP_GEMM_BF16: return "GEMM_BF16";
    case OP_QKV: return "QKV";
    case OP_ROPE: return "ROPE";
    case OP_KVSTORE: return "KVSTORE";
    case OP_ATT_DECODE: return "ATT_DECODE";
    case OP_ATT_PREFILL: return "ATT_PREFILL";
    case OP_SOFTMAX_CAUSAL: return "SOFTMAX_CAUSAL";
    case OP_SILU_MUL: return "SILU_MUL";
    case OP_CONV4: return "CONV4";
    case OP_LIN_DEC: return "LIN_DEC";
    case OP_LIN_PRE: return "LIN_PRE";
    case OP_COPY: return "COPY";
    case OP_SAMPLE: return "SAMPLE";
    case OP_NOP: return "NOP";
    default: return "?";
    }
}

int mm_plan_dump(const mm_plan *p, char *buf, size_t n)
{
    int o = 0;
    const char *ph = p->phase == PH_PREFILL ? "prefill"
                     : p->phase == PH_DECODE ? "decode" : "verify";
    o += snprintf(buf + o, n - (size_t)o,
                  "plan[%s M=%u] %u ops%s\n", ph, p->M, p->n_ops,
                  p->captured ? " (graph captured)" : " (direct)");
    for (uint32_t i = 0; i < p->n_ops && o < (int)n - 96; i++) {
        const mm_op_desc *op = &p->ops[i];
        o += snprintf(buf + o, n - (size_t)o,
                      "  %3u %-13s L%-2d a=%p b=%p c=%p d=%p "
                      "n0=%u n1=%u n2=%u  %s\n",
                      i, op_name(op->op), op->layer, op->a, op->b, op->c,
                      op->d, op->n0, op->n1, op->n2, op->note);
    }
    return o;
}

/* --------------------------------------------------- op-list emitters */

/* y = a @ W^T with W looked up in the registry; emits the fp4 or bf16 gemm
 * opcode. a: [M][K] bf16; c: [M][cols]. */
static mm_status emit_gemm(mm_engine *e, mm_plan *p, int layer,
                           const char *anote, const void *a, const char *wname,
                           void *c, uint32_t M, uint32_t K, uint32_t cols)
{
    const mm_tens *t = mm_tens_find(&e->reg, wname);
    char note[64];
    if (!t) {
        MM_LOGE("plan: missing tensor \"%s\"", wname);
        return MM_ERR_STATE;
    }
    snprintf(note, sizeof note, "%s@%s", anote, wname);
    if (t->is_fp4)
        return emit(p, OP_GEMM_F4, layer, a, t->w.data, c, t->w.scale, M, K,
                    cols, note);
    return emit(p, OP_GEMM_BF16, layer, a, t->v.data, c, NULL, M, K, cols,
                note);
}

/* MLP tail: normed -> (gate,up) -> silu -> down -> x. */
static mm_status emit_mlp(mm_engine *e, mm_plan *p, uint32_t li, uint32_t M)
{
    const mm_model_cfg *mc = &e->mc;
    mm_abufs *ab = &e->ab;
    char nm[64];
    mm_status s;

    snprintf(nm, sizeof nm, "model.layers.%u.mlp.gate_proj.weight", li);
    s = emit_gemm(e, p, (int)li, "gate", ab->normed, nm, ab->gate, M,
                  mc->hidden, mc->inter);
    if (s != MM_OK)
        return s;
    snprintf(nm, sizeof nm, "model.layers.%u.mlp.up_proj.weight", li);
    s = emit_gemm(e, p, (int)li, "up", ab->normed, nm, ab->up, M, mc->hidden,
                  mc->inter);
    if (s != MM_OK)
        return s;
    s = emit(p, OP_SILU_MUL, -1, ab->gate, ab->up, ab->mlp, NULL, M,
             mc->inter, 0, "silu(gate)*up");
    if (s != MM_OK)
        return s;
    snprintf(nm, sizeof nm, "model.layers.%u.mlp.down_proj.weight", li);
    return emit_gemm(e, p, (int)li, "down", ab->mlp, nm, ab->x, M,
                     mc->inter, mc->hidden);
}

/* Full-attention sublayer: normed -> o_proj(x). Writes the projected output
 * to e->ab.x. */
static mm_status emit_full_attn(mm_engine *e, mm_plan *p, uint32_t li,
                                uint32_t M, mm_phase ph)
{
    const mm_model_cfg *mc = &e->mc;
    mm_abufs *ab = &e->ab;
    char nm[64];
    const uint8_t *qkv = (const uint8_t *)ab->qkv;
    const size_t bf16 = 2;
    uint32_t qdim = mc->q_heads * mc->head_dim;
    uint32_t kvdim = mc->kv_heads * mc->head_dim;
    /* The pool is writable; the kernel reads e->kv for the V base. */
    void *kb = (void *)(size_t)(e->kv ? e->kv->k_base[li] : NULL);
    const mm_tens *t;
    mm_status s;

    snprintf(nm, sizeof nm, "model.layers.%u.self_attn.qkv.weight", li);
    t = mm_tens_find(&e->reg, nm);
    if (!t) {
        MM_LOGE("plan: missing %s", nm);
        return MM_ERR_STATE;
    }
    s = emit(p, OP_QKV, (int)li, ab->normed, t->is_fp4 ? t->w.data
                                                       : t->v.data,
             ab->qkv, t->is_fp4 ? t->w.scale : NULL, M, mc->hidden,
             qkv_row_dim(mc), nm);
    if (s != MM_OK)
        return s;

    /* split the fused row: [q | k | v] */
    s = emit(p, OP_COPY, -1, qkv, ab->q, NULL, NULL, M * qdim * bf16, 0, 0,
             "qkv->q");
    if (s != MM_OK)
        return s;
    s = emit(p, OP_COPY, -1, qkv + M * qdim * bf16, ab->k, NULL, NULL,
             M * kvdim * bf16, 0, 0, "qkv->k");
    if (s != MM_OK)
        return s;
    s = emit(p, OP_COPY, -1, qkv + M * (qdim + kvdim) * bf16, ab->v, NULL,
             NULL, M * kvdim * bf16, 0, 0, "qkv->v");
    if (s != MM_OK)
        return s;

    s = emit(p, OP_ROPE, (int)li, ab->q, ab->k, NULL, NULL, M, mc->rope_dim,
             0, "rope(q,k)");
    if (s != MM_OK)
        return s;
    s = emit(p, OP_KVSTORE, (int)li, ab->k, ab->v, kb, NULL, li, M, 0,
             "kvstore");
    if (s != MM_OK)
        return s;

    if (ph == PH_PREFILL)
        s = emit(p, OP_ATT_PREFILL, (int)li, kb, ab->q, ab->o, NULL, M,
                 mc->q_heads, 0, "att_prefill");
    else
        s = emit(p, OP_ATT_DECODE, (int)li, kb, ab->q, ab->o, NULL, li, M, 0,
                 "att_decode");
    if (s != MM_OK)
        return s;

    snprintf(nm, sizeof nm, "model.layers.%u.self_attn.o_proj.weight", li);
    return emit_gemm(e, p, (int)li, "o_proj", ab->o, nm, ab->x, M, qdim,
                     mc->hidden);
}

/* Gated-DeltaNet sublayer: normed -> in_proj -> conv4 -> recurrence(x). */
static mm_status emit_lin_attn(mm_engine *e, mm_plan *p, uint32_t li,
                               uint32_t M, mm_phase ph)
{
    const mm_model_cfg *mc = &e->mc;
    mm_abufs *ab = &e->ab;
    char nm[64];
    uint32_t in_dim = 2u * mc->lin_k_heads * mc->lin_k_dim
                   + mc->lin_v_heads * mc->lin_v_dim
                   + 2u * mc->lin_v_heads;   /* q,k,v + g,beta gates */
    uint32_t vdim = mc->lin_v_heads * mc->lin_v_dim;
    /* State/conv pools are indexed by the COMPACT linear-layer slot, not the
     * raw layer index (which skips full layers and can reach n_lin_layers). */
    const uint32_t lidx = lin_index(mc, li);
    const void *st = e->st ? (const void *)((const uint8_t *)e->st->base
                                            + (size_t)lidx * e->st->layer_bytes)
                           : NULL;
    /* Conv state is [slot][layer][conv_k-1][ch]; the baked pointer is the
     * slot-0 layer base (the active slot's offset is added by the kernel at
     * run time). conv_bytes is the per-slot total, so the per-layer stride
     * is conv_bytes / n_lin_layers — the same layout as the state pointer
     * above. (Using the full conv_bytes here was an off-by-n_lin_layers
     * stride bug: for li >= 1 the pointer fell outside the conv buffer.) */
    const void *cv = NULL;
    if (e->st) {
        const size_t cv_layer = e->st->conv_bytes /
                                (size_t)(e->st->mc->n_lin_layers ?
                                         e->st->mc->n_lin_layers : 1);
        cv = (const void *)((const uint8_t *)e->st->conv + (size_t)lidx * cv_layer);
    }
    const mm_tens *t;
    mm_status s;

    snprintf(nm, sizeof nm, "model.layers.%u.linear_attn.in_proj.weight", li);
    t = mm_tens_find(&e->reg, nm);
    if (!t) {
        MM_LOGE("plan: missing %s", nm);
        return MM_ERR_STATE;
    }
    s = emit(p, OP_QKV, (int)li, ab->normed, t->is_fp4 ? t->w.data
                                                       : t->v.data,
             ab->qkv, t->is_fp4 ? t->w.scale : NULL, M, mc->hidden, in_dim,
             nm);
    if (s != MM_OK)
        return s;

    snprintf(nm, sizeof nm, "model.layers.%u.linear_attn.conv1d.weight", li);
    t = mm_tens_find(&e->reg, nm);
    if (!t) {
        MM_LOGE("plan: missing %s", nm);
        return MM_ERR_STATE;
    }
    s = emit(p, OP_CONV4, (int)li, ab->qkv, t->v.data, ab->qkv, cv, M,
             in_dim, 0, "conv4");
    if (s != MM_OK)
        return s;

    s = emit(p, ph == PH_PREFILL ? OP_LIN_PRE : OP_LIN_DEC, (int)li, st,
             ab->qkv, ab->o, cv, M, mc->lin_v_heads, 0,
             ph == PH_PREFILL ? "lin_pre" : "lin_dec");
    if (s != MM_OK)
        return s;

    snprintf(nm, sizeof nm, "model.layers.%u.linear_attn.out_proj.weight", li);
    return emit_gemm(e, p, (int)li, "out_proj", ab->o, nm, ab->x, M, vdim,
                     mc->hidden);
}

/* ------------------------------------------------------------ the build */

mm_status mm_plan_build(mm_engine *e, mm_plan *p, mm_phase ph, uint32_t M)
{
    const mm_model_cfg *mc = &e->mc;
    mm_abufs *ab = &e->ab;
    char nm[80];
    const mm_tens *t;
    mm_status s;

    MM_REQUIRE(e && p && M > 0 && M <= e->cfg.chunk, MM_ERR_STATE);
    MM_REQUIRE(ab->x, MM_ERR_STATE);     /* acts reserved */
    p->phase = ph;
    p->M = M;
    p->n_ops = 0;
    p->captured = 0;

    t = mm_tens_find(&e->reg, "model.embed_tokens.weight");
    if (!t) {
        MM_LOGE("plan: missing embed_tokens");
        return MM_ERR_STATE;
    }
    s = emit(p, OP_EMBED, -1, t->is_fp4 ? t->w.data : t->v.data, NULL,
             ab->res, NULL, M, mc->hidden, t->is_fp4 ? 1 : 0, "embed");
    if (s != MM_OK)
        return s;

    for (uint32_t li = 0; li < mc->layers; li++)
    {
        /* input norm. Layer 0 normalises the embedding; later layers first
         * add the previous sublayer's output (held in e->ab.x). */
        snprintf(nm, sizeof nm,
                 "model.layers.%u.input_layernorm.weight", li);
        t = mm_tens_find(&e->reg, nm);
        if (!t) {
            MM_LOGE("plan: missing %s", nm);
            return MM_ERR_STATE;
        }
        if (li == 0)
            s = emit(p, OP_RMSNORM, (int)li, ab->res, t->v.data, ab->normed,
                     NULL, M, mc->hidden, 0, nm);
        else
            s = emit(p, OP_ADD_RMSNORM, (int)li, ab->res, ab->x, ab->normed,
                     t->v.data, M, mc->hidden, 0, nm);
        if (s != MM_OK)
            return s;

        if (mm_model_cfg_layer_is_full(mc, li + 1))
            s = emit_full_attn(e, p, li, M, ph);
        else
            s = emit_lin_attn(e, p, li, M, ph);
        if (s != MM_OK)
            return s;

        /* post-attention: res += x (attn out); normed = rmsnorm(res, ln2) */
        snprintf(nm, sizeof nm,
                 "model.layers.%u.post_attention_layernorm.weight", li);
        t = mm_tens_find(&e->reg, nm);
        if (!t) {
            MM_LOGE("plan: missing %s", nm);
            return MM_ERR_STATE;
        }
        s = emit(p, OP_ADD_RMSNORM, (int)li, ab->res, ab->x, ab->normed,
                 t->v.data, M, mc->hidden, 0, nm);
        if (s != MM_OK)
            return s;

        s = emit_mlp(e, p, li, M);
        if (s != MM_OK)
            return s;
    }

    /* final norm + lm head + sample */
    t = mm_tens_find(&e->reg, "model.norm.weight");
    if (!t) {
        MM_LOGE("plan: missing model.norm");
        return MM_ERR_STATE;
    }
    s = emit(p, OP_ADD_RMSNORM, -1, ab->res, ab->x, ab->normed, t->v.data,
             M, mc->hidden, 0, "final norm");
    if (s != MM_OK)
        return s;
    s = emit_gemm(e, p, -1, "lmhead", ab->normed, "lm_head.weight",
                  ab->logits, M, mc->hidden, mc->vocab);
    if (s != MM_OK)
        return s;
    s = emit(p, OP_SAMPLE, -1, ab->logits, ab->toks_dev, NULL, NULL, M, 0, 0,
             "sample");
    return s;
}

/* ------------------------------------------------------------- capture */

/* Capture one built plan into a CUDA graph (GPU only). In the host build
 * and with --no-graph this is a no-op: plans run by direct dispatch. */
static mm_status capture_one(mm_engine *e, mm_plan *p)
{
    if (e->on_host || e->cfg.no_graph)
        return MM_OK;
    if (p->n_ops == 0 || p->captured)
        return MM_OK;
    MM_CHECK(mm_graph_create(&p->graph));
    MM_CHECK(mm_graph_capture_begin(p->graph));
    MM_CHECK(plan_run_ops(e, p));
    MM_CHECK(mm_graph_commit(p->graph, NULL, 0, 0));
    p->captured = 1;
    return MM_OK;
}

/* Capture every built plan: the prefill cache, decode by n_slots
 * (1..concurrency), verify by draft window. Unbuilt slots (n_ops == 0)
 * are skipped, so this is safe to call at load after the common plans
 * have been built. */
mm_status mm_plan_capture_all(mm_engine *e)
{
    MM_REQUIRE(e, MM_ERR_STATE);
    for (int i = 0; i < 8; i++)
        MM_CHECK(capture_one(e, &e->pre_c[i]));
    for (uint32_t n = 0; n <= MM_MAX_CONCURRENCY; n++)
        MM_CHECK(capture_one(e, &e->dec[n]));
    for (uint32_t d = 0; d <= MM_DRAFT_MAX; d++)
        MM_CHECK(capture_one(e, &e->ver[d]));
    return MM_OK;
}