/*
 * Tensor registry implementation (see src/model/tensor_registry.h).
 *
 * mm_tens_add / mm_tens_find are the engine's weight ownership layer: a
 * fixed-size (MM_MAX_TENSORS) array of (name, view) entries, linear
 * lookup. No hashmap, no tree, no per-tensor allocation — the table is the
 * owner of the registry and the mm_tensor/mm_wt views it holds are
 * zero-copy references into the weight arena (which owns the bytes).
 */
#include "mimfer/engine.h"
#include "mimfer/tensor.h"
#include "tensor_registry.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---------------------------------------------------------------- find */

/* Linear scan. Qwen-only scale (a few hundred names) makes this a
 * non-issue: it runs once per tensor at load, never in the token path. */
const mm_tens *mm_tens_find(const mm_tensreg *r, const char *name)
{
    if (!r || !name)
        return NULL;
    for (uint32_t i = 0; i < r->n; i++)
        if (strncmp(r->t[i].name, name, sizeof r->t[i].name) == 0)
            return &r->t[i];
    return NULL;
}

/* Append one entry. Fails on a full table or a name already present
 * (duplicates would make find() ambiguous). `v`/`w` are borrowed views,
 * not owned (the arena owns the bytes). */
mm_status mm_tens_add(mm_tensreg *r, const char *name, int is_fp4,
                      const mm_tensor *v, const mm_wt *w)
{
    if (!r || !name)
        return MM_ERR_STATE;
    if (r->n >= MM_MAX_TENSORS)
        return MM_ERR_CAPACITY;
    if (strnlen(name, sizeof r->t[0].name) >= sizeof r->t[0].name)
        return MM_ERR_RANGE;       /* name would truncate */
    if (mm_tens_find(r, name))
        return MM_ERR_STATE;       /* duplicate name */
    mm_tens *t = &r->t[r->n++];
    memset(t, 0, sizeof *t);
    snprintf(t->name, sizeof t->name, "%s", name);
    t->is_fp4 = is_fp4;
    t->v = v ? *v : (mm_tensor){0};
    t->w = w ? *w : (mm_wt){0};
    return MM_OK;
}

/* ------------------------------------------------------------ model set */
/* The standard Qwen3.8 tensor layout. Both the arena sizing and the fake
 * fill enumerate it, so it lives in one place. `rows` = output features,
 * `cols` = input features (a 1-D tensor has cols == 1). */

typedef enum { T_WT = 0, T_NORM, T_BIAS, T_ALOG } tens_kind;

typedef struct tensor_spec {
    char     name[48];
    uint32_t rows, cols;
    tens_kind kind;
} tensor_spec;

/* Push one spec, bounds-checked. `fmt` may carry a %u for the layer index
 * (li); names without a specifier simply ignore it. Returns 0 on ok, -1 on
 * overflow. */
static int put_spec(tensor_spec *o, uint32_t *n, uint32_t cap, const char *fmt,
                    uint32_t li, uint32_t rows, uint32_t cols, tens_kind kind)
{
    if (*n >= cap)
        return -1;
    /* fmt carries at most one %u (the layer index); extra args are ignored
     * when the name has no specifier, so a plain snprintf is correct. */
    snprintf(o[(*n)].name, sizeof o[0].name, fmt, (unsigned)li);
    o[(*n)].rows = rows;
    o[(*n)].cols = cols;
    o[(*n)].kind = kind;
    (*n)++;
    return 0;
}

/* Emit every tensor spec for the model into out[0..cap). Returns the count
 * or -1 if cap is too small. */
static int gen_all_specs(const mm_model_cfg *mc, tensor_spec *out,
                         uint32_t cap)
{
    uint32_t n = 0, li;
    uint32_t ch = 2u * mc->lin_k_heads * mc->lin_k_dim +
                 mc->lin_v_heads * mc->lin_v_dim;      /* conv channels */

    if (put_spec(out, &n, cap, "model.embed_tokens.weight", 0, mc->vocab,
                 mc->hidden, T_WT) ||
        put_spec(out, &n, cap, "lm_head.weight", 0, mc->vocab, mc->hidden,
                 T_WT) ||
        put_spec(out, &n, cap, "model.norm.weight", 0, mc->hidden, 1, T_NORM))
        return -1;

    for (li = 0; li < mc->layers; li++) {
        if (put_spec(out, &n, cap,
                     "model.layers.%u.input_layernorm.weight", li,
                     mc->hidden, 1, T_NORM) ||
            put_spec(out, &n, cap,
                     "model.layers.%u.post_attention_layernorm.weight", li,
                     mc->hidden, 1, T_NORM) ||
            put_spec(out, &n, cap, "model.layers.%u.mlp.gate_proj.weight", li,
                     mc->inter, mc->hidden, T_WT) ||
            put_spec(out, &n, cap, "model.layers.%u.mlp.up_proj.weight", li,
                     mc->inter, mc->hidden, T_WT) ||
            put_spec(out, &n, cap, "model.layers.%u.mlp.down_proj.weight", li,
                     mc->hidden, mc->inter, T_WT))
            return -1;

        if (mm_model_cfg_layer_is_full(mc, li + 1)) {
            uint32_t qkv = (mc->q_heads + 2u * mc->kv_heads) * mc->head_dim;
            if (put_spec(out, &n, cap, "model.layers.%u.self_attn.qkv.weight",
                         li, qkv, mc->hidden, T_WT) ||
                put_spec(out, &n, cap,
                         "model.layers.%u.self_attn.o_proj.weight", li,
                         mc->hidden, mc->q_heads * mc->head_dim, T_WT))
                return -1;
        } else {
            uint32_t in_dim = ch + 2u * mc->lin_v_heads;  /* qkv + g,beta */
            if (put_spec(out, &n, cap,
                         "model.layers.%u.linear_attn.in_proj.weight", li,
                         in_dim, mc->hidden, T_WT) ||
                put_spec(out, &n, cap,
                         "model.layers.%u.linear_attn.conv1d.weight", li,
                         ch, mc->conv_k, T_WT) ||
                put_spec(out, &n, cap, "model.layers.%u.linear_attn.dt_bias",
                         li, 2u * mc->lin_v_heads, 1, T_BIAS) ||
                put_spec(out, &n, cap, "model.layers.%u.linear_attn.A_log", li,
                         mc->lin_v_heads, 1, T_ALOG) ||
                put_spec(out, &n, cap,
                         "model.layers.%u.linear_attn.norm.weight", li,
                         mc->lin_v_heads * mc->lin_v_dim, 1, T_NORM) ||
                put_spec(out, &n, cap,
                         "model.layers.%u.linear_attn.out_proj.weight", li,
                         mc->hidden, mc->lin_v_heads * mc->lin_v_dim, T_WT))
                return -1;
        }
    }
    return (int)n;
}

/* Total bf16 bytes for the standard tensor set (0 if it would not fit). */
size_t mm_model_weights_bytes(const mm_model_cfg *mc)
{
    tensor_spec *sp;
    size_t total = 0;
    int n, i;
    uint32_t cap;

    if (!mc)
        return 0;
    cap = mc->layers * 12u + 16u;
    sp = malloc(cap * sizeof *sp);
    if (!sp)
        return 0;
    n = gen_all_specs(mc, sp, cap);
    if (n > 0)
        for (i = 0; i < n; i++)
            total += (size_t)sp[i].rows * sp[i].cols * 2;
    free(sp);
    return n > 0 ? total : 0;
}

/* ------------------------------------------------------------ fake fill */

/* Deterministic tiny-weight generator (splitmix64, see mimfer.h). One
 * stream per fill; the same seed reproduces the same weights. Weights are
 * small (|v| < 0.1) so values stay bounded through the layer stack (the
 * per-layer RMSNorm re-normalises to unit RMS regardless). */
static void fill_tensor(uint16_t *dst, size_t nelem, tens_kind kind,
                        uint64_t *st)
{
    for (size_t i = 0; i < nelem; i++) {
        float v;
        switch (kind) {
        case T_NORM: v = 1.0f; break;
        case T_BIAS: v = 0.0f; break;
        case T_ALOG: v = 0.1f; break;
        default: {
            uint64_t r = mm_splitmix64(st);
            v = ((float)((r >> 40) & 0xffff) / 65536.0f - 0.5f) * 0.2f;
            break;
        }
        }
        dst[i] = mm_bf16_from_f32(v);
    }
}

mm_status mm_model_fill_fake(mm_engine *e, uint64_t seed)
{
    const mm_model_cfg *mc = &e->mc;
    mm_arena *w = &e->w_arena;
    uint32_t cap;
    uint64_t st;
    tensor_spec *sp;
    int n, i;

    MM_REQUIRE(e && w->base, MM_ERR_STATE);
    cap = mc->layers * 12u + 16u;
    st = seed ? seed : 1;
    sp = malloc(cap * sizeof *sp);
    if (!sp)
        return MM_ERR_NOMEM;
    n = gen_all_specs(mc, sp, cap);
    if (n <= 0) {
        free(sp);
        return MM_ERR_SHAPE;
    }
    for (i = 0; i < n; i++) {
        const tensor_spec *s = &sp[i];
        size_t bytes = (size_t)s->rows * s->cols * 2;
        void *p;
        void *hp;
        mm_tensor v;

        MM_CHECK(mm_arena_alloc(w, bytes, 256, s->name, &p));
        /* Fill through the arena's host-side view: in the MM_WITH_CUDA
         * build the arena's canonical pointers are device addresses, so
         * the host writes the mirror; the engine H2D's the used region
         * after load. Host builds: identity mapping, the same pointer. */
        hp = mm_arena_host_ptr(w, p);
        if (!hp)
            return MM_ERR_STATE;
        fill_tensor((uint16_t *)hp, (size_t)s->rows * s->cols, s->kind, &st);
        v = mm_t_view(p, MM_DT_BF16, 2, s->rows, s->cols, 0, 0, bytes);
        MM_CHECK(mm_tens_add(&e->reg, s->name, 0, &v, NULL));
    }
    free(sp);
    return MM_OK;
}

