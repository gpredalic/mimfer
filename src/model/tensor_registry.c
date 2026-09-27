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

/* ------------------------------------------------- canonical tensor set */
/*
 * Public wrappers over gen_all_specs (load order): the packer fixture
 * builders and the weights loader enumerate the canonical set through
 * these so a single source of truth defines the tensor layout.
 */
size_t mm_model_spec_count(const mm_model_cfg *mc)
{
    uint32_t cap;
    int n;
    tensor_spec *sp;

    if (!mc)
        return 0;
    cap = mc->layers * 12u + 16u;
    sp = malloc(cap * sizeof *sp);
    if (!sp)
        return 0;
    n = gen_all_specs(mc, sp, cap);
    free(sp);
    return n > 0 ? (size_t)n : 0;
}

mm_status mm_model_spec_at(const mm_model_cfg *mc, size_t i, char *name,
                           size_t name_sz, uint32_t *rows, uint32_t *cols)
{
    uint32_t cap;
    int n;
    tensor_spec *sp;

    if (!mc || !name || name_sz < 48 || !rows || !cols)
        return MM_ERR_RANGE;
    if (i >= (size_t)(mc->layers * 12u + 16u))
        return MM_ERR_RANGE;
    cap = mc->layers * 12u + 16u;
    sp = malloc(cap * sizeof *sp);
    if (!sp)
        return MM_ERR_NOMEM;
    n = gen_all_specs(mc, sp, cap);
    if (n <= 0 || i >= (size_t)n) {
        free(sp);
        return MM_ERR_RANGE;
    }
    snprintf(name, name_sz, "%s", sp[i].name);
    *rows = sp[i].rows;
    *cols = sp[i].cols;
    free(sp);
    return MM_OK;
}

/* ---------------------------------------------------- ART_MODEL parse */
/*
 * Little-endian fixed layout (documented in tensor_registry.h). The
 * target is little-endian and the format is LE, so these are memcpy
 * (same convention as src/artifact/artifact.c).
 */
static uint32_t am_rd32(const uint8_t *p)
{
    uint32_t v;
    memcpy(&v, p, 4);
    return v;
}
static float am_rd32f(const uint8_t *p)
{
    float f;
    memcpy(&f, p, 4);
    return f;
}

mm_status mm_art_model_parse(const uint8_t *data, size_t len,
                             mm_model_cfg *out)
{
    if (!data || !out)
        return MM_ERR_STATE;
    if (len < MM_ART_MODEL_FIXED_SZ) {
        MM_LOGE("art_model: payload %zu bytes < fixed layout %u bytes",
                len, (unsigned)MM_ART_MODEL_FIXED_SZ);
        return MM_ERR_CORRUPT;
    }
    if (am_rd32(data + 0) != MM_ART_MODEL_VERSION) {
        MM_LOGE("art_model: unsupported version %u (this build reads %u)",
                am_rd32(data + 0), (unsigned)MM_ART_MODEL_VERSION);
        return MM_ERR_CORRUPT;
    }
    if (am_rd32(data + 4) != 0) {
        MM_LOGE("art_model: unsupported flags %u (only 0 is defined)",
                am_rd32(data + 4));
        return MM_ERR_CORRUPT;
    }
    if (len > MM_ART_MODEL_FIXED_SZ)
        MM_LOGI("art_model: %zu reserved extra bytes after the fixed "
                "layout (ignored by v1)", len - MM_ART_MODEL_FIXED_SZ);

    memset(out, 0, sizeof *out);
    out->hidden          = am_rd32(data + 8);
    out->inter           = am_rd32(data + 12);
    out->layers          = am_rd32(data + 16);
    out->q_heads         = am_rd32(data + 20);
    out->kv_heads        = am_rd32(data + 24);
    out->head_dim        = am_rd32(data + 28);
    out->rope_dim        = am_rd32(data + 32);
    out->lin_k_heads     = am_rd32(data + 36);
    out->lin_k_dim       = am_rd32(data + 40);
    out->lin_v_heads     = am_rd32(data + 44);
    out->lin_v_dim       = am_rd32(data + 48);
    out->conv_k          = am_rd32(data + 52);
    out->vocab           = am_rd32(data + 56);
    out->max_pos         = am_rd32(data + 60);
    out->rope_theta      = am_rd32f(data + 64);
    out->norm_eps        = am_rd32f(data + 68);
    out->full_layer_step = am_rd32(data + 72);
    out->full_layer_offset = am_rd32(data + 76);
    out->mtp_layers      = am_rd32(data + 80);
    return MM_OK;
}

/* --------------------------------------------------- ART_WEIGHTS load */
/*
 * Real artifact weight loading (see tensor_registry.h for the on-disk
 * index layout and the validation gates). The payload passed in starts
 * at the tensor index (the engine strips any sub-checksum block first;
 * the section's checksums were already gated by mm_art_verify at
 * create). All reads are bounds-checked before use; every failure logs
 * the offending entry and returns with no partial success.
 */
static uint16_t aw_rd16(const uint8_t *p)
{
    uint16_t v;
    memcpy(&v, p, 2);
    return v;
}
static uint64_t aw_rd64(const uint8_t *p)
{
    uint64_t v;
    memcpy(&v, p, 8);
    return v;
}

mm_status mm_model_weights_load(mm_engine *e, const uint8_t *data, size_t len)
{
    const mm_model_cfg *mc;
    uint32_t cap, n, i;
    size_t p, total = 0;
    int nspec;
    mm_status s = MM_OK;
    tensor_spec *sp;
    struct wtent {
        char     name[48];
        uint32_t rows, cols;
        uint64_t off, nbytes;
    } *ents;

    if (!e || !data || !e->w_arena.base) {
        MM_LOGE("art_weights: bad state (engine/arena/payload null)");
        return MM_ERR_STATE;
    }
    if (len < 12) {
        MM_LOGE("art_weights: payload %zu bytes < 12-byte index header",
                len);
        return MM_ERR_CORRUPT;
    }
    if (am_rd32(data + 0) != MM_ART_WEIGHTS_VERSION) {
        MM_LOGE("art_weights: unsupported index version %u (this build "
                "reads %u)", am_rd32(data + 0),
                (unsigned)MM_ART_WEIGHTS_VERSION);
        return MM_ERR_CORRUPT;
    }
    if (am_rd32(data + 4) != 0) {
        MM_LOGE("art_weights: unsupported flags %u (only 0 is defined)",
                am_rd32(data + 4));
        return MM_ERR_CORRUPT;
    }
    n = am_rd32(data + 8);
    if (n > MM_MAX_TENSORS) {
        MM_LOGE("art_weights: n_tensors %u > MM_MAX_TENSORS %d", n,
                MM_MAX_TENSORS);
        return MM_ERR_CORRUPT;
    }

    /* Count gate: the index must carry exactly the canonical set of the
     * model section's shape (a mismatch means the weights belong to a
     * different model than the metadata claims). */
    mc = &e->mc;
    cap = mc->layers * 12u + 16u;
    sp = malloc(cap * sizeof *sp);
    ents = calloc(n, sizeof *ents);
    if (!sp || !ents) {
        MM_LOGE("art_weights: host allocation failure (n_tensors %u)", n);
        s = MM_ERR_NOMEM;
        goto fail;
    }
    nspec = gen_all_specs(mc, sp, cap);
    if (nspec <= 0 || n != (uint32_t)nspec) {
        MM_LOGE("art_weights: n_tensors %u != canonical count %d for the "
                "model shape (weights/model mismatch)", n, nspec);
        s = MM_ERR_SHAPE;
        goto fail;
    }

    /* Index walk: bounds, names, geometry, dtype, payload containment. */
    p = 12;
    for (i = 0; i < n; i++) {
        uint16_t nl;
        uint8_t dt;

        if (p + 2 > len)
            goto truncated;
        nl = aw_rd16(data + p);
        p += 2;
        if (nl < 1 || nl > 48) {
            MM_LOGE("art_weights: entry %u: bad name_len %u (1..48)",
                    i, nl);
            s = MM_ERR_CORRUPT;
            goto fail;
        }
        if (p + (size_t)nl + 28 > len)
            goto truncated;
        if (data[p + (size_t)nl - 1] != 0) {
            MM_LOGE("art_weights: entry %u: name not NUL-terminated", i);
            s = MM_ERR_CORRUPT;
            goto fail;
        }
        memcpy(ents[i].name, data + p, nl);
        p += nl;
        ents[i].rows = am_rd32(data + p);
        p += 4;
        ents[i].cols = am_rd32(data + p);
        p += 4;
        dt = data[p];
        if (dt == (uint8_t)MM_DT_FP4E2M1) {
            MM_LOGE("art_weights: entry %u (\"%s\"): fp4 weights are not "
                    "implemented in this build (bf16 only)", i,
                    ents[i].name);
            s = MM_ERR_UNSUPPORTED;
            goto fail;
        }
        if (dt != (uint8_t)MM_DT_BF16) {
            MM_LOGE("art_weights: entry %u (\"%s\"): dtype %u is not "
                    "bf16 (unsupported)", i, ents[i].name, dt);
            s = MM_ERR_UNSUPPORTED;
            goto fail;
        }
        p += 4;
        ents[i].off = aw_rd64(data + p);
        p += 8;
        ents[i].nbytes = aw_rd64(data + p);
        p += 8;
        if (ents[i].nbytes != (uint64_t)ents[i].rows * ents[i].cols * 2) {
            MM_LOGE("art_weights: entry %u (\"%s\"): nbytes %llu != "
                    "rows*cols*2 (%u x %u)", i, ents[i].name,
                    (unsigned long long)ents[i].nbytes, ents[i].rows,
                    ents[i].cols);
            s = MM_ERR_CORRUPT;
            goto fail;
        }
        /* Containment: after the index table, inside the payload. */
        if (ents[i].off < p || ents[i].off + ents[i].nbytes > len) {
            MM_LOGE("art_weights: entry %u (\"%s\"): payload region "
                    "[%llu..%llu) outside the %zu-byte payload (index "
                    "ends at %zu)", i, ents[i].name,
                    (unsigned long long)ents[i].off,
                    (unsigned long long)(ents[i].off + ents[i].nbytes),
                    len, p);
            s = MM_ERR_CORRUPT;
            goto fail;
        }
        total += (size_t)ents[i].nbytes;
    }
    if (p > len)
        goto truncated;

    /* Order gate: entry i must be canonical spec i (name + rows + cols).
     * With the count gate this is the metadata/runtime consistency
     * check: the index must describe exactly the model section's tensor
     * set, in load order. */
    for (i = 0; i < n; i++) {
        if (strcmp(ents[i].name, sp[i].name) != 0 ||
            ents[i].rows != sp[i].rows || ents[i].cols != sp[i].cols) {
            MM_LOGE("art_weights: entry %u: got \"%s\" [%ux%u], canonical "
                    "spec is \"%s\" [%ux%u] (weights do not match the "
                    "model section)", i, ents[i].name, ents[i].rows,
                    ents[i].cols, sp[i].name, sp[i].rows, sp[i].cols);
            s = MM_ERR_SHAPE;
            goto fail;
        }
    }

    /* Stage: arena slot (256-aligned) + host-mirror copy + bf16 view.
     * Same allocation pattern as the fake fill, so the arena capacity
     * computed from mm_model_weights_bytes + pool headroom holds. On a
     * staging failure the arena holds a partial set; the caller owns
     * the whole create/destroy cycle, so destroy cleans it up. */
    for (i = 0; i < n; i++) {
        size_t bytes = (size_t)ents[i].nbytes;
        void *slot, *hp;
        mm_tensor v;

        s = mm_arena_alloc(&e->w_arena, bytes, 256, ents[i].name, &slot);
        if (s != MM_OK) {
            MM_LOGE("art_weights: arena exhausted at tensor %u/%u "
                    "(\"%s\", %zu bytes)", i, n, ents[i].name, bytes);
            goto fail;
        }
        hp = mm_arena_host_ptr(&e->w_arena, slot);
        if (!hp) {
            MM_LOGE("art_weights: no host mirror for \"%s\" (staging "
                    "tensor %u/%u)", ents[i].name, i, n);
            s = MM_ERR_STATE;
            goto fail;
        }
        memcpy(hp, data + ents[i].off, bytes);
        v = mm_t_view(slot, MM_DT_BF16, 2, ents[i].rows, ents[i].cols,
                      0, 0, bytes);
        s = mm_tens_add(&e->reg, ents[i].name, 0, &v, NULL);
        if (s != MM_OK) {
            MM_LOGE("art_weights: registry add failed for \"%s\" "
                    "(staging tensor %u/%u)", ents[i].name, i, n);
            goto fail;
        }
    }
    MM_LOGW("art_weights: staged %u tensors (%zu bytes) into w_arena",
            n, total);
    s = MM_OK;
    goto fail;
fail:
    free(sp);
    free(ents);
    return s;
truncated:
    MM_LOGE("art_weights: index truncated at entry %u (offset %zu of a "
            "%zu-byte payload)", i, p, len);
    s = MM_ERR_CORRUPT;
    goto fail;
}

