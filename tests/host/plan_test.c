/*
 * Host-side validation of the plan builder (src/plan/plan.c).
 *
 * Builds a real plan for a tiny model with no GPU: populates a fake tensor
 * registry, reserves the fixed activation map, then runs mm_plan_build()
 * for a prefill and a decode plan and checks the emitted op list. Links the
 * real plan.c + alloc.c + config.c + mimfer.c + cuda_rt.c (host stubs); the
 * single symbol the engine would otherwise provide (mm_tens_find) is stubbed
 * here as a plain linear search, which is all that implementation is.
 *
 * Build: gcc -std=c11 -Wall -Wextra -Werror -Iinclude \
 *          tests/host/plan_test.c src/plan/plan.c src/alloc/alloc.c \
 *          src/config/config.c src/core/mimfer.c src/cuda/cuda_rt.c \
 *          src/sampling/sampling.c -o /tmp/plan_test
 */
#include "mimfer/engine.h"
#include "mimfer/kernels.h"
#include "mimfer/plan.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Provided by the engine in a full build; a plain linear search. */
const mm_tens *mm_tens_find(const mm_tensreg *r, const char *name)
{
    for (uint32_t i = 0; i < r->n; i++)
        if (strcmp(r->t[i].name, name) == 0)
            return &r->t[i];
    return NULL;
}

/* The kernel dispatcher (src/kernels/kx.c) is not built yet; the test only
 * builds plans and never launches, so a stub satisfies the linker. */
mm_status mm_kx_invoke(const mm_kcall *kc)
{
    (void)kc;
    return MM_OK;
}

static int g_fail = 0;
#define CHECK(cond, msg)                                                      \
    do {                                                                      \
        if (!(cond)) {                                                        \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg);     \
            g_fail = 1;                                                       \
        }                                                                     \
    } while (0)

/* Register one bf16 tensor pointing into a shared dummy buffer (never
 * dereferenced at build time). */
static void add_tens(mm_tensreg *r, char *dummy, size_t *off, const char *nm)
{
    CHECK(r->n < MM_MAX_TENSORS, "registry full");
    mm_tens *t = &r->t[r->n++];
    snprintf(t->name, sizeof t->name, "%s", nm);
    t->is_fp4 = 0;
    t->v.data = dummy + *off;
    *off += 256;
}

int main(void)
{
    mm_engine e;
    char *dummy;
    size_t off = 0;
    char nm[96];
    mm_plan pre, dec;
    int i;

    memset(&e, 0, sizeof e);
    e.on_host = 1;
    e.cfg.chunk = 8;

    /* Tiny hybrid model: [lin,lin,lin,full] x 2. */
    e.mc.hidden = 64;
    e.mc.inter = 128;
    e.mc.layers = 8;
    e.mc.q_heads = 8;
    e.mc.kv_heads = 2;
    e.mc.head_dim = 32;
    e.mc.rope_dim = 16;
    e.mc.lin_k_heads = 4;
    e.mc.lin_k_dim = 16;
    e.mc.lin_v_heads = 8;
    e.mc.lin_v_dim = 16;
    e.mc.conv_k = 4;
    e.mc.vocab = 512;
    e.mc.max_pos = 1024;
    e.mc.rope_theta = 1e7f;
    e.mc.norm_eps = 1e-6f;
    e.mc.full_layer_step = 4;
    e.mc.full_layer_offset = 0;
    e.mc.mtp_layers = 0;
    CHECK(mm_model_cfg_finalize(&e.mc) == MM_OK, "model cfg finalize");
    CHECK(e.mc.n_full_layers == 2 && e.mc.n_lin_layers == 6,
          "expected 2 full / 6 lin layers");

    /* Host activation arena. */
    CHECK(mm_arena_init_host(&e.a_arena, 1 << 20, "a") == MM_OK,
          "arena init");

    /* Fake tensor registry: every name the plan builder looks up. */
    dummy = malloc(4 << 20);
    CHECK(dummy != NULL, "dummy buffer");
    add_tens(&e.reg, dummy, &off, "model.embed_tokens.weight");
    for (uint32_t li = 0; li < e.mc.layers; li++) {
        snprintf(nm, sizeof nm, "model.layers.%u.input_layernorm.weight", li);
        add_tens(&e.reg, dummy, &off, nm);
        if (mm_model_cfg_layer_is_full(&e.mc, li + 1)) {
            snprintf(nm, sizeof nm, "model.layers.%u.self_attn.qkv.weight", li);
            add_tens(&e.reg, dummy, &off, nm);
            snprintf(nm, sizeof nm,
                     "model.layers.%u.self_attn.o_proj.weight", li);
            add_tens(&e.reg, dummy, &off, nm);
        } else {
            snprintf(nm, sizeof nm,
                     "model.layers.%u.linear_attn.in_proj.weight", li);
            add_tens(&e.reg, dummy, &off, nm);
            snprintf(nm, sizeof nm,
                     "model.layers.%u.linear_attn.conv1d.weight", li);
            add_tens(&e.reg, dummy, &off, nm);
            snprintf(nm, sizeof nm,
                     "model.layers.%u.linear_attn.out_proj.weight", li);
            add_tens(&e.reg, dummy, &off, nm);
        }
        snprintf(nm, sizeof nm, "model.layers.%u.mlp.gate_proj.weight", li);
        add_tens(&e.reg, dummy, &off, nm);
        snprintf(nm, sizeof nm, "model.layers.%u.mlp.up_proj.weight", li);
        add_tens(&e.reg, dummy, &off, nm);
        snprintf(nm, sizeof nm, "model.layers.%u.mlp.down_proj.weight", li);
        add_tens(&e.reg, dummy, &off, nm);
        snprintf(nm, sizeof nm,
                 "model.layers.%u.post_attention_layernorm.weight", li);
        add_tens(&e.reg, dummy, &off, nm);
    }
    add_tens(&e.reg, dummy, &off, "model.norm.weight");
    add_tens(&e.reg, dummy, &off, "lm_head.weight");
    /* Reserve the fixed activation map. */
    CHECK(mm_plan_reserve_acts(&e, e.cfg.chunk) == MM_OK, "reserve_acts");
    CHECK(e.ab.x && e.ab.res && e.ab.normed && e.ab.qkv, "core abufs set");
    CHECK(e.ab.q && e.ab.k && e.ab.v && e.ab.o, "attn split buffers set");
    CHECK(e.ab.gate && e.ab.up && e.ab.mlp, "mlp buffers set");
    CHECK(e.ab.S && e.ab.part && e.ab.scratch && e.ab.logits, "aux set");
    CHECK(e.ab.toks_dev != NULL, "toks_dev set");

    /* Build the prefill plan (M = chunk). */
    memset(&pre, 0, sizeof pre);
    CHECK(mm_plan_build(&e, &pre, PH_PREFILL, e.cfg.chunk) == MM_OK,
          "build prefill");
    CHECK(pre.n_ops > 0 && pre.n_ops < MM_MAX_OPS, "prefill op count sane");
    CHECK(pre.ops[0].op == OP_EMBED, "first op is EMBED");
    CHECK(pre.ops[pre.n_ops - 1].op == OP_SAMPLE, "last op is SAMPLE");
    {
        int n_full_attn = 0, n_lin = 0;
        for (i = 0; i < (int)pre.n_ops; i++) {
            if (pre.ops[i].op == OP_ATT_PREFILL)
                n_full_attn++;
            else if (pre.ops[i].op == OP_LIN_PRE)
                n_lin++;
        }
        CHECK(n_full_attn == (int)e.mc.n_full_layers,
              "one ATT_PREFILL per full layer");
        CHECK(n_lin == (int)e.mc.n_lin_layers, "one LIN_PRE per lin layer");
    }

    /* Build the decode plan (M = 1 active slot). */
    memset(&dec, 0, sizeof dec);
    CHECK(mm_plan_build(&e, &dec, PH_DECODE, 1) == MM_OK, "build decode");
    CHECK(dec.n_ops > 0, "decode op count sane");
    {
        int n_att_dec = 0, n_att_pre = 0, n_lin_dec = 0;
        for (i = 0; i < (int)dec.n_ops; i++) {
            if (dec.ops[i].op == OP_ATT_DECODE)
                n_att_dec++;
            else if (dec.ops[i].op == OP_ATT_PREFILL)
                n_att_pre++;
            else if (dec.ops[i].op == OP_LIN_DEC)
                n_lin_dec++;
        }
        CHECK(n_att_dec == (int)e.mc.n_full_layers && n_att_pre == 0,
              "decode uses ATT_DECODE only, no ATT_PREFILL");
        CHECK(n_lin_dec == (int)e.mc.n_lin_layers,
              "one LIN_DEC per lin layer");
    }

    /* Same-layer op counts must match between prefill and decode. */
    CHECK(pre.n_ops == dec.n_ops,
          "prefill and decode emit the same number of ops");

    /* Rebuild to confirm the builder is idempotent. */
    memset(&pre, 0, sizeof pre);
    CHECK(mm_plan_build(&e, &pre, PH_PREFILL, e.cfg.chunk) == MM_OK,
          "rebuild prefill");

    /* Human-readable dump of the (re)built prefill plan. */
    {
        static char buf[1 << 16];
        mm_plan_dump(&pre, buf, sizeof buf);
        printf("%s", buf);
    }

    free(dummy);
    if (g_fail) {
        fprintf(stderr, "PLAN TEST FAILED\n");
        return 1;
    }
    printf("PLAN TEST PASSED (prefill %u ops, decode %u ops)\n",
           (unsigned)pre.n_ops, (unsigned)dec.n_ops);
    return 0;
}