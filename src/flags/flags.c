/*
 * CLI parsing: the server flag surface (see include/mimfer/flags.h).
 *
 * One table owns the grammar; the parser is a single pass over argv.
 * Scalar flags store straight into mm_engine_cfg by offset; the three
 * string-valued enums (--kv-dtype, --spec, --weights-profile) map a
 * token to an int with an explicit error that lists the accepted
 * values. mm_engine_cfg_validate runs last, so cross-field rules
 * (--spec without --draft-tokens, --lm-head-draft without --spec, the
 * YaRN factor) are enforced in one place (config.c).
 */
#include "mimfer/flags.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* flag value kinds */
enum { FV_BOOL, FV_U32, FV_F32, FV_U64, FV_INT, FV_STR, FV_KV,
       FV_SPEC, FV_PROFILE };

typedef struct fv {
    const char *name;    /* "--chunk"                                 */
    int         kind;    /* value kind (enum above)                   */
    size_t      off;     /* offsetof(mm_engine_cfg, field)            */
    long        lo, hi;  /* numeric range; -1 = no check on that side */
    const char *help;    /* one line for --help                       */
} fv;

static const fv FLAG_TAB[] = {
    { "--artifact",         FV_STR,     offsetof(mm_engine_cfg, artifact),
      -1, -1,
      "path to the .mimfer artifact: the engine opens + verifies the "
      "container and builds the tokenizer from its ART_TOK section. "
      "When absent, the built-in tiny shape + deterministic fake weights "
      "are used (the parity/golden path)" },
    { "--max-ctx",          FV_U32,     offsetof(mm_engine_cfg, max_ctx),
      1, MM_MAX_SEQ,
      "per-sequence context limit (1..262144; default 32768)" },
    { "--kv-capacity",      FV_U32,     offsetof(mm_engine_cfg, kv_capacity),
      -1, -1,
      "KV pool capacity in tokens; 0 = auto (default 0)" },
    { "--kv-dtype",         FV_KV,      offsetof(mm_engine_cfg, kv_dtype),
      -1, -1,
      "full-attention KV dtype: bf16 | fp8 (default bf16)" },
    { "--concurrency",      FV_U32,     offsetof(mm_engine_cfg, concurrency),
      1, MM_MAX_CONCURRENCY,
      "active decode slots (1..4; default 1)" },
    { "--chunk",            FV_U32,     offsetof(mm_engine_cfg, chunk),
      1, MM_MAX_PREFILL_CH,
      "prefill chunk in tokens (1..4096; default 2048)" },
    { "--draft-tokens",     FV_U32,     offsetof(mm_engine_cfg, draft),
      0, MM_DRAFT_MAX,
      "speculative draft window (0..8; default 0; must be 1..8 with "
      "--spec != off)" },
    { "--temperature",      FV_F32,     offsetof(mm_engine_cfg, temperature),
      -1, -1,
      "sampling temperature; 0 = greedy (default 0)" },
    { "--top-k",            FV_INT,     offsetof(mm_engine_cfg, top_k),
      0, 1 << 24,
      "top-k sampling; 0 = off (default 0)" },
    { "--top-p",            FV_F32,     offsetof(mm_engine_cfg, top_p),
      -1, -1,
      "nucleus sampling; 1.0 = off (default 1.0)" },
    { "--seed",             FV_U64,     offsetof(mm_engine_cfg, seed),
      -1, -1,
      "deterministic RNG seed (default 0x4d1bfe57)" },
    { "--verbose",          FV_INT,     offsetof(mm_engine_cfg, verbose),
      MM_LOG_ERR, MM_LOG_DBG,
      "log level: 0=err 1=warn 2=info 3=dbg (default 1)" },
    { "--no-graph",         FV_BOOL,    offsetof(mm_engine_cfg, no_graph),
      -1, -1,
      "disable CUDA graph capture (debug)" },
    { "--weights-profile",  FV_PROFILE, offsetof(mm_engine_cfg, profile_id),
      -1, -1,
      "Qwen3.8-27B NVFP4 checkpoint profile: quasar | neroued (applies "
      "the profile's context defaults; explicit flags win)" },
    { "--rope-yarn",        FV_BOOL,    offsetof(mm_engine_cfg, rope_yarn),
      -1, -1,
      "enable YaRN RoPE scaling (with --rope-yarn-factor)" },
    { "--rope-yarn-factor", FV_F32,     offsetof(mm_engine_cfg, rope_factor),
      -1, -1,
      "YaRN scale factor s (> 1.0; default 4.0)" },
    { "--rope-yarn-ctx",    FV_U32,     offsetof(mm_engine_cfg, rope_orig_ctx),
      0, MM_MAX_SEQ,
      "original context before scaling (0 = model max_pos; default 0)" },
    { "--spec",             FV_SPEC,    offsetof(mm_engine_cfg, spec),
      -1, -1,
      "speculative backend: off | mtp | dflash2 (validated scaffolding: "
      "non-off backends are refused at start until the draft/verify loop "
      "lands - docs/dflash2.md; default off)" },
    { "--lm-head-draft",    FV_BOOL,    offsetof(mm_engine_cfg,
                                                lm_head_draft),
      -1, -1,
      "score the draft window with the draft LM head (needs --spec)" },
    { "--vision",           FV_BOOL,    offsetof(mm_engine_cfg, vision),
      -1, -1,
      "enable the vision pipeline hook (image submission returns an "
      "explicit unsupported error until the pipeline lands)" },
};
#define N_FLAGS ((int)(sizeof FLAG_TAB / sizeof FLAG_TAB[0]))

static const fv *fv_find(const char *name)
{
    int i;
    for (i = 0; i < N_FLAGS; i++)
        if (!strcmp(FLAG_TAB[i].name, name))
            return &FLAG_TAB[i];
    return NULL;
}
/* Strict numeric parse: full consumption, no junk after the number. */
static int parse_uint(const char *s, unsigned long *out)
{
    char *end;
    unsigned long v = strtoul(s, &end, 10);
    if (*end || v > 0xfffffffful)
        return 0;
    *out = v;
    return 1;
}

static int parse_float(const char *s, float *out)
{
    char *end;
    float v = strtof(s, &end);
    if (*end || v != v)   /* trailing junk or NaN */
        return 0;
    *out = v;
    return 1;
}

size_t mm_flags_help(char *buf, size_t n)
{
    char h[MM_FLAGS_HELP_SZ];
    size_t off = 0;
    int i;

    off += (size_t)snprintf(h + off, MM_FLAGS_HELP_SZ - off,
        "mimfer " MIMFER_VERSION
        " — Qwen3.8-27B NVFP4 inference appliance\n"
        "            (one RTX PRO 4000 Blackwell, sm_120; the host build\n"
        "            runs the bit-deterministic CPU reference)\n\n"
        "usage: mimfer [flags]\n\n");
    for (i = 0; i < N_FLAGS; i++)
        off += (size_t)snprintf(h + off, MM_FLAGS_HELP_SZ - off,
                                "  %-21s %s\n", FLAG_TAB[i].name,
                                FLAG_TAB[i].help);
    off += (size_t)snprintf(h + off, MM_FLAGS_HELP_SZ - off,
                            "  %-21s %s\n", "--help, -h",
                            "print this text and exit\n");

    if (n > 0)
        memcpy(buf, h, (off + 1 < n) ? off + 1 : n);
    return off + 1;
}

mm_status mm_flags_parse(int argc, char **argv, mm_engine_cfg *cfg, int *help)
{
    mm_engine_cfg c;
    const mm_profile *prof = NULL;
    int have_max_ctx = 0, have_chunk = 0;
    int i;

    if (help)
        *help = 0;
    if (!cfg || argc < 1)
        return MM_ERR_STATE;
    mm_engine_cfg_default(&c);

    for (i = 1; i < argc; i++) {
        const fv *f = fv_find(argv[i]);
        if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            char h[MM_FLAGS_HELP_SZ];
            if (help)
                *help = 1;
            mm_flags_help(h, sizeof h);
            fputs(h, stdout);
            return MM_OK;
        }
        if (!f) {
            MM_LOGE("flags: unknown flag '%s' (try --help)", argv[i]);
            return MM_ERR_RANGE;
        }
        if (f->kind == FV_BOOL) {
            *(int *)(uintptr_t)((char *)&c + f->off) = 1;
            continue;
        }
        if (i + 1 >= argc) {
            MM_LOGE("flags: %s needs a value", f->name);
            return MM_ERR_STATE;
        }
        i++;
        switch (f->kind) {
        case FV_U32: {
            unsigned long v;
            if (!parse_uint(argv[i], &v)) {
                MM_LOGE("flags: %s: bad unsigned value '%s'",
                        f->name, argv[i]);
                return MM_ERR_RANGE;
            }
            if (f->lo >= 0 && v < (unsigned long)f->lo) {
                MM_LOGE("flags: %s: %lu below minimum %ld",
                        f->name, v, f->lo);
                return MM_ERR_RANGE;
            }
            if (f->hi >= 0 && v > (unsigned long)f->hi) {
                MM_LOGE("flags: %s: %lu above maximum %ld",
                        f->name, v, f->hi);
                return MM_ERR_RANGE;
            }
            *(uint32_t *)(uintptr_t)((char *)&c + f->off) = (uint32_t)v;
            if (!strcmp(f->name, "--max-ctx"))
                have_max_ctx = 1;
            if (!strcmp(f->name, "--chunk"))
                have_chunk = 1;
            break;
        }
        case FV_U64: {
            unsigned long long v;
            char *end;
            if (argv[i][0] == '-') {
                MM_LOGE("flags: %s: negative values are not allowed",
                        f->name);
                return MM_ERR_RANGE;
            }
            errno = 0;
            v = strtoull(argv[i], &end, 10);
            if (*end || errno == ERANGE) {
                MM_LOGE("flags: %s: bad unsigned value '%s'",
                        f->name, argv[i]);
                return MM_ERR_RANGE;
            }
            *(uint64_t *)(uintptr_t)((char *)&c + f->off) =
                (uint64_t)v;
            break;
        }
        case FV_INT: {
            long v;
            char *end;
            v = strtol(argv[i], &end, 10);
            if (*end || v < f->lo || v > f->hi) {
                MM_LOGE("flags: %s: value out of range [%ld..%ld]",
                        f->name, f->lo, f->hi);
                return MM_ERR_RANGE;
            }
            *(int *)(uintptr_t)((char *)&c + f->off) = (int)v;
            break;
        }
        case FV_F32: {
            float v;
            if (!parse_float(argv[i], &v)) {
                MM_LOGE("flags: %s: bad float value '%s'", f->name,
                        argv[i]);
                return MM_ERR_RANGE;
            }
            *(float *)(uintptr_t)((char *)&c + f->off) = v;
            break;
        }
        case FV_STR:
            *(const char **)(uintptr_t)((char *)&c + f->off) = argv[i];
            break;
        case FV_KV:
            if (!strcmp(argv[i], "bf16"))
                *(int *)(uintptr_t)((char *)&c + f->off) = MM_KV_BF16;
            else if (!strcmp(argv[i], "fp8"))
                *(int *)(uintptr_t)((char *)&c + f->off) = MM_KV_FP8;
            else {
                MM_LOGE("flags: %s: expected bf16 or fp8 (got '%s')",
                        f->name, argv[i]);
                return MM_ERR_RANGE;
            }
            break;
        case FV_SPEC:
            if (!strcmp(argv[i], "off"))
                *(int *)(uintptr_t)((char *)&c + f->off) = 0;
            else if (!strcmp(argv[i], "mtp"))
                *(int *)(uintptr_t)((char *)&c + f->off) = 1;
            else if (!strcmp(argv[i], "dflash2"))
                *(int *)(uintptr_t)((char *)&c + f->off) = 2;
            else {
                MM_LOGE("flags: %s: expected off, mtp or dflash2 (got '%s')",
                        f->name, argv[i]);
                return MM_ERR_RANGE;
            }
            break;
        case FV_PROFILE:
            prof = mm_profile_by_name(argv[i]);
            if (!prof) {
                MM_LOGE("flags: %s: unknown profile '%s' (available: %s, %s)",
                        f->name, argv[i], MM_PROFILES[0].name,
                        MM_PROFILES[1].name);
                return MM_ERR_RANGE;
            }
            *(uint32_t *)(uintptr_t)((char *)&c + f->off) = prof->id;
            break;
        default:
            return MM_ERR_STATE;
        }
    }

    /* Profile context defaults: applied only where the user was explicit. */
    if (prof) {
        if (!have_max_ctx)
            c.max_ctx = prof->def_ctx;
        if (!have_chunk)
            c.chunk = prof->def_chunk;
    }

    MM_CHECK(mm_engine_cfg_validate(&c));
    *cfg = c;
    return MM_OK;
}

