/*
 * Host test for the CLI flag surface (src/flags/flags.c,
 * include/mimfer/flags.h): every flag is parsed, range-checked and
 * mapped onto mm_engine_cfg; malformed input fails with an explicit
 * error; the weights profile applies its context defaults only where
 * the user was not explicit; the help text lists every flag.
 *
 * The cross-field rules (--spec vs --draft-tokens vs --lm-head-draft,
 * the YaRN factor) live in mm_engine_cfg_validate and are exercised
 * through the parser here; the engine-level refusals (spec backends,
 * vision) are covered by engine_features_test.c.
 */
#include "mimfer/flags.h"

#include <stdio.h>
#include <string.h>

static int g_fail = 0;
#define CHECK(cond, msg)                                                      \
    do {                                                                      \
        if (!(cond)) {                                                        \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg);     \
            g_fail = 1;                                                       \
        }                                                                     \
    } while (0)

/* Parse n flags (string array) with "mimfer" as argv[0]. */
static mm_status parse(char *const flags[], int n, mm_engine_cfg *cfg,
                       int *help)
{
    char *av[64];
    int i;

    if (n + 1 > 64)
        return MM_ERR_STATE;
    av[0] = (char *)"mimfer";
    for (i = 0; i < n; i++)
        av[i + 1] = flags[i];
    return mm_flags_parse(n + 1, av, cfg, help);
}

static void test_defaults(void)
{
    mm_engine_cfg cfg;
    mm_engine_cfg def;

    CHECK(parse(NULL, 0, &cfg, NULL) == MM_OK, "no flags -> OK");
    mm_engine_cfg_default(&def);
    CHECK(cfg.max_ctx == def.max_ctx, "default max_ctx");
    CHECK(cfg.chunk == def.chunk, "default chunk");
    CHECK(cfg.concurrency == def.concurrency, "default concurrency");
    CHECK(cfg.kv_dtype == def.kv_dtype, "default kv_dtype");
    CHECK(cfg.temperature == def.temperature, "default temperature");
    CHECK(cfg.top_k == def.top_k && cfg.top_p == def.top_p,
          "default policy");
    CHECK(cfg.seed == def.seed, "default seed");
    CHECK(cfg.verbose == def.verbose, "default verbose");
    CHECK(cfg.no_graph == 0, "default no_graph");
    CHECK(cfg.rope_yarn == 0, "default rope_yarn off");
    CHECK(cfg.rope_factor == 4.0f, "default rope_factor 4.0");
    CHECK(cfg.rope_orig_ctx == 0, "default rope_orig_ctx 0");
    CHECK(cfg.spec == 0 && cfg.draft == 0 && cfg.lm_head_draft == 0,
          "default spec off");
    CHECK(cfg.profile_id == 0, "default profile none");
    CHECK(cfg.vision == 0, "default vision off");
}

static void test_scalars(void)
{
    char *f[] = {
        "--artifact", "/tmp/x.mimfer",
        "--max-ctx", "4096",
        "--kv-capacity", "8192",
        "--kv-dtype", "fp8",
        "--concurrency", "2",
        "--chunk", "512",
        "--temperature", "0.7",
        "--top-k", "50",
        "--top-p", "0.9",
        "--seed", "42",
        "--verbose", "3",
        "--no-graph",
    };
    mm_engine_cfg cfg;

    CHECK(parse(f, (int)(sizeof f / sizeof f[0]), &cfg, NULL) == MM_OK,
          "scalar flags -> OK");
    CHECK(cfg.artifact != NULL && !strcmp(cfg.artifact, "/tmp/x.mimfer"),
          "artifact path stored");
    CHECK(cfg.max_ctx == 4096, "max_ctx");
    CHECK(cfg.kv_capacity == 8192, "kv_capacity");
    CHECK(cfg.kv_dtype == MM_KV_FP8, "kv_dtype fp8");
    CHECK(cfg.concurrency == 2, "concurrency");
    CHECK(cfg.chunk == 512, "chunk");
    CHECK(cfg.temperature == 0.7f, "temperature");
    CHECK(cfg.top_k == 50, "top_k");
    CHECK(cfg.top_p == 0.9f, "top_p");
    CHECK(cfg.seed == 42, "seed");
    CHECK(cfg.verbose == MM_LOG_DBG, "verbose");
    CHECK(cfg.no_graph == 1, "no_graph");
}

static void test_yarn_flags(void)
{
    char *f1[] = { "--rope-yarn", "--rope-yarn-factor", "2.0",
                   "--rope-yarn-ctx", "131072" };
    char *f2[] = { "--rope-yarn" };
    char *f3[] = { "--rope-yarn-factor", "1.0" };
    char *f4[] = { "--rope-yarn-factor", "abc" };
    mm_engine_cfg cfg;

    CHECK(parse(f1, 5, &cfg, NULL) == MM_OK, "yarn flags -> OK");
    CHECK(cfg.rope_yarn == 1, "rope_yarn on");
    CHECK(cfg.rope_factor == 2.0f, "rope_factor 2.0");
    CHECK(cfg.rope_orig_ctx == 131072, "rope_orig_ctx");

    /* yarn on, factor falls back to the default 4.0 (still valid). */
    CHECK(parse(f2, 1, &cfg, NULL) == MM_OK, "yarn without factor -> OK");
    CHECK(cfg.rope_factor == 4.0f, "default factor applied");

    /* factor 1.0 is not an extension: the validator refuses. */
    CHECK(parse(f3, 2, &cfg, NULL) == MM_ERR_RANGE,
          "yarn factor 1.0 refused");
    /* malformed float */
    CHECK(parse(f4, 2, &cfg, NULL) == MM_ERR_RANGE, "bad factor refused");
}

static void test_spec_flags(void)
{
    char *ok1[] = { "--spec", "dflash2", "--draft-tokens", "4" };
    char *ok2[] = { "--spec", "mtp", "--draft-tokens", "1",
                    "--lm-head-draft" };
    char *e1[] = { "--spec", "dflash2" };
    char *e2[] = { "--draft-tokens", "4" };
    char *e3[] = { "--lm-head-draft" };
    char *e4[] = { "--spec", "nope" };
    char *e5[] = { "--spec", "dflash2", "--draft-tokens", "9" };
    mm_engine_cfg cfg;

    /* The flags parse and validate: the backends are real options. */
    CHECK(parse(ok1, 4, &cfg, NULL) == MM_OK, "dflash2 + window -> OK");
    CHECK(cfg.spec == 2, "spec dflash2");
    CHECK(cfg.draft == 4, "draft window 4");

    CHECK(parse(ok2, 5, &cfg, NULL) == MM_OK, "mtp + draft head -> OK");
    CHECK(cfg.spec == 1, "spec mtp");
    CHECK(cfg.lm_head_draft == 1, "lm_head_draft");

    /* Cross-field refusals (mm_engine_cfg_validate). */
    CHECK(parse(e1, 2, &cfg, NULL) == MM_ERR_RANGE,
          "spec without a window refused");
    CHECK(parse(e2, 2, &cfg, NULL) == MM_ERR_RANGE,
          "window without a backend refused");
    CHECK(parse(e3, 1, &cfg, NULL) == MM_ERR_RANGE,
          "draft head without a backend refused");
    CHECK(parse(e4, 2, &cfg, NULL) == MM_ERR_RANGE,
          "unknown backend refused");
    CHECK(parse(e5, 4, &cfg, NULL) == MM_ERR_RANGE,
          "window above MM_DRAFT_MAX refused");
}

static void test_profile_flags(void)
{
    char *f1[] = { "--weights-profile", "quasar" };
    char *f2[] = { "--weights-profile", "quasar", "--max-ctx", "8192" };
    char *f3[] = { "--weights-profile", "QUASAR" };
    char *f4[] = { "--weights-profile", "bogus" };
    const mm_profile *p;
    mm_engine_cfg cfg;

    CHECK(parse(f1, 2, &cfg, NULL) == MM_OK, "profile quasar -> OK");
    CHECK(cfg.profile_id == 1, "profile id 1");
    p = mm_profile_get(cfg.profile_id);
    CHECK(p != NULL && !strcmp(p->name, "quasar"), "profile resolves");
    CHECK(cfg.max_ctx == p->def_ctx, "profile default max_ctx applied");
    CHECK(cfg.chunk == p->def_chunk, "profile default chunk applied");
    CHECK(!strcmp(p->repo, "MirkoCovizzi/Qwen3.8-27B-QUASAR-NVFP4-NInfer"),
          "quasar checkpoint identity");
    CHECK(p->max_ctx == 262144, "quasar native ceiling");
    CHECK(p->mtp_layers == 1, "quasar MTP layers");
    CHECK(p->dflash2_max >= 1 && p->dflash2_max <= (uint32_t)MM_DRAFT_MAX,
          "quasar dflash2 ceiling in range");

    /* Explicit flags win over the profile defaults. */
    CHECK(parse(f2, 4, &cfg, NULL) == MM_OK, "explicit max_ctx wins");
    CHECK(cfg.profile_id == 1 && cfg.max_ctx == 8192, "explicit max_ctx");
    CHECK(cfg.chunk == p->def_chunk, "chunk still from the profile");

    /* Case-insensitive name. */
    CHECK(parse(f3, 2, &cfg, NULL) == MM_OK, "case-insensitive profile");
    CHECK(cfg.profile_id == 1, "id 1 regardless of case");

    /* Unknown profile. */
    CHECK(parse(f4, 2, &cfg, NULL) == MM_ERR_RANGE, "unknown profile refused");
    CHECK(mm_profile_by_name(NULL) == NULL, "NULL name refused");
    CHECK(mm_profile_get(0) == NULL, "id 0 = none");
    CHECK(mm_profile_get(99) == NULL, "unknown id refused");

    /* The second profile. */
    {
        const mm_profile *n = mm_profile_by_name("neroued");
        CHECK(n != NULL && n->id == 2, "neroued resolves to id 2");
        CHECK(!strcmp(n->repo, "Neroued/Qwen3.8-27B-nvfp4-NInfer"),
              "neroued checkpoint identity");
    }
}

static void test_vision_flags(void)
{
    char *f[] = { "--vision" };
    mm_engine_cfg cfg;

    CHECK(parse(f, 1, &cfg, NULL) == MM_OK, "vision flag -> OK");
    CHECK(cfg.vision == 1, "vision on");
}

static void test_errors(void)
{
    char *e1[] = { "--bogus", "1" };
    char *e2[] = { "--chunk" };
    char *e3[] = { "--max-ctx", "0" };
    char *e4[] = { "--max-ctx", "99999999999999" };
    char *e5[] = { "--max-ctx", "abc" };
    char *e6[] = { "--max-ctx", "-1" };
    char *e7[] = { "--verbose", "9" };
    char *e8[] = { "--verbose", "-1" };
    char *e9[] = { "--seed", "-1" };
    char *e10[] = { "--chunk", "4097" };
    char *e11[] = { "--kv-dtype", "int8" };
    mm_engine_cfg cfg;

    CHECK(parse(e1, 2, &cfg, NULL) == MM_ERR_RANGE, "unknown flag refused");
    CHECK(parse(e2, 1, &cfg, NULL) == MM_ERR_STATE, "missing value refused");
    CHECK(parse(e3, 2, &cfg, NULL) == MM_ERR_RANGE, "max-ctx 0 refused");
    CHECK(parse(e4, 2, &cfg, NULL) == MM_ERR_RANGE, "overflow refused");
    CHECK(parse(e5, 2, &cfg, NULL) == MM_ERR_RANGE, "non-numeric refused");
    CHECK(parse(e6, 2, &cfg, NULL) == MM_ERR_RANGE, "negative u32 refused");
    CHECK(parse(e7, 2, &cfg, NULL) == MM_ERR_RANGE, "verbose high refused");
    CHECK(parse(e8, 2, &cfg, NULL) == MM_ERR_RANGE, "verbose low refused");
    CHECK(parse(e9, 2, &cfg, NULL) == MM_ERR_RANGE, "negative seed refused");
    CHECK(parse(e10, 2, &cfg, NULL) == MM_ERR_RANGE, "chunk over max refused");
    CHECK(parse(e11, 2, &cfg, NULL) == MM_ERR_RANGE, "bad kv dtype refused");
}

static void test_help(void)
{
    char buf[MM_FLAGS_HELP_SZ];
    size_t n;
    static const char *names[] = {
        "--artifact", "--max-ctx", "--kv-capacity", "--kv-dtype",
        "--concurrency", "--chunk", "--draft-tokens", "--temperature",
        "--top-k", "--top-p", "--seed", "--verbose", "--no-graph",
        "--weights-profile", "--rope-yarn", "--rope-yarn-factor",
        "--rope-yarn-ctx", "--spec", "--lm-head-draft", "--vision",
    };
    int i;
    mm_engine_cfg cfg;
    int help = -1;

    n = mm_flags_help(buf, sizeof buf);
    CHECK(n > 0 && n < sizeof buf, "help fits the buffer");
    CHECK(strstr(buf, "usage:") != NULL, "help has usage");
    for (i = 0; i < (int)(sizeof names / sizeof names[0]); i++)
        CHECK(strstr(buf, names[i]) != NULL, "help lists the flag");
    CHECK(strstr(buf, "quasar") != NULL && strstr(buf, "neroued") != NULL,
          "help lists the profiles");
    CHECK(strstr(buf, "dflash2") != NULL && strstr(buf, "mtp") != NULL,
          "help lists the spec backends");

    /* --help parses, sets the help flag and leaves the defaults. */
    {
        char *h[] = { "--help" };
        CHECK(parse(h, 1, &cfg, &help) == MM_OK, "--help -> OK");
        CHECK(help == 1, "help flag set");
    }
}

int main(void)
{
    test_defaults();
    test_scalars();
    test_yarn_flags();
    test_spec_flags();
    test_profile_flags();
    test_vision_flags();
    test_errors();
    test_help();

    if (g_fail) {
        fprintf(stderr, "FLAGS TEST FAILED\n");
        return 1;
    }
    printf("FLAGS TEST PASSED\n");
    return 0;
}
