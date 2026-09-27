/*
 * Fixture-based tests for the I/O modules: the .mimfer artifact container
 * (src/artifact/artifact.c), the byte-BPE tokenizer (src/tokenizer/
 * tokenizer.c), the telemetry ring (src/telemetry/telemetry.c) -- and the
 * engine's real --artifact path (open + verify + tokenizer load).
 *
 * All fixtures are generated in-process, deterministically:
 *   - the .mimfer files are written with the modules' own writer
 *     (mm_art_write); no checked-in binary blobs,
 *   - the ART_TOK payload is a hand-built 260-entry byte-BPE vocab
 *     ("hello" world: 256 byte tokens + 4 merges + 2 specials),
 *   - the mbuf buffer is a byte-exact stream covering every tag.
 * The corruption cases are made by patching the on-disk fixture bytes and
 * re-opening -- exactly what a bad download would produce.
 *
 * Sections:
 *   A. container     writer -> open -> fields -> verify -> find/data -> dump
 *   B. integrity     section + sub-checksum verification, superblock / TOC
 *                    / payload corruption, geometry errors, version gates,
 *                    unknown-section skip semantics
 *   C. tokenizer     load from the ART_TOK fixture payload (the same call
 *                    the engine makes), encode/decode round-trips, specials,
 *                    truncation, the zero-merge tokenizer, load errors
 *   D. mbuf codec    every tag, nested list/map, malformed inputs
 *   E. telemetry     counters, ring, wrap-around, exact summary/report text
 *   F. engine path   --artifact: create + load + step determinism and the
 *                    hard-fail negatives (missing file, bad superblock
 *                    checksum, missing ART_TOK section)
 *
 * Build (host, no GPU): the Makefile 'artifact-test' target; CMake's
 * 'artifact-test' target runs the same binary.
 */
#include "mimfer/engine.h"

#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int g_fail = 0;
#define CHECK(cond, msg)                                                     \
    do {                                                                     \
        if (!(cond)) {                                                       \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg);    \
            g_fail = 1;                                                      \
        }                                                                    \
    } while (0)

/* Superblock / TOC field offsets (the on-disk format contract from
 * include/mimfer/artifact.h). */
enum {
    SB_MAGIC = 0, SB_MAJOR = 4, SB_MINOR = 6, SB_MINREADER = 12,
    SB_NSEC = 16, SB_FSIZE = 24, SB_CK = 32,
    SB_SZ = 4096,
    TOC_SZ = 128, TOC_TYPE = 32, TOC_FLAGS = 36, TOC_LEN = 48, TOC_CK = 56,
};

/* --------------------------------------------------------------- helpers */

static void le16(uint8_t *p, uint16_t v) { memcpy(p, &v, 2); }
static void le32(uint8_t *p, uint32_t v) { memcpy(p, &v, 4); }
static void le64(uint8_t *p, uint64_t v) { memcpy(p, &v, 8); }

static int patch_file(const char *path, size_t off, const void *bytes, size_t n)
{
    int fd = open(path, O_WRONLY);
    ssize_t w;
    if (fd < 0)
        return -1;
    w = pwrite(fd, bytes, n, (off_t)off);
    close(fd);
    return w == (ssize_t)n ? 0 : -1;
}

static int copy_file(const char *src, const char *dst)
{
    char buf[65536];
    int in, out, rc = 0;
    ssize_t r;

    in = open(src, O_RDONLY);
    if (in < 0)
        return -1;
    out = open(dst, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (out < 0) {
        close(in);
        return -1;
    }
    for (;;) {
        r = read(in, buf, sizeof buf);
        if (r < 0) {
            rc = -1;
            break;
        }
        if (r == 0)
            break;
        if (write(out, buf, (size_t)r) != r) {
            rc = -1;
            break;
        }
    }
    close(in);
    close(out);
    return rc;
}

/* ------------------------------------------------- fixture payloads (A-F) */

/* The tokenizer fixture: a 260-entry byte-BPE vocab.
 *   merges (rank order): (108,108) "ll" -> 256, (104,101) "he" -> 257,
 *       (257,256) "hell" -> 258, (258,111) "hello" -> 259
 *   specials: 300 "<pad>", 301 "<eos>" (the last entry is the eos id)
 * vstr packing: the special strings pack at the FRONT, each NUL-terminated
 * (mm_tok_decode takes their length via strlen), then the 256 byte-token
 * strings, then the merge strings; the last vocab string ("hello") ends at
 * the end of vstr (the decode length contract, tokenizer.h).
 *   [0..6)   "<pad>\0"     (special 300 @ 0)
 *   [6..12)  "<eos>\0"     (special 301 @ 6)
 *   [12..268) byte i @ 12+i
 *   [268) "ll" [270) "he" [272) "hell" [276) "hello" (ends at 281) */
#define FIX_TOK_VOCAB  260
#define FIX_TOK_VSTR   281
static uint8_t g_tok_payload[1400];
static size_t g_tok_len;

static void build_tok_payload(void)
{
    uint8_t *p = g_tok_payload;
    size_t off = 0;
    uint32_t voff[FIX_TOK_VOCAB];
    int i;

    le32(p + off, FIX_TOK_VOCAB); off += 4;            /* vocab */
    le32(p + off, 4); off += 4;                        /* n_merges */
    le32(p + off, 2); off += 4;                        /* n_special */
    le32(p + off, 108); le32(p + off + 4, 108); off += 8;  /* (l, l) */
    le32(p + off, 104); le32(p + off + 4, 101); off += 8;  /* (h, e) */
    le32(p + off, 257); le32(p + off + 4, 256); off += 8;  /* (he, ll) */
    le32(p + off, 258); le32(p + off + 4, 111); off += 8;  /* (hell, o) */
    le32(p + off, 300); le32(p + off + 4, 0);   off += 8;  /* <pad> @ 0 */
    le32(p + off, 301); le32(p + off + 4, 6);   off += 8;  /* <eos> @ 6 */
    for (i = 0; i < 256; i++)
        voff[i] = (uint32_t)(12 + i);      /* byte tokens @ 12..267 */
    voff[256] = 268;   /* "ll"    */
    voff[257] = 270;   /* "he"    */
    voff[258] = 272;   /* "hell"  */
    voff[259] = 276;   /* "hello" (ends at 281) */
    for (i = 0; i < FIX_TOK_VOCAB; i++) {
        le32(p + off, voff[i]);
        off += 4;
    }
    memcpy(p + off, "<pad>", 5);
    p[off + 5] = 0;
    memcpy(p + off + 6, "<eos>", 5);
    p[off + 11] = 0;
    for (i = 0; i < 256; i++)
        *(p + off + 12 + i) = (uint8_t)i;
    memcpy(p + off + 268, "ll", 2);
    memcpy(p + off + 270, "he", 2);
    memcpy(p + off + 272, "hell", 4);
    memcpy(p + off + 276, "hello", 5);
    g_tok_len = off + FIX_TOK_VSTR;
}

/* A minimal payload with ZERO merges (pure byte tokenizer) + one special:
 * guards the mcap == 0 path of the merge hash (tok_rank). The special
 * string is NUL-terminated (decode strlen contract).
 *   [0..4) "<e>\0" (special 999 @ 0), [4..260) byte i @ 4+i */
static uint8_t g_min_payload[2048];
static size_t g_min_len;

static void build_min_payload(void)
{
    uint8_t *p = g_min_payload;
    size_t off = 0;
    int i;

    le32(p + off, 256); off += 4;   /* vocab */
    le32(p + off, 0); off += 4;     /* n_merges */
    le32(p + off, 1); off += 4;     /* n_special */
    le32(p + off, 999); le32(p + off + 4, 0); off += 8;   /* 999 -> "<e>" @ 0 */
    for (i = 0; i < 256; i++) {
        le32(p + off, (uint32_t)(4 + i));
        off += 4;
    }
    memcpy(p + off, "<e>", 3);
    p[off + 3] = 0;
    for (i = 0; i < 256; i++)
        *(p + off + 4 + i) = (uint8_t)i;
    g_min_len = off + 260;
}
/* The ART_WEIGHTS fixture: 2 x 4 KiB sub-checksummed 8 KiB payload
 * (deterministic, non-constant data so value-dependent faults would show). */
#define FIX_WT_DATA    8192
#define FIX_WT_SUBSTEP 4096
static uint8_t g_weights[16 + FIX_WT_DATA];

static void build_weights(void)
{
    int i;
    for (i = 0; i < FIX_WT_DATA; i++)
        g_weights[16 + i] = (uint8_t)(i * 7 + 3);
    le64(g_weights + 0, mm_fnv1a64(g_weights + 16, FIX_WT_SUBSTEP));
    le64(g_weights + 8, mm_fnv1a64(g_weights + 16 + FIX_WT_SUBSTEP,
                                   FIX_WT_DATA - FIX_WT_SUBSTEP));
}

static uint8_t g_model_blob[64];

static void build_model_blob(void)
{
    int i;
    for (i = 0; i < (int)sizeof g_model_blob; i++)
        g_model_blob[i] = (uint8_t)(i * 3);
}

/* The fixture artifact. Sections (TOC order, with_tok layout):
 *   model (ART_MODEL, required), tok (ART_TOK, required, if with_tok),
 *   chat (ART_CHAT, optional), gen (ART_GENCFG, optional),
 *   weights (ART_WEIGHTS, required, 2 sub-checksums),
 *   future (unknown type 0x3F, skippable -- forward-compat skip). */
#define FIX_FUTURE_TYPE 0x3Fu

static mm_status write_fixture(const char *path, int with_tok)
{
    mm_arts secs[6];
    const uint8_t *payloads[6];
    static const char *chat = "{system}\n{history}\n{user}";
    static const char *gen = "temperature 0.6\ntop_p 0.9\n";
    uint32_t cap = (1u << ART_MODEL) | (1u << ART_TOK) | (1u << ART_CHAT) |
                   (1u << ART_GENCFG) | (1u << ART_WEIGHTS);
    uint32_t n = 0;

    memset(secs, 0, sizeof secs);
    strcpy(secs[n].name, "model");
    secs[n].type = ART_MODEL;
    secs[n].flags = MM_ART_FLAG_REQUIRED;
    secs[n].length = sizeof g_model_blob;
    payloads[n] = g_model_blob;
    n++;
    if (with_tok) {
        strcpy(secs[n].name, "tok");
        secs[n].type = ART_TOK;
        secs[n].flags = MM_ART_FLAG_REQUIRED;
        secs[n].length = (uint64_t)g_tok_len;
        payloads[n] = g_tok_payload;
        n++;
    }
    strcpy(secs[n].name, "chat");
    secs[n].type = ART_CHAT;
    secs[n].flags = MM_ART_FLAG_OPTIONAL;
    secs[n].length = (uint64_t)strlen(chat);
    payloads[n] = (const uint8_t *)chat;
    n++;
    strcpy(secs[n].name, "gen");
    secs[n].type = ART_GENCFG;
    secs[n].flags = MM_ART_FLAG_OPTIONAL;
    secs[n].length = (uint64_t)strlen(gen);
    payloads[n] = (const uint8_t *)gen;
    n++;
    strcpy(secs[n].name, "weights");
    secs[n].type = ART_WEIGHTS;
    secs[n].flags = MM_ART_FLAG_REQUIRED;
    secs[n].length = 16 + FIX_WT_DATA;
    secs[n].sub_step = FIX_WT_SUBSTEP;
    secs[n].n_subck = 2;
    payloads[n] = g_weights;
    n++;
    strcpy(secs[n].name, "future");
    secs[n].type = FIX_FUTURE_TYPE;
    secs[n].flags = MM_ART_FLAG_SKIPPABLE;
    secs[n].length = 16;
    payloads[n] = g_model_blob;
    n++;

    return mm_art_write(path, 2, 0, cap, "qwen3.8-27b", "nvfp4",
                        "artifact-test/fixture-1", secs, n, payloads);
}

/* ------------------------------------------------------ A: container ---- */

static void section_container(const char *path)
{
    mm_artifact *a = NULL;
    const mm_arts *s;
    const uint8_t *data;
    size_t len;
    char buf[16384];
    int n;

    printf("== artifact: container (fixture open + verify + dump)\n");
    CHECK(write_fixture(path, 1) == MM_OK, "fixture write (6 sections)");
    CHECK(mm_art_open(path, &a) == MM_OK, "open fixture");
    if (!a)
        return;
    CHECK(a->major == 2 && a->minor == 0, "format version 2.0");
    CHECK(strcmp(a->model_id, "qwen3.8-27b") == 0, "model_id round-trips");
    CHECK(strcmp(a->weights_id, "nvfp4") == 0, "weights_id round-trips");
    CHECK(strcmp(a->recipe, "artifact-test/fixture-1") == 0, "recipe");
    CHECK(a->n_toc == 6, "TOC section count");
    CHECK(mm_art_verify(a) == MM_OK, "verify: superblock + all sections");
    CHECK(mm_art_find(a, ART_MTP) == NULL, "find: absent type is NULL");

    s = mm_art_find(a, ART_MODEL);
    CHECK(s && strcmp(s->name, "model") == 0 &&
          s->flags == MM_ART_FLAG_REQUIRED, "find: model section");
    data = mm_art_data(a, s, &len);
    CHECK(len == sizeof g_model_blob &&
          memcmp(data, g_model_blob, len) == 0,
          "data: model payload byte-identical");

    s = mm_art_find(a, ART_TOK);
    CHECK(s && strcmp(s->name, "tok") == 0 &&
          (size_t)s->length == g_tok_len, "find: tok section");
    data = mm_art_data(a, s, &len);
    CHECK(memcmp(data, g_tok_payload, g_tok_len) == 0, "data: tok payload");

    s = mm_art_find(a, ART_WEIGHTS);
    CHECK(s && s->sub_step == FIX_WT_SUBSTEP && s->n_subck == 2,
          "find: weights sub-checksum geometry");
    CHECK(mm_art_verify_section(a, s) == MM_OK,
          "verify_section: weights (whole ck + 2 sub-ck)");

    s = mm_art_find(a, FIX_FUTURE_TYPE);
    CHECK(s && s->flags == MM_ART_FLAG_SKIPPABLE,
          "find: unknown skippable section visible to readers");

    n = mm_art_dump(a, buf, sizeof buf);
    CHECK(n > 0 && strstr(buf, "qwen3.8-27b") && strstr(buf, "nvfp4") &&
          strstr(buf, "model") && strstr(buf, "weights") &&
          strstr(buf, "+subck") && strstr(buf, "skippable"),
          "dump: manifest carries ids + full section table");
    mm_art_close(a);
}

/* ------------------------------------------- B: integrity (corruption) --- */

static void section_integrity_1(const char *path)
{
    char p[512];
    uint8_t v[2];
    uint64_t u64;
    mm_artifact *a;
    const mm_arts *s;

    printf("== artifact: integrity (payload + checksum corruption)\n");

    {
        /* Corrupt one byte of the chat payload: open() stays layout-only,
         * both verify_section and full verify must reject. */
        snprintf(p, sizeof p, "%s.b_chat", path);
        CHECK(copy_file(path, p) == 0, "copy: bad chat payload");
        CHECK(mm_art_open(p, &a) == MM_OK, "open: layout-only succeeds");
        s = mm_art_find(a, ART_CHAT);
        CHECK(patch_file(p, (size_t)s->offset + 1, "\x01", 1) == 0, "patch byte");
        CHECK(mm_art_verify_section(a, s) == MM_ERR_CORRUPT,
              "verify_section: corrupt payload rejected");
        CHECK(mm_art_verify(a) == MM_ERR_CORRUPT,
              "verify: corrupt payload rejected");
        mm_art_close(a);
    }
    {
        /* Weights: the on-disk whole ck is re-computed to MATCH the
         * corrupted payload, so the sub-checksum path is what must catch
         * the flip (whole ck alone would pass). TOC index of weights in
         * the with_tok layout is 4. */
        snprintf(p, sizeof p, "%s.b_subck", path);
        copy_file(path, p);
        {
            uint8_t *wp;
            size_t wlen;
            size_t woff;
            CHECK(mm_art_open(p, &a) == MM_OK, "open: subck fixture");
            s = mm_art_find(a, ART_WEIGHTS);
            wlen = s->length;
            woff = s->offset;
            wp = malloc(wlen);
            memcpy(wp, a->map + woff, wlen);
            mm_art_close(a);
            wp[16 + FIX_WT_SUBSTEP + 100] ^= 0xFF;   /* inside sub-chunk 1 */
            u64 = mm_fnv1a64(wp, wlen);
            CHECK(patch_file(p, woff, wp, wlen) == 0, "patch: payload");
            CHECK(patch_file(p, (size_t)SB_SZ + 4 * TOC_SZ + TOC_CK,
                             &u64, 8) == 0,
                  "patch: TOC whole ck re-computed to match");
            free(wp);
        }
        CHECK(mm_art_open(p, &a) == MM_OK, "open: subck fixture re-open");
        s = mm_art_find(a, ART_WEIGHTS);
        CHECK(mm_art_verify_section(a, s) == MM_ERR_CORRUPT,
              "verify_section: sub-checksum mismatch rejected "
              "(whole ck matched)");
        mm_art_close(a);
    }
    {
        /* TOC entry checksum (payload untouched). verify_section() compares
         * the payload FNV to the TOC ck cached in the in-memory struct at open
         * time, so the on-disk TOC ck must be corrupted BEFORE opening; the
         * handle then loads the bad ck and rejects the clean payload. TOC
         * index of chat in the with_tok layout is 2. */
        snprintf(p, sizeof p, "%s.b_tocck", path);
        CHECK(copy_file(path, p) == 0, "copy: tocck fixture");
        {
            mm_artifact *c = NULL;
            const mm_arts *cs;
            CHECK(mm_art_open(path, &c) == MM_OK, "open: clean (read ck)");
            cs = mm_art_find(c, ART_CHAT);
            u64 = cs->ck ^ 0xDEADBEEFCAFEBABEu;
            mm_art_close(c);
        }
        CHECK(patch_file(p, (size_t)SB_SZ + 2 * TOC_SZ + TOC_CK,
                         &u64, 8) == 0, "patch: TOC ck (pre-open)");
        CHECK(mm_art_open(p, &a) == MM_OK, "open: tocck (layout-only)");
        s = mm_art_find(a, ART_CHAT);
        CHECK(mm_art_verify_section(a, s) == MM_ERR_CORRUPT,
              "verify_section: TOC ck mismatch rejected");
        mm_art_close(a);
    }
    {
        /* Superblock checksum: open() succeeds (layout-only), verify fails. */
        snprintf(p, sizeof p, "%s.b_sbck", path);
        copy_file(path, p);
        u64 = 0;
        CHECK(patch_file(p, (size_t)SB_CK, &u64, 8) == 0, "patch: sb ck");
        CHECK(mm_art_open(p, &a) == MM_OK, "open: sb ck zeroed still opens");
        CHECK(mm_art_verify(a) == MM_ERR_CORRUPT,
              "verify: zeroed superblock ck rejected");
        mm_art_close(a);
    }
    {
        /* Bad magic: open() itself must fail. */
        snprintf(p, sizeof p, "%s.b_magic", path);
        copy_file(path, p);
        CHECK(patch_file(p, (size_t)SB_MAGIC, "XXXX", 4) == 0, "patch: magic");
        CHECK(mm_art_open(p, &a) == MM_ERR_CORRUPT, "open: bad magic");
    }
    {
        /* Major version mismatch: the reader refuses (2.x != 3.x). */
        snprintf(p, sizeof p, "%s.b_major", path);
        copy_file(path, p);
        v[0] = 3; v[1] = 0;                       /* major = 3 */
        CHECK(patch_file(p, (size_t)SB_MAJOR, v, 2) == 0, "patch: major");
        CHECK(mm_art_open(p, &a) == MM_ERR_CORRUPT, "open: major mismatch");
    }
}
static void section_integrity_2(const char *path)
{
    char p[512];
    uint32_t u32;
    uint64_t u64;
    uint8_t small[100];
    mm_artifact *a;
    const mm_arts *s;
    int fd;

    printf("== artifact: integrity (version gates + geometry)\n");

    {
        /* min_reader ahead of this reader (3.0 > 2.0): refuse. */
        snprintf(p, sizeof p, "%s.b_minreader", path);
        copy_file(path, p);
        u32 = 3u << 16;
        CHECK(patch_file(p, (size_t)SB_MINREADER, &u32, 4) == 0,
              "patch: min_reader");
        CHECK(mm_art_open(p, &a) == MM_ERR_CORRUPT,
              "open: min_reader beyond reader");
    }
    {
        /* Stored file_size disagrees with the actual size. */
        snprintf(p, sizeof p, "%s.b_fsize", path);
        copy_file(path, p);
        u64 = 1;
        CHECK(patch_file(p, (size_t)SB_FSIZE, &u64, 8) == 0, "patch: fsize");
        CHECK(mm_art_open(p, &a) == MM_ERR_CORRUPT, "open: fsize mismatch");
    }
    {
        /* TOC geometry: nsec runs far past EOF. */
        snprintf(p, sizeof p, "%s.b_nsec", path);
        copy_file(path, p);
        u32 = 0x100000;
        CHECK(patch_file(p, (size_t)SB_NSEC, &u32, 4) == 0, "patch: nsec");
        CHECK(mm_art_open(p, &a) == MM_ERR_CORRUPT, "open: nsec past EOF");
    }
    {
        /* A section whose stored length runs past EOF (model section,
         * TOC index 0; patch its TOC length field). */
        snprintf(p, sizeof p, "%s.b_pasteof", path);
        copy_file(path, p);
        CHECK(mm_art_open(p, &a) == MM_OK, "open: pasteof (clean read)");
        s = mm_art_find(a, ART_MODEL);
        u64 = (uint64_t)(a->map_len - s->offset) + 100;
        mm_art_close(a);
        CHECK(patch_file(p, (size_t)SB_SZ + TOC_LEN, &u64, 8) == 0,
              "patch: section length");
        CHECK(mm_art_open(p, &a) == MM_ERR_CORRUPT, "open: section past EOF");
    }
    {
        /* An UNKNOWN section that is not skippable must be refused
         * (future format must not smuggle in data this reader would
         * skip). The 'future' section is TOC index 5 in with_tok layout. */
        snprintf(p, sizeof p, "%s.b_future_ns", path);
        copy_file(path, p);
        u32 = MM_ART_FLAG_REQUIRED;
        CHECK(patch_file(p, (size_t)SB_SZ + 5 * TOC_SZ + TOC_FLAGS,
                         &u32, 4) == 0, "patch: unknown flags");
        CHECK(mm_art_open(p, &a) == MM_ERR_CORRUPT,
              "open: unknown non-skippable section refused");
    }
    {
        /* The skippable unknown section: the whole-file verify() must
         * skip it (not checksum it) -- open + verify on the clean file. */
        CHECK(mm_art_open(path, &a) == MM_OK, "open: clean fixture");
        s = mm_art_find(a, FIX_FUTURE_TYPE);
        CHECK(s && mm_art_verify(a) == MM_OK,
              "verify: unknown skippable section skipped, not checksummed");
        mm_art_close(a);
    }
    {
        /* Truncated header: smaller than superblock + one TOC entry. */
        snprintf(p, sizeof p, "%s.b_small", path);
        memset(small, 0xAB, sizeof small);
        fd = open(p, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        CHECK(fd >= 0, "create: tiny file");
        CHECK(write(fd, small, sizeof small) == (ssize_t)sizeof small,
              "write: tiny file");
        close(fd);
        CHECK(mm_art_open(p, &a) == MM_ERR_CORRUPT, "open: tiny file");
    }
    {
        /* Missing path: I/O error, not corruption. */
        snprintf(p, sizeof p, "%s.nosuch", path);
        CHECK(mm_art_open(p, &a) == MM_ERR_IO, "open: missing file -> IO");
    }
}
/* ------------------------------------------------------ C: tokenizer --- */

static void section_tokenizer(void)
{
    mm_tok *t = NULL;
    mm_tok *tm = NULL;
    uint32_t out[64];
    uint32_t n_out;
    int trunc;
    char buf[128];
    int d;

    printf("== artifact: tokenizer (load / encode / decode / trunc)\n");
    CHECK(mm_tok_load(g_tok_payload, g_tok_len, &t) == MM_OK,
          "load: fixture tokenizer");
    if (!t)
        return;
    CHECK(t->vocab == 260 && t->n_merges == 4 && t->n_special == 2,
          "load: geometry (vocab/merges/special)");
    CHECK(t->eos == 301 && mm_tok_eos(t) == 301, "load: eos id");
    CHECK(t->vstr_len == FIX_TOK_VSTR, "load: vstr_len");

    /* Encode: greedy lowest-rank merges. */
    CHECK(mm_tok_encode(t, (const uint8_t *)"hello", 5, out, 64, &n_out,
                        &trunc) == MM_OK,
          "enc: hello");
    CHECK(n_out == 1 && !trunc && out[0] == 259, "enc: hello -> 259");
    CHECK(mm_tok_encode(t, (const uint8_t *)"hell", 4, out, 64, &n_out,
                        &trunc) == MM_OK && n_out == 1 && out[0] == 258,
          "enc: hell -> 258");
    CHECK(mm_tok_encode(t, (const uint8_t *)"he", 2, out, 64, &n_out,
                        &trunc) == MM_OK && n_out == 1 && out[0] == 257,
          "enc: he -> 257");
    CHECK(mm_tok_encode(t, (const uint8_t *)"ll", 2, out, 64, &n_out,
                        &trunc) == MM_OK && n_out == 1 && out[0] == 256,
          "enc: ll -> 256");
    CHECK(mm_tok_encode(t, (const uint8_t *)"h", 1, out, 64, &n_out,
                        &trunc) == MM_OK && n_out == 1 && out[0] == 104,
          "enc: h -> 104 (no merge)");
    CHECK(mm_tok_encode(t, (const uint8_t *)"hello hello", 11, out, 64,
                        &n_out, &trunc) == MM_OK && n_out == 3 && !trunc &&
          out[0] == 259 && out[1] == 32 && out[2] == 259,
          "enc: 'hello hello' -> 259 32 259");
    CHECK(mm_tok_encode(t, (const uint8_t *)"hello hello", 11, out, 2,
                        &n_out, &trunc) == MM_OK && n_out == 2 && trunc &&
          out[0] == 259 && out[1] == 32,
          "enc: cap=2 truncates (n_out=2, trunc=1)");
    CHECK(mm_tok_encode(t, (const uint8_t *)"x", 0, out, 64, &n_out,
                        &trunc) == MM_OK && n_out == 0 && !trunc,
          "enc: empty input -> n_out=0");

    /* Decode: exact string concatenation. */
    {
        uint32_t v[8];
        v[0] = 259;
        d = mm_tok_decode(t, v, 1, buf, sizeof buf);
        CHECK(d == 5 && strcmp(buf, "hello") == 0, "dec: 259 -> 'hello'");
        v[0] = 259; v[1] = 32; v[2] = 259;
        d = mm_tok_decode(t, v, 3, buf, sizeof buf);
        CHECK(d == 11 && strcmp(buf, "hello hello") == 0,
              "dec: 259 32 259 -> 'hello hello'");
        v[0] = 104; v[1] = 101; v[2] = 108; v[3] = 108; v[4] = 111;
        d = mm_tok_decode(t, v, 5, buf, sizeof buf);
        CHECK(d == 5 && strcmp(buf, "hello") == 0,
              "dec: byte tokens -> 'hello'");
        v[0] = 301;
        d = mm_tok_decode(t, v, 1, buf, sizeof buf);
        CHECK(d == 5 && strcmp(buf, "<eos>") == 0, "dec: special 301");
        v[0] = 300;
        d = mm_tok_decode(t, v, 1, buf, sizeof buf);
        CHECK(d == 5 && strcmp(buf, "<pad>") == 0, "dec: special 300");
        v[0] = 259; v[1] = 301;
        d = mm_tok_decode(t, v, 2, buf, sizeof buf);
        CHECK(d == 10 && strcmp(buf, "hello<eos>") == 0,
              "dec: vocab+special mix");
        CHECK(mm_tok_decode(t, v, 2, buf, 8) == -1,
              "dec: cap too small -> -1");
    }

    /* Round-trip: encode then decode reproduces the input bytes. */
    {
        const char *in = "hello hello world";
        uint32_t rt[64];
        uint32_t rn;
        int rt_trunc;
        char rtbuf[128];
        int rtd;

        CHECK(mm_tok_encode(t, (const uint8_t *)in, strlen(in), rt, 64, &rn,
                            &rt_trunc) == MM_OK && !rt_trunc &&
              rn == 9 && rt[0] == 259 && rt[1] == 32 && rt[2] == 259 &&
              rt[3] == 32 && rt[4] == 119 && rt[5] == 111 && rt[6] == 114 &&
              rt[7] == 108 && rt[8] == 100,
              "roundtrip: encode 'hello hello world'");
        rtd = mm_tok_decode(t, rt, rn, rtbuf, sizeof rtbuf);
        CHECK(rtd == (int)strlen(in) && strcmp(rtbuf, in) == 0,
              "roundtrip: decode restores input");
    }

    /* Token-string accessor (specials are NUL-terminated by layout). */
    CHECK(strcmp(mm_tok_str(t, 301), "<eos>") == 0, "str: eos");
    CHECK(strcmp(mm_tok_str(t, 300), "<pad>") == 0, "str: pad");
    CHECK(strcmp(mm_tok_str(t, 999999), "") == 0, "str: unknown id -> \"\"");

    mm_tok_free(t);

    /* Minimal payload: zero merges (mcap == 0 path of the merge hash). */
    CHECK(mm_tok_load(g_min_payload, g_min_len, &tm) == MM_OK,
          "load: minimal (no merges)");
    if (tm) {
        CHECK(tm->vocab == 256 && tm->n_merges == 0 && tm->n_special == 1 &&
              tm->eos == 999 && tm->mcap == 0,
              "load: minimal geometry (mcap=0)");
        CHECK(mm_tok_encode(tm, (const uint8_t *)"ab", 2, out, 64, &n_out,
                            &trunc) == MM_OK && n_out == 2 && !trunc &&
              out[0] == 97 && out[1] == 98,
              "enc: 'ab' -> 97 98 (byte pass-through)");
        d = mm_tok_decode(tm, out, 2, buf, sizeof buf);
        CHECK(d == 2 && strcmp(buf, "ab") == 0, "dec: 97 98 -> 'ab'");
        d = mm_tok_decode(tm, (const uint32_t[]){999}, 1, buf, sizeof buf);
        CHECK(d == 3 && strcmp(buf, "<e>") == 0, "dec: special 999 -> '<e>'");
        mm_tok_unload(tm);
        free(tm);
    }
}
/* ------------------------------------------------------- D: mbuf codec -- */

/* A byte-exact mbuf stream: map(10) covering every value tag, a nested
 * list, and a trailing END. Layout:
 *   map(10):
 *     ("n8",u8 7) ("n16",u16 300) ("n32",u32 42)
 *     ("n64",u64 0x1122334455667788) ("i64",i64 -12345) ("f",f32 1.5)
 *     ("s",str "hi") ("blob",blob <1,2,3>)
 *     ("list",list(2): u32 1, u32 2) ("ref",ref 99)
 *   end */
static size_t mbuf_build(uint8_t *b)
{
    size_t o = 0;
    float f;

    b[o++] = (uint8_t)MBUF_MAP; le32(b + o, 10); o += 4;
    b[o++] = (uint8_t)MBUF_STR; le16(b + o, 2); o += 2; b[o++] = 'n'; b[o++] = '8';
    b[o++] = (uint8_t)MBUF_U8; b[o++] = 7;
    b[o++] = (uint8_t)MBUF_STR; le16(b + o, 3); o += 2;
    b[o++] = 'n'; b[o++] = '1'; b[o++] = '6';
    b[o++] = (uint8_t)MBUF_U16; le16(b + o, 300); o += 2;
    b[o++] = (uint8_t)MBUF_STR; le16(b + o, 3); o += 2;
    b[o++] = 'n'; b[o++] = '3'; b[o++] = '2';
    b[o++] = (uint8_t)MBUF_U32; le32(b + o, 42); o += 4;
    b[o++] = (uint8_t)MBUF_STR; le16(b + o, 3); o += 2;
    b[o++] = 'n'; b[o++] = '6'; b[o++] = '4';
    b[o++] = (uint8_t)MBUF_U64; le64(b + o, 0x1122334455667788ull); o += 8;
    b[o++] = (uint8_t)MBUF_STR; le16(b + o, 3); o += 2;
    b[o++] = 'i'; b[o++] = '6'; b[o++] = '4';
    b[o++] = (uint8_t)MBUF_I64; le64(b + o, (uint64_t)(int64_t)-12345); o += 8;
    b[o++] = (uint8_t)MBUF_STR; le16(b + o, 1); o += 2; b[o++] = 'f';
    b[o++] = (uint8_t)MBUF_F32; f = 1.5f; memcpy(b + o, &f, 4); o += 4;
    b[o++] = (uint8_t)MBUF_STR; le16(b + o, 1); o += 2; b[o++] = 's';
    b[o++] = (uint8_t)MBUF_STR; le16(b + o, 2); o += 2; b[o++] = 'h'; b[o++] = 'i';
    b[o++] = (uint8_t)MBUF_STR; le16(b + o, 4); o += 2;
    b[o++] = 'b'; b[o++] = 'l'; b[o++] = 'o'; b[o++] = 'b';
    b[o++] = (uint8_t)MBUF_BLOB; le32(b + o, 3); o += 4;
    b[o++] = 1; b[o++] = 2; b[o++] = 3;
    b[o++] = (uint8_t)MBUF_STR; le16(b + o, 4); o += 2;
    b[o++] = 'l'; b[o++] = 'i'; b[o++] = 's'; b[o++] = 't';
    b[o++] = (uint8_t)MBUF_LIST; le32(b + o, 2); o += 4;
    b[o++] = (uint8_t)MBUF_U32; le32(b + o, 1); o += 4;
    b[o++] = (uint8_t)MBUF_U32; le32(b + o, 2); o += 4;
    b[o++] = (uint8_t)MBUF_STR; le16(b + o, 3); o += 2;
    b[o++] = 'r'; b[o++] = 'e'; b[o++] = 'f';
    b[o++] = (uint8_t)MBUF_REF; le32(b + o, 99); o += 4;
    b[o++] = (uint8_t)MBUF_END;
    return o;
}
static void section_mbuf(void)
{
    uint8_t b[512];
    size_t n;
    const uint8_t *p, *end, *sp;
    mm_mbuf_tag tag;
    uint64_t iv;
    float fv;
    uint32_t sl;

    printf("== artifact: mbuf codec (every tag + malformed inputs)\n");
    n = mbuf_build(b);
    p = b;
    end = b + n;

    CHECK(mm_mbuf_read(&p, end, &tag, &iv, &fv, &sp, &sl) == MM_OK &&
          tag == MBUF_MAP && iv == 10, "mbuf: map(10) header");

    CHECK(mm_mbuf_read(&p, end, &tag, &iv, &fv, &sp, &sl) == MM_OK &&
          tag == MBUF_STR && sl == 2 && sp[0] == 'n' && sp[1] == '8',
          "mbuf: key 'n8'");
    CHECK(mm_mbuf_read(&p, end, &tag, &iv, &fv, &sp, &sl) == MM_OK &&
          tag == MBUF_U8 && iv == 7, "mbuf: u8 7");

    CHECK(mm_mbuf_read(&p, end, &tag, &iv, &fv, &sp, &sl) == MM_OK &&
          tag == MBUF_STR && sl == 3 && sp[1] == '1' && sp[2] == '6',
          "mbuf: key 'n16'");
    CHECK(mm_mbuf_read(&p, end, &tag, &iv, &fv, &sp, &sl) == MM_OK &&
          tag == MBUF_U16 && iv == 300, "mbuf: u16 300");

    CHECK(mm_mbuf_read(&p, end, &tag, &iv, &fv, &sp, &sl) == MM_OK &&
          tag == MBUF_STR && sl == 3 && sp[1] == '3' && sp[2] == '2',
          "mbuf: key 'n32'");
    CHECK(mm_mbuf_read(&p, end, &tag, &iv, &fv, &sp, &sl) == MM_OK &&
          tag == MBUF_U32 && iv == 42, "mbuf: u32 42");

    CHECK(mm_mbuf_read(&p, end, &tag, &iv, &fv, &sp, &sl) == MM_OK &&
          tag == MBUF_STR && sl == 3 && sp[1] == '6' && sp[2] == '4',
          "mbuf: key 'n64'");
    CHECK(mm_mbuf_read(&p, end, &tag, &iv, &fv, &sp, &sl) == MM_OK &&
          tag == MBUF_U64 && iv == 0x1122334455667788ull, "mbuf: u64 wide");

    CHECK(mm_mbuf_read(&p, end, &tag, &iv, &fv, &sp, &sl) == MM_OK &&
          tag == MBUF_STR && sl == 3 && sp[0] == 'i', "mbuf: key 'i64'");
    CHECK(mm_mbuf_read(&p, end, &tag, &iv, &fv, &sp, &sl) == MM_OK &&
          tag == MBUF_I64 && (int64_t)iv == -12345, "mbuf: i64 -12345");

    CHECK(mm_mbuf_read(&p, end, &tag, &iv, &fv, &sp, &sl) == MM_OK &&
          tag == MBUF_STR && sl == 1 && sp[0] == 'f', "mbuf: key 'f'");
    CHECK(mm_mbuf_read(&p, end, &tag, &iv, &fv, &sp, &sl) == MM_OK &&
          tag == MBUF_F32 && fv == 1.5f, "mbuf: f32 1.5");

    CHECK(mm_mbuf_read(&p, end, &tag, &iv, &fv, &sp, &sl) == MM_OK &&
          tag == MBUF_STR && sl == 1 && sp[0] == 's', "mbuf: key 's'");
    CHECK(mm_mbuf_read(&p, end, &tag, &iv, &fv, &sp, &sl) == MM_OK &&
          tag == MBUF_STR && sl == 2 && sp[0] == 'h' && sp[1] == 'i',
          "mbuf: str value 'hi'");

    CHECK(mm_mbuf_read(&p, end, &tag, &iv, &fv, &sp, &sl) == MM_OK &&
          tag == MBUF_STR && sl == 4 && sp[0] == 'b' && sp[3] == 'b',
          "mbuf: key 'blob'");
    CHECK(mm_mbuf_read(&p, end, &tag, &iv, &fv, &sp, &sl) == MM_OK &&
          tag == MBUF_BLOB && sl == 3 && sp[0] == 1 && sp[1] == 2 &&
          sp[2] == 3, "mbuf: blob <1,2,3>");

    CHECK(mm_mbuf_read(&p, end, &tag, &iv, &fv, &sp, &sl) == MM_OK &&
          tag == MBUF_STR && sl == 4 && sp[0] == 'l' && sp[3] == 't',
          "mbuf: key 'list'");
    CHECK(mm_mbuf_read(&p, end, &tag, &iv, &fv, &sp, &sl) == MM_OK &&
          tag == MBUF_LIST && iv == 2, "mbuf: list(2) header");
    CHECK(mm_mbuf_read(&p, end, &tag, &iv, &fv, &sp, &sl) == MM_OK &&
          tag == MBUF_U32 && iv == 1, "mbuf: list[0] = 1");
    CHECK(mm_mbuf_read(&p, end, &tag, &iv, &fv, &sp, &sl) == MM_OK &&
          tag == MBUF_U32 && iv == 2, "mbuf: list[1] = 2");

    CHECK(mm_mbuf_read(&p, end, &tag, &iv, &fv, &sp, &sl) == MM_OK &&
          tag == MBUF_STR && sl == 3 && sp[0] == 'r' && sp[2] == 'f',
          "mbuf: key 'ref'");
    CHECK(mm_mbuf_read(&p, end, &tag, &iv, &fv, &sp, &sl) == MM_OK &&
          tag == MBUF_REF && iv == 99, "mbuf: ref 99");

    CHECK(mm_mbuf_read(&p, end, &tag, &iv, &fv, &sp, &sl) == MM_OK &&
          tag == MBUF_END, "mbuf: trailing END");
    CHECK(p == end, "mbuf: stream consumed exactly");

    /* Malformed inputs: each must be MM_ERR_CORRUPT, never a crash. */
    {
        const uint8_t bad1[1] = { 0xFF };
        const uint8_t *q = bad1;
        CHECK(mm_mbuf_read(&q, bad1 + 1, &tag, &iv, &fv, &sp, &sl) ==
              MM_ERR_CORRUPT, "mbuf: bad tag -> CORRUPT");
    }
    {
        const uint8_t t3[3] = { MBUF_U32, 1, 2 };
        const uint8_t *q = t3;
        CHECK(mm_mbuf_read(&q, t3 + 3, &tag, &iv, &fv, &sp, &sl) ==
              MM_ERR_CORRUPT, "mbuf: truncated u32 -> CORRUPT");
    }
    {
        /* str len 256 with only one payload byte present. */
        const uint8_t ts[4] = { MBUF_STR, 0x00, 0x01, 'x' };
        const uint8_t *q = ts;
        CHECK(mm_mbuf_read(&q, ts + 4, &tag, &iv, &fv, &sp, &sl) ==
              MM_ERR_CORRUPT, "mbuf: str len past end -> CORRUPT");
    }
    {
        const uint8_t *q = b;
        CHECK(mm_mbuf_read(&q, q, &tag, &iv, &fv, &sp, &sl) ==
              MM_ERR_CORRUPT, "mbuf: empty buffer -> CORRUPT");
    }
    {
        const uint8_t e1[1] = { MBUF_END };
        const uint8_t *q = e1;
        CHECK(mm_mbuf_read(&q, e1 + 1, &tag, &iv, &fv, &sp, &sl) == MM_OK &&
              tag == MBUF_END, "mbuf: bare END");
    }
}
/* ----------------------------------------------------- E: telemetry ---- */

static void section_telemetry(void)
{
    /* ~160 KiB each (the 4096-sample ring): keep them off the stack. */
    static mm_tel t, w;
    char buf[4096];
    int n;
    uint32_t i;

    printf("== artifact: telemetry (counters / ring / report)\n");
    mm_tel_init(&t);
    CHECK(t.rounds == 0 && t.head == 0 && t.n_tokens == 0, "tel: init zeroed");

    n = mm_tel_summary(&t, buf, sizeof buf);
    CHECK(n > 0 && strcmp(buf, "no rounds") == 0, "tel: summary empty");
    n = mm_tel_report(&t, buf, sizeof buf);
    CHECK(n > 0 && strstr(buf, "rounds 0") != NULL, "tel: report empty");

    mm_tel_round(&t, TEL_PREFILL, 8, 1, 0, 100, 150);
    mm_tel_round(&t, TEL_DECODE, 1, 1, 1, 150, 165);
    mm_tel_round(&t, TEL_DECODE, 1, 1, 0, 165, 170);
    mm_tel_round(&t, TEL_DECODE, 1, 2, 2, 170, 210);
    mm_tel_round(&t, TEL_DECODE, 1, 2, 0, 210, 215);

    CHECK(t.rounds == 5 && t.n_tokens == 12 && t.n_prefill == 1 &&
          t.n_decode == 4 && t.cache_hits == 3, "tel: counters");
    CHECK(t.t_first_ms == 100 && t.t_end_ms == 215, "tel: t_first/t_end");

    /* wall=115 ms, 12 tokens -> 104 tok/s (truncated); decode itl {15,5,40,5}
     * -> p50 = 5. Exact one-line summary. */
    n = mm_tel_summary(&t, buf, sizeof buf);
    CHECK(n > 0 &&
          strcmp(buf, "104 tok/s | 4 decode rounds | itl p50 5 ms | "
                      "kv rounds 115") == 0,
          "tel: summary exact");

    n = mm_tel_report(&t, buf, sizeof buf);
    CHECK(n > 0 &&
          strstr(buf, "rounds 5 (prefill 1, decode 4)  tokens 12") &&
          strstr(buf, "prefix cache: 3 hits / 0 misses (100.0%)") &&
          strstr(buf, "last 32 rounds:") &&
          strstr(buf, "prefill") && strstr(buf, "decode"),
          "tel: report table (counters + per-round rows)");

    /* Truncation: the return is the FULL length even when buf is small;
     * n < 2 is rejected. */
    {
        char small[64];
        CHECK(mm_tel_report(&t, small, sizeof small) > 64,
              "tel: report returns full length when truncated");
        CHECK(mm_tel_report(&t, small, 1) == -1, "tel: report n<2 -> -1");
    }

    /* Ring wrap-around: after MM_TEL_RING rounds head wraps to 0 and the
     * next sample overwrites ring[0] (carrying round number MM_TEL_RING). */
    mm_tel_init(&w);
    for (i = 0; i < MM_TEL_RING; i++)
        mm_tel_round(&w, TEL_DECODE, 1, 1, 0, (uint64_t)i * 10,
                     (uint64_t)i * 10 + 1);
    CHECK(w.head == 0 && w.rounds == MM_TEL_RING, "tel: ring wraps head");
    mm_tel_round(&w, TEL_DECODE, 1, 1, 0, (uint64_t)MM_TEL_RING * 10,
                 (uint64_t)MM_TEL_RING * 10 + 1);
    CHECK(w.head == 1 && w.rounds == MM_TEL_RING + 1, "tel: post-wrap step");
    CHECK(w.ring[0].round == MM_TEL_RING &&
          w.ring[0].t_ms == (uint64_t)MM_TEL_RING * 10,
          "tel: oldest slot overwritten by newest");
}
/* ------------------------------------------------------ F: engine path -- */

/* One full engine run against a .mimfer artifact on the CPU reference:
 * create (open + verify + tokenizer load) -> load -> submit -> step to
 * idle. On success the tokenizer must have been built from the artifact's
 * ART_TOK section (vocab/merges/special/eos) and e->art must be open. */
static mm_status engine_artifact_run(const char *artifact, uint32_t max_out,
                                     uint64_t seed, int *tok_ok,
                                     uint32_t *out, uint32_t cap,
                                     uint32_t *n_out)
{
    mm_engine_cfg cfg;
    mm_engine *e = NULL;
    const uint32_t prompt[4] = { 5, 9, 17, 42 };
    uint32_t n = 0, rounds = 0;
    mm_status s;

    *tok_ok = 0;
    mm_engine_cfg_default(&cfg);
    cfg.artifact = artifact;
    cfg.max_ctx = 128;
    cfg.kv_capacity = 0;
    cfg.kv_dtype = MM_KV_BF16;
    cfg.concurrency = 1;
    cfg.chunk = 8;
    cfg.draft = 0;
    cfg.temperature = 0.0f;
    cfg.top_k = 0;
    cfg.top_p = 1.0f;
    cfg.seed = seed;
    cfg.verbose = 0;
    cfg.no_graph = 1;

    s = mm_engine_create(&cfg, &e);
    if (s != MM_OK)
        return s;
    if (e->tok.vocab != 260 || e->tok.n_merges != 4 ||
        e->tok.n_special != 2 || e->tok.eos != 301 || !e->art)
    {
        mm_engine_destroy(e);
        return MM_ERR_STATE;
    }
    *tok_ok = 1;

    s = mm_engine_load(e);
    if (s != MM_OK)
    {
        mm_engine_destroy(e);
        return s;
    }
    if (mm_sched_submit(&e->sched, prompt, 4, max_out) != MM_OK)
    {
        mm_engine_destroy(e);
        return MM_ERR_STATE;
    }
    while (mm_sched_active(&e->sched) || mm_sched_queued(&e->sched))
    {
        s = mm_engine_step(e);
        if (s != MM_OK)
        {
            mm_engine_destroy(e);
            return s;
        }
        if (n < cap)
            out[n] = e->last_tok[0];
        n++;
        if (++rounds > 1024)
        {
            mm_engine_destroy(e);
            return MM_ERR_STATE;
        }
    }
    *n_out = n;
    mm_engine_destroy(e);
    return MM_OK;
}

static void section_engine(const char *path)
{
    char p[512];
    uint32_t a[1024], b[1024];
    uint32_t na = 0, nb = 0;
    int tk = 0, i;
    int same;
    mm_status s;

    printf("== artifact: engine --artifact path (CPU reference)\n");

    /* Positive: two independent runs, same seed -> byte-identical stream. */
    s = engine_artifact_run(path, 8, 777, &tk, a, 1024, &na);
    CHECK(s == MM_OK && tk, "engine: create+load+step from artifact");
    s = engine_artifact_run(path, 8, 777, &tk, b, 1024, &nb);
    CHECK(s == MM_OK && tk, "engine: second run (tokenizer loaded)");
    CHECK(na == (uint32_t)(1 + 8) && nb == na,
          "engine: rounds = 1 prefill + 8 decode");
    same = (na == nb);
    for (i = 0; same && i < (int)na; i++)
        if (a[i] != b[i])
            same = 0;
    CHECK(same, "engine: deterministic token stream across runs");
    printf("  artifact run: rounds=%u stream=[", (unsigned)na);
    for (i = 0; i < (int)na && i < 8; i++)
        printf("%s%u", i ? " " : "", a[i]);
    printf("]\n");

    /* Negatives: each must hard-fail at create with the right code. */
    {
        int t = 0;
        snprintf(p, sizeof p, "%s.eng_nosuch", path);
        s = engine_artifact_run(p, 8, 777, &t, a, 1024, &na);
        CHECK(s == MM_ERR_IO && !t, "engine: missing file -> IO");
    }
    {
        int t = 0;
        uint64_t zero = 0;
        snprintf(p, sizeof p, "%s.eng_sbck", path);
        CHECK(copy_file(path, p) == 0, "engine: copy for sb ck case");
        CHECK(patch_file(p, (size_t)SB_CK, &zero, 8) == 0,
              "engine: patch sb ck");
        s = engine_artifact_run(p, 8, 777, &t, a, 1024, &na);
        CHECK(s == MM_ERR_CORRUPT && !t, "engine: bad sb ck -> CORRUPT");
    }
    {
        int t = 0;
        snprintf(p, sizeof p, "%s.eng_magic", path);
        CHECK(copy_file(path, p) == 0, "engine: copy for magic case");
        CHECK(patch_file(p, (size_t)SB_MAGIC, "XXXX", 4) == 0,
              "engine: patch magic");
        s = engine_artifact_run(p, 8, 777, &t, a, 1024, &na);
        CHECK(s == MM_ERR_CORRUPT && !t, "engine: bad magic -> CORRUPT");
    }
    {
        int t = 0;
        snprintf(p, sizeof p, "%s.eng_notok", path);
        CHECK(write_fixture(p, 0) == MM_OK, "engine: write no-tok fixture");
        s = engine_artifact_run(p, 8, 777, &t, a, 1024, &na);
        CHECK(s == MM_ERR_CORRUPT && !t, "engine: no ART_TOK -> CORRUPT");
    }
}
/* --------------------------------------------------------------- main --- */

/* Recursively remove the temp fixture directory (best effort). */
static void rm_rf(const char *dir)
{
    struct dirent *d;
    struct stat st;
    char path[600];
    DIR *dp;

    dp = opendir(dir);
    if (!dp)
        return;
    while ((d = readdir(dp)) != NULL) {
        if (!strcmp(d->d_name, ".") || !strcmp(d->d_name, ".."))
            continue;
        snprintf(path, sizeof path, "%s/%s", dir, d->d_name);
        if (stat(path, &st) == 0) {
            if (S_ISDIR(st.st_mode))
                rm_rf(path);
            else
                unlink(path);
        }
    }
    closedir(dp);
    rmdir(dir);
}

int main(void)
{
    static char tmpl[] = "/tmp/mimfer_art_XXXXXX";
    char path[600];
    char *dir;

    dir = mkdtemp(tmpl);
    if (!dir) {
        fprintf(stderr, "artifact_test: mkdtemp failed\n");
        return 1;
    }

    build_tok_payload();
    build_min_payload();
    build_weights();
    build_model_blob();

    snprintf(path, sizeof path, "%s/fixture.mimfer", dir);
    if (write_fixture(path, 1) != MM_OK) {
        fprintf(stderr, "artifact_test: fixture write failed\n");
        rm_rf(dir);
        return 1;
    }

    section_container(path);
    section_integrity_1(path);
    section_integrity_2(path);
    section_tokenizer();
    section_mbuf();
    section_telemetry();
    section_engine(path);

    rm_rf(dir);

    if (g_fail) {
        fprintf(stderr, "ARTIFACT TEST FAILED\n");
        return 1;
    }
    printf("ARTIFACT TEST PASSED\n");
    return 0;
}
