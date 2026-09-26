/*
 * The .mimfer artifact container (see artifact.h for the on-disk format).
 *
 * The file is mmap'd read-only; sections are never copied. Weight loading
 * streams from the mmap into device memory in slices, so the page cache
 * does the rest.
 */
#include "mimfer/artifact.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>

#define MM_ART_MAGIC     0x464D494Du   /* "MIMF" */
#define MM_ART_SUPER_SZ  4096u
#define MM_ART_TOC_ENTRY 128u
#define MM_ART_MAJOR     2u

/* Superblock offsets (artifact.h documents these). */
enum {
    SO_MAGIC = 0, SO_MAJOR = 4, SO_MINOR = 6, SO_CAP = 8,
    SO_MINREADER = 12, SO_NSEC = 16, SO_TOCOFF = 20, SO_FSIZE = 24,
    SO_CK = 32, SO_MODEL = 40, SO_WTS = 104, SO_RECIPE = 136,
};
/* TOC entry offsets. */
enum {
    TE_NAME = 0, TE_TYPE = 32, TE_FLAGS = 36, TE_OFF = 40,
    TE_LEN = 48, TE_CK = 56, TE_SUBSTEP = 64, TE_NSUBCK = 68,
};

/* Target is little-endian; the format is LE, so these are memcpy. */
static uint64_t rd_u64(const uint8_t *p)
{
    uint64_t v;
    memcpy(&v, p, 8);
    return v;
}
static uint32_t rd_u32(const uint8_t *p)
{
    uint32_t v;
    memcpy(&v, p, 4);
    return v;
}
static uint16_t rd_u16(const uint8_t *p)
{
    uint16_t v;
    memcpy(&v, p, 2);
    return v;
}
static void wr_u64(uint8_t *p, uint64_t v) { memcpy(p, &v, 8); }
static void wr_u32(uint8_t *p, uint32_t v) { memcpy(p, &v, 4); }
static void wr_u16(uint8_t *p, uint16_t v) { memcpy(p, &v, 2); }

mm_status mm_art_open(const char *path, mm_artifact **out)
{
    mm_artifact *a;
    struct stat st;
    int fd;
    const uint8_t *sb;
    uint32_t nsec, tocoff, i;
    uint64_t fsize;

    fd = open(path, O_RDONLY);
    if (fd < 0) {
        MM_LOGE("artifact: open %s failed: %s", path, strerror(errno));
        return MM_ERR_IO;
    }
    if (fstat(fd, &st) || st.st_size < MM_ART_SUPER_SZ + MM_ART_TOC_ENTRY) {
        close(fd);
        MM_LOGE("artifact: %s too small", path);
        return MM_ERR_CORRUPT;
    }
    a = calloc(1, sizeof *a);
    if (!a) {
        close(fd);
        return MM_ERR_NOMEM;
    }
    a->fd = fd;
    a->map_len = (size_t)st.st_size;
    a->map = mmap(NULL, a->map_len, PROT_READ, MAP_PRIVATE, fd, 0);
    if (a->map == MAP_FAILED) {
        free(a);
        close(fd);
        return MM_ERR_IO;
    }
    sb = a->map;

    if (rd_u32(sb + SO_MAGIC) != MM_ART_MAGIC) {
        MM_LOGE("artifact: bad magic in %s", path);
        goto corrupt;
    }
    a->major = rd_u16(sb + SO_MAJOR);
    a->minor = rd_u16(sb + SO_MINOR);
    if (a->major != MM_ART_MAJOR) {
        MM_LOGE("artifact: format v%u, reader is v%u (major mismatch)",
                a->major, MM_ART_MAJOR);
        goto corrupt;
    }
    a->cap = rd_u32(sb + SO_CAP);
    a->min_reader = rd_u32(sb + SO_MINREADER);
    {
        uint16_t mr = (uint16_t)(a->min_reader >> 16), mn = (uint16_t)(a->min_reader & 0xffff);
        if (mr > a->major || (mr == a->major && mn > a->minor)) {
            MM_LOGE("artifact: needs reader >= %u.%u, this is %u.%u",
                    mr, mn, a->major, a->minor);
            goto corrupt;
        }
    }
    nsec = rd_u32(sb + SO_NSEC);
    tocoff = rd_u32(sb + SO_TOCOFF);
    fsize = rd_u64(sb + SO_FSIZE);
    if (fsize != (uint64_t)st.st_size) {
        MM_LOGE("artifact: file_size %llu != actual %zu",
                (unsigned long long)fsize, a->map_len);
        goto corrupt;
    }
    if (tocoff % MM_PAGE_4K || (uint64_t)tocoff + (uint64_t)nsec *
                                       MM_ART_TOC_ENTRY > a->map_len ||
        nsec > 256) {
        MM_LOGE("artifact: bad TOC geometry (nsec %u, off %u)", nsec, tocoff);
        goto corrupt;
    }
    snprintf(a->model_id, sizeof a->model_id, "%s", (const char *)sb + SO_MODEL);
    snprintf(a->weights_id, sizeof a->weights_id, "%s",
             (const char *)sb + SO_WTS);
    snprintf(a->recipe, sizeof a->recipe, "%s", (const char *)sb + SO_RECIPE);
    a->n_toc = nsec;

    for (i = 0; i < nsec; i++) {
        const uint8_t *e = a->map + tocoff + (size_t)i * MM_ART_TOC_ENTRY;
        mm_arts *t = &a->toc[i];
        uint64_t off, len;

        snprintf(t->name, sizeof t->name, "%s", (const char *)e + TE_NAME);
        t->type = rd_u32(e + TE_TYPE);
        t->flags = rd_u32(e + TE_FLAGS);
        off = rd_u64(e + TE_OFF);
        len = rd_u64(e + TE_LEN);
        t->ck = rd_u64(e + TE_CK);
        t->sub_step = rd_u32(e + TE_SUBSTEP);
        t->n_subck = rd_u32(e + TE_NSUBCK);
        t->offset = off;
        t->length = len;
        if (t->type >= ART_MAX) {
            if (t->flags != MM_ART_FLAG_SKIPPABLE) {
                MM_LOGE("artifact: unknown section type %u \"%s\" "
                        "(not skippable)", t->type, t->name);
                goto corrupt;
            }
            MM_LOGI("artifact: skipping unknown section \"%s\" (type %u)",
                    t->name, t->type);
        }
        if (off + len > a->map_len) {
            MM_LOGE("artifact: section \"%s\" runs past EOF", t->name);
            goto corrupt;
        }
    }
    *out = a;
    return MM_OK;

corrupt:
    mm_art_close(a);
    return MM_ERR_CORRUPT;
}

void mm_art_close(mm_artifact *a)
{
    if (!a)
        return;
    if (a->map)
        munmap(a->map, a->map_len);
    if (a->fd >= 0)
        close(a->fd);
    free(a);
}

const mm_arts *mm_art_find(const mm_artifact *a, uint32_t type)
{
    for (uint32_t i = 0; i < a->n_toc; i++)
        if (a->toc[i].type == type)
            return &a->toc[i];
    return NULL;
}

const uint8_t *mm_art_data(const mm_artifact *a, const mm_arts *s, size_t *len)
{
    *len = (size_t)s->length;
    return a->map + s->offset;
}

mm_status mm_art_verify(const mm_artifact *a)
{
    uint64_t ck = mm_fnv1a64(a->map, 32);
    if (ck != rd_u64(a->map + SO_CK)) {
        MM_LOGE("artifact: superblock checksum mismatch");
        return MM_ERR_CORRUPT;
    }
    for (uint32_t i = 0; i < a->n_toc; i++) {
        const mm_arts *s = &a->toc[i];
        if (s->type >= ART_MAX)
            continue;
        MM_CHECK(mm_art_verify_section(a, s));
    }
    return MM_OK;
}

mm_status mm_art_verify_section(const mm_artifact *a, const mm_arts *s)
{
    const uint8_t *p = a->map + s->offset;

    if (mm_fnv1a64(p, (size_t)s->length) != s->ck) {
        MM_LOGE("artifact: checksum mismatch in section \"%s\"", s->name);
        return MM_ERR_CORRUPT;
    }
    if (s->n_subck == 0)
        return MM_OK;
    /* The payload starts with n_subck u64 FNV values; each covers the
     * following sub_step bytes of the data (last chunk may be short). */
    {
        size_t ckblk = (size_t)s->n_subck * 8;
        const uint8_t *data;
        uint64_t dlen;

        if (ckblk > s->length) {
            MM_LOGE("artifact: section \"%s\" sub-ck block past payload",
                    s->name);
            return MM_ERR_CORRUPT;
        }
        data = p + ckblk;
        dlen = (size_t)s->length - ckblk;
        for (uint32_t i = 0; i < s->n_subck; i++) {
            size_t off = (size_t)i * s->sub_step;
            size_t len = s->sub_step;
            uint64_t exp, act;
            if (off > dlen)
                break;
            if (off + len > dlen)
                len = (size_t)dlen - off;
            exp = rd_u64(p + i * 8);
            act = mm_fnv1a64(data + off, len);
            if (exp != act) {
                MM_LOGE("artifact: sub-checksum %u/%u mismatch in \"%s\"",
                        i, s->n_subck, s->name);
                return MM_ERR_CORRUPT;
            }
        }
    }
    return MM_OK;
}

int mm_art_dump(const mm_artifact *a, char *buf, size_t n)
{
    int o = 0;
    o += snprintf(buf + o, n - (size_t)o,
                  ".mimfer v%u.%u  %s / %s (recipe: %s)\n",
                  a->major, a->minor, a->model_id, a->weights_id,
                  a->recipe);
    o += snprintf(buf + o, n - (size_t)o,
                  "  %-12s %-28s %12s %s\n", "type", "name", "size", "flags");
    for (uint32_t i = 0; i < a->n_toc && o < (int)n - 96; i++) {
        const mm_arts *s = &a->toc[i];
        char line[160];
        snprintf(line, sizeof line, "  %-12u %-28s %12zu %s%s\n",
                 s->type, s->name, (size_t)s->length,
                 s->flags == MM_ART_FLAG_REQUIRED ? "required" :
                 s->flags == MM_ART_FLAG_OPTIONAL ? "optional" : "skippable",
                 s->n_subck ? " +subck" : "");
        o += snprintf(buf + o, n - (size_t)o, "%s", line);
    }
    return o;
}

/* ---------------------------------------------------------------- write */
/*
 * Writer (packer). Two passes: (1) stream FNV over every payload to fill
 * in the TOC checksums, (2) pwrite the superblock, the TOC, then the
 * payloads at 4 KiB-aligned offsets. If secs[i].n_subck > 0, payloads[i]
 * must already begin with the sub-checksum block; the writer only
 * computes the whole-section ck.
 */
mm_status mm_art_write(const char *path, uint16_t major, uint16_t minor,
                       uint32_t cap, const char *model_id,
                       const char *weights_id, const char *recipe,
                       mm_arts *secs, uint32_t n_secs,
                       const uint8_t *const *payloads)
{
    int fd;
    uint64_t fsize, pos;
    uint8_t sb[MM_ART_SUPER_SZ];

    MM_REQUIRE(payloads && n_secs <= 256, MM_ERR_STATE);

    for (uint32_t i = 0; i < n_secs; i++)
        secs[i].ck = mm_fnv1a64(payloads[i], (size_t)secs[i].length);

    {
        uint64_t toc_bytes = (uint64_t)n_secs * MM_ART_TOC_ENTRY;
        pos = mm_align_up((uint64_t)MM_ART_SUPER_SZ + toc_bytes, MM_PAGE_4K);
        for (uint32_t i = 0; i < n_secs; i++) {
            secs[i].offset = pos;
            pos = mm_align_up(pos + secs[i].length, MM_PAGE_4K);
        }
        fsize = pos;
    }

    fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        MM_LOGE("artifact: create %s: %s", path, strerror(errno));
        return MM_ERR_IO;
    }
    memset(sb, 0, sizeof sb);
    wr_u32(sb + SO_MAGIC, MM_ART_MAGIC);
    wr_u16(sb + SO_MAJOR, major);
    wr_u16(sb + SO_MINOR, minor);
    wr_u32(sb + SO_CAP, cap);
    wr_u32(sb + SO_MINREADER, (uint32_t)major << 16 | minor);
    wr_u32(sb + SO_NSEC, n_secs);
    wr_u32(sb + SO_TOCOFF, MM_ART_SUPER_SZ);
    wr_u64(sb + SO_FSIZE, fsize);
    snprintf((char *)sb + SO_MODEL, 64, "%s", model_id ? model_id : "");
    snprintf((char *)sb + SO_WTS, 32, "%s", weights_id ? weights_id : "");
    snprintf((char *)sb + SO_RECIPE, 64, "%s", recipe ? recipe : "");
    wr_u64(sb + SO_CK, mm_fnv1a64(sb, 32));

    if (pwrite(fd, sb, sizeof sb, 0) != (ssize_t)sizeof sb)
        goto io;
    for (uint32_t i = 0; i < n_secs; i++) {
        uint8_t e[MM_ART_TOC_ENTRY];
        memset(e, 0, sizeof e);
        snprintf((char *)e + TE_NAME, 32, "%s", secs[i].name);
        wr_u32(e + TE_TYPE, secs[i].type);
        wr_u32(e + TE_FLAGS, secs[i].flags);
        wr_u64(e + TE_OFF, secs[i].offset);
        wr_u64(e + TE_LEN, secs[i].length);
        wr_u64(e + TE_CK, secs[i].ck);
        wr_u32(e + TE_SUBSTEP, secs[i].sub_step);
        wr_u32(e + TE_NSUBCK, secs[i].n_subck);
        if (pwrite(fd, e, sizeof e,
                   MM_ART_SUPER_SZ + (uint64_t)i * MM_ART_TOC_ENTRY)
            != (ssize_t)sizeof e)
            goto io;
        if (pwrite(fd, payloads[i], (size_t)secs[i].length,
                   secs[i].offset) != (ssize_t)secs[i].length)
            goto io;
    }
    if (close(fd) != 0)
        goto io;
    return MM_OK;
io:
    close(fd);
    MM_LOGE("artifact: write %s: %s", path, strerror(errno));
    return MM_ERR_IO;
}

/* --------------------------------------------------------------- mbuf */

mm_status mm_mbuf_read(const uint8_t **pp, const uint8_t *end,
                       mm_mbuf_tag *tag, uint64_t *ival, float *fval,
                       const uint8_t **sval, uint32_t *slen)
{
    const uint8_t *p = *pp;
    uint8_t t;

    if (p + 1 > end)
        return MM_ERR_CORRUPT;
    t = *p++;
    if (t > MBUF_REF)
        return MM_ERR_CORRUPT;
    *tag = (mm_mbuf_tag)t;
    switch (t) {
    case MBUF_END:
        break;
    case MBUF_U8:
        if (p + 1 > end) return MM_ERR_CORRUPT;
        if (ival) *ival = *p;
        p += 1;
        break;
    case MBUF_U16:
        if (p + 2 > end) return MM_ERR_CORRUPT;
        if (ival) *ival = rd_u16(p);
        p += 2;
        break;
    case MBUF_U32:
        if (p + 4 > end) return MM_ERR_CORRUPT;
        if (ival) *ival = rd_u32(p);
        p += 4;
        break;
    case MBUF_U64:
    case MBUF_I64:
        if (p + 8 > end) return MM_ERR_CORRUPT;
        if (ival) *ival = rd_u64(p);
        p += 8;
        break;
    case MBUF_F32: {
        float f;
        if (p + 4 > end) return MM_ERR_CORRUPT;
        memcpy(&f, p, 4);
        if (fval) *fval = f;
        p += 4;
        break;
    }
    case MBUF_STR: {
        uint16_t len;
        if (p + 2 > end) return MM_ERR_CORRUPT;
        len = rd_u16(p);
        p += 2;
        if (p + len > end) return MM_ERR_CORRUPT;
        if (sval) *sval = p;
        if (slen) *slen = len;
        p += len;
        break;
    }
    case MBUF_BLOB: {
        uint32_t len;
        if (p + 4 > end) return MM_ERR_CORRUPT;
        len = rd_u32(p);
        p += 4;
        if (p + len > end) return MM_ERR_CORRUPT;
        if (sval) *sval = p;
        if (slen) *slen = len;
        p += len;
        break;
    }
    case MBUF_LIST:
    case MBUF_MAP: {
        /* Returns the element count in *ival; the caller then reads the
         * values (pairs for MAP) and finally an MBUF_END at the same
         * nesting level. */
        uint32_t n;
        if (p + 4 > end) return MM_ERR_CORRUPT;
        n = rd_u32(p);
        p += 4;
        if (ival) *ival = n;
        break;
    }
    case MBUF_REF:
        if (p + 4 > end) return MM_ERR_CORRUPT;
        if (ival) *ival = rd_u32(p);
        p += 4;
        break;
    }
    *pp = p;
    return MM_OK;
}
