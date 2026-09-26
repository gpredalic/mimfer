/*
 * The .mimfer artifact container — original format, no dependencies.
 *
 * Layout (all integers little-endian):
 *
 *   offset 0        superblock, 4096 B:
 *     u32  magic      0x464D494D ("MIMF")
 *     u16  fmt_major  2
 *     u16  fmt_minor  0
 *     u32  cap        capability bitmap: bit i set => section type i
 *                     exists (or is understood); readers skip unknowns
 *     u32  min_reader minimum reader major.minor (u16 u16) + 0
 *     u32  n_sections
 *     u32  toc_off    4 KiB aligned offset of the TOC
 *     u64  file_size
 *     u64  super_ck   FNV-1a64 of superblock bytes [0..32)
 *     u8   model_id[64]     "qwen3.8-27b"
 *     u8   weights_id[32]   "nvfp4"
 *     u8   recipe[64]       conversion recipe tag (provenance)
 *     u8   reserved[..]     zero
 *
 *   toc_off         n_sections x 128 B entries:
 *     u8   name[32]
 *     u32  type       (mm_arts_type)
 *     u32  flags      0 required, 1 optional, 2 skippable (fwd-compat)
 *     u64  offset
 *     u64  length
 *     u64  ck         FNV-1a64 of the whole payload
 *     u32  sub_step   sub-checksum granularity in bytes (0 = none)
 *     u32  n_subck
 *     u64  reserved
 *     pad to 128
 *
 *   payloads        4 KiB aligned; the ART_WEIGHTS payload carries its
 *                   own internal tensor index (see design.md "Weight
 *                   memory layout") so weights load by offset without
 *                   parsing the whole file.
 *
 * Integrity: superblock + TOC entry + per-section FNV-1a64; large weight
 * sections additionally carry 64 MiB sub-checksums so a download can be
 * verified in parallel and resumed.
 * Compatibility: readers load sections whose type bit is set in `cap`;
 * unknown types with the SKIPPABLE flag are ignored (forward), and a
 * reader older than min_reader refuses the file (backward).
 */

#ifdef __cplusplus
extern "C" {
#endif

#ifndef MIMFER_ARTIFACT_H
#define MIMFER_ARTIFACT_H

#include "mimfer.h"

/* Section types. Bits 0..63 map 1:1 into the capability word. */
typedef enum mm_arts_type {
    ART_MODEL = 0,    /* mm_model_cfg fixed layout + extras                */
    ART_TOK,          /* byte-BPE: merges + vocab strings                  */
    ART_CHAT,         /* chat template (UTF-8 text)                        */
    ART_WEIGHTS,      /* packed NVFP4 weights + internal tensor index      */
    ART_MTP,          /* MTP draft head (lazy-loaded for --spec mtp)       */
    ART_GENCFG,       /* default sampling policy (JSON-ish text)           */
    /* --- V2 artifact family ------------------------------------------- */
    ART_TEMPLATE,     /* prompt template (slots + render rules)            */
    ART_CTX,          /* context bundle: token stream + block hash chain   */
    ART_CONV,         /* cached conversation: history + hash chain         */
    ART_SESSION,      /* reusable session: state snapshot + policy         */
    /* --- V3 artifact family ------------------------------------------- */
    ART_MEMORY,       /* structured memory (mbuf graph)                    */
    ART_WORKFLOW,     /* workflow graph (steps + artifact refs)            */
    ART_AGENT,        /* agent state (slots, counters, RNG, open loops)    */
    ART_CKPT,         /* execution checkpoint (engine + gen snapshot)      */
    ART_REASON,       /* persistent reasoning metadata                     */
    ART_TASK,         /* reusable task bundle                              */
    ART_MAX
} mm_arts_type;

#define MM_ART_FLAG_REQUIRED  0u
#define MM_ART_FLAG_OPTIONAL  1u
#define MM_ART_FLAG_SKIPPABLE 2u

typedef struct mm_arts {
    char    name[32];
    uint32_t type;
    uint32_t flags;
    uint64_t offset;
    uint64_t length;
    uint64_t ck;
    uint32_t sub_step;
    uint32_t n_subck;
} mm_arts;

typedef struct mm_artifact {
    int       fd;
    uint8_t  *map;         /* mmap, read-only                             */
    size_t    map_len;
    uint16_t  major, minor;
    uint32_t  cap;
    uint32_t  min_reader;
    char      model_id[64];
    char      weights_id[32];
    char      recipe[64];
    mm_arts   toc[256];
    uint32_t  n_toc;
} mm_artifact;

mm_status mm_art_open(const char *path, mm_artifact **out);
void      mm_art_close(mm_artifact *a);
const mm_arts *mm_art_find(const mm_artifact *a, uint32_t type);
/* Payload pointer (no copy; the file is mmap'd). */
const uint8_t *mm_art_data(const mm_artifact *a, const mm_arts *s, size_t *len);
/* Verify superblock + TOC integrity. */
mm_status mm_art_verify(const mm_artifact *a);
/* Verify one section's checksum (and sub-checksums). */
mm_status mm_art_verify_section(const mm_artifact *a, const mm_arts *s);
/* Human-readable manifest (CLI: mimfer check). */
int       mm_art_dump(const mm_artifact *a, char *buf, size_t n);

/* Writer (used by tools/pack). payloads[i] must cover toc[i].length. */
mm_status mm_art_write(const char *path, uint16_t major, uint16_t minor,
                       uint32_t cap, const char *model_id,
                       const char *weights_id, const char *recipe,
                       mm_arts *secs, uint32_t n_secs,
                       const uint8_t *const *payloads);

/* ---------------------------------------------------- mbuf (v3) */
/*
 * mbuf: the tagged-value encoding for structured sections (memory,
 * workflow, agent, checkpoint, reasoning, task). Little-endian; all
 * strings UTF-8; every value is self-delimiting, so readers can skip
 * unknown tags (forward compatibility). Tags:
 *   0x00 end   0x01 u8   0x02 u16  0x03 u32  0x04 u64
 *   0x05 i64   0x06 f32  0x07 str(u16 len)  0x08 blob(u32 len)
 *   0x09 list(u32 n)  0x0A map(u32 n of str,value)  0x0B ref(u32 id)
 */
typedef enum {
    MBUF_END = 0, MBUF_U8, MBUF_U16, MBUF_U32, MBUF_U64, MBUF_I64,
    MBUF_F32, MBUF_STR, MBUF_BLOB, MBUF_LIST, MBUF_MAP, MBUF_REF
} mm_mbuf_tag;

/* Parse one value at *p (advances). Returns MM_ERR_CORRUPT on bad tag/len. */
mm_status mm_mbuf_read(const uint8_t **p, const uint8_t *end,
                       mm_mbuf_tag *tag, uint64_t *ival, float *fval,
                       const uint8_t **sval, uint32_t *slen);

#endif /* MIMFER_ARTIFACT_H */
#ifdef __cplusplus
}
#endif
