/*
 * Byte-BPE tokenizer (encode + decode).
 *
 * The artifact's ART_TOK section carries the full model tokenizer
 * (model data, not engine code):
 *
 *   u32  vocab_size
 *   u32  n_merges
 *   u32  n_special
 *   merges:   [n_merges][2] u32   (pair of ids -> new id)
 *   special:  [n_special][2] {u32 id, u32 str_off}
 *   vstr_off: [vocab_size] u32    offset into vstr
 *   vstr:     packed UTF-8 strings
 *
 * Byte tokens are ids 0..255 (single byte values); merge 0 produces id
 * 256, merge i produces id 256+i, matching the Qwen BPE layout.
 *
 * Encode: bytes -> greedy merge by lowest merge rank. A binary heap over
 * (rank, position) pairs gives O(n log n); prompts are far smaller than
 * the model, so this host cost is irrelevant to latency (it runs once
 * per prompt, on the submit path).
 *
 * Decode: string concatenation by construction (the Qwen vocab uses
 * byte-level BPE, so concatenation is exact).
 */

#ifdef __cplusplus
extern "C" {
#endif

#ifndef MIMFER_TOKENIZER_H
#define MIMFER_TOKENIZER_H

#include "mimfer.h"

typedef struct mm_tok {
    uint32_t     vocab;
    uint32_t     n_merges;
    const uint32_t *merges;   /* [n_merges][2]                            */
    uint32_t     n_special;
    const uint32_t *special;  /* [n_special][2]: id, str_off; the LAST    */
                              /* entry is eos                             */
    const uint32_t *voff;     /* [vocab] offset into vstr                 */
    const uint8_t  *vstr;
    uint32_t     vstr_len;
    uint32_t     eos;
    /* open-addressing merge hash (built at load): key = (a<<32)|b, +1 to
     * keep 0 as the empty marker; val = merge rank. Writable (calloc'd). */
    uint64_t *mkey;
    uint32_t *mval;
    uint32_t       mcap;
} mm_tok;

/* data = ART_TOK payload. Takes ownership of nothing (mmap-backed). */
mm_status mm_tok_load(const uint8_t *data, size_t n, mm_tok **out);
void mm_tok_free(mm_tok *t);

/* Encode up to cap tokens; *trunc = 1 if the input was cut. */
mm_status mm_tok_encode(const mm_tok *t, const uint8_t *text, size_t n,
                        uint32_t *out, uint32_t cap, int *trunc);
/* Decode into buf (NUL-terminated); returns bytes written or -1. */
int mm_tok_decode(const mm_tok *t, const uint32_t *toks, uint32_t n,
                  char *buf, size_t cap);
/* Token string (may be empty for special ids). */
const char *mm_tok_str(const mm_tok *t, uint32_t id);
uint32_t mm_tok_eos(const mm_tok *t);

#endif /* MIMFER_TOKENIZER_H */
#ifdef __cplusplus
}
#endif
