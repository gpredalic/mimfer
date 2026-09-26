/*
 * Byte-BPE tokenizer.
 *
 * Encode: bytes -> greedy lowest-rank merges over a binary heap of
 * (rank, pos, a, b) entries with a doubly-linked list of live positions.
 * Stale heap entries (their pair no longer adjacent/alive) are skipped.
 */
#include "mimfer/tokenizer.h"
#include <stdlib.h>
#include <string.h>

static uint32_t next_pow2(uint32_t v)
{
    v--;
    v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16;
    return v + 1;
}

mm_status mm_tok_load(const uint8_t *data, size_t n, mm_tok **out)
{
    mm_tok *t;
    uint32_t vocab, n_merges, n_special, off, mcap, i;

    MM_REQUIRE(data && out, MM_ERR_STATE);
    if (n < 12)
        return MM_ERR_CORRUPT;
    memcpy(&vocab, data + 0, 4);
    memcpy(&n_merges, data + 4, 4);
    memcpy(&n_special, data + 8, 4);
    off = 12;
    if ((size_t)n_merges * 8 > n - off) return MM_ERR_CORRUPT;
    off += n_merges * 8;
    if ((size_t)n_special * 8 > n - off) return MM_ERR_CORRUPT;
    off += n_special * 8;
    if (vocab == 0 || (size_t)vocab * 4 > n - off) return MM_ERR_CORRUPT;
    off += (size_t)vocab * 4;
    if (n_special < 1 || vocab < 256)
        return MM_ERR_CORRUPT;

    t = calloc(1, sizeof *t);
    if (!t)
        return MM_ERR_NOMEM;
    t->vocab = vocab;
    t->n_merges = n_merges;
    t->merges = (const uint32_t *)(data + 12);
    t->n_special = n_special;
    t->special = (const uint32_t *)(data + 12 + (size_t)n_merges * 8);
    t->voff = (const uint32_t *)(data + off - (size_t)vocab * 4);
    t->vstr = data + off;
    t->vstr_len = (uint32_t)(n - off);
    t->eos = t->special[(size_t)n_special * 2 - 2];   /* last entry */

    /* Merge hash. */
    if (n_merges > 0) {
        mcap = next_pow2((uint32_t)((size_t)n_merges * 2));
        t->mkey = calloc(mcap, 8);
        t->mval = calloc(mcap, 4);
        if (!t->mkey || !t->mval) {
            free(t->mkey);
            free(t->mval);
            free(t);
            return MM_ERR_NOMEM;
        }
        t->mcap = mcap;
        for (i = 0; i < n_merges; i++) {
            uint64_t key = ((uint64_t)t->merges[i * 2] << 32) |
                           (t->merges[i * 2 + 1] + 1);
            uint32_t s = (uint32_t)key & (mcap - 1);
            for (;;) {
                if (t->mkey[s] == 0) {
                    t->mkey[s] = key;
                    t->mval[s] = i;
                    break;
                }
                s = (s + 1) & (mcap - 1);
            }
        }
    }
    *out = t;
    return MM_OK;
}

void mm_tok_free(mm_tok *t)
{
    if (!t)
        return;
    free(t->mkey);
    free(t->mval);
    free(t);
}

uint32_t mm_tok_eos(const mm_tok *t)
{
    return t->eos;
}

const char *mm_tok_str(const mm_tok *t, uint32_t id)
{
    if (id < t->vocab)
        return (const char *)t->vstr + t->voff[id];
    for (uint32_t i = 0; i < t->n_special; i++) {
        if (t->special[i * 2] == id)
            return (const char *)t->vstr + t->special[i * 2 + 1];
    }
    return "";
}

/* Rank of pair (a, b), or -1 for none. */
static int64_t tok_rank(const mm_tok *t, uint32_t a, uint32_t b)
{
    uint64_t key = ((uint64_t)a << 32) | (b + 1);
    uint32_t s = (uint32_t)key & (t->mcap - 1);

    for (;;) {
        if (t->mkey[s] == 0)
            return -1;
        if (t->mkey[s] == key)
            return t->mval[s];
        s = (s + 1) & (t->mcap - 1);
    }
}

/* ------------------------------------------------------------- encode */

typedef struct {
    int64_t rank;
    uint32_t pos;
    uint32_t a;
    uint32_t b;
} hnode;

static void heap_push(hnode *h, uint32_t *nh, const hnode n)
{
    uint32_t i = (*nh)++;
    h[i] = n;
    while (i > 0) {
        uint32_t p = (i - 1) / 2;
        if (h[p].rank <= h[i].rank)
            break;
        {
            hnode tmp = h[p];
            h[p] = h[i];
            h[i] = tmp;
        }
        i = p;
    }
}

static hnode heap_pop(hnode *h, uint32_t *nh)
{
    hnode top = h[0];
    h[0] = h[--(*nh)];
    {
        uint32_t i = 0;
        for (;;) {
            uint32_t l = 2 * i + 1, r = l + 1, m = i;
            if (l < *nh && h[l].rank < h[m].rank)
                m = l;
            if (r < *nh && h[r].rank < h[m].rank)
                m = r;
            if (m == i)
                break;
            {
                hnode tmp = h[m];
                h[m] = h[i];
                h[i] = tmp;
            }
            i = m;
        }
    }
    return top;
}

mm_status mm_tok_encode(const mm_tok *t, const uint8_t *text, size_t n,
                        uint32_t *out, uint32_t cap, int *trunc)
{
    uint32_t *ids;
    int32_t *prev, *next;
    uint8_t *dead;
    hnode *heap;
    uint32_t nh = 0;
    int64_t r;

    MM_REQUIRE(text && out, MM_ERR_STATE);
    *trunc = 0;
    if (n == 0)
        return MM_OK;
    ids = malloc(n * 4);
    prev = malloc(n * 4);
    next = malloc(n * 4);
    dead = malloc(n);
    heap = malloc(n * sizeof(hnode));
    if (!ids || !prev || !next || !dead || !heap) {
        free(ids); free(prev); free(next); free(dead); free(heap);
        return MM_ERR_NOMEM;
    }
    memcpy(ids, text, n);
    for (uint32_t i = 0; i < (uint32_t)n; i++) {
        prev[i] = (int32_t)(i - 1);
        next[i] = i + 1 < (uint32_t)n ? (int32_t)(i + 1) : -1;
        dead[i] = 0;
    }

    for (uint32_t i = 0; i + 1 < (uint32_t)n; i++) {
        r = tok_rank(t, ids[i], ids[i + 1]);
        if (r >= 0)
            heap_push(heap, &nh, (hnode){r, i, ids[i], ids[i + 1]});
    }

    while (nh > 0) {
        hnode e = heap_pop(heap, &nh);
        int32_t q;

        if (dead[e.pos])
            continue;
        q = next[e.pos];
        if (q < 0 || dead[q] || ids[e.pos] != e.a || ids[q] != e.b)
            continue;
        /* Merge e.pos and q. */
        ids[e.pos] = 256 + (uint32_t)e.rank;
        next[e.pos] = next[q];
        if (next[q] >= 0)
            prev[next[q]] = e.pos;
        dead[q] = 1;

        if (prev[e.pos] >= 0 && !dead[prev[e.pos]]) {
            r = tok_rank(t, ids[prev[e.pos]], ids[e.pos]);
            if (r >= 0)
                heap_push(heap, &nh,
                          (hnode){r, (uint32_t)prev[e.pos],
                                  ids[prev[e.pos]], ids[e.pos]});
        }
        if (next[e.pos] >= 0 && !dead[next[e.pos]]) {
            r = tok_rank(t, ids[e.pos], ids[next[e.pos]]);
            if (r >= 0)
                heap_push(heap, &nh,
                          (hnode){r, e.pos, ids[e.pos], ids[next[e.pos]]});
        }
    }

    {
        uint32_t o = 0, i;
        for (i = 0; i < (uint32_t)n; i++) {
            if (!dead[i]) {
                if (o >= cap) {
                    *trunc = 1;
                    break;
                }
                out[o++] = ids[i];
            }
        }
    }
    free(ids); free(prev); free(next); free(dead); free(heap);
    return MM_OK;
}

/* ------------------------------------------------------------- decode */

int mm_tok_decode(const mm_tok *t, const uint32_t *toks, uint32_t n,
                  char *buf, size_t cap)
{
    size_t o = 0;

    if (n > 0 && cap == 0)
        return -1;
    for (uint32_t i = 0; i < n; i++) {
        const char *s;
        size_t len;

        if (toks[i] < t->vocab) {
            s = (const char *)t->vstr + t->voff[toks[i]];
            len = toks[i] + 1 < t->vocab
                      ? (size_t)(t->voff[toks[i] + 1] - t->voff[toks[i]])
                      : (size_t)t->vstr_len - t->voff[toks[i]];
        } else {
            s = mm_tok_str(t, toks[i]);
            len = strlen(s);
        }
        if (o + len + 1 > cap)
            return -1;
        memcpy(buf + o, s, len);
        o += len;
    }
    buf[o] = '\0';
    return (int)o;
}