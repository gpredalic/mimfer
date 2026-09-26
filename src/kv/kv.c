/*
 * Sequence state: paged KV pool, recurrent state pool, prefix registry.
 *
 * Host-side bookkeeping (block allocation, LRU) runs at most once per 64
 * tokens per sequence — it is never on the token-latency path. Device
 * structures (block tables, state bases) are fixed at create time so the
 * CUDA graphs capture them.
 */
#include "mimfer/kv.h"
#include "mimfer/cuda_rt.h"
#include <stdlib.h>
#include <string.h>

/* ----------------------------------------------------------- KV pool */

/* One layer, one of K or V: [kv_head][block][tok_in_block][head_dim]. */
static size_t kv_layer_bytes(const mm_model_cfg *mc, uint32_t n_blocks,
                             mm_kv_dtype dt)
{
    size_t elem = (dt == MM_KV_FP8) ? 1 : 2;
    return (size_t)mc->kv_heads * n_blocks * MM_KV_BLOCK_TOK *
           mc->head_dim * elem;
}

mm_status mm_kvpool_create(const mm_model_cfg *mc, uint32_t cap_tokens,
                           mm_kv_dtype dt, uint32_t max_ctx,
                           uint32_t concurrency, mm_arena *dev_arena,
                           mm_kvpool **out)
{
    mm_kvpool *p;
    uint32_t n_blocks, i;
    size_t layer_bytes, total, off, tab;
    void *base = NULL;

    MM_REQUIRE(mc && dev_arena && out, MM_ERR_STATE);
    MM_REQUIRE(cap_tokens % MM_KV_BLOCK_TOK == 0 && cap_tokens > 0,
               MM_ERR_RANGE);
    n_blocks = cap_tokens / MM_KV_BLOCK_TOK;

    p = calloc(1, sizeof *p);
    if (!p)
        return MM_ERR_NOMEM;
    p->mc = mc;
    p->dtype = dt;
    p->n_blocks = n_blocks;
    p->cap_tokens = (uint32_t)(n_blocks * MM_KV_BLOCK_TOK);
    p->max_blocks_seq = (max_ctx + MM_KV_BLOCK_TOK - 1) / MM_KV_BLOCK_TOK;
    p->concurrency = concurrency;

    layer_bytes = kv_layer_bytes(mc, n_blocks, dt);
    p->tok_stride = (size_t)mc->head_dim * ((dt == MM_KV_FP8) ? 1 : 2);
    p->block_stride = MM_KV_BLOCK_TOK * p->tok_stride;
    p->head_stride = (size_t)n_blocks * p->block_stride;

    total = (size_t)mc->n_full_layers * 2 * layer_bytes;
    MM_CHECK(mm_arena_alloc(dev_arena, mm_align_up(total, MM_PAGE_4K),
                            MM_PAGE_4K, "kv.pool", &base));

    /* Lay the 16 full layers out; k_base/v_base stay NULL for lin layers. */
    off = 0;
    for (i = 0; i < mc->layers; i++) {
        if (!mm_model_cfg_layer_is_full(mc, i + 1))
            continue;
        p->k_base[i] = (const char *)base + off;
        p->v_base[i] = (const char *)base + off + layer_bytes;
        off += 2 * layer_bytes;
    }

    /* Block tables: one fixed device array + host mirror. */
    tab = (size_t)concurrency * p->max_blocks_seq * 4;
    MM_CHECK(mm_arena_alloc(dev_arena, mm_align_up(tab, 256), 256, "kv.tab",
                            (void **)&p->blk_tab_dev));
    p->blk_tab_host = calloc(p->max_blocks_seq * concurrency, 4);
    /* Block ids run 1..n_blocks (id 0 is the "no block" sentinel), so the
     * id-indexed bookkeeping needs n_blocks+1 slots. */
    p->block_used = calloc(n_blocks + 1, 1);
    p->block_ref = calloc(n_blocks + 1, 2);
    p->free_list = calloc(n_blocks, 4);
    if (!p->blk_tab_host || !p->block_used || !p->block_ref ||
        !p->free_list) {
        mm_kvpool_destroy(p);
        return MM_ERR_NOMEM;
    }
    for (i = 0; i < n_blocks; i++)
        p->free_list[i] = i + 1;   /* id 0 == "no block" */
    p->free_top = n_blocks;

    *out = p;
    return MM_OK;
}

void mm_kvpool_destroy(mm_kvpool *p)
{
    if (!p)
        return;
    free(p->blk_tab_host);
    free(p->block_used);
    free(p->block_ref);
    free(p->free_list);
    free(p);
}

uint32_t mm_kvpool_free_blocks(const mm_kvpool *p)
{
    return p->free_top;
}

mm_status mm_kvpool_alloc_block(mm_kvpool *p, int slot, uint32_t block_idx,
                                uint32_t *blk_out)
{
    uint32_t idx, blk;

    MM_REQUIRE(slot >= 0 && slot < (int)p->concurrency, MM_ERR_RANGE);
    MM_REQUIRE(block_idx < p->max_blocks_seq, MM_ERR_RANGE);
    idx = (uint32_t)slot * p->max_blocks_seq + block_idx;
    if (p->blk_tab_host[idx]) {
        *blk_out = p->blk_tab_host[idx];
        return MM_OK;      /* already allocated for this slot */
    }
    if (p->free_top == 0)
        return MM_ERR_NOMEM;   /* engine: evict prefixes, retry */
    blk = p->free_list[--p->free_top];
    p->block_used[blk] = 1;
    p->block_ref[blk]++;
    p->blk_tab_host[idx] = blk;
    *blk_out = blk;
    return MM_OK;
}

mm_status mm_kvpool_free_block(mm_kvpool *p, uint32_t blk)
{
    MM_REQUIRE(blk > 0 && blk <= p->n_blocks, MM_ERR_STATE);
    if (p->block_ref[blk] > 0)
        p->block_ref[blk]--;
    if (p->block_ref[blk] == 0 && p->block_used[blk]) {
        p->block_used[blk] = 0;
        p->free_list[p->free_top++] = blk;
    }
    return MM_OK;
}

mm_status mm_kvpool_release_slot(mm_kvpool *p, int slot)
{
    uint32_t i;
    MM_REQUIRE(slot >= 0 && slot < (int)p->concurrency, MM_ERR_RANGE);
    for (i = 0; i < p->max_blocks_seq; i++) {
        uint32_t blk = p->blk_tab_host[(uint32_t)slot * p->max_blocks_seq + i];
        if (blk) {
            p->blk_tab_host[(uint32_t)slot * p->max_blocks_seq + i] = 0;
            MM_CHECK(mm_kvpool_free_block(p, blk));
        }
    }
    return MM_OK;
}

mm_status mm_kvpool_push_tab(mm_kvpool *p, int slot, uint32_t a, uint32_t b)
{
    MM_REQUIRE(a <= b && b <= p->max_blocks_seq, MM_ERR_RANGE);
    return mm_h2d_async((char *)p->blk_tab_dev +
                           (size_t)slot * p->max_blocks_seq * 4 + a * 4,
                       (const char *)p->blk_tab_host +
                           (size_t)slot * p->max_blocks_seq * 4 + a * 4,
                       (size_t)(b - a) * 4, MM_ST_XFER);
}

/* ------------------------------------------------ linear state pool */

mm_status mm_linstate_create(const mm_model_cfg *mc, uint32_t n_slots,
                             mm_arena *dev_arena, mm_linstate **out)
{
    mm_linstate *s;

    MM_REQUIRE(mc && dev_arena && out && n_slots > 0, MM_ERR_STATE);
    s = calloc(1, sizeof *s);
    if (!s)
        return MM_ERR_NOMEM;
    s->mc = mc;
    s->n_slots = n_slots;
    s->layer_bytes = (size_t)mc->lin_v_heads * mc->lin_k_dim * mc->lin_v_dim *
                     4u;
    s->seq_bytes = (size_t)mc->n_lin_layers * s->layer_bytes;
    s->conv_bytes = (size_t)mc->n_lin_layers * (mc->conv_k - 1) *
                    ((size_t)2 * mc->lin_k_heads * mc->lin_k_dim +
                     (size_t)mc->lin_v_heads * mc->lin_v_dim) * 2u;

    MM_CHECK(mm_arena_alloc(dev_arena,
                            mm_align_up((size_t)n_slots * s->seq_bytes,
                                        MM_PAGE_2M),
                            MM_PAGE_2M, "st.pool", &s->base));
    MM_CHECK(mm_arena_alloc(dev_arena,
                            mm_align_up((size_t)n_slots * s->conv_bytes,
                                        MM_PAGE_4K),
                            MM_PAGE_4K, "st.conv", &s->conv));
    *out = s;
    return MM_OK;
}

void mm_linstate_destroy(mm_linstate *s)
{
    free(s);
}

mm_status mm_linstate_reset(mm_linstate *s, int slot)
{
    MM_REQUIRE(slot >= 0 && slot < (int)s->n_slots, MM_ERR_RANGE);
    MM_CHECK(mm_mset_async((char *)s->base + (size_t)slot * s->seq_bytes, 0,
                           s->seq_bytes, MM_ST_COMPUTE));
    MM_CHECK(mm_mset_async((char *)s->conv + (size_t)slot * s->conv_bytes, 0,
                           s->conv_bytes, MM_ST_COMPUTE));
    return MM_OK;
}

/* ---------------------------------------------------- prefix registry */
/*
 * Open-addressed table, power-of-two capacity. Key: the chained FNV hash
 * of the 64-token block; value: the pool block id that already holds the
 * KV for that prefix. A deterministic content -> block mapping is what
 * makes reuse safe: two sequences that share a token prefix share the
 * exact same KV blocks.
 */
struct mm_prefix {
    uint32_t  cap;      /* power of two */
    uint32_t  mask;
    uint32_t  count;
    uint64_t  seed;     /* h_0 = FNV(model id) */
    uint64_t *h;
    uint32_t *blk;      /* 0 = empty */
    uint64_t *tick;     /* LRU tick */
};

mm_status mm_prefix_create(uint32_t max_blocks, const char *model_id,
                           mm_prefix **out)
{
    mm_prefix *p;
    uint32_t cap = 8;

    while (cap < max_blocks * 2)
        cap *= 2;
    p = calloc(1, sizeof *p);
    if (!p)
        return MM_ERR_NOMEM;
    p->cap = cap;
    p->mask = cap - 1;
    p->seed = mm_fnv1a64(model_id, strlen(model_id));
    p->h = calloc(cap, 8);
    p->blk = calloc(cap, 4);
    p->tick = calloc(cap, 8);
    if (!p->h || !p->blk || !p->tick) {
        free(p->h);
        free(p->blk);
        free(p->tick);
        free(p);
        return MM_ERR_NOMEM;
    }
    *out = p;
    return MM_OK;
}

void mm_prefix_destroy(mm_prefix *p)
{
    if (!p)
        return;
    free(p->h);
    free(p->blk);
    free(p->tick);
    free(p);
}

uint64_t mm_prefix_seed(const mm_prefix *p)
{
    return p->seed;
}

/* Chained FNV-1a continuation: extend `prev` over the block's token words.
 * h_i depends on the whole prefix, not just block i — that is what makes
 * the content->block mapping deterministic across sequences. */
uint64_t mm_prefix_hash_block(uint64_t prev, const uint32_t *toks, uint32_t n)
{
    uint64_t h = prev;
    const uint8_t *p = (const uint8_t *)toks;
    size_t m = (size_t)n * 4, i;
    for (i = 0; i < m; i++) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

static uint32_t slot_of(const mm_prefix *p, uint64_t h)
{
    /* Splitmix the hash for a good probe pattern in a 2^k table. */
    uint64_t x = h;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    x ^= x >> 31;
    return (uint32_t)x & p->mask;
}

uint32_t mm_prefix_lookup(mm_prefix *p, uint64_t h)
{
    uint32_t i = slot_of(p, h);

    for (;;) {
        if (p->h[i] == 0)
            return 0;      /* empty slot: the probe ran past the cluster */
        if (p->h[i] == h)
            return p->blk[i];
        i = (i + 1) & p->mask;
    }
}

mm_status mm_prefix_insert(mm_prefix *p, uint64_t h, uint32_t blk,
                           uint64_t tick)
{
    uint32_t i = slot_of(p, h);

    for (;;) {
        if (p->h[i] == 0 && p->blk[i] == 0) {
            p->h[i] = h;
            p->blk[i] = blk;
            p->tick[i] = tick;
            p->count++;
            return MM_OK;
        }
        if (p->h[i] == h) {
            /* Same prefix already mapped: content determinism means it
             * must be the same block; touch and agree. */
            if (p->blk[i] != blk)
                return MM_ERR_STATE;
            p->tick[i] = tick;
            return MM_OK;
        }
        i = (i + 1) & p->mask;
    }
}

void mm_prefix_touch(mm_prefix *p, uint64_t h, uint64_t tick)
{
    uint32_t i = slot_of(p, h);
    for (;;) {
        if (p->h[i] == 0 && p->blk[i] == 0)
            return;
        if (p->h[i] == h) {
            p->tick[i] = tick;
            return;
        }
        i = (i + 1) & p->mask;
    }
}

/*
 * Free up to `need` LRU-evictable blocks: full scan (the table is small
 * and this only runs when the pool is actually full), collect the oldest
 * ticks below `tick`, mark them empty, hand the block ids back through
 * freed_blks (capacity `need`). The engine calls mm_kvpool_free_block on
 * each id — the registry does not own the pool.
 */
uint32_t mm_prefix_evict(mm_prefix *p, uint64_t tick, uint32_t need,
                         uint32_t *freed_blks)
{
    uint32_t i, n = 0;

    if (need == 0)
        return 0;
    for (i = 0; i < p->cap && n < need; i++) {
        if (p->h[i] == 0 || p->blk[i] == 0)
            continue;
        if (p->tick[i] >= tick)
            continue;
        freed_blks[n++] = p->blk[i];
        p->h[i] = 0;
        p->blk[i] = 0;
        p->tick[i] = 0;
        p->count--;
    }
    return n;
}

void mm_prefix_clear(mm_prefix *p)
{
    memset(p->h, 0, p->cap * 8);
    memset(p->blk, 0, p->cap * 4);
    memset(p->tick, 0, p->cap * 8);
    p->count = 0;
}

size_t mm_prefix_count(const mm_prefix *p)
{
    return p->count;
}
