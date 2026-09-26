/*
 * Sequence state: paged KV + recurrent states + prefix registry.
 *
 * The Qwen3.8-27B text core is hybrid, so there are two state systems:
 *
 *  (1) KV POOL — the 16 full-attention layers. Paged: 64 tokens per page.
 *     One big device allocation. Layout per layer, K and V separately:
 *         [kv_head][block][tok_in_block][head_dim]
 *     so a decode CTA reading one (kv_head, t) row gets a contiguous
 *     head_dim-stride row (coalesced). Block tables are per-slot device
 *     arrays of u32 block ids, mirrored on the host.
 *     Size math (bf16, per token, all 16 layers):
 *         16 layers x 4 kv_heads x 2 (K,V) x 256 dim x 2 B = 65536 B
 *     => 32768-token pool = 2 GiB. See design.md "Memory flow".
 *
 *  (2) STATE POOL — the 48 Gated-DeltaNet layers. Per sequence slot:
 *         per layer: [lin_v_heads][lin_k_dim][lin_v_dim] float
 *                   = 48 x 128 x 128 x 4 B = 3145728 B
 *         per seq  : 48 layers x 3 MiB = 151 MiB
 *         conv tail: (q+k+v = 10240 channels) x (conv_k-1) x 2 B
 *     States are O(1) per token; this pool is why the hybrid model is a
 *     great fit for a 24 GiB appliance: context costs 64 KB/token of KV
 *     and the linear layers cost nothing beyond the fixed pool.
 *
 *  (3) PREFIX REGISTRY — host-side open-address table mapping a chained
 *     FNV-1a64 over 64-token blocks to the KV block holding that prefix.
 *     h_0 = FNV(model id); h_i = FNV_extend(h_{i-1}, block_i). On prefill
 *     every 64-token boundary is looked up: hits reuse the block (ref +1,
 *     no recompute), misses allocate. Eviction is LRU over the block list
 *     — a full scan is < 1 us and only runs when the pool is actually full.
 */
#ifndef MIMFER_KV_H
#define MIMFER_KV_H

#include "mimfer.h"
#include "config.h"
#include "alloc.h"

/* --------------------------------------------------------- KV pool */

typedef struct mm_kvpool {
    const mm_model_cfg *mc;
    mm_kv_dtype         dtype;
    uint32_t            n_blocks;        /* total pages in the pool        */
    uint32_t            cap_tokens;      /* n_blocks x MM_KV_BLOCK_TOK     */
    uint32_t            max_blocks_seq;  /* per-slot block table length    */
    uint32_t            concurrency;

    /* One base per layer for K and V; head/block offsets are computed in
     * kernels from the constants below (no per-call host math). */
    const void *k_base[MM_Q38_LAYERS];
    const void *v_base[MM_Q38_LAYERS];
    size_t      head_stride;   /* bytes per kv_head (all blocks)           */
    size_t      block_stride;  /* bytes per block within one kv_head       */
    size_t      tok_stride;    /* bytes per token within a block           */

    /* Block allocation: bitmap + freelist (u32, host-side; block ops are
     * rare — at most one per 64 tokens per seq). block_ref counts owners
     * (slot tables + prefix registry); a block is reusable only at 0. */
    uint8_t   *block_used;
    uint16_t  *block_ref;
    uint32_t  *free_list;
    uint32_t   free_top;

    /* Per-slot block tables: device (kernel-visible) + host mirror. */
    uint32_t  *blk_tab_dev;   /* [concurrency][max_blocks_seq]            */
    uint32_t  *blk_tab_host;  /* same shape, host mirror                  */
} mm_kvpool;

mm_status mm_kvpool_create(const mm_model_cfg *mc, uint32_t cap_tokens,
                           mm_kv_dtype dt, uint32_t max_ctx,
                           uint32_t concurrency, mm_arena *dev_arena,
                           mm_kvpool **out);
void      mm_kvpool_destroy(mm_kvpool *p);
uint32_t  mm_kvpool_free_blocks(const mm_kvpool *p);
/* Allocate a block for slot s at position pos (block index = pos/64).
 * Block id 0 is reserved for "no block". */
mm_status mm_kvpool_alloc_block(mm_kvpool *p, int slot, uint32_t block_idx,
                                uint32_t *blk_out);
/* Release a block (refcount 0 only). */
mm_status mm_kvpool_free_block(mm_kvpool *p, uint32_t blk);
/* Copy the slot's block table slice [a..b) to device (async). */
mm_status mm_kvpool_push_tab(mm_kvpool *p, int slot, uint32_t a, uint32_t b);
/* Drop a whole slot's table (sequence finished). */
mm_status mm_kvpool_release_slot(mm_kvpool *p, int slot);

/* ------------------------------------------------- linear state pool */

typedef struct mm_linstate {
    const mm_model_cfg *mc;
    uint32_t            n_slots;
    size_t              layer_bytes;   /* one linear layer's state         */
    size_t              seq_bytes;     /* all linear layers, one slot      */
    size_t              conv_bytes;    /* conv tail, one slot              */
    void               *base;          /* dev: [slot][layer][h][k][v]      */
    void               *conv;          /* dev: [slot][layer][conv_k-1][C]  */
} mm_linstate;

mm_status mm_linstate_create(const mm_model_cfg *mc, uint32_t n_slots,
                             mm_arena *dev_arena, mm_linstate **out);
void      mm_linstate_destroy(mm_linstate *s);
/* Zero one slot (new sequence). Async on the compute stream. */
mm_status mm_linstate_reset(mm_linstate *s, int slot);

/* ---------------------------------------------------- prefix registry */

typedef struct mm_prefix mm_prefix;
mm_status mm_prefix_create(uint32_t max_blocks, const char *model_id,
                           mm_prefix **out);
void      mm_prefix_destroy(mm_prefix *p);

/* Chain the block hash. Caller passes the previous hash (start: seed). */
uint64_t  mm_prefix_hash_block(uint64_t prev, const uint32_t *toks,
                               uint32_t n);
/* The h_0 seed (FNV of the model id) — start the chain with this. */
uint64_t  mm_prefix_seed(const mm_prefix *p);

/* 0 = miss. */
uint32_t  mm_prefix_lookup(mm_prefix *p, uint64_t h);
/* Claim block id for hash h (must be free or already owned). */
mm_status mm_prefix_insert(mm_prefix *p, uint64_t h, uint32_t blk,
                           uint64_t tick);
/* Bump the LRU tick of a prefix's block. */
void      mm_prefix_touch(mm_prefix *p, uint64_t h, uint64_t tick);
/* Free up to `need` LRU-evictable blocks; returns how many freed. */
uint32_t  mm_prefix_evict(mm_prefix *p, uint64_t tick, uint32_t need,
                          uint32_t *freed_blks);
/* Drop everything (engine teardown / benchmark reset). */
void      mm_prefix_clear(mm_prefix *p);
size_t    mm_prefix_count(const mm_prefix *p);

#endif /* MIMFER_KV_H */
