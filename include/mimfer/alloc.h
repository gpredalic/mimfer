/*
 * Memory subsystem.
 *
 * The whole device-side memory story, stated once:
 *
 *   Exactly FOUR cudaMalloc calls exist in the process, made once at load:
 *
 *     w_arena    weights + embed + lm-head + MTP (NVFP4 packed, resident,
 *                never touched after load except reads by kernels)
 *     kv_pool    paged KV for the 16 full-attention layers (one allocation,
 *                block-allocated at runtime, see kv.h)
 *     state_pool recurrent states of the 48 linear-attention layers, one
 *                slot per active sequence (fp32), plus conv state
 *     a_arena    activations + scratch for one prefill chunk (bump arena,
 *                reset every step; never grows past load-time sizing)
 *
 *   ...plus one small cudaHostAlloc for token I/O (pinned, async copy).
 *
 * Consequences:
 *   - no per-tensor allocation, ever  -> zero fragmentation by construction
 *   - every kernel argument pointer is FIXED at load time  -> CUDA Graphs
 *     capture cleanly; the runtime path is graph launch + bookkeeping only
 *   - residency is the slot table itself; `mimfer check --map` prints it
 *   - the activation arena is the only "mutable" region and it is reset by
 *     bump, so a step can never allocate more than the previous step
 *
 * Host memory (token buffers, prefix registry, artifact TOC) uses plain
 * calloc/free — it is tiny (< 256 MiB worst case) and off the hot path.
 */
#ifndef MIMFER_ALLOC_H
#define MIMFER_ALLOC_H

#include "mimfer.h"

typedef struct mm_slot {
    size_t      off;       /* offset from arena base                        */
    size_t      bytes;
    const char *name;      /* "w.layers.07.qkv" etc; NULL for anonymous     */
    int         pinned;    /* survives reset()                              */
} mm_slot;

typedef struct mm_arena {
    void       *base;       /* device or host pointer (canonical)            */
    void       *hbase;      /* host-side view: == base on host arenas; on
                              * MM_WITH_CUDA device arenas a calloc'd mirror
                              * of the same size (load-time staging: weights
                              * are filled on hbase, then one H2D; readbacks
                              * stage back from base). Never dereferenced
                              * from a device kernel. */
    size_t      cap;
    size_t      used;
    size_t      mark;       /* reset point for resettable arenas             */
    int         is_device;
    int         resettable;
    mm_slot    *slots;
    uint32_t    n_slots;
    uint32_t    cap_slots;
    char        tag[16];    /* "w", "kv", "st", "a"                         */
} mm_arena;

/* Device arenas call cudaMalloc once; host arenas calloc. */
mm_status mm_arena_init_device(mm_arena *a, size_t cap, const char *tag);
mm_status mm_arena_init_host(mm_arena *a, size_t cap, const char *tag);
void      mm_arena_free(mm_arena *a);

/* Bump-allocate `bytes` aligned to `align` (power of two, >= 16).
 * Returns MM_ERR_NOMEM when the arena would overflow. */
mm_status mm_arena_alloc(mm_arena *a, size_t bytes, size_t align,
                         const char *name, void **out);
/* Convenience for the common bf16 row-major [r x c] case. */
mm_status mm_arena_alloc_2d(mm_arena *a, size_t r, size_t c, size_t elem,
                            size_t align, const char *name, void **out);

/* Mark the slot containing p as pinned (survives reset()). */
void mm_arena_pin(mm_arena *a, const void *p);

void  *mm_arena_at(const mm_arena *a, size_t off);
/* Host-side twin of an offset / canonical pointer (see hbase). Host arenas
 * are identity mappings; device arenas map into the host mirror. NULL out
 * when the arena is empty or the pointer is outside the arena's range. */
void  *mm_arena_at_host(const mm_arena *a, size_t off);
void  *mm_arena_host_ptr(const mm_arena *a, const void *p);
size_t mm_arena_used(const mm_arena *a);
size_t mm_arena_free_bytes(const mm_arena *a);

/* Resettable arenas (a_arena): mark() saves the high-water mark,
 * reset() returns used to the mark and drops non-pinned slots. */
mm_status mm_arena_mark(mm_arena *a);
mm_status mm_arena_reset(mm_arena *a);

/* Human-readable slot table into buf (returns bytes written, or -1).
 * Used by `mimfer check --map` and startup logging. */
int mm_arena_dump(const mm_arena *a, char *buf, size_t n);

#endif /* MIMFER_ALLOC_H */
