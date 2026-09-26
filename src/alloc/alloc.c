/*
 * Arena allocator: the four-allocation memory model (see alloc.h).
 *
 * Device arenas call cudaMalloc exactly once at init; host arenas calloc.
 * All later allocation is a bump. The resettable arena (a_arena) resets
 * to its mark every step; Pinned slots survive a reset (their off is kept
 * and used never falls back below the highest pinned slot end).
 */
#include "mimfer/alloc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(MM_WITH_CUDA)
#include <cuda_runtime.h>
#endif

/* ----------------------------------------------------------- backends */

static mm_status dev_alloc(size_t n, void **out)
{
#if defined(MM_WITH_CUDA)
    cudaError_t e = cudaMalloc(out, n);
    if (e != cudaSuccess) {
        MM_LOGE("cudaMalloc(%zu MiB): %s", n >> 20, cudaGetErrorString(e));
        return MM_ERR_NOMEM;
    }
    return MM_OK;
#else
    (void)n;
    (void)out;
    return MM_ERR_UNSUPPORTED;
#endif
}

static void dev_free(void *p)
{
#if defined(MM_WITH_CUDA)
    if (p)
        cudaFree(p);
#else
    (void)p;
#endif
}

/* ------------------------------------------------------------- init */

static mm_status arena_init_common(mm_arena *a, size_t cap, const char *tag,
                                   int is_device)
{
    memset(a, 0, sizeof *a);
    snprintf(a->tag, sizeof a->tag, "%s", tag ? tag : "?");
    a->cap = cap;
    a->is_device = is_device;
    a->slots = calloc(1024, sizeof(mm_slot));
    if (!a->slots)
        return MM_ERR_NOMEM;
    a->cap_slots = 1024;
    if (is_device) {
        if (dev_alloc(cap, &a->base) != MM_OK)
            return MM_ERR_NOMEM;
        /* Host mirror of the same size: the CUDA build's load-time staging
         * surface (weights are filled here, then one H2D; see alloc.h). On
         * a non-CUDA build dev_alloc already failed above, so this branch
         * is only reachable in MM_WITH_CUDA builds. */
        a->hbase = calloc(1, cap);
        if (!a->hbase) {
            dev_free(a->base);
            a->base = NULL;
            return MM_ERR_NOMEM;
        }
        return MM_OK;
    }
    a->base = calloc(1, cap);
    a->hbase = a->base;    /* identity mapping: host arenas have no twin */
    return a->base ? MM_OK : MM_ERR_NOMEM;
}

mm_status mm_arena_init_device(mm_arena *a, size_t cap, const char *tag)
{
    return arena_init_common(a, cap, tag, 1);
}

mm_status mm_arena_init_host(mm_arena *a, size_t cap, const char *tag)
{
    a->resettable = 0;
    return arena_init_common(a, cap, tag, 0);
}

void mm_arena_free(mm_arena *a)
{
    if (!a->base && !a->hbase)
        return;
    if (a->is_device && a->base)
        dev_free(a->base);
    else if (a->base)
        free(a->base);
    if (a->hbase && a->hbase != a->base)
        free(a->hbase);
    free(a->slots);
    memset(a, 0, sizeof *a);
}

/* -------------------------------------------------------- allocation */

mm_status mm_arena_alloc(mm_arena *a, size_t bytes, size_t align,
                         const char *name, void **out)
{
    size_t off, end;

    MM_REQUIRE(a->base && out, MM_ERR_STATE);
    MM_REQUIRE(align >= 16 && (align & (align - 1)) == 0, MM_ERR_STATE);
    off = mm_align_up(a->used, align);
    end = off + bytes;
    if (end > a->cap) {
        MM_LOGE("arena %s: %zu + %zu MiB exceeds %zu MiB (%s)",
                a->tag, a->used >> 20, bytes >> 20, a->cap >> 20,
                name ? name : "anon");
        return MM_ERR_NOMEM;
    }
    a->used = end;
    *out = (char *)a->base + off;
    if (a->n_slots == a->cap_slots) {
        a->cap_slots *= 2;
        a->slots = realloc(a->slots, a->cap_slots * sizeof(mm_slot));
        if (!a->slots)
            return MM_ERR_NOMEM;
    }
    a->slots[a->n_slots].off = off;
    a->slots[a->n_slots].bytes = bytes;
    a->slots[a->n_slots].name = name;
    a->slots[a->n_slots].pinned = 0;
    a->n_slots++;
    return MM_OK;
}

mm_status mm_arena_alloc_2d(mm_arena *a, size_t r, size_t c, size_t elem,
                            size_t align, const char *name, void **out)
{
    return mm_arena_alloc(a, r * c * elem, align, name, out);
}

/* Pin a slot so it survives reset(). The slot is identified by pointer. */
void mm_arena_pin(mm_arena *a, const void *p)
{
    ptrdiff_t off = (const char *)p - (const char *)a->base;
    for (uint32_t i = 0; i < a->n_slots; i++) {
        if (a->slots[i].off == (size_t)off) {
            a->slots[i].pinned = 1;
            return;
        }
    }
    MM_LOGW("arena %s: pin: pointer not a slot start", a->tag);
}

void *mm_arena_at(const mm_arena *a, size_t off)
{
    return (char *)a->base + off;
}

void *mm_arena_at_host(const mm_arena *a, size_t off)
{
    if (!a || !a->base || off > a->cap)
        return NULL;
    return (char *)(a->hbase ? a->hbase : a->base) + off;
}

void *mm_arena_host_ptr(const mm_arena *a, const void *p)
{
    ptrdiff_t off;

    if (!a || !a->base || !p)
        return NULL;
    off = (const char *)p - (const char *)a->base;
    if (off < 0 || (size_t)off > a->cap)
        return NULL;       /* not a pointer into this arena */
    return (char *)(a->hbase ? a->hbase : a->base) + off;
}

size_t mm_arena_used(const mm_arena *a)
{
    return a->used;
}

size_t mm_arena_free_bytes(const mm_arena *a)
{
    return a->cap - a->used;
}

/* ------------------------------------------------------------ reset */

mm_status mm_arena_mark(mm_arena *a)
{
    MM_REQUIRE(a->resettable, MM_ERR_STATE);
    a->mark = a->used;
    return MM_OK;
}

mm_status mm_arena_reset(mm_arena *a)
{
    size_t hi = a->mark;

    MM_REQUIRE(a->resettable, MM_ERR_STATE);
    /* Pinned slots keep their addresses: used must not fall below the end
     * of the highest pinned slot. */
    for (uint32_t i = 0; i < a->n_slots; i++) {
        if (a->slots[i].pinned) {
            size_t end = a->slots[i].off + a->slots[i].bytes;
            if (end > hi)
                hi = end;
        }
    }
    a->used = hi;
    /* Drop slot records past the new watermark (they are anonymous). */
    while (a->n_slots > 0 &&
           (a->slots[a->n_slots - 1].off + a->slots[a->n_slots - 1].bytes)
               > hi &&
           !a->slots[a->n_slots - 1].pinned)
        a->n_slots--;
    return MM_OK;
}

/* ------------------------------------------------------------ dump */

int mm_arena_dump(const mm_arena *a, char *buf, size_t n)
{
    int o = 0;
    o += snprintf(buf + o, n - (size_t)o,
                  "arena %s: cap %zu MiB, used %zu MiB, %u slots\n",
                  a->tag, a->cap >> 20, a->used >> 20, a->n_slots);
    for (uint32_t i = 0; i < a->n_slots && o < (int)n - 96; i++) {
        const mm_slot *s = &a->slots[i];
        snprintf(buf + o, n - (size_t)o, "  %7zu + %10zu  %s%s\n",
                 s->off >> 10, s->bytes >> 10,
                 s->pinned ? "[pin] " : "", s->name ? s->name : "(anon)");
        o = (int)strlen(buf + o);
    }
    return o;
}
