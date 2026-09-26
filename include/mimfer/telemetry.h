/*
 * Telemetry: one fixed ring + counters. No allocation at runtime; every
 * round appends one sample. `mimfer stat` prints the table; the run loop
 * prints a one-line summary on exit.
 *
 * Metrics and their definitions (see benchmark_plan.md):
 *   ttft        submit -> first generated token (per prompt)
 *   itl         wall time of a decode round / tokens produced
 *   tok/s       tokens / wall time over the run
 *   cache hit   prefix-block reuse rate (blocks hit / blocks requested)
 *   kv occ      KV pool occupancy at the end of the round
 *
 * The ring is single-producer (the scheduler thread); readers are the CLI
 * at process end. No locks by construction.
 */

#ifdef __cplusplus
extern "C" {
#endif

#ifndef MIMFER_TELEMETRY_H
#define MIMFER_TELEMETRY_H

#include "mimfer.h"

typedef enum { TEL_PREFILL = 0, TEL_DECODE, TEL_VERIFY } mm_tel_kind;

typedef struct mm_tel_sample {
    uint64_t t_ms;       /* round start                                  */
    uint64_t t_end;      /* round end (after the only host sync)         */
    uint32_t round;      /* global tick                                  */
    uint16_t tokens;     /* tokens produced by this round                */
    uint32_t kv_used;    /* pool occupancy (blocks)                      */
    uint32_t cache_hits; /* prefix blocks reused this round              */
    uint8_t  kind;       /* mm_tel_kind                                  */
} mm_tel_sample;

typedef struct mm_tel {
    mm_tel_sample ring[MM_TEL_RING];
    uint32_t      head;
    uint64_t      rounds, n_tokens, n_prefill, n_decode;
    uint64_t      cache_hits, cache_misses;
    uint64_t      t_first_ms, t_end_ms;   /* for the one-line summary    */
} mm_tel;

void mm_tel_init(mm_tel *t);
void mm_tel_round(mm_tel *t, uint8_t kind, uint32_t tokens, uint32_t kv,
                  uint32_t cache_hits, uint64_t t_start, uint64_t t_end);
/* Human-readable table into buf; returns bytes written or -1. */
int  mm_tel_report(const mm_tel *t, char *buf, size_t n);
/* One-line summary: "123.4 tok/s, ttft 412 ms, itl p50 16.9 ms" */
int  mm_tel_summary(const mm_tel *t, char *buf, size_t n);

#endif /* MIMFER_TELEMETRY_H */
#ifdef __cplusplus
}
#endif
