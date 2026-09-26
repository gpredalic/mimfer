/*
 * Telemetry: fixed ring + counters. Single producer (scheduler thread);
 * readers are the CLI at process end. No locks by construction.
 */
#include "mimfer/telemetry.h"
#include "mimfer/mimfer.h"
#include <stdio.h>
#include <string.h>

void mm_tel_init(mm_tel *t)
{
    memset(t, 0, sizeof *t);
}

void mm_tel_round(mm_tel *t, uint8_t kind, uint32_t tokens, uint32_t kv,
                  uint32_t cache_hits, uint64_t t_start, uint64_t t_end)
{
    mm_tel_sample *s = &t->ring[t->head];
    t->head = (t->head + 1) & (MM_TEL_RING - 1);

    s->t_ms = t_start;
    s->t_end = t_end;
    s->round = (uint32_t)(t->rounds);
    s->tokens = tokens;
    s->kv_used = kv;
    s->cache_hits = cache_hits;
    s->kind = kind;

    t->rounds++;
    t->n_tokens += tokens;
    if (kind == TEL_PREFILL) t->n_prefill++;
    if (kind == TEL_DECODE)  t->n_decode++;
    t->cache_hits += cache_hits;
    if (t->t_first_ms == 0) t->t_first_ms = t_start;
    t->t_end_ms = t_end;
}

/* Percentile over decode round durations (itl). Static 1024-element
 * workspace; the ring holds 4096 so we take the first 1024 decode samples.
 * Cold path (process end), so a plain insertion sort is fine. */
static uint64_t percent_decode_ms(const mm_tel *t, double p)
{
    static uint64_t ws[1024];
    uint64_t vals = 0, i, k;

    for (i = 0; i < MM_TEL_RING && vals < 1024; i++) {
        const mm_tel_sample *s = &t->ring[i];
        if (s->kind == TEL_DECODE && s->t_end > s->t_ms)
            ws[vals++] = s->t_end - s->t_ms;
    }
    if (vals == 0)
        return 0;
    k = (uint64_t)(p * (double)vals);
    if (k < 1) k = 1;
    if (k > vals) k = vals;
    for (i = 1; i < vals; i++) {
        uint64_t v = ws[i], j = i;
        while (j > 0 && ws[j - 1] > v) {
            ws[j] = ws[j - 1];
            j--;
        }
        ws[j] = v;
    }
    return ws[k - 1];
}

int mm_tel_summary(const mm_tel *t, char *buf, size_t n)
{
    uint64_t wall, p50;
    double tps;

    if (t->rounds == 0)
        return snprintf(buf, n, "no rounds");
    wall = t->t_end_ms - t->t_first_ms;
    if (wall == 0) wall = 1;
    tps = (double)t->n_tokens / (wall / 1000.0);
    p50 = percent_decode_ms(t, 0.50);
    return snprintf(buf, n,
                    "%llu tok/s | %llu decode rounds | itl p50 %llu ms | "
                    "kv rounds %llu",
                    (unsigned long long)tps, (unsigned long long)t->n_decode,
                    (unsigned long long)p50, (unsigned long long)wall);
}

int mm_tel_report(const mm_tel *t, char *buf, size_t n)
{
    int o = 0, i;
    uint64_t hits = t->cache_hits, misses = t->cache_misses;
    double rate = hits + misses ? 100.0 * (double)hits / (double)(hits + misses) : 0.0;
    char line[256];

    o += snprintf(buf + o, n - (size_t)o,
                  "rounds %llu (prefill %llu, decode %llu)  tokens %llu\n",
                  (unsigned long long)t->rounds,
                  (unsigned long long)t->n_prefill,
                  (unsigned long long)t->n_decode,
                  (unsigned long long)t->n_tokens);
    o += snprintf(buf + o, n - (size_t)o,
                  "prefix cache: %llu hits / %llu misses (%.1f%%)\n",
                  (unsigned long long)hits, (unsigned long long)misses, rate);
    o += snprintf(buf + o, n - (size_t)o,
                  "last 32 rounds:\n  %-8s %-6s %-9s %-9s %s\n",
                  "kind", "toks", "ms", "kv_used", "cache_hits");
    for (i = 0; i < 32; i++) {
        uint32_t idx = (t->head + MM_TEL_RING - 1 - i) & (MM_TEL_RING - 1);
        const mm_tel_sample *s = &t->ring[idx];
        const char *k = s->kind == TEL_PREFILL ? "prefill"
                        : s->kind == TEL_VERIFY ? "verify" : "decode";
        if (s->t_ms == 0 && s->t_end == 0)
            break;
        snprintf(line, sizeof line, "  %-8s %-6u %-9llu %-9u %u\n",
                 k, s->tokens,
                 (unsigned long long)(s->t_end - s->t_ms),
                 s->kv_used, s->cache_hits);
        o += snprintf(buf + o, n - (size_t)o, "%s", line);
    }
    return o;
}
