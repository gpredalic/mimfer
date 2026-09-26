/*
 * Round scheduler: queue, slot table, round selection (see sched.h).
 * Pure host state — no CUDA, no allocation beyond the fixed arrays.
 */
#include "mimfer/sched.h"
#include <string.h>

#define QCAP (MM_MAX_CONCURRENCY + 1)

void mm_sched_init(mm_sched *s, uint32_t n)
{
    uint32_t i;
    memset(s, 0, sizeof *s);
    s->n = n;
    for (i = 0; i < n; i++)
        s->slots[i].state = RS_FREE;
}

uint32_t mm_sched_queued(const mm_sched *s)
{
    return (uint32_t)((s->q_tail - s->q_head) % QCAP);
}

uint32_t mm_sched_active(const mm_sched *s)
{
    uint32_t i, n = 0;
    for (i = 0; i < s->n; i++)
        if (s->slots[i].state == RS_DECODE)
            n++;
    return n;
}

mm_status mm_sched_submit(mm_sched *s, const uint32_t *toks, uint32_t n,
                          uint32_t max_out)
{
    MM_REQUIRE(toks && n > 0, MM_ERR_STATE);
    if (mm_sched_queued(s) >= s->n)
        return MM_ERR_CAPACITY;
    s->q_tok[s->q_tail] = toks;
    s->q_len[s->q_tail] = n;
    s->q_maxout[s->q_tail] = max_out;
    s->q_tail = (s->q_tail + 1) % QCAP;
    return MM_OK;
}

/* Total order (see sched.h): 1 = prefill round, 2 = decode round, 0 = idle. */
int mm_sched_pick_round(mm_sched *s, int *slot, uint32_t *n_decode)
{
    uint32_t i;
    *slot = -1;
    *n_decode = 0;

    if (mm_sched_queued(s) > 0) {
        for (i = 0; i < s->n; i++) {
            if (s->slots[i].state == RS_FREE) {
                *slot = (int)i;
                return 1;
            }
        }
    }
    for (i = 0; i < s->n; i++) {
        if (s->slots[i].state == RS_DECODE)
            (*n_decode)++;
    }
    if (*n_decode)
        return 2;
    return 0;
}

/* Move the queue head into a free slot. */
mm_status mm_sched_assign(mm_sched *s, int slot)
{
    mm_seq *q = &s->slots[slot];
    MM_REQUIRE(q->state == RS_FREE && mm_sched_queued(s) > 0, MM_ERR_STATE);
    q->ptoks = s->q_tok[s->q_head];
    q->plen = s->q_len[s->q_head];
    q->max_out = s->q_maxout[s->q_head];
    s->q_head = (s->q_head + 1) % QCAP;
    q->len = 0;
    q->prefilled = 0;
    q->gen = 0;
    q->finished = 0;
    q->state = RS_PREFILL;
    return MM_OK;
}

mm_status mm_sched_done(mm_sched *s, int slot)
{
    mm_seq *q = &s->slots[slot];
    MM_REQUIRE(slot >= 0 && slot < (int)s->n, MM_ERR_STATE);
    q->ptoks = NULL;
    q->state = RS_DONE;
    /* Slots are reusable: the engine treats RS_DONE as free. */
    q->state = RS_FREE;
    return MM_OK;
}
