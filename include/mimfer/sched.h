/*
 * Scheduler: one CPU thread, round-based, no condition variables.
 *
 * A ROUND is exactly one of:
 *   PREFILL round  the queued prompt's next chunk runs through the
 *                  prefill plan for one free slot (async; no host sync
 *                  until the round that consumes output)
 *   DECODE round   all active slots run one decode step (batched in one
 *                  plan), then a single host sync — the tokens are needed
 *                  to advance state, so this is the ONLY sync per step
 *
 * Round selection is total order and dumb:
 *   1. if a prompt is queued and a slot is free          -> prefill round
 *   2. else if any slot is active                        -> decode round
 *   3. else                                              -> idle (200 us)
 *
 * All cleverness lives in the plans and kernels; the scheduler only moves
 * tokens between the host, the control buffer and the slot state. That
 * split is what keeps it testable: the CPU reference build exercises the
 * exact same scheduler against a tiny fake model.
 */

#ifdef __cplusplus
extern "C" {
#endif

#ifndef MIMFER_SCHED_H
#define MIMFER_SCHED_H

#include "mimfer.h"

typedef enum mm_rstate {
    RS_FREE = 0,     /* slot unused                                */
    RS_WAIT,         /* prompt queued, no slot yet                 */
    RS_PREFILL,      /* slot assigned, prefill in flight           */
    RS_DECODE,       /* generating                                 */
    RS_DONE,         /* finished (EOS, length or error)            */
} mm_rstate;

typedef struct mm_seq {
    int       state;
    uint32_t  len;        /* committed context tokens (prompt+gen)    */
    uint32_t  prefilled;  /* tokens that passed through the prefill   */
    uint32_t  max_out;    /* generation budget                        */
    uint32_t  gen;        /* tokens generated so far                  */
    /* prompt (host-owned; submitted by the caller) */
    const uint32_t *ptoks;
    uint32_t  plen;
    int       finished;   /* EOS / length / error reason set          */
} mm_seq;

typedef struct mm_sched {
    uint32_t  n;                     /* = cfg.concurrency              */
    mm_seq    slots[MM_MAX_CONCURRENCY];
    /* waiting queue (capacity n+1, so up to n prompts can wait).
     * Prompts are host-owned; the submitter keeps them alive until the
     * slot is done. */
    const uint32_t *q_tok[MM_MAX_CONCURRENCY + 1];
    uint32_t        q_len[MM_MAX_CONCURRENCY + 1];
    uint32_t        q_maxout[MM_MAX_CONCURRENCY + 1];
    uint32_t        q_head, q_tail;  /* ring indices */
} mm_sched;

void      mm_sched_init(mm_sched *s, uint32_t n);
/* Queue a prompt. Returns MM_ERR_CAPACITY when the queue is full. */
mm_status mm_sched_submit(mm_sched *s, const uint32_t *toks, uint32_t n,
                          uint32_t max_out);
/* Decide the next round. Fills *slot (prefill) or *n_decode.
 * Returns 0 when idle. */
int       mm_sched_pick_round(mm_sched *s, int *slot, uint32_t *n_decode);
/* Assign the queue head to a free slot (called by the engine). */
mm_status mm_sched_assign(mm_sched *s, int slot);
/* Mark a slot finished; frees the prompt. */
mm_status mm_sched_done(mm_sched *s, int slot);
uint32_t  mm_sched_active(const mm_sched *s);
uint32_t  mm_sched_queued(const mm_sched *s);

#endif /* MIMFER_SCHED_H */
#ifdef __cplusplus
}
#endif
