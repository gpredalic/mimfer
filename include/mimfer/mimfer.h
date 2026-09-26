/*
 * mimfer — a small inference appliance.
 *
 * One NVIDIA RTX PRO 4000 Blackwell (sm_120a, 24 GiB GDDR7).
 * One Linux process. One CUDA device. One resident model:
 * Qwen3.8-27B (Qwen3_5 text core), NVFP4 quantized.
 *
 * This header is the only contract every subsystem may depend on. It
 * carries the error model (mm_status), logging (mm_log), build constants
 * (limits, feature gates) and small utilities (min/max, align, timing).
 *
 * No C++, no exceptions, no third-party headers. CUDA enters only through
 * <cuda_runtime.h> inside src/cuda/ and src/kernels/.
 */
#ifndef MIMFER_H
#define MIMFER_H

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L   /* clock_gettime under -std=c11 */
#endif

#include <stdint.h>
#include <stddef.h>
#include <time.h>

#define MIMFER_VERSION  "0.1.0"
#define MIMFER_NAME     "mimfer"

/* Arena alignment granules. */
#define MM_PAGE_4K      ((size_t)4096)
#define MM_PAGE_2M      ((size_t)(2 * 1024 * 1024))

/* Hard engine limits. Compile-time: the appliance is built for one
 * workload shape and refuses to be reconfigured into another. */
#define MM_MAX_SEQ          262144   /* model native max_position_embeddings */
#define MM_MAX_CONCURRENCY  4        /* active decode slots, fixed at start  */
#define MM_MAX_PREFILL_CH  4096      /* prefill chunk length (tokens)        */
#define MM_KV_BLOCK_TOK     64       /* tokens per KV page                   */
#define MM_MAX_KV_BLOCKS   (MM_MAX_SEQ / MM_KV_BLOCK_TOK)
#define MM_DRAFT_MAX        8        /* speculative draft window             */
#define MM_TEL_RING         4096     /* telemetry ring entries               */

/* ---------------------------------------------------------------- status */

typedef enum mm_status {
    MM_OK = 0,
    MM_ERR_NOMEM,        /* arena exhausted / cudaMalloc failed              */
    MM_ERR_IO,           /* file read/write failed                           */
    MM_ERR_CORRUPT,      /* checksum or magic mismatch                       */
    MM_ERR_SHAPE,        /* tensor shape/stride invalid                      */
    MM_ERR_CUDA,         /* driver/runtime API failure                       */
    MM_ERR_DEVICE,       /* device not the target profile                    */
    MM_ERR_UNSUPPORTED,  /* feature not implemented on this build            */
    MM_ERR_RANGE,        /* scalar out of range                              */
    MM_ERR_STATE,       /* object not in the required state                 */
    MM_ERR_BUSY,         /* resource held by another owner                   */
    MM_ERR_TIMEOUT,      /* host-side wait exceeded its budget               */
    MM_ERR_CAPACITY,     /* KV/state pool full, no reusable prefix space     */
    MM_ERR_EOF,          /* input ended (not an error)                       */
    MM_ERR_LAST
} mm_status;

const char *mm_status_str(mm_status s);

/* ---------------------------------------------------------------- logging */

typedef enum mm_log_level {
    MM_LOG_ERR = 0,
    MM_LOG_WARN,
    MM_LOG_INFO,
    MM_LOG_DBG
} mm_log_level;

extern mm_log_level g_mm_log_level;

/* mm_log writes "level file:line: message\n" to stderr. Formatted on the
 * host only; kernels never call it. The level filter is set once at
 * startup (CLI flag), never at runtime. */
void mm_log(mm_log_level lvl, const char *file, int line, const char *fmt,
            ...) __attribute__((format(printf, 4, 5)));

#define MM_LOGE(...) mm_log(MM_LOG_ERR,  __FILE__, __LINE__, __VA_ARGS__)
#define MM_LOGW(...) mm_log(MM_LOG_WARN, __FILE__, __LINE__, __VA_ARGS__)
#define MM_LOGI(...) mm_log(MM_LOG_INFO, __FILE__, __LINE__, __VA_ARGS__)
#define MM_LOGD(...) mm_log(MM_LOG_DBG,  __FILE__, __LINE__, __VA_ARGS__)

/* --------------------------------------------------------------- errors */

/* Record status into a pointer (NULL = ignore) and return it. */
#define MM_SET_ERR(st, code) ((st) != (code) ? (code) : (st))

/* Evaluate e; on non-MM_OK log once and return the status from the
 * enclosing function. This is the entire error-handling policy: no
 * exceptions, no unwinding, no per-call branching beyond this. */
#define MM_CHECK(expr)                                              \
    do {                                                            \
        mm_status _s = (expr);                                      \
        if (_s != MM_OK) {                                          \
            MM_LOGE("%s:%d: %s: %s", __FILE__, __LINE__, #expr,     \
                    mm_status_str(_s));                             \
            return _s;                                              \
        }                                                           \
    } while (0)

/* Assert a condition that must hold by construction; log + return err. */
#define MM_REQUIRE(cond, code)                                      \
    do {                                                            \
        if (!(cond)) {                                              \
            MM_LOGE("%s:%d: required: %s", __FILE__, __LINE__,      \
                    #cond);                                         \
            return code;                                            \
        }                                                           \
    } while (0)

/* ------------------------------------------------------------------ misc */

static inline uint64_t mm_min64(uint64_t a, uint64_t b) { return a < b ? a : b; }
static inline uint64_t mm_max64(uint64_t a, uint64_t b) { return a > b ? a : b; }
static inline size_t   mm_align_up(size_t v, size_t a) { return (v + a - 1) & ~(a - 1); }

/* Monotonic clock, milliseconds. Single source of timing for telemetry. */
static inline uint64_t mm_now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000ull + (uint64_t)(ts.tv_nsec / 1000000ull);
}

/* FNV-1a 64. Prefix hashing, artifact checksums, RNG seeding.
 * Deliberately slow: a correctness primitive, never on the hot path. */
static inline uint64_t mm_fnv1a64(const void *data, size_t n)
{
    const unsigned char *p = (const unsigned char *)data;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

/* Splitmix64 finalizer — turns a 64-bit seed into a well-mixed RNG state.
 * Used by the deterministic RNG (see sampling/rng). */
static inline uint64_t mm_splitmix64(uint64_t *state)
{
    uint64_t z = (*state += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

#endif /* MIMFER_H */

