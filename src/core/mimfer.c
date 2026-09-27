/*
 * Core: status strings, logging, dtype metadata.
 *
 * This file is the only place in the codebase that formats a log line.
 * Kernels never log; the scheduler is the only hot-path caller (one line
 * per prompt at most).
 */
#include "mimfer/mimfer.h"
#include "mimfer/tensor.h"
#include <stdio.h>
#include <stdarg.h>
#include <string.h>

mm_log_level g_mm_log_level = MM_LOG_WARN;

const char *mm_status_str(mm_status s)
{
    switch (s) {
    case MM_OK:            return "ok";
    case MM_ERR_NOMEM:     return "out of memory";
    case MM_ERR_IO:        return "i/o failure";
    case MM_ERR_CORRUPT:   return "checksum or magic mismatch";
    case MM_ERR_SHAPE:     return "bad shape or stride";
    case MM_ERR_CUDA:      return "cuda runtime failure";
    case MM_ERR_DEVICE:    return "device not the target profile";
    case MM_ERR_UNSUPPORTED: return "unsupported on this build";
    case MM_ERR_RANGE:     return "scalar out of range";
    case MM_ERR_STATE:     return "bad object state";
    case MM_ERR_BUSY:      return "resource held by another owner";
    case MM_ERR_TIMEOUT:   return "host-side wait exceeded its budget";
    case MM_ERR_CAPACITY:  return "pool exhausted, no reusable prefix space";
    case MM_ERR_EOF:       return "end of input";
    default:               return "unknown status";
    }
}

void mm_log(mm_log_level lvl, const char *file, int line, const char *fmt,
            ...)
{
    static const char tag[4] = { 'E', 'W', 'I', 'D' };
    const char *base;
    va_list ap;

    /* Severity order: MM_LOG_ERR (0) is the most severe, MM_LOG_DBG (3)
     * the least. Print when the message is at least as severe as the
     * configured level (i.e. its numeric level does not exceed it);
     * dropping the message on `lvl > g_mm_log_level` is the only reading
     * under which the default (MM_LOG_WARN) still shows errors. The
     * previous `lvl < g_mm_log_level` inverted the severity and silently
     * suppressed every MM_LOGE at the default level. */
    if ((int)lvl < 0 || (int)lvl > MM_LOG_DBG || lvl > g_mm_log_level)
        return;
    base = strrchr(file, '/');
    base = base ? base + 1 : file;
    fprintf(stderr, "%c %s:%d: ", tag[lvl], base, line);
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
}

/* ------------------------------------------------------------ dtypes */

const char *mm_dtype_str(mm_dtype dt)
{
    switch (dt) {
    case MM_DT_U8:       return "u8";
    case MM_DT_I32:      return "i32";
    case MM_DT_I64:      return "i64";
    case MM_DT_F32:      return "f32";
    case MM_DT_BF16:     return "bf16";
    case MM_DT_FP4E2M1:  return "nvfp4";
    case MM_DT_FP8E4M3:  return "e4m3";
    default:             return "?";
    }
}

/* Nominal bytes per element. FP4 is packed 2/byte: use /2 on the element
 * count for byte math (the layout is defined in tensor.h). */
size_t mm_dtype_bytes(mm_dtype dt)
{
    switch (dt) {
    case MM_DT_U8:       return 1;
    case MM_DT_I32:      return 4;
    case MM_DT_I64:      return 8;
    case MM_DT_F32:      return 4;
    case MM_DT_BF16:     return 2;
    case MM_DT_FP4E2M1:  return 0;   /* packed; see mm_wt layout */
    case MM_DT_FP8E4M3:  return 1;
    default:             return 0;
    }
}
