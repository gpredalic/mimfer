/*
 * Tensor views and numeric formats.
 *
 * A mm_tensor is a zero-copy view: it points into an arena allocation and
 * never frees memory. All tensors that outlive a kernel launch live in the
 * weight arena (read-only, resident) or the activation arena (reused each
 * step). There is no per-tensor allocation: that is the whole memory story.
 *
 * Numeric formats are the NVIDIA NVFP4 family:
 *   MM_DT_FP4E2M1  4-bit float, e2m1: 1 sign, 2 exponent, 1 mantissa.
 *                  Packed 2 per byte (low nibble = even index).
 *                  Values: {0, .5, 1, 1.5, 2, 3, 4, 6} x {+,-}
 *   MM_DT_FP8E4M3  8-bit float, e4m3: 1 sign, 4 exponent (bias 7), 3 mant.
 *                  No inf; NaN only at 0x7f/0xff; max |448|.
 *                  Used for NVFP4 block scales (one per 16 fp4 elements).
 *
 * NVFP4 weight tensor layout (see docs/design.md "Weight memory layout"):
 *   data:  [rows][cols/2] bytes, row-major, col pairs packed low/high
 *   scale: [rows][cols/16] e4m3, same row order
 * The kernel that consumes a weight tile dequantizes in registers; no
 * materialized bf16 weight copy exists (it would not fit in 24 GiB).
 */

#ifdef __cplusplus
extern "C" {
#endif

#ifndef MIMFER_TENSOR_H
#define MIMFER_TENSOR_H

#include "mimfer.h"

typedef enum mm_dtype {
    MM_DT_U8,
    MM_DT_I32,
    MM_DT_I64,
    MM_DT_F32,
    MM_DT_BF16,
    MM_DT_FP4E2M1,
    MM_DT_FP8E4M3,
    MM_DT_N
} mm_dtype;

const char *mm_dtype_str(mm_dtype dt);
size_t      mm_dtype_bytes(mm_dtype dt);

/* Zero-copy view. `bytes` is the allocation size, not ndim*shape (the last
 * dimension may be packed, e.g. fp4). Keep this struct cache-line small:
 * it is passed by value everywhere. */
typedef struct mm_tensor {
    const void  *data;
    size_t       bytes;
    mm_dtype     dtype;
    int          ndim;
    uint32_t     shape[4];
} mm_tensor;

static inline size_t mm_tensor_elems(const mm_tensor *t)
{
    size_t n = 1;
    for (int i = 0; i < t->ndim; i++) n *= t->shape[i];
    return n;
}

static inline mm_tensor mm_t_view(const void *p, mm_dtype dt, int ndim,
                                  uint32_t d0, uint32_t d1, uint32_t d2,
                                  uint32_t d3, size_t bytes)
{
    mm_tensor t = { p, bytes, dt, ndim, { d0, d1, d2, d3 } };
    return t;
}

/* ------------------------------------------------------- numeric decoders
 * These are the single source of truth for the float formats. Host code
 * (tests, packer, detokenizer) and device kernels (via a __device__ copy in
 * kernels/fp4.cuh) must agree; tests enforce byte-for-byte agreement. */

#if defined(__CUDACC__)
#define MM_HOSTDEV __host__ __device__
#else
#define MM_HOSTDEV
#endif

/* e2m1 magnitude table, indexed by the 3-bit magnitude code (0..7).
 * code = bits[2:0] with bit2 = exp_hi, bit1 = exp_lo, bit0 = mantissa:
 *   exp 0: 0.m  -> 0, 0.5
 *   exp 1: 1.m  -> 1, 1.5
 *   exp 2: 1.m+1 -> 2, 3
 *   exp 3: 1.m+1 -> 4, 6 */
MM_HOSTDEV static inline float mm_fp4_decode(unsigned nibble)
{
    static const float mag[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
    float v = mag[nibble & 7u];
    return (nibble & 8u) ? -v : v;
}

/* e4m3 decode. Bit layout: s eeee mmm, bias 7. 0x7f/0xff are NaN -> 0. */
MM_HOSTDEV static inline float mm_fp8e4m3_decode(unsigned char b)
{
    if (b == 0x7f || b == 0xff) return 0.0f;
    int    sgn  = b & 0x80;
    int    exp  = (b >> 3) & 0x0f;
    int    man  = b & 0x07;
    float  mag  = (float)man / 8.0f;
    if (exp == 0) {
        /* subnormal: 0.mmm * 2^-6 */
        mag *= 0.015625f; /* 1/64 */
    } else {
        /* normal: 1.mmm * 2^(e-7) */
        mag = 1.0f + mag;
        float p = 1.0f;
        int   e = exp - 7;
        if (e >= 0) { while (e-- > 0) p *= 2.0f; }
        else        { while (e++ < 0) p *= 0.5f; }
        mag *= p;
    }
    return sgn ? -mag : mag;
}

/* Weight views need both payload and scale pointers. */
typedef struct mm_wt {
    const void *data;   /* fp4 payload, [rows][cols/2]                       */
    const void *scale;  /* e4m3 scales, [rows][cols/16]                      */
    uint32_t    rows;   /* output features                                   */
    uint32_t    cols;   /* input features, always a multiple of 16          */
} mm_wt;

/* NVFP4 weight accessors. `row`/`col` index the logical [rows x cols]
 * matrix; `rows` is the output feature, `cols` the input feature. */
MM_HOSTDEV static inline float mm_fp4_wget(const mm_wt *w, int row, int col)
{
    const unsigned char *p = (const unsigned char *)w->data;
    unsigned byte  = p[row * (w->cols / 2) + (col >> 1)];
    unsigned nib   = (col & 1) ? (byte >> 4) : (byte & 0x0f);
    /* group scale: one e4m3 per 16 columns */
    const unsigned char *s = (const unsigned char *)w->scale;
    float sc = mm_fp8e4m3_decode(s[row * (w->cols / 16) + (col >> 4)]);
    return mm_fp4_decode(nib) * sc;
}

/* bf16 (bfloat16): 1 sign, 8 exponent (bias 127), 7 mantissa — the upper
 * 16 bits of an f32. All activation kernels in the CPU reference build
 * read bf16 tensors and accumulate in f32. Round-to-nearest-even on the
 * f32 -> bf16 path (the low 16 mantissa bits are dropped). Union type-pun
 * so no <string.h> is pulled into this header. */
MM_HOSTDEV static inline uint16_t mm_bf16_from_f32(float f)
{
    union { float f; uint32_t u; } v;
    v.f = f;
    v.u += 0x7fffu + ((v.u >> 16) & 1u);   /* RNE: round, ties to even */
    return (uint16_t)(v.u >> 16);
}

MM_HOSTDEV static inline float mm_f32_from_bf16(uint16_t b)
{
    union { float f; uint32_t u; } v;
    v.u = ((uint32_t)b) << 16;
    return v.f;
}

#endif /* MIMFER_TENSOR_H */
#ifdef __cplusplus
}
#endif
