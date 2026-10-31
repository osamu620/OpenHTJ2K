// Copyright (c) 2019 - 2026, Osamu Watanabe
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
//    modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
//    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
//    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
//    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
//    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once
#include <cstdint>
#include <cfloat>
#include <cmath>
#include "utils.hpp"

#define ALPHA_R 0.299
#define ALPHA_B 0.114
#define ALPHA_RB (ALPHA_R + ALPHA_B)
#define ALPHA_G (1 - ALPHA_RB)
#define CR_FACT_R (2 * (1 - ALPHA_R))
#define CB_FACT_B (2 * (1 - ALPHA_B))
#define CR_FACT_G (2 * ALPHA_R * (1 - ALPHA_R) / ALPHA_G)
#define CB_FACT_G (2 * ALPHA_B * (1 - ALPHA_B) / ALPHA_G)

typedef void (*cvt_color_func)(int32_t *, int32_t *, int32_t *, uint32_t, uint32_t);
// Float-domain inverse color transform (used in decode to avoid float→int32 intermediate copy).
// sp0/sp1/sp2 point to float sample buffers; stride = row stride in floats.
typedef void (*cvt_color_float_func)(float *, float *, float *, uint32_t, uint32_t, uint32_t);
// Fused int32→float + forward color transform (used in encode to avoid int32 intermediate copy).
// sp0/sp1/sp2: int32 input; dp0/dp1/dp2: float output; all share the same row stride.
typedef void (*cvt_color_i32_to_f_func)(const int32_t *, const int32_t *, const int32_t *, float *,
                                        float *, float *, uint32_t, uint32_t, uint32_t);

// Per-component parameters for the float→int32 finalization step.
struct FinalizeParams {
  int16_t ds;      // FRACBITS - bit depth for an irreversible transform, 0 for a reversible one
  float scale;     // 2^-ds, one output LSB in Q(FRACBITS) units (finalize_scale(ds))
  int32_t dc;      // DC offset added after rounding (e.g. 128 for 8-bit unsigned)
  int32_t maxval;  // clamp upper bound
  int32_t minval;  // clamp lower bound
};

// Lossy finalize rounding.  An irreversible sample v is in Q(FRACBITS) units, so one
// output LSB is 2^ds of them, ds = FRACBITS - bit depth (negative above 13 bits).  The
// output is the nearest integer with ties rounded up, floor(v * 2^-ds + 0.5): for an
// integer v and ds > 0 that is exactly (v + 2^(ds-1)) >> ds, the fixed-point
// convention, and the same formula serves every sign of ds, so one loop covers every
// bit depth.  A reversible transform leaves integer-valued floats, which scale 1
// reproduces exactly.  The vector forms below compute the same thing (multiply, add
// 0.5, floor, convert); keep every variant identical.
inline float finalize_scale(int32_t ds) { return std::ldexp(1.0f, -ds); }
inline int32_t finalize_round(float v, float scale) {
  return static_cast<int32_t>(std::floor(v * scale + 0.5f));
}
#if defined(OPENHTJ2K_TRY_AVX2) && defined(__AVX512F__)
static inline __m512i finalize_round_avx512(__m512 v, __m512 vscale, __m512 vhalf) {
  return _mm512_cvt_roundps_epi32(_mm512_fmadd_ps(v, vscale, vhalf),
                                  _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
}
#endif
#if defined(OPENHTJ2K_TRY_AVX2) && defined(__AVX2__)
static inline __m256i finalize_round_avx2(__m256 v, __m256 vscale, __m256 vhalf) {
  return _mm256_cvttps_epi32(_mm256_floor_ps(_mm256_add_ps(_mm256_mul_ps(v, vscale), vhalf)));
}
#endif
#if defined(OPENHTJ2K_ENABLE_ARM_NEON) || defined(__ARM_NEON) || defined(__ARM_NEON__)
  #include <arm_neon.h>
static inline int32x4_t finalize_round_neon(float32x4_t v, float32x4_t vscale, float32x4_t vhalf) {
  #if defined(__aarch64__) || defined(_M_ARM64)
  return vcvtmq_s32_f32(vfmaq_f32(vhalf, v, vscale));
  #else
  // 32-bit ARM NEON (ARMv7) has neither FMA nor a round-down conversion.  vscale is
  // a power of two, so v * vscale is exact and a separate multiply-add rounds like
  // the FMA; floor() is the truncation minus 1 wherever it lies above the value.
  const float32x4_t x = vmlaq_f32(vhalf, v, vscale);
  const int32x4_t t   = vcvtq_s32_f32(x);
  return vaddq_s32(t, vreinterpretq_s32_u32(vcgtq_f32(vcvtq_f32_s32(t), x)));
  #endif
}
#endif
#if defined(OPENHTJ2K_ENABLE_WASM_SIMD)
  #include <wasm_simd128.h>
static inline v128_t finalize_round_wasm(v128_t v, v128_t vscale, v128_t vhalf) {
  return wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_floor(wasm_f32x4_add(wasm_f32x4_mul(v, vscale), vhalf)));
}
#endif
#if defined(OPENHTJ2K_ENABLE_SSE2)
// SSE2 has no floor: truncate, convert back and step down where that went up.  Raising
// the input to -2^31 first keeps every result out of int32 range (and NaN, for which
// maxps returns its second operand) at INT32_MIN, which the scalar conversion
// (cvttss2si) gives too, instead of stepping it down to INT32_MAX.
static inline __m128i finalize_round_sse2(__m128 v, __m128 vscale, __m128 vhalf) {
  const __m128 x  = _mm_max_ps(_mm_add_ps(_mm_mul_ps(v, vscale), vhalf), _mm_set1_ps(-2147483648.0f));
  const __m128i t = _mm_cvttps_epi32(x);
  return _mm_add_epi32(t, _mm_castps_si128(_mm_cmpgt_ps(_mm_cvtepi32_ps(t), x)));
}
// min(max(x, lo), hi) per int32 lane; SSE2 has no pmaxsd/pminsd, so select by mask.
static inline __m128i clamp_epi32_sse2(__m128i x, __m128i lo, __m128i hi) {
  __m128i m = _mm_cmplt_epi32(x, lo);
  x         = _mm_or_si128(_mm_and_si128(m, lo), _mm_andnot_si128(m, x));
  m         = _mm_cmpgt_epi32(x, hi);
  return _mm_or_si128(_mm_and_si128(m, hi), _mm_andnot_si128(m, x));
}
#endif

// Fused inverse color transform (float-domain) + float→int32 finalize.
// Reads Y/Cb/Cr from ring buffer (read-only), applies inverse MCT, applies per-component
// finalize (shift + DC_OFFSET + clamp), and writes R/G/B as int32.
// fp[3]: finalize params for component 0 (R), 1 (G), 2 (B).
typedef void (*fused_mct_finalize_func)(const float *, const float *, const float *, int32_t *,
                                        int32_t *, int32_t *, uint32_t, const FinalizeParams *);
#if defined(OPENHTJ2K_TRY_AVX2) && defined(__AVX512F__)
void cvt_rgb_to_ycbcr_rev_avx512(int32_t *sp0, int32_t *sp1, int32_t *sp2, uint32_t width, uint32_t height);
void cvt_rgb_to_ycbcr_irrev_avx512(int32_t *sp0, int32_t *sp1, int32_t *sp2, uint32_t width, uint32_t height);
void cvt_ycbcr_to_rgb_rev_avx512(int32_t *sp0, int32_t *sp1, int32_t *sp2, uint32_t width, uint32_t height);
void cvt_ycbcr_to_rgb_irrev_avx512(int32_t *sp0, int32_t *sp1, int32_t *sp2, uint32_t width, uint32_t height);
void cvt_ycbcr_to_rgb_rev_float_avx512(float *sp0, float *sp1, float *sp2, uint32_t width, uint32_t height, uint32_t stride);
void cvt_ycbcr_to_rgb_irrev_float_avx512(float *sp0, float *sp1, float *sp2, uint32_t width, uint32_t height, uint32_t stride);
void cvt_rgb_to_ycbcr_rev_float_avx512(const int32_t *sp0, const int32_t *sp1, const int32_t *sp2,
                                       float *dp0, float *dp1, float *dp2,
                                       uint32_t width, uint32_t height, uint32_t stride);
void cvt_rgb_to_ycbcr_irrev_float_avx512(const int32_t *sp0, const int32_t *sp1, const int32_t *sp2,
                                         float *dp0, float *dp1, float *dp2,
                                         uint32_t width, uint32_t height, uint32_t stride);
void fused_ycbcr_irrev_to_rgb_i32_avx512(const float *y, const float *cb, const float *cr,
                                          int32_t *r, int32_t *g, int32_t *b, uint32_t width,
                                          const FinalizeParams *fp);
void fused_ycbcr_rev_to_rgb_i32_avx512(const float *y, const float *cb, const float *cr,
                                        int32_t *r, int32_t *g, int32_t *b, uint32_t width,
                                        const FinalizeParams *fp);
#elif defined(OPENHTJ2K_TRY_AVX2) && defined(__AVX2__)
/**
 * @brief Forward reversible color transform (RCT) with AVX2 intrinsics
 * @param sp0 pointer to Red samples (shall be aligned and multiple of 8 samples)
 * @param sp1 pointer to Green samples (shall be aligned and multiple of 8 samples)
 * @param sp2 pointer to Blue samples (shall be aligned and multiple of 8 samples)
 * @param width original width (may not be multiple of 8)
 * @param height original height
 */
void cvt_rgb_to_ycbcr_rev_avx2(int32_t *sp0, int32_t *sp1, int32_t *sp2, uint32_t width, uint32_t height);
/**
 * @brief Forward irreversible color transform (ICT) with AVX2 intrinsics
 * @param sp0 pointer to Red samples (shall be aligned and multiple of 8 samples)
 * @param sp1 pointer to Green samples (shall be aligned and multiple of 8 samples)
 * @param sp2 pointer to Blue samples (shall be aligned and multiple of 8 samples)
 * @param width original width (may not be multiple of 8)
 * @param height original height
 */
void cvt_rgb_to_ycbcr_irrev_avx2(int32_t *sp0, int32_t *sp1, int32_t *sp2, uint32_t width, uint32_t height);
/**
 * @brief Inverse reversible color transform (RCT) with AVX2 intrinsics
 * @param sp0 pointer to Y samples (shall be aligned and multiple of 8 samples)
 * @param sp1 pointer to Cb samples (shall be aligned and multiple of 8 samples)
 * @param sp2 pointer to Cr samples (shall be aligned and multiple of 8 samples)
 * @param width original width (may not be multiple of 8)
 * @param height original height
 */
void cvt_ycbcr_to_rgb_rev_avx2(int32_t *sp0, int32_t *sp1, int32_t *sp2, uint32_t width, uint32_t height);
/**
 * @brief Inverse irreversible color transform (ICT) with AVX2 intrinsics
 * @param sp0 pointer to Y samples (shall be aligned and multiple of 8 samples)
 * @param sp1 pointer to Cb samples (shall be aligned and multiple of 8 samples)
 * @param sp2 pointer to Cr samples (shall be aligned and multiple of 8 samples)
 * @param width original width (may not be multiple of 8)
 * @param height original height
 */
void cvt_ycbcr_to_rgb_irrev_avx2(int32_t *sp0, int32_t *sp1, int32_t *sp2, uint32_t width, uint32_t height);
void cvt_ycbcr_to_rgb_rev_float_avx2(float *sp0, float *sp1, float *sp2, uint32_t width, uint32_t height, uint32_t stride);
void cvt_ycbcr_to_rgb_irrev_float_avx2(float *sp0, float *sp1, float *sp2, uint32_t width, uint32_t height, uint32_t stride);
void cvt_rgb_to_ycbcr_rev_float_avx2(const int32_t *sp0, const int32_t *sp1, const int32_t *sp2,
                                     float *dp0, float *dp1, float *dp2,
                                     uint32_t width, uint32_t height, uint32_t stride);
void cvt_rgb_to_ycbcr_irrev_float_avx2(const int32_t *sp0, const int32_t *sp1, const int32_t *sp2,
                                       float *dp0, float *dp1, float *dp2,
                                       uint32_t width, uint32_t height, uint32_t stride);
// Fused inverse ICT (lossy) + float→int32 finalize (AVX2)
void fused_ycbcr_irrev_to_rgb_i32_avx2(const float *y, const float *cb, const float *cr,
                                        int32_t *r, int32_t *g, int32_t *b, uint32_t width,
                                        const FinalizeParams *fp);
// Fused inverse RCT (lossless) + int32 finalize (AVX2)
void fused_ycbcr_rev_to_rgb_i32_avx2(const float *y, const float *cb, const float *cr,
                                      int32_t *r, int32_t *g, int32_t *b, uint32_t width,
                                      const FinalizeParams *fp);
#elif defined(OPENHTJ2K_ENABLE_ARM_NEON)
/**
 * @brief Forward reversible color transform (RCT) with NEON intrinsics
 * @param sp0 pointer to Red samples (shall be aligned and multiple of 8 samples)
 * @param sp1 pointer to Green samples (shall be aligned and multiple of 8 samples)
 * @param sp2 pointer to Blue samples (shall be aligned and multiple of 8 samples)
 * @param width original width (may not be multiple of 8)
 * @param height original height
 */
void cvt_rgb_to_ycbcr_rev_neon(int32_t *sp0, int32_t *sp1, int32_t *sp2, uint32_t width, uint32_t height);
/**
 * @brief Forward irreversible color transform (ICT) with NEON intrinsics
 * @param sp0 pointer to Red samples (shall be aligned and multiple of 8 samples)
 * @param sp1 pointer to Green samples (shall be aligned and multiple of 8 samples)
 * @param sp2 pointer to Blue samples (shall be aligned and multiple of 8 samples)
 * @param width original width (may not be multiple of 8)
 * @param height original height
 */
void cvt_rgb_to_ycbcr_irrev_neon(int32_t *sp0, int32_t *sp1, int32_t *sp2, uint32_t width, uint32_t height);
/**
 * @brief Inverse reversible color transform (RCT) with NEON intrinsics
 * @param sp0 pointer to Y samples (shall be aligned and multiple of 8 samples)
 * @param sp1 pointer to Cb samples (shall be aligned and multiple of 8 samples)
 * @param sp2 pointer to Cr samples (shall be aligned and multiple of 8 samples)
 * @param width original width (may not be multiple of 8)
 * @param height original height
 */
void cvt_ycbcr_to_rgb_rev_neon(int32_t *sp0, int32_t *sp1, int32_t *sp2, uint32_t width, uint32_t height);
/**
 * @brief Inverse irreversible color transform (ICT) with NEON intrinsics
 * @param sp0 pointer to Y samples (shall be aligned and multiple of 8 samples)
 * @param sp1 pointer to Cb samples (shall be aligned and multiple of 8 samples)
 * @param sp2 pointer to Cr samples (shall be aligned and multiple of 8 samples)
 * @param width original width (may not be multiple of 8)
 * @param height original height
 */
void cvt_ycbcr_to_rgb_irrev_neon(int32_t *sp0, int32_t *sp1, int32_t *sp2, uint32_t width, uint32_t height);
void cvt_ycbcr_to_rgb_rev_float_neon(float *sp0, float *sp1, float *sp2, uint32_t width, uint32_t height, uint32_t stride);
void cvt_ycbcr_to_rgb_irrev_float_neon(float *sp0, float *sp1, float *sp2, uint32_t width, uint32_t height, uint32_t stride);
void cvt_rgb_to_ycbcr_rev_float_neon(const int32_t *sp0, const int32_t *sp1, const int32_t *sp2,
                                     float *dp0, float *dp1, float *dp2,
                                     uint32_t width, uint32_t height, uint32_t stride);
void cvt_rgb_to_ycbcr_irrev_float_neon(const int32_t *sp0, const int32_t *sp1, const int32_t *sp2,
                                       float *dp0, float *dp1, float *dp2,
                                       uint32_t width, uint32_t height, uint32_t stride);
// Fused inverse ICT (lossy) + float→int32 finalize (NEON)
void fused_ycbcr_irrev_to_rgb_i32_neon(const float *y, const float *cb, const float *cr,
                                        int32_t *r, int32_t *g, int32_t *b, uint32_t width,
                                        const FinalizeParams *fp);
// Fused inverse RCT (lossless) + int32 finalize (NEON)
void fused_ycbcr_rev_to_rgb_i32_neon(const float *y, const float *cb, const float *cr,
                                      int32_t *r, int32_t *g, int32_t *b, uint32_t width,
                                      const FinalizeParams *fp);
#elif defined(OPENHTJ2K_ENABLE_WASM_SIMD)
void cvt_rgb_to_ycbcr_rev_wasm(int32_t *sp0, int32_t *sp1, int32_t *sp2, uint32_t width, uint32_t height);
void cvt_rgb_to_ycbcr_irrev_wasm(int32_t *sp0, int32_t *sp1, int32_t *sp2, uint32_t width, uint32_t height);
void cvt_ycbcr_to_rgb_rev_wasm(int32_t *sp0, int32_t *sp1, int32_t *sp2, uint32_t width, uint32_t height);
void cvt_ycbcr_to_rgb_irrev_wasm(int32_t *sp0, int32_t *sp1, int32_t *sp2, uint32_t width, uint32_t height);
void cvt_ycbcr_to_rgb_rev_float_wasm(float *sp0, float *sp1, float *sp2, uint32_t width, uint32_t height, uint32_t stride);
void cvt_ycbcr_to_rgb_irrev_float_wasm(float *sp0, float *sp1, float *sp2, uint32_t width, uint32_t height, uint32_t stride);
void cvt_rgb_to_ycbcr_rev_float_wasm(const int32_t *sp0, const int32_t *sp1, const int32_t *sp2,
                                     float *dp0, float *dp1, float *dp2,
                                     uint32_t width, uint32_t height, uint32_t stride);
void cvt_rgb_to_ycbcr_irrev_float_wasm(const int32_t *sp0, const int32_t *sp1, const int32_t *sp2,
                                       float *dp0, float *dp1, float *dp2,
                                       uint32_t width, uint32_t height, uint32_t stride);
void fused_ycbcr_irrev_to_rgb_i32_wasm(const float *y, const float *cb, const float *cr,
                                        int32_t *r, int32_t *g, int32_t *b, uint32_t width,
                                        const FinalizeParams *fp);
void fused_ycbcr_rev_to_rgb_i32_wasm(const float *y, const float *cb, const float *cr,
                                      int32_t *r, int32_t *g, int32_t *b, uint32_t width,
                                      const FinalizeParams *fp);
#else
void cvt_rgb_to_ycbcr_rev(int32_t *sp0, int32_t *sp1, int32_t *sp2, uint32_t width, uint32_t height);
void cvt_rgb_to_ycbcr_irrev(int32_t *sp0, int32_t *sp1, int32_t *sp2, uint32_t width, uint32_t height);
void cvt_ycbcr_to_rgb_rev(int32_t *sp0, int32_t *sp1, int32_t *sp2, uint32_t width, uint32_t height);
void cvt_ycbcr_to_rgb_irrev(int32_t *sp0, int32_t *sp1, int32_t *sp2, uint32_t width, uint32_t height);
void cvt_ycbcr_to_rgb_rev_float(float *sp0, float *sp1, float *sp2, uint32_t width, uint32_t height, uint32_t stride);
void cvt_ycbcr_to_rgb_irrev_float(float *sp0, float *sp1, float *sp2, uint32_t width, uint32_t height, uint32_t stride);
void cvt_rgb_to_ycbcr_rev_float(const int32_t *sp0, const int32_t *sp1, const int32_t *sp2,
                                float *dp0, float *dp1, float *dp2,
                                uint32_t width, uint32_t height, uint32_t stride);
void cvt_rgb_to_ycbcr_irrev_float(const int32_t *sp0, const int32_t *sp1, const int32_t *sp2,
                                  float *dp0, float *dp1, float *dp2,
                                  uint32_t width, uint32_t height, uint32_t stride);
void fused_ycbcr_irrev_to_rgb_i32(const float *y, const float *cb, const float *cr,
                                   int32_t *r, int32_t *g, int32_t *b, uint32_t width,
                                   const FinalizeParams *fp);
void fused_ycbcr_rev_to_rgb_i32(const float *y, const float *cb, const float *cr,
                                 int32_t *r, int32_t *g, int32_t *b, uint32_t width,
                                 const FinalizeParams *fp);
  #if defined(OPENHTJ2K_ENABLE_SSE2)
void fused_ycbcr_irrev_to_rgb_i32_sse2(const float *y, const float *cb, const float *cr, int32_t *r,
                                       int32_t *g, int32_t *b, uint32_t width, const FinalizeParams *fp);
  #endif
#endif

inline int32_t round_d(double val) {
  if (fabs(val) < DBL_EPSILON) {
    return 0;
  } else {
    return static_cast<int32_t>(val + ((val > 0) ? 0.5 : -0.5));
  }
}
