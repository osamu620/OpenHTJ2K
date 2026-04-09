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
  int16_t ds;      // downshift: > 0 → right shift, < 0 → left shift, 0 → no shift
  int16_t rnd;     // rounding offset added before right shift (= (1<<ds)>>1); 0 when ds <= 0
  int32_t dc;      // DC offset added after shift (e.g. 128 for 8-bit unsigned)
  int32_t maxval;  // clamp upper bound
  int32_t minval;  // clamp lower bound
};

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
#endif

inline int32_t round_d(double val) {
  if (fabs(val) < DBL_EPSILON) {
    return 0;
  } else {
    return static_cast<int32_t>(val + ((val > 0) ? 0.5 : -0.5));
  }
}
