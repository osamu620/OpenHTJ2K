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
#include <cstddef>
#include "open_htj2k_typedef.hpp"
#include "j2kmarkers.hpp"
// utils.hpp brings in <x86intrin.h> / <arm_neon.h> conditionally on the active
// SIMD backend, which dwt_pse_fill_inplace_simd needs for the intrinsics below.
#include "utils.hpp"
#if defined(OPENHTJ2K_ENABLE_WASM_SIMD)
  #include <wasm_simd128.h>
#endif
#if defined(OPENHTJ2K_TRY_AVX2) && defined(__AVX512F__)
  #define OPENHTJ2K_ENABLE_AVX512
#endif
#if defined(OPENHTJ2K_TRY_AVX2) && defined(__AVX2__)
  #define OPENHTJ2K_ENABLE_AVX2
#endif
#define SIMD_PADDING 32
// DWT_RIGHT_SLACK is declared in utils.hpp (accessible to coding_units.cpp without
// pulling in the transform headers).  It must equal SIMD_PADDING because both
// describe the same per-row right-edge padding the in-place horizontal DWT
// requires.  This static_assert keeps the two constants in sync without
// creating a header dependency from utils.hpp on dwt.hpp.
static_assert(DWT_RIGHT_SLACK == SIMD_PADDING,
              "DWT_RIGHT_SLACK (utils.hpp) must equal SIMD_PADDING (dwt.hpp)");
constexpr int32_t DWT_VERT_STRIP = 64;  // column-strip width for vertical DWT (multiple of 8)

constexpr float fA = -1.586134342059924f;
constexpr float fB = -0.052980118572961f;
constexpr float fC = 0.882911075530934f;
constexpr float fD = 0.443506852043971f;

constexpr int16_t Acoeff_simd      = -19206;  // need to -1
constexpr int16_t Bcoeff_simd      = -3472;   // need to >> 1
constexpr int16_t Bcoeff_simd_avx2 = -13888;  // need to (out+4) >> 3
constexpr int16_t Ccoeff_simd      = 28931;
constexpr int16_t Dcoeff_simd      = 14533;

constexpr int32_t Acoeff = -25987;
constexpr int32_t Bcoeff = -3472;
constexpr int32_t Ccoeff = 28931;
constexpr int32_t Dcoeff = 29066;

constexpr int32_t Aoffset = 8192;
constexpr int32_t Boffset = 32767;
constexpr int32_t Coffset = 16384;
constexpr int32_t Doffset = 32767;

constexpr int32_t Ashift = 14;
constexpr int32_t Bshift = 16;
constexpr int32_t Cshift = 15;
constexpr int32_t Dshift = 16;

// define pointer to FDWT functions
typedef void (*fdwt_1d_filtr_func_fixed)(sprec_t *, const int32_t, const int32_t, const int32_t);
typedef void (*fdwt_ver_filtr_func_fixed)(sprec_t *, const int32_t, const int32_t, const int32_t,
                                          const int32_t, const int32_t stride, sprec_t *pse_scratch,
                                          sprec_t **buf_scratch);
// define pointer to IDWT functions
typedef void (*idwt_1d_filtd_func_fixed)(sprec_t *, const int32_t, const int32_t, const int32_t);
typedef void (*idwt_ver_filtd_func_fixed)(sprec_t *, const int32_t, const int32_t, const int32_t,
                                          const int32_t, const int32_t stride, sprec_t *pse_scratch,
                                          sprec_t **buf_scratch);

// symmetric extension
static inline int32_t PSEo(const int32_t i, const int32_t i0, const int32_t i1) {
  const int32_t tmp0    = 2 * (i1 - i0 - 1);
  const int32_t tmp1    = ((i - i0) < 0) ? i0 - i : i - i0;
  const int32_t mod_val = tmp1 % tmp0;
  const int32_t min_val = mod_val < tmp0 - mod_val ? mod_val : tmp0 - mod_val;
  return min_val;
}

// In-place whole-sample symmetric extension (WSSE) of row[].
//
// Writes:
//   row[-i] = row[+i]              for i = 1..8   (left mirror around 0)
//   row[width+k] = row[width-2-k]  for k = 0..7   (right mirror around width-1)
//
// The horizontal DWT filter only reads at most LEFT, RIGHT in {3, 4} samples
// past each end (parity-dependent for the 9/7 and 5/3 filters), so the upper
// 4-5 write lanes spill into the caller-owned scratch region and are harmless.
//
// Preconditions:
//   - row[0..width-1] holds valid data
//   - row[1..8] and row[width-9..width-2] are valid (width >= 9 — caller must
//     dispatch narrow widths to the scalar PSEo() path)
//   - row[-8..-1] and row[width..width+7] are writable scratch (8 floats each side):
//       * line-based path: ring-buffer slot has IDWT_RING_PSE_LEFT = 8 prefix
//         and SIMD_PADDING = 32 suffix
//       * batch path: caller saves left_save[8] / right_save[SIMD_PADDING]
//
// Implementation: load 8 floats per side, reverse via constant-pattern permute,
// store. ~3 instructions per side on AVX2 / NEON / WASM.
static inline void dwt_pse_fill_inplace_simd(sprec_t *row, int32_t width) {
#if defined(OPENHTJ2K_ENABLE_AVX2)
  // AVX2 (and AVX-512 builds): 8-lane reverse via vpermps.
  const __m256i rev = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
  __m256 lv = _mm256_loadu_ps(row + 1);                       // [row[1]..row[8]]
  __m256 lr = _mm256_permutevar8x32_ps(lv, rev);              // [row[8]..row[1]]
  _mm256_storeu_ps(row - 8, lr);                              // row[-8..-1]
  __m256 rv = _mm256_loadu_ps(row + width - 9);               // [row[w-9]..row[w-2]]
  __m256 rr = _mm256_permutevar8x32_ps(rv, rev);              // [row[w-2]..row[w-9]]
  _mm256_storeu_ps(row + width, rr);                          // row[w..w+7]
#elif defined(OPENHTJ2K_ENABLE_ARM_NEON)
  // NEON: 4-lane reverse via vrev64q + vextq combo, two halves per side.
  float32x4_t lo = vld1q_f32(row + 1);                        // [row[1],row[2],row[3],row[4]]
  float32x4_t hi = vld1q_f32(row + 5);                        // [row[5],row[6],row[7],row[8]]
  float32x4_t lo_rev = vrev64q_f32(vextq_f32(lo, lo, 2));     // [row[4],row[3],row[2],row[1]]
  float32x4_t hi_rev = vrev64q_f32(vextq_f32(hi, hi, 2));     // [row[8],row[7],row[6],row[5]]
  vst1q_f32(row - 4, lo_rev);                                 // row[-4..-1]
  vst1q_f32(row - 8, hi_rev);                                 // row[-8..-5] (slack)
  float32x4_t r_lo = vld1q_f32(row + width - 9);              // [row[w-9]..row[w-6]]
  float32x4_t r_hi = vld1q_f32(row + width - 5);              // [row[w-5]..row[w-2]]
  float32x4_t r_hi_rev = vrev64q_f32(vextq_f32(r_hi, r_hi, 2));  // [row[w-2],row[w-3],row[w-4],row[w-5]]
  float32x4_t r_lo_rev = vrev64q_f32(vextq_f32(r_lo, r_lo, 2));  // [row[w-6],row[w-7],row[w-8],row[w-9]]
  vst1q_f32(row + width,     r_hi_rev);                       // row[w..w+3]
  vst1q_f32(row + width + 4, r_lo_rev);                       // row[w+4..w+7] (slack)
#elif defined(OPENHTJ2K_ENABLE_WASM_SIMD)
  // WASM SIMD128: 4-lane shuffle, two halves per side.
  v128_t lo = wasm_v128_load(row + 1);
  v128_t hi = wasm_v128_load(row + 5);
  v128_t lo_rev = wasm_i32x4_shuffle(lo, lo, 3, 2, 1, 0);
  v128_t hi_rev = wasm_i32x4_shuffle(hi, hi, 3, 2, 1, 0);
  wasm_v128_store(row - 4, lo_rev);
  wasm_v128_store(row - 8, hi_rev);
  v128_t r_lo = wasm_v128_load(row + width - 9);
  v128_t r_hi = wasm_v128_load(row + width - 5);
  v128_t r_hi_rev = wasm_i32x4_shuffle(r_hi, r_hi, 3, 2, 1, 0);
  v128_t r_lo_rev = wasm_i32x4_shuffle(r_lo, r_lo, 3, 2, 1, 0);
  wasm_v128_store(row + width,     r_hi_rev);
  wasm_v128_store(row + width + 4, r_lo_rev);
#else
  // Scalar fallback: 8-iter unrolled, no PSEo() / modulo / branch.
  for (int32_t i = 1; i <= 8; ++i) row[-i]            = row[i];
  for (int32_t i = 0; i <  8; ++i) row[width + i]     = row[width - 2 - i];
#endif
}
template <class T>
static inline void dwt_1d_extr_fixed(T *extbuf, T *buf, const int32_t left, const int32_t right,
                                     const int32_t i0, const int32_t i1) {
  memcpy(extbuf + left, buf, sizeof(T) * static_cast<size_t>((i1 - i0)));
  for (int32_t i = 1; i <= left; ++i) {
    extbuf[left - i] = buf[PSEo(i0 - i, i0, i1)];
  }
  for (int32_t i = 1; i <= right; ++i) {
    extbuf[left + (i1 - i0) + i - 1] = buf[PSEo(i1 - i0 + i - 1 + i0, i0, i1)];
  }
}

// FDWT
#if defined(OPENHTJ2K_ENABLE_ARM_NEON)
void fdwt_1d_filtr_irrev97_fixed_neon(sprec_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
void fdwt_1d_filtr_rev53_fixed_neon(sprec_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
void fdwt_irrev_ver_sr_fixed_neon(sprec_t *in, int32_t u0, int32_t u1, int32_t v0, int32_t v1,
                                  int32_t stride, sprec_t *pse_scratch, sprec_t **buf_scratch);
void fdwt_rev_ver_sr_fixed_neon(sprec_t *in, int32_t u0, int32_t u1, int32_t v0, int32_t v1,
                                int32_t stride, sprec_t *pse_scratch, sprec_t **buf_scratch);
// Single-row reversible (5/3) FDWT vertical lifting steps.
void fdwt_rev_ver_hp_step_neon(int32_t n, const float *prev, const float *next, float *tgt);
void fdwt_rev_ver_lp_step_neon(int32_t n, const float *prev, const float *next, float *tgt);

#elif defined(OPENHTJ2K_ENABLE_AVX2)
void fdwt_1d_filtr_irrev97_fixed_avx2(sprec_t *X, const int32_t left, const int32_t u_i0,
                                      const int32_t u_i1);
void fdwt_1d_filtr_rev53_fixed_avx2(sprec_t *X, const int32_t left, const int32_t u_i0, const int32_t u_i1);
void fdwt_irrev_ver_sr_fixed_avx2(sprec_t *in, const int32_t u0, const int32_t u1, const int32_t v0,
                                  const int32_t v1, const int32_t stride, sprec_t *pse_scratch,
                                  sprec_t **buf_scratch);
void fdwt_rev_ver_sr_fixed_avx2(sprec_t *in, const int32_t u0, const int32_t u1, const int32_t v0,
                                const int32_t v1, const int32_t stride, sprec_t *pse_scratch,
                                sprec_t **buf_scratch);
// Single-row reversible (5/3) FDWT vertical lifting steps.
void fdwt_rev_ver_hp_step_avx2(int32_t n, const float *prev, const float *next, float *tgt);
void fdwt_rev_ver_lp_step_avx2(int32_t n, const float *prev, const float *next, float *tgt);
#else
void fdwt_1d_filtr_irrev97_fixed(sprec_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
void fdwt_1d_filtr_rev53_fixed(sprec_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
void fdwt_irrev_ver_sr_fixed(sprec_t *in, int32_t u0, int32_t u1, int32_t v0, int32_t v1, int32_t stride, sprec_t *pse_scratch, sprec_t **buf_scratch);
void fdwt_rev_ver_sr_fixed(sprec_t *in, int32_t u0, int32_t u1, int32_t v0, int32_t v1, int32_t stride, sprec_t *pse_scratch, sprec_t **buf_scratch);
#endif

void fdwt_2d_sr_fixed(sprec_t *previousLL, sprec_t *LL, sprec_t *HL, sprec_t *LH, sprec_t *HH, int32_t u0,
                      int32_t u1, int32_t v0, int32_t v1, uint8_t transformation, sprec_t *pse_scratch,
                      sprec_t **buf_scratch);

// FDWT AVX-512 horizontal and vertical
#if defined(OPENHTJ2K_ENABLE_AVX512)
void fdwt_1d_filtr_irrev97_fixed_avx512(sprec_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
void fdwt_1d_filtr_rev53_fixed_avx512(sprec_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
void fdwt_irrev_ver_sr_fixed_avx512(sprec_t *in, int32_t u0, int32_t u1, int32_t v0, int32_t v1,
                                    int32_t stride, sprec_t *pse_scratch, sprec_t **buf_scratch);
void fdwt_rev_ver_sr_fixed_avx512(sprec_t *in, int32_t u0, int32_t u1, int32_t v0, int32_t v1,
                                  int32_t stride, sprec_t *pse_scratch, sprec_t **buf_scratch);
#endif

// IDWT
#if defined(OPENHTJ2K_ENABLE_AVX512)
void idwt_1d_filtr_rev53_fixed_avx512(sprec_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
void idwt_1d_filtr_irrev97_fixed_avx512(sprec_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
void idwt_1d_filtr_irrev53_fixed_avx512(sprec_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
void idwt_irrev_ver_sr_fixed_avx512(sprec_t *in, int32_t u0, int32_t u1, int32_t v0, int32_t v1,
                                    int32_t stride, sprec_t *pse_scratch, sprec_t **buf_scratch);
void idwt_rev_ver_sr_fixed_avx512(sprec_t *in, int32_t u0, int32_t u1, int32_t v0, int32_t v1,
                                  int32_t stride, sprec_t *pse_scratch, sprec_t **buf_scratch);
void idwt_irrev53_ver_sr_fixed_avx512(sprec_t *in, int32_t u0, int32_t u1, int32_t v0, int32_t v1,
                                      int32_t stride, sprec_t *pse_scratch, sprec_t **buf_scratch);
void idwt_irrev_ver_step_fixed_avx512(int32_t n, float *prev, float *next, float *tgt, float coeff);
void idwt_rev_ver_lp_step_avx512(int32_t n, const float *prev, const float *next, float *tgt);
void idwt_rev_ver_hp_step_avx512(int32_t n, const float *prev, const float *next, float *tgt);
// Single-row reversible (5/3) FDWT vertical lifting steps.
void fdwt_rev_ver_hp_step_avx512(int32_t n, const float *prev, const float *next, float *tgt);
void fdwt_rev_ver_lp_step_avx512(int32_t n, const float *prev, const float *next, float *tgt);
#elif defined(OPENHTJ2K_ENABLE_ARM_NEON)
void idwt_1d_filtr_rev53_fixed_neon(sprec_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
void idwt_1d_filtr_irrev97_fixed_neon(sprec_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
void idwt_1d_filtr_irrev53_fixed_neon(sprec_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
void idwt_irrev_ver_sr_fixed_neon(sprec_t *in, int32_t u0, int32_t u1, int32_t v0, int32_t v1,
                                  int32_t stride, sprec_t *pse_scratch, sprec_t **buf_scratch);
void idwt_rev_ver_sr_fixed_neon(sprec_t *in, int32_t u0, int32_t u1, int32_t v0, int32_t v1,
                                int32_t stride, sprec_t *pse_scratch, sprec_t **buf_scratch);
void idwt_irrev53_ver_sr_fixed_neon(sprec_t *in, int32_t u0, int32_t u1, int32_t v0, int32_t v1,
                                    int32_t stride, sprec_t *pse_scratch, sprec_t **buf_scratch);
// Single-row irreversible vertical lifting step: tgt[i] -= coeff*(prev[i]+next[i]) using FMA.
// Uses SIMD for multiples of 4 elements, scalar for the tail.
void idwt_irrev_ver_step_fixed_neon(int32_t n, float *prev, float *next, float *tgt, float coeff);
// Single-row reversible (5/3) vertical lifting steps.
void idwt_rev_ver_lp_step_neon(int32_t n, const float *prev, const float *next, float *tgt);
void idwt_rev_ver_hp_step_neon(int32_t n, const float *prev, const float *next, float *tgt);
#elif defined(OPENHTJ2K_ENABLE_AVX2)
void idwt_1d_filtr_rev53_fixed_avx2(sprec_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
void idwt_1d_filtr_irrev97_fixed_avx2(sprec_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
void idwt_1d_filtr_irrev53_fixed_avx2(sprec_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
void idwt_irrev_ver_sr_fixed_avx2(sprec_t *in, int32_t u0, int32_t u1, int32_t v0, int32_t v1,
                                  int32_t stride, sprec_t *pse_scratch, sprec_t **buf_scratch);
void idwt_rev_ver_sr_fixed_avx2(sprec_t *in, int32_t u0, int32_t u1, int32_t v0, int32_t v1,
                                int32_t stride, sprec_t *pse_scratch, sprec_t **buf_scratch);
void idwt_irrev53_ver_sr_fixed_avx2(sprec_t *in, int32_t u0, int32_t u1, int32_t v0, int32_t v1,
                                    int32_t stride, sprec_t *pse_scratch, sprec_t **buf_scratch);
// Single-row irreversible vertical lifting step: tgt[i] -= coeff*(prev[i]+next[i]) using FMA.
// Uses SIMD for multiples of 8 elements, scalar for the tail.
void idwt_irrev_ver_step_fixed_avx2(int32_t n, float *prev, float *next, float *tgt, float coeff);
// Single-row reversible (5/3) vertical lifting steps.
void idwt_rev_ver_lp_step_avx2(int32_t n, const float *prev, const float *next, float *tgt);
void idwt_rev_ver_hp_step_avx2(int32_t n, const float *prev, const float *next, float *tgt);
#else
void idwt_1d_filtr_rev53_fixed(sprec_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
void idwt_1d_filtr_irrev97_fixed(sprec_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
void idwt_irrev_ver_sr_fixed(sprec_t *in, int32_t u0, int32_t u1, int32_t v0, int32_t v1, int32_t stride, sprec_t *pse_scratch, sprec_t **buf_scratch);
void idwt_rev_ver_sr_fixed(sprec_t *in, int32_t u0, int32_t u1, int32_t v0, int32_t v1, int32_t stride, sprec_t *pse_scratch, sprec_t **buf_scratch);
#endif

// WASM-SIMD DWT kernels (EMSCRIPTEN builds only, no NEON dependency).
#if defined(OPENHTJ2K_ENABLE_WASM_SIMD)
// horizontal
void fdwt_1d_filtr_irrev97_fixed_wasm(sprec_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
void fdwt_1d_filtr_rev53_fixed_wasm(sprec_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
void idwt_1d_filtr_irrev97_fixed_wasm(sprec_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
void idwt_1d_filtr_rev53_fixed_wasm(sprec_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
// vertical (batch)
void fdwt_irrev_ver_sr_fixed_wasm(sprec_t *in, int32_t u0, int32_t u1, int32_t v0, int32_t v1,
                                  int32_t stride, sprec_t *pse_scratch, sprec_t **buf_scratch);
void fdwt_rev_ver_sr_fixed_wasm(sprec_t *in, int32_t u0, int32_t u1, int32_t v0, int32_t v1,
                                int32_t stride, sprec_t *pse_scratch, sprec_t **buf_scratch);
void idwt_irrev_ver_sr_fixed_wasm(sprec_t *in, int32_t u0, int32_t u1, int32_t v0, int32_t v1,
                                  int32_t stride, sprec_t *pse_scratch, sprec_t **buf_scratch);
void idwt_rev_ver_sr_fixed_wasm(sprec_t *in, int32_t u0, int32_t u1, int32_t v0, int32_t v1,
                                int32_t stride, sprec_t *pse_scratch, sprec_t **buf_scratch);
// single-row vertical step (for streaming idwt_2d_state)
void idwt_irrev_ver_step_fixed_wasm(int32_t n, float *prev, float *next, float *tgt, float coeff);
void idwt_rev_ver_lp_step_wasm(int32_t n, const float *prev, const float *next, float *tgt);
void idwt_rev_ver_hp_step_wasm(int32_t n, const float *prev, const float *next, float *tgt);
// single-row vertical step (for streaming fdwt_2d_state)
void fdwt_rev_ver_hp_step_wasm(int32_t n, const float *prev, const float *next, float *tgt);
void fdwt_rev_ver_lp_step_wasm(int32_t n, const float *prev, const float *next, float *tgt);
#endif

void idwt_2d_sr_fixed(sprec_t *nextLL, sprec_t *LL, sprec_t *HL, sprec_t *LH, sprec_t *HH, int32_t u0,
                      int32_t u1, int32_t v0, int32_t v1, uint8_t transformation, sprec_t *pse_scratch,
                      sprec_t **buf_scratch);

// DFS HORZ level: interleave LL (even x) and H (odd x), apply horizontal 1D IDWT only.
// LL and H must have strides = round_up(their_width, 32); nextLL stride = round_up(u1-u0, 32).
void idwt_horz_only_sr_fixed(sprec_t *nextLL, const sprec_t *LL, const sprec_t *H, int32_t u0, int32_t u1,
                              int32_t v0, int32_t v1, uint8_t transformation);

// DFS VERT level: interleave LL (even y) and H (odd y), apply vertical 1D IDWT only.
// LL, H, and nextLL all have stride = round_up(u1-u0, 32).
void idwt_vert_only_sr_fixed(sprec_t *nextLL, const sprec_t *LL, const sprec_t *H, int32_t u0, int32_t u1,
                              int32_t v0, int32_t v1, uint8_t transformation, sprec_t *pse_scratch,
                              sprec_t **buf_scratch);

// In-place variant for ring buffer slots: requires writable PSE scratch at row[-left..-1]
// and row[u1-u0..u1-u0+right-1] (guaranteed by IDWT_RING_PSE_LEFT slot prefix and SIMD_PADDING suffix).
// left and right are precomputed PSE counts (function of u0%2, u1%2, and transformation).
void idwt_1d_row_inplace(sprec_t *row, int32_t left, int32_t right,
                         int32_t u0, int32_t u1, uint8_t transformation);

// ─────────────────────────────────────────────────────────────────────────────
// Streaming 2D IDWT — produces one output row per call via pull_row().
//
// The caller supplies a row-source callback that returns one horizontally-
// synthesised interleaved row (LL+HL interleaved for LP rows, LH+HH for HP)
// for absolute row indices v0..v1-1.  Vertical lifting is driven internally
// using a sliding ring buffer and delay-line d_level tracking.
//
// Memory per level  ≈  (RING_DEPTH + top_pse + bottom_pse) × stride × 4 B
// ─────────────────────────────────────────────────────────────────────────────

// Callback: write the horizontally-synthesised interleaved row at absolute
// index abs_row ∈ [v0, v1) into out_row[0..u1-u0-1].
typedef void (*idwt_row_src_fn)(void *ctx, int32_t abs_row, sprec_t *out_row);

// Ring depth: must hold enough real rows for steady-state output.
// PSE rows use separate top_pse_buf/bot_pse_buf and never occupy ring slots.
// For 9/7 (max_dl=2): need rows r..r+4 simultaneously → 5 ring slots; RING_DEPTH=8 is safe.
// For 5/3 (max_dl=1): need rows r..r+2 simultaneously → 3 ring slots; RING_DEPTH=8 is more than enough.
// Reducing from 12 → 8 cuts level-1 ring buffer (4K: 3840 floats/row) from 180KB to 120KB,
// improving L2 cache utilization for vertical lifting steps.
constexpr int32_t IDWT_STATE_RING_DEPTH = 8;

// Extra floats reserved before each ring buffer slot for in-place horizontal PSE.
// Must be >= max(left PSE) = 4 and a multiple of 8 (8×4B=32B) for AVX2 alignment.
constexpr int32_t IDWT_RING_PSE_LEFT = 8;

struct idwt_2d_state {
  // ── geometry ──────────────────────────────────────────────────────────────
  int32_t u0, u1, v0, v1;
  int32_t stride;          // round_up(u1-u0, SIMD_PADDING) — data width per row
  int32_t slot_stride;     // IDWT_RING_PSE_LEFT + round_up(u1-u0+SIMD_PADDING, SIMD_PADDING)
  uint8_t transformation;  // 0 = irrev 9/7, 1 = rev 5/3, 2+ = ATK irrev
  dwt_type dir;            // DWT_BIDIR (full 2D), DWT_HORZ (horizontal only), DWT_NO (passthrough)
  int8_t  top_pse;         // PSE rows above v0  (3 or 4 for 9/7; 1 or 2 for 5/3 / ATK)
  int8_t  bottom_pse;      // PSE rows below v1-1

  // ── PSE scratch (separate from the ring, BIDIR only) ──────────────────────
  // top_pse_buf[0] ↔ physical row v0-1, [1] ↔ v0-2, …
  // bot_pse_buf[0] ↔ physical row v1,   [1] ↔ v1+1, …
  sprec_t *top_pse_buf;        // top_pse    × stride sprec_t (SIMD-aligned); nullptr for HORZ/NO
  sprec_t *bot_pse_buf;        // bottom_pse × stride sprec_t; nullptr for HORZ/NO
  int8_t   top_dlevel[4];      // d_level per top-PSE slot (-1 = unfilled)
  int8_t   bot_dlevel[4];      // d_level per bot-PSE slot (-1 = unfilled)

  // ── sliding ring for real rows [v0, v1) (BIDIR only) ─────────────────────
  // Slot for absolute row r : r % IDWT_STATE_RING_DEPTH
  // Each ring slot is slot_stride floats wide; the data portion (post-horizontal-IDWT)
  // starts at offset IDWT_RING_PSE_LEFT within the slot, providing scratch space
  // for the in-place horizontal PSE fill and filter (no separate ext_buf needed).
  sprec_t *ring_buf;                           // IDWT_STATE_RING_DEPTH × slot_stride; nullptr for HORZ/NO
  int32_t  ring_origin;                         // abs row mapped to slot 0
  int8_t   d_level[IDWT_STATE_RING_DEPTH];     // 0=raw, 1=step1, 2=step2, -1=unused

  // ── single-row output buffer (HORZ and NO only) ───────────────────────────
  // Allocated with IDWT_RING_PSE_LEFT prefix for in-place horizontal IDWT.
  // Data area starts at horz_out_buf + IDWT_RING_PSE_LEFT.
  sprec_t *horz_out_buf;   // nullptr for BIDIR

  // ── cursors ───────────────────────────────────────────────────────────────
  int32_t next_out;    // next output row (v0 ≤ next_out < v1)
  int32_t next_fetch;  // next real row to fetch from source (v0 ≤ next_fetch ≤ v1)

  // ── source ────────────────────────────────────────────────────────────────
  idwt_row_src_fn get_src_row;
  void           *src_ctx;
};

// Initialise the streaming IDWT state.
// For dir=DWT_BIDIR: allocates ring_buf, top_pse_buf, bot_pse_buf (full 2D vertical+horizontal).
// For dir=DWT_HORZ: allocates only horz_out_buf; no vertical lifting, horizontal IDWT only.
// For dir=DWT_NO:   allocates only horz_out_buf (passthrough — no filtering).
// DWT_VERT is not supported in the streaming path.
void idwt_2d_state_init(idwt_2d_state *s,
                        int32_t u0, int32_t u1, int32_t v0, int32_t v1,
                        uint8_t transformation, dwt_type dir,
                        idwt_row_src_fn src_fn, void *src_ctx);

// Free buffers allocated by idwt_2d_state_init.
void idwt_2d_state_free(idwt_2d_state *s);

// Rewind streaming cursors (next_out / next_fetch / ring_origin / d_level /
// top_dlevel / bot_dlevel) to the post-init state without freeing any
// buffers.  Used by the single-tile reuse path; a subsequent pull_row
// call restarts at row v0.  Safe to call on any dir.
void idwt_2d_state_rewind(idwt_2d_state *s);

// Pull the next output row into out[0..u1-u0-1].
// Returns true while rows remain; false when all v1-v0 rows have been produced.
bool idwt_2d_state_pull_row(idwt_2d_state *s, sprec_t *out);

// Zero-copy variant: returns a pointer into the internal ring buffer for the next
// output row (u1-u0 elements).  The pointer is valid until the next call to
// pull_row_ref or pull_row for this state.  Returns nullptr when exhausted.
// The caller MAY modify the returned row (e.g. in-place colour transform).
sprec_t *idwt_2d_state_pull_row_ref(idwt_2d_state *s);

// ─────────────────────────────────────────────────────────────────────────────
// Streaming 2D FDWT — consumes one input row per push_row() call.
//
// The caller pushes image rows one at a time (v0..v1-1).  When vertical
// lifting completes a row, horizontal analysis is applied and the result is
// delivered to the sink callback (is_hp=false → LP/LL+HL row; true → HP/LH+HH
// row).  The sink receives a pointer to an interleaved row that is valid only
// for the duration of the callback.
//
// Call fdwt_2d_state_flush() after the last push_row() to drain any remaining
// rows that depend on the bottom-PSE extension.
// ─────────────────────────────────────────────────────────────────────────────

// Callback: delivers a completed row after both V and H FDWT.
// is_hp=false → LP row (even abs_phys_row), is_hp=true → HP row (odd).
// interleaved_row: u1-u0 samples at stride alignment; valid only in callback.
typedef void (*fdwt_row_sink_fn)(void *ctx, bool is_hp, int32_t abs_phys_row,
                                 const sprec_t *interleaved_row);

constexpr int32_t FDWT_STATE_RING_DEPTH = 12;

struct fdwt_2d_state {
  // ── geometry ──────────────────────────────────────────────────────────────
  int32_t u0, u1, v0, v1;
  int32_t stride;            // round_up(u1-u0, SIMD_PADDING)
  int32_t horiz_left;        // horizontal-DWT left PSE length
  int32_t horiz_right;       // horizontal-DWT right PSE length
  uint8_t transformation;    // 0 = irrev 9/7, 1 = rev 5/3
  int8_t  top_pse;           // PSE rows above v0
  int8_t  bottom_pse;        // PSE rows below v1-1

  // ── PSE scratch ───────────────────────────────────────────────────────────
  sprec_t *top_pse_buf;      // top_pse    × stride sprec_t
  sprec_t *bot_pse_buf;      // bottom_pse × stride sprec_t
  int8_t   top_dlevel[4];    // d_level per top-PSE slot (-1 = unfilled)
  int8_t   bot_dlevel[4];    // d_level per bot-PSE slot (-1 = unfilled)

  // ── sliding ring ──────────────────────────────────────────────────────────
  sprec_t *ring_buf;                          // FDWT_STATE_RING_DEPTH × stride
  int32_t  ring_origin;
  int8_t   d_level[FDWT_STATE_RING_DEPTH];   // 0=raw, 1=step1, 2=step2, -1=unused

  // ── horizontal-DWT temp buffer ────────────────────────────────────────────
  // Size: horiz_left + stride + horiz_right + SIMD_PADDING
  sprec_t *horiz_tmp;

  // ── cursors ───────────────────────────────────────────────────────────────
  int32_t next_in;    // next row to accept via push_row() [v0, v1]
  int32_t next_emit;  // next completed row waiting to be emitted [v0, v1)

  // ── sink ──────────────────────────────────────────────────────────────────
  fdwt_row_sink_fn put_row;
  void            *sink_ctx;
};

// Initialise (allocates ring_buf, PSE buffers, horiz_tmp).
void fdwt_2d_state_init(fdwt_2d_state *s,
                        int32_t u0, int32_t u1, int32_t v0, int32_t v1,
                        uint8_t transformation,
                        fdwt_row_sink_fn sink_fn, void *sink_ctx);

// Free buffers allocated by fdwt_2d_state_init.
void fdwt_2d_state_free(fdwt_2d_state *s);

// Push one input row in[0..u1-u0-1].  May trigger sink callbacks.
void fdwt_2d_state_push_row(fdwt_2d_state *s, const sprec_t *in);

// Finalise: fill bottom PSE, run remaining cascade, emit all pending rows.
// Must be called after the last push_row().
void fdwt_2d_state_flush(fdwt_2d_state *s);