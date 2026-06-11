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
// Column-strip width for vertical DWT (must be a multiple of 16).
// 128 on Apple Silicon to match M-series 128-byte cache lines (128 floats × 4B = 512B = 4 lines);
// 64 elsewhere to fit x86/Cortex-A 64-byte cache lines.
#if defined(__APPLE__) && defined(__aarch64__)
constexpr int32_t DWT_VERT_STRIP = 128;
#else
constexpr int32_t DWT_VERT_STRIP = 64;
#endif

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

// Integer variant of dwt_pse_fill_inplace_simd for use_i32 buffers.
// Identical algorithm but uses integer SIMD intrinsics — avoids strict-aliasing
// UB from applying float-typed operations to int32_t data.
static inline void dwt_pse_fill_inplace_i32(int32_t *row, int32_t width) {
#if defined(OPENHTJ2K_ENABLE_AVX2)
  const __m256i rev = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
  __m256i lv = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(row + 1));
  __m256i lr = _mm256_permutevar8x32_epi32(lv, rev);
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(row - 8), lr);
  __m256i rv = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(row + width - 9));
  __m256i rr = _mm256_permutevar8x32_epi32(rv, rev);
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(row + width), rr);
#elif defined(OPENHTJ2K_ENABLE_ARM_NEON)
  int32x4_t lo = vld1q_s32(row + 1);
  int32x4_t hi = vld1q_s32(row + 5);
  int32x4_t lo_rev = vrev64q_s32(vextq_s32(lo, lo, 2));
  int32x4_t hi_rev = vrev64q_s32(vextq_s32(hi, hi, 2));
  vst1q_s32(row - 4, lo_rev);
  vst1q_s32(row - 8, hi_rev);
  int32x4_t r_lo = vld1q_s32(row + width - 9);
  int32x4_t r_hi = vld1q_s32(row + width - 5);
  int32x4_t r_hi_rev = vrev64q_s32(vextq_s32(r_hi, r_hi, 2));
  int32x4_t r_lo_rev = vrev64q_s32(vextq_s32(r_lo, r_lo, 2));
  vst1q_s32(row + width,     r_hi_rev);
  vst1q_s32(row + width + 4, r_lo_rev);
#elif defined(OPENHTJ2K_ENABLE_WASM_SIMD)
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
  for (int32_t i = 1; i <= 8; ++i) row[-i]        = row[i];
  for (int32_t i = 0; i <  8; ++i) row[width + i] = row[width - 2 - i];
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
// int32 variants of the reversible (5/3) primitives — same algorithm operating
// on native int32 storage instead of float-with-integer-values.  Skip the
// mul-by-0.5 / floor / cast emulation of integer divide-by-2 in the float
// versions; use a single arithmetic right shift.  See PR description for
// rationale.  Currently unused — wired up by a follow-up commit.
void fdwt_1d_filtr_rev53_i32_neon(int32_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
void fdwt_rev_ver_hp_step_i32_neon(int32_t n, const int32_t *prev, const int32_t *next, int32_t *tgt);
void fdwt_rev_ver_lp_step_i32_neon(int32_t n, const int32_t *prev, const int32_t *next, int32_t *tgt);
// Encoder planar mirror (see the AVX2 declarations below for the contract):
// fused horizontal FDWT reading the natural-domain ring row, writing LP/HP
// planes.  Dispatched from emit_ready_f, which guarantees u0 even and
// u1/2 - u0/2 >= 12 (the 4-lane 9/7 warmup loads j = 0..7 unconditionally).
void fdwt_1d_filtr_irrev97_planar_neon(sprec_t *lp, sprec_t *hp, const sprec_t *in, int32_t u0, int32_t u1);
void fdwt_1d_filtr_rev53_planar_i32_neon(int32_t *lp, int32_t *hp, const int32_t *in, int32_t u0,
                                         int32_t u1);

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
// int32 variants of the reversible (5/3) primitives — see NEON block above.
void fdwt_1d_filtr_rev53_i32_avx2(int32_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
void fdwt_rev_ver_hp_step_i32_avx2(int32_t n, const int32_t *prev, const int32_t *next, int32_t *tgt);
void fdwt_rev_ver_lp_step_i32_avx2(int32_t n, const int32_t *prev, const int32_t *next, int32_t *tgt);
#else
void fdwt_1d_filtr_irrev97_fixed(sprec_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
void fdwt_1d_filtr_rev53_fixed(sprec_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
void fdwt_irrev_ver_sr_fixed(sprec_t *in, int32_t u0, int32_t u1, int32_t v0, int32_t v1, int32_t stride, sprec_t *pse_scratch, sprec_t **buf_scratch);
void fdwt_rev_ver_sr_fixed(sprec_t *in, int32_t u0, int32_t u1, int32_t v0, int32_t v1, int32_t stride, sprec_t *pse_scratch, sprec_t **buf_scratch);
// int32 scalar variant of the reversible (5/3) 1D primitive — see NEON block
// above.  (No vertical-step scalar counterparts: the streaming line-based DWT
// path requires SIMD step functions and the scalar fdwt build uses the batch
// fdwt_rev_ver_sr_fixed instead.)
void fdwt_1d_filtr_rev53_i32(int32_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
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
void idwt_1d_filtr_rev53_i32_avx512(int32_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
void idwt_rev_ver_lp_step_i32_avx512(int32_t n, const int32_t *prev, const int32_t *next, int32_t *tgt);
void idwt_rev_ver_hp_step_i32_avx512(int32_t n, const int32_t *prev, const int32_t *next, int32_t *tgt);
// Planar-input horizontal synthesis — 16-lane variant of the AVX2 planar
// kernels (see the standalone OPENHTJ2K_ENABLE_AVX2 block below for those).
// Bit-identical to the AVX2 planar kernels and the in-place fallback, so the
// dispatcher picks per-row: these for N = u1/2 - u0/2 >= 32 (the 16-lane
// warmup loads j = 0..31 unconditionally), the AVX2 kernels for 16 <= N < 32.
void idwt_1d_filtr_irrev97_planar_avx512(sprec_t *out, const sprec_t *lp, const sprec_t *hp, int32_t u0,
                                         int32_t u1);
void idwt_1d_filtr_rev53_planar_i32_avx512(int32_t *out, const int32_t *lp, const int32_t *hp, int32_t u0,
                                           int32_t u1);
// Encoder mirror: 16-lane variants of the AVX2 planar FDWT kernels (see the
// OPENHTJ2K_ENABLE_AVX2 block below).  Bit-identical to the AVX2 planar
// kernels, so emit_ready_f picks per-state: these for N = u1/2 - u0/2 >= 32
// (the 16-lane warmup loads j = 0..31 unconditionally), the AVX2 kernels for
// 16 <= N < 32.
void fdwt_1d_filtr_irrev97_planar_avx512(sprec_t *lp, sprec_t *hp, const sprec_t *in, int32_t u0,
                                         int32_t u1);
void fdwt_1d_filtr_rev53_planar_i32_avx512(int32_t *lp, int32_t *hp, const int32_t *in, int32_t u0,
                                           int32_t u1);
// Single-row reversible (5/3) FDWT vertical lifting steps.
void fdwt_rev_ver_hp_step_avx512(int32_t n, const float *prev, const float *next, float *tgt);
void fdwt_rev_ver_lp_step_avx512(int32_t n, const float *prev, const float *next, float *tgt);
// int32 variants of the reversible (5/3) primitives — see NEON block above.
void fdwt_1d_filtr_rev53_i32_avx512(int32_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
void fdwt_rev_ver_hp_step_i32_avx512(int32_t n, const int32_t *prev, const int32_t *next, int32_t *tgt);
void fdwt_rev_ver_lp_step_i32_avx512(int32_t n, const int32_t *prev, const int32_t *next, int32_t *tgt);
#elif defined(OPENHTJ2K_ENABLE_ARM_NEON)
void idwt_1d_filtr_rev53_fixed_neon(sprec_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
void idwt_1d_filtr_irrev97_fixed_neon(sprec_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
void idwt_1d_filtr_irrev53_fixed_neon(sprec_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
// Planar-input horizontal synthesis: read the LP plane (E[j] = lp[j]) and HP
// plane (O[j] = hp[j]) directly and write the synthesised natural-domain row
// to out[0..u1-u0-1] — no interleave pass, vld1q instead of vld2q.  Same fused
// lifting pipeline and single-rounded ops as the interleaved kernels, so the
// output is bit-identical.  Dispatched from idwt_1d_row_from_planar, which
// guarantees: u0 even, u1/2 - u0/2 >= 12, out with >= IDWT_RING_PSE_LEFT
// writable floats before index 0 and >= 8 after index u1-u0-1.
void idwt_1d_filtr_irrev97_planar_neon(sprec_t *out, const sprec_t *lp, const sprec_t *hp, int32_t u0,
                                       int32_t u1);
void idwt_1d_filtr_rev53_planar_i32_neon(int32_t *out, const int32_t *lp, const int32_t *hp, int32_t u0,
                                         int32_t u1);
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
void idwt_1d_filtr_rev53_i32_neon(int32_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
void idwt_rev_ver_lp_step_i32_neon(int32_t n, const int32_t *prev, const int32_t *next, int32_t *tgt);
void idwt_rev_ver_hp_step_i32_neon(int32_t n, const int32_t *prev, const int32_t *next, int32_t *tgt);
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
void idwt_1d_filtr_rev53_i32_avx2(int32_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
void idwt_rev_ver_lp_step_i32_avx2(int32_t n, const int32_t *prev, const int32_t *next, int32_t *tgt);
void idwt_rev_ver_hp_step_i32_avx2(int32_t n, const int32_t *prev, const int32_t *next, int32_t *tgt);
#else
void idwt_1d_filtr_rev53_fixed(sprec_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
void idwt_1d_filtr_irrev97_fixed(sprec_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
void idwt_irrev_ver_sr_fixed(sprec_t *in, int32_t u0, int32_t u1, int32_t v0, int32_t v1, int32_t stride, sprec_t *pse_scratch, sprec_t **buf_scratch);
void idwt_rev_ver_sr_fixed(sprec_t *in, int32_t u0, int32_t u1, int32_t v0, int32_t v1, int32_t stride, sprec_t *pse_scratch, sprec_t **buf_scratch);
#endif

// Planar-input horizontal synthesis for AVX2 — 8-lane port of the NEON planar
// kernels (see the NEON declarations above for the full contract).  Declared
// outside the AVX-512/NEON/AVX2 #elif chain because OPENHTJ2K_ENABLE_AVX2 is
// also defined on AVX-512 builds, where these kernels serve the planar fast
// path until a dedicated _avx512 variant exists (the in-place dispatch tables
// still prefer AVX-512).  Dispatched from idwt_1d_row_from_planar, which
// guarantees: u0 even, u1/2 - u0/2 >= 16 (the 8-lane warmup loads j = 0..15
// unconditionally), out with >= IDWT_RING_PSE_LEFT writable floats before
// index 0 and >= 8 after index u1-u0-1.
#if defined(OPENHTJ2K_ENABLE_AVX2)
void idwt_1d_filtr_irrev97_planar_avx2(sprec_t *out, const sprec_t *lp, const sprec_t *hp, int32_t u0,
                                       int32_t u1);
void idwt_1d_filtr_rev53_planar_i32_avx2(int32_t *out, const int32_t *lp, const int32_t *hp, int32_t u0,
                                         int32_t u1);
// Encoder mirror: fused 9/7 horizontal FDWT reading the natural-domain row
// (read-only, no PSE margins — boundary taps mirror within the row) and
// writing LP/HP planes with plain stores.  Dispatched from emit_ready_f,
// which guarantees: u0 even, u1/2 - u0/2 >= 16 (the 8-lane warmup loads
// j = 0..15 unconditionally), lp/hp with >= 8 writable floats of slack past
// their plane widths.
void fdwt_1d_filtr_irrev97_planar_avx2(sprec_t *lp, sprec_t *hp, const sprec_t *in, int32_t u0, int32_t u1);
// Reversible 5/3 int32 variant (lossless pipe), same dispatch guarantees as
// the 9/7 planar kernel.  All shifts arithmetic — bit-exact vs the
// interleaved fdwt_1d_filtr_rev53_i32_avx2 + sink-deinterleave path.
void fdwt_1d_filtr_rev53_planar_i32_avx2(int32_t *lp, int32_t *hp, const int32_t *in, int32_t u0,
                                         int32_t u1);
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
void idwt_1d_filtr_rev53_i32_wasm(int32_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
void idwt_rev_ver_lp_step_i32_wasm(int32_t n, const int32_t *prev, const int32_t *next, int32_t *tgt);
void idwt_rev_ver_hp_step_i32_wasm(int32_t n, const int32_t *prev, const int32_t *next, int32_t *tgt);
// Planar-input horizontal synthesis — 4-lane transcription of the NEON planar
// kernels (see the NEON declarations above for the full contract; same N >= 12
// dispatch guard).  The 9/7 kernel uses separately-rounded mul+sub to match
// the in-place WASM kernel (SIMD128 has no FMA) — bit-identical to the
// fallback on this platform, not to NEON/AVX2 hosts.
void idwt_1d_filtr_irrev97_planar_wasm(sprec_t *out, const sprec_t *lp, const sprec_t *hp, int32_t u0,
                                       int32_t u1);
void idwt_1d_filtr_rev53_planar_i32_wasm(int32_t *out, const int32_t *lp, const int32_t *hp, int32_t u0,
                                         int32_t u1);
// single-row vertical step (for streaming fdwt_2d_state)
void fdwt_rev_ver_hp_step_wasm(int32_t n, const float *prev, const float *next, float *tgt);
void fdwt_rev_ver_lp_step_wasm(int32_t n, const float *prev, const float *next, float *tgt);
// int32 variants of the reversible (5/3) primitives — see NEON block above.
void fdwt_1d_filtr_rev53_i32_wasm(int32_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
void fdwt_rev_ver_hp_step_i32_wasm(int32_t n, const int32_t *prev, const int32_t *next, int32_t *tgt);
void fdwt_rev_ver_lp_step_i32_wasm(int32_t n, const int32_t *prev, const int32_t *next, int32_t *tgt);
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
void idwt_1d_row_inplace_i32(int32_t *row, int32_t left, int32_t right,
                             int32_t u0, int32_t u1);

// Horizontal synthesis of one streaming row from planar LP/HP subband rows.
// Writes the synthesised natural-domain row to out[0..u1-u0-1].  `out` must be
// a ring-slot data pointer (IDWT_RING_PSE_LEFT writable floats before index 0,
// >= SIMD_PADDING after index u1-u0-1) and must not alias lp/hp.  On platforms
// with a planar kernel (NEON) and a supported geometry the LP/HP planes are
// lifted directly — no interleave pass; otherwise the planes are interleaved
// into `out` and the existing in-place kernels run.  Both paths produce
// bit-identical rows.  For use_i32 the column range is ignored (full-width
// lifting), matching the historical idwt_1d_row_inplace_i32 behaviour;
// lp/hp/out are then reinterpret_cast'd int32_t buffers.
void idwt_1d_row_from_planar(sprec_t *out, const sprec_t *lp, const sprec_t *hp,
                             int32_t lp_width, int32_t hp_width,
                             int32_t u0, int32_t u1, uint8_t transformation, bool use_i32,
                             int32_t h_pse_left, int32_t h_pse_right,
                             int32_t col_lo, int32_t col_hi);

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
  bool    use_i32;         // when true, ring/PSE buffers hold int32_t (reinterpret_cast'd from sprec_t*)
  int8_t  top_pse;         // PSE rows above v0  (3 or 4 for 9/7; 1 or 2 for 5/3 / ATK)
  int8_t  bottom_pse;      // PSE rows below v1-1

  // ── PSE scratch (separate from the ring, BIDIR only) ──────────────────────
  // top_pse_buf[0] ↔ physical row v0-1, [1] ↔ v0-2, …
  // bot_pse_buf[0] ↔ physical row v1,   [1] ↔ v1+1, …
  void *top_pse_buf;           // top_pse    × stride sprec_t (SIMD-aligned); nullptr for HORZ/NO
  void *bot_pse_buf;           // bottom_pse × stride sprec_t; nullptr for HORZ/NO
  int8_t   top_dlevel[4];      // d_level per top-PSE slot (-1 = unfilled)
  int8_t   bot_dlevel[4];      // d_level per bot-PSE slot (-1 = unfilled)

  // ── sliding ring for real rows [v0, v1) (BIDIR only) ─────────────────────
  // Slot for absolute row r : r & (IDWT_STATE_RING_DEPTH - 1)
  // Each ring slot is slot_stride floats wide; the data portion (post-horizontal-IDWT)
  // starts at offset IDWT_RING_PSE_LEFT within the slot, providing scratch space
  // for the in-place horizontal PSE fill and filter (no separate ext_buf needed).
  void *ring_buf;                              // IDWT_STATE_RING_DEPTH × slot_stride; nullptr for HORZ/NO
  int32_t  ring_origin;                         // abs row mapped to slot 0
  int8_t   d_level[IDWT_STATE_RING_DEPTH];     // 0=raw, 1=step1, 2=step2, -1=unused

  // ── single-row output buffer (HORZ and NO only) ───────────────────────────
  // Allocated with IDWT_RING_PSE_LEFT prefix for in-place horizontal IDWT.
  // Data area starts at horz_out_buf + IDWT_RING_PSE_LEFT.
  void *horz_out_buf;      // nullptr for BIDIR

  // ── zero-row tracking (Phase 4 IDWT skip for absent JPIP precincts) ──────
  // When a fetched source row is entirely zero (absent precinct), the zero
  // flag is set.  Vertical lifting is skipped when both neighbor rows are
  // zero, saving ~60 % of decode time for sparse JPIP frames.
  bool row_zero[IDWT_STATE_RING_DEPTH];
  bool top_pse_zero[4];
  bool bot_pse_zero[4];

  // ── cursors ───────────────────────────────────────────────────────────────
  int32_t next_out;    // next output row (v0 ≤ next_out < v1)
  int32_t next_fetch;  // next real row to fetch from source (v0 ≤ next_fetch ≤ v1)

  // ── column-range for vertical lifting (default = full [u0, u1]) ───────────
  // When JPIP region decode restricts the horizontal span, vertical lifting
  // is clipped to [col_lo, col_hi] — columns outside the range are left
  // untouched in the ring buffer.  Only the caller's read range within
  // [col_lo, col_hi] is guaranteed valid.  For default (full) decode the
  // range equals [u0, u1] and kernels run unchanged.
  int32_t col_lo;
  int32_t col_hi;

  // ── row-range for vertical viewport (default = full [v0, v1]) ─────────────
  // When the caller restricts the viewport's vertical extent (set_row_range
  // on the decoder), pull_row_ref fast-forwards its cursors to row_lo and
  // stops producing output once row_hi is reached.  The widen-by-halving
  // margin for 9/7 cascade dependencies is applied by the caller (see
  // j2k_tile_component::set_line_decode_row_range) before this state sees
  // the range, so row_lo here is already shifted to cover filter support.
  int32_t row_lo;
  int32_t row_hi;

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
                        idwt_row_src_fn src_fn, void *src_ctx,
                        bool use_i32 = false);

// Free buffers allocated by idwt_2d_state_init.
void idwt_2d_state_free(idwt_2d_state *s);

// Restrict vertical lifting to columns [col_lo, col_hi).  Passing the default
// [u0, u1] restores full-width lifting.  Must be called after _init and before
// any pull_row().  See idwt_2d_state::col_lo / col_hi comments.
void idwt_2d_state_set_col_range(idwt_2d_state *s, int32_t col_lo, int32_t col_hi);

// Restrict output row range to [row_lo, row_hi).  Passing the default [v0, v1]
// restores full-height decoding.  Must be called after _init and before any
// pull_row().  Caller is responsible for applying the 9/7 filter-support
// margin (see j2k_tile_component::set_line_decode_row_range) so the cascade's
// top-edge dependencies are preserved.
void idwt_2d_state_set_row_range(idwt_2d_state *s, int32_t row_lo, int32_t row_hi);

// Sub-range horizontal 1D IDWT.  col_lo / col_hi are target-valid-output
// columns in ROW coords [0, u1-u0].  When the caller passes the full range
// (col_lo <= 0 && col_hi >= u1 - u0) this function is a thin wrapper around
// idwt_1d_row_inplace() and produces byte-identical output — that is the
// default JPIP-unaware path.  When a narrower range is requested, a scalar
// sub-range lifter runs over [col_lo - pse, col_hi + pse] (clamped to
// [0, u1 - u0]) and skips the columns outside that window.  The caller
// guarantees the samples outside the processing range are zero on entry and
// must remain zero on exit; for JPIP this is true because interleave zeros
// all samples outside the precinct-populated region.
void idwt_1d_row_inplace_range(sprec_t *row, int32_t left, int32_t right,
                               int32_t u0, int32_t u1, uint8_t transformation,
                               int32_t col_lo, int32_t col_hi);

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

// ── Inline helpers for the IDWT ring state machine ────────────────────────
// These are called millions of times per frame from cascade() and must be
// inlined to eliminate function-call overhead on the hot decode path.

// Pointer to the row buffer for physical row r (ring, top-PSE, or bot-PSE).
static inline sprec_t *idwt_rptr(const idwt_2d_state *s, int32_t r) {
  if (r >= s->v0 && r < s->v1)
    return static_cast<sprec_t *>(s->ring_buf)
           + static_cast<ptrdiff_t>(r & (IDWT_STATE_RING_DEPTH - 1)) * s->slot_stride
           + IDWT_RING_PSE_LEFT;
  if (r < s->v0)
    return static_cast<sprec_t *>(s->top_pse_buf) + static_cast<ptrdiff_t>(s->v0 - 1 - r) * s->stride;
  return static_cast<sprec_t *>(s->bot_pse_buf) + static_cast<ptrdiff_t>(r - s->v1) * s->stride;
}

// d_level for physical row r (-1 = unfilled / out of range).
static inline int8_t idwt_get_dl(const idwt_2d_state *s, int32_t r) {
  if (r >= s->v0 && r < s->v1) {
    if (r < s->ring_origin || r >= s->ring_origin + IDWT_STATE_RING_DEPTH) return -1;
    return s->d_level[r & (IDWT_STATE_RING_DEPTH - 1)];
  }
  if (r >= s->v0 - s->top_pse && r < s->v0) return s->top_dlevel[s->v0 - 1 - r];
  if (r >= s->v1 && r < s->v1 + s->bottom_pse) return s->bot_dlevel[r - s->v1];
  return -1;
}

// Set d_level for physical row r.
static inline void idwt_set_dl(idwt_2d_state *s, int32_t r, int8_t lv) {
  if (r >= s->v0 && r < s->v1) {
    s->d_level[r & (IDWT_STATE_RING_DEPTH - 1)] = lv;
    return;
  }
  if (r >= s->v0 - s->top_pse && r < s->v0) { s->top_dlevel[s->v0 - 1 - r] = lv; return; }
  if (r >= s->v1 && r < s->v1 + s->bottom_pse) { s->bot_dlevel[r - s->v1] = lv; }
}

// True if physical row r is tracked as all-zero (absent-precinct optimisation).
static inline bool idwt_is_zero(const idwt_2d_state *s, int32_t r) {
  if (r >= s->v0 && r < s->v1) return s->row_zero[r & (IDWT_STATE_RING_DEPTH - 1)];
  if (r >= s->v0 - s->top_pse && r < s->v0) return s->top_pse_zero[s->v0 - 1 - r];
  if (r >= s->v1 && r < s->v1 + s->bottom_pse) return s->bot_pse_zero[r - s->v1];
  return true;
}

static inline void idwt_set_zero(idwt_2d_state *s, int32_t r, bool z) {
  if (r >= s->v0 && r < s->v1) { s->row_zero[r & (IDWT_STATE_RING_DEPTH - 1)] = z; return; }
  if (r >= s->v0 - s->top_pse && r < s->v0) { s->top_pse_zero[s->v0 - 1 - r] = z; return; }
  if (r >= s->v1 && r < s->v1 + s->bottom_pse) { s->bot_pse_zero[r - s->v1] = z; }
}

// Physical source row for PSE position p via periodic symmetric extension.
static inline int32_t idwt_pse_source(int32_t p, int32_t v0, int32_t v1) {
  return v0 + PSEo(p, v0, v1);
}

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

// Planar variant: receives the horizontally-filtered row as separate LP/HP
// planes (lp_row[j] = LP sample j, hp_row[j] = HP sample j), skipping the
// interleave→deinterleave round trip.  Optional — only called when the state
// owner sets fdwt_2d_state::put_planes and the planar kernel guards hold.
// On the int32 pipe (use_i32) the plane bytes are int32_t behind the sprec_t
// pointers, same as fdwt_row_sink_fn's interleaved_row.
typedef void (*fdwt_row_sink_planes_fn)(void *ctx, bool is_hp, int32_t abs_phys_row, const sprec_t *lp_row,
                                        const sprec_t *hp_row);

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
  // When true: ring_buf / top_pse_buf / bot_pse_buf / horiz_tmp contain
  // int32_t (reinterpret_cast'd from sprec_t* storage).  Lifting dispatches
  // to fdwt_{1d_filtr_rev53,rev_ver_{hp,lp}_step}_i32_* primitives instead of
  // float ones.  Only valid for transformation == 1 (rev 5/3).  Set at init.
  bool    use_i32;

  // ── PSE scratch ───────────────────────────────────────────────────────────
  void *top_pse_buf;         // top_pse    × stride sprec_t (or int32_t if use_i32)
  void *bot_pse_buf;         // bottom_pse × stride sprec_t (or int32_t if use_i32)
  int8_t   top_dlevel[4];    // d_level per top-PSE slot (-1 = unfilled)
  int8_t   bot_dlevel[4];    // d_level per bot-PSE slot (-1 = unfilled)

  // ── sliding ring ──────────────────────────────────────────────────────────
  void *ring_buf;                             // FDWT_STATE_RING_DEPTH × stride
  int32_t  ring_origin;
  int8_t   d_level[FDWT_STATE_RING_DEPTH];   // 0=raw, 1=step1, 2=step2, -1=unused

  // ── horizontal-DWT temp buffer ────────────────────────────────────────────
  // Size: horiz_left + stride + horiz_right + SIMD_PADDING
  void *horiz_tmp;

  // ── cursors ───────────────────────────────────────────────────────────────
  int32_t next_in;    // next row to accept via push_row() [v0, v1]
  int32_t next_emit;  // next completed row waiting to be emitted [v0, v1)

  // ── sink ──────────────────────────────────────────────────────────────────
  fdwt_row_sink_fn put_row;
  void            *sink_ctx;

  // ── planar horizontal fast path (9/7 float + rev53 int32) ─────────────────
  // put_planes: optional planes sink set by the state owner (nullptr = off).
  // planar_lp/planar_hp: per-state plane scratch (allocated by init when the
  // geometry is planar-eligible; views into one allocation).  emit_ready_f
  // takes the planar path only when put_planes and planar_lp are non-null,
  // and additionally requires use_i32 for rev53 (the int32 opt-in happens
  // after init; a float rev53 state keeps the interleaved path).
  fdwt_row_sink_planes_fn put_planes;
  sprec_t *planar_lp;
  sprec_t *planar_hp;
};

// Initialise (allocates ring_buf, PSE buffers, horiz_tmp).
// `use_i32`: pipe int32 storage through the state.  Only valid for
// transformation == 1 (rev 5/3); ignored otherwise.  When true, callers
// must pass int32_t* (reinterpret_cast'd) to push_row, and the sink
// callback receives interleaved_row as int32_t* (reinterpret_cast back).
void fdwt_2d_state_init(fdwt_2d_state *s,
                        int32_t u0, int32_t u1, int32_t v0, int32_t v1,
                        uint8_t transformation, bool use_i32,
                        fdwt_row_sink_fn sink_fn, void *sink_ctx);

// Free buffers allocated by fdwt_2d_state_init.
void fdwt_2d_state_free(fdwt_2d_state *s);

// Push one input row in[0..u1-u0-1].  May trigger sink callbacks.
// For use_i32 states, the caller reinterpret_casts an int32_t* row.
void fdwt_2d_state_push_row(fdwt_2d_state *s, const sprec_t *in);

// Finalise: fill bottom PSE, run remaining cascade, emit all pending rows.
// Must be called after the last push_row().
void fdwt_2d_state_flush(fdwt_2d_state *s);