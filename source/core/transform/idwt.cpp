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

#include <cstring>
#include <cmath>
#include <utility>
#include "dwt.hpp"
#include "utils.hpp"
#if defined(OPENHTJ2K_ENABLE_WASM_SIMD)
  #include <wasm_simd128.h>
#endif
// Forward declarations for static ATK filter functions defined later in this file.
[[maybe_unused]] static void idwt_1d_filtr_irrev53_fixed(sprec_t *X, int32_t left, int32_t u_i0,
                                                         int32_t u_i1);
[[maybe_unused]] static void idwt_irrev53_ver_sr_fixed(sprec_t *in, int32_t u0, int32_t u1,
                                                       int32_t v0, int32_t v1, int32_t stride,
                                                       sprec_t *pse_scratch, sprec_t **buf_scratch);
#if defined(OPENHTJ2K_ENABLE_WASM_SIMD)
static idwt_1d_filtd_func_fixed idwt_1d_filtr_fixed[2] = {idwt_1d_filtr_irrev97_fixed_wasm,
                                                           idwt_1d_filtr_rev53_fixed_wasm};
static idwt_ver_filtd_func_fixed idwt_ver_sr_fixed[2]  = {idwt_irrev_ver_sr_fixed_wasm,
                                                           idwt_rev_ver_sr_fixed_wasm};
static idwt_1d_filtd_func_fixed idwt_1d_filtr_irrev53_fn = idwt_1d_filtr_irrev53_fixed;
static idwt_ver_filtd_func_fixed idwt_ver_irrev53_fn    = idwt_irrev53_ver_sr_fixed;
typedef void (*adv_irrev_step_fn)(int32_t, float *, float *, float *, float);
static adv_irrev_step_fn adv_irrev_ver_step_fn = idwt_irrev_ver_step_fixed_wasm;
typedef void (*adv_rev_step_fn)(int32_t, const float *, const float *, float *);
static adv_rev_step_fn adv_rev_ver_lp_step_fn = idwt_rev_ver_lp_step_wasm;
static adv_rev_step_fn adv_rev_ver_hp_step_fn = idwt_rev_ver_hp_step_wasm;
#elif defined(OPENHTJ2K_ENABLE_AVX512)
static idwt_1d_filtd_func_fixed idwt_1d_filtr_fixed[2] = {idwt_1d_filtr_irrev97_fixed_avx512,
                                                           idwt_1d_filtr_rev53_fixed_avx512};
static idwt_ver_filtd_func_fixed idwt_ver_sr_fixed[2]  = {idwt_irrev_ver_sr_fixed_avx512,
                                                           idwt_rev_ver_sr_fixed_avx512};
static idwt_1d_filtd_func_fixed idwt_1d_filtr_irrev53_fn = idwt_1d_filtr_irrev53_fixed_avx512;
static idwt_ver_filtd_func_fixed idwt_ver_irrev53_fn    = idwt_irrev53_ver_sr_fixed_avx512;
typedef void (*adv_irrev_step_fn)(int32_t, float *, float *, float *, float);
static adv_irrev_step_fn adv_irrev_ver_step_fn = idwt_irrev_ver_step_fixed_avx512;
typedef void (*adv_rev_step_fn)(int32_t, const float *, const float *, float *);
static adv_rev_step_fn adv_rev_ver_lp_step_fn = idwt_rev_ver_lp_step_avx512;
static adv_rev_step_fn adv_rev_ver_hp_step_fn = idwt_rev_ver_hp_step_avx512;
#elif defined(OPENHTJ2K_ENABLE_ARM_NEON)
static idwt_1d_filtd_func_fixed idwt_1d_filtr_fixed[2] = {idwt_1d_filtr_irrev97_fixed_neon,
                                                          idwt_1d_filtr_rev53_fixed_neon};
static idwt_ver_filtd_func_fixed idwt_ver_sr_fixed[2]  = {idwt_irrev_ver_sr_fixed_neon,
                                                          idwt_rev_ver_sr_fixed_neon};
static idwt_1d_filtd_func_fixed idwt_1d_filtr_irrev53_fn = idwt_1d_filtr_irrev53_fixed_neon;
static idwt_ver_filtd_func_fixed idwt_ver_irrev53_fn    = idwt_irrev53_ver_sr_fixed_neon;
typedef void (*adv_irrev_step_fn)(int32_t, float *, float *, float *, float);
static adv_irrev_step_fn adv_irrev_ver_step_fn = idwt_irrev_ver_step_fixed_neon;
typedef void (*adv_rev_step_fn)(int32_t, const float *, const float *, float *);
static adv_rev_step_fn adv_rev_ver_lp_step_fn = idwt_rev_ver_lp_step_neon;
static adv_rev_step_fn adv_rev_ver_hp_step_fn = idwt_rev_ver_hp_step_neon;
#elif defined(OPENHTJ2K_ENABLE_AVX2)
static idwt_1d_filtd_func_fixed idwt_1d_filtr_fixed[2] = {idwt_1d_filtr_irrev97_fixed_avx2,
                                                          idwt_1d_filtr_rev53_fixed_avx2};
static idwt_ver_filtd_func_fixed idwt_ver_sr_fixed[2]  = {idwt_irrev_ver_sr_fixed_avx2,
                                                          idwt_rev_ver_sr_fixed_avx2};
static idwt_1d_filtd_func_fixed idwt_1d_filtr_irrev53_fn = idwt_1d_filtr_irrev53_fixed_avx2;
static idwt_ver_filtd_func_fixed idwt_ver_irrev53_fn    = idwt_irrev53_ver_sr_fixed_avx2;
typedef void (*adv_irrev_step_fn)(int32_t, float *, float *, float *, float);
static adv_irrev_step_fn adv_irrev_ver_step_fn = idwt_irrev_ver_step_fixed_avx2;
typedef void (*adv_rev_step_fn)(int32_t, const float *, const float *, float *);
static adv_rev_step_fn adv_rev_ver_lp_step_fn = idwt_rev_ver_lp_step_avx2;
static adv_rev_step_fn adv_rev_ver_hp_step_fn = idwt_rev_ver_hp_step_avx2;
#else
static idwt_1d_filtd_func_fixed idwt_1d_filtr_fixed[2] = {idwt_1d_filtr_irrev97_fixed,
                                                          idwt_1d_filtr_rev53_fixed};
static idwt_ver_filtd_func_fixed idwt_ver_sr_fixed[2]  = {idwt_irrev_ver_sr_fixed, idwt_rev_ver_sr_fixed};
static idwt_1d_filtd_func_fixed idwt_1d_filtr_irrev53_fn = idwt_1d_filtr_irrev53_fixed;
static idwt_ver_filtd_func_fixed idwt_ver_irrev53_fn    = idwt_irrev53_ver_sr_fixed;
static void adv_irrev_ver_step_scalar(int32_t n, float *prev, float *next, float *tgt, float coeff) {
  for (int32_t i = 0; i < n; ++i) tgt[i] -= coeff * (prev[i] + next[i]);
}
typedef void (*adv_irrev_step_fn)(int32_t, float *, float *, float *, float);
static adv_irrev_step_fn adv_irrev_ver_step_fn = adv_irrev_ver_step_scalar;
static void adv_rev_ver_lp_step_scalar(int32_t n, const float *prev, const float *next, float *tgt) {
  for (int32_t i = 0; i < n; ++i) tgt[i] -= floorf((prev[i] + next[i] + 2.0f) * 0.25f);
}
static void adv_rev_ver_hp_step_scalar(int32_t n, const float *prev, const float *next, float *tgt) {
  for (int32_t i = 0; i < n; ++i) tgt[i] += floorf((prev[i] + next[i]) * 0.5f);
}
typedef void (*adv_rev_step_fn)(int32_t, const float *, const float *, float *);
static adv_rev_step_fn adv_rev_ver_lp_step_fn = adv_rev_ver_lp_step_scalar;
static adv_rev_step_fn adv_rev_ver_hp_step_fn = adv_rev_ver_hp_step_scalar;
#endif

void idwt_1d_filtr_irrev97_fixed(sprec_t *X, const int32_t left, const int32_t u_i0, const int32_t u_i1) {
  const auto i0        = static_cast<int32_t>(u_i0);
  const auto i1        = static_cast<int32_t>(u_i1);
  const int32_t start  = i0 / 2;
  const int32_t stop   = i1 / 2;
  const int32_t offset = left - i0 % 2;

  float sum;
  /* K and 1/K have been already done by dequantization */
  for (int32_t n = -2 + offset, i = start - 1; i < stop + 2; i++, n += 2) {
    sum = X[n - 1];
    sum += X[n + 1];
    X[n] = X[n] - fD * sum;
  }
  int16_t a[16];
  memcpy(a, X - 2 + offset, sizeof(int16_t) * 16);
  for (int32_t n = -2 + offset, i = start - 1; i < stop + 1; i++, n += 2) {
    sum = X[n];
    sum += X[n + 2];
    X[n + 1] = X[n + 1] - fC * sum;
  }
  for (int32_t n = 0 + offset, i = start; i < stop + 1; i++, n += 2) {
    sum = X[n - 1];
    sum += X[n + 1];
    X[n] = X[n] - fB * sum;
  }
  for (int32_t n = 0 + offset, i = start; i < stop; i++, n += 2) {
    sum = X[n];
    sum += X[n + 2];
    X[n + 1] = X[n + 1] - fA * sum;
  }
}

void idwt_1d_filtr_rev53_fixed(sprec_t *X, const int32_t left, const int32_t u_i0, const int32_t u_i1) {
  const auto i0        = static_cast<int32_t>(u_i0);
  const auto i1        = static_cast<int32_t>(u_i1);
  const int32_t start  = i0 / 2;
  const int32_t stop   = i1 / 2;
  const int32_t offset = left - i0 % 2;

  for (int32_t n = 0 + offset, i = start; i < stop + 1; ++i, n += 2) {
    X[n] -= floorf((X[n - 1] + X[n + 1] + 2) * 0.25f);
  }

  for (int32_t n = 0 + offset, i = start; i < stop; ++i, n += 2) {
    X[n + 1] += floorf((X[n] + X[n + 2]) * 0.5f);
  }
}

// ATK irreversible 5/3 synthesis (from OpenJPH gen_irv_horz_syn pattern):
// Analysis (forward, steps in REVERSE order, alternating LP/HP due to buffer swap):
//   step[1] A=-0.5: HP[k] += -0.5*(LP[k]+LP[k+1])   [HP modified first]
//   step[0] A=+0.25: LP[k] += 0.25*(HP_mod[k-1]+HP_mod[k])  [LP modified using modified HP]
// Synthesis (steps in FORWARD order, negated, LP first, then HP using modified LP):
//   Step 1 (undo step[0]): LP[k] -= 0.25*(HP[k-1]+HP[k])   [modifies LP using original HP]
//   Step 2 (undo step[1]): HP[k] += 0.5*(LP_mod[k]+LP_mod[k+1])  [modifies HP using modified LP]
[[maybe_unused]] static void idwt_1d_filtr_irrev53_fixed(sprec_t *X, const int32_t left,
                                                        const int32_t u_i0, const int32_t u_i1) {
  const int32_t lp_count  = ceil_int(u_i1, 2) - ceil_int(u_i0, 2);  // LP sample count
  const int32_t hp_count  = u_i1 / 2 - u_i0 / 2;                    // HP sample count
  const int32_t offset    = left - u_i0 % 2;                         // base for HP step loop
  const int32_t lp_offset = offset + (u_i0 % 2) * 2;                // first LP sample position
  // Step 1: LP -= 0.25*(HP_left + HP_right)  [using original HP values]
  for (int32_t k = 0, n = lp_offset; k < lp_count; ++k, n += 2)
    X[n] -= 0.25f * (X[n - 1] + X[n + 1]);
  // Step 2: HP += 0.5*(LP_mod_left + LP_mod_right)  [using LP values modified in step 1]
  for (int32_t k = 0, n = offset; k < hp_count; ++k, n += 2)
    X[n + 1] += 0.5f * (X[n] + X[n + 2]);
}

// In-place 1-D IDWT for all rows.
// Operates directly on in[-left..width+SIMD_PADDING-1] without copying to/from an external buffer.
// Precondition: those memory locations are within the tile allocation. The j2k_subband /
// j2k_resolution allocators add DWT_LEFT_SLACK floats before the first row and DWT_RIGHT_SLACK
// floats after the last row of the buffer (border slack), which is what makes the precondition
// hold for the first/last rows of a tile. Interior rows always satisfied it because the slack
// regions overlap adjacent rows' data — save/restore below preserves them.
static inline void idwt_1d_sr_inplace(sprec_t *in, const int32_t left, const int32_t right,
                                      const int32_t i0, const int32_t i1,
                                      const uint8_t transformation) {
  const int32_t width = i1 - i0;
  // Save regions that the filter will temporarily overwrite with PSE data or SIMD tail writes.
  // left_save[8] covers the dwt_pse_fill_inplace_simd write window (8 floats per side).
  // right_save[SIMD_PADDING] (32 floats) also accommodates tail writes from the widest SIMD
  // register (AVX-512 = 16 floats per ZMM), which can write up to 15 floats past width.
  sprec_t left_save[8];
  sprec_t right_save[SIMD_PADDING];
  for (int32_t i = 0; i < 8; ++i) left_save[i] = in[-8 + i];
  for (int32_t i = 0; i < SIMD_PADDING; ++i) right_save[i] = in[width + i];
  // Fill left PSE into in[-left..-1] and right PSE into in[width..width+right-1].
  if (width >= 9) {
    dwt_pse_fill_inplace_simd(in, width);
  } else {
    for (int32_t i = 1; i <= left; ++i)
      in[-i] = in[PSEo(i0 - i, i0, i1)];
    for (int32_t i = 1; i <= right; ++i)
      in[width + i - 1] = in[PSEo(i1 - i0 + i - 1 + i0, i0, i1)];
  }
  // Filter in-place: in-left is the extended buffer (left PSE | data | right PSE).
  if (transformation < 2)
    idwt_1d_filtr_fixed[transformation](in - left, left, i0, i1);
  else
    idwt_1d_filtr_irrev53_fn(in - left, left, i0, i1);
  // Restore the saved regions (IDWT output is in in[0..width-1], boundary regions are scratch).
  for (int32_t i = 0; i < 8; ++i) in[-8 + i] = left_save[i];
  for (int32_t i = 0; i < SIMD_PADDING; ++i) in[width + i] = right_save[i];
}

static void idwt_hor_sr_fixed(sprec_t *in, const int32_t u0, const int32_t u1, const int32_t v0,
                              const int32_t v1, const uint8_t transformation, const int32_t stride) {
  constexpr int32_t num_pse_i0[2][2] = {{3, 1}, {4, 2}};
  constexpr int32_t num_pse_i1[2][2] = {{4, 2}, {3, 1}};
  // ATK (transformation>=2) uses same PSE lengths as rev53 (CDF 5/3, 2-step filter)
  const uint8_t eff = (transformation < 2) ? transformation : 1;
  const int32_t left  = num_pse_i0[u0 % 2][eff];
  const int32_t right = num_pse_i1[u1 % 2][eff];

  if (u0 == u1 - 1) {
    // one sample case
    for (int32_t row = 0; row < v1 - v0; ++row) {
      if (u0 % 2 != 0 && transformation == 1) {
        in[row * stride] = static_cast<sprec_t>(in[row * stride] / 2.0f);
      }
    }
  } else {
    // All rows use the in-place horizontal IDWT.  The j2k_subband / j2k_resolution
    // allocators add DWT_LEFT_SLACK + DWT_RIGHT_SLACK floats of border slack to
    // the buffer (and offset the user-visible i_samples pointer by DWT_LEFT_SLACK),
    // so the first row's in[-left..-1] and the last row's in[width..width+SIMD_PADDING-1]
    // both fall inside valid memory.  Interior rows are handled by save/restore inside
    // idwt_1d_sr_inplace.  This eliminates the per-tile-per-level Yext allocation and
    // the redundant memcpy of every row that the legacy copy path performed.
    const int32_t nrows = v1 - v0;
    for (int32_t row = 0; row < nrows; ++row) {
      idwt_1d_sr_inplace(in, left, right, u0, u1, transformation);
      in += stride;
    }
  }
}

void idwt_irrev_ver_sr_fixed(sprec_t *in, const int32_t u0, const int32_t u1, const int32_t v0,
                             const int32_t v1, const int32_t stride, sprec_t *pse_scratch,
                             sprec_t **buf_scratch) {
  constexpr int32_t num_pse_i0[2] = {3, 4};
  constexpr int32_t num_pse_i1[2] = {4, 3};
  const int32_t top               = num_pse_i0[v0 % 2];
  const int32_t bottom            = num_pse_i1[v1 % 2];
  if (v0 == v1 - 1) {
    // one sample case
    for (int32_t col = 0; col < u1 - u0; ++col) {
      // in[col] >>= (v0 % 2 == 0) ? 0 : 0;
    }
  } else {
    const int32_t len = round_up(stride, SIMD_PADDING);
    sprec_t **buf     = buf_scratch;
    for (int32_t i = 1; i <= top; ++i) {
      // buf[top - i] =
      //     static_cast<sprec_t *>(aligned_mem_alloc(sizeof(sprec_t) * static_cast<size_t>(len), 32));
      buf[top - i] = pse_scratch + (i - 1) * len;
      memcpy(buf[top - i], &in[PSEo(v0 - i, v0, v1) * stride],
             sizeof(sprec_t) * static_cast<size_t>(stride));
    }
    for (int32_t row = 0; row < v1 - v0; ++row) {
      buf[top + row] = &in[row * stride];
    }
    for (int32_t i = 1; i <= bottom; i++) {
      // buf[top + (v1 - v0) + i - 1] =
      //     static_cast<sprec_t *>(aligned_mem_alloc(sizeof(sprec_t) * static_cast<size_t>(len), 32));
      buf[top + (v1 - v0) + i - 1] = pse_scratch + (top + i - 1) * len;
      memcpy(buf[top + (v1 - v0) + i - 1], &in[PSEo(v1 - v0 + i - 1 + v0, v0, v1) * stride],
             sizeof(sprec_t) * static_cast<size_t>(stride));
    }
    const int32_t start  = v0 / 2;
    const int32_t stop   = v1 / 2;
    const int32_t offset = top - v0 % 2;

    const int32_t width = u1 - u0;
    for (int32_t cs = 0; cs < width; cs += DWT_VERT_STRIP) {
      const int32_t ce = (cs + DWT_VERT_STRIP < width) ? cs + DWT_VERT_STRIP : width;
      for (int32_t n = -2 + offset, i = start - 1; i < stop + 2; i++, n += 2) {
        for (int32_t col = cs; col < ce; ++col) {
          buf[n][col] -= fD * (buf[n - 1][col] + buf[n + 1][col]);
        }
      }
      for (int32_t n = -2 + offset, i = start - 1; i < stop + 1; i++, n += 2) {
        for (int32_t col = cs; col < ce; ++col) {
          buf[n + 1][col] -= fC * (buf[n][col] + buf[n + 2][col]);
        }
      }
      for (int32_t n = 0 + offset, i = start; i < stop + 1; i++, n += 2) {
        for (int32_t col = cs; col < ce; ++col) {
          buf[n][col] -= fB * (buf[n - 1][col] + buf[n + 1][col]);
        }
      }
      for (int32_t n = 0 + offset, i = start; i < stop; i++, n += 2) {
        for (int32_t col = cs; col < ce; ++col) {
          buf[n + 1][col] -= fA * (buf[n][col] + buf[n + 2][col]);
        }
      }
    }

    // for (int32_t i = 1; i <= top; ++i) {
    //   aligned_mem_free(buf[top - i]);
    // }
    // for (int32_t i = 1; i <= bottom; i++) {
    //   aligned_mem_free(buf[top + (v1 - v0) + i - 1]);
    // }
  }
}

void idwt_rev_ver_sr_fixed(sprec_t *in, const int32_t u0, const int32_t u1, const int32_t v0,
                           const int32_t v1, const int32_t stride, sprec_t *pse_scratch,
                           sprec_t **buf_scratch) {
  constexpr int32_t num_pse_i0[2] = {1, 2};
  constexpr int32_t num_pse_i1[2] = {2, 1};
  const int32_t top               = num_pse_i0[v0 % 2];
  const int32_t bottom            = num_pse_i1[v1 % 2];
  if (v0 == v1 - 1 && (v0 % 2)) {
    // one sample case
    for (int32_t col = 0; col < u1 - u0; ++col) {
      in[col] = floorf(in[col] * 0.5f);
    }
  } else {
    const int32_t len = round_up(stride, SIMD_PADDING);
    sprec_t **buf     = buf_scratch;
    for (int32_t i = 1; i <= top; ++i) {
      // buf[top - i] =
      //     static_cast<sprec_t *>(aligned_mem_alloc(sizeof(sprec_t) * static_cast<size_t>(len), 32));
      buf[top - i] = pse_scratch + (i - 1) * len;
      memcpy(buf[top - i], &in[PSEo(v0 - i, v0, v1) * stride],
             sizeof(sprec_t) * static_cast<size_t>(stride));
    }
    for (int32_t row = 0; row < v1 - v0; ++row) {
      buf[top + row] = &in[row * stride];
    }
    for (int32_t i = 1; i <= bottom; i++) {
      // buf[top + (v1 - v0) + i - 1] =
      //     static_cast<sprec_t *>(aligned_mem_alloc(sizeof(sprec_t) * static_cast<size_t>(len), 32));
      buf[top + (v1 - v0) + i - 1] = pse_scratch + (top + i - 1) * len;
      memcpy(buf[top + (v1 - v0) + i - 1], &in[PSEo(v1 - v0 + i - 1 + v0, v0, v1) * stride],
             sizeof(sprec_t) * static_cast<size_t>(stride));
    }
    const int32_t start  = v0 / 2;
    const int32_t stop   = v1 / 2;
    const int32_t offset = top - v0 % 2;

    const int32_t width = u1 - u0;
    for (int32_t cs = 0; cs < width; cs += DWT_VERT_STRIP) {
      const int32_t ce = (cs + DWT_VERT_STRIP < width) ? cs + DWT_VERT_STRIP : width;
      for (int32_t n = 0 + offset, i = start; i < stop + 1; ++i, n += 2) {
        for (int32_t col = cs; col < ce; ++col) {
          buf[n][col] -= floorf((buf[n - 1][col] + buf[n + 1][col] + 2.0f) * 0.25f);
        }
      }
      for (int32_t n = 0 + offset, i = start; i < stop; ++i, n += 2) {
        for (int32_t col = cs; col < ce; ++col) {
          buf[n + 1][col] += floorf((buf[n][col] + buf[n + 2][col]) * 0.5f);
        }
      }
    }

    // for (int32_t i = 1; i <= top; ++i) {
    //   aligned_mem_free(buf[top - i]);
    // }
    // for (int32_t i = 1; i <= bottom; i++) {
    //   aligned_mem_free(buf[top + (v1 - v0) + i - 1]);
    // }
  }
}

// ATK irreversible 5/3 vertical synthesis (two-step, matching gen_irv_vert_step pattern):
// Step 1: LP rows -= 0.25*(HP_above + HP_below)  [using original HP rows]
// Step 2: HP rows += 0.5*(LP_mod_above + LP_mod_below)  [using LP rows modified in step 1]
static void idwt_irrev53_ver_sr_fixed(sprec_t *in, const int32_t u0, const int32_t u1, const int32_t v0,
                                      const int32_t v1, const int32_t stride, sprec_t *pse_scratch,
                                      sprec_t **buf_scratch) {
  constexpr int32_t num_pse_i0[2] = {1, 2};
  constexpr int32_t num_pse_i1[2] = {2, 1};
  const int32_t top               = num_pse_i0[v0 % 2];
  const int32_t bottom            = num_pse_i1[v1 % 2];
  if (v0 == v1 - 1) {
    // single row: nothing to do (PSE has no valid neighbours for lifting)
  } else {
    const int32_t len = round_up(stride, SIMD_PADDING);
    sprec_t **buf     = buf_scratch;
    for (int32_t i = 1; i <= top; ++i) {
      buf[top - i] = pse_scratch + (i - 1) * len;
      memcpy(buf[top - i], &in[PSEo(v0 - i, v0, v1) * stride],
             sizeof(sprec_t) * static_cast<size_t>(stride));
    }
    for (int32_t row = 0; row < v1 - v0; ++row) buf[top + row] = &in[row * stride];
    for (int32_t i = 1; i <= bottom; i++) {
      buf[top + (v1 - v0) + i - 1] = pse_scratch + (top + i - 1) * len;
      memcpy(buf[top + (v1 - v0) + i - 1], &in[PSEo(v1 - v0 + i - 1 + v0, v0, v1) * stride],
             sizeof(sprec_t) * static_cast<size_t>(stride));
    }
    const int32_t lp_count = ceil_int(v1, 2) - ceil_int(v0, 2);  // LP row count
    const int32_t hp_count = v1 / 2 - v0 / 2;                    // HP row count
    const int32_t offset   = top - v0 % 2;                        // base for HP step loop
    const int32_t lp_n0    = top + v0 % 2;                        // first LP row index
    const int32_t width    = u1 - u0;
    for (int32_t cs = 0; cs < width; cs += DWT_VERT_STRIP) {
      const int32_t ce = (cs + DWT_VERT_STRIP < width) ? cs + DWT_VERT_STRIP : width;
      // Step 1: LP -= 0.25*(HP_above + HP_below)  [original HP values]
      for (int32_t k = 0, n = lp_n0; k < lp_count; ++k, n += 2) {
        for (int32_t col = cs; col < ce; ++col)
          buf[n][col] -= 0.25f * (buf[n - 1][col] + buf[n + 1][col]);
      }
      // Step 2: HP += 0.5*(LP_mod_above + LP_mod_below)  [modified LP from step 1]
      for (int32_t k = 0, n = offset; k < hp_count; ++k, n += 2) {
        for (int32_t col = cs; col < ce; ++col)
          buf[n + 1][col] += 0.5f * (buf[n][col] + buf[n + 2][col]);
      }
    }
  }
}

static void idwt_2d_interleave_fixed(sprec_t *buf, sprec_t *LL, sprec_t *HL, sprec_t *LH, sprec_t *HH,
                                     int32_t u0, int32_t u1, int32_t v0, int32_t v1, const int32_t stride) {
  const int32_t v_offset   = v0 % 2;
  const int32_t u_offset   = u0 % 2;
  sprec_t *sp[4]           = {LL, HL, LH, HH};
  const int32_t vstart[4]  = {ceil_int(v0, 2), ceil_int(v0, 2), v0 / 2, v0 / 2};
  const int32_t vstop[4]   = {ceil_int(v1, 2), ceil_int(v1, 2), v1 / 2, v1 / 2};
  const int32_t ustart[4]  = {ceil_int(u0, 2), u0 / 2, ceil_int(u0, 2), u0 / 2};
  const int32_t ustop[4]   = {ceil_int(u1, 2), u1 / 2, ceil_int(u1, 2), u1 / 2};
  const int32_t voffset[4] = {v_offset, v_offset, 1 - v_offset, 1 - v_offset};
  const int32_t uoffset[4] = {u_offset, 1 - u_offset, u_offset, 1 - u_offset};
  const int32_t stride2[4] = {round_up(ustop[0] - ustart[0], 32), round_up(ustop[1] - ustart[1], 32),
                              round_up(ustop[2] - ustart[2], 32), round_up(ustop[3] - ustart[3], 32)};

#if defined(OPENHTJ2K_ENABLE_WASM_SIMD)
  // WASM-SIMD interleave: use wasm_i32x4_shuffle instead of vzip1q/vzip2q.
  auto wasm_interleave_pair = [](sprec_t *buf, sprec_t *bp0, sprec_t *bp1, int32_t s0, int32_t s1,
                                  int32_t common_len, int32_t vstart_b, int32_t vstop_b, int32_t voffset_b,
                                  int32_t stride) {
    for (int32_t v = 0, vb = vstart_b; vb < vstop_b; ++vb, ++v) {
      sprec_t *dp    = buf + (2 * v + voffset_b) * stride;
      size_t len     = static_cast<size_t>(common_len);
      sprec_t *line0 = bp0 + static_cast<ptrdiff_t>(v) * s0;
      sprec_t *line1 = bp1 + static_cast<ptrdiff_t>(v) * s1;
      for (; len >= 8; len -= 8) {
        v128_t a0 = wasm_v128_load(line0);     v128_t b0 = wasm_v128_load(line1);
        v128_t a1 = wasm_v128_load(line0 + 4); v128_t b1 = wasm_v128_load(line1 + 4);
        wasm_v128_store(dp,      wasm_i32x4_shuffle(a0, b0, 0, 4, 1, 5));
        wasm_v128_store(dp + 4,  wasm_i32x4_shuffle(a0, b0, 2, 6, 3, 7));
        wasm_v128_store(dp + 8,  wasm_i32x4_shuffle(a1, b1, 0, 4, 1, 5));
        wasm_v128_store(dp + 12, wasm_i32x4_shuffle(a1, b1, 2, 6, 3, 7));
        line0 += 8; line1 += 8; dp += 16;
      }
      for (; len > 0; --len) { *dp++ = *line0++; *dp++ = *line1++; }
    }
  };
  {
    const int32_t len0 = ustop[0] - ustart[0], len1 = ustop[1] - ustart[1];
    const int32_t common_len = len0 < len1 ? len0 : len1;
    sprec_t *bp0 = sp[0]; sprec_t *bp1 = sp[1]; int32_t s0 = stride2[0], s1 = stride2[1];
    if (uoffset[0] > uoffset[1]) { std::swap(bp0, bp1); std::swap(s0, s1); }
    wasm_interleave_pair(buf, bp0, bp1, s0, s1, common_len, vstart[0], vstop[0], voffset[0], stride);
    for (uint8_t b = 0; b < 2; ++b) {
      if ((ustop[b] - ustart[b]) > common_len) {
        for (int32_t v = 0, vb = vstart[b]; vb < vstop[b]; ++vb, ++v)
          buf[2 * common_len + uoffset[b] + (2 * v + voffset[b]) * stride] =
              sp[b][static_cast<ptrdiff_t>(v) * stride2[b] + common_len];
        break;
      }
    }
  }
  {
    const int32_t len2 = ustop[2] - ustart[2], len3 = ustop[3] - ustart[3];
    const int32_t common_len = len2 < len3 ? len2 : len3;
    sprec_t *bp2 = sp[2]; sprec_t *bp3 = sp[3]; int32_t s2 = stride2[2], s3 = stride2[3];
    if (uoffset[2] > uoffset[3]) { std::swap(bp2, bp3); std::swap(s2, s3); }
    wasm_interleave_pair(buf, bp2, bp3, s2, s3, common_len, vstart[2], vstop[2], voffset[2], stride);
    for (uint8_t b = 2; b < 4; ++b) {
      if ((ustop[b] - ustart[b]) > common_len) {
        for (int32_t v = 0, vb = vstart[b]; vb < vstop[b]; ++vb, ++v)
          buf[2 * common_len + uoffset[b] + (2 * v + voffset[b]) * stride] =
              sp[b][static_cast<ptrdiff_t>(v) * stride2[b] + common_len];
        break;
      }
    }
  }
#elif defined(OPENHTJ2K_ENABLE_ARM_NEON)
  {
    // Band pair {LL, HL}: NEON zip for common length, scalar for extra sample.
    const int32_t len0       = ustop[0] - ustart[0];
    const int32_t len1       = ustop[1] - ustart[1];
    const int32_t common_len = len0 < len1 ? len0 : len1;
    sprec_t *bp0 = sp[0], *bp1 = sp[1];
    int32_t s0 = stride2[0], s1 = stride2[1];
    if (uoffset[0] > uoffset[1]) {
      std::swap(bp0, bp1);
      std::swap(s0, s1);
    }
    float32x4_t vfirst0, vfirst1, vsecond0, vsecond1;
    for (int32_t v = 0, vb = vstart[0]; vb < vstop[0]; ++vb, ++v) {
      sprec_t *dp    = buf + (2 * v + voffset[0]) * stride;
      size_t len     = static_cast<size_t>(common_len);
      sprec_t *line0 = bp0 + static_cast<ptrdiff_t>(v) * s0;
      sprec_t *line1 = bp1 + static_cast<ptrdiff_t>(v) * s1;
      for (; len >= 8; len -= 8) {
        vfirst0  = vld1q_f32(line0);
        vsecond0 = vld1q_f32(line1);
        vst1q_f32(dp, vzip1q_f32(vfirst0, vsecond0));
        vst1q_f32(dp + 4, vzip2q_f32(vfirst0, vsecond0));
        vfirst1  = vld1q_f32(line0 + 4);
        vsecond1 = vld1q_f32(line1 + 4);
        vst1q_f32(dp + 8, vzip1q_f32(vfirst1, vsecond1));
        vst1q_f32(dp + 12, vzip2q_f32(vfirst1, vsecond1));
        line0 += 8;
        line1 += 8;
        dp += 16;
      }
      for (; len > 0; --len) {
        *dp++ = *line0++;
        *dp++ = *line1++;
      }
    }
    for (uint8_t b = 0; b < 2; ++b) {
      if ((ustop[b] - ustart[b]) > common_len) {
        for (int32_t v = 0, vb = vstart[b]; vb < vstop[b]; ++vb, ++v) {
          buf[2 * common_len + uoffset[b] + (2 * v + voffset[b]) * stride] =
              sp[b][static_cast<ptrdiff_t>(v) * stride2[b] + common_len];
        }
        break;
      }
    }
  }

  {
    // Band pair {LH, HH}: same treatment.
    const int32_t len2       = ustop[2] - ustart[2];
    const int32_t len3       = ustop[3] - ustart[3];
    const int32_t common_len = len2 < len3 ? len2 : len3;
    sprec_t *bp2 = sp[2], *bp3 = sp[3];
    int32_t s2 = stride2[2], s3 = stride2[3];
    if (uoffset[2] > uoffset[3]) {
      std::swap(bp2, bp3);
      std::swap(s2, s3);
    }
    float32x4_t vfirst0, vfirst1, vsecond0, vsecond1;
    for (int32_t v = 0, vb = vstart[2]; vb < vstop[2]; ++vb, ++v) {
      sprec_t *dp    = buf + (2 * v + voffset[2]) * stride;
      size_t len     = static_cast<size_t>(common_len);
      sprec_t *line0 = bp2 + static_cast<ptrdiff_t>(v) * s2;
      sprec_t *line1 = bp3 + static_cast<ptrdiff_t>(v) * s3;
      for (; len >= 8; len -= 8) {
        vfirst0  = vld1q_f32(line0);
        vsecond0 = vld1q_f32(line1);
        vst1q_f32(dp, vzip1q_f32(vfirst0, vsecond0));
        vst1q_f32(dp + 4, vzip2q_f32(vfirst0, vsecond0));
        vfirst1  = vld1q_f32(line0 + 4);
        vsecond1 = vld1q_f32(line1 + 4);
        vst1q_f32(dp + 8, vzip1q_f32(vfirst1, vsecond1));
        vst1q_f32(dp + 12, vzip2q_f32(vfirst1, vsecond1));
        line0 += 8;
        line1 += 8;
        dp += 16;
      }
      for (; len > 0; --len) {
        *dp++ = *line0++;
        *dp++ = *line1++;
      }
    }
    for (uint8_t b = 2; b < 4; ++b) {
      if ((ustop[b] - ustart[b]) > common_len) {
        for (int32_t v = 0, vb = vstart[b]; vb < vstop[b]; ++vb, ++v) {
          buf[2 * common_len + uoffset[b] + (2 * v + voffset[b]) * stride] =
              sp[b][static_cast<ptrdiff_t>(v) * stride2[b] + common_len];
        }
        break;
      }
    }
  }
#elif defined(OPENHTJ2K_TRY_AVX2) && defined(__AVX2__)
  {
    // Band pair {LL, HL}: may have unequal widths (ceil vs floor) when tile width is odd.
    // Use AVX2 zip for the common length, then handle the extra sample if present.
    const int32_t len0       = ustop[0] - ustart[0];
    const int32_t len1       = ustop[1] - ustart[1];
    const int32_t common_len = len0 < len1 ? len0 : len1;
    // Ensure bp0/s0 is the even-position band, bp1/s1 is the odd-position band.
    sprec_t *bp0 = sp[0], *bp1 = sp[1];
    int32_t s0 = stride2[0], s1 = stride2[1];
    if (uoffset[0] > uoffset[1]) {
      std::swap(bp0, bp1);
      std::swap(s0, s1);
    }
    for (int32_t v = 0, vb = vstart[0]; vb < vstop[0]; ++vb, ++v) {
      sprec_t *dp    = buf + (2 * v + voffset[0]) * stride;
      size_t len     = static_cast<size_t>(common_len);
      sprec_t *line0 = bp0 + static_cast<ptrdiff_t>(v) * s0;
      sprec_t *line1 = bp1 + static_cast<ptrdiff_t>(v) * s1;
      // SSE version
      //  for (; len >= 8; len -= 8) {
      //    auto vfirst  = _mm_loadu_si128((__m128i *)line0);
      //    auto vsecond = _mm_loadu_si128((__m128i *)line1);
      //    auto vtmp0   = _mm_unpacklo_epi16(vfirst, vsecond);
      //    auto vtmp1   = _mm_unpackhi_epi16(vfirst, vsecond);
      //    _mm_storeu_si128((__m128i *)dp, vtmp0);
      //    _mm_storeu_si128((__m128i *)(dp + 8), vtmp1);
      //    line0 += 8;
      //    line1 += 8;
      //    dp += 16;
      // }

      // AVX2 version
      __m256i vfirst, vsecond;
      for (; len >= 8; len -= 8) {
        vfirst     = _mm256_loadu_si256((__m256i *)line0);
        vsecond    = _mm256_loadu_si256((__m256i *)line1);
        auto vtmp0 = _mm256_unpacklo_epi32(vfirst, vsecond);
        auto vtmp1 = _mm256_unpackhi_epi32(vfirst, vsecond);

        _mm256_storeu_si256((__m256i *)dp, _mm256_permute2x128_si256(vtmp0, vtmp1, 0x20));
        _mm256_storeu_si256((__m256i *)dp + 1, _mm256_permute2x128_si256(vtmp0, vtmp1, 0x31));
        line0 += 8;
        line1 += 8;
        dp += 16;
      }
      for (; len > 0; --len) {
        *dp++ = *line0++;
        *dp++ = *line1++;
      }
    }
    // Write the one extra sample from the wider band (at most one band has len0 != len1).
    for (uint8_t b = 0; b < 2; ++b) {
      if ((ustop[b] - ustart[b]) > common_len) {
        for (int32_t v = 0, vb = vstart[b]; vb < vstop[b]; ++vb, ++v) {
          buf[2 * common_len + uoffset[b] + (2 * v + voffset[b]) * stride] =
              sp[b][static_cast<ptrdiff_t>(v) * stride2[b] + common_len];
        }
        break;
      }
    }
  }

  {
    // Band pair {LH, HH}: same treatment — AVX2 zip for common length, scalar for extra sample.
    const int32_t len2       = ustop[2] - ustart[2];
    const int32_t len3       = ustop[3] - ustart[3];
    const int32_t common_len = len2 < len3 ? len2 : len3;
    sprec_t *bp2 = sp[2], *bp3 = sp[3];
    int32_t s2 = stride2[2], s3 = stride2[3];
    if (uoffset[2] > uoffset[3]) {
      std::swap(bp2, bp3);
      std::swap(s2, s3);
    }
    for (int32_t v = 0, vb = vstart[2]; vb < vstop[2]; ++vb, ++v) {
      sprec_t *dp    = buf + (2 * v + voffset[2]) * stride;
      size_t len     = static_cast<size_t>(common_len);
      sprec_t *line0 = bp2 + static_cast<ptrdiff_t>(v) * s2;
      sprec_t *line1 = bp3 + static_cast<ptrdiff_t>(v) * s3;
      // SSE version
      //  for (; len >= 8; len -= 8) {
      //    auto vfirst  = _mm_loadu_si128((__m128i *)line0);
      //    auto vsecond = _mm_loadu_si128((__m128i *)line1);
      //    auto vtmp0   = _mm_unpacklo_epi16(vfirst, vsecond);
      //    auto vtmp1   = _mm_unpackhi_epi16(vfirst, vsecond);
      //    _mm_storeu_si128((__m128i *)dp, vtmp0);
      //    _mm_storeu_si128((__m128i *)(dp + 8), vtmp1);
      //    line0 += 8;
      //    line1 += 8;
      //    dp += 16;
      // }

      // AVX2 version
      __m256i vfirst, vsecond;
      for (; len >= 8; len -= 8) {
        vfirst     = _mm256_loadu_si256((__m256i *)line0);
        vsecond    = _mm256_loadu_si256((__m256i *)line1);
        auto vtmp0 = _mm256_unpacklo_epi32(vfirst, vsecond);
        auto vtmp1 = _mm256_unpackhi_epi32(vfirst, vsecond);

        _mm256_storeu_si256((__m256i *)dp, _mm256_permute2x128_si256(vtmp0, vtmp1, 0x20));
        _mm256_storeu_si256((__m256i *)dp + 1, _mm256_permute2x128_si256(vtmp0, vtmp1, 0x31));
        line0 += 8;
        line1 += 8;
        dp += 16;
      }
      for (; len > 0; --len) {
        *dp++ = *line0++;
        *dp++ = *line1++;
      }
    }
    // Write the one extra sample from the wider band (at most one band has len2 != len3).
    for (uint8_t b = 2; b < 4; ++b) {
      if ((ustop[b] - ustart[b]) > common_len) {
        for (int32_t v = 0, vb = vstart[b]; vb < vstop[b]; ++vb, ++v) {
          buf[2 * common_len + uoffset[b] + (2 * v + voffset[b]) * stride] =
              sp[b][static_cast<ptrdiff_t>(v) * stride2[b] + common_len];
        }
        break;
      }
    }
  }
#else
  for (uint8_t b = 0; b < 4; ++b) {
    for (int32_t v = 0, vb = vstart[b]; vb < vstop[b]; ++vb, ++v) {
      sprec_t *line = sp[b] + v * stride2[b];
      for (int32_t u = 0, ub = ustart[b]; ub < ustop[b]; ++ub, ++u) {
        buf[2 * u + uoffset[b] + (2 * v + voffset[b]) * stride] = *(line++);
      }
    }
  }
#endif
}

void idwt_2d_sr_fixed(sprec_t *nextLL, sprec_t *LL, sprec_t *HL, sprec_t *LH, sprec_t *HH, const int32_t u0,
                      const int32_t u1, const int32_t v0, const int32_t v1, const uint8_t transformation,
                      sprec_t *pse_scratch, sprec_t **buf_scratch) {
  const int32_t stride     = round_up(u1 - u0, 32);
  sprec_t *src             = nextLL;
  idwt_2d_interleave_fixed(src, LL, HL, LH, HH, u0, u1, v0, v1, stride);
  idwt_hor_sr_fixed(src, u0, u1, v0, v1, transformation, stride);

  // Vertical DWT (pse_scratch provided by caller, sized for 8 * round_up(stride, SIMD_LEN_I32))
  if (transformation < 2)
    idwt_ver_sr_fixed[transformation](src, u0, u1, v0, v1, stride, pse_scratch, buf_scratch);
  else
    idwt_irrev53_ver_sr_fixed(src, u0, u1, v0, v1, stride, pse_scratch, buf_scratch);
}

void idwt_horz_only_sr_fixed(sprec_t *nextLL, const sprec_t *LL, const sprec_t *H, const int32_t u0,
                              const int32_t u1, const int32_t v0, const int32_t v1,
                              const uint8_t transformation) {
  if (u0 >= u1 || v0 >= v1) return;
  const int32_t width     = u1 - u0;
  const int32_t stride    = round_up(width, 32);
  const int32_t ll_width  = ceil_int(u1, 2) - ceil_int(u0, 2);
  const int32_t h_width   = u1 / 2 - u0 / 2;
  const int32_t ll_stride = round_up(ll_width, 32);
  const int32_t h_stride  = round_up(h_width, 32);
  const int32_t u_par     = u0 % 2;  // 0: LL at even positions; 1: LL at odd positions

  for (int32_t v = v0; v < v1; ++v) {
    const sprec_t *ll_row = LL + static_cast<ptrdiff_t>(v - v0) * ll_stride;
    const sprec_t *h_row  = H + static_cast<ptrdiff_t>(v - v0) * h_stride;
    sprec_t *out          = nextLL + static_cast<ptrdiff_t>(v - v0) * stride;
    for (int32_t k = 0, il = 0, ih = 0; k < width; ++k) {
      if (k % 2 == u_par)
        out[k] = ll_row[il++];
      else
        out[k] = h_row[ih++];
    }
  }
  idwt_hor_sr_fixed(nextLL, u0, u1, v0, v1, transformation, stride);
}

void idwt_vert_only_sr_fixed(sprec_t *nextLL, const sprec_t *LL, const sprec_t *H, const int32_t u0,
                              const int32_t u1, const int32_t v0, const int32_t v1,
                              const uint8_t transformation, sprec_t *pse_scratch, sprec_t **buf_scratch) {
  if (u0 >= u1 || v0 >= v1) return;
  const int32_t width  = u1 - u0;
  const int32_t stride = round_up(width, 32);
  const int32_t height = v1 - v0;
  const int32_t v_par  = v0 % 2;  // 0: LL at even rows; 1: LL at odd rows

  for (int32_t b = 0, il = 0, ih = 0; b < height; ++b) {
    sprec_t *out_row = nextLL + static_cast<ptrdiff_t>(b) * stride;
    if (b % 2 == v_par)
      memcpy(out_row, LL + static_cast<ptrdiff_t>(il++) * stride, sizeof(sprec_t) * static_cast<size_t>(width));
    else
      memcpy(out_row, H + static_cast<ptrdiff_t>(ih++) * stride, sizeof(sprec_t) * static_cast<size_t>(width));
  }
  if (transformation < 2)
    idwt_ver_sr_fixed[transformation](nextLL, u0, u1, v0, v1, stride, pse_scratch, buf_scratch);
  else
    idwt_ver_irrev53_fn(nextLL, u0, u1, v0, v1, stride, pse_scratch, buf_scratch);
}

// In-place 1-D IDWT for rows that have writable PSE scratch space immediately before and
// after the data area (ring buffer slots with IDWT_RING_PSE_LEFT prefix).
// row[-left..-1] and row[width..width+right-1] must be writable (within the slot's PSE areas).
// After this call, row[0..u1-u0-1] holds the filtered output.
void idwt_1d_row_inplace(sprec_t *row, const int32_t left, const int32_t right,
                         const int32_t u0, const int32_t u1, const uint8_t transformation) {
  const int32_t width = u1 - u0;
  if (width >= 9) {
    // Constant-pattern SIMD reflection — see dwt_pse_fill_inplace_simd in dwt.hpp.
    // The slot has IDWT_RING_PSE_LEFT = 8 floats prefix and >= SIMD_PADDING suffix,
    // so the 8-lane writes are always within the slot's scratch area.
    dwt_pse_fill_inplace_simd(row, width);
  } else {
    // Narrow-row scalar fallback (rare in practice; widths < 9 only at small subbands).
    for (int32_t i = 1; i <= left; ++i)
      row[-i] = row[PSEo(u0 - i, u0, u1)];
    for (int32_t i = 1; i <= right; ++i)
      row[width + i - 1] = row[PSEo(u1 - u0 + i - 1 + u0, u0, u1)];
  }
  // Apply horizontal IDWT filter in-place (X = row - left, data at X[left..left+width-1]).
  if (transformation < 2)
    idwt_1d_filtr_fixed[transformation](row - left, left, u0, u1);
  else
    idwt_1d_filtr_irrev53_fn(row - left, left, u0, u1);
}

// =============================================================================
// Streaming 2D IDWT — idwt_2d_state
// =============================================================================
//
// PSE counts: [v0%2][transform] for top, [v1%2][transform] for bottom.
// Indexed [parity][0=irrev97, 1=rev53].
static constexpr int8_t kPseTop[2][2] = {{3, 1}, {4, 2}};
static constexpr int8_t kPseBot[2][2] = {{4, 2}, {3, 1}};

// Max d_level a row reaches before it is output-ready (2 for 9/7, 1 for 5/3).
static inline int8_t max_dl(uint8_t transform) { return (transform == 0) ? 2 : 1; }

// True if physical row r is a low-pass (LP) row at this level.
// LP rows are always at even absolute positions, regardless of v0.
static inline bool is_lp(int32_t r) { return (r & 1) == 0; }

// Physical source row for PSE position p via periodic symmetric extension.
static inline int32_t pse_source(int32_t p, int32_t v0, int32_t v1) {
  return v0 + PSEo(p, v0, v1);
}

// Pointer to the row buffer for physical row r (ring, top-PSE, or bot-PSE).
// Ring slot is r % IDWT_STATE_RING_DEPTH (fixed per row, independent of ring_origin).
// For ring rows, returns a pointer to the DATA area (offset IDWT_RING_PSE_LEFT into the slot),
// which is 32-byte aligned because IDWT_RING_PSE_LEFT=8 floats=32 bytes and ring_buf is
// 32-byte aligned with slot_stride also a multiple of 8 floats.
static sprec_t *rptr(const idwt_2d_state *s, int32_t r) {
  if (r >= s->v0 && r < s->v1)
    return s->ring_buf + static_cast<ptrdiff_t>(r % IDWT_STATE_RING_DEPTH) * s->slot_stride
           + IDWT_RING_PSE_LEFT;
  if (r < s->v0)
    return s->top_pse_buf + static_cast<ptrdiff_t>(s->v0 - 1 - r) * s->stride;
  return s->bot_pse_buf + static_cast<ptrdiff_t>(r - s->v1) * s->stride;
}

// d_level for physical row r (-1 = unfilled / out of range).
static int8_t get_dl(const idwt_2d_state *s, int32_t r) {
  if (r >= s->v0 && r < s->v1) {
    if (r < s->ring_origin || r >= s->ring_origin + IDWT_STATE_RING_DEPTH) return -1;
    return s->d_level[r % IDWT_STATE_RING_DEPTH];
  }
  if (r >= s->v0 - s->top_pse && r < s->v0) return s->top_dlevel[s->v0 - 1 - r];
  if (r >= s->v1 && r < s->v1 + s->bottom_pse) return s->bot_dlevel[r - s->v1];
  return -1;
}

static void set_dl(idwt_2d_state *s, int32_t r, int8_t lv) {
  if (r >= s->v0 && r < s->v1) {
    s->d_level[r % IDWT_STATE_RING_DEPTH] = lv;
    return;
  }
  if (r >= s->v0 - s->top_pse && r < s->v0) { s->top_dlevel[s->v0 - 1 - r] = lv; return; }
  if (r >= s->v1 && r < s->v1 + s->bottom_pse) { s->bot_dlevel[r - s->v1] = lv; }
}

// Required d_level of neighbor rows for row r to advance one level (irrev 9/7 only).
//   LP: step D (cur 0→1) needs HP neighbors @0; step B (cur 1→2) needs HP @1
//   HP: step C (cur 0→1) needs LP neighbors @1; step A (cur 1→2) needs LP @2
// Simplified: for LP need = cur, for HP need = cur + 1.
// (Currently unused — kept for documentation / future use.)
// static inline int8_t needed_neighbor_dl_97(bool lp, int8_t cur) {
//   return lp ? cur : static_cast<int8_t>(cur + 1);
// }

// Apply one lifting step to row r and increment its d_level.
// cur must be the current d_level of row r (caller already fetched it).
static void adv_step(idwt_2d_state *s, int32_t r, int8_t cur) {
  const bool    lp   = is_lp(r);
  sprec_t *tgt  = rptr(s, r);
  sprec_t *prev = rptr(s, r - 1);
  sprec_t *next = rptr(s, r + 1);
  const int32_t w = s->u1 - s->u0;

  if (s->transformation == 0) {  // irrev 9/7
    const float coeff = lp ? (cur == 0 ? fD : fB) : (cur == 0 ? fC : fA);
    adv_irrev_ver_step_fn(w, prev, next, tgt, coeff);
  } else if (s->transformation >= 2) {  // ATK irrev (e.g. irrev53): no floor, 2-step filter
    // irrev53 synthesis: LP[k] -= 0.25*(HP[k-1]+HP[k]);  HP[k] += 0.5*(LP[k]+LP[k+1])
    const float coeff = lp ? 0.25f : -0.5f;  // adv_irrev_ver does: tgt -= coeff*(prev+next)
    adv_irrev_ver_step_fn(w, prev, next, tgt, coeff);
  } else {  // rev 5/3
    if (lp) {
      adv_rev_ver_lp_step_fn(w, prev, next, tgt);
    } else {
      adv_rev_ver_hp_step_fn(w, prev, next, tgt);
    }
  }
  set_dl(s, r, cur + 1);
}

// Fill any PSE slots whose source is physical row r (called after row r is fetched).
static void fill_pse(idwt_2d_state *s, int32_t r) {
  const size_t nb = sizeof(sprec_t) * static_cast<size_t>(s->stride);
  const sprec_t *src = rptr(s, r);
  for (int8_t i = 1; i <= s->top_pse; ++i) {
    if (s->top_dlevel[i - 1] < 0 && pse_source(s->v0 - i, s->v0, s->v1) == r) {
      memcpy(s->top_pse_buf + static_cast<ptrdiff_t>(i - 1) * s->stride, src, nb);
      s->top_dlevel[i - 1] = 0;
    }
  }
  for (int8_t i = 0; i < s->bottom_pse; ++i) {
    if (s->bot_dlevel[i] < 0 && pse_source(s->v1 + i, s->v0, s->v1) == r) {
      memcpy(s->bot_pse_buf + static_cast<ptrdiff_t>(i) * s->stride, src, nb);
      s->bot_dlevel[i] = 0;
    }
  }
}

// Run the cascade: advance every row that can advance, until stable.
//
// Key optimisation — dynamic lo:
//   The naive lo = v0 - top_pse is fixed, so the scan window grows O(n) per call
//   and the total cascade work is O(n²) in the tile height.  Once ring_origin has
//   advanced past the initial PSE processing zone, all rows below ring_origin are
//   either evicted (d_level=-1) or at max_dl; scanning them is pure wasted work.
//   Using lo = max(v0-top_pse, ring_origin - margin) caps the window to a constant
//   number of rows per call, making total cascade work O(n).
//
// For rev 5/3 (max_dl=1): the cascade is strictly 2-phase:
//   Phase 1 — LP Update  (even rows): needs HP neighbors at dl >= 0
//   Phase 2 — HP Predict (odd  rows): needs LP neighbors at dl >= 1
// Two dedicated single passes replace the while(progress) loop.
//
// For irrev 9/7 (max_dl=2): use the generic while(progress) loop (also with dynamic lo).
static void cascade(idwt_2d_state *s) {
  // Margin: enough to cover PSE rows and propagation distance for this transform.
  // 5/3: top_pse≤2, propagation=2  → margin=6
  // 9/7: top_pse≤4, propagation=4  → margin=10
  const int32_t margin  = (int32_t)s->top_pse + max_dl(s->transformation) * 2 + 2;
  const int32_t lo_full = s->v0 - (int32_t)s->top_pse;
  const int32_t lo      = (s->ring_origin - margin > lo_full)
                          ? s->ring_origin - margin : lo_full;
  const int32_t hi      = (s->next_fetch < s->v1) ? s->next_fetch + s->bottom_pse
                                                   : s->v1 + s->bottom_pse;

  if (s->transformation != 0) {  // 2-step filters: rev53 and ATK irrev53 (max_dl=1)
    // Rev 5/3: exactly two phases — LP Update then HP Predict. No while(progress) needed.
    // All rows that advance start at dl=0, so cur=0 is known.

    // Phase 1: Update LP rows (even absolute index) — need HP neighbors at dl >= 0.
    const int32_t lp0 = lo + (lo & 1);   // first even row >= lo  (works for negative lo)
    for (int32_t r = lp0; r < hi; r += 2) {
      if (get_dl(s, r) == 0 && get_dl(s, r - 1) >= 0 && get_dl(s, r + 1) >= 0)
        adv_step(s, r, 0);
    }

    // Phase 2: Predict HP rows (odd absolute index) — need LP neighbors at dl >= 1.
    const int32_t hp0 = lo + (1 - (lo & 1));   // first odd row >= lo
    for (int32_t r = hp0; r < hi; r += 2) {
      if (get_dl(s, r) == 0 && get_dl(s, r - 1) >= 1 && get_dl(s, r + 1) >= 1)
        adv_step(s, r, 0);
    }
  } else {
    // Irrev 9/7: 4 dedicated phases replace the generic while(progress) loop.
    // Dependency chain (max_dl=2):
    //   Phase 1 (D): LP rows dl=0, need HP neighbors @0 → advance to dl=1
    //   Phase 2 (C): HP rows dl=0, need LP neighbors @1 → advance to dl=1
    //   Phase 3 (B): LP rows dl=1, need HP neighbors @1 → advance to dl=2
    //   Phase 4 (A): HP rows dl=1, need LP neighbors @2 → advance to dl=2
    const int32_t lp0 = lo + (lo & 1);        // first even row >= lo
    const int32_t hp0 = lo + (1 - (lo & 1));   // first odd row >= lo

    // Phase 1 (D): LP dl=0 → dl=1
    for (int32_t r = lp0; r < hi; r += 2) {
      if (get_dl(s, r) == 0 && get_dl(s, r - 1) >= 0 && get_dl(s, r + 1) >= 0)
        adv_step(s, r, 0);
    }
    // Phase 2 (C): HP dl=0 → dl=1
    for (int32_t r = hp0; r < hi; r += 2) {
      if (get_dl(s, r) == 0 && get_dl(s, r - 1) >= 1 && get_dl(s, r + 1) >= 1)
        adv_step(s, r, 0);
    }
    // Phase 3 (B): LP dl=1 → dl=2
    for (int32_t r = lp0; r < hi; r += 2) {
      if (get_dl(s, r) == 1 && get_dl(s, r - 1) >= 1 && get_dl(s, r + 1) >= 1)
        adv_step(s, r, 1);
    }
    // Phase 4 (A): HP dl=1 → dl=2
    for (int32_t r = hp0; r < hi; r += 2) {
      if (get_dl(s, r) == 1 && get_dl(s, r - 1) >= 2 && get_dl(s, r + 1) >= 2)
        adv_step(s, r, 1);
    }
  }
}

// Fetch the next real row from the source callback into the ring.
// Ring eviction happens here before fetching if the ring is full and we've
// already output the eviction candidate.
static void fetch_one(idwt_2d_state *s) {
  const int32_t r = s->next_fetch;
  if (r >= s->v1) return;

  // Make room in the ring: advance ring_origin over output rows.
  while (r >= s->ring_origin + IDWT_STATE_RING_DEPTH) {
    if (s->ring_origin >= s->next_out) break;  // can't evict un-output rows
    s->d_level[s->ring_origin % IDWT_STATE_RING_DEPTH] = -1;
    ++s->ring_origin;
  }

  const int32_t slot = r % IDWT_STATE_RING_DEPTH;
  s->d_level[slot]   = -1;
  s->get_src_row(s->src_ctx, r, rptr(s, r));
  s->d_level[slot]   = 0;
  ++s->next_fetch;
  fill_pse(s, r);
  cascade(s);
}

// ─────────────────────────────────────────────────────────────────────────────

void idwt_2d_state_init(idwt_2d_state *s,
                        const int32_t u0, const int32_t u1,
                        const int32_t v0, const int32_t v1,
                        const uint8_t transformation, const dwt_type dir,
                        idwt_row_src_fn src_fn, void *src_ctx) {
  s->u0            = u0;  s->u1 = u1;
  s->v0            = v0;  s->v1 = v1;
  s->stride        = round_up(u1 - u0, SIMD_PADDING);
  s->slot_stride   = IDWT_RING_PSE_LEFT + round_up(u1 - u0 + SIMD_PADDING, SIMD_PADDING);
  s->transformation = transformation;
  s->dir           = dir;
  // ATK (transformation>=2) is a 2-step filter like rev53 — use same PSE counts (eff=1).
  const uint8_t eff = (transformation < 2) ? transformation : 1;
  s->top_pse       = kPseTop[v0 % 2][eff];
  s->bottom_pse    = kPseBot[v1 % 2][eff];

  s->ring_buf      = nullptr;
  s->top_pse_buf   = nullptr;
  s->bot_pse_buf   = nullptr;
  s->horz_out_buf  = nullptr;

  if (dir == DWT_BIDIR) {
    const size_t row_bytes  = sizeof(sprec_t) * static_cast<size_t>(s->stride);
    const size_t slot_bytes = sizeof(sprec_t) * static_cast<size_t>(s->slot_stride);
    s->ring_buf    = static_cast<sprec_t *>(aligned_mem_alloc(IDWT_STATE_RING_DEPTH * slot_bytes, 32));
    s->top_pse_buf = (s->top_pse    > 0) ? static_cast<sprec_t *>(aligned_mem_alloc(static_cast<size_t>(s->top_pse)    * row_bytes, 32)) : nullptr;
    s->bot_pse_buf = (s->bottom_pse > 0) ? static_cast<sprec_t *>(aligned_mem_alloc(static_cast<size_t>(s->bottom_pse) * row_bytes, 32)) : nullptr;
  } else {
    // HORZ or NO: one output row at a time — single scratch buffer with PSE prefix.
    const size_t slot_bytes = sizeof(sprec_t) * static_cast<size_t>(s->slot_stride);
    s->horz_out_buf = static_cast<sprec_t *>(aligned_mem_alloc(slot_bytes, 32));
  }

  s->ring_origin = v0;
  for (int32_t i = 0; i < IDWT_STATE_RING_DEPTH; ++i) s->d_level[i]    = -1;
  for (int32_t i = 0; i < 4;                     ++i) s->top_dlevel[i] = -1;
  for (int32_t i = 0; i < 4;                     ++i) s->bot_dlevel[i] = -1;

  s->next_out    = v0;
  s->next_fetch  = v0;
  s->get_src_row = src_fn;
  s->src_ctx     = src_ctx;
}

void idwt_2d_state_free(idwt_2d_state *s) {
  aligned_mem_free(s->ring_buf);     s->ring_buf     = nullptr;
  aligned_mem_free(s->top_pse_buf);  s->top_pse_buf  = nullptr;
  aligned_mem_free(s->bot_pse_buf);  s->bot_pse_buf  = nullptr;
  aligned_mem_free(s->horz_out_buf); s->horz_out_buf = nullptr;
}

// Rewind the streaming cursors to their post-init state without touching
// any of the aligned buffers (ring_buf / top_pse_buf / bot_pse_buf /
// horz_out_buf).  Used by the single-tile reuse path to return a state
// that has already produced v1-v0 rows back to "nothing emitted yet".
// Geometry, transformation, dir, PSE counts and src pointers remain set,
// so a subsequent pull_row call restarts from row v0.
void idwt_2d_state_rewind(idwt_2d_state *s) {
  s->ring_origin = s->v0;
  s->next_out    = s->v0;
  s->next_fetch  = s->v0;
  for (int32_t i = 0; i < IDWT_STATE_RING_DEPTH; ++i) s->d_level[i]    = -1;
  for (int32_t i = 0; i < 4;                     ++i) s->top_dlevel[i] = -1;
  for (int32_t i = 0; i < 4;                     ++i) s->bot_dlevel[i] = -1;
}

bool idwt_2d_state_pull_row(idwt_2d_state *s, sprec_t *out) {
  const sprec_t *ptr = idwt_2d_state_pull_row_ref(s);
  if (!ptr) return false;
  memcpy(out, ptr, sizeof(sprec_t) * static_cast<size_t>(s->u1 - s->u0));
  return true;
}

sprec_t *idwt_2d_state_pull_row_ref(idwt_2d_state *s) {
  if (s->next_out >= s->v1) return nullptr;

  // ── HORZ: no vertical DWT — one horizontal IDWT per row ───────────────────
  // The source callback interleaves LL and H and applies the horizontal IDWT.
  // horz_out_buf has IDWT_RING_PSE_LEFT prefix so the callback can fill PSE in-place.
  if (s->dir == DWT_HORZ) {
    sprec_t *data = s->horz_out_buf + IDWT_RING_PSE_LEFT;
    s->get_src_row(s->src_ctx, s->next_out, data);
    ++s->next_out;
    return data;
  }

  // ── NO_DWT: pure passthrough — source callback copies LL row to scratch ───
  if (s->dir == DWT_NO) {
    sprec_t *data = s->horz_out_buf + IDWT_RING_PSE_LEFT;
    s->get_src_row(s->src_ctx, s->next_out, data);
    ++s->next_out;
    return data;
  }

  // ── BIDIR: full 2D IDWT with ring buffer and vertical lifting ─────────────

  // Special case: empty or trivial tile.
  if (s->v1 <= s->v0) return nullptr;

  // Special case: single-row tile (v1 == v0+1) — write into ring buffer slot.
  if (s->v1 == s->v0 + 1) {
    sprec_t *dst = rptr(s, s->v0);
    s->get_src_row(s->src_ctx, s->v0, dst);
    if (s->transformation == 1 && (s->v0 % 2) == 1) {
      for (int32_t c = 0; c < s->u1 - s->u0; ++c) dst[c] = floorf(dst[c] * 0.5f);
    }
    ++s->next_out;
    return dst;
  }

  const int8_t mxdl = max_dl(s->transformation);

  // Fetch source rows (and run cascade) until next_out is ready.
  while (get_dl(s, s->next_out) < mxdl) {
    if (s->next_fetch >= s->v1) {
      cascade(s);  // final pass to drain bottom PSE
      break;
    }
    // Advance ring_origin past already-output rows to make room for new fetches.
    while (s->ring_origin < s->next_out &&
           s->next_fetch >= s->ring_origin + IDWT_STATE_RING_DEPTH) {
      s->d_level[s->ring_origin % IDWT_STATE_RING_DEPTH] = -1;
      ++s->ring_origin;
    }
    fetch_one(s);
  }

  if (get_dl(s, s->next_out) < mxdl) return nullptr;  // should not happen

  sprec_t *result = rptr(s, s->next_out);
  ++s->next_out;
  return result;
}
