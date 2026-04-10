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
#if defined(OPENHTJ2K_ENABLE_WASM_SIMD)
static fdwt_1d_filtr_func_fixed fdwt_1d_filtr_fixed[2] = {fdwt_1d_filtr_irrev97_fixed_wasm,
                                                           fdwt_1d_filtr_rev53_fixed_wasm};
static fdwt_ver_filtr_func_fixed fdwt_ver_sr_fixed[2]  = {fdwt_irrev_ver_sr_fixed_wasm,
                                                           fdwt_rev_ver_sr_fixed_wasm};
// Single-row vertical lifting step: tgt[i] -= coeff*(prev[i]+next[i]).
// FDWT calls with -coeff when the step is additive (tgt[i] += coeff*(prev[i]+next[i])).
typedef void (*adv_fdwt_irrev_step_fn_t)(int32_t, float *, float *, float *, float);
static adv_fdwt_irrev_step_fn_t adv_fdwt_irrev_step_fn = idwt_irrev_ver_step_fixed_wasm;
typedef void (*adv_fdwt_rev_step_fn_t)(int32_t, const float *, const float *, float *);
static adv_fdwt_rev_step_fn_t adv_fdwt_rev_hp_step_fn = fdwt_rev_ver_hp_step_wasm;
static adv_fdwt_rev_step_fn_t adv_fdwt_rev_lp_step_fn = fdwt_rev_ver_lp_step_wasm;
#elif defined(OPENHTJ2K_ENABLE_AVX512)
static fdwt_1d_filtr_func_fixed fdwt_1d_filtr_fixed[2] = {fdwt_1d_filtr_irrev97_fixed_avx512,
                                                          fdwt_1d_filtr_rev53_fixed_avx512};
static fdwt_ver_filtr_func_fixed fdwt_ver_sr_fixed[2]  = {fdwt_irrev_ver_sr_fixed_avx512,
                                                          fdwt_rev_ver_sr_fixed_avx512};
typedef void (*adv_fdwt_irrev_step_fn_t)(int32_t, float *, float *, float *, float);
static adv_fdwt_irrev_step_fn_t adv_fdwt_irrev_step_fn = idwt_irrev_ver_step_fixed_avx512;
typedef void (*adv_fdwt_rev_step_fn_t)(int32_t, const float *, const float *, float *);
static adv_fdwt_rev_step_fn_t adv_fdwt_rev_hp_step_fn = fdwt_rev_ver_hp_step_avx512;
static adv_fdwt_rev_step_fn_t adv_fdwt_rev_lp_step_fn = fdwt_rev_ver_lp_step_avx512;
#elif defined(OPENHTJ2K_ENABLE_ARM_NEON)
static fdwt_1d_filtr_func_fixed fdwt_1d_filtr_fixed[2] = {fdwt_1d_filtr_irrev97_fixed_neon,
                                                          fdwt_1d_filtr_rev53_fixed_neon};
static fdwt_ver_filtr_func_fixed fdwt_ver_sr_fixed[2]  = {fdwt_irrev_ver_sr_fixed_neon,
                                                          fdwt_rev_ver_sr_fixed_neon};
typedef void (*adv_fdwt_irrev_step_fn_t)(int32_t, float *, float *, float *, float);
static adv_fdwt_irrev_step_fn_t adv_fdwt_irrev_step_fn = idwt_irrev_ver_step_fixed_neon;
typedef void (*adv_fdwt_rev_step_fn_t)(int32_t, const float *, const float *, float *);
static adv_fdwt_rev_step_fn_t adv_fdwt_rev_hp_step_fn = fdwt_rev_ver_hp_step_neon;
static adv_fdwt_rev_step_fn_t adv_fdwt_rev_lp_step_fn = fdwt_rev_ver_lp_step_neon;
#elif defined(OPENHTJ2K_ENABLE_AVX2)
static fdwt_1d_filtr_func_fixed fdwt_1d_filtr_fixed[2] = {fdwt_1d_filtr_irrev97_fixed_avx2,
                                                          fdwt_1d_filtr_rev53_fixed_avx2};
static fdwt_ver_filtr_func_fixed fdwt_ver_sr_fixed[2]  = {fdwt_irrev_ver_sr_fixed_avx2,
                                                          fdwt_rev_ver_sr_fixed_avx2};
typedef void (*adv_fdwt_irrev_step_fn_t)(int32_t, float *, float *, float *, float);
static adv_fdwt_irrev_step_fn_t adv_fdwt_irrev_step_fn = idwt_irrev_ver_step_fixed_avx2;
typedef void (*adv_fdwt_rev_step_fn_t)(int32_t, const float *, const float *, float *);
static adv_fdwt_rev_step_fn_t adv_fdwt_rev_hp_step_fn = fdwt_rev_ver_hp_step_avx2;
static adv_fdwt_rev_step_fn_t adv_fdwt_rev_lp_step_fn = fdwt_rev_ver_lp_step_avx2;
#else
static fdwt_1d_filtr_func_fixed fdwt_1d_filtr_fixed[2] = {fdwt_1d_filtr_irrev97_fixed,
                                                          fdwt_1d_filtr_rev53_fixed};
static fdwt_ver_filtr_func_fixed fdwt_ver_sr_fixed[2]  = {fdwt_irrev_ver_sr_fixed, fdwt_rev_ver_sr_fixed};
static void adv_fdwt_irrev_step_scalar(int32_t n, float *prev, float *next, float *tgt, float coeff) {
  for (int32_t i = 0; i < n; ++i) tgt[i] -= coeff * (prev[i] + next[i]);
}
typedef void (*adv_fdwt_irrev_step_fn_t)(int32_t, float *, float *, float *, float);
static adv_fdwt_irrev_step_fn_t adv_fdwt_irrev_step_fn = adv_fdwt_irrev_step_scalar;
static void adv_fdwt_rev_hp_step_scalar(int32_t n, const float *prev, const float *next, float *tgt) {
  for (int32_t i = 0; i < n; ++i) tgt[i] -= floorf((prev[i] + next[i]) * 0.5f);
}
static void adv_fdwt_rev_lp_step_scalar(int32_t n, const float *prev, const float *next, float *tgt) {
  for (int32_t i = 0; i < n; ++i) tgt[i] += floorf((prev[i] + next[i] + 2.0f) * 0.25f);
}
typedef void (*adv_fdwt_rev_step_fn_t)(int32_t, const float *, const float *, float *);
static adv_fdwt_rev_step_fn_t adv_fdwt_rev_hp_step_fn = adv_fdwt_rev_hp_step_scalar;
static adv_fdwt_rev_step_fn_t adv_fdwt_rev_lp_step_fn = adv_fdwt_rev_lp_step_scalar;
#endif
// irreversible FDWT
void fdwt_1d_filtr_irrev97_fixed(sprec_t *X, const int32_t left, const int32_t u_i0, const int32_t u_i1) {
  const auto i0       = static_cast<int32_t>(u_i0);
  const auto i1       = static_cast<int32_t>(u_i1);
  const int32_t start = ceil_int(i0, 2);
  const int32_t stop  = ceil_int(i1, 2);

  const int32_t offset = left + i0 % 2;
  for (int32_t n = -4 + offset, i = start - 2; i < stop + 1; i++, n += 2) {
    float sum = X[n];
    sum += X[n + 2];
    // X[n + 1] = static_cast<sprec_t>(X[n + 1] + ((Acoeff * sum + Aoffset) >> Ashift));
    X[n + 1] += fA * sum;
  }
  for (int32_t n = -2 + offset, i = start - 1; i < stop + 1; i++, n += 2) {
    float sum = X[n - 1];
    sum += X[n + 1];
    // X[n] = static_cast<sprec_t>(X[n] + ((Bcoeff * sum + Boffset) >> Bshift));
    X[n] += fB * sum;
  }
  for (int32_t n = -2 + offset, i = start - 1; i < stop; i++, n += 2) {
    float sum = X[n];
    sum += X[n + 2];
    // X[n + 1] = static_cast<sprec_t>(X[n + 1] + ((Ccoeff * sum + Coffset) >> Cshift));
    X[n + 1] += fC * sum;
  }
  for (int32_t n = 0 + offset, i = start; i < stop; i++, n += 2) {
    float sum = X[n - 1];
    sum += X[n + 1];
    // X[n] = static_cast<sprec_t>(X[n] + ((Dcoeff * sum + Doffset) >> Dshift));
    X[n] += fD * sum;
  }
};

// reversible FDWT
void fdwt_1d_filtr_rev53_fixed(sprec_t *X, const int32_t left, const int32_t u_i0, const int32_t u_i1) {
  const auto i0       = static_cast<int32_t>(u_i0);
  const auto i1       = static_cast<int32_t>(u_i1);
  const int32_t start = ceil_int(i0, 2);
  const int32_t stop  = ceil_int(i1, 2);
  // X += left - i0 % 2;
  const int32_t offset = left + i0 % 2;
  for (int32_t n = -2 + offset, i = start - 1; i < stop; ++i, n += 2) {
    float sum = X[n] + X[n + 2];
    X[n + 1] -= floorf(sum * 0.5f);
  }
  for (int32_t n = 0 + offset, i = start; i < stop; ++i, n += 2) {
    float sum = X[n - 1] + X[n + 1];
    X[n] += floorf((sum + 2) * 0.25f);
  }
};

// ATK irreversible 5/3 horizontal FDWT (analysis): 2-step without floor.
// Step 1: HP[k] -= 0.5*(LP[k] + LP[k+1])   [predict using original LP]
// Step 2: LP[k] += 0.25*(HP_mod[k-1] + HP_mod[k])  [update using modified HP]
static void fdwt_1d_filtr_irrev53_fixed(sprec_t *X, const int32_t left, const int32_t u_i0,
                                        const int32_t u_i1) {
  const int32_t i0     = static_cast<int32_t>(u_i0);
  const int32_t i1     = static_cast<int32_t>(u_i1);
  const int32_t start  = ceil_int(i0, 2);
  const int32_t stop   = ceil_int(i1, 2);
  const int32_t offset = left + i0 % 2;
  for (int32_t n = -2 + offset, i = start - 1; i < stop; ++i, n += 2)
    X[n + 1] -= 0.5f * (X[n] + X[n + 2]);
  for (int32_t n = 0 + offset, i = start; i < stop; ++i, n += 2)
    X[n] += 0.25f * (X[n - 1] + X[n + 1]);
}

// In-place 1-D FDWT for all rows.
// Operates directly on in[-left..width+SIMD_LEN_I32-1] without copying to/from an external buffer.
// Precondition: those memory locations are within the tile allocation. The j2k_subband /
// j2k_resolution allocators add DWT_LEFT_SLACK floats before the first row and DWT_RIGHT_SLACK
// floats after the last row of the buffer (border slack), which is what makes the precondition
// hold for the first/last rows of a tile. Interior rows always satisfied it because the slack
// regions overlap adjacent rows' data — save/restore below preserves them.
static inline void fdwt_1d_sr_inplace(sprec_t *in, const int32_t left, const int32_t right,
                                      const int32_t i0, const int32_t i1,
                                      const uint8_t transformation) {
  const int32_t width = i1 - i0;
  // Save regions that the filter will temporarily overwrite with PSE data or SIMD tail writes.
  // left_save[8] covers the dwt_pse_fill_inplace_simd write window (8 floats per side).
  // right_save[SIMD_LEN_I32=8] also matches that window exactly.
  sprec_t left_save[8];
  sprec_t right_save[SIMD_LEN_I32];
  for (int32_t i = 0; i < 8; ++i) left_save[i] = in[-8 + i];
  for (int32_t i = 0; i < SIMD_LEN_I32; ++i) right_save[i] = in[width + i];
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
    fdwt_1d_filtr_fixed[transformation](in - left, left, i0, i1);
  else
    fdwt_1d_filtr_irrev53_fixed(in - left, left, i0, i1);
  // Restore the saved regions (DWT output is in in[0..width-1], boundary regions are scratch).
  for (int32_t i = 0; i < 8; ++i) in[-8 + i] = left_save[i];
  for (int32_t i = 0; i < SIMD_LEN_I32; ++i) in[width + i] = right_save[i];
}

// FDWT for horizontal direction
static void fdwt_hor_sr_fixed(sprec_t *in, const int32_t u0, const int32_t u1, const int32_t v0,
                              const int32_t v1, const uint8_t transformation, const int32_t stride) {
  constexpr int32_t num_pse_i0[2][2] = {{4, 2}, {3, 1}};
  constexpr int32_t num_pse_i1[2][2] = {{3, 1}, {4, 2}};
  // ATK (transformation>=2) uses same PSE sizes as rev53 (2-step filter).
  const int32_t cls   = (transformation == 0) ? 0 : 1;
  const int32_t left  = num_pse_i0[u0 % 2][cls];
  const int32_t right = num_pse_i1[u1 % 2][cls];

  if (u0 == u1 - 1) {
    // one sample case: irrev53 (ATK) needs no scaling; rev53 HP *= 2; irrev97 no-op.
    for (int32_t row = 0; row < v1 - v0; ++row) {
      if (u0 % 2 != 0) {
        if (transformation == 1) in[row * stride] = floorf(in[row * stride] * 2.0f);
      }
    }
  } else {
    // All rows use the in-place horizontal DWT.  The j2k_subband / j2k_resolution
    // allocators add DWT_LEFT_SLACK + DWT_RIGHT_SLACK floats of border slack to
    // the buffer (and offset the user-visible i_samples pointer by DWT_LEFT_SLACK),
    // so the first row's in[-left..-1] and the last row's in[width..width+SIMD_LEN_I32-1]
    // both fall inside valid memory.  Interior rows are handled by save/restore inside
    // fdwt_1d_sr_inplace.  This eliminates the per-tile-per-level Xext allocation and
    // the redundant memcpy of every row that the legacy copy path performed.
    const int32_t nrows = v1 - v0;
    for (int32_t row = 0; row < nrows; ++row) {
      fdwt_1d_sr_inplace(in, left, right, u0, u1, transformation);
      in += stride;
    }
  }
}

void fdwt_irrev_ver_sr_fixed(sprec_t *in, const int32_t u0, const int32_t u1, const int32_t v0,
                             const int32_t v1, const int32_t stride, sprec_t *pse_scratch,
                             sprec_t **buf_scratch) {
  constexpr int32_t num_pse_i0[2] = {4, 3};
  constexpr int32_t num_pse_i1[2] = {3, 4};
  const int32_t top               = num_pse_i0[v0 % 2];
  const int32_t bottom            = num_pse_i1[v1 % 2];

  if (v0 == v1 - 1) {
    // one sample case
    for (int32_t col = 0; col < u1 - u0; ++col) {
      if (v0 % 2) {
        // in[col] <<= 0;
      }
    }
  } else {
    const int32_t len = round_up(stride, SIMD_LEN_I32);
    sprec_t **buf     = buf_scratch;
    for (int32_t i = 1; i <= top; ++i) {
      // buf[top - i] =
      //     static_cast<sprec_t *>(aligned_mem_alloc(sizeof(sprec_t) * static_cast<size_t>(len), 32));
      buf[top - i] = pse_scratch + (i - 1) * len;
      memcpy(buf[top - i], &in[PSEo(v0 - i, v0, v1) * stride],
             sizeof(sprec_t) * static_cast<size_t>(stride));
      // buf[top - i] = &in[(PSEo(v0 - i, v0, v1) - v0) * stride];
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
    const int32_t start  = ceil_int(v0, 2);
    const int32_t stop   = ceil_int(v1, 2);
    const int32_t offset = top + v0 % 2;

    const int32_t width = u1 - u0;
    for (int32_t cs = 0; cs < width; cs += DWT_VERT_STRIP) {
      const int32_t ce = (cs + DWT_VERT_STRIP < width) ? cs + DWT_VERT_STRIP : width;
      for (int32_t n = -4 + offset, i = start - 2; i < stop + 1; i++, n += 2) {
        for (int32_t col = cs; col < ce; ++col) {
          buf[n + 1][col] += fA * (buf[n][col] + buf[n + 2][col]);
        }
      }
      for (int32_t n = -2 + offset, i = start - 1; i < stop + 1; i++, n += 2) {
        for (int32_t col = cs; col < ce; ++col) {
          buf[n][col] += fB * (buf[n - 1][col] + buf[n + 1][col]);
        }
      }
      for (int32_t n = -2 + offset, i = start - 1; i < stop; i++, n += 2) {
        for (int32_t col = cs; col < ce; ++col) {
          buf[n + 1][col] += fC * (buf[n][col] + buf[n + 2][col]);
        }
      }
      for (int32_t n = 0 + offset, i = start; i < stop; i++, n += 2) {
        for (int32_t col = cs; col < ce; ++col) {
          buf[n][col] += fD * (buf[n - 1][col] + buf[n + 1][col]);
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

void fdwt_rev_ver_sr_fixed(sprec_t *in, const int32_t u0, const int32_t u1, const int32_t v0,
                           const int32_t v1, const int32_t stride, sprec_t *pse_scratch,
                           sprec_t **buf_scratch) {
  constexpr int32_t num_pse_i0[2] = {2, 1};
  constexpr int32_t num_pse_i1[2] = {1, 2};
  const int32_t top               = num_pse_i0[v0 % 2];
  const int32_t bottom            = num_pse_i1[v1 % 2];

  if (v0 == v1 - 1) {
    // one sample case
    for (int32_t col = 0; col < u1 - u0; ++col) {
      if (v0 % 2) {
        in[col] = floorf(in[col] * 2.0f);
      }
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
      // buf[top - i] = &in[(PSEo(v0 - i, v0, v1) - v0) * stride];
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
    const int32_t start  = ceil_int(v0, 2);
    const int32_t stop   = ceil_int(v1, 2);
    const int32_t offset = top + v0 % 2;

    const int32_t width = u1 - u0;
    for (int32_t cs = 0; cs < width; cs += DWT_VERT_STRIP) {
      const int32_t ce = (cs + DWT_VERT_STRIP < width) ? cs + DWT_VERT_STRIP : width;
      for (int32_t n = -2 + offset, i = start - 1; i < stop; ++i, n += 2) {
        for (int32_t col = cs; col < ce; ++col) {
          buf[n + 1][col] -= floorf((buf[n][col] + buf[n + 2][col]) * 0.5f);
        }
      }
      for (int32_t n = 0 + offset, i = start; i < stop; ++i, n += 2) {
        for (int32_t col = cs; col < ce; ++col) {
          buf[n][col] += floorf((buf[n - 1][col] + buf[n + 1][col] + 2) * 0.25f);
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

// ATK irreversible 5/3 vertical FDWT (analysis): 2-step without floor.
// Step 1: HP rows -= 0.5*(LP_above + LP_below)   [predict using original LP rows]
// Step 2: LP rows += 0.25*(HP_mod_above + HP_mod_below)  [update using modified HP rows]
static void fdwt_irrev53_ver_sr_fixed(sprec_t *in, const int32_t u0, const int32_t u1, const int32_t v0,
                                      const int32_t v1, const int32_t stride, sprec_t *pse_scratch,
                                      sprec_t **buf_scratch) {
  constexpr int32_t num_pse_i0[2] = {2, 1};
  constexpr int32_t num_pse_i1[2] = {1, 2};
  const int32_t top               = num_pse_i0[v0 % 2];
  const int32_t bottom            = num_pse_i1[v1 % 2];
  if (v0 == v1 - 1) {
    // single row: nothing to do (no neighbouring rows for lifting)
  } else {
    const int32_t len = round_up(stride, SIMD_LEN_I32);
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
    const int32_t start  = ceil_int(v0, 2);
    const int32_t stop   = ceil_int(v1, 2);
    const int32_t offset = top + v0 % 2;
    const int32_t width  = u1 - u0;
    for (int32_t cs = 0; cs < width; cs += DWT_VERT_STRIP) {
      const int32_t ce = (cs + DWT_VERT_STRIP < width) ? cs + DWT_VERT_STRIP : width;
      // Step 1: HP -= 0.5*(LP_left + LP_right)
      for (int32_t n = -2 + offset, i = start - 1; i < stop; ++i, n += 2)
        for (int32_t col = cs; col < ce; ++col)
          buf[n + 1][col] -= 0.5f * (buf[n][col] + buf[n + 2][col]);
      // Step 2: LP += 0.25*(HP_mod_left + HP_mod_right)
      for (int32_t n = 0 + offset, i = start; i < stop; ++i, n += 2)
        for (int32_t col = cs; col < ce; ++col)
          buf[n][col] += 0.25f * (buf[n - 1][col] + buf[n + 1][col]);
    }
  }
}

// Deinterleaving to devide coefficients into subbands
static void fdwt_2d_deinterleave_fixed(sprec_t *buf, sprec_t *const LL, sprec_t *const HL,
                                       sprec_t *const LH, sprec_t *const HH, const int32_t u0,
                                       const int32_t u1, const int32_t v0, const int32_t v1,
                                       const int32_t stride) {
  const int32_t v_offset   = v0 % 2;
  const int32_t u_offset   = u0 % 2;
  sprec_t *dp[4]           = {LL, HL, LH, HH};
  const int32_t vstart[4]  = {ceil_int(v0, 2), ceil_int(v0, 2), v0 / 2, v0 / 2};
  const int32_t vstop[4]   = {ceil_int(v1, 2), ceil_int(v1, 2), v1 / 2, v1 / 2};
  const int32_t ustart[4]  = {ceil_int(u0, 2), u0 / 2, ceil_int(u0, 2), u0 / 2};
  const int32_t ustop[4]   = {ceil_int(u1, 2), u1 / 2, ceil_int(u1, 2), u1 / 2};
  const int32_t voffset[4] = {v_offset, v_offset, 1 - v_offset, 1 - v_offset};
  const int32_t uoffset[4] = {u_offset, 1 - u_offset, u_offset, 1 - u_offset};
  const int32_t stride2[4] = {round_up(ustop[0] - ustart[0], 32), round_up(ustop[1] - ustart[1], 32),
                              round_up(ustop[2] - ustart[2], 32), round_up(ustop[3] - ustart[3], 32)};

#if defined(OPENHTJ2K_ENABLE_WASM_SIMD)
  // WASM-SIMD deinterleave: use wasm_i32x4_shuffle instead of vld2q_f32.
  auto wasm_deinterleave_pair = [](sprec_t *buf, sprec_t *bdp0, sprec_t *bdp1, int32_t s0, int32_t s1,
                                   int32_t common_len, int32_t vstart_b, int32_t vstop_b, int32_t voffset_b,
                                   int32_t stride) {
    for (int32_t v = 0, vb = vstart_b; vb < vstop_b; ++vb, ++v) {
      sprec_t *sp    = buf + (2 * v + voffset_b) * stride;
      size_t len     = static_cast<size_t>(common_len);
      sprec_t *line0 = bdp0 + static_cast<ptrdiff_t>(v) * s0;
      sprec_t *line1 = bdp1 + static_cast<ptrdiff_t>(v) * s1;
      for (; len >= 8; len -= 8) {
        v128_t lo0 = wasm_v128_load(sp);      v128_t hi0 = wasm_v128_load(sp + 4);
        v128_t lo1 = wasm_v128_load(sp + 8);  v128_t hi1 = wasm_v128_load(sp + 12);
        wasm_v128_store(line0,     wasm_i32x4_shuffle(lo0, hi0, 0, 2, 4, 6));
        wasm_v128_store(line1,     wasm_i32x4_shuffle(lo0, hi0, 1, 3, 5, 7));
        wasm_v128_store(line0 + 4, wasm_i32x4_shuffle(lo1, hi1, 0, 2, 4, 6));
        wasm_v128_store(line1 + 4, wasm_i32x4_shuffle(lo1, hi1, 1, 3, 5, 7));
        line0 += 8; line1 += 8; sp += 16;
      }
      for (; len >= 4; len -= 4) {
        v128_t lo = wasm_v128_load(sp); v128_t hi = wasm_v128_load(sp + 4);
        wasm_v128_store(line0, wasm_i32x4_shuffle(lo, hi, 0, 2, 4, 6));
        wasm_v128_store(line1, wasm_i32x4_shuffle(lo, hi, 1, 3, 5, 7));
        line0 += 4; line1 += 4; sp += 8;
      }
      for (; len > 0; --len) { *line0++ = *sp++; *line1++ = *sp++; }
    }
  };
  {
    const int32_t len0 = ustop[0] - ustart[0], len1 = ustop[1] - ustart[1];
    const int32_t common_len = len0 < len1 ? len0 : len1;
    sprec_t *bdp0 = dp[0]; sprec_t *bdp1 = dp[1]; int32_t s0 = stride2[0], s1 = stride2[1];
    if (uoffset[0] > uoffset[1]) { std::swap(bdp0, bdp1); std::swap(s0, s1); }
    wasm_deinterleave_pair(buf, bdp0, bdp1, s0, s1, common_len, vstart[0], vstop[0], voffset[0], stride);
    for (uint8_t b = 0; b < 2; ++b) {
      if ((ustop[b] - ustart[b]) > common_len) {
        for (int32_t v = 0, vb = vstart[b]; vb < vstop[b]; ++vb, ++v)
          dp[b][static_cast<ptrdiff_t>(v) * stride2[b] + common_len] =
              buf[2 * common_len + uoffset[b] + (2 * v + voffset[b]) * stride];
        break;
      }
    }
  }
  {
    const int32_t len2 = ustop[2] - ustart[2], len3 = ustop[3] - ustart[3];
    const int32_t common_len = len2 < len3 ? len2 : len3;
    sprec_t *bdp2 = dp[2]; sprec_t *bdp3 = dp[3]; int32_t s2 = stride2[2], s3 = stride2[3];
    if (uoffset[2] > uoffset[3]) { std::swap(bdp2, bdp3); std::swap(s2, s3); }
    wasm_deinterleave_pair(buf, bdp2, bdp3, s2, s3, common_len, vstart[2], vstop[2], voffset[2], stride);
    for (uint8_t b = 2; b < 4; ++b) {
      if ((ustop[b] - ustart[b]) > common_len) {
        for (int32_t v = 0, vb = vstart[b]; vb < vstop[b]; ++vb, ++v)
          dp[b][static_cast<ptrdiff_t>(v) * stride2[b] + common_len] =
              buf[2 * common_len + uoffset[b] + (2 * v + voffset[b]) * stride];
        break;
      }
    }
  }
#elif defined(OPENHTJ2K_ENABLE_ARM_NEON)
  {
    // Band pair {LL, HL}: NEON deinterleave for common length, scalar for extra sample.
    const int32_t len0       = ustop[0] - ustart[0];
    const int32_t len1       = ustop[1] - ustart[1];
    const int32_t common_len = len0 < len1 ? len0 : len1;
    sprec_t *bdp0 = dp[0], *bdp1 = dp[1];
    int32_t s0 = stride2[0], s1 = stride2[1];
    if (uoffset[0] > uoffset[1]) {
      std::swap(bdp0, bdp1);
      std::swap(s0, s1);
    }
    for (int32_t v = 0, vb = vstart[0]; vb < vstop[0]; ++vb, ++v) {
      sprec_t *sp    = buf + (2 * v + voffset[0]) * stride;
      size_t len     = static_cast<size_t>(common_len);
      sprec_t *line0 = bdp0 + static_cast<ptrdiff_t>(v) * s0;
      sprec_t *line1 = bdp1 + static_cast<ptrdiff_t>(v) * s1;
      // 2× unrolled: two vld2q_f32 per iteration to hide load latency.
      for (; len >= 8; len -= 8) {
        auto vline0 = vld2q_f32(sp);
        auto vline1 = vld2q_f32(sp + 8);
        vst1q_f32(line0, vline0.val[0]);
        vst1q_f32(line1, vline0.val[1]);
        vst1q_f32(line0 + 4, vline1.val[0]);
        vst1q_f32(line1 + 4, vline1.val[1]);
        line0 += 8;
        line1 += 8;
        sp += 16;
      }
      for (; len >= 4; len -= 4) {
        auto vline = vld2q_f32(sp);
        vst1q_f32(line0, vline.val[0]);
        vst1q_f32(line1, vline.val[1]);
        line0 += 4;
        line1 += 4;
        sp += 8;
      }
      for (; len > 0; --len) {
        *line0++ = *sp++;
        *line1++ = *sp++;
      }
    }
    for (uint8_t b = 0; b < 2; ++b) {
      if ((ustop[b] - ustart[b]) > common_len) {
        for (int32_t v = 0, vb = vstart[b]; vb < vstop[b]; ++vb, ++v) {
          dp[b][static_cast<ptrdiff_t>(v) * stride2[b] + common_len] =
              buf[2 * common_len + uoffset[b] + (2 * v + voffset[b]) * stride];
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
    sprec_t *bdp2 = dp[2], *bdp3 = dp[3];
    int32_t s2 = stride2[2], s3 = stride2[3];
    if (uoffset[2] > uoffset[3]) {
      std::swap(bdp2, bdp3);
      std::swap(s2, s3);
    }
    for (int32_t v = 0, vb = vstart[2]; vb < vstop[2]; ++vb, ++v) {
      sprec_t *sp    = buf + (2 * v + voffset[2]) * stride;
      size_t len     = static_cast<size_t>(common_len);
      sprec_t *line0 = bdp2 + static_cast<ptrdiff_t>(v) * s2;
      sprec_t *line1 = bdp3 + static_cast<ptrdiff_t>(v) * s3;
      // 2× unrolled: two vld2q_f32 per iteration to hide load latency.
      for (; len >= 8; len -= 8) {
        auto vline0 = vld2q_f32(sp);
        auto vline1 = vld2q_f32(sp + 8);
        vst1q_f32(line0, vline0.val[0]);
        vst1q_f32(line1, vline0.val[1]);
        vst1q_f32(line0 + 4, vline1.val[0]);
        vst1q_f32(line1 + 4, vline1.val[1]);
        line0 += 8;
        line1 += 8;
        sp += 16;
      }
      for (; len >= 4; len -= 4) {
        auto vline = vld2q_f32(sp);
        vst1q_f32(line0, vline.val[0]);
        vst1q_f32(line1, vline.val[1]);
        line0 += 4;
        line1 += 4;
        sp += 8;
      }
      for (; len > 0; --len) {
        *line0++ = *sp++;
        *line1++ = *sp++;
      }
    }
    for (uint8_t b = 2; b < 4; ++b) {
      if ((ustop[b] - ustart[b]) > common_len) {
        for (int32_t v = 0, vb = vstart[b]; vb < vstop[b]; ++vb, ++v) {
          dp[b][static_cast<ptrdiff_t>(v) * stride2[b] + common_len] =
              buf[2 * common_len + uoffset[b] + (2 * v + voffset[b]) * stride];
        }
        break;
      }
    }
  }
#elif defined(OPENHTJ2K_TRY_AVX2) && defined(__AVX2__)
  {
    // Band pair {LL, HL}: AVX2 deinterleave for common length, scalar for the extra sample.
    const int32_t len0       = ustop[0] - ustart[0];
    const int32_t len1       = ustop[1] - ustart[1];
    const int32_t common_len = len0 < len1 ? len0 : len1;
    sprec_t *bdp0 = dp[0], *bdp1 = dp[1];
    int32_t s0 = stride2[0], s1 = stride2[1];
    if (uoffset[0] > uoffset[1]) {
      std::swap(bdp0, bdp1);
      std::swap(s0, s1);
    }
    for (int32_t v = 0, vb = vstart[0]; vb < vstop[0]; ++vb, ++v) {
      sprec_t *sp    = buf + (2 * v + voffset[0]) * stride;
      size_t len     = static_cast<size_t>(common_len);
      sprec_t *line0 = bdp0 + static_cast<ptrdiff_t>(v) * s0;
      sprec_t *line1 = bdp1 + static_cast<ptrdiff_t>(v) * s1;
      for (; len >= 4; len -= 4) {
        __m256i vline = _mm256_loadu_si256((__m256i *)sp);
        vline         = _mm256_shuffle_epi32(vline, 0xD8);
        vline         = _mm256_permute4x64_epi64(vline, 0xD8);
        _mm256_storeu2_m128i((__m128i *)line1, (__m128i *)line0, vline);
        line0 += 4;
        line1 += 4;
        sp += 8;
      }
      for (; len > 0; --len) {
        *line0++ = *sp++;
        *line1++ = *sp++;
      }
    }
    // Write the one extra sample from the wider band into its sub-band array.
    for (uint8_t b = 0; b < 2; ++b) {
      if ((ustop[b] - ustart[b]) > common_len) {
        for (int32_t v = 0, vb = vstart[b]; vb < vstop[b]; ++vb, ++v) {
          dp[b][static_cast<ptrdiff_t>(v) * stride2[b] + common_len] =
              buf[2 * common_len + uoffset[b] + (2 * v + voffset[b]) * stride];
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
    sprec_t *bdp2 = dp[2], *bdp3 = dp[3];
    int32_t s2 = stride2[2], s3 = stride2[3];
    if (uoffset[2] > uoffset[3]) {
      std::swap(bdp2, bdp3);
      std::swap(s2, s3);
    }
    for (int32_t v = 0, vb = vstart[2]; vb < vstop[2]; ++vb, ++v) {
      sprec_t *sp    = buf + (2 * v + voffset[2]) * stride;
      size_t len     = static_cast<size_t>(common_len);
      sprec_t *line0 = bdp2 + static_cast<ptrdiff_t>(v) * s2;
      sprec_t *line1 = bdp3 + static_cast<ptrdiff_t>(v) * s3;
      for (; len >= 4; len -= 4) {
        __m256i vline = _mm256_loadu_si256((__m256i *)sp);
        vline         = _mm256_shuffle_epi32(vline, 0xD8);
        vline         = _mm256_permute4x64_epi64(vline, 0xD8);
        _mm256_storeu2_m128i((__m128i *)line1, (__m128i *)line0, vline);
        line0 += 4;
        line1 += 4;
        sp += 8;
      }
      for (; len > 0; --len) {
        *line0++ = *sp++;
        *line1++ = *sp++;
      }
    }
    // Write the one extra sample from the wider band.
    for (uint8_t b = 2; b < 4; ++b) {
      if ((ustop[b] - ustart[b]) > common_len) {
        for (int32_t v = 0, vb = vstart[b]; vb < vstop[b]; ++vb, ++v) {
          dp[b][static_cast<ptrdiff_t>(v) * stride2[b] + common_len] =
              buf[2 * common_len + uoffset[b] + (2 * v + voffset[b]) * stride];
        }
        break;
      }
    }
  }
#else
  for (uint8_t b = 0; b < 4; ++b) {
    for (int32_t v = 0, vb = vstart[b]; vb < vstop[b]; ++vb, ++v) {
      sprec_t *line = dp[b] + v * stride2[b];
      for (int32_t u = 0, ub = ustart[b]; ub < ustop[b]; ++ub, ++u) {
        *(line++) = buf[2 * u + uoffset[b] + (2 * v + voffset[b]) * stride];
      }
    }
  }
#endif
}

// 2D FDWT function
void fdwt_2d_sr_fixed(sprec_t *previousLL, sprec_t *LL, sprec_t *HL, sprec_t *LH, sprec_t *HH,
                      const int32_t u0, const int32_t u1, const int32_t v0, const int32_t v1,
                      const uint8_t transformation, sprec_t *pse_scratch, sprec_t **buf_scratch) {
  const int32_t stride = round_up(u1 - u0, 32);
  sprec_t *src         = previousLL;

  // Vertical DWT (pse_scratch provided by caller, sized for 8 * round_up(stride, SIMD_LEN_I32))
  if (transformation < 2)
    fdwt_ver_sr_fixed[transformation](src, u0, u1, v0, v1, stride, pse_scratch, buf_scratch);
  else
    fdwt_irrev53_ver_sr_fixed(src, u0, u1, v0, v1, stride, pse_scratch, buf_scratch);

  // Horizontal DWT
  fdwt_hor_sr_fixed(src, u0, u1, v0, v1, transformation, stride);

  fdwt_2d_deinterleave_fixed(src, LL, HL, LH, HH, u0, u1, v0, v1, stride);
}

// =============================================================================
// Streaming 2D FDWT — fdwt_2d_state
// =============================================================================
//
// Vertical analysis PSE counts: [v0%2][transform].
// Note: FDWT PSE is the mirror of IDWT (top↔bottom).
static constexpr int8_t kPseFdwtTop[2][2] = {{4, 2}, {3, 1}};
static constexpr int8_t kPseFdwtBot[2][2] = {{3, 1}, {4, 2}};

// Horizontal PSE sizes per [edge_parity][transform]:
// num_pse_i0[u0%2][transform] / num_pse_i1[u1%2][transform] in fdwt_hor_sr_fixed.
static constexpr int32_t kHorizLeft[2][2]  = {{4, 2}, {3, 1}};
static constexpr int32_t kHorizRight[2][2] = {{3, 1}, {4, 2}};

// For FDWT, LP rows are at even absolute row indices, HP at odd.
static inline bool is_lp_fdwt(int32_t r) { return (r % 2) == 0; }

// Max d_level before row is output-ready: 2 for 9/7, 1 for 5/3.
static inline int8_t max_dl_fdwt(uint8_t transform) { return (transform == 0) ? 2 : 1; }

// Physical source row for FDWT PSE position p.
static inline int32_t pse_src_fdwt(int32_t p, int32_t v0, int32_t v1) {
  return v0 + PSEo(p, v0, v1);
}

// Ring slot is r % FDWT_STATE_RING_DEPTH (fixed per row, independent of ring_origin).
// Pointer to ring / PSE row buffer for physical row r.
static sprec_t *rptr_f(const fdwt_2d_state *s, int32_t r) {
  if (r >= s->v0 && r < s->v1)
    return s->ring_buf + static_cast<ptrdiff_t>(r % FDWT_STATE_RING_DEPTH) * s->stride;
  if (r < s->v0)
    return s->top_pse_buf + static_cast<ptrdiff_t>(s->v0 - 1 - r) * s->stride;
  return s->bot_pse_buf + static_cast<ptrdiff_t>(r - s->v1) * s->stride;
}

static int8_t get_dl_f(const fdwt_2d_state *s, int32_t r) {
  if (r >= s->v0 && r < s->v1) {
    if (r < s->ring_origin || r >= s->ring_origin + FDWT_STATE_RING_DEPTH) return -1;
    return s->d_level[r % FDWT_STATE_RING_DEPTH];
  }
  if (r >= s->v0 - s->top_pse && r < s->v0) return s->top_dlevel[s->v0 - 1 - r];
  if (r >= s->v1 && r < s->v1 + s->bottom_pse) return s->bot_dlevel[r - s->v1];
  return -1;
}

static void set_dl_f(fdwt_2d_state *s, int32_t r, int8_t lv) {
  if (r >= s->v0 && r < s->v1) {
    s->d_level[r % FDWT_STATE_RING_DEPTH] = lv; return;
  }
  if (r >= s->v0 - s->top_pse && r < s->v0) { s->top_dlevel[s->v0 - 1 - r] = lv; return; }
  if (r >= s->v1 && r < s->v1 + s->bottom_pse) { s->bot_dlevel[r - s->v1] = lv; }
}

// FDWT dependency rule:
//   HP rows (odd): step A (cur=0→1) needs LP@0; step C (cur=1→2) needs LP@1.
//   LP rows (even): step B (cur=0→1) needs HP@1; step D (cur=1→2) needs HP@2.
static int8_t needed_neighbor_dl_f(const fdwt_2d_state *s, int32_t r) {
  const bool lp  = is_lp_fdwt(r);
  const int8_t cur = get_dl_f(s, r);
  if (s->transformation == 0) {  // irrev 9/7
    if (!lp) return (cur == 0) ? 0 : 1;   // HP: step A needs LP@0, step C needs LP@1
    else     return (cur == 0) ? 1 : 2;   // LP: step B needs HP@1, step D needs HP@2
  } else {                                 // rev 5/3
    return lp ? 1 : 0;                    // LP: update needs HP@1, HP: predict needs LP@0
  }
}

static bool can_adv_f(const fdwt_2d_state *s, int32_t r) {
  const int8_t cur = get_dl_f(s, r);
  if (cur < 0 || cur >= max_dl_fdwt(s->transformation)) return false;
  const int8_t need = needed_neighbor_dl_f(s, r);
  return get_dl_f(s, r - 1) >= need && get_dl_f(s, r + 1) >= need;
}

static void adv_step_f(fdwt_2d_state *s, int32_t r) {
  const bool   lp  = is_lp_fdwt(r);
  const int8_t cur = get_dl_f(s, r);
  sprec_t *tgt  = rptr_f(s, r);
  sprec_t *prev = rptr_f(s, r - 1);
  sprec_t *next = rptr_f(s, r + 1);
  const int32_t w = s->u1 - s->u0;

  if (s->transformation == 0) {  // irrev 9/7
    // HP: step A (coeff fA), step C (coeff fC); LP: step B (coeff fB), step D (coeff fD)
    // FDWT does tgt += coeff*(prev+next); dispatch fn does tgt -= coeff*(prev+next) → negate.
    const float coeff = lp ? (cur == 0 ? fB : fD) : (cur == 0 ? fA : fC);
    adv_fdwt_irrev_step_fn(w, prev, next, tgt, -coeff);
  } else if (s->transformation >= 2) {  // ATK irrev 5/3
    // HP predict: HP -= 0.5*(LP[r-1]+LP[r+1]) → coeff=0.5 (fn subtracts → pass 0.5)
    // LP update:  LP += 0.25*(HP[r-1]+HP[r+1]) → fn subtracts → pass -0.25
    const float coeff = lp ? -0.25f : 0.5f;
    adv_fdwt_irrev_step_fn(w, prev, next, tgt, coeff);
  } else {  // rev 5/3
    if (!lp) {  // HP predict: HP -= floor((LP[r-1] + LP[r+1]) * 0.5f)
      adv_fdwt_rev_hp_step_fn(w, prev, next, tgt);
    } else {  // LP update: LP += floor((HP[r-1] + HP[r+1] + 2) * 0.25f)
      adv_fdwt_rev_lp_step_fn(w, prev, next, tgt);
    }
  }
  set_dl_f(s, r, cur + 1);
}

// Fill PSE slots reflecting from just-pushed row r.
static void fill_pse_f(fdwt_2d_state *s, int32_t r) {
  const size_t nb = sizeof(sprec_t) * static_cast<size_t>(s->stride);
  const sprec_t *src = rptr_f(s, r);
  for (int8_t i = 1; i <= s->top_pse; ++i) {
    if (s->top_dlevel[i - 1] < 0 && pse_src_fdwt(s->v0 - i, s->v0, s->v1) == r) {
      memcpy(s->top_pse_buf + static_cast<ptrdiff_t>(i - 1) * s->stride, src, nb);
      s->top_dlevel[i - 1] = 0;
    }
  }
  for (int8_t i = 0; i < s->bottom_pse; ++i) {
    if (s->bot_dlevel[i] < 0 && pse_src_fdwt(s->v1 + i, s->v0, s->v1) == r) {
      memcpy(s->bot_pse_buf + static_cast<ptrdiff_t>(i) * s->stride, src, nb);
      s->bot_dlevel[i] = 0;
    }
  }
}

// Run cascade until stable.
//
// Key optimisation — dynamic lo:
//   The naive lo = v0 - top_pse is fixed, so the scan window grows O(n) per call
//   and the total cascade work is O(n²) in the tile height.  Once ring_origin has
//   advanced past the initial PSE zone, scanning earlier rows is wasted work.
//   Using lo = max(v0-top_pse, ring_origin-margin) caps the window to a constant
//   number of rows per call, making total cascade work O(n).
//
// For rev 5/3 and ATK (max_dl=1): the cascade is strictly 2-phase:
//   Phase 1 — HP Predict (odd  rows): needs LP neighbors at dl >= 0
//   Phase 2 — LP Update  (even rows): needs HP neighbors at dl >= 1
// Two dedicated single passes replace the while(progress) loop.
//
// For irrev 9/7 (max_dl=2): use the generic while(progress) loop (also with dynamic lo).
static void cascade_f(fdwt_2d_state *s) {
  const int32_t margin  = (int32_t)s->top_pse + max_dl_fdwt(s->transformation) * 2 + 2;
  const int32_t lo_full = s->v0 - (int32_t)s->top_pse;
  const int32_t lo      = (s->ring_origin - margin > lo_full)
                          ? s->ring_origin - margin : lo_full;
  const int32_t hi      = (s->next_in < s->v1) ? s->next_in + s->bottom_pse
                                                : s->v1 + s->bottom_pse;

  if (s->transformation != 0) {  // 2-step filters: rev53 and ATK irrev53 (max_dl=1)
    // Phase 1: Predict HP rows (odd absolute index) — need LP neighbors at dl >= 0.
    const int32_t hp0 = lo + (1 - (lo & 1));  // first odd row >= lo
    for (int32_t r = hp0; r < hi; r += 2) {
      if (get_dl_f(s, r) == 0 && get_dl_f(s, r - 1) >= 0 && get_dl_f(s, r + 1) >= 0)
        adv_step_f(s, r);
    }
    // Phase 2: Update LP rows (even absolute index) — need HP neighbors at dl >= 1.
    const int32_t lp0 = lo + (lo & 1);  // first even row >= lo
    for (int32_t r = lp0; r < hi; r += 2) {
      if (get_dl_f(s, r) == 0 && get_dl_f(s, r - 1) >= 1 && get_dl_f(s, r + 1) >= 1)
        adv_step_f(s, r);
    }
  } else {
    // Irrev 9/7: generic while(progress) loop (max_dl=2, cascade ≤ 4 passes).
    bool progress = true;
    while (progress) {
      progress = false;
      for (int32_t r = lo; r < hi; ++r) {
        if (can_adv_f(s, r)) { adv_step_f(s, r); progress = true; }
      }
    }
  }
}

// Emit all rows that have reached max_dl, in order from next_emit.
static void emit_ready_f(fdwt_2d_state *s) {
  const int8_t mxdl = max_dl_fdwt(s->transformation);
  while (s->next_emit < s->v1 && get_dl_f(s, s->next_emit) >= mxdl) {
    const int32_t r = s->next_emit;
    const bool    is_hp = !is_lp_fdwt(r);

    // Copy ring row into horiz_tmp without modifying the ring (other rows may
    // still reference r as a vertical-lifting neighbour).
    const sprec_t *ring_row = rptr_f(s, r);

    if (s->u1 == s->u0 + 1) {
      // Single-column: skip PSE/filter (PSEo has UB for length-1 signals).
      // Match fdwt_hor_sr_fixed: LP even u0 → no-op; HP odd u0 → *2 for 5/3.
      sprec_t *out = s->horiz_tmp + s->horiz_left;
      out[0] = ring_row[0];
      if (s->transformation == 1 && (s->u0 % 2 != 0)) out[0] = floorf(out[0] * 2.0f);
    } else {
      dwt_1d_extr_fixed(s->horiz_tmp, const_cast<sprec_t *>(ring_row),
                        s->horiz_left, s->horiz_right, s->u0, s->u1);
      if (s->transformation < 2)
        fdwt_1d_filtr_fixed[s->transformation](s->horiz_tmp, s->horiz_left, s->u0, s->u1);
      else
        fdwt_1d_filtr_irrev53_fixed(s->horiz_tmp, s->horiz_left, s->u0, s->u1);
    }

    s->put_row(s->sink_ctx, is_hp, r, s->horiz_tmp + s->horiz_left);
    ++s->next_emit;

    // Reclaim ring slots that are at least 4 rows behind next_emit (safe look-back).
    while (s->ring_origin < s->next_emit - 4 &&
           get_dl_f(s, s->ring_origin) >= mxdl) {
      s->d_level[s->ring_origin % FDWT_STATE_RING_DEPTH] = -1;
      ++s->ring_origin;
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────

void fdwt_2d_state_init(fdwt_2d_state *s,
                        const int32_t u0, const int32_t u1,
                        const int32_t v0, const int32_t v1,
                        const uint8_t transformation,
                        fdwt_row_sink_fn sink_fn, void *sink_ctx) {
  s->u0            = u0;  s->u1 = u1;
  s->v0            = v0;  s->v1 = v1;
  s->stride        = round_up(u1 - u0, SIMD_PADDING);
  s->transformation = transformation;
  const int32_t cls = (transformation == 0) ? 0 : 1;
  s->top_pse       = kPseFdwtTop[v0 % 2][cls];
  s->bottom_pse    = kPseFdwtBot[v1 % 2][cls];
  s->horiz_left    = kHorizLeft[u0 % 2][cls];
  s->horiz_right   = kHorizRight[u1 % 2][cls];

  const size_t row_bytes = sizeof(sprec_t) * static_cast<size_t>(s->stride);
  s->ring_buf    = static_cast<sprec_t *>(aligned_mem_alloc(FDWT_STATE_RING_DEPTH * row_bytes, 32));
  s->top_pse_buf = (s->top_pse    > 0) ? static_cast<sprec_t *>(aligned_mem_alloc(static_cast<size_t>(s->top_pse)    * row_bytes, 32)) : nullptr;
  s->bot_pse_buf = (s->bottom_pse > 0) ? static_cast<sprec_t *>(aligned_mem_alloc(static_cast<size_t>(s->bottom_pse) * row_bytes, 32)) : nullptr;

  const size_t htmp_bytes = sizeof(sprec_t) *
      static_cast<size_t>(s->horiz_left + s->stride + s->horiz_right + SIMD_PADDING);
  s->horiz_tmp = static_cast<sprec_t *>(aligned_mem_alloc(htmp_bytes, 32));

  s->ring_origin = v0;
  for (int32_t i = 0; i < FDWT_STATE_RING_DEPTH; ++i) s->d_level[i]    = -1;
  for (int32_t i = 0; i < 4;                     ++i) s->top_dlevel[i] = -1;
  for (int32_t i = 0; i < 4;                     ++i) s->bot_dlevel[i] = -1;

  s->next_in   = v0;
  s->next_emit = v0;
  s->put_row   = sink_fn;
  s->sink_ctx  = sink_ctx;
}

void fdwt_2d_state_free(fdwt_2d_state *s) {
  aligned_mem_free(s->ring_buf);    s->ring_buf    = nullptr;
  aligned_mem_free(s->top_pse_buf); s->top_pse_buf = nullptr;
  aligned_mem_free(s->bot_pse_buf); s->bot_pse_buf = nullptr;
  aligned_mem_free(s->horiz_tmp);   s->horiz_tmp   = nullptr;
}

void fdwt_2d_state_push_row(fdwt_2d_state *s, const sprec_t *in) {
  if (s->next_in >= s->v1) return;

  // Special case: single-row tile (handled by flush()).
  if (s->v1 == s->v0 + 1) {
    // Still store the row in ring_buf so flush() can read it.
    const int32_t slot = s->next_in % FDWT_STATE_RING_DEPTH;
    memcpy(s->ring_buf + static_cast<ptrdiff_t>(slot) * s->stride, in,
           sizeof(sprec_t) * static_cast<size_t>(s->u1 - s->u0));
    ++s->next_in;
    return;
  }

  const int32_t r    = s->next_in;
  const int32_t slot = r % FDWT_STATE_RING_DEPTH;

  memcpy(s->ring_buf + static_cast<ptrdiff_t>(slot) * s->stride, in,
         sizeof(sprec_t) * static_cast<size_t>(s->u1 - s->u0));
  s->d_level[slot] = 0;
  ++s->next_in;

  fill_pse_f(s, r);
  cascade_f(s);
  emit_ready_f(s);
}

void fdwt_2d_state_flush(fdwt_2d_state *s) {
  // Handle single-row tile special case.
  if (s->v1 == s->v0 + 1 && s->next_in == s->v0 + 1 && s->next_emit == s->v0) {
    // Single-row: LL-only or HP-only scaling.
    // For 5/3 with v0 odd: HP *= 2; for irrev: no-op.
    const bool is_hp = !is_lp_fdwt(s->v0);
    const sprec_t *src = s->ring_buf + static_cast<ptrdiff_t>(s->v0 % FDWT_STATE_RING_DEPTH) * s->stride;
    sprec_t *out = s->horiz_tmp + s->horiz_left;
    if (s->u1 == s->u0 + 1) {
      // Single-column: PSEo has UB for length-1 — bypass filter entirely.
      // Match fdwt_hor_sr_fixed: even u0 (LP) → no-op; odd u0 (HP) → *2 for 5/3.
      out[0] = src[0];
      if (s->transformation == 1) {
        if (s->v0 % 2 != 0) out[0] = floorf(out[0] * 2.0f);  // vertical HP: match fdwt_rev_ver_sr_fixed
        if (s->u0 % 2 != 0) out[0] = floorf(out[0] * 2.0f);  // horizontal HP: match fdwt_hor_sr_fixed
      }
      // ATK (transformation>=2) irrev53: no scaling needed for single-sample case.
    } else {
      // Match fdwt_rev_ver_sr_fixed: scale HP row by 2 BEFORE horizontal DWT.
      // (floorf makes 5/3 non-linear, so order matters.)
      if (s->transformation == 1 && is_hp) {
        sprec_t *ms = s->ring_buf + static_cast<ptrdiff_t>(s->v0 % FDWT_STATE_RING_DEPTH) * s->stride;
        const int32_t w = s->u1 - s->u0;
        for (int32_t c = 0; c < w; ++c) ms[c] = floorf(ms[c] * 2.0f);
      }
      dwt_1d_extr_fixed(s->horiz_tmp, const_cast<sprec_t *>(src),
                        s->horiz_left, s->horiz_right, s->u0, s->u1);
      if (s->transformation < 2)
        fdwt_1d_filtr_fixed[s->transformation](s->horiz_tmp, s->horiz_left, s->u0, s->u1);
      else
        fdwt_1d_filtr_irrev53_fixed(s->horiz_tmp, s->horiz_left, s->u0, s->u1);
    }
    s->put_row(s->sink_ctx, is_hp, s->v0, out);
    ++s->next_emit;
    return;
  }

  // After last push, bottom PSE may not yet be filled; fill them now.
  // (Some bottom PSE sources reflect to early rows already in ring.)
  for (int8_t i = 0; i < s->bottom_pse; ++i) {
    if (s->bot_dlevel[i] < 0) {
      const int32_t src_r = pse_src_fdwt(s->v1 + i, s->v0, s->v1);
      if (src_r >= s->ring_origin && src_r < s->ring_origin + FDWT_STATE_RING_DEPTH &&
          get_dl_f(s, src_r) >= 0) {
        fill_pse_f(s, src_r);
      }
    }
  }
  cascade_f(s);
  emit_ready_f(s);
}