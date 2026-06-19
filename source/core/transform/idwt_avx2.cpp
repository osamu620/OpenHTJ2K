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

#include "open_htj2k_typedef.hpp"
#include "utils.hpp"
#if defined(OPENHTJ2K_TRY_AVX2) && defined(__AVX2__)
  #include "dwt.hpp"
  #include <cstring>
  #include <cmath>
  #include <vector>
/********************************************************************************
 * horizontal transforms
 *******************************************************************************/
// irreversible IDWT
auto idwt_irrev97_fixed_avx2_hor_step = [](const int32_t init_pos, const int32_t simdlen, float *const X,
                                            const int32_t n0, const int32_t n1, const float fV) {
  auto vcoeff = _mm256_set1_ps(fV);
  int32_t n = init_pos, i = 0;
  // 2× unrolled main loop: two independent 4-sample groups per iteration for better ILP.
  // slli_epi64(sum,32) replaces blend(0xAA)+slli_si256(4): shifts each 64-bit lane left 32 bits,
  // which places even-indexed sums at odd (update) positions and zeros the even positions.
  // fnmadd(sum, coeff, xin0) = xin0 - coeff*sum, fusing mul+sub into one FMA instruction.
  for (; i + 4 < simdlen; i += 8, n += 16) {
    auto xin0a = _mm256_loadu_ps(X + n + n0);
    auto xin2a = _mm256_loadu_ps(X + n + n1);
    auto xin0b = _mm256_loadu_ps(X + n + 8 + n0);
    auto xin2b = _mm256_loadu_ps(X + n + 8 + n1);
    auto xsuma = _mm256_castsi256_ps(_mm256_slli_epi64(_mm256_castps_si256(_mm256_add_ps(xin0a, xin2a)), 32));
    auto xsumb = _mm256_castsi256_ps(_mm256_slli_epi64(_mm256_castps_si256(_mm256_add_ps(xin0b, xin2b)), 32));
    xin0a = _mm256_fnmadd_ps(xsuma, vcoeff, xin0a);
    xin0b = _mm256_fnmadd_ps(xsumb, vcoeff, xin0b);
    _mm256_storeu_ps(X + n + n0, xin0a);
    _mm256_storeu_ps(X + n + 8 + n0, xin0b);
  }
  for (; i < simdlen; i += 4, n += 8) {
    auto xin0 = _mm256_loadu_ps(X + n + n0);
    auto xin2 = _mm256_loadu_ps(X + n + n1);
    auto xsum = _mm256_castsi256_ps(_mm256_slli_epi64(_mm256_castps_si256(_mm256_add_ps(xin0, xin2)), 32));
    xin0      = _mm256_fnmadd_ps(xsum, vcoeff, xin0);
    _mm256_storeu_ps(X + n + n0, xin0);
  }
};

void idwt_1d_filtr_irrev97_fixed_avx2(sprec_t *X, const int32_t left, const int32_t u_i0,
                                      const int32_t u_i1) {
  const auto i0        = static_cast<int32_t>(u_i0);
  const auto i1        = static_cast<int32_t>(u_i1);
  const int32_t start  = i0 / 2;
  const int32_t stop   = i1 / 2;
  const int32_t offset = left - i0 % 2;

  // step 1
  int32_t simdlen = stop + 2 - (start - 1);
  idwt_irrev97_fixed_avx2_hor_step(offset - 2, simdlen, X, -1, 1, fD);

  // step 2
  simdlen = stop + 1 - (start - 1);
  idwt_irrev97_fixed_avx2_hor_step(offset - 2, simdlen, X, 0, 2, fC);

  // step 3
  simdlen = stop + 1 - start;
  idwt_irrev97_fixed_avx2_hor_step(offset, simdlen, X, -1, 1, fB);

  // step 4
  simdlen = stop - start;
  idwt_irrev97_fixed_avx2_hor_step(offset, simdlen, X, 0, 2, fA);
}

// reversible IDWT
void idwt_1d_filtr_rev53_fixed_avx2(sprec_t *X, const int32_t left, const int32_t u_i0,
                                    const int32_t u_i1) {
  const auto i0        = static_cast<int32_t>(u_i0);
  const auto i1        = static_cast<int32_t>(u_i1);
  const int32_t start  = i0 / 2;
  const int32_t stop   = i1 / 2;
  const int32_t offset = left - i0 % 2;

  // step 1: L[k] -= floor((H[k-1] + H[k] + 2) * 0.25)
  // slli_epi64(sum,32): within each 64-bit lane, shifts the even-indexed float to the odd slot,
  // zeroing the even slot — replacing blend(0xAA)+slli_si256(4) with one instruction.
  int32_t simdlen = stop + 1 - start;
  sprec_t *sp     = X + offset;
  const auto xtwo = _mm256_set1_ps(2.0f);
  const auto x025 = _mm256_set1_ps(0.25f);
  int32_t i = 0;
  for (; i + 4 < simdlen; i += 8, sp += 16) {
    auto xin0a = _mm256_loadu_ps(sp - 1);
    auto xin2a = _mm256_loadu_ps(sp + 1);
    auto xin0b = _mm256_loadu_ps(sp + 7);
    auto xin2b = _mm256_loadu_ps(sp + 9);
    auto xsuma = _mm256_castsi256_ps(_mm256_slli_epi64(
        _mm256_castps_si256(_mm256_add_ps(_mm256_add_ps(xin0a, xin2a), xtwo)), 32));
    auto xsumb = _mm256_castsi256_ps(_mm256_slli_epi64(
        _mm256_castps_si256(_mm256_add_ps(_mm256_add_ps(xin0b, xin2b), xtwo)), 32));
    xin0a = _mm256_sub_ps(xin0a, _mm256_floor_ps(_mm256_mul_ps(xsuma, x025)));
    xin0b = _mm256_sub_ps(xin0b, _mm256_floor_ps(_mm256_mul_ps(xsumb, x025)));
    _mm256_storeu_ps(sp - 1, xin0a);
    _mm256_storeu_ps(sp + 7, xin0b);
  }
  for (; i < simdlen; i += 4, sp += 8) {
    auto xin0 = _mm256_loadu_ps(sp - 1);
    auto xin2 = _mm256_loadu_ps(sp + 1);
    auto xsum = _mm256_castsi256_ps(_mm256_slli_epi64(
        _mm256_castps_si256(_mm256_add_ps(_mm256_add_ps(xin0, xin2), xtwo)), 32));
    xin0 = _mm256_sub_ps(xin0, _mm256_floor_ps(_mm256_mul_ps(xsum, x025)));
    _mm256_storeu_ps(sp - 1, xin0);
  }

  // step 2: H[k] += floor((L[k] + L[k+1]) * 0.5)
  simdlen = stop - start;
  sp      = X + offset;
  const auto x05 = _mm256_set1_ps(0.5f);
  i = 0;
  for (; i + 4 < simdlen; i += 8, sp += 16) {
    auto xin0a = _mm256_loadu_ps(sp);
    auto xin2a = _mm256_loadu_ps(sp + 2);
    auto xin0b = _mm256_loadu_ps(sp + 8);
    auto xin2b = _mm256_loadu_ps(sp + 10);
    auto xsuma = _mm256_castsi256_ps(_mm256_slli_epi64(
        _mm256_castps_si256(_mm256_add_ps(xin0a, xin2a)), 32));
    auto xsumb = _mm256_castsi256_ps(_mm256_slli_epi64(
        _mm256_castps_si256(_mm256_add_ps(xin0b, xin2b)), 32));
    xin0a = _mm256_add_ps(xin0a, _mm256_floor_ps(_mm256_mul_ps(xsuma, x05)));
    xin0b = _mm256_add_ps(xin0b, _mm256_floor_ps(_mm256_mul_ps(xsumb, x05)));
    _mm256_storeu_ps(sp, xin0a);
    _mm256_storeu_ps(sp + 8, xin0b);
  }
  for (; i < simdlen; i += 4, sp += 8) {
    auto xin0 = _mm256_loadu_ps(sp);
    auto xin2 = _mm256_loadu_ps(sp + 2);
    auto xsum = _mm256_castsi256_ps(_mm256_slli_epi64(
        _mm256_castps_si256(_mm256_add_ps(xin0, xin2)), 32));
    xin0 = _mm256_add_ps(xin0, _mm256_floor_ps(_mm256_mul_ps(xsum, x05)));
    _mm256_storeu_ps(sp, xin0);
  }
}

// ATK irreversible 5/3 horizontal IDWT — same structure as rev53 but without floor().
// Step 1: LP[k] -= 0.25*(HP[k-1] + HP[k])
// Step 2: HP[k] += 0.5*(LP_mod[k] + LP_mod[k+1])
// slli_epi64 trick: shifts each 64-bit lane left 32 bits, placing the even-element sum at the odd
// (update-target) slot and zeroing the even (pass-through) slot — no blend or separate scatter needed.
void idwt_1d_filtr_irrev53_fixed_avx2(sprec_t *X, const int32_t left, const int32_t u_i0,
                                      const int32_t u_i1) {
  const int32_t i0     = static_cast<int32_t>(u_i0);
  const int32_t i1     = static_cast<int32_t>(u_i1);
  const int32_t start  = i0 / 2;
  const int32_t stop   = i1 / 2;
  const int32_t offset = left - i0 % 2;

  // Step 1: LP[k] -= 0.25*(HP[k-1]+HP[k])
  // sp-1: [HP[k-1], LP[k], HP[k], LP[k+1], ...];  sp+1: [HP[k], LP[k+1], HP[k+1], ...]
  // slli places HP sums at LP (odd) positions; fnmadd updates only those, passes HP through.
  int32_t simdlen = stop + 1 - start;
  sprec_t *sp     = X + offset;
  const auto x025 = _mm256_set1_ps(0.25f);
  int32_t i = 0;
  for (; i + 4 < simdlen; i += 8, sp += 16) {
    auto xin0a = _mm256_loadu_ps(sp - 1);
    auto xin2a = _mm256_loadu_ps(sp + 1);
    auto xin0b = _mm256_loadu_ps(sp + 7);
    auto xin2b = _mm256_loadu_ps(sp + 9);
    auto xsuma = _mm256_castsi256_ps(_mm256_slli_epi64(_mm256_castps_si256(_mm256_add_ps(xin0a, xin2a)), 32));
    auto xsumb = _mm256_castsi256_ps(_mm256_slli_epi64(_mm256_castps_si256(_mm256_add_ps(xin0b, xin2b)), 32));
    xin0a = _mm256_fnmadd_ps(xsuma, x025, xin0a);
    xin0b = _mm256_fnmadd_ps(xsumb, x025, xin0b);
    _mm256_storeu_ps(sp - 1, xin0a);
    _mm256_storeu_ps(sp + 7, xin0b);
  }
  for (; i < simdlen; i += 4, sp += 8) {
    auto xin0 = _mm256_loadu_ps(sp - 1);
    auto xin2 = _mm256_loadu_ps(sp + 1);
    auto xsum = _mm256_castsi256_ps(_mm256_slli_epi64(_mm256_castps_si256(_mm256_add_ps(xin0, xin2)), 32));
    xin0      = _mm256_fnmadd_ps(xsum, x025, xin0);
    _mm256_storeu_ps(sp - 1, xin0);
  }

  // Step 2: HP[k] += 0.5*(LP_mod[k]+LP_mod[k+1])
  // sp: [LP[k], HP[k], ...];  sp+2: [LP[k+1], HP[k+1], ...]
  // slli places LP sums at HP (odd) positions; fmadd updates only those, passes LP through.
  simdlen = stop - start;
  sp      = X + offset;
  const auto x05 = _mm256_set1_ps(0.5f);
  i = 0;
  for (; i + 4 < simdlen; i += 8, sp += 16) {
    auto xin0a = _mm256_loadu_ps(sp);
    auto xin2a = _mm256_loadu_ps(sp + 2);
    auto xin0b = _mm256_loadu_ps(sp + 8);
    auto xin2b = _mm256_loadu_ps(sp + 10);
    auto xsuma = _mm256_castsi256_ps(_mm256_slli_epi64(_mm256_castps_si256(_mm256_add_ps(xin0a, xin2a)), 32));
    auto xsumb = _mm256_castsi256_ps(_mm256_slli_epi64(_mm256_castps_si256(_mm256_add_ps(xin0b, xin2b)), 32));
    xin0a = _mm256_fmadd_ps(xsuma, x05, xin0a);
    xin0b = _mm256_fmadd_ps(xsumb, x05, xin0b);
    _mm256_storeu_ps(sp, xin0a);
    _mm256_storeu_ps(sp + 8, xin0b);
  }
  for (; i < simdlen; i += 4, sp += 8) {
    auto xin0 = _mm256_loadu_ps(sp);
    auto xin2 = _mm256_loadu_ps(sp + 2);
    auto xsum = _mm256_castsi256_ps(_mm256_slli_epi64(_mm256_castps_si256(_mm256_add_ps(xin0, xin2)), 32));
    xin0      = _mm256_fmadd_ps(xsum, x05, xin0);
    _mm256_storeu_ps(sp, xin0);
  }
}

/********************************************************************************
 * vertical transform
 *******************************************************************************/
// irreversible IDWT
auto idwt_irrev97_fixed_avx2_ver_step = [](const int32_t simdlen, float *const Xin0, float *const Xin1,
                                           float *const Xout, const float coeff) {
  auto vcoeff  = _mm256_set1_ps(coeff);
  for (int32_t n = 0; n < simdlen; n += 8) {
    auto xin0 = _mm256_load_ps(Xin0 + n);
    auto xin2 = _mm256_load_ps(Xin1 + n);
    auto xout  = _mm256_load_ps(Xout + n);
    auto xsum = _mm256_add_ps(xin0, xin2);
    xout = _mm256_fnmadd_ps(xsum, vcoeff, xout);
    _mm256_store_ps(Xout + n, xout);
  }
};

// Single-row reversible (5/3) LP vertical lifting: tgt[i] -= floor((prev[i]+next[i]+2)*0.25)
// Single-row reversible (5/3) LP vertical lifting: tgt[i] -= floor((prev[i]+next[i]+2)*0.25)
// Called only from adv_step() with ring-buffer row pointers; those rows are 32-byte aligned
// (slot_stride is a multiple of 8 floats and ring_buf is 32-byte aligned), so _mm256_load_ps is safe.
void idwt_rev_ver_lp_step_avx2(int32_t n, const float *prev, const float *next, float *tgt) {
  const __m256 k025 = _mm256_set1_ps(0.25f);
  const __m256 k2   = _mm256_set1_ps(2.0f);
  int32_t i = 0;
  for (; i + 8 <= n; i += 8) {
    __m256 a = _mm256_load_ps(prev + i);
    __m256 b = _mm256_load_ps(next + i);
    __m256 t = _mm256_load_ps(tgt  + i);
    t = _mm256_sub_ps(t, _mm256_floor_ps(_mm256_mul_ps(_mm256_add_ps(_mm256_add_ps(a, b), k2), k025)));
    _mm256_store_ps(tgt + i, t);
  }
  for (; i < n; ++i)
    tgt[i] -= floorf((prev[i] + next[i] + 2.0f) * 0.25f);
}

// Single-row reversible (5/3) HP vertical lifting: tgt[i] += floor((prev[i]+next[i])*0.5)
// Same alignment guarantee as idwt_rev_ver_lp_step_avx2 above.
void idwt_rev_ver_hp_step_avx2(int32_t n, const float *prev, const float *next, float *tgt) {
  const __m256 k05 = _mm256_set1_ps(0.5f);
  int32_t i = 0;
  for (; i + 8 <= n; i += 8) {
    __m256 a = _mm256_load_ps(prev + i);
    __m256 b = _mm256_load_ps(next + i);
    __m256 t = _mm256_load_ps(tgt  + i);
    t = _mm256_add_ps(t, _mm256_floor_ps(_mm256_mul_ps(_mm256_add_ps(a, b), k05)));
    _mm256_store_ps(tgt + i, t);
  }
  for (; i < n; ++i)
    tgt[i] += floorf((prev[i] + next[i]) * 0.5f);
}

// Single-row irreversible vertical lifting step for idwt_2d_state::adv_step().
// Applies tgt[i] -= coeff*(prev[i]+next[i]) using FMA, matching the batch path exactly.
// n is the row width; the ring-buffer rows are 32-byte aligned so load_ps is safe.
void idwt_irrev_ver_step_fixed_avx2(int32_t n, float *prev, float *next, float *tgt, float coeff) {
  auto vcoeff = _mm256_set1_ps(coeff);
  int32_t i   = 0;
  for (; i + 8 <= n; i += 8) {
    auto xin0 = _mm256_load_ps(prev + i);
    auto xin1 = _mm256_load_ps(next + i);
    auto xout = _mm256_load_ps(tgt  + i);
    xout = _mm256_fnmadd_ps(_mm256_add_ps(xin0, xin1), vcoeff, xout);
    _mm256_store_ps(tgt + i, xout);
  }
  for (; i < n; ++i)
    tgt[i] -= coeff * (prev[i] + next[i]);
}

void idwt_irrev_ver_sr_fixed_avx2(sprec_t *in, const int32_t u0, const int32_t u1, const int32_t v0,
                                  const int32_t v1, const int32_t stride, sprec_t *pse_scratch, sprec_t **buf_scratch) {
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
      buf[top - i] = pse_scratch + (i - 1) * len;
      memcpy(buf[top - i], &in[PSEo(v0 - i, v0, v1) * stride],
             sizeof(sprec_t) * static_cast<size_t>(stride));
    }
    for (int32_t row = 0; row < v1 - v0; ++row) {
      buf[top + row] = &in[row * stride];
    }
    for (int32_t i = 1; i <= bottom; i++) {
      buf[top + (v1 - v0) + i - 1] = pse_scratch + (top + i - 1) * len;
      memcpy(buf[top + (v1 - v0) + i - 1], &in[PSEo(v1 - v0 + i - 1 + v0, v0, v1) * stride],
             sizeof(sprec_t) * static_cast<size_t>(stride));
    }
    const int32_t start  = v0 / 2;
    const int32_t stop   = v1 / 2;
    const int32_t offset = top - v0 % 2;

    const int32_t width = u1 - u0;
    for (int32_t cs = 0; cs < width; cs += DWT_VERT_STRIP) {
      const int32_t ce        = (cs + DWT_VERT_STRIP < width) ? cs + DWT_VERT_STRIP : width;
      const int32_t simdlen_s = (ce - cs) - (ce - cs) % 8;
      for (int32_t n = -2 + offset, i = start - 1; i < stop + 2; i++, n += 2) {
        idwt_irrev97_fixed_avx2_ver_step(simdlen_s, buf[n - 1] + cs, buf[n + 1] + cs, buf[n] + cs, fD);
        for (int32_t col = cs + simdlen_s; col < ce; ++col) {
          buf[n][col] -= fD * (buf[n - 1][col] + buf[n + 1][col]);
        }
      }
      for (int32_t n = -2 + offset, i = start - 1; i < stop + 1; i++, n += 2) {
        idwt_irrev97_fixed_avx2_ver_step(simdlen_s, buf[n] + cs, buf[n + 2] + cs, buf[n + 1] + cs, fC);
        for (int32_t col = cs + simdlen_s; col < ce; ++col) {
          buf[n + 1][col] -= fC * (buf[n][col] + buf[n + 2][col]);
        }
      }
      for (int32_t n = 0 + offset, i = start; i < stop + 1; i++, n += 2) {
        idwt_irrev97_fixed_avx2_ver_step(simdlen_s, buf[n - 1] + cs, buf[n + 1] + cs, buf[n] + cs, fB);
        for (int32_t col = cs + simdlen_s; col < ce; ++col) {
          buf[n][col] -= fB * (buf[n - 1][col] + buf[n + 1][col]);
        }
      }
      for (int32_t n = 0 + offset, i = start; i < stop; i++, n += 2) {
        idwt_irrev97_fixed_avx2_ver_step(simdlen_s, buf[n] + cs, buf[n + 2] + cs, buf[n + 1] + cs, fA);
        for (int32_t col = cs + simdlen_s; col < ce; ++col) {
          buf[n + 1][col] -= fA * (buf[n][col] + buf[n + 2][col]);
        }
      }
    }
  }
}

// reversible IDWT
void idwt_rev_ver_sr_fixed_avx2(sprec_t *in, const int32_t u0, const int32_t u1, const int32_t v0,
                                const int32_t v1, const int32_t stride, sprec_t *pse_scratch, sprec_t **buf_scratch) {
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
      buf[top - i] = pse_scratch + (i - 1) * len;
      memcpy(buf[top - i], &in[PSEo(v0 - i, v0, v1) * stride],
             sizeof(sprec_t) * static_cast<size_t>(stride));
    }
    for (int32_t row = 0; row < v1 - v0; ++row) {
      buf[top + row] = &in[row * stride];
    }
    for (int32_t i = 1; i <= bottom; i++) {
      buf[top + (v1 - v0) + i - 1] = pse_scratch + (top + i - 1) * len;
      memcpy(buf[top + (v1 - v0) + i - 1], &in[PSEo(v1 - v0 + i - 1 + v0, v0, v1) * stride],
             sizeof(sprec_t) * static_cast<size_t>(stride));
    }
    const int32_t start  = v0 / 2;
    const int32_t stop   = v1 / 2;
    const int32_t offset = top - v0 % 2;

    const int32_t width = u1 - u0;
    for (int32_t cs = 0; cs < width; cs += DWT_VERT_STRIP) {
      const int32_t ce        = (cs + DWT_VERT_STRIP < width) ? cs + DWT_VERT_STRIP : width;
      const int32_t simdlen_s = (ce - cs) - (ce - cs) % 8;
      const __m256 xtwo = _mm256_set1_ps(2.0f);
      const __m256 x025 = _mm256_set1_ps(0.25f);
      for (int32_t n = 0 + offset, i = start; i < stop + 1; ++i, n += 2) {
        for (int32_t col = 0; col < simdlen_s; col += 8) {
          __m256 x0   = _mm256_load_ps(buf[n - 1] + cs + col);
          __m256 x2   = _mm256_load_ps(buf[n + 1] + cs + col);
          __m256 x1   = _mm256_load_ps(buf[n] + cs + col);
          auto xfloor = _mm256_floor_ps(_mm256_mul_ps(_mm256_add_ps(_mm256_add_ps(x0, x2), xtwo), x025));
          x1          = _mm256_sub_ps(x1, xfloor);
          _mm256_store_ps(buf[n] + cs + col, x1);
        }
        for (int32_t col = cs + simdlen_s; col < ce; ++col) {
          buf[n][col] -= floorf((buf[n - 1][col] + buf[n + 1][col] + 2.0f) * 0.25f);
        }
      }
      const __m256 x05 = _mm256_set1_ps(0.5f);
      for (int32_t n = 0 + offset, i = start; i < stop; ++i, n += 2) {
        for (int32_t col = 0; col < simdlen_s; col += 8) {
          __m256 x0   = _mm256_load_ps(buf[n] + cs + col);
          __m256 x2   = _mm256_load_ps(buf[n + 2] + cs + col);
          __m256 x1   = _mm256_load_ps(buf[n + 1] + cs + col);
          auto xfloor = _mm256_floor_ps(_mm256_mul_ps(_mm256_add_ps(x0, x2), x05));
          x1          = _mm256_add_ps(x1, xfloor);
          _mm256_store_ps(buf[n + 1] + cs + col, x1);
        }
        for (int32_t col = cs + simdlen_s; col < ce; ++col) {
          buf[n + 1][col] += floorf((buf[n][col] + buf[n + 2][col]) * 0.5f);
        }
      }
    }
  }
}

// ATK irreversible 5/3 vertical IDWT — same structure as rev53 vertical but without floor().
// Step 1: LP[k] -= 0.25*(HP_above + HP_below)   [original HP rows]
// Step 2: HP[k] += 0.5*(LP_mod_above + LP_mod_below)  [LP rows modified by step 1]
// Row buffers (ring/PSE) are 32-byte aligned, so _mm256_load_ps is safe.
void idwt_irrev53_ver_sr_fixed_avx2(sprec_t *in, const int32_t u0, const int32_t u1, const int32_t v0,
                                    const int32_t v1, const int32_t stride, sprec_t *pse_scratch,
                                    sprec_t **buf_scratch) {
  constexpr int32_t num_pse_i0[2] = {1, 2};
  constexpr int32_t num_pse_i1[2] = {2, 1};
  const int32_t top    = num_pse_i0[v0 % 2];
  const int32_t bottom = num_pse_i1[v1 % 2];
  if (v0 == v1 - 1) {
    // single row: nothing to do
  } else {
    const int32_t len = round_up(stride, SIMD_PADDING);
    sprec_t **buf     = buf_scratch;
    for (int32_t i = 1; i <= top; ++i) {
      buf[top - i] = pse_scratch + (i - 1) * len;
      memcpy(buf[top - i], &in[PSEo(v0 - i, v0, v1) * stride], sizeof(sprec_t) * static_cast<size_t>(stride));
    }
    for (int32_t row = 0; row < v1 - v0; ++row) buf[top + row] = &in[row * stride];
    for (int32_t i = 1; i <= bottom; i++) {
      buf[top + (v1 - v0) + i - 1] = pse_scratch + (top + i - 1) * len;
      memcpy(buf[top + (v1 - v0) + i - 1], &in[PSEo(v1 - v0 + i - 1 + v0, v0, v1) * stride],
             sizeof(sprec_t) * static_cast<size_t>(stride));
    }
    // Per-step extension semantics: the LP pass also lifts the even PSE rows adjacent to the
    // data (same bounds as the rev53 vertical kernel) so the HP pass reads post-LP values.
    const int32_t lp_count = v1 / 2 - v0 / 2 + 1;  // LP rows incl. even PSE rows at the edges
    const int32_t hp_count = v1 / 2 - v0 / 2;      // HP row count
    const int32_t offset   = top - v0 % 2;         // first LP row (may be a PSE row)
    const int32_t width    = u1 - u0;
    for (int32_t cs = 0; cs < width; cs += DWT_VERT_STRIP) {
      const int32_t ce        = (cs + DWT_VERT_STRIP < width) ? cs + DWT_VERT_STRIP : width;
      const int32_t simdlen_s = (ce - cs) - (ce - cs) % 8;
      const __m256 x025       = _mm256_set1_ps(0.25f);
      for (int32_t k = 0, n = offset; k < lp_count; ++k, n += 2) {
        for (int32_t col = 0; col < simdlen_s; col += 8) {
          __m256 x0 = _mm256_load_ps(buf[n - 1] + cs + col);
          __m256 x2 = _mm256_load_ps(buf[n + 1] + cs + col);
          __m256 x1 = _mm256_load_ps(buf[n]     + cs + col);
          x1 = _mm256_fnmadd_ps(_mm256_add_ps(x0, x2), x025, x1);
          _mm256_store_ps(buf[n] + cs + col, x1);
        }
        for (int32_t col = cs + simdlen_s; col < ce; ++col)
          buf[n][col] -= 0.25f * (buf[n - 1][col] + buf[n + 1][col]);
      }
      const __m256 x05 = _mm256_set1_ps(0.5f);
      for (int32_t k = 0, n = offset; k < hp_count; ++k, n += 2) {
        for (int32_t col = 0; col < simdlen_s; col += 8) {
          __m256 x0 = _mm256_load_ps(buf[n]     + cs + col);
          __m256 x2 = _mm256_load_ps(buf[n + 2] + cs + col);
          __m256 x1 = _mm256_load_ps(buf[n + 1] + cs + col);
          x1 = _mm256_fmadd_ps(_mm256_add_ps(x0, x2), x05, x1);
          _mm256_store_ps(buf[n + 1] + cs + col, x1);
        }
        for (int32_t col = cs + simdlen_s; col < ce; ++col)
          buf[n + 1][col] += 0.5f * (buf[n][col] + buf[n + 2][col]);
      }
    }
  }
}

void idwt_1d_filtr_rev53_i32_avx2(int32_t *X, const int32_t left, const int32_t i0, const int32_t i1) {
  const int32_t start  = i0 / 2;
  const int32_t stop   = i1 / 2;
  const int32_t offset = left - i0 % 2;

  // step 1 (undo forward LP update): LP -= (HP_left + HP_right + 2) >> 2
  int32_t simdlen = stop + 1 - start;
  int32_t i = 0, n = offset;
  const __m256i vtwo = _mm256_set1_epi32(2);
  for (; i + 4 < simdlen; i += 8, n += 16) {
    __m256i xin0a = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(X + n - 1));
    __m256i xin2a = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(X + n + 1));
    __m256i xin0b = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(X + n + 7));
    __m256i xin2b = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(X + n + 9));
    __m256i xsuma = _mm256_slli_epi64(
        _mm256_add_epi32(_mm256_add_epi32(xin0a, xin2a), vtwo), 32);
    __m256i xsumb = _mm256_slli_epi64(
        _mm256_add_epi32(_mm256_add_epi32(xin0b, xin2b), vtwo), 32);
    xsuma = _mm256_srai_epi32(xsuma, 2);
    xsumb = _mm256_srai_epi32(xsumb, 2);
    xin0a = _mm256_sub_epi32(xin0a, xsuma);
    xin0b = _mm256_sub_epi32(xin0b, xsumb);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(X + n - 1), xin0a);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(X + n + 7), xin0b);
  }
  for (; i < simdlen; i += 4, n += 8) {
    __m256i xin0 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(X + n - 1));
    __m256i xin2 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(X + n + 1));
    __m256i xsum = _mm256_slli_epi64(
        _mm256_add_epi32(_mm256_add_epi32(xin0, xin2), vtwo), 32);
    xsum = _mm256_srai_epi32(xsum, 2);
    xin0 = _mm256_sub_epi32(xin0, xsum);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(X + n - 1), xin0);
  }

  // step 2 (undo forward HP predict): HP += (LP_left + LP_right) >> 1
  simdlen = stop - start;
  i = 0; n = offset;
  for (; i + 4 < simdlen; i += 8, n += 16) {
    __m256i xin0a = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(X + n));
    __m256i xin2a = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(X + n + 2));
    __m256i xin0b = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(X + n + 8));
    __m256i xin2b = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(X + n + 10));
    __m256i xsuma = _mm256_slli_epi64(_mm256_add_epi32(xin0a, xin2a), 32);
    __m256i xsumb = _mm256_slli_epi64(_mm256_add_epi32(xin0b, xin2b), 32);
    xsuma = _mm256_srai_epi32(xsuma, 1);
    xsumb = _mm256_srai_epi32(xsumb, 1);
    xin0a = _mm256_add_epi32(xin0a, xsuma);
    xin0b = _mm256_add_epi32(xin0b, xsumb);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(X + n), xin0a);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(X + n + 8), xin0b);
  }
  for (; i < simdlen; i += 4, n += 8) {
    __m256i xin0 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(X + n));
    __m256i xin2 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(X + n + 2));
    __m256i xsum = _mm256_slli_epi64(_mm256_add_epi32(xin0, xin2), 32);
    xsum = _mm256_srai_epi32(xsum, 1);
    xin0 = _mm256_add_epi32(xin0, xsum);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(X + n), xin0);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Planar-input horizontal synthesis — the LP and HP subband rows are read
// directly (E[j] = lp[j], O[j] = hp[j]) and the synthesised natural-domain row
// is written to out[].  8-lane port of the NEON planar kernels
// (idwt_neon.cpp): one plain load per plane replaces the in-place kernels'
// four overlapping passes over the interleaved row (which waste half their
// lanes on pass-through elements), and the caller's interleave pass
// disappears.  Boundary taps outside the planes use WSSE mirroring: positions
// u0-k and u0+k have equal parity, so each plane extends within itself;
// PSEo() yields the mirrored in-range position, whose half is the plane
// index.  Every lifting stage is the same single-rounded op on the same
// inputs as the in-place AVX2 kernels (_mm256_fnmadd_ps(sum, coeff, x) ==
// fmaf(-coeff, sum, x)) — output is bit-identical.
//
// Contract (enforced by idwt_1d_row_from_planar): u0 even, N = u1/2 - u0/2
// >= 16 (the 8-lane warmup loads blocks 0/1, j = 0..15, unconditionally),
// lp has ceil(u1/2) - u0/2 valid samples, hp has N, out has >= 2 writable
// floats before index 0 and >= 8 after index u1-u0-1.
// ─────────────────────────────────────────────────────────────────────────────

// Cross-element shifts with carry-in (the AVX2 equivalent of vextq_f32).
// shl1: [a1..a7, b0] — shift left one element, carry-in from next vector b.
static inline __m256 avx2_shl1_ps(__m256 a, __m256 b) {
  __m256 t = _mm256_permute2f128_ps(a, b, 0x21);  // [a4..a7, b0..b3]
  return _mm256_castsi256_ps(_mm256_alignr_epi8(_mm256_castps_si256(t), _mm256_castps_si256(a), 4));
}
// shr1: [p7, a0..a6] — shift right one element, carry-in from previous vector p.
static inline __m256 avx2_shr1_ps(__m256 p, __m256 a) {
  __m256 t = _mm256_permute2f128_ps(p, a, 0x21);  // [p4..p7, a0..a3]
  return _mm256_castsi256_ps(_mm256_alignr_epi8(_mm256_castps_si256(a), _mm256_castps_si256(t), 12));
}
static inline __m256i avx2_shl1_epi32(__m256i a, __m256i b) {
  __m256i t = _mm256_permute2x128_si256(a, b, 0x21);
  return _mm256_alignr_epi8(t, a, 4);
}
static inline __m256i avx2_shr1_epi32(__m256i p, __m256i a) {
  __m256i t = _mm256_permute2x128_si256(p, a, 0x21);
  return _mm256_alignr_epi8(a, t, 12);
}

// Interleaved store of one finished block: out[2j+0,2,..] = e, out[2j+1,3,..] = o.
// Inverse of the deinterleave pattern in interleave_row_planes (idwt.cpp).
static inline void avx2_store_interleaved_ps(float *dst, __m256 e, __m256 o) {
  __m256 lo = _mm256_unpacklo_ps(e, o);  // e0,o0,e1,o1, e4,o4,e5,o5
  __m256 hi = _mm256_unpackhi_ps(e, o);  // e2,o2,e3,o3, e6,o6,e7,o7
  _mm256_storeu_ps(dst, _mm256_permute2f128_ps(lo, hi, 0x20));
  _mm256_storeu_ps(dst + 8, _mm256_permute2f128_ps(lo, hi, 0x31));
}

void idwt_1d_filtr_irrev97_planar_avx2(sprec_t *out, const sprec_t *lp, const sprec_t *hp, const int32_t u0,
                                       const int32_t u1) {
  const int32_t N = u1 / 2 - u0 / 2;
  // Mirrored raw-plane accessors (used only for warmup/drain boundary taps).
  auto E = [&](int32_t j) -> float { return lp[PSEo(u0 + 2 * j, u0, u1) >> 1]; };
  auto O = [&](int32_t j) -> float { return hp[PSEo(u0 + 2 * j + 1, u0, u1) >> 1]; };

  const __m256 vA = _mm256_set1_ps(fA), vB = _mm256_set1_ps(fB);
  const __m256 vC = _mm256_set1_ps(fC), vD = _mm256_set1_ps(fD);

  // Warmup: j = -1 scalars, then S1 of blocks 0 and 1, S2/S3 of block 0.
  const float om1  = O(-1);
  const float s1m1 = std::fmaf(-fD, O(-2) + om1, E(-1));  // S1[-1]
  __m256 O_b0      = _mm256_loadu_ps(hp);
  __m256 S1_b0     = _mm256_fnmadd_ps(_mm256_add_ps(avx2_shr1_ps(_mm256_set1_ps(om1), O_b0), O_b0), vD,
                                      _mm256_loadu_ps(lp));
  const float s2m1 = std::fmaf(-fC, s1m1 + _mm256_cvtss_f32(S1_b0), om1);  // S2[-1]
  out[-2]          = s1m1;                                                 // final E[-1]
  out[-1]          = s2m1;                                                 // final O[-1]

  __m256 O_b1 = _mm256_loadu_ps(hp + 8);
  __m256 S1_b1 =
      _mm256_fnmadd_ps(_mm256_add_ps(_mm256_loadu_ps(hp + 7), O_b1), vD, _mm256_loadu_ps(lp + 8));
  __m256 S2_b0 = _mm256_fnmadd_ps(_mm256_add_ps(S1_b0, avx2_shl1_ps(S1_b0, S1_b1)), vC, O_b0);
  __m256 S3_b0 =
      _mm256_fnmadd_ps(_mm256_add_ps(avx2_shr1_ps(_mm256_set1_ps(s2m1), S2_b0), S2_b0), vB, S1_b0);

  // Steady state: iteration n loads input block n (j = 8n..8n+7) with one
  // unaligned load per plane and emits finished block n-2 with one interleaved
  // store.  Raw O[j-1] is an unaligned reload from hp + 8n - 1 (j-1 >= 0 in
  // steady state — cheaper than shuffling on x86); the pipeline-internal
  // shifts use avx2_shl1/shr1.  The bound is 8n+7 <= N-1 because the planes
  // have no PSE-filled margins to read past — the drain covers the rest scalar.
  __m256 O_nm1 = O_b1, S1_nm1 = S1_b1, S2_nm2 = S2_b0, S3_nm2 = S3_b0;
  int32_t n = 2;
  for (; 8 * n + 7 <= N - 1; ++n) {
    __m256 O_n    = _mm256_loadu_ps(hp + 8 * n);
    __m256 S1_n   = _mm256_fnmadd_ps(_mm256_add_ps(_mm256_loadu_ps(hp + 8 * n - 1), O_n), vD,
                                     _mm256_loadu_ps(lp + 8 * n));
    __m256 S2_nm1 = _mm256_fnmadd_ps(_mm256_add_ps(S1_nm1, avx2_shl1_ps(S1_nm1, S1_n)), vC, O_nm1);
    __m256 S3_nm1 = _mm256_fnmadd_ps(_mm256_add_ps(avx2_shr1_ps(S2_nm2, S2_nm1), S2_nm1), vB, S1_nm1);
    __m256 S4_nm2 = _mm256_fnmadd_ps(_mm256_add_ps(S3_nm2, avx2_shl1_ps(S3_nm2, S3_nm1)), vA, S2_nm2);
    avx2_store_interleaved_ps(out + 16 * (n - 2), S3_nm2, S4_nm2);
    O_nm1  = O_n;
    S1_nm1 = S1_n;
    S2_nm2 = S2_nm1;
    S3_nm2 = S3_nm1;
  }

  // Drain: scalar finish for blocks n-2, n-1 (still in registers) and the
  // ragged tail.  Loop exit bounds m = 8n to [N-7, N], so with base = m-16
  // every stage index j-base fits in 32 (max N+1-base = N+1-m+16 <= 24).
  // s1t/s2t/s3t[i] hold S1/S2/S3[base+i]; s1t[0..7] are unused (block n-2's
  // S1 was consumed).
  {
    const int32_t m    = 8 * n;
    const int32_t base = m - 16;
    float s1t[32], s2t[32], s3t[32];
    _mm256_storeu_ps(s1t + 8, S1_nm1);  // S1[m-8..m-1]
    _mm256_storeu_ps(s2t, S2_nm2);      // S2[m-16..m-9]
    _mm256_storeu_ps(s3t, S3_nm2);      // S3[m-16..m-9]
    for (int32_t j = m; j <= N + 1; ++j) s1t[j - base] = std::fmaf(-fD, O(j - 1) + O(j), E(j));
    for (int32_t j = m - 8; j <= N; ++j)
      s2t[j - base] = std::fmaf(-fC, s1t[j - base] + s1t[j - base + 1], O(j));
    for (int32_t j = m - 8; j <= N; ++j)
      s3t[j - base] = std::fmaf(-fB, s2t[j - base - 1] + s2t[j - base], s1t[j - base]);
    // Final row: E[j] = S3[j], O[j] = S4[j]; then O[N] = S2[N], E[N+1] = S1[N+1].
    for (int32_t j = base; j <= N - 1; ++j) {
      out[2 * j]     = s3t[j - base];
      out[2 * j + 1] = std::fmaf(-fA, s3t[j - base] + s3t[j - base + 1], s2t[j - base]);
    }
    out[2 * N]       = s3t[N - base];
    out[2 * N + 1]   = s2t[N - base];
    out[2 * (N + 1)] = s1t[N + 1 - base];
  }
}

// Sub-range 9/7 planar synthesis (JPIP column window).  Lifts straight from the
// LP/HP planes over the widened column window and writes the natural-domain row
// in place — eliminating the full-width interleave_row_planes pass the in-place
// fallback pays per row (it interleaves the whole row even for a tiny window).
// Bit-identical to interleave + idwt_1d_row_inplace_range (fixed_range) for
// every column the caller reads: the §3.1 recurrences are the same
// single-rounded fnmadd/fmaf sequence as the full planar kernel, and boundary
// taps mirror within the plane via PSEo exactly as the in-place kernel's
// PSE-filled margins do (= idwt_1d_filtr_irrev97_planar_avx2 restricted).
//
// Strategy: stage S1/S2/S3 into thread-local scratch over [J0-1, J1+2] in four
// passes.  When the window (widened) is strictly interior to the row, every
// plane read is in-bounds, so the passes are pure 8-lane SIMD with direct
// unaligned plane loads (the common centred-zoom case).  Edge-touching windows
// fall to the PSEo-mirrored scalar path — correct but rarer.
void idwt_1d_filtr_irrev97_planar_sr_avx2(sprec_t *out, const sprec_t *lp, const sprec_t *hp,
                                          const int32_t u0, const int32_t u1, const int32_t col_lo,
                                          const int32_t col_hi) {
  const int32_t N     = u1 / 2 - u0 / 2;
  const int32_t width = u1 - u0;
  // Output columns the caller reads, widened by the 9/7 filter support (4) and
  // clamped to the row — identical to idwt_1d_row_inplace_range's row_lo/row_hi.
  int32_t row_lo = col_lo - u0 - 4;
  int32_t row_hi = col_hi - u0 - 1 + 4;
  if (row_lo < 0) row_lo = 0;
  if (row_hi > width - 1) row_hi = width - 1;
  if (row_lo > row_hi) return;
  const int32_t J0 = row_lo >> 1;  // first subband index produced
  const int32_t J1 = row_hi >> 1;  // last  subband index produced

  // S1[J0-1 .. J1+2], S2[J0-1 .. J1+1], S3[J0 .. J1+1] staged in thread-local
  // scratch indexed by absolute j (offset by `base`).
  const int32_t base = J0 - 1;
  const int32_t M    = (J1 + 2) - base + 1;
  thread_local std::vector<float> s1v, s2v, s3v;
  if (static_cast<int32_t>(s1v.size()) < M) {
    s1v.resize(M);
    s2v.resize(M);
    s3v.resize(M);
  }
  float *const s1 = s1v.data() - base;  // valid for j in [base, J1+2]
  float *const s2 = s2v.data() - base;
  float *const s3 = s3v.data() - base;

  if (J0 >= 2 && J1 <= N - 3) {
    // Interior fast path: hp[J0-2 .. J1+2] and lp[J0-1 .. J1+2] are all in range
    // (PSEo would be the identity here), so read the planes directly.  fnmadd is
    // single-rounded and equal to fmaf(-coeff, sum, x), so vector lanes and the
    // scalar remainders below are bit-identical.
    const __m256 vA = _mm256_set1_ps(fA), vB = _mm256_set1_ps(fB);
    const __m256 vC = _mm256_set1_ps(fC), vD = _mm256_set1_ps(fD);
    int32_t j;
    // Pass 1: S1[j] = E[j] - fD*(O[j-1] + O[j]),  j in [J0-1, J1+2]
    for (j = J0 - 1; j + 8 <= J1 + 3; j += 8)
      _mm256_storeu_ps(
          s1 + j, _mm256_fnmadd_ps(_mm256_add_ps(_mm256_loadu_ps(hp + j - 1), _mm256_loadu_ps(hp + j)), vD,
                                   _mm256_loadu_ps(lp + j)));
    for (; j <= J1 + 2; ++j) s1[j] = std::fmaf(-fD, hp[j - 1] + hp[j], lp[j]);
    // Pass 2: S2[j] = O[j] - fC*(S1[j] + S1[j+1]),  j in [J0-1, J1+1]
    for (j = J0 - 1; j + 8 <= J1 + 2; j += 8)
      _mm256_storeu_ps(
          s2 + j, _mm256_fnmadd_ps(_mm256_add_ps(_mm256_loadu_ps(s1 + j), _mm256_loadu_ps(s1 + j + 1)), vC,
                                   _mm256_loadu_ps(hp + j)));
    for (; j <= J1 + 1; ++j) s2[j] = std::fmaf(-fC, s1[j] + s1[j + 1], hp[j]);
    // Pass 3: S3[j] = S1[j] - fB*(S2[j-1] + S2[j]),  j in [J0, J1+1]
    for (j = J0; j + 8 <= J1 + 2; j += 8)
      _mm256_storeu_ps(
          s3 + j, _mm256_fnmadd_ps(_mm256_add_ps(_mm256_loadu_ps(s2 + j - 1), _mm256_loadu_ps(s2 + j)), vB,
                                   _mm256_loadu_ps(s1 + j)));
    for (; j <= J1 + 1; ++j) s3[j] = std::fmaf(-fB, s2[j - 1] + s2[j], s1[j]);
    // Pass 4 + interleaved store: out[2j]=S3[j], out[2j+1]=S2[j]-fA*(S3[j]+S3[j+1])
    for (j = J0; j + 8 <= J1 + 1; j += 8) {
      __m256 s3j = _mm256_loadu_ps(s3 + j);
      __m256 s4j =
          _mm256_fnmadd_ps(_mm256_add_ps(s3j, _mm256_loadu_ps(s3 + j + 1)), vA, _mm256_loadu_ps(s2 + j));
      avx2_store_interleaved_ps(out + 2 * j, s3j, s4j);
    }
    for (; j <= J1; ++j) {
      out[2 * j]     = s3[j];
      out[2 * j + 1] = std::fmaf(-fA, s3[j] + s3[j + 1], s2[j]);
    }
  } else {
    // Edge path: window touches an image boundary → mirror via PSEo (scalar).
    auto E = [&](int32_t j) -> float { return lp[PSEo(u0 + 2 * j, u0, u1) >> 1]; };
    auto O = [&](int32_t j) -> float { return hp[PSEo(u0 + 2 * j + 1, u0, u1) >> 1]; };
    for (int32_t j = J0 - 1; j <= J1 + 2; ++j) s1[j] = std::fmaf(-fD, O(j - 1) + O(j), E(j));
    for (int32_t j = J0 - 1; j <= J1 + 1; ++j) s2[j] = std::fmaf(-fC, s1[j] + s1[j + 1], O(j));
    for (int32_t j = J0; j <= J1 + 1; ++j) s3[j] = std::fmaf(-fB, s2[j - 1] + s2[j], s1[j]);
    for (int32_t j = J0; j <= J1; ++j) {
      out[2 * j]     = s3[j];
      out[2 * j + 1] = std::fmaf(-fA, s3[j] + s3[j + 1], s2[j]);
    }
  }
}

void idwt_1d_filtr_rev53_planar_i32_avx2(int32_t *out, const int32_t *lp, const int32_t *hp,
                                         const int32_t u0, const int32_t u1) {
  const int32_t N = u1 / 2 - u0 / 2;
  auto E          = [&](int32_t j) -> int32_t { return lp[PSEo(u0 + 2 * j, u0, u1) >> 1]; };
  auto O          = [&](int32_t j) -> int32_t { return hp[PSEo(u0 + 2 * j + 1, u0, u1) >> 1]; };

  //   S1[j] = E[j] - ((O[j-1] + O[j] + 2) >> 2)   for j in [0, N]   (undo LP update)
  //   S2[j] = O[j] + ((S1[j] + S1[j+1]) >> 1)     for j in [0, N)   (undo HP predict)
  // Integer ops are exact, so this matches idwt_1d_filtr_rev53_i32_avx2 bit
  // for bit.  Loop bound j+8 <= N-1 keeps every plane read in range (the
  // s1_next lookahead reads lp[j+8] / hp[j+8]); the scalar tail mirrors.
  const __m256i vtwo = _mm256_set1_epi32(2);
  const int32_t o_m1 = O(-1);  // mirrored O[-1] for the first block only
  int32_t j          = 0;
  for (; j + 8 <= N - 1; j += 8) {
    __m256i Ob = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(hp + j));
    // O[j-1..j+6]: lane 0 is the mirrored O(-1) on the first block; later
    // blocks reload hp + j - 1 (j >= 8, all in range — cheaper than shuffling).
    __m256i Ojm1 = (j == 0) ? avx2_shr1_epi32(_mm256_set1_epi32(o_m1), Ob)
                            : _mm256_loadu_si256(reinterpret_cast<const __m256i *>(hp + j - 1));
    __m256i S1b =
        _mm256_sub_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(lp + j)),
                         _mm256_srai_epi32(_mm256_add_epi32(_mm256_add_epi32(Ojm1, Ob), vtwo), 2));
    // S1[j+8] from raw memory (those positions are not yet written this pass)
    const int32_t s1_next = lp[j + 8] - ((hp[j + 7] + hp[j + 8] + 2) >> 2);
    __m256i S1n           = avx2_shl1_epi32(S1b, _mm256_set1_epi32(s1_next));  // S1[j+1..j+8]
    __m256i S2b           = _mm256_add_epi32(Ob, _mm256_srai_epi32(_mm256_add_epi32(S1b, S1n), 1));
    __m256i lo            = _mm256_unpacklo_epi32(S1b, S2b);
    __m256i hi            = _mm256_unpackhi_epi32(S1b, S2b);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(out + 2 * j), _mm256_permute2x128_si256(lo, hi, 0x20));
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(out + 2 * j + 8),
                        _mm256_permute2x128_si256(lo, hi, 0x31));
  }
  // Scalar tail (at most 9 S1 values: loop exits with N - j <= 8).
  {
    int32_t s1t[12];
    int32_t o_prev = O(j - 1);
    for (int32_t t = j; t <= N; ++t) {
      const int32_t o = O(t);
      s1t[t - j]      = E(t) - ((o_prev + o + 2) >> 2);
      o_prev          = o;
    }
    for (int32_t t = j; t <= N; ++t) out[2 * t] = s1t[t - j];
    for (int32_t t = j; t < N; ++t) out[2 * t + 1] = O(t) + ((s1t[t - j] + s1t[t - j + 1]) >> 1);
  }
}

void idwt_rev_ver_lp_step_i32_avx2(int32_t n, const int32_t *prev, const int32_t *next, int32_t *tgt) {
  const __m256i vtwo = _mm256_set1_epi32(2);
  int32_t i = 0;
  for (; i + 8 <= n; i += 8) {
    __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(prev + i));
    __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(next + i));
    __m256i t = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(tgt  + i));
    t = _mm256_sub_epi32(t,
        _mm256_srai_epi32(_mm256_add_epi32(_mm256_add_epi32(a, b), vtwo), 2));
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(tgt + i), t);
  }
  for (; i < n; ++i) tgt[i] -= (prev[i] + next[i] + 2) >> 2;
}

void idwt_rev_ver_hp_step_i32_avx2(int32_t n, const int32_t *prev, const int32_t *next, int32_t *tgt) {
  int32_t i = 0;
  for (; i + 8 <= n; i += 8) {
    __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(prev + i));
    __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(next + i));
    __m256i t = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(tgt  + i));
    t = _mm256_add_epi32(t, _mm256_srai_epi32(_mm256_add_epi32(a, b), 1));
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(tgt + i), t);
  }
  for (; i < n; ++i) tgt[i] += (prev[i] + next[i]) >> 1;
}
#endif
