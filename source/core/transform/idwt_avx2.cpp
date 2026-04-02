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
    const int32_t lp_count = ceil_int(v1, 2) - ceil_int(v0, 2);
    const int32_t hp_count = v1 / 2 - v0 / 2;
    const int32_t offset   = top - v0 % 2;
    const int32_t lp_n0    = top + v0 % 2;
    const int32_t width    = u1 - u0;
    for (int32_t cs = 0; cs < width; cs += DWT_VERT_STRIP) {
      const int32_t ce        = (cs + DWT_VERT_STRIP < width) ? cs + DWT_VERT_STRIP : width;
      const int32_t simdlen_s = (ce - cs) - (ce - cs) % 8;
      const __m256 x025       = _mm256_set1_ps(0.25f);
      for (int32_t k = 0, n = lp_n0; k < lp_count; ++k, n += 2) {
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
#endif
