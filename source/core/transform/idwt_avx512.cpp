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
#if defined(OPENHTJ2K_TRY_AVX2) && defined(__AVX512F__)
  #include "dwt.hpp"
  #include <cstring>
  #include <cmath>

/********************************************************************************
 * Horizontal transforms
 *******************************************************************************/

// Irreversible 9/7 horizontal IDWT lifting step — AVX-512 variant.
// Processes 8 LP-HP pairs per ZMM register (16 floats/iter).
// Same slli_epi64(32) trick as AVX2: within each 64-bit lane the sum of the two
// neighbouring samples is shifted to the odd (update-target) slot, zeroing the even
// (pass-through) slot, so fnmadd updates only the target slot in one fused instruction.
auto idwt_irrev97_fixed_avx512_hor_step = [](const int32_t init_pos, const int32_t simdlen,
                                              float *const X, const int32_t n0, const int32_t n1,
                                              const float fV) {
  const auto vcoeff = _mm512_set1_ps(fV);
  int32_t n = init_pos, i = 0;
  // 2× unrolled: two independent 8-pair groups per iteration for ILP.
  for (; i + 8 < simdlen; i += 16, n += 32) {
    auto xin0a = _mm512_loadu_ps(X + n + n0);
    auto xin2a = _mm512_loadu_ps(X + n + n1);
    auto xin0b = _mm512_loadu_ps(X + n + 16 + n0);
    auto xin2b = _mm512_loadu_ps(X + n + 16 + n1);
    auto xsuma = _mm512_castsi512_ps(
        _mm512_slli_epi64(_mm512_castps_si512(_mm512_add_ps(xin0a, xin2a)), 32));
    auto xsumb = _mm512_castsi512_ps(
        _mm512_slli_epi64(_mm512_castps_si512(_mm512_add_ps(xin0b, xin2b)), 32));
    xin0a = _mm512_fnmadd_ps(xsuma, vcoeff, xin0a);
    xin0b = _mm512_fnmadd_ps(xsumb, vcoeff, xin0b);
    _mm512_storeu_ps(X + n + n0, xin0a);
    _mm512_storeu_ps(X + n + 16 + n0, xin0b);
  }
  // Cleanup: 8 pairs per ZMM.  SIMD_PADDING provides enough right-margin for safe overshoot.
  for (; i < simdlen; i += 8, n += 16) {
    auto xin0 = _mm512_loadu_ps(X + n + n0);
    auto xin2 = _mm512_loadu_ps(X + n + n1);
    auto xsum = _mm512_castsi512_ps(
        _mm512_slli_epi64(_mm512_castps_si512(_mm512_add_ps(xin0, xin2)), 32));
    xin0 = _mm512_fnmadd_ps(xsum, vcoeff, xin0);
    _mm512_storeu_ps(X + n + n0, xin0);
  }
};

void idwt_1d_filtr_irrev97_fixed_avx512(sprec_t *X, const int32_t left, const int32_t u_i0,
                                        const int32_t u_i1) {
  const auto i0        = static_cast<int32_t>(u_i0);
  const auto i1        = static_cast<int32_t>(u_i1);
  const int32_t start  = i0 / 2;
  const int32_t stop   = i1 / 2;
  const int32_t offset = left - i0 % 2;

  int32_t simdlen = stop + 2 - (start - 1);
  idwt_irrev97_fixed_avx512_hor_step(offset - 2, simdlen, X, -1, 1, fD);
  simdlen = stop + 1 - (start - 1);
  idwt_irrev97_fixed_avx512_hor_step(offset - 2, simdlen, X, 0, 2, fC);
  simdlen = stop + 1 - start;
  idwt_irrev97_fixed_avx512_hor_step(offset, simdlen, X, -1, 1, fB);
  simdlen = stop - start;
  idwt_irrev97_fixed_avx512_hor_step(offset, simdlen, X, 0, 2, fA);
}

// Reversible 5/3 horizontal IDWT — AVX-512 variant.
// Step 1: LP[k] -= floor((HP[k-1] + HP[k] + 2) * 0.25)
// Step 2: HP[k] += floor((LP[k] + LP[k+1]) * 0.5)
void idwt_1d_filtr_rev53_fixed_avx512(sprec_t *X, const int32_t left, const int32_t u_i0,
                                      const int32_t u_i1) {
  const auto i0        = static_cast<int32_t>(u_i0);
  const auto i1        = static_cast<int32_t>(u_i1);
  const int32_t start  = i0 / 2;
  const int32_t stop   = i1 / 2;
  const int32_t offset = left - i0 % 2;

  int32_t simdlen = stop + 1 - start;
  sprec_t *sp     = X + offset;
  const auto xtwo = _mm512_set1_ps(2.0f);
  const auto x025 = _mm512_set1_ps(0.25f);
  int32_t i = 0;
  for (; i + 8 < simdlen; i += 16, sp += 32) {
    auto xin0a = _mm512_loadu_ps(sp - 1);
    auto xin2a = _mm512_loadu_ps(sp + 1);
    auto xin0b = _mm512_loadu_ps(sp + 15);
    auto xin2b = _mm512_loadu_ps(sp + 17);
    auto xsuma = _mm512_castsi512_ps(_mm512_slli_epi64(
        _mm512_castps_si512(_mm512_add_ps(_mm512_add_ps(xin0a, xin2a), xtwo)), 32));
    auto xsumb = _mm512_castsi512_ps(_mm512_slli_epi64(
        _mm512_castps_si512(_mm512_add_ps(_mm512_add_ps(xin0b, xin2b), xtwo)), 32));
    xin0a = _mm512_sub_ps(xin0a, _mm512_floor_ps(_mm512_mul_ps(xsuma, x025)));
    xin0b = _mm512_sub_ps(xin0b, _mm512_floor_ps(_mm512_mul_ps(xsumb, x025)));
    _mm512_storeu_ps(sp - 1, xin0a);
    _mm512_storeu_ps(sp + 15, xin0b);
  }
  for (; i < simdlen; i += 8, sp += 16) {
    auto xin0 = _mm512_loadu_ps(sp - 1);
    auto xin2 = _mm512_loadu_ps(sp + 1);
    auto xsum = _mm512_castsi512_ps(_mm512_slli_epi64(
        _mm512_castps_si512(_mm512_add_ps(_mm512_add_ps(xin0, xin2), xtwo)), 32));
    xin0 = _mm512_sub_ps(xin0, _mm512_floor_ps(_mm512_mul_ps(xsum, x025)));
    _mm512_storeu_ps(sp - 1, xin0);
  }

  simdlen = stop - start;
  sp      = X + offset;
  const auto x05 = _mm512_set1_ps(0.5f);
  i = 0;
  for (; i + 8 < simdlen; i += 16, sp += 32) {
    auto xin0a = _mm512_loadu_ps(sp);
    auto xin2a = _mm512_loadu_ps(sp + 2);
    auto xin0b = _mm512_loadu_ps(sp + 16);
    auto xin2b = _mm512_loadu_ps(sp + 18);
    auto xsuma = _mm512_castsi512_ps(
        _mm512_slli_epi64(_mm512_castps_si512(_mm512_add_ps(xin0a, xin2a)), 32));
    auto xsumb = _mm512_castsi512_ps(
        _mm512_slli_epi64(_mm512_castps_si512(_mm512_add_ps(xin0b, xin2b)), 32));
    xin0a = _mm512_add_ps(xin0a, _mm512_floor_ps(_mm512_mul_ps(xsuma, x05)));
    xin0b = _mm512_add_ps(xin0b, _mm512_floor_ps(_mm512_mul_ps(xsumb, x05)));
    _mm512_storeu_ps(sp, xin0a);
    _mm512_storeu_ps(sp + 16, xin0b);
  }
  for (; i < simdlen; i += 8, sp += 16) {
    auto xin0 = _mm512_loadu_ps(sp);
    auto xin2 = _mm512_loadu_ps(sp + 2);
    auto xsum = _mm512_castsi512_ps(
        _mm512_slli_epi64(_mm512_castps_si512(_mm512_add_ps(xin0, xin2)), 32));
    xin0 = _mm512_add_ps(xin0, _mm512_floor_ps(_mm512_mul_ps(xsum, x05)));
    _mm512_storeu_ps(sp, xin0);
  }
}

// ATK irreversible 5/3 horizontal IDWT — AVX-512 variant.
// Step 1: LP[k] -= 0.25*(HP[k-1]+HP[k])
// Step 2: HP[k] += 0.5*(LP_mod[k]+LP_mod[k+1])
void idwt_1d_filtr_irrev53_fixed_avx512(sprec_t *X, const int32_t left, const int32_t u_i0,
                                        const int32_t u_i1) {
  const int32_t i0     = static_cast<int32_t>(u_i0);
  const int32_t i1     = static_cast<int32_t>(u_i1);
  const int32_t start  = i0 / 2;
  const int32_t stop   = i1 / 2;
  const int32_t offset = left - i0 % 2;

  int32_t simdlen = stop + 1 - start;
  sprec_t *sp     = X + offset;
  const auto x025 = _mm512_set1_ps(0.25f);
  int32_t i = 0;
  for (; i + 8 < simdlen; i += 16, sp += 32) {
    auto xin0a = _mm512_loadu_ps(sp - 1);
    auto xin2a = _mm512_loadu_ps(sp + 1);
    auto xin0b = _mm512_loadu_ps(sp + 15);
    auto xin2b = _mm512_loadu_ps(sp + 17);
    auto xsuma = _mm512_castsi512_ps(
        _mm512_slli_epi64(_mm512_castps_si512(_mm512_add_ps(xin0a, xin2a)), 32));
    auto xsumb = _mm512_castsi512_ps(
        _mm512_slli_epi64(_mm512_castps_si512(_mm512_add_ps(xin0b, xin2b)), 32));
    xin0a = _mm512_fnmadd_ps(xsuma, x025, xin0a);
    xin0b = _mm512_fnmadd_ps(xsumb, x025, xin0b);
    _mm512_storeu_ps(sp - 1, xin0a);
    _mm512_storeu_ps(sp + 15, xin0b);
  }
  for (; i < simdlen; i += 8, sp += 16) {
    auto xin0 = _mm512_loadu_ps(sp - 1);
    auto xin2 = _mm512_loadu_ps(sp + 1);
    auto xsum = _mm512_castsi512_ps(
        _mm512_slli_epi64(_mm512_castps_si512(_mm512_add_ps(xin0, xin2)), 32));
    xin0 = _mm512_fnmadd_ps(xsum, x025, xin0);
    _mm512_storeu_ps(sp - 1, xin0);
  }

  simdlen = stop - start;
  sp      = X + offset;
  const auto x05 = _mm512_set1_ps(0.5f);
  i = 0;
  for (; i + 8 < simdlen; i += 16, sp += 32) {
    auto xin0a = _mm512_loadu_ps(sp);
    auto xin2a = _mm512_loadu_ps(sp + 2);
    auto xin0b = _mm512_loadu_ps(sp + 16);
    auto xin2b = _mm512_loadu_ps(sp + 18);
    auto xsuma = _mm512_castsi512_ps(
        _mm512_slli_epi64(_mm512_castps_si512(_mm512_add_ps(xin0a, xin2a)), 32));
    auto xsumb = _mm512_castsi512_ps(
        _mm512_slli_epi64(_mm512_castps_si512(_mm512_add_ps(xin0b, xin2b)), 32));
    xin0a = _mm512_fmadd_ps(xsuma, x05, xin0a);
    xin0b = _mm512_fmadd_ps(xsumb, x05, xin0b);
    _mm512_storeu_ps(sp, xin0a);
    _mm512_storeu_ps(sp + 16, xin0b);
  }
  for (; i < simdlen; i += 8, sp += 16) {
    auto xin0 = _mm512_loadu_ps(sp);
    auto xin2 = _mm512_loadu_ps(sp + 2);
    auto xsum = _mm512_castsi512_ps(
        _mm512_slli_epi64(_mm512_castps_si512(_mm512_add_ps(xin0, xin2)), 32));
    xin0 = _mm512_fmadd_ps(xsum, x05, xin0);
    _mm512_storeu_ps(sp, xin0);
  }
}

/********************************************************************************
 * Vertical transforms
 *******************************************************************************/

// Inner vertical lifting step: tgt[i] -= coeff * (prev[i] + next[i]).
// Processes 16 floats per ZMM iteration.  Row pointers are at least 32-byte aligned.
auto idwt_irrev97_fixed_avx512_ver_step = [](const int32_t simdlen, float *const Xin0,
                                              float *const Xin1, float *const Xout,
                                              const float coeff) {
  const auto vcoeff = _mm512_set1_ps(coeff);
  for (int32_t n = 0; n < simdlen; n += 16) {
    auto xin0 = _mm512_loadu_ps(Xin0 + n);
    auto xin2 = _mm512_loadu_ps(Xin1 + n);
    auto xout = _mm512_loadu_ps(Xout + n);
    xout      = _mm512_fnmadd_ps(_mm512_add_ps(xin0, xin2), vcoeff, xout);
    _mm512_storeu_ps(Xout + n, xout);
  }
};

// Masked tail: handle remaining 1-15 columns with AVX-512 mask instead of scalar loop.
// tgt[i] -= coeff * (prev[i] + next[i]) for i in [0, remain).
auto irrev97_masked_tail = [](const int32_t remain, float *prev, float *next, float *tgt,
                              const float coeff) {
  if (remain <= 0) return;
  const __mmask16 mask = static_cast<__mmask16>((1U << remain) - 1U);
  auto vcoeff          = _mm512_set1_ps(coeff);
  auto a               = _mm512_maskz_loadu_ps(mask, prev);
  auto b               = _mm512_maskz_loadu_ps(mask, next);
  auto t               = _mm512_maskz_loadu_ps(mask, tgt);
  t                    = _mm512_fnmadd_ps(_mm512_add_ps(a, b), vcoeff, t);
  _mm512_mask_storeu_ps(tgt, mask, t);
};

// Single-row irrev streaming vertical lifting: tgt[i] -= coeff*(prev[i]+next[i]).
void idwt_irrev_ver_step_fixed_avx512(int32_t n, float *prev, float *next, float *tgt, float coeff) {
  const auto vcoeff = _mm512_set1_ps(coeff);
  int32_t i = 0;
  for (; i + 16 <= n; i += 16) {
    auto xin0 = _mm512_loadu_ps(prev + i);
    auto xin1 = _mm512_loadu_ps(next + i);
    auto xout = _mm512_loadu_ps(tgt + i);
    xout      = _mm512_fnmadd_ps(_mm512_add_ps(xin0, xin1), vcoeff, xout);
    _mm512_storeu_ps(tgt + i, xout);
  }
  if (i < n) {
    __mmask16 mask = static_cast<__mmask16>((1U << (n - i)) - 1U);
    auto xin0 = _mm512_maskz_loadu_ps(mask, prev + i);
    auto xin1 = _mm512_maskz_loadu_ps(mask, next + i);
    auto xout = _mm512_maskz_loadu_ps(mask, tgt + i);
    xout      = _mm512_fnmadd_ps(_mm512_add_ps(xin0, xin1), vcoeff, xout);
    _mm512_mask_storeu_ps(tgt + i, mask, xout);
  }
}

// Single-row rev53 LP vertical lifting: tgt[i] -= floor((prev[i]+next[i]+2)*0.25).
void idwt_rev_ver_lp_step_avx512(int32_t n, const float *prev, const float *next, float *tgt) {
  const __m512 k025 = _mm512_set1_ps(0.25f);
  const __m512 k2   = _mm512_set1_ps(2.0f);
  int32_t i = 0;
  for (; i + 16 <= n; i += 16) {
    __m512 a = _mm512_loadu_ps(prev + i);
    __m512 b = _mm512_loadu_ps(next + i);
    __m512 t = _mm512_loadu_ps(tgt + i);
    t = _mm512_sub_ps(t, _mm512_floor_ps(_mm512_mul_ps(_mm512_add_ps(_mm512_add_ps(a, b), k2), k025)));
    _mm512_storeu_ps(tgt + i, t);
  }
  if (i < n) {
    __mmask16 mask = static_cast<__mmask16>((1U << (n - i)) - 1U);
    auto a = _mm512_maskz_loadu_ps(mask, prev + i);
    auto b = _mm512_maskz_loadu_ps(mask, next + i);
    auto t = _mm512_maskz_loadu_ps(mask, tgt + i);
    t = _mm512_sub_ps(t, _mm512_floor_ps(_mm512_mul_ps(_mm512_add_ps(_mm512_add_ps(a, b), k2), k025)));
    _mm512_mask_storeu_ps(tgt + i, mask, t);
  }
}

// Single-row rev53 HP vertical lifting: tgt[i] += floor((prev[i]+next[i])*0.5).
void idwt_rev_ver_hp_step_avx512(int32_t n, const float *prev, const float *next, float *tgt) {
  const __m512 k05 = _mm512_set1_ps(0.5f);
  int32_t i = 0;
  for (; i + 16 <= n; i += 16) {
    __m512 a = _mm512_loadu_ps(prev + i);
    __m512 b = _mm512_loadu_ps(next + i);
    __m512 t = _mm512_loadu_ps(tgt + i);
    t = _mm512_add_ps(t, _mm512_floor_ps(_mm512_mul_ps(_mm512_add_ps(a, b), k05)));
    _mm512_storeu_ps(tgt + i, t);
  }
  if (i < n) {
    __mmask16 mask = static_cast<__mmask16>((1U << (n - i)) - 1U);
    auto a = _mm512_maskz_loadu_ps(mask, prev + i);
    auto b = _mm512_maskz_loadu_ps(mask, next + i);
    auto t = _mm512_maskz_loadu_ps(mask, tgt + i);
    t = _mm512_add_ps(t, _mm512_floor_ps(_mm512_mul_ps(_mm512_add_ps(a, b), k05)));
    _mm512_mask_storeu_ps(tgt + i, mask, t);
  }
}

// Irreversible 9/7 vertical IDWT — AVX-512 variant.
// Column-strip loop (DWT_VERT_STRIP columns per pass) with 16 floats per ZMM iteration.
void idwt_irrev_ver_sr_fixed_avx512(sprec_t *in, const int32_t u0, const int32_t u1, const int32_t v0,
                                    const int32_t v1, const int32_t stride, sprec_t *pse_scratch,
                                    sprec_t **buf_scratch) {
  constexpr int32_t num_pse_i0[2] = {3, 4};
  constexpr int32_t num_pse_i1[2] = {4, 3};
  const int32_t top               = num_pse_i0[v0 % 2];
  const int32_t bottom            = num_pse_i1[v1 % 2];
  if (v0 == v1 - 1) {
    // single row — nothing to do
  } else {
    const int32_t len = round_up(stride, SIMD_PADDING);
    sprec_t **buf     = buf_scratch;
    for (int32_t i = 1; i <= top; ++i) {
      buf[top - i] = pse_scratch + (i - 1) * len;
      memcpy(buf[top - i], &in[PSEo(v0 - i, v0, v1) * stride],
             sizeof(sprec_t) * static_cast<size_t>(stride));
    }
    for (int32_t row = 0; row < v1 - v0; ++row)
      buf[top + row] = &in[row * stride];
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
      const int32_t simdlen_s = (ce - cs) - (ce - cs) % 16;
      const int32_t tail      = ce - cs - simdlen_s;
      for (int32_t n = -2 + offset, i = start - 1; i < stop + 2; i++, n += 2) {
        idwt_irrev97_fixed_avx512_ver_step(simdlen_s, buf[n - 1] + cs, buf[n + 1] + cs, buf[n] + cs,
                                           fD);
        irrev97_masked_tail(tail, buf[n - 1] + cs + simdlen_s, buf[n + 1] + cs + simdlen_s,
                            buf[n] + cs + simdlen_s, fD);
      }
      for (int32_t n = -2 + offset, i = start - 1; i < stop + 1; i++, n += 2) {
        idwt_irrev97_fixed_avx512_ver_step(simdlen_s, buf[n] + cs, buf[n + 2] + cs, buf[n + 1] + cs,
                                           fC);
        irrev97_masked_tail(tail, buf[n] + cs + simdlen_s, buf[n + 2] + cs + simdlen_s,
                            buf[n + 1] + cs + simdlen_s, fC);
      }
      for (int32_t n = 0 + offset, i = start; i < stop + 1; i++, n += 2) {
        idwt_irrev97_fixed_avx512_ver_step(simdlen_s, buf[n - 1] + cs, buf[n + 1] + cs, buf[n] + cs,
                                           fB);
        irrev97_masked_tail(tail, buf[n - 1] + cs + simdlen_s, buf[n + 1] + cs + simdlen_s,
                            buf[n] + cs + simdlen_s, fB);
      }
      for (int32_t n = 0 + offset, i = start; i < stop; i++, n += 2) {
        idwt_irrev97_fixed_avx512_ver_step(simdlen_s, buf[n] + cs, buf[n + 2] + cs, buf[n + 1] + cs,
                                           fA);
        irrev97_masked_tail(tail, buf[n] + cs + simdlen_s, buf[n + 2] + cs + simdlen_s,
                            buf[n + 1] + cs + simdlen_s, fA);
      }
    }
  }
}

// Reversible 5/3 vertical IDWT — AVX-512 variant.
void idwt_rev_ver_sr_fixed_avx512(sprec_t *in, const int32_t u0, const int32_t u1, const int32_t v0,
                                  const int32_t v1, const int32_t stride, sprec_t *pse_scratch,
                                  sprec_t **buf_scratch) {
  constexpr int32_t num_pse_i0[2] = {1, 2};
  constexpr int32_t num_pse_i1[2] = {2, 1};
  const int32_t top               = num_pse_i0[v0 % 2];
  const int32_t bottom            = num_pse_i1[v1 % 2];
  if (v0 == v1 - 1 && (v0 % 2)) {
    for (int32_t col = 0; col < u1 - u0; ++col)
      in[col] = floorf(in[col] * 0.5f);
  } else {
    const int32_t len = round_up(stride, SIMD_PADDING);
    sprec_t **buf     = buf_scratch;
    for (int32_t i = 1; i <= top; ++i) {
      buf[top - i] = pse_scratch + (i - 1) * len;
      memcpy(buf[top - i], &in[PSEo(v0 - i, v0, v1) * stride],
             sizeof(sprec_t) * static_cast<size_t>(stride));
    }
    for (int32_t row = 0; row < v1 - v0; ++row)
      buf[top + row] = &in[row * stride];
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
      const int32_t simdlen_s = (ce - cs) - (ce - cs) % 16;
      const int32_t tail      = ce - cs - simdlen_s;
      const __m512 xtwo = _mm512_set1_ps(2.0f);
      const __m512 x025 = _mm512_set1_ps(0.25f);
      const __mmask16 tmask = tail > 0 ? static_cast<__mmask16>((1U << tail) - 1U) : 0;
      for (int32_t n = 0 + offset, i = start; i < stop + 1; ++i, n += 2) {
        for (int32_t col = 0; col < simdlen_s; col += 16) {
          __m512 x0     = _mm512_loadu_ps(buf[n - 1] + cs + col);
          __m512 x2     = _mm512_loadu_ps(buf[n + 1] + cs + col);
          __m512 x1     = _mm512_loadu_ps(buf[n] + cs + col);
          auto xfloor   = _mm512_floor_ps(_mm512_mul_ps(_mm512_add_ps(_mm512_add_ps(x0, x2), xtwo), x025));
          x1            = _mm512_sub_ps(x1, xfloor);
          _mm512_storeu_ps(buf[n] + cs + col, x1);
        }
        if (tail > 0) {
          auto x0 = _mm512_maskz_loadu_ps(tmask, buf[n - 1] + cs + simdlen_s);
          auto x2 = _mm512_maskz_loadu_ps(tmask, buf[n + 1] + cs + simdlen_s);
          auto x1 = _mm512_maskz_loadu_ps(tmask, buf[n] + cs + simdlen_s);
          x1      = _mm512_sub_ps(x1, _mm512_floor_ps(_mm512_mul_ps(_mm512_add_ps(_mm512_add_ps(x0, x2), xtwo), x025)));
          _mm512_mask_storeu_ps(buf[n] + cs + simdlen_s, tmask, x1);
        }
      }
      const __m512 x05 = _mm512_set1_ps(0.5f);
      for (int32_t n = 0 + offset, i = start; i < stop; ++i, n += 2) {
        for (int32_t col = 0; col < simdlen_s; col += 16) {
          __m512 x0   = _mm512_loadu_ps(buf[n] + cs + col);
          __m512 x2   = _mm512_loadu_ps(buf[n + 2] + cs + col);
          __m512 x1   = _mm512_loadu_ps(buf[n + 1] + cs + col);
          auto xfloor = _mm512_floor_ps(_mm512_mul_ps(_mm512_add_ps(x0, x2), x05));
          x1          = _mm512_add_ps(x1, xfloor);
          _mm512_storeu_ps(buf[n + 1] + cs + col, x1);
        }
        if (tail > 0) {
          auto x0 = _mm512_maskz_loadu_ps(tmask, buf[n] + cs + simdlen_s);
          auto x2 = _mm512_maskz_loadu_ps(tmask, buf[n + 2] + cs + simdlen_s);
          auto x1 = _mm512_maskz_loadu_ps(tmask, buf[n + 1] + cs + simdlen_s);
          x1      = _mm512_add_ps(x1, _mm512_floor_ps(_mm512_mul_ps(_mm512_add_ps(x0, x2), x05)));
          _mm512_mask_storeu_ps(buf[n + 1] + cs + simdlen_s, tmask, x1);
        }
      }
    }
  }
}

// ATK irreversible 5/3 vertical IDWT — AVX-512 variant.
// Step 1: LP[k] -= 0.25*(HP_above + HP_below)
// Step 2: HP[k] += 0.5*(LP_mod_above + LP_mod_below)
void idwt_irrev53_ver_sr_fixed_avx512(sprec_t *in, const int32_t u0, const int32_t u1,
                                      const int32_t v0, const int32_t v1, const int32_t stride,
                                      sprec_t *pse_scratch, sprec_t **buf_scratch) {
  constexpr int32_t num_pse_i0[2] = {1, 2};
  constexpr int32_t num_pse_i1[2] = {2, 1};
  const int32_t top    = num_pse_i0[v0 % 2];
  const int32_t bottom = num_pse_i1[v1 % 2];
  if (v0 == v1 - 1) {
    // single row — nothing to do
  } else {
    const int32_t len = round_up(stride, SIMD_PADDING);
    sprec_t **buf     = buf_scratch;
    for (int32_t i = 1; i <= top; ++i) {
      buf[top - i] = pse_scratch + (i - 1) * len;
      memcpy(buf[top - i], &in[PSEo(v0 - i, v0, v1) * stride],
             sizeof(sprec_t) * static_cast<size_t>(stride));
    }
    for (int32_t row = 0; row < v1 - v0; ++row)
      buf[top + row] = &in[row * stride];
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
      const int32_t simdlen_s = (ce - cs) - (ce - cs) % 16;
      const int32_t tail      = ce - cs - simdlen_s;
      const __m512 x025       = _mm512_set1_ps(0.25f);
      for (int32_t k = 0, n = offset; k < lp_count; ++k, n += 2) {
        for (int32_t col = 0; col < simdlen_s; col += 16) {
          __m512 x0 = _mm512_loadu_ps(buf[n - 1] + cs + col);
          __m512 x2 = _mm512_loadu_ps(buf[n + 1] + cs + col);
          __m512 x1 = _mm512_loadu_ps(buf[n] + cs + col);
          x1        = _mm512_fnmadd_ps(_mm512_add_ps(x0, x2), x025, x1);
          _mm512_storeu_ps(buf[n] + cs + col, x1);
        }
        irrev97_masked_tail(tail, buf[n - 1] + cs + simdlen_s, buf[n + 1] + cs + simdlen_s,
                            buf[n] + cs + simdlen_s, 0.25f);
      }
      const __m512 x05 = _mm512_set1_ps(0.5f);
      for (int32_t k = 0, n = offset; k < hp_count; ++k, n += 2) {
        for (int32_t col = 0; col < simdlen_s; col += 16) {
          __m512 x0 = _mm512_loadu_ps(buf[n] + cs + col);
          __m512 x2 = _mm512_loadu_ps(buf[n + 2] + cs + col);
          __m512 x1 = _mm512_loadu_ps(buf[n + 1] + cs + col);
          x1        = _mm512_fmadd_ps(_mm512_add_ps(x0, x2), x05, x1);
          _mm512_storeu_ps(buf[n + 1] + cs + col, x1);
        }
        irrev97_masked_tail(tail, buf[n] + cs + simdlen_s, buf[n + 2] + cs + simdlen_s,
                            buf[n + 1] + cs + simdlen_s, -0.5f);
      }
    }
  }
}

void idwt_1d_filtr_rev53_i32_avx512(int32_t *X, const int32_t left, const int32_t i0, const int32_t i1) {
  const int32_t start  = i0 / 2;
  const int32_t stop   = i1 / 2;
  const int32_t offset = left - i0 % 2;

  // step 1 (undo forward LP update): LP -= (HP_left + HP_right + 2) >> 2
  int32_t simdlen = stop + 1 - start;
  int32_t i = 0, n = offset;
  const __m512i vtwo = _mm512_set1_epi32(2);
  for (; i + 8 < simdlen; i += 16, n += 32) {
    __m512i xin0a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(X + n - 1));
    __m512i xin2a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(X + n + 1));
    __m512i xin0b = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(X + n + 15));
    __m512i xin2b = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(X + n + 17));
    __m512i xsuma = _mm512_slli_epi64(
        _mm512_add_epi32(_mm512_add_epi32(xin0a, xin2a), vtwo), 32);
    __m512i xsumb = _mm512_slli_epi64(
        _mm512_add_epi32(_mm512_add_epi32(xin0b, xin2b), vtwo), 32);
    xsuma = _mm512_srai_epi32(xsuma, 2);
    xsumb = _mm512_srai_epi32(xsumb, 2);
    xin0a = _mm512_sub_epi32(xin0a, xsuma);
    xin0b = _mm512_sub_epi32(xin0b, xsumb);
    _mm512_storeu_si512(reinterpret_cast<__m512i *>(X + n - 1), xin0a);
    _mm512_storeu_si512(reinterpret_cast<__m512i *>(X + n + 15), xin0b);
  }
  for (; i < simdlen; i += 8, n += 16) {
    __m512i xin0 = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(X + n - 1));
    __m512i xin2 = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(X + n + 1));
    __m512i xsum = _mm512_slli_epi64(
        _mm512_add_epi32(_mm512_add_epi32(xin0, xin2), vtwo), 32);
    xsum = _mm512_srai_epi32(xsum, 2);
    xin0 = _mm512_sub_epi32(xin0, xsum);
    _mm512_storeu_si512(reinterpret_cast<__m512i *>(X + n - 1), xin0);
  }

  // step 2 (undo forward HP predict): HP += (LP_left + LP_right) >> 1
  simdlen = stop - start;
  i = 0; n = offset;
  for (; i + 8 < simdlen; i += 16, n += 32) {
    __m512i xin0a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(X + n));
    __m512i xin2a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(X + n + 2));
    __m512i xin0b = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(X + n + 16));
    __m512i xin2b = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(X + n + 18));
    __m512i xsuma = _mm512_slli_epi64(_mm512_add_epi32(xin0a, xin2a), 32);
    __m512i xsumb = _mm512_slli_epi64(_mm512_add_epi32(xin0b, xin2b), 32);
    xsuma = _mm512_srai_epi32(xsuma, 1);
    xsumb = _mm512_srai_epi32(xsumb, 1);
    xin0a = _mm512_add_epi32(xin0a, xsuma);
    xin0b = _mm512_add_epi32(xin0b, xsumb);
    _mm512_storeu_si512(reinterpret_cast<__m512i *>(X + n), xin0a);
    _mm512_storeu_si512(reinterpret_cast<__m512i *>(X + n + 16), xin0b);
  }
  for (; i < simdlen; i += 8, n += 16) {
    __m512i xin0 = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(X + n));
    __m512i xin2 = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(X + n + 2));
    __m512i xsum = _mm512_slli_epi64(_mm512_add_epi32(xin0, xin2), 32);
    xsum = _mm512_srai_epi32(xsum, 1);
    xin0 = _mm512_add_epi32(xin0, xsum);
    _mm512_storeu_si512(reinterpret_cast<__m512i *>(X + n), xin0);
  }
}

void idwt_rev_ver_lp_step_i32_avx512(int32_t n, const int32_t *prev, const int32_t *next, int32_t *tgt) {
  const __m512i vtwo = _mm512_set1_epi32(2);
  int32_t i = 0;
  for (; i + 16 <= n; i += 16) {
    __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(prev + i));
    __m512i b = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(next + i));
    __m512i t = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(tgt  + i));
    t = _mm512_sub_epi32(t,
        _mm512_srai_epi32(_mm512_add_epi32(_mm512_add_epi32(a, b), vtwo), 2));
    _mm512_storeu_si512(reinterpret_cast<__m512i *>(tgt + i), t);
  }
  if (i < n) {
    __mmask16 mask = static_cast<__mmask16>((1U << (n - i)) - 1U);
    __m512i a = _mm512_maskz_loadu_epi32(mask, prev + i);
    __m512i b = _mm512_maskz_loadu_epi32(mask, next + i);
    __m512i t = _mm512_maskz_loadu_epi32(mask, tgt  + i);
    t = _mm512_sub_epi32(t,
        _mm512_srai_epi32(_mm512_add_epi32(_mm512_add_epi32(a, b), vtwo), 2));
    _mm512_mask_storeu_epi32(tgt + i, mask, t);
  }
}

void idwt_rev_ver_hp_step_i32_avx512(int32_t n, const int32_t *prev, const int32_t *next, int32_t *tgt) {
  int32_t i = 0;
  for (; i + 16 <= n; i += 16) {
    __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(prev + i));
    __m512i b = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(next + i));
    __m512i t = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(tgt  + i));
    t = _mm512_add_epi32(t, _mm512_srai_epi32(_mm512_add_epi32(a, b), 1));
    _mm512_storeu_si512(reinterpret_cast<__m512i *>(tgt + i), t);
  }
  if (i < n) {
    __mmask16 mask = static_cast<__mmask16>((1U << (n - i)) - 1U);
    __m512i a = _mm512_maskz_loadu_epi32(mask, prev + i);
    __m512i b = _mm512_maskz_loadu_epi32(mask, next + i);
    __m512i t = _mm512_maskz_loadu_epi32(mask, tgt  + i);
    t = _mm512_add_epi32(t, _mm512_srai_epi32(_mm512_add_epi32(a, b), 1));
    _mm512_mask_storeu_epi32(tgt + i, mask, t);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Planar-input horizontal synthesis — 16-lane variant of the AVX2 planar
// kernels (idwt_avx2.cpp; NEON original in idwt_neon.cpp).  The LP and HP
// subband rows are read directly (E[j] = lp[j], O[j] = hp[j]) and the
// synthesised natural-domain row is written to out[].  AVX-512 advantages
// over the 8-lane version: _mm512_alignr_epi32 concatenates across the full
// register, so the pipeline-internal cross-element shifts are one instruction
// instead of permute2f128+alignr, and vpermt2ps does the interleave-on-store
// in one shuffle per half.  Every lifting stage is the same single-rounded
// _mm512_fnmadd_ps / std::fmaf sequence per element as the AVX2 planar kernel
// and the in-place kernels — output is bit-identical, which is what lets the
// dispatcher pick per-row between this kernel (N >= 32) and the AVX2 one
// (16 <= N < 32).
//
// Contract (enforced by idwt_1d_row_from_planar): u0 even, N = u1/2 - u0/2
// >= 32 (the 16-lane warmup loads blocks 0/1, j = 0..31, unconditionally),
// lp has ceil(u1/2) - u0/2 valid samples, hp has N, out has >= 2 writable
// floats before index 0 and >= 8 after index u1-u0-1.
// ─────────────────────────────────────────────────────────────────────────────

// Cross-element shifts with carry-in — full-register valignd.
// shl1: [a1..a15, b0] — shift left one element, carry-in from next vector b.
static inline __m512 avx512_shl1_ps(__m512 a, __m512 b) {
  return _mm512_castsi512_ps(_mm512_alignr_epi32(_mm512_castps_si512(b), _mm512_castps_si512(a), 1));
}
// shr1: [p15, a0..a14] — shift right one element, carry-in from previous vector p.
static inline __m512 avx512_shr1_ps(__m512 p, __m512 a) {
  return _mm512_castsi512_ps(_mm512_alignr_epi32(_mm512_castps_si512(a), _mm512_castps_si512(p), 15));
}

// Interleaved store of one finished block: out[2j+0,2,..] = e, out[2j+1,3,..] = o.
static inline void avx512_store_interleaved_ps(float *dst, __m512 e, __m512 o) {
  const __m512i idx_lo = _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
  const __m512i idx_hi = _mm512_setr_epi32(8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
  _mm512_storeu_ps(dst, _mm512_permutex2var_ps(e, idx_lo, o));
  _mm512_storeu_ps(dst + 16, _mm512_permutex2var_ps(e, idx_hi, o));
}

void idwt_1d_filtr_irrev97_planar_avx512(sprec_t *out, const sprec_t *lp, const sprec_t *hp,
                                         const int32_t u0, const int32_t u1) {
  const int32_t N = u1 / 2 - u0 / 2;
  // Mirrored raw-plane accessors (used only for warmup/drain boundary taps).
  auto E = [&](int32_t j) -> float { return lp[PSEo(u0 + 2 * j, u0, u1) >> 1]; };
  auto O = [&](int32_t j) -> float { return hp[PSEo(u0 + 2 * j + 1, u0, u1) >> 1]; };

  const __m512 vA = _mm512_set1_ps(fA), vB = _mm512_set1_ps(fB);
  const __m512 vC = _mm512_set1_ps(fC), vD = _mm512_set1_ps(fD);

  // Warmup: j = -1 scalars, then S1 of blocks 0 and 1, S2/S3 of block 0.
  const float om1  = O(-1);
  const float s1m1 = std::fmaf(-fD, O(-2) + om1, E(-1));  // S1[-1]
  __m512 O_b0      = _mm512_loadu_ps(hp);
  __m512 S1_b0     = _mm512_fnmadd_ps(_mm512_add_ps(avx512_shr1_ps(_mm512_set1_ps(om1), O_b0), O_b0), vD,
                                      _mm512_loadu_ps(lp));
  const float s2m1 = std::fmaf(-fC, s1m1 + _mm512_cvtss_f32(S1_b0), om1);  // S2[-1]
  out[-2]          = s1m1;                                                 // final E[-1]
  out[-1]          = s2m1;                                                 // final O[-1]

  __m512 O_b1 = _mm512_loadu_ps(hp + 16);
  __m512 S1_b1 =
      _mm512_fnmadd_ps(_mm512_add_ps(_mm512_loadu_ps(hp + 15), O_b1), vD, _mm512_loadu_ps(lp + 16));
  __m512 S2_b0 = _mm512_fnmadd_ps(_mm512_add_ps(S1_b0, avx512_shl1_ps(S1_b0, S1_b1)), vC, O_b0);
  __m512 S3_b0 =
      _mm512_fnmadd_ps(_mm512_add_ps(avx512_shr1_ps(_mm512_set1_ps(s2m1), S2_b0), S2_b0), vB, S1_b0);

  // Steady state: iteration n loads input block n (j = 16n..16n+15) with one
  // unaligned load per plane and emits finished block n-2 with one interleaved
  // store.  Raw O[j-1] is an unaligned reload from hp + 16n - 1 (j-1 >= 0 in
  // steady state); the pipeline-internal shifts are single valignd ops.  The
  // bound is 16n+15 <= N-1 because the planes have no PSE-filled margins to
  // read past — the drain covers the rest scalar.
  __m512 O_nm1 = O_b1, S1_nm1 = S1_b1, S2_nm2 = S2_b0, S3_nm2 = S3_b0;
  int32_t n = 2;
  for (; 16 * n + 15 <= N - 1; ++n) {
    __m512 O_n    = _mm512_loadu_ps(hp + 16 * n);
    __m512 S1_n   = _mm512_fnmadd_ps(_mm512_add_ps(_mm512_loadu_ps(hp + 16 * n - 1), O_n), vD,
                                     _mm512_loadu_ps(lp + 16 * n));
    __m512 S2_nm1 = _mm512_fnmadd_ps(_mm512_add_ps(S1_nm1, avx512_shl1_ps(S1_nm1, S1_n)), vC, O_nm1);
    __m512 S3_nm1 = _mm512_fnmadd_ps(_mm512_add_ps(avx512_shr1_ps(S2_nm2, S2_nm1), S2_nm1), vB, S1_nm1);
    __m512 S4_nm2 = _mm512_fnmadd_ps(_mm512_add_ps(S3_nm2, avx512_shl1_ps(S3_nm2, S3_nm1)), vA, S2_nm2);
    avx512_store_interleaved_ps(out + 32 * (n - 2), S3_nm2, S4_nm2);
    O_nm1  = O_n;
    S1_nm1 = S1_n;
    S2_nm2 = S2_nm1;
    S3_nm2 = S3_nm1;
  }

  // Drain: scalar finish for blocks n-2, n-1 (still in registers) and the
  // ragged tail.  Loop exit bounds m = 16n to [N-15, N], so with base = m-32
  // every stage index j-base fits in 64 (max N+1-base = N+1-m+32 <= 48).
  // s1t/s2t/s3t[i] hold S1/S2/S3[base+i]; s1t[0..15] are unused (block n-2's
  // S1 was consumed).
  {
    const int32_t m    = 16 * n;
    const int32_t base = m - 32;
    float s1t[64], s2t[64], s3t[64];
    _mm512_storeu_ps(s1t + 16, S1_nm1);  // S1[m-16..m-1]
    _mm512_storeu_ps(s2t, S2_nm2);       // S2[m-32..m-17]
    _mm512_storeu_ps(s3t, S3_nm2);       // S3[m-32..m-17]
    for (int32_t j = m; j <= N + 1; ++j) s1t[j - base] = std::fmaf(-fD, O(j - 1) + O(j), E(j));
    for (int32_t j = m - 16; j <= N; ++j)
      s2t[j - base] = std::fmaf(-fC, s1t[j - base] + s1t[j - base + 1], O(j));
    for (int32_t j = m - 16; j <= N; ++j)
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

void idwt_1d_filtr_rev53_planar_i32_avx512(int32_t *out, const int32_t *lp, const int32_t *hp,
                                           const int32_t u0, const int32_t u1) {
  const int32_t N = u1 / 2 - u0 / 2;
  auto E          = [&](int32_t j) -> int32_t { return lp[PSEo(u0 + 2 * j, u0, u1) >> 1]; };
  auto O          = [&](int32_t j) -> int32_t { return hp[PSEo(u0 + 2 * j + 1, u0, u1) >> 1]; };

  //   S1[j] = E[j] - ((O[j-1] + O[j] + 2) >> 2)   for j in [0, N]   (undo LP update)
  //   S2[j] = O[j] + ((S1[j] + S1[j+1]) >> 1)     for j in [0, N)   (undo HP predict)
  // Integer ops are exact, so this matches the AVX2 planar kernel and the
  // in-place kernels bit for bit.  Loop bound j+16 <= N-1 keeps every plane
  // read in range (the s1_next lookahead reads lp[j+16] / hp[j+16]); the
  // scalar tail mirrors.
  const __m512i vtwo   = _mm512_set1_epi32(2);
  const __m512i idx_lo = _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
  const __m512i idx_hi = _mm512_setr_epi32(8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
  const int32_t o_m1   = O(-1);  // mirrored O[-1] for the first block only
  int32_t j            = 0;
  for (; j + 16 <= N - 1; j += 16) {
    __m512i Ob = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(hp + j));
    // O[j-1..j+14]: lane 0 is the mirrored O(-1) on the first block; later
    // blocks reload hp + j - 1 (j >= 16, all in range).
    __m512i Ojm1 = (j == 0) ? _mm512_alignr_epi32(Ob, _mm512_set1_epi32(o_m1), 15)
                            : _mm512_loadu_si512(reinterpret_cast<const __m512i *>(hp + j - 1));
    __m512i S1b =
        _mm512_sub_epi32(_mm512_loadu_si512(reinterpret_cast<const __m512i *>(lp + j)),
                         _mm512_srai_epi32(_mm512_add_epi32(_mm512_add_epi32(Ojm1, Ob), vtwo), 2));
    // S1[j+16] from raw memory (those positions are not yet written this pass)
    const int32_t s1_next = lp[j + 16] - ((hp[j + 15] + hp[j + 16] + 2) >> 2);
    __m512i S1n           = _mm512_alignr_epi32(_mm512_set1_epi32(s1_next), S1b, 1);  // S1[j+1..j+16]
    __m512i S2b           = _mm512_add_epi32(Ob, _mm512_srai_epi32(_mm512_add_epi32(S1b, S1n), 1));
    _mm512_storeu_si512(reinterpret_cast<__m512i *>(out + 2 * j),
                        _mm512_permutex2var_epi32(S1b, idx_lo, S2b));
    _mm512_storeu_si512(reinterpret_cast<__m512i *>(out + 2 * j + 16),
                        _mm512_permutex2var_epi32(S1b, idx_hi, S2b));
  }
  // Scalar tail (at most 17 S1 values: loop exits with N - j <= 16).
  {
    int32_t s1t[20];
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

#endif  // OPENHTJ2K_TRY_AVX2 && __AVX512F__
