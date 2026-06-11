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

// Irreversible 9/7 horizontal FDWT lifting step — AVX-512 variant.
// Processes 8 LP-HP pairs per ZMM register (16 floats/iter).
// Same slli_epi64(32) trick as AVX2: within each 64-bit lane the sum of the two
// neighbouring samples is shifted to the odd (update-target) slot, zeroing the even
// (pass-through) slot, so fmadd updates only the target slot in one fused instruction.
auto fdwt_irrev97_fixed_avx512_hor_step = [](const int32_t init_pos, const int32_t simdlen,
                                              float *const X, const int32_t n0, const int32_t n1,
                                              float coeff) {
  auto vcoeff = _mm512_set1_ps(coeff);
  int32_t n = init_pos, i = 0;
  // 2x unrolled: two independent 8-pair groups per iteration for ILP.
  for (; i + 8 < simdlen; i += 16, n += 32) {
    auto xin0a = _mm512_loadu_ps(X + n + n0);
    auto xin2a = _mm512_loadu_ps(X + n + n1);
    auto xin0b = _mm512_loadu_ps(X + n + 16 + n0);
    auto xin2b = _mm512_loadu_ps(X + n + 16 + n1);
    auto xsuma = _mm512_castsi512_ps(
        _mm512_slli_epi64(_mm512_castps_si512(_mm512_add_ps(xin0a, xin2a)), 32));
    auto xsumb = _mm512_castsi512_ps(
        _mm512_slli_epi64(_mm512_castps_si512(_mm512_add_ps(xin0b, xin2b)), 32));
    xin0a = _mm512_fmadd_ps(xsuma, vcoeff, xin0a);
    xin0b = _mm512_fmadd_ps(xsumb, vcoeff, xin0b);
    _mm512_storeu_ps(X + n + n0, xin0a);
    _mm512_storeu_ps(X + n + 16 + n0, xin0b);
  }
  // Cleanup: 8 pairs per ZMM.  SIMD_PADDING provides enough right-margin for safe overshoot.
  for (; i < simdlen; i += 8, n += 16) {
    auto xin0 = _mm512_loadu_ps(X + n + n0);
    auto xin2 = _mm512_loadu_ps(X + n + n1);
    auto xsum = _mm512_castsi512_ps(
        _mm512_slli_epi64(_mm512_castps_si512(_mm512_add_ps(xin0, xin2)), 32));
    xin0 = _mm512_fmadd_ps(xsum, vcoeff, xin0);
    _mm512_storeu_ps(X + n + n0, xin0);
  }
};

void fdwt_1d_filtr_irrev97_fixed_avx512(sprec_t *X, const int32_t left, const int32_t u_i0,
                                         const int32_t u_i1) {
  const auto i0       = static_cast<int32_t>(u_i0);
  const auto i1       = static_cast<int32_t>(u_i1);
  const int32_t start = ceil_int(i0, 2);
  const int32_t stop  = ceil_int(i1, 2);

  const int32_t offset = left + i0 % 2;

  // step 1
  int32_t simdlen = stop + 1 - (start - 2);
  fdwt_irrev97_fixed_avx512_hor_step(offset - 4, simdlen, X, 0, 2, fA);

  // step 2
  simdlen = stop + 1 - (start - 1);
  fdwt_irrev97_fixed_avx512_hor_step(offset - 2, simdlen, X, -1, 1, fB);

  // step 3
  simdlen = stop - (start - 1);
  fdwt_irrev97_fixed_avx512_hor_step(offset - 2, simdlen, X, 0, 2, fC);

  // step 4
  simdlen = stop - start;
  fdwt_irrev97_fixed_avx512_hor_step(offset, simdlen, X, -1, 1, fD);
}

// Reversible 5/3 horizontal FDWT — AVX-512 variant.
// Step 1: H[k] -= floor((L[k] + L[k+1]) * 0.5)
// Step 2: L[k] += floor((H[k-1] + H[k] + 2) * 0.25)
void fdwt_1d_filtr_rev53_fixed_avx512(sprec_t *X, const int32_t left, const int32_t u_i0,
                                       const int32_t u_i1) {
  const auto i0        = static_cast<int32_t>(u_i0);
  const auto i1        = static_cast<int32_t>(u_i1);
  const int32_t start  = ceil_int(i0, 2);
  const int32_t stop   = ceil_int(i1, 2);
  const int32_t offset = left + i0 % 2;

  auto x05 = _mm512_set1_ps(0.5f);
  // step 1: H[k] -= floor((L[k] + L[k+1]) * 0.5)
  int32_t simdlen = stop - (start - 1);
  int32_t i = 0, n = -2 + offset;
  for (; i + 8 < simdlen; i += 16, n += 32) {
    auto xin0a = _mm512_loadu_ps(X + n);
    auto xin2a = _mm512_loadu_ps(X + n + 2);
    auto xin0b = _mm512_loadu_ps(X + n + 16);
    auto xin2b = _mm512_loadu_ps(X + n + 18);
    auto xsuma = _mm512_castsi512_ps(
        _mm512_slli_epi64(_mm512_castps_si512(_mm512_add_ps(xin0a, xin2a)), 32));
    auto xsumb = _mm512_castsi512_ps(
        _mm512_slli_epi64(_mm512_castps_si512(_mm512_add_ps(xin0b, xin2b)), 32));
    xin0a = _mm512_sub_ps(xin0a, _mm512_floor_ps(_mm512_mul_ps(xsuma, x05)));
    xin0b = _mm512_sub_ps(xin0b, _mm512_floor_ps(_mm512_mul_ps(xsumb, x05)));
    _mm512_storeu_ps(X + n, xin0a);
    _mm512_storeu_ps(X + n + 16, xin0b);
  }
  for (; i < simdlen; i += 8, n += 16) {
    auto xin0 = _mm512_loadu_ps(X + n);
    auto xin2 = _mm512_loadu_ps(X + n + 2);
    auto xsum = _mm512_castsi512_ps(
        _mm512_slli_epi64(_mm512_castps_si512(_mm512_add_ps(xin0, xin2)), 32));
    xin0 = _mm512_sub_ps(xin0, _mm512_floor_ps(_mm512_mul_ps(xsum, x05)));
    _mm512_storeu_ps(X + n, xin0);
  }

  // step 2: L[k] += floor((H[k-1] + H[k] + 2) * 0.25)
  simdlen   = stop - start;
  i = 0; n = 0 + offset;
  auto xtwo = _mm512_set1_ps(2.0f);
  auto x025 = _mm512_set1_ps(0.25f);
  for (; i + 8 < simdlen; i += 16, n += 32) {
    auto xin0a = _mm512_loadu_ps(X + n - 1);
    auto xin2a = _mm512_loadu_ps(X + n + 1);
    auto xin0b = _mm512_loadu_ps(X + n + 15);
    auto xin2b = _mm512_loadu_ps(X + n + 17);
    auto xsuma = _mm512_castsi512_ps(_mm512_slli_epi64(
        _mm512_castps_si512(_mm512_add_ps(_mm512_add_ps(xin0a, xin2a), xtwo)), 32));
    auto xsumb = _mm512_castsi512_ps(_mm512_slli_epi64(
        _mm512_castps_si512(_mm512_add_ps(_mm512_add_ps(xin0b, xin2b), xtwo)), 32));
    xin0a = _mm512_add_ps(xin0a, _mm512_floor_ps(_mm512_mul_ps(xsuma, x025)));
    xin0b = _mm512_add_ps(xin0b, _mm512_floor_ps(_mm512_mul_ps(xsumb, x025)));
    _mm512_storeu_ps(X + n - 1, xin0a);
    _mm512_storeu_ps(X + n + 15, xin0b);
  }
  for (; i < simdlen; i += 8, n += 16) {
    auto xin0 = _mm512_loadu_ps(X + n - 1);
    auto xin2 = _mm512_loadu_ps(X + n + 1);
    auto xsum = _mm512_castsi512_ps(_mm512_slli_epi64(
        _mm512_castps_si512(_mm512_add_ps(_mm512_add_ps(xin0, xin2), xtwo)), 32));
    xin0 = _mm512_add_ps(xin0, _mm512_floor_ps(_mm512_mul_ps(xsum, x025)));
    _mm512_storeu_ps(X + n - 1, xin0);
  }
}

/********************************************************************************
 * Vertical transforms
 *******************************************************************************/

// Inner vertical lifting step for irrev 9/7: tgt[i] += coeff * (prev[i] + next[i]).
// Processes 16 floats per ZMM iteration.  Row pointers are at least 32-byte aligned.
auto fdwt_irrev97_fixed_avx512_ver_step = [](const int32_t simdlen, float *const Xin0,
                                              float *const Xin1, float *const Xout,
                                              const float coeff) {
  const auto vcoeff = _mm512_set1_ps(coeff);
  for (int32_t n = 0; n < simdlen; n += 16) {
    auto xin0 = _mm512_load_ps(Xin0 + n);
    auto xin2 = _mm512_load_ps(Xin1 + n);
    auto xsum = _mm512_add_ps(xin0, xin2);
    auto xout = _mm512_load_ps(Xout + n);
    xout = _mm512_fmadd_ps(xsum, vcoeff, xout);
    _mm512_store_ps(Xout + n, xout);
  }
};

// Masked tail: handle remaining 1-15 columns with AVX-512 mask instead of scalar loop.
// tgt[i] += coeff * (prev[i] + next[i]) for i in [0, remain).
auto fdwt_irrev97_masked_tail = [](const int32_t remain, float *prev, float *next, float *tgt,
                                    const float coeff) {
  if (remain <= 0) return;
  const __mmask16 mask = static_cast<__mmask16>((1U << remain) - 1U);
  auto vcoeff          = _mm512_set1_ps(coeff);
  auto a               = _mm512_maskz_loadu_ps(mask, prev);
  auto b               = _mm512_maskz_loadu_ps(mask, next);
  auto t               = _mm512_maskz_loadu_ps(mask, tgt);
  t                    = _mm512_fmadd_ps(_mm512_add_ps(a, b), vcoeff, t);
  _mm512_mask_storeu_ps(tgt, mask, t);
};

// Irreversible 9/7 vertical FDWT — AVX-512 variant.
// Column-strip loop (DWT_VERT_STRIP columns per pass) with 16 floats per ZMM iteration.
void fdwt_irrev_ver_sr_fixed_avx512(sprec_t *in, const int32_t u0, const int32_t u1, const int32_t v0,
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
    const int32_t start  = ceil_int(v0, 2);
    const int32_t stop   = ceil_int(v1, 2);
    const int32_t offset = top + v0 % 2;

    const int32_t width = u1 - u0;
    for (int32_t cs = 0; cs < width; cs += DWT_VERT_STRIP) {
      const int32_t ce        = (cs + DWT_VERT_STRIP < width) ? cs + DWT_VERT_STRIP : width;
      const int32_t simdlen_s = (ce - cs) - (ce - cs) % 16;
      const int32_t tail      = ce - cs - simdlen_s;
      for (int32_t n = -4 + offset, i = start - 2; i < stop + 1; i++, n += 2) {
        fdwt_irrev97_fixed_avx512_ver_step(simdlen_s, buf[n] + cs, buf[n + 2] + cs,
                                           buf[n + 1] + cs, fA);
        fdwt_irrev97_masked_tail(tail, buf[n] + cs + simdlen_s, buf[n + 2] + cs + simdlen_s,
                                 buf[n + 1] + cs + simdlen_s, fA);
      }
      for (int32_t n = -2 + offset, i = start - 1; i < stop + 1; i++, n += 2) {
        fdwt_irrev97_fixed_avx512_ver_step(simdlen_s, buf[n - 1] + cs, buf[n + 1] + cs,
                                           buf[n] + cs, fB);
        fdwt_irrev97_masked_tail(tail, buf[n - 1] + cs + simdlen_s, buf[n + 1] + cs + simdlen_s,
                                 buf[n] + cs + simdlen_s, fB);
      }
      for (int32_t n = -2 + offset, i = start - 1; i < stop; i++, n += 2) {
        fdwt_irrev97_fixed_avx512_ver_step(simdlen_s, buf[n] + cs, buf[n + 2] + cs,
                                           buf[n + 1] + cs, fC);
        fdwt_irrev97_masked_tail(tail, buf[n] + cs + simdlen_s, buf[n + 2] + cs + simdlen_s,
                                 buf[n + 1] + cs + simdlen_s, fC);
      }
      for (int32_t n = 0 + offset, i = start; i < stop; i++, n += 2) {
        fdwt_irrev97_fixed_avx512_ver_step(simdlen_s, buf[n - 1] + cs, buf[n + 1] + cs,
                                           buf[n] + cs, fD);
        fdwt_irrev97_masked_tail(tail, buf[n - 1] + cs + simdlen_s, buf[n + 1] + cs + simdlen_s,
                                 buf[n] + cs + simdlen_s, fD);
      }
    }
  }
}

// Reversible 5/3 vertical FDWT — AVX-512 variant.
void fdwt_rev_ver_sr_fixed_avx512(sprec_t *in, const int32_t u0, const int32_t u1, const int32_t v0,
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
    const int32_t start  = ceil_int(v0, 2);
    const int32_t stop   = ceil_int(v1, 2);
    const int32_t offset = top + v0 % 2;

    const int32_t width = u1 - u0;
    for (int32_t cs = 0; cs < width; cs += DWT_VERT_STRIP) {
      const int32_t ce        = (cs + DWT_VERT_STRIP < width) ? cs + DWT_VERT_STRIP : width;
      const int32_t simdlen_s = (ce - cs) - (ce - cs) % 16;
      const int32_t tail      = ce - cs - simdlen_s;
      const __mmask16 tmask   = tail > 0 ? static_cast<__mmask16>((1U << tail) - 1U) : 0;
      const __m512 x05 = _mm512_set1_ps(0.5f);
      // step 1: H[k] -= floor((L[k] + L[k+1]) * 0.5)
      for (int32_t n = -2 + offset, i = start - 1; i < stop; ++i, n += 2) {
        for (int32_t col = 0; col < simdlen_s; col += 16) {
          __m512 x0   = _mm512_load_ps(buf[n] + cs + col);
          __m512 x2   = _mm512_load_ps(buf[n + 2] + cs + col);
          __m512 x1   = _mm512_load_ps(buf[n + 1] + cs + col);
          auto xfloor = _mm512_floor_ps(_mm512_mul_ps(_mm512_add_ps(x0, x2), x05));
          x1          = _mm512_sub_ps(x1, xfloor);
          _mm512_store_ps(buf[n + 1] + cs + col, x1);
        }
        if (tail > 0) {
          auto x0 = _mm512_maskz_loadu_ps(tmask, buf[n] + cs + simdlen_s);
          auto x2 = _mm512_maskz_loadu_ps(tmask, buf[n + 2] + cs + simdlen_s);
          auto x1 = _mm512_maskz_loadu_ps(tmask, buf[n + 1] + cs + simdlen_s);
          x1      = _mm512_sub_ps(x1, _mm512_floor_ps(_mm512_mul_ps(_mm512_add_ps(x0, x2), x05)));
          _mm512_mask_storeu_ps(buf[n + 1] + cs + simdlen_s, tmask, x1);
        }
      }
      // step 2: L[k] += floor((H[k-1] + H[k] + 2) * 0.25)
      const __m512 xtwo = _mm512_set1_ps(2.0f);
      const __m512 x025 = _mm512_set1_ps(0.25f);
      for (int32_t n = 0 + offset, i = start; i < stop; ++i, n += 2) {
        for (int32_t col = 0; col < simdlen_s; col += 16) {
          __m512 x0   = _mm512_load_ps(buf[n - 1] + cs + col);
          __m512 x2   = _mm512_load_ps(buf[n + 1] + cs + col);
          __m512 x1   = _mm512_load_ps(buf[n] + cs + col);
          auto xfloor = _mm512_floor_ps(_mm512_mul_ps(_mm512_add_ps(_mm512_add_ps(x0, x2), xtwo), x025));
          x1          = _mm512_add_ps(x1, xfloor);
          _mm512_store_ps(buf[n] + cs + col, x1);
        }
        if (tail > 0) {
          auto x0 = _mm512_maskz_loadu_ps(tmask, buf[n - 1] + cs + simdlen_s);
          auto x2 = _mm512_maskz_loadu_ps(tmask, buf[n + 1] + cs + simdlen_s);
          auto x1 = _mm512_maskz_loadu_ps(tmask, buf[n] + cs + simdlen_s);
          x1      = _mm512_add_ps(x1, _mm512_floor_ps(_mm512_mul_ps(_mm512_add_ps(_mm512_add_ps(x0, x2), xtwo), x025)));
          _mm512_mask_storeu_ps(buf[n] + cs + simdlen_s, tmask, x1);
        }
      }
    }
  }
}

// Single-row rev53 FDWT HP vertical lifting: tgt[i] -= floor((prev[i]+next[i])*0.5).
void fdwt_rev_ver_hp_step_avx512(int32_t n, const float *prev, const float *next, float *tgt) {
  const __m512 k05 = _mm512_set1_ps(0.5f);
  int32_t i = 0;
  for (; i + 16 <= n; i += 16) {
    __m512 a = _mm512_loadu_ps(prev + i);
    __m512 b = _mm512_loadu_ps(next + i);
    __m512 t = _mm512_loadu_ps(tgt + i);
    t = _mm512_sub_ps(t, _mm512_floor_ps(_mm512_mul_ps(_mm512_add_ps(a, b), k05)));
    _mm512_storeu_ps(tgt + i, t);
  }
  if (i < n) {
    __mmask16 mask = static_cast<__mmask16>((1U << (n - i)) - 1U);
    auto a = _mm512_maskz_loadu_ps(mask, prev + i);
    auto b = _mm512_maskz_loadu_ps(mask, next + i);
    auto t = _mm512_maskz_loadu_ps(mask, tgt + i);
    t = _mm512_sub_ps(t, _mm512_floor_ps(_mm512_mul_ps(_mm512_add_ps(a, b), k05)));
    _mm512_mask_storeu_ps(tgt + i, mask, t);
  }
}

// Single-row rev53 FDWT LP vertical lifting: tgt[i] += floor((prev[i]+next[i]+2)*0.25).
void fdwt_rev_ver_lp_step_avx512(int32_t n, const float *prev, const float *next, float *tgt) {
  const __m512 k025 = _mm512_set1_ps(0.25f);
  const __m512 k2   = _mm512_set1_ps(2.0f);
  int32_t i = 0;
  for (; i + 16 <= n; i += 16) {
    __m512 a = _mm512_loadu_ps(prev + i);
    __m512 b = _mm512_loadu_ps(next + i);
    __m512 t = _mm512_loadu_ps(tgt + i);
    t = _mm512_add_ps(t, _mm512_floor_ps(_mm512_mul_ps(_mm512_add_ps(_mm512_add_ps(a, b), k2), k025)));
    _mm512_storeu_ps(tgt + i, t);
  }
  if (i < n) {
    __mmask16 mask = static_cast<__mmask16>((1U << (n - i)) - 1U);
    auto a = _mm512_maskz_loadu_ps(mask, prev + i);
    auto b = _mm512_maskz_loadu_ps(mask, next + i);
    auto t = _mm512_maskz_loadu_ps(mask, tgt + i);
    t = _mm512_add_ps(t, _mm512_floor_ps(_mm512_mul_ps(_mm512_add_ps(_mm512_add_ps(a, b), k2), k025)));
    _mm512_mask_storeu_ps(tgt + i, mask, t);
  }
}

// ============================================================================
// int32 5/3 reversible DWT primitives (lossless path).
//
// The reversible 5/3 lifting is integer arithmetic by spec:
//   H[k] -= (L[k] + L[k+1]) >> 1                          // predict
//   L[k] += (H[k-1] + H[k] + 2) >> 2                      // update
// The existing fdwt_1d_filtr_rev53_fixed_avx512 emulates these in float via
// mul-by-0.5/0.25 + floor_ps, which adds latency vs srai_epi32 (1 cycle).
// These int32 variants are equivalent and intended to replace the float
// versions on the lossless pipeline.  Currently dormant — no callers.
//
// Layout note: the same _mm512_slli_epi64(..., 32) lane-mask trick from the
// float version works identically on the integer reinterpretation, since the
// data layout interleaves L/H values at adjacent int32 positions.
// ============================================================================

void fdwt_1d_filtr_rev53_i32_avx512(int32_t *X, const int32_t left, const int32_t u_i0,
                                    const int32_t u_i1) {
  const int32_t i0     = u_i0;
  const int32_t i1     = u_i1;
  const int32_t start  = ceil_int(i0, 2);
  const int32_t stop   = ceil_int(i1, 2);
  const int32_t offset = left + i0 % 2;

  // step 1: H[k] -= (L[k] + L[k+1]) >> 1
  int32_t simdlen = stop - (start - 1);
  int32_t i = 0, n = -2 + offset;
  for (; i + 8 < simdlen; i += 16, n += 32) {
    __m512i xin0a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(X + n));
    __m512i xin2a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(X + n + 2));
    __m512i xin0b = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(X + n + 16));
    __m512i xin2b = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(X + n + 18));
    __m512i xsuma = _mm512_slli_epi64(_mm512_add_epi32(xin0a, xin2a), 32);
    __m512i xsumb = _mm512_slli_epi64(_mm512_add_epi32(xin0b, xin2b), 32);
    xsuma = _mm512_srai_epi32(xsuma, 1);
    xsumb = _mm512_srai_epi32(xsumb, 1);
    xin0a = _mm512_sub_epi32(xin0a, xsuma);
    xin0b = _mm512_sub_epi32(xin0b, xsumb);
    _mm512_storeu_si512(reinterpret_cast<__m512i *>(X + n), xin0a);
    _mm512_storeu_si512(reinterpret_cast<__m512i *>(X + n + 16), xin0b);
  }
  for (; i < simdlen; i += 8, n += 16) {
    __m512i xin0 = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(X + n));
    __m512i xin2 = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(X + n + 2));
    __m512i xsum = _mm512_slli_epi64(_mm512_add_epi32(xin0, xin2), 32);
    xsum = _mm512_srai_epi32(xsum, 1);
    xin0 = _mm512_sub_epi32(xin0, xsum);
    _mm512_storeu_si512(reinterpret_cast<__m512i *>(X + n), xin0);
  }

  // step 2: L[k] += (H[k-1] + H[k] + 2) >> 2
  simdlen   = stop - start;
  i = 0; n = 0 + offset;
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
    xin0a = _mm512_add_epi32(xin0a, xsuma);
    xin0b = _mm512_add_epi32(xin0b, xsumb);
    _mm512_storeu_si512(reinterpret_cast<__m512i *>(X + n - 1), xin0a);
    _mm512_storeu_si512(reinterpret_cast<__m512i *>(X + n + 15), xin0b);
  }
  for (; i < simdlen; i += 8, n += 16) {
    __m512i xin0 = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(X + n - 1));
    __m512i xin2 = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(X + n + 1));
    __m512i xsum = _mm512_slli_epi64(
        _mm512_add_epi32(_mm512_add_epi32(xin0, xin2), vtwo), 32);
    xsum = _mm512_srai_epi32(xsum, 2);
    xin0 = _mm512_add_epi32(xin0, xsum);
    _mm512_storeu_si512(reinterpret_cast<__m512i *>(X + n - 1), xin0);
  }
}

// Single-row rev53 FDWT HP vertical lifting (int32): tgt[i] -= (prev[i]+next[i]) >> 1.
void fdwt_rev_ver_hp_step_i32_avx512(int32_t n, const int32_t *prev, const int32_t *next, int32_t *tgt) {
  int32_t i = 0;
  for (; i + 16 <= n; i += 16) {
    __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(prev + i));
    __m512i b = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(next + i));
    __m512i t = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(tgt  + i));
    t = _mm512_sub_epi32(t, _mm512_srai_epi32(_mm512_add_epi32(a, b), 1));
    _mm512_storeu_si512(reinterpret_cast<__m512i *>(tgt + i), t);
  }
  if (i < n) {
    __mmask16 mask = static_cast<__mmask16>((1U << (n - i)) - 1U);
    __m512i a = _mm512_maskz_loadu_epi32(mask, prev + i);
    __m512i b = _mm512_maskz_loadu_epi32(mask, next + i);
    __m512i t = _mm512_maskz_loadu_epi32(mask, tgt  + i);
    t = _mm512_sub_epi32(t, _mm512_srai_epi32(_mm512_add_epi32(a, b), 1));
    _mm512_mask_storeu_epi32(tgt + i, mask, t);
  }
}

// Single-row rev53 FDWT LP vertical lifting (int32): tgt[i] += (prev[i]+next[i]+2) >> 2.
void fdwt_rev_ver_lp_step_i32_avx512(int32_t n, const int32_t *prev, const int32_t *next, int32_t *tgt) {
  const __m512i vtwo = _mm512_set1_epi32(2);
  int32_t i = 0;
  for (; i + 16 <= n; i += 16) {
    __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(prev + i));
    __m512i b = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(next + i));
    __m512i t = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(tgt  + i));
    t = _mm512_add_epi32(t,
        _mm512_srai_epi32(_mm512_add_epi32(_mm512_add_epi32(a, b), vtwo), 2));
    _mm512_storeu_si512(reinterpret_cast<__m512i *>(tgt + i), t);
  }
  if (i < n) {
    __mmask16 mask = static_cast<__mmask16>((1U << (n - i)) - 1U);
    __m512i a = _mm512_maskz_loadu_epi32(mask, prev + i);
    __m512i b = _mm512_maskz_loadu_epi32(mask, next + i);
    __m512i t = _mm512_maskz_loadu_epi32(mask, tgt  + i);
    t = _mm512_add_epi32(t,
        _mm512_srai_epi32(_mm512_add_epi32(_mm512_add_epi32(a, b), vtwo), 2));
    _mm512_mask_storeu_epi32(tgt + i, mask, t);
  }
}

/********************************************************************************
 * Planar horizontal FDWT — 16-lane variants of the AVX2 planar kernels
 * (fdwt_avx2.cpp).  Same structure; the full-register valignd replaces the
 * permute2f128+alignr pair for the cross-element shifts, and the
 * deinterleaving load is one vpermt2ps per plane.  Every lifting stage is the
 * same single-rounded _mm512_fmadd_ps / std::fmaf (or exact integer) sequence
 * per element as the AVX2 planar kernels — output is bit-identical, which is
 * what lets emit_ready_f pick per-state between this kernel (N >= 32, the
 * 16-lane warmup loads j = 0..31 unconditionally) and the AVX2 one
 * (16 <= N < 32).
 *******************************************************************************/
// Cross-element shifts with carry-in — full-register valignd.
static inline __m512 fdwt_shl1_ps_512(__m512 a, __m512 b) {  // [a1..a15, b0]
  return _mm512_castsi512_ps(_mm512_alignr_epi32(_mm512_castps_si512(b), _mm512_castps_si512(a), 1));
}
static inline __m512 fdwt_shr1_ps_512(__m512 p, __m512 a) {  // [p15, a0..a14]
  return _mm512_castsi512_ps(_mm512_alignr_epi32(_mm512_castps_si512(a), _mm512_castps_si512(p), 15));
}
static inline __m512i fdwt_shl1_epi32_512(__m512i a, __m512i b) { return _mm512_alignr_epi32(b, a, 1); }
static inline __m512i fdwt_shr1_epi32_512(__m512i p, __m512i a) { return _mm512_alignr_epi32(a, p, 15); }
// Deinterleaving load of one block: e = even lanes in[2j..], o = odd lanes.
static inline void fdwt_load_deint_ps_512(const float *p, __m512 &e, __m512 &o) {
  const __m512i idx_e = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
  const __m512i idx_o = _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);
  __m512 a            = _mm512_loadu_ps(p);
  __m512 b            = _mm512_loadu_ps(p + 16);
  e                   = _mm512_permutex2var_ps(a, idx_e, b);
  o                   = _mm512_permutex2var_ps(a, idx_o, b);
}
static inline void fdwt_load_deint_epi32_512(const int32_t *p, __m512i &e, __m512i &o) {
  __m512 ef, of;
  fdwt_load_deint_ps_512(reinterpret_cast<const float *>(p), ef, of);
  e = _mm512_castps_si512(ef);
  o = _mm512_castps_si512(of);
}

// Fused single-pass 9/7 analysis, planar output.  Stage recurrences as in the
// AVX2 kernel (even u0, E[j] = in[2j], O[j] = in[2j+1], NL = ceil(u1/2)-u0/2,
// NH = u1/2-u0/2):
//   T1[j] = O[j]  + fA*(E[j]    + E[j+1])   j in [-2, NL]
//   T2[j] = E[j]  + fB*(T1[j-1] + T1[j])    j in [-1, NL]
//   T3[j] = T1[j] + fC*(T2[j]   + T2[j+1])  j in [-1, NL-1]
//   T4[j] = T2[j] + fD*(T3[j-1] + T3[j])    j in [ 0, NL-1]
//   hp[j] = T3[j] (j < NH),  lp[j] = T4[j] (j < NL)
void fdwt_1d_filtr_irrev97_planar_avx512(sprec_t *lp, sprec_t *hp, const sprec_t *in, const int32_t u0,
                                         const int32_t u1) {
  const int32_t NH = u1 / 2 - u0 / 2;
  const int32_t NL = ceil_int(u1, 2) - u0 / 2;
  auto E           = [&](int32_t j) -> float { return in[PSEo(u0 + 2 * j, u0, u1) - u0]; };
  auto O           = [&](int32_t j) -> float { return in[PSEo(u0 + 2 * j + 1, u0, u1) - u0]; };

  const __m512 vA = _mm512_set1_ps(fA), vB = _mm512_set1_ps(fB);
  const __m512 vC = _mm512_set1_ps(fC), vD = _mm512_set1_ps(fD);

  // Warmup: boundary scalars, blocks 0/1 loads, T1/T2 of block 0.
  const float t1m2 = std::fmaf(fA, E(-2) + E(-1), O(-2));  // T1[-2]
  const float t1m1 = std::fmaf(fA, E(-1) + E(0), O(-1));   // T1[-1]
  __m512 E0, O0, E1, O1;
  fdwt_load_deint_ps_512(in, E0, O0);
  fdwt_load_deint_ps_512(in + 32, E1, O1);
  __m512 T1_0 = _mm512_fmadd_ps(_mm512_add_ps(E0, fdwt_shl1_ps_512(E0, E1)), vA, O0);
  __m512 T2_0 = _mm512_fmadd_ps(_mm512_add_ps(fdwt_shr1_ps_512(_mm512_set1_ps(t1m1), T1_0), T1_0), vB, E0);
  const float t2m1 = std::fmaf(fB, t1m2 + t1m1, E(-1));                   // T2[-1]
  const float t3m1 = std::fmaf(fC, t2m1 + _mm512_cvtss_f32(T2_0), t1m1);  // T3[-1]

  // Steady state: iteration n deinterleave-loads input block n and emits
  // finished plane block n-2 with two plain stores.
  __m512 E_nm1 = E1, O_nm1 = O1, T1_nm2 = T1_0, T2_nm2 = T2_0;
  __m512 T3_nm3 = _mm512_set1_ps(t3m1);  // only its top lane is consumed (shr1)
  int32_t n     = 2;
  for (; 16 * n + 15 <= NH - 1; ++n) {
    __m512 E_n, O_n;
    fdwt_load_deint_ps_512(in + 32 * n, E_n, O_n);
    __m512 T1_nm1 = _mm512_fmadd_ps(_mm512_add_ps(E_nm1, fdwt_shl1_ps_512(E_nm1, E_n)), vA, O_nm1);
    __m512 T2_nm1 = _mm512_fmadd_ps(_mm512_add_ps(fdwt_shr1_ps_512(T1_nm2, T1_nm1), T1_nm1), vB, E_nm1);
    __m512 T3_nm2 = _mm512_fmadd_ps(_mm512_add_ps(T2_nm2, fdwt_shl1_ps_512(T2_nm2, T2_nm1)), vC, T1_nm2);
    __m512 T4_nm2 = _mm512_fmadd_ps(_mm512_add_ps(fdwt_shr1_ps_512(T3_nm3, T3_nm2), T3_nm2), vD, T2_nm2);
    _mm512_storeu_ps(hp + 16 * (n - 2), T3_nm2);
    _mm512_storeu_ps(lp + 16 * (n - 2), T4_nm2);
    E_nm1  = E_n;
    O_nm1  = O_n;
    T1_nm2 = T1_nm1;
    T2_nm2 = T2_nm1;
    T3_nm3 = T3_nm2;
  }

  // Drain: scalar finish from the carried registers + mirrored accessors.
  // P = 16*(n-2) is the first plane index not yet stored; the loop-exit bound
  // gives NL - P <= 48, so with base = P - 16 every stage index stays within
  // the arrays (max NL - base <= 64).
  {
    const int32_t P    = 16 * (n - 2);
    const int32_t base = P - 16;
    float t1[80], t2[80], t3[80];
    _mm512_storeu_ps(t1 + (P - base), T1_nm2);  // T1[P..P+15]
    _mm512_storeu_ps(t2 + (P - base), T2_nm2);  // T2[P..P+15]
    _mm512_storeu_ps(t3, T3_nm3);               // T3[P-16..P-1] (top lane = T3[P-1])
    for (int32_t j = P + 16; j <= NL; ++j) t1[j - base] = std::fmaf(fA, E(j) + E(j + 1), O(j));
    for (int32_t j = P + 16; j <= NL; ++j)
      t2[j - base] = std::fmaf(fB, t1[j - base - 1] + t1[j - base], E(j));
    for (int32_t j = P; j <= NL - 1; ++j)
      t3[j - base] = std::fmaf(fC, t2[j - base] + t2[j - base + 1], t1[j - base]);
    for (int32_t j = P; j <= NL - 1; ++j)
      lp[j] = std::fmaf(fD, t3[j - base - 1] + t3[j - base], t2[j - base]);
    for (int32_t j = P; j <= NH - 1; ++j) hp[j] = t3[j - base];
  }
}

// Fused single-pass reversible 5/3 analysis, planar int32 output — 16-lane
// variant of fdwt_1d_filtr_rev53_planar_i32_avx2 (integer ops are exact, so
// all variants match bit for bit):
//   T1[j] = O[j] - ((E[j]    + E[j+1]) >> 1)      j in [-1, NL-1]
//   T2[j] = E[j] + ((T1[j-1] + T1[j] + 2) >> 2)   j in [ 0, NL-1]
//   hp[j] = T1[j] (j < NH),  lp[j] = T2[j] (j < NL)
void fdwt_1d_filtr_rev53_planar_i32_avx512(int32_t *lp, int32_t *hp, const int32_t *in, const int32_t u0,
                                           const int32_t u1) {
  const int32_t NH = u1 / 2 - u0 / 2;
  const int32_t NL = ceil_int(u1, 2) - u0 / 2;
  auto E           = [&](int32_t j) -> int32_t { return in[PSEo(u0 + 2 * j, u0, u1) - u0]; };
  auto O           = [&](int32_t j) -> int32_t { return in[PSEo(u0 + 2 * j + 1, u0, u1) - u0]; };

  const __m512i vtwo = _mm512_set1_epi32(2);

  // Warmup: boundary scalar T1[-1], block 0 load.
  const int32_t t1m1 = O(-1) - ((E(-1) + E(0)) >> 1);
  __m512i E_nm1, O_nm1;
  fdwt_load_deint_epi32_512(in, E_nm1, O_nm1);
  __m512i T1_nm2 = _mm512_set1_epi32(t1m1);  // only its top lane is consumed (shr1)

  // Steady state: iteration n deinterleave-loads input block n and emits
  // finished plane block n-1 with two plain stores.
  int32_t n = 1;
  for (; 16 * n + 15 <= NH - 1; ++n) {
    __m512i E_n, O_n;
    fdwt_load_deint_epi32_512(in + 32 * n, E_n, O_n);
    __m512i T1_nm1 = _mm512_sub_epi32(
        O_nm1, _mm512_srai_epi32(_mm512_add_epi32(E_nm1, fdwt_shl1_epi32_512(E_nm1, E_n)), 1));
    __m512i T2_nm1 = _mm512_add_epi32(
        E_nm1,
        _mm512_srai_epi32(
            _mm512_add_epi32(_mm512_add_epi32(fdwt_shr1_epi32_512(T1_nm2, T1_nm1), T1_nm1), vtwo), 2));
    _mm512_storeu_si512(reinterpret_cast<__m512i *>(hp + 16 * (n - 1)), T1_nm1);
    _mm512_storeu_si512(reinterpret_cast<__m512i *>(lp + 16 * (n - 1)), T2_nm1);
    E_nm1  = E_n;
    O_nm1  = O_n;
    T1_nm2 = T1_nm1;
  }

  // Drain: scalar finish from the carried boundary lane + mirrored accessors.
  // P = 16*(n-1) is the first plane index not yet stored; the loop-exit bound
  // gives NH - P <= 31, so NL - P <= 32 and t1buf stays within bounds.
  {
    const int32_t P = 16 * (n - 1);
    // T1[P-1]: rotate lane 15 of the carried block into lane 0 (== t1m1 when P == 0).
    const int32_t t1_prev = _mm512_cvtsi512_si32(_mm512_alignr_epi32(T1_nm2, T1_nm2, 15));
    int32_t t1buf[32];
    for (int32_t j = P; j <= NL - 1; ++j) t1buf[j - P] = O(j) - ((E(j) + E(j + 1)) >> 1);
    for (int32_t j = P; j <= NH - 1; ++j) hp[j] = t1buf[j - P];
    for (int32_t j = P; j <= NL - 1; ++j) {
      const int32_t t1l = (j == P) ? t1_prev : t1buf[j - P - 1];
      lp[j]             = E(j) + ((t1l + t1buf[j - P] + 2) >> 2);
    }
  }
}

#endif  // OPENHTJ2K_TRY_AVX2 && __AVX512F__
