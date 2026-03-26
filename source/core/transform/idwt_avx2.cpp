// Copyright (c) 2019 - 2021, Osamu Watanabe
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
  auto vzero = _mm256_setzero_ps();
  int32_t n = init_pos, i = 0;
  // 2× unrolled main loop: two independent 4-sample groups per iteration for better ILP.
  for (; i + 4 < simdlen; i += 8, n += 16) {
    auto xin0a = _mm256_loadu_ps(X + n + n0);
    auto xin2a = _mm256_loadu_ps(X + n + n1);
    auto xin0b = _mm256_loadu_ps(X + n + 8 + n0);
    auto xin2b = _mm256_loadu_ps(X + n + 8 + n1);
    auto xsuma = _mm256_add_ps(xin0a, xin2a);
    auto xsumb = _mm256_add_ps(xin0b, xin2b);
    xsuma = _mm256_blend_ps(xsuma, vzero, 0xAA);
    xsumb = _mm256_blend_ps(xsumb, vzero, 0xAA);
    xsuma = _mm256_mul_ps(xsuma, vcoeff);
    xsumb = _mm256_mul_ps(xsumb, vcoeff);
    xsuma = _mm256_castsi256_ps(_mm256_slli_si256(_mm256_castps_si256(xsuma), 4));
    xsumb = _mm256_castsi256_ps(_mm256_slli_si256(_mm256_castps_si256(xsumb), 4));
    xin0a = _mm256_sub_ps(xin0a, xsuma);
    xin0b = _mm256_sub_ps(xin0b, xsumb);
    _mm256_storeu_ps(X + n + n0, xin0a);
    _mm256_storeu_ps(X + n + 8 + n0, xin0b);
  }
  for (; i < simdlen; i += 4, n += 8) {
    auto xin0 = _mm256_loadu_ps(X + n + n0);
    auto xin2 = _mm256_loadu_ps(X + n + n1);
    auto xsum = _mm256_add_ps(xin0, xin2);
    xsum      = _mm256_blend_ps(xsum, vzero, 0xAA);
    xsum      = _mm256_mul_ps(xsum, vcoeff);
    xsum      = _mm256_castsi256_ps(_mm256_slli_si256(_mm256_castps_si256(xsum), 4));
    xin0      = _mm256_sub_ps(xin0, xsum);
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

  // step 1
  int32_t simdlen = stop + 1 - start;
  sprec_t *sp     = X + offset;
  const auto xzero = _mm256_setzero_ps();
  const auto xtwo = _mm256_set1_ps(2.0f);
  const auto x025 = _mm256_set1_ps(0.25f);
  int32_t i = 0;
  // 2× unrolled main loop: two independent 4-sample groups per iteration for better ILP.
  for (; i + 4 < simdlen; i += 8, sp += 16) {
    auto xin0a = _mm256_loadu_ps(sp - 1);
    auto xin2a = _mm256_loadu_ps(sp + 1);
    auto xin0b = _mm256_loadu_ps(sp + 7);
    auto xin2b = _mm256_loadu_ps(sp + 9);
    auto xsuma = _mm256_add_ps(_mm256_add_ps(xin0a, xin2a), xtwo);
    auto xsumb = _mm256_add_ps(_mm256_add_ps(xin0b, xin2b), xtwo);
    xsuma      = _mm256_blend_ps(xsuma, xzero, 0xAA);
    xsumb      = _mm256_blend_ps(xsumb, xzero, 0xAA);
    auto xfa   = _mm256_floor_ps(_mm256_mul_ps(xsuma, x025));
    auto xfb   = _mm256_floor_ps(_mm256_mul_ps(xsumb, x025));
    xfa        = _mm256_castsi256_ps(_mm256_slli_si256(_mm256_castps_si256(xfa), 4));
    xfb        = _mm256_castsi256_ps(_mm256_slli_si256(_mm256_castps_si256(xfb), 4));
    xin0a      = _mm256_sub_ps(xin0a, xfa);
    xin0b      = _mm256_sub_ps(xin0b, xfb);
    _mm256_storeu_ps(sp - 1, xin0a);
    _mm256_storeu_ps(sp + 7, xin0b);
  }
  for (; i < simdlen; i += 4, sp += 8) {
    auto xin0 = _mm256_loadu_ps(sp - 1);
    auto xin2 = _mm256_loadu_ps(sp + 1);
    auto xsum = _mm256_add_ps(_mm256_add_ps(xin0, xin2), xtwo);
    xsum      = _mm256_blend_ps(xsum, xzero, 0xAA);
    auto xfloor = _mm256_floor_ps(_mm256_mul_ps(xsum, x025));
    xsum      = _mm256_castsi256_ps(_mm256_slli_si256(_mm256_castps_si256(xfloor), 4));
    xin0      = _mm256_sub_ps(xin0, xsum);
    _mm256_storeu_ps(sp - 1, xin0);
  }

  // step 2
  simdlen = stop - start;
  sp      = X + offset;
  auto x05 = _mm256_set1_ps(0.5f);
  i = 0;
  // 2× unrolled main loop: two independent 4-sample groups per iteration for better ILP.
  for (; i + 4 < simdlen; i += 8, sp += 16) {
    auto xin0a = _mm256_loadu_ps(sp);
    auto xin2a = _mm256_loadu_ps(sp + 2);
    auto xin0b = _mm256_loadu_ps(sp + 8);
    auto xin2b = _mm256_loadu_ps(sp + 10);
    auto xsuma = _mm256_add_ps(xin0a, xin2a);
    auto xsumb = _mm256_add_ps(xin0b, xin2b);
    xsuma      = _mm256_blend_ps(xsuma, xzero, 0xAA);
    xsumb      = _mm256_blend_ps(xsumb, xzero, 0xAA);
    auto xfa   = _mm256_floor_ps(_mm256_mul_ps(xsuma, x05));
    auto xfb   = _mm256_floor_ps(_mm256_mul_ps(xsumb, x05));
    xfa        = _mm256_castsi256_ps(_mm256_slli_si256(_mm256_castps_si256(xfa), 4));
    xfb        = _mm256_castsi256_ps(_mm256_slli_si256(_mm256_castps_si256(xfb), 4));
    xin0a      = _mm256_add_ps(xin0a, xfa);
    xin0b      = _mm256_add_ps(xin0b, xfb);
    _mm256_storeu_ps(sp, xin0a);
    _mm256_storeu_ps(sp + 8, xin0b);
  }
  for (; i < simdlen; i += 4, sp += 8) {
    auto xin0 = _mm256_loadu_ps(sp);
    auto xin2 = _mm256_loadu_ps(sp + 2);

    auto xsum = _mm256_add_ps(xin0, xin2);
    xsum      = _mm256_blend_ps(xsum, xzero, 0xAA);
    auto xfloor = _mm256_floor_ps(_mm256_mul_ps(xsum, x05));
    xsum      = _mm256_castsi256_ps(_mm256_slli_si256(_mm256_castps_si256(xfloor), 4));
    xin0      = _mm256_add_ps(xin0, xsum);
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
    xout = _mm256_fnmadd_ps(xsum, vcoeff,xout);
    _mm256_store_ps(Xout + n, xout);
  }
};

void idwt_irrev_ver_sr_fixed_avx2(sprec_t *in, const int32_t u0, const int32_t u1, const int32_t v0,
                                  const int32_t v1, const int32_t stride, sprec_t *pse_scratch) {
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
    auto **buf        = new sprec_t *[static_cast<size_t>(top + v1 - v0 + bottom)];
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
    delete[] buf;
  }
}

// reversible IDWT
void idwt_rev_ver_sr_fixed_avx2(sprec_t *in, const int32_t u0, const int32_t u1, const int32_t v0,
                                const int32_t v1, const int32_t stride, sprec_t *pse_scratch) {
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
    auto **buf        = new sprec_t *[static_cast<size_t>(top + v1 - v0 + bottom)];
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
    delete[] buf;
  }
}
#endif
