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

#include "utils.hpp"
#if defined(OPENHTJ2K_TRY_AVX2) && defined(__AVX2__)
  #include "dwt.hpp"
  #include <cstring>
/********************************************************************************
 * horizontal transforms
 *******************************************************************************/
// irreversible IDWT
auto idwt_irrev97_fixed_avx2_hor_step = [](const int32_t init_pos, const int32_t simdlen, float *const X,
                                            const int32_t n0, const int32_t n1, const float fV) {
  auto vcoeff = _mm256_set1_ps(fV);
  for (int32_t n = init_pos, i = 0; i < simdlen; i += 4, n += 8) {
    auto xin0 = _mm256_loadu_ps(X + n + n0);
    auto xin2 = _mm256_loadu_ps(X + n + n1);
    auto xsum = _mm256_add_ps(xin0, xin2);
    xsum      = _mm256_blend_ps(xsum, _mm256_setzero_ps(), 0xAA);
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
  for (; simdlen > 0; simdlen -= 4) {
    auto xin0 = _mm256_loadu_ps(sp - 1);
    auto xin2 = _mm256_loadu_ps(sp + 1);
    auto xsum = _mm256_add_ps(_mm256_add_ps(xin0, xin2), xtwo);
    xsum      = _mm256_blend_ps(xsum, xzero, 0xAA);
    auto xfloor = _mm256_floor_ps(_mm256_mul_ps(xsum, x025));
    xsum      = _mm256_castsi256_ps(_mm256_slli_si256(_mm256_castps_si256(xfloor), 4));
    xin0      = _mm256_sub_ps(xin0, xsum);
    _mm256_storeu_ps(sp - 1, xin0);
    sp += 8;
  }

  // step 2
  simdlen = stop - start;
  sp      = X + offset;
  auto x05 = _mm256_set1_ps(0.5f);
  for (; simdlen > 0; simdlen -= 4) {
    auto xin0 = _mm256_loadu_ps(sp);
    auto xin2 = _mm256_loadu_ps(sp + 2);

    auto xsum = _mm256_add_ps(xin0, xin2);
    xsum      = _mm256_blend_ps(xsum, xzero, 0xAA);
    auto xfloor = _mm256_floor_ps(_mm256_mul_ps(xsum, x05));
    xsum      = _mm256_castsi256_ps(_mm256_slli_si256(_mm256_castps_si256(xfloor), 4));
    xin0      = _mm256_add_ps(xin0, xsum);
    _mm256_storeu_ps(sp, xin0);
    sp += 8;
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
                                  const int32_t v1, const int32_t stride) {
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
      buf[top - i] =
          static_cast<sprec_t *>(aligned_mem_alloc(sizeof(sprec_t) * static_cast<size_t>(len), 32));
      memcpy(buf[top - i], &in[PSEo(v0 - i, v0, v1) * stride],
             sizeof(sprec_t) * static_cast<size_t>(stride));
    }
    for (int32_t row = 0; row < v1 - v0; ++row) {
      buf[top + row] = &in[row * stride];
    }
    for (int32_t i = 1; i <= bottom; i++) {
      buf[top + (v1 - v0) + i - 1] =
          static_cast<sprec_t *>(aligned_mem_alloc(sizeof(sprec_t) * static_cast<size_t>(len), 32));
      memcpy(buf[top + (v1 - v0) + i - 1], &in[PSEo(v1 - v0 + i - 1 + v0, v0, v1) * stride],
             sizeof(sprec_t) * static_cast<size_t>(stride));
    }
    const int32_t start  = v0 / 2;
    const int32_t stop   = v1 / 2;
    const int32_t offset = top - v0 % 2;

    const int32_t simdlen = (u1 - u0) - (u1 - u0) % 8;
    for (int32_t n = -2 + offset, i = start - 1; i < stop + 2; i++, n += 2) {
      idwt_irrev97_fixed_avx2_ver_step(simdlen, buf[n - 1], buf[n + 1], buf[n], fD);
      for (int32_t col = simdlen; col < u1 - u0; ++col) {
        float sum = buf[n - 1][col];
        sum += buf[n + 1][col];
        buf[n][col] = buf[n][col] - fD * sum;
      }
    }
    for (int32_t n = -2 + offset, i = start - 1; i < stop + 1; i++, n += 2) {
      idwt_irrev97_fixed_avx2_ver_step(simdlen, buf[n], buf[n + 2], buf[n + 1], fC);
      for (int32_t col = simdlen; col < u1 - u0; ++col) {
        float sum = buf[n][col];
        sum += buf[n + 2][col];
        buf[n + 1][col] = buf[n + 1][col] - fC * sum;
      }
    }
    for (int32_t n = 0 + offset, i = start; i < stop + 1; i++, n += 2) {
      idwt_irrev97_fixed_avx2_ver_step(simdlen, buf[n - 1], buf[n + 1], buf[n], fB);
      for (int32_t col = simdlen; col < u1 - u0; ++col) {
        float sum = buf[n - 1][col];
        sum += buf[n + 1][col];
        buf[n][col] = buf[n][col] - fB * sum;
      }
    }
    for (int32_t n = 0 + offset, i = start; i < stop; i++, n += 2) {
      idwt_irrev97_fixed_avx2_ver_step(simdlen, buf[n], buf[n + 2], buf[n + 1], fA);
      for (int32_t col = simdlen; col < u1 - u0; ++col) {
        float sum = buf[n][col];
        sum += buf[n + 2][col];
        buf[n + 1][col] = buf[n + 1][col] - fA * sum;
      }
    }

    for (int32_t i = 1; i <= top; ++i) {
      aligned_mem_free(buf[top - i]);
    }
    for (int32_t i = 1; i <= bottom; i++) {
      aligned_mem_free(buf[top + (v1 - v0) + i - 1]);
    }
    delete[] buf;
  }
}

// reversible IDWT
void idwt_rev_ver_sr_fixed_avx2(sprec_t *in, const int32_t u0, const int32_t u1, const int32_t v0,
                                const int32_t v1, const int32_t stride) {
  constexpr int32_t num_pse_i0[2] = {1, 2};
  constexpr int32_t num_pse_i1[2] = {2, 1};
  const int32_t top               = num_pse_i0[v0 % 2];
  const int32_t bottom            = num_pse_i1[v1 % 2];
  if (v0 == v1 - 1 && (v0 % 2)) {
    // one sample case
    for (int32_t col = 0; col < u1 - u0; ++col) {
      in[col] = static_cast<sprec_t>((int32_t)in[col] >> 1);
    }
  } else {
    const int32_t len = round_up(stride, SIMD_PADDING);
    auto **buf        = new sprec_t *[static_cast<size_t>(top + v1 - v0 + bottom)];
    for (int32_t i = 1; i <= top; ++i) {
      buf[top - i] =
          static_cast<sprec_t *>(aligned_mem_alloc(sizeof(sprec_t) * static_cast<size_t>(len), 32));
      memcpy(buf[top - i], &in[PSEo(v0 - i, v0, v1) * stride],
             sizeof(sprec_t) * static_cast<size_t>(stride));
    }
    for (int32_t row = 0; row < v1 - v0; ++row) {
      buf[top + row] = &in[row * stride];
    }
    for (int32_t i = 1; i <= bottom; i++) {
      buf[top + (v1 - v0) + i - 1] =
          static_cast<sprec_t *>(aligned_mem_alloc(sizeof(sprec_t) * static_cast<size_t>(len), 32));
      memcpy(buf[top + (v1 - v0) + i - 1], &in[PSEo(v1 - v0 + i - 1 + v0, v0, v1) * stride],
             sizeof(sprec_t) * static_cast<size_t>(stride));
    }
    const int32_t start  = v0 / 2;
    const int32_t stop   = v1 / 2;
    const int32_t offset = top - v0 % 2;

    const int32_t simdlen = (u1 - u0) - (u1 - u0) % 8;
    const __m256 xtwo = _mm256_set1_ps(2.0f);
    const __m256 x025 = _mm256_set1_ps(0.25f);
    __m256 x0, x1, x2;
    for (int32_t n = 0 + offset, i = start; i < stop + 1; ++i, n += 2) {
      float *xp0 = buf[n - 1];
      float *xp1 = buf[n];
      float *xp2 = buf[n + 1];
      for (int32_t col = 0; col < simdlen; col += 8) {
        x0           = _mm256_load_ps(xp0);
        x2           = _mm256_load_ps(xp2);
        x1           = _mm256_load_ps(xp1);
        auto xfloor = _mm256_floor_ps(_mm256_mul_ps(_mm256_add_ps(_mm256_add_ps(x0, x2), xtwo), x025));
        x1 = _mm256_sub_ps(x1, xfloor);
        // __m256i vout = _mm256_add_epi32(vone, _mm256_srai_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(x0), _mm256_cvtps_epi32(x2)), 1));
        // vout         = _mm256_srai_epi32(vout, 1);
        // x1           = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtps_epi32(x1), vout));
        _mm256_store_ps(xp1, x1);
        // _mm_prefetch(reinterpret_cast<char *>((__m256 *)xp0 + 2), _MM_HINT_NTA);
        // _mm_prefetch(reinterpret_cast<char *>((__m256 *)xp1 + 2), _MM_HINT_NTA);
        // _mm_prefetch(reinterpret_cast<char *>((__m256 *)xp2 + 2), _MM_HINT_NTA);
        xp0 += 8;
        xp1 += 8;
        xp2 += 8;
      }
      for (int32_t col = simdlen; col < u1 - u0; ++col) {
        int32_t sum = (int32_t)*xp0++;
        sum += (int32_t)*xp2++;
        *xp1 = static_cast<sprec_t>((int32_t)*xp1 - ((sum + 2) >> 2));
        xp1++;
      }
    }
    const __m256 x05 = _mm256_set1_ps(0.5f);
    for (int32_t n = 0 + offset, i = start; i < stop; ++i, n += 2) {
      float *xp0 = buf[n];
      float *xp1 = buf[n + 1];
      float *xp2 = buf[n + 2];
      for (int32_t col = 0; col < simdlen; col += 8) {
        x0 = _mm256_load_ps(xp0);
        x2 = _mm256_load_ps(xp2);
        x1 = _mm256_load_ps(xp1);
        auto xfloor = _mm256_floor_ps(_mm256_mul_ps(_mm256_add_ps(x0, x2), x05));
        x1 = _mm256_add_ps(x1, xfloor);
        // x1 = _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_cvtps_epi32(x1), _mm256_srai_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(x0), _mm256_cvtps_epi32(x2)), 1)));
        _mm256_store_ps(xp1, x1);
        // _mm_prefetch(reinterpret_cast<char *>((__m256 *)xp0 + 2), _MM_HINT_NTA);
        // _mm_prefetch(reinterpret_cast<char *>((__m256 *)xp1 + 2), _MM_HINT_NTA);
        // _mm_prefetch(reinterpret_cast<char *>((__m256 *)xp2 + 2), _MM_HINT_NTA);
        xp0 += 8;
        xp1 += 8;
        xp2 += 8;
      }
      for (int32_t col = simdlen; col < u1 - u0; ++col) {
        int32_t sum = (int32_t)*xp0++;
        sum += (int32_t)*xp2++;
        *xp1 = static_cast<sprec_t>((int32_t)*xp1 + (sum >> 1));
        xp1++;
      }
    }

    for (int32_t i = 1; i <= top; ++i) {
      aligned_mem_free(buf[top - i]);
    }
    for (int32_t i = 1; i <= bottom; i++) {
      aligned_mem_free(buf[top + (v1 - v0) + i - 1]);
    }
    delete[] buf;
  }
}
#endif
