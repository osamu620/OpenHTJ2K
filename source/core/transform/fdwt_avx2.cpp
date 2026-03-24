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
// irreversible FDWT
auto fdwt_irrev97_fixed_avx2_hor_step = [](const int32_t init_pos, const int32_t simdlen, float *const X,
                                            const int32_t n0, const int32_t n1, float coeff) {
  auto vcoeff = _mm256_set1_ps(coeff);
  for (int32_t n = init_pos, i = 0; i < simdlen; i += 4, n += 8) {
    auto xin0 = _mm256_loadu_ps(X + n + n0);
    auto xin2 = _mm256_loadu_ps(X + n + n1);
    auto xsum = _mm256_add_ps(xin0, xin2);
    xsum      = _mm256_blend_ps(xsum, _mm256_setzero_ps(), 0xAA);
    xsum      = _mm256_mul_ps(xsum, vcoeff);
    xsum      = _mm256_cvtepi32_ps(_mm256_slli_si256(_mm256_cvtps_epi32(xsum), 4));
    xin0      = _mm256_add_ps(xsum, xin0);
    _mm256_storeu_ps(X + n + n0, xin0);
  }
};

void fdwt_1d_filtr_irrev97_fixed_avx2(sprec_t *X, const int32_t left, const int32_t u_i0,
                                      const int32_t u_i1) {
  const auto i0       = static_cast<int32_t>(u_i0);
  const auto i1       = static_cast<int32_t>(u_i1);
  const int32_t start = ceil_int(i0, 2);
  const int32_t stop  = ceil_int(i1, 2);

  const int32_t offset = left + i0 % 2;

  // step 1
  int32_t simdlen = stop + 1 - (start - 2);
  fdwt_irrev97_fixed_avx2_hor_step(offset - 4, simdlen, X, 0, 2,fA);

  // step 2
  simdlen = stop + 1 - (start - 1);
  fdwt_irrev97_fixed_avx2_hor_step(offset - 2, simdlen, X, -1, 1, fB);

  // step 3
  simdlen = stop - (start - 1);
  fdwt_irrev97_fixed_avx2_hor_step(offset - 2, simdlen, X, 0, 2, fC);

  // step 4
  simdlen = stop - start;
  fdwt_irrev97_fixed_avx2_hor_step(offset, simdlen, X, -1, 1, fD);
}

// reversible FDWT
void fdwt_1d_filtr_rev53_fixed_avx2(sprec_t *X, const int32_t left, const int32_t u_i0,
                                    const int32_t u_i1) {
  const auto i0        = static_cast<int32_t>(u_i0);
  const auto i1        = static_cast<int32_t>(u_i1);
  const int32_t start  = ceil_int(i0, 2);
  const int32_t stop   = ceil_int(i1, 2);
  const int32_t offset = left + i0 % 2;

  // step 1
  int32_t simdlen = stop - (start - 1);
  for (int32_t n = -2 + offset, i = 0; i < simdlen; i += 4, n += 8) {
    auto xin0 = _mm256_loadu_ps(X + n);
    auto xin2 = _mm256_loadu_ps(X + n + 2);
    auto xsum = _mm256_add_epi32(_mm256_cvtps_epi32(xin0), _mm256_cvtps_epi32(xin2));
    xsum      = _mm256_blend_epi32(xsum, _mm256_setzero_si256(), 0xAA);
    xsum      = _mm256_srai_epi32(xsum, 1);
    xsum      = _mm256_slli_si256(xsum, 4);
    xin0      = _mm256_sub_ps(xin0, _mm256_cvtepi32_ps(xsum));
    _mm256_storeu_ps(X + n, xin0);
  }

  // step 2
  simdlen   = stop - start;
  auto xtwo = _mm256_set1_epi32(2);
  for (int32_t n = 0 + offset, i = 0; i < simdlen; i += 4, n += 8) {
    auto xin0 = _mm256_loadu_ps(X + n - 1);
    auto xin2 = _mm256_loadu_ps(X + n + 1);
    auto xsum = _mm256_add_epi32(_mm256_cvtps_epi32(xin0), _mm256_cvtps_epi32(xin2));
    xsum      = _mm256_add_epi32(xsum, xtwo);
    xsum      = _mm256_blend_epi32(xsum, _mm256_setzero_si256(), 0xAA);
    xsum      = _mm256_srai_epi32(xsum, 2);
    xsum      = _mm256_slli_si256(xsum, 4);
    xin0      = _mm256_add_ps(xin0, _mm256_cvtepi32_ps(xsum));
    _mm256_storeu_ps(X + n - 1, xin0);
  }
}



/********************************************************************************
 * vertical transforms
 *******************************************************************************/
// irreversible FDWT
auto fdwt_irrev97_fixed_avx2_ver_step = [](const int32_t simdlen, float *const Xin0, float *const Xin1,
                                            float *const Xout, float coeff) {
  auto vcoeff = _mm256_set1_ps(coeff);
  for (int32_t n = 0; n < simdlen; n += 8) {
    auto xin0 = _mm256_loadu_ps(Xin0 + n);
    auto xin2 = _mm256_loadu_ps(Xin1 + n);
    auto xsum = _mm256_add_ps(xin0, xin2);
    auto xtmp = _mm256_mul_ps(xsum, vcoeff);
    auto xout = _mm256_loadu_ps(Xout + n);
    xout      = _mm256_add_ps(xout, xtmp);
    _mm256_storeu_ps(Xout + n, xout);
  }
};

void fdwt_irrev_ver_sr_fixed_avx2(sprec_t *in, const int32_t u0, const int32_t u1, const int32_t v0,
                                  const int32_t v1, const int32_t stride) {
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
    auto **buf        = new sprec_t *[static_cast<size_t>(top + v1 - v0 + bottom)];
    for (int32_t i = 1; i <= top; ++i) {
      buf[top - i] =
          static_cast<sprec_t *>(aligned_mem_alloc(sizeof(sprec_t) * static_cast<size_t>(len), 32));
      memcpy(buf[top - i], &in[PSEo(v0 - i, v0, v1) * stride],
             sizeof(sprec_t) * static_cast<size_t>(stride));
      // buf[top - i] = &in[(PSEo(v0 - i, v0, v1) - v0) * stride];
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
    const int32_t start  = ceil_int(v0, 2);
    const int32_t stop   = ceil_int(v1, 2);
    const int32_t offset = top + v0 % 2;

    const int32_t simdlen = (u1 - u0) - (u1 - u0) % 8;
    for (int32_t n = -4 + offset, i = start - 2; i < stop + 1; i++, n += 2) {
      fdwt_irrev97_fixed_avx2_ver_step(simdlen, buf[n], buf[n + 2], buf[n + 1], fA);
      for (int32_t col = simdlen; col < u1 - u0; ++col) {
        float sum = buf[n][col];
        sum += buf[n + 2][col];
        buf[n + 1][col] = buf[n + 1][col] + fA * sum;
      }
    }
    for (int32_t n = -2 + offset, i = start - 1; i < stop + 1; i++, n += 2) {
      fdwt_irrev97_fixed_avx2_ver_step(simdlen, buf[n - 1], buf[n + 1], buf[n], fB);
      for (int32_t col = simdlen; col < u1 - u0; ++col) {
        float sum = buf[n - 1][col];
        sum += buf[n + 1][col];
        buf[n][col] = buf[n][col] + fB * sum;
      }
    }
    for (int32_t n = -2 + offset, i = start - 1; i < stop; i++, n += 2) {
      fdwt_irrev97_fixed_avx2_ver_step(simdlen, buf[n], buf[n + 2], buf[n + 1], fC);
      for (int32_t col = simdlen; col < u1 - u0; ++col) {
        float sum = buf[n][col];
        sum += buf[n + 2][col];
        buf[n + 1][col] = buf[n + 1][col] + fC * sum;
      }
    }
    for (int32_t n = 0 + offset, i = start; i < stop; i++, n += 2) {
      fdwt_irrev97_fixed_avx2_ver_step(simdlen, buf[n - 1], buf[n + 1], buf[n], fD);
      for (int32_t col = simdlen; col < u1 - u0; ++col) {
        float sum = buf[n - 1][col];
        sum += buf[n + 1][col];
        buf[n][col] = buf[n][col] + fD * sum;
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

// reversible FDWT
void fdwt_rev_ver_sr_fixed_avx2(sprec_t *in, const int32_t u0, const int32_t u1, const int32_t v0,
                                const int32_t v1, const int32_t stride) {
  constexpr int32_t num_pse_i0[2] = {2, 1};
  constexpr int32_t num_pse_i1[2] = {1, 2};
  const int32_t top               = num_pse_i0[v0 % 2];
  const int32_t bottom            = num_pse_i1[v1 % 2];

  if (v0 == v1 - 1) {
    // one sample case
    for (int32_t col = 0; col < u1 - u0; ++col) {
      if (v0 % 2) {
        in[col] = static_cast<sprec_t>((int32_t)in[col] << 1);
      }
    }
  } else {
    const int32_t len = round_up(stride, SIMD_PADDING);
    auto **buf        = new sprec_t *[static_cast<size_t>(top + v1 - v0 + bottom)];
    for (int32_t i = 1; i <= top; ++i) {
      buf[top - i] =
          static_cast<sprec_t *>(aligned_mem_alloc(sizeof(sprec_t) * static_cast<size_t>(len), 32));
      memcpy(buf[top - i], &in[PSEo(v0 - i, v0, v1) * stride],
             sizeof(sprec_t) * static_cast<size_t>(stride));
      // buf[top - i] = &in[(PSEo(v0 - i, v0, v1) - v0) * stride];
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
    const int32_t start  = ceil_int(v0, 2);
    const int32_t stop   = ceil_int(v1, 2);
    const int32_t offset = top + v0 % 2;

    const int32_t simdlen = (u1 - u0) - (u1 - u0) % 8;
    for (int32_t n = -2 + offset, i = start - 1; i < stop; ++i, n += 2) {
      for (int32_t col = 0; col < simdlen; col += 8) {
        __m256 x0 = _mm256_loadu_ps(buf[n] + col);
        __m256 x2 = _mm256_loadu_ps(buf[n + 2] + col);
        __m256 x1 = _mm256_loadu_ps(buf[n + 1] + col);
        x1         = _mm256_cvtepi32_ps(
          _mm256_sub_epi32(_mm256_cvtps_epi32(x1),
            _mm256_srai_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(x0), _mm256_cvtps_epi32(x2)), 1)
            )
            );
        _mm256_storeu_ps(buf[n + 1] + col, x1);
      }
      for (int32_t col = simdlen; col < u1 - u0; ++col) {
        int32_t sum = (int32_t)buf[n][col];
        sum += (int32_t)buf[n + 2][col];
        buf[n + 1][col] = static_cast<sprec_t>((int32_t)buf[n + 1][col] - (sum >> 1));
      }
    }
    for (int32_t n = 0 + offset, i = start; i < stop; ++i, n += 2) {
      for (int32_t col = 0; col < simdlen; col += 8) {
        __m256 x0 = _mm256_loadu_ps(buf[n - 1] + col);
        __m256 x2 = _mm256_loadu_ps(buf[n + 1] + col);
        __m256 x1 = _mm256_loadu_ps(buf[n] + col);
        __m256i vout =
            _mm256_add_epi32(_mm256_set1_epi32(1), _mm256_srai_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(x0), _mm256_cvtps_epi32(x2)), 1));
        vout = _mm256_srai_epi32(vout, 1);
        x1   = _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_cvtps_epi32(x1), vout));
        _mm256_storeu_ps(buf[n] + col, x1);
      }
      for (int32_t col = simdlen; col < u1 - u0; ++col) {
        int32_t sum = (int32_t)buf[n - 1][col];
        sum += (int32_t)buf[n + 1][col];
        sum >>= 1;
        sum += 1;
        sum >>= 1;
        buf[n][col] = static_cast<sprec_t>((int32_t)buf[n][col] + sum);
        // buf[n][col] += ((sum >> 1) + 1) >> 1;  //((sum + 2) >> 2);
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