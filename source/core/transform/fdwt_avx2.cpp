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
// irreversible FDWT
auto fdwt_irrev97_fixed_avx2_hor_step = [](int32_t init_pos, int32_t simdlen, int16_t *X, int32_t n0,
                                           int32_t n1, int32_t coeff, int32_t offset, int32_t shift) {
  auto vcoeff  = _mm256_set1_epi32(coeff);
  auto voffset = _mm256_set1_epi32(offset);
  for (int32_t n = init_pos, i = 0; i < simdlen; i += 8, n += 16) {
    auto xin0    = _mm256_loadu_si256((__m256i *)(X + n + n0));
    auto xin2    = _mm256_loadu_si256((__m256i *)(X + n + n1));
    auto xin_tmp = _mm256_permutevar8x32_epi32(
        _mm256_shufflelo_epi16(_mm256_shufflehi_epi16(xin0, 0b11011000), 0b11011000),
        _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7));
    auto xin00 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(xin_tmp, 0));
    auto xin01 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(xin_tmp, 1));
    xin_tmp    = _mm256_permutevar8x32_epi32(
           _mm256_shufflelo_epi16(_mm256_shufflehi_epi16(xin2, 0b11011000), 0b11011000),
           _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7));
    auto xin20 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(xin_tmp, 0));
    auto vsum  = _mm256_add_epi32(xin00, xin20);
    xin01      = _mm256_add_epi32(
             _mm256_srai_epi32(_mm256_add_epi32(_mm256_mullo_epi32(vsum, vcoeff), voffset), shift), xin01);
    auto xout_even_odd = _mm256_shuffle_epi32(_mm256_packs_epi32(xin00, xin01), 0b11011000);
    auto xout_interleaved =
        _mm256_shufflelo_epi16(_mm256_shufflehi_epi16(xout_even_odd, 0b11011000), 0b11011000);
    _mm256_storeu_si256((__m256i *)(X + n + n0), xout_interleaved);
  }
};

auto fdwt_irrev97_fixed_avx2_ver_step = [](int32_t simdlen, int16_t *Xin0, int16_t *Xin1, int16_t *Xout,
                                           int32_t coeff, int32_t offset, int32_t shift) {
  auto vcoeff  = _mm256_set1_epi32(coeff);
  auto voffset = _mm256_set1_epi32(offset);
  for (int32_t n = 0; n < simdlen; n += 16) {
    auto xin0_16 = _mm256_loadu_si256((__m256i *)(Xin0 + n));
    auto xin2_16 = _mm256_loadu_si256((__m256i *)(Xin1 + n));
    auto xout16  = _mm256_loadu_si256((__m256i *)(Xout + n));
    // low
    auto xin0_32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(xin0_16, 0));
    auto xin2_32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(xin2_16, 0));
    auto xout32  = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(xout16, 0));
    auto vsum32  = _mm256_add_epi32(xin0_32, xin2_32);
    auto xout32l = _mm256_add_epi32(
        _mm256_srai_epi32(_mm256_add_epi32(_mm256_mullo_epi32(vsum32, vcoeff), voffset), shift), xout32);

    // high
    xin0_32      = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(xin0_16, 1));
    xin2_32      = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(xin2_16, 1));
    xout32       = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(xout16, 1));
    vsum32       = _mm256_add_epi32(xin0_32, xin2_32);
    auto xout32h = _mm256_add_epi32(
        _mm256_srai_epi32(_mm256_add_epi32(_mm256_mullo_epi32(vsum32, vcoeff), voffset), shift), xout32);

    // pack and store
    _mm256_storeu_si256((__m256i *)(Xout + n),
                        _mm256_permute4x64_epi64(_mm256_packs_epi32(xout32l, xout32h), 0b11011000));
  }
};

void fdwt_1d_filtr_irrev97_fixed_avx2(sprec_t *X, const int32_t left, const int32_t right,
                                      const uint32_t u_i0, const uint32_t u_i1) {
  const auto i0       = static_cast<const int32_t>(u_i0);
  const auto i1       = static_cast<const int32_t>(u_i1);
  const int32_t start = ceil_int(i0, 2);
  const int32_t stop  = ceil_int(i1, 2);

  const int32_t offset = left + i0 % 2;

  // step 1
  int32_t simdlen = stop + 1 - (start - 2);
  fdwt_irrev97_fixed_avx2_hor_step(offset - 4, simdlen, X, 0, 2, Acoeff, Aoffset, Ashift);

  // step 2
  simdlen = stop + 1 - (start - 1);
  fdwt_irrev97_fixed_avx2_hor_step(offset - 2, simdlen, X, -1, 1, Bcoeff, Boffset, Bshift);

  // step 3
  simdlen = stop - (start - 1);
  fdwt_irrev97_fixed_avx2_hor_step(offset - 2, simdlen, X, 0, 2, Ccoeff, Coffset, Cshift);

  // step 4
  simdlen = stop - start;
  fdwt_irrev97_fixed_avx2_hor_step(offset, simdlen, X, -1, 1, Dcoeff, Doffset, Dshift);
}

// reversible FDWT
void fdwt_1d_filtr_rev53_fixed_avx2(sprec_t *X, const int32_t left, const int32_t right,
                                    const uint32_t u_i0, const uint32_t u_i1) {
  const auto i0        = static_cast<const int32_t>(u_i0);
  const auto i1        = static_cast<const int32_t>(u_i1);
  const int32_t start  = ceil_int(i0, 2);
  const int32_t stop   = ceil_int(i1, 2);
  const int32_t offset = left + i0 % 2;

  // step 1
  int32_t simdlen = stop - (start - 1);
  for (int32_t n = -2 + offset, i = 0; i < simdlen; i += 8, n += 16) {
    auto xin0  = _mm256_loadu_si256((__m256i *)(X + n));
    auto xin2  = _mm256_loadu_si256((__m256i *)(X + n + 2));
    auto xin02 = _mm256_permutevar8x32_epi32(
        _mm256_shufflelo_epi16(_mm256_shufflehi_epi16(xin0, 0b11011000), 0b11011000),
        _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7));
    auto xeven0 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(xin02, 0));
    auto xodd0  = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(xin02, 1));
    auto xin22  = _mm256_permutevar8x32_epi32(
         _mm256_shufflelo_epi16(_mm256_shufflehi_epi16(xin2, 0b11011000), 0b11011000),
         _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7));
    auto xeven1        = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(xin22, 0));
    auto vsum          = _mm256_add_epi32(xeven0, xeven1);
    auto xout          = _mm256_sub_epi32(xodd0, _mm256_srai_epi32(vsum, 1));
    auto xout_even_odd = _mm256_shuffle_epi32(_mm256_packs_epi32(xeven0, xout), 0b11011000);
    auto xout_interleaved =
        _mm256_shufflelo_epi16(_mm256_shufflehi_epi16(xout_even_odd, 0b11011000), 0b11011000);
    _mm256_storeu_si256((__m256i *)(X + n), xout_interleaved);
  }

  // step 2
  simdlen = stop - start;
  for (int32_t n = 0 + offset, i = 0; i < simdlen; i += 8, n += 16) {
    auto xin0  = _mm256_loadu_si256((__m256i *)(X + n - 1));
    auto xin2  = _mm256_loadu_si256((__m256i *)(X + n + 1));
    auto xin02 = _mm256_permutevar8x32_epi32(
        _mm256_shufflelo_epi16(_mm256_shufflehi_epi16(xin0, 0b11011000), 0b11011000),
        _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7));
    auto xodd0  = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(xin02, 0));
    auto xeven0 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(xin02, 1));
    auto xin22  = _mm256_permutevar8x32_epi32(
         _mm256_shufflelo_epi16(_mm256_shufflehi_epi16(xin2, 0b11011000), 0b11011000),
         _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7));
    auto xodd1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(xin22, 0));
    auto vsum  = _mm256_add_epi32(xodd0, xodd1);
    auto xout =
        _mm256_add_epi32(xeven0, _mm256_srai_epi32(_mm256_add_epi32(vsum, _mm256_set1_epi32(2)), 2));
    auto xout_odd_even = _mm256_shuffle_epi32(_mm256_packs_epi32(xodd0, xout), 0b11011000);
    auto xout_interleaved =
        _mm256_shufflelo_epi16(_mm256_shufflehi_epi16(xout_odd_even, 0b11011000), 0b11011000);
    _mm256_storeu_si256((__m256i *)(X + n - 1), xout_interleaved);
  }
}

// vertical transforms
void fdwt_irrev_ver_sr_fixed_avx2(sprec_t *in, const uint32_t u0, const uint32_t u1, const uint32_t v0,
                                  const uint32_t v1) {
  const uint32_t stride           = u1 - u0;
  constexpr int32_t num_pse_i0[2] = {4, 3};
  constexpr int32_t num_pse_i1[2] = {3, 4};
  const int32_t top               = num_pse_i0[v0 % 2];
  const int32_t bottom            = num_pse_i1[v1 % 2];

  if (v0 == v1 - 1) {
    // one sample case
    constexpr float K  = 1.2301741 / 2;  // 04914001;
    constexpr float K1 = 0.8128931;      // 066115961;
    for (uint32_t col = 0; col < u1 - u0; ++col) {
      if (v0 % 2) {
        in[col] <<= 1;
      }
    }
  } else {
    const uint32_t len = round_up(stride, SIMD_LEN_I32);
    auto **buf         = new sprec_t *[top + v1 - v0 + bottom];
    for (uint32_t i = 1; i <= top; ++i) {
      buf[top - i] = static_cast<sprec_t *>(aligned_mem_alloc(sizeof(sprec_t) * len, 32));
      memcpy(buf[top - i], &in[(PSEo(v0 - i, v0, v1) - v0) * stride], sizeof(sprec_t) * stride);
      // buf[top - i] = &in[(PSEo(v0 - i, v0, v1) - v0) * stride];
    }
    for (uint32_t row = 0; row < v1 - v0; ++row) {
      buf[top + row] = &in[row * stride];
    }
    for (uint32_t i = 1; i <= bottom; i++) {
      buf[top + (v1 - v0) + i - 1] = static_cast<sprec_t *>(aligned_mem_alloc(sizeof(sprec_t) * len, 32));
      memcpy(buf[top + (v1 - v0) + i - 1], &in[(PSEo(v1 - v0 + i - 1 + v0, v0, v1) - v0) * stride],
             sizeof(sprec_t) * stride);
    }
    const int32_t start  = ceil_int(v0, 2);
    const int32_t stop   = ceil_int(v1, 2);
    const int32_t offset = top + v0 % 2;

    const int32_t simdlen = (u1 - u0) - (u1 - u0) % 16;
    for (int32_t n = -4 + offset, i = start - 2; i < stop + 1; i++, n += 2) {
      fdwt_irrev97_fixed_avx2_ver_step(simdlen, buf[n], buf[n + 2], buf[n + 1], Acoeff, Aoffset, Ashift);
      for (uint32_t col = simdlen; col < u1 - u0; ++col) {
        int32_t sum = buf[n][col];
        sum += buf[n + 2][col];
        buf[n + 1][col] += (sprec_t)((Acoeff * sum + Aoffset) >> Ashift);
      }
    }
    for (int32_t n = -2 + offset, i = start - 1; i < stop + 1; i++, n += 2) {
      fdwt_irrev97_fixed_avx2_ver_step(simdlen, buf[n - 1], buf[n + 1], buf[n], Bcoeff, Boffset, Bshift);
      for (uint32_t col = simdlen; col < u1 - u0; ++col) {
        int32_t sum = buf[n - 1][col];
        sum += buf[n + 1][col];
        buf[n][col] += (sprec_t)((Bcoeff * sum + Boffset) >> Bshift);
      }
    }
    for (int32_t n = -2 + offset, i = start - 1; i < stop; i++, n += 2) {
      fdwt_irrev97_fixed_avx2_ver_step(simdlen, buf[n], buf[n + 2], buf[n + 1], Ccoeff, Coffset, Cshift);
      for (uint32_t col = simdlen; col < u1 - u0; ++col) {
        int32_t sum = buf[n][col];
        sum += buf[n + 2][col];
        buf[n + 1][col] += (sprec_t)((Ccoeff * sum + Coffset) >> Cshift);
      }
    }
    for (int32_t n = 0 + offset, i = start; i < stop; i++, n += 2) {
      fdwt_irrev97_fixed_avx2_ver_step(simdlen, buf[n - 1], buf[n + 1], buf[n], Dcoeff, Doffset, Dshift);
      for (uint32_t col = simdlen; col < u1 - u0; ++col) {
        int32_t sum = buf[n - 1][col];
        sum += buf[n + 1][col];
        buf[n][col] += (sprec_t)((Dcoeff * sum + Doffset) >> Dshift);
      }
    }

    for (uint32_t i = 1; i <= top; ++i) {
      aligned_mem_free(buf[top - i]);
    }
    for (uint32_t i = 1; i <= bottom; i++) {
      aligned_mem_free(buf[top + (v1 - v0) + i - 1]);
    }
    delete[] buf;
  }
}

void fdwt_rev_ver_sr_fixed_avx2(sprec_t *in, const uint32_t u0, const uint32_t u1, const uint32_t v0,
                                const uint32_t v1) {
  const uint32_t stride           = u1 - u0;
  constexpr int32_t num_pse_i0[2] = {2, 1};
  constexpr int32_t num_pse_i1[2] = {1, 2};
  const int32_t top               = num_pse_i0[v0 % 2];
  const int32_t bottom            = num_pse_i1[v1 % 2];

  if (v0 == v1 - 1) {
    // one sample case
    for (uint32_t col = 0; col < u1 - u0; ++col) {
      if (v0 % 2) {
        in[col] <<= 1;
      }
    }
  } else {
    const uint32_t len = round_up(stride, SIMD_PADDING);
    auto **buf         = new sprec_t *[top + v1 - v0 + bottom];
    for (uint32_t i = 1; i <= top; ++i) {
      buf[top - i] = static_cast<sprec_t *>(aligned_mem_alloc(sizeof(sprec_t) * len, 32));
      memcpy(buf[top - i], &in[(PSEo(v0 - i, v0, v1) - v0) * stride], sizeof(sprec_t) * stride);
      // buf[top - i] = &in[(PSEo(v0 - i, v0, v1) - v0) * stride];
    }
    for (uint32_t row = 0; row < v1 - v0; ++row) {
      buf[top + row] = &in[row * stride];
    }
    for (uint32_t i = 1; i <= bottom; i++) {
      buf[top + (v1 - v0) + i - 1] = static_cast<sprec_t *>(aligned_mem_alloc(sizeof(sprec_t) * len, 32));
      memcpy(buf[top + (v1 - v0) + i - 1], &in[(PSEo(v1 - v0 + i - 1 + v0, v0, v1) - v0) * stride],
             sizeof(sprec_t) * stride);
    }
    const int32_t start  = ceil_int(v0, 2);
    const int32_t stop   = ceil_int(v1, 2);
    const int32_t offset = top + v0 % 2;

    for (int32_t n = -2 + offset, i = start - 1; i < stop; ++i, n += 2) {
      for (uint32_t col = 0; col < u1 - u0; ++col) {
        int32_t sum = buf[n][col];
        sum += buf[n + 2][col];
        buf[n + 1][col] -= (sum >> 1);
      }
    }
    for (int32_t n = 0 + offset, i = start; i < stop; ++i, n += 2) {
      for (uint32_t col = 0; col < u1 - u0; ++col) {
        int32_t sum = buf[n - 1][col];
        sum += buf[n + 1][col];
        buf[n][col] += ((sum + 2) >> 2);
      }
    }

    for (uint32_t i = 1; i <= top; ++i) {
      aligned_mem_free(buf[top - i]);
    }
    for (uint32_t i = 1; i <= bottom; i++) {
      aligned_mem_free(buf[top + (v1 - v0) + i - 1]);
    }
    delete[] buf;
  }
}
#endif