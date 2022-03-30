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
#if defined(OPENHTJ2K_ENABLE_AVX2)
  #include "dwt.hpp"

// irreversible FDWT
void fdwt_1d_filtr_irrev97_fixed_avx2(sprec_t *X, const int32_t left, const int32_t right,
                                      const uint32_t u_i0, const uint32_t u_i1) {
  const auto i0       = static_cast<const int32_t>(u_i0);
  const auto i1       = static_cast<const int32_t>(u_i1);
  const int32_t start = ceil_int(i0, 2);
  const int32_t stop  = ceil_int(i1, 2);

  const int32_t offset = left + i0 % 2;

  // step 1
  int32_t simdlen = stop + 1 - (start - 2);
  for (int32_t n = -4 + offset, i = 0; i < simdlen; i += 8, n += 16) {
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
    auto xeven1  = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(xin22, 0));
    auto vsum    = _mm256_add_epi32(xeven0, xeven1);
    auto vcoeff  = _mm256_set1_epi32(Acoeff);
    auto voffset = _mm256_set1_epi32(Aoffset);
    auto xout    = _mm256_add_epi32(
           _mm256_srai_epi32(_mm256_add_epi32(_mm256_mullo_epi32(vsum, vcoeff), voffset), Ashift), xodd0);
    auto xout_even_odd = _mm256_shuffle_epi32(_mm256_packs_epi32(xeven0, xout), 0b11011000);
    auto xout_interleaved =
        _mm256_shufflelo_epi16(_mm256_shufflehi_epi16(xout_even_odd, 0b11011000), 0b11011000);
    _mm256_storeu_si256((__m256i *)(X + n), xout_interleaved);
  }

  // step 2
  simdlen = stop + 1 - (start - 1);
  for (int32_t n = -2 + offset, i = 0; i < simdlen; i += 8, n += 16) {
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
    auto xodd1   = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(xin22, 0));
    auto vsum    = _mm256_add_epi32(xodd0, xodd1);
    auto vcoeff  = _mm256_set1_epi32(Bcoeff);
    auto voffset = _mm256_set1_epi32(Boffset);
    auto xout    = _mm256_add_epi32(
           _mm256_srai_epi32(_mm256_add_epi32(_mm256_mullo_epi32(vsum, vcoeff), voffset), Bshift), xeven0);
    auto xout_odd_even = _mm256_shuffle_epi32(_mm256_packs_epi32(xodd0, xout), 0b11011000);
    auto xout_interleaved =
        _mm256_shufflelo_epi16(_mm256_shufflehi_epi16(xout_odd_even, 0b11011000), 0b11011000);
    _mm256_storeu_si256((__m256i *)(X + n - 1), xout_interleaved);
  }

  // step 3
  simdlen = stop - (start - 1);
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
    auto xeven1  = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(xin22, 0));
    auto vsum    = _mm256_add_epi32(xeven0, xeven1);
    auto vcoeff  = _mm256_set1_epi32(Ccoeff);
    auto voffset = _mm256_set1_epi32(Coffset);
    auto xout    = _mm256_add_epi32(
           _mm256_srai_epi32(_mm256_add_epi32(_mm256_mullo_epi32(vsum, vcoeff), voffset), Cshift), xodd0);
    auto xout_even_odd = _mm256_shuffle_epi32(_mm256_packs_epi32(xeven0, xout), 0b11011000);
    auto xout_interleaved =
        _mm256_shufflelo_epi16(_mm256_shufflehi_epi16(xout_even_odd, 0b11011000), 0b11011000);
    _mm256_storeu_si256((__m256i *)(X + n), xout_interleaved);
  }

  // step 4
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
    auto xodd1   = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(xin22, 0));
    auto vsum    = _mm256_add_epi32(xodd0, xodd1);
    auto vcoeff  = _mm256_set1_epi32(Dcoeff);
    auto voffset = _mm256_set1_epi32(Doffset);
    auto xout    = _mm256_add_epi32(
           _mm256_srai_epi32(_mm256_add_epi32(_mm256_mullo_epi32(vsum, vcoeff), voffset), Dshift), xeven0);
    auto xout_odd_even = _mm256_shuffle_epi32(_mm256_packs_epi32(xodd0, xout), 0b11011000);
    auto xout_interleaved =
        _mm256_shufflelo_epi16(_mm256_shufflehi_epi16(xout_odd_even, 0b11011000), 0b11011000);
    _mm256_storeu_si256((__m256i *)(X + n - 1), xout_interleaved);
  }
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
#endif