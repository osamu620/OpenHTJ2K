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

#if defined(OPENHTJ2K_TRY_AVX2) && defined(__AVX2__)
  #if defined(_MSC_VER)
    #include <intrin.h>
  #else
    #include <x86intrin.h>
  #endif
  #include "color.hpp"

// lossless: forward RCT
void cvt_rgb_to_ycbcr_rev_avx2(int32_t *sp0, int32_t *sp1, int32_t *sp2, uint32_t width, uint32_t height) {
  for (uint32_t y = 0; y < height; ++y) {
    int32_t *p0 = sp0 + y * round_up(width, 32U);
    int32_t *p1 = sp1 + y * round_up(width, 32U);
    int32_t *p2 = sp2 + y * round_up(width, 32U);
    int32_t len = static_cast<int32_t>(width);
    for (; len > 0; len -= 8) {
      __m256i mR       = *((__m256i *)p0);
      __m256i mG       = *((__m256i *)p1);
      __m256i mB       = *((__m256i *)p2);
      __m256i mY       = _mm256_add_epi32(mR, mB);
      mY               = _mm256_add_epi32(mG, mY);
      mY               = _mm256_add_epi32(mG, mY);  // Y = R + 2 * G + B;
      *((__m256i *)p1) = _mm256_sub_epi32(mB, mG);
      *((__m256i *)p2) = _mm256_sub_epi32(mR, mG);
      *((__m256i *)p0) = _mm256_srai_epi32(mY, 2);  // Y = (R + 2 * G + B) >> 2;
      p0 += 8;
      p1 += 8;
      p2 += 8;
    }
  }
}

// lossy: forward ICT
void cvt_rgb_to_ycbcr_irrev_avx2(int32_t *sp0, int32_t *sp1, int32_t *sp2, uint32_t width,
                                 uint32_t height) {
  const __m256 mALPHA_R = _mm256_set1_ps(static_cast<float>(ALPHA_R));
  const __m256 mALPHA_G = _mm256_set1_ps(static_cast<float>(ALPHA_G));
  const __m256 mALPHA_B = _mm256_set1_ps(static_cast<float>(ALPHA_B));
  const __m256 mCB_FACT = _mm256_set1_ps(static_cast<float>(1.0 / CB_FACT_B));
  const __m256 mCR_FACT = _mm256_set1_ps(static_cast<float>(1.0 / CR_FACT_R));
  for (uint32_t y = 0; y < height; ++y) {
    int32_t *p0 = sp0 + y * round_up(width, 32U);
    int32_t *p1 = sp1 + y * round_up(width, 32U);
    int32_t *p2 = sp2 + y * round_up(width, 32U);
    int32_t len = static_cast<int32_t>(width);
    for (; len > 0; len -= 8) {
      __m256 mR        = _mm256_cvtepi32_ps(*((__m256i *)p0));
      __m256 mG        = _mm256_cvtepi32_ps(*((__m256i *)p1));
      __m256 mB        = _mm256_cvtepi32_ps(*((__m256i *)p2));
      __m256 mY        = _mm256_mul_ps(mG, mALPHA_G);
      mY               = _mm256_fmadd_ps(mR, mALPHA_R, mY);
      mY               = _mm256_fmadd_ps(mB, mALPHA_B, mY);
      __m256 mCb       = _mm256_mul_ps(mCB_FACT, _mm256_sub_ps(mB, mY));
      __m256 mCr       = _mm256_mul_ps(mCR_FACT, _mm256_sub_ps(mR, mY));
      *((__m256i *)p0) = _mm256_cvtps_epi32(_mm256_round_ps(mY, 0));
      *((__m256i *)p1) = _mm256_cvtps_epi32(_mm256_round_ps(mCb, 0));
      *((__m256i *)p2) = _mm256_cvtps_epi32(_mm256_round_ps(mCr, 0));
      p0 += 8;
      p1 += 8;
      p2 += 8;
    }
  }
}

// lossless: inverse RCT
void cvt_ycbcr_to_rgb_rev_avx2(int32_t *sp0, int32_t *sp1, int32_t *sp2, uint32_t width, uint32_t height) {
  for (uint32_t y = 0; y < height; ++y) {
    int32_t *p0 = sp0 + y * round_up(width, 32U);
    int32_t *p1 = sp1 + y * round_up(width, 32U);
    int32_t *p2 = sp2 + y * round_up(width, 32U);
    int32_t len = static_cast<int32_t>(width);
    for (; len > 0; len -= 8) {
      __m256i mCb = *((__m256i *)p1);
      __m256i mCr = *((__m256i *)p2);
      __m256i mY  = *((__m256i *)p0);
      __m256i tmp = _mm256_add_epi32(mCb, mCr);
      tmp         = _mm256_srai_epi32(tmp, 2);  //(Cb + Cr) >> 2
      __m256i mG  = _mm256_sub_epi32(mY, tmp);
      // *((__m256i *)p1) = mG;
      // *((__m256i *)p0) = _mm256_add_epi32(mCr, mG);
      // *((__m256i *)p2) = _mm256_add_epi32(mCb, mG);
      _mm256_stream_si256((__m256i *)p1, mG);
      _mm256_stream_si256((__m256i *)p0, _mm256_add_epi32(mCr, mG));
      _mm256_stream_si256((__m256i *)p2, _mm256_add_epi32(mCb, mG));
      p0 += 8;
      p1 += 8;
      p2 += 8;
    }
  }
}

// lossy: inverse ICT
void cvt_ycbcr_to_rgb_irrev_avx2(int32_t *sp0, int32_t *sp1, int32_t *sp2, uint32_t width,
                                 uint32_t height) {
  __m256 mCR_FACT_R = _mm256_set1_ps(static_cast<float>(CR_FACT_R));
  __m256 mCR_FACT_G = _mm256_set1_ps(static_cast<float>(CR_FACT_G));
  __m256 mCB_FACT_B = _mm256_set1_ps(static_cast<float>(CB_FACT_B));
  __m256 mCB_FACT_G = _mm256_set1_ps(static_cast<float>(CB_FACT_G));
  for (uint32_t y = 0; y < height; ++y) {
    int32_t *p0 = sp0 + y * round_up(width, 32U);
    int32_t *p1 = sp1 + y * round_up(width, 32U);
    int32_t *p2 = sp2 + y * round_up(width, 32U);
    int32_t len = static_cast<int32_t>(width);
    for (; len > 0; len -= 8) {
      __m256 mY  = _mm256_cvtepi32_ps(*((__m256i *)p0));
      __m256 mCb = _mm256_cvtepi32_ps(*((__m256i *)p1));
      __m256 mCr = _mm256_cvtepi32_ps(*((__m256i *)p2));
      __m256 mR  = _mm256_fmadd_ps(mCr, mCR_FACT_R, mY);
      __m256 mB  = _mm256_fmadd_ps(mCb, mCB_FACT_B, mY);
      __m256 mG  = _mm256_fnmadd_ps(mCr, mCR_FACT_G, mY);
      mG         = _mm256_fnmadd_ps(mCb, mCB_FACT_G, mG);
      // *((__m256i *)p0) = _mm256_cvtps_epi32(_mm256_round_ps(mR, 0));
      // *((__m256i *)p1) = _mm256_cvtps_epi32(_mm256_round_ps(mG, 0));
      // *((__m256i *)p2) = _mm256_cvtps_epi32(_mm256_round_ps(mB, 0));
      _mm256_stream_si256((__m256i *)p0, _mm256_cvtps_epi32(_mm256_round_ps(mR, 0)));
      _mm256_stream_si256((__m256i *)p1, _mm256_cvtps_epi32(_mm256_round_ps(mG, 0)));
      _mm256_stream_si256((__m256i *)p2, _mm256_cvtps_epi32(_mm256_round_ps(mB, 0)));
      p0 += 8;
      p1 += 8;
      p2 += 8;
    }
  }
}

// lossless: inverse RCT on float buffers (values are exact integers stored as float)
void cvt_ycbcr_to_rgb_rev_float_avx2(float *sp0, float *sp1, float *sp2, uint32_t width, uint32_t height,
                                     uint32_t stride) {
  for (uint32_t y = 0; y < height; ++y) {
    float *p0   = sp0 + y * stride;
    float *p1   = sp1 + y * stride;
    float *p2   = sp2 + y * stride;
    int32_t len = static_cast<int32_t>(width);
    for (; len >= 8; len -= 8) {
      __m256i iY  = _mm256_cvtps_epi32(_mm256_load_ps(p0));
      __m256i iCb = _mm256_cvtps_epi32(_mm256_load_ps(p1));
      __m256i iCr = _mm256_cvtps_epi32(_mm256_load_ps(p2));
      __m256i tmp = _mm256_srai_epi32(_mm256_add_epi32(iCb, iCr), 2);
      __m256i iG  = _mm256_sub_epi32(iY, tmp);
      _mm256_store_ps(p0, _mm256_cvtepi32_ps(_mm256_add_epi32(iCr, iG)));
      _mm256_store_ps(p1, _mm256_cvtepi32_ps(iG));
      _mm256_store_ps(p2, _mm256_cvtepi32_ps(_mm256_add_epi32(iCb, iG)));
      p0 += 8;
      p1 += 8;
      p2 += 8;
    }
    for (; len > 0; --len) {
      int32_t Y  = static_cast<int32_t>(*p0);
      int32_t Cb = static_cast<int32_t>(*p1);
      int32_t Cr = static_cast<int32_t>(*p2);
      int32_t G  = Y - ((Cb + Cr) >> 2);
      *p0++      = static_cast<float>(Cr + G);
      *p1++      = static_cast<float>(G);
      *p2++      = static_cast<float>(Cb + G);
    }
  }
}

// lossy: inverse ICT on float buffers
void cvt_ycbcr_to_rgb_irrev_float_avx2(float *sp0, float *sp1, float *sp2, uint32_t width, uint32_t height,
                                       uint32_t stride) {
  const __m256 mCR_FACT_R = _mm256_set1_ps(static_cast<float>(CR_FACT_R));
  const __m256 mCR_FACT_G = _mm256_set1_ps(static_cast<float>(CR_FACT_G));
  const __m256 mCB_FACT_B = _mm256_set1_ps(static_cast<float>(CB_FACT_B));
  const __m256 mCB_FACT_G = _mm256_set1_ps(static_cast<float>(CB_FACT_G));
  for (uint32_t y = 0; y < height; ++y) {
    float *p0   = sp0 + y * stride;
    float *p1   = sp1 + y * stride;
    float *p2   = sp2 + y * stride;
    int32_t len = static_cast<int32_t>(width);
    for (; len >= 8; len -= 8) {
      __m256 mY  = _mm256_load_ps(p0);
      __m256 mCb = _mm256_load_ps(p1);
      __m256 mCr = _mm256_load_ps(p2);
      __m256 mR  = _mm256_fmadd_ps(mCr, mCR_FACT_R, mY);
      __m256 mB  = _mm256_fmadd_ps(mCb, mCB_FACT_B, mY);
      __m256 mG  = _mm256_fnmadd_ps(mCr, mCR_FACT_G, mY);
      mG         = _mm256_fnmadd_ps(mCb, mCB_FACT_G, mG);
      _mm256_store_ps(p0, mR);
      _mm256_store_ps(p1, mG);
      _mm256_store_ps(p2, mB);
      p0 += 8;
      p1 += 8;
      p2 += 8;
    }
    for (; len > 0; --len) {
      float Y  = *p0;
      float Cb = *p1;
      float Cr = *p2;
      *p0++    = Y + static_cast<float>(CR_FACT_R) * Cr;
      *p1++    = Y - static_cast<float>(CR_FACT_G) * Cr - static_cast<float>(CB_FACT_G) * Cb;
      *p2++    = Y + static_cast<float>(CB_FACT_B) * Cb;
    }
  }
}
#endif