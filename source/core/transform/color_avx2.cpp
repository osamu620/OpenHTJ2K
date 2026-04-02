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
      __m256i iY  = _mm256_cvtps_epi32(_mm256_loadu_ps(p0));
      __m256i iCb = _mm256_cvtps_epi32(_mm256_loadu_ps(p1));
      __m256i iCr = _mm256_cvtps_epi32(_mm256_loadu_ps(p2));
      __m256i tmp = _mm256_srai_epi32(_mm256_add_epi32(iCb, iCr), 2);
      __m256i iG  = _mm256_sub_epi32(iY, tmp);
      _mm256_storeu_ps(p0, _mm256_cvtepi32_ps(_mm256_add_epi32(iCr, iG)));
      _mm256_storeu_ps(p1, _mm256_cvtepi32_ps(iG));
      _mm256_storeu_ps(p2, _mm256_cvtepi32_ps(_mm256_add_epi32(iCb, iG)));
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
      __m256 mY  = _mm256_loadu_ps(p0);
      __m256 mCb = _mm256_loadu_ps(p1);
      __m256 mCr = _mm256_loadu_ps(p2);
      __m256 mR  = _mm256_fmadd_ps(mCr, mCR_FACT_R, mY);
      __m256 mB  = _mm256_fmadd_ps(mCb, mCB_FACT_B, mY);
      __m256 mG  = _mm256_fnmadd_ps(mCr, mCR_FACT_G, mY);
      mG         = _mm256_fnmadd_ps(mCb, mCB_FACT_G, mG);
      _mm256_storeu_ps(p0, mR);
      _mm256_storeu_ps(p1, mG);
      _mm256_storeu_ps(p2, mB);
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

// lossless: fused int32→float + forward RCT
void cvt_rgb_to_ycbcr_rev_float_avx2(const int32_t *sp0, const int32_t *sp1, const int32_t *sp2,
                                     float *dp0, float *dp1, float *dp2,
                                     uint32_t width, uint32_t height, uint32_t stride) {
  for (uint32_t y = 0; y < height; ++y) {
    const int32_t *p0 = sp0 + y * stride;
    const int32_t *p1 = sp1 + y * stride;
    const int32_t *p2 = sp2 + y * stride;
    float *d0         = dp0 + y * stride;
    float *d1         = dp1 + y * stride;
    float *d2         = dp2 + y * stride;
    int32_t len       = static_cast<int32_t>(width);
    for (; len >= 8; len -= 8) {
      __m256i mR  = _mm256_loadu_si256((__m256i *)p0);
      __m256i mG  = _mm256_loadu_si256((__m256i *)p1);
      __m256i mB  = _mm256_loadu_si256((__m256i *)p2);
      __m256i mY  = _mm256_add_epi32(_mm256_add_epi32(mR, mB), _mm256_add_epi32(mG, mG));
      mY          = _mm256_srai_epi32(mY, 2);  // (R + 2G + B) >> 2
      _mm256_storeu_ps(d0, _mm256_cvtepi32_ps(mY));
      _mm256_storeu_ps(d1, _mm256_cvtepi32_ps(_mm256_sub_epi32(mB, mG)));
      _mm256_storeu_ps(d2, _mm256_cvtepi32_ps(_mm256_sub_epi32(mR, mG)));
      p0 += 8; p1 += 8; p2 += 8;
      d0 += 8; d1 += 8; d2 += 8;
    }
    for (; len > 0; --len) {
      int32_t R = *p0++, G = *p1++, B = *p2++;
      *d0++ = static_cast<float>((R + 2 * G + B) >> 2);
      *d1++ = static_cast<float>(B - G);
      *d2++ = static_cast<float>(R - G);
    }
  }
}

// lossy: fused int32→float + forward ICT
void cvt_rgb_to_ycbcr_irrev_float_avx2(const int32_t *sp0, const int32_t *sp1, const int32_t *sp2,
                                       float *dp0, float *dp1, float *dp2,
                                       uint32_t width, uint32_t height, uint32_t stride) {
  const __m256 mALPHA_R = _mm256_set1_ps(static_cast<float>(ALPHA_R));
  const __m256 mALPHA_G = _mm256_set1_ps(static_cast<float>(ALPHA_G));
  const __m256 mALPHA_B = _mm256_set1_ps(static_cast<float>(ALPHA_B));
  const __m256 mCB_FACT = _mm256_set1_ps(static_cast<float>(1.0 / CB_FACT_B));
  const __m256 mCR_FACT = _mm256_set1_ps(static_cast<float>(1.0 / CR_FACT_R));
  for (uint32_t y = 0; y < height; ++y) {
    const int32_t *p0 = sp0 + y * stride;
    const int32_t *p1 = sp1 + y * stride;
    const int32_t *p2 = sp2 + y * stride;
    float *d0         = dp0 + y * stride;
    float *d1         = dp1 + y * stride;
    float *d2         = dp2 + y * stride;
    int32_t len       = static_cast<int32_t>(width);
    for (; len >= 8; len -= 8) {
      __m256 mR  = _mm256_cvtepi32_ps(_mm256_loadu_si256((__m256i *)p0));
      __m256 mG  = _mm256_cvtepi32_ps(_mm256_loadu_si256((__m256i *)p1));
      __m256 mB  = _mm256_cvtepi32_ps(_mm256_loadu_si256((__m256i *)p2));
      __m256 mY  = _mm256_fmadd_ps(mR, mALPHA_R, _mm256_mul_ps(mG, mALPHA_G));
      mY         = _mm256_fmadd_ps(mB, mALPHA_B, mY);
      _mm256_storeu_ps(d0, mY);
      _mm256_storeu_ps(d1, _mm256_mul_ps(mCB_FACT, _mm256_sub_ps(mB, mY)));
      _mm256_storeu_ps(d2, _mm256_mul_ps(mCR_FACT, _mm256_sub_ps(mR, mY)));
      p0 += 8; p1 += 8; p2 += 8;
      d0 += 8; d1 += 8; d2 += 8;
    }
    for (; len > 0; --len) {
      float R = static_cast<float>(*p0++);
      float G = static_cast<float>(*p1++);
      float B = static_cast<float>(*p2++);
      float Y = static_cast<float>(ALPHA_R) * R + static_cast<float>(ALPHA_G) * G
                + static_cast<float>(ALPHA_B) * B;
      *d0++ = Y;
      *d1++ = static_cast<float>(1.0 / CB_FACT_B) * (B - Y);
      *d2++ = static_cast<float>(1.0 / CR_FACT_R) * (R - Y);
    }
  }
}

// Fused inverse ICT (lossy, float-domain) + float→int32 finalize.
// Reads Y/Cb/Cr from ring buffer (read-only), applies ICT coefficients, then applies per-component
// finalize (rounding right-shift + DC_OFFSET + clamp), writing R/G/B as int32.
// fp[3]: FinalizeParams for component 0 (R), 1 (G), 2 (B).
void fused_ycbcr_irrev_to_rgb_i32_avx2(const float *y, const float *cb, const float *cr,
                                        int32_t *r, int32_t *g, int32_t *b, uint32_t width,
                                        const FinalizeParams *fp) {
  const __m256 mCR_FACT_R = _mm256_set1_ps(static_cast<float>(CR_FACT_R));
  const __m256 mCR_FACT_G = _mm256_set1_ps(static_cast<float>(CR_FACT_G));
  const __m256 mCB_FACT_B = _mm256_set1_ps(static_cast<float>(CB_FACT_B));
  const __m256 mCB_FACT_G = _mm256_set1_ps(static_cast<float>(CB_FACT_G));

  const __m256i vrnd0 = _mm256_set1_epi32(fp[0].rnd);
  const __m256i vdc0  = _mm256_set1_epi32(fp[0].dc);
  const __m256i vmx0  = _mm256_set1_epi32(fp[0].maxval);
  const __m256i vmn0  = _mm256_set1_epi32(fp[0].minval);
  const __m256i vrnd1 = _mm256_set1_epi32(fp[1].rnd);
  const __m256i vdc1  = _mm256_set1_epi32(fp[1].dc);
  const __m256i vmx1  = _mm256_set1_epi32(fp[1].maxval);
  const __m256i vmn1  = _mm256_set1_epi32(fp[1].minval);
  const __m256i vrnd2 = _mm256_set1_epi32(fp[2].rnd);
  const __m256i vdc2  = _mm256_set1_epi32(fp[2].dc);
  const __m256i vmx2  = _mm256_set1_epi32(fp[2].maxval);
  const __m256i vmn2  = _mm256_set1_epi32(fp[2].minval);

  uint32_t n = 0;
  // Common case: all three components have a positive downshift (standard lossy decode).
  if (fp[0].ds > 0 && fp[1].ds > 0 && fp[2].ds > 0) {
    const __m128i vsh0 = _mm_cvtsi32_si128(fp[0].ds);
    const __m128i vsh1 = _mm_cvtsi32_si128(fp[1].ds);
    const __m128i vsh2 = _mm_cvtsi32_si128(fp[2].ds);
    for (; n + 8 <= width; n += 8) {
      __m256 mY  = _mm256_loadu_ps(y + n);
      __m256 mCb = _mm256_loadu_ps(cb + n);
      __m256 mCr = _mm256_loadu_ps(cr + n);
      __m256 mR  = _mm256_fmadd_ps(mCr, mCR_FACT_R, mY);
      __m256 mB  = _mm256_fmadd_ps(mCb, mCB_FACT_B, mY);
      __m256 mG  = _mm256_fnmadd_ps(mCr, mCR_FACT_G, mY);
      mG         = _mm256_fnmadd_ps(mCb, mCB_FACT_G, mG);
      __m256i vR = _mm256_sra_epi32(_mm256_add_epi32(_mm256_cvttps_epi32(mR), vrnd0), vsh0);
      __m256i vG = _mm256_sra_epi32(_mm256_add_epi32(_mm256_cvttps_epi32(mG), vrnd1), vsh1);
      __m256i vB = _mm256_sra_epi32(_mm256_add_epi32(_mm256_cvttps_epi32(mB), vrnd2), vsh2);
      vR = _mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(vR, vdc0), vmn0), vmx0);
      vG = _mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(vG, vdc1), vmn1), vmx1);
      vB = _mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(vB, vdc2), vmn2), vmx2);
      _mm256_storeu_si256((__m256i *)(r + n), vR);
      _mm256_storeu_si256((__m256i *)(g + n), vG);
      _mm256_storeu_si256((__m256i *)(b + n), vB);
    }
  } else if (fp[0].ds == 0 && fp[1].ds == 0 && fp[2].ds == 0) {
    // Lossless-like path: no shift (ds == 0 for all components).
    for (; n + 8 <= width; n += 8) {
      __m256 mY  = _mm256_loadu_ps(y + n);
      __m256 mCb = _mm256_loadu_ps(cb + n);
      __m256 mCr = _mm256_loadu_ps(cr + n);
      __m256 mR  = _mm256_fmadd_ps(mCr, mCR_FACT_R, mY);
      __m256 mB  = _mm256_fmadd_ps(mCb, mCB_FACT_B, mY);
      __m256 mG  = _mm256_fnmadd_ps(mCr, mCR_FACT_G, mY);
      mG         = _mm256_fnmadd_ps(mCb, mCB_FACT_G, mG);
      __m256i vR = _mm256_cvttps_epi32(mR);
      __m256i vG = _mm256_cvttps_epi32(mG);
      __m256i vB = _mm256_cvttps_epi32(mB);
      vR = _mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(vR, vdc0), vmn0), vmx0);
      vG = _mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(vG, vdc1), vmn1), vmx1);
      vB = _mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(vB, vdc2), vmn2), vmx2);
      _mm256_storeu_si256((__m256i *)(r + n), vR);
      _mm256_storeu_si256((__m256i *)(g + n), vG);
      _mm256_storeu_si256((__m256i *)(b + n), vB);
    }
  }
  // Scalar tail (also handles ds < 0 edge case for high-bitdepth images).
  auto finalize_scalar = [](float v, const FinalizeParams &p) -> int32_t {
    int32_t x = static_cast<int32_t>(v);
    if (p.ds > 0) x = (x + p.rnd) >> p.ds;
    else if (p.ds < 0) x <<= -p.ds;
    x += p.dc;
    if (x > p.maxval) x = p.maxval;
    if (x < p.minval) x = p.minval;
    return x;
  };
  for (; n < width; ++n) {
    float Y = y[n], Cb = cb[n], Cr = cr[n];
    r[n] = finalize_scalar(Y + static_cast<float>(CR_FACT_R) * Cr, fp[0]);
    g[n] = finalize_scalar(
        Y - static_cast<float>(CR_FACT_G) * Cr - static_cast<float>(CB_FACT_G) * Cb, fp[1]);
    b[n] = finalize_scalar(Y + static_cast<float>(CB_FACT_B) * Cb, fp[2]);
  }
}

// Fused inverse RCT (lossless) + int32 finalize.
// Reads Y/Cb/Cr from ring buffer (float, but integer-valued for 5/3 IDWT), applies RCT,
// then applies per-component DC_OFFSET + clamp. ds must be 0 for all components.
void fused_ycbcr_rev_to_rgb_i32_avx2(const float *y, const float *cb, const float *cr,
                                      int32_t *r, int32_t *g, int32_t *b, uint32_t width,
                                      const FinalizeParams *fp) {
  const __m256i vdc0 = _mm256_set1_epi32(fp[0].dc);
  const __m256i vmx0 = _mm256_set1_epi32(fp[0].maxval);
  const __m256i vmn0 = _mm256_set1_epi32(fp[0].minval);
  const __m256i vdc1 = _mm256_set1_epi32(fp[1].dc);
  const __m256i vmx1 = _mm256_set1_epi32(fp[1].maxval);
  const __m256i vmn1 = _mm256_set1_epi32(fp[1].minval);
  const __m256i vdc2 = _mm256_set1_epi32(fp[2].dc);
  const __m256i vmx2 = _mm256_set1_epi32(fp[2].maxval);
  const __m256i vmn2 = _mm256_set1_epi32(fp[2].minval);

  uint32_t n = 0;
  for (; n + 8 <= width; n += 8) {
    // After 5/3 IDWT, float values are integer-valued; use round-to-nearest (cvtps) to be safe.
    __m256i iY  = _mm256_cvtps_epi32(_mm256_loadu_ps(y + n));
    __m256i iCb = _mm256_cvtps_epi32(_mm256_loadu_ps(cb + n));
    __m256i iCr = _mm256_cvtps_epi32(_mm256_loadu_ps(cr + n));
    // RCT: G = Y - (Cb + Cr) >> 2; R = Cr + G; B = Cb + G
    __m256i iG  = _mm256_sub_epi32(iY, _mm256_srai_epi32(_mm256_add_epi32(iCb, iCr), 2));
    __m256i iR  = _mm256_add_epi32(iCr, iG);
    __m256i iB  = _mm256_add_epi32(iCb, iG);
    __m256i vR  = _mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(iR, vdc0), vmn0), vmx0);
    __m256i vG  = _mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(iG, vdc1), vmn1), vmx1);
    __m256i vB  = _mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(iB, vdc2), vmn2), vmx2);
    _mm256_storeu_si256((__m256i *)(r + n), vR);
    _mm256_storeu_si256((__m256i *)(g + n), vG);
    _mm256_storeu_si256((__m256i *)(b + n), vB);
  }
  for (; n < width; ++n) {
    int32_t Y  = static_cast<int32_t>(y[n]);
    int32_t Cb = static_cast<int32_t>(cb[n]);
    int32_t Cr = static_cast<int32_t>(cr[n]);
    int32_t G  = Y - ((Cb + Cr) >> 2);
    auto clamp = [](int32_t v, const FinalizeParams &p) -> int32_t {
      v += p.dc;
      if (v > p.maxval) v = p.maxval;
      if (v < p.minval) v = p.minval;
      return v;
    };
    r[n] = clamp(Cr + G, fp[0]);
    g[n] = clamp(G, fp[1]);
    b[n] = clamp(Cb + G, fp[2]);
  }
}
#endif