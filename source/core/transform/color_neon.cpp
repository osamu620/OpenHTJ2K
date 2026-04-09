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

#if defined(OPENHTJ2K_ENABLE_ARM_NEON)
  #include <arm_neon.h>
  #include "color.hpp"

  #define neon_alphaR (static_cast<int32_t>(0.5 + ALPHA_R * (1 << 15)))
  #define neon_alphaB (static_cast<int32_t>(0.5 + ALPHA_B * (1 << 15)))
  #define neon_alphaG (static_cast<int32_t>(0.5 + ALPHA_G * (1 << 15)))
  #define neon_CBfact (static_cast<int32_t>(0.5 + CB_FACT * (1 << 15)))
  #define neon_CRfact (static_cast<int32_t>(0.5 + CR_FACT * (1 << 15)))
  #define neon_CRfactR (static_cast<int32_t>(0.5 + (CR_FACT_R - 1) * (1 << 15)))
  #define neon_CBfactB (static_cast<int32_t>(0.5 + (CB_FACT_B - 1) * (1 << 15)))
  #define neon_neg_CRfactG (static_cast<int32_t>(0.5 - CR_FACT_G * (1 << 15)))
  #define neon_neg_CBfactG (static_cast<int32_t>(0.5 - CB_FACT_G * (1 << 15)))

// lossless: forward RCT
void cvt_rgb_to_ycbcr_rev_neon(int32_t *sp0, int32_t *sp1, int32_t *sp2, uint32_t width, uint32_t height) {
  // process two vectors at a time
  for (uint32_t y = 0; y < height; ++y) {
    int32_t *p0 = sp0 + y * round_up(width, 32U);
    int32_t *p1 = sp1 + y * round_up(width, 32U);
    int32_t *p2 = sp2 + y * round_up(width, 32U);
    int32_t len = static_cast<int32_t>(width);
    for (; len > 0; len -= 8) {
      auto vR0 = vld1q_s32(p0);
      auto vR1 = vld1q_s32(p0 + 4);
      auto vG0 = vld1q_s32(p1);
      auto vG1 = vld1q_s32(p1 + 4);
      auto vB0 = vld1q_s32(p2);
      auto vB1 = vld1q_s32(p2 + 4);
      auto vY0 = vshrq_n_s32(vaddq_s32(vaddq_s32(vR0, vG0), vaddq_s32(vG0, vB0)), 2);
      //(vR0 + 2 * vG0 + vB0) >> 2;
      auto vY1 = vshrq_n_s32(vaddq_s32(vaddq_s32(vR1, vG1), vaddq_s32(vG1, vB1)), 2);
      //(vR1 + 2 * vG1 + vB1) >> 2;
      auto vCb0 = vsubq_s32(vB0, vG0);
      auto vCb1 = vsubq_s32(vB1, vG1);
      auto vCr0 = vsubq_s32(vR0, vG0);
      auto vCr1 = vsubq_s32(vR1, vG1);
      vst1q_s32(p0, vY0);
      vst1q_s32(p0 + 4, vY1);
      vst1q_s32(p1, vCb0);
      vst1q_s32(p1 + 4, vCb1);
      vst1q_s32(p2, vCr0);
      vst1q_s32(p2 + 4, vCr1);
      p0 += 8;
      p1 += 8;
      p2 += 8;
    }
  }
}

// lossy: forward ICT
void cvt_rgb_to_ycbcr_irrev_neon(int32_t *sp0, int32_t *sp1, int32_t *sp2, uint32_t width,
                                 uint32_t height) {
  int32x4_t R0, G0, B0, R1, G1, B1;
  float32x4_t Y0, Cb0, Cr0, fR0, fG0, fB0, Y1, Cb1, Cr1, fR1, fG1, fB1;
  const float32x4_t a0 = vdupq_n_f32(static_cast<float32_t>(ALPHA_R));
  const float32x4_t a1 = vdupq_n_f32(static_cast<float32_t>(ALPHA_G));
  const float32x4_t a2 = vdupq_n_f32(static_cast<float32_t>(ALPHA_B));
  const float32x4_t a3 = vdupq_n_f32(static_cast<float32_t>(1.0 / CB_FACT_B));
  const float32x4_t a4 = vdupq_n_f32(static_cast<float32_t>(1.0 / CR_FACT_R));
  // process two vectors at a time

  for (uint32_t y = 0; y < height; ++y) {
    int32_t *p0 = sp0 + y * round_up(width, 32U);
    int32_t *p1 = sp1 + y * round_up(width, 32U);
    int32_t *p2 = sp2 + y * round_up(width, 32U);
    int32_t len = static_cast<int32_t>(width);
    for (; len > 0; len -= 8) {
      R0  = vld1q_s32(p0);
      R1  = vld1q_s32(p0 + 4);
      G0  = vld1q_s32(p1);
      G1  = vld1q_s32(p1 + 4);
      B0  = vld1q_s32(p2);
      B1  = vld1q_s32(p2 + 4);
      fR0 = vcvtq_f32_s32(R0);
      fR1 = vcvtq_f32_s32(R1);
      fG0 = vcvtq_f32_s32(G0);
      fG1 = vcvtq_f32_s32(G1);
      fB0 = vcvtq_f32_s32(B0);
      fB1 = vcvtq_f32_s32(B1);
      Y0  = vmulq_f32(fR0, a0);
      Y0  = vfmaq_f32(Y0, fG0, a1);
      Y0  = vfmaq_f32(Y0, fB0, a2);
      Y1  = vmulq_f32(fR1, a0);
      Y1  = vfmaq_f32(Y1, fG1, a1);
      Y1  = vfmaq_f32(Y1, fB1, a2);
      // Y0  = fR0 * a0 + fG0 * a1 + fB0 * a2;
      Cb0 = vmulq_f32(vsubq_f32(fB0, Y0), a3);
      Cb1 = vmulq_f32(vsubq_f32(fB1, Y1), a3);
      Cr0 = vmulq_f32(vsubq_f32(fR0, Y0), a4);
      Cr1 = vmulq_f32(vsubq_f32(fR1, Y1), a4);

      // TODO: need to consider precision and setting FPSCR register value
      vst1q_s32(p0, vcvtnq_s32_f32(Y0));
      vst1q_s32(p0 + 4, vcvtnq_s32_f32(Y1));
      vst1q_s32(p1, vcvtnq_s32_f32(Cb0));
      vst1q_s32(p1 + 4, vcvtnq_s32_f32(Cb1));
      vst1q_s32(p2, vcvtnq_s32_f32(Cr0));
      vst1q_s32(p2 + 4, vcvtnq_s32_f32(Cr1));
      p0 += 8;
      p1 += 8;
      p2 += 8;
    }
  }
}

// lossless: inverse RCT
void cvt_ycbcr_to_rgb_rev_neon(int32_t *sp0, int32_t *sp1, int32_t *sp2, uint32_t width, uint32_t height) {
  int32x4_t vY0, vCb0, vCr0, vG0, vR0, vB0;
  int32x4_t vY1, vCb1, vCr1, vG1, vR1, vB1;

  // process two vectors at a time
  for (uint32_t y = 0; y < height; ++y) {
    int32_t *p0 = sp0 + y * round_up(width, 32U);
    int32_t *p1 = sp1 + y * round_up(width, 32U);
    int32_t *p2 = sp2 + y * round_up(width, 32U);
    int32_t len = static_cast<int32_t>(width);

    for (; len >= 8; len -= 8) {
      vY0  = vld1q_s32(p0);
      vCb0 = vld1q_s32(p1);
      vCr0 = vld1q_s32(p2);
      vY1  = vld1q_s32(p0 + 4);
      vCb1 = vld1q_s32(p1 + 4);
      vCr1 = vld1q_s32(p2 + 4);
      vG0  = vsubq_s32(vY0, vshrq_n_s32(vaddq_s32(vCb0, vCr0), 2));
      vG1  = vsubq_s32(vY1, vshrq_n_s32(vaddq_s32(vCb1, vCr1), 2));
      vR0  = vaddq_s32(vCr0, vG0);
      vR1  = vaddq_s32(vCr1, vG1);
      vB0  = vaddq_s32(vCb0, vG0);
      vB1  = vaddq_s32(vCb1, vG1);

      vst1q_s32(p0, vR0);
      vst1q_s32(p0 + 4, vR1);
      vst1q_s32(p1, vG0);
      vst1q_s32(p1 + 4, vG1);
      vst1q_s32(p2, vB0);
      vst1q_s32(p2 + 4, vB1);
      p0 += 8;
      p1 += 8;
      p2 += 8;
    }
    for (; len > 0; --len) {
      int32_t Y  = *p0;
      int32_t Cb = *p1;
      int32_t Cr = *p2;
      int32_t G  = Y - ((Cb + Cr) >> 2);
      *p0++      = Cr + G;
      *p1++      = G;
      *p2++      = Cb + G;
    }
  }
}

// lossy: inverse ICT
void cvt_ycbcr_to_rgb_irrev_neon(int32_t *sp0, int32_t *sp1, int32_t *sp2, uint32_t width,
                                 uint32_t height) {
  const float32x4_t fCR_FACT_R = vdupq_n_f32(static_cast<float32_t>(CR_FACT_R));
  const float32x4_t fCB_FACT_B = vdupq_n_f32(static_cast<float32_t>(CB_FACT_B));
  const float32x4_t fCR_FACT_G = vdupq_n_f32(static_cast<float32_t>(CR_FACT_G));
  const float32x4_t fCB_FACT_G = vdupq_n_f32(static_cast<float32_t>(CB_FACT_G));
  for (uint32_t y = 0; y < height; ++y) {
    int32_t *p0 = sp0 + y * round_up(width, 32U);
    int32_t *p1 = sp1 + y * round_up(width, 32U);
    int32_t *p2 = sp2 + y * round_up(width, 32U);
    int32_t len = static_cast<int32_t>(width);
    for (; len >= 8; len -= 8) {
      auto Y0  = vcvtq_f32_s32(vld1q_s32(p0));
      auto Y1  = vcvtq_f32_s32(vld1q_s32(p0 + 4));
      auto Cb0 = vcvtq_f32_s32(vld1q_s32(p1));
      auto Cb1 = vcvtq_f32_s32(vld1q_s32(p1 + 4));
      auto Cr0 = vcvtq_f32_s32(vld1q_s32(p2));
      auto Cr1 = vcvtq_f32_s32(vld1q_s32(p2 + 4));

      // TODO: need to consider precision and setting FPSCR register value
      vst1q_s32(p0, vcvtnq_s32_f32(vfmaq_f32(Y0, Cr0, fCR_FACT_R)));
      vst1q_s32(p0 + 4, vcvtnq_s32_f32(vfmaq_f32(Y1, Cr1, fCR_FACT_R)));
      vst1q_s32(p2, vcvtnq_s32_f32(vfmaq_f32(Y0, Cb0, fCB_FACT_B)));
      vst1q_s32(p2 + 4, vcvtnq_s32_f32(vfmaq_f32(Y1, Cb1, fCB_FACT_B)));
      Y0 = vfmsq_f32(Y0, Cr0, fCR_FACT_G);
      vst1q_s32(p1, vcvtnq_s32_f32(vfmsq_f32(Y0, Cb0, fCB_FACT_G)));
      Y1 = vfmsq_f32(Y1, Cr1, fCR_FACT_G);
      vst1q_s32(p1 + 4, vcvtnq_s32_f32(vfmsq_f32(Y1, Cb1, fCB_FACT_G)));
      p0 += 8;
      p1 += 8;
      p2 += 8;
    }
    for (; len > 0; --len) {
      float Y  = static_cast<float>(*p0);
      float Cb = static_cast<float>(*p1);
      float Cr = static_cast<float>(*p2);
      *p0++    = static_cast<int32_t>(std::roundf(Y + static_cast<float>(CR_FACT_R) * Cr));
      *p2++    = static_cast<int32_t>(std::roundf(Y + static_cast<float>(CB_FACT_B) * Cb));
      *p1++    = static_cast<int32_t>(
          std::roundf(Y - static_cast<float>(CR_FACT_G) * Cr - static_cast<float>(CB_FACT_G) * Cb));
    }
  }
}

// lossless: inverse RCT on float buffers
void cvt_ycbcr_to_rgb_rev_float_neon(float *sp0, float *sp1, float *sp2, uint32_t width, uint32_t height,
                                     uint32_t stride) {
  for (uint32_t y = 0; y < height; ++y) {
    float *p0   = sp0 + y * stride;
    float *p1   = sp1 + y * stride;
    float *p2   = sp2 + y * stride;
    int32_t len = static_cast<int32_t>(width);
    for (; len >= 8; len -= 8) {
      int32x4_t iY0  = vcvtq_s32_f32(vld1q_f32(p0));
      int32x4_t iCb0 = vcvtq_s32_f32(vld1q_f32(p1));
      int32x4_t iCr0 = vcvtq_s32_f32(vld1q_f32(p2));
      int32x4_t iY1  = vcvtq_s32_f32(vld1q_f32(p0 + 4));
      int32x4_t iCb1 = vcvtq_s32_f32(vld1q_f32(p1 + 4));
      int32x4_t iCr1 = vcvtq_s32_f32(vld1q_f32(p2 + 4));
      int32x4_t iG0  = vsubq_s32(iY0, vshrq_n_s32(vaddq_s32(iCb0, iCr0), 2));
      int32x4_t iG1  = vsubq_s32(iY1, vshrq_n_s32(vaddq_s32(iCb1, iCr1), 2));
      vst1q_f32(p0, vcvtq_f32_s32(vaddq_s32(iCr0, iG0)));
      vst1q_f32(p0 + 4, vcvtq_f32_s32(vaddq_s32(iCr1, iG1)));
      vst1q_f32(p1, vcvtq_f32_s32(iG0));
      vst1q_f32(p1 + 4, vcvtq_f32_s32(iG1));
      vst1q_f32(p2, vcvtq_f32_s32(vaddq_s32(iCb0, iG0)));
      vst1q_f32(p2 + 4, vcvtq_f32_s32(vaddq_s32(iCb1, iG1)));
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
void cvt_ycbcr_to_rgb_irrev_float_neon(float *sp0, float *sp1, float *sp2, uint32_t width, uint32_t height,
                                       uint32_t stride) {
  const float32x4_t fCR_FACT_R = vdupq_n_f32(static_cast<float>(CR_FACT_R));
  const float32x4_t fCB_FACT_B = vdupq_n_f32(static_cast<float>(CB_FACT_B));
  const float32x4_t fCR_FACT_G = vdupq_n_f32(static_cast<float>(CR_FACT_G));
  const float32x4_t fCB_FACT_G = vdupq_n_f32(static_cast<float>(CB_FACT_G));
  for (uint32_t y = 0; y < height; ++y) {
    float *p0   = sp0 + y * stride;
    float *p1   = sp1 + y * stride;
    float *p2   = sp2 + y * stride;
    int32_t len = static_cast<int32_t>(width);
    for (; len >= 8; len -= 8) {
      float32x4_t Y0  = vld1q_f32(p0);
      float32x4_t Y1  = vld1q_f32(p0 + 4);
      float32x4_t Cb0 = vld1q_f32(p1);
      float32x4_t Cb1 = vld1q_f32(p1 + 4);
      float32x4_t Cr0 = vld1q_f32(p2);
      float32x4_t Cr1 = vld1q_f32(p2 + 4);
      vst1q_f32(p0, vfmaq_f32(Y0, Cr0, fCR_FACT_R));
      vst1q_f32(p0 + 4, vfmaq_f32(Y1, Cr1, fCR_FACT_R));
      vst1q_f32(p2, vfmaq_f32(Y0, Cb0, fCB_FACT_B));
      vst1q_f32(p2 + 4, vfmaq_f32(Y1, Cb1, fCB_FACT_B));
      Y0 = vfmsq_f32(Y0, Cr0, fCR_FACT_G);
      vst1q_f32(p1, vfmsq_f32(Y0, Cb0, fCB_FACT_G));
      Y1 = vfmsq_f32(Y1, Cr1, fCR_FACT_G);
      vst1q_f32(p1 + 4, vfmsq_f32(Y1, Cb1, fCB_FACT_G));
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
void cvt_rgb_to_ycbcr_rev_float_neon(const int32_t *sp0, const int32_t *sp1, const int32_t *sp2,
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
      int32x4_t mR0 = vld1q_s32(p0),     mR1 = vld1q_s32(p0 + 4);
      int32x4_t mG0 = vld1q_s32(p1),     mG1 = vld1q_s32(p1 + 4);
      int32x4_t mB0 = vld1q_s32(p2),     mB1 = vld1q_s32(p2 + 4);
      int32x4_t mY0 = vshrq_n_s32(vaddq_s32(vaddq_s32(mR0, mB0), vaddq_s32(mG0, mG0)), 2);
      int32x4_t mY1 = vshrq_n_s32(vaddq_s32(vaddq_s32(mR1, mB1), vaddq_s32(mG1, mG1)), 2);
      vst1q_f32(d0,     vcvtq_f32_s32(mY0));
      vst1q_f32(d0 + 4, vcvtq_f32_s32(mY1));
      vst1q_f32(d1,     vcvtq_f32_s32(vsubq_s32(mB0, mG0)));
      vst1q_f32(d1 + 4, vcvtq_f32_s32(vsubq_s32(mB1, mG1)));
      vst1q_f32(d2,     vcvtq_f32_s32(vsubq_s32(mR0, mG0)));
      vst1q_f32(d2 + 4, vcvtq_f32_s32(vsubq_s32(mR1, mG1)));
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
void cvt_rgb_to_ycbcr_irrev_float_neon(const int32_t *sp0, const int32_t *sp1, const int32_t *sp2,
                                       float *dp0, float *dp1, float *dp2,
                                       uint32_t width, uint32_t height, uint32_t stride) {
  const float32x4_t fALPHA_R = vdupq_n_f32(static_cast<float>(ALPHA_R));
  const float32x4_t fALPHA_G = vdupq_n_f32(static_cast<float>(ALPHA_G));
  const float32x4_t fALPHA_B = vdupq_n_f32(static_cast<float>(ALPHA_B));
  const float32x4_t fCB_FACT = vdupq_n_f32(static_cast<float>(1.0 / CB_FACT_B));
  const float32x4_t fCR_FACT = vdupq_n_f32(static_cast<float>(1.0 / CR_FACT_R));
  for (uint32_t y = 0; y < height; ++y) {
    const int32_t *p0 = sp0 + y * stride;
    const int32_t *p1 = sp1 + y * stride;
    const int32_t *p2 = sp2 + y * stride;
    float *d0         = dp0 + y * stride;
    float *d1         = dp1 + y * stride;
    float *d2         = dp2 + y * stride;
    int32_t len       = static_cast<int32_t>(width);
    for (; len >= 8; len -= 8) {
      float32x4_t mR0 = vcvtq_f32_s32(vld1q_s32(p0));
      float32x4_t mR1 = vcvtq_f32_s32(vld1q_s32(p0 + 4));
      float32x4_t mG0 = vcvtq_f32_s32(vld1q_s32(p1));
      float32x4_t mG1 = vcvtq_f32_s32(vld1q_s32(p1 + 4));
      float32x4_t mB0 = vcvtq_f32_s32(vld1q_s32(p2));
      float32x4_t mB1 = vcvtq_f32_s32(vld1q_s32(p2 + 4));
      float32x4_t mY0 = vfmaq_f32(vfmaq_f32(vmulq_f32(mR0, fALPHA_R), mG0, fALPHA_G), mB0, fALPHA_B);
      float32x4_t mY1 = vfmaq_f32(vfmaq_f32(vmulq_f32(mR1, fALPHA_R), mG1, fALPHA_G), mB1, fALPHA_B);
      vst1q_f32(d0,     mY0);
      vst1q_f32(d0 + 4, mY1);
      vst1q_f32(d1,     vmulq_f32(fCB_FACT, vsubq_f32(mB0, mY0)));
      vst1q_f32(d1 + 4, vmulq_f32(fCB_FACT, vsubq_f32(mB1, mY1)));
      vst1q_f32(d2,     vmulq_f32(fCR_FACT, vsubq_f32(mR0, mY0)));
      vst1q_f32(d2 + 4, vmulq_f32(fCR_FACT, vsubq_f32(mR1, mY1)));
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

void fused_ycbcr_irrev_to_rgb_i32_neon(const float *y, const float *cb, const float *cr,
                                        int32_t *r, int32_t *g, int32_t *b, uint32_t width,
                                        const FinalizeParams *fp) {
  const float32x4_t mCR_FACT_R = vdupq_n_f32(static_cast<float>(CR_FACT_R));
  const float32x4_t mCR_FACT_G = vdupq_n_f32(static_cast<float>(CR_FACT_G));
  const float32x4_t mCB_FACT_B = vdupq_n_f32(static_cast<float>(CB_FACT_B));
  const float32x4_t mCB_FACT_G = vdupq_n_f32(static_cast<float>(CB_FACT_G));

  const int32x4_t vrnd0 = vdupq_n_s32(fp[0].rnd);
  const int32x4_t vdc0  = vdupq_n_s32(fp[0].dc);
  const int32x4_t vmx0  = vdupq_n_s32(fp[0].maxval);
  const int32x4_t vmn0  = vdupq_n_s32(fp[0].minval);
  const int32x4_t vrnd1 = vdupq_n_s32(fp[1].rnd);
  const int32x4_t vdc1  = vdupq_n_s32(fp[1].dc);
  const int32x4_t vmx1  = vdupq_n_s32(fp[1].maxval);
  const int32x4_t vmn1  = vdupq_n_s32(fp[1].minval);
  const int32x4_t vrnd2 = vdupq_n_s32(fp[2].rnd);
  const int32x4_t vdc2  = vdupq_n_s32(fp[2].dc);
  const int32x4_t vmx2  = vdupq_n_s32(fp[2].maxval);
  const int32x4_t vmn2  = vdupq_n_s32(fp[2].minval);

  uint32_t n = 0;
  if (fp[0].ds > 0 && fp[1].ds > 0 && fp[2].ds > 0) {
    const int32x4_t vs0 = vdupq_n_s32(-fp[0].ds);
    const int32x4_t vs1 = vdupq_n_s32(-fp[1].ds);
    const int32x4_t vs2 = vdupq_n_s32(-fp[2].ds);
    for (; n + 4 <= width; n += 4) {
      float32x4_t mY  = vld1q_f32(y + n);
      float32x4_t mCb = vld1q_f32(cb + n);
      float32x4_t mCr = vld1q_f32(cr + n);
      float32x4_t mR  = vmlaq_f32(mY, mCr, mCR_FACT_R);
      float32x4_t mB  = vmlaq_f32(mY, mCb, mCB_FACT_B);
      float32x4_t mG  = vmlsq_f32(mY, mCr, mCR_FACT_G);
      mG              = vmlsq_f32(mG, mCb, mCB_FACT_G);
      int32x4_t vR    = vshlq_s32(vaddq_s32(vcvtq_s32_f32(mR), vrnd0), vs0);
      int32x4_t vG    = vshlq_s32(vaddq_s32(vcvtq_s32_f32(mG), vrnd1), vs1);
      int32x4_t vB    = vshlq_s32(vaddq_s32(vcvtq_s32_f32(mB), vrnd2), vs2);
      vR = vmaxq_s32(vminq_s32(vaddq_s32(vR, vdc0), vmx0), vmn0);
      vG = vmaxq_s32(vminq_s32(vaddq_s32(vG, vdc1), vmx1), vmn1);
      vB = vmaxq_s32(vminq_s32(vaddq_s32(vB, vdc2), vmx2), vmn2);
      vst1q_s32(r + n, vR);
      vst1q_s32(g + n, vG);
      vst1q_s32(b + n, vB);
    }
  } else if (fp[0].ds == 0 && fp[1].ds == 0 && fp[2].ds == 0) {
    for (; n + 4 <= width; n += 4) {
      float32x4_t mY  = vld1q_f32(y + n);
      float32x4_t mCb = vld1q_f32(cb + n);
      float32x4_t mCr = vld1q_f32(cr + n);
      float32x4_t mR  = vmlaq_f32(mY, mCr, mCR_FACT_R);
      float32x4_t mB  = vmlaq_f32(mY, mCb, mCB_FACT_B);
      float32x4_t mG  = vmlsq_f32(mY, mCr, mCR_FACT_G);
      mG              = vmlsq_f32(mG, mCb, mCB_FACT_G);
      int32x4_t vR   = vmaxq_s32(vminq_s32(vaddq_s32(vcvtq_s32_f32(mR), vdc0), vmx0), vmn0);
      int32x4_t vG   = vmaxq_s32(vminq_s32(vaddq_s32(vcvtq_s32_f32(mG), vdc1), vmx1), vmn1);
      int32x4_t vB   = vmaxq_s32(vminq_s32(vaddq_s32(vcvtq_s32_f32(mB), vdc2), vmx2), vmn2);
      vst1q_s32(r + n, vR);
      vst1q_s32(g + n, vG);
      vst1q_s32(b + n, vB);
    }
  }
  auto finalize_one = [](float v, const FinalizeParams &p) -> int32_t {
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
    r[n] = finalize_one(Y + static_cast<float>(CR_FACT_R) * Cr, fp[0]);
    g[n] = finalize_one(
        Y - static_cast<float>(CR_FACT_G) * Cr - static_cast<float>(CB_FACT_G) * Cb, fp[1]);
    b[n] = finalize_one(Y + static_cast<float>(CB_FACT_B) * Cb, fp[2]);
  }
}

void fused_ycbcr_rev_to_rgb_i32_neon(const float *y, const float *cb, const float *cr,
                                      int32_t *r, int32_t *g, int32_t *b, uint32_t width,
                                      const FinalizeParams *fp) {
  const int32x4_t vdc0 = vdupq_n_s32(fp[0].dc);
  const int32x4_t vmx0 = vdupq_n_s32(fp[0].maxval);
  const int32x4_t vmn0 = vdupq_n_s32(fp[0].minval);
  const int32x4_t vdc1 = vdupq_n_s32(fp[1].dc);
  const int32x4_t vmx1 = vdupq_n_s32(fp[1].maxval);
  const int32x4_t vmn1 = vdupq_n_s32(fp[1].minval);
  const int32x4_t vdc2 = vdupq_n_s32(fp[2].dc);
  const int32x4_t vmx2 = vdupq_n_s32(fp[2].maxval);
  const int32x4_t vmn2 = vdupq_n_s32(fp[2].minval);

  uint32_t n = 0;
  for (; n + 4 <= width; n += 4) {
    int32x4_t iY  = vcvtq_s32_f32(vld1q_f32(y + n));
    int32x4_t iCb = vcvtq_s32_f32(vld1q_f32(cb + n));
    int32x4_t iCr = vcvtq_s32_f32(vld1q_f32(cr + n));
    int32x4_t iG  = vsubq_s32(iY, vshrq_n_s32(vaddq_s32(iCb, iCr), 2));
    int32x4_t iR  = vaddq_s32(iCr, iG);
    int32x4_t iB  = vaddq_s32(iCb, iG);
    vst1q_s32(r + n, vmaxq_s32(vminq_s32(vaddq_s32(iR, vdc0), vmx0), vmn0));
    vst1q_s32(g + n, vmaxq_s32(vminq_s32(vaddq_s32(iG, vdc1), vmx1), vmn1));
    vst1q_s32(b + n, vmaxq_s32(vminq_s32(vaddq_s32(iB, vdc2), vmx2), vmn2));
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