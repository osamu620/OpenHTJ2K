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

#if defined(OPENHTJ2K_ENABLE_ARM_NEON)
  #include <arm_neon.h>
  #include "color.hpp"

// lossless: forward RCT
void cvt_rgb_to_ycbcr_rev_neon(int32_t *sp0, int32_t *sp1, int32_t *sp2, uint32_t width, uint32_t height) {
  int32_t R, G, B;
  int32_t Y, Cb, Cr;

  // process two vectors at a time
  for (uint32_t y = 0; y < height; ++y) {
    int32_t *p0 = sp0 + y * round_up(width, 32);
    int32_t *p1 = sp1 + y * round_up(width, 32);
    int32_t *p2 = sp2 + y * round_up(width, 32);
    int32_t len = static_cast<int32_t>(width);
    for (; len >= 8; len -= 8) {
      auto vR0  = vld1q_s32(p0);
      auto vR1  = vld1q_s32(p0 + 4);
      auto vG0  = vld1q_s32(p1);
      auto vG1  = vld1q_s32(p1 + 4);
      auto vB0  = vld1q_s32(p2);
      auto vB1  = vld1q_s32(p2 + 4);
      auto vY0  = (vR0 + 2 * vG0 + vB0) >> 2;
      auto vY1  = (vR1 + 2 * vG1 + vB1) >> 2;
      auto vCb0 = vB0 - vG0;
      auto vCb1 = vB1 - vG1;
      auto vCr0 = vR0 - vG0;
      auto vCr1 = vR1 - vG1;
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
    for (; len > 0; --len) {
      R     = *p0;
      G     = *p1;
      B     = *p2;
      Y     = (R + 2 * G + B) >> 2;
      Cb    = B - G;
      Cr    = R - G;
      *p0++ = Y;
      *p1++ = Cb;
      *p2++ = Cr;
    }
  }
}

// lossy: forward ICT
void cvt_rgb_to_ycbcr_irrev_neon(int32_t *sp0, int32_t *sp1, int32_t *sp2, uint32_t width,
                                 uint32_t height) {
  double fRed, fGrn, fBlu;
  double fY, fCb, fCr;
  int32x4_t R0, G0, B0, R1, G1, B1;
  float32x4_t Y0, Cb0, Cr0, fR0, fG0, fB0, Y1, Cb1, Cr1, fR1, fG1, fB1;
  const float32x4_t a0 = vdupq_n_f32(static_cast<float32_t>(ALPHA_R));
  const float32x4_t a1 = vdupq_n_f32(static_cast<float32_t>(ALPHA_G));
  const float32x4_t a2 = vdupq_n_f32(static_cast<float32_t>(ALPHA_B));
  const float32x4_t a3 = vdupq_n_f32(static_cast<float32_t>(1.0 / CB_FACT_B));
  const float32x4_t a4 = vdupq_n_f32(static_cast<float32_t>(1.0 / CR_FACT_R));
  // process two vectors at a time
  for (uint32_t y = 0; y < height; ++y) {
    int32_t *p0 = sp0 + y * round_up(width, 32);
    int32_t *p1 = sp1 + y * round_up(width, 32);
    int32_t *p2 = sp2 + y * round_up(width, 32);
    int32_t len = static_cast<int32_t>(width);
    for (; len >= 8; len -= 8) {
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
    for (; len > 0; --len) {
      fRed  = static_cast<double>(p0[0]);
      fGrn  = static_cast<double>(p1[0]);
      fBlu  = static_cast<double>(p2[0]);
      fY    = ALPHA_R * fRed + ALPHA_G * fGrn + ALPHA_B * fBlu;
      fCb   = (1.0 / CB_FACT_B) * (fBlu - fY);
      fCr   = (1.0 / CR_FACT_R) * (fRed - fY);
      p0[0] = round_d(fY);
      p1[0] = round_d(fCb);
      p2[0] = round_d(fCr);
      p0++;
      p1++;
      p2++;
    }
  }
}

// lossless: inverse RCT
void cvt_ycbcr_to_rgb_rev_neon(int32_t *sp0, int32_t *sp1, int32_t *sp2, uint32_t width, uint32_t height) {
  int32_t R, G, B;
  int32_t Y, Cb, Cr;
  int32x4_t vY0, vCb0, vCr0, vG0, vR0, vB0;
  int32x4_t vY1, vCb1, vCr1, vG1, vR1, vB1;

  // process two vectors at a time
  for (uint32_t y = 0; y < height; ++y) {
    int32_t *p0 = sp0 + y * round_up(width, 32);
    int32_t *p1 = sp1 + y * round_up(width, 32);
    int32_t *p2 = sp2 + y * round_up(width, 32);
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
      vY0  = vld1q_s32(sp0);
      vCb0 = vld1q_s32(sp1);
      vCr0 = vld1q_s32(sp2);
      vY1  = vld1q_s32(sp0 + 4);
      vCb1 = vld1q_s32(sp1 + 4);
      vCr1 = vld1q_s32(sp2 + 4);
    }
    for (; len > 0; --len) {
      Y     = *p0;
      Cb    = *p1;
      Cr    = *p2;
      G     = Y - ((Cb + Cr) >> 2);
      R     = Cr + G;
      B     = Cb + G;
      *p0++ = R;
      *p1++ = G;
      *p2++ = B;
    }
  }
}

// lossy: inverse ICT
void cvt_ycbcr_to_rgb_irrev_neon(int32_t *sp0, int32_t *sp1, int32_t *sp2, uint32_t width,
                                 uint32_t height) {
  int32_t R, G, B;
  double fY, fCb, fCr;
  const float32x4_t fCR_FACT_R = vdupq_n_f32(static_cast<float32_t>(CR_FACT_R));
  const float32x4_t fCB_FACT_B = vdupq_n_f32(static_cast<float32_t>(CB_FACT_B));
  const float32x4_t fCR_FACT_G = vdupq_n_f32(static_cast<float32_t>(CR_FACT_G));
  const float32x4_t fCB_FACT_G = vdupq_n_f32(static_cast<float32_t>(CB_FACT_G));
  for (uint32_t y = 0; y < height; ++y) {
    int32_t *p0 = sp0 + y * round_up(width, 32);
    int32_t *p1 = sp1 + y * round_up(width, 32);
    int32_t *p2 = sp2 + y * round_up(width, 32);
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
      fY    = static_cast<double>(*p0);
      fCb   = static_cast<double>(*p1);
      fCr   = static_cast<double>(*p2);
      R     = static_cast<int32_t>(round_d(fY + CR_FACT_R * fCr));
      B     = static_cast<int32_t>(round_d(fY + CB_FACT_B * fCb));
      G     = static_cast<int32_t>(round_d(fY - CR_FACT_G * fCr - CB_FACT_G * fCb));
      *p0++ = R;
      *p1++ = G;
      *p2++ = B;
    }
  }
}
#endif