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
#if defined(OPENHTJ2K_ENABLE_ARM_NEON)
  #include "color.hpp"

void cvt_rgb_to_ycbcr_rev_neon(int32_t *sp0, int32_t *sp1, int32_t *sp2, uint32_t num_tc_samples) {
  int32_t R, G, B;
  int32_t Y, Cb, Cr;
  for (; num_tc_samples > 4; num_tc_samples -= 4) {
    auto vR  = vld1q_s32(sp0);
    auto vG  = vld1q_s32(sp1);
    auto vB  = vld1q_s32(sp2);
    auto vY  = (vR + 2 * vG + vB) >> 2;
    auto vCb = vB - vG;
    auto vCr = vR - vG;
    vst1q_s32(sp0, vY);
    vst1q_s32(sp1, vCb);
    vst1q_s32(sp2, vCr);
    sp0 += 4;
    sp1 += 4;
    sp2 += 4;
  }
  for (; num_tc_samples > 0; --num_tc_samples) {
    R      = *sp0;
    G      = *sp1;
    B      = *sp2;
    Y      = (R + 2 * G + B) >> 2;
    Cb     = B - G;
    Cr     = R - G;
    *sp0++ = Y;
    *sp1++ = Cb;
    *sp2++ = Cr;
  }
}

void cvt_rgb_to_ycbcr_irrev_neon(int32_t *sp0, int32_t *sp1, int32_t *sp2, uint32_t num_tc_samples) {
  double fRed, fGrn, fBlu;
  double fY, fCb, fCr;
  int32x4_t R, G, B;
  float32x4_t Y, Cb, Cr, fR, fG, fB;
  float32x4_t a0 = vdupq_n_f32(static_cast<float32_t>(ALPHA_R));
  float32x4_t a1 = vdupq_n_f32(static_cast<float32_t>(ALPHA_G));
  float32x4_t a2 = vdupq_n_f32(static_cast<float32_t>(ALPHA_B));
  for (; num_tc_samples >= 4; num_tc_samples -= 4) {
    R  = vld1q_s32(sp0);
    G  = vld1q_s32(sp1);
    B  = vld1q_s32(sp2);
    fR = vcvtq_f32_s32(R);
    fG = vcvtq_f32_s32(G);
    fB = vcvtq_f32_s32(B);
    Y  = fR * a0;
    Y  = vmlaq_f32(Y, fG, a1);
    Y  = vmlaq_f32(Y, fB, a2);
    // Y  = fR * a0 + fG * a1 + fB * a2;
    Cb = vmulq_n_f32(vsubq_f32(fB, Y), static_cast<float32_t>(1.0 / CB_FACT_B));
    Cr = vmulq_n_f32(vsubq_f32(fR, Y), static_cast<float32_t>(1.0 / CR_FACT_R));

    // TODO: need to consider precision and setting FPSCR register value
    vst1q_s32(sp0, vcvtnq_s32_f32(Y));
    vst1q_s32(sp1, vcvtnq_s32_f32(Cb));
    vst1q_s32(sp2, vcvtnq_s32_f32(Cr));
    sp0 += 4;
    sp1 += 4;
    sp2 += 4;
  }

  for (; num_tc_samples > 0; --num_tc_samples) {
    fRed   = static_cast<double>(sp0[0]);
    fGrn   = static_cast<double>(sp1[0]);
    fBlu   = static_cast<double>(sp2[0]);
    fY     = ALPHA_R * fRed + ALPHA_G * fGrn + ALPHA_B * fBlu;
    fCb    = (1.0 / CB_FACT_B) * (fBlu - fY);
    fCr    = (1.0 / CR_FACT_R) * (fRed - fY);
    sp0[0] = round_d(fY);
    sp1[0] = round_d(fCb);
    sp2[0] = round_d(fCr);
    sp0++;
    sp1++;
    sp2++;
  }
}

void cvt_ycbcr_to_rgb_rev_neon(int32_t *sp0, int32_t *sp1, int32_t *sp2, uint32_t num_tc_samples) {
  int32_t R, G, B;
  int32_t Y, Cb, Cr;
  for (; num_tc_samples >= 4; num_tc_samples -= 4) {
    auto vY  = vld1q_s32(sp0);
    auto vCb = vld1q_s32(sp1);
    auto vCr = vld1q_s32(sp2);
    //    auto vG  = vY - ((vCb + vCr) >> 2);
    auto vG = vsubq_s32(vY, (vhaddq_s32(vCb, vCr) >> 1));
    auto vR = vaddq_s32(vCr, vG);
    auto vB = vaddq_s32(vCb, vG);
    vst1q_s32(sp0, vR);
    vst1q_s32(sp1, vG);
    vst1q_s32(sp2, vB);
    sp0 += 4;
    sp1 += 4;
    sp2 += 4;
  }
  for (; num_tc_samples > 0; --num_tc_samples) {
    Y      = *sp0;
    Cb     = *sp1;
    Cr     = *sp2;
    G      = Y - ((Cb + Cr) >> 2);
    R      = Cr + G;
    B      = Cb + G;
    *sp0++ = R;
    *sp1++ = G;
    *sp2++ = B;
  }
}

void cvt_ycbcr_to_rgb_irrev_neon(int32_t *sp0, int32_t *sp1, int32_t *sp2, uint32_t num_tc_samples) {
  int32_t R, G, B;
  double fY, fCb, fCr;
  for (; num_tc_samples >= 4; num_tc_samples -= 4) {
    auto vY  = vcvtq_f32_s32(vld1q_s32(sp0));
    auto vCb = vcvtq_f32_s32(vld1q_s32(sp1));
    auto vCr = vcvtq_f32_s32(vld1q_s32(sp2));

    // TODO: need to consider precision and setting FPSCR register value
    vst1q_s32(sp0, vcvtnq_s32_f32(vaddq_f32(vY, vmulq_n_f32(vCr, static_cast<float32_t>(CR_FACT_R)))));
    vst1q_s32(sp2, vcvtnq_s32_f32(vaddq_f32(vY, vmulq_n_f32(vCb, static_cast<float32_t>(CB_FACT_B)))));
    vst1q_s32(
        sp1, vcvtnq_s32_f32(vsubq_f32(vY, vaddq_f32(vmulq_n_f32(vCr, static_cast<float32_t>(CR_FACT_G)),
                                                    vmulq_n_f32(vCb, static_cast<float32_t>(CB_FACT_G))))));
    sp0 += 4;
    sp1 += 4;
    sp2 += 4;
  }
  for (; num_tc_samples > 0; --num_tc_samples) {
    fY     = static_cast<double>(*sp0);
    fCb    = static_cast<double>(*sp1);
    fCr    = static_cast<double>(*sp2);
    R      = static_cast<int32_t>(round_d(fY + CR_FACT_R * fCr));
    B      = static_cast<int32_t>(round_d(fY + CB_FACT_B * fCb));
    G      = static_cast<int32_t>(round_d(fY - CR_FACT_G * fCr - CB_FACT_G * fCb));
    *sp0++ = R;
    *sp1++ = G;
    *sp2++ = B;
  }
}
#endif