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
  for (uint32_t n = 0; n < num_tc_samples - num_tc_samples % 4; n += 4) {
    auto vR  = vld1q_s32(sp0 + n);
    auto vG  = vld1q_s32(sp1 + n);
    auto vB  = vld1q_s32(sp2 + n);
    auto vY  = (vR + 2 * vG + vB) >> 2;
    auto vCb = vB - vG;
    auto vCr = vR - vG;
    vst1q_s32(sp0 + n, vY);
    vst1q_s32(sp1 + n, vCb);
    vst1q_s32(sp2 + n, vCr);
  }
  for (uint32_t n = num_tc_samples - num_tc_samples % 4; n < num_tc_samples; ++n) {
    R      = sp0[n];
    G      = sp1[n];
    B      = sp2[n];
    Y      = (R + 2 * G + B) >> 2;
    Cb     = B - G;
    Cr     = R - G;
    sp0[n] = Y;
    sp1[n] = Cb;
    sp2[n] = Cr;
  }
}

void cvt_rgb_to_ycbcr_irrev_neon(int32_t *sp0, int32_t *sp1, int32_t *sp2, uint32_t num_tc_samples) {
  double fR, fG, fB;
  double fY, fCb, fCr;

  for (uint32_t n = 0; n < num_tc_samples - num_tc_samples % 4; n += 4) {
    int32x4_t s32x4_R   = vld1q_s32(sp0 + n);
    int32x4_t s32x4_G   = vld1q_s32(sp1 + n);
    int32x4_t s32x4_B   = vld1q_s32(sp2 + n);
    float32x4_t f32x4_R = vcvtq_f32_s32(s32x4_R);
    float32x4_t f32x4_G = vcvtq_f32_s32(s32x4_G);
    float32x4_t f32x4_B = vcvtq_f32_s32(s32x4_B);

    float32x4_t a0 = vdupq_n_f32(ALPHA_R);
    float32x4_t a1 = vdupq_n_f32(ALPHA_G);
    float32x4_t a2 = vdupq_n_f32(ALPHA_B);

    float32x4_t f32x4_Y  = f32x4_R * a0 + f32x4_G * a1 + f32x4_B * a2;
    float32x4_t f32x4_Cb = vmulq_n_f32(vsubq_f32(f32x4_B, f32x4_Y), 1.0 / CB_FACT_B);
    float32x4_t f32x4_Cr = vmulq_n_f32(vsubq_f32(f32x4_R, f32x4_Y), 1.0 / CR_FACT_R);

    // TODO: need to consider precision and setting FPSCR register value
    vst1q_s32(sp0 + n, vcvtnq_s32_f32(f32x4_Y));
    vst1q_s32(sp1 + n, vcvtnq_s32_f32(f32x4_Cb));
    vst1q_s32(sp2 + n, vcvtnq_s32_f32(f32x4_Cr));
  }
  for (uint32_t n = num_tc_samples - num_tc_samples % 4; n < num_tc_samples; n++) {
    fR     = static_cast<double>(sp0[n]);
    fG     = static_cast<double>(sp1[n]);
    fB     = static_cast<double>(sp2[n]);
    fY     = ALPHA_R * fR + ALPHA_G * fG + ALPHA_B * fB;
    fCb    = (1.0 / CB_FACT_B) * (fB - fY);
    fCr    = (1.0 / CR_FACT_R) * (fR - fY);
    sp0[n] = round_d(fY);
    sp1[n] = round_d(fCb);
    sp2[n] = round_d(fCr);
  }
}

void cvt_ycbcr_to_rgb_rev_neon(int32_t *sp0, int32_t *sp1, int32_t *sp2, uint32_t num_tc_samples) {
  int32_t R, G, B;
  int32_t Y, Cb, Cr;
  for (uint32_t n = 0; n < num_tc_samples - num_tc_samples % 4; n += 4) {
    auto vY  = vld1q_s32(sp0 + n);
    auto vCb = vld1q_s32(sp1 + n);
    auto vCr = vld1q_s32(sp2 + n);
    auto vG  = vY - ((vCb + vCr) >> 2);
    auto vR  = vCr + vG;
    auto vB  = vCb + vG;
    vst1q_s32(sp0 + n, vR);
    vst1q_s32(sp1 + n, vG);
    vst1q_s32(sp2 + n, vB);
  }
  for (uint32_t n = num_tc_samples - num_tc_samples % 4; n < num_tc_samples; ++n) {
    Y      = sp0[n];
    Cb     = sp1[n];
    Cr     = sp2[n];
    G      = Y - ((Cb + Cr) >> 2);
    R      = Cr + G;
    B      = Cb + G;
    sp0[n] = R;
    sp1[n] = G;
    sp2[n] = B;
  }
}

void cvt_ycbcr_to_rgb_irrev_neon(int32_t *sp0, int32_t *sp1, int32_t *sp2, uint32_t num_tc_samples) {
  int32_t R, G, B;
  double fY, fCb, fCr;
  for (uint32_t n = 0; n < num_tc_samples - num_tc_samples % 4; n += 4) {
    int32x4_t s32x4_Y    = vld1q_s32(sp0 + n);
    int32x4_t s32x4_Cb   = vld1q_s32(sp1 + n);
    int32x4_t s32x4_Cr   = vld1q_s32(sp2 + n);
    float32x4_t f32x4_Y  = vcvtq_f32_s32(s32x4_Y);
    float32x4_t f32x4_Cb = vcvtq_f32_s32(s32x4_Cb);
    float32x4_t f32x4_Cr = vcvtq_f32_s32(s32x4_Cr);

    // TODO: need to consider precision and setting FPSCR register value
    vst1q_s32(sp0 + n, vcvtnq_s32_f32(vaddq_f32(f32x4_Y, vmulq_n_f32(f32x4_Cr, CR_FACT_R))));
    vst1q_s32(sp2 + n, vcvtnq_s32_f32(vaddq_f32(f32x4_Y, vmulq_n_f32(f32x4_Cb, CB_FACT_B))));
    vst1q_s32(sp1 + n, vcvtnq_s32_f32(vsubq_f32(f32x4_Y, vaddq_f32(vmulq_n_f32(f32x4_Cr, CR_FACT_G),
                                                                   vmulq_n_f32(f32x4_Cb, CB_FACT_G)))));
  }
  for (uint32_t n = num_tc_samples - num_tc_samples % 4; n < num_tc_samples; ++n) {
    fY     = static_cast<double>(sp0[n]);
    fCb    = static_cast<double>(sp1[n]);
    fCr    = static_cast<double>(sp2[n]);
    R      = static_cast<int32_t>(round_d(fY + CR_FACT_R * fCr));
    B      = static_cast<int32_t>(round_d(fY + CB_FACT_B * fCb));
    G      = static_cast<int32_t>(round_d(fY - CR_FACT_G * fCr - CB_FACT_G * fCb));
    sp0[n] = R;
    sp1[n] = G;
    sp2[n] = B;
  }
}
#endif