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

#if defined(__ARM_NEON__)
  #include "utils.hpp"
  #include "color.hpp"

void cvt_rgb_to_ycbcr_irrev_neon(int32_t *sp0, int32_t *sp1, int32_t *sp2, uint32_t num_tc_samples) {
  double fR, fG, fB;
  double fY, fCb, fCr;

  for (uint32_t n = 0; n < num_tc_samples - num_tc_samples % 4; n += 4) {
    int32x4_t s32x4_R   = vld1q_s32(sp0);
    int32x4_t s32x4_G   = vld1q_s32(sp1);
    int32x4_t s32x4_B   = vld1q_s32(sp2);
    float32x4_t f32x4_R = vcvtq_f32_s32(s32x4_R);
    float32x4_t f32x4_G = vcvtq_f32_s32(s32x4_G);
    float32x4_t f32x4_B = vcvtq_f32_s32(s32x4_B);

    float32x4_t a0 = vdupq_n_f32(ALPHA_R);
    float32x4_t a1 = vdupq_n_f32(ALPHA_G);
    float32x4_t a2 = vdupq_n_f32(ALPHA_B);

    float32x4_t f32x4_Y = f32x4_R * a0 + f32x4_G * a1 + f32x4_B * a2;

    a0                   = vdupq_n_f32(1.0 / CB_FACT_B);
    float32x4_t f32x4_Cb = vmulq_f32(a0, vsubq_f32(f32x4_B, f32x4_Y));
    a1                   = vdupq_n_f32(1.0 / CR_FACT_R);
    float32x4_t f32x4_Cr = vmulq_f32(a1, vsubq_f32(f32x4_R, f32x4_Y));

    // TODO: need to consider precision and setting FPSCR register value
    vst1q_s32(sp0, vcvtq_s32_f32(f32x4_Y));
    vst1q_s32(sp1, vcvtq_s32_f32(f32x4_Cb));
    vst1q_s32(sp2, vcvtq_s32_f32(f32x4_Cr));
    sp0 += 4;
    sp1 += 4;
    sp2 += 4;
  }
  for (uint32_t n = 0; n < num_tc_samples % 4; n++) {
    fR     = static_cast<double>(sp0[0]);
    fG     = static_cast<double>(sp1[0]);
    fB     = static_cast<double>(sp2[0]);
    fY     = ALPHA_R * fR + ALPHA_G * fG + ALPHA_B * fB;
    fCb    = (1.0 / CB_FACT_B) * (fB - fY);
    fCr    = (1.0 / CR_FACT_R) * (fR - fY);
    sp0[0] = round_d(fY);
    sp1[0] = round_d(fCb);
    sp2[0] = round_d(fCr);
    sp0++;
    sp1++;
    sp2++;
  }
}

void cvt_ycbcr_to_rgb_irrev_neon(int32_t *sp0, int32_t *sp1, int32_t *sp2, uint32_t num_tc_samples) {
  int32_t R, G, B;
  double fY, fCb, fCr;
  for (uint32_t n = 0; n < num_tc_samples - num_tc_samples % 4; n += 4) {
    int32x4_t s32x4_Y    = vld1q_s32(sp0);
    int32x4_t s32x4_Cb   = vld1q_s32(sp1);
    int32x4_t s32x4_Cr   = vld1q_s32(sp2);
    float32x4_t f32x4_Y  = vcvtq_f32_s32(s32x4_Y);
    float32x4_t f32x4_Cb = vcvtq_f32_s32(s32x4_Cb);
    float32x4_t f32x4_Cr = vcvtq_f32_s32(s32x4_Cr);

    float32x4_t a0 = vdupq_n_f32(CR_FACT_R);
    float32x4_t a1 = vdupq_n_f32(CB_FACT_B);
    float32x4_t a2 = vdupq_n_f32(CR_FACT_G);
    float32x4_t a3 = vdupq_n_f32(CB_FACT_G);

    // TODO: need to consider precision and setting FPSCR register value
    vst1q_s32(sp0, vcvtq_s32_f32(vaddq_f32(f32x4_Y, vmulq_f32(a0, f32x4_Cr))));
    vst1q_s32(sp2, vcvtq_s32_f32(vaddq_f32(f32x4_Y, vmulq_f32(a1, f32x4_Cb))));
    vst1q_s32(sp1, vcvtq_s32_f32(
                       vsubq_f32(f32x4_Y, vaddq_f32(vmulq_f32(a2, f32x4_Cr), vmulq_f32(a3, f32x4_Cb)))));
    sp0 += 4;
    sp1 += 4;
    sp2 += 4;
  }
  for (uint32_t n = 0; n < num_tc_samples % 4; n++) {
    fY     = static_cast<double>(sp0[0]);
    fCb    = static_cast<double>(sp1[0]);
    fCr    = static_cast<double>(sp2[0]);
    R      = static_cast<int32_t>(round_d(fY + CR_FACT_R * fCr));
    B      = static_cast<int32_t>(round_d(fY + CB_FACT_B * fCb));
    G      = static_cast<int32_t>(round_d(fY - CR_FACT_G * fCr - CB_FACT_G * fCb));
    sp0[0] = R;
    sp1[0] = G;
    sp2[0] = B;
    sp0++;
    sp1++;
    sp2++;
  }
}
#endif