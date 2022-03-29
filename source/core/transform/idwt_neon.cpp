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
  #include "dwt.hpp"
  #include "utils.hpp"

void idwt_1d_filtr_irrev97_fixed_neon(sprec_t *X, const int32_t left, const int32_t right,
                                      const uint32_t u_i0, const uint32_t u_i1) {
  const auto i0        = static_cast<const int32_t>(u_i0);
  const auto i1        = static_cast<const int32_t>(u_i1);
  const int32_t start  = i0 / 2;
  const int32_t stop   = i1 / 2;
  const int32_t offset = left - i0 % 2;

  // step 1
  int32_t simdlen = stop + 2 - (start - 1);
  for (int32_t n = -2 + offset, i = 0; i < simdlen; i += 8, n += 16) {
    auto xl0     = vld2q_s16(X + n - 1);
    auto xl1     = vld2q_s16(X + n + 1);
    auto vcoeff  = vdupq_n_s32(Dcoeff);
    auto voffset = vdupq_n_s32(Doffset);
    auto x0      = vreinterpretq_s32_s16(xl0.val[0]);
    auto x0l     = vmovl_s16(vreinterpret_s16_s32(vget_low_s32(x0)));
    auto x0h     = vmovl_s16(vreinterpret_s16_s32(vget_high_s32(x0)));
    auto x2      = vreinterpretq_s32_s16(xl1.val[0]);
    auto x2l     = vmovl_s16(vreinterpret_s16_s32(vget_low_s32(x2)));
    auto x2h     = vmovl_s16(vreinterpret_s16_s32(vget_high_s32(x2)));
    auto xoutl   = ((x0l + x2l) * vcoeff + voffset) >> Dshift;
    auto xouth   = ((x0h + x2h) * vcoeff + voffset) >> Dshift;
    xl0.val[1] -= vcombine_s16(vmovn_s32(xoutl), vmovn_s32(xouth));
    vst2q_s16(X + n - 1, xl0);
  }

  // step 2
  simdlen = stop + 1 - (start - 1);
  for (int32_t n = -2 + offset, i = 0; i < simdlen; i += 8, n += 16) {
    auto xl0     = vld2q_s16(X + n);
    auto xl1     = vld2q_s16(X + n + 2);
    auto vcoeff  = vdupq_n_s32(Ccoeff);
    auto voffset = vdupq_n_s32(Coffset);
    auto x0      = vreinterpretq_s32_s16(xl0.val[0]);
    auto x0l     = vmovl_s16(vreinterpret_s16_s32(vget_low_s32(x0)));
    auto x0h     = vmovl_s16(vreinterpret_s16_s32(vget_high_s32(x0)));
    auto x2      = vreinterpretq_s32_s16(xl1.val[0]);
    auto x2l     = vmovl_s16(vreinterpret_s16_s32(vget_low_s32(x2)));
    auto x2h     = vmovl_s16(vreinterpret_s16_s32(vget_high_s32(x2)));
    auto xoutl   = ((x0l + x2l) * vcoeff + voffset) >> Cshift;
    auto xouth   = ((x0h + x2h) * vcoeff + voffset) >> Cshift;
    xl0.val[1] -= vcombine_s16(vmovn_s32(xoutl), vmovn_s32(xouth));
    vst2q_s16(X + n, xl0);
  }

  // step 3
  simdlen = stop + 1 - start;
  for (int32_t n = 0 + offset, i = 0; i < simdlen; i += 8, n += 16) {
    auto xl0     = vld2q_s16(X + n - 1);
    auto xl1     = vld2q_s16(X + n + 1);
    auto vcoeff  = vdupq_n_s32(Bcoeff);
    auto voffset = vdupq_n_s32(Boffset);
    auto x0      = vreinterpretq_s32_s16(xl0.val[0]);
    auto x0l     = vmovl_s16(vreinterpret_s16_s32(vget_low_s32(x0)));
    auto x0h     = vmovl_s16(vreinterpret_s16_s32(vget_high_s32(x0)));
    auto x2      = vreinterpretq_s32_s16(xl1.val[0]);
    auto x2l     = vmovl_s16(vreinterpret_s16_s32(vget_low_s32(x2)));
    auto x2h     = vmovl_s16(vreinterpret_s16_s32(vget_high_s32(x2)));
    auto xoutl   = ((x0l + x2l) * vcoeff + voffset) >> Bshift;
    auto xouth   = ((x0h + x2h) * vcoeff + voffset) >> Bshift;
    xl0.val[1] -= vcombine_s16(vmovn_s32(xoutl), vmovn_s32(xouth));
    vst2q_s16(X + n - 1, xl0);
  }

  // step 4
  simdlen = stop - start;
  for (int32_t n = 0 + offset, i = 0; i < simdlen; i += 8, n += 16) {
    auto xl0     = vld2q_s16(X + n);
    auto xl1     = vld2q_s16(X + n + 2);
    auto vcoeff  = vdupq_n_s32(Acoeff);
    auto voffset = vdupq_n_s32(Aoffset);
    auto x0      = vreinterpretq_s32_s16(xl0.val[0]);
    auto x0l     = vmovl_s16(vreinterpret_s16_s32(vget_low_s32(x0)));
    auto x0h     = vmovl_s16(vreinterpret_s16_s32(vget_high_s32(x0)));
    auto x2      = vreinterpretq_s32_s16(xl1.val[0]);
    auto x2l     = vmovl_s16(vreinterpret_s16_s32(vget_low_s32(x2)));
    auto x2h     = vmovl_s16(vreinterpret_s16_s32(vget_high_s32(x2)));
    auto xoutl   = ((x0l + x2l) * vcoeff + voffset) >> Ashift;
    auto xouth   = ((x0h + x2h) * vcoeff + voffset) >> Ashift;
    xl0.val[1] -= vcombine_s16(vmovn_s32(xoutl), vmovn_s32(xouth));
    vst2q_s16(X + n, xl0);
  }
}

void idwt_1d_filtr_rev53_fixed_neon(sprec_t *X, const int32_t left, const int32_t right,
                                    const uint32_t u_i0, const uint32_t u_i1) {
  const auto i0        = static_cast<const int32_t>(u_i0);
  const auto i1        = static_cast<const int32_t>(u_i1);
  const int32_t start  = i0 / 2;
  const int32_t stop   = i1 / 2;
  const int32_t offset = left - i0 % 2;

  // step 1
  int32_t simdlen = stop + 1 - start;
  for (int32_t n = 0 + offset, i = 0; i < simdlen; i += 8, n += 16) {
    auto xl0  = vld2q_s16(X + n - 1);
    auto xl1  = vld2q_s16(X + n + 1);
    auto xout = (xl0.val[0] + xl1.val[0] + 2) >> 2;
    xl0.val[1] -= xout;
    vst2q_s16(X + n - 1, xl0);
  }

  // step 2
  simdlen = stop - start;
  for (int32_t n = 0 + offset, i = 0; i < simdlen; i += 8, n += 16) {
    auto xl0  = vld2q_s16(X + n);
    auto xl1  = vld2q_s16(X + n + 2);
    auto xout = vhaddq_s16(xl0.val[0], xl1.val[0]);
    xl0.val[1] += xout;
    vst2q_s16(X + n, xl0);
  }
}
#endif