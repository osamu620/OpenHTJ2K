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

#if !defined(OPENHTJ2K_ENABLE_ARM_NEON) && (!defined(__AVX2__) || !defined(OPENHTJ2K_TRY_AVX2))

#include "block_dequant.hpp"
#include <cassert>

void j2k_dequant(int32_t *sample_buf, size_t blksampl_stride, const uint8_t *block_states,
                 size_t blkstate_stride, sprec_t *i_samples, uint32_t band_stride, uint32_t width,
                 uint32_t height, int32_t M_b, uint8_t ROIshift, uint8_t transformation,
                 float stepsize) {
  int32_t N_b;
  const int32_t pLSB  = 31 - M_b;
  const uint32_t mask = UINT32_MAX >> (M_b + 1);
  int32_t r_val;
  int32_t offset = 0;

  float fscale = stepsize;
  fscale *= (1 << FRACBITS);
  if (M_b <= 31) {
    fscale /= (static_cast<float>(1 << (31 - M_b)));
  } else {
    fscale *= (static_cast<float>(1 << (M_b - 31)));
  }
  constexpr int32_t downshift = 15;
  fscale *= (float)(1 << 16) * (float)(1 << downshift);
  const auto scale = (int32_t)(fscale + 0.5);

  if (transformation == 1) {
    // reversible path
    for (uint32_t y = 0; y < height; y++) {
      for (uint32_t x = 0; x < width; x++) {
        const uint32_t n = x + y * band_stride;
        int32_t *val     = &sample_buf[x + y * blksampl_stride];
        uint8_t state    = block_states[(x + 1) + (y + 1) * blkstate_stride];
        sprec_t *dst     = i_samples + n;
        int32_t sign     = *val & INT32_MIN;
        *val &= INT32_MAX;
        if (ROIshift && (((uint32_t)*val & ~mask) == 0)) {
          *val <<= ROIshift;
        }
        if (ROIshift) {
          N_b = 30 - pLSB + 1;
        } else {
          N_b = 30 - (state >> 3) + 1;
        }
        offset        = (M_b > N_b) ? M_b - N_b : 0;
        r_val         = 1 << (pLSB - 1 + offset);
        if (*val != 0 && N_b < M_b) {
          *val |= r_val;
        }
        int32_t smask = sign >> 31;
        *val          = (*val ^ smask) - smask;
        assert(pLSB >= 0);
        int32_t QF32 = *val >> pLSB;
        *dst         = static_cast<float>(QF32);
      }
    }
  } else {
    // irreversible path
    for (uint32_t y = 0; y < height; y++) {
      for (uint32_t x = 0; x < width; x++) {
        const uint32_t n = x + y * band_stride;
        int32_t *val     = &sample_buf[x + y * blksampl_stride];
        uint8_t state    = block_states[(x + 1) + (y + 1) * blkstate_stride];
        sprec_t *dst     = i_samples + n;
        int32_t sign     = *val & INT32_MIN;
        *val &= INT32_MAX;
        if (ROIshift && (((uint32_t)*val & ~mask) == 0)) {
          *val <<= ROIshift;
        }
        if (ROIshift) {
          N_b = 30 - pLSB + 1;
        } else {
          N_b = 30 - (state >> 3) + 1;
        }
        offset        = (M_b > N_b) ? M_b - N_b : 0;
        r_val         = 1 << (pLSB - 1 + offset);
        if (*val != 0) {
          *val |= r_val;
        }
        *val          = (*val + (1 << 15)) >> 16;
        *val         *= scale;
        int32_t QF32  = (int32_t)((*val + (1 << (downshift - 1))) >> downshift);
        int32_t smask = sign >> 31;
        QF32          = (QF32 ^ smask) - smask;
        *dst          = static_cast<float>(QF32);
      }
    }
  }
}

#endif
