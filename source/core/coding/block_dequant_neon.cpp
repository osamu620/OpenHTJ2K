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

#include "block_dequant.hpp"
#include <arm_neon.h>
#include <cassert>
#include <cstddef>

void j2k_dequant(int32_t *sample_buf, size_t blksampl_stride, const uint8_t *block_states,
                 size_t blkstate_stride, sprec_t *i_samples, uint32_t band_stride, uint32_t width,
                 uint32_t height, int32_t M_b, uint8_t ROIshift, uint8_t transformation,
                 float stepsize) {
  int32_t N_b;
  const int32_t pLSB    = 31 - M_b;
  const uint32_t mask   = UINT32_MAX >> (M_b + 1);
  const int32_t pLSB_m1 = pLSB - 1;

  // Precompute irreversible scale
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
    // Reversible path
    const int32x4_t v_M_b     = vdupq_n_s32(M_b);
    const int32x4_t v_30      = vdupq_n_s32(30);
    const int32x4_t v_1       = vdupq_n_s32(1);
    const int32x4_t v_pLSB_m1 = vdupq_n_s32(pLSB_m1);
    const int32x4_t v_neg_pLSB = vdupq_n_s32(-pLSB);

    for (uint32_t y = 0; y < height; y++) {
      int32_t *val_row        = sample_buf + y * blksampl_stride;
      const uint8_t *st_row   = block_states + (y + 1) * blkstate_stride + 1;
      sprec_t *dst_row        = i_samples + y * band_stride;
      uint32_t x              = 0;

      for (; x + 4 <= width; x += 4) {
        // Load 4 samples
        int32x4_t v_val = vld1q_s32(val_row + x);

        // Extract sign (bit 31)
        int32x4_t v_sign = vshrq_n_s32(v_val, 31);  // 0 or -1

        // Clear sign bit
        v_val = vandq_s32(v_val, vdupq_n_s32(INT32_MAX));

        if (ROIshift) {
          // ROI handling: scalar fallback for this rare case
          int32_t tmp[4];
          vst1q_s32(tmp, v_val);
          for (int i = 0; i < 4; i++) {
            if (((uint32_t)tmp[i] & ~mask) == 0) {
              tmp[i] <<= ROIshift;
            }
          }
          v_val = vld1q_s32(tmp);
        }

        // Load 4 state bytes and extract N_b = 31 - (state >> 3)
        uint8_t st_bytes[4] = {st_row[x], st_row[x + 1], st_row[x + 2], st_row[x + 3]};
        int32x4_t v_state;
        if (ROIshift) {
          v_state = vdupq_n_s32(30 - pLSB + 1);
        } else {
          int32_t st_vals[4] = {st_bytes[0] >> 3, st_bytes[1] >> 3, st_bytes[2] >> 3,
                                st_bytes[3] >> 3};
          int32x4_t v_st = vld1q_s32(st_vals);
          v_state        = vaddq_s32(vsubq_s32(v_30, v_st), v_1);  // 30 - (state>>3) + 1
        }

        // offset = max(M_b - N_b, 0)
        int32x4_t v_offset = vmaxq_s32(vsubq_s32(v_M_b, v_state), vdupq_n_s32(0));

        // r_val = 1 << (pLSB - 1 + offset)
        int32x4_t v_shift = vaddq_s32(v_pLSB_m1, v_offset);
        int32x4_t v_rval  = vshlq_s32(v_1, v_shift);  // per-element variable shift

        // Add r_val if val != 0 && N_b < M_b
        uint32x4_t v_nonzero = vcgtq_s32(v_val, vdupq_n_s32(0));  // val > 0 (sign cleared)
        uint32x4_t v_nb_lt   = vcltq_s32(v_state, v_M_b);         // N_b < M_b
        uint32x4_t v_cond    = vandq_u32(v_nonzero, v_nb_lt);
        v_val = vorrq_s32(v_val, vandq_s32(v_rval, vreinterpretq_s32_u32(v_cond)));

        // Sign-magnitude to two's complement (branchless)
        v_val = vsubq_s32(veorq_s32(v_val, v_sign), v_sign);

        // Right shift by pLSB
        int32x4_t v_qf32 = vshlq_s32(v_val, v_neg_pLSB);

        // Convert to float and store
        float32x4_t v_out = vcvtq_f32_s32(v_qf32);
        vst1q_f32(dst_row + x, v_out);
      }

      // Scalar tail
      for (; x < width; x++) {
        int32_t *val = val_row + x;
        int32_t sign = *val & INT32_MIN;
        *val &= INT32_MAX;
        if (ROIshift && (((uint32_t)*val & ~mask) == 0)) {
          *val <<= ROIshift;
        }
        uint8_t state = st_row[x];
        if (ROIshift) {
          N_b = 30 - pLSB + 1;
        } else {
          N_b = 30 - (state >> 3) + 1;
        }
        int32_t offset = (M_b > N_b) ? M_b - N_b : 0;
        int32_t r_val  = 1 << (pLSB_m1 + offset);
        if (*val != 0 && N_b < M_b) {
          *val |= r_val;
        }
        int32_t smask = sign >> 31;
        *val          = (*val ^ smask) - smask;
        assert(pLSB >= 0);
        int32_t QF32       = *val >> pLSB;
        dst_row[x]         = static_cast<float>(QF32);
      }
    }
  } else {
    // Irreversible path
    const int32x4_t v_M_b     = vdupq_n_s32(M_b);
    const int32x4_t v_30      = vdupq_n_s32(30);
    const int32x4_t v_1       = vdupq_n_s32(1);
    const int32x4_t v_pLSB_m1 = vdupq_n_s32(pLSB_m1);
    const int32x4_t v_scale   = vdupq_n_s32(scale);
    const int32x4_t v_round16 = vdupq_n_s32(1 << 15);
    const int32x4_t v_roundds = vdupq_n_s32(1 << (downshift - 1));

    for (uint32_t y = 0; y < height; y++) {
      int32_t *val_row        = sample_buf + y * blksampl_stride;
      const uint8_t *st_row   = block_states + (y + 1) * blkstate_stride + 1;
      sprec_t *dst_row        = i_samples + y * band_stride;
      uint32_t x              = 0;

      for (; x + 4 <= width; x += 4) {
        // Load 4 samples
        int32x4_t v_val = vld1q_s32(val_row + x);

        // Extract sign (bit 31) as mask: 0 or -1
        int32x4_t v_sign = vshrq_n_s32(v_val, 31);

        // Clear sign bit
        v_val = vandq_s32(v_val, vdupq_n_s32(INT32_MAX));

        if (ROIshift) {
          int32_t tmp[4];
          vst1q_s32(tmp, v_val);
          for (int i = 0; i < 4; i++) {
            if (((uint32_t)tmp[i] & ~mask) == 0) {
              tmp[i] <<= ROIshift;
            }
          }
          v_val = vld1q_s32(tmp);
        }

        // Load state bytes and compute N_b
        uint8_t st_bytes[4] = {st_row[x], st_row[x + 1], st_row[x + 2], st_row[x + 3]};
        int32x4_t v_state;
        if (ROIshift) {
          v_state = vdupq_n_s32(30 - pLSB + 1);
        } else {
          int32_t st_vals[4] = {st_bytes[0] >> 3, st_bytes[1] >> 3, st_bytes[2] >> 3,
                                st_bytes[3] >> 3};
          int32x4_t v_st = vld1q_s32(st_vals);
          v_state        = vaddq_s32(vsubq_s32(v_30, v_st), v_1);
        }

        // offset = max(M_b - N_b, 0)
        int32x4_t v_offset = vmaxq_s32(vsubq_s32(v_M_b, v_state), vdupq_n_s32(0));

        // r_val = 1 << (pLSB - 1 + offset)
        int32x4_t v_shift = vaddq_s32(v_pLSB_m1, v_offset);
        int32x4_t v_rval  = vshlq_s32(v_1, v_shift);

        // Add r_val if val != 0 (irreversible always adds when non-zero)
        uint32x4_t v_nonzero = vcgtq_s32(v_val, vdupq_n_s32(0));
        v_val = vorrq_s32(v_val, vandq_s32(v_rval, vreinterpretq_s32_u32(v_nonzero)));

        // Truncate to int16_t range: val = (val + (1<<15)) >> 16
        v_val = vshrq_n_s32(vaddq_s32(v_val, v_round16), 16);

        // Dequantize: val *= scale
        v_val = vmulq_s32(v_val, v_scale);

        // Downshift with rounding: QF32 = (val + (1 << (downshift-1))) >> downshift
        int32x4_t v_qf32 = vshrq_n_s32(vaddq_s32(v_val, v_roundds), downshift);

        // Apply sign (branchless)
        v_qf32 = vsubq_s32(veorq_s32(v_qf32, v_sign), v_sign);

        // Convert to float and store
        float32x4_t v_out = vcvtq_f32_s32(v_qf32);
        vst1q_f32(dst_row + x, v_out);
      }

      // Scalar tail
      for (; x < width; x++) {
        int32_t *val = val_row + x;
        int32_t sign = *val & INT32_MIN;
        *val &= INT32_MAX;
        if (ROIshift && (((uint32_t)*val & ~mask) == 0)) {
          *val <<= ROIshift;
        }
        uint8_t state = st_row[x];
        if (ROIshift) {
          N_b = 30 - pLSB + 1;
        } else {
          N_b = 30 - (state >> 3) + 1;
        }
        int32_t offset = (M_b > N_b) ? M_b - N_b : 0;
        int32_t r_val  = 1 << (pLSB_m1 + offset);
        if (*val != 0) {
          *val |= r_val;
        }
        *val          = (*val + (1 << 15)) >> 16;
        *val         *= scale;
        int32_t QF32  = (int32_t)((*val + (1 << (downshift - 1))) >> downshift);
        int32_t smask = sign >> 31;
        QF32          = (QF32 ^ smask) - smask;
        dst_row[x]    = static_cast<float>(QF32);
      }
    }
  }
}

#endif  // OPENHTJ2K_ENABLE_ARM_NEON
