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

template <bool StoreI32>
static void j2k_dequant_rev_neon(const int32_t *sample_buf, size_t blksampl_stride, sprec_t *band_buf,
                                 uint32_t band_stride, uint32_t width, uint32_t height, int32_t M_b,
                                 uint8_t ROIshift) {
  const int32_t pLSB  = 31 - M_b;
  const uint32_t mask = UINT32_MAX >> (M_b + 1);
  assert(pLSB >= 0);

  const int32x4_t v_neg_pLSB = vdupq_n_s32(-pLSB);
  const int32x4_t v_int32max = vdupq_n_s32(INT32_MAX);

  for (uint32_t y = 0; y < height; y++) {
    const int32_t *val_row = sample_buf + y * blksampl_stride;
    sprec_t *dst_row       = band_buf + y * band_stride;
    uint32_t x             = 0;

    for (; x + 4 <= width; x += 4) {
      int32x4_t v_val  = vld1q_s32(val_row + x);
      int32x4_t v_sign = vshrq_n_s32(v_val, 31);
      v_val            = vandq_s32(v_val, v_int32max);
      int32x4_t v_qf32;
      if (ROIshift) {
        // All bit-planes count as decoded, so no mid-point is added.
        int32_t tmp[4];
        vst1q_s32(tmp, v_val);
        for (int i = 0; i < 4; i++) {
          if (((uint32_t)tmp[i] & ~mask) == 0) {
            tmp[i] <<= ROIshift;
          }
        }
        v_val  = vld1q_s32(tmp);
        v_val  = vsubq_s32(veorq_s32(v_val, v_sign), v_sign);
        v_qf32 = vshlq_s32(v_val, v_neg_pLSB);
      } else {
        // A marker at or above pLSB is the mid-point; one below pLSB is shifted out.
        v_qf32 = vreinterpretq_s32_u32(vshlq_u32(vreinterpretq_u32_s32(v_val), v_neg_pLSB));
        v_qf32 = vsubq_s32(veorq_s32(v_qf32, v_sign), v_sign);
      }
      OPENHTJ2K_IF_CONSTEXPR (StoreI32)
        vst1q_s32(reinterpret_cast<int32_t *>(dst_row + x), v_qf32);
      else
        vst1q_f32(dst_row + x, vcvtq_f32_s32(v_qf32));
    }

    for (; x < width; x++) {
      const int32_t smask = val_row[x] >> 31;
      int32_t mag         = val_row[x] & INT32_MAX;
      int32_t QF32;
      if (ROIshift) {
        if (((uint32_t)mag & ~mask) == 0) mag <<= ROIshift;
        QF32 = ((mag ^ smask) - smask) >> pLSB;
      } else {
        QF32 = ((mag >> pLSB) ^ smask) - smask;
      }
      OPENHTJ2K_IF_CONSTEXPR (StoreI32)
        reinterpret_cast<int32_t *>(dst_row)[x] = QF32;
      else
        dst_row[x] = static_cast<float>(QF32);
    }
  }
}

// Lossy, no ROI: magnitude -> float -> scale, sign applied by XOR (the HT decoder's
// dequant_store_neon does the same).
static void j2k_dequant_irrev_neon(const int32_t *sample_buf, size_t blksampl_stride, sprec_t *band_buf,
                                   uint32_t band_stride, uint32_t width, uint32_t height,
                                   float fscale_direct) {
  const float32x4_t v_fscale = vdupq_n_f32(fscale_direct);
  const int32x4_t v_magmask  = vdupq_n_s32(INT32_MAX);
  const int32x4_t v_signmask = vdupq_n_s32(INT32_MIN);
  for (uint32_t y = 0; y < height; y++) {
    const int32_t *val_row = sample_buf + y * blksampl_stride;
    sprec_t *dst_row       = band_buf + y * band_stride;
    uint32_t x             = 0;
    for (; x + 4 <= width; x += 4) {
      const int32x4_t v = vld1q_s32(val_row + x);
      float32x4_t f     = vmulq_f32(vcvtq_f32_s32(vandq_s32(v, v_magmask)), v_fscale);
      f = vreinterpretq_f32_s32(veorq_s32(vreinterpretq_s32_f32(f), vandq_s32(v, v_signmask)));
      vst1q_f32(dst_row + x, f);
    }
    for (; x < width; x++) {
      const int32_t v = val_row[x];
      const float f   = static_cast<float>(v & INT32_MAX) * fscale_direct;
      dst_row[x]      = (v < 0) ? -f : f;
    }
  }
}

void j2k_dequant(const int32_t *sample_buf, size_t blksampl_stride, sprec_t *band_buf, uint32_t band_stride,
                 uint32_t width, uint32_t height, int32_t M_b, uint8_t ROIshift, uint8_t transformation,
                 float stepsize, bool dequant_i32) {
  const int32_t pLSB  = 31 - M_b;
  const uint32_t mask = UINT32_MAX >> (M_b + 1);

  // Direct float scale factor: the decoded magnitude is in Q(31-M_b) fixed-point
  // and the subband sample must be in Q(FRACBITS).
  float fscale_direct = stepsize;
  fscale_direct *= static_cast<float>(1 << FRACBITS);
  if (M_b <= 31) {
    fscale_direct /= (static_cast<float>(1 << (31 - M_b)));
  } else {
    fscale_direct *= (static_cast<float>(1 << (M_b - 31)));
  }
  // Integer scale for the ROI path below.
  constexpr int32_t downshift = 15;
  float fscale                = fscale_direct;
  fscale *= (float)(1 << 16) * (float)(1 << downshift);
  const auto scale = (int32_t)(fscale + 0.5);

  if (transformation == 1) {
    if (dequant_i32)
      j2k_dequant_rev_neon<true>(sample_buf, blksampl_stride, band_buf, band_stride, width, height, M_b,
                                 ROIshift);
    else
      j2k_dequant_rev_neon<false>(sample_buf, blksampl_stride, band_buf, band_stride, width, height, M_b,
                                  ROIshift);
  } else if (ROIshift == 0) {
    // Lossy, no ROI (the common case): one float multiply per sample, the same
    // arithmetic as the HT decoder's dequantize, so Part 1 and HT code-blocks that
    // hold the same quantization indices reconstruct to the same values.  The
    // integer pipeline of the ROI path rounds every sample to the Q(FRACBITS) grid
    // and drops the index bits below bit 16 when M_b > 15, which cost up to 1 LSB
    // at 8-bit and tens of LSBs at 16-bit depth against the HT path.
    j2k_dequant_irrev_neon(sample_buf, blksampl_stride, band_buf, band_stride, width, height,
                           fscale_direct);
  } else {
    // Irreversible path with an ROI shift: all bit-planes count as decoded and the
    // mid-point is half the LSB.
    const int32x4_t v_scale    = vdupq_n_s32(scale);
    const int32x4_t v_round16  = vdupq_n_s32(1 << 15);
    const int32x4_t v_roundds  = vdupq_n_s32(1 << (downshift - 1));
    const int32x4_t v_int32max = vdupq_n_s32(INT32_MAX);
    const int32x4_t v_roi_half = vdupq_n_s32(ROIshift ? 1 << (pLSB - 1) : 0);

    for (uint32_t y = 0; y < height; y++) {
      const int32_t *val_row = sample_buf + y * blksampl_stride;
      sprec_t *dst_row       = band_buf + y * band_stride;
      uint32_t x             = 0;

      for (; x + 4 <= width; x += 4) {
        // Load 4 samples
        int32x4_t v_val = vld1q_s32(val_row + x);

        // Extract sign (bit 31) as mask: 0 or -1
        int32x4_t v_sign = vshrq_n_s32(v_val, 31);

        // Clear sign bit
        v_val = vandq_s32(v_val, v_int32max);

        if (ROIshift) {
          int32_t tmp[4];
          vst1q_s32(tmp, v_val);
          for (int i = 0; i < 4; i++) {
            if (((uint32_t)tmp[i] & ~mask) == 0) {
              tmp[i] <<= ROIshift;
            }
          }
          v_val                = vld1q_s32(tmp);
          uint32x4_t v_nonzero = vcgtq_s32(v_val, vdupq_n_s32(0));
          v_val                = vorrq_s32(v_val, vandq_s32(v_roi_half, vreinterpretq_s32_u32(v_nonzero)));
        }

        // Truncate to int16_t range: val = (val + (1<<15)) >> 16
        v_val = vshrq_n_s32(vaddq_s32(v_val, v_round16), 16);

        // Dequantize: val *= scale
        v_val = vmulq_s32(v_val, v_scale);

        // Downshift with rounding: QF32 = (val + (1 << (downshift-1))) >> downshift
        int32x4_t v_qf32 = vshrq_n_s32(vaddq_s32(v_val, v_roundds), downshift);

        // Apply sign (branchless)
        v_qf32 = vsubq_s32(veorq_s32(v_qf32, v_sign), v_sign);

        // Convert to float and store
        vst1q_f32(dst_row + x, vcvtq_f32_s32(v_qf32));
      }

      // Scalar tail.  Match the SIMD path's modular 32-bit multiply via uint32_t
      // arithmetic — `int32_t * int32_t` is signed-overflow UB and MSVC ARM64
      // exploits it to produce different bits from vmulq_s32 in some inlining
      // contexts, which surfaces as the lbs_p1_05 1-LSB drift.
      for (; x < width; x++) {
        const int32_t smask = val_row[x] >> 31;
        int32_t val         = val_row[x] & INT32_MAX;
        if (ROIshift) {
          if (((uint32_t)val & ~mask) == 0) {
            val <<= ROIshift;
          }
          if (val != 0) {
            val |= 1 << (pLSB - 1);
          }
        }
        val = (val + (1 << 15)) >> 16;
        // Modular 32-bit multiply, matching vmulq_s32 exactly (no signed-overflow UB).
        val          = static_cast<int32_t>(static_cast<uint32_t>(val) * static_cast<uint32_t>(scale));
        int32_t QF32 = (int32_t)((val + (1 << (downshift - 1))) >> downshift);
        QF32         = (QF32 ^ smask) - smask;
        dst_row[x]   = static_cast<float>(QF32);
      }
    }
  }
}

#endif  // OPENHTJ2K_ENABLE_ARM_NEON
