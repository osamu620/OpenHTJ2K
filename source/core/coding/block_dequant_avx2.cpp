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

#if defined(OPENHTJ2K_TRY_AVX2) && defined(__AVX2__) && !defined(__AVX512F__)

  #if defined(_MSC_VER) || defined(__MINGW64__)
    #include <intrin.h>
  #else
    #include <x86intrin.h>
  #endif
  #include "block_dequant.hpp"
  #include <cassert>

template <bool StoreI32>
static void j2k_dequant_rev_avx2(const int32_t *sample_buf, size_t blksampl_stride, sprec_t *band_buf,
                                 uint32_t band_stride, uint32_t width, uint32_t height, int32_t M_b,
                                 uint8_t ROIshift) {
  const int32_t pLSB  = 31 - M_b;
  const uint32_t mask = UINT32_MAX >> (M_b + 1);
  assert(pLSB >= 0);

  const __m256i v_int32max = _mm256_set1_epi32(INT32_MAX);
  const __m256i v_notmask  = _mm256_set1_epi32((int32_t)~mask);
  const __m256i v_zero     = _mm256_setzero_si256();

  for (uint32_t y = 0; y < height; y++) {
    const int32_t *val_row = sample_buf + y * blksampl_stride;
    sprec_t *dst_row       = band_buf + y * band_stride;
    uint32_t x             = 0;

    for (; x + 8 <= width; x += 8) {
      __m256i v_val  = _mm256_loadu_si256((const __m256i *)(val_row + x));
      __m256i v_sign = _mm256_srai_epi32(v_val, 31);
      v_val          = _mm256_and_si256(v_val, v_int32max);
      __m256i v_qf32;
      if (ROIshift) {
        // All bit-planes count as decoded, so no mid-point is added.
        const __m256i v_cond_roi = _mm256_cmpeq_epi32(_mm256_and_si256(v_val, v_notmask), v_zero);
        v_val  = _mm256_blendv_epi8(v_val, _mm256_slli_epi32(v_val, ROIshift), v_cond_roi);
        v_val  = _mm256_sub_epi32(_mm256_xor_si256(v_val, v_sign), v_sign);
        v_qf32 = _mm256_srai_epi32(v_val, pLSB);
      } else {
        // A marker at or above pLSB is the mid-point; one below pLSB is shifted out.
        v_qf32 = _mm256_srli_epi32(v_val, pLSB);
        v_qf32 = _mm256_sub_epi32(_mm256_xor_si256(v_qf32, v_sign), v_sign);
      }
      OPENHTJ2K_IF_CONSTEXPR (StoreI32)
        _mm256_storeu_si256((__m256i *)(dst_row + x), v_qf32);
      else
        _mm256_storeu_ps(dst_row + x, _mm256_cvtepi32_ps(v_qf32));
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

void j2k_dequant(const int32_t *sample_buf, size_t blksampl_stride, sprec_t *band_buf, uint32_t band_stride,
                 uint32_t width, uint32_t height, int32_t M_b, uint8_t ROIshift, uint8_t transformation,
                 float stepsize, bool dequant_i32) {
  const int32_t pLSB  = 31 - M_b;
  const uint32_t mask = UINT32_MAX >> (M_b + 1);

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
    if (dequant_i32)
      j2k_dequant_rev_avx2<true>(sample_buf, blksampl_stride, band_buf, band_stride, width, height, M_b,
                                 ROIshift);
    else
      j2k_dequant_rev_avx2<false>(sample_buf, blksampl_stride, band_buf, band_stride, width, height, M_b,
                                  ROIshift);
  } else {
    // Irreversible path.  Without an ROI shift the reconstruction marker is the
    // mid-point already; with one, all bit-planes count as decoded and the mid-point
    // is half the LSB.
    const __m256i v_scale    = _mm256_set1_epi32(scale);
    const __m256i v_round16  = _mm256_set1_epi32(1 << 15);
    const __m256i v_roundds  = _mm256_set1_epi32(1 << (downshift - 1));
    const __m256i v_int32max = _mm256_set1_epi32(INT32_MAX);
    const __m256i v_notmask  = _mm256_set1_epi32((int32_t)~mask);
    const __m256i v_zero     = _mm256_setzero_si256();
    const __m256i v_roi_half = _mm256_set1_epi32(ROIshift ? 1 << (pLSB - 1) : 0);

    for (uint32_t y = 0; y < height; y++) {
      const int32_t *val_row = sample_buf + y * blksampl_stride;
      sprec_t *dst_row       = band_buf + y * band_stride;
      uint32_t x             = 0;

      for (; x + 8 <= width; x += 8) {
        __m256i v_val = _mm256_loadu_si256((const __m256i *)(val_row + x));

        // Extract sign (bit 31) as mask: 0 or -1
        __m256i v_sign = _mm256_srai_epi32(v_val, 31);

        // Clear sign bit
        v_val = _mm256_and_si256(v_val, v_int32max);

        if (ROIshift) {
          // Upshift lanes where (val & ~mask) == 0. Pure vector; no scalar round-trip
          // through a stack array — MSVC miscompiles the store-loop-reload pattern
          // via __m256i* / int32_t* aliasing (observed on VS 2022 /arch:AVX2 /O2).
          const __m256i v_cond_roi = _mm256_cmpeq_epi32(_mm256_and_si256(v_val, v_notmask), v_zero);
          v_val = _mm256_blendv_epi8(v_val, _mm256_slli_epi32(v_val, ROIshift), v_cond_roi);
          const __m256i v_nonzero = _mm256_cmpgt_epi32(v_val, v_zero);
          v_val                   = _mm256_or_si256(v_val, _mm256_and_si256(v_roi_half, v_nonzero));
        }

        // Truncate: val = (val + (1<<15)) >> 16
        v_val = _mm256_srai_epi32(_mm256_add_epi32(v_val, v_round16), 16);

        // Dequantize: val *= scale
        v_val = _mm256_mullo_epi32(v_val, v_scale);

        // Downshift with rounding
        __m256i v_qf32 = _mm256_srai_epi32(_mm256_add_epi32(v_val, v_roundds), downshift);

        // Apply sign (branchless)
        v_qf32 = _mm256_sub_epi32(_mm256_xor_si256(v_qf32, v_sign), v_sign);

        // Convert to float and store
        _mm256_storeu_ps(dst_row + x, _mm256_cvtepi32_ps(v_qf32));
      }

      // Scalar tail
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
        val *= scale;
        int32_t QF32 = (int32_t)((val + (1 << (downshift - 1))) >> downshift);
        QF32         = (QF32 ^ smask) - smask;
        dst_row[x]   = static_cast<float>(QF32);
      }
    }
  }
}

#endif  // OPENHTJ2K_TRY_AVX2 && __AVX2__ && !__AVX512F__
