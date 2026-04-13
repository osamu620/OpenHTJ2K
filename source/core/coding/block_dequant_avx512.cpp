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

#if defined(OPENHTJ2K_TRY_AVX2) && defined(__AVX512F__)

#if defined(_MSC_VER) || defined(__MINGW64__)
  #include <intrin.h>
#else
  #include <x86intrin.h>
#endif
#include "block_dequant.hpp"
#include <cassert>

void j2k_dequant(int32_t *sample_buf, size_t blksampl_stride, const uint8_t *block_states,
                 size_t blkstate_stride, sprec_t *i_samples, uint32_t band_stride, uint32_t width,
                 uint32_t height, int32_t M_b, uint8_t ROIshift, uint8_t transformation,
                 float stepsize) {
  int32_t N_b;
  const int32_t pLSB    = 31 - M_b;
  const uint32_t mask   = UINT32_MAX >> (M_b + 1);
  const int32_t pLSB_m1 = pLSB - 1;

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
    const __m512i v_M_b      = _mm512_set1_epi32(M_b);
    const __m512i v_30       = _mm512_set1_epi32(30);
    const __m512i v_1        = _mm512_set1_epi32(1);
    const __m512i v_pLSB_m1  = _mm512_set1_epi32(pLSB_m1);
    const __m512i v_int32max = _mm512_set1_epi32(INT32_MAX);
    const __m512i v_zero     = _mm512_setzero_si512();

    for (uint32_t y = 0; y < height; y++) {
      int32_t *val_row      = sample_buf + y * blksampl_stride;
      const uint8_t *st_row = block_states + (y + 1) * blkstate_stride + 1;
      sprec_t *dst_row      = i_samples + y * band_stride;
      uint32_t x            = 0;

      for (; x + 16 <= width; x += 16) {
        __m512i v_val = _mm512_loadu_si512(val_row + x);

        // Extract sign (bit 31) as mask: 0 or -1
        __m512i v_sign = _mm512_srai_epi32(v_val, 31);

        // Clear sign bit
        v_val = _mm512_and_si512(v_val, v_int32max);

        if (ROIshift) {
          // Upshift lanes where (val & ~mask) == 0. Pure vector via AVX-512 mask ops;
          // avoids the scalar store-loop-reload round-trip that MSVC miscompiles on
          // its AVX2 counterpart (see block_dequant_avx2.cpp).
          const __m512i v_notmask = _mm512_set1_epi32((int32_t)~mask);
          const __mmask16 k_roi =
              _mm512_cmpeq_epi32_mask(_mm512_and_si512(v_val, v_notmask), v_zero);
          v_val = _mm512_mask_slli_epi32(v_val, k_roi, v_val, ROIshift);
        }

        // Load 16 state bytes, widen to int32, compute N_b = 30 - (state >> 3) + 1
        __m512i v_state;
        if (ROIshift) {
          v_state = _mm512_set1_epi32(30 - pLSB + 1);
        } else {
          __m128i v_st8 = _mm_loadu_si128((__m128i *)(st_row + x));
          __m512i v_st  = _mm512_cvtepu8_epi32(v_st8);
          v_st          = _mm512_srli_epi32(v_st, 3);
          v_state       = _mm512_add_epi32(_mm512_sub_epi32(v_30, v_st), v_1);
        }

        // offset = max(M_b - N_b, 0)
        __m512i v_offset = _mm512_max_epi32(_mm512_sub_epi32(v_M_b, v_state), v_zero);

        // r_val = 1 << (pLSB - 1 + offset)
        __m512i v_shift = _mm512_add_epi32(v_pLSB_m1, v_offset);
        __m512i v_rval  = _mm512_sllv_epi32(v_1, v_shift);

        // Add r_val if val != 0 && N_b < M_b (using mask registers)
        __mmask16 k_nonzero = _mm512_cmpgt_epi32_mask(v_val, v_zero);
        __mmask16 k_nb_lt   = _mm512_cmpgt_epi32_mask(v_M_b, v_state);
        __mmask16 k_cond    = _kand_mask16(k_nonzero, k_nb_lt);
        v_val = _mm512_mask_or_epi32(v_val, k_cond, v_val, v_rval);

        // Sign-magnitude to two's complement (branchless)
        v_val = _mm512_sub_epi32(_mm512_xor_si512(v_val, v_sign), v_sign);

        // Right shift by pLSB
        __m512i v_qf32 = _mm512_srai_epi32(v_val, static_cast<unsigned int>(pLSB));

        // Convert to float and store
        __m512 v_out = _mm512_cvtepi32_ps(v_qf32);
        _mm512_storeu_ps(dst_row + x, v_out);
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
        int32_t QF32 = *val >> pLSB;
        dst_row[x]   = static_cast<float>(QF32);
      }
    }
  } else {
    // Irreversible path
    const __m512i v_M_b      = _mm512_set1_epi32(M_b);
    const __m512i v_30       = _mm512_set1_epi32(30);
    const __m512i v_1        = _mm512_set1_epi32(1);
    const __m512i v_pLSB_m1  = _mm512_set1_epi32(pLSB_m1);
    const __m512i v_scale    = _mm512_set1_epi32(scale);
    const __m512i v_round16  = _mm512_set1_epi32(1 << 15);
    const __m512i v_roundds  = _mm512_set1_epi32(1 << (downshift - 1));
    const __m512i v_int32max = _mm512_set1_epi32(INT32_MAX);
    const __m512i v_zero     = _mm512_setzero_si512();

    for (uint32_t y = 0; y < height; y++) {
      int32_t *val_row      = sample_buf + y * blksampl_stride;
      const uint8_t *st_row = block_states + (y + 1) * blkstate_stride + 1;
      sprec_t *dst_row      = i_samples + y * band_stride;
      uint32_t x            = 0;

      for (; x + 16 <= width; x += 16) {
        __m512i v_val = _mm512_loadu_si512(val_row + x);

        // Extract sign (bit 31) as mask: 0 or -1
        __m512i v_sign = _mm512_srai_epi32(v_val, 31);

        // Clear sign bit
        v_val = _mm512_and_si512(v_val, v_int32max);

        if (ROIshift) {
          // Upshift lanes where (val & ~mask) == 0. Pure vector via AVX-512 mask ops;
          // avoids the scalar store-loop-reload round-trip that MSVC miscompiles on
          // its AVX2 counterpart (see block_dequant_avx2.cpp).
          const __m512i v_notmask = _mm512_set1_epi32((int32_t)~mask);
          const __mmask16 k_roi =
              _mm512_cmpeq_epi32_mask(_mm512_and_si512(v_val, v_notmask), v_zero);
          v_val = _mm512_mask_slli_epi32(v_val, k_roi, v_val, ROIshift);
        }

        // Load state bytes and compute N_b
        __m512i v_state;
        if (ROIshift) {
          v_state = _mm512_set1_epi32(30 - pLSB + 1);
        } else {
          __m128i v_st8 = _mm_loadu_si128((__m128i *)(st_row + x));
          __m512i v_st  = _mm512_cvtepu8_epi32(v_st8);
          v_st          = _mm512_srli_epi32(v_st, 3);
          v_state       = _mm512_add_epi32(_mm512_sub_epi32(v_30, v_st), v_1);
        }

        // offset = max(M_b - N_b, 0)
        __m512i v_offset = _mm512_max_epi32(_mm512_sub_epi32(v_M_b, v_state), v_zero);

        // r_val = 1 << (pLSB - 1 + offset)
        __m512i v_shift = _mm512_add_epi32(v_pLSB_m1, v_offset);
        __m512i v_rval  = _mm512_sllv_epi32(v_1, v_shift);

        // Add r_val if val != 0 (irreversible always adds when non-zero)
        __mmask16 k_nonzero = _mm512_cmpgt_epi32_mask(v_val, v_zero);
        v_val = _mm512_mask_or_epi32(v_val, k_nonzero, v_val, v_rval);

        // Truncate: val = (val + (1<<15)) >> 16
        v_val = _mm512_srai_epi32(_mm512_add_epi32(v_val, v_round16), 16);

        // Dequantize: val *= scale
        v_val = _mm512_mullo_epi32(v_val, v_scale);

        // Downshift with rounding
        __m512i v_qf32 = _mm512_srai_epi32(_mm512_add_epi32(v_val, v_roundds), downshift);

        // Apply sign (branchless)
        v_qf32 = _mm512_sub_epi32(_mm512_xor_si512(v_qf32, v_sign), v_sign);

        // Convert to float and store
        __m512 v_out = _mm512_cvtepi32_ps(v_qf32);
        _mm512_storeu_ps(dst_row + x, v_out);
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

#endif  // OPENHTJ2K_TRY_AVX2 && __AVX512F__
