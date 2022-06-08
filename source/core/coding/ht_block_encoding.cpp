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

#include <algorithm>
#include "coding_units.hpp"
#include "ht_block_encoding.hpp"
#include "coding_local.hpp"
#include "enc_CxtVLC_tables.hpp"
#include "utils.hpp"

#ifdef _OPENMP
  #include <omp.h>
#endif

#define Q0 0
#define Q1 1

//#define HTSIMD
//#define ENABLE_SP_MR

void j2k_codeblock::set_MagSgn_and_sigma(uint32_t &or_val) {
  const uint32_t height = this->size.y;
  const uint32_t width  = this->size.x + (this->size.x % 2);
  const uint32_t stride = this->band_stride;
  const int32_t pshift  = (refsegment) ? 1 : 0;
  const int32_t pLSB    = (1 << (pshift - 1));

  for (uint16_t i = 0; i < static_cast<uint16_t>(height); ++i) {
    sprec_t *const sp  = this->i_samples + i * stride;
    int32_t *const dp  = this->sample_buf.get() + i * blksampl_stride;
    size_t block_index = (i + 1U) * (blkstate_stride) + 1U;
    uint8_t *dstblk    = block_states.get() + block_index;
#if defined(OPENHTJ2K_ENABLE_ARM_NEON)
    // vetorized for ARM NEON
    uint16x8_t vpLSB = vdupq_n_u16((uint16_t)pLSB);
    int16x8_t vone   = vdupq_n_s16(1);
    auto vzero       = vdupq_n_s16(0);
    // simd
    int16_t simdlen = round_down(static_cast<int16_t>(this->size.x), 8);
    for (int16_t j = 0; j < simdlen; j += 8) {
      int16x8_t coeff16   = vld1q_s16(sp + j);
      uint8x8_t vblkstate = vget_low_u8(vld1q_u8(dstblk + j));
      uint16x8_t vsign    = vcltzq_s16(coeff16) >> 15;
      uint8x8_t vsmag     = vmovn_u16(vandq_s16(coeff16, vpLSB));
      uint8x8_t vssgn     = vmovn_u16(vsign);
      vblkstate |= vsmag << SHIFT_SMAG;
      vblkstate |= vssgn << SHIFT_SSGN;
      int16x8_t vabsmag     = (vabsq_s16(coeff16) & 0x7FFF) >> pshift;
      vzero                 = vorrq_s16(vzero, vabsmag);
      int16x8_t vmasked_one = (vceqzq_s16(vabsmag) ^ 0xFFFF) & vone;
      vblkstate |= vmovn_u16(vmasked_one);
      vst1_u8(dstblk + j, vblkstate);
      vabsmag -= vmasked_one;
      vabsmag <<= vmasked_one;
      vabsmag += vsign * vmasked_one;
      int32x4_t coeff32     = vreinterpretq_s32_s16(vabsmag);
      int32x4_t coeff32low  = vmovl_s16(vreinterpret_s16_s32(vget_low_s32(coeff32)));
      int32x4_t coeff32high = vmovl_s16(vreinterpret_s16_s32(vget_high_s32(coeff32)));
      vst1q_s32(dp + j, coeff32low);
      vst1q_s32(dp + j + 4, coeff32high);
    }
    or_val |= static_cast<unsigned int>(vmaxvq_s16(vzero));
    // remaining
    for (int16_t j = simdlen; j < static_cast<int16_t>(this->size.x); ++j) {
      int32_t temp  = sp[j];
      uint32_t sign = static_cast<uint32_t>(temp) & 0x80000000;
      dstblk[j] |= static_cast<uint8_t>(((temp & pLSB) & 1) << SHIFT_SMAG);
      dstblk[j] |= static_cast<uint8_t>((sign >> 31) << SHIFT_SSGN);
      temp = (temp < 0) ? -temp : temp;
      temp &= 0x7FFFFFFF;
      temp >>= pshift;
      if (temp) {
        or_val |= 1;
        dstblk[j] |= 1;
        temp--;
        temp <<= 1;
        temp += static_cast<uint8_t>(sign >> 31);
        dp[j] = temp;
      }
    }

#elif defined(OPENHTJ2K_TRY_AVX2) && defined(__AVX2__)
    auto vpLSB     = _mm256_set1_epi16((int16_t)pLSB);
    auto vone      = _mm256_set1_epi16((int16_t)1);
    auto vtwo      = _mm256_set1_epi16((int16_t)2);
    auto vsignmask = _mm256_set1_epi16((int16_t)0x8000);
    auto vabsmask  = _mm256_set1_epi16((int16_t)0x7FFF);
    auto vzero     = _mm256_setzero_si256();
    // simd
    int32_t simdlen = round_down(static_cast<int32_t>(this->size.x), 16);
    for (int32_t j = 0; j < simdlen; j += 16) {
      auto coeff16 = _mm256_loadu_si256((__m256i *)(sp + j));
      // auto vsmag   = _mm256_and_si256(coeff16, vpLSB);
      auto vsign = _mm256_srli_epi16(_mm256_and_si256(coeff16, vsignmask), 15);
      auto vsmag = _mm256_slli_epi16(_mm256_and_si256(coeff16, vpLSB), SHIFT_SMAG);

      auto vabsmag = _mm256_srli_epi16(_mm256_and_si256(_mm256_abs_epi16(coeff16), vabsmask), pshift);

      auto vmasked_one =
          _mm256_and_si256(_mm256_xor_si256(_mm256_cmpeq_epi16(vabsmag, _mm256_setzero_si256()),
                                            _mm256_set1_epi16((int16_t)0xFFFF)),
                           vone);
      vzero            = _mm256_or_si256(vzero, vabsmag);
      auto vmasked_two = _mm256_mullo_epi16(vmasked_one, vtwo);
      auto vblkstate =
          _mm256_or_si256(vmasked_one, _mm256_or_si256(vsmag, _mm256_slli_epi16(vsign, SHIFT_SSGN)));
      auto vblkstate_low  = _mm256_extracti128_si256(vblkstate, 0);
      auto vblkstate_high = _mm256_extracti128_si256(vblkstate, 1);
      auto vblkstate_u8   = _mm_packs_epi16(vblkstate_low, vblkstate_high);
      _mm_storeu_si128((__m128i *)(dstblk + j), vblkstate_u8);
      auto vcoeff      = _mm256_add_epi16(_mm256_mullo_epi16(_mm256_subs_epu16(vabsmag, vone), vmasked_two),
                                          _mm256_mullo_epi16(vsign, vmasked_one));
      auto vcoeff_low  = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(vcoeff, 0));
      auto vcoeff_high = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(vcoeff, 1));
      _mm256_storeu_si256((__m256i *)(dp + j), vcoeff_low);
      _mm256_storeu_si256((__m256i *)(dp + j + 8), vcoeff_high);
    }
    or_val |= !_mm256_testz_si256(vzero, vabsmask);
    // remaining
    for (int32_t j = simdlen; j < static_cast<int32_t>(this->size.x); ++j) {
      int32_t temp  = sp[j];
      uint32_t sign = (static_cast<uint32_t>(temp) & 0x80000000) >> 31;
      dstblk[j] |= static_cast<uint8_t>((temp & pLSB) << SHIFT_SMAG);
      dstblk[j] |= static_cast<uint8_t>((sign) << SHIFT_SSGN);
      temp = (temp < 0) ? -temp : temp;
      temp &= 0x7FFFFFFF;
      temp >>= pshift;
      if (temp) {
        or_val |= 1;
        dstblk[j] |= 1;
        temp--;
        temp <<= 1;
        temp  = temp + static_cast<int32_t>(sign);
        dp[j] = temp;
      }
      block_index++;
    }
#else
    for (uint16_t j = 0; j < width; ++j) {
      int32_t temp  = sp[j];
      uint32_t sign = static_cast<uint32_t>(temp) & 0x80000000;
      dstblk[j] |= (temp & pLSB) << SHIFT_SMAG;
      dstblk[j] |= (sign >> 31) << SHIFT_SSGN;
      temp = (temp < 0) ? -temp : temp;
      temp &= 0x7FFFFFFF;
      temp >>= pshift;
      if (temp) {
        or_val |= 1;
        dstblk[j] |= 1;
        // convert sample value to MagSgn
        //        temp = (temp < 0) ? -temp : temp;
        //        temp &= 0x7FFFFFFF;
        temp--;
        temp <<= 1;
        temp += sign >> 31;
        dp[j] = temp;
      }
      block_index++;
    }
#endif
  }
}

/********************************************************************************
 * state_MS_enc: member functions
 *******************************************************************************/
#ifdef MSNAIVE
void state_MS_enc::emitMagSgnBits(uint32_t cwd, uint8_t len) {
  /* naive implementation */
  uint8_t b;
  for (; len > 0;) {
    b = cwd & 1;
    cwd >>= 1;
    --len;
    tmp |= b << bits;
    bits++;
    if (bits == max) {
      buf[pos] = tmp;
      pos++;
      max  = (tmp == 0xFF) ? 7 : 8;
      tmp  = 0;
      bits = 0;
    }
  }
  /* slightly faster implementation */
  //  for (; len > 0;) {
  //    int32_t t = std::min(max - bits, (int32_t)len);
  //    tmp |= (cwd & ((1 << t) - 1)) << bits;
  //    bits += t;
  //    cwd >>= t;
  //    len -= t;
  //    if (bits >= max) {
  //      buf[pos] = tmp;
  //      pos++;
  //      max  = (tmp == 0xFF) ? 7 : 8;
  //      tmp  = 0;
  //      bits = 0;
  //    }
  //  }
}
#else
void state_MS_enc::emitMagSgnBits(uint32_t cwd, uint8_t len, uint8_t emb_1) {
  int32_t temp = emb_1 << len;
  cwd -= static_cast<uint32_t>(temp);
  Creg |= static_cast<uint64_t>(cwd) << ctreg;
  ctreg += len;
  while (ctreg >= 32) {
    emit_dword();
  }
}
void state_MS_enc::emit_dword() {
  for (int i = 0; i < 4; ++i) {
    if (last == 0xFF) {
      last = static_cast<uint8_t>(Creg & 0x7F);
      Creg >>= 7;
      ctreg -= 7;
    } else {
      last = static_cast<uint8_t>(Creg & 0xFF);
      Creg >>= 8;
      ctreg -= 8;
    }
    buf[pos++] = last;
  }
}
#endif

int32_t state_MS_enc::termMS() {
#ifdef MSNAIVE
  /* naive implementation */
  if (bits > 0) {
    for (; bits < max; bits++) {
      tmp |= 1 << bits;
    }
    if (tmp != 0xFF) {
      buf[pos] = tmp;
      pos++;
    }
  } else if (max == 7) {
    pos--;
  }
#else
  while (true) {
    if (last == 0xFF) {
      if (ctreg < 7) break;
      last = static_cast<uint8_t>(Creg & 0x7F);
      Creg >>= 7;
      ctreg -= 7;
    } else {
      if (ctreg < 8) break;
      last = static_cast<uint8_t>(Creg & 0xFF);
      Creg >>= 8;
      ctreg -= 8;
    }
    buf[pos++] = last;
  }
  bool last_was_FF = (last == 0xFF);
  uint8_t fill_mask, cwd;
  if (ctreg > 0) {
    fill_mask = static_cast<uint8_t>(0xFF << ctreg);
    if (last_was_FF) {
      fill_mask &= 0x7F;
    }
    cwd = static_cast<uint8_t>(Creg |= fill_mask);
    if (cwd != 0xFF) {
      buf[pos++] = cwd;
    }
  } else if (last_was_FF) {
    pos--;
    buf[pos] = 0x00;  // may be not necessary
  }
#endif
  return pos;  // return current position as Pcup
}

/********************************************************************************
 * state_MEL_enc: member functions
 *******************************************************************************/
void state_MEL_enc::emitMELbit(uint8_t bit) {
  tmp = static_cast<uint8_t>((tmp << 1) + bit);
  rem--;
  if (rem == 0) {
    buf[pos] = tmp;
    pos++;
    rem = (tmp == 0xFF) ? 7 : 8;
    tmp = 0;
  }
}

void state_MEL_enc::encodeMEL(uint8_t smel) {
  uint8_t eval;
  switch (smel) {
    case 0:
      MEL_run++;
      if (MEL_run >= MEL_t) {
        emitMELbit(1);
        MEL_run = 0;
        MEL_k   = (int8_t)std::min(12, MEL_k + 1);
        eval    = MEL_E[MEL_k];
        MEL_t   = static_cast<uint8_t>(1 << eval);
      }
      break;

    default:
      emitMELbit(0);
      eval = MEL_E[MEL_k];
      while (eval > 0) {
        eval--;
        // (MEL_run >> eval) & 1 = msb
        emitMELbit((MEL_run >> eval) & 1);
      }
      MEL_run = 0;
      MEL_k   = (int8_t)std::max(0, MEL_k - 1);
      eval    = MEL_E[MEL_k];
      MEL_t   = static_cast<uint8_t>(1 << eval);
      break;
  }
}

void state_MEL_enc::termMEL() {
  if (MEL_run > 0) {
    emitMELbit(1);
  }
}

/********************************************************************************
 * state_VLC_enc: member functions
 *******************************************************************************/
void state_VLC_enc::emitVLCBits(uint16_t cwd, uint8_t len) {
  int32_t len32 = len;
  for (; len32 > 0;) {
    int32_t available_bits = 8 - (last > 0x8F) - bits;
    int32_t t              = std::min(available_bits, len32);
    tmp |= static_cast<uint8_t>((cwd & ((1 << t) - 1)) << bits);
    bits = static_cast<uint8_t>(bits + t);
    available_bits -= t;
    len32 -= t;
    cwd = static_cast<uint16_t>(cwd >> t);
    if (available_bits == 0) {
      if ((last > 0x8f) && tmp != 0x7F) {
        last = 0x00;
        continue;
      }
      buf[pos] = tmp;
      pos--;  // reverse order
      last = tmp;
      tmp  = 0;
      bits = 0;
    }
  }
  //  uint8_t b;
  //  for (; len > 0;) {
  //    b = cwd & 1;
  //    cwd >>= 1;
  //    len--;
  //    tmp |= b << bits;
  //    bits++;
  //    if ((last > 0x8F) && (tmp == 0x7F)) {
  //      bits++;
  //    }
  //    if (bits == 8) {
  //      buf[pos] = tmp;
  //      pos--;  // reverse order
  //      last = tmp;
  //      tmp  = 0;
  //      bits = 0;
  //    }
  //  }
}

/********************************************************************************
 * HT cleanup encoding: helper functions
 *******************************************************************************/
#if defined(OPENHTJ2K_TRY_AVX2) && defined(__AVX2__)
// the following two functions are taken from https://primenumber.hatenadiary.jp/entry/2016/12/13/011832
inline __m256i bsr_256_32_cvtfloat_impl(__m256i x, int32_t sub) {
  __m256i cvt_fl  = _mm256_castps_si256(_mm256_cvtepi32_ps(x));
  __m256i shifted = _mm256_srli_epi32(cvt_fl, 23);
  return _mm256_sub_epi32(shifted, _mm256_set1_epi32(sub));
}
inline __m256i bsr_256_32_cvtfloat(__m256i x) {
  x              = _mm256_andnot_si256(_mm256_srli_epi32(x, 1), x);
  __m256i result = bsr_256_32_cvtfloat_impl(x, 127);
  result         = _mm256_or_si256(result, _mm256_srai_epi32(x, 31));
  return _mm256_and_si256(result, _mm256_set1_epi32(0x0000001F));
}
#endif
auto make_storage = [](const j2k_codeblock *const block, const uint16_t qy, const uint16_t qx,
                       uint8_t *const sigma_n, uint32_t *const v_n, int32_t *const E_n,
                       uint8_t *const rho_q) {
// This function shall be called on the assumption that there are two quads
#if defined(OPENHTJ2K_ENABLE_ARM_NEON)
  const uint32_t QWx2                = block->size.x + block->size.x % 2;
  alignas(32) const int8_t nshift[8] = {0, 1, 2, 3, 0, 1, 2, 3};
  uint8_t *const sp0 = block->block_states.get() + (2 * qy + 1U) * (block->blkstate_stride) + 2 * qx + 1;
  uint8_t *const sp1 = block->block_states.get() + (2 * qy + 2U) * (block->blkstate_stride) + 2 * qx + 1;
  auto v_u8_0        = vld1_u8(sp0);
  auto v_u8_1        = vld1_u8(sp1);
  auto v_u8_zip      = vzip1_u8(v_u8_0, v_u8_1);
  auto vmask         = vdup_n_u8(1);
  auto v_u8_out      = vand_u8(v_u8_zip, vmask);
  vst1_u8(sigma_n, v_u8_out);
  auto v_u8_shift = vld1_s8(nshift);
  auto vtmp       = vpadd_u8(vpadd_u8(vshl_u8(v_u8_out, v_u8_shift), vshl_u8(v_u8_out, v_u8_shift)),
                             vpadd_u8(vshl_u8(v_u8_out, v_u8_shift), vshl_u8(v_u8_out, v_u8_shift)));
  rho_q[0]        = vdupb_lane_u8(vtmp, 0);
  rho_q[1]        = vdupb_lane_u8(vtmp, 1);
  auto v_s32_0    = vld1q_s32(block->sample_buf.get() + 2 * qx + 2 * qy * block->blksampl_stride);
  auto v_s32_1    = vld1q_s32(block->sample_buf.get() + 2 * qx + (2 * qy + 1U) * block->blksampl_stride);
  auto v_s32_out  = vzipq_s32(v_s32_0, v_s32_1);
  vst1q_u32(v_n, v_s32_out.val[0]);
  vst1q_u32(v_n + 4, v_s32_out.val[1]);
  auto vsig0 = vmovl_u16(vget_low_u16(vmovl_u8(v_u8_out)));
  auto vsig1 = vmovl_u16(vget_high_u16(vmovl_u8(v_u8_out)));
  vst1q_s32(E_n, (32 - vclzq_u32(vshlq_n_s32(vshrq_n_s32(v_s32_out.val[0], 1), 1) + 1)) * vsig0);
  vst1q_s32(E_n + 4, (32 - vclzq_u32(vshlq_n_s32(vshrq_n_s32(v_s32_out.val[1], 1), 1) + 1)) * vsig1);
#elif defined(OPENHTJ2K_TRY_AVX2) && defined(__AVX2__)
  const uint32_t QWx2 = block->size.x + block->size.x % 2;
  // const int8_t nshift[8] = {0, 1, 2, 3, 0, 1, 2, 3};
  uint8_t *const sp0 = block->block_states.get() + (2U * qy + 1U) * (block->blkstate_stride) + 2U * qx + 1U;
  uint8_t *const sp1 = block->block_states.get() + (2U * qy + 2U) * (block->blkstate_stride) + 2U * qx + 1U;
  auto v_u8_0 = _mm_set1_epi64x(*((int64_t *)sp0));
  auto v_u8_1 = _mm_set1_epi64x(*((int64_t *)sp1));
  auto v_u8_zip = _mm_unpacklo_epi8(v_u8_0, v_u8_1);
  auto vmask = _mm_set1_epi8(1);
  auto v_u8_out = _mm_and_si128(v_u8_zip, vmask);
  *((int64_t *)sigma_n) = _mm_cvtsi128_si64(v_u8_out);

  rho_q[0] = static_cast<uint8_t>(sigma_n[0] + (sigma_n[1] << 1) + (sigma_n[2] << 2) + (sigma_n[3] << 3));
  rho_q[1] = static_cast<uint8_t>(sigma_n[4] + (sigma_n[5] << 1) + (sigma_n[6] << 2) + (sigma_n[7] << 3));

  alignas(32) uint32_t sig32[8];
  _mm256_store_si256(((__m256i *)sig32), _mm256_cvtepu8_epi32(v_u8_out));
  auto v_s32_0 =
      _mm_loadu_si128((__m128i *)(block->sample_buf.get() + 2U * qx + 2U * qy * block->blksampl_stride));
  auto v_s32_1 = _mm_loadu_si128(
      (__m128i *)(block->sample_buf.get() + 2U * qx + (2U * qy + 1U) * block->blksampl_stride));
  *((__m128i *)(v_n)) = _mm_unpacklo_epi32(v_s32_0, v_s32_1);
  *((__m128i *)(v_n + 4)) = _mm_unpackhi_epi32(v_s32_0, v_s32_1);

  auto vv_n = _mm256_load_si256((__m256i *)v_n);
  auto vsig = _mm256_load_si256((__m256i *)sig32);
  auto vone = _mm256_set1_epi32(1);
  // auto v32 = _mm256_set1_epi32(32);
  auto vvv = bsr_256_32_cvtfloat(_mm256_add_epi32(_mm256_slli_epi32(_mm256_srai_epi32(vv_n, 1), 1), vone));
  auto vtmp = _mm256_mullo_epi32(_mm256_add_epi32(vvv, vone), vsig);
  _mm256_store_si256((__m256i *)E_n, vtmp);
#else
  const int32_t x[8] = {2 * qx,       2 * qx,       2 * qx + 1,       2 * qx + 1,
                        2 * (qx + 1), 2 * (qx + 1), 2 * (qx + 1) + 1, 2 * (qx + 1) + 1};
  const int32_t y[8] = {2 * qy, 2 * qy + 1, 2 * qy, 2 * qy + 1, 2 * qy, 2 * qy + 1, 2 * qy, 2 * qy + 1};
  // First quad
  for (int i = 0; i < 4; ++i) {
    sigma_n[i] = block->get_state(Sigma, (int16_t)y[i], (int16_t)x[i]);
  }
  // Second quad
  for (int i = 4; i < 8; ++i) {
    sigma_n[i] = block->get_state(Sigma, (int16_t)y[i], (int16_t)x[i]);
  }
  rho_q[0] = static_cast<uint8_t>(sigma_n[0] + (sigma_n[1] << 1) + (sigma_n[2] << 2) + (sigma_n[3] << 3));
  rho_q[1] = static_cast<uint8_t>(sigma_n[4] + (sigma_n[5] << 1) + (sigma_n[6] << 2) + (sigma_n[7] << 3));
  for (int i = 0; i < 8; ++i) {
    v_n[i] = static_cast<uint32_t>(block->sample_buf[(uint32_t)x[i] + (uint32_t)y[i] * block->size.x]);
  }
  for (int i = 0; i < 8; ++i) {
    E_n[i] = static_cast<int32_t>((32 - count_leading_zeros(((v_n[i] >> 1) << 1) + 1)) * sigma_n[i]);
  }
#endif
};

static inline void make_storage_one(const j2k_codeblock *const block, const uint16_t qy, const uint16_t qx,
                                    uint8_t *const sigma_n, uint32_t *const v_n, int32_t *const E_n,
                                    uint8_t *const rho_q) {
  const int16_t x[4] = {static_cast<int16_t>(2 * qx), static_cast<int16_t>(2 * qx),
                        static_cast<int16_t>(2 * qx + 1), static_cast<int16_t>(2 * qx + 1)};
  const int16_t y[4] = {static_cast<int16_t>(2 * qy), static_cast<int16_t>(2 * qy + 1),
                        static_cast<int16_t>(2 * qy), static_cast<int16_t>(2 * qy + 1)};

  for (int i = 0; i < 4; ++i) {
    if ((x[i] >= 0 && x[i] < static_cast<int16_t>(block->size.x))
        && (y[i] >= 0 && y[i] < static_cast<int16_t>(block->size.y))) {
      sigma_n[i] = block->get_state(Sigma, y[i], x[i]);
    } else {
      sigma_n[i] = 0;
    }
  }
  rho_q[0] = static_cast<uint8_t>(sigma_n[0] + (sigma_n[1] << 1) + (sigma_n[2] << 2) + (sigma_n[3] << 3));

  for (int i = 0; i < 4; ++i) {
    if ((x[i] >= 0 && x[i] < static_cast<int16_t>(block->size.x))
        && (y[i] >= 0 && y[i] < static_cast<int16_t>(block->size.y))) {
      v_n[i] = static_cast<uint32_t>(
          block->sample_buf[(uint32_t)x[i] + (uint32_t)y[i] * (block->blksampl_stride)]);
    } else {
      v_n[i] = 0;
    }
  }

  for (int i = 0; i < 4; ++i) {
    E_n[i] = static_cast<int32_t>((32 - count_leading_zeros(((v_n[i] >> 1) << 1) + 1)) * sigma_n[i]);
  }
}

// UVLC encoding for initial line pair
auto encode_UVLC0 = [](uint16_t &cwd, uint8_t &lw, int32_t u1, int32_t u2 = 0) {
  int32_t tmp;
  tmp = (int32_t)enc_UVLC_table0[u1 + (u2 << 5)];
  lw  = static_cast<uint8_t>(tmp & 0xFF);
  cwd = static_cast<uint16_t>(tmp >> 8);
};

// UVLC encoding for non-initial line pair
auto encode_UVLC1 = [](uint16_t &cwd, uint8_t &lw, int32_t u1, int32_t u2 = 0) {
  int32_t tmp;
  tmp = (int32_t)enc_UVLC_table1[u1 + (u2 << 5)];
  lw  = static_cast<uint8_t>(tmp & 0xFF);
  cwd = static_cast<uint16_t>(tmp >> 8);
};

// joint termination of MEL and VLC
int32_t termMELandVLC(state_VLC_enc &VLC, state_MEL_enc &MEL) {
  uint8_t MEL_mask, VLC_mask, fuse;
  MEL.tmp  = static_cast<uint8_t>(MEL.tmp << MEL.rem);
  MEL_mask = static_cast<uint8_t>((0xFF << MEL.rem) & 0xFF);
  VLC_mask = static_cast<uint8_t>(0xFF >> (8 - VLC.bits));
  if ((MEL_mask | VLC_mask) != 0) {
    fuse = MEL.tmp | VLC.tmp;
    if (((((fuse ^ MEL.tmp) & MEL_mask) | ((fuse ^ VLC.tmp) & VLC_mask)) == 0) && (fuse != 0xFF)) {
      MEL.buf[MEL.pos] = fuse;
    } else {
      MEL.buf[MEL.pos] = MEL.tmp;
      VLC.buf[VLC.pos] = VLC.tmp;
      VLC.pos--;  // reverse order
    }
    MEL.pos++;
  }
  // concatenate MEL and VLC buffers
  memmove(&MEL.buf[MEL.pos], &VLC.buf[VLC.pos + 1], static_cast<size_t>(MAX_Scup - VLC.pos - 1));
  // return Scup
  return (MEL.pos + MAX_Scup - VLC.pos - 1);
}

// joint termination of SP and MR
int32_t termSPandMR(SP_enc &SP, MR_enc &MR) {
  uint8_t SP_mask = static_cast<uint8_t>(0xFF >> (8 - SP.bits));  // if SP_bits is 0, SP_mask = 0
  SP_mask =
      static_cast<uint8_t>(SP_mask | ((1 << SP.max) & 0x80));  // Auguments SP_mask to cover any stuff bit
  uint8_t MR_mask = static_cast<uint8_t>(0xFF >> (8 - MR.bits));  // if MR_bits is 0, MR_mask = 0
  if ((SP_mask | MR_mask) == 0) {
    // last SP byte cannot be 0xFF, since then SP_max would be 7
    memmove(&SP.buf[SP.pos], &MR.buf[MR.pos + 1], MAX_Lref - MR.pos);
    return static_cast<int32_t>(SP.pos + MAX_Lref - MR.pos);
  }
  uint8_t fuse = SP.tmp | MR.tmp;
  if ((((fuse ^ SP.tmp) & SP_mask) | ((fuse ^ MR.tmp) & MR_mask)) == 0) {
    SP.buf[SP.pos] = fuse;  // fuse always < 0x80 here; no false marker risk
  } else {
    SP.buf[SP.pos] = SP.tmp;  // SP_tmp cannot be 0xFF
    MR.buf[MR.pos] = MR.tmp;
    MR.pos--;  // MR buf gorws reverse order
  }
  SP.pos++;
  memmove(&SP.buf[SP.pos], &MR.buf[MR.pos + 1], MAX_Lref - MR.pos);
  return static_cast<int32_t>(SP.pos + MAX_Lref - MR.pos);
}

#if defined(OPENHTJ2K_ENABLE_ARM_NEON)
  #define MAKE_STORAGE()                                                                                    \
    {                                                                                                       \
      const uint32_t QWx2    = block->size.x + block->size.x % 2;                                           \
      const int8_t nshift[8] = {0, 1, 2, 3, 0, 1, 2, 3};                                                    \
      uint8_t *const sp0     = block->block_states.get() + (2 * qy + 1) * (block->size.x + 2) + 2 * qx + 1; \
      uint8_t *const sp1     = block->block_states.get() + (2 * qy + 2) * (block->size.x + 2) + 2 * qx + 1; \
      auto v_u8_0            = vld1_u8(sp0);                                                                \
      auto v_u8_1            = vld1_u8(sp1);                                                                \
      auto v_u8_zip          = vzip1_u8(v_u8_0, v_u8_1);                                                    \
      auto vmask             = vdup_n_u8(1);                                                                \
      auto v_u8_out          = vand_u8(v_u8_zip, vmask);                                                    \
      vst1_u8(sigma_n, v_u8_out);                                                                           \
      auto v_u8_shift = vld1_s8(nshift);                                                                    \
      auto vtmp       = vpadd_u8(vpadd_u8(vshl_u8(v_u8_out, v_u8_shift), vshl_u8(v_u8_out, v_u8_shift)),    \
                                 vpadd_u8(vshl_u8(v_u8_out, v_u8_shift), vshl_u8(v_u8_out, v_u8_shift)));   \
      rho_q[0]        = vdupb_lane_u8(vtmp, 0);                                                             \
      rho_q[1]        = vdupb_lane_u8(vtmp, 1);                                                             \
      auto v_s32_0    = vld1q_s32(block->sample_buf.get() + 2 * qx + 2 * qy * QWx2);                        \
      auto v_s32_1    = vld1q_s32(block->sample_buf.get() + 2 * qx + (2 * qy + 1) * QWx2);                  \
      auto v_s32_out  = vzipq_s32(v_s32_0, v_s32_1);                                                        \
      vst1q_u32(v_n, v_s32_out.val[0]);                                                                     \
      vst1q_u32(v_n + 4, v_s32_out.val[1]);                                                                 \
      auto vsig0 = vmovl_u16(vget_low_u16(vmovl_u8(v_u8_out)));                                             \
      auto vsig1 = vmovl_u16(vget_high_u16(vmovl_u8(v_u8_out)));                                            \
      vst1q_s32(E_n, (32 - vclzq_u32(vshlq_n_s32(vshrq_n_s32(v_s32_out.val[0], 1), 1) + 1)) * vsig0);       \
      vst1q_s32(E_n + 4, (32 - vclzq_u32(vshlq_n_s32(vshrq_n_s32(v_s32_out.val[1], 1), 1) + 1)) * vsig1);   \
    }
#else
  #define MAKE_STORAGE()                                                                             \
    {                                                                                                \
      const int32_t x[8] = {2 * qx,       2 * qx,       2 * qx + 1,       2 * qx + 1,                \
                            2 * (qx + 1), 2 * (qx + 1), 2 * (qx + 1) + 1, 2 * (qx + 1) + 1};         \
      const int32_t y[8] = {                                                                         \
          2 * qy, 2 * qy + 1, 2 * qy, 2 * qy + 1, 2 * qy, 2 * qy + 1, 2 * qy, 2 * qy + 1};           \
      for (int i = 0; i < 4; ++i)                                                                    \
        sigma_n[i] =                                                                                 \
            (block->block_states[(y[i] + 1) * (block->size.x + 2) + (x[i] + 1)] >> SHIFT_SIGMA) & 1; \
      rho_q[0] = sigma_n[0] + (sigma_n[1] << 1) + (sigma_n[2] << 2) + (sigma_n[3] << 3);             \
      for (int i = 4; i < 8; ++i)                                                                    \
        sigma_n[i] =                                                                                 \
            (block->block_states[(y[i] + 1) * (block->size.x + 2) + (x[i] + 1)] >> SHIFT_SIGMA) & 1; \
      rho_q[1] = sigma_n[4] + (sigma_n[5] << 1) + (sigma_n[6] << 2) + (sigma_n[7] << 3);             \
      for (int i = 0; i < 8; ++i) {                                                                  \
        if ((x[i] >= 0 && x[i] < (block->size.x)) && (y[i] >= 0 && y[i] < (block->size.y)))          \
          v_n[i] = block->sample_buf[x[i] + y[i] * block->size.x];                                   \
        else                                                                                         \
          v_n[i] = 0;                                                                                \
      }                                                                                              \
      for (int i = 0; i < 8; ++i)                                                                    \
        E_n[i] = (32 - count_leading_zeros(((v_n[i] >> 1) << 1) + 1)) * sigma_n[i];                  \
    }
#endif

/********************************************************************************
 * HT cleanup encoding
 *******************************************************************************/
int32_t htj2k_cleanup_encode(j2k_codeblock *const block, const uint8_t ROIshift) noexcept {
  // length of HT cleanup pass
  int32_t Lcup;
  // length of MagSgn buffer
  int32_t Pcup;
  // length of MEL buffer + VLC buffer
  int32_t Scup;
  // used as a flag to invoke HT Cleanup encoding
  uint32_t or_val = 0;
  if (ROIshift) {
    printf("WARNING: Encoding with ROI is not supported.\n");
  }

  const uint16_t QW = static_cast<uint16_t>(ceil_int(static_cast<int16_t>(block->size.x), 2));
  const uint16_t QH = static_cast<uint16_t>(ceil_int(static_cast<int16_t>(block->size.y), 2));

  block->set_MagSgn_and_sigma(or_val);

  if (!or_val) {
    // nothing to do here because this codeblock is empty
    // set length of coding passes
    block->length         = 0;
    block->pass_length[0] = 0;
    // set number of coding passes
    block->num_passes      = 0;
    block->layer_passes[0] = 0;
    block->layer_start[0]  = 0;
    // set number of zero-bitplanes (=Zblk)
    block->num_ZBP = static_cast<uint8_t>(block->get_Mb() - 1);
    return static_cast<int32_t>(block->length);
  }

  // buffers shall be zeroed.
  std::unique_ptr<uint8_t[]> fwd_buf = MAKE_UNIQUE<uint8_t[]>(MAX_Lcup);
  std::unique_ptr<uint8_t[]> rev_buf = MAKE_UNIQUE<uint8_t[]>(MAX_Scup);
  memset(fwd_buf.get(), 0, sizeof(uint8_t) * (MAX_Lcup));
  memset(rev_buf.get(), 0, sizeof(uint8_t) * MAX_Scup);

  state_MS_enc MagSgn_encoder(fwd_buf.get());
  state_MEL_enc MEL_encoder(rev_buf.get());
  state_VLC_enc VLC_encoder(rev_buf.get());

  alignas(32) uint32_t v_n[8];
  std::unique_ptr<int32_t[]> Eadj = MAKE_UNIQUE<int32_t[]>(round_up(block->size.x, 2U) + 2);
  memset(Eadj.get(), 0, round_up(block->size.x, 2U) + 2);
  std::unique_ptr<uint8_t[]> sigma_adj = MAKE_UNIQUE<uint8_t[]>(round_up(block->size.x, 2U) + 2);
  memset(sigma_adj.get(), 0, round_up(block->size.x, 2U) + 2);
  alignas(32) uint8_t sigma_n[8] = {0}, rho_q[2] = {0}, gamma[2] = {0}, emb_k, emb_1, lw, m_n[8] = {0};
  alignas(32) uint16_t c_q[2] = {0, 0}, n_q[2] = {0}, CxtVLC[2] = {0}, cwd;
  alignas(32) int32_t E_n[8] = {0}, Emax_q[2] = {0}, U_q[2] = {0}, u_q[2] = {0}, uoff_q[2] = {0},
                      emb[2] = {0}, kappa = 1;

  // Initial line pair
  int32_t *ep = Eadj.get();
  ep++;
  uint8_t *sp = sigma_adj.get();
  sp++;
  for (uint16_t qx = 0; qx < QW - 1; qx = static_cast<uint16_t>(qx + 2U)) {
    const int16_t qy = 0;
    // MAKE_STORAGE()
    make_storage(block, qy, qx, sigma_n, v_n, E_n, rho_q);
    // MEL encoding for the first quad
    if (c_q[Q0] == 0) {
      MEL_encoder.encodeMEL((rho_q[Q0] != 0));
    }

    Emax_q[Q0] = find_max(E_n[0], E_n[1], E_n[2], E_n[3]);
    U_q[Q0]    = std::max((int32_t)Emax_q[Q0], kappa);
    u_q[Q0]    = U_q[Q0] - kappa;
    uoff_q[Q0] = (u_q[Q0]) ? 1 : 0;
#if defined(OPENHTJ2K_ENABLE_ARM_NEON)
    auto vE_n     = vld1q_s32(E_n);
    auto vEmax_q  = vdupq_n_s32(Emax_q[Q0]);
    auto vuoff_q  = vdupq_n_s32(uoff_q[Q0]);
    auto vmask    = vceqq_s32(vE_n, vEmax_q);
    int32_t vs[4] = {1, 2, 4, 8};
    auto vshift   = vld1q_s32(vs);
    emb[Q0]       = vaddvq_s32(vmulq_s32(vuoff_q, vshift) & vmask);
#elif defined(OPENHTJ2K_TRY_AVX2) && defined(__AVX2__)
    auto vE_n = _mm_load_si128((__m128i *)E_n);
    auto vEmax_q = _mm_set1_epi32(Emax_q[Q0]);
    auto vuoff_q = _mm_set1_epi32(uoff_q[Q0]);
    auto vmask = _mm_cmpeq_epi32(vE_n, vEmax_q);
    auto vshift = _mm_setr_epi32(0, 1, 2, 3);
    auto vtmp = _mm_and_si128(_mm_sllv_epi32(vuoff_q, vshift), vmask);
    emb[Q0] = _mm_cvtsi128_si32(_mm_hadd_epi32(_mm_hadd_epi32(vtmp, vtmp), _mm_hadd_epi32(vtmp, vtmp)));
#else
    emb[Q0] = (E_n[0] == Emax_q[Q0]) ? uoff_q[Q0] : 0;
    emb[Q0] += (E_n[1] == Emax_q[Q0]) ? uoff_q[Q0] << 1 : 0;
    emb[Q0] += (E_n[2] == Emax_q[Q0]) ? uoff_q[Q0] << 2 : 0;
    emb[Q0] += (E_n[3] == Emax_q[Q0]) ? uoff_q[Q0] << 3 : 0;
#endif

    n_q[Q0]    = static_cast<uint16_t>(emb[Q0] + (rho_q[Q0] << 4) + (c_q[Q0] << 8));
    CxtVLC[Q0] = enc_CxtVLC_table0[n_q[Q0]];
    emb_k      = CxtVLC[Q0] & 0xF;
    emb_1      = static_cast<uint8_t>(n_q[Q0] % 16 & emb_k);

    for (int i = 0; i < 4; ++i) {
      m_n[i] = static_cast<uint8_t>(sigma_n[i] * U_q[Q0] - ((emb_k >> i) & 1));
#ifdef MSNAIVE
      MagSgn_encoder.emitMagSgnBits(v_n[i], m_n[i]);
#else
      MagSgn_encoder.emitMagSgnBits(v_n[i], m_n[i], (emb_1 >> i) & 1);
#endif
    }

    CxtVLC[Q0] = static_cast<uint16_t>(CxtVLC[Q0] >> 4);
    lw         = CxtVLC[Q0] & 0x07;
    CxtVLC[Q0] = static_cast<uint16_t>(CxtVLC[Q0] >> 3);
    cwd        = CxtVLC[Q0];

    ep[2 * qx]     = E_n[1];
    ep[2 * qx + 1] = E_n[3];

    sp[2 * qx]     = sigma_n[1];
    sp[2 * qx + 1] = sigma_n[3];

    VLC_encoder.emitVLCBits(cwd, lw);

    // context for 1st quad of next quad-pair
    c_q[Q0] = static_cast<uint16_t>((sigma_n[4] | sigma_n[5]) + (sigma_n[6] << 1) + (sigma_n[7] << 2));
    // context for 2nd quad of current quad pair
    c_q[Q1] = static_cast<uint16_t>((sigma_n[0] | sigma_n[1]) + (sigma_n[2] << 1) + (sigma_n[3] << 2));

    Emax_q[Q1] = find_max(E_n[4], E_n[5], E_n[6], E_n[7]);
    U_q[Q1]    = std::max((int32_t)Emax_q[Q1], kappa);
    u_q[Q1]    = U_q[Q1] - kappa;
    uoff_q[Q1] = (u_q[Q1]) ? 1 : 0;
    // MEL encoding of the second quad
    if (c_q[Q1] == 0) {
      if (rho_q[Q1] != 0) {
        MEL_encoder.encodeMEL(1);
      } else {
        if (std::min(u_q[Q0], u_q[Q1]) > 2) {
          MEL_encoder.encodeMEL(1);
        } else {
          MEL_encoder.encodeMEL(0);
        }
      }
    } else if (uoff_q[Q0] == 1 && uoff_q[Q1] == 1) {
      if (std::min(u_q[Q0], u_q[Q1]) > 2) {
        MEL_encoder.encodeMEL(1);
      } else {
        MEL_encoder.encodeMEL(0);
      }
    }
#if defined(OPENHTJ2K_ENABLE_ARM_NEON)
    vE_n    = vld1q_s32(E_n + 4);
    vEmax_q = vdupq_n_s32(Emax_q[Q1]);
    vuoff_q = vdupq_n_s32(uoff_q[Q1]);
    vmask   = vceqq_s32(vE_n, vEmax_q);
    emb[Q1] = vaddvq_s32(vmulq_s32(vuoff_q, vshift) & vmask);
#elif defined(OPENHTJ2K_TRY_AVX2) && defined(__AVX2__)
    vE_n = _mm_load_si128((__m128i *)(E_n + 4));
    vEmax_q = _mm_set1_epi32(Emax_q[Q1]);
    vuoff_q = _mm_set1_epi32(uoff_q[Q1]);
    vmask = _mm_cmpeq_epi32(vE_n, vEmax_q);
    vtmp = _mm_and_si128(_mm_sllv_epi32(vuoff_q, vshift), vmask);
    emb[Q1] = _mm_cvtsi128_si32(_mm_hadd_epi32(_mm_hadd_epi32(vtmp, vtmp), _mm_hadd_epi32(vtmp, vtmp)));
#else
    emb[Q1] = (E_n[4] == Emax_q[Q1]) ? uoff_q[Q1] : 0;
    emb[Q1] += (E_n[5] == Emax_q[Q1]) ? uoff_q[Q1] << 1 : 0;
    emb[Q1] += (E_n[6] == Emax_q[Q1]) ? uoff_q[Q1] << 2 : 0;
    emb[Q1] += (E_n[7] == Emax_q[Q1]) ? uoff_q[Q1] << 3 : 0;
#endif
    n_q[Q1]    = static_cast<uint16_t>(emb[Q1] + (rho_q[Q1] << 4) + (c_q[Q1] << 8));
    CxtVLC[Q1] = enc_CxtVLC_table0[n_q[Q1]];
    emb_k      = CxtVLC[Q1] & 0xF;
    emb_1      = static_cast<uint8_t>(n_q[Q1] % 16 & emb_k);
    for (int i = 0; i < 4; ++i) {
      m_n[4 + i] = static_cast<uint8_t>(sigma_n[4 + i] * U_q[Q1] - ((emb_k >> i) & 1));
#ifdef MSNAIVE
      MagSgn_encoder.emitMagSgnBits(v_n[4 + i], m_n[4 + i]);
#else
      MagSgn_encoder.emitMagSgnBits(v_n[4 + i], m_n[4 + i], (emb_1 >> i) & 1);
#endif
    }

    CxtVLC[Q1] = static_cast<uint16_t>(CxtVLC[Q1] >> 4);
    lw         = CxtVLC[Q1] & 0x07;
    CxtVLC[Q1] = static_cast<uint16_t>(CxtVLC[Q1] >> 3);
    cwd        = CxtVLC[Q1];

    VLC_encoder.emitVLCBits(cwd, lw);
    encode_UVLC0(cwd, lw, u_q[Q0], u_q[Q1]);
    VLC_encoder.emitVLCBits(cwd, lw);
    ep[2 * (qx + 1)]     = E_n[5];
    ep[2 * (qx + 1) + 1] = E_n[7];

    sp[2 * (qx + 1)]     = sigma_n[5];
    sp[2 * (qx + 1) + 1] = sigma_n[7];
  }
  if (QW & 1) {
    uint16_t qx = static_cast<uint16_t>(QW - 1);
    make_storage_one(block, 0, qx, sigma_n, v_n, E_n, rho_q);
    // MEL encoding for the first quad
    if (c_q[Q0] == 0) {
      MEL_encoder.encodeMEL((rho_q[Q0] != 0));
    }
    Emax_q[Q0] = find_max(E_n[0], E_n[1], E_n[2], E_n[3]);
    U_q[Q0]    = std::max((int32_t)Emax_q[Q0], kappa);
    u_q[Q0]    = U_q[Q0] - kappa;
    uoff_q[Q0] = (u_q[Q0]) ? 1 : 0;
#if defined(OPENHTJ2K_ENABLE_ARM_NEON)
    auto vE_n     = vld1q_s32(E_n);
    auto vEmax_q  = vdupq_n_s32(Emax_q[Q0]);
    auto vuoff_q  = vdupq_n_s32(uoff_q[Q0]);
    auto vmask    = vceqq_s32(vE_n, vEmax_q);
    int32_t vs[4] = {1, 2, 4, 8};
    auto vshift   = vld1q_s32(vs);
    emb[Q0]       = vaddvq_s32(vmulq_s32(vuoff_q, vshift) & vmask);
#elif defined(OPENHTJ2K_TRY_AVX2) && defined(__AVX2__)
    auto vE_n = _mm_load_si128((__m128i *)E_n);
    auto vEmax_q = _mm_set1_epi32(Emax_q[Q0]);
    auto vuoff_q = _mm_set1_epi32(uoff_q[Q0]);
    auto vmask = _mm_cmpeq_epi32(vE_n, vEmax_q);
    auto vshift = _mm_setr_epi32(0, 1, 2, 3);
    auto vtmp = _mm_and_si128(_mm_sllv_epi32(vuoff_q, vshift), vmask);
    emb[Q0] = _mm_cvtsi128_si32(_mm_hadd_epi32(_mm_hadd_epi32(vtmp, vtmp), _mm_hadd_epi32(vtmp, vtmp)));
#else
    emb[Q0] = (E_n[0] == Emax_q[Q0]) ? uoff_q[Q0] : 0;
    emb[Q0] += (E_n[1] == Emax_q[Q0]) ? uoff_q[Q0] << 1 : 0;
    emb[Q0] += (E_n[2] == Emax_q[Q0]) ? uoff_q[Q0] << 2 : 0;
    emb[Q0] += (E_n[3] == Emax_q[Q0]) ? uoff_q[Q0] << 3 : 0;
#endif
    n_q[Q0]    = static_cast<uint16_t>(emb[Q0] + (rho_q[Q0] << 4) + (c_q[Q0] << 8));
    CxtVLC[Q0] = enc_CxtVLC_table0[n_q[Q0]];
    emb_k      = CxtVLC[Q0] & 0xF;
    emb_1      = static_cast<uint8_t>(n_q[Q0] % 16 & emb_k);
    for (int i = 0; i < 4; ++i) {
      m_n[i] = static_cast<uint8_t>(sigma_n[i] * U_q[Q0] - ((emb_k >> i) & 1));
#ifdef MSNAIVE
      MagSgn_encoder.emitMagSgnBits(v_n[i], m_n[i]);
#else
      MagSgn_encoder.emitMagSgnBits(v_n[i], m_n[i], (emb_1 >> i) & 1);
#endif
    }

    CxtVLC[Q0] = static_cast<uint16_t>(CxtVLC[Q0] >> 4);
    lw         = CxtVLC[Q0] & 0x07;
    CxtVLC[Q0] = static_cast<uint16_t>(CxtVLC[Q0] >> 3);
    cwd        = CxtVLC[Q0];

    ep[2 * qx]     = E_n[1];
    ep[2 * qx + 1] = E_n[3];

    sp[2 * qx]     = sigma_n[1];
    sp[2 * qx + 1] = sigma_n[3];

    VLC_encoder.emitVLCBits(cwd, lw);
    encode_UVLC0(cwd, lw, u_q[Q0]);
    VLC_encoder.emitVLCBits(cwd, lw);
  }

  // Non-initial line pair
  for (uint16_t qy = 1; qy < QH; qy++) {
    ep = Eadj.get();
    ep++;
    sp = sigma_adj.get();
    sp++;
    E_n[7]     = 0;
    sigma_n[6] = sigma_n[7] = 0;
    for (uint16_t qx = 0; qx < QW - 1; qx = static_cast<uint16_t>(qx + 2)) {
      // E_n[7] shall be saved because ep[2*qx-1] can't be changed before kappa calculation
      int32_t E7     = E_n[7];
      uint8_t sigma7 = sigma_n[7];
      // context for 1st quad of current quad pair
      c_q[Q0] = static_cast<uint16_t>((sp[2 * qx + 1] | sp[2 * qx + 2]) << 2);
      c_q[Q0] = static_cast<uint16_t>(c_q[Q0] + ((sigma_n[6] | sigma_n[7]) << 1));
      c_q[Q0] = static_cast<uint16_t>(c_q[Q0] + (sp[2 * qx - 1] | sp[2 * qx]));

      // MAKE_STORAGE()
      make_storage(block, qy, qx, sigma_n, v_n, E_n, rho_q);

      // context for 2nd quad of current quad pair
      c_q[Q1] = static_cast<uint16_t>((sp[2 * (qx + 1) + 1] | sp[2 * (qx + 1) + 2]) << 2);
      c_q[Q1] = static_cast<uint16_t>(c_q[Q1] + ((sigma_n[2] | sigma_n[3]) << 1));
      c_q[Q1] = static_cast<uint16_t>(c_q[Q1] + (sp[2 * (qx + 1) - 1] | sp[2 * (qx + 1)]));
      // MEL encoding of the first quad
      if (c_q[Q0] == 0) {
        MEL_encoder.encodeMEL((rho_q[Q0] != 0));
      }

      gamma[Q0] = (popcount32((uint32_t)rho_q[Q0]) > 1) ? 1 : 0;
      kappa     = std::max(
              (find_max(ep[2 * qx - 1], ep[2 * qx], ep[2 * qx + 1], ep[2 * qx + 2]) - 1) * gamma[Q0], 1);

      ep[2 * qx] = E_n[1];
      // if (qx > 0) {
      ep[2 * qx - 1] = E7;  // put back saved E_n
      //}

      sp[2 * qx] = sigma_n[1];
      // if (qx > 0) {
      sp[2 * qx - 1] = sigma7;  // put back saved E_n
      //}

      Emax_q[Q0] = find_max(E_n[0], E_n[1], E_n[2], E_n[3]);
      U_q[Q0]    = std::max((int32_t)Emax_q[Q0], kappa);
      u_q[Q0]    = U_q[Q0] - kappa;
      uoff_q[Q0] = (u_q[Q0]) ? 1 : 0;
#if defined(OPENHTJ2K_ENABLE_ARM_NEON)
      auto vE_n     = vld1q_s32(E_n);
      auto vEmax_q  = vdupq_n_s32(Emax_q[Q0]);
      auto vuoff_q  = vdupq_n_s32(uoff_q[Q0]);
      auto vmask    = vceqq_s32(vE_n, vEmax_q);
      int32_t vs[4] = {1, 2, 4, 8};
      auto vshift   = vld1q_s32(vs);
      emb[Q0]       = vaddvq_s32(vmulq_s32(vuoff_q, vshift) & vmask);
#elif defined(OPENHTJ2K_TRY_AVX2) && defined(__AVX2__)
      auto vE_n = _mm_load_si128((__m128i *)E_n);
      auto vEmax_q = _mm_set1_epi32(Emax_q[Q0]);
      auto vuoff_q = _mm_set1_epi32(uoff_q[Q0]);
      auto vmask = _mm_cmpeq_epi32(vE_n, vEmax_q);
      auto vshift = _mm_setr_epi32(0, 1, 2, 3);
      auto vtmp = _mm_and_si128(_mm_sllv_epi32(vuoff_q, vshift), vmask);
      emb[Q0] = _mm_cvtsi128_si32(_mm_hadd_epi32(_mm_hadd_epi32(vtmp, vtmp), _mm_hadd_epi32(vtmp, vtmp)));
#else
      emb[Q0] = (E_n[0] == Emax_q[Q0]) ? uoff_q[Q0] : 0;
      emb[Q0] += (E_n[1] == Emax_q[Q0]) ? uoff_q[Q0] << 1 : 0;
      emb[Q0] += (E_n[2] == Emax_q[Q0]) ? uoff_q[Q0] << 2 : 0;
      emb[Q0] += (E_n[3] == Emax_q[Q0]) ? uoff_q[Q0] << 3 : 0;
#endif
      n_q[Q0]    = static_cast<uint16_t>(emb[Q0] + (rho_q[Q0] << 4) + (c_q[Q0] << 8));
      CxtVLC[Q0] = enc_CxtVLC_table1[n_q[Q0]];
      emb_k      = CxtVLC[Q0] & 0xF;
      emb_1      = static_cast<uint8_t>(n_q[Q0] % 16 & emb_k);
      for (int i = 0; i < 4; ++i) {
        m_n[i] = static_cast<uint8_t>(sigma_n[i] * U_q[Q0] - ((emb_k >> i) & 1));
#ifdef MSNAIVE
        MagSgn_encoder.emitMagSgnBits(v_n[i], m_n[i]);
#else
        MagSgn_encoder.emitMagSgnBits(v_n[i], m_n[i], (emb_1 >> i) & 1);
#endif
      }

      CxtVLC[Q0] = static_cast<uint16_t>(CxtVLC[Q0] >> 4);
      lw         = CxtVLC[Q0] & 0x07;
      CxtVLC[Q0] = static_cast<uint16_t>(CxtVLC[Q0] >> 3);
      cwd        = CxtVLC[Q0];

      VLC_encoder.emitVLCBits(cwd, lw);

      // MEL encoding of the second quad
      if (c_q[Q1] == 0) {
        MEL_encoder.encodeMEL((rho_q[Q1] != 0));
      }
      gamma[Q1] = (popcount32((uint32_t)rho_q[Q1]) > 1) ? 1 : 0;
      kappa     = std::max(
              (find_max(ep[2 * (qx + 1) - 1], ep[2 * (qx + 1)], ep[2 * (qx + 1) + 1], ep[2 * (qx + 1) + 2]) - 1)
                  * gamma[Q1],
              1);

      ep[2 * (qx + 1) - 1] = E_n[3];
      ep[2 * (qx + 1)]     = E_n[5];
      if (qx + 1 == QW - 1) {  // if this quad (2nd quad) is the end of the line-pair
        ep[2 * (qx + 1) + 1] = E_n[7];
      }
      sp[2 * (qx + 1) - 1] = sigma_n[3];
      sp[2 * (qx + 1)]     = sigma_n[5];
      if (qx + 1 == QW - 1) {  // if this quad (2nd quad) is the end of the line-pair
        sp[2 * (qx + 1) + 1] = sigma_n[7];
      }

      Emax_q[Q1] = find_max(E_n[4], E_n[5], E_n[6], E_n[7]);
      U_q[Q1]    = std::max((int32_t)Emax_q[Q1], kappa);
      u_q[Q1]    = U_q[Q1] - kappa;
      uoff_q[Q1] = (u_q[Q1]) ? 1 : 0;
#if defined(OPENHTJ2K_ENABLE_ARM_NEON)
      vE_n    = vld1q_s32(E_n + 4);
      vEmax_q = vdupq_n_s32(Emax_q[Q1]);
      vuoff_q = vdupq_n_s32(uoff_q[Q1]);
      vmask   = vceqq_s32(vE_n, vEmax_q);
      emb[Q1] = vaddvq_s32(vmulq_s32(vuoff_q, vshift) & vmask);
#elif defined(OPENHTJ2K_TRY_AVX2) && defined(__AVX2__)
      vE_n = _mm_load_si128((__m128i *)(E_n + 4));
      vEmax_q = _mm_set1_epi32(Emax_q[Q1]);
      vuoff_q = _mm_set1_epi32(uoff_q[Q1]);
      vmask = _mm_cmpeq_epi32(vE_n, vEmax_q);
      vtmp = _mm_and_si128(_mm_sllv_epi32(vuoff_q, vshift), vmask);
      emb[Q1] = _mm_cvtsi128_si32(_mm_hadd_epi32(_mm_hadd_epi32(vtmp, vtmp), _mm_hadd_epi32(vtmp, vtmp)));
#else
      emb[Q1] = (E_n[4] == Emax_q[Q1]) ? uoff_q[Q1] : 0;
      emb[Q1] += (E_n[5] == Emax_q[Q1]) ? uoff_q[Q1] << 1 : 0;
      emb[Q1] += (E_n[6] == Emax_q[Q1]) ? uoff_q[Q1] << 2 : 0;
      emb[Q1] += (E_n[7] == Emax_q[Q1]) ? uoff_q[Q1] << 3 : 0;
#endif
      n_q[Q1]    = static_cast<uint16_t>(emb[Q1] + (rho_q[Q1] << 4) + (c_q[Q1] << 8));
      CxtVLC[Q1] = enc_CxtVLC_table1[n_q[Q1]];
      emb_k      = CxtVLC[Q1] & 0xF;
      emb_1      = static_cast<uint8_t>(n_q[Q1] % 16 & emb_k);
      for (int i = 0; i < 4; ++i) {
        m_n[4 + i] = static_cast<uint8_t>(sigma_n[4 + i] * U_q[Q1] - ((emb_k >> i) & 1));
#ifdef MSNAIVE
        MagSgn_encoder.emitMagSgnBits(v_n[4 + i], m_n[4 + i]);
#else
        MagSgn_encoder.emitMagSgnBits(v_n[4 + i], m_n[4 + i], (emb_1 >> i) & 1);
#endif
      }

      CxtVLC[Q1] = static_cast<uint16_t>(CxtVLC[Q1] >> 4);
      lw         = CxtVLC[Q1] & 0x07;
      CxtVLC[Q1] = static_cast<uint16_t>(CxtVLC[Q1] >> 3);
      cwd        = CxtVLC[Q1];

      VLC_encoder.emitVLCBits(cwd, lw);
      encode_UVLC1(cwd, lw, u_q[Q0], u_q[Q1]);
      VLC_encoder.emitVLCBits(cwd, lw);
    }
    if (QW & 1) {
      uint16_t qx = static_cast<uint16_t>(QW - 1);
      // E_n[7] shall be saved because ep[2*qx-1] can't be changed before kappa calculation
      int32_t E7     = E_n[7];
      uint8_t sigma7 = sigma_n[7];
      // context for current quad
      c_q[Q0] = static_cast<uint16_t>((sp[2 * qx + 1] | sp[2 * qx + 2]) << 2);
      c_q[Q0] = static_cast<uint16_t>(c_q[Q0] + ((sigma_n[6] | sigma_n[7]) << 1));
      c_q[Q0] = static_cast<uint16_t>(c_q[Q0] + (sp[2 * qx - 1] | sp[2 * qx]));
      make_storage_one(block, qy, qx, sigma_n, v_n, E_n, rho_q);
      // MEL encoding of the first quad
      if (c_q[Q0] == 0) {
        MEL_encoder.encodeMEL((rho_q[Q0] != 0));
      }

      gamma[Q0] = (popcount32((uint32_t)rho_q[Q0]) > 1) ? 1 : 0;
      kappa     = std::max(
              (find_max(ep[2 * qx - 1], ep[2 * qx], ep[2 * qx + 1], ep[2 * qx + 2]) - 1) * gamma[Q0], 1);

      ep[2 * qx] = E_n[1];
      // if (qx > 0) {
      ep[2 * qx - 1] = E7;  // put back saved E_n
      //}
      // this quad (first) is the end of the line-pair
      ep[2 * qx + 1] = E_n[3];

      sp[2 * qx] = sigma_n[1];
      // if (qx > 0) {
      sp[2 * qx - 1] = sigma7;  // put back saved E_n
      //}
      // this quad (first) is the end of the line-pair
      sp[2 * qx + 1] = sigma_n[3];

      Emax_q[Q0] = find_max(E_n[0], E_n[1], E_n[2], E_n[3]);
      U_q[Q0]    = std::max((int32_t)Emax_q[Q0], kappa);
      u_q[Q0]    = U_q[Q0] - kappa;
      uoff_q[Q0] = (u_q[Q0]) ? 1 : 0;
#if defined(OPENHTJ2K_ENABLE_ARM_NEON)
      auto vE_n     = vld1q_s32(E_n);
      auto vEmax_q  = vdupq_n_s32(Emax_q[Q0]);
      auto vuoff_q  = vdupq_n_s32(uoff_q[Q0]);
      auto vmask    = vceqq_s32(vE_n, vEmax_q);
      int32_t vs[4] = {1, 2, 4, 8};
      auto vshift   = vld1q_s32(vs);
      emb[Q0]       = vaddvq_s32(vmulq_s32(vuoff_q, vshift) & vmask);
#elif defined(OPENHTJ2K_TRY_AVX2) && defined(__AVX2__)
      auto vE_n = _mm_load_si128((__m128i *)E_n);
      auto vEmax_q = _mm_set1_epi32(Emax_q[Q0]);
      auto vuoff_q = _mm_set1_epi32(uoff_q[Q0]);
      auto vmask = _mm_cmpeq_epi32(vE_n, vEmax_q);
      auto vshift = _mm_setr_epi32(0, 1, 2, 3);
      auto vtmp = _mm_and_si128(_mm_sllv_epi32(vuoff_q, vshift), vmask);
      emb[Q0] = _mm_cvtsi128_si32(_mm_hadd_epi32(_mm_hadd_epi32(vtmp, vtmp), _mm_hadd_epi32(vtmp, vtmp)));
#else
      emb[Q0] = (E_n[0] == Emax_q[Q0]) ? uoff_q[Q0] : 0;
      emb[Q0] += (E_n[1] == Emax_q[Q0]) ? uoff_q[Q0] << 1 : 0;
      emb[Q0] += (E_n[2] == Emax_q[Q0]) ? uoff_q[Q0] << 2 : 0;
      emb[Q0] += (E_n[3] == Emax_q[Q0]) ? uoff_q[Q0] << 3 : 0;
#endif
      n_q[Q0]    = static_cast<uint16_t>(emb[Q0] + (rho_q[Q0] << 4) + (c_q[Q0] << 8));
      CxtVLC[Q0] = enc_CxtVLC_table1[n_q[Q0]];
      emb_k      = CxtVLC[Q0] & 0xF;
      emb_1      = static_cast<uint8_t>(n_q[Q0] % 16 & emb_k);
      for (int i = 0; i < 4; ++i) {
        m_n[i] = static_cast<uint8_t>(sigma_n[i] * U_q[Q0] - ((emb_k >> i) & 1));
#ifdef MSNAIVE
        MagSgn_encoder.emitMagSgnBits(v_n[i], m_n[i]);
#else
        MagSgn_encoder.emitMagSgnBits(v_n[i], m_n[i], (emb_1 >> i) & 1);
#endif
      }

      CxtVLC[Q0] = static_cast<uint16_t>(CxtVLC[Q0] >> 4);
      lw         = CxtVLC[Q0] & 0x07;
      CxtVLC[Q0] = static_cast<uint16_t>(CxtVLC[Q0] >> 3);
      cwd        = CxtVLC[Q0];

      VLC_encoder.emitVLCBits(cwd, lw);
      encode_UVLC1(cwd, lw, u_q[Q0]);
      VLC_encoder.emitVLCBits(cwd, lw);
    }
  }

  Pcup = MagSgn_encoder.termMS();
  MEL_encoder.termMEL();
  Scup = termMELandVLC(VLC_encoder, MEL_encoder);
  memcpy(&fwd_buf[static_cast<size_t>(Pcup)], &rev_buf[0], static_cast<size_t>(Scup));
  Lcup = Pcup + Scup;

  fwd_buf[static_cast<size_t>(Lcup - 1)] = static_cast<uint8_t>(Scup >> 4);
  fwd_buf[static_cast<size_t>(Lcup - 2)] =
      (fwd_buf[static_cast<size_t>(Lcup - 2)] & 0xF0) | static_cast<uint8_t>(Scup & 0x0f);

  // transfer Dcup[] to block->compressed_data
  block->set_compressed_data(fwd_buf.get(), static_cast<uint16_t>(Lcup), MAX_Lref);
  // set length of compressed data
  block->length         = static_cast<uint32_t>(Lcup);
  block->pass_length[0] = static_cast<unsigned int>(Lcup);
  // set number of coding passes
  block->num_passes      = 1;
  block->layer_passes[0] = 1;
  block->layer_start[0]  = 0;
  // set number of zero-bit planes (=Zblk)
  block->num_ZBP = static_cast<uint8_t>(block->get_Mb() - 1);
  return static_cast<int32_t>(block->length);
}
/********************************************************************************
 * HT sigprop encoding
 *******************************************************************************/
auto process_stripes_block_enc = [](SP_enc &SigProp, j2k_codeblock *block, const uint16_t i_start,
                                    const uint16_t j_start, const uint16_t width, const uint16_t height) {
  uint8_t *sp;
  uint8_t causal_cond = 0;
  uint8_t bit;
  uint8_t mbr;
  // uint32_t mbr_info;  // NOT USED
  const auto block_width  = static_cast<uint16_t>(j_start + width);
  const auto block_height = static_cast<uint16_t>(i_start + height);
  for (int16_t j = (int16_t)j_start; j < block_width; j++) {
    // mbr_info = 0;
    for (int16_t i = (int16_t)i_start; i < block_height; i++) {
      sp          = &block->block_states[(static_cast<uint32_t>(j + 1))
                                + (static_cast<uint32_t>(i + 1)) * (block->size.x + 2)];
      causal_cond = (((block->Cmodes & CAUSAL) == 0) || (i != i_start + height - 1));
      mbr         = 0;
      if (block->get_state(Sigma, i, j) == 0) {
        mbr = block->calc_mbr(i, j, causal_cond);
      }
      // mbr_info >>= 3;
      if (mbr != 0) {
        bit = (*sp >> SHIFT_SMAG) & 1;
        SigProp.emitSPBit(bit);
        block->modify_state(refinement_indicator, 1, i, j);
        block->modify_state(refinement_value, bit, i, j);
      }
      block->modify_state(scan, 1, i, j);
    }
  }
  for (int16_t j = (int16_t)j_start; j < block_width; j++) {
    for (int16_t i = (int16_t)i_start; i < block_height; i++) {
      sp = &block->block_states[(static_cast<uint32_t>(j + 1))
                                + (static_cast<uint32_t>(i + 1)) * (block->size.x + 2)];
      // encode sign
      if (block->get_state(Refinement_value, i, j)) {
        bit = (*sp >> SHIFT_SSGN) & 1;
        SigProp.emitSPBit(bit);
      }
    }
  }
};

void ht_sigprop_encode(j2k_codeblock *block, SP_enc &SigProp) {
  const uint16_t num_v_stripe = static_cast<uint16_t>(block->size.y / 4);
  const uint16_t num_h_stripe = static_cast<uint16_t>(block->size.x / 4);
  uint16_t i_start            = 0, j_start;
  uint16_t width              = 4;
  uint16_t width_last;
  uint16_t height = 4;

  // encode full-height (=4) stripes
  for (uint16_t n1 = 0; n1 < num_v_stripe; n1++) {
    j_start = 0;
    for (uint16_t n2 = 0; n2 < num_h_stripe; n2++) {
      process_stripes_block_enc(SigProp, block, i_start, j_start, width, height);
      j_start = static_cast<uint16_t>(j_start + 4);
    }
    width_last = block->size.x % 4;
    if (width_last) {
      process_stripes_block_enc(SigProp, block, i_start, j_start, width_last, height);
    }
    i_start = static_cast<uint16_t>(i_start + 4);
  }
  // encode remaining height stripes
  height  = block->size.y % 4;
  j_start = 0;
  for (uint16_t n2 = 0; n2 < num_h_stripe; n2++) {
    process_stripes_block_enc(SigProp, block, i_start, j_start, width, height);
    j_start = static_cast<uint16_t>(j_start + 4);
  }
  width_last = block->size.x % 4;
  if (width_last) {
    process_stripes_block_enc(SigProp, block, i_start, j_start, width_last, height);
  }
}
/********************************************************************************
 * HT magref encoding
 *******************************************************************************/
void ht_magref_encode(j2k_codeblock *block, MR_enc &MagRef) {
  const uint16_t blk_height   = static_cast<uint16_t>(block->size.y);
  const uint16_t blk_width    = static_cast<uint16_t>(block->size.x);
  const uint16_t num_v_stripe = static_cast<uint16_t>(block->size.y / 4);
  uint16_t i_start            = 0;
  uint16_t height             = 4;
  uint8_t *sp;
  uint8_t bit;

  for (int16_t n1 = 0; n1 < num_v_stripe; n1++) {
    for (int16_t j = 0; j < blk_width; j++) {
      for (int16_t i = (int16_t)i_start; i < (int16_t)i_start + height; i++) {
        sp = &block->block_states[static_cast<uint32_t>(j + 1)
                                  + static_cast<uint32_t>(i + 1) * (block->size.x + 2)];
        if (block->get_state(Sigma, i, j) != 0) {
          bit = (sp[0] >> SHIFT_SMAG) & 1;
          MagRef.emitMRBit(bit);
          block->modify_state(refinement_indicator, 1, i, j);
        }
      }
    }
    i_start = static_cast<uint16_t>(i_start + 4);
  }
  height = blk_height % 4;
  for (int16_t j = 0; j < blk_width; j++) {
    for (int16_t i = (int16_t)i_start; i < (int16_t)i_start + height; i++) {
      sp = &block->block_states[static_cast<uint32_t>(j + 1)
                                + static_cast<uint32_t>(i + 1) * (block->size.x + 2)];
      if (block->get_state(Sigma, i, j) != 0) {
        bit = (sp[0] >> SHIFT_SMAG) & 1;
        MagRef.emitMRBit(bit);
        block->modify_state(refinement_indicator, 1, i, j);
      }
    }
  }
}

/********************************************************************************
 * HT encoding
 *******************************************************************************/
int32_t htj2k_encode(j2k_codeblock *block, uint8_t ROIshift) noexcept {
#ifdef ENABLE_SP_MR
  block->refsegment = true;
#endif
  int32_t Lcup = htj2k_cleanup_encode(block, ROIshift);
  if (Lcup && block->refsegment) {
    uint8_t Dref[2047] = {0};
    SP_enc SigProp(Dref);
    MR_enc MagRef(Dref);
    int32_t HTMagRefLength = 0;
    // SigProp encoding
    ht_sigprop_encode(block, SigProp);
    // MagRef encoding
    ht_magref_encode(block, MagRef);
    if (MagRef.get_length()) {
      HTMagRefLength         = termSPandMR(SigProp, MagRef);
      block->num_passes      = static_cast<uint8_t>(block->num_passes + 2);
      block->layer_passes[0] = static_cast<uint8_t>(block->layer_passes[0] + 2);
      block->pass_length.push_back(SigProp.get_length());
      block->pass_length.push_back(MagRef.get_length());
    } else {
      SigProp.termSP();
      HTMagRefLength         = static_cast<int32_t>(SigProp.get_length());
      block->num_passes      = static_cast<uint8_t>(block->num_passes + 1);
      block->layer_passes[0] = static_cast<uint8_t>(block->layer_passes[0] + 1);
      block->pass_length.push_back(SigProp.get_length());
    }
    if (HTMagRefLength) {
      block->length += static_cast<unsigned int>(HTMagRefLength);
      block->num_ZBP = static_cast<uint8_t>(block->num_ZBP - (block->refsegment));
      block->set_compressed_data(Dref, static_cast<uint16_t>(HTMagRefLength));
    }
    //    // debugging
    //    printf("SP length = %d\n", SigProp.get_length());
    //    printf("MR length = %d\n", MagRef.get_length());
    //    printf("HT MAgRef length = %d\n", HTMagRefLength);
    //    for (int i = 0; i < HTMagRefLength; ++i) {
    //      printf("%02X ", Dref[i]);
    //    }
    //    printf("\n");
  }
  return EXIT_SUCCESS;
}
