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

#include "coding_units.hpp"
#include "dec_CxtVLC_tables.hpp"
#include "ht_block_decoding.hpp"
#include "coding_local.hpp"
#include "utils.hpp"

#ifdef _OPENMP
  #include <omp.h>
#endif

#if defined(OPENHTJ2K_ENABLE_ARM_NEON)
  #include <arm_neon.h>
#elif defined(OPENHTJ2K_TRY_AVX2) && defined(__AVX2__)
  #if defined(_MSC_VER) || defined(__MINGW64__)
    #include <intrin.h>
  #else
    #include <x86intrin.h>
  #endif
#endif

#define Q0 0
#define Q1 1

uint8_t j2k_codeblock::calc_mbr(const int16_t i, const int16_t j, const uint8_t causal_cond) const {
  const int16_t im1 = static_cast<int16_t>(i - 1);
  const int16_t jm1 = static_cast<int16_t>(j - 1);
  const int16_t ip1 = static_cast<int16_t>(i + 1);
  const int16_t jp1 = static_cast<int16_t>(j + 1);
  uint8_t mbr       = get_state(Sigma, im1, jm1);
  mbr               = mbr | get_state(Sigma, im1, j);
  mbr               = mbr | get_state(Sigma, im1, jp1);
  mbr               = mbr | get_state(Sigma, i, jm1);
  mbr               = mbr | get_state(Sigma, i, jp1);
  mbr               = mbr | static_cast<uint8_t>(get_state(Sigma, ip1, jm1) * causal_cond);
  mbr               = mbr | static_cast<uint8_t>(get_state(Sigma, ip1, j) * causal_cond);
  mbr               = mbr | static_cast<uint8_t>(get_state(Sigma, ip1, jp1) * causal_cond);

  mbr = mbr | static_cast<uint8_t>(get_state(Refinement_value, im1, jm1) * get_state(Scan, im1, jm1));
  mbr = mbr | static_cast<uint8_t>(get_state(Refinement_value, im1, j) * get_state(Scan, im1, j));
  mbr = mbr | static_cast<uint8_t>(get_state(Refinement_value, im1, jp1) * get_state(Scan, im1, jp1));
  mbr = mbr | static_cast<uint8_t>(get_state(Refinement_value, i, jm1) * get_state(Scan, i, jm1));
  mbr = mbr | static_cast<uint8_t>(get_state(Refinement_value, i, jp1) * get_state(Scan, i, jp1));
  mbr = mbr
        | static_cast<uint8_t>(get_state(Refinement_value, ip1, jm1) * get_state(Scan, ip1, jm1)
                               * causal_cond);
  mbr = mbr
        | static_cast<uint8_t>(get_state(Refinement_value, ip1, j) * get_state(Scan, ip1, j) * causal_cond);
  mbr = mbr
        | static_cast<uint8_t>(get_state(Refinement_value, ip1, jp1) * get_state(Scan, ip1, jp1)
                               * causal_cond);
  return mbr;
}

/********************************************************************************
 * functions for state_MS: state class for MagSgn decoding
 *******************************************************************************/
void state_MS_dec::loadByte() {
  tmp  = 0xFF;
  bits = (last == 0xFF) ? 7 : 8;
  if (pos < length) {
    tmp = buf[pos];
    pos++;
    last = tmp;
  }

  Creg |= static_cast<uint64_t>(tmp) << ctreg;
  ctreg = static_cast<uint8_t>(ctreg + bits);
}
void state_MS_dec::close(int32_t num_bits) {
  Creg >>= num_bits;
  ctreg = static_cast<uint8_t>(ctreg - static_cast<uint8_t>(num_bits));
  while (ctreg < 32) {
    loadByte();
  }
}

uint8_t state_MS_dec::importMagSgnBit() {
  uint8_t val;
  if (bits == 0) {
    bits = (last == 0xFF) ? 7 : 8;
    if (pos < length) {
      tmp = *(buf + pos);  // modDcup(MS->pos, Lcup);
      if ((static_cast<uint16_t>(tmp) & static_cast<uint16_t>(1 << bits)) != 0) {
        printf("ERROR: importMagSgnBit error\n");
        throw std::exception();
      }
    } else if (pos == length) {
      tmp = 0xFF;
    } else {
      printf("ERROR: importMagSgnBit error\n");
      throw std::exception();
    }
    last = tmp;
    pos++;
  }
  val = tmp & 1;
  tmp = static_cast<uint8_t>(tmp >> 1);
  --bits;
  return val;
}

int32_t state_MS_dec::decodeMagSgnValue(int32_t m_n, int32_t i_n) {
  int32_t val = 0;
  // uint8_t bit;
  if (m_n > 0) {
    val = static_cast<int32_t>(bitmask32[m_n] & (int32_t)Creg);
    //      for (int i = 0; i < m_n; i++) {
    //        bit = MS->importMagSgnBit();
    //        val += (bit << i);
    //      }
    val += (i_n << m_n);
    close(m_n);
  } else {
    val = 0;
  }
  return val;
}

/********************************************************************************
 * functions for state_MEL_unPacker and state_MEL: state classes for MEL decoding
 *******************************************************************************/
uint8_t state_MEL_unPacker::importMELbit() {
  if (bits == 0) {
    bits = (tmp == 0xFF) ? 7 : 8;
    if (pos < length) {
      tmp = *(buf + pos);  //+ modDcup(MEL_unPacker->pos, Lcup);
      //        MEL_unPacker->tmp = modDcup()
      pos++;
    } else {
      tmp = 0xFF;
    }
  }
  bits--;
  return (tmp >> bits) & 1;
}

uint8_t state_MEL_decoder::decodeMELSym() {
  uint8_t eval;
  uint8_t bit;
  if (MEL_run == 0 && MEL_one == 0) {
    eval = this->MEL_E[MEL_k];
    bit  = MEL_unPacker->importMELbit();
    if (bit == 1) {
      MEL_run = static_cast<uint8_t>(1 << eval);
      MEL_k   = static_cast<uint8_t>((12 < MEL_k + 1) ? 12 : MEL_k + 1);
    } else {
      MEL_run = 0;
      while (eval > 0) {
        bit     = MEL_unPacker->importMELbit();
        MEL_run = static_cast<uint8_t>((MEL_run << 1) + bit);
        eval--;
      }
      MEL_k   = static_cast<uint8_t>((0 > MEL_k - 1) ? 0 : MEL_k - 1);
      MEL_one = 1;
    }
  }
  if (MEL_run > 0) {
    MEL_run--;
    return 0;
  } else {
    MEL_one = 0;
    return 1;
  }
}

/********************************************************************************
 * functions for state_VLC: state class for VLC decoding
 *******************************************************************************/
#ifndef ADVANCED
uint8_t state_VLC::importVLCBit() {
  uint8_t val;
  if (bits == 0) {
    if (pos >= rev_length) {
      tmp = *(buf + pos);  // modDcup(VLC->pos, Lcup);
    } else {
      printf("ERROR: import VLCBits error\n");
      throw std::exception();
    }
    bits = 8;
    if (last > 0x8F && (tmp & 0x7F) == 0x7F) {
      bits = 7;  // bit-un-stuffing
    }
    last = tmp;
    // To prevent overflow of pos
    if (pos > 0) {
      pos--;
    }
  }
  val = tmp & 1;
  tmp >>= 1;
  bits--;
  return val;
}
#else
void state_VLC_dec::load_bytes() {
  uint64_t load_val = 0;
  int32_t new_bits  = 32;
  last              = buf[pos + 1];
  if (pos >= 3) {  // Common case; we have at least 4 bytes available
    load_val = buf[pos - 3];
    load_val = (load_val << 8) | buf[pos - 2];
    load_val = (load_val << 8) | buf[pos - 1];
    load_val = (load_val << 8) | buf[pos];
    load_val = (load_val << 8) | last;  // For stuffing bit detection
    pos -= 4;
  } else {
    if (pos >= 2) {
      load_val = buf[pos - 2];
    }
    if (pos >= 1) {
      load_val = (load_val << 8) | buf[pos - 1];
    }
    if (pos >= 0) {
      load_val = (load_val << 8) | buf[pos];
    }
    pos      = 0;
    load_val = (load_val << 8) | last;  // For stuffing bit detection
  }
  // Now remove any stuffing bits, shifting things down as we go
  if ((load_val & 0x7FFF000000) > 0x7F8F000000) {
    load_val &= 0x7FFFFFFFFF;
    new_bits--;
  }
  if ((load_val & 0x007FFF0000) > 0x007F8F0000) {
    load_val = (load_val & 0x007FFFFFFF) + ((load_val & 0xFF00000000) >> 1);
    new_bits--;
  }
  if ((load_val & 0x00007FFF00) > 0x00007F8F00) {
    load_val = (load_val & 0x00007FFFFF) + ((load_val & 0xFFFF000000) >> 1);
    new_bits--;
  }
  if ((load_val & 0x0000007FFF) > 0x0000007F8F) {
    load_val = (load_val & 0x0000007FFF) + ((load_val & 0xFFFFFF0000) >> 1);
    new_bits--;
  }
  load_val >>= 8;  // Shifts away the extra byte we imported
  Creg |= (load_val << ctreg);
  ctreg += new_bits;
}

uint8_t state_VLC_dec::getVLCbit() {
  // "bits" is not actually bits, but a bit
  bits = (uint8_t)(Creg & 0x01);
  close32(1);
  return bits;
}

void state_VLC_dec::close32(int32_t num_bits) {
  Creg >>= num_bits;
  ctreg -= num_bits;
  while (ctreg < 32) {
    load_bytes();
  }
}
#endif

void state_VLC_dec::decodeCxtVLC(const uint16_t &context, uint8_t (&u_off)[2], uint8_t (&rho)[2],
                                 uint8_t (&emb_k)[2], uint8_t (&emb_1)[2], const uint8_t &first_or_second,
                                 const uint16_t *dec_CxtVLC_table) {
#ifndef ADVANCED
  uint8_t b_low = tmp;
  uint8_t b_upp = *(buf + pos);  // modDcup(VLC->pos, Lcup);
  uint16_t word = (b_upp << bits) + b_low;
  uint8_t cwd   = word & 0x7F;
#else
  uint8_t cwd = Creg & 0x7f;
#endif
  uint16_t idx           = static_cast<uint16_t>(cwd + (context << 7));
  uint16_t value         = dec_CxtVLC_table[idx];
  u_off[first_or_second] = value & 1;
  // value >>= 1;
  // uint8_t len = value & 0x07;
  // value >>= 3;
  // rho[first_or_second] = value & 0x0F;
  // value >>= 4;
  // emb_k[first_or_second] = value & 0x0F;
  // value >>= 4;
  // emb_1[first_or_second] = value & 0x0F;
  uint8_t len            = static_cast<uint8_t>((value & 0x000F) >> 1);
  rho[first_or_second]   = static_cast<uint8_t>((value & 0x00F0) >> 4);
  emb_k[first_or_second] = static_cast<uint8_t>((value & 0x0F00) >> 8);
  emb_1[first_or_second] = static_cast<uint8_t>((value & 0xF000) >> 12);

#ifndef ADVANCED
  for (int i = 0; i < len; i++) {
    importVLCBit();
  }
#else
  close32(len);
#endif
}

uint8_t state_VLC_dec::decodeUPrefix() {
  if (getbitfunc == 1) {
    return 1;
  }
  if (getbitfunc == 1) {
    return 2;
  }
  if (getbitfunc == 1) {
    return 3;
  } else {
    return 5;
  }
}

uint8_t state_VLC_dec::decodeUSuffix(const uint32_t &u_pfx) {
  uint8_t bit, val;
  if (u_pfx < 3) {
    return 0;
  }
  val = getbitfunc;
  if (u_pfx == 3) {
    return val;
  }
  for (int i = 1; i < 5; i++) {
    bit = getbitfunc;
    val = static_cast<uint8_t>(val + (bit << i));
  }
  return val;
}
uint8_t state_VLC_dec::decodeUExtension(const uint32_t &u_sfx) {
  uint8_t bit, val;
  if (u_sfx < 28) {
    return 0;
  }
  val = getbitfunc;
  for (int i = 1; i < 4; i++) {
    bit = getbitfunc;
    val = static_cast<uint8_t>(val + (bit << i));
  }
  return val;
}
/********************************************************************************
 * functions for SP_dec: state class for HT SigProp decoding
 *******************************************************************************/
uint8_t SP_dec::importSigPropBit() {
  uint8_t val;
  if (bits == 0) {
    bits = (last == 0xFF) ? 7 : 8;
    if (pos < Lref) {
      tmp = *(Dref + pos);
      pos++;
      if ((tmp & (1 << bits)) != 0) {
        printf("ERROR: importSigPropBit error\n");
        throw std::exception();
      }
    } else {
      tmp = 0;
    }
    last = tmp;
  }
  val = tmp & 1;
  tmp = static_cast<uint8_t>(tmp >> 1);
  bits--;
  return val;
}

/********************************************************************************
 * MR_dec: state class for HT MagRef decoding
 *******************************************************************************/
uint8_t MR_dec::importMagRefBit() {
  uint8_t val;
  if (bits == 0) {
    if (pos >= 0) {
      tmp = *(Dref + pos);
      pos--;
    } else {
      tmp = 0;
    }
    bits = 8;
    if (last > 0x8F && (tmp & 0x7F) == 0x7F) {
      bits = 7;
    }
    last = tmp;
  }
  val = tmp & 1;
  tmp = static_cast<uint8_t>(tmp >> 1);
  bits--;
  return val;
}

auto decodeSigEMB = [](state_MEL_decoder &MEL_decoder, rev_buf &VLC_dec, const uint16_t &context,
                       uint8_t (&u_off)[2], uint8_t (&rho)[2], uint8_t (&emb_k)[2], uint8_t (&emb_1)[2],
                       const uint8_t &first_or_second, const uint16_t *dec_CxtVLC_table) {
  uint8_t sym;
  if (context == 0) {
    sym = MEL_decoder.decodeMELSym();
    if (sym == 0) {
      rho[first_or_second] = u_off[first_or_second] = emb_k[first_or_second] = emb_1[first_or_second] = 0;
      return;
    }
  }
  uint32_t vlcval        = VLC_dec.fetch();
  uint16_t value         = dec_CxtVLC_table[(vlcval & 0x7F) + (context << 7)];
  u_off[first_or_second] = value & 1;
  uint32_t len           = static_cast<uint8_t>((value & 0x000F) >> 1);
  rho[first_or_second]   = static_cast<uint8_t>((value & 0x00F0) >> 4);
  emb_k[first_or_second] = static_cast<uint8_t>((value & 0x0F00) >> 8);
  emb_1[first_or_second] = static_cast<uint8_t>((value & 0xF000) >> 12);
  VLC_dec.advance(len);
  //  VLC_dec.decodeCxtVLC(context, u_off, rho, emb_k, emb_1, first_or_second, dec_CxtVLC_table);
};

void ht_cleanup_decode(j2k_codeblock *block, const uint8_t &pLSB, fwd_buf<0xFF> &MagSgn, MEL_dec &MEL,
                       rev_buf &VLC_dec) {
  const uint16_t QW = static_cast<uint16_t>(ceil_int(static_cast<int16_t>(block->size.x), 2));
  const uint16_t QH = static_cast<uint16_t>(ceil_int(static_cast<int16_t>(block->size.y), 2));

  alignas(32) int32_t sigma_quads[8];
  alignas(32) uint32_t mu_quads[8];
  alignas(32) uint32_t m_quads[8];
  alignas(32) uint32_t v_quads[8];
  auto mp0 = block->sample_buf.get();
  auto mp1 = block->sample_buf.get() + block->blksampl_stride;
  auto sp0 = block->block_states.get() + 1 + block->blkstate_stride;
  auto sp1 = block->block_states.get() + 1 + 2 * block->blkstate_stride;

  alignas(32) int32_t rho[2];
  alignas(32) uint8_t u_off[2];
  alignas(32) uint8_t emb_k[2];
  alignas(32) uint8_t emb_1[2];
  alignas(32) uint32_t u[2];
  alignas(32) uint32_t u_pfx[2];
  alignas(32) uint32_t u_sfx[2];
  alignas(32) uint32_t u_ext[2];
  alignas(32) uint32_t U[2];
  alignas(32) uint8_t gamma[2];
  alignas(32) uint32_t kappa[2] = {1, 1};  // kappa is always 1 for initial line-pair

  const uint16_t *dec_table0, *dec_table1;
  dec_table0 = dec_CxtVLC_table0_fast_16;
  dec_table1 = dec_CxtVLC_table1_fast_16;

  alignas(32) uint32_t m_n[2], known_1[2];

  alignas(32) auto rholine = MAKE_UNIQUE<int32_t[]>(QW + 2);
  memset(rholine.get(), 0, sizeof(int32_t) * (QW + 2));
  auto rho_p             = rholine.get() + 1;
  alignas(32) auto Eline = MAKE_UNIQUE<int32_t[]>(2 * QW + 2);
  memset(Eline.get(), 0, sizeof(int32_t) * (2 * QW + 2));
  auto E_p = Eline.get() + 1;

  uint16_t context = 0;
  uint32_t vlcval;
  int32_t mel_run = MEL.get_run();

  // Initial line-pair
  for (int32_t q = 0; q < QW - 1; q += 2) {
    // Decoding of significance and EMB patterns and unsigned residual offsets
    vlcval       = VLC_dec.fetch();
    uint16_t tv0 = dec_table0[(vlcval & 0x7F) + (context << 7)];
    if (context == 0) {
      mel_run -= 2;
      tv0 = (mel_run == -1) ? tv0 : 0;
      if (mel_run < 0) {
        mel_run = MEL.get_run();
      }
    }
    rho[0]   = static_cast<uint8_t>((tv0 & 0x00F0) >> 4);
    emb_k[0] = static_cast<uint8_t>((tv0 & 0x0F00) >> 8);
    emb_1[0] = static_cast<uint8_t>((tv0 & 0xF000) >> 12);
#if defined(OPENHTJ2K_ENABLE_ARM_NEON)
    int32_t mask[4] = {1, 2, 4, 8};
    auto v0         = vdupq_n_s32(rho[Q0]);
    auto vm         = vld1q_s32(mask);
    auto vone       = vdupq_n_s32(1);
    vst1q_s32(sigma_quads, vandq_s32(vtstq_s32(v0, vm), vone));
#elif defined(__AVX2__)
    // do nothing
#else
    for (uint32_t i = 0; i < 4; i++) {
      sigma_quads[i] = (rho[Q0] >> i) & 1;
    }
#endif
    *sp0++   = (rho[Q0] >> 0) & 1;
    *sp0++   = (rho[Q0] >> 2) & 1;
    *sp1++   = (rho[Q0] >> 1) & 1;
    *sp1++   = (rho[Q0] >> 3) & 1;
    *rho_p++ = rho[Q0];
    // calculate context for the next quad
    context = (rho[Q0] >> 1) | (rho[Q0] & 1);

    // Decoding of significance and EMB patterns and unsigned residual offsets
    vlcval       = VLC_dec.advance(static_cast<uint8_t>((tv0 & 0x000F) >> 1));
    uint16_t tv1 = dec_table0[(vlcval & 0x7F) + (context << 7)];
    if (context == 0) {
      mel_run -= 2;
      tv1 = (mel_run == -1) ? tv1 : 0;
      if (mel_run < 0) {
        mel_run = MEL.get_run();
      }
    }
    rho[1]   = static_cast<uint8_t>((tv1 & 0x00F0) >> 4);
    emb_k[1] = static_cast<uint8_t>((tv1 & 0x0F00) >> 8);
    emb_1[1] = static_cast<uint8_t>((tv1 & 0xF000) >> 12);
#if defined(OPENHTJ2K_ENABLE_ARM_NEON)
    auto v1 = vdupq_n_s32(rho[Q1]);
    vst1q_s32(sigma_quads + 4, vandq_s32(vtstq_s32(v1, vm), vone));
#elif defined(__AVX2__)
    auto vrho   = _mm256_setr_epi32(rho[0], rho[0], rho[0], rho[0], rho[1], rho[1], rho[1], rho[1]);
    auto vsigma = _mm256_and_si256(_mm256_srav_epi32(vrho, _mm256_setr_epi32(0, 1, 2, 3, 0, 1, 2, 3)),
                                   _mm256_set1_epi32(1));
    _mm256_store_si256((__m256i *)sigma_quads, vsigma);
#else
    for (uint32_t i = 0; i < 4; i++) {
      sigma_quads[i + 4] = (rho[Q1] >> i) & 1;
    }
#endif

    *sp0++   = (rho[Q1] >> 0) & 1;
    *sp0++   = (rho[Q1] >> 2) & 1;
    *sp1++   = (rho[Q1] >> 1) & 1;
    *sp1++   = (rho[Q1] >> 3) & 1;
    *rho_p++ = rho[Q1];
    // calculate context for the next quad
    context = (rho[Q1] >> 1) | (rho[Q1] & 1);

    vlcval   = VLC_dec.advance(static_cast<uint8_t>((tv1 & 0x000F) >> 1));
    u_off[0] = tv0 & 1;
    u_off[1] = tv1 & 1;

    if (u_off[Q0] == 1 && u_off[Q1] == 1) {
      mel_run -= 2;
      bool mel_cond = (mel_run == -1);
      if (mel_run < 0) {
        mel_run = MEL.get_run();
      }
      if (mel_cond) {
        u_pfx[Q0] = VLC_dec.decodeUPrefix();
        u_pfx[Q1] = VLC_dec.decodeUPrefix();
        u_sfx[Q0] = VLC_dec.decodeUSuffix(u_pfx[Q0]);
        u_sfx[Q1] = VLC_dec.decodeUSuffix(u_pfx[Q1]);
        u_ext[Q0] = VLC_dec.decodeUExtension(u_sfx[Q0]);
        u_ext[Q1] = VLC_dec.decodeUExtension(u_sfx[Q1]);
        u[Q0]     = 2 + u_pfx[Q0] + u_sfx[Q0] + (u_ext[Q0] << 2);
        u[Q1]     = 2 + u_pfx[Q1] + u_sfx[Q1] + (u_ext[Q1] << 2);
      } else {
        u_pfx[Q0] = VLC_dec.decodeUPrefix();
        if (u_pfx[Q0] > 2) {
          u[Q1]     = VLC_dec.importVLCBit() + 1;  // block->VLC->importVLCBit() + 1;
          u_sfx[Q0] = VLC_dec.decodeUSuffix(u_pfx[Q0]);
          u_ext[Q0] = VLC_dec.decodeUExtension(u_sfx[Q0]);
        } else {
          u_pfx[Q1] = VLC_dec.decodeUPrefix();
          u_sfx[Q0] = VLC_dec.decodeUSuffix(u_pfx[Q0]);
          u_sfx[Q1] = VLC_dec.decodeUSuffix(u_pfx[Q1]);
          u_ext[Q0] = VLC_dec.decodeUExtension(u_sfx[Q0]);
          u_ext[Q1] = VLC_dec.decodeUExtension(u_sfx[Q1]);
          u[Q1]     = u_pfx[Q1] + u_sfx[Q1] + (u_ext[Q1] << 2);
        }
        u[Q0] = u_pfx[Q0] + u_sfx[Q0] + (u_ext[Q0] << 2);
      }
    } else if (u_off[Q0] == 1 && u_off[Q1] == 0) {
      u_pfx[Q0] = VLC_dec.decodeUPrefix();
      u_sfx[Q0] = VLC_dec.decodeUSuffix(u_pfx[Q0]);
      u_ext[Q0] = VLC_dec.decodeUExtension(u_sfx[Q0]);
      u[Q0]     = u_pfx[Q0] + u_sfx[Q0] + (u_ext[Q0] << 2);
      u[Q1]     = 0;
    } else if (u_off[Q0] == 0 && u_off[Q1] == 1) {
      u_pfx[Q1] = VLC_dec.decodeUPrefix();
      u_sfx[Q1] = VLC_dec.decodeUSuffix(u_pfx[Q1]);
      u_ext[Q1] = VLC_dec.decodeUExtension(u_sfx[Q1]);
      u[Q0]     = 0;
      u[Q1]     = u_pfx[Q1] + u_sfx[Q1] + (u_ext[Q1] << 2);
    } else {
      u[Q0] = 0;
      u[Q1] = 0;
    }

    U[Q0] = kappa[Q0] + u[Q0];
    U[Q1] = kappa[Q1] + u[Q1];
#if defined(OPENHTJ2K_ENABLE_ARM_NEON)
    v0          = vandq_s32(vtstq_s32(vdupq_n_s32(emb_k[Q0]), vm), vone);
    auto vsigma = vld1q_s32(sigma_quads);
    vst1q_u32(m_quads, vsubq_s32(vmulq_s32(vsigma, vdupq_n_s32(U[Q0])), v0));
    v0     = vandq_s32(vtstq_s32(vdupq_n_s32(emb_k[Q1]), vm), vone);
    vsigma = vld1q_s32(sigma_quads + 4);
    vst1q_u32(m_quads + 4, vsubq_s32(vmulq_s32(vsigma, vdupq_n_s32(U[Q1])), v0));
#elif defined(__AVX2__)
    auto vemb_k =
        _mm256_and_si256(_mm256_srav_epi32(_mm256_setr_epi32(emb_k[0], emb_k[0], emb_k[0], emb_k[0],
                                                             emb_k[1], emb_k[1], emb_k[1], emb_k[1]),
                                           _mm256_setr_epi32(0, 1, 2, 3, 0, 1, 2, 3)),
                         _mm256_set1_epi32(1));
    auto vU = _mm256_setr_epi32(U[0], U[0], U[0], U[0], U[1], U[1], U[1], U[1]);
    _mm256_store_si256((__m256i *)m_quads, _mm256_sub_epi32(_mm256_mullo_epi32(vsigma, vU), vemb_k));
#else
    for (uint32_t i = 0; i < 4; i++) {
      m_quads[i]     = sigma_quads[i] * U[Q0] - ((emb_k[Q0] >> i) & 1);
      m_quads[i + 4] = sigma_quads[i + 4] * U[Q1] - ((emb_k[Q1] >> i) & 1);
    }
#endif

    // recoverMagSgnValue
    for (uint32_t i = 0; i < 4; i++) {
      uint32_t msval = MagSgn.fetch();
      m_n[Q0]        = m_quads[i];
      known_1[Q0]    = (emb_1[Q0] >> i) & 1;
      v_quads[i]     = msval & ((1 << m_n[Q0]) - 1);
      v_quads[i] |= known_1[Q0] << m_n[Q0];
      MagSgn.advance(m_n[Q0]);

      if (m_n[Q0] != 0) {
        mu_quads[i] = static_cast<uint32_t>((v_quads[i] >> 1) + 1);
        mu_quads[i] <<= pLSB;
        mu_quads[i] |= static_cast<uint32_t>((v_quads[i] & 1) << 31);  // sign bit
      } else {
        mu_quads[i] = 0;
      }
    }
    *mp0++ = static_cast<int>(mu_quads[0]);
    *mp0++ = static_cast<int>(mu_quads[2]);
    *mp1++ = static_cast<int>(mu_quads[1]);
    *mp1++ = static_cast<int>(mu_quads[3]);

    for (uint32_t i = 0; i < 4; i++) {
      uint32_t msval = MagSgn.fetch();
      m_n[Q1]        = m_quads[i + 4];
      known_1[Q1]    = (emb_1[Q1] >> i) & 1;
      v_quads[i + 4] = msval & ((1 << m_n[Q1]) - 1);
      v_quads[i + 4] |= known_1[Q1] << m_n[Q1];
      MagSgn.advance(m_n[Q1]);

      if (m_n[Q1] != 0) {
        mu_quads[i + 4] = static_cast<uint32_t>((v_quads[i + 4] >> 1) + 1);
        mu_quads[i + 4] <<= pLSB;
        mu_quads[i + 4] |= static_cast<uint32_t>((v_quads[i + 4] & 1) << 31);  // sign bit
      } else {
        mu_quads[i + 4] = 0;
      }
    }
    *mp0++ = static_cast<int>(mu_quads[0 + 4]);
    *mp0++ = static_cast<int>(mu_quads[2 + 4]);
    *mp1++ = static_cast<int>(mu_quads[1 + 4]);
    *mp1++ = static_cast<int>(mu_quads[3 + 4]);
    *E_p++ = static_cast<int32_t>(32 - count_leading_zeros(static_cast<uint32_t>(v_quads[1])));
    *E_p++ = static_cast<int32_t>(32 - count_leading_zeros(static_cast<uint32_t>(v_quads[3])));
    *E_p++ = static_cast<int32_t>(32 - count_leading_zeros(static_cast<uint32_t>(v_quads[5])));
    *E_p++ = static_cast<int32_t>(32 - count_leading_zeros(static_cast<uint32_t>(v_quads[7])));
  }
  // if QW is odd number ..
  if (QW % 2 == 1) {
    // Decoding of significance and EMB patterns and unsigned residual offsets
    vlcval       = VLC_dec.fetch();
    uint16_t tv0 = dec_table0[(vlcval & 0x7F) + (context << 7)];
    if (context == 0) {
      mel_run -= 2;
      tv0 = (mel_run == -1) ? tv0 : 0;
      if (mel_run < 0) {
        mel_run = MEL.get_run();
      }
    }
    rho[0]   = static_cast<uint8_t>((tv0 & 0x00F0) >> 4);
    emb_k[0] = static_cast<uint8_t>((tv0 & 0x0F00) >> 8);
    emb_1[0] = static_cast<uint8_t>((tv0 & 0xF000) >> 12);
    *rho_p++ = rho[Q0];

#if defined(OPENHTJ2K_ENABLE_ARM_NEON)
    int32_t mask[4] = {1, 2, 4, 8};
    auto v0         = vdupq_n_s32(rho[Q0]);
    auto vm         = vld1q_s32(mask);
    auto vone       = vdupq_n_s32(1);
    vst1q_s32(sigma_quads, vandq_s32(vtstq_s32(v0, vm), vone));
#elif defined(__AVX2__)
    for (uint32_t i = 0; i < 4; i++) {
      sigma_quads[i] = (rho[Q0] >> i) & 1;
    }
#else
    for (uint32_t i = 0; i < 4; i++) {
      sigma_quads[i] = (rho[Q0] >> i) & 1;
    }
#endif
    *sp0++ = (rho[Q0] >> 0) & 1;
    *sp0++ = (rho[Q0] >> 2) & 1;
    *sp1++ = (rho[Q0] >> 1) & 1;
    *sp1++ = (rho[Q0] >> 3) & 1;
    vlcval = VLC_dec.advance(static_cast<uint8_t>((tv0 & 0x000F) >> 1));

    u_off[0] = tv0 & 1;

    if (u_off[Q0] == 1) {
      u_pfx[Q0] = VLC_dec.decodeUPrefix();
      u_sfx[Q0] = VLC_dec.decodeUSuffix(u_pfx[Q0]);
      u_ext[Q0] = VLC_dec.decodeUExtension(u_sfx[Q0]);
      u[Q0]     = u_pfx[Q0] + u_sfx[Q0] + (u_ext[Q0] << 2);
    } else {
      u[Q0] = 0;
    }

    U[Q0] = kappa[Q0] + u[Q0];

#if defined(OPENHTJ2K_ENABLE_ARM_NEON)
    v0          = vandq_s32(vtstq_s32(vdupq_n_s32(emb_k[Q0]), vm), vone);
    auto vsigma = vld1q_s32(sigma_quads);
    vst1q_u32(m_quads, vsubq_s32(vmulq_s32(vsigma, vdupq_n_s32(U[Q0])), v0));
#elif defined(__AVX2__)
    for (uint32_t i = 0; i < 4; i++) {
      m_quads[i] = sigma_quads[i] * U[Q0] - ((emb_k[Q0] >> i) & 1);
    }
#else
    for (uint32_t i = 0; i < 4; i++) {
      m_quads[i] = sigma_quads[i] * U[Q0] - ((emb_k[Q0] >> i) & 1);
    }
#endif

    // recoverMagSgnValue
    for (uint32_t i = 0; i < 4; i++) {
      uint32_t msval = MagSgn.fetch();
      m_n[Q0]        = m_quads[i];
      known_1[Q0]    = (emb_1[Q0] >> i) & 1;
      v_quads[i]     = msval & ((1 << m_n[Q0]) - 1);
      v_quads[i] |= known_1[Q0] << m_n[Q0];
      MagSgn.advance(m_n[Q0]);

      if (m_n[Q0] != 0) {
        mu_quads[i] = static_cast<uint32_t>((v_quads[i] >> 1) + 1);
        mu_quads[i] <<= pLSB;
        mu_quads[i] |= static_cast<uint32_t>((v_quads[i] & 1) << 31);  // sign bit
      } else {
        mu_quads[i] = 0;
      }
    }
    *mp0++ = static_cast<int>(mu_quads[0]);
    *mp0++ = static_cast<int>(mu_quads[2]);
    *mp1++ = static_cast<int>(mu_quads[1]);
    *mp1++ = static_cast<int>(mu_quads[3]);
    *E_p++ = static_cast<int32_t>(32 - count_leading_zeros(static_cast<uint32_t>(v_quads[1])));
    *E_p++ = static_cast<int32_t>(32 - count_leading_zeros(static_cast<uint32_t>(v_quads[3])));
  }  // Initial line-pair end

  // Non-initial line-pair
  uint16_t context1, context2;
  for (uint16_t row = 1; row < QH; row++) {
    rho_p      = rholine.get() + 1;
    E_p        = Eline.get() + 1;
    mp0        = block->sample_buf.get() + (row * 2) * block->blksampl_stride;
    mp1        = block->sample_buf.get() + (row * 2 + 1) * block->blksampl_stride;
    sp0        = block->block_states.get() + (row * 2 + 1) * block->blkstate_stride + 1;
    sp1        = block->block_states.get() + (row * 2 + 2) * block->blkstate_stride + 1;
    int32_t qx = 0;
    for (qx = 0; qx < QW - 1; qx += 2) {
      // calculate context for the current quad
      context1 = static_cast<uint16_t>((rho_p[qx - 1] >> 3)
                                       | ((rho_p[qx] >> 1) & 1)
                                             + (((rho_p[qx] >> 3) | ((rho_p[qx + 1] >> 1) & 1)) << 2));
      if (qx > 0) {
        context1 |= (((rho[Q1] >> 2) & 1) | ((rho[Q1] >> 3) & 1)) << 1;
      }

      // Decoding of significance and EMB patterns and unsigned residual offsets
      vlcval       = VLC_dec.fetch();
      uint16_t tv0 = dec_table1[(vlcval & 0x7F) + (context1 << 7)];
      if (context1 == 0) {
        mel_run -= 2;
        tv0 = (mel_run == -1) ? tv0 : 0;
        if (mel_run < 0) {
          mel_run = MEL.get_run();
        }
      }
      rho[0]   = static_cast<uint8_t>((tv0 & 0x00F0) >> 4);
      emb_k[0] = static_cast<uint8_t>((tv0 & 0x0F00) >> 8);
      emb_1[0] = static_cast<uint8_t>((tv0 & 0xF000) >> 12);
      vlcval   = VLC_dec.advance(static_cast<uint8_t>((tv0 & 0x000F) >> 1));
      if (qx > 0) {
        rho_p[qx - 1] = rho[Q1];
      }

#if defined(OPENHTJ2K_ENABLE_ARM_NEON)
      int32_t mask[4] = {1, 2, 4, 8};
      auto v0         = vdupq_n_s32(rho[Q0]);
      auto vm         = vld1q_s32(mask);
      auto vone       = vdupq_n_s32(1);
      vst1q_s32(sigma_quads, vandq_s32(vtstq_s32(v0, vm), vone));
#elif defined(__AVX2__)
      // do nothing
#else
      for (uint32_t i = 0; i < 4; i++) {
        sigma_quads[i] = (rho[Q0] >> i) & 1;
      }
#endif
      *sp0++ = (rho[Q0] >> 0) & 1;
      *sp0++ = (rho[Q0] >> 2) & 1;
      *sp1++ = (rho[Q0] >> 1) & 1;
      *sp1++ = (rho[Q0] >> 3) & 1;
      // calculate context for the current quad
      context2 = static_cast<uint16_t>((rho_p[qx] >> 3)
                                       | ((rho_p[qx + 1] >> 1) & 1)
                                             + (((rho_p[qx + 1] >> 3) | ((rho_p[qx + 2] >> 1) & 1)) << 2));
      context2 |= (((rho[Q0] >> 2) & 1) | ((rho[Q0] >> 3) & 1)) << 1;

      // Decoding of significance and EMB patterns and unsigned residual offsets
      uint16_t tv1 = dec_table1[(vlcval & 0x7F) + (context2 << 7)];
      if (context2 == 0) {
        mel_run -= 2;
        tv1 = (mel_run == -1) ? tv1 : 0;
        if (mel_run < 0) {
          mel_run = MEL.get_run();
        }
      }
      rho[1]   = static_cast<uint8_t>((tv1 & 0x00F0) >> 4);
      emb_k[1] = static_cast<uint8_t>((tv1 & 0x0F00) >> 8);
      emb_1[1] = static_cast<uint8_t>((tv1 & 0xF000) >> 12);
      vlcval   = VLC_dec.advance(static_cast<uint8_t>((tv1 & 0x000F) >> 1));
#if defined(OPENHTJ2K_ENABLE_ARM_NEON)
      auto v1 = vdupq_n_s32(rho[Q1]);
      vst1q_s32(sigma_quads + 4, vandq_s32(vtstq_s32(v1, vm), vone));
#elif defined(__AVX2__)
      auto vrho   = _mm256_setr_epi32(rho[0], rho[0], rho[0], rho[0], rho[1], rho[1], rho[1], rho[1]);
      auto vsigma = _mm256_and_si256(_mm256_srav_epi32(vrho, _mm256_setr_epi32(0, 1, 2, 3, 0, 1, 2, 3)),
                                     _mm256_set1_epi32(1));
      _mm256_store_si256((__m256i *)sigma_quads, vsigma);
#else
      for (uint32_t i = 0; i < 4; i++) {
        sigma_quads[i + 4] = (rho[Q1] >> i) & 1;
      }
#endif
      *sp0++ = (rho[Q1] >> 0) & 1;
      *sp0++ = (rho[Q1] >> 2) & 1;
      *sp1++ = (rho[Q1] >> 1) & 1;
      *sp1++ = (rho[Q1] >> 3) & 1;

      u_off[0] = tv0 & 1;
      u_off[1] = tv1 & 1;

      if (u_off[Q0] == 1 && u_off[Q1] == 1) {
        u_pfx[Q0] = VLC_dec.decodeUPrefix();
        u_pfx[Q1] = VLC_dec.decodeUPrefix();
        u_sfx[Q0] = VLC_dec.decodeUSuffix(u_pfx[Q0]);
        u_sfx[Q1] = VLC_dec.decodeUSuffix(u_pfx[Q1]);
        u_ext[Q0] = VLC_dec.decodeUExtension(u_sfx[Q0]);
        u_ext[Q1] = VLC_dec.decodeUExtension(u_sfx[Q1]);
        u[Q0]     = u_pfx[Q0] + u_sfx[Q0] + (u_ext[Q0] << 2);
        u[Q1]     = u_pfx[Q1] + u_sfx[Q1] + (u_ext[Q1] << 2);
      } else if (u_off[Q0] == 1 && u_off[Q1] == 0) {
        u_pfx[Q0] = VLC_dec.decodeUPrefix();
        u_sfx[Q0] = VLC_dec.decodeUSuffix(u_pfx[Q0]);
        u_ext[Q0] = VLC_dec.decodeUExtension(u_sfx[Q0]);
        u[Q0]     = u_pfx[Q0] + u_sfx[Q0] + (u_ext[Q0] << 2);
        u[Q1]     = 0;
      } else if (u_off[Q0] == 0 && u_off[Q1] == 1) {
        u_pfx[Q1] = VLC_dec.decodeUPrefix();
        u_sfx[Q1] = VLC_dec.decodeUSuffix(u_pfx[Q1]);
        u_ext[Q1] = VLC_dec.decodeUExtension(u_sfx[Q1]);
        u[Q0]     = 0;
        u[Q1]     = u_pfx[Q1] + u_sfx[Q1] + (u_ext[Q1] << 2);
      } else {
        u[Q0] = 0;
        u[Q1] = 0;
      }

      gamma[Q0]     = (popcount32(rho[Q0]) < 2) ? 0 : 1;
      gamma[Q1]     = (popcount32(rho[Q1]) < 2) ? 0 : 1;
      int32_t Emax0 = find_max(E_p[2 * qx - 1], E_p[2 * qx], E_p[2 * qx + 1], E_p[2 * qx + 2]);
      int32_t Emax1 =
          find_max(E_p[2 * (qx + 1) - 1], E_p[2 * (qx + 1)], E_p[2 * (qx + 1) + 1], E_p[2 * (qx + 1) + 2]);
      if (qx > 0) {
        E_p[2 * qx - 1] = static_cast<int32_t>(32 - count_leading_zeros(static_cast<uint32_t>(v_quads[7])));
      }
      kappa[Q0] = (1 > gamma[Q0] * (Emax0 - 1)) ? 1U : static_cast<uint8_t>(gamma[Q0] * (Emax0 - 1));
      kappa[Q1] = (1 > gamma[Q1] * (Emax1 - 1)) ? 1U : static_cast<uint8_t>(gamma[Q1] * (Emax1 - 1));
      U[Q0]     = kappa[Q0] + u[Q0];
      U[Q1]     = kappa[Q1] + u[Q1];

#if defined(OPENHTJ2K_ENABLE_ARM_NEON)
      v0          = vandq_s32(vtstq_s32(vdupq_n_s32(emb_k[Q0]), vm), vone);
      auto vsigma = vld1q_s32(sigma_quads);
      vst1q_u32(m_quads, vsubq_s32(vmulq_s32(vsigma, vdupq_n_s32(U[Q0])), v0));
      v0     = vandq_s32(vtstq_s32(vdupq_n_s32(emb_k[Q1]), vm), vone);
      vsigma = vld1q_s32(sigma_quads + 4);
      vst1q_u32(m_quads + 4, vsubq_s32(vmulq_s32(vsigma, vdupq_n_s32(U[Q1])), v0));
#elif defined(__AVX2__)
      auto vemb_k =
          _mm256_and_si256(_mm256_srav_epi32(_mm256_setr_epi32(emb_k[0], emb_k[0], emb_k[0], emb_k[0],
                                                               emb_k[1], emb_k[1], emb_k[1], emb_k[1]),
                                             _mm256_setr_epi32(0, 1, 2, 3, 0, 1, 2, 3)),
                           _mm256_set1_epi32(1));
      auto vU = _mm256_setr_epi32(U[0], U[0], U[0], U[0], U[1], U[1], U[1], U[1]);
      _mm256_store_si256((__m256i *)m_quads, _mm256_sub_epi32(_mm256_mullo_epi32(vsigma, vU), vemb_k));
#else
      for (uint32_t i = 0; i < 4; i++) {
        m_quads[i]     = sigma_quads[i] * U[Q0] - ((emb_k[Q0] >> i) & 1);
        m_quads[i + 4] = sigma_quads[i + 4] * U[Q1] - ((emb_k[Q1] >> i) & 1);
      }
#endif

      // recoverMagSgnValue
      for (uint32_t i = 0; i < 4; i++) {
        uint32_t msval = MagSgn.fetch();
        m_n[Q0]        = m_quads[i];
        known_1[Q0]    = (emb_1[Q0] >> i) & 1;
        v_quads[i]     = msval & ((1 << m_n[Q0]) - 1);
        v_quads[i] |= known_1[Q0] << m_n[Q0];
        MagSgn.advance(m_n[Q0]);

        if (m_n[Q0] != 0) {
          mu_quads[i] = static_cast<uint32_t>((v_quads[i] >> 1) + 1);
          mu_quads[i] <<= pLSB;
          mu_quads[i] |= static_cast<uint32_t>((v_quads[i] & 1) << 31);  // sign bit
        } else {
          mu_quads[i] = 0;
        }
      }
      *mp0++ = static_cast<int>(mu_quads[0]);
      *mp0++ = static_cast<int>(mu_quads[2]);
      *mp1++ = static_cast<int>(mu_quads[1]);
      *mp1++ = static_cast<int>(mu_quads[3]);
      for (uint32_t i = 0; i < 4; i++) {
        uint32_t msval = MagSgn.fetch();
        m_n[Q1]        = m_quads[i + 4];
        known_1[Q1]    = (emb_1[Q1] >> i) & 1;
        v_quads[i + 4] = msval & ((1 << m_n[Q1]) - 1);
        v_quads[i + 4] |= known_1[Q1] << m_n[Q1];
        MagSgn.advance(m_n[Q1]);
        if (m_n[Q1] != 0) {
          mu_quads[i + 4] = static_cast<uint32_t>((v_quads[i + 4] >> 1) + 1);
          mu_quads[i + 4] <<= pLSB;
          mu_quads[i + 4] |= static_cast<uint32_t>((v_quads[i + 4] & 1) << 31);  // sign bit
        } else {
          mu_quads[i + 4] = 0;
        }
      }
      *mp0++          = static_cast<int>(mu_quads[0 + 4]);
      *mp0++          = static_cast<int>(mu_quads[2 + 4]);
      *mp1++          = static_cast<int>(mu_quads[1 + 4]);
      *mp1++          = static_cast<int>(mu_quads[3 + 4]);
      rho_p[qx]       = rho[Q0];
      E_p[2 * qx]     = static_cast<int32_t>(32 - count_leading_zeros(static_cast<uint32_t>(v_quads[1])));
      E_p[2 * qx + 1] = static_cast<int32_t>(32 - count_leading_zeros(static_cast<uint32_t>(v_quads[3])));
      E_p[2 * qx + 2] = static_cast<int32_t>(32 - count_leading_zeros(static_cast<uint32_t>(v_quads[5])));
      if (qx + 1 == QW - 1) {
        rho_p[qx + 1]   = rho[Q1];
        E_p[2 * qx + 3] = static_cast<int32_t>(32 - count_leading_zeros(static_cast<uint32_t>(v_quads[7])));
      }
    }
    // if QW is odd number ..
    if (QW % 2 == 1) {
      // calculate context for the current quad
      context1 = static_cast<uint16_t>((rho_p[qx - 1] >> 3)
                                       | ((rho_p[qx] >> 1) & 1)
                                             + (((rho_p[qx] >> 3) | ((rho_p[qx + 1] >> 1) & 1)) << 2));
      if (qx > 0) {
        context1 |= (((rho[Q1] >> 2) & 1) | ((rho[Q1] >> 3) & 1)) << 1;
      }

      // Decoding of significance and EMB patterns and unsigned residual offsets
      vlcval       = VLC_dec.fetch();
      uint16_t tv0 = dec_table1[(vlcval & 0x7F) + (context1 << 7)];
      if (context1 == 0) {
        mel_run -= 2;
        tv0 = (mel_run == -1) ? tv0 : 0;
        if (mel_run < 0) {
          mel_run = MEL.get_run();
        }
      }
      rho[0]   = static_cast<uint8_t>((tv0 & 0x00F0) >> 4);
      emb_k[0] = static_cast<uint8_t>((tv0 & 0x0F00) >> 8);
      emb_1[0] = static_cast<uint8_t>((tv0 & 0xF000) >> 12);

      if (qx > 0) {
        rho_p[qx - 1] = rho[Q1];
      }

#if defined(OPENHTJ2K_ENABLE_ARM_NEON)
      int32_t mask[4] = {1, 2, 4, 8};
      auto v0         = vdupq_n_s32(rho[Q0]);
      auto vm         = vld1q_s32(mask);
      auto vone       = vdupq_n_s32(1);
      vst1q_s32(sigma_quads, vandq_s32(vtstq_s32(v0, vm), vone));
#elif defined(__AVX2__)
      for (uint32_t i = 0; i < 4; i++) {
        sigma_quads[i] = (rho[Q0] >> i) & 1;
      }
#else
      for (uint32_t i = 0; i < 4; i++) {
        sigma_quads[i] = (rho[Q0] >> i) & 1;
      }
#endif
      *sp0++   = (rho[Q0] >> 0) & 1;
      *sp0++   = (rho[Q0] >> 2) & 1;
      *sp1++   = (rho[Q0] >> 1) & 1;
      *sp1++   = (rho[Q0] >> 3) & 1;
      vlcval   = VLC_dec.advance(static_cast<uint8_t>((tv0 & 0x000F) >> 1));
      u_off[0] = tv0 & 1;

      if (u_off[Q0] == 1) {
        u_pfx[Q0] = VLC_dec.decodeUPrefix();
        u_sfx[Q0] = VLC_dec.decodeUSuffix(u_pfx[Q0]);
        u_ext[Q0] = VLC_dec.decodeUExtension(u_sfx[Q0]);
        u[Q0]     = u_pfx[Q0] + u_sfx[Q0] + (u_ext[Q0] << 2);
      } else {
        u[Q0] = 0;
      }

      gamma[Q0]     = (popcount32(rho[Q0]) < 2) ? 0 : 1;
      int32_t Emax0 = find_max(E_p[2 * qx - 1], E_p[2 * qx], E_p[2 * qx + 1], E_p[2 * qx + 2]);
      kappa[Q0]     = (1 > gamma[Q0] * (Emax0 - 1)) ? 1U : static_cast<uint8_t>(gamma[Q0] * (Emax0 - 1));
      U[Q0]         = kappa[Q0] + u[Q0];

#if defined(OPENHTJ2K_ENABLE_ARM_NEON)
      v0          = vandq_s32(vtstq_s32(vdupq_n_s32(emb_k[Q0]), vm), vone);
      auto vsigma = vld1q_s32(sigma_quads);
      vst1q_u32(m_quads, vsubq_s32(vmulq_s32(vsigma, vdupq_n_s32(U[Q0])), v0));
#elif defined(__AVX2__)
      for (uint32_t i = 0; i < 4; i++) {
        m_quads[i] = sigma_quads[i] * U[Q0] - ((emb_k[Q0] >> i) & 1);
      }
#else
      for (uint32_t i = 0; i < 4; i++) {
        m_quads[i] = sigma_quads[i] * U[Q0] - ((emb_k[Q0] >> i) & 1);
      }
#endif

      // recoverMagSgnValue
      for (uint32_t i = 0; i < 4; i++) {
        uint32_t msval = MagSgn.fetch();
        m_n[Q0]        = m_quads[i];
        known_1[Q0]    = (emb_1[Q0] >> i) & 1;
        v_quads[i]     = msval & ((1 << m_n[Q0]) - 1);
        v_quads[i] |= known_1[Q0] << m_n[Q0];
        MagSgn.advance(m_n[Q0]);

        if (m_n[Q0] != 0) {
          mu_quads[i] = static_cast<uint32_t>((v_quads[i] >> 1) + 1);
          mu_quads[i] <<= pLSB;
          mu_quads[i] |= static_cast<uint32_t>((v_quads[i] & 1) << 31);
        } else {
          mu_quads[i] = 0;
        }
      }
      *mp0++    = static_cast<int>(mu_quads[0]);
      *mp0++    = static_cast<int>(mu_quads[2]);
      *mp1++    = static_cast<int>(mu_quads[1]);
      *mp1++    = static_cast<int>(mu_quads[3]);
      rho_p[qx] = rho[Q0];
      if (qx > 0) {
        E_p[2 * qx - 1] = static_cast<int32_t>(32 - count_leading_zeros(static_cast<uint32_t>(v_quads[7])));
      }
      E_p[2 * qx]     = static_cast<int32_t>(32 - count_leading_zeros(static_cast<uint32_t>(v_quads[1])));
      E_p[2 * qx + 1] = static_cast<int32_t>(32 - count_leading_zeros(static_cast<uint32_t>(v_quads[3])));
    }  // Non-Initial line-pair end
  }
}

auto process_stripes_block_dec = [](SP_dec &SigProp, j2k_codeblock *block, const int32_t i_start,
                                    const int32_t j_start, const uint16_t width, const uint16_t height,
                                    const uint8_t &pLSB) {
  int32_t *sp;
  uint8_t causal_cond = 0;
  uint8_t bit;
  uint8_t mbr;
  // uint32_t mbr_info;
  const auto block_width  = static_cast<uint16_t>(j_start + width);
  const auto block_height = static_cast<uint16_t>(i_start + height);

  for (int16_t j = (int16_t)j_start; j < block_width; j++) {
    // mbr_info = 0;
    //     for (int16_t i = height; i > -2; --i) {
    //       causal_cond = (((block->Cmodes & CAUSAL) == 0) || (i != height));
    //       mbr_info <<= 3;
    //       mbr_info |= (block->get_sigma(i_start + i, j - 1) + (block->get_sigma(i_start + i, j) << 1)
    //                    + (block->get_sigma(i_start + i, j + 1) << 2))
    //                   * causal_cond;
    //     }
    for (int16_t i = (int16_t)i_start; i < block_height; i++) {
      sp = &block->sample_buf[static_cast<uint32_t>(j) + static_cast<uint32_t>(i) * block->blksampl_stride];
      causal_cond = (((block->Cmodes & CAUSAL) == 0) || (i != block_height - 1));
      mbr         = 0;
      if (block->get_state(Sigma, i, j) == 0) {
        mbr = block->calc_mbr(i, j, causal_cond);
      }
      // mbr_info >>= 3;
      if (mbr != 0) {
        block->modify_state(refinement_indicator, 1, i, j);
        bit = SigProp.importSigPropBit();
        block->modify_state(refinement_value, bit, i, j);
        // block->set_refinement_value(bit, i, j);
        *sp |= bit << pLSB;
      }
      block->modify_state(scan, 1, i, j);
      // block->update_scan_state(1, i, j);
    }
  }
  for (int16_t j = (int16_t)j_start; j < block_width; j++) {
    for (int16_t i = (int16_t)i_start; i < block_height; i++) {
      sp = &block->sample_buf[static_cast<uint32_t>(j) + static_cast<uint32_t>(i) * block->blksampl_stride];
      // decode sign
      if ((*sp & (1 << pLSB)) != 0) {
        *sp = (*sp & 0x7FFFFFFF) | (SigProp.importSigPropBit() << 31);
      }
    }
  }
};

void ht_sigprop_decode(j2k_codeblock *block, uint8_t *HT_magref_segment, uint32_t magref_length,
                       const uint8_t &pLSB) {
  SP_dec SigProp(HT_magref_segment, magref_length);
  const uint16_t num_v_stripe = static_cast<uint16_t>(block->size.y / 4);
  const uint16_t num_h_stripe = static_cast<uint16_t>(block->size.x / 4);
  int32_t i_start             = 0, j_start;
  uint16_t width              = 4;
  uint16_t width_last;
  uint16_t height = 4;

  // decode full-height (=4) stripes
  for (uint16_t n1 = 0; n1 < num_v_stripe; n1++) {
    j_start = 0;
    for (uint16_t n2 = 0; n2 < num_h_stripe; n2++) {
      process_stripes_block_dec(SigProp, block, i_start, j_start, width, height, pLSB);
      j_start += 4;
    }
    width_last = block->size.x % 4;
    if (width_last) {
      process_stripes_block_dec(SigProp, block, i_start, j_start, width_last, height, pLSB);
    }
    i_start += 4;
  }
  // decode remaining height stripes
  height  = block->size.y % 4;
  j_start = 0;
  for (uint16_t n2 = 0; n2 < num_h_stripe; n2++) {
    process_stripes_block_dec(SigProp, block, i_start, j_start, width, height, pLSB);
    j_start += 4;
  }
  width_last = block->size.x % 4;
  if (width_last) {
    process_stripes_block_dec(SigProp, block, i_start, j_start, width_last, height, pLSB);
  }
}

void ht_magref_decode(j2k_codeblock *block, uint8_t *HT_magref_segment, uint32_t magref_length,
                      const uint8_t &pLSB) {
  MR_dec MagRef(HT_magref_segment, magref_length);
  const uint16_t blk_height   = static_cast<uint16_t>(block->size.y);
  const uint16_t blk_width    = static_cast<uint16_t>(block->size.x);
  const uint16_t num_v_stripe = static_cast<uint16_t>(block->size.y / 4);
  int16_t i_start             = 0;
  int16_t height              = 4;
  int32_t *sp;

  for (int16_t n1 = 0; n1 < num_v_stripe; n1++) {
    for (int16_t j = 0; j < blk_width; j++) {
      for (int16_t i = i_start; i < i_start + height; i++) {
        sp =
            &block
                 ->sample_buf[static_cast<uint32_t>(j) + static_cast<uint32_t>(i) * block->blksampl_stride];
        if (block->get_state(Sigma, i, j) != 0) {
          block->modify_state(refinement_indicator, 1, i, j);
          sp[0] |= MagRef.importMagRefBit() << pLSB;
        }
      }
    }
    i_start = static_cast<int16_t>(i_start + 4);
  }
  height = static_cast<int16_t>(blk_height % 4);
  for (int16_t j = 0; j < blk_width; j++) {
    for (int16_t i = i_start; i < i_start + height; i++) {
      sp = &block->sample_buf[static_cast<uint32_t>(j) + static_cast<uint32_t>(i) * block->blksampl_stride];
      if (block->get_state(Sigma, i, j) != 0) {
        block->modify_state(refinement_indicator, 1, i, j);
        sp[0] |= MagRef.importMagRefBit() << pLSB;
      }
    }
  }
}

void j2k_codeblock::dequantize(uint8_t S_blk, uint8_t ROIshift) {
  /* ready for ROI adjustment and dequantization */

  // number of decoded magnitude bit‐planes
  const int32_t pLSB = 31 - M_b;  // indicates binary point;

  // bit mask for ROI detection
  const uint32_t mask = UINT32_MAX >> (M_b + 1);
  // reconstruction parameter defined in E.1.1.2 of the spec

  float fscale = this->stepsize;
  fscale *= (1 << FRACBITS);
  if (M_b <= 31) {
    fscale /= (static_cast<float>(1 << (31 - M_b)));
  } else {
    fscale *= (static_cast<float>(1 << (M_b - 31)));
  }
  constexpr int32_t downshift = 15;
  fscale *= (float)(1 << 16) * (float)(1 << downshift);
  const auto scale = (int32_t)(fscale + 0.5);
  if (this->transformation) {
    // lossless path
    for (size_t i = 0; i < static_cast<size_t>(this->size.y); i++) {
      int32_t *val      = this->sample_buf.get() + i * this->blksampl_stride;
      sprec_t *dst      = this->i_samples + i * this->band_stride;
      uint8_t *blkstate = this->block_states.get() + (i + 1) * this->blkstate_stride + 1;
#if defined(OPENHTJ2K_ENABLE_ARM_NEON)
      // TODO: test will fail with gcc
      size_t simdlen = static_cast<size_t>(this->size.x) - static_cast<size_t>(this->size.x) % 8;
      auto vmask     = vdupq_n_s32(static_cast<int32_t>(~mask));
      for (size_t j = 0; j < simdlen; j += 8) {
        auto vsrc0  = vld1q_s32(val);
        auto vsrc1  = vld1q_s32(val + 4);
        auto vsign0 = vcltzq_s32(vsrc0) >> 31;
        auto vsign1 = vcltzq_s32(vsrc1) >> 31;
        vsrc0       = vsrc0 & INT32_MAX;
        vsrc1       = vsrc1 & INT32_MAX;
        // upshift background region, if necessary
        auto vROImask = vandq_s32(vsrc0, vmask);
        vROImask      = vceqzq_s32(vROImask);
        vROImask &= vdupq_n_s32(ROIshift);
        vsrc0    = vshlq_s32(vsrc0, vROImask);
        vROImask = vandq_s32(vsrc1, vmask);
        vROImask = vceqzq_s32(vROImask);
        vROImask &= vdupq_n_s32(ROIshift);
        vsrc1 = vshlq_s32(vsrc1, vROImask);

        // retrieve number of decoded magnitude bit-planes
        auto vstate = vld1_u8(blkstate);
        vstate >>= 2;
        vstate &= 1;
        auto vNb0 = vdupq_n_s32(S_blk + 1) + vmovl_s16(vget_low_s16(vmovl_s8(vstate)));
        auto vNb1 = vdupq_n_s32(S_blk + 1) + vmovl_s16(vget_high_s16(vmovl_s8(vstate)));

        // add reconstruction value, if necessary (it will happen for a truncated codestream)
        auto vMb           = vdupq_n_s32(M_b);
        auto v_recval_mask = vcgtq_s32(vMb, vNb0);
        v_recval_mask &= vcgtzq_s32(vsrc0);
        auto vrecval0 = (1 << (31 - vNb0 - 1)) & v_recval_mask;
        v_recval_mask = vcgtq_s32(vMb, vNb1);
        v_recval_mask &= vcgtzq_s32(vsrc1);
        auto vrecval1 = (1 << (31 - vNb1 - 1)) & v_recval_mask;
        vsrc0 |= vrecval0;
        vsrc1 |= vrecval1;

        // convert vlues from sign-magnitude form to two's complement one
        auto vnegmask = vcltzq_s32(vsrc0 | (vsign0 << 31));
        auto vposmask = ~vnegmask;
        auto vdst0    = (vnegq_s32(vsrc0) & vnegmask) + (vsrc0 & vposmask);
        vnegmask      = vcltzq_s32(vsrc1 | (vsign1 << 31));
        vposmask      = ~vnegmask;
        auto vdst1    = (vnegq_s32(vsrc1) & vnegmask) + (vsrc1 & vposmask);
        vst1q_s16(dst, vcombine_s16(vmovn_s32(vdst0 >> pLSB), vmovn_s32(vdst1 >> pLSB)));
        val += 8;
        dst += 8;
        blkstate += 8;
      }
      for (size_t j = static_cast<size_t>(this->size.x) - static_cast<size_t>(this->size.x) % 8;
           j < static_cast<size_t>(this->size.x); j++) {
        int32_t sign = *val & INT32_MIN;
        *val &= INT32_MAX;
        // detect background region and upshift it
        if (ROIshift && (((uint32_t)*val & ~mask) == 0)) {
          *val <<= ROIshift;
        }
        // do adjustment of the position indicating 0.5
        int32_t N_b = S_blk + 1 + ((*blkstate >> 2) & 1);
        if (ROIshift) {
          N_b = M_b;
        }
        if (N_b < M_b && *val) {
          *val |= 1 << (31 - N_b - 1);
        }
        // bring sign back
        *val |= sign;
        // convert sign-magnitude to two's complement form
        if (*val < 0) {
          *val = -(*val & INT32_MAX);
        }

        assert(pLSB >= 0);  // assure downshift is not negative
        *dst = static_cast<int16_t>(*val >> pLSB);
        val++;
        dst++;
        blkstate++;
      }
#elif defined(OPENHTJ2K_TRY_AVX2) && defined(__AVX2__)
      size_t simdlen = static_cast<size_t>(this->size.x) - static_cast<size_t>(this->size.x) % 16;
      for (size_t j = 0; j < simdlen; j += 16) {
        auto vsrc0 = _mm256_loadu_si256((__m256i *)val);
        auto vsrc1 = _mm256_loadu_si256((__m256i *)(val + 8));
        auto vsign0 =
            _mm256_or_si256(_mm256_and_si256(vsrc0, _mm256_set1_epi32(INT32_MIN)), _mm256_set1_epi32(1));
        auto vsign1 =
            _mm256_or_si256(_mm256_and_si256(vsrc1, _mm256_set1_epi32(INT32_MIN)), _mm256_set1_epi32(1));
        vsrc0 = _mm256_and_si256(vsrc0, _mm256_set1_epi32(0x7FFFFFFF));
        vsrc1 = _mm256_and_si256(vsrc1, _mm256_set1_epi32(0x7FFFFFFF));
        // upshift background region, if necessary
        auto vROImask = _mm256_and_si256(vsrc0, _mm256_set1_epi32(static_cast<int32_t>(~mask)));
        vROImask      = _mm256_cmpeq_epi32(vROImask, _mm256_setzero_si256());
        vROImask      = _mm256_and_si256(vROImask, _mm256_set1_epi32(ROIshift));
        vsrc0         = _mm256_sllv_epi32(vsrc0, vROImask);
        vROImask      = _mm256_and_si256(vsrc1, _mm256_set1_epi32(static_cast<int32_t>(~mask)));
        vROImask      = _mm256_cmpeq_epi32(vROImask, _mm256_setzero_si256());
        vROImask      = _mm256_and_si256(vROImask, _mm256_set1_epi32(ROIshift));
        vsrc1         = _mm256_sllv_epi32(vsrc1, vROImask);

        // retrieve number of decoded magnitude bit-planes
        auto vstate      = _mm_loadu_si128((__m128i *)blkstate);
        auto vstate_low  = _mm256_cvtepi8_epi32(vstate);
        auto vstate_high = _mm256_cvtepi8_epi32(_mm_srli_si128(vstate, 8));
        vstate_low       = _mm256_and_si256(_mm256_srai_epi32(vstate_low, 2), _mm256_set1_epi32(1));
        vstate_high      = _mm256_and_si256(_mm256_srai_epi32(vstate_high, 2), _mm256_set1_epi32(1));
        auto vNb0        = _mm256_add_epi32(_mm256_set1_epi32(S_blk + 1), vstate_low);
        auto vNb1        = _mm256_add_epi32(_mm256_set1_epi32(S_blk + 1), vstate_high);

        // add reconstruction value, if necessary (it will happen for a truncated codestream)
        auto vMb           = _mm256_set1_epi32(M_b);
        auto v_recval_mask = _mm256_cmpgt_epi32(vMb, vNb0);
        v_recval_mask = _mm256_and_si256(v_recval_mask, _mm256_cmpgt_epi32(vsrc0, _mm256_setzero_si256()));
        auto vrecval0 = _mm256_and_si256(
            _mm256_sllv_epi32(_mm256_set1_epi32(1), _mm256_sub_epi32(_mm256_set1_epi32(30), vNb0)),
            v_recval_mask);
        v_recval_mask = _mm256_cmpgt_epi32(vMb, vNb1);
        v_recval_mask = _mm256_and_si256(v_recval_mask, _mm256_cmpgt_epi32(vsrc1, _mm256_setzero_si256()));
        auto vrecval1 = _mm256_and_si256(
            _mm256_sllv_epi32(_mm256_set1_epi32(1), _mm256_sub_epi32(_mm256_set1_epi32(30), vNb1)),
            v_recval_mask);
        vsrc0 = _mm256_or_si256(vsrc0, vrecval0);
        vsrc1 = _mm256_or_si256(vsrc1, vrecval1);

        // convert values from sign-magnitude form to two's complement one
        auto vdst0 = _mm256_srai_epi32(_mm256_sign_epi32(vsrc0, vsign0), pLSB);
        auto vdst1 = _mm256_srai_epi32(_mm256_sign_epi32(vsrc1, vsign1), pLSB);
        _mm256_storeu_si256((__m256i *)dst,
                            _mm256_permute4x64_epi64(_mm256_packs_epi32(vdst0, vdst1), 0xD8));
        val += 16;
        dst += 16;
        blkstate += 16;
      }
      for (size_t j = static_cast<size_t>(this->size.x) - static_cast<size_t>(this->size.x) % 16;
           j < static_cast<size_t>(this->size.x); j++) {
        int32_t sign = *val & INT32_MIN;
        *val &= INT32_MAX;
        // detect background region and upshift it
        if (ROIshift && (((uint32_t)*val & ~mask) == 0)) {
          *val <<= ROIshift;
        }
        // do adjustment of the position indicating 0.5
        int32_t N_b = S_blk + 1 + ((*blkstate >> 2) & 1);
        if (ROIshift) {
          N_b = M_b;
        }
        if (N_b < M_b && *val) {
          *val |= 1 << (31 - N_b - 1);
        }
        // bring sign back
        *val |= sign;
        // convert sign-magnitude to two's complement form
        if (*val < 0) {
          *val = -(*val & INT32_MAX);
        }

        assert(pLSB >= 0);  // assure downshift is not negative
        *dst = static_cast<int16_t>(*val >> pLSB);
        val++;
        dst++;
        blkstate++;
      }
#else
      for (size_t j = 0; j < static_cast<size_t>(this->size.x); j++) {
        int32_t sign = *val & INT32_MIN;
        *val &= INT32_MAX;
        // detect background region and upshift it
        if (ROIshift && (((uint32_t)*val & ~mask) == 0)) {
          *val <<= ROIshift;
        }
        // do adjustment of the position indicating 0.5
        int32_t N_b = S_blk + 1 + ((*blkstate >> 2) & 1);
        if (ROIshift) {
          N_b = M_b;
        }
        if (N_b < M_b && *val) {
          *val |= 1 << (31 - N_b - 1);
        }
        // bring sign back
        *val |= sign;
        // convert sign-magnitude to two's complement form
        if (*val < 0) {
          *val = -(*val & INT32_MAX);
        }

        assert(pLSB >= 0);  // assure downshift is not negative
        *dst = static_cast<int16_t>(*val >> pLSB);
        val++;
        dst++;
        blkstate++;
      }
#endif
    }
  } else {
    // lossy path
    int32_t ROImask = 0;
    if (ROIshift) {
      ROImask = static_cast<int32_t>(0xFFFFFFFF);
    }
    //    auto vROIshift = vdupq_n_s32(ROImask);
    for (size_t i = 0; i < static_cast<size_t>(this->size.y); i++) {
      int32_t *val      = this->sample_buf.get() + i * this->blksampl_stride;
      sprec_t *dst      = this->i_samples + i * this->band_stride;
      uint8_t *blkstate = this->block_states.get() + (i + 1) * this->blkstate_stride + 1;
#if defined(OPENHTJ2K_ENABLE_ARM_NEON)
      size_t simdlen = static_cast<size_t>(this->size.x) - static_cast<size_t>(this->size.x) % 8;
      auto vmask     = vdupq_n_s32(static_cast<int32_t>(~mask));
      for (size_t j = 0; j < simdlen; j += 8) {
        auto vsrc0  = vld1q_s32(val);
        auto vsrc1  = vld1q_s32(val + 4);
        auto vsign0 = vcltzq_s32(vsrc0) >> 31;
        auto vsign1 = vcltzq_s32(vsrc1) >> 31;
        vsrc0       = vsrc0 & INT32_MAX;
        vsrc1       = vsrc1 & INT32_MAX;
        // upshift background region, if necessary
        auto vROImask = vandq_s32(vsrc0, vmask);
        vROImask      = vceqzq_s32(vROImask);
        vROImask &= vdupq_n_s32(ROIshift);
        vsrc0    = vshlq_s32(vsrc0, vROImask);
        vROImask = vandq_s32(vsrc1, vmask);
        vROImask = vceqzq_s32(vROImask);
        vROImask &= vdupq_n_s32(ROIshift);
        vsrc1 = vshlq_s32(vsrc1, vROImask);

        // retrieve number of decoded magnitude bit-planes
        auto vstate = vld1_u8(blkstate);
        vstate >>= 2;
        vstate &= 1;
        auto vNb0 = vdupq_n_s32(S_blk + 1) + vmovl_s16(vget_low_s16(vmovl_s8(vstate)));
        auto vNb1 = vdupq_n_s32(S_blk + 1) + vmovl_s16(vget_high_s16(vmovl_s8(vstate)));
        if (ROIshift) {
          vNb0 = vdupq_n_s32(M_b);
          vNb1 = vdupq_n_s32(M_b);
        }
        // add reconstruction value, if necessary (it will happen for a truncated codestream)
        auto v_recval_mask = vcgtzq_s32(vsrc0);
        auto vrecval0      = (1 << (31 - vNb0 - 1)) & v_recval_mask;
        v_recval_mask      = vcgtzq_s32(vsrc1);
        auto vrecval1      = (1 << (31 - vNb1 - 1)) & v_recval_mask;
        vsrc0 |= vrecval0;
        vsrc1 |= vrecval1;

        // to prevent overflow, truncate to int16_t range
        vsrc0 = (vsrc0 + (1 << 15)) >> 16;
        vsrc1 = (vsrc1 + (1 << 15)) >> 16;

        // dequantization
        vsrc0 = vmulq_s32(vsrc0, vdupq_n_s32(scale));
        vsrc1 = vmulq_s32(vsrc1, vdupq_n_s32(scale));

        // downshift and convert values from sign-magnitude form to two's complement one
        auto vdst     = vcombine_s16(vmovn_s32((vsrc0 + (1 << (downshift - 1))) >> downshift),
                                     vmovn_s32((vsrc1 + (1 << (downshift - 1))) >> downshift));
        auto vsign    = vcombine_s16(vmovn_s32(vsign0), vmovn_s32(vsign1));
        auto vnegmask = vcltzq_s16(vdst | (vsign << 15));
        auto vposmask = ~vnegmask;
        vdst          = (vnegq_s16(vdst) & vnegmask) + (vdst & vposmask);
        vst1q_s16(dst, vdst);

        val += 8;
        dst += 8;
        blkstate += 8;
      }
      for (size_t j = static_cast<size_t>(this->size.x) - static_cast<size_t>(this->size.x) % 8;
           j < static_cast<size_t>(this->size.x); j++) {
        int32_t sign = *val & INT32_MIN;
        *val &= INT32_MAX;
        // detect background region and upshift it
        if (ROIshift && (((uint32_t)*val & ~mask) == 0)) {
          *val <<= ROIshift;
        }
        // do adjustment of the position indicating 0.5
        int32_t N_b = S_blk + 1 + ((*blkstate >> 2) & 1);
        if (ROIshift) {
          N_b = M_b;
        }
        if (*val) {
          *val |= 1 << (31 - N_b - 1);
        }

        // to prevent overflow, truncate to int16_t
        *val = (*val + (1 << 15)) >> 16;
        //  dequantization
        *val *= scale;
        // downshift
        *dst = (int16_t)((*val + (1 << (downshift - 1))) >> downshift);
        // convert sign-magnitude to two's complement form
        if (sign) {
          *dst = static_cast<int16_t>(-(*dst));
        }
        val++;
        dst++;
        blkstate++;
      }
#elif defined(OPENHTJ2K_TRY_AVX2) && defined(__AVX2__)
      size_t simdlen = static_cast<size_t>(this->size.x) - static_cast<size_t>(this->size.x) % 16;
      for (size_t j = 0; j < simdlen; j += 16) {
        auto vsrc0 = _mm256_loadu_si256((__m256i *)val);
        auto vsrc1 = _mm256_loadu_si256((__m256i *)(val + 8));
        auto vsign0 =
            _mm256_or_si256(_mm256_and_si256(vsrc0, _mm256_set1_epi32(INT32_MIN)), _mm256_set1_epi32(1));
        auto vsign1 =
            _mm256_or_si256(_mm256_and_si256(vsrc1, _mm256_set1_epi32(INT32_MIN)), _mm256_set1_epi32(1));
        vsrc0 = _mm256_and_si256(vsrc0, _mm256_set1_epi32(0x7FFFFFFF));
        vsrc1 = _mm256_and_si256(vsrc1, _mm256_set1_epi32(0x7FFFFFFF));
        // upshift background region, if necessary
        auto vROImask = _mm256_and_si256(vsrc0, _mm256_set1_epi32(static_cast<int32_t>(~mask)));
        vROImask      = _mm256_cmpeq_epi32(vROImask, _mm256_setzero_si256());
        vROImask      = _mm256_and_si256(vROImask, _mm256_set1_epi32(ROIshift));
        vsrc0         = _mm256_sllv_epi32(vsrc0, vROImask);
        vROImask      = _mm256_and_si256(vsrc1, _mm256_set1_epi32(static_cast<int32_t>(~mask)));
        vROImask      = _mm256_cmpeq_epi32(vROImask, _mm256_setzero_si256());
        vROImask      = _mm256_and_si256(vROImask, _mm256_set1_epi32(ROIshift));
        vsrc1         = _mm256_sllv_epi32(vsrc1, vROImask);

        // retrieve number of decoded magnitude bit-planes
        auto vstate      = _mm_loadu_si128((__m128i *)blkstate);
        auto vstate_low  = _mm256_cvtepi8_epi32(vstate);
        auto vstate_high = _mm256_cvtepi8_epi32(_mm_srli_si128(vstate, 8));
        vstate_low       = _mm256_and_si256(_mm256_srai_epi32(vstate_low, 2), _mm256_set1_epi32(1));
        vstate_high      = _mm256_and_si256(_mm256_srai_epi32(vstate_high, 2), _mm256_set1_epi32(1));
        auto vNb0        = _mm256_add_epi32(_mm256_set1_epi32(S_blk + 1), vstate_low);
        auto vNb1        = _mm256_add_epi32(_mm256_set1_epi32(S_blk + 1), vstate_high);
        if (ROIshift) {
          vNb0 = _mm256_set1_epi32(M_b);
          vNb1 = _mm256_set1_epi32(M_b);
        }

        // add reconstruction value, if necessary (it will happen for a truncated codestream)
        auto v_recval_mask = _mm256_cmpgt_epi32(vsrc0, _mm256_setzero_si256());
        auto vrecval0      = _mm256_and_si256(
                 _mm256_sllv_epi32(_mm256_set1_epi32(1), _mm256_sub_epi32(_mm256_set1_epi32(30), vNb0)),
                 v_recval_mask);
        v_recval_mask = _mm256_cmpgt_epi32(vsrc1, _mm256_setzero_si256());
        auto vrecval1 = _mm256_and_si256(
            _mm256_sllv_epi32(_mm256_set1_epi32(1), _mm256_sub_epi32(_mm256_set1_epi32(30), vNb1)),
            v_recval_mask);
        vsrc0 = _mm256_or_si256(vsrc0, vrecval0);
        vsrc1 = _mm256_or_si256(vsrc1, vrecval1);

        // to prevent overflow, truncate to int16_t range
        vsrc0 = _mm256_srai_epi32(_mm256_add_epi32(vsrc0, _mm256_set1_epi32(1 << 15)), 16);
        vsrc1 = _mm256_srai_epi32(_mm256_add_epi32(vsrc1, _mm256_set1_epi32(1 << 15)), 16);

        // dequantization
        vsrc0 = _mm256_mullo_epi32(vsrc0, _mm256_set1_epi32(scale));
        vsrc1 = _mm256_mullo_epi32(vsrc1, _mm256_set1_epi32(scale));

        // downshift and convert values from sign-magnitude form to two's complement one
        vsrc0 =
            _mm256_srai_epi32(_mm256_add_epi32(vsrc0, _mm256_set1_epi32(1 << (downshift - 1))), downshift);
        vsrc1 =
            _mm256_srai_epi32(_mm256_add_epi32(vsrc1, _mm256_set1_epi32(1 << (downshift - 1))), downshift);

        vsrc0 = _mm256_sign_epi32(vsrc0, vsign0);
        vsrc1 = _mm256_sign_epi32(vsrc1, vsign1);

        _mm256_storeu_si256((__m256i *)dst,
                            _mm256_permute4x64_epi64(_mm256_packs_epi32(vsrc0, vsrc1), 0xD8));

        val += 16;
        dst += 16;
        blkstate += 16;
      }
      for (size_t j = static_cast<size_t>(this->size.x) - static_cast<size_t>(this->size.x) % 16;
           j < static_cast<size_t>(this->size.x); j++) {
        int32_t sign = *val & INT32_MIN;
        *val &= INT32_MAX;
        // detect background region and upshift it
        if (ROIshift && (((uint32_t)*val & ~mask) == 0)) {
          *val <<= ROIshift;
        }
        // do adjustment of the position indicating 0.5
        int32_t N_b = S_blk + 1 + ((*blkstate >> 2) & 1);
        if (ROIshift) {
          N_b = M_b;
        }
        if (*val) {
          *val |= 1 << (31 - N_b - 1);
        }

        // to prevent overflow, truncate to int16_t
        *val = (*val + (1 << 15)) >> 16;
        //  dequantization
        *val *= scale;
        // downshift
        *dst = (int16_t)((*val + (1 << (downshift - 1))) >> downshift);
        // convert sign-magnitude to two's complement form
        if (sign) {
          *dst = static_cast<int16_t>(-(*dst));
        }
        val++;
        dst++;
        blkstate++;
      }
#else
      for (size_t j = 0; j < static_cast<size_t>(this->size.x); j++) {
        int32_t sign = *val & INT32_MIN;
        *val &= INT32_MAX;
        // detect background region and upshift it
        if (ROIshift && (((uint32_t)*val & ~mask) == 0)) {
          *val <<= ROIshift;
        }
        // do adjustment of the position indicating 0.5
        int32_t N_b = S_blk + 1 + ((*blkstate >> 2) & 1);
        if (ROIshift) {
          N_b = M_b;
        }
        if (*val) {
          *val |= 1 << (31 - N_b - 1);
        }

        // to prevent overflow, truncate to int16_t
        *val = (*val + (1 << 15)) >> 16;
        //  dequantization
        *val *= scale;
        // downshift
        *dst = (int16_t)((*val + (1 << (downshift - 1))) >> downshift);
        // convert sign-magnitude to two's complement form
        if (sign) {
          *dst = static_cast<int16_t>(-(*dst));
        }
        val++;
        dst++;
        blkstate++;
      }
#endif
    }
  }
}

bool htj2k_decode(j2k_codeblock *block, const uint8_t ROIshift) {
  // number of placeholder pass
  uint8_t P0 = 0;
  // length of HT Cleanup segment
  uint32_t Lcup = 0;
  // length of HT Refinement segment
  uint32_t Lref = 0;
  // number of HT Sets preceding the given(this) HT Set
  const uint8_t S_skip = 0;

  if (block->num_passes > 3) {
    for (uint32_t i = 0; i < block->pass_length.size(); i++) {
      if (block->pass_length[i] != 0) {
        break;
      }
      P0++;
    }
    P0 /= 3;
  } else if (block->length == 0 && block->num_passes != 0) {
    P0 = 1;
  } else {
    P0 = 0;
  }
  const uint8_t empty_passes = static_cast<uint8_t>(P0 * 3);
  if (block->num_passes < empty_passes) {
    printf("WARNING: number of passes %d exceeds number of empty passes %d", block->num_passes,
           empty_passes);
    return false;
  }
  // number of ht coding pass (Z_blk in the spec)
  const uint8_t num_ht_passes = static_cast<uint8_t>(block->num_passes - empty_passes);
  // pointer to buffer for HT Cleanup segment
  uint8_t *Dcup;
  // pointer to buffer for HT Refinement segment
  uint8_t *Dref;

  if (num_ht_passes > 0) {
    std::vector<uint8_t> all_segments;
    all_segments.reserve(3);
    for (uint32_t i = 0; i < block->pass_length.size(); i++) {
      if (block->pass_length[i] != 0) {
        all_segments.push_back(static_cast<uint8_t>(i));
      }
    }
    Lcup += block->pass_length[all_segments[0]];
    if (Lcup < 2) {
      printf("WARNING: Cleanup pass length must be at least 2 bytes in length.\n");
      return false;
    }
    for (uint32_t i = 1; i < all_segments.size(); i++) {
      Lref += block->pass_length[all_segments[i]];
    }
    Dcup = block->get_compressed_data();

    if (block->num_passes > 1 && all_segments.size() > 1) {
      Dref = block->get_compressed_data() + Lcup;
    } else {
      Dref = nullptr;
    }
    // number of (skipped) magnitude bitplanes
    const uint8_t S_blk = static_cast<uint8_t>(P0 + block->num_ZBP + S_skip);
    if (S_blk >= 30) {
      printf("WARNING: Number of skipped mag bitplanes %d is too large.\n", S_blk);
      return false;
    }
    // Suffix length (=MEL + VLC) of HT Cleanup pass
    const uint32_t Scup = static_cast<uint32_t>((Dcup[Lcup - 1] << 4) + (Dcup[Lcup - 2] & 0x0F));
    if (Scup < 2 || Scup > Lcup || Scup > 4079) {
      printf("WARNING: cleanup pass suffix length %d is invalid.\n", Scup);
      return false;
    }
    // modDcup (shall be done before the creation of state_VLC instance)
    Dcup[Lcup - 1] = 0xFF;
    Dcup[Lcup - 2] |= 0x0F;
    const int32_t Pcup = static_cast<const int32_t>(Lcup - Scup);
    //    state_MS_dec MS     = state_MS_dec(Dcup, Pcup);
    fwd_buf<0xFF> MagSgn(Dcup, Pcup);
    //    state_MEL_unPacker MEL_unPacker = state_MEL_unPacker(Dcup, Lcup, Pcup);
    //    state_MEL_decoder MEL_decoder   = state_MEL_decoder(MEL_unPacker);
    //    state_VLC_dec VLC               = state_VLC_dec(Dcup, Lcup, Pcup);
    MEL_dec MEL(Dcup, Lcup, Scup);
    rev_buf VLCdec(Dcup, Lcup, Scup);
    ht_cleanup_decode(block, static_cast<uint8_t>(30 - S_blk), MagSgn, MEL, VLCdec);
    if (num_ht_passes > 1) {
      ht_sigprop_decode(block, Dref, Lref, static_cast<uint8_t>(30 - (S_blk + 1)));
    }
    if (num_ht_passes > 2) {
      ht_magref_decode(block, Dref, Lref, static_cast<uint8_t>(30 - (S_blk + 1)));
    }

    // dequantization
    block->dequantize(S_blk, ROIshift);

  }  // end

  return true;
}
