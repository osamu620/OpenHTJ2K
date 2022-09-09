// Copyright (c) 2019 - 2022, Osamu Watanabe
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
  #include "coding_units.hpp"
  #include "dec_CxtVLC_tables.hpp"
  #include "ht_block_decoding.hpp"
  #include "coding_local.hpp"
  #include "utils.hpp"

  #include <arm_neon.h>

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

template <int N>
FORCE_INLINE int32x4_t decode_one_quad(int32x4_t &qinf, int32x4_t U_q, uint8_t pLSB, fwd_buf<0xFF> &MagSgn,
                                       int32x4_t &v_n) {
  const int32x4_t ones = vdupq_n_s32(1);
  uint8x16_t select;
  if (N == 0) {
    select = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
  } else if (N == 1) {
    select = {4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7};
  }
  int32x4_t mu_n;
  // shuffle s32
  int32x4_t w0     = vreinterpretq_s32_u8(vqtbl1q_u8(vreinterpretq_u8_s32(qinf), select));
  int32x4_t mask   = {0x1110, 0x2220, 0x4440, 0x8880};
  int32x4_t flags  = vandq_s32(w0, mask);
  int32x4_t insig  = vceqzq_s32(flags);
  int16x8_t shift  = {3, 3, 2, 2, 1, 1, 0, 0};
  flags            = vreinterpretq_s32_s16(vshlq_s16(vreinterpretq_s16_s32(flags), shift));
  w0               = vshrq_n_s32(flags, 15);  // emb_k
  U_q              = vreinterpretq_s32_u8(vqtbl1q_u8(vreinterpretq_u8_s32(U_q), select));
  int32x4_t m_n    = vsubq_s32(U_q, w0);
  m_n              = vbicq_s32(m_n, insig);
  w0               = vandq_s32(vshrq_n_s32(flags, 11), ones);  // emb_1
  mask             = vsubq_s32(vshlq_s32(ones, m_n), ones);
  int32x4_t ms_vec = MagSgn.fetch(m_n);
  ms_vec           = vandq_s32(ms_vec, mask);
  ms_vec           = vorrq_u32(ms_vec, vshlq_u32(w0, m_n));  // v = 2(mu-1) + sign (0 or 1)
  mu_n             = vaddq_u32(ms_vec, vdupq_n_s32(2));      // 2(mu-1) + sign + 2 = 2mu + sign
  // Add center bin (would be used for lossy and truncated lossless codestreams)
  mu_n = vorrq_s32(mu_n, ones);  // This cancels the effect of a sign bit in LSB
  mu_n = vshlq_u32(mu_n, vdupq_n_s32(pLSB - 1));
  mu_n = vorrq_u32(mu_n, vshlq_n_u32(ms_vec, 31));
  mu_n = vbicq_u32(mu_n, insig);

  w0 = ms_vec;
  if (N == 0) {
    select = {0x4, 0x5, 0x6, 0x7, 0xC, 0xD, 0xE, 0xF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
    w0     = vreinterpretq_s32_u8(vqtbl1q_u8(vreinterpretq_u8_s32(w0), select));
  } else if (N == 1) {
    select = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x4, 0x5, 0x6, 0x7, 0xC, 0xD, 0xE, 0xF};
    w0     = vreinterpretq_s32_u8(vqtbl1q_u8(vreinterpretq_u8_s32(w0), select));
  }
  v_n = vorrq_s32(v_n, w0);
  return mu_n;
}

void ht_cleanup_decode(j2k_codeblock *block, const uint8_t &pLSB, const int32_t Lcup, const int32_t Pcup,
                       const int32_t Scup) {
  const uint16_t QW = static_cast<uint16_t>(ceil_int(static_cast<int16_t>(block->size.x), 2));
  const uint16_t QH = static_cast<uint16_t>(ceil_int(static_cast<int16_t>(block->size.y), 2));

  const uint16_t *dec_table0, *dec_table1;
  dec_table0 = dec_CxtVLC_table0_fast_16;
  dec_table1 = dec_CxtVLC_table1_fast_16;

  alignas(32) auto Eline = MAKE_UNIQUE<int32_t[]>(2U * QW + 6U);
  Eline[0]               = 0;
  auto E_p               = Eline.get() + 1;

  alignas(32) uint16_t scratch[8 * 513] = {0};
  int32_t sstr = static_cast<int32_t>(((block->size.x + 2) + 7u) & ~7u);  // multiples of 8
  uint16_t *sp;

  int32_t qx;
  /*******************************************************************************************************************/
  // VLC, UVLC and MEL decoding
  /*******************************************************************************************************************/
  MEL_dec MEL(block->get_compressed_data(), Lcup, Scup);
  rev_buf VLC_dec(block->get_compressed_data(), Lcup, Scup);
  auto sp0 = block->block_states.get() + 1 + block->blkstate_stride;
  auto sp1 = block->block_states.get() + 1 + 2 * block->blkstate_stride;
  uint32_t u_off0, u_off1;
  uint32_t u0, u1;
  uint32_t context = 0;
  uint32_t vlcval;

  // Initial line-pair
  sp              = scratch;
  int32_t mel_run = MEL.get_run();
  for (qx = QW; qx > 0; qx -= 2, sp += 4) {
    // Decoding of significance and EMB patterns and unsigned residual offsets
    vlcval       = VLC_dec.fetch();
    uint16_t tv0 = dec_table0[(vlcval & 0x7F) + context];
    if (context == 0) {
      mel_run -= 2;
      tv0 = (mel_run == -1) ? tv0 : 0;
      if (mel_run < 0) {
        mel_run = MEL.get_run();
      }
    }
    sp[0] = tv0;

    // calculate context for the next quad, Eq. (1) in the spec
    context = ((tv0 & 0xE0U) << 2) | ((tv0 & 0x10U) << 3);  // = context << 7

    // Decoding of significance and EMB patterns and unsigned residual offsets
    vlcval       = VLC_dec.advance(static_cast<uint8_t>((tv0 & 0x000F) >> 1));
    uint16_t tv1 = dec_table0[(vlcval & 0x7F) + context];
    if (context == 0 && qx > 1) {
      mel_run -= 2;
      tv1 = (mel_run == -1) ? tv1 : 0;
      if (mel_run < 0) {
        mel_run = MEL.get_run();
      }
    }
    tv1   = (qx > 1) ? tv1 : 0;
    sp[2] = tv1;

    // store sigma
    *sp0++ = ((tv0 >> 4) >> 0) & 1;
    *sp0++ = ((tv0 >> 4) >> 2) & 1;
    *sp0++ = ((tv1 >> 4) >> 0) & 1;
    *sp0++ = ((tv1 >> 4) >> 2) & 1;
    *sp1++ = ((tv0 >> 4) >> 1) & 1;
    *sp1++ = ((tv0 >> 4) >> 3) & 1;
    *sp1++ = ((tv1 >> 4) >> 1) & 1;
    *sp1++ = ((tv1 >> 4) >> 3) & 1;

    // calculate context for the next quad, Eq. (1) in the spec
    context = ((tv1 & 0xE0U) << 2) | ((tv1 & 0x10U) << 3);  // = context << 7

    // UVLC decoding
    vlcval = VLC_dec.advance((tv1 & 0x000F) >> 1);
    u_off0 = tv0 & 1;
    u_off1 = tv1 & 1;

    uint32_t mel_offset = 0;
    if (u_off0 == 1 && u_off1 == 1) {
      mel_run -= 2;
      mel_offset = (mel_run == -1) ? 0x40 : 0;
      if (mel_run < 0) {
        mel_run = MEL.get_run();
      }
    }
    uint32_t idx         = (vlcval & 0x3F) + (u_off0 << 6U) + (u_off1 << 7U) + mel_offset;
    uint32_t uvlc_result = uvlc_dec_0[idx];
    // remove total prefix length
    vlcval = VLC_dec.advance(uvlc_result & 0x7);
    uvlc_result >>= 3;
    // extract suffixes for quad 0 and 1
    uint32_t len = uvlc_result & 0xF;            // suffix length for 2 quads (up to 10 = 5 + 5)
    uint32_t tmp = vlcval & ((1U << len) - 1U);  // suffix value for 2 quads
    vlcval       = VLC_dec.advance(len);
    uvlc_result >>= 4;
    // quad 0 length
    len = uvlc_result & 0x7;  // quad 0 suffix length
    uvlc_result >>= 3;
    u0 = 1 + (uvlc_result & 7) + (tmp & ~(0xFFU << len));  // always kappa = 1 in initial line pair
    u1 = 1 + (uvlc_result >> 3) + (tmp >> len);            // always kappa = 1 in initial line pair

    sp[1] = static_cast<uint16_t>(u0);
    sp[3] = static_cast<uint16_t>(u1);
  }

  // Non-initial line-pair
  for (uint16_t row = 1; row < QH; row++) {
    sp0 = block->block_states.get() + (row * 2U + 1U) * block->blkstate_stride + 1U;
    sp1 = block->block_states.get() + (row * 2U + 2U) * block->blkstate_stride + 1U;

    sp = scratch + row * sstr;
    // calculate context for the next quad
    context = ((sp[0 - sstr] & 0xA0U) << 2)
              | ((sp[2 - sstr] & 0x20U) << 4);  // w, sw, nw are always 0 at the head of a row
    for (qx = QW; qx > 0; qx -= 2, sp += 4) {
      // Decoding of significance and EMB patterns and unsigned residual offsets
      vlcval       = VLC_dec.fetch();
      uint16_t tv0 = dec_table1[(vlcval & 0x7F) + context];
      if (context == 0) {
        mel_run -= 2;
        tv0 = (mel_run == -1) ? tv0 : 0;
        if (mel_run < 0) {
          mel_run = MEL.get_run();
        }
      }
      // calculate context for the next quad, Eq. (2) in the spec
      context = ((tv0 & 0x40U) << 2) | ((tv0 & 0x80U) << 1);              // (w | sw) << 8
      context |= (sp[0 - sstr] & 0x80U) | ((sp[2 - sstr] & 0xA0U) << 2);  // ((nw | n) << 7) | (ne << 9)
      context |= (sp[4 - sstr] & 0x20U) << 4;                             // ( nf) << 9

      sp[0] = tv0;

      vlcval = VLC_dec.advance((tv0 & 0x000F) >> 1);

      // Decoding of significance and EMB patterns and unsigned residual offsets
      uint16_t tv1 = dec_table1[(vlcval & 0x7F) + context];
      if (context == 0 && qx > 1) {
        mel_run -= 2;
        tv1 = (mel_run == -1) ? tv1 : 0;
        if (mel_run < 0) {
          mel_run = MEL.get_run();
        }
      }
      tv1 = (qx > 1) ? tv1 : 0;
      // calculate context for the next quad, Eq. (2) in the spec
      context = ((tv1 & 0x40U) << 2) | ((tv1 & 0x80U) << 1);              // (w | sw) << 8
      context |= (sp[2 - sstr] & 0x80U) | ((sp[4 - sstr] & 0xA0U) << 2);  // ((nw | n) << 7) | (ne << 9)
      context |= (sp[6 - sstr] & 0x20U) << 4;                             // ( nf) << 9

      sp[2] = tv1;

      // store sigma
      *sp0++ = ((tv0 >> 4) >> 0) & 1;
      *sp0++ = ((tv0 >> 4) >> 2) & 1;
      *sp0++ = ((tv1 >> 4) >> 0) & 1;
      *sp0++ = ((tv1 >> 4) >> 2) & 1;
      *sp1++ = ((tv0 >> 4) >> 1) & 1;
      *sp1++ = ((tv0 >> 4) >> 3) & 1;
      *sp1++ = ((tv1 >> 4) >> 1) & 1;
      *sp1++ = ((tv1 >> 4) >> 3) & 1;

      vlcval = VLC_dec.advance((tv1 & 0x000F) >> 1);

      // UVLC decoding
      u_off0       = tv0 & 1;
      u_off1       = tv1 & 1;
      uint32_t idx = (vlcval & 0x3F) + (u_off0 << 6U) + (u_off1 << 7U);

      uint32_t uvlc_result = uvlc_dec_1[idx];
      // remove total prefix length
      vlcval = VLC_dec.advance(uvlc_result & 0x7);
      uvlc_result >>= 3;
      // extract suffixes for quad 0 and 1
      uint32_t len = uvlc_result & 0xF;            // suffix length for 2 quads (up to 10 = 5 + 5)
      uint32_t tmp = vlcval & ((1U << len) - 1U);  // suffix value for 2 quads
      VLC_dec.advance(len);

      uvlc_result >>= 4;
      // quad 0 length
      len = uvlc_result & 0x7;  // quad 0 suffix length
      uvlc_result >>= 3;
      u0    = (uvlc_result & 7) + (tmp & ~(0xFFU << len));
      u1    = (uvlc_result >> 3) + (tmp >> len);
      sp[1] = static_cast<uint16_t>(u0);
      sp[3] = static_cast<uint16_t>(u1);
    }
  }

  /*******************************************************************************************************************/
  // MagSgn decoding
  /*******************************************************************************************************************/
  fwd_buf<0xFF> MagSgn(block->get_compressed_data(), Pcup);
  auto mp0 = block->sample_buf.get();
  auto mp1 = block->sample_buf.get() + block->blksampl_stride;
  int32x4_t v_n, qinf, U_q, mu0_n, mu1_n;
  const int32x4_t zeros = vdupq_n_s32(0);
  const int32x4_t ones  = vdupq_n_s32(1);
  sp                    = scratch;
  for (qx = QW; qx > 0; qx -= 2, sp += 4) {
    v_n   = zeros;
    qinf  = vreinterpretq_s32_u16(vld1q_u16(sp));
    U_q   = vshrq_n_u32(vreinterpretq_u32_s32(qinf), 16);
    mu0_n = decode_one_quad<0>(qinf, U_q, pLSB, MagSgn, v_n);
    mu1_n = decode_one_quad<1>(qinf, U_q, pLSB, MagSgn, v_n);

    // store mu
    auto vvv = vzipq_s32(mu0_n, mu1_n);
    vst1q_s32(mp0, vzip1q_s32(vvv.val[0], vvv.val[1]));
    vst1q_s32(mp1, vzip2q_s32(vvv.val[0], vvv.val[1]));
    mp0 += 4;
    mp1 += 4;

    // update Exponent
    v_n = vsubq_s32(vdupq_n_s32(32), vclzq_s32(v_n));
    vst1q_s32(E_p, v_n);
    E_p += 4;
  }

  // Non-initial line-pair
  for (uint16_t row = 1; row < QH; row++) {
    E_p = Eline.get() + 1;
    mp0 = block->sample_buf.get() + (row * 2U) * block->blksampl_stride;
    mp1 = block->sample_buf.get() + (row * 2U + 1U) * block->blksampl_stride;

    sp = scratch + row * sstr;
    int32_t Emax0, Emax1;
    // calculate Emax for the next two quads
    Emax0 = vmaxvq_s32(vld1q_s32(E_p - 1));
    Emax1 = vmaxvq_s32(vld1q_s32(E_p + 1));

    for (qx = QW; qx > 0; qx -= 2, sp += 4) {
      v_n  = zeros;
      qinf = vreinterpretq_s32_u16(vld1q_u16(sp));
      {
        int32x4_t gamma, emax, kappa, u_q, w0;  // local
        gamma = vandq_s32(qinf, vdupq_n_s32(0xF0));
        w0    = vsubq_s32(gamma, ones);
        gamma = vandq_s32(gamma, w0);
        gamma = vceqzq_s32(gamma);

        emax = {Emax0 - 1, Emax1 - 1, 0, 0};
        //        emax  = vsetq_lane_s32(Emax0 - 1, emax, 0);
        //        emax  = vsetq_lane_s32(Emax1 - 1, emax, 1);
        emax  = vbicq_s32(emax, gamma);
        kappa = vmaxq_s32(emax, ones);

        u_q = vshrq_n_s32(qinf, 16);
        U_q = vaddq_s32(u_q, kappa);
      }
      mu0_n = decode_one_quad<0>(qinf, U_q, pLSB, MagSgn, v_n);
      mu1_n = decode_one_quad<1>(qinf, U_q, pLSB, MagSgn, v_n);

      // store mu
      auto vvv = vzipq_s32(mu0_n, mu1_n);
      vst1q_s32(mp0, vzip1q_s32(vvv.val[0], vvv.val[1]));
      vst1q_s32(mp1, vzip2q_s32(vvv.val[0], vvv.val[1]));
      mp0 += 4;
      mp1 += 4;

      // calculate Emax for the next two quads
      Emax0 = vmaxvq_s32(vld1q_s32(E_p + 3));
      Emax1 = vmaxvq_s32(vld1q_s32(E_p + 5));

      // Update Exponent
      v_n = vsubq_s32(vdupq_n_s32(32), vclzq_s32(v_n));
      vst1q_s32(E_p, v_n);
      E_p += 4;
    }
  }  // Non-Initial line-pair end
}  // Cleanup decoding end

auto process_stripes_block_dec = [](SP_dec &SigProp, j2k_codeblock *block, const int32_t i_start,
                                    const int32_t j_start, const uint16_t width, const uint16_t height,
                                    const uint8_t &pLSB) {
  int32_t *sp;
  uint8_t causal_cond = 0;
  uint8_t bit;
  uint8_t mbr;
  const auto block_width  = static_cast<uint16_t>(j_start + width);
  const auto block_height = static_cast<uint16_t>(i_start + height);

  // Decode magnitude
  for (int16_t j = (int16_t)j_start; j < block_width; j++) {
    for (int16_t i = (int16_t)i_start; i < block_height; i++) {
      sp = &block->sample_buf[static_cast<size_t>(j) + static_cast<size_t>(i) * block->blksampl_stride];
      causal_cond = (((block->Cmodes & CAUSAL) == 0) || (i != block_height - 1));
      mbr         = 0;
      if (block->get_state(Sigma, i, j) == 0) {
        mbr = block->calc_mbr(i, j, causal_cond);
      }
      if (mbr != 0) {
        block->modify_state(refinement_indicator, 1, i, j);
        bit = SigProp.importSigPropBit();
        block->modify_state(refinement_value, bit, i, j);
        *sp |= bit << pLSB;
        *sp |= bit << (pLSB - 1);  // new bin center ( = 0.5)
      }
      block->modify_state(scan, 1, i, j);
    }
  }
  // Decode sign
  for (int16_t j = (int16_t)j_start; j < block_width; j++) {
    for (int16_t i = (int16_t)i_start; i < block_height; i++) {
      sp = &block->sample_buf[static_cast<size_t>(j) + static_cast<size_t>(i) * block->blksampl_stride];
      //      if ((*sp & (1 << pLSB)) != 0) {
      if (block->get_state(Refinement_value, i, j)) {
        *sp |= static_cast<int32_t>(SigProp.importSigPropBit()) << 31;
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
  int32_t bit;
  int32_t tmp;
  for (int16_t n1 = 0; n1 < num_v_stripe; n1++) {
    for (int16_t j = 0; j < blk_width; j++) {
      for (int16_t i = i_start; i < i_start + height; i++) {
        sp = &block->sample_buf[static_cast<size_t>(j) + static_cast<size_t>(i) * block->blksampl_stride];
        if (block->get_state(Sigma, i, j) != 0) {
          block->modify_state(refinement_indicator, 1, i, j);
          bit = MagRef.importMagRefBit();
          tmp = static_cast<int32_t>(0xFFFFFFFE | static_cast<unsigned int>(bit));
          tmp <<= pLSB;
          sp[0] &= tmp;
          sp[0] |= 1 << (pLSB - 1);  // new bin center ( = 0.5)
        }
      }
    }
    i_start = static_cast<int16_t>(i_start + 4);
  }
  height = static_cast<int16_t>(blk_height % 4);
  for (int16_t j = 0; j < blk_width; j++) {
    for (int16_t i = i_start; i < i_start + height; i++) {
      sp = &block->sample_buf[static_cast<size_t>(j) + static_cast<size_t>(i) * block->blksampl_stride];
      if (block->get_state(Sigma, i, j) != 0) {
        block->modify_state(refinement_indicator, 1, i, j);
        bit = MagRef.importMagRefBit();
        tmp = static_cast<int32_t>(0xFFFFFFFE | static_cast<unsigned int>(bit));
        tmp <<= pLSB;
        sp[0] &= tmp;
        sp[0] |= 1 << (pLSB - 1);  // new bin center ( = 0.5)
      }
    }
  }
}

void j2k_codeblock::dequantize(uint8_t ROIshift) const {
  // number of decoded magnitude bit‐planes
  const int32_t pLSB = 31 - M_b;  // indicates binary point;

  // bit mask for ROI detection
  const uint32_t mask  = UINT32_MAX >> (M_b + 1);
  const auto vmask     = vdupq_n_s32(static_cast<int32_t>(~mask));
  const auto vROIshift = vdupq_n_s32(ROIshift);

  // vdst0, vdst1 cannot be auto for gcc
  int32x4_t v0, v1, s0, s1, vROImask, vmagmask, vdst0, vdst1;
  vmagmask = vdupq_n_s32(INT32_MAX);
  if (this->transformation) {
    // lossless path
    for (size_t i = 0; i < static_cast<size_t>(this->size.y); i++) {
      int32_t *val = this->sample_buf.get() + i * this->blksampl_stride;
      sprec_t *dst = this->i_samples + i * this->band_stride;
      size_t len   = this->size.x;
      for (; len >= 8; len -= 8) {  // dequantize two vectors at a time
        v0 = vld1q_s32(val);
        v1 = vld1q_s32(val + 4);
        s0 = vshrq_n_s32(v0, 31);  // generate a mask for negative values
        s1 = vshrq_n_s32(v1, 31);  // generate a mask for negative values
        v0 = vandq_s32(v0, vmagmask);
        v1 = vandq_s32(v1, vmagmask);
        // upshift background region, if necessary
        vROImask = vandq_s32(v0, vmask);
        vROImask = vceqzq_s32(vROImask);
        vROImask &= vROIshift;
        v0       = vshlq_s32(v0, vROImask - pLSB);
        vROImask = vandq_s32(v1, vmask);
        vROImask = vceqzq_s32(vROImask);
        vROImask &= vROIshift;
        v1 = vshlq_s32(v1, vROImask - pLSB);
        // convert values from sign-magnitude form to two's complement one
        vdst0 = vbslq_s32(vreinterpretq_u32_s32(s0), vnegq_s32(v0), v0);
        vdst1 = vbslq_s32(vreinterpretq_u32_s32(s1), vnegq_s32(v1), v1);
        vst1q_s16(dst, vcombine_s16(vmovn_s32(vdst0), vmovn_s32(vdst1)));
        val += 8;
        dst += 8;
      }
      for (; len > 0; --len) {
        int32_t sign = *val & INT32_MIN;
        *val &= INT32_MAX;
        // upshift background region, if necessary
        if (ROIshift && (((uint32_t)*val & ~mask) == 0)) {
          *val <<= ROIshift;
        }
        *val >>= pLSB;
        // convert sign-magnitude to two's complement form
        if (sign) {
          *val = -(*val & INT32_MAX);
        }
        assert(pLSB >= 0);  // assure downshift is not negative
        *dst = static_cast<int16_t>(*val);
        val++;
        dst++;
      }
    }
  } else {
    // lossy path
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

    for (size_t i = 0; i < static_cast<size_t>(this->size.y); i++) {
      int32_t *val = this->sample_buf.get() + i * this->blksampl_stride;
      sprec_t *dst = this->i_samples + i * this->band_stride;
      size_t len   = this->size.x;
      for (; len >= 8; len -= 8) {  // dequantize two vectors at a time
        v0 = vld1q_s32(val);
        v1 = vld1q_s32(val + 4);
        s0 = vshrq_n_s32(v0, 31);  // generate a mask for negative values
        s1 = vshrq_n_s32(v1, 31);  // generate a mask for negative values
        v0 = vandq_s32(v0, vmagmask);
        v1 = vandq_s32(v1, vmagmask);
        // upshift background region, if necessary
        vROImask = vandq_s32(v0, vmask);
        vROImask = vceqzq_s32(vROImask);
        vROImask &= vROIshift;
        v0       = vshlq_s32(v0, vROImask);
        vROImask = vandq_s32(v1, vmask);
        vROImask = vceqzq_s32(vROImask);
        vROImask &= vROIshift;
        v1 = vshlq_s32(v1, vROImask);
        // to prevent overflow, truncate to int16_t range
        v0 = vrshrq_n_s32(v0, 16);  // (v0 + (1 << 15)) >> 16;
        v1 = vrshrq_n_s32(v1, 16);  // (v1 + (1 << 15)) >> 16;
        // dequantization
        v0 = vmulq_s32(v0, vdupq_n_s32(scale));
        v1 = vmulq_s32(v1, vdupq_n_s32(scale));
        // downshift and convert values from sign-magnitude form to two's complement one
        v0    = (v0 + (1 << (downshift - 1))) >> downshift;
        v1    = (v1 + (1 << (downshift - 1))) >> downshift;
        vdst0 = vbslq_s32(vreinterpretq_u32_s32(s0), vnegq_s32(v0), v0);
        vdst1 = vbslq_s32(vreinterpretq_u32_s32(s1), vnegq_s32(v1), v1);
        vst1q_s16(dst, vcombine_s16(vmovn_s32(vdst0), vmovn_s32(vdst1)));
        val += 8;
        dst += 8;
      }
      for (; len > 0; --len) {
        int32_t sign = *val & INT32_MIN;
        *val &= INT32_MAX;
        // upshift background region, if necessary
        if (ROIshift && (((uint32_t)*val & ~mask) == 0)) {
          *val <<= ROIshift;
        }
        // to prevent overflow, truncate to int16_t
        *val = (*val + (1 << 15)) >> 16;
        //  dequantization
        *val *= scale;
        // downshift
        *val = (int16_t)((*val + (1 << (downshift - 1))) >> downshift);
        // convert sign-magnitude to two's complement form
        if (sign) {
          *val = -(*val & INT32_MAX);
        }
        *dst = static_cast<int16_t>(*val);
        val++;
        dst++;
      }
    }
  }
}

bool htj2k_decode(j2k_codeblock *block, const uint8_t ROIshift) {
  // number of placeholder pass
  uint8_t P0 = 0;
  // length of HT Cleanup segment
  int32_t Lcup = 0;
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
  // number of (skipped) magnitude bitplanes
  const auto S_blk = static_cast<uint8_t>(P0 + block->num_ZBP + S_skip);
  if (S_blk >= 30) {
    printf("WARNING: Number of skipped mag bitplanes %d is too large.\n", S_blk);
    return false;
  }

  const auto empty_passes = static_cast<uint8_t>(P0 * 3);
  if (block->num_passes < empty_passes) {
    printf("WARNING: number of passes %d exceeds number of empty passes %d", block->num_passes,
           empty_passes);
    return false;
  }
  // number of ht coding pass (Z_blk in the spec)
  const auto num_ht_passes = static_cast<uint8_t>(block->num_passes - empty_passes);
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
    Lcup += static_cast<int32_t>(block->pass_length[all_segments[0]]);
    if (Lcup < 2) {
      printf("WARNING: Cleanup pass length must be at least 2 bytes in length.\n");
      return false;
    }
    Dcup = block->get_compressed_data();
    // Suffix length (=MEL + VLC) of HT Cleanup pass
    const auto Scup = static_cast<int32_t>((Dcup[Lcup - 1] << 4) + (Dcup[Lcup - 2] & 0x0F));
    // modDcup (shall be done before the creation of state_VLC instance)
    Dcup[Lcup - 1] = 0xFF;
    Dcup[Lcup - 2] |= 0x0F;

    if (Scup < 2 || Scup > Lcup || Scup > 4079) {
      printf("WARNING: cleanup pass suffix length %d is invalid.\n", Scup);
      return false;
    }
    // Prefix length (=MagSgn) of HT Cleanup pass
    const auto Pcup = static_cast<int32_t>(Lcup - Scup);

    for (uint32_t i = 1; i < all_segments.size(); i++) {
      Lref += block->pass_length[all_segments[i]];
    }
    if (block->num_passes > 1 && all_segments.size() > 1) {
      Dref = block->get_compressed_data() + Lcup;
    } else {
      Dref = nullptr;
    }

    ht_cleanup_decode(block, static_cast<uint8_t>(30 - S_blk), Lcup, Pcup, Scup);
    if (num_ht_passes > 1) {
      ht_sigprop_decode(block, Dref, Lref, static_cast<uint8_t>(30 - (S_blk + 1)));
    }
    if (num_ht_passes > 2) {
      ht_magref_decode(block, Dref, Lref, static_cast<uint8_t>(30 - (S_blk + 1)));
    }

    // dequantization
    block->dequantize(ROIshift);

  }  // end

  return true;
}
#endif
