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

#if !defined(OPENHTJ2K_ENABLE_ARM_NEON) && (!defined(__AVX2__) || !defined(OPENHTJ2K_TRY_AVX2))
  #include "coding_units.hpp"
  #include "dec_CxtVLC_tables.hpp"
  #include "ht_block_decoding.hpp"
  #include "coding_local.hpp"
  #include "utils.hpp"

  #define Q0 0
  #define Q1 1

uint8_t j2k_codeblock::calc_mbr(const uint32_t i, const uint32_t j, const uint8_t causal_cond) const {
  uint8_t *state_p0 = block_states + static_cast<size_t>(i) * blkstate_stride + j;
  uint8_t *state_p1 = block_states + static_cast<size_t>(i + 1) * blkstate_stride + j;
  uint8_t *state_p2 = block_states + static_cast<size_t>(i + 2) * blkstate_stride + j;

  uint8_t mbr0 = state_p0[0] | state_p0[1] | state_p0[2];
  uint8_t mbr1 = state_p1[0] | state_p1[2];
  uint8_t mbr2 = state_p2[0] | state_p2[1] | state_p2[2];
  uint8_t mbr  = mbr0 | mbr1 | (mbr2 & causal_cond);
  mbr |= (mbr0 >> SHIFT_REF) & (mbr0 >> SHIFT_SCAN);
  mbr |= (mbr1 >> SHIFT_REF) & (mbr1 >> SHIFT_SCAN);
  mbr |= (mbr2 >> SHIFT_REF) & (mbr2 >> SHIFT_SCAN) & causal_cond;
  return mbr & 1;
}

void ht_cleanup_decode(j2k_codeblock *block, const uint8_t &pLSB, const int32_t Lcup, const int32_t Pcup,
                       const int32_t Scup) {
  uint8_t *compressed_data = block->get_compressed_data();
  const uint16_t QW        = static_cast<uint16_t>(ceil_int(static_cast<int16_t>(block->size.x), 2));
  const uint16_t QH        = static_cast<uint16_t>(ceil_int(static_cast<int16_t>(block->size.y), 2));

  uint16_t scratch[8 * 513] = {0};
  int32_t sstr              = static_cast<int32_t>(((block->size.x + 2) + 7u) & ~7u);  // multiples of 8
  uint16_t *sp;
  int32_t qx;
  /*******************************************************************************************************************/
  // VLC, UVLC and MEL decoding
  /*******************************************************************************************************************/
  MEL_dec MEL(compressed_data, Lcup, Scup);
  rev_buf VLC_dec(compressed_data, Lcup, Scup);
  auto sp0 = block->block_states + 1 + block->blkstate_stride;
  auto sp1 = block->block_states + 1 + 2 * block->blkstate_stride;
  uint32_t u_off0, u_off1;
  uint32_t u0, u1;
  uint32_t context = 0;
  uint32_t vlcval;

  const uint16_t *dec_table;
  // Initial line-pair
  dec_table       = dec_CxtVLC_table0_fast_16;
  sp              = scratch;
  int32_t mel_run = MEL.get_run();
  for (qx = QW; qx > 0; qx -= 2, sp += 4) {
    // Decoding of significance and EMB patterns and unsigned residual offsets
    vlcval       = VLC_dec.fetch();
    uint16_t tv0 = dec_table[(vlcval & 0x7F) + context];
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
    vlcval       = VLC_dec.advance((tv0 & 0x000F) >> 1);
    uint16_t tv1 = dec_table[(vlcval & 0x7F) + context];
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

    // UVLC decoding
    uint32_t idx         = (vlcval & 0x3F) + (u_off0 << 6U) + (u_off1 << 7U) + mel_offset;
    uint32_t uvlc_result = uvlc_dec_0[idx];
    // remove total prefix length
    vlcval = VLC_dec.advance(uvlc_result & 0x7);
    uvlc_result >>= 3;
    // extract suffixes for quad 0 and 1
    uint32_t len = uvlc_result & 0xF;  // suffix length for 2 quads (up to 10 = 5 + 5)
    //  ((1U << len) - 1U) can be replaced with _bzhi_u32(UINT32_MAX, len); not fast
    uint32_t tmp = vlcval & ((1U << len) - 1U);  // suffix value for 2 quads
    vlcval       = VLC_dec.advance(len);
    uvlc_result >>= 4;
    // quad 0 length
    len = uvlc_result & 0x7;  // quad 0 suffix length
    uvlc_result >>= 3;
    // U = 1+ u
    u0 = 1 + (uvlc_result & 7) + (tmp & ~(0xFFU << len));  // always kappa = 1 in initial line pair
    u1 = 1 + (uvlc_result >> 3) + (tmp >> len);            // always kappa = 1 in initial line pair

    sp[1] = static_cast<uint16_t>(u0);
    sp[3] = static_cast<uint16_t>(u1);
  }
  // sp[0] = sp[1] = 0;

  // Non-initial line-pair
  dec_table = dec_CxtVLC_table1_fast_16;
  for (uint16_t row = 1; row < QH; row++) {
    sp0 = block->block_states + (row * 2U + 1U) * block->blkstate_stride + 1U;
    sp1 = sp0 + block->blkstate_stride;

    sp = scratch + row * sstr;
    // calculate context for the next quad: w, sw, nw are always 0 at the head of a row
    context = ((sp[0 - sstr] & 0xA0U) << 2) | ((sp[2 - sstr] & 0x20U) << 4);
    for (qx = QW; qx > 0; qx -= 2, sp += 4) {
      // Decoding of significance and EMB patterns and unsigned residual offsets
      vlcval       = VLC_dec.fetch();
      uint16_t tv0 = dec_table[(vlcval & 0x7F) + context];
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
      uint16_t tv1 = dec_table[(vlcval & 0x7F) + context];
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
      uint32_t len = uvlc_result & 0xF;  // suffix length for 2 quads (up to 10 = 5 + 5)
      //  ((1U << len) - 1U) can be replaced with _bzhi_u32(UINT32_MAX, len); not fast
      uint32_t tmp = vlcval & ((1U << len) - 1U);  // suffix value for 2 quads
      vlcval       = VLC_dec.advance(len);
      uvlc_result >>= 4;
      // quad 0 length
      len = uvlc_result & 0x7;  // quad 0 suffix length
      uvlc_result >>= 3;
      u0 = (uvlc_result & 7) + (tmp & ~(0xFFU << len));
      u1 = (uvlc_result >> 3) + (tmp >> len);

      sp[1] = static_cast<uint16_t>(u0);
      sp[3] = static_cast<uint16_t>(u1);
    }
    // sp[0] = sp[1] = 0;
  }

  /*******************************************************************************************************************/
  // MagSgn decoding
  /*******************************************************************************************************************/
  {
    // We allocate a scratch row for storing v_n values.
    // We have 512 quads horizontally.
    // We need an extra entry to handle the case of vp[1]
    // when vp is at the last column.
    // Here, we allocate 4 instead of 1 to make the buffer size
    // a multipled of 16 bytes.
    const int v_n_size             = 512 + 4;
    uint32_t v_n_scratch[v_n_size] = {0};  // 2+ kB

    fwd_buf<0xFF> MagSgn(compressed_data, Pcup);

    uint16_t *sp = scratch;
    uint32_t *vp = v_n_scratch;
    int32_t *dp  = block->sample_buf;

    uint32_t prev_v_n = 0;
    for (uint32_t x = 0; x < block->size.x; sp += 2, ++vp) {
      uint32_t inf = sp[0];
      uint32_t U_q = sp[1];
      if (U_q > ((30 - pLSB) + 2)) {
        printf("ERROR\n");
      }

      uint32_t v_n;
      uint32_t val = 0;
      uint32_t bit = 0;
      if (inf & (1 << (4 + bit))) {
        // get 32 bits of magsgn data
        uint32_t ms_val = MagSgn.fetch();
        uint32_t m_n    = U_q - ((inf >> (12 + bit)) & 1);  // remove e_k
        MagSgn.advance(m_n);                                // consume m_n

        val = ms_val << 31;                      // get sign bit
        v_n = ms_val & ((1 << m_n) - 1);         // keep only m_n bits
        v_n |= ((inf >> (8 + bit)) & 1) << m_n;  // add EMB e_1 as MSB
        v_n |= 1;                                // add center of bin
        // v_n now has 2 * (\mu - 1) + 0.5 with correct sign bit
        // add 2 to make it 2*\mu+0.5, shift it up to missing MSBs
        val |= (v_n + 2) << (pLSB - 1);
      }
      dp[0] = val;

      v_n = 0;
      val = 0;
      bit = 1;
      if (inf & (1 << (4 + bit))) {
        // get 32 bits of magsgn data
        uint32_t ms_val = MagSgn.fetch();
        uint32_t m_n    = U_q - ((inf >> (12 + bit)) & 1);  // remove e_k
        MagSgn.advance(m_n);                                // consume m_n

        val = ms_val << 31;                      // get sign bit
        v_n = ms_val & ((1 << m_n) - 1);         // keep only m_n bits
        v_n |= ((inf >> (8 + bit)) & 1) << m_n;  // add EMB e_1 as MSB
        v_n |= 1;                                // add center of bin
        // v_n now has 2 * (\mu - 1) + 0.5 with correct sign bit
        // add 2 to make it 2*\mu+0.5, shift it up to missing MSBs
        val |= (v_n + 2) << (pLSB - 1);
      }
      dp[block->blksampl_stride] = val;
      vp[0]                      = prev_v_n | v_n;
      prev_v_n                   = 0;
      ++dp;
      if (++x >= block->size.x) {
        ++vp;
        break;
      }

      val = 0;
      bit = 2;
      if (inf & (1 << (4 + bit))) {
        // get 32 bits of magsgn data
        uint32_t ms_val = MagSgn.fetch();
        uint32_t m_n    = U_q - ((inf >> (12 + bit)) & 1);  // remove e_k
        MagSgn.advance(m_n);                                // consume m_n

        val = ms_val << 31;                      // get sign bit
        v_n = ms_val & ((1 << m_n) - 1);         // keep only m_n bits
        v_n |= ((inf >> (8 + bit)) & 1) << m_n;  // add EMB e_1 as MSB
        v_n |= 1;                                // add center of bin
        // v_n now has 2 * (\mu - 1) + 0.5 with correct sign bit
        // add 2 to make it 2*\mu+0.5, shift it up to missing MSBs
        val |= (v_n + 2) << (pLSB - 1);
      }
      dp[0] = val;

      v_n = 0;
      val = 0;
      bit = 3;
      if (inf & (1 << (4 + bit))) {
        // get 32 bits of magsgn data
        uint32_t ms_val = MagSgn.fetch();
        uint32_t m_n    = U_q - ((inf >> (12 + bit)) & 1);  // remove e_k
        MagSgn.advance(m_n);                                // consume m_n

        val = ms_val << 31;                      // get sign bit
        v_n = ms_val & ((1 << m_n) - 1);         // keep only m_n bits
        v_n |= ((inf >> (8 + bit)) & 1) << m_n;  // add EMB e_1 as MSB
        v_n |= 1;                                // add center of bin
        // v_n now has 2 * (\mu - 1) + 0.5 with correct sign bit
        // add 2 to make it 2*\mu+0.5, shift it up to missing MSBs
        val |= (v_n + 2) << (pLSB - 1);
      }
      dp[block->blksampl_stride] = val;
      prev_v_n                   = v_n;
      ++dp;
      ++x;
    }
    vp[0] = prev_v_n;

    for (uint32_t y = 2; y < block->size.y; y += 2) {
      uint16_t *sp = scratch + (y >> 1) * sstr;
      uint32_t *vp = v_n_scratch;
      int32_t *dp  = block->sample_buf + y * block->blksampl_stride;

      prev_v_n = 0;
      for (uint32_t x = 0; x < block->size.x; sp += 2, ++vp) {
        uint32_t inf = sp[0];
        uint32_t u_q = sp[1];

        uint32_t gamma = inf & 0xF0;
        gamma &= gamma - 0x10;  // is gamma_q 1?
        uint32_t emax  = vp[0] | vp[1];
        emax           = 31 - count_leading_zeros(emax | 2);  // emax - 1
        uint32_t kappa = gamma ? emax : 1;

        uint32_t U_q = u_q + kappa;
        if (U_q > ((30 - pLSB) + 2)) {
          printf("ERROR\n");
        }

        uint32_t v_n;
        uint32_t val = 0;
        uint32_t bit = 0;
        if (inf & (1 << (4 + bit))) {
          // get 32 bits of magsgn data
          uint32_t ms_val = MagSgn.fetch();
          uint32_t m_n    = U_q - ((inf >> (12 + bit)) & 1);  // remove e_k
          MagSgn.advance(m_n);                                // consume m_n

          val = ms_val << 31;                      // get sign bit
          v_n = ms_val & ((1 << m_n) - 1);         // keep only m_n bits
          v_n |= ((inf >> (8 + bit)) & 1) << m_n;  // add EMB e_1 as MSB
          v_n |= 1;                                // add center of bin
          // v_n now has 2 * (\mu - 1) + 0.5 with correct sign bit
          // add 2 to make it 2*\mu+0.5, shift it up to missing MSBs
          val |= (v_n + 2) << (pLSB - 1);
        }
        dp[0] = val;

        v_n = 0;
        val = 0;
        bit = 1;
        if (inf & (1 << (4 + bit))) {
          // get 32 bits of magsgn data
          uint32_t ms_val = MagSgn.fetch();
          uint32_t m_n    = U_q - ((inf >> (12 + bit)) & 1);  // remove e_k
          MagSgn.advance(m_n);                                // consume m_n

          val = ms_val << 31;                      // get sign bit
          v_n = ms_val & ((1 << m_n) - 1);         // keep only m_n bits
          v_n |= ((inf >> (8 + bit)) & 1) << m_n;  // add EMB e_1 as MSB
          v_n |= 1;                                // add center of bin
          // v_n now has 2 * (\mu - 1) + 0.5 with correct sign bit
          // add 2 to make it 2*\mu+0.5, shift it up to missing MSBs
          val |= (v_n + 2) << (pLSB - 1);
        }
        dp[block->blksampl_stride] = val;
        vp[0]                      = prev_v_n | v_n;
        prev_v_n                   = 0;
        ++dp;
        if (++x >= block->size.x) {
          ++vp;
          break;
        }

        val = 0;
        bit = 2;
        if (inf & (1 << (4 + bit))) {
          // get 32 bits of magsgn data
          uint32_t ms_val = MagSgn.fetch();
          uint32_t m_n    = U_q - ((inf >> (12 + bit)) & 1);  // remove e_k
          MagSgn.advance(m_n);                                // consume m_n

          val = ms_val << 31;                      // get sign bit
          v_n = ms_val & ((1 << m_n) - 1);         // keep only m_n bits
          v_n |= ((inf >> (8 + bit)) & 1) << m_n;  // add EMB e_1 as MSB
          v_n |= 1;                                // add center of bin
          // v_n now has 2 * (\mu - 1) + 0.5 with correct sign bit
          // add 2 to make it 2*\mu+0.5, shift it up to missing MSBs
          val |= (v_n + 2) << (pLSB - 1);
        }
        dp[0] = val;

        v_n = 0;
        val = 0;
        bit = 3;
        if (inf & (1 << (4 + bit))) {
          // get 32 bits of magsgn data
          uint32_t ms_val = MagSgn.fetch();
          uint32_t m_n    = U_q - ((inf >> (12 + bit)) & 1);  // remove e_k
          MagSgn.advance(m_n);                                // consume m_n

          val = ms_val << 31;                      // get sign bit
          v_n = ms_val & ((1 << m_n) - 1);         // keep only m_n bits
          v_n |= ((inf >> (8 + bit)) & 1) << m_n;  // add EMB e_1 as MSB
          v_n |= 1;                                // add center of bin
          // v_n now has 2 * (\mu - 1) + 0.5 with correct sign bit
          // add 2 to make it 2*\mu+0.5, shift it up to missing MSBs
          val |= (v_n + 2) << (pLSB - 1);
        }
        dp[block->blksampl_stride] = val;
        prev_v_n                   = v_n;
        ++dp;
        ++x;
      }
      vp[0] = prev_v_n;
    }
  }
}

void ht_cleanup_decode2(j2k_codeblock *block, const uint8_t &pLSB, const int32_t Lcup, const int32_t Pcup,
                        const int32_t Scup) {
  fwd_buf<0xFF> MagSgn(block->get_compressed_data(), Pcup);
  MEL_dec MEL(block->get_compressed_data(), Lcup, Scup);
  rev_buf VLC_dec(block->get_compressed_data(), Lcup, Scup);
  const uint16_t QW = static_cast<uint16_t>(ceil_int(static_cast<int16_t>(block->size.x), 2));
  const uint16_t QH = static_cast<uint16_t>(ceil_int(static_cast<int16_t>(block->size.y), 2));

  alignas(32) uint32_t m_quads[8];
  alignas(32) uint32_t msval[8];
  alignas(32) uint32_t sigma_quads[8];
  alignas(32) uint32_t mu_quads[8];
  alignas(32) uint32_t v_quads[8];
  alignas(32) uint32_t known_1[2];

  auto mp0 = block->sample_buf;
  auto mp1 = block->sample_buf + block->blksampl_stride;
  auto sp0 = block->block_states + 1 + block->blkstate_stride;
  auto sp1 = block->block_states + 1 + 2 * block->blkstate_stride;

  uint32_t rho0, rho1;
  uint32_t u_off0, u_off1;
  uint32_t emb_k_0, emb_k_1;
  uint32_t emb_1_0, emb_1_1;
  uint32_t u0, u1;
  uint32_t U0, U1;
  uint8_t gamma0, gamma1;
  uint32_t kappa0 = 1, kappa1 = 1;  // kappa is always 1 for initial line-pair

  const uint16_t *dec_table0, *dec_table1;
  dec_table0 = dec_CxtVLC_table0_fast_16;
  dec_table1 = dec_CxtVLC_table1_fast_16;

  alignas(32) auto rholine = MAKE_UNIQUE<uint32_t[]>(QW + 4U);
  rholine[0]               = 0;
  auto rho_p               = rholine.get() + 1;
  alignas(32) auto Eline   = MAKE_UNIQUE<int32_t[]>(2U * QW + 8U);
  Eline[0]                 = 0;
  auto E_p                 = Eline.get() + 1;

  uint32_t context = 0;
  uint32_t vlcval;
  int32_t mel_run = MEL.get_run();

  int32_t qx;
  // Initial line-pair
  for (qx = QW; qx > 0; qx -= 2) {
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

    rho0    = (tv0 & 0x00F0) >> 4;
    emb_k_0 = (tv0 & 0xF000) >> 12;
    emb_1_0 = (tv0 & 0x0F00) >> 8;

    *rho_p++ = rho0;
    // calculate context for the next quad
    context = ((tv0 & 0xE0U) << 2) | ((tv0 & 0x10U) << 3);

    // Decoding of significance and EMB patterns and unsigned residual offsets
    vlcval       = VLC_dec.advance((tv0 & 0x000F) >> 1);
    uint16_t tv1 = dec_table0[(vlcval & 0x7F) + context];
    if (context == 0 && qx > 1) {
      mel_run -= 2;
      tv1 = (mel_run == -1) ? tv1 : 0;
      if (mel_run < 0) {
        mel_run = MEL.get_run();
      }
    }
    tv1     = (qx > 1) ? tv1 : 0;
    rho1    = (tv1 & 0x00F0) >> 4;
    emb_k_1 = (tv1 & 0xF000) >> 12;
    emb_1_1 = (tv1 & 0x0F00) >> 8;

    *rho_p++ = rho1;

    for (uint32_t i = 0; i < 4; i++) {
      sigma_quads[i] = (rho0 >> i) & 1;
    }
    for (uint32_t i = 0; i < 4; i++) {
      sigma_quads[i + 4] = (rho1 >> i) & 1;
    }

    // calculate context for the next quad
    context = ((tv1 & 0xE0U) << 2) | ((tv1 & 0x10U) << 3);

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
    u0 = (uvlc_result & 7) + (tmp & ~(0xFFU << len));
    u1 = (uvlc_result >> 3) + (tmp >> len);

    U0 = kappa0 + u0;
    U1 = kappa1 + u1;

    for (uint32_t i = 0; i < 4; i++) {
      m_quads[i]     = sigma_quads[i] * U0 - ((emb_k_0 >> i) & 1);
      m_quads[i + 4] = sigma_quads[i + 4] * U1 - ((emb_k_1 >> i) & 1);
    }

    // recoverMagSgnValue
    for (uint32_t i = 0; i < 8; i++) {
      msval[i] = MagSgn.fetch();
      MagSgn.advance(m_quads[i]);
    }

    for (uint32_t i = 0; i < 4; i++) {
      known_1[Q0] = (emb_1_0 >> i) & 1;
      v_quads[i]  = msval[i] & ((1 << m_quads[i]) - 1U);
      v_quads[i] |= known_1[Q0] << m_quads[i];
      if (m_quads[i] != 0) {
        mu_quads[i] = v_quads[i] + 2;
        mu_quads[i] |= 1;
        mu_quads[i] <<= pLSB - 1;
        mu_quads[i] |= (v_quads[i] & 1) << 31;  // sign bit
      } else {
        mu_quads[i] = 0;
      }
    }
    for (uint32_t i = 0; i < 4; i++) {
      known_1[Q1]    = (emb_1_1 >> i) & 1;
      v_quads[i + 4] = msval[i + 4] & ((1 << m_quads[i + 4]) - 1U);
      v_quads[i + 4] |= known_1[Q1] << m_quads[i + 4];
      if (m_quads[i + 4] != 0) {
        mu_quads[i + 4] = v_quads[i + 4] + 2;
        mu_quads[i + 4] |= 1;
        mu_quads[i + 4] <<= pLSB - 1;
        mu_quads[i + 4] |= (v_quads[i + 4] & 1) << 31;  // sign bit
      } else {
        mu_quads[i + 4] = 0;
      }
    }
    *mp0++ = static_cast<int>(mu_quads[0]);
    *mp0++ = static_cast<int>(mu_quads[2]);
    *mp0++ = static_cast<int>(mu_quads[0 + 4]);
    *mp0++ = static_cast<int>(mu_quads[2 + 4]);
    *mp1++ = static_cast<int>(mu_quads[1]);
    *mp1++ = static_cast<int>(mu_quads[3]);
    *mp1++ = static_cast<int>(mu_quads[1 + 4]);
    *mp1++ = static_cast<int>(mu_quads[3 + 4]);

    *sp0++ = (rho0 >> 0) & 1;
    *sp0++ = (rho0 >> 2) & 1;
    *sp0++ = (rho1 >> 0) & 1;
    *sp0++ = (rho1 >> 2) & 1;
    *sp1++ = (rho0 >> 1) & 1;
    *sp1++ = (rho0 >> 3) & 1;
    *sp1++ = (rho1 >> 1) & 1;
    *sp1++ = (rho1 >> 3) & 1;

    *E_p++ = static_cast<int32_t>(32 - count_leading_zeros(v_quads[1]));
    *E_p++ = static_cast<int32_t>(32 - count_leading_zeros(v_quads[3]));
    *E_p++ = static_cast<int32_t>(32 - count_leading_zeros(v_quads[5]));
    *E_p++ = static_cast<int32_t>(32 - count_leading_zeros(v_quads[7]));
  }  // Initial line-pair end

  /*******************************************************************************************************************/
  // Non-initial line-pair
  /*******************************************************************************************************************/

  for (uint16_t row = 1; row < QH; row++) {
    rho_p = rholine.get() + 1;
    E_p   = Eline.get() + 1;
    mp0   = block->sample_buf + (row * 2U) * block->blksampl_stride;
    mp1   = block->sample_buf + (row * 2U + 1U) * block->blksampl_stride;
    sp0   = block->block_states + (row * 2U + 1U) * block->blkstate_stride + 1U;
    sp1   = block->block_states + (row * 2U + 2U) * block->blkstate_stride + 1U;
    rho1  = 0;

    int32_t Emax0, Emax1;
    // Calculate Emax for the next two quads
    Emax0 = find_max(E_p[-1], E_p[0], E_p[1], E_p[2]);
    Emax1 = find_max(E_p[1], E_p[2], E_p[3], E_p[4]);

    // calculate context for the next quad
    context = ((rho1 & 0x4) << 6) | ((rho1 & 0x8) << 5);            // (w | sw) << 8
    context |= ((rho_p[-1] & 0x8) << 4) | ((rho_p[0] & 0x2) << 6);  // (nw | n) << 7
    context |= ((rho_p[0] & 0x8) << 6) | ((rho_p[1] & 0x2) << 8);   // (ne | nf) << 9

    for (qx = QW; qx > 0; qx -= 2) {
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

      rho0    = (tv0 & 0x00F0) >> 4;
      emb_k_0 = (tv0 & 0xF000) >> 12;
      emb_1_0 = (tv0 & 0x0F00) >> 8;

      vlcval = VLC_dec.advance((tv0 & 0x000F) >> 1);

      // calculate context for the next quad
      context = ((rho0 & 0x4) << 6) | ((rho0 & 0x8) << 5);           // (w | sw) << 8
      context |= ((rho_p[0] & 0x8) << 4) | ((rho_p[1] & 0x2) << 6);  // (nw | n) << 7
      context |= ((rho_p[1] & 0x8) << 6) | ((rho_p[2] & 0x2) << 8);  // (ne | nf) << 9

      // Decoding of significance and EMB patterns and unsigned residual offsets
      uint16_t tv1 = dec_table1[(vlcval & 0x7F) + context];
      if (context == 0 && qx > 1) {
        mel_run -= 2;
        tv1 = (mel_run == -1) ? tv1 : 0;
        if (mel_run < 0) {
          mel_run = MEL.get_run();
        }
      }
      tv1     = (qx > 1) ? tv1 : 0;
      rho1    = (tv1 & 0x00F0) >> 4;
      emb_k_1 = (tv1 & 0xF000) >> 12;
      emb_1_1 = (tv1 & 0x0F00) >> 8;

      // calculate context for the next quad
      context = ((rho1 & 0x4) << 6) | ((rho1 & 0x8) << 5);           // (w | sw) << 8
      context |= ((rho_p[1] & 0x8) << 4) | ((rho_p[2] & 0x2) << 6);  // (nw | n) << 7
      context |= ((rho_p[2] & 0x8) << 6) | ((rho_p[3] & 0x2) << 8);  // (ne | nf) << 9

      vlcval = VLC_dec.advance((tv1 & 0x000F) >> 1);

      for (uint32_t i = 0; i < 4; i++) {
        sigma_quads[i] = (rho0 >> i) & 1;
      }
      for (uint32_t i = 0; i < 4; i++) {
        sigma_quads[i + 4] = (rho1 >> i) & 1;
      }

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
      vlcval       = VLC_dec.advance(len);
      uvlc_result >>= 4;
      // quad 0 length
      len = uvlc_result & 0x7;  // quad 0 suffix length
      uvlc_result >>= 3;
      u0 = (uvlc_result & 7) + (tmp & ~(0xFFU << len));
      u1 = (uvlc_result >> 3) + (tmp >> len);

      gamma0 = ((rho0 & (rho0 - 1)) == 0) ? 0 : 1;  // (popcount32(rho0) < 2) ? 0 : 1;
      gamma1 = ((rho1 & (rho1 - 1)) == 0) ? 0 : 1;  // (popcount32(rho1) < 2) ? 0 : 1;
      kappa0 = (1 > gamma0 * (Emax0 - 1)) ? 1U : static_cast<uint8_t>(gamma0 * (Emax0 - 1));
      kappa1 = (1 > gamma1 * (Emax1 - 1)) ? 1U : static_cast<uint8_t>(gamma1 * (Emax1 - 1));
      U0     = kappa0 + u0;
      U1     = kappa1 + u1;

      for (uint32_t i = 0; i < 4; i++) {
        m_quads[i]     = sigma_quads[i] * U0 - ((emb_k_0 >> i) & 1);
        m_quads[i + 4] = sigma_quads[i + 4] * U1 - ((emb_k_1 >> i) & 1);
      }

      // recoverMagSgnValue
      for (uint32_t i = 0; i < 8; i++) {
        msval[i] = MagSgn.fetch();
        MagSgn.advance(m_quads[i]);
      }

      for (uint32_t i = 0; i < 4; i++) {
        known_1[Q0] = (emb_1_0 >> i) & 1;
        v_quads[i]  = msval[i] & ((1 << m_quads[i]) - 1U);
        v_quads[i] |= known_1[Q0] << m_quads[i];
        if (m_quads[i] != 0) {
          mu_quads[i] = v_quads[i] + 2;
          mu_quads[i] |= 1;
          mu_quads[i] <<= pLSB - 1;
          mu_quads[i] |= (v_quads[i] & 1) << 31;  // sign bit
        } else {
          mu_quads[i] = 0;
        }
      }
      for (uint32_t i = 0; i < 4; i++) {
        known_1[Q1]    = (emb_1_1 >> i) & 1;
        v_quads[i + 4] = msval[i + 4] & ((1 << m_quads[i + 4]) - 1U);
        v_quads[i + 4] |= known_1[Q1] << m_quads[i + 4];
        if (m_quads[i + 4] != 0) {
          mu_quads[i + 4] = v_quads[i + 4] + 2;
          mu_quads[i + 4] |= 1;
          mu_quads[i + 4] <<= pLSB - 1;
          mu_quads[i + 4] |= (v_quads[i + 4] & 1) << 31;  // sign bit
        } else {
          mu_quads[i + 4] = 0;
        }
      }
      *mp0++ = static_cast<int>(mu_quads[0]);
      *mp0++ = static_cast<int>(mu_quads[2]);
      *mp0++ = static_cast<int>(mu_quads[0 + 4]);
      *mp0++ = static_cast<int>(mu_quads[2 + 4]);
      *mp1++ = static_cast<int>(mu_quads[1]);
      *mp1++ = static_cast<int>(mu_quads[3]);
      *mp1++ = static_cast<int>(mu_quads[1 + 4]);
      *mp1++ = static_cast<int>(mu_quads[3 + 4]);

      *sp0++ = (rho0 >> 0) & 1;
      *sp0++ = (rho0 >> 2) & 1;
      *sp0++ = (rho1 >> 0) & 1;
      *sp0++ = (rho1 >> 2) & 1;
      *sp1++ = (rho0 >> 1) & 1;
      *sp1++ = (rho0 >> 3) & 1;
      *sp1++ = (rho1 >> 1) & 1;
      *sp1++ = (rho1 >> 3) & 1;

      *rho_p++ = rho0;
      *rho_p++ = rho1;

      Emax0  = find_max(E_p[3], E_p[4], E_p[5], E_p[6]);
      Emax1  = find_max(E_p[5], E_p[6], E_p[7], E_p[8]);
      E_p[0] = static_cast<int32_t>(32 - count_leading_zeros(v_quads[1]));
      E_p[1] = static_cast<int32_t>(32 - count_leading_zeros(v_quads[3]));
      E_p[2] = static_cast<int32_t>(32 - count_leading_zeros(v_quads[5]));
      E_p[3] = static_cast<int32_t>(32 - count_leading_zeros(v_quads[7]));
      E_p += 4;
    }
  }  // Non-Initial line-pair end
}

auto process_stripes_block_dec = [](SP_dec &SigProp, j2k_codeblock *block, const uint32_t i_start,
                                    const uint32_t j_start, const uint32_t width, const uint32_t height,
                                    const uint8_t &pLSB) {
  int32_t *sp;
  uint8_t causal_cond = 0;
  uint8_t bit;
  uint8_t mbr;
  const auto block_width  = j_start + width;
  const auto block_height = i_start + height;

  // Decode magnitude
  for (uint32_t j = j_start; j < block_width; j++) {
    for (uint32_t i = i_start; i < block_height; i++) {
      sp               = &block->sample_buf[j + i * block->blksampl_stride];
      causal_cond      = (((block->Cmodes & CAUSAL) == 0) || (i != block_height - 1));
      mbr              = 0;
      uint8_t *state_p = block->block_states + (i + 1) * block->blkstate_stride + (j + 1);
      if ((state_p[0] >> SHIFT_SIGMA & 1) == 0) {
        mbr = block->calc_mbr(i, j, causal_cond);
      }
      if (mbr != 0) {
        //        block->modify_state(refinement_indicator, 1, i, j);
        state_p[0] |= 1 << SHIFT_PI_;
        bit = SigProp.importSigPropBit();
        //        block->modify_state(refinement_value, bit, i, j);
        state_p[0] |= bit << SHIFT_REF;
        *sp |= bit << pLSB;
        *sp |= bit << (pLSB - 1);  // new bin center ( = 0.5)
      }
      //      block->modify_state(scan, 1, i, j);
      state_p[0] |= 1 << SHIFT_SCAN;
    }
  }
  // Decode sign
  for (uint32_t j = j_start; j < block_width; j++) {
    for (uint32_t i = i_start; i < block_height; i++) {
      sp               = &block->sample_buf[j + i * block->blksampl_stride];
      uint8_t *state_p = block->block_states + (i + 1) * block->blkstate_stride + (j + 1);
      //      if ((*sp & (1 << pLSB)) != 0) {
      if ((state_p[0] >> SHIFT_REF) & 1) {
        *sp |= static_cast<int32_t>(SigProp.importSigPropBit()) << 31;
      }
    }
  }
};

void ht_sigprop_decode(j2k_codeblock *block, uint8_t *HT_magref_segment, uint32_t magref_length,
                       const uint8_t &pLSB) {
  SP_dec SigProp(HT_magref_segment, magref_length);
  const uint32_t num_v_stripe = block->size.y / 4;
  const uint32_t num_h_stripe = block->size.x / 4;
  uint32_t i_start            = 0, j_start;
  uint32_t width              = 4;
  uint32_t width_last;
  uint32_t height = 4;

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
  const uint32_t blk_height   = block->size.y;
  const uint32_t blk_width    = block->size.x;
  const uint32_t num_v_stripe = block->size.y / 4;
  uint32_t i_start            = 0;
  uint32_t height             = 4;
  int32_t *sp;
  int32_t bit;
  int32_t tmp;
  for (uint32_t n1 = 0; n1 < num_v_stripe; n1++) {
    for (uint32_t j = 0; j < blk_width; j++) {
      for (uint32_t i = i_start; i < i_start + height; i++) {
        sp               = &block->sample_buf[j + i * block->blksampl_stride];
        uint8_t *state_p = block->block_states + (i + 1) * block->blkstate_stride + (j + 1);
        //        if (block->get_state(Sigma, i, j) != 0) {
        if ((state_p[0] >> SHIFT_SIGMA & 1) != 0) {
          //          block->modify_state(refinement_indicator, 1, i, j);
          state_p[0] |= 1 << SHIFT_PI_;
          bit = MagRef.importMagRefBit();
          tmp = static_cast<int32_t>(0xFFFFFFFE | static_cast<unsigned int>(bit));
          tmp <<= pLSB;
          sp[0] &= tmp;
          sp[0] |= 1 << (pLSB - 1);  // new bin center ( = 0.5)
        }
      }
    }
    i_start += 4;
  }
  height = blk_height % 4;
  for (uint32_t j = 0; j < blk_width; j++) {
    for (uint32_t i = i_start; i < i_start + height; i++) {
      sp               = &block->sample_buf[j + i * block->blksampl_stride];
      uint8_t *state_p = block->block_states + (i + 1) * block->blkstate_stride + (j + 1);
      //        if (block->get_state(Sigma, i, j) != 0) {
      if ((state_p[0] >> SHIFT_SIGMA & 1) != 0) {
        //          block->modify_state(refinement_indicator, 1, i, j);
        state_p[0] |= 1 << SHIFT_PI_;
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
      int32_t *val = this->sample_buf + i * this->blksampl_stride;
      sprec_t *dst = this->i_samples + i * this->band_stride;
      size_t len   = this->size.x;
      for (; len > 0; --len) {
        int32_t sign = *val & INT32_MIN;
        *val &= INT32_MAX;
        // detect background region and upshift it
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
    [[maybe_unused]] int32_t ROImask = 0;
    if (ROIshift) {
      ROImask = static_cast<int32_t>(0xFFFFFFFF);
    }
    //    auto vROIshift = vdupq_n_s32(ROImask);
    for (size_t i = 0; i < static_cast<size_t>(this->size.y); i++) {
      int32_t *val = this->sample_buf + i * this->blksampl_stride;
      sprec_t *dst = this->i_samples + i * this->band_stride;
      size_t len   = this->size.x;

      for (; len > 0; --len) {
        int32_t sign = *val & INT32_MIN;
        *val &= INT32_MAX;
        // detect background region and upshift it
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
    Lcup += static_cast<int32_t>(block->pass_length[all_segments[0]]);
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
    const int32_t Scup = static_cast<int32_t>((Dcup[Lcup - 1] << 4) + (Dcup[Lcup - 2] & 0x0F));
    if (Scup < 2 || Scup > Lcup || Scup > 4079) {
      printf("WARNING: cleanup pass suffix length %d is invalid.\n", Scup);
      return false;
    }
    // modDcup (shall be done before the creation of state_VLC instance)
    Dcup[Lcup - 1] = 0xFF;
    Dcup[Lcup - 2] |= 0x0F;
    const int32_t Pcup = static_cast<int32_t>(Lcup - Scup);

    // HT block decoding
    ht_cleanup_decode(block, static_cast<uint8_t>(30 - S_blk), Lcup, Pcup, Scup);
    if (num_ht_passes > 1) {
      ht_sigprop_decode(block, Dref, Lref, static_cast<uint8_t>(30 - (S_blk + 1)));
    }
    if (num_ht_passes > 2) {
      ht_magref_decode(block, Dref, Lref, static_cast<uint8_t>(30 - (S_blk + 1)));
    }

    // dequantization
    block->dequantize(ROIshift);

  }  // block decoding end

  return true;
}
#endif