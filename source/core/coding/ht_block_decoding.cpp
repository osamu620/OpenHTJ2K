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

#if !defined(OPENHTJ2K_ENABLE_ARM_NEON) && (!defined(__AVX2__) || !defined(OPENHTJ2K_TRY_AVX2)) \
    && !defined(OPENHTJ2K_ENABLE_WASM_SIMD)
  #include "coding_units.hpp"
  #include "dec_CxtVLC_tables.hpp"
  #include "ht_block_decoding.hpp"
  #include "block_decoding.hpp"
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

// Scalar fused dequantize-and-store: convert sign-magnitude int32 to dequantized float.
template <bool StoreI32 = false>
static FORCE_INLINE void dequant_store_scalar(void *dst, int32_t val, uint8_t transformation,
                                              int32_t pLSB_dq, float fscale_direct) {
  if (transformation == 1) {
    int32_t sign = val & INT32_MIN;
    val &= INT32_MAX;
    val >>= pLSB_dq;
    if (sign) val = -(val & INT32_MAX);
    if constexpr (StoreI32)
      *reinterpret_cast<int32_t *>(dst) = val;
    else
      *reinterpret_cast<float *>(dst) = static_cast<float>(val);
  } else {
    int32_t sign = val & INT32_MIN;
    float f      = static_cast<float>(val & INT32_MAX) * fscale_direct;
    if (sign) f = -f;
    *reinterpret_cast<float *>(dst) = f;
  }
}

// Step-2 of the HT cleanup pass: MagSgn decoding over the (tv, u) scratch
// written by ht_cleanup_step1_nway (the former phase 1 of this function).
// Kept per-block: the fwd_buf destuff scratch is thread-local (constructing a
// second fwd_buf on the same thread invalidates the first), and step-2 is
// throughput-bound — there is nothing to gain from interleaving it.
template <bool fuse_dequant = false, bool store_i32 = false>
static void ht_cleanup_step2(j2k_codeblock *block, const uint8_t pLSB, const int32_t Pcup,
                             uint16_t *scratch, const int32_t sstr) {
  uint8_t *compressed_data = block->get_compressed_data();
  /*******************************************************************************************************************/
  // MagSgn decoding
  /*******************************************************************************************************************/
  {
    // Fused dequantize setup
    int32_t pLSB_dq     = 0;
    float fscale_direct = 0.0f;
    uint32_t out_stride = block->blksampl_stride;
    if constexpr (fuse_dequant) {
      const int32_t M_b_val = block->get_Mb();
      pLSB_dq               = 31 - M_b_val;
      out_stride            = block->band_stride;
      if (block->transformation != 1) {
        // lossy path (transformation==0 for irrev97, transformation>=2 for ATK irrev)
        fscale_direct = block->stepsize;
        fscale_direct *= static_cast<float>(1 << FRACBITS);
        if (M_b_val <= 31)
          fscale_direct /= static_cast<float>(1 << (31 - M_b_val));
        else
          fscale_direct *= static_cast<float>(1 << (M_b_val - 31));
      }
    }

    // We allocate a scratch row for storing v_n values.
    // We have 512 quads horizontally.
    // We need an extra entry to handle the case of vp[1]
    // when vp is at the last column.
    // Here, we allocate 4 instead of 1 to make the buffer size
    // a multipled of 16 bytes.
    const int v_n_size             = 512 + 4;
    uint32_t v_n_scratch[v_n_size] = {0};  // 2+ kB

    fwd_buf<0xFF> MagSgn(compressed_data, Pcup);

    // Helper to store a decoded sample, optionally fusing dequantize.
    auto store_sample = [&](int32_t *dst, int32_t ival) {
      if constexpr (fuse_dequant) {
        dequant_store_scalar<store_i32>(dst, ival, block->transformation, pLSB_dq, fscale_direct);
      } else {
        *dst = ival;
      }
    };

    uint16_t *sp = scratch;
    uint32_t *vp = v_n_scratch;
    int32_t *dp  = fuse_dequant ? reinterpret_cast<int32_t *>(block->band_buf) : block->sample_buf;

    uint32_t prev_v_n = 0;
    for (uint32_t x = 0; x < block->size.x; sp += 2, ++vp) {
      uint32_t inf = sp[0];
      uint32_t U_q = sp[1];
      if (U_q > ((30U - pLSB) + 2U)) {
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
        v_n = ms_val & ((1U << m_n) - 1U);       // keep only m_n bits
        v_n |= ((inf >> (8 + bit)) & 1) << m_n;  // add EMB e_1 as MSB
        v_n |= 1;                                // add center of bin
        // v_n now has 2 * (\mu - 1) + 0.5 with correct sign bit
        // add 2 to make it 2*\mu+0.5, shift it up to missing MSBs
        val |= (v_n + 2) << (pLSB - 1);
      }
      store_sample(&dp[0], static_cast<int32_t>(val));

      v_n = 0;
      val = 0;
      bit = 1;
      if (inf & (1 << (4 + bit))) {
        // get 32 bits of magsgn data
        uint32_t ms_val = MagSgn.fetch();
        uint32_t m_n    = U_q - ((inf >> (12 + bit)) & 1);  // remove e_k
        MagSgn.advance(m_n);                                // consume m_n

        val = ms_val << 31;                      // get sign bit
        v_n = ms_val & ((1U << m_n) - 1U);       // keep only m_n bits
        v_n |= ((inf >> (8 + bit)) & 1) << m_n;  // add EMB e_1 as MSB
        v_n |= 1;                                // add center of bin
        // v_n now has 2 * (\mu - 1) + 0.5 with correct sign bit
        // add 2 to make it 2*\mu+0.5, shift it up to missing MSBs
        val |= (v_n + 2) << (pLSB - 1);
      }
      store_sample(&dp[out_stride], static_cast<int32_t>(val));
      vp[0]    = prev_v_n | v_n;
      prev_v_n = 0;
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
        v_n = ms_val & ((1U << m_n) - 1U);       // keep only m_n bits
        v_n |= ((inf >> (8 + bit)) & 1) << m_n;  // add EMB e_1 as MSB
        v_n |= 1;                                // add center of bin
        // v_n now has 2 * (\mu - 1) + 0.5 with correct sign bit
        // add 2 to make it 2*\mu+0.5, shift it up to missing MSBs
        val |= (v_n + 2) << (pLSB - 1);
      }
      store_sample(&dp[0], static_cast<int32_t>(val));

      v_n = 0;
      val = 0;
      bit = 3;
      if (inf & (1 << (4 + bit))) {
        // get 32 bits of magsgn data
        uint32_t ms_val = MagSgn.fetch();
        uint32_t m_n    = U_q - ((inf >> (12 + bit)) & 1);  // remove e_k
        MagSgn.advance(m_n);                                // consume m_n

        val = ms_val << 31;                      // get sign bit
        v_n = ms_val & ((1U << m_n) - 1U);       // keep only m_n bits
        v_n |= ((inf >> (8 + bit)) & 1) << m_n;  // add EMB e_1 as MSB
        v_n |= 1;                                // add center of bin
        // v_n now has 2 * (\mu - 1) + 0.5 with correct sign bit
        // add 2 to make it 2*\mu+0.5, shift it up to missing MSBs
        val |= (v_n + 2) << (pLSB - 1);
      }
      store_sample(&dp[out_stride], static_cast<int32_t>(val));
      prev_v_n = v_n;
      ++dp;
      ++x;
    }
    vp[0] = prev_v_n;

    for (uint32_t y = 2; y < block->size.y; y += 2) {
      uint16_t *sp = scratch + (y >> 1) * static_cast<uint32_t>(sstr);
      uint32_t *vp = v_n_scratch;
      int32_t *dp  = fuse_dequant ? reinterpret_cast<int32_t *>(block->band_buf) + y * block->band_stride
                                  : block->sample_buf + y * block->blksampl_stride;

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
        if (U_q > ((30U - pLSB) + 2U)) {
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
          v_n = ms_val & ((1U << m_n) - 1U);       // keep only m_n bits
          v_n |= ((inf >> (8 + bit)) & 1) << m_n;  // add EMB e_1 as MSB
          v_n |= 1;                                // add center of bin
          // v_n now has 2 * (\mu - 1) + 0.5 with correct sign bit
          // add 2 to make it 2*\mu+0.5, shift it up to missing MSBs
          val |= (v_n + 2) << (pLSB - 1);
        }
        store_sample(&dp[0], static_cast<int32_t>(val));

        v_n = 0;
        val = 0;
        bit = 1;
        if (inf & (1 << (4 + bit))) {
          // get 32 bits of magsgn data
          uint32_t ms_val = MagSgn.fetch();
          uint32_t m_n    = U_q - ((inf >> (12 + bit)) & 1);  // remove e_k
          MagSgn.advance(m_n);                                // consume m_n

          val = ms_val << 31;                      // get sign bit
          v_n = ms_val & ((1U << m_n) - 1U);       // keep only m_n bits
          v_n |= ((inf >> (8 + bit)) & 1) << m_n;  // add EMB e_1 as MSB
          v_n |= 1;                                // add center of bin
          // v_n now has 2 * (\mu - 1) + 0.5 with correct sign bit
          // add 2 to make it 2*\mu+0.5, shift it up to missing MSBs
          val |= (v_n + 2) << (pLSB - 1);
        }
        store_sample(&dp[out_stride], static_cast<int32_t>(val));
        vp[0]    = prev_v_n | v_n;
        prev_v_n = 0;
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
          v_n = ms_val & ((1U << m_n) - 1U);       // keep only m_n bits
          v_n |= ((inf >> (8 + bit)) & 1) << m_n;  // add EMB e_1 as MSB
          v_n |= 1;                                // add center of bin
          // v_n now has 2 * (\mu - 1) + 0.5 with correct sign bit
          // add 2 to make it 2*\mu+0.5, shift it up to missing MSBs
          val |= (v_n + 2) << (pLSB - 1);
        }
        store_sample(&dp[0], static_cast<int32_t>(val));

        v_n = 0;
        val = 0;
        bit = 3;
        if (inf & (1 << (4 + bit))) {
          // get 32 bits of magsgn data
          uint32_t ms_val = MagSgn.fetch();
          uint32_t m_n    = U_q - ((inf >> (12 + bit)) & 1);  // remove e_k
          MagSgn.advance(m_n);                                // consume m_n

          val = ms_val << 31;                      // get sign bit
          v_n = ms_val & ((1U << m_n) - 1U);       // keep only m_n bits
          v_n |= ((inf >> (8 + bit)) & 1) << m_n;  // add EMB e_1 as MSB
          v_n |= 1;                                // add center of bin
          // v_n now has 2 * (\mu - 1) + 0.5 with correct sign bit
          // add 2 to make it 2*\mu+0.5, shift it up to missing MSBs
          val |= (v_n + 2) << (pLSB - 1);
        }
        store_sample(&dp[out_stride], static_cast<int32_t>(val));
        prev_v_n = v_n;
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

  alignas(32) uint32_t rholine[516];  // QW_max + 4, QW_max = 512
  std::memset(rholine, 0, (QW + 4U) * sizeof(uint32_t));
  uint32_t *rho_p = rholine + 1;
  alignas(32) int32_t Eline[1032];  // 2 * QW_max + 8, QW_max = 512
  std::memset(Eline, 0, (2U * QW + 8U) * sizeof(int32_t));
  int32_t *E_p = Eline + 1;

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

    // Branchless MEL offset: replace compound branch with mask
    uint32_t both_off = u_off0 & u_off1;
    int32_t om        = -static_cast<int32_t>(both_off);
    mel_run -= om & 2;
    uint32_t mel_offset = static_cast<uint32_t>(-(mel_run == -1) & om) & 0x40;
    if (mel_run < 0) mel_run = MEL.get_run();
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
    rho_p = rholine + 1;
    E_p   = Eline + 1;
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
  if (pLSB == 0) return;  // no plane below the LSB; mirrors ht_magref_decode (avoids 1 << (pLSB-1) UB)
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
  if (pLSB == 0) return;
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
          tmp = static_cast<int32_t>((0xFFFFFFFEU | static_cast<unsigned int>(bit)) << pLSB);
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
        tmp = static_cast<int32_t>((0xFFFFFFFEU | static_cast<unsigned int>(bit)) << pLSB);
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

  // Direct float scale factor: decoded magnitude is in Q(31-M_b) fixed-point;
  // the result must be in Q(FRACBITS).
  float fscale_direct = this->stepsize;
  fscale_direct *= static_cast<float>(1 << FRACBITS);
  if (M_b <= 31) {
    fscale_direct /= (static_cast<float>(1 << (31 - M_b)));
  } else {
    fscale_direct *= (static_cast<float>(1 << (M_b - 31)));
  }
  if (this->transformation == 1) {
    auto rev_loop = [&](auto store_fn) {
      for (size_t i = 0; i < static_cast<size_t>(this->size.y); i++) {
        int32_t *val = this->sample_buf + i * this->blksampl_stride;
        sprec_t *dst = this->band_buf + i * this->band_stride;
        size_t len   = this->size.x;
        for (; len > 0; --len) {
          int32_t sign = *val & INT32_MIN;
          *val &= INT32_MAX;
          if (ROIshift && (((uint32_t)*val & ~mask) == 0)) {
            *val <<= ROIshift;
          }
          *val >>= pLSB;
          if (sign) {
            *val = -(*val & INT32_MAX);
          }
          assert(pLSB >= 0);
          store_fn(dst, *val);
          val++;
          dst++;
        }
      }
    };
    if (this->dequant_i32)
      rev_loop([](sprec_t *d, int32_t v) { *reinterpret_cast<int32_t *>(d) = v; });
    else
      rev_loop([](sprec_t *d, int32_t v) { *d = static_cast<sprec_t>(v); });
  } else if (ROIshift == 0) {
    // Lossy, no ROI (common case): direct float multiply, matching the fused
    // dequant path (dequant_store_scalar) and the AVX2/NEON/WASM variants of
    // this function bit-exactly.  The previous integer fixed-point pipeline
    // here rounded every sample to an integer in the Q(FRACBITS) domain, so
    // blocks taking this fallback (multiple HT passes, odd height) were
    // dequantized with different precision than fused blocks and than SIMD
    // builds, causing up to ±1 LSB divergence in decoded images.
    for (size_t i = 0; i < static_cast<size_t>(this->size.y); i++) {
      int32_t *val = this->sample_buf + i * this->blksampl_stride;
      sprec_t *dst = this->band_buf + i * this->band_stride;
      size_t len   = this->size.x;
      for (; len > 0; --len) {
        const int32_t sign = *val & INT32_MIN;
        float f            = static_cast<float>(*val & INT32_MAX) * fscale_direct;
        if (sign) f = -f;
        *dst = f;
        val++;
        dst++;
      }
    }
  } else {
    // Lossy ROI path — rarely used; keep the integer-arithmetic approach.
    constexpr int32_t downshift = 15;
    float fscale                = fscale_direct;
    fscale *= static_cast<float>(1 << 16) * static_cast<float>(1 << downshift);
    const auto scale = static_cast<int32_t>(fscale + 0.5f);
    for (size_t i = 0; i < static_cast<size_t>(this->size.y); i++) {
      int32_t *val = this->sample_buf + i * this->blksampl_stride;
      sprec_t *dst = this->band_buf + i * this->band_stride;
      size_t len   = this->size.x;

      for (; len > 0; --len) {
        int32_t sign = *val & INT32_MIN;
        *val &= INT32_MAX;
        // detect background region and upshift it
        if (((uint32_t)*val & ~mask) == 0) {
          *val <<= ROIshift;
        }
        // to prevent overflow, truncate to int16_t
        *val = (*val + (1 << 15)) >> 16;
        //  dequantization
        *val *= scale;
        // downshift
        *val = (int32_t)((*val + (1 << (downshift - 1))) >> downshift);
        // convert sign-magnitude to two's complement form
        if (sign) {
          *val = -(*val & INT32_MAX);
        }
        *dst = static_cast<sprec_t>(*val);
        val++;
        dst++;
      }
    }
  }
}

// Per-block decode setup: segment validation, Lcup/Lref/Scup/Pcup derivation
// and the modDcup buffer mutation.  Factored out of htj2k_decode so the
// 1-way and batched entries share one source of truth.
struct ht_dec_setup {
  int32_t Lcup, Pcup, Scup;
  uint32_t Lref;
  uint8_t *Dref;  // nullptr when single-segment
  uint8_t S_blk;
  uint8_t num_ht_passes;
  bool ok;     // false → htj2k_decode returns false
  bool empty;  // num_ht_passes == 0 → success with no work
};

static ht_dec_setup htj2k_dec_setup(j2k_codeblock *block) {
  ht_dec_setup su;
  su.Lcup  = 0;
  su.Pcup  = 0;
  su.Scup  = 0;
  su.Lref  = 0;
  su.Dref  = nullptr;
  su.S_blk = 0;
  su.ok    = false;
  su.empty = false;

  // number of placeholder pass
  uint8_t P0 = 0;
  // number of HT Sets preceding the given(this) HT Set
  const uint8_t S_skip = 0;

  if (block->num_passes > 3) {
    for (uint32_t i = 0; i < block->pass_length_count; i++) {
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
    return su;
  }
  // number of ht coding pass (Z_blk in the spec)
  su.num_ht_passes = static_cast<uint8_t>(block->num_passes - empty_passes);
  if (su.num_ht_passes == 0) {
    su.ok    = true;
    su.empty = true;
    return su;
  }

  // HT defines at most two segments per codeblock (Cleanup + optional
  // Refinement); `all_segments[4]` is over-provisioned.  A malformed
  // input with pass_length_count > 4 non-zero entries used to write
  // past the array and smash the stack — guard the write and the
  // later `all_segments[0]` read.  Reported by IM JUN SEO (KISIA) and
  // OH HAN GUEL (SANGMYUNG UNIVERSITY).
  uint8_t all_segments[4];
  uint32_t num_segments = 0;
  for (uint32_t i = 0; i < block->pass_length_count; i++) {
    if (block->pass_length[i] != 0) {
      if (num_segments >= 4) {
        printf("WARNING: too many HT coding-pass segments (>4) — malformed input.\n");
        return su;
      }
      all_segments[num_segments++] = static_cast<uint8_t>(i);
    }
  }
  if (num_segments == 0) {
    printf("WARNING: no non-empty HT coding-pass segments.\n");
    return su;
  }
  su.Lcup += static_cast<int32_t>(block->pass_length[all_segments[0]]);
  if (su.Lcup < 2) {
    printf("WARNING: Cleanup pass length must be at least 2 bytes in length.\n");
    return su;
  }
  // Bound the attacker-controlled cleanup length by the (already clamped)
  // codeblock byte count before any Dcup[Lcup-1] access / modDcup write.
  if (static_cast<uint32_t>(su.Lcup) > block->length) {
    printf("WARNING: HT cleanup pass length %d exceeds codeblock bytes %u — malformed input.\n", su.Lcup,
           block->length);
    return su;
  }
  for (uint32_t i = 1; i < num_segments; i++) {
    su.Lref += block->pass_length[all_segments[i]];
  }
  // The refinement segments are read from Dcup + Lcup; keep them in the buffer.
  if (static_cast<uint32_t>(su.Lref) > block->length - static_cast<uint32_t>(su.Lcup)) {
    printf("WARNING: HT refinement length exceeds remaining codeblock bytes %u — malformed input.\n",
           block->length - static_cast<uint32_t>(su.Lcup));
    return su;
  }
  uint8_t *Dcup = block->get_compressed_data();

  if (block->num_passes > 1 && num_segments > 1) {
    su.Dref = block->get_compressed_data() + su.Lcup;
  } else {
    su.Dref = nullptr;
  }
  // number of (skipped) magnitude bitplanes
  su.S_blk = static_cast<uint8_t>(P0 + block->num_ZBP + S_skip);
  if (su.S_blk >= 30) {
    printf("WARNING: Number of skipped mag bitplanes %d is too large.\n", su.S_blk);
    return su;
  }
  // Suffix length (=MEL + VLC) of HT Cleanup pass
  su.Scup = static_cast<int32_t>((Dcup[su.Lcup - 1] << 4) + (Dcup[su.Lcup - 2] & 0x0F));
  if (su.Scup < 2 || su.Scup > su.Lcup || su.Scup > 4079) {
    printf("WARNING: cleanup pass suffix length %d is invalid.\n", su.Scup);
    return su;
  }
  // modDcup (shall be done before the creation of state_VLC instance)
  Dcup[su.Lcup - 1] = 0xFF;
  Dcup[su.Lcup - 2] |= 0x0F;
  su.Pcup = static_cast<int32_t>(su.Lcup - su.Scup);
  su.ok   = true;
  return su;
}

// Everything after step-1: step-2 (MagSgn) variant dispatch, SigProp/MagRef
// refinement passes, and dequantization.  scratch/sstr hold the step-1 output.
static bool htj2k_dec_finish(j2k_codeblock *block, const ht_dec_setup &su, const uint8_t ROIshift,
                             uint16_t *scratch, const int32_t sstr) {
  const uint8_t pLSB = static_cast<uint8_t>(30 - su.S_blk);

  // HT block decoding
  bool dequant_done = false;
  // The fused-dequant scalar/NEON/AVX2/WASM kernels process samples in
  // 2-row quad pairs and write both rows of every pair unconditionally to
  // `block->band_buf`.  When `block->size.y` is odd the last pair's
  // second-row write lands `band_stride` bytes past the block's final
  // row — into the NEXT block's band_buf region when codeblocks tile
  // vertically inside the subband (e.g. 1×1 blocks stacked in a narrow
  // subband from a horizontally-subsampled component).  Under single-
  // threaded decode the overflow is harmlessly overwritten by the next
  // block's legitimate row-0 write, but under multi-threaded dispatch
  // the blocks finish out-of-order and the stale overflow clobbers
  // adjacent blocks' output.  Fall back to the non-fused path (which
  // writes into the block-local sample_buf scratch, not band_buf) when
  // height is odd; the separate dequant pass below handles bounds
  // correctly.
  const bool fuseable = (su.num_ht_passes == 1) && (ROIshift == 0) && ((block->size.y & 1u) == 0u);
  if (fuseable) {
    if (block->dequant_i32)
      ht_cleanup_step2<true, true>(block, pLSB, su.Pcup, scratch, sstr);
    else
      ht_cleanup_step2<true>(block, pLSB, su.Pcup, scratch, sstr);
    dequant_done = true;
  } else {
    ht_cleanup_step2<>(block, pLSB, su.Pcup, scratch, sstr);
  }
  if (su.num_ht_passes > 1) {
    ht_sigprop_decode(block, su.Dref, su.Lref, static_cast<uint8_t>(30 - (su.S_blk + 1)));
  }
  if (su.num_ht_passes > 2) {
    ht_magref_decode(block, su.Dref, su.Lref, static_cast<uint8_t>(30 - (su.S_blk + 1)));
  }

  // dequantization (skipped when already fused into MagSgn output)
  if (!dequant_done) {
    block->dequantize(ROIshift);
  }

  return true;
}

// Decode one block from an already-computed setup.  htj2k_dec_setup is NOT
// idempotent (modDcup mutates the compressed buffer, so a re-run reads a
// corrupted Scup) — every setup must be consumed by exactly one decode.
static bool htj2k_decode_su(j2k_codeblock *block, const ht_dec_setup &su, const uint8_t ROIshift) {
  if (!su.ok) return false;
  if (su.empty) return true;

  const uint16_t QW  = static_cast<uint16_t>(ceil_int(static_cast<int16_t>(block->size.x), 2));
  const uint16_t QH  = static_cast<uint16_t>(ceil_int(static_cast<int16_t>(block->size.y), 2));
  const int32_t sstr = static_cast<int32_t>(((block->size.x + 2) + 7u) & ~7u);  // multiples of 8

  uint16_t scratch[8 * 513];
  ht_step1_lane ln;
  ln.Dcup            = block->get_compressed_data();
  ln.Lcup            = su.Lcup;
  ln.Scup            = su.Scup;
  ln.scratch         = scratch;
  ln.block_states    = block->block_states;
  ln.blkstate_stride = block->blkstate_stride;
  if (su.num_ht_passes == 1) {
    ht_cleanup_step1_nway<1, true>(&ln, QW, QH, sstr);
  } else {
    ht_cleanup_step1_nway<1, false>(&ln, QW, QH, sstr);
  }

  return htj2k_dec_finish(block, su, ROIshift, scratch, sstr);
}

bool htj2k_decode(j2k_codeblock *block, const uint8_t ROIshift) {
  return htj2k_decode_su(block, htj2k_dec_setup(block), ROIshift);
}

// Number of codeblocks whose step-1 chains are decoded in lockstep.
  #ifndef OPENHTJ2K_HT_DEC_BATCH_N
    #define OPENHTJ2K_HT_DEC_BATCH_N 2
  #endif

// Batched entry point (see block_decoding.hpp).  Walks the block list and
// decodes OPENHTJ2K_HT_DEC_BATCH_N consecutive blocks with the N-way
// lockstep step-1 kernel whenever they share dimensions and pass-class and
// all validate; everything else falls back to the 1-way path.  Output is
// byte-identical to per-block decoding (the lockstep kernel preserves each
// lane's operation sequence, and step-2 runs per block in list order).
bool htj2k_decode_batch(j2k_codeblock *const *blocks, uint32_t n, uint8_t ROIshift, bool *results) {
  constexpr uint32_t BN = OPENHTJ2K_HT_DEC_BATCH_N;
  bool all_ok           = true;
  uint32_t i            = 0;
  while (i < n) {
    // A group needs BN consecutive blocks of equal dimensions (⇒ shared
    // QW/QH/sstr) — check the cheap key before running any setup.
    bool key = (i + BN <= n);
    for (uint32_t k = 1; key && k < BN; ++k) {
      key = blocks[i + k]->size.x == blocks[i]->size.x && blocks[i + k]->size.y == blocks[i]->size.y;
    }
    if (!key) {
      results[i] = htj2k_decode(blocks[i], ROIshift);
      all_ok &= results[i];
      ++i;
      continue;
    }

    // Setup mutates each block's buffer (modDcup): from here on, every one
    // of the BN setups is consumed below, batched or not.
    ht_dec_setup su[BN];
    bool group = true;
    for (uint32_t k = 0; k < BN; ++k) {
      su[k] = htj2k_dec_setup(blocks[i + k]);
      group &= su[k].ok && !su[k].empty;
    }
    // skip_sigma is a step-1 template parameter: lanes must share pass-class.
    for (uint32_t k = 1; group && k < BN; ++k) {
      group &= (su[k].num_ht_passes == 1) == (su[0].num_ht_passes == 1);
    }
    if (!group) {
      for (uint32_t k = 0; k < BN; ++k) {
        results[i + k] = htj2k_decode_su(blocks[i + k], su[k], ROIshift);
        all_ok &= results[i + k];
      }
      i += BN;
      continue;
    }

    const j2k_codeblock *b0 = blocks[i];
    const uint16_t QW       = static_cast<uint16_t>(ceil_int(static_cast<int16_t>(b0->size.x), 2));
    const uint16_t QH       = static_cast<uint16_t>(ceil_int(static_cast<int16_t>(b0->size.y), 2));
    const int32_t sstr      = static_cast<int32_t>(((b0->size.x + 2) + 7u) & ~7u);  // multiples of 8
    const bool skip_sigma   = su[0].num_ht_passes == 1;

    uint16_t scratch[BN][8 * 513];
    ht_step1_lane ln[BN];
    for (uint32_t k = 0; k < BN; ++k) {
      ln[k].Dcup            = blocks[i + k]->get_compressed_data();
      ln[k].Lcup            = su[k].Lcup;
      ln[k].Scup            = su[k].Scup;
      ln[k].scratch         = scratch[k];
      ln[k].block_states    = blocks[i + k]->block_states;
      ln[k].blkstate_stride = blocks[i + k]->blkstate_stride;
    }

    bool step1_done = true;
    try {
      if (skip_sigma) {
        ht_cleanup_step1_nway<BN, true>(ln, QW, QH, sstr);
      } else {
        ht_cleanup_step1_nway<BN, false>(ln, QW, QH, sstr);
      }
    } catch (...) {
      // Malformed input: one lane's rev_buf underflowed mid-lockstep.  Redo
      // every lane 1-way from its saved setup so per-block results and
      // exceptions match the non-batched path exactly (per-lane step-1 work
      // is idempotent: scratch is fully rewritten, sigma stores are pure
      // assignments).  This path never runs on valid streams.
      step1_done = false;
    }
    for (uint32_t k = 0; k < BN; ++k) {
      if (!step1_done) {
        ht_step1_lane l1 = ln[k];
        if (skip_sigma) {
          ht_cleanup_step1_nway<1, true>(&l1, QW, QH, sstr);  // may throw, like the 1-way path
        } else {
          ht_cleanup_step1_nway<1, false>(&l1, QW, QH, sstr);
        }
      }
      results[i + k] = htj2k_dec_finish(blocks[i + k], su[k], ROIshift, scratch[k], sstr);
      all_ok &= results[i + k];
    }
    i += BN;
  }
  return all_ok;
}

const uint32_t htj2k_dec_batch_lanes = OPENHTJ2K_HT_DEC_BATCH_N;
#endif