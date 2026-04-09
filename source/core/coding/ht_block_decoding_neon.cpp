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
  #include "coding_units.hpp"
  #include "dec_CxtVLC_tables.hpp"
  #include "ht_block_decoding.hpp"
  #include "coding_local.hpp"
  #include "utils.hpp"

  #include <arm_neon.h>

uint8_t j2k_codeblock::calc_mbr(const uint32_t i, const uint32_t j, const uint8_t causal_cond) const {
  uint8_t *state_p0 = block_states + i * blkstate_stride + j;
  uint8_t *state_p1 = block_states + (i + 1) * blkstate_stride + j;
  uint8_t *state_p2 = block_states + (i + 2) * blkstate_stride + j;

  uint8_t mbr0 = state_p0[0] | state_p0[1] | state_p0[2];
  uint8_t mbr1 = state_p1[0] | state_p1[2];
  uint8_t mbr2 = state_p2[0] | state_p2[1] | state_p2[2];
  uint8_t mbr  = mbr0 | mbr1 | (mbr2 & causal_cond);
  mbr |= (mbr0 >> SHIFT_REF) & (mbr0 >> SHIFT_SCAN);
  mbr |= (mbr1 >> SHIFT_REF) & (mbr1 >> SHIFT_SCAN);
  mbr |= (mbr2 >> SHIFT_REF) & (mbr2 >> SHIFT_SCAN) & causal_cond;

  //  uint8_t mbr = state_p0[0];
  //  mbr |= state_p0[1];
  //  mbr |= state_p0[2];
  //  mbr |= state_p1[0];
  //  mbr |= state_p1[2];
  //  mbr |= (state_p2[0]) & causal_cond;
  //  mbr |= (state_p2[1]) & causal_cond;
  //  mbr |= (state_p2[2]) & causal_cond;
  //
  //  mbr |= ((state_p0[0] >> SHIFT_REF)) & ((state_p0[0] >> SHIFT_SCAN));
  //  mbr |= ((state_p0[1] >> SHIFT_REF)) & ((state_p0[1] >> SHIFT_SCAN));
  //  mbr |= ((state_p0[2] >> SHIFT_REF)) & ((state_p0[2] >> SHIFT_SCAN));
  //
  //  mbr |= ((state_p1[0] >> SHIFT_REF)) & ((state_p1[0] >> SHIFT_SCAN));
  //  mbr |= ((state_p1[2] >> SHIFT_REF)) & ((state_p1[2] >> SHIFT_SCAN));
  //
  //  mbr |= ((state_p2[0] >> SHIFT_REF)) & ((state_p2[0] >> SHIFT_SCAN)) & causal_cond;
  //  mbr |= ((state_p2[1] >> SHIFT_REF)) & ((state_p2[1] >> SHIFT_SCAN)) & causal_cond;
  //  mbr |= ((state_p2[2] >> SHIFT_REF)) & ((state_p2[2] >> SHIFT_SCAN)) & causal_cond;

  //  const int16_t im1 = static_cast<int16_t>(i - 1);
  //  const int16_t jm1 = static_cast<int16_t>(j - 1);
  //  const int16_t ip1 = static_cast<int16_t>(i + 1);
  //  const int16_t jp1 = static_cast<int16_t>(j + 1);
  //  uint8_t mbr       = get_state(Sigma, im1, jm1);
  //  mbr         = mbr | get_state(Sigma, im1, j);
  //  mbr = mbr | get_state(Sigma, im1, jp1);
  //  mbr = mbr | get_state(Sigma, i, jm1);
  //  mbr = mbr | get_state(Sigma, i, jp1);
  //  mbr = mbr | static_cast<uint8_t>(get_state(Sigma, ip1, jm1) * causal_cond);
  //  mbr = mbr | static_cast<uint8_t>(get_state(Sigma, ip1, j) * causal_cond);
  //  mbr = mbr | static_cast<uint8_t>(get_state(Sigma, ip1, jp1) * causal_cond);
  //  mbr = mbr | static_cast<uint8_t>(get_state(Refinement_value, im1, jm1) * get_state(Scan, im1, jm1));
  //  mbr = mbr | static_cast<uint8_t>(get_state(Refinement_value, im1, j) * get_state(Scan, im1, j));
  //  mbr = mbr | static_cast<uint8_t>(get_state(Refinement_value, im1, jp1) * get_state(Scan, im1, jp1));
  //  mbr = mbr | static_cast<uint8_t>(get_state(Refinement_value, i, jm1) * get_state(Scan, i, jm1));
  //  mbr = mbr | static_cast<uint8_t>(get_state(Refinement_value, i, jp1) * get_state(Scan, i, jp1));
  //  mbr = mbr
  //        | static_cast<uint8_t>(get_state(Refinement_value, ip1, jm1) * get_state(Scan, ip1, jm1)
  //                               * causal_cond);
  //  mbr = mbr
  //        | static_cast<uint8_t>(get_state(Refinement_value, ip1, j) * get_state(Scan, ip1, j) *
  //        causal_cond);
  //  mbr = mbr
  //        | static_cast<uint8_t>(get_state(Refinement_value, ip1, jp1) * get_state(Scan, ip1, jp1)
  //                               * causal_cond);
  return mbr & 1;
}

// Fused dequantize-and-store for 4 × int32 MagSgn samples → 4 × float.
// Lossless (transformation==1): sign-magnitude → two's-complement shift → float.
// Lossy   (transformation==0): magnitude → float → scale → apply sign via XOR.
static FORCE_INLINE void dequant_store_neon(int32_t *dst, int32x4_t val, uint8_t transformation,
                                            int32_t pLSB_dq, float32x4_t vfscale, int32x4_t vmagmask,
                                            int32x4_t vsignmask) {
  if (transformation == 1) {
    int32x4_t mag     = vandq_s32(val, vmagmask);
    int32x4_t shifted = vshlq_s32(mag, vdupq_n_s32(-pLSB_dq));
    uint32x4_t neg    = vreinterpretq_u32_s32(vshrq_n_s32(val, 31));
    int32x4_t res     = vbslq_s32(neg, vnegq_s32(shifted), shifted);
    vst1q_f32(reinterpret_cast<float *>(dst), vcvtq_f32_s32(res));
  } else {
    int32x4_t mag  = vandq_s32(val, vmagmask);
    float32x4_t f  = vmulq_f32(vcvtq_f32_s32(mag), vfscale);
    f              = vreinterpretq_f32_s32(
        veorq_s32(vreinterpretq_s32_f32(f), vandq_s32(val, vsignmask)));
    vst1q_f32(reinterpret_cast<float *>(dst), f);
  }
}

// Returns {max(a[0..3]), max(b[0..3])} as int32x2_t using NEON pairwise max.
// Avoids the SIMD-to-scalar extraction that vmaxvq_s32 requires.
static inline int32x2_t max4_pair(int32x4_t a, int32x4_t b) {
  int32x2_t ra = vpmax_s32(vget_low_s32(a), vget_high_s32(a));
  int32x2_t rb = vpmax_s32(vget_low_s32(b), vget_high_s32(b));
  return vpmax_s32(ra, rb);
}

template <bool skip_sigma, bool fuse_dequant = false>
void ht_cleanup_decode(j2k_codeblock *block, const uint8_t &pLSB, const int32_t Lcup, const int32_t Pcup,
                       const int32_t Scup) {
  fwd_buf<0xFF> MagSgn(block->get_compressed_data(), Pcup);
  MEL_dec MEL(block->get_compressed_data(), Lcup, Scup);
  rev_buf VLC_dec(block->get_compressed_data(), Lcup, Scup);

  const uint16_t QW = static_cast<uint16_t>(ceil_int(static_cast<int16_t>(block->size.x), 2));
  const uint16_t QH = static_cast<uint16_t>(ceil_int(static_cast<int16_t>(block->size.y), 2));

  int32x4_t vExp;
  const int32_t mask[4]  = {1, 2, 4, 8};
  const int32x4_t vm     = vld1q_s32(mask);
  const int32x4_t vone   = vdupq_n_s32(1);
  const int32x4_t vtwo   = vdupq_n_s32(2);
  const int32x4_t vshift = vdupq_n_s32(pLSB - 1);

  // Fused dequantize setup: when fuse_dequant is true, we write dequantized float values
  // directly to i_samples, eliminating the separate dequantize pass.
  // pLSB_dq is the dequantization shift (31 - M_b), distinct from the MagSgn pLSB.
  int32_t pLSB_dq         = 0;
  float32x4_t vfscale_dq  = vdupq_n_f32(0.0f);
  int32x4_t vmagmask_dq   = vdupq_n_s32(0);
  int32x4_t vsignmask_dq  = vdupq_n_s32(0);
  if constexpr (fuse_dequant) {
    const int32_t M_b_val = block->get_Mb();
    pLSB_dq               = 31 - M_b_val;
    vmagmask_dq            = vdupq_n_s32(0x7FFFFFFF);
    vsignmask_dq           = vdupq_n_s32(INT32_MIN);
    if (block->transformation != 1) {
      // lossy path (transformation==0 for irrev97, transformation>=2 for ATK irrev)
      float fscale_direct = block->stepsize;
      fscale_direct *= static_cast<float>(1 << FRACBITS);
      if (M_b_val <= 31)
        fscale_direct /= static_cast<float>(1 << (31 - M_b_val));
      else
        fscale_direct *= static_cast<float>(1 << (M_b_val - 31));
      vfscale_dq = vdupq_n_f32(fscale_direct);
    }
  }

  int32_t *const sample_buf = block->sample_buf;
  int32_t *mp0 = fuse_dequant ? reinterpret_cast<int32_t *>(block->i_samples) : sample_buf;
  int32_t *mp1 = mp0 + (fuse_dequant ? block->band_stride : block->blksampl_stride);
  auto sp0 = block->block_states + 1 + block->blkstate_stride;
  auto sp1 = block->block_states + 1 + 2 * block->blkstate_stride;

  uint32_t rho0, rho1;
  uint32_t u_off0, u_off1;
  uint32_t emb_k_0, emb_k_1;
  uint32_t emb_1_0, emb_1_1;
  uint32_t u0, u1;
  uint32_t U0, U1;
  uint32_t kappa0 = 1, kappa1 = 1;  // kappa is always 1 for initial line-pair

  const uint16_t *dec_table0, *dec_table1;
  dec_table0 = dec_CxtVLC_table0_fast_16;
  dec_table1 = dec_CxtVLC_table1_fast_16;

  alignas(32) uint32_t rholine[516];  // QW_max + 4, QW_max = 512
  std::memset(rholine, 0, (QW + 4U) * sizeof(uint32_t));
  uint32_t *rho_p    = rholine + 1;
  alignas(32) int32_t Eline[1032];   // 2 * QW_max + 8, QW_max = 512
  std::memset(Eline, 0, (2U * QW + 8U) * sizeof(int32_t));
  int32_t *E_p       = Eline + 1;

  uint32_t context = 0;
  uint32_t vlcval;
  int32_t mel_run = MEL.get_run();

  int32_t qx;
  // Initial line-pair
  for (qx = QW; qx > 0; qx -= 2) {
    // Decoding of significance and EMB patterns and unsigned residual offsets
    vlcval       = VLC_dec.fetch();
    uint16_t tv0 = dec_table0[(vlcval & 0x7F) + context];
    {
      // Branchless context-0 MEL handling: replace unpredictable branch with mask
      int32_t cm = -static_cast<int32_t>(context == 0);
      mel_run -= cm & 2;
      tv0 &= static_cast<uint16_t>(-(mel_run == -1) | ~cm);
      if (mel_run < 0) mel_run = MEL.get_run();
    }

    rho0    = (tv0 & 0x00F0) >> 4;
    emb_k_0 = (tv0 & 0xF000) >> 12;
    emb_1_0 = (tv0 & 0x0F00) >> 8;

    // calculate context for the next quad
    context = ((tv0 & 0xE0U) << 2) | ((tv0 & 0x10U) << 3);

    // Decoding of significance and EMB patterns and unsigned residual offsets
    vlcval       = VLC_dec.advance((tv0 & 0x000F) >> 1);
    uint16_t tv1 = dec_table0[(vlcval & 0x7F) + context];
    {
      int32_t cm = -static_cast<int32_t>((context == 0) & (qx > 1));
      mel_run -= cm & 2;
      tv1 &= static_cast<uint16_t>(-(mel_run == -1) | ~cm);
      if (mel_run < 0) mel_run = MEL.get_run();
    }
    tv1     = (qx > 1) ? tv1 : 0;
    rho1    = (tv1 & 0x00F0) >> 4;
    emb_k_1 = (tv1 & 0xF000) >> 12;
    emb_1_1 = (tv1 & 0x0F00) >> 8;

    // store sigma
    if (!skip_sigma) {
      *sp0++ = (rho0 >> 0) & 1;
      *sp0++ = (rho0 >> 2) & 1;
      *sp0++ = (rho1 >> 0) & 1;
      *sp0++ = (rho1 >> 2) & 1;
      *sp1++ = (rho0 >> 1) & 1;
      *sp1++ = (rho0 >> 3) & 1;
      *sp1++ = (rho1 >> 1) & 1;
      *sp1++ = (rho1 >> 3) & 1;
    }
    *rho_p++ = rho0;
    *rho_p++ = rho1;

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
    VLC_dec.advance(len);
    uvlc_result >>= 4;
    // quad 0 length
    len = uvlc_result & 0x7;  // quad 0 suffix length
    uvlc_result >>= 3;
    u0 = (uvlc_result & 7) + (tmp & ~(0xFFU << len));
    u1 = (uvlc_result >> 3) + (tmp >> len);

    U0 = kappa0 + u0;
    U1 = kappa1 + u1;

    if (pLSB > 16) {
      // 16-bit fast path: batch bit extraction via vqtbl1q_u8
      const uint8_t pLSB_adj = pLSB - 16;
      int16x4_t vn_16        = vdup_n_s16(0);
      int16x8_t row16 =
          MagSgn.decode_two_quads_16bit(tv0, tv1, static_cast<uint16_t>(U0),
                                        static_cast<uint16_t>(U1), pLSB_adj, vn_16);
      // Deinterleave row0/row1 and expand int16 -> int32 (sign bit 15 -> bit 31).
      int16x4_t lo      = vget_low_s16(row16);
      int16x4_t hi      = vget_high_s16(row16);
      int16x4_t row0_16 = vuzp1_s16(lo, hi);
      int16x4_t row1_16 = vuzp2_s16(lo, hi);
      if constexpr (fuse_dequant) {
        dequant_store_neon(mp0, vshll_n_s16(row0_16, 16), block->transformation, pLSB_dq,
                           vfscale_dq, vmagmask_dq, vsignmask_dq);
        dequant_store_neon(mp1, vshll_n_s16(row1_16, 16), block->transformation, pLSB_dq,
                           vfscale_dq, vmagmask_dq, vsignmask_dq);
      } else {
        vst1q_s32(mp0, vshll_n_s16(row0_16, 16));
        vst1q_s32(mp1, vshll_n_s16(row1_16, 16));
      }
      mp0 += 4;
      mp1 += 4;
      // Expand v_n to int32 and compute E_p.
      int32x4_t vn32 = vreinterpretq_s32_u32(vmovl_u16(vreinterpret_u16_s16(vn_16)));
      vExp           = vsubq_s32(vdupq_n_s32(32), vclzq_s32(vn32));
      vst1q_s32(E_p, vExp);
      E_p += 4;
    } else {
      // Existing 32-bit path
      int32x4_t vmask1, sig0, sig1, vtmp, m_n_0, m_n_1, msvec, v_n_0, v_n_1, mu0, mu1;

      sig0 = vdupq_n_u32(rho0);
      sig0 = vtstq_s32(sig0, vm);
      vtmp  = vandq_s32(vtstq_s32(vdupq_n_u32(emb_k_0), vm), vone);
      m_n_0 = vsubq_s32(vandq_s32(sig0, vdupq_n_u32(U0)), vtmp);
      sig1  = vdupq_n_u32(rho1);
      sig1  = vtstq_s32(sig1, vm);
      vtmp  = vandq_s32(vtstq_s32(vdupq_n_u32(emb_k_1), vm), vone);
      m_n_1 = vsubq_s32(vandq_s32(sig1, vdupq_n_u32(U1)), vtmp);

      vmask1 = vsubq_u32(vshlq_u32(vone, m_n_0), vone);
      msvec  = MagSgn.fetch(m_n_0);
      v_n_0  = vandq_u32(msvec, vmask1);
      vtmp   = vandq_s32(vtstq_s32(vdupq_n_u32(emb_1_0), vm), vone);
      v_n_0  = vorrq_u32(v_n_0, vshlq_u32(vtmp, m_n_0));
      mu0    = vaddq_u32(v_n_0, vtwo);
      mu0    = vorrq_s32(mu0, vone);
      mu0    = vshlq_u32(mu0, vshift);
      mu0    = vorrq_u32(mu0, vshlq_n_u32(v_n_0, 31));
      mu0    = vandq_u32(mu0, sig0);

      vmask1 = vsubq_u32(vshlq_u32(vone, m_n_1), vone);
      msvec  = MagSgn.fetch(m_n_1);
      v_n_1  = vandq_u32(msvec, vmask1);
      vtmp   = vandq_s32(vtstq_s32(vdupq_n_u32(emb_1_1), vm), vone);
      v_n_1  = vorrq_u32(v_n_1, vshlq_u32(vtmp, m_n_1));
      mu1    = vaddq_u32(v_n_1, vtwo);
      mu1    = vorrq_s32(mu1, vone);
      mu1    = vshlq_u32(mu1, vshift);
      mu1    = vorrq_u32(mu1, vshlq_n_u32(v_n_1, 31));
      mu1    = vandq_u32(mu1, sig1);

      if constexpr (fuse_dequant) {
        dequant_store_neon(mp0, vuzp1q_s32(mu0, mu1), block->transformation, pLSB_dq,
                           vfscale_dq, vmagmask_dq, vsignmask_dq);
        dequant_store_neon(mp1, vuzp2q_s32(mu0, mu1), block->transformation, pLSB_dq,
                           vfscale_dq, vmagmask_dq, vsignmask_dq);
      } else {
        vst1q_s32(mp0, vuzp1q_s32(mu0, mu1));
        vst1q_s32(mp1, vuzp2q_s32(mu0, mu1));
      }
      mp0 += 4;
      mp1 += 4;

      vExp = vsubq_s32(vdupq_n_s32(32), vclzq_s32(vuzp2q_s32(v_n_0, v_n_1)));
      vst1q_s32(E_p, vExp);
      E_p += 4;
    }
  }

  // Initial line-pair end

  /*******************************************************************************************************************/
  // Non-initial line-pair
  /*******************************************************************************************************************/
  for (uint16_t row = 1; row < QH; row++) {
    rho_p = rholine + 1;
    E_p   = Eline + 1;
    if constexpr (fuse_dequant) {
      mp0 = reinterpret_cast<int32_t *>(block->i_samples) + (row * 2U) * block->band_stride;
      mp1 = mp0 + block->band_stride;
    } else {
      mp0 = sample_buf + (row * 2U) * block->blksampl_stride;
      mp1 = sample_buf + (row * 2U + 1U) * block->blksampl_stride;
    }
    sp0   = block->block_states + (row * 2U + 1U) * block->blkstate_stride + 1U;
    sp1   = block->block_states + (row * 2U + 2U) * block->blkstate_stride + 1U;
    rho1  = 0;

    // {max(E_p[-1..2]), max(E_p[1..4])} — kept in NEON to avoid scalar round-trip
    int32x2_t vEmax = max4_pair(vld1q_s32(E_p - 1), vld1q_s32(E_p + 1));

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

      // store sigma
      if (!skip_sigma) {
        *sp0++ = (rho0 >> 0) & 1;
        *sp0++ = (rho0 >> 2) & 1;
        *sp0++ = (rho1 >> 0) & 1;
        *sp0++ = (rho1 >> 2) & 1;
        *sp1++ = (rho0 >> 1) & 1;
        *sp1++ = (rho0 >> 3) & 1;
        *sp1++ = (rho1 >> 1) & 1;
        *sp1++ = (rho1 >> 3) & 1;
      }
      // Update rho_p
      *rho_p++ = rho0;
      *rho_p++ = rho1;

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
      u0 = (uvlc_result & 7) + (tmp & ~(0xFFU << len));
      u1 = (uvlc_result >> 3) + (tmp >> len);

      {
        // gamma: 0 if popcount(rho) < 2 (single bit or zero), else 1
        int32x2_t vrho    = vset_lane_s32((int32_t)rho1, vdup_n_s32((int32_t)rho0), 1);
        int32x2_t vrho_m1 = vsub_s32(vrho, vdup_n_s32(1));
        // rho & (rho-1) == 0  ↔  popcount < 2  ↔  gamma = 0
        uint32x2_t vz     = vceq_u32(vreinterpret_u32_s32(vand_s32(vrho, vrho_m1)), vdup_n_u32(0));
        int32x2_t vgamma  = vreinterpret_s32_u32(vbic_u32(vdup_n_u32(1), vz));
        // kappa = max(1, gamma * (Emax - 1))
        int32x2_t vkappa  = vmax_s32(vmul_s32(vgamma, vsub_s32(vEmax, vdup_n_s32(1))), vdup_n_s32(1));
        U0 = (uint32_t)vget_lane_s32(vkappa, 0) + u0;
        U1 = (uint32_t)vget_lane_s32(vkappa, 1) + u1;
      }

      if (pLSB > 16) {
        // 16-bit fast path: batch bit extraction via vqtbl1q_u8
        const uint8_t pLSB_adj = pLSB - 16;
        int16x4_t vn_16        = vdup_n_s16(0);
        int16x8_t row16 =
            MagSgn.decode_two_quads_16bit(tv0, tv1, static_cast<uint16_t>(U0),
                                          static_cast<uint16_t>(U1), pLSB_adj, vn_16);
        int16x4_t lo      = vget_low_s16(row16);
        int16x4_t hi      = vget_high_s16(row16);
        int16x4_t row0_16 = vuzp1_s16(lo, hi);
        int16x4_t row1_16 = vuzp2_s16(lo, hi);
        if constexpr (fuse_dequant) {
          dequant_store_neon(mp0, vshll_n_s16(row0_16, 16), block->transformation, pLSB_dq,
                             vfscale_dq, vmagmask_dq, vsignmask_dq);
          dequant_store_neon(mp1, vshll_n_s16(row1_16, 16), block->transformation, pLSB_dq,
                             vfscale_dq, vmagmask_dq, vsignmask_dq);
        } else {
          vst1q_s32(mp0, vshll_n_s16(row0_16, 16));
          vst1q_s32(mp1, vshll_n_s16(row1_16, 16));
        }
        mp0 += 4;
        mp1 += 4;

        vEmax = max4_pair(vld1q_s32(E_p + 3), vld1q_s32(E_p + 5));

        int32x4_t vn32 = vreinterpretq_s32_u32(vmovl_u16(vreinterpret_u16_s16(vn_16)));
        vExp           = vsubq_s32(vdupq_n_s32(32), vclzq_s32(vn32));
        vst1q_s32(E_p, vExp);
        E_p += 4;
      } else {
        // Existing 32-bit path
        int32x4_t vmask1, sig0, sig1, vtmp, m_n_0, m_n_1, msvec, v_n_0, v_n_1, mu0, mu1;

        sig0 = vdupq_n_u32(rho0);
        sig0 = vtstq_s32(sig0, vm);
        vtmp  = vandq_s32(vtstq_s32(vdupq_n_u32(emb_k_0), vm), vone);
        m_n_0 = vsubq_s32(vandq_s32(sig0, vdupq_n_u32(U0)), vtmp);
        sig1  = vdupq_n_u32(rho1);
        sig1  = vtstq_s32(sig1, vm);
        vtmp  = vandq_s32(vtstq_s32(vdupq_n_u32(emb_k_1), vm), vone);
        m_n_1 = vsubq_s32(vandq_s32(sig1, vdupq_n_u32(U1)), vtmp);

        vmask1 = vsubq_u32(vshlq_u32(vone, m_n_0), vone);
        msvec  = MagSgn.fetch(m_n_0);
        v_n_0  = vandq_u32(msvec, vmask1);
        vtmp   = vandq_s32(vtstq_s32(vdupq_n_u32(emb_1_0), vm), vone);
        v_n_0  = vorrq_u32(v_n_0, vshlq_u32(vtmp, m_n_0));
        mu0    = vaddq_u32(v_n_0, vtwo);
        mu0    = vorrq_s32(mu0, vone);
        mu0    = vshlq_u32(mu0, vshift);
        mu0    = vorrq_u32(mu0, vshlq_n_u32(v_n_0, 31));
        mu0    = vandq_u32(mu0, sig0);

        vmask1 = vsubq_u32(vshlq_u32(vone, m_n_1), vone);
        msvec  = MagSgn.fetch(m_n_1);
        v_n_1  = vandq_u32(msvec, vmask1);
        vtmp   = vandq_s32(vtstq_s32(vdupq_n_u32(emb_1_1), vm), vone);
        v_n_1  = vorrq_u32(v_n_1, vshlq_u32(vtmp, m_n_1));
        mu1    = vaddq_u32(v_n_1, vtwo);
        mu1    = vorrq_s32(mu1, vone);
        mu1    = vshlq_u32(mu1, vshift);
        mu1    = vorrq_u32(mu1, vshlq_n_u32(v_n_1, 31));
        mu1    = vandq_u32(mu1, sig1);

        if constexpr (fuse_dequant) {
          dequant_store_neon(mp0, vuzp1q_s32(mu0, mu1), block->transformation, pLSB_dq,
                             vfscale_dq, vmagmask_dq, vsignmask_dq);
          dequant_store_neon(mp1, vuzp2q_s32(mu0, mu1), block->transformation, pLSB_dq,
                             vfscale_dq, vmagmask_dq, vsignmask_dq);
        } else {
          vst1q_s32(mp0, vuzp1q_s32(mu0, mu1));
          vst1q_s32(mp1, vuzp2q_s32(mu0, mu1));
        }
        mp0 += 4;
        mp1 += 4;

        vEmax = max4_pair(vld1q_s32(E_p + 3), vld1q_s32(E_p + 5));

        vExp = vsubq_s32(vdupq_n_s32(32), vclzq_s32(vuzp2q_s32(v_n_0, v_n_1)));
        vst1q_s32(E_p, vExp);
        E_p += 4;
      }
    }
  }  // Non-Initial line-pair end
}  // Cleanup decoding end

// Pack block_states SIGMA bits into nibble-packed uint16_t sigma array.
// Layout: sigma[qy * mstr + qx] holds 16 bits for the 4×4 block at (qy*4, qx*4).
// Nibbles: bits[0:3] = column 0, bits[4:7] = column 1, bits[8:11] = column 2, bits[12:15] = column 3.
// Within each nibble: bit 0 = row 0, bit 1 = row 1, bit 2 = row 2, bit 3 = row 3.
static void pack_sigma(const uint8_t *states, size_t bstride, uint32_t width, uint32_t height,
                       uint16_t *sigma, uint32_t mstr) {
  const uint32_t qw = (width + 3) >> 2;
  const uint32_t qh = (height + 3) >> 2;
  for (uint32_t qy = 0; qy < qh; qy++) {
    const uint32_t y0 = qy * 4;
    const uint32_t sh = (height - y0 < 4) ? (height - y0) : 4;
    for (uint32_t qx = 0; qx < qw; qx++) {
      const uint32_t x0 = qx * 4;
      const uint32_t sw = (width - x0 < 4) ? (width - x0) : 4;
      uint16_t s        = 0;
      for (uint32_t col = 0; col < sw; col++) {
        const uint8_t *p = states + (y0 + 1) * bstride + (x0 + col + 1);
        for (uint32_t row = 0; row < sh; row++) {
          s |= static_cast<uint16_t>((p[0] & 1) << (col * 4 + row));
          p += bstride;
        }
      }
      sigma[qy * mstr + qx] = s;
    }
    sigma[qy * mstr + qw] = 0;
  }
  for (uint32_t qx = 0; qx <= qw; qx++) {
    sigma[qh * mstr + qx] = 0;
  }
}

void ht_sigprop_decode(j2k_codeblock *block, uint8_t *HT_magref_segment, uint32_t magref_length,
                       const uint8_t &pLSB, uint16_t *sigma, uint32_t mstr) {
  SP_dec SigProp(HT_magref_segment, magref_length);
  const uint32_t height       = block->size.y;
  const uint32_t width        = block->size.x;
  const size_t sstride        = block->blksampl_stride;
  int32_t *samples            = block->sample_buf;
  const bool non_causal       = (block->Cmodes & CAUSAL) == 0;
  const int32_t spp_mask      = 3 << (pLSB - 1);
  uint16_t prev_row_sig[264];
  memset(prev_row_sig, 0, sizeof(prev_row_sig));

  for (uint32_t y = 0; y < height; y += 4) {
    const uint32_t sh = (height - y < 4) ? (height - y) : 4;
    uint32_t pattern  = 0xFFFFu;
    if (sh < 4) pattern = (sh == 3) ? 0x7777u : (sh == 2) ? 0x3333u : 0x1111u;

    uint32_t prev     = 0;
    uint16_t *cur_sig = sigma + (y >> 2) * mstr;

    for (uint32_t x = 0; x < width; x += 4) {
      const uint32_t qx = x >> 2;
      uint32_t pat      = pattern;
      int32_t excess    = static_cast<int32_t>(x + 4) - static_cast<int32_t>(width);
      if (excess > 0) pat >>= (excess * 4);

      uint32_t cs = *reinterpret_cast<const uint32_t *>(cur_sig + qx);
      uint32_t ps = *reinterpret_cast<const uint32_t *>(prev_row_sig + qx);
      uint32_t ns = *reinterpret_cast<const uint32_t *>(cur_sig + mstr + qx);

      uint32_t u = (ps & 0x88888888u) >> 3;
      if (non_causal) u |= (ns & 0x11111111u) << 3;

      uint32_t mbr = cs;
      mbr |= (cs & 0x77777777u) << 1;
      mbr |= (cs & 0xEEEEEEEEu) >> 1;
      mbr |= u;
      uint32_t t = mbr;
      mbr |= t << 4;
      mbr |= t >> 4;
      mbr |= prev >> 12;

      mbr &= pat;
      mbr &= ~cs;

      uint32_t new_sig = 0;
      if (mbr) {
        static const uint32_t row_masks[4] = {0x33u, 0x76u, 0xECu, 0xC8u};
        uint32_t inv_sig = ~cs & pat;
        while (mbr) {
          uint32_t pos      = static_cast<uint32_t>(openhtj2k_ctz32(mbr));
          uint32_t smask    = 1u << pos;
          mbr &= ~smask;
          uint32_t bit      = SigProp.importSigPropBit();
          uint32_t bit_mask = static_cast<uint32_t>(-static_cast<int32_t>(bit));
          new_sig |= smask & bit_mask;
          uint32_t neighbor = row_masks[pos & 3] << (pos & ~3u);
          mbr |= neighbor & inv_sig & ~new_sig & bit_mask;
        }

        if (new_sig) {
          uint32_t bits = new_sig;
          while (bits) {
            uint32_t pos = static_cast<uint32_t>(openhtj2k_ctz32(bits));
            bits &= bits - 1;
            samples[(y + (pos & 3)) * sstride + (x + (pos >> 2))] |= spp_mask;
          }
          bits = new_sig;
          while (bits) {
            uint32_t pos = static_cast<uint32_t>(openhtj2k_ctz32(bits));
            bits &= bits - 1;
            samples[(y + (pos & 3)) * sstride + (x + (pos >> 2))] |=
                static_cast<int32_t>(SigProp.importSigPropBit()) << 31;
          }
        }
      }

      new_sig |= cs & 0xFFFFu;
      prev_row_sig[qx] = static_cast<uint16_t>(new_sig);

      t = new_sig;
      new_sig |= (t & 0x7777u) << 1;
      new_sig |= (t & 0xEEEEu) >> 1;
      prev = (new_sig | u) & 0xF000u;
    }
  }
}

void ht_magref_decode(j2k_codeblock *block, uint8_t *HT_magref_segment, uint32_t magref_length,
                      const uint8_t &pLSB, const uint16_t *sigma, uint32_t mstr) {
  if (pLSB == 0) return;
  MR_dec MagRef(HT_magref_segment, magref_length);
  const uint32_t height = block->size.y;
  const uint32_t width  = block->size.x;
  const size_t sstride  = block->blksampl_stride;
  int32_t *samples      = block->sample_buf;

  for (uint32_t y = 0; y < height; y += 4) {
    const uint16_t *csig = sigma + (y >> 2) * mstr;

    for (uint32_t x = 0; x < width; x += 4) {
      uint16_t sig = csig[x >> 2];
      if (!sig) continue;

      uint32_t s = sig;
      while (s) {
        uint32_t pos = static_cast<uint32_t>(openhtj2k_ctz32(s));
        s &= s - 1;
        int32_t *sp  = samples + (y + (pos & 3)) * sstride + (x + (pos >> 2));
        int32_t bit  = MagRef.importMagRefBit();
        int32_t mask = static_cast<int32_t>(0xFFFFFFFE | static_cast<unsigned int>(bit));
        mask <<= pLSB;
        sp[0] &= mask;
        sp[0] |= 1 << (pLSB - 1);
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
  int32x4_t v0, v1, s0, s1, vROImask, vmagmask, vdst0, vdst1, vpLSB;
  vpLSB    = vdupq_n_s32(pLSB);
  vmagmask = vdupq_n_s32(INT32_MAX);
  if (this->transformation == 1) {
    // lossless path
    for (size_t i = 0; i < static_cast<size_t>(this->size.y); i++) {
      int32_t *val = this->sample_buf + i * this->blksampl_stride;
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
        vROImask = vandq_s32(vROImask, vROIshift);
        v0       = vshlq_s32(v0, vsubq_s32(vROImask, vpLSB));
        vROImask = vandq_s32(v1, vmask);
        vROImask = vceqzq_s32(vROImask);
        vROImask = vandq_s32(vROImask, vROIshift);
        v1       = vshlq_s32(v1, vsubq_s32(vROImask, vpLSB));
        // convert values from sign-magnitude form to two's complement one
        vdst0 = vbslq_s32(vreinterpretq_u32_s32(s0), vnegq_s32(v0), v0);
        vdst1 = vbslq_s32(vreinterpretq_u32_s32(s1), vnegq_s32(v1), v1);
        // vst1q_s16(dst, vcombine_s16(vmovn_s32(vdst0), vmovn_s32(vdst1)));
        vst1q_f32(dst, vcvtq_f32_s32(vdst0));
        vst1q_f32(dst + 4, vcvtq_f32_s32(vdst1));
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
        *dst = static_cast<float>(*val);
        val++;
        dst++;
      }
    }
  } else {
    // lossy path: compute the direct float scale factor.
    // decoded magnitude is in Q(31-M_b) fixed-point; result must be in Q(FRACBITS)
    float fscale_direct = this->stepsize;
    fscale_direct *= static_cast<float>(1 << FRACBITS);
    if (M_b <= 31)
      fscale_direct /= static_cast<float>(1 << (31 - M_b));
    else
      fscale_direct *= static_cast<float>(1 << (M_b - 31));

    if (ROIshift == 0) {
      // Common case: no ROI — direct float multiply, sign via XOR.
      // Eliminates integer truncate→mul→shift pipeline: saves ~5 ops per 4 elements.
      const float32x4_t vfscale  = vdupq_n_f32(fscale_direct);
      const int32x4_t vsignmask  = vdupq_n_s32(INT32_MIN);
      for (size_t i = 0; i < static_cast<size_t>(this->size.y); i++) {
        int32_t *val = this->sample_buf + i * this->blksampl_stride;
        sprec_t *dst = this->i_samples + i * this->band_stride;
        size_t len   = this->size.x;
        // 2× unrolled: 8 elements per iteration for better ILP.
        for (; len >= 8; len -= 8) {
          int32x4_t a0   = vld1q_s32(val);
          int32x4_t a1   = vld1q_s32(val + 4);
          int32x4_t m0   = vandq_s32(a0, vmagmask);
          int32x4_t m1   = vandq_s32(a1, vmagmask);
          float32x4_t f0 = vmulq_f32(vcvtq_f32_s32(m0), vfscale);
          float32x4_t f1 = vmulq_f32(vcvtq_f32_s32(m1), vfscale);
          // XOR sign bit from input integer into float result.
          f0 = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(f0),
                                               vreinterpretq_u32_s32(vandq_s32(a0, vsignmask))));
          f1 = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(f1),
                                               vreinterpretq_u32_s32(vandq_s32(a1, vsignmask))));
          vst1q_f32(dst, f0);
          vst1q_f32(dst + 4, f1);
          val += 8;
          dst += 8;
        }
        for (; len > 0; --len) {
          int32_t sign = *val & INT32_MIN;
          float f      = static_cast<float>(*val & INT32_MAX) * fscale_direct;
          if (sign) f  = -f;
          *dst++ = f;
          val++;
        }
      }
    } else {
      // ROI path — rarely used; keep integer-arithmetic approach for correctness.
      float fscale = fscale_direct;
      constexpr int32_t downshift = 15;
      fscale *= static_cast<float>(1 << 16) * static_cast<float>(1 << downshift);
      const auto scale = static_cast<int32_t>(fscale + 0.5f);
      for (size_t i = 0; i < static_cast<size_t>(this->size.y); i++) {
        int32_t *val = this->sample_buf + i * this->blksampl_stride;
        sprec_t *dst = this->i_samples + i * this->band_stride;
        size_t len   = this->size.x;
        for (; len >= 8; len -= 8) {
          v0 = vld1q_s32(val);
          v1 = vld1q_s32(val + 4);
          s0 = vshrq_n_s32(v0, 31);
          s1 = vshrq_n_s32(v1, 31);
          v0 = vandq_s32(v0, vmagmask);
          v1 = vandq_s32(v1, vmagmask);
          // upshift background region
          vROImask = vandq_s32(v0, vmask);
          vROImask = vceqzq_s32(vROImask);
          vROImask = vandq_s32(vROImask, vROIshift);
          v0       = vshlq_s32(v0, vROImask);
          vROImask = vandq_s32(v1, vmask);
          vROImask = vceqzq_s32(vROImask);
          vROImask = vandq_s32(vROImask, vROIshift);
          v1       = vshlq_s32(v1, vROImask);
          // truncate to int16 range
          v0 = vrshrq_n_s32(v0, 16);
          v1 = vrshrq_n_s32(v1, 16);
          // dequantization
          v0 = vmulq_s32(v0, vdupq_n_s32(scale));
          v1 = vmulq_s32(v1, vdupq_n_s32(scale));
          // downshift and sign
          v0    = vrshrq_n_s32(v0, downshift);
          v1    = vrshrq_n_s32(v1, downshift);
          vdst0 = vbslq_s32(vreinterpretq_u32_s32(s0), vnegq_s32(v0), v0);
          vdst1 = vbslq_s32(vreinterpretq_u32_s32(s1), vnegq_s32(v1), v1);
          vst1q_f32(dst, vcvtq_f32_s32(vdst0));
          vst1q_f32(dst + 4, vcvtq_f32_s32(vdst1));
          val += 8;
          dst += 8;
        }
        for (; len > 0; --len) {
          int32_t sign = *val & INT32_MIN;
          *val &= INT32_MAX;
          if (((uint32_t)*val & ~mask) == 0) *val <<= ROIshift;
          *val = (*val + (1 << 15)) >> 16;
          *val *= scale;
          *val = static_cast<int32_t>((*val + (1 << (downshift - 1))) >> downshift);
          if (sign) *val = -(*val & INT32_MAX);
          *dst = static_cast<float>(*val);
          val++;
          dst++;
        }
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
    uint8_t  all_segments[4];
    uint32_t num_segments = 0;
    for (uint32_t i = 0; i < block->pass_length_count; i++) {
      if (block->pass_length[i] != 0) {
        all_segments[num_segments++] = static_cast<uint8_t>(i);
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

    for (uint32_t i = 1; i < num_segments; i++) {
      Lref += block->pass_length[all_segments[i]];
    }
    if (block->num_passes > 1 && num_segments > 1) {
      Dref = block->get_compressed_data() + Lcup;
    } else {
      Dref = nullptr;
    }

    // Single HT pass with no ROI: use fused dequantize path to eliminate
    // the separate dequantize pass over sample_buf.
    bool dequant_done = false;
    // Fused dequant gate: NEON stores write 4 elements (128-bit); when block width
    // is not a multiple of 4, the overshoot corrupts adjacent blocks in parallel decode.
    if (num_ht_passes == 1 && ROIshift == 0 && (block->size.x & 3) == 0) {
      ht_cleanup_decode<true, true>(block, static_cast<uint8_t>(30 - S_blk), Lcup, Pcup, Scup);
      dequant_done = true;
    } else if (num_ht_passes == 1) {
      ht_cleanup_decode<true>(block, static_cast<uint8_t>(30 - S_blk), Lcup, Pcup, Scup);
    } else {
      ht_cleanup_decode<false>(block, static_cast<uint8_t>(30 - S_blk), Lcup, Pcup, Scup);

      const uint32_t qw   = (block->size.x + 3) >> 2;
      const uint32_t mstr = ((qw + 2) + 7u) & ~7u;
      uint16_t sigma_buf[(17 + 1) * 24] = {};
      pack_sigma(block->block_states, block->blkstate_stride, block->size.x, block->size.y,
                 sigma_buf, mstr);
      ht_sigprop_decode(block, Dref, Lref, static_cast<uint8_t>(30 - (S_blk + 1)), sigma_buf, mstr);
      if (num_ht_passes > 2) {
        ht_magref_decode(block, Dref, Lref, static_cast<uint8_t>(30 - (S_blk + 1)), sigma_buf, mstr);
      }
    }

    // dequantization (skipped when already fused into MagSgn output)
    if (!dequant_done) {
      block->dequantize(ROIshift);
    }

  }  // end

  return true;
}
#endif
