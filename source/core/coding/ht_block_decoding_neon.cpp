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
  #include "block_decoding.hpp"
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
template <bool StoreI32 = false>
static FORCE_INLINE void dequant_store_neon(int32_t *dst, int32x4_t val, uint8_t transformation,
                                            int32_t pLSB_dq, float32x4_t vfscale, int32x4_t vmagmask,
                                            int32x4_t vsignmask) {
  if (transformation == 1) {
    int32x4_t mag     = vandq_s32(val, vmagmask);
    int32x4_t shifted = vshlq_s32(mag, vdupq_n_s32(-pLSB_dq));
    uint32x4_t neg    = vreinterpretq_u32_s32(vshrq_n_s32(val, 31));
    int32x4_t res     = vbslq_s32(neg, vnegq_s32(shifted), shifted);
    if constexpr (StoreI32)
      vst1q_s32(dst, res);
    else
      vst1q_f32(reinterpret_cast<float *>(dst), vcvtq_f32_s32(res));
  } else {
    int32x4_t mag = vandq_s32(val, vmagmask);
    float32x4_t f = vmulq_f32(vcvtq_f32_s32(mag), vfscale);
    f             = vreinterpretq_f32_s32(veorq_s32(vreinterpretq_s32_f32(f), vandq_s32(val, vsignmask)));
    vst1q_f32(reinterpret_cast<float *>(dst), f);
  }
}

// Step-2 of the HT cleanup pass: MagSgn decoding over the (tv, u) scratch
// written by ht_cleanup_step1_nway (the former phase 1 of the fused kernel).
// Kept per-block: the fwd_buf destuff scratch is thread-local (constructing a
// second fwd_buf on the same thread invalidates the first), and step-2 is
// throughput-bound — there is nothing to gain from interleaving it.
template <bool fuse_dequant = false, bool store_i32 = false>
static void ht_cleanup_step2(j2k_codeblock *block, const uint8_t pLSB, const int32_t Pcup,
                             uint16_t *scratch, const int32_t sstr) {
  fwd_buf<0xFF> MagSgn(block->get_compressed_data(), Pcup);

  const uint16_t QW = static_cast<uint16_t>(ceil_int(static_cast<int16_t>(block->size.x), 2));
  const uint16_t QH = static_cast<uint16_t>(ceil_int(static_cast<int16_t>(block->size.y), 2));

  int32x4_t vExp;
  const int32_t mask[4]  = {1, 2, 4, 8};
  const int32x4_t vm     = vld1q_s32(mask);
  const int32x4_t vone   = vdupq_n_s32(1);
  const int32x4_t vtwo   = vdupq_n_s32(2);
  const int32x4_t vshift = vdupq_n_s32(pLSB - 1);

  // Fused dequantize setup: when fuse_dequant is true, we write dequantized float values
  // directly to band_buf, eliminating the separate dequantize pass.
  // pLSB_dq is the dequantization shift (31 - M_b), distinct from the MagSgn pLSB.
  int32_t pLSB_dq        = 0;
  float32x4_t vfscale_dq = vdupq_n_f32(0.0f);
  int32x4_t vmagmask_dq  = vdupq_n_s32(0);
  int32x4_t vsignmask_dq = vdupq_n_s32(0);
  if constexpr (fuse_dequant) {
    const int32_t M_b_val = block->get_Mb();
    pLSB_dq               = 31 - M_b_val;
    vmagmask_dq           = vdupq_n_s32(0x7FFFFFFF);
    vsignmask_dq          = vdupq_n_s32(INT32_MIN);
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
  int32_t *mp0              = fuse_dequant ? reinterpret_cast<int32_t *>(block->band_buf) : sample_buf;
  int32_t *mp1              = mp0 + (fuse_dequant ? block->band_stride : block->blksampl_stride);

  uint32_t rho0, rho1;
  uint32_t emb_k_0, emb_k_1;
  uint32_t emb_1_0, emb_1_1;
  uint32_t u0, u1;
  uint32_t U0, U1;

  alignas(32) int32_t Eline[1032];  // 2 * QW_max + 8, QW_max = 512
  std::memset(Eline, 0, (2U * QW + 8U) * sizeof(int32_t));
  int32_t *E_p = Eline + 1;

  int32_t qx;
  // Pre-load NEON constant vectors once for the entire codeblock.
  const typename fwd_buf<0xFF>::DecodeConstants dc;
  uint16_t *sp = scratch;
  // Initial line-pair
  for (qx = QW; qx > 0; qx -= 2, sp += 4) {
    const uint16_t tv0 = sp[0];
    const uint16_t tv1 = sp[2];
    rho0               = (tv0 & 0x00F0) >> 4;
    emb_k_0            = (tv0 & 0xF000) >> 12;
    emb_1_0            = (tv0 & 0x0F00) >> 8;
    rho1               = (tv1 & 0x00F0) >> 4;
    emb_k_1            = (tv1 & 0xF000) >> 12;
    emb_1_1            = (tv1 & 0x0F00) >> 8;

    // step-1 already folded kappa (always 1 in the initial line-pair) into u
    U0 = sp[1];
    U1 = sp[3];

    if (pLSB > 16) {
      // 16-bit fast path: batch bit extraction via vqtbl1q_u8
      const uint8_t pLSB_adj = pLSB - 16;
      int16x4_t vn_16        = vdup_n_s16(0);
      int16x8_t row16 = MagSgn.decode_two_quads_16bit(tv0, tv1, static_cast<uint16_t>(U0),
                                                      static_cast<uint16_t>(U1), pLSB_adj, vn_16, dc);
      // Deinterleave row0/row1 and expand int16 -> int32 (sign bit 15 -> bit 31).
      int16x4_t lo      = vget_low_s16(row16);
      int16x4_t hi      = vget_high_s16(row16);
      int16x4_t row0_16 = vuzp1_s16(lo, hi);
      int16x4_t row1_16 = vuzp2_s16(lo, hi);
      if constexpr (fuse_dequant) {
        dequant_store_neon<store_i32>(mp0, vshll_n_s16(row0_16, 16), block->transformation, pLSB_dq,
                                      vfscale_dq, vmagmask_dq, vsignmask_dq);
        dequant_store_neon<store_i32>(mp1, vshll_n_s16(row1_16, 16), block->transformation, pLSB_dq,
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

      sig0  = vdupq_n_u32(rho0);
      sig0  = vtstq_s32(sig0, vm);
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
        dequant_store_neon<store_i32>(mp0, vuzp1q_s32(mu0, mu1), block->transformation, pLSB_dq, vfscale_dq,
                                      vmagmask_dq, vsignmask_dq);
        dequant_store_neon<store_i32>(mp1, vuzp2q_s32(mu0, mu1), block->transformation, pLSB_dq, vfscale_dq,
                                      vmagmask_dq, vsignmask_dq);
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
    E_p = Eline + 1;
    if constexpr (fuse_dequant) {
      mp0 = reinterpret_cast<int32_t *>(block->band_buf) + (row * 2U) * block->band_stride;
      mp1 = mp0 + block->band_stride;
    } else {
      mp0 = sample_buf + (row * 2U) * block->blksampl_stride;
      mp1 = sample_buf + (row * 2U + 1U) * block->blksampl_stride;
    }
    sp = scratch + row * static_cast<uint32_t>(sstr);

    // Pre-compute Emax for first 4 quads (vectorized sliding max).
    int32x4_t vEmax4;
    {
      int32x2_t m01 = vpmax_s32(vget_low_s32(vmaxq_s32(vld1q_s32(E_p - 1), vld1q_s32(E_p + 1))),
                                vget_high_s32(vmaxq_s32(vld1q_s32(E_p - 1), vld1q_s32(E_p + 1))));
      int32x2_t m23 = vpmax_s32(vget_low_s32(vmaxq_s32(vld1q_s32(E_p + 3), vld1q_s32(E_p + 5))),
                                vget_high_s32(vmaxq_s32(vld1q_s32(E_p + 3), vld1q_s32(E_p + 5))));
      vEmax4        = vcombine_s32(m01, m23);
    }

    for (qx = QW; qx > 0; qx -= 2, sp += 4) {
      const uint16_t tv0 = sp[0];
      const uint16_t tv1 = sp[2];
      rho0               = (tv0 & 0x00F0) >> 4;
      emb_k_0            = (tv0 & 0xF000) >> 12;
      emb_1_0            = (tv0 & 0x0F00) >> 8;
      rho1               = (tv1 & 0x00F0) >> 4;
      emb_k_1            = (tv1 & 0xF000) >> 12;
      emb_1_1            = (tv1 & 0x0F00) >> 8;
      u0                 = sp[1];
      u1                 = sp[3];

      {
        // gamma: 0 if popcount(rho) < 2 (single bit or zero), else 1
        // Use vEmax4[0..1] for this quad-pair (consumed in order).
        int32x2_t vEmax   = vget_low_s32(vEmax4);
        int32x2_t vrho    = vset_lane_s32((int32_t)rho1, vdup_n_s32((int32_t)rho0), 1);
        int32x2_t vrho_m1 = vsub_s32(vrho, vdup_n_s32(1));
        uint32x2_t vz     = vceq_u32(vreinterpret_u32_s32(vand_s32(vrho, vrho_m1)), vdup_n_u32(0));
        int32x2_t vgamma  = vreinterpret_s32_u32(vbic_u32(vdup_n_u32(1), vz));
        int32x2_t vkappa  = vmax_s32(vmul_s32(vgamma, vsub_s32(vEmax, vdup_n_s32(1))), vdup_n_s32(1));
        // Store kappa to stack to avoid vget_lane cross-pipeline penalty (~4c each).
        int32_t kappa_arr[2];
        vst1_s32(kappa_arr, vkappa);
        U0 = static_cast<uint32_t>(kappa_arr[0]) + u0;
        U1 = static_cast<uint32_t>(kappa_arr[1]) + u1;
        // Shift vEmax4: next quad-pair will use [2..3] via vget_low after shift.
        vEmax4 = vextq_s32(vEmax4, vEmax4, 2);
      }

      if (pLSB > 16) {
        // 16-bit fast path: batch bit extraction via vqtbl1q_u8
        const uint8_t pLSB_adj = pLSB - 16;
        int16x4_t vn_16        = vdup_n_s16(0);
        int16x8_t row16   = MagSgn.decode_two_quads_16bit(tv0, tv1, static_cast<uint16_t>(U0),
                                                          static_cast<uint16_t>(U1), pLSB_adj, vn_16, dc);
        int16x4_t lo      = vget_low_s16(row16);
        int16x4_t hi      = vget_high_s16(row16);
        int16x4_t row0_16 = vuzp1_s16(lo, hi);
        int16x4_t row1_16 = vuzp2_s16(lo, hi);
        if constexpr (fuse_dequant) {
          dequant_store_neon<store_i32>(mp0, vshll_n_s16(row0_16, 16), block->transformation, pLSB_dq,
                                        vfscale_dq, vmagmask_dq, vsignmask_dq);
          dequant_store_neon<store_i32>(mp1, vshll_n_s16(row1_16, 16), block->transformation, pLSB_dq,
                                        vfscale_dq, vmagmask_dq, vsignmask_dq);
        } else {
          vst1q_s32(mp0, vshll_n_s16(row0_16, 16));
          vst1q_s32(mp1, vshll_n_s16(row1_16, 16));
        }
        mp0 += 4;
        mp1 += 4;

        // Read-ahead Emax: reload 4-quad Emax every other iteration (after the
        // shifted vEmax4's high pair was consumed).  The shift in the kappa block
        // moved [2..3]→[0..1]; after this iteration consumes [0..1], both pairs
        // are spent.  Reload from E_p+3 (which still holds the previous row's values).
        {
          int32x2_t nm01 = vpmax_s32(vget_low_s32(vmaxq_s32(vld1q_s32(E_p + 3), vld1q_s32(E_p + 5))),
                                     vget_high_s32(vmaxq_s32(vld1q_s32(E_p + 3), vld1q_s32(E_p + 5))));
          int32x2_t nm23 = vpmax_s32(vget_low_s32(vmaxq_s32(vld1q_s32(E_p + 7), vld1q_s32(E_p + 9))),
                                     vget_high_s32(vmaxq_s32(vld1q_s32(E_p + 7), vld1q_s32(E_p + 9))));
          vEmax4         = vcombine_s32(nm01, nm23);
        }

        int32x4_t vn32 = vreinterpretq_s32_u32(vmovl_u16(vreinterpret_u16_s16(vn_16)));
        vExp           = vsubq_s32(vdupq_n_s32(32), vclzq_s32(vn32));
        vst1q_s32(E_p, vExp);
        E_p += 4;
      } else {
        // Existing 32-bit path
        int32x4_t vmask1, sig0, sig1, vtmp, m_n_0, m_n_1, msvec, v_n_0, v_n_1, mu0, mu1;

        sig0  = vdupq_n_u32(rho0);
        sig0  = vtstq_s32(sig0, vm);
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
          dequant_store_neon<store_i32>(mp0, vuzp1q_s32(mu0, mu1), block->transformation, pLSB_dq,
                                        vfscale_dq, vmagmask_dq, vsignmask_dq);
          dequant_store_neon<store_i32>(mp1, vuzp2q_s32(mu0, mu1), block->transformation, pLSB_dq,
                                        vfscale_dq, vmagmask_dq, vsignmask_dq);
        } else {
          vst1q_s32(mp0, vuzp1q_s32(mu0, mu1));
          vst1q_s32(mp1, vuzp2q_s32(mu0, mu1));
        }
        mp0 += 4;
        mp1 += 4;

        {
          int32x2_t nm01 = vpmax_s32(vget_low_s32(vmaxq_s32(vld1q_s32(E_p + 3), vld1q_s32(E_p + 5))),
                                     vget_high_s32(vmaxq_s32(vld1q_s32(E_p + 3), vld1q_s32(E_p + 5))));
          int32x2_t nm23 = vpmax_s32(vget_low_s32(vmaxq_s32(vld1q_s32(E_p + 7), vld1q_s32(E_p + 9))),
                                     vget_high_s32(vmaxq_s32(vld1q_s32(E_p + 7), vld1q_s32(E_p + 9))));
          vEmax4         = vcombine_s32(nm01, nm23);
        }

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
  if (pLSB == 0) return;  // no plane below the LSB; mirrors ht_magref_decode (avoids 1 << (pLSB-1) UB)
  SP_dec SigProp(HT_magref_segment, magref_length);
  const uint32_t height  = block->size.y;
  const uint32_t width   = block->size.x;
  const size_t sstride   = block->blksampl_stride;
  int32_t *samples       = block->sample_buf;
  const bool non_causal  = (block->Cmodes & CAUSAL) == 0;
  const int32_t spp_mask = 3 << (pLSB - 1);
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
        uint32_t inv_sig                   = ~cs & pat;
        while (mbr) {
          uint32_t pos   = static_cast<uint32_t>(openhtj2k_ctz32(mbr));
          uint32_t smask = 1u << pos;
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
    auto rev_dequant_neon = [&](auto simd_store, auto scalar_store) {
      for (size_t i = 0; i < static_cast<size_t>(this->size.y); i++) {
        int32_t *val = this->sample_buf + i * this->blksampl_stride;
        sprec_t *dst = this->band_buf + i * this->band_stride;
        size_t len   = this->size.x;
        for (; len >= 8; len -= 8) {
          v0       = vld1q_s32(val);
          v1       = vld1q_s32(val + 4);
          s0       = vshrq_n_s32(v0, 31);
          s1       = vshrq_n_s32(v1, 31);
          v0       = vandq_s32(v0, vmagmask);
          v1       = vandq_s32(v1, vmagmask);
          vROImask = vandq_s32(v0, vmask);
          vROImask = vceqzq_s32(vROImask);
          vROImask = vandq_s32(vROImask, vROIshift);
          v0       = vshlq_s32(v0, vsubq_s32(vROImask, vpLSB));
          vROImask = vandq_s32(v1, vmask);
          vROImask = vceqzq_s32(vROImask);
          vROImask = vandq_s32(vROImask, vROIshift);
          v1       = vshlq_s32(v1, vsubq_s32(vROImask, vpLSB));
          vdst0    = vbslq_s32(vreinterpretq_u32_s32(s0), vnegq_s32(v0), v0);
          vdst1    = vbslq_s32(vreinterpretq_u32_s32(s1), vnegq_s32(v1), v1);
          simd_store(dst, vdst0, vdst1);
          val += 8;
          dst += 8;
        }
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
          scalar_store(dst, *val);
          val++;
          dst++;
        }
      }
    };
    if (this->dequant_i32) {
      rev_dequant_neon(
          [](sprec_t *d, int32x4_t a, int32x4_t b) {
            vst1q_s32(reinterpret_cast<int32_t *>(d), a);
            vst1q_s32(reinterpret_cast<int32_t *>(d + 4), b);
          },
          [](sprec_t *d, int32_t v) { *reinterpret_cast<int32_t *>(d) = v; });
    } else {
      rev_dequant_neon(
          [](sprec_t *d, int32x4_t a, int32x4_t b) {
            vst1q_f32(d, vcvtq_f32_s32(a));
            vst1q_f32(d + 4, vcvtq_f32_s32(b));
          },
          [](sprec_t *d, int32_t v) { *d = static_cast<float>(v); });
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
      const float32x4_t vfscale = vdupq_n_f32(fscale_direct);
      const int32x4_t vsignmask = vdupq_n_s32(INT32_MIN);
      for (size_t i = 0; i < static_cast<size_t>(this->size.y); i++) {
        int32_t *val = this->sample_buf + i * this->blksampl_stride;
        sprec_t *dst = this->band_buf + i * this->band_stride;
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
          f0 = vreinterpretq_f32_u32(
              veorq_u32(vreinterpretq_u32_f32(f0), vreinterpretq_u32_s32(vandq_s32(a0, vsignmask))));
          f1 = vreinterpretq_f32_u32(
              veorq_u32(vreinterpretq_u32_f32(f1), vreinterpretq_u32_s32(vandq_s32(a1, vsignmask))));
          vst1q_f32(dst, f0);
          vst1q_f32(dst + 4, f1);
          val += 8;
          dst += 8;
        }
        for (; len > 0; --len) {
          int32_t sign = *val & INT32_MIN;
          float f      = static_cast<float>(*val & INT32_MAX) * fscale_direct;
          if (sign) f = -f;
          *dst++ = f;
          val++;
        }
      }
    } else {
      // ROI path — rarely used; keep integer-arithmetic approach for correctness.
      float fscale                = fscale_direct;
      constexpr int32_t downshift = 15;
      fscale *= static_cast<float>(1 << 16) * static_cast<float>(1 << downshift);
      const auto scale = static_cast<int32_t>(fscale + 0.5f);
      for (size_t i = 0; i < static_cast<size_t>(this->size.y); i++) {
        int32_t *val = this->sample_buf + i * this->blksampl_stride;
        sprec_t *dst = this->band_buf + i * this->band_stride;
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

  // Single HT pass with no ROI: use fused dequantize path to eliminate
  // the separate dequantize pass over sample_buf.
  bool dequant_done = false;
  // Fused dequant gate: NEON stores write 4 elements (128-bit); when block width
  // is not a multiple of 4, the overshoot corrupts adjacent blocks in parallel decode.
  // Also gate on even height: the kernel writes row-pairs unconditionally and odd
  // height overflows one row into the next block's region.
  if (su.num_ht_passes == 1 && ROIshift == 0 && (block->size.x & 3) == 0 && (block->size.y & 1u) == 0) {
    if (block->dequant_i32)
      ht_cleanup_step2<true, true>(block, pLSB, su.Pcup, scratch, sstr);
    else
      ht_cleanup_step2<true>(block, pLSB, su.Pcup, scratch, sstr);
    dequant_done = true;
  } else if (su.num_ht_passes == 1) {
    ht_cleanup_step2<>(block, pLSB, su.Pcup, scratch, sstr);
  } else {
    ht_cleanup_step2<>(block, pLSB, su.Pcup, scratch, sstr);

    // Pack block_states SIGMA bits into nibble-packed sigma array
    const uint32_t qw                 = (block->size.x + 3) >> 2;
    const uint32_t mstr               = ((qw + 2) + 7u) & ~7u;
    uint16_t sigma_buf[(17 + 1) * 24] = {};
    pack_sigma(block->block_states, block->blkstate_stride, block->size.x, block->size.y, sigma_buf, mstr);

    ht_sigprop_decode(block, su.Dref, su.Lref, static_cast<uint8_t>(30 - (su.S_blk + 1)), sigma_buf, mstr);
    if (su.num_ht_passes > 2) {
      ht_magref_decode(block, su.Dref, su.Lref, static_cast<uint8_t>(30 - (su.S_blk + 1)), sigma_buf, mstr);
    }
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
