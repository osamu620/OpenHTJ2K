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

#if defined(OPENHTJ2K_ENABLE_WASM_SIMD)
  #include "coding_units.hpp"
  #include "dec_CxtVLC_tables.hpp"
  #include "ht_block_decoding.hpp"
  #include "coding_local.hpp"
  #include "utils.hpp"
  #include <wasm_simd128.h>

/******************************************************************************/
// WASM SIMD helper functions
/******************************************************************************/
static inline v128_t wasm_i32x4_shlv(v128_t a, v128_t b) {
  return wasm_i32x4_make(
      (int32_t)((uint32_t)wasm_i32x4_extract_lane(a, 0) << wasm_i32x4_extract_lane(b, 0)),
      (int32_t)((uint32_t)wasm_i32x4_extract_lane(a, 1) << wasm_i32x4_extract_lane(b, 1)),
      (int32_t)((uint32_t)wasm_i32x4_extract_lane(a, 2) << wasm_i32x4_extract_lane(b, 2)),
      (int32_t)((uint32_t)wasm_i32x4_extract_lane(a, 3) << wasm_i32x4_extract_lane(b, 3)));
}

static inline v128_t wasm_i32x4_vshl(v128_t a, v128_t b) {
  auto shift_lane = [](int32_t val, int32_t n) -> int32_t {
    return n >= 0 ? val << n : val >> (-n);
  };
  return wasm_i32x4_make(
      shift_lane(wasm_i32x4_extract_lane(a, 0), wasm_i32x4_extract_lane(b, 0)),
      shift_lane(wasm_i32x4_extract_lane(a, 1), wasm_i32x4_extract_lane(b, 1)),
      shift_lane(wasm_i32x4_extract_lane(a, 2), wasm_i32x4_extract_lane(b, 2)),
      shift_lane(wasm_i32x4_extract_lane(a, 3), wasm_i32x4_extract_lane(b, 3)));
}

static inline int32_t wasm_i32x4_reduce_max(v128_t a) {
  v128_t t = wasm_i32x4_max(a, wasm_i32x4_shuffle(a, a, 2, 3, 0, 1));
  t        = wasm_i32x4_max(t, wasm_i32x4_shuffle(t, t, 1, 0, 3, 2));
  return wasm_i32x4_extract_lane(t, 0);
}

// Branchless vectorized CLZ via float exponent: clz32(a) = min(158 - (float_bits(a) >> 23), 32).
// Valid for non-negative int32 inputs; correctly returns 32 for a == 0.
static inline v128_t wasm_u32x4_clz(v128_t a) {
  v128_t vf  = wasm_f32x4_convert_i32x4(a);                             // int32 → float (non-negative)
  v128_t exp = wasm_u32x4_shr(vf, 23);                                  // extract biased exponent
  v128_t clz = wasm_i32x4_sub(wasm_i32x4_const_splat(158), exp);        // 158 - biased_exp = clz
  return wasm_i32x4_min(clz, wasm_i32x4_const_splat(32));               // clamp: handles a == 0
}

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
  return mbr & 1;
}

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

// WASM SIMD fused dequantize-and-store: 4 × int32 sign-magnitude → 4 × float.
static inline void dequant_store_wasm(int32_t *dst, v128_t val, uint8_t transformation, int32_t pLSB_dq,
                                      v128_t vfscale, v128_t vmagmask, v128_t vsignmask) {
  if (transformation == 1) {
    v128_t mag     = wasm_v128_and(val, vmagmask);
    v128_t shifted = wasm_i32x4_shr(mag, pLSB_dq);
    // Apply sign: negate where val is negative (sign bit set)
    v128_t neg     = wasm_i32x4_lt(val, wasm_i32x4_const_splat(0));
    v128_t negated = wasm_i32x4_sub(wasm_i32x4_const_splat(0), shifted);
    v128_t res     = wasm_v128_bitselect(negated, shifted, neg);
    wasm_v128_store(dst, wasm_f32x4_convert_i32x4(res));
  } else {
    v128_t mag = wasm_v128_and(val, vmagmask);
    v128_t f   = wasm_f32x4_mul(wasm_f32x4_convert_i32x4(mag), vfscale);
    // Apply sign via XOR with sign bit
    f = wasm_v128_xor(f, wasm_v128_and(val, vsignmask));
    wasm_v128_store(dst, f);
  }
}

template <bool skip_sigma, bool fuse_dequant = false>
void ht_cleanup_decode(j2k_codeblock *block, const uint8_t &pLSB, const int32_t Lcup, const int32_t Pcup,
                       const int32_t Scup) {
  fwd_buf<0xFF> MagSgn(block->get_compressed_data(), Pcup);
  MEL_dec MEL(block->get_compressed_data(), Lcup, Scup);
  rev_buf VLC_dec(block->get_compressed_data(), Lcup, Scup);

  const uint16_t QW = static_cast<uint16_t>(ceil_int(static_cast<int16_t>(block->size.x), 2));
  const uint16_t QH = static_cast<uint16_t>(ceil_int(static_cast<int16_t>(block->size.y), 2));

  // Fused dequantize setup
  int32_t pLSB_dq  = 0;
  v128_t vfscale_dq  = wasm_f32x4_const_splat(0.0f);
  v128_t vmagmask_dq = wasm_i32x4_const_splat(0);
  v128_t vsignmask_dq = wasm_i32x4_const_splat(0);
  if constexpr (fuse_dequant) {
    const int32_t M_b_val = block->get_Mb();
    pLSB_dq               = 31 - M_b_val;
    vmagmask_dq            = wasm_i32x4_const_splat(0x7FFFFFFF);
    vsignmask_dq           = wasm_i32x4_const_splat((int32_t)0x80000000u);
    if (block->transformation != 1) {
      float fscale_direct = block->stepsize;
      fscale_direct *= static_cast<float>(1 << FRACBITS);
      if (M_b_val <= 31)
        fscale_direct /= static_cast<float>(1 << (31 - M_b_val));
      else
        fscale_direct *= static_cast<float>(1 << (M_b_val - 31));
      vfscale_dq = wasm_f32x4_splat(fscale_direct);
    }
  }

  v128_t vExp;
  const v128_t vm     = wasm_i32x4_const(1, 2, 4, 8);
  const v128_t vone   = wasm_i32x4_const_splat(1);
  const v128_t vtwo   = wasm_i32x4_const_splat(2);
  const v128_t vshift = wasm_i32x4_splat(pLSB - 1);

  auto mp0 = fuse_dequant ? reinterpret_cast<int32_t *>(block->i_samples) : block->sample_buf;
  auto mp1 = mp0 + (fuse_dequant ? block->band_stride : block->blksampl_stride);
  auto sp0 = block->block_states + 1 + block->blkstate_stride;
  auto sp1 = block->block_states + 1 + 2 * block->blkstate_stride;

  uint32_t rho0, rho1;
  uint32_t u_off0, u_off1;
  uint32_t emb_k_0, emb_k_1;
  uint32_t emb_1_0, emb_1_1;
  uint32_t u0, u1;
  uint32_t U0, U1;
  uint8_t gamma0, gamma1;
  uint32_t kappa0 = 1, kappa1 = 1;

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

    context = ((tv0 & 0xE0U) << 2) | ((tv0 & 0x10U) << 3);

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

    context = ((tv1 & 0xE0U) << 2) | ((tv1 & 0x10U) << 3);

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
    vlcval               = VLC_dec.advance(uvlc_result & 0x7);
    uvlc_result >>= 3;
    uint32_t len = uvlc_result & 0xF;
    uint32_t tmp = vlcval & ((1U << len) - 1U);
    VLC_dec.advance(len);
    uvlc_result >>= 4;
    len = uvlc_result & 0x7;
    uvlc_result >>= 3;
    u0 = (uvlc_result & 7) + (tmp & ~(0xFFU << len));
    u1 = (uvlc_result >> 3) + (tmp >> len);

    U0 = kappa0 + u0;
    U1 = kappa1 + u1;

    if (pLSB > 16) {
      // 16-bit fast path: batch bit extraction via wasm_i8x16_swizzle
      const uint8_t pLSB_adj = static_cast<uint8_t>(pLSB - 16);
      const v128_t zero16    = wasm_i32x4_const_splat(0);
      v128_t vn_16           = zero16;
      v128_t row16           = MagSgn.decode_two_quads_16bit_wasm(
          tv0, tv1, static_cast<uint16_t>(U0), static_cast<uint16_t>(U1), pLSB_adj, vn_16);
      // Deinterleave: even lanes → row0, odd lanes → row1
      v128_t row0_16 = wasm_i16x8_shuffle(row16, zero16, 0, 2, 4, 6, 8, 8, 8, 8);
      v128_t row1_16 = wasm_i16x8_shuffle(row16, zero16, 1, 3, 5, 7, 8, 8, 8, 8);
      // Expand int16 → int32 in sign-magnitude format (sign at bit 31)
      v128_t mu0_32  = wasm_i32x4_shl(wasm_i32x4_extend_low_i16x8(row0_16), 16);
      v128_t mu1_32  = wasm_i32x4_shl(wasm_i32x4_extend_low_i16x8(row1_16), 16);
      if constexpr (fuse_dequant) {
        dequant_store_wasm(mp0, mu0_32, block->transformation, pLSB_dq, vfscale_dq, vmagmask_dq, vsignmask_dq);
        dequant_store_wasm(mp1, mu1_32, block->transformation, pLSB_dq, vfscale_dq, vmagmask_dq, vsignmask_dq);
      } else {
        wasm_v128_store(mp0, mu0_32);
        wasm_v128_store(mp1, mu1_32);
      }
      mp0 += 4;
      mp1 += 4;
      // Update exponent: widen v_n int16 → uint32 then CLZ
      v128_t vn32 = wasm_u32x4_extend_low_u16x8(vn_16);
      vExp        = wasm_i32x4_sub(wasm_i32x4_const_splat(32), wasm_u32x4_clz(vn32));
      wasm_v128_store(E_p, vExp);
      E_p += 4;
    } else {
      // 32-bit path
      v128_t vmask1, sig0, sig1, vtmp, m_n_0, m_n_1, msvec, v_n_0, v_n_1, mu0, mu1;

      sig0  = wasm_i32x4_ne(wasm_v128_and(wasm_i32x4_splat((int32_t)rho0), vm), wasm_i32x4_const_splat(0));
      vtmp  = wasm_v128_and(wasm_i32x4_ne(wasm_v128_and(wasm_i32x4_splat((int32_t)emb_k_0), vm),
                                         wasm_i32x4_const_splat(0)),
                           vone);
      m_n_0 = wasm_i32x4_sub(wasm_v128_and(sig0, wasm_i32x4_splat((int32_t)U0)), vtmp);
      sig1  = wasm_i32x4_ne(wasm_v128_and(wasm_i32x4_splat((int32_t)rho1), vm), wasm_i32x4_const_splat(0));
      vtmp  = wasm_v128_and(wasm_i32x4_ne(wasm_v128_and(wasm_i32x4_splat((int32_t)emb_k_1), vm),
                                         wasm_i32x4_const_splat(0)),
                           vone);
      m_n_1 = wasm_i32x4_sub(wasm_v128_and(sig1, wasm_i32x4_splat((int32_t)U1)), vtmp);

      vmask1 = wasm_i32x4_make((1 << wasm_i32x4_extract_lane(m_n_0, 0)) - 1,
                               (1 << wasm_i32x4_extract_lane(m_n_0, 1)) - 1,
                               (1 << wasm_i32x4_extract_lane(m_n_0, 2)) - 1,
                               (1 << wasm_i32x4_extract_lane(m_n_0, 3)) - 1);
      msvec = MagSgn.fetch(m_n_0);
      v_n_0 = wasm_v128_and(msvec, vmask1);
      vtmp  = wasm_v128_and(wasm_i32x4_ne(wasm_v128_and(wasm_i32x4_splat((int32_t)emb_1_0), vm),
                                         wasm_i32x4_const_splat(0)),
                           vone);
      v_n_0 = wasm_v128_or(v_n_0, wasm_i32x4_shlv(vtmp, m_n_0));
      mu0   = wasm_i32x4_add(v_n_0, vtwo);
      mu0   = wasm_v128_or(mu0, vone);
      mu0   = wasm_i32x4_shlv(mu0, vshift);
      mu0   = wasm_v128_or(mu0, wasm_i32x4_shl(v_n_0, 31));
      mu0   = wasm_v128_and(mu0, sig0);

      vmask1 = wasm_i32x4_make((1 << wasm_i32x4_extract_lane(m_n_1, 0)) - 1,
                               (1 << wasm_i32x4_extract_lane(m_n_1, 1)) - 1,
                               (1 << wasm_i32x4_extract_lane(m_n_1, 2)) - 1,
                               (1 << wasm_i32x4_extract_lane(m_n_1, 3)) - 1);
      msvec = MagSgn.fetch(m_n_1);
      v_n_1 = wasm_v128_and(msvec, vmask1);
      vtmp  = wasm_v128_and(wasm_i32x4_ne(wasm_v128_and(wasm_i32x4_splat((int32_t)emb_1_1), vm),
                                         wasm_i32x4_const_splat(0)),
                           vone);
      v_n_1 = wasm_v128_or(v_n_1, wasm_i32x4_shlv(vtmp, m_n_1));
      mu1   = wasm_i32x4_add(v_n_1, vtwo);
      mu1   = wasm_v128_or(mu1, vone);
      mu1   = wasm_i32x4_shlv(mu1, vshift);
      mu1   = wasm_v128_or(mu1, wasm_i32x4_shl(v_n_1, 31));
      mu1   = wasm_v128_and(mu1, sig1);

      if constexpr (fuse_dequant) {
        dequant_store_wasm(mp0, wasm_i32x4_shuffle(mu0, mu1, 0, 2, 4, 6), block->transformation, pLSB_dq,
                           vfscale_dq, vmagmask_dq, vsignmask_dq);
        dequant_store_wasm(mp1, wasm_i32x4_shuffle(mu0, mu1, 1, 3, 5, 7), block->transformation, pLSB_dq,
                           vfscale_dq, vmagmask_dq, vsignmask_dq);
      } else {
        wasm_v128_store(mp0, wasm_i32x4_shuffle(mu0, mu1, 0, 2, 4, 6));
        wasm_v128_store(mp1, wasm_i32x4_shuffle(mu0, mu1, 1, 3, 5, 7));
      }
      mp0 += 4;
      mp1 += 4;
      // Update exponent
      vExp = wasm_i32x4_sub(wasm_i32x4_const_splat(32),
                            wasm_u32x4_clz(wasm_i32x4_shuffle(v_n_0, v_n_1, 1, 3, 5, 7)));
      wasm_v128_store(E_p, vExp);
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
      mp1 = reinterpret_cast<int32_t *>(block->i_samples) + (row * 2U + 1U) * block->band_stride;
    } else {
      mp0 = block->sample_buf + (row * 2U) * block->blksampl_stride;
      mp1 = block->sample_buf + (row * 2U + 1U) * block->blksampl_stride;
    }
    sp0   = block->block_states + (row * 2U + 1U) * block->blkstate_stride + 1U;
    sp1   = block->block_states + (row * 2U + 2U) * block->blkstate_stride + 1U;
    rho1  = 0;

    int32_t Emax0, Emax1;
    Emax0 = wasm_i32x4_reduce_max(wasm_v128_load(E_p - 1));
    Emax1 = wasm_i32x4_reduce_max(wasm_v128_load(E_p + 1));

    context = ((rho1 & 0x4) << 6) | ((rho1 & 0x8) << 5);
    context |= ((rho_p[-1] & 0x8) << 4) | ((rho_p[0] & 0x2) << 6);
    context |= ((rho_p[0] & 0x8) << 6) | ((rho_p[1] & 0x2) << 8);

    for (qx = QW; qx > 0; qx -= 2) {
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

      context = ((rho0 & 0x4) << 6) | ((rho0 & 0x8) << 5);
      context |= ((rho_p[0] & 0x8) << 4) | ((rho_p[1] & 0x2) << 6);
      context |= ((rho_p[1] & 0x8) << 6) | ((rho_p[2] & 0x2) << 8);

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

      context = ((rho1 & 0x4) << 6) | ((rho1 & 0x8) << 5);
      context |= ((rho_p[1] & 0x8) << 4) | ((rho_p[2] & 0x2) << 6);
      context |= ((rho_p[2] & 0x8) << 6) | ((rho_p[3] & 0x2) << 8);

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

      vlcval = VLC_dec.advance((tv1 & 0x000F) >> 1);

      u_off0          = tv0 & 1;
      u_off1          = tv1 & 1;
      uint32_t idx    = (vlcval & 0x3F) + (u_off0 << 6U) + (u_off1 << 7U);
      uint32_t uvlc_result = uvlc_dec_1[idx];
      vlcval          = VLC_dec.advance(uvlc_result & 0x7);
      uvlc_result >>= 3;
      uint32_t len    = uvlc_result & 0xF;
      uint32_t tmp    = vlcval & ((1U << len) - 1U);
      VLC_dec.advance(len);
      uvlc_result >>= 4;
      len = uvlc_result & 0x7;
      uvlc_result >>= 3;
      u0 = (uvlc_result & 7) + (tmp & ~(0xFFU << len));
      u1 = (uvlc_result >> 3) + (tmp >> len);

      gamma0 = ((rho0 & (rho0 - 1)) == 0) ? 0 : 1;
      gamma1 = ((rho1 & (rho1 - 1)) == 0) ? 0 : 1;
      kappa0 = (1 > gamma0 * (Emax0 - 1)) ? 1U : static_cast<uint32_t>(Emax0 - 1);
      kappa1 = (1 > gamma1 * (Emax1 - 1)) ? 1U : static_cast<uint32_t>(Emax1 - 1);
      U0     = kappa0 + u0;
      U1     = kappa1 + u1;

      if (pLSB > 16 && U0 + pLSB <= 30 && U1 + pLSB <= 30) {
        // 16-bit fast path: batch bit extraction via wasm_i8x16_swizzle
        // Guard: U + pLSB <= 30 ensures magnitude fits in int16 and total_mn <= 128.
        const uint8_t pLSB_adj = static_cast<uint8_t>(pLSB - 16);
        const v128_t zero16    = wasm_i32x4_const_splat(0);
        v128_t vn_16           = zero16;
        v128_t row16           = MagSgn.decode_two_quads_16bit_wasm(
            tv0, tv1, static_cast<uint16_t>(U0), static_cast<uint16_t>(U1), pLSB_adj, vn_16);
        v128_t row0_16 = wasm_i16x8_shuffle(row16, zero16, 0, 2, 4, 6, 8, 8, 8, 8);
        v128_t row1_16 = wasm_i16x8_shuffle(row16, zero16, 1, 3, 5, 7, 8, 8, 8, 8);
        v128_t mu0_32  = wasm_i32x4_shl(wasm_i32x4_extend_low_i16x8(row0_16), 16);
        v128_t mu1_32  = wasm_i32x4_shl(wasm_i32x4_extend_low_i16x8(row1_16), 16);
        if constexpr (fuse_dequant) {
          dequant_store_wasm(mp0, mu0_32, block->transformation, pLSB_dq, vfscale_dq, vmagmask_dq, vsignmask_dq);
          dequant_store_wasm(mp1, mu1_32, block->transformation, pLSB_dq, vfscale_dq, vmagmask_dq, vsignmask_dq);
        } else {
          wasm_v128_store(mp0, mu0_32);
          wasm_v128_store(mp1, mu1_32);
        }
        mp0 += 4;
        mp1 += 4;
        // Update exponent (before E_p advance, matching 32-bit path)
        v128_t vn32 = wasm_u32x4_extend_low_u16x8(vn_16);
        vExp        = wasm_i32x4_sub(wasm_i32x4_const_splat(32), wasm_u32x4_clz(vn32));
        Emax0       = wasm_i32x4_reduce_max(wasm_v128_load(E_p + 3));
        Emax1       = wasm_i32x4_reduce_max(wasm_v128_load(E_p + 5));
        wasm_v128_store(E_p, vExp);
        E_p += 4;
      } else {
        // 32-bit path
        v128_t vmask1, sig0, sig1, vtmp, m_n_0, m_n_1, msvec, v_n_0, v_n_1, mu0, mu1;

        sig0  = wasm_i32x4_ne(wasm_v128_and(wasm_i32x4_splat((int32_t)rho0), vm), wasm_i32x4_const_splat(0));
        vtmp  = wasm_v128_and(wasm_i32x4_ne(wasm_v128_and(wasm_i32x4_splat((int32_t)emb_k_0), vm),
                                           wasm_i32x4_const_splat(0)),
                             vone);
        m_n_0 = wasm_i32x4_sub(wasm_v128_and(sig0, wasm_i32x4_splat((int32_t)U0)), vtmp);
        sig1  = wasm_i32x4_ne(wasm_v128_and(wasm_i32x4_splat((int32_t)rho1), vm), wasm_i32x4_const_splat(0));
        vtmp  = wasm_v128_and(wasm_i32x4_ne(wasm_v128_and(wasm_i32x4_splat((int32_t)emb_k_1), vm),
                                           wasm_i32x4_const_splat(0)),
                             vone);
        m_n_1 = wasm_i32x4_sub(wasm_v128_and(sig1, wasm_i32x4_splat((int32_t)U1)), vtmp);

        vmask1 = wasm_i32x4_make((1 << wasm_i32x4_extract_lane(m_n_0, 0)) - 1,
                                 (1 << wasm_i32x4_extract_lane(m_n_0, 1)) - 1,
                                 (1 << wasm_i32x4_extract_lane(m_n_0, 2)) - 1,
                                 (1 << wasm_i32x4_extract_lane(m_n_0, 3)) - 1);
        msvec = MagSgn.fetch(m_n_0);
        v_n_0 = wasm_v128_and(msvec, vmask1);
        vtmp  = wasm_v128_and(wasm_i32x4_ne(wasm_v128_and(wasm_i32x4_splat((int32_t)emb_1_0), vm),
                                           wasm_i32x4_const_splat(0)),
                             vone);
        v_n_0 = wasm_v128_or(v_n_0, wasm_i32x4_shlv(vtmp, m_n_0));
        mu0   = wasm_i32x4_add(v_n_0, vtwo);
        mu0   = wasm_v128_or(mu0, vone);
        mu0   = wasm_i32x4_shlv(mu0, vshift);
        mu0   = wasm_v128_or(mu0, wasm_i32x4_shl(v_n_0, 31));
        mu0   = wasm_v128_and(mu0, sig0);

        vmask1 = wasm_i32x4_make((1 << wasm_i32x4_extract_lane(m_n_1, 0)) - 1,
                                 (1 << wasm_i32x4_extract_lane(m_n_1, 1)) - 1,
                                 (1 << wasm_i32x4_extract_lane(m_n_1, 2)) - 1,
                                 (1 << wasm_i32x4_extract_lane(m_n_1, 3)) - 1);
        msvec = MagSgn.fetch(m_n_1);
        v_n_1 = wasm_v128_and(msvec, vmask1);
        vtmp  = wasm_v128_and(wasm_i32x4_ne(wasm_v128_and(wasm_i32x4_splat((int32_t)emb_1_1), vm),
                                           wasm_i32x4_const_splat(0)),
                             vone);
        v_n_1 = wasm_v128_or(v_n_1, wasm_i32x4_shlv(vtmp, m_n_1));
        mu1   = wasm_i32x4_add(v_n_1, vtwo);
        mu1   = wasm_v128_or(mu1, vone);
        mu1   = wasm_i32x4_shlv(mu1, vshift);
        mu1   = wasm_v128_or(mu1, wasm_i32x4_shl(v_n_1, 31));
        mu1   = wasm_v128_and(mu1, sig1);

        if constexpr (fuse_dequant) {
          dequant_store_wasm(mp0, wasm_i32x4_shuffle(mu0, mu1, 0, 2, 4, 6), block->transformation,
                             pLSB_dq, vfscale_dq, vmagmask_dq, vsignmask_dq);
          dequant_store_wasm(mp1, wasm_i32x4_shuffle(mu0, mu1, 1, 3, 5, 7), block->transformation,
                             pLSB_dq, vfscale_dq, vmagmask_dq, vsignmask_dq);
        } else {
          wasm_v128_store(mp0, wasm_i32x4_shuffle(mu0, mu1, 0, 2, 4, 6));
          wasm_v128_store(mp1, wasm_i32x4_shuffle(mu0, mu1, 1, 3, 5, 7));
        }
        mp0 += 4;
        mp1 += 4;

        Emax0 = wasm_i32x4_reduce_max(wasm_v128_load(E_p + 3));
        Emax1 = wasm_i32x4_reduce_max(wasm_v128_load(E_p + 5));
        vExp  = wasm_i32x4_sub(wasm_i32x4_const_splat(32),
                               wasm_u32x4_clz(wasm_i32x4_shuffle(v_n_0, v_n_1, 1, 3, 5, 7)));
        wasm_v128_store(E_p, vExp);
        E_p += 4;
      }
    }
  }  // Non-Initial line-pair end
}  // Cleanup decoding end

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

      // Do NOT update sigma — it must retain cleanup-only significance for MRP
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
    const uint16_t *csig  = sigma + (y >> 2) * mstr;

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
  const int32_t pLSB = 31 - M_b;

  const uint32_t mask  = UINT32_MAX >> (M_b + 1);
  const v128_t vmask   = wasm_i32x4_splat(static_cast<int32_t>(~mask));
  const v128_t vROIshift = wasm_i32x4_splat(ROIshift);

  v128_t v0, v1, s0, s1, vROImask, vmagmask, vdst0, vdst1, vpLSB;
  vpLSB    = wasm_i32x4_splat(pLSB);
  vmagmask = wasm_i32x4_const_splat(INT32_MAX);
  if (this->transformation == 1) {
    // lossless path
    for (size_t i = 0; i < static_cast<size_t>(this->size.y); i++) {
      int32_t *val = this->sample_buf + i * this->blksampl_stride;
      sprec_t *dst = this->i_samples + i * this->band_stride;
      size_t len   = this->size.x;
      for (; len >= 8; len -= 8) {
        v0       = wasm_v128_load(val);
        v1       = wasm_v128_load(val + 4);
        s0       = wasm_i32x4_shr(v0, 31);
        s1       = wasm_i32x4_shr(v1, 31);
        v0       = wasm_v128_and(v0, vmagmask);
        v1       = wasm_v128_and(v1, vmagmask);
        vROImask = wasm_v128_and(v0, vmask);
        vROImask = wasm_i32x4_eq(vROImask, wasm_i32x4_const_splat(0));
        vROImask = wasm_v128_and(vROImask, vROIshift);
        v0       = wasm_i32x4_vshl(v0, wasm_i32x4_sub(vROImask, vpLSB));
        vROImask = wasm_v128_and(v1, vmask);
        vROImask = wasm_i32x4_eq(vROImask, wasm_i32x4_const_splat(0));
        vROImask = wasm_v128_and(vROImask, vROIshift);
        v1       = wasm_i32x4_vshl(v1, wasm_i32x4_sub(vROImask, vpLSB));
        vdst0    = wasm_v128_bitselect(wasm_i32x4_neg(v0), v0, s0);
        vdst1    = wasm_v128_bitselect(wasm_i32x4_neg(v1), v1, s1);
        wasm_v128_store(dst, wasm_f32x4_convert_i32x4(vdst0));
        wasm_v128_store(dst + 4, wasm_f32x4_convert_i32x4(vdst1));
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
        *dst = static_cast<int32_t>(*val);
        val++;
        dst++;
      }
    }
  } else {
    // lossy path
    float fscale_direct = this->stepsize;
    fscale_direct *= (1 << FRACBITS);
    if (M_b <= 31) {
      fscale_direct /= static_cast<float>(1 << (31 - M_b));
    } else {
      fscale_direct *= static_cast<float>(1 << (M_b - 31));
    }

    if (ROIshift == 0) {
      // Fast path: direct float multiply avoids integer approximation and ROI overhead
      const v128_t vfscale   = wasm_f32x4_splat(fscale_direct);
      const v128_t vsignmask = wasm_i32x4_const_splat(INT32_MIN);

      for (size_t i = 0; i < static_cast<size_t>(this->size.y); i++) {
        int32_t *val = this->sample_buf + i * this->blksampl_stride;
        sprec_t *dst = this->i_samples + i * this->band_stride;
        size_t len   = this->size.x;
        for (; len >= 8; len -= 8) {
          v0 = wasm_v128_load(val);
          v1 = wasm_v128_load(val + 4);
          // magnitude (strip sign bit), convert to float, scale
          v128_t m0 = wasm_v128_and(v0, vmagmask);
          v128_t m1 = wasm_v128_and(v1, vmagmask);
          v128_t f0 = wasm_f32x4_mul(wasm_f32x4_convert_i32x4(m0), vfscale);
          v128_t f1 = wasm_f32x4_mul(wasm_f32x4_convert_i32x4(m1), vfscale);
          // apply sign via XOR of sign bit into float sign bit
          wasm_v128_store(dst,     wasm_v128_xor(f0, wasm_v128_and(v0, vsignmask)));
          wasm_v128_store(dst + 4, wasm_v128_xor(f1, wasm_v128_and(v1, vsignmask)));
          val += 8;
          dst += 8;
        }
        for (; len > 0; --len) {
          float mag = static_cast<float>(*val & INT32_MAX);
          float f   = mag * fscale_direct;
          *dst      = (*val < 0) ? -f : f;
          val++;
          dst++;
        }
      }
    } else {
      // ROI path: integer approximation avoids int32 overflow when ROI shift inflates magnitude
      constexpr int32_t downshift = 15;
      float fscale                = fscale_direct * (float)(1 << 16) * (float)(1 << downshift);
      const auto scale            = static_cast<int32_t>(fscale + 0.5f);

      for (size_t i = 0; i < static_cast<size_t>(this->size.y); i++) {
        int32_t *val = this->sample_buf + i * this->blksampl_stride;
        sprec_t *dst = this->i_samples + i * this->band_stride;
        size_t len   = this->size.x;

        for (; len >= 8; len -= 8) {
          v0       = wasm_v128_load(val);
          v1       = wasm_v128_load(val + 4);
          s0       = wasm_i32x4_shr(v0, 31);
          s1       = wasm_i32x4_shr(v1, 31);
          v0       = wasm_v128_and(v0, vmagmask);
          v1       = wasm_v128_and(v1, vmagmask);
          vROImask = wasm_v128_and(v0, vmask);
          vROImask = wasm_i32x4_eq(vROImask, wasm_i32x4_const_splat(0));
          vROImask = wasm_v128_and(vROImask, vROIshift);
          v0       = wasm_i32x4_vshl(v0, vROImask);
          vROImask = wasm_v128_and(v1, vmask);
          vROImask = wasm_i32x4_eq(vROImask, wasm_i32x4_const_splat(0));
          vROImask = wasm_v128_and(vROImask, vROIshift);
          v1       = wasm_i32x4_vshl(v1, vROImask);
          // to prevent overflow, truncate to int16_t range
          v0       = wasm_i32x4_shr(wasm_i32x4_add(v0, wasm_i32x4_const_splat(1 << 15)), 16);
          v1       = wasm_i32x4_shr(wasm_i32x4_add(v1, wasm_i32x4_const_splat(1 << 15)), 16);
          // dequantization
          v0       = wasm_i32x4_mul(v0, wasm_i32x4_splat(scale));
          v1       = wasm_i32x4_mul(v1, wasm_i32x4_splat(scale));
          // downshift
          v0 = wasm_i32x4_shr(wasm_i32x4_add(v0, wasm_i32x4_const_splat(1 << (downshift - 1))), downshift);
          v1 = wasm_i32x4_shr(wasm_i32x4_add(v1, wasm_i32x4_const_splat(1 << (downshift - 1))), downshift);
          vdst0 = wasm_v128_bitselect(wasm_i32x4_neg(v0), v0, s0);
          vdst1 = wasm_v128_bitselect(wasm_i32x4_neg(v1), v1, s1);
          wasm_v128_store(dst, wasm_f32x4_convert_i32x4(vdst0));
          wasm_v128_store(dst + 4, wasm_f32x4_convert_i32x4(vdst1));
          val += 8;
          dst += 8;
        }
        for (; len > 0; --len) {
          int32_t sign = *val & INT32_MIN;
          *val &= INT32_MAX;
          if (((uint32_t)*val & ~mask) == 0) {
            *val <<= ROIshift;
          }
          *val = (*val + (1 << 15)) >> 16;
          *val *= scale;
          *val = static_cast<int32_t>((*val + (1 << (downshift - 1))) >> downshift);
          if (sign) {
            *val = -(*val & INT32_MAX);
          }
          *dst = static_cast<int32_t>(*val);
          val++;
          dst++;
        }
      }
    }
  }
}

bool htj2k_decode(j2k_codeblock *block, const uint8_t ROIshift) {
  uint8_t P0     = 0;
  int32_t Lcup   = 0;
  uint32_t Lref  = 0;
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
  const auto num_ht_passes = static_cast<uint8_t>(block->num_passes - empty_passes);
  uint8_t *Dcup;
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
    Dcup                 = block->get_compressed_data();
    const auto Scup      = static_cast<int32_t>((Dcup[Lcup - 1] << 4) + (Dcup[Lcup - 2] & 0x0F));
    Dcup[Lcup - 1]       = 0xFF;
    Dcup[Lcup - 2]      |= 0x0F;

    if (Scup < 2 || Scup > Lcup || Scup > 4079) {
      printf("WARNING: cleanup pass suffix length %d is invalid.\n", Scup);
      return false;
    }
    const auto Pcup = static_cast<int32_t>(Lcup - Scup);

    for (uint32_t i = 1; i < num_segments; i++) {
      Lref += block->pass_length[all_segments[i]];
    }
    if (block->num_passes > 1 && num_segments > 1) {
      Dref = block->get_compressed_data() + Lcup;
    } else {
      Dref = nullptr;
    }

    bool dequant_done = false;
    // Fused dequant gate: WASM SIMD stores write 4 elements (128-bit); when block width
    // is not a multiple of 4, the overshoot corrupts adjacent blocks in parallel decode.
    if (num_ht_passes == 1 && ROIshift == 0 && (block->size.x & 3) == 0) {
      ht_cleanup_decode<true, true>(block, static_cast<uint8_t>(30 - S_blk), Lcup, Pcup, Scup);
      dequant_done = true;
    } else if (num_ht_passes == 1) {
      ht_cleanup_decode<true>(block, static_cast<uint8_t>(30 - S_blk), Lcup, Pcup, Scup);
    } else {
      ht_cleanup_decode<false>(block, static_cast<uint8_t>(30 - S_blk), Lcup, Pcup, Scup);

      // Pack block_states SIGMA bits into nibble-packed sigma array
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

    if (!dequant_done) {
      block->dequantize(ROIshift);
    }

  }  // end

  return true;
}
#endif
