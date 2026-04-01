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

static inline v128_t wasm_u32x4_clz(v128_t a) {
  auto clz32 = [](uint32_t x) -> int32_t { return x == 0 ? 32 : __builtin_clz(x); };
  return wasm_i32x4_make(clz32((uint32_t)wasm_i32x4_extract_lane(a, 0)),
                         clz32((uint32_t)wasm_i32x4_extract_lane(a, 1)),
                         clz32((uint32_t)wasm_i32x4_extract_lane(a, 2)),
                         clz32((uint32_t)wasm_i32x4_extract_lane(a, 3)));
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

template <bool skip_sigma>
void ht_cleanup_decode(j2k_codeblock *block, const uint8_t &pLSB, const int32_t Lcup, const int32_t Pcup,
                       const int32_t Scup) {
  fwd_buf<0xFF> MagSgn(block->get_compressed_data(), Pcup);
  MEL_dec MEL(block->get_compressed_data(), Lcup, Scup);
  rev_buf VLC_dec(block->get_compressed_data(), Lcup, Scup);

  const uint16_t QW = static_cast<uint16_t>(ceil_int(static_cast<int16_t>(block->size.x), 2));
  const uint16_t QH = static_cast<uint16_t>(ceil_int(static_cast<int16_t>(block->size.y), 2));

  v128_t vExp;
  const v128_t vm     = wasm_i32x4_const(1, 2, 4, 8);
  const v128_t vone   = wasm_i32x4_const_splat(1);
  const v128_t vtwo   = wasm_i32x4_const_splat(2);
  const v128_t vshift = wasm_i32x4_splat(pLSB - 1);

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
  uint32_t kappa0 = 1, kappa1 = 1;

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

    context = ((tv0 & 0xE0U) << 2) | ((tv0 & 0x10U) << 3);

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

    if constexpr (!skip_sigma) {
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

    // WASM section
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

    wasm_v128_store(mp0, wasm_i32x4_shuffle(mu0, mu1, 0, 2, 4, 6));
    wasm_v128_store(mp1, wasm_i32x4_shuffle(mu0, mu1, 1, 3, 5, 7));
    mp0 += 4;
    mp1 += 4;

    // update Exponent
    vExp = wasm_i32x4_sub(wasm_i32x4_const_splat(32),
                          wasm_u32x4_clz(wasm_i32x4_shuffle(v_n_0, v_n_1, 1, 3, 5, 7)));
    wasm_v128_store(E_p, vExp);
    E_p += 4;
  }

  // Initial line-pair end

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

      if constexpr (!skip_sigma) {
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

      // WASM section
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

      wasm_v128_store(mp0, wasm_i32x4_shuffle(mu0, mu1, 0, 2, 4, 6));
      wasm_v128_store(mp1, wasm_i32x4_shuffle(mu0, mu1, 1, 3, 5, 7));
      mp0 += 4;
      mp1 += 4;

      Emax0 = wasm_i32x4_reduce_max(wasm_v128_load(E_p + 3));
      Emax1 = wasm_i32x4_reduce_max(wasm_v128_load(E_p + 5));

      // Update Exponent
      vExp = wasm_i32x4_sub(wasm_i32x4_const_splat(32),
                            wasm_u32x4_clz(wasm_i32x4_shuffle(v_n_0, v_n_1, 1, 3, 5, 7)));
      wasm_v128_store(E_p, vExp);
      E_p += 4;
    }
  }  // Non-Initial line-pair end
}  // Cleanup decoding end

auto process_stripes_block_dec = [](SP_dec &SigProp, j2k_codeblock *block, const uint32_t i_start,
                                    const uint32_t j_start, const uint32_t width, const uint32_t height,
                                    const uint8_t &pLSB) {
  int32_t *sp;
  uint8_t causal_cond     = 0;
  uint8_t bit;
  uint8_t mbr;
  const auto block_width  = j_start + width;
  const auto block_height = i_start + height;

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
        state_p[0] |= 1 << SHIFT_PI_;
        bit = SigProp.importSigPropBit();
        state_p[0] |= bit << SHIFT_REF;
        *sp |= bit << pLSB;
        *sp |= bit << (pLSB - 1);
      }
      state_p[0] |= 1 << SHIFT_SCAN;
    }
  }
  for (uint32_t j = j_start; j < block_width; j++) {
    for (uint32_t i = i_start; i < block_height; i++) {
      sp               = &block->sample_buf[j + i * block->blksampl_stride];
      uint8_t *state_p = block->block_states + (i + 1) * block->blkstate_stride + (j + 1);
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
        if ((state_p[0] >> SHIFT_SIGMA & 1) != 0) {
          state_p[0] |= 1 << SHIFT_PI_;
          bit = MagRef.importMagRefBit();
          tmp = static_cast<int32_t>(0xFFFFFFFE | static_cast<unsigned int>(bit));
          tmp <<= pLSB;
          sp[0] &= tmp;
          sp[0] |= 1 << (pLSB - 1);
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
      if ((state_p[0] >> SHIFT_SIGMA & 1) != 0) {
        state_p[0] |= 1 << SHIFT_PI_;
        bit = MagRef.importMagRefBit();
        tmp = static_cast<int32_t>(0xFFFFFFFE | static_cast<unsigned int>(bit));
        tmp <<= pLSB;
        sp[0] &= tmp;
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
  if (this->transformation) {
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
        v0       = wasm_i32x4_shr(wasm_i32x4_add(v0, wasm_i32x4_const_splat(1 << (downshift - 1))),
                                  downshift);
        v1       = wasm_i32x4_shr(wasm_i32x4_add(v1, wasm_i32x4_const_splat(1 << (downshift - 1))),
                                  downshift);
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
        *val = (*val + (1 << 15)) >> 16;
        *val *= scale;
        *val = (int32_t)((*val + (1 << (downshift - 1))) >> downshift);
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

    if (num_ht_passes == 1) {
      ht_cleanup_decode<true>(block, static_cast<uint8_t>(30 - S_blk), Lcup, Pcup, Scup);
    } else {
      ht_cleanup_decode<false>(block, static_cast<uint8_t>(30 - S_blk), Lcup, Pcup, Scup);
    }
    if (num_ht_passes > 1) {
      ht_sigprop_decode(block, Dref, Lref, static_cast<uint8_t>(30 - (S_blk + 1)));
    }
    if (num_ht_passes > 2) {
      ht_magref_decode(block, Dref, Lref, static_cast<uint8_t>(30 - (S_blk + 1)));
    }

    block->dequantize(ROIshift);

  }  // end

  return true;
}
#endif
