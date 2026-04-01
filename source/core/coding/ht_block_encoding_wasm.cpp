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
  #include "ht_block_encoding_wasm.hpp"
  #include "coding_local.hpp"
  #include "enc_CxtVLC_tables.hpp"
  #include "utils.hpp"
  #include <wasm_simd128.h>

// Uncomment for experimental use of HT SigProp and MagRef encoding (does not work)
// #define ENABLE_SP_MR

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

static inline int32_t wasm_i32x4_reduce_add(v128_t a) {
  v128_t t = wasm_i32x4_add(a, wasm_i32x4_shuffle(a, a, 2, 3, 0, 1));
  t        = wasm_i32x4_add(t, wasm_i32x4_shuffle(t, t, 1, 0, 3, 2));
  return wasm_i32x4_extract_lane(t, 0);
}

static inline int32_t wasm_i32x4_reduce_max(v128_t a) {
  v128_t t = wasm_i32x4_max(a, wasm_i32x4_shuffle(a, a, 2, 3, 0, 1));
  t        = wasm_i32x4_max(t, wasm_i32x4_shuffle(t, t, 1, 0, 3, 2));
  return wasm_i32x4_extract_lane(t, 0);
}

static inline v128_t wasm_u32x4_qsub(v128_t a, v128_t b) {
  v128_t diff = wasm_i32x4_sub(a, b);
  v128_t mask = wasm_u32x4_ge(a, b);
  return wasm_v128_and(diff, mask);
}

static inline v128_t wasm_u32x4_clz(v128_t a) {
  auto clz32 = [](uint32_t x) -> int32_t { return x == 0 ? 32 : __builtin_clz(x); };
  return wasm_i32x4_make(clz32((uint32_t)wasm_i32x4_extract_lane(a, 0)),
                         clz32((uint32_t)wasm_i32x4_extract_lane(a, 1)),
                         clz32((uint32_t)wasm_i32x4_extract_lane(a, 2)),
                         clz32((uint32_t)wasm_i32x4_extract_lane(a, 3)));
}

// Quantize DWT coefficients and transfer them to codeblock buffer in a form of MagSgn value
void j2k_codeblock::quantize(uint32_t &or_val) {
  float fscale = 1.0f / this->stepsize;
  fscale /= (1 << (FRACBITS));
  if (transformation) fscale = 1.0f;

  const uint32_t height = this->size.y;
  const uint32_t stride = this->band_stride;
  v128_t vscale         = wasm_f32x4_splat(fscale);
  v128_t vorval         = wasm_i32x4_const_splat(0);
  for (uint16_t i = 0; i < static_cast<uint16_t>(height); ++i) {
    sprec_t *sp        = this->i_samples + i * stride;
    int32_t *dp        = this->sample_buf + i * blksampl_stride;
    size_t block_index = (i + 1U) * (blkstate_stride) + 1U;
    uint8_t *dstblk    = block_states + block_index;
    int16_t len        = static_cast<int16_t>(this->size.x);
    for (; len >= 8; len -= 8) {
      v128_t fv0 = wasm_v128_load(sp);
      v128_t fv1 = wasm_v128_load(sp + 4);
      // Quantization
      v128_t v0 = wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_mul(fv0, vscale));
      v128_t v1 = wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_mul(fv1, vscale));
      // Take sign bit
      v128_t s0 = wasm_u32x4_shr(v0, 31);
      v128_t s1 = wasm_u32x4_shr(v1, 31);
      // Absolute value
      v0 = wasm_i32x4_abs(v0);
      v1 = wasm_i32x4_abs(v1);
      // Block states
      v128_t narrow0     = wasm_i16x8_narrow_i32x4(v0, v0);
      v128_t narrow1     = wasm_i16x8_narrow_i32x4(v1, v1);
      v128_t combined16  = wasm_i8x16_shuffle(narrow0, narrow1, 0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20,
                                             21, 22, 23);
      v128_t gt0_mask    = wasm_i16x8_gt(combined16, wasm_i16x8_const_splat(0));
      v128_t vblkstate_v = wasm_v128_and(gt0_mask, wasm_i16x8_const_splat(1));
      v128_t narrow8     = wasm_i8x16_narrow_i16x8(vblkstate_v, vblkstate_v);
      *(uint64_t *)dstblk = (uint64_t)wasm_i64x2_extract_lane(narrow8, 0);
      dstblk += 8;
      // Check emptiness of a block
      vorval = wasm_v128_or(vorval, v0);
      vorval = wasm_v128_or(vorval, v1);
      // Convert two's complement to MagSgn form
      v0 = wasm_u32x4_qsub(v0, wasm_i32x4_const_splat(1));
      v1 = wasm_u32x4_qsub(v1, wasm_i32x4_const_splat(1));
      v0 = wasm_i32x4_shl(v0, 1);
      v1 = wasm_i32x4_shl(v1, 1);
      v0 = wasm_i32x4_add(v0, s0);
      v1 = wasm_i32x4_add(v1, s1);
      // Store
      wasm_v128_store(dp, v0);
      wasm_v128_store(dp + 4, v1);
      sp += 8;
      dp += 8;
    }
    // Check emptiness of a block
    or_val |= static_cast<unsigned int>(wasm_i32x4_reduce_max(vorval));
    // process leftover
    for (; len > 0; --len) {
      int32_t temp;
      temp             = static_cast<int32_t>(static_cast<float>(sp[0]) * fscale);
      uint32_t sign    = static_cast<uint32_t>(temp) & 0x80000000;
      temp             = (temp < 0) ? -temp : temp;
      temp &= 0x7FFFFFFF;
      if (temp) {
        or_val |= 1;
        dstblk[0] |= 1;
        temp--;
        temp <<= 1;
        temp += static_cast<uint8_t>(sign >> 31);
        dp[0] = temp;
      }
      ++sp;
      ++dp;
      ++dstblk;
    }
  }
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
 * HT cleanup encoding: helper functions
 *******************************************************************************/
auto make_storage = [](uint8_t *ssp0, uint8_t *ssp1, int32_t *sp0, int32_t *sp1, v128_t &sig0,
                       v128_t &sig1, v128_t &v0, v128_t &v1, v128_t &E0, v128_t &E1, int32_t &rho0,
                       int32_t &rho1) {
  v128_t t0 = wasm_v128_load(sp0);
  v128_t t1 = wasm_v128_load(sp1);
  v0        = wasm_i32x4_shuffle(t0, t1, 0, 4, 1, 5);
  v1        = wasm_i32x4_shuffle(t0, t1, 2, 6, 3, 7);

  v128_t vssp0_l = wasm_v128_load64_zero(ssp0);
  v128_t vssp1_l = wasm_v128_load64_zero(ssp1);
  v128_t sig01_v = wasm_v128_and(
      wasm_i8x16_shuffle(vssp0_l, vssp1_l, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23),
      wasm_i8x16_const_splat(1));
  // sig0 from bytes 0-3 of sig01_v
  v128_t sig01_u16 = wasm_u16x8_extend_low_u8x16(sig01_v);
  sig0             = wasm_i32x4_gt(wasm_u32x4_extend_low_u16x8(sig01_u16), wasm_i32x4_const_splat(0));
  // sig1 from bytes 4-7 of sig01_v
  v128_t sig01_hi     = wasm_i8x16_shuffle(sig01_v, sig01_v, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5,
                                           6, 7);
  v128_t sig01_hi_u16 = wasm_u16x8_extend_low_u8x16(sig01_hi);
  sig1                = wasm_i32x4_gt(wasm_u32x4_extend_low_u16x8(sig01_hi_u16), wasm_i32x4_const_splat(0));
  // rho computation
  uint64_t sig_bits  = (uint64_t)wasm_i64x2_extract_lane(sig01_v, 0);
  uint8_t rho01_byte = 0;
  for (int k = 0; k < 8; k++) rho01_byte |= (uint8_t)(((sig_bits >> (8 * k)) & 1) << k);
  rho0 = rho01_byte & 0xF;
  rho1 = rho01_byte >> 4;

  E0 = wasm_v128_and(wasm_i32x4_sub(wasm_i32x4_const_splat(32), wasm_u32x4_clz(v0)), sig0);
  E1 = wasm_v128_and(wasm_i32x4_sub(wasm_i32x4_const_splat(32), wasm_u32x4_clz(v1)), sig1);
};

auto make_storage_one = [](uint8_t *ssp0, uint8_t *ssp1, int32_t *sp0, int32_t *sp1, v128_t &sig0,
                           v128_t &v0, v128_t &E0, int32_t &rho0) {
  v0              = wasm_i32x4_make(sp0[0], sp1[0], sp0[1], sp1[1]);
  v128_t sig      = wasm_i32x4_make(ssp0[0] & 1, ssp1[0] & 1, ssp0[1] & 1, ssp1[1] & 1);
  const v128_t shift = wasm_i32x4_const(0, 1, 2, 3);
  rho0  = (int32_t)((uint32_t)wasm_i32x4_reduce_add(wasm_i32x4_shlv(sig, shift))) & 0xF;
  sig0  = wasm_i32x4_gt(sig, wasm_i32x4_const_splat(0));
  E0    = wasm_v128_and(wasm_i32x4_sub(wasm_i32x4_const_splat(32), wasm_u32x4_clz(v0)), sig0);
};

// joint termination of MEL and VLC
int32_t termMELandVLC(state_VLC_enc &VLC, state_MEL_enc &MEL) {
  VLC.termVLC();
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
      VLC.pos--;
    }
    MEL.pos++;
  }
  memmove(&MEL.buf[MEL.pos], &VLC.buf[VLC.pos + 1], static_cast<size_t>(MAX_Scup - VLC.pos - 1));
  return (MEL.pos + MAX_Scup - VLC.pos - 1);
}

// joint termination of SP and MR
int32_t termSPandMR(SP_enc &SP, MR_enc &MR) {
  uint8_t SP_mask = static_cast<uint8_t>(0xFF >> (8 - SP.bits));
  SP_mask         = static_cast<uint8_t>(SP_mask | ((1 << SP.max) & 0x80));
  uint8_t MR_mask = static_cast<uint8_t>(0xFF >> (8 - MR.bits));
  if ((SP_mask | MR_mask) == 0) {
    memmove(&SP.buf[SP.pos], &MR.buf[MR.pos + 1], MAX_Lref - MR.pos);
    return static_cast<int32_t>(SP.pos + MAX_Lref - MR.pos);
  }
  uint8_t fuse = SP.tmp | MR.tmp;
  if ((((fuse ^ SP.tmp) & SP_mask) | ((fuse ^ MR.tmp) & MR_mask)) == 0) {
    SP.buf[SP.pos] = fuse;
  } else {
    SP.buf[SP.pos] = SP.tmp;
    MR.buf[MR.pos] = MR.tmp;
    MR.pos--;
  }
  SP.pos++;
  memmove(&SP.buf[SP.pos], &MR.buf[MR.pos + 1], MAX_Lref - MR.pos);
  return static_cast<int32_t>(SP.pos + MAX_Lref - MR.pos);
}

/*********************************************************************************************************************/
// HT Cleanup encoding
/*********************************************************************************************************************/
int32_t htj2k_cleanup_encode(j2k_codeblock *const block, const uint8_t ROIshift) noexcept {
  int32_t Lcup;
  int32_t Pcup;
  int32_t Scup;
  uint32_t or_val = 0;
  if (ROIshift) {
    printf("WARNING: Encoding with ROI is not supported.\n");
  }

  const uint32_t QW = ceil_int(block->size.x, 2U);
  const uint32_t QH = ceil_int(block->size.y, 2U);

  block->quantize(or_val);

  if (!or_val) {
    block->length         = 0;
    block->pass_length[0] = 0;
    block->num_passes      = 0;
    block->layer_passes[0] = 0;
    block->layer_start[0]  = 0;
    block->num_ZBP         = static_cast<uint8_t>(block->get_Mb() - 1);
    return static_cast<int32_t>(block->length);
  }

  alignas(4) static thread_local uint8_t fwd_buf[MAX_Lcup];
  alignas(4) static thread_local uint8_t rev_buf[MAX_Scup];

  state_MS_enc MagSgn_encoder(fwd_buf);
  state_MEL_enc MEL_encoder(rev_buf);
  state_VLC_enc VLC_encoder(rev_buf);

  alignas(32) int32_t Eline[2 * 512 + 6];
  std::fill_n(Eline, 2U * QW + 6U, int32_t{0});
  int32_t *E_p = Eline + 1;
  alignas(32) int32_t rholine[512 + 3];
  std::fill_n(rholine, QW + 3U, int32_t{0});
  int32_t *rho_p = rholine + 1;

  int32_t gamma;
  int32_t context = 0, n_q;
  uint32_t CxtVLC, lw, cwd;
  int32_t Emax_q;
  int32_t rho0, rho1, U0, U1;
  int32_t u_q, uoff, u_min, uvlc_idx, kappa = 1;
  int32_t emb_pattern, embk_0, embk_1, emb1_0, emb1_1;

  const v128_t lshift = wasm_i32x4_const(0, 1, 2, 3);
  const v128_t rshift = wasm_i32x4_const(0, -1, -2, -3);
  const v128_t vone   = wasm_i32x4_const_splat(1);
  v128_t v0, v1, E0, E1, sig0, sig1, Etmp, vuoff, mask, m0, m1, known1_0, known1_1;

  /*******************************************************************************************************************/
  // Initial line-pair
  /*******************************************************************************************************************/
  uint8_t *ssp0 = block->block_states + 1U * (block->blkstate_stride) + 1U;
  uint8_t *ssp1 = ssp0 + block->blkstate_stride;
  int32_t *sp0  = block->sample_buf;
  int32_t *sp1  = sp0 + block->blksampl_stride;
  uint32_t qx;
  for (qx = QW; qx >= 2; qx -= 2) {
    bool uoff_flag = true;

    make_storage(ssp0, ssp1, sp0, sp1, sig0, sig1, v0, v1, E0, E1, rho0, rho1);
    // update Eline
    wasm_v128_store(E_p, wasm_i32x4_shuffle(E0, E1, 1, 3, 5, 7));
    E_p += 4;
    // MEL encoding for the first quad
    if (context == 0) {
      MEL_encoder.encodeMEL((rho0 != 0));
    }
    // calculate u_off values
    Emax_q   = wasm_i32x4_reduce_max(E0);
    U0       = std::max(Emax_q, kappa);
    u_q      = U0 - kappa;
    u_min    = u_q;
    uvlc_idx = u_q;
    uoff     = (u_q) ? 1 : 0;
    uoff_flag &= uoff;
    Etmp        = wasm_i32x4_splat(Emax_q);
    vuoff       = wasm_i32x4_splat(uoff);
    mask        = wasm_i32x4_eq(E0, Etmp);
    emb_pattern = wasm_i32x4_reduce_add(wasm_v128_and(wasm_i32x4_shlv(vuoff, lshift), mask));
    n_q         = emb_pattern + (rho0 << 4) + (context << 8);
    // prepare VLC encoding of quad 0
    CxtVLC = enc_CxtVLC_table0[n_q];
    embk_0 = CxtVLC & 0xF;
    emb1_0 = emb_pattern & embk_0;
    lw     = (CxtVLC >> 4) & 0x07;
    cwd    = CxtVLC >> 7;

    // context for the next quad
    context = (rho0 >> 1) | (rho0 & 0x1);

    Emax_q = wasm_i32x4_reduce_max(E1);
    U1     = std::max(Emax_q, kappa);
    u_q    = U1 - kappa;
    u_min  = (u_min < u_q) ? u_min : u_q;
    uvlc_idx += u_q << 5;
    uoff = (u_q) ? 1 : 0;
    uoff_flag &= uoff;
    Etmp        = wasm_i32x4_splat(Emax_q);
    vuoff       = wasm_i32x4_splat(uoff);
    mask        = wasm_i32x4_eq(E1, Etmp);
    emb_pattern = wasm_i32x4_reduce_add(wasm_v128_and(wasm_i32x4_shlv(vuoff, lshift), mask));
    n_q         = emb_pattern + (rho1 << 4) + (context << 8);
    // VLC encoding of quads 0 and 1
    VLC_encoder.emitVLCBits(cwd, lw);  // quad 0
    CxtVLC = enc_CxtVLC_table0[n_q];
    embk_1 = CxtVLC & 0xF;
    emb1_1 = emb_pattern & embk_1;
    lw     = (CxtVLC >> 4) & 0x07;
    cwd    = CxtVLC >> 7;
    VLC_encoder.emitVLCBits(cwd, lw);  // quad 1
    // UVLC encoding
    uint32_t tmp = enc_UVLC_table0[uvlc_idx];
    lw           = tmp & 0xFF;
    cwd          = tmp >> 8;
    VLC_encoder.emitVLCBits(cwd, lw);

    // MEL encoding of the second quad
    if (context == 0) {
      if (rho1 != 0) {
        MEL_encoder.encodeMEL(1);
      } else {
        if (u_min > 2) {
          MEL_encoder.encodeMEL(1);
        } else {
          MEL_encoder.encodeMEL(0);
        }
      }
    } else if (uoff_flag) {
      if (u_min > 2) {
        MEL_encoder.encodeMEL(1);
      } else {
        MEL_encoder.encodeMEL(0);
      }
    }

    // MagSgn encoding
    m0 = wasm_i32x4_sub(wasm_v128_and(sig0, wasm_i32x4_splat(U0)),
                        wasm_v128_and(wasm_i32x4_vshl(wasm_i32x4_splat(embk_0), rshift), vone));
    m1 = wasm_i32x4_sub(wasm_v128_and(sig1, wasm_i32x4_splat(U1)),
                        wasm_v128_and(wasm_i32x4_vshl(wasm_i32x4_splat(embk_1), rshift), vone));
    known1_0 = wasm_v128_and(wasm_i32x4_vshl(wasm_i32x4_splat(emb1_0), rshift), vone);
    known1_1 = wasm_v128_and(wasm_i32x4_vshl(wasm_i32x4_splat(emb1_1), rshift), vone);
    MagSgn_encoder.emitBits(v0, m0, known1_0);
    MagSgn_encoder.emitBits(v1, m1, known1_1);

    // context for the next quad
    context = (rho1 >> 1) | (rho1 & 0x1);
    // update rho_line
    *rho_p++ = rho0;
    *rho_p++ = rho1;
    // update pointer to line buffer
    ssp0 += 4;
    ssp1 += 4;
    sp0 += 4;
    sp1 += 4;
  }
  if (qx) {
    make_storage_one(ssp0, ssp1, sp0, sp1, sig0, v0, E0, rho0);
    *E_p++ = wasm_i32x4_extract_lane(E0, 1);
    *E_p++ = wasm_i32x4_extract_lane(E0, 3);

    // MEL encoding
    if (context == 0) {
      MEL_encoder.encodeMEL((rho0 != 0));
    }

    Emax_q   = wasm_i32x4_reduce_max(E0);
    U0       = std::max(Emax_q, kappa);
    u_q      = U0 - kappa;
    uvlc_idx = u_q;
    uoff     = (u_q) ? 1 : 0;

    Etmp        = wasm_i32x4_splat(Emax_q);
    vuoff       = wasm_i32x4_splat(uoff);
    mask        = wasm_i32x4_eq(E0, Etmp);
    emb_pattern = wasm_i32x4_reduce_add(wasm_v128_and(wasm_i32x4_shlv(vuoff, lshift), mask));
    n_q         = emb_pattern + (rho0 << 4) + (context << 8);
    // VLC encoding
    CxtVLC = enc_CxtVLC_table0[n_q];
    embk_0 = CxtVLC & 0xF;
    emb1_0 = emb_pattern & embk_0;
    lw     = (CxtVLC >> 4) & 0x07;
    cwd    = CxtVLC >> 7;
    VLC_encoder.emitVLCBits(cwd, lw);
    // UVLC encoding
    uint32_t tmp = enc_UVLC_table0[uvlc_idx];
    lw           = tmp & 0xFF;
    cwd          = tmp >> 8;
    VLC_encoder.emitVLCBits(cwd, lw);

    // MagSgn encoding
    m0 = wasm_i32x4_sub(wasm_v128_and(sig0, wasm_i32x4_splat(U0)),
                        wasm_v128_and(wasm_i32x4_vshl(wasm_i32x4_splat(embk_0), rshift), vone));
    known1_0 = wasm_v128_and(wasm_i32x4_vshl(wasm_i32x4_splat(emb1_0), rshift), vone);
    MagSgn_encoder.emitBits(v0, m0, known1_0);

    // update rho_line
    *rho_p++ = rho0;
  }

  /*******************************************************************************************************************/
  // Non-initial line-pair
  /*******************************************************************************************************************/
  int32_t Emax0, Emax1;
  for (uint32_t qy = 1; qy < QH; ++qy) {
    E_p   = Eline + 1;
    rho_p = rholine + 1;
    rho1  = 0;

    Emax0 = find_max(E_p[-1], E_p[0], E_p[1], E_p[2]);
    Emax1 = find_max(E_p[1], E_p[2], E_p[3], E_p[4]);

    context = ((rho1 & 0x4) << 7) | ((rho1 & 0x8) << 6);
    context |= ((rho_p[-1] & 0x8) << 5) | ((rho_p[0] & 0x2) << 7);
    context |= ((rho_p[0] & 0x8) << 7) | ((rho_p[1] & 0x2) << 9);

    ssp0 = block->block_states + (2U * qy + 1U) * (block->blkstate_stride) + 1U;
    ssp1 = ssp0 + block->blkstate_stride;
    sp0  = block->sample_buf + 2U * (qy * block->blksampl_stride);
    sp1  = sp0 + block->blksampl_stride;
    for (qx = QW; qx >= 2; qx -= 2) {
      make_storage(ssp0, ssp1, sp0, sp1, sig0, sig1, v0, v1, E0, E1, rho0, rho1);
      // MEL encoding of the first quad
      if (context == 0) {
        MEL_encoder.encodeMEL((rho0 != 0));
      }
      gamma       = ((rho0 & (rho0 - 1)) == 0) ? 0 : 1;
      kappa       = std::max((Emax0 - 1) * gamma, 1);
      Emax_q      = wasm_i32x4_reduce_max(E0);
      U0          = std::max(Emax_q, kappa);
      u_q         = U0 - kappa;
      uvlc_idx    = u_q;
      uoff        = (u_q) ? 1 : 0;
      Etmp        = wasm_i32x4_splat(Emax_q);
      vuoff       = wasm_i32x4_splat(uoff);
      mask        = wasm_i32x4_eq(E0, Etmp);
      emb_pattern = wasm_i32x4_reduce_add(wasm_v128_and(wasm_i32x4_shlv(vuoff, lshift), mask));
      n_q         = emb_pattern + (rho0 << 4) + (context << 0);
      // prepare VLC encoding of quad 0
      CxtVLC = enc_CxtVLC_table1[n_q];
      embk_0 = CxtVLC & 0xF;
      emb1_0 = emb_pattern & embk_0;
      lw     = (CxtVLC >> 4) & 0x07;
      cwd    = CxtVLC >> 7;

      // calculate context for the next quad
      context = ((rho0 & 0x4) << 7) | ((rho0 & 0x8) << 6);
      context |= ((rho_p[0] & 0x8) << 5) | ((rho_p[1] & 0x2) << 7);
      context |= ((rho_p[1] & 0x8) << 7) | ((rho_p[2] & 0x2) << 9);
      // MEL encoding of the second quad
      if (context == 0) {
        MEL_encoder.encodeMEL((rho1 != 0));
      }
      gamma  = ((rho1 & (rho1 - 1)) == 0) ? 0 : 1;
      kappa  = std::max((Emax1 - 1) * gamma, 1);
      Emax_q = wasm_i32x4_reduce_max(E1);
      U1     = std::max(Emax_q, kappa);
      u_q    = U1 - kappa;
      uvlc_idx += u_q << 5;
      uoff        = (u_q) ? 1 : 0;
      Etmp        = wasm_i32x4_splat(Emax_q);
      vuoff       = wasm_i32x4_splat(uoff);
      mask        = wasm_i32x4_eq(E1, Etmp);
      emb_pattern = wasm_i32x4_reduce_add(wasm_v128_and(wasm_i32x4_shlv(vuoff, lshift), mask));
      n_q         = emb_pattern + (rho1 << 4) + (context << 0);
      // VLC encoding of quads 0 and 1
      VLC_encoder.emitVLCBits(cwd, lw);  // quad 0
      CxtVLC = enc_CxtVLC_table1[n_q];
      embk_1 = CxtVLC & 0xF;
      emb1_1 = emb_pattern & embk_1;
      lw     = (CxtVLC >> 4) & 0x07;
      cwd    = CxtVLC >> 7;
      VLC_encoder.emitVLCBits(cwd, lw);  // quad 1
      // UVLC encoding
      uint32_t tmp = enc_UVLC_table1[uvlc_idx];
      lw           = tmp & 0xFF;
      cwd          = tmp >> 8;
      VLC_encoder.emitVLCBits(cwd, lw);

      // MagSgn encoding
      m0 = wasm_i32x4_sub(wasm_v128_and(sig0, wasm_i32x4_splat(U0)),
                          wasm_v128_and(wasm_i32x4_vshl(wasm_i32x4_splat(embk_0), rshift), vone));
      m1 = wasm_i32x4_sub(wasm_v128_and(sig1, wasm_i32x4_splat(U1)),
                          wasm_v128_and(wasm_i32x4_vshl(wasm_i32x4_splat(embk_1), rshift), vone));
      known1_0 = wasm_v128_and(wasm_i32x4_vshl(wasm_i32x4_splat(emb1_0), rshift), vone);
      known1_1 = wasm_v128_and(wasm_i32x4_vshl(wasm_i32x4_splat(emb1_1), rshift), vone);
      MagSgn_encoder.emitBits(v0, m0, known1_0);
      MagSgn_encoder.emitBits(v1, m1, known1_1);

      Emax0 = wasm_i32x4_reduce_max(wasm_v128_load(E_p + 3));
      Emax1 = wasm_i32x4_reduce_max(wasm_v128_load(E_p + 5));
      wasm_v128_store(E_p, wasm_i32x4_shuffle(E0, E1, 1, 3, 5, 7));
      E_p += 4;

      // calculate context for the next quad
      context = ((rho1 & 0x4) << 7) | ((rho1 & 0x8) << 6);
      context |= ((rho_p[1] & 0x8) << 5) | ((rho_p[2] & 0x2) << 7);
      context |= ((rho_p[2] & 0x8) << 7) | ((rho_p[3] & 0x2) << 9);

      // update rho_line
      *rho_p++ = rho0;
      *rho_p++ = rho1;
      // update pointer to line buffer
      ssp0 += 4;
      ssp1 += 4;
      sp0 += 4;
      sp1 += 4;
    }
    if (qx) {
      make_storage_one(ssp0, ssp1, sp0, sp1, sig0, v0, E0, rho0);
      *E_p++ = wasm_i32x4_extract_lane(E0, 1);
      *E_p++ = wasm_i32x4_extract_lane(E0, 3);

      // MEL encoding of the first quad
      if (context == 0) {
        MEL_encoder.encodeMEL((rho0 != 0));
      }

      gamma    = (popcount32((uint32_t)rho0) > 1) ? 1 : 0;
      kappa    = std::max((Emax0 - 1) * gamma, 1);
      Emax_q   = wasm_i32x4_reduce_max(E0);
      U0       = std::max(Emax_q, kappa);
      u_q      = U0 - kappa;
      uvlc_idx = u_q;
      uoff     = (u_q) ? 1 : 0;

      Etmp        = wasm_i32x4_splat(Emax_q);
      vuoff       = wasm_i32x4_splat(uoff);
      mask        = wasm_i32x4_eq(E0, Etmp);
      emb_pattern = wasm_i32x4_reduce_add(wasm_v128_and(wasm_i32x4_shlv(vuoff, lshift), mask));
      n_q         = emb_pattern + (rho0 << 4) + (context << 0);
      // VLC encoding
      CxtVLC = enc_CxtVLC_table1[n_q];
      embk_0 = CxtVLC & 0xF;
      emb1_0 = emb_pattern & embk_0;
      lw     = (CxtVLC >> 4) & 0x07;
      cwd    = CxtVLC >> 7;
      VLC_encoder.emitVLCBits(cwd, lw);
      // UVLC encoding
      uint32_t tmp = enc_UVLC_table1[uvlc_idx];
      lw           = tmp & 0xFF;
      cwd          = tmp >> 8;
      VLC_encoder.emitVLCBits(cwd, lw);

      // MagSgn encoding
      m0 = wasm_i32x4_sub(wasm_v128_and(sig0, wasm_i32x4_splat(U0)),
                          wasm_v128_and(wasm_i32x4_vshl(wasm_i32x4_splat(embk_0), rshift), vone));
      known1_0 = wasm_v128_and(wasm_i32x4_vshl(wasm_i32x4_splat(emb1_0), rshift), vone);
      MagSgn_encoder.emitBits(v0, m0, known1_0);

      // update rho_line
      *rho_p++ = rho0;
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

  block->set_compressed_data(fwd_buf, static_cast<uint16_t>(Lcup), MAX_Lref);
  block->length         = static_cast<uint32_t>(Lcup);
  block->pass_length[0] = static_cast<unsigned int>(Lcup);
  block->num_passes      = 1;
  block->layer_passes[0] = 1;
  block->layer_start[0]  = 0;
  block->num_ZBP         = static_cast<uint8_t>(block->get_Mb() - 1);
  return static_cast<int32_t>(block->length);
}
/********************************************************************************
 * HT sigprop encoding
 *******************************************************************************/
auto process_stripes_block_enc = [](SP_enc &SigProp, j2k_codeblock *block, const uint32_t i_start,
                                    const uint32_t j_start, const uint32_t width,
                                    const uint32_t height) {
  uint8_t *sp;
  uint8_t causal_cond     = 0;
  uint8_t bit;
  uint8_t mbr;
  const auto block_width  = j_start + width;
  const auto block_height = i_start + height;
  for (uint32_t j = j_start; j < block_width; j++) {
    for (uint32_t i = i_start; i < block_height; i++) {
      sp          = block->block_states + (i + 1) * block->blkstate_stride + (j + 1);
      causal_cond = (((block->Cmodes & CAUSAL) == 0) || (i != i_start + height - 1));
      mbr         = 0;
      if ((sp[0] >> SHIFT_SIGMA & 1) == 0) {
        mbr = block->calc_mbr(i, j, causal_cond);
      }
      if (mbr != 0) {
        bit = (*sp >> SHIFT_SMAG) & 1;
        SigProp.emitSPBit(bit);
        sp[0] |= 1 << SHIFT_PI_;
        sp[0] |= bit << SHIFT_REF;
      }
      sp[0] |= 1 << SHIFT_SCAN;
    }
  }
  for (uint32_t j = j_start; j < block_width; j++) {
    for (uint32_t i = i_start; i < block_height; i++) {
      sp = block->block_states + (i + 1) * block->blkstate_stride + (j + 1);
      if ((sp[0] >> SHIFT_REF) & 1) {
        bit = (sp[0] >> SHIFT_SSGN) & 1;
        SigProp.emitSPBit(bit);
      }
    }
  }
};

void ht_sigprop_encode(j2k_codeblock *block, SP_enc &SigProp) {
  const uint32_t num_v_stripe = block->size.y / 4;
  const uint32_t num_h_stripe = block->size.x / 4;
  uint32_t i_start            = 0, j_start;
  uint32_t width              = 4;
  uint32_t width_last;
  uint32_t height = 4;

  for (uint32_t n1 = 0; n1 < num_v_stripe; n1++) {
    j_start = 0;
    for (uint32_t n2 = 0; n2 < num_h_stripe; n2++) {
      process_stripes_block_enc(SigProp, block, i_start, j_start, width, height);
      j_start += 4;
    }
    width_last = block->size.x % 4;
    if (width_last) {
      process_stripes_block_enc(SigProp, block, i_start, j_start, width_last, height);
    }
    i_start += 4;
  }
  height  = block->size.y % 4;
  j_start = 0;
  for (uint32_t n2 = 0; n2 < num_h_stripe; n2++) {
    process_stripes_block_enc(SigProp, block, i_start, j_start, width, height);
    j_start += 4;
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
  const uint32_t blk_height   = block->size.y;
  const uint32_t blk_width    = block->size.x;
  const uint32_t num_v_stripe = block->size.y / 4;
  uint32_t i_start            = 0;
  uint32_t height             = 4;
  uint8_t *sp;
  uint8_t bit;

  for (uint32_t n1 = 0; n1 < num_v_stripe; n1++) {
    for (uint32_t j = 0; j < blk_width; j++) {
      for (uint32_t i = i_start; i < i_start + height; i++) {
        sp = block->block_states + (i + 1) * block->blkstate_stride + (j + 1);
        if ((sp[0] >> SHIFT_SIGMA & 1) != 0) {
          bit = (sp[0] >> SHIFT_SMAG) & 1;
          MagRef.emitMRBit(bit);
          sp[0] |= 1 << SHIFT_PI_;
        }
      }
    }
    i_start += 4;
  }
  height = blk_height % 4;
  for (uint32_t j = 0; j < blk_width; j++) {
    for (uint32_t i = i_start; i < i_start + height; i++) {
      sp = block->block_states + (i + 1) * block->blkstate_stride + (j + 1);
      if ((sp[0] >> SHIFT_SIGMA & 1) != 0) {
        bit = (sp[0] >> SHIFT_SMAG) & 1;
        MagRef.emitMRBit(bit);
        sp[0] |= 1 << SHIFT_PI_;
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
    ht_sigprop_encode(block, SigProp);
    ht_magref_encode(block, MagRef);
    if (MagRef.get_length()) {
      HTMagRefLength         = termSPandMR(SigProp, MagRef);
      block->num_passes      = static_cast<uint8_t>(block->num_passes + 2);
      block->layer_passes[0] = static_cast<uint8_t>(block->layer_passes[0] + 2);
      block->pass_length[block->pass_length_count++] = SigProp.get_length();
      block->pass_length[block->pass_length_count++] = MagRef.get_length();
    } else {
      SigProp.termSP();
      HTMagRefLength         = static_cast<int32_t>(SigProp.get_length());
      block->num_passes      = static_cast<uint8_t>(block->num_passes + 1);
      block->layer_passes[0] = static_cast<uint8_t>(block->layer_passes[0] + 1);
      block->pass_length[block->pass_length_count++] = SigProp.get_length();
    }
    if (HTMagRefLength) {
      block->length += static_cast<unsigned int>(HTMagRefLength);
      block->num_ZBP = static_cast<uint8_t>(block->num_ZBP - (block->refsegment));
      block->set_compressed_data(Dref, static_cast<uint16_t>(HTMagRefLength));
    }
  }
  return EXIT_SUCCESS;
}
#endif
