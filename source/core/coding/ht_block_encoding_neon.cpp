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

#if defined(OPENHTJ2K_ENABLE_ARM_NEON)
  #include <algorithm>
  #include <cmath>
  #include "coding_units.hpp"
  #include "ht_block_encoding.hpp"
  #include "coding_local.hpp"
  #include "enc_CxtVLC_tables.hpp"
  #include "utils.hpp"

  #include <arm_neon.h>

  #define Q0 0
  #define Q1 1

//#define HTSIMD
//#define ENABLE_SP_MR

// Quantize DWT coefficients and transfer them to codeblock buffer in a form of MagSgn value
void j2k_codeblock::quantize(uint32_t &or_val) {
  // TODO: check the way to quantize in terms of precision and reconstruction quality
  float fscale = 1.0f / this->stepsize;
  fscale /= (1 << (FRACBITS));
  // Set fscale = 1.0 in lossless coding instead of skipping quantization
  // to avoid if-branch in the following SIMD processing
  if (transformation) fscale = 1.0f;

  const uint32_t height = this->size.y;
  const uint32_t stride = this->band_stride;
  const int32_t pshift  = (refsegment) ? 1 : 0;
  const int32_t pLSB    = (refsegment) ? 2 : 1;

  for (uint16_t i = 0; i < static_cast<uint16_t>(height); ++i) {
    sprec_t *sp        = this->i_samples + i * stride;
    int32_t *dp        = this->sample_buf.get() + i * blksampl_stride;
    size_t block_index = (i + 1U) * (blkstate_stride) + 1U;
    uint8_t *dstblk    = block_states.get() + block_index;

    float32x4_t vscale = vdupq_n_f32(fscale);
    auto vorval        = vdupq_n_s32(0);
    int32x4_t vpLSB    = vdupq_n_s32(pLSB);
    int32x4_t vone     = vdupq_n_s32(1);

    int16_t len = static_cast<int16_t>(this->size.x);
    for (; len >= 8; len -= 8) {
      int16x8_t coeff16 = vld1q_s16(sp);
      int32x4_t v0      = vmovl_s16(vget_low_s16(coeff16));
      int32x4_t v1      = vmovl_high_s16(coeff16);
      // Quantization
      v0 = vcvtq_s32_f32(vmulq_f32(vcvtq_f32_s32(v0), vscale));
      v1 = vcvtq_s32_f32(vmulq_f32(vcvtq_f32_s32(v1), vscale));
      // Take sign bit
      int32x4_t s0 = vandq_s32(vshrq_n_s32(v0, 31), vone);
      int32x4_t s1 = vandq_s32(vshrq_n_s32(v1, 31), vone);
      // Absolute value
      v0           = vabsq_s32(v0);
      v1           = vabsq_s32(v1);
      int32x4_t z0 = vandq_s32(v0, vpLSB);  // only for SigProp and MagRef
      int32x4_t z1 = vandq_s32(v1, vpLSB);  // only for SigProp and MagRef
      // Down-shift if other than HT Cleanup pass exists
      v0 = v0 >> pshift;
      v1 = v1 >> pshift;
      // Generate masks for sigma
      int32x4_t mask0 = vcgtzq_s32(v0);
      int32x4_t mask1 = vcgtzq_s32(v1);
      // Check emptiness of a block
      vorval = vorrq_s32(vorval, v0);
      vorval = vorrq_s32(vorval, v1);
      // Convert two's compliment to MagSgn form
      int32x4_t vone0 = vandq_s32(mask0, vone);
      int32x4_t vone1 = vandq_s32(mask1, vone);
      v0              = vsubq_u32(v0, vone0);
      v1              = vsubq_u32(v1, vone1);
      v0              = vshlq_n_s32(v0, 1);
      v1              = vshlq_n_s32(v1, 1);
      v0              = vaddq_s32(v0, vandq_s32(s0, mask0));
      v1              = vaddq_s32(v1, vandq_s32(s1, mask1));
      // Store
      vst1q_s32(dp, v0);
      vst1q_s32(dp + 4, v1);
      sp += 8;
      dp += 8;
      // for Block states
      uint8x8_t vblkstate = vdup_n_u8(0);
      vblkstate |= vmovn_s16(vandq_s16(vcombine_s16(vmovn_s32(mask0), vmovn_s32(mask1)), vdupq_n_s16(1)));
      // bits in lowest bitplane, only for SigProp and MagRef TODO: test this line
      vblkstate |= vmovn_s16(
          vshlq_n_s16(vandq_s16(vcombine_s16(vmovn_s32(z0), vmovn_s32(z1)), vdupq_n_s16(1)), SHIFT_SMAG));
      // sign-bits, only for SigProp and MagRef  TODO: test this line
      vblkstate |= vmovn_s16(
          vshlq_n_s16(vandq_s16(vcombine_s16(vmovn_s32(s0), vmovn_s32(s1)), vdupq_n_s16(1)), SHIFT_SSGN));
      //      uint8x8_t vblkstate = vget_low_u8(vld1q_u8(dstblk + j));
      //      uint16x8_t vsign = vcltzq_s16(coeff16) >> 15;
      //      uint8x8_t vsmag  = vmovn_u16(vandq_s16(coeff16, vpLSB));
      //      uint8x8_t vssgn  = vmovn_u16(vsign);
      //      vblkstate |= vsmag << SHIFT_SMAG;
      //      vblkstate |= vssgn << SHIFT_SSGN;
      //      int16x8_t vabsmag     = (vabsq_s16(coeff16) & 0x7FFF) >> pshift;
      //      vzero                 = vorrq_s16(vzero, vabsmag);
      //      int16x8_t vmasked_one = (vceqzq_s16(vabsmag) ^ 0xFFFF) & vone;
      //      vblkstate |= vmovn_u16(vmasked_one);

      vst1_u8(dstblk, vblkstate);
      dstblk += 8;
    }
    // Check emptiness of a block
    or_val |= static_cast<unsigned int>(vmaxvq_s16(vorval));
    // process leftover
    for (; len > 0; --len) {
      int32_t temp;
      temp = static_cast<int32_t>(static_cast<float>(sp[0]) * fscale);  // needs to be rounded towards zero
      uint32_t sign = static_cast<uint32_t>(temp) & 0x80000000;
      dstblk[0] |= static_cast<uint8_t>(((temp & pLSB) & 1) << SHIFT_SMAG);
      dstblk[0] |= static_cast<uint8_t>((sign >> 31) << SHIFT_SSGN);
      temp = (temp < 0) ? -temp : temp;
      temp &= 0x7FFFFFFF;
      temp >>= pshift;
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
 * state_MS_enc: member functions
 *******************************************************************************/
void state_MS_enc::emitMagSgnBits(uint32_t cwd, uint8_t len, uint8_t emb_1) {
  int32_t temp = emb_1 << len;
  cwd -= static_cast<uint32_t>(temp);
  //  auto v0 = vld1_u64(&Creg);
  //  v0 = vorr_u64(v0, vdup_n_u64(cwd) << ctreg);
  //  vst1_u64(&Creg, v0);
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

int32_t state_MS_enc::termMS() {
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
auto make_storage = [](const j2k_codeblock *const block, const uint16_t qy, const uint16_t qx,
                       int32x4_t &sig0, int32x4_t &sig1, uint32_t *const v_n, int32x4_t &E0, int32x4_t &E1,
                       uint8_t *const rho_q) {
  // This function shall be called on the assumption that there are two quads
  alignas(32) const int8_t nshift[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  uint8_t *const ssp0 =
      block->block_states.get() + (2U * qy + 1U) * (block->blkstate_stride) + 2U * qx + 1U;
  uint8_t *const ssp1 = ssp0 + block->blkstate_stride;
  int32_t *sp0        = block->sample_buf.get() + 2U * (qx + qy * block->blksampl_stride);
  int32_t *sp1        = sp0 + block->blksampl_stride;
  auto v_u8_zip       = vzip1_u8(vld1_u8(ssp0), vld1_u8(ssp1));
  auto vmask          = vdup_n_u8(1);
  auto v_u8_out       = vand_u8(v_u8_zip, vmask);
  //  vst1_u8(sigma_n, v_u8_out);
  auto v_u8_shift = vld1_s8(nshift);
  auto vtmp       = vshl_u8(v_u8_out, v_u8_shift);
  rho_q[0]        = vaddv_u8(vtmp) & 0xF;
  rho_q[1]        = vaddv_u8(vtmp) >> 4;
  auto v0         = vld1q_s32(sp0);
  auto v1         = vld1q_s32(sp1);
  auto v          = vzipq_s32(v0, v1);
  vst1q_u32(v_n, v.val[0]);
  vst1q_u32(v_n + 4, v.val[1]);
  sig0     = vmovl_u16(vget_low_u16(vmovl_u8(v_u8_out)));
  sig1     = vmovl_u16(vget_high_u16(vmovl_u8(v_u8_out)));
  v.val[0] = vaddq_s32(vshlq_n_s32(vshrq_n_s32(v.val[0], 1), 1), vdupq_n_s32(1));
  E0       = vsubq_u32(vdupq_n_s32(32), vclzq_u32(v.val[0])) * sig0;
  v.val[1] = vaddq_s32(vshlq_n_s32(vshrq_n_s32(v.val[1], 1), 1), vdupq_n_s32(1));
  E1       = vsubq_u32(vdupq_n_s32(32), vclzq_u32(v.val[1])) * sig1;
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
      v_n[i] =
          static_cast<uint32_t>(block->sample_buf[(size_t)x[i] + (size_t)y[i] * (block->blksampl_stride)]);
    } else {
      v_n[i] = 0;
    }
  }

  for (int i = 0; i < 4; ++i) {
    E_n[i] = static_cast<int32_t>((32 - count_leading_zeros(((v_n[i] >> 1) << 1) + 1)) * sigma_n[i]);
  }
}

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

  block->quantize(or_val);

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
  alignas(32) uint8_t sigma_n[8] = {0}, rho_q[2] = {0}, m_n[8] = {0};
  alignas(32) int32_t E_n[8] = {0}, U_q[2] = {0}, emb[2] = {0};

  int32x4_t vshift = {0, 1, 2, 3};
  // Initial line pair
  alignas(32) auto Eline   = MAKE_UNIQUE<int32_t[]>(2U * QW + 6U);
  Eline[0]                 = 0;
  auto E_p                 = Eline.get() + 1;
  alignas(32) auto rholine = MAKE_UNIQUE<int32_t[]>(QW + 3U);
  rholine[0]               = 0;
  auto rho_p               = rholine.get() + 1;

  uint8_t lw, gamma;
  uint16_t context = 0, n_q, CxtVLC, cwd;
  int32x4_t E0, E1, sig0, sig1, Etmp, vuoff, mask;
  int32_t Emax_q;
  int32_t u_q, uoff, u_min, uvlc_idx, kappa = 1;
  int32_t emb_pattern, embk_0, embk_1, emb1_0, emb1_1;
  for (uint16_t qx = 0; qx < QW - 1; qx = static_cast<uint16_t>(qx + 2U)) {
    const int16_t qy = 0;
    bool uoff_flag   = true;

    // MAKE_STORAGE()
    make_storage(block, qy, qx, sig0, sig1, v_n, E0, E1, rho_q);
    // update Eline
    vst1q_s32(E_p, vzip2q_s32(vzip1q_s32(E0, E1), vzip2q_s32(E0, E1)));
    E_p += 4;
    // MEL encoding for the first quad
    if (context == 0) {
      MEL_encoder.encodeMEL((rho_q[Q0] != 0));
    }
    // calculate u_off values
    Emax_q   = vmaxvq_s32(E0);
    U_q[Q0]  = std::max(Emax_q, kappa);
    u_q      = U_q[Q0] - kappa;
    u_min    = u_q;
    uvlc_idx = u_q;
    uoff     = (u_q) ? 1 : 0;
    uoff_flag &= uoff;
    Etmp        = vdupq_n_s32(Emax_q);
    vuoff       = vdupq_n_s32(uoff);
    mask        = vceqq_s32(E0, Etmp);
    emb_pattern = vaddvq_s32(vshlq_s32(vuoff, vshift) & mask);
    n_q         = static_cast<uint16_t>(emb_pattern + (rho_q[Q0] << 4) + (context << 8));
    // VLC encoding
    CxtVLC = enc_CxtVLC_table0[n_q];
    embk_0 = CxtVLC & 0xF;
    emb1_0 = emb_pattern & embk_0;
    lw     = (CxtVLC >> 4) & 0x07;
    cwd    = static_cast<uint16_t>(CxtVLC >> 7);
    VLC_encoder.emitVLCBits(cwd, lw);

    // context for the next quad
    context = (rho_q[Q0] >> 1) | (rho_q[Q0] & 0x1);

    Emax_q  = vmaxvq_s32(E1);
    U_q[Q1] = std::max(Emax_q, kappa);
    u_q     = U_q[Q1] - kappa;
    u_min   = (u_min < u_q) ? u_min : u_q;
    uvlc_idx += u_q << 5;
    uoff = (u_q) ? 1 : 0;
    uoff_flag &= uoff;
    Etmp        = vdupq_n_s32(Emax_q);
    vuoff       = vdupq_n_s32(uoff);
    mask        = vceqq_s32(E1, Etmp);
    emb_pattern = vaddvq_s32(vshlq_s32(vuoff, vshift) & mask);
    n_q         = static_cast<uint16_t>(emb_pattern + (rho_q[Q1] << 4) + (context << 8));
    // VLC encoding
    CxtVLC = enc_CxtVLC_table0[n_q];
    embk_1 = CxtVLC & 0xF;
    emb1_1 = emb_pattern & embk_1;
    lw     = (CxtVLC >> 4) & 0x07;
    cwd    = static_cast<uint16_t>(CxtVLC >> 7);
    VLC_encoder.emitVLCBits(cwd, lw);

    // UVLC encoding
    int32_t tmp = static_cast<int32_t>(enc_UVLC_table0[uvlc_idx]);
    lw          = static_cast<uint8_t>(tmp & 0xFF);
    cwd         = static_cast<uint16_t>(tmp >> 8);
    VLC_encoder.emitVLCBits(cwd, lw);

    // MEL encoding of the second quad
    if (context == 0) {
      if (rho_q[Q1] != 0) {
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
    int32_t m32_n[4];
    int32x4_t vshift2 = {0, -1, -2, -3};
    vst1q_s32(m32_n, vsubq_s32(vmulq_s32(sig0, vdupq_n_s32(U_q[0])),
                               vandq_s32(vshlq_s32(vdupq_n_s32(embk_0), vshift2), vdupq_n_s32(1))));
    for (int i = 0; i < 4; ++i) {
      MagSgn_encoder.emitMagSgnBits(v_n[i], (uint8_t)m32_n[i], (emb1_0 >> i) & 1);
    }
    vst1q_s32(m32_n, vsubq_s32(vmulq_s32(sig1, vdupq_n_s32(U_q[1])),
                               vandq_s32(vshlq_s32(vdupq_n_s32(embk_1), vshift2), vdupq_n_s32(1))));
    for (int i = 0; i < 4; ++i) {
      MagSgn_encoder.emitMagSgnBits(v_n[4 + i], (uint8_t)m32_n[i], (emb1_1 >> i) & 1);
    }

    // context for the next quad
    context = (rho_q[Q1] >> 1) | (rho_q[Q1] & 0x1);
    // update rho_line
    *rho_p++ = rho_q[0];
    *rho_p++ = rho_q[1];
  }
  if (QW & 1) {
    uint16_t qx = static_cast<uint16_t>(QW - 1);
    make_storage_one(block, 0, qx, sigma_n, v_n, E_n, rho_q);
    E0     = vld1q_s32(E_n);
    *E_p++ = E_n[1];
    *E_p++ = E_n[3];

    // MEL encoding
    if (context == 0) {
      MEL_encoder.encodeMEL((rho_q[Q0] != 0));
    }

    Emax_q   = vmaxvq_s32(E0);
    U_q[Q0]  = std::max(Emax_q, kappa);
    u_q      = U_q[Q0] - kappa;
    uvlc_idx = u_q;
    uoff     = (u_q) ? 1 : 0;

    Etmp        = vdupq_n_s32(Emax_q);
    vuoff       = vdupq_n_s32(uoff);
    mask        = vceqq_s32(E0, Etmp);
    emb_pattern = vaddvq_s32(vshlq_s32(vuoff, vshift) & mask);
    n_q         = static_cast<uint16_t>(emb_pattern + (rho_q[Q0] << 4) + (context << 8));
    // VLC encoding
    CxtVLC = enc_CxtVLC_table0[n_q];
    embk_0 = CxtVLC & 0xF;
    emb1_0 = emb_pattern & embk_0;
    lw     = (CxtVLC >> 4) & 0x07;
    cwd    = static_cast<uint16_t>(CxtVLC >> 7);
    VLC_encoder.emitVLCBits(cwd, lw);
    // UVLC encoding
    int32_t tmp = static_cast<int32_t>(enc_UVLC_table0[uvlc_idx]);
    lw          = static_cast<uint8_t>(tmp & 0xFF);
    cwd         = static_cast<uint16_t>(tmp >> 8);
    VLC_encoder.emitVLCBits(cwd, lw);

    for (int i = 0; i < 4; ++i) {
      m_n[i] = static_cast<uint8_t>(sigma_n[i] * U_q[Q0] - ((embk_0 >> i) & 1));
      MagSgn_encoder.emitMagSgnBits(v_n[i], m_n[i], (emb1_0 >> i) & 1);
    }
    // update rho_line
    *rho_p++ = rho_q[0];
  }

  /*******************************************************************************************************************/
  // Non-initial line-pair
  /*******************************************************************************************************************/
  int32_t Emax0, Emax1;
  for (uint16_t qy = 1; qy < QH; qy++) {
    E_p      = Eline.get() + 1;
    rho_p    = rholine.get() + 1;
    rho_q[1] = 0;

    Emax0 = find_max(E_p[-1], E_p[0], E_p[1], E_p[2]);
    Emax1 = find_max(E_p[1], E_p[2], E_p[3], E_p[4]);

    // calculate context for the next quad
    context = static_cast<uint16_t>(((rho_q[1] & 0x4) << 7) | ((rho_q[1] & 0x8) << 6));  // (w | sw) << 9
    context |= ((rho_p[-1] & 0x8) << 5) | ((rho_p[0] & 0x2) << 7);                       // (nw | n) << 8
    context |= ((rho_p[0] & 0x8) << 7) | ((rho_p[1] & 0x2) << 9);                        // (ne | nf) << 10
    for (uint16_t qx = 0; qx < QW - 1; qx = static_cast<uint16_t>(qx + 2)) {
      make_storage(block, qy, qx, sig0, sig1, v_n, E0, E1, rho_q);
      // MEL encoding of the first quad
      if (context == 0) {
        MEL_encoder.encodeMEL((rho_q[Q0] != 0));
      }
      gamma       = (popcount32((uint32_t)rho_q[Q0]) > 1) ? 1 : 0;
      kappa       = std::max((Emax0 - 1) * gamma, 1);
      Emax_q      = vmaxvq_s32(E0);
      U_q[Q0]     = std::max(Emax_q, kappa);
      u_q         = U_q[Q0] - kappa;
      uvlc_idx    = u_q;
      uoff        = (u_q) ? 1 : 0;
      Etmp        = vdupq_n_s32(Emax_q);
      vuoff       = vdupq_n_s32(uoff);
      mask        = vceqq_s32(E0, Etmp);
      emb_pattern = vaddvq_s32(vshlq_s32(vuoff, vshift) & mask);
      n_q         = static_cast<uint16_t>(emb_pattern + (rho_q[Q0] << 4) + (context << 0));
      // VLC encoding
      CxtVLC = enc_CxtVLC_table1[n_q];
      embk_0 = CxtVLC & 0xF;
      emb1_0 = emb_pattern & embk_0;
      lw     = (CxtVLC >> 4) & 0x07;
      cwd    = static_cast<uint16_t>(CxtVLC >> 7);
      VLC_encoder.emitVLCBits(cwd, lw);

      // calculate context for the next quad
      context = static_cast<uint16_t>(((rho_q[0] & 0x4) << 7) | ((rho_q[0] & 0x8) << 6));  // (w | sw) << 9
      context |= ((rho_p[0] & 0x8) << 5) | ((rho_p[1] & 0x2) << 7);                        // (nw | n) << 8
      context |= ((rho_p[1] & 0x8) << 7) | ((rho_p[2] & 0x2) << 9);  // (ne | nf) << 10
      // MEL encoding of the second quad
      if (context == 0) {
        MEL_encoder.encodeMEL((rho_q[Q1] != 0));
      }
      gamma   = (popcount32((uint32_t)rho_q[Q1]) > 1) ? 1 : 0;
      kappa   = std::max((Emax1 - 1) * gamma, 1);
      Emax_q  = vmaxvq_s32(E1);
      U_q[Q1] = std::max(Emax_q, kappa);
      u_q     = U_q[Q1] - kappa;
      uvlc_idx += u_q << 5;
      uoff        = (u_q) ? 1 : 0;
      Etmp        = vdupq_n_s32(Emax_q);
      vuoff       = vdupq_n_s32(uoff);
      mask        = vceqq_s32(E1, Etmp);
      emb_pattern = vaddvq_s32(vshlq_s32(vuoff, vshift) & mask);
      n_q         = static_cast<uint16_t>(emb_pattern + (rho_q[Q1] << 4) + (context << 0));
      // VLC encoding
      CxtVLC = enc_CxtVLC_table1[n_q];
      embk_1 = CxtVLC & 0xF;
      emb1_1 = emb_pattern & embk_1;
      lw     = (CxtVLC >> 4) & 0x07;
      cwd    = static_cast<uint16_t>(CxtVLC >> 7);
      VLC_encoder.emitVLCBits(cwd, lw);
      // UVLC encoding
      int32_t tmp = static_cast<int32_t>(enc_UVLC_table1[uvlc_idx]);
      lw          = static_cast<uint8_t>(tmp & 0xFF);
      cwd         = static_cast<uint16_t>(tmp >> 8);
      VLC_encoder.emitVLCBits(cwd, lw);

      // MagSgn encoding
      for (int i = 0; i < 4; ++i) {
        m_n[i] = static_cast<uint8_t>(sig0[i] * U_q[Q0] - ((embk_0 >> i) & 1));
        MagSgn_encoder.emitMagSgnBits(v_n[i], m_n[i], (emb1_0 >> i) & 1);
      }
      for (int i = 0; i < 4; ++i) {
        m_n[4 + i] = static_cast<uint8_t>(sig1[i] * U_q[Q1] - ((embk_1 >> i) & 1));
        MagSgn_encoder.emitMagSgnBits(v_n[4 + i], m_n[4 + i], (emb1_1 >> i) & 1);
      }

      Emax0 = find_max(E_p[3], E_p[4], E_p[5], E_p[6]);
      Emax1 = find_max(E_p[5], E_p[6], E_p[7], E_p[8]);
      vst1q_s32(E_p, vzip2q_s32(vzip1q_s32(E0, E1), vzip2q_s32(E0, E1)));
      E_p += 4;

      // calculate context for the next quad
      context = ((rho_q[1] & 0x4) << 7) | ((rho_q[1] & 0x8) << 6);   // (w | sw) << 9
      context |= ((rho_p[1] & 0x8) << 5) | ((rho_p[2] & 0x2) << 7);  // (nw | n) << 8
      context |= ((rho_p[2] & 0x8) << 7) | ((rho_p[3] & 0x2) << 9);  // (ne | nf) << 10

      *rho_p++ = rho_q[0];
      *rho_p++ = rho_q[1];
    }
    if (QW & 1) {
      uint16_t qx = static_cast<uint16_t>(QW - 1);

      make_storage_one(block, qy, qx, sigma_n, v_n, E_n, rho_q);
      E0     = vld1q_s32(E_n);
      *E_p++ = E_n[1];
      *E_p++ = E_n[3];

      // MEL encoding of the first quad
      if (context == 0) {
        MEL_encoder.encodeMEL((rho_q[Q0] != 0));
      }

      gamma    = (popcount32((uint32_t)rho_q[Q0]) > 1) ? 1 : 0;
      kappa    = std::max((Emax0 - 1) * gamma, 1);
      Emax_q   = find_max(E_n[0], E_n[1], E_n[2], E_n[3]);
      U_q[Q0]  = std::max(Emax_q, kappa);
      u_q      = U_q[Q0] - kappa;
      uvlc_idx = u_q;
      uoff     = (u_q) ? 1 : 0;

      Etmp        = vdupq_n_s32(Emax_q);
      vuoff       = vdupq_n_s32(uoff);
      mask        = vceqq_s32(E0, Etmp);
      emb_pattern = vaddvq_s32(vshlq_s32(vuoff, vshift) & mask);
      n_q         = static_cast<uint16_t>(emb_pattern + (rho_q[Q0] << 4) + (context << 0));
      // VLC encoding
      CxtVLC = enc_CxtVLC_table1[n_q];
      embk_0 = CxtVLC & 0xF;
      emb1_0 = emb_pattern & embk_0;
      lw     = (CxtVLC >> 4) & 0x07;
      cwd    = static_cast<uint16_t>(CxtVLC >> 7);
      VLC_encoder.emitVLCBits(cwd, lw);
      // UVLC encoding
      int32_t tmp = static_cast<int32_t>(enc_UVLC_table1[uvlc_idx]);
      lw          = static_cast<uint8_t>(tmp & 0xFF);
      cwd         = static_cast<uint16_t>(tmp >> 8);
      VLC_encoder.emitVLCBits(cwd, lw);

      for (int i = 0; i < 4; ++i) {
        m_n[i] = static_cast<uint8_t>(sigma_n[i] * U_q[Q0] - ((embk_0 >> i) & 1));
        MagSgn_encoder.emitMagSgnBits(v_n[i], m_n[i], (emb1_0 >> i) & 1);
      }
      // update rho_line
      *rho_p++ = rho_q[0];
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
#endif