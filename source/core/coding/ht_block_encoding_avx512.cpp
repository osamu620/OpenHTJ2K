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

#if defined(OPENHTJ2K_TRY_AVX2) && defined(__AVX512F__)
  #include <algorithm>
  #include <cmath>
  #include "coding_units.hpp"
  #include "ht_block_encoding_avx2.hpp"
  #include "coding_local.hpp"
  #include "enc_CxtVLC_tables.hpp"
  #include "utils.hpp"

  #define Q0 0
  #define Q1 1

// Uncomment for experimental use of HT SigProp and MagRef encoding (does not work)
//#define ENABLE_SP_MR

// Quantize DWT coefficients and transfer them to codeblock buffer in a form of MagSgn value
void j2k_codeblock::quantize(uint32_t &or_val) {
  const uint32_t height  = this->size.y;
  const uint32_t width   = this->size.x;
  const uint32_t stride  = this->band_stride;
  const bool lossless    = (this->transformation != 0);

  float fscale = 1.0f;
  if (!lossless) {
    fscale = 1.0f / this->stepsize;
    fscale /= (1 << (FRACBITS));
  }

  #if defined(ENABLE_SP_MR)
  const int32_t pshift = (refsegment) ? 1 : 0;
  const int32_t pLSB   = (refsegment) ? 1 : 1;
  #endif
  const __m256i vone  = _mm256_set1_epi32(1);
  const __m256 vscale = _mm256_set1_ps(fscale);
  __m256i vor_val = _mm256_setzero_si256();
  for (uint16_t i = 0; i < static_cast<uint16_t>(height); ++i) {
    sprec_t *sp        = this->i_samples + i * stride;
    int32_t *dp        = this->sample_buf + i * blksampl_stride;
    size_t block_index = (i + 1U) * (blkstate_stride) + 1U;
    uint8_t *dstblk    = block_states + block_index;
  #if defined(ENABLE_SP_MR)
    const __m256i vpLSB = _mm256_set1_epi32(pLSB);
  #endif
    int32_t len = static_cast<int32_t>(width);
    for (; len >= 16; len -= 16) {
      __m256i v0, v1;
      if (lossless) {
        v0 = _mm256_cvttps_epi32(_mm256_loadu_ps(sp));
        v1 = _mm256_cvttps_epi32(_mm256_loadu_ps(sp + 8));
      } else {
        v0 = _mm256_cvttps_epi32(_mm256_mul_ps(_mm256_loadu_ps(sp), vscale));
        v1 = _mm256_cvttps_epi32(_mm256_mul_ps(_mm256_loadu_ps(sp + 8), vscale));
      }
      // Take sign bit
      __m256i s0 = _mm256_srli_epi32(v0, 31);
      __m256i s1 = _mm256_srli_epi32(v1, 31);
      v0         = _mm256_abs_epi32(v0);
      v1         = _mm256_abs_epi32(v1);
  #if defined(ENABLE_SP_MR)
      __m256i z0 = _mm256_and_si256(v0, vpLSB);  // only for SigProp and MagRef
      __m256i z1 = _mm256_and_si256(v1, vpLSB);  // only for SigProp and MagRef

      // Down-shift if other than HT Cleanup pass exists
      v0 = _mm256_srai_epi32(v0, pshift);
      v1 = _mm256_srai_epi32(v1, pshift);
  #endif
      // Generate masks for sigma
      __m256i mask0 = _mm256_cmpgt_epi32(v0, _mm256_setzero_si256());
      __m256i mask1 = _mm256_cmpgt_epi32(v1, _mm256_setzero_si256());
      // Accumulate or_val in a vector register; no scalar dependency in hot loop
      vor_val = _mm256_or_si256(vor_val, mask0);
      vor_val = _mm256_or_si256(vor_val, mask1);

      // Convert two's compliment to MagSgn form
      __m256i vone0 = _mm256_and_si256(mask0, vone);
      __m256i vone1 = _mm256_and_si256(mask1, vone);
      v0            = _mm256_sub_epi32(v0, vone0);
      v1            = _mm256_sub_epi32(v1, vone1);
      v0            = _mm256_slli_epi32(v0, 1);
      v1            = _mm256_slli_epi32(v1, 1);
      v0            = _mm256_add_epi32(v0, _mm256_and_si256(s0, mask0));
      v1            = _mm256_add_epi32(v1, _mm256_and_si256(s1, mask1));
      // Store
      _mm256_storeu_si256((__m256i *)dp, v0);
      _mm256_storeu_si256((__m256i *)(dp + 8), v1);
      sp += 16;
      dp += 16;
      // for Block states
      v0 = _mm256_packs_epi32(vone0, vone1);  // re-use v0 as sigma
      v0 = _mm256_permute4x64_epi64(v0, 0xD8);
  #if defined(ENABLE_SP_MR)
      vone0 = _mm256_packs_epi32(z0, z1);  // re-use vone0 as z
      vone0 = _mm256_permute4x64_epi64(vone0, 0xD8);
      vone1 = _mm256_packs_epi32(s0, s1);  // re-use vone1 as sign
      vone1 = _mm256_permute4x64_epi64(vone1, 0xD8);
      v0    = _mm256_or_si256(v0, _mm256_slli_epi16(vone0, SHIFT_SMAG));
      v0    = _mm256_or_si256(v0, _mm256_slli_epi16(vone1, SHIFT_SSGN));
  #endif
      v0        = _mm256_packs_epi16(v0, v0);  // re-use vone0
      v0        = _mm256_permute4x64_epi64(v0, 0xD8);
      __m128i v = _mm256_extracti128_si256(v0, 0);
      // _mm256_zeroupper(); // does not work on GCC, TODO: find a solution with __m128i v
      _mm_storeu_si128((__m128i *)dstblk, v);
      dstblk += 16;
    }
    for (; len > 0; --len) {
      int32_t temp;
      if (lossless)
        temp = static_cast<int32_t>(sp[0]);
      else
        temp = static_cast<int32_t>(static_cast<float>(sp[0]) * fscale);
      uint32_t sign = static_cast<uint32_t>(temp) & 0x80000000;
  #if defined(ENABLE_SP_MR)
      dstblk[0] |= static_cast<uint8_t>(((temp & pLSB) & 1) << SHIFT_SMAG);
      dstblk[0] |= static_cast<uint8_t>((sign >> 31) << SHIFT_SSGN);
  #endif
      temp = (temp < 0) ? -temp : temp;
      temp &= 0x7FFFFFFF;
  #if defined(ENABLE_SP_MR)
      temp >>= pshift;
  #endif
      if (temp) {
        or_val |= 1;
        dstblk[0] |= 1;
        temp--;
        temp <<= 1;
        temp += static_cast<uint8_t>(sign >> 31);
      }
      dp[0] = temp;
      ++sp;
      ++dp;
      ++dstblk;
    }
    if (blksampl_stride > width)
      memset(dp, 0, (blksampl_stride - width) * sizeof(int32_t));
  }
  const uint32_t QHx2 = (height + 7U) & ~7U;
  for (uint32_t i = height; i < QHx2; ++i)
    memset(this->sample_buf + i * blksampl_stride, 0, blksampl_stride * sizeof(int32_t));
  if (!_mm256_testz_si256(vor_val, vor_val)) {
    or_val |= 1;
  }
}

/********************************************************************************
 * HT cleanup encoding: helper functions
 *******************************************************************************/

// https://stackoverflow.com/a/58827596
inline __m128i sse_lzcnt_epi32(__m128i v) {
  // prevent value from being rounded up to the next power of two
  v = _mm_andnot_si128(_mm_srli_epi32(v, 8), v);  // keep 8 MSB

  v = _mm_castps_si128(_mm_cvtepi32_ps(v));    // convert an integer to float
  v = _mm_srli_epi32(v, 23);                   // shift down the exponent
  v = _mm_subs_epu16(_mm_set1_epi32(158), v);  // undo bias
  v = _mm_min_epi16(v, _mm_set1_epi32(32));    // clamp at 32

  return v;
}

auto make_storage = [](const uint8_t *ssp0, const uint8_t *ssp1, const int32_t *sp0, const int32_t *sp1,
                       __m128i &sig0, __m128i &sig1, __m128i &v0, __m128i &v1, __m128i &E0, __m128i &E1,
                       int32_t &rho0, int32_t &rho1) {
  // This function shall be called on the assumption that there are two quads
  const __m128i zero = _mm_setzero_si128();
  __m128i t0         = _mm_set1_epi64x(*((int64_t *)ssp0));
  __m128i t1         = _mm_set1_epi64x(*((int64_t *)ssp1));
  __m128i t          = _mm_unpacklo_epi8(t0, t1);
  __m128i v_u8_out   = _mm_and_si128(t, _mm_set1_epi8(1));
  v_u8_out           = _mm_cmpgt_epi8(v_u8_out, zero);
  sig0               = _mm_cvtepu8_epi32(v_u8_out);
  sig1               = _mm_cvtepu8_epi32(_mm_srli_si128(v_u8_out, 4));
  rho0               = _mm_movemask_epi8(_mm_packus_epi16(_mm_packus_epi32(sig0, zero), zero));
  rho1               = _mm_movemask_epi8(_mm_packus_epi16(_mm_packus_epi32(sig1, zero), zero));

  sig0 = _mm_cmpgt_epi32(sig0, zero);
  sig1 = _mm_cmpgt_epi32(sig1, zero);

  t0 = _mm_loadu_si128((__m128i *)sp0);
  t1 = _mm_loadu_si128((__m128i *)sp1);
  v0 = _mm_unpacklo_epi32(t0, t1);
  v1 = _mm_unpackhi_epi32(t0, t1);

  t0 = _mm_sub_epi32(_mm_set1_epi32(32), sse_lzcnt_epi32(v0));
  E0 = _mm_and_si128(t0, sig0);
  t1 = _mm_sub_epi32(_mm_set1_epi32(32), sse_lzcnt_epi32(v1));
  E1 = _mm_and_si128(t1, sig1);
};

auto make_storage_one = [](const uint8_t *ssp0, const uint8_t *ssp1, const int32_t *sp0, const int32_t *sp1,
                           __m128i &sig0, __m128i &v0, __m128i &E0, int32_t &rho0) {
  sig0 = _mm_setr_epi32(ssp0[0] & 1, ssp1[0] & 1, ssp0[1] & 1, ssp1[1] & 1);

  __m128i shift = _mm_setr_epi32(7, 7, 7, 7);
  __m128i t0    = _mm_sllv_epi32(sig0, shift);
  __m128i zero  = _mm_setzero_si128();
  rho0          = _mm_movemask_epi8(_mm_packus_epi16(_mm_packus_epi32(t0, zero), zero));

  v0 = _mm_setr_epi32(sp0[0], sp1[0], sp0[1], sp1[1]);

  sig0 = _mm_cmpgt_epi32(sig0, zero);
  t0   = _mm_sub_epi32(_mm_set1_epi32(32), sse_lzcnt_epi32(v0));
  E0   = _mm_and_si128(t0, sig0);
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

  if (!block->pre_quantized)
    block->quantize(or_val);
  else
    or_val = block->pre_or_val;

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

  // Thread-local scratch buffers: one allocation per thread for the lifetime of the program.
  // Must be zeroed before each codeblock (encoders rely on zero-initialized state).
  // Thread-local scratch buffers: one allocation per thread for the lifetime of the program.
  // All positions written before they are read; no zeroing needed between codeblocks.
  alignas(32) static thread_local uint8_t fwd_buf[MAX_Lcup];
  alignas(32) static thread_local uint8_t rev_buf[MAX_Scup];

  state_MS_enc MagSgn_encoder(fwd_buf);
  state_MEL_enc MEL_encoder(rev_buf);
  state_VLC_enc VLC_encoder(rev_buf);

  int32_t rho0, rho1, U0, U1;

  // Deferred MagSgn: store v/m/known1 per quad, emit in a tight loop after all
  // context/VLC/MEL work for the line pair is done.  This keeps the MagSgn
  // accumulator hot in icache and improves branch-predictor utilization.
  alignas(32) __m128i ms_v[512], ms_m[512], ms_k1[512];
  int32_t ms_count;

  /*******************************************************************************************************************/
  // Initial line-pair
  /*******************************************************************************************************************/
  uint8_t *ssp0 = block->block_states + 1U * (block->blkstate_stride) + 1U;
  uint8_t *ssp1 = ssp0 + block->blkstate_stride;
  int32_t *sp0  = block->sample_buf;
  int32_t *sp1  = sp0 + block->blksampl_stride;

  // Stack-allocate Eline/rholine: bounded by max codeblock width (1024 → QW ≤ 512).
  // Zero exactly the used range (matches make_unique<int32_t[]> zero-initialization).
  alignas(32) int32_t Eline[2 * 512 + 6];
  std::fill_n(Eline, 2U * QW + 6U, int32_t{0});
  int32_t *E_p = Eline + 1;
  alignas(32) int32_t rholine[512 + 3];
  std::fill_n(rholine, QW + 3U, int32_t{0});
  int32_t *rho_p = rholine + 1;

  int32_t context = 0, n_q;
  uint32_t CxtVLC, lw, cwd;
  int32_t Emax_q;
  int32_t u_q, uoff, u_min, uvlc_idx, kappa = 1;
  const __m128i vshift = _mm_setr_epi32(0, 1, 2, 3);
  const __m128i vone   = _mm_set1_epi32(1);
  int32_t emb_pattern, embk_0, embk_1, emb1_0, emb1_1;
  __m128i sig0, sig1, v0, v1, E0, E1, m0, m1, known1_0, known1_1;
  __m128i Etmp, vuoff, mask, vtmp;

  ms_count  = 0;
  int32_t qx = QW;
  for (; qx >= 2; qx -= 2) {
    bool uoff_flag = true;
    make_storage(ssp0, ssp1, sp0, sp1, sig0, sig1, v0, v1, E0, E1, rho0, rho1);
    // MEL encoding for the first quad
    if (context == 0) {
      MEL_encoder.encodeMEL((rho0 != 0));
    }

    Emax_q   = hMax(E0);
    U0       = std::max((int32_t)Emax_q, kappa);
    u_q      = U0 - kappa;
    u_min    = u_q;
    uvlc_idx = u_q;
    uoff     = (u_q) ? 1 : 0;
    uoff_flag &= uoff;
    Etmp        = _mm_set1_epi32(Emax_q);
    vuoff       = _mm_set1_epi32(uoff << 7);
    mask        = _mm_cmpeq_epi32(E0, Etmp);
    vtmp        = _mm_and_si128(vuoff, mask);
    emb_pattern = _mm_movemask_epi8(
        _mm_packus_epi16(_mm_packus_epi32(vtmp, _mm_setzero_si128()), _mm_setzero_si128()));
    n_q = emb_pattern + (rho0 << 4) + (context << 8);
    // prepare VLC encoding of quad 0
    CxtVLC = enc_CxtVLC_table0[n_q];
    embk_0 = CxtVLC & 0xF;
    emb1_0 = emb_pattern & embk_0;
    uint32_t lw0 = (CxtVLC >> 4) & 0x07;
    uint32_t cwd0 = CxtVLC >> 7;

    // context for the next quad
    context = (rho0 >> 1) | (rho0 & 0x1);

    Emax_q = hMax(E1);
    U1     = std::max((int32_t)Emax_q, kappa);
    u_q    = U1 - kappa;
    u_min  = (u_min < u_q) ? u_min : u_q;
    uvlc_idx += u_q << 5;
    uoff = (u_q) ? 1 : 0;
    uoff_flag &= uoff;
    Etmp        = _mm_set1_epi32(Emax_q);
    vuoff       = _mm_set1_epi32(uoff << 7);
    mask        = _mm_cmpeq_epi32(E1, Etmp);
    vtmp        = _mm_and_si128(vuoff, mask);
    emb_pattern = _mm_movemask_epi8(
        _mm_packus_epi16(_mm_packus_epi32(vtmp, _mm_setzero_si128()), _mm_setzero_si128()));
    n_q = emb_pattern + (rho1 << 4) + (context << 8);
    CxtVLC = enc_CxtVLC_table0[n_q];
    embk_1 = CxtVLC & 0xF;
    emb1_1 = emb_pattern & embk_1;
    lw     = (CxtVLC >> 4) & 0x07;
    cwd    = CxtVLC >> 7;
    // Batched VLC + UVLC encoding
    uint32_t uvlc_tmp = enc_UVLC_table0[uvlc_idx];
    uint32_t uvlc_lw  = uvlc_tmp & 0xFF;
    uint32_t uvlc_cwd = uvlc_tmp >> 8;
    uint64_t vlc_all  = cwd0 | (static_cast<uint64_t>(cwd) << lw0)
                        | (static_cast<uint64_t>(uvlc_cwd) << (lw0 + lw));
    VLC_encoder.emitVLCBits(vlc_all, lw0 + lw + uvlc_lw);

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

    // Defer MagSgn encoding
    m0       = _mm_sub_epi32(_mm_and_si128(sig0, _mm_set1_epi32(U0)),
                             _mm_and_si128(_mm_srlv_epi32(_mm_set1_epi32(embk_0), vshift), vone));
    m1       = _mm_sub_epi32(_mm_and_si128(sig1, _mm_set1_epi32(U1)),
                             _mm_and_si128(_mm_srlv_epi32(_mm_set1_epi32(embk_1), vshift), vone));
    known1_0 = _mm_and_si128(_mm_srlv_epi32(_mm_set1_epi32(emb1_0), vshift), vone);
    known1_1 = _mm_and_si128(_mm_srlv_epi32(_mm_set1_epi32(emb1_1), vshift), vone);
    ms_v[ms_count] = v0; ms_m[ms_count] = m0; ms_k1[ms_count] = known1_0; ms_count++;
    ms_v[ms_count] = v1; ms_m[ms_count] = m1; ms_k1[ms_count] = known1_1; ms_count++;

    // context for the next quad
    context = (rho1 >> 1) | (rho1 & 0x1);
    // update rho_line
    *rho_p++ = rho0;
    *rho_p++ = rho1;
    // update Eline
    E0 = _mm_shuffle_epi32(E0, 0xD8);
    E1 = _mm_shuffle_epi32(E1, 0xD8);
    _mm_storeu_si128((__m128i *)E_p, _mm_unpackhi_epi32(E0, E1));
    E_p += 4;
    // update pointer to line buffer
    ssp0 += 4;
    ssp1 += 4;
    sp0 += 4;
    sp1 += 4;
  }
  if (qx) {
    make_storage_one(ssp0, ssp1, sp0, sp1, sig0, v0, E0, rho0);
    // MEL encoding for the first quad
    if (context == 0) {
      MEL_encoder.encodeMEL((rho0 != 0));
    }
    Emax_q      = hMax(E0);
    U0          = std::max((int32_t)Emax_q, kappa);
    u_q         = U0 - kappa;
    uvlc_idx    = u_q;
    uoff        = (u_q) ? 1 : 0;
    Etmp        = _mm_set1_epi32(Emax_q);
    vuoff       = _mm_set1_epi32(uoff << 7);
    mask        = _mm_cmpeq_epi32(E0, Etmp);
    vtmp        = _mm_and_si128(vuoff, mask);
    emb_pattern = _mm_movemask_epi8(
        _mm_packus_epi16(_mm_packus_epi32(vtmp, _mm_setzero_si128()), _mm_setzero_si128()));
    n_q = emb_pattern + (rho0 << 4) + (context << 8);
    CxtVLC = enc_CxtVLC_table0[n_q];
    embk_0 = CxtVLC & 0xF;
    emb1_0 = emb_pattern & embk_0;
    lw     = (CxtVLC >> 4) & 0x07;
    cwd    = CxtVLC >> 7;
    {
      uint32_t uvlc_tmp = enc_UVLC_table0[uvlc_idx];
      uint32_t uvlc_lw  = uvlc_tmp & 0xFF;
      uint32_t uvlc_cwd = uvlc_tmp >> 8;
      VLC_encoder.emitVLCBits(cwd | (static_cast<uint64_t>(uvlc_cwd) << lw), lw + uvlc_lw);
    }

    m0       = _mm_sub_epi32(_mm_and_si128(sig0, _mm_set1_epi32(U0)),
                             _mm_and_si128(_mm_srlv_epi32(_mm_set1_epi32(embk_0), vshift), vone));
    known1_0 = _mm_and_si128(_mm_srlv_epi32(_mm_set1_epi32(emb1_0), vshift), vone);
    ms_v[ms_count] = v0; ms_m[ms_count] = m0; ms_k1[ms_count] = known1_0; ms_count++;

    *E_p++ = _mm_extract_epi32(E0, 1);
    *E_p++ = _mm_extract_epi32(E0, 3);
    // update rho_line
    *rho_p++ = rho0;
  }
  // Emit deferred MagSgn for initial line-pair
  for (int32_t i = 0; i < ms_count; i++)
    MagSgn_encoder.emitBits(ms_v[i], ms_m[i], ms_k1[i]);

  /*******************************************************************************************************************/
  // Non-initial line-pair
  /*******************************************************************************************************************/
  // Pre-computed per-quad arrays for the two-pass architecture.
  // Sized for max codeblock width (1024 → QW ≤ 512).
  alignas(32) int32_t  rho_a[512];
  alignas(32) int32_t  ctx_a[512];
  alignas(32) int32_t  U_a[512];
  alignas(32) int32_t  u_q_a[512];
  alignas(32) int32_t  embk_a[512];
  alignas(32) int32_t  emb1_a[512];
  alignas(32) uint32_t vlc_cwd_a[512];
  alignas(32) uint32_t vlc_lw_a[512];
  alignas(64) int32_t  v_flat[512 * 4];   // 4 MagSgn values per quad, interleaved [r0c0,r1c0,r0c1,r1c1]
  alignas(64) int32_t  sig_flat[512 * 4]; // 4 significance flags per quad (0 or -1)
  alignas(64) int32_t  E_flat[512 * 4];   // 4 E values per quad
  // __m128i views for backward compatibility with Phase 2c/3 that use hMax/__m128i
  auto v_a   = reinterpret_cast<__m128i *>(v_flat);
  auto sig_a = reinterpret_cast<__m128i *>(sig_flat);
  auto E_a   = reinterpret_cast<__m128i *>(E_flat);

  for (uint16_t qy = 1; qy < QH; qy++) {
    ms_count = 0;
    E_p   = Eline + 1;
    rho_p = rholine + 1;

    ssp0 = block->block_states + (2U * qy + 1U) * (block->blkstate_stride) + 1U;
    ssp1 = ssp0 + block->blkstate_stride;
    sp0  = block->sample_buf + 2U * (qy * block->blksampl_stride);
    sp1  = sp0 + block->blksampl_stride;

    // ===== PHASE 1: Pre-compute rho/E/v/sig for ALL quads (AVX-512, 8 quads/iter) =====
    {
      uint8_t *s0p = ssp0, *s1p = ssp1;
      int32_t *p0 = sp0, *p1 = sp1;
      int32_t q = 0;
#ifdef __AVX512CD__
      // Permutation index for interleaving two __m512i of 16 columns from row0/row1
      // into quad order: [r0c0,r1c0,r0c1,r1c1] for 4 quads per output register.
      // unpacklo gives: [r0[0],r1[0],r0[1],r1[1] | r0[4],r1[4],r0[5],r1[5] | ...]
      // unpackhi gives: [r0[2],r1[2],r0[3],r1[3] | r0[6],r1[6],r0[7],r1[7] | ...]
      // We need quads in order: q0=[c0,c1], q1=[c2,c3], q2=[c4,c5], ...
      // So output0 = {lo_lane0, hi_lane0, lo_lane1, hi_lane1} = quads 0,1,2,3
      // output1 = {lo_lane2, hi_lane2, lo_lane3, hi_lane3} = quads 4,5,6,7
      const __m512i perm_lo = _mm512_setr_epi32(
          0,1,2,3, 16,17,18,19, 4,5,6,7, 20,21,22,23);
      const __m512i perm_hi = _mm512_setr_epi32(
          8,9,10,11, 24,25,26,27, 12,13,14,15, 28,29,30,31);
      const __m512i V32 = _mm512_set1_epi32(32);

      for (; q + 7 < QW; q += 8) {
        // --- Significance from block_states (16 bytes per row) ---
        // Load 16 uint8 from each row, AND with 1 to extract sigma bit
        __m128i st0_raw = _mm_loadu_si128(reinterpret_cast<const __m128i *>(s0p));
        __m128i st1_raw = _mm_loadu_si128(reinterpret_cast<const __m128i *>(s1p));
        __m128i ones_8  = _mm_set1_epi8(1);
        __m128i st0     = _mm_and_si128(st0_raw, ones_8);  // sig row0[0..15]
        __m128i st1     = _mm_and_si128(st1_raw, ones_8);  // sig row1[0..15]

        // Compute rho for 8 quads from 16 columns:
        // rho[q] = sig_r0[2q] | (sig_r1[2q] << 1) | (sig_r0[2q+1] << 2) | (sig_r1[2q+1] << 3)
        // Deinterleave even/odd bytes:
        // even: positions 0,2,4,...,14 → sig_r0_even[0..7], sig_r1_even[0..7]
        // odd:  positions 1,3,5,...,15 → sig_r0_odd[0..7], sig_r1_odd[0..7]
        const __m128i shuf_even = _mm_setr_epi8(0,2,4,6,8,10,12,14, -1,-1,-1,-1,-1,-1,-1,-1);
        const __m128i shuf_odd  = _mm_setr_epi8(1,3,5,7,9,11,13,15, -1,-1,-1,-1,-1,-1,-1,-1);
        __m128i s0_even = _mm_shuffle_epi8(st0, shuf_even);  // row0 even cols
        __m128i s0_odd  = _mm_shuffle_epi8(st0, shuf_odd);   // row0 odd cols
        __m128i s1_even = _mm_shuffle_epi8(st1, shuf_even);  // row1 even cols
        __m128i s1_odd  = _mm_shuffle_epi8(st1, shuf_odd);   // row1 odd cols

        // rho = s0_even | (s1_even << 1) | (s0_odd << 2) | (s1_odd << 3)
        // All values are 0 or 1, so shifts within byte are safe
        __m128i rho_bytes = _mm_or_si128(
            _mm_or_si128(s0_even, _mm_slli_epi16(s1_even, 1)),
            _mm_or_si128(_mm_slli_epi16(s0_odd, 2), _mm_slli_epi16(s1_odd, 3)));
        // rho_bytes low 8 bytes contain rho[0..7] as uint8 (values 0-15)
        // Zero-extend to int32 and store
        __m256i rho_32 = _mm256_cvtepu8_epi32(rho_bytes);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(rho_a + q), rho_32);

        // --- Significance masks (int32, 0 or -1) in quad order ---
        // Target per-quad: [r0c0, r1c0, r0c1, r1c1]
        // Build pairs: even_pairs=[s0e[0],s1e[0], s0e[1],s1e[1],...] (16-bit chunks)
        //              odd_pairs =[s0o[0],s1o[0], s0o[1],s1o[1],...]
        // Then unpack at 16-bit to interleave pairs:
        //   [s0e[0],s1e[0], s0o[0],s1o[0], ...] = [r0c0,r1c0,r0c1,r1c1, ...]
        __m128i even_pairs = _mm_unpacklo_epi8(s0_even, s1_even);
        __m128i odd_pairs  = _mm_unpacklo_epi8(s0_odd, s1_odd);
        __m128i sig_lo_bytes = _mm_unpacklo_epi16(even_pairs, odd_pairs);  // quads 0-3
        __m128i sig_hi_bytes = _mm_unpackhi_epi16(even_pairs, odd_pairs);  // quads 4-7

        // Expand bytes (0 or 1) to int32 sign-extended masks (0 or -1)
        __m512i sig_lo = _mm512_cvtepi8_epi32(sig_lo_bytes);
        sig_lo = _mm512_srai_epi32(_mm512_slli_epi32(sig_lo, 31), 31);

        __m512i sig_hi = _mm512_cvtepi8_epi32(sig_hi_bytes);
        sig_hi = _mm512_srai_epi32(_mm512_slli_epi32(sig_hi, 31), 31);

        _mm512_storeu_si512(sig_flat + q * 4, sig_lo);
        _mm512_storeu_si512(sig_flat + q * 4 + 16, sig_hi);

        // --- Sample values in quad order ---
        __m512i row0 = _mm512_loadu_si512(p0);  // sp0[0..15]
        __m512i row1 = _mm512_loadu_si512(p1);  // sp1[0..15]

        // unpacklo/hi at 32-bit granularity (within 128-bit lanes):
        __m512i lo = _mm512_unpacklo_epi32(row0, row1);
        __m512i hi = _mm512_unpackhi_epi32(row0, row1);

        // Permute to get consecutive quads:
        __m512i v_q0123 = _mm512_permutex2var_epi32(lo, perm_lo, hi);  // quads 0-3
        __m512i v_q4567 = _mm512_permutex2var_epi32(lo, perm_hi, hi);  // quads 4-7

        _mm512_storeu_si512(v_flat + q * 4, v_q0123);
        _mm512_storeu_si512(v_flat + q * 4 + 16, v_q4567);

        // --- E = (32 - lzcnt(v)) & sig ---
        __m512i lz_lo = _mm512_lzcnt_epi32(v_q0123);
        __m512i lz_hi = _mm512_lzcnt_epi32(v_q4567);
        __m512i E_lo = _mm512_and_epi32(_mm512_sub_epi32(V32, lz_lo), sig_lo);
        __m512i E_hi = _mm512_and_epi32(_mm512_sub_epi32(V32, lz_hi), sig_hi);

        _mm512_storeu_si512(E_flat + q * 4, E_lo);
        _mm512_storeu_si512(E_flat + q * 4 + 16, E_hi);

        s0p += 16; s1p += 16; p0 += 16; p1 += 16;
      }
#endif  // __AVX512CD__
      // Scalar tail for remaining quads
      for (; q + 1 < QW; q += 2) {
        make_storage(s0p, s1p, p0, p1,
                     sig_a[q], sig_a[q + 1], v_a[q], v_a[q + 1],
                     E_a[q], E_a[q + 1], rho_a[q], rho_a[q + 1]);
        s0p += 4; s1p += 4; p0 += 4; p1 += 4;
      }
      if (q < QW) {
        make_storage_one(s0p, s1p, p0, p1, sig_a[q], v_a[q], E_a[q], rho_a[q]);
      }
    }

    // ===== PHASE 2a: Compute context for ALL quads (AVX-512 vectorized) =====
    {
#ifdef __AVX512CD__
      int32_t q = 0;
      // rho_west[q] = rho_a[q-1] with rho_a[-1] = 0
      // rho_p[q-1], rho_p[q], rho_p[q+1] are from the previous line-pair's rholine
      for (; q + 15 < QW; q += 16) {
        // Load rho_a[q-1..q+14] as rho_west (shifted by 1)
        __m512i rw;
        if (q == 0) {
          alignas(64) int32_t rw_tmp[16];
          rw_tmp[0] = 0;
          memcpy(rw_tmp + 1, rho_a, 15 * sizeof(int32_t));
          rw = _mm512_load_si512(rw_tmp);
        } else {
          rw = _mm512_loadu_si512(rho_a + q - 1);
        }
        __m512i rp_m1 = _mm512_loadu_si512(rho_p + q - 1);
        __m512i rp_0  = _mm512_loadu_si512(rho_p + q);
        __m512i rp_p1 = _mm512_loadu_si512(rho_p + q + 1);

        // ctx = ((rw & 0x4) << 7) | ((rw & 0x8) << 6)
        //     | ((rp_m1 & 0x8) << 5) | ((rp_0 & 0xa) << 7)
        //     | ((rp_p1 & 0x2) << 9)
        __m512i ctx = _mm512_or_epi32(
            _mm512_or_epi32(
                _mm512_slli_epi32(_mm512_and_epi32(rw, _mm512_set1_epi32(0x4)), 7),
                _mm512_slli_epi32(_mm512_and_epi32(rw, _mm512_set1_epi32(0x8)), 6)),
            _mm512_or_epi32(
                _mm512_or_epi32(
                    _mm512_slli_epi32(_mm512_and_epi32(rp_m1, _mm512_set1_epi32(0x8)), 5),
                    _mm512_slli_epi32(_mm512_and_epi32(rp_0, _mm512_set1_epi32(0xa)), 7)),
                _mm512_slli_epi32(_mm512_and_epi32(rp_p1, _mm512_set1_epi32(0x2)), 9)));
        _mm512_storeu_si512(ctx_a + q, ctx);
      }
      // Scalar tail
      int32_t rho_west = (q > 0) ? rho_a[q - 1] : 0;
      for (; q < QW; q++) {
        ctx_a[q] = ((rho_west & 0x4) << 7) | ((rho_west & 0x8) << 6)
                   | ((rho_p[q - 1] & 0x8) << 5) | ((rho_p[q] & 0xa) << 7)
                   | ((rho_p[q + 1] & 0x2) << 9);
        rho_west = rho_a[q];
      }
#else
      int32_t rho_west = 0;
      for (int32_t q = 0; q < QW; q++) {
        ctx_a[q] = ((rho_west & 0x4) << 7) | ((rho_west & 0x8) << 6)
                   | ((rho_p[q - 1] & 0x8) << 5) | ((rho_p[q] & 0xa) << 7)
                   | ((rho_p[q + 1] & 0x2) << 9);
        rho_west = rho_a[q];
      }
#endif
    }

    // ===== PHASE 2b: Read ALL Emax from old Eline before any updates =====
    // Emax(q) = max(E_p[2q-1], E_p[2q], E_p[2q+1], E_p[2q+2])
    alignas(64) int32_t Emax_a[512];
    for (int32_t q = 0; q < QW; q++)
      Emax_a[q] = find_max(E_p[2 * q - 1], E_p[2 * q], E_p[2 * q + 1], E_p[2 * q + 2]);

    // ===== PHASE 2c: Compute kappa, U, emb_pattern, VLC (AVX-512 gather); update Eline + rholine =====
    {
      const __m512i VONE = _mm512_set1_epi32(1);
      const __m512i VZERO = _mm512_setzero_si512();
      int32_t q = 0;
      for (; q + 15 < QW; q += 16) {
        __m512i rho_v = _mm512_loadu_si512(rho_a + q);
        __m512i emax_line_v = _mm512_loadu_si512(Emax_a + q);

        // gamma = (popcount(rho) > 1) ? -1 : 0
        __m512i rho_m1 = _mm512_sub_epi32(rho_v, VONE);
        __m512i gamma_v = _mm512_and_epi32(rho_v, rho_m1);
        __mmask16 gamma_mask = _mm512_cmpneq_epi32_mask(gamma_v, VZERO);
        gamma_v = _mm512_maskz_set1_epi32(gamma_mask, -1);

        // kappa = max((Emax_line - 1) & gamma, 1)
        __m512i kappa_v = _mm512_and_epi32(_mm512_sub_epi32(emax_line_v, VONE), gamma_v);
        kappa_v = _mm512_max_epi32(kappa_v, VONE);

        // Emax_q = hMax(E_a[q]) for each quad — horizontal max of 4 elements
        // E_a[q] is __m128i; extract and compute per-quad (scalar for now)
        alignas(64) int32_t emax_q_arr[16];
        for (int i = 0; i < 16; i++)
          emax_q_arr[i] = hMax(E_a[q + i]);
        __m512i emax_q_v = _mm512_load_si512(emax_q_arr);

        // U = max(Emax_q, kappa)
        __m512i U_v = _mm512_max_epi32(emax_q_v, kappa_v);
        __m512i u_q_v = _mm512_sub_epi32(U_v, kappa_v);
        _mm512_storeu_si512(U_a + q, U_v);
        _mm512_storeu_si512(u_q_a + q, u_q_v);

        // emb_pattern: for each quad, check which E values == Emax_q AND uoff
        // This is per-quad (4 elements each), done scalar
        alignas(64) int32_t emb_arr[16], nq_arr[16];
        for (int i = 0; i < 16; i++) {
          int32_t uoff_i = (u_q_a[q + i] > 0) ? 1 : 0;
          __m128i Et = _mm_set1_epi32(emax_q_arr[i]);
          __m128i vu = _mm_set1_epi32(uoff_i << 7);
          __m128i mk = _mm_cmpeq_epi32(E_a[q + i], Et);
          __m128i vt = _mm_and_si128(vu, mk);
          emb_arr[i] = _mm_movemask_epi8(
              _mm_packus_epi16(_mm_packus_epi32(vt, _mm_setzero_si128()), _mm_setzero_si128()));
          nq_arr[i] = emb_arr[i] + (rho_a[q + i] << 4) + ctx_a[q + i];
        }

        // AVX-512 GATHER: 16 VLC table lookups at once (uint16_t table → scale=2, mask to 16 bits)
        __m512i nq_v = _mm512_load_si512(nq_arr);
        __m512i cv_v = _mm512_and_epi32(
            _mm512_i32gather_epi32(nq_v, enc_CxtVLC_table1, 2),
            _mm512_set1_epi32(0xFFFF));

        // Extract embk, emb1, vlc_cwd, vlc_lw from gathered CxtVLC values
        __m512i embk_v = _mm512_and_epi32(cv_v, _mm512_set1_epi32(0xF));
        _mm512_storeu_si512(embk_a + q, embk_v);

        __m512i emb_v = _mm512_load_si512(emb_arr);
        _mm512_storeu_si512(emb1_a + q, _mm512_and_epi32(emb_v, embk_v));

        __m512i vlc_cwd_v = _mm512_srli_epi32(cv_v, 7);
        _mm512_storeu_si512(vlc_cwd_a + q, vlc_cwd_v);
        __m512i vlc_lw_v = _mm512_and_epi32(_mm512_srli_epi32(cv_v, 4), _mm512_set1_epi32(0x07));
        _mm512_storeu_si512(vlc_lw_a + q, vlc_lw_v);

        // Update Eline + rholine
        for (int i = 0; i < 16; i++) {
          __m128i Es = _mm_shuffle_epi32(E_a[q + i], 0xD8);
          E_p[2 * (q + i)]     = _mm_extract_epi32(Es, 2);
          E_p[2 * (q + i) + 1] = _mm_extract_epi32(Es, 3);
          rho_p[q + i] = rho_a[q + i];
        }
      }
      // Scalar tail for remaining quads
      for (; q < QW; q++) {
        int32_t rq = rho_a[q];
        int32_t gamma_q = ((rq & (rq - 1)) == 0) ? 0 : static_cast<int32_t>(0xFFFFFFFF);
        int32_t kappa_q = std::max((Emax_a[q] - 1) & gamma_q, 1);
        int32_t Emax_q = hMax(E_a[q]);
        int32_t Uq = std::max(Emax_q, kappa_q);
        int32_t uq = Uq - kappa_q;
        U_a[q] = Uq;
        u_q_a[q] = uq;

        int32_t uoff_q = (uq > 0) ? 1 : 0;
        __m128i Etmp_q = _mm_set1_epi32(Emax_q);
        __m128i vuoff_q = _mm_set1_epi32(uoff_q << 7);
        __m128i mask_q = _mm_cmpeq_epi32(E_a[q], Etmp_q);
        __m128i vtmp_q = _mm_and_si128(vuoff_q, mask_q);
        int32_t emb_p = _mm_movemask_epi8(
            _mm_packus_epi16(_mm_packus_epi32(vtmp_q, _mm_setzero_si128()), _mm_setzero_si128()));
        int32_t nq = emb_p + (rq << 4) + (ctx_a[q] << 0);
        uint32_t cv = enc_CxtVLC_table1[nq];
        embk_a[q] = cv & 0xF;
        emb1_a[q] = emb_p & embk_a[q];
        vlc_cwd_a[q] = cv >> 7;
        vlc_lw_a[q] = (cv >> 4) & 0x07;

        __m128i Es = _mm_shuffle_epi32(E_a[q], 0xD8);
        E_p[2 * q]     = _mm_extract_epi32(Es, 2);
        E_p[2 * q + 1] = _mm_extract_epi32(Es, 3);
        rho_p[q] = rq;
      }
    }


    // ===== PHASE 3: Serial MEL + batched VLC + deferred MagSgn =====
    for (int32_t q = 0; q + 1 < QW; q += 2) {
      // MEL for first quad
      if (ctx_a[q] == 0)
        MEL_encoder.encodeMEL((rho_a[q] != 0));

      // Batched VLC + UVLC
      int32_t uvlc_idx_pair = u_q_a[q] + (u_q_a[q + 1] << 5);
      uint32_t uvlc_tmp = enc_UVLC_table1[uvlc_idx_pair];
      uint32_t uvlc_lw_val  = uvlc_tmp & 0xFF;
      uint32_t uvlc_cwd_val = uvlc_tmp >> 8;
      uint64_t vlc_all = vlc_cwd_a[q]
          | (static_cast<uint64_t>(vlc_cwd_a[q + 1]) << vlc_lw_a[q])
          | (static_cast<uint64_t>(uvlc_cwd_val) << (vlc_lw_a[q] + vlc_lw_a[q + 1]));
      VLC_encoder.emitVLCBits(vlc_all, vlc_lw_a[q] + vlc_lw_a[q + 1] + uvlc_lw_val);

      // MEL for second quad
      if (ctx_a[q + 1] == 0)
        MEL_encoder.encodeMEL((rho_a[q + 1] != 0));

      // Defer MagSgn
      for (int32_t s = 0; s < 2; s++) {
        int32_t qi = q + s;
        __m128i mq = _mm_sub_epi32(_mm_and_si128(sig_a[qi], _mm_set1_epi32(U_a[qi])),
                                   _mm_and_si128(_mm_srlv_epi32(_mm_set1_epi32(embk_a[qi]), vshift), vone));
        __m128i kq = _mm_and_si128(_mm_srlv_epi32(_mm_set1_epi32(emb1_a[qi]), vshift), vone);
        ms_v[ms_count] = v_a[qi]; ms_m[ms_count] = mq; ms_k1[ms_count] = kq; ms_count++;
      }
    }
    // Odd trailing quad
    if (QW & 1) {
      int32_t q = QW - 1;
      if (ctx_a[q] == 0)
        MEL_encoder.encodeMEL(rho_a[q] != 0);
      uint32_t uvlc_tmp = enc_UVLC_table1[u_q_a[q]];
      uint32_t uvlc_lw_val  = uvlc_tmp & 0xFF;
      uint32_t uvlc_cwd_val = uvlc_tmp >> 8;
      VLC_encoder.emitVLCBits(
          vlc_cwd_a[q] | (static_cast<uint64_t>(uvlc_cwd_val) << vlc_lw_a[q]),
          vlc_lw_a[q] + uvlc_lw_val);
      __m128i mq = _mm_sub_epi32(_mm_and_si128(sig_a[q], _mm_set1_epi32(U_a[q])),
                                 _mm_and_si128(_mm_srlv_epi32(_mm_set1_epi32(embk_a[q]), vshift), vone));
      __m128i kq = _mm_and_si128(_mm_srlv_epi32(_mm_set1_epi32(emb1_a[q]), vshift), vone);
      ms_v[ms_count] = v_a[q]; ms_m[ms_count] = mq; ms_k1[ms_count] = kq; ms_count++;
    }

    // ===== PHASE 4: Emit deferred MagSgn =====
    // Pre-extract to flat scalar arrays — eliminates SSE register pressure
    // so the compiler can keep Creg/ctreg in GPRs during the emit loop.
    {
      alignas(32) uint32_t flat_v[512 * 4], flat_m[512 * 4];
      int32_t flat_count = 0;
      for (int32_t i = 0; i < ms_count; i++) {
        __m128i tmp = _mm_sllv_epi32(ms_k1[i], ms_m[i]);
        __m128i v = _mm_sub_epi32(ms_v[i], tmp);
        __m128i m = ms_m[i];
        flat_v[flat_count]   = static_cast<uint32_t>(_mm_extract_epi32(v, 0));
        flat_m[flat_count++] = static_cast<uint32_t>(_mm_extract_epi32(m, 0));
        flat_v[flat_count]   = static_cast<uint32_t>(_mm_extract_epi32(v, 1));
        flat_m[flat_count++] = static_cast<uint32_t>(_mm_extract_epi32(m, 1));
        flat_v[flat_count]   = static_cast<uint32_t>(_mm_extract_epi32(v, 2));
        flat_m[flat_count++] = static_cast<uint32_t>(_mm_extract_epi32(m, 2));
        flat_v[flat_count]   = static_cast<uint32_t>(_mm_extract_epi32(v, 3));
        flat_m[flat_count++] = static_cast<uint32_t>(_mm_extract_epi32(m, 3));
      }
      MagSgn_encoder.emitFlat(flat_v, flat_m, flat_count);
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
  block->set_compressed_data(fwd_buf, static_cast<uint16_t>(Lcup), MAX_Lref);
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
auto process_stripes_block_enc = [](SP_enc &SigProp, j2k_codeblock *block, const uint32_t i_start,
                                    const uint32_t j_start, const uint32_t width, const uint32_t height) {
  uint8_t *sp;
  uint8_t causal_cond = 0;
  uint8_t bit;
  uint8_t mbr;
  // uint32_t mbr_info;  // NOT USED
  const auto block_width  = j_start + width;
  const auto block_height = i_start + height;
  for (uint32_t j = j_start; j < block_width; j++) {
    // mbr_info = 0;
    for (uint32_t i = i_start; i < block_height; i++) {
      sp          = block->block_states + (i + 1) * block->blkstate_stride + (j + 1);
      causal_cond = (((block->Cmodes & CAUSAL) == 0) || (i != i_start + height - 1));
      mbr         = 0;
      //      if (block->get_state(Sigma, i, j) == 0) {
      if ((sp[0] >> SHIFT_SIGMA & 1) == 0) {
        mbr = block->calc_mbr(i, j, causal_cond);
      }
      // mbr_info >>= 3;
      if (mbr != 0) {
        bit = (*sp >> SHIFT_SMAG) & 1;
        SigProp.emitSPBit(bit);
        //        block->modify_state(refinement_indicator, 1, i, j);
        sp[0] |= 1 << SHIFT_PI_;
        //        block->modify_state(refinement_value, bit, i, j);
        sp[0] |= static_cast<uint8_t>(bit << SHIFT_REF);
      }
      //      block->modify_state(scan, 1, i, j);
      sp[0] |= 1 << SHIFT_SCAN;
    }
  }
  for (uint32_t j = j_start; j < block_width; j++) {
    for (uint32_t i = i_start; i < block_height; i++) {
      sp = block->block_states + (i + 1) * block->blkstate_stride + (j + 1);
      // encode sign
      //      if (block->get_state(Refinement_value, i, j)) {
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

  // encode full-height (=4) stripes
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
  // encode remaining height stripes
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
        //        sp               = &block->block_states[j + 1 + (i + 1) * (block->size.x + 2)];
        if ((sp[0] >> SHIFT_SIGMA & 1) != 0) {
          bit = (sp[0] >> SHIFT_SMAG) & 1;
          MagRef.emitMRBit(bit);
          //          block->modify_state(refinement_indicator, 1, i, j);
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
        //        block->modify_state(refinement_indicator, 1, i, j);
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
    // SigProp encoding
    ht_sigprop_encode(block, SigProp);
    // MagRef encoding
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