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

#if defined(OPENHTJ2K_TRY_AVX2) && defined(__AVX2__)
  #include "coding_units.hpp"
  #include "dec_CxtVLC_tables.hpp"
  #include "ht_block_decoding.hpp"
  #include "coding_local.hpp"
  #include "utils.hpp"

  #if defined(_MSC_VER) || defined(__MINGW64__)
    #include <intrin.h>
  #else
    #include <x86intrin.h>
  #endif

static FORCE_INLINE uint8_t calc_mbr_inline(const uint8_t *block_states, size_t blkstate_stride,
                                             uint32_t i, uint32_t j, uint8_t causal_cond) {
  const uint8_t *p0 = block_states + static_cast<size_t>(i) * blkstate_stride + j;
  const uint8_t *p1 = p0 + blkstate_stride;
  const uint8_t *p2 = p1 + blkstate_stride;

  uint32_t mbr0 = p0[0] | p0[1] | p0[2];
  uint32_t mbr1 = p1[0] | p1[2];
  uint32_t mbr2 = p2[0] | p2[1] | p2[2];
  uint32_t mbr  = mbr0 | mbr1 | (mbr2 & causal_cond);
  mbr |= (mbr0 >> SHIFT_REF) & (mbr0 >> SHIFT_SCAN);
  mbr |= (mbr1 >> SHIFT_REF) & (mbr1 >> SHIFT_SCAN);
  mbr |= (mbr2 >> SHIFT_REF) & (mbr2 >> SHIFT_SCAN) & causal_cond;
  return mbr & 1;
}

// Pointer-based MBR: state_p points to state[(i+1)*stride + (j+1)].
static FORCE_INLINE uint8_t calc_mbr_p(const uint8_t *state_p, size_t stride, uint8_t causal_cond) {
  const uint8_t *p0 = state_p - stride - 1;
  const uint8_t *p1 = state_p - 1;
  const uint8_t *p2 = state_p + stride - 1;

  uint32_t mbr0 = p0[0] | p0[1] | p0[2];
  uint32_t mbr1 = p1[0] | p1[2];
  uint32_t mbr2 = p2[0] | p2[1] | p2[2];
  uint32_t mbr  = mbr0 | mbr1 | (mbr2 & causal_cond);
  mbr |= (mbr0 >> SHIFT_REF) & (mbr0 >> SHIFT_SCAN);
  mbr |= (mbr1 >> SHIFT_REF) & (mbr1 >> SHIFT_SCAN);
  mbr |= (mbr2 >> SHIFT_REF) & (mbr2 >> SHIFT_SCAN) & causal_cond;
  return mbr & 1;
}

// Keep the member function for ABI compatibility (used by non-AVX2 callers).
uint8_t j2k_codeblock::calc_mbr(const uint32_t i, const uint32_t j, const uint8_t causal_cond) const {
  return calc_mbr_inline(block_states, blkstate_stride, i, j, causal_cond);
}

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

// AVX2 256-bit leading-zero count for 8 × uint32_t values.
inline __m256i avx2_lzcnt_epi32(__m256i v) {
  v = _mm256_andnot_si256(_mm256_srli_epi32(v, 8), v);  // keep 8 MSB
  v = _mm256_castps_si256(_mm256_cvtepi32_ps(v));
  v = _mm256_srli_epi32(v, 23);
  v = _mm256_subs_epu16(_mm256_set1_epi32(158), v);
  v = _mm256_min_epi16(v, _mm256_set1_epi32(32));
  return v;
}

// Build __m256i with lower 128 = broadcast of qinf[0], upper = broadcast of qinf[1].
static FORCE_INLINE __m256i expand_two_quads(__m128i qinf128) {
  return _mm256_set_m128i(_mm_shuffle_epi32(qinf128, _MM_SHUFFLE(1, 1, 1, 1)),
                          _mm_shuffle_epi32(qinf128, _MM_SHUFFLE(0, 0, 0, 0)));
}

// Fused dequantize-and-store for 8 × int32 MagSgn samples → 8 × float.
// Lossless (transformation==1): sign-magnitude → two's-complement shift → float.
// Lossy   (transformation==0): magnitude → float → scale → apply sign via XOR.
static FORCE_INLINE void dequant_store_256(int32_t *dst, __m256i val, uint8_t transformation,
                                           int32_t pLSB_dq, __m256 vfscale, __m256i vmagmask,
                                           __m256i vsignmask) {
  if (transformation == 1) {
    __m256i mag  = _mm256_and_si256(val, vmagmask);
    __m256i res  = _mm256_sign_epi32(_mm256_srai_epi32(mag, pLSB_dq), val);
    _mm256_storeu_ps(reinterpret_cast<float *>(dst), _mm256_cvtepi32_ps(res));
  } else {
    __m256i mag = _mm256_and_si256(val, vmagmask);
    __m256 f    = _mm256_mul_ps(_mm256_cvtepi32_ps(mag), vfscale);
    f           = _mm256_xor_ps(f, _mm256_castsi256_ps(_mm256_and_si256(val, vsignmask)));
    _mm256_storeu_ps(reinterpret_cast<float *>(dst), f);
  }
}

// SSE 128-bit variant: 4 × int32 → 4 × float.
static FORCE_INLINE void dequant_store_128(int32_t *dst, __m128i val, uint8_t transformation,
                                           int32_t pLSB_dq, __m256 vfscale256, __m128i vmagmask,
                                           __m128i vsignmask) {
  if (transformation == 1) {
    __m128i mag = _mm_and_si128(val, vmagmask);
    __m128i res = _mm_sign_epi32(_mm_srai_epi32(mag, pLSB_dq), val);
    _mm_storeu_ps(reinterpret_cast<float *>(dst), _mm_cvtepi32_ps(res));
  } else {
    __m128i mag = _mm_and_si128(val, vmagmask);
    __m128 f    = _mm_mul_ps(_mm_cvtepi32_ps(mag), _mm256_castps256_ps128(vfscale256));
    f           = _mm_xor_ps(f, _mm_castsi128_ps(_mm_and_si128(val, vsignmask)));
    _mm_storeu_ps(reinterpret_cast<float *>(dst), f);
  }
}

template <bool skip_sigma, bool fuse_dequant = false>
void ht_cleanup_decode(j2k_codeblock *block, const uint8_t &pLSB, const int32_t Lcup, const int32_t Pcup,
                       const int32_t Scup) {
  uint8_t *compressed_data = block->get_compressed_data();
  const uint16_t QW        = static_cast<uint16_t>(ceil_int(static_cast<int16_t>(block->size.x), 2));
  const uint16_t QH        = static_cast<uint16_t>(ceil_int(static_cast<int16_t>(block->size.y), 2));

  uint16_t scratch[8 * 513];
  int32_t sstr = static_cast<int32_t>(((block->size.x + 2) + 7u) & ~7u);  // multiples of 8
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
    {
      // Branchless context-0 MEL handling: replace unpredictable branch with mask
      int32_t cm = -static_cast<int32_t>(context == 0);
      mel_run -= cm & 2;
      tv0 &= static_cast<uint16_t>(-(mel_run == -1) | ~cm);
      if (mel_run < 0) mel_run = MEL.get_run();
    }
    sp[0] = tv0;

    // calculate context for the next quad, Eq. (1) in the spec
    context = ((tv0 & 0xE0U) << 2) | ((tv0 & 0x10U) << 3);  // = context << 7

    // Decoding of significance and EMB patterns and unsigned residual offsets
    vlcval       = VLC_dec.advance((tv0 & 0x000F) >> 1);
    uint16_t tv1 = dec_table[(vlcval & 0x7F) + context];
    {
      int32_t cm = -static_cast<int32_t>((context == 0) & (qx > 1));
      mel_run -= cm & 2;
      tv1 &= static_cast<uint16_t>(-(mel_run == -1) | ~cm);
      if (mel_run < 0) mel_run = MEL.get_run();
    }
    tv1   = (qx > 1) ? tv1 : 0;
    sp[2] = tv1;

    // store sigma
    if (!skip_sigma) {
      *sp0++ = ((tv0 >> 4) >> 0) & 1;
      *sp0++ = ((tv0 >> 4) >> 2) & 1;
      *sp0++ = ((tv1 >> 4) >> 0) & 1;
      *sp0++ = ((tv1 >> 4) >> 2) & 1;
      *sp1++ = ((tv0 >> 4) >> 1) & 1;
      *sp1++ = ((tv0 >> 4) >> 3) & 1;
      *sp1++ = ((tv1 >> 4) >> 1) & 1;
      *sp1++ = ((tv1 >> 4) >> 3) & 1;
    }

    // calculate context for the next quad, Eq. (1) in the spec
    context = ((tv1 & 0xE0U) << 2) | ((tv1 & 0x10U) << 3);  // = context << 7

    vlcval = VLC_dec.advance((tv1 & 0x000F) >> 1);
    u_off0 = tv0 & 1;
    u_off1 = tv1 & 1;

    // Branchless MEL offset: replace compound branch with mask
    uint32_t both_off = u_off0 & u_off1;
    int32_t om        = -static_cast<int32_t>(both_off);
    mel_run -= om & 2;
    uint32_t mel_offset = static_cast<uint32_t>(-(mel_run == -1) & om) & 0x40;
    if (mel_run < 0) mel_run = MEL.get_run();

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
  // Zero the 8-byte guard past the last VLC row: the MagSgn SSE load reads 4 extra
  // uint16_t beyond the written region in the final iteration of each row.
  std::memset(sp, 0, sizeof(uint16_t) * 4);

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
      if (!skip_sigma) {
        *sp0++ = ((tv0 >> 4) >> 0) & 1;
        *sp0++ = ((tv0 >> 4) >> 2) & 1;
        *sp0++ = ((tv1 >> 4) >> 0) & 1;
        *sp0++ = ((tv1 >> 4) >> 2) & 1;
        *sp1++ = ((tv0 >> 4) >> 1) & 1;
        *sp1++ = ((tv0 >> 4) >> 3) & 1;
        *sp1++ = ((tv1 >> 4) >> 1) & 1;
        *sp1++ = ((tv1 >> 4) >> 3) & 1;
      }

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
    // Zero the 8-byte guard: same reason as initial row.
    std::memset(sp, 0, sizeof(uint16_t) * 4);
  }

  /*******************************************************************************************************************/
  // MagSgn decoding
  /*******************************************************************************************************************/

  // Fused dequantize setup: when fuse_dequant is true, we write dequantized float values
  // directly to i_samples, eliminating the separate dequantize pass.
  // pLSB_dq is the dequantization shift (31 - M_b), distinct from the MagSgn pLSB.
  int32_t pLSB_dq        = 0;
  float fscale_direct     = 0.0f;
  __m256 vfscale          = _mm256_setzero_ps();
  __m256i vsignmask_dq    = _mm256_setzero_si256();
  __m256i vmagmask_dq     = _mm256_setzero_si256();
  if constexpr (fuse_dequant) {
    const int32_t M_b_val = block->get_Mb();
    pLSB_dq               = 31 - M_b_val;
    vmagmask_dq            = _mm256_set1_epi32(0x7FFFFFFF);
    vsignmask_dq           = _mm256_set1_epi32(INT32_MIN);
    if (block->transformation != 1) {
      // lossy path (transformation==0 for irrev97, transformation>=2 for ATK irrev)
      fscale_direct = block->stepsize;
      fscale_direct *= static_cast<float>(1 << FRACBITS);
      if (M_b_val <= 31)
        fscale_direct /= static_cast<float>(1 << (31 - M_b_val));
      else
        fscale_direct *= static_cast<float>(1 << (M_b_val - 31));
      vfscale = _mm256_set1_ps(fscale_direct);
    }
  }

  int32_t *const sample_buf = block->sample_buf;
  // When fusing dequantize, output pointers target i_samples (float*) instead of sample_buf (int32_t*).
  // Both are 32-bit wide so pointer arithmetic is identical.
  int32_t *mp0 = fuse_dequant ? reinterpret_cast<int32_t *>(block->i_samples) : sample_buf;
  int32_t *mp1 = mp0 + (fuse_dequant ? block->band_stride : block->blksampl_stride);

  alignas(32) int32_t Eline[1040];  // 2 * QW_max + 16, QW_max = 512
  std::memset(Eline, 0, (2U * QW + 16U) * sizeof(int32_t));
  int32_t *E_p = Eline + 1;

  __m128i v_n, mu0_n, mu1_n;
  fwd_buf<0xFF> MagSgn(compressed_data, Pcup);

  // Shuffle constants to expand 4-quad 16-bit results to 32-bit.
  // decode_four_quads returns row256 as 16 × int16_t where consecutive pairs
  // within each quad are (row0_sample, row1_sample). Placing each int16_t in the
  // upper 16 bits of a 32-bit slot moves the sign from bit 15 to bit 31, which
  // matches OpenHTJ2K's fixed-point format (sign at bit 31).
  // shuffle_r0 extracts even-indexed 16-bit elements (row 0 samples).
  // shuffle_r1 extracts odd-indexed 16-bit elements (row 1 samples).
  const __m256i shuffle_r0 = _mm256_set_epi8(
      0x0D, 0x0C, -1, -1, 0x09, 0x08, -1, -1, 0x05, 0x04, -1, -1, 0x01, 0x00, -1, -1,
      0x0D, 0x0C, -1, -1, 0x09, 0x08, -1, -1, 0x05, 0x04, -1, -1, 0x01, 0x00, -1, -1);
  const __m256i shuffle_r1 = _mm256_set_epi8(
      0x0F, 0x0E, -1, -1, 0x0B, 0x0A, -1, -1, 0x07, 0x06, -1, -1, 0x03, 0x02, -1, -1,
      0x0F, 0x0E, -1, -1, 0x0B, 0x0A, -1, -1, 0x07, 0x06, -1, -1, 0x03, 0x02, -1, -1);

  // When pLSB > 16 (mmsbp2 = 32 - pLSB < 16), decoded sample values fit in 16 bits
  // so we use the faster 4-quad 16-bit path which processes twice as many quads per
  // SIMD iteration.  For pLSB <= 16 we fall back to the 32-bit 2-quad path.
  if (pLSB > 16) {
    const uint8_t pLSB_adj = pLSB - 16;

    // Initial line-pair — 4 quads at a time
    sp = scratch;
    for (qx = QW; qx >= 4; qx -= 4, sp += 8, mp0 += 8, mp1 += 8) {
      v_n              = _mm_setzero_si128();
      __m128i qinf128  = _mm_loadu_si128((__m128i *)sp);
      __m128i U_q128   = _mm_srli_epi32(qinf128, 16);
      __m256i row256   = MagSgn.decode_four_quads(qinf128, U_q128, pLSB_adj, v_n);

      if constexpr (fuse_dequant) {
        dequant_store_256(mp0, _mm256_shuffle_epi8(row256, shuffle_r0), block->transformation, pLSB_dq,
                          vfscale, vmagmask_dq, vsignmask_dq);
        dequant_store_256(mp1, _mm256_shuffle_epi8(row256, shuffle_r1), block->transformation, pLSB_dq,
                          vfscale, vmagmask_dq, vsignmask_dq);
      } else {
        _mm256_storeu_si256((__m256i *)mp0, _mm256_shuffle_epi8(row256, shuffle_r0));
        _mm256_storeu_si256((__m256i *)mp1, _mm256_shuffle_epi8(row256, shuffle_r1));
      }

      __m256i vn32 = _mm256_cvtepu16_epi32(v_n);
      vn32 = avx2_lzcnt_epi32(vn32);
      vn32 = _mm256_sub_epi32(_mm256_set1_epi32(32), vn32);
      _mm256_storeu_si256((__m256i *)E_p, vn32);
      E_p += 8;
    }
    // Handle remaining 0 or 2 quads — use decode_one_quad per quad so E_p is
    // written per-column (consistent with the 4-quad loop's per-column format).
    for (; qx > 0; qx -= 2, sp += 4, mp0 += 4, mp1 += 4) {
      v_n             = _mm_setzero_si128();
      __m128i qinf128 = _mm_loadu_si128((__m128i *)sp);
      __m128i U_q128  = _mm_srli_epi32(qinf128, 16);
      mu0_n           = MagSgn.decode_one_quad<0>(qinf128, U_q128, pLSB, v_n);
      mu1_n           = MagSgn.decode_one_quad<1>(qinf128, U_q128, pLSB, v_n);
      // v_n now has per-column (row-1 only) values for 4 columns.

      auto t0 = _mm_unpacklo_epi32(mu0_n, mu1_n);
      auto t1 = _mm_unpackhi_epi32(mu0_n, mu1_n);
      mu0_n   = _mm_unpacklo_epi32(t0, t1);
      mu1_n   = _mm_unpackhi_epi32(t0, t1);
      if constexpr (fuse_dequant) {
        dequant_store_128(mp0, mu0_n, block->transformation, pLSB_dq, vfscale,
                          _mm256_castsi256_si128(vmagmask_dq), _mm256_castsi256_si128(vsignmask_dq));
        dequant_store_128(mp1, mu1_n, block->transformation, pLSB_dq, vfscale,
                          _mm256_castsi256_si128(vmagmask_dq), _mm256_castsi256_si128(vsignmask_dq));
      } else {
        _mm_storeu_si128((__m128i *)mp0, mu0_n);
        _mm_storeu_si128((__m128i *)mp1, mu1_n);
      }
      v_n = sse_lzcnt_epi32(v_n);
      v_n = _mm_sub_epi32(_mm_set1_epi32(32), v_n);
      _mm_storeu_si128((__m128i *)E_p, v_n);
      E_p += 4;
    }

    // Non-initial line-pairs
    for (uint16_t row = 1; row < QH; row++) {
      E_p = Eline + 1;
      if constexpr (fuse_dequant) {
        mp0 = reinterpret_cast<int32_t *>(block->i_samples) + (row * 2U) * block->band_stride;
        mp1 = mp0 + block->band_stride;
      } else {
        mp0 = sample_buf + (row * 2U) * block->blksampl_stride;
        mp1 = mp0 + block->blksampl_stride;
      }
      sp  = scratch + row * sstr;

      // Vectorized Emax: sliding max over 4-column windows using AVX2.
      // Emax[q] = max(E_p[2q-1], E_p[2q], E_p[2q+1], E_p[2q+2]).
      // Two 8-wide loads offset by 2 → max → per-lane shift → pairwise max → permute.
      const __m256i perm_emax = _mm256_set_epi32(0, 0, 0, 0, 6, 4, 2, 0);
      __m256i elo     = _mm256_loadu_si256((__m256i *)(E_p - 1));
      __m256i ehi     = _mm256_loadu_si256((__m256i *)(E_p + 1));
      __m256i emx     = _mm256_max_epi32(elo, ehi);
      __m256i epr     = _mm256_max_epi32(emx, _mm256_srli_si256(emx, 4));
      __m128i emax128 = _mm256_castsi256_si128(
          _mm256_permutevar8x32_epi32(epr, perm_emax));

      for (qx = QW; qx >= 4; qx -= 4, sp += 8, mp0 += 8, mp1 += 8) {
        v_n             = _mm_setzero_si128();
        __m128i qinf128 = _mm_loadu_si128((__m128i *)sp);

        // Compute kappa for all 4 quads using vectorized emax128.
        __m128i rho128   = _mm_and_si128(qinf128, _mm_set1_epi32(0x00F0));
        __m128i gm1      = _mm_sub_epi32(rho128, _mm_set1_epi32(1));
        __m128i gamma128 = _mm_cmpeq_epi32(_mm_and_si128(rho128, gm1), _mm_setzero_si128());
        __m128i em1      = _mm_sub_epi32(emax128, _mm_set1_epi32(1));
        em1              = _mm_andnot_si128(gamma128, em1);
        __m128i kappa128 = _mm_max_epi32(em1, _mm_set1_epi32(1));
        __m128i U_q128   = _mm_add_epi32(_mm_srli_epi32(qinf128, 16), kappa128);

        __m256i row256 = MagSgn.decode_four_quads(qinf128, U_q128, pLSB_adj, v_n);

        if constexpr (fuse_dequant) {
          dequant_store_256(mp0, _mm256_shuffle_epi8(row256, shuffle_r0), block->transformation, pLSB_dq,
                            vfscale, vmagmask_dq, vsignmask_dq);
          dequant_store_256(mp1, _mm256_shuffle_epi8(row256, shuffle_r1), block->transformation, pLSB_dq,
                            vfscale, vmagmask_dq, vsignmask_dq);
        } else {
          _mm256_storeu_si256((__m256i *)mp0, _mm256_shuffle_epi8(row256, shuffle_r0));
          _mm256_storeu_si256((__m256i *)mp1, _mm256_shuffle_epi8(row256, shuffle_r1));
        }

        // Read-ahead: vectorized Emax for next 4 quads BEFORE writing E_p.
        elo     = _mm256_loadu_si256((__m256i *)(E_p + 7));
        ehi     = _mm256_loadu_si256((__m256i *)(E_p + 9));
        emx     = _mm256_max_epi32(elo, ehi);
        epr     = _mm256_max_epi32(emx, _mm256_srli_si256(emx, 4));
        emax128 = _mm256_castsi256_si128(
            _mm256_permutevar8x32_epi32(epr, perm_emax));

        __m256i vn32 = _mm256_cvtepu16_epi32(v_n);
        vn32 = avx2_lzcnt_epi32(vn32);
        vn32 = _mm256_sub_epi32(_mm256_set1_epi32(32), vn32);
        _mm256_storeu_si256((__m256i *)E_p, vn32);
        E_p += 8;
      }
      // Remaining 0 or 2 quads — emax128[0..1] already hold the correct Emax values.
      for (; qx > 0; qx -= 2, sp += 4, mp0 += 4, mp1 += 4) {
        v_n             = _mm_setzero_si128();
        __m128i qinf128 = _mm_loadu_si128((__m128i *)sp);
        __m256i qinf256 = expand_two_quads(qinf128);
        __m256i U_q256;
        {
          // Compute kappa without scalar extracts: broadcast Emax-1 per quad lane.
          __m128i emax_m1 = _mm_sub_epi32(emax128, _mm_set1_epi32(1));
          __m256i gamma   = _mm256_and_si256(qinf256, _mm256_set1_epi32(0xF0));
          __m256i gm1     = _mm256_sub_epi32(gamma, _mm256_set1_epi32(1));
          gamma           = _mm256_and_si256(gamma, gm1);
          gamma           = _mm256_cmpeq_epi32(gamma, _mm256_setzero_si256());
          __m256i emax256 = _mm256_set_m128i(_mm_shuffle_epi32(emax_m1, _MM_SHUFFLE(1, 1, 1, 1)),
                                             _mm_shuffle_epi32(emax_m1, _MM_SHUFFLE(0, 0, 0, 0)));
          emax256         = _mm256_andnot_si256(gamma, emax256);
          __m256i kappa   = _mm256_max_epi32(emax256, _mm256_set1_epi32(1));
          U_q256          = _mm256_add_epi32(_mm256_srli_epi32(qinf256, 16), kappa);
        }
        __m128i U_q128 = _mm_unpacklo_epi32(_mm256_castsi256_si128(U_q256),
                                             _mm256_extracti128_si256(U_q256, 1));
        mu0_n          = MagSgn.decode_one_quad<0>(qinf128, U_q128, pLSB, v_n);
        mu1_n          = MagSgn.decode_one_quad<1>(qinf128, U_q128, pLSB, v_n);

        auto t0 = _mm_unpacklo_epi32(mu0_n, mu1_n);
        auto t1 = _mm_unpackhi_epi32(mu0_n, mu1_n);
        mu0_n   = _mm_unpacklo_epi32(t0, t1);
        mu1_n   = _mm_unpackhi_epi32(t0, t1);
        if constexpr (fuse_dequant) {
          dequant_store_128(mp0, mu0_n, block->transformation, pLSB_dq, vfscale,
                            _mm256_castsi256_si128(vmagmask_dq), _mm256_castsi256_si128(vsignmask_dq));
          dequant_store_128(mp1, mu1_n, block->transformation, pLSB_dq, vfscale,
                            _mm256_castsi256_si128(vmagmask_dq), _mm256_castsi256_si128(vsignmask_dq));
        } else {
          _mm_storeu_si128((__m128i *)mp0, mu0_n);
          _mm_storeu_si128((__m128i *)mp1, mu1_n);
        }
        // Read-ahead Emax for next 2 quads.
        __m128i elo128 = _mm_loadu_si128((__m128i *)(E_p + 3));
        __m128i ehi128 = _mm_loadu_si128((__m128i *)(E_p + 5));
        __m128i emx128 = _mm_max_epi32(elo128, ehi128);
        __m128i epr128 = _mm_max_epi32(emx128, _mm_srli_si128(emx128, 4));
        emax128 = _mm_shuffle_epi32(epr128, _MM_SHUFFLE(3, 3, 2, 0));
        v_n   = sse_lzcnt_epi32(v_n);
        v_n   = _mm_sub_epi32(_mm_set1_epi32(32), v_n);
        _mm_storeu_si128((__m128i *)E_p, v_n);
        E_p += 4;
      }
    }
  } else {
    // 32-bit 2-quad path (used when pLSB <= 16, i.e. high-precision / heavy lossy)
    // Initial line-pair
    sp = scratch;
    for (qx = QW; qx > 0; qx -= 2, sp += 4, mp0 += 4, mp1 += 4) {
      v_n             = _mm_setzero_si128();
      __m128i qinf128 = _mm_loadu_si128((__m128i *)sp);
      __m256i qinf256 = expand_two_quads(qinf128);
      __m256i U_q256  = _mm256_srli_epi32(qinf256, 16);
      __m256i row256  = MagSgn.decode_two_quads(qinf256, U_q256, pLSB, v_n);

      mu0_n   = _mm256_castsi256_si128(row256);
      mu1_n   = _mm256_extracti128_si256(row256, 1);
      auto t0 = _mm_unpacklo_epi32(mu0_n, mu1_n);
      auto t1 = _mm_unpackhi_epi32(mu0_n, mu1_n);
      mu0_n   = _mm_unpacklo_epi32(t0, t1);
      mu1_n   = _mm_unpackhi_epi32(t0, t1);
      if constexpr (fuse_dequant) {
        dequant_store_128(mp0, mu0_n, block->transformation, pLSB_dq, vfscale,
                          _mm256_castsi256_si128(vmagmask_dq), _mm256_castsi256_si128(vsignmask_dq));
        dequant_store_128(mp1, mu1_n, block->transformation, pLSB_dq, vfscale,
                          _mm256_castsi256_si128(vmagmask_dq), _mm256_castsi256_si128(vsignmask_dq));
      } else {
        _mm_storeu_si128((__m128i *)mp0, mu0_n);
        _mm_storeu_si128((__m128i *)mp1, mu1_n);
      }
      v_n = sse_lzcnt_epi32(v_n);
      v_n = _mm_sub_epi32(_mm_set1_epi32(32), v_n);
      _mm_storeu_si128((__m128i *)E_p, v_n);
      E_p += 4;
    }
    // Non-initial line-pairs
    for (uint16_t row = 1; row < QH; row++) {
      E_p = Eline + 1;
      if constexpr (fuse_dequant) {
        mp0 = reinterpret_cast<int32_t *>(block->i_samples) + (row * 2U) * block->band_stride;
        mp1 = mp0 + block->band_stride;
      } else {
        mp0 = sample_buf + (row * 2U) * block->blksampl_stride;
        mp1 = mp0 + block->blksampl_stride;
      }
      sp  = scratch + row * sstr;

      // Vectorized Emax for 2-quad path: sliding max over 4-column windows.
      // epr128 positions {0, 2} hold {Emax[0], Emax[1]}; kept as vector to avoid scalar extracts.
      __m128i elo128 = _mm_loadu_si128((__m128i *)(E_p - 1));
      __m128i ehi128 = _mm_loadu_si128((__m128i *)(E_p + 1));
      __m128i emx128 = _mm_max_epi32(elo128, ehi128);
      __m128i epr128 = _mm_max_epi32(emx128, _mm_srli_si128(emx128, 4));
      for (qx = QW; qx > 0; qx -= 2, sp += 4, mp0 += 4, mp1 += 4) {
        v_n             = _mm_setzero_si128();
        __m128i qinf128 = _mm_loadu_si128((__m128i *)sp);
        __m256i qinf256 = expand_two_quads(qinf128);
        __m256i U_q256;
        {
          // Compute kappa without scalar extracts: broadcast (Emax-1) per quad lane.
          __m128i emax_m1 = _mm_sub_epi32(epr128, _mm_set1_epi32(1));
          __m256i gamma   = _mm256_and_si256(qinf256, _mm256_set1_epi32(0xF0));
          __m256i gm1     = _mm256_sub_epi32(gamma, _mm256_set1_epi32(1));
          gamma           = _mm256_and_si256(gamma, gm1);
          gamma           = _mm256_cmpeq_epi32(gamma, _mm256_setzero_si256());
          __m256i emax256 = _mm256_set_m128i(_mm_shuffle_epi32(emax_m1, _MM_SHUFFLE(2, 2, 2, 2)),
                                             _mm_shuffle_epi32(emax_m1, _MM_SHUFFLE(0, 0, 0, 0)));
          emax256         = _mm256_andnot_si256(gamma, emax256);
          __m256i kappa   = _mm256_max_epi32(emax256, _mm256_set1_epi32(1));
          U_q256          = _mm256_add_epi32(_mm256_srli_epi32(qinf256, 16), kappa);
        }
        __m256i row256 = MagSgn.decode_two_quads(qinf256, U_q256, pLSB, v_n);
        mu0_n          = _mm256_castsi256_si128(row256);
        mu1_n          = _mm256_extracti128_si256(row256, 1);
        auto t0        = _mm_unpacklo_epi32(mu0_n, mu1_n);
        auto t1        = _mm_unpackhi_epi32(mu0_n, mu1_n);
        mu0_n          = _mm_unpacklo_epi32(t0, t1);
        mu1_n          = _mm_unpackhi_epi32(t0, t1);
        if constexpr (fuse_dequant) {
          dequant_store_128(mp0, mu0_n, block->transformation, pLSB_dq, vfscale,
                            _mm256_castsi256_si128(vmagmask_dq), _mm256_castsi256_si128(vsignmask_dq));
          dequant_store_128(mp1, mu1_n, block->transformation, pLSB_dq, vfscale,
                            _mm256_castsi256_si128(vmagmask_dq), _mm256_castsi256_si128(vsignmask_dq));
        } else {
          _mm_storeu_si128((__m128i *)mp0, mu0_n);
          _mm_storeu_si128((__m128i *)mp1, mu1_n);
        }
        // Read-ahead: vectorized Emax for next 2 quads BEFORE writing E_p.
        elo128 = _mm_loadu_si128((__m128i *)(E_p + 3));
        ehi128 = _mm_loadu_si128((__m128i *)(E_p + 5));
        emx128 = _mm_max_epi32(elo128, ehi128);
        epr128 = _mm_max_epi32(emx128, _mm_srli_si128(emx128, 4));
        v_n    = sse_lzcnt_epi32(v_n);
        v_n    = _mm_sub_epi32(_mm_set1_epi32(32), v_n);
        _mm_storeu_si128((__m128i *)E_p, v_n);
        E_p += 4;
      }
    }
  }  // end if (pLSB > 16)
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
  // Zero the extra row below the block (needed for below-stripe MBR context)
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

      // Load current + right-neighbor sigma as 32 bits (for horizontal MBR context)
      uint32_t cs = *reinterpret_cast<const uint32_t *>(cur_sig + qx);
      uint32_t ps = *reinterpret_cast<const uint32_t *>(prev_row_sig + qx);
      uint32_t ns = *reinterpret_cast<const uint32_t *>(cur_sig + mstr + qx);

      // Context from stripe above (bottom row) and below (top row)
      uint32_t u = (ps & 0x88888888u) >> 3;
      if (non_causal) u |= (ns & 0x11111111u) << 3;

      // Vertical MBR integration
      uint32_t mbr = cs;
      mbr |= (cs & 0x77777777u) << 1;
      mbr |= (cs & 0xEEEEEEEEu) >> 1;
      mbr |= u;
      // Horizontal MBR integration
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
        // CTZ iteration: process only set mbr bits in column-major order
        // (nibble layout guarantees ascending bit positions = column-major).
        // Branchless bit-result handling eliminates the unpredictable
        // if(importSigPropBit()) branch (~50% mispredict rate).
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

        // Write magnitude + sign using CTZ (avoids 32 branch checks)
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

      // Update prev_row_sig for above-stripe context in next stripe
      // Do NOT update sigma — it must retain cleanup-only significance for MRP
      new_sig |= cs & 0xFFFFu;
      prev_row_sig[qx] = static_cast<uint16_t>(new_sig);

      // Prepare left-neighbor context for next quad
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

      // CTZ iteration: process only significant samples (column-major order
      // preserved by nibble layout). Avoids 16 per-quad if(sig & bit) checks.
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
  const uint32_t mask = UINT32_MAX >> (M_b + 1);

  const __m256i magmask = _mm256_set1_epi32(0x7FFFFFFF);
  const __m256i vmask   = _mm256_set1_epi32(static_cast<int32_t>(~mask));
  const __m256i zero    = _mm256_setzero_si256();
  const __m256i shift   = _mm256_set1_epi32(ROIshift);
  __m256i v0, v1, s0, s1, vdst0, vdst1, vROImask;
  if (this->transformation == 1) {
    // lossless path
    for (size_t i = 0; i < static_cast<size_t>(this->size.y); i++) {
      int32_t *val = this->sample_buf + i * this->blksampl_stride;
      sprec_t *dst = this->i_samples + i * this->band_stride;
      size_t len   = this->size.x;
      if (ROIshift == 0) {
        // Common case: no ROI — skip the ROI upshift entirely.
        // val = sample_buf + i * blksampl_stride; blksampl_stride = round_up(width,8) is a
        // multiple of 8 int32 = 32 bytes, and sample_buf is 32-byte aligned → _mm256_load_si256 ok.
        for (; len >= 16; len -= 16) {
          v0    = _mm256_load_si256((__m256i *)val);
          v1    = _mm256_load_si256((__m256i *)(val + 8));
          s0    = v0;
          s1    = v1;
          v0    = _mm256_and_si256(v0, magmask);
          v1    = _mm256_and_si256(v1, magmask);
          vdst0 = _mm256_sign_epi32(_mm256_srai_epi32(v0, pLSB), s0);
          vdst1 = _mm256_sign_epi32(_mm256_srai_epi32(v1, pLSB), s1);
          _mm256_storeu_ps(dst, _mm256_cvtepi32_ps(vdst0));
          _mm256_storeu_ps(dst + 8, _mm256_cvtepi32_ps(vdst1));
          val += 16;
          dst += 16;
        }
        for (; len > 0; --len) {
          int32_t sign = *val & INT32_MIN;
          *val &= INT32_MAX;
          *val >>= pLSB;
          if (sign) *val = -(*val & INT32_MAX);
          *dst = static_cast<float>(*val);
          val++;
          dst++;
        }
      } else {
        for (; len >= 16; len -= 16) {
          v0 = _mm256_load_si256((__m256i *)val);
          v1 = _mm256_load_si256((__m256i *)(val + 8));
          s0 = v0;
          s1 = v1;
          v0 = _mm256_and_si256(v0, magmask);
          v1 = _mm256_and_si256(v1, magmask);
          // upshift background region, if necessary
          vROImask = _mm256_and_si256(v0, vmask);
          vROImask = _mm256_cmpeq_epi32(vROImask, zero);
          vROImask = _mm256_and_si256(vROImask, shift);
          v0       = _mm256_sllv_epi32(v0, vROImask);
          vROImask = _mm256_and_si256(v1, vmask);
          vROImask = _mm256_cmpeq_epi32(vROImask, zero);
          vROImask = _mm256_and_si256(vROImask, shift);
          v1       = _mm256_sllv_epi32(v1, vROImask);
          // convert values from sign-magnitude form to two's complement one
          vdst0 = _mm256_sign_epi32(_mm256_srai_epi32(v0, pLSB), s0);
          vdst1 = _mm256_sign_epi32(_mm256_srai_epi32(v1, pLSB), s1);
          _mm256_storeu_ps(dst, _mm256_cvtepi32_ps(vdst0));
          _mm256_storeu_ps(dst + 8, _mm256_cvtepi32_ps(vdst1));
          val += 16;
          dst += 16;
        }
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
          *dst = static_cast<float>(*val);
          val++;
          dst++;
        }
      }
    }
  } else {
    // lossy path: compute the direct float scale factor
    // decoded magnitude is in Q(31-M_b) fixed-point; result must be in Q(FRACBITS)
    float fscale_direct = this->stepsize;
    fscale_direct *= static_cast<float>(1 << FRACBITS);
    if (M_b <= 31)
      fscale_direct /= static_cast<float>(1 << (31 - M_b));
    else
      fscale_direct *= static_cast<float>(1 << (M_b - 31));

    if (ROIshift == 0) {
      // Common case: no ROI — direct float multiply, sign via XOR.
      const __m256 vfscale   = _mm256_set1_ps(fscale_direct);
      const __m256i vsignmask = _mm256_set1_epi32(INT32_MIN);
      for (size_t i = 0; i < static_cast<size_t>(this->size.y); i++) {
        int32_t *val = this->sample_buf + i * this->blksampl_stride;
        sprec_t *dst = this->i_samples + i * this->band_stride;
        size_t len   = this->size.x;
        // 2× unrolled: 4 vectors (32 elements) per iteration for better ILP
        // val = sample_buf + i*blksampl_stride; 32-byte aligned → _mm256_load_si256 ok.
        for (; len >= 32; len -= 32) {
          __m256i a0 = _mm256_load_si256((__m256i *)val);
          __m256i a1 = _mm256_load_si256((__m256i *)(val + 8));
          __m256i a2 = _mm256_load_si256((__m256i *)(val + 16));
          __m256i a3 = _mm256_load_si256((__m256i *)(val + 24));
          __m256i m0 = _mm256_and_si256(a0, magmask);
          __m256i m1 = _mm256_and_si256(a1, magmask);
          __m256i m2 = _mm256_and_si256(a2, magmask);
          __m256i m3 = _mm256_and_si256(a3, magmask);
          __m256 f0  = _mm256_mul_ps(_mm256_cvtepi32_ps(m0), vfscale);
          __m256 f1  = _mm256_mul_ps(_mm256_cvtepi32_ps(m1), vfscale);
          __m256 f2  = _mm256_mul_ps(_mm256_cvtepi32_ps(m2), vfscale);
          __m256 f3  = _mm256_mul_ps(_mm256_cvtepi32_ps(m3), vfscale);
          f0 = _mm256_xor_ps(f0, _mm256_castsi256_ps(_mm256_and_si256(a0, vsignmask)));
          f1 = _mm256_xor_ps(f1, _mm256_castsi256_ps(_mm256_and_si256(a1, vsignmask)));
          f2 = _mm256_xor_ps(f2, _mm256_castsi256_ps(_mm256_and_si256(a2, vsignmask)));
          f3 = _mm256_xor_ps(f3, _mm256_castsi256_ps(_mm256_and_si256(a3, vsignmask)));
          _mm256_storeu_ps(dst, f0);
          _mm256_storeu_ps(dst + 8, f1);
          _mm256_storeu_ps(dst + 16, f2);
          _mm256_storeu_ps(dst + 24, f3);
          val += 32;
          dst += 32;
        }
        for (; len >= 8; len -= 8) {
          __m256i a0 = _mm256_load_si256((__m256i *)val);
          __m256i m0 = _mm256_and_si256(a0, magmask);
          __m256 f0  = _mm256_mul_ps(_mm256_cvtepi32_ps(m0), vfscale);
          f0 = _mm256_xor_ps(f0, _mm256_castsi256_ps(_mm256_and_si256(a0, vsignmask)));
          _mm256_storeu_ps(dst, f0);
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
        for (; len >= 16; len -= 16) {
          v0 = _mm256_load_si256((__m256i *)val);
          v1 = _mm256_load_si256((__m256i *)(val + 8));
          s0 = v0;
          s1 = v1;
          v0 = _mm256_and_si256(v0, magmask);
          v1 = _mm256_and_si256(v1, magmask);
          // upshift background region
          vROImask = _mm256_and_si256(v0, vmask);
          vROImask = _mm256_cmpeq_epi32(vROImask, zero);
          vROImask = _mm256_and_si256(vROImask, shift);
          v0       = _mm256_sllv_epi32(v0, vROImask);
          vROImask = _mm256_and_si256(v1, vmask);
          vROImask = _mm256_cmpeq_epi32(vROImask, zero);
          vROImask = _mm256_and_si256(vROImask, shift);
          v1       = _mm256_sllv_epi32(v1, vROImask);
          // truncate to int16 range
          v0 = _mm256_srai_epi32(_mm256_add_epi32(v0, _mm256_set1_epi32(1 << 15)), 16);
          v1 = _mm256_srai_epi32(_mm256_add_epi32(v1, _mm256_set1_epi32(1 << 15)), 16);
          // dequantization
          v0 = _mm256_mullo_epi32(v0, _mm256_set1_epi32(scale));
          v1 = _mm256_mullo_epi32(v1, _mm256_set1_epi32(scale));
          // downshift and sign
          v0 = _mm256_srai_epi32(_mm256_add_epi32(v0, _mm256_set1_epi32(1 << (downshift - 1))), downshift);
          v1 = _mm256_srai_epi32(_mm256_add_epi32(v1, _mm256_set1_epi32(1 << (downshift - 1))), downshift);
          v0 = _mm256_sign_epi32(v0, s0);
          v1 = _mm256_sign_epi32(v1, s1);
          _mm256_storeu_ps(dst, _mm256_cvtepi32_ps(v0));
          _mm256_storeu_ps(dst + 8, _mm256_cvtepi32_ps(v1));
          val += 16;
          dst += 16;
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
    uint8_t  all_segments[4] = {};
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
    for (uint32_t i = 1; i < num_segments; i++) {
      Lref += block->pass_length[all_segments[i]];
    }
    Dcup = block->get_compressed_data();

    if (block->num_passes > 1 && num_segments > 1) {
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

    // Single HT pass with no ROI: use fused dequantize path to eliminate
    // the separate dequantize pass over sample_buf.
    bool dequant_done = false;
    // Fused dequant: the MagSgn SIMD stores write in units of 4 (128-bit) or 8 (256-bit)
    // elements.  When block width is not a multiple of 4, the extra elements overflow into
    // adjacent blocks' column range in the shared output buffer (subband or ring buffer).
    // This is safe in single-threaded decode (sequential order overwrites correctly) but
    // causes a data race in multi-threaded decode.  Gate on width % 4 == 0 to avoid this.
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

    // dequantization (skipped when already fused into MagSgn output)
    if (!dequant_done) {
      block->dequantize(ROIshift);
    }

  }  // end

  return true;
}
#endif