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

uint8_t j2k_codeblock::calc_mbr(const uint32_t i, const uint32_t j, const uint8_t causal_cond) const {
  uint8_t *state_p0 = block_states + static_cast<size_t>(i) * blkstate_stride + j;
  uint8_t *state_p1 = block_states + static_cast<size_t>(i + 1) * blkstate_stride + j;
  uint8_t *state_p2 = block_states + static_cast<size_t>(i + 2) * blkstate_stride + j;

  uint32_t mbr0 = state_p0[0] | state_p0[1] | state_p0[2];
  uint32_t mbr1 = state_p1[0] | state_p1[2];
  uint32_t mbr2 = state_p2[0] | state_p2[1] | state_p2[2];
  uint32_t mbr  = mbr0 | mbr1 | (mbr2 & causal_cond);
  mbr |= (mbr0 >> SHIFT_REF) & (mbr0 >> SHIFT_SCAN);
  mbr |= (mbr1 >> SHIFT_REF) & (mbr1 >> SHIFT_SCAN);
  mbr |= (mbr2 >> SHIFT_REF) & (mbr2 >> SHIFT_SCAN) & causal_cond;
  return mbr & 1;
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

// Build __m256i with lower 128 = broadcast of qinf[0], upper = broadcast of qinf[1].
static FORCE_INLINE __m256i expand_two_quads(__m128i qinf128) {
  return _mm256_set_m128i(_mm_shuffle_epi32(qinf128, _MM_SHUFFLE(1, 1, 1, 1)),
                          _mm_shuffle_epi32(qinf128, _MM_SHUFFLE(0, 0, 0, 0)));
}

template <bool skip_sigma>
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
  int32_t *const sample_buf = block->sample_buf;
  int32_t *mp0              = sample_buf;
  int32_t *mp1              = sample_buf + block->blksampl_stride;

  alignas(32) int32_t Eline[1032];  // 2 * QW_max + 8, QW_max = 512
  std::memset(Eline, 0, (2U * QW + 8U) * sizeof(int32_t));
  int32_t *E_p = Eline + 1;

  __m128i v_n, mu0_n, mu1_n;
  fwd_buf<0xFF> MagSgn(compressed_data, Pcup);

  // Initial line-pair
  sp = scratch;
  for (qx = QW; qx > 0; qx -= 2, sp += 4) {
    v_n              = _mm_setzero_si128();
    __m128i qinf128  = _mm_loadu_si128((__m128i *)sp);
    __m256i qinf256  = expand_two_quads(qinf128);
    __m256i U_q256   = _mm256_srli_epi32(qinf256, 16);
    __m256i row256   = MagSgn.decode_two_quads(qinf256, U_q256, pLSB, v_n);

    // Transpose and store: lower 128 = quad 0, upper 128 = quad 1.
    mu0_n = _mm256_castsi256_si128(row256);
    mu1_n = _mm256_extracti128_si256(row256, 1);
    auto t0 = _mm_unpacklo_epi32(mu0_n, mu1_n);  // 0, 4, 1, 5
    auto t1 = _mm_unpackhi_epi32(mu0_n, mu1_n);  // 2, 6, 3, 7
    mu0_n   = _mm_unpacklo_epi32(t0, t1);        // 0, 2, 4, 6
    mu1_n   = _mm_unpackhi_epi32(t0, t1);        // 1, 3, 5, 7
    _mm_storeu_si128((__m128i *)mp0, mu0_n);
    _mm_storeu_si128((__m128i *)mp1, mu1_n);
    mp0 += 4;
    mp1 += 4;

    // Update Exponent
    v_n = sse_lzcnt_epi32(v_n);
    v_n = _mm_sub_epi32(_mm_set1_epi32(32), v_n);
    _mm_storeu_si128((__m128i *)E_p, v_n);
    E_p += 4;
  }
  // Non-initial line-pair
  for (uint16_t row = 1; row < QH; row++) {
    E_p = Eline + 1;
    mp0 = sample_buf + (row * 2U) * block->blksampl_stride;
    mp1 = mp0 + block->blksampl_stride;

    sp = scratch + row * sstr;

    // Calculate Emax for the next two quads
    int32_t Emax0, Emax1;
    Emax0 = hMax(_mm_loadu_si128((__m128i *)(E_p - 1)));
    Emax1 = hMax(_mm_loadu_si128((__m128i *)(E_p + 1)));
    for (qx = QW; qx > 0; qx -= 2, sp += 4) {
      v_n             = _mm_setzero_si128();
      __m128i qinf128 = _mm_loadu_si128((__m128i *)sp);
      __m256i qinf256 = expand_two_quads(qinf128);
      __m256i U_q256;
      {
        // 256-bit gamma/kappa computation — both quads in one pass.
        // gamma: popcount(rho) < 2 → 0, else 1.  Computed as (x & (x-1)) == 0.
        __m256i gamma  = _mm256_and_si256(qinf256, _mm256_set1_epi32(0xF0));
        __m256i gm1    = _mm256_sub_epi32(gamma, _mm256_set1_epi32(1));
        gamma          = _mm256_and_si256(gamma, gm1);
        gamma          = _mm256_cmpeq_epi32(gamma, _mm256_setzero_si256());

        // emax: quad 0 uses Emax0-1, quad 1 uses Emax1-1.
        __m256i emax256 = _mm256_set_m128i(_mm_set1_epi32(Emax1 - 1),
                                            _mm_set1_epi32(Emax0 - 1));
        emax256         = _mm256_andnot_si256(gamma, emax256);
        __m256i kappa   = _mm256_max_epi16(emax256, _mm256_set1_epi32(1));

        U_q256 = _mm256_add_epi32(_mm256_srli_epi32(qinf256, 16), kappa);
      }
      __m256i row256 = MagSgn.decode_two_quads(qinf256, U_q256, pLSB, v_n);

      // Transpose and store
      mu0_n = _mm256_castsi256_si128(row256);
      mu1_n = _mm256_extracti128_si256(row256, 1);
      auto t0 = _mm_unpacklo_epi32(mu0_n, mu1_n);
      auto t1 = _mm_unpackhi_epi32(mu0_n, mu1_n);
      mu0_n   = _mm_unpacklo_epi32(t0, t1);
      mu1_n   = _mm_unpackhi_epi32(t0, t1);
      _mm_storeu_si128((__m128i *)mp0, mu0_n);
      _mm_storeu_si128((__m128i *)mp1, mu1_n);
      mp0 += 4;
      mp1 += 4;

      // Update Exponent
      Emax0 = hMax(_mm_loadu_si128((__m128i *)(E_p + 3)));
      Emax1 = hMax(_mm_loadu_si128((__m128i *)(E_p + 5)));
      v_n = sse_lzcnt_epi32(v_n);
      v_n = _mm_sub_epi32(_mm_set1_epi32(32), v_n);
      _mm_storeu_si128((__m128i *)E_p, v_n);
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
        state_p[0] |= static_cast<uint8_t>(bit << SHIFT_REF);
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
  for (uint32_t n1 = 0; n1 < num_v_stripe; n1++) {
    j_start = 0;
    for (uint32_t n2 = 0; n2 < num_h_stripe; n2++) {
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
  for (uint32_t n2 = 0; n2 < num_h_stripe; n2++) {
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

    // dequantization
    block->dequantize(ROIshift);

  }  // end

  return true;
}
#endif