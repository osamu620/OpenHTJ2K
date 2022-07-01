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

uint8_t j2k_codeblock::calc_mbr(const int16_t i, const int16_t j, const uint8_t causal_cond) const {
  const int16_t im1 = static_cast<int16_t>(i - 1);
  const int16_t jm1 = static_cast<int16_t>(j - 1);
  const int16_t ip1 = static_cast<int16_t>(i + 1);
  const int16_t jp1 = static_cast<int16_t>(j + 1);
  uint8_t mbr       = get_state(Sigma, im1, jm1);
  mbr               = mbr | get_state(Sigma, im1, j);
  mbr               = mbr | get_state(Sigma, im1, jp1);
  mbr               = mbr | get_state(Sigma, i, jm1);
  mbr               = mbr | get_state(Sigma, i, jp1);
  mbr               = mbr | static_cast<uint8_t>(get_state(Sigma, ip1, jm1) * causal_cond);
  mbr               = mbr | static_cast<uint8_t>(get_state(Sigma, ip1, j) * causal_cond);
  mbr               = mbr | static_cast<uint8_t>(get_state(Sigma, ip1, jp1) * causal_cond);

  mbr = mbr | static_cast<uint8_t>(get_state(Refinement_value, im1, jm1) * get_state(Scan, im1, jm1));
  mbr = mbr | static_cast<uint8_t>(get_state(Refinement_value, im1, j) * get_state(Scan, im1, j));
  mbr = mbr | static_cast<uint8_t>(get_state(Refinement_value, im1, jp1) * get_state(Scan, im1, jp1));
  mbr = mbr | static_cast<uint8_t>(get_state(Refinement_value, i, jm1) * get_state(Scan, i, jm1));
  mbr = mbr | static_cast<uint8_t>(get_state(Refinement_value, i, jp1) * get_state(Scan, i, jp1));
  mbr = mbr
        | static_cast<uint8_t>(get_state(Refinement_value, ip1, jm1) * get_state(Scan, ip1, jm1)
                               * causal_cond);
  mbr = mbr
        | static_cast<uint8_t>(get_state(Refinement_value, ip1, j) * get_state(Scan, ip1, j) * causal_cond);
  mbr = mbr
        | static_cast<uint8_t>(get_state(Refinement_value, ip1, jp1) * get_state(Scan, ip1, jp1)
                               * causal_cond);
  return mbr;
}

void ht_cleanup_decode(j2k_codeblock *block, const uint8_t &pLSB, const int32_t Lcup, const int32_t Pcup,
                       const int32_t Scup) {
  fwd_buf<0xFF> MagSgn(block->get_compressed_data(), Pcup);
  MEL_dec MEL(block->get_compressed_data(), Lcup, Scup);
  rev_buf VLC_dec(block->get_compressed_data(), Lcup, Scup);
  const uint16_t QW = static_cast<uint16_t>(ceil_int(static_cast<int16_t>(block->size.x), 2));
  const uint16_t QH = static_cast<uint16_t>(ceil_int(static_cast<int16_t>(block->size.y), 2));

  __m128i vExp;
  auto mp0 = block->sample_buf.get();
  auto mp1 = block->sample_buf.get() + block->blksampl_stride;
  auto sp0 = block->block_states.get() + 1 + block->blkstate_stride;
  auto sp1 = block->block_states.get() + 1 + 2 * block->blkstate_stride;

  int32_t rho0, rho1;
  uint32_t u_off0, u_off1;
  int32_t emb_k_0, emb_k_1;
  int32_t emb_1_0, emb_1_1;
  uint32_t u0, u1;
  uint32_t U0, U1;
  uint8_t gamma0, gamma1;
  uint32_t kappa0 = 1, kappa1 = 1;  // kappa is always 1 for initial line-pair

  const uint16_t *dec_table0, *dec_table1;
  dec_table0 = dec_CxtVLC_table0_fast_16;
  dec_table1 = dec_CxtVLC_table1_fast_16;

  alignas(32) auto rholine = MAKE_UNIQUE<int32_t[]>(QW + 3U);
  rholine[0]               = 0;
  auto rho_p               = rholine.get() + 1;
  alignas(32) auto Eline   = MAKE_UNIQUE<int32_t[]>(2U * QW + 6U);
  Eline[0]                 = 0;
  auto E_p                 = Eline.get() + 1;

  int32_t context = 0;
  uint32_t vlcval;
  int32_t mel_run = MEL.get_run();

  // Initial line-pair
  for (int32_t q = 0; q < QW - 1; q += 2) {
    // Decoding of significance and EMB patterns and unsigned residual offsets
    vlcval       = VLC_dec.fetch();
    uint16_t tv0 = dec_table0[(vlcval & 0x7F) + (static_cast<unsigned int>(context << 7))];
    if (context == 0) {
      mel_run -= 2;
      tv0 = (mel_run == -1) ? tv0 : 0;
      if (mel_run < 0) {
        mel_run = MEL.get_run();
      }
    }

    rho0    = static_cast<uint8_t>(_pext_u32(tv0, 0x00F0U));
    emb_k_0 = static_cast<uint8_t>(_pext_u32(tv0, 0x0F00U));
    emb_1_0 = static_cast<uint8_t>(_pext_u32(tv0, 0xF000U));

    *rho_p++ = rho0;
    // calculate context for the next quad
    context = static_cast<uint16_t>((rho0 >> 1) | (rho0 & 1));

    // Decoding of significance and EMB patterns and unsigned residual offsets
    vlcval       = VLC_dec.advance(static_cast<uint8_t>((tv0 & 0x000F) >> 1));
    uint16_t tv1 = dec_table0[(vlcval & 0x7F) + (static_cast<unsigned int>(context << 7))];
    if (context == 0) {
      mel_run -= 2;
      tv1 = (mel_run == -1) ? tv1 : 0;
      if (mel_run < 0) {
        mel_run = MEL.get_run();
      }
    }

    rho1    = static_cast<uint8_t>(_pext_u32(tv1, 0x00F0U));
    emb_k_1 = static_cast<uint8_t>(_pext_u32(tv1, 0x0F00U));
    emb_1_1 = static_cast<uint8_t>(_pext_u32(tv1, 0xF000U));

    *rho_p++ = rho1;

    // calculate context for the next quad
    context = static_cast<uint16_t>((rho1 >> 1) | (rho1 & 1));

    vlcval = VLC_dec.advance(static_cast<uint8_t>((tv1 & 0x000F) >> 1));
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
    uint32_t idx        = (vlcval & 0x3F) + (u_off0 << 6U) + (u_off1 << 7U) + mel_offset;
    int32_t uvlc_result = uvlc_dec_0[idx];
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
    u1 = static_cast<uint32_t>(uvlc_result >> 3) + (tmp >> len);

    U0 = kappa0 + u0;
    U1 = kappa1 + u1;

    auto vrho      = _mm256_inserti128_si256(_mm256_set1_epi32(rho0), _mm_set1_epi32(rho1), 1);
    auto vsigma    = _mm256_and_si256(_mm256_srav_epi32(vrho, _mm256_setr_epi32(0, 1, 2, 3, 0, 1, 2, 3)),
                                      _mm256_set1_epi32(1));
    auto vtmp      = _mm256_inserti128_si256(_mm256_set1_epi32(emb_k_0), _mm_set1_epi32(emb_k_1), 1);
    auto vemb_k    = _mm256_and_si256(_mm256_srav_epi32(vtmp, _mm256_setr_epi32(0, 1, 2, 3, 0, 1, 2, 3)),
                                      _mm256_set1_epi32(1));
    auto v_m_quads = _mm256_inserti128_si256(_mm256_set1_epi32(static_cast<int32_t>(U0)),
                                             _mm_set1_epi32(static_cast<int32_t>(U1)), 1);
    v_m_quads      = _mm256_sub_epi32(_mm256_mullo_epi32(vsigma, v_m_quads), vemb_k);

    uint32_t mmqq0 = static_cast<uint32_t>(_mm256_extract_epi32(
        _mm256_hadd_epi32(_mm256_hadd_epi32(v_m_quads, v_m_quads), _mm256_hadd_epi32(v_m_quads, v_m_quads)),
        0));
    uint32_t mmqq1 = static_cast<uint32_t>(_mm256_extract_epi32(
        _mm256_hadd_epi32(_mm256_hadd_epi32(v_m_quads, v_m_quads), _mm256_hadd_epi32(v_m_quads, v_m_quads)),
        4));

    // recoverMagSgnValue
    auto mmm0 = MagSgn.fetch(_mm256_extracti128_si256(v_m_quads, 0));
    MagSgn.advance(mmqq0);
    auto mmm1 = MagSgn.fetch(_mm256_extracti128_si256(v_m_quads, 1));
    MagSgn.advance(mmqq1);

    auto vknown_1 = _mm256_and_si256(
        _mm256_srav_epi32(_mm256_inserti128_si256(_mm256_set1_epi32(emb_1_0), _mm_set1_epi32(emb_1_1), 1),
                          _mm256_setr_epi32(0, 1, 2, 3, 0, 1, 2, 3)),
        _mm256_set1_epi32(1));
    auto vmsval = _mm256_inserti128_si256(_mm256_broadcastsi128_si256(mmm0), mmm1, 1);
    auto vmask = _mm256_sub_epi32(_mm256_sllv_epi32(_mm256_set1_epi32(1), v_m_quads), _mm256_set1_epi32(1));
    auto v_v_quads = _mm256_and_si256(vmsval, vmask);
    v_v_quads      = _mm256_or_si256(v_v_quads, _mm256_sllv_epi32(vknown_1, v_m_quads));
    vmask = _mm256_xor_si256(_mm256_cmpeq_epi32(v_m_quads, _mm256_setzero_si256()), _mm256_set1_epi32(-1));
    auto v_mu = _mm256_add_epi32(_mm256_srai_epi32(v_v_quads, 1), _mm256_set1_epi32(1));
    v_mu      = _mm256_slli_epi32(v_mu, pLSB);
    v_mu = _mm256_or_si256(v_mu, _mm256_slli_epi32(_mm256_and_si256(v_v_quads, _mm256_set1_epi32(1)), 31));
    v_mu = _mm256_and_si256(v_mu, vmask);

    // store mu
    // 0, 2, 4, 6, 1, 3, 5, 7
    v_mu = _mm256_permutevar8x32_epi32(v_mu, _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7));
    _mm256_storeu2_m128i((__m128i *)mp1, (__m128i *)mp0, v_mu);
    mp0 += 4;
    mp1 += 4;

    // store sigma
    *sp0++ = (rho0 >> 0) & 1;
    *sp0++ = (rho0 >> 2) & 1;
    *sp0++ = (rho1 >> 0) & 1;
    *sp0++ = (rho1 >> 2) & 1;
    *sp1++ = (rho0 >> 1) & 1;
    *sp1++ = (rho0 >> 3) & 1;
    *sp1++ = (rho1 >> 1) & 1;
    *sp1++ = (rho1 >> 3) & 1;

    // Update Exponent
    v_v_quads = _mm256_sub_epi32(_mm256_set1_epi32(32), avx2_lzcnt2_epi32(v_v_quads));
    v_v_quads = _mm256_permutevar8x32_epi32(v_v_quads, _mm256_setr_epi32(1, 3, 5, 7, 0, 2, 4, 6));
    vExp      = _mm256_extracti128_si256(v_v_quads, 0);
    _mm256_zeroupper();
    _mm_storeu_si128((__m128i *)E_p, vExp);
    E_p += 4;
  }

  // process the last block (left over)
  if (QW % 2) {
    // Decoding of significance and EMB patterns and unsigned residual offsets
    vlcval       = VLC_dec.fetch();
    uint16_t tv0 = dec_table0[(vlcval & 0x7F) + (static_cast<unsigned int>(context << 7))];
    if (context == 0) {
      mel_run -= 2;
      tv0 = (mel_run == -1) ? tv0 : 0;
      if (mel_run < 0) {
        mel_run = MEL.get_run();
      }
    }
    rho0    = static_cast<uint8_t>((tv0 & 0x00F0) >> 4);
    emb_k_0 = static_cast<uint8_t>((tv0 & 0x0F00) >> 8);
    emb_1_0 = static_cast<uint8_t>((tv0 & 0xF000) >> 12);

    *rho_p++ = rho0;

    // UVLC decoding
    vlcval = VLC_dec.advance(static_cast<uint8_t>((tv0 & 0x000F) >> 1));

    u_off0 = tv0 & 1;

    uint32_t idx        = (vlcval & 0x3F) + (u_off0 << 6U);
    int32_t uvlc_result = uvlc_dec_0[idx];
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

    U0 = kappa0 + u0;

    auto vtmp   = _mm256_inserti128_si256(_mm256_set1_epi32(emb_k_0), _mm_setzero_si128(), 1U);
    auto vemb_k = _mm256_and_si256(_mm256_srav_epi32(vtmp, _mm256_setr_epi32(0, 1, 2, 3, 0, 1, 2, 3)),
                                   _mm256_set1_epi32(1));
    auto vrho   = _mm256_inserti128_si256(_mm256_set1_epi32(rho0), _mm_setzero_si128(), 1);
    auto vsigma = _mm256_and_si256(_mm256_srav_epi32(vrho, _mm256_setr_epi32(0, 1, 2, 3, 0, 1, 2, 3)),
                                   _mm256_set1_epi32(1));
    auto v_m_quads =
        _mm256_inserti128_si256(_mm256_set1_epi32(static_cast<int32_t>(U0)), _mm_setzero_si128(), 1U);
    v_m_quads      = _mm256_sub_epi32(_mm256_mullo_epi32(vsigma, v_m_quads), vemb_k);
    uint32_t mmqq0 = static_cast<uint32_t>(_mm256_extract_epi32(
        _mm256_hadd_epi32(_mm256_hadd_epi32(v_m_quads, v_m_quads), _mm256_hadd_epi32(v_m_quads, v_m_quads)),
        0));

    // recoverMagSgnValue
    auto mmm = MagSgn.fetch(_mm256_extracti128_si256(v_m_quads, 0));
    MagSgn.advance(mmqq0);

    auto vknown_1 = _mm256_and_si256(
        _mm256_srav_epi32(_mm256_inserti128_si256(_mm256_set1_epi32(emb_1_0), _mm_set1_epi32(emb_1_1), 1),
                          _mm256_setr_epi32(0, 1, 2, 3, 0, 1, 2, 3)),
        _mm256_set1_epi32(1));
    auto vmsval = _mm256_inserti128_si256(_mm256_broadcastsi128_si256(mmm), _mm_setzero_si128(), 1);
    auto vmask = _mm256_sub_epi32(_mm256_sllv_epi32(_mm256_set1_epi32(1), v_m_quads), _mm256_set1_epi32(1));
    auto v_v_quads = _mm256_and_si256(vmsval, vmask);
    v_v_quads      = _mm256_or_si256(v_v_quads, _mm256_sllv_epi32(vknown_1, v_m_quads));
    vmask = _mm256_xor_si256(_mm256_cmpeq_epi32(v_m_quads, _mm256_setzero_si256()), _mm256_set1_epi32(-1));
    auto v_mu = _mm256_add_epi32(_mm256_srai_epi32(v_v_quads, 1), _mm256_set1_epi32(1));
    v_mu      = _mm256_slli_epi32(v_mu, pLSB);
    v_mu = _mm256_or_si256(v_mu, _mm256_slli_epi32(_mm256_and_si256(v_v_quads, _mm256_set1_epi32(1)), 31));
    v_mu = _mm256_and_si256(v_mu, vmask);

    // store mu
    *mp0++ = _mm256_extract_epi32(v_mu, 0);
    *mp0++ = _mm256_extract_epi32(v_mu, 2);
    *mp1++ = _mm256_extract_epi32(v_mu, 1);
    *mp1++ = _mm256_extract_epi32(v_mu, 3);

    // store sigma
    *sp0++ = (rho0 >> 0) & 1;
    *sp0++ = (rho0 >> 2) & 1;
    *sp1++ = (rho0 >> 1) & 1;
    *sp1++ = (rho0 >> 3) & 1;

    // Update Exponent
    *E_p++ = static_cast<int32_t>(
        32 - count_leading_zeros(static_cast<uint32_t>(_mm256_extract_epi32(v_v_quads, 1))));
    *E_p++ = static_cast<int32_t>(
        32 - count_leading_zeros(static_cast<uint32_t>(_mm256_extract_epi32(v_v_quads, 3))));
  }  // Initial line-pair end

  /*******************************************************************************************************************/
  // Non-initial line-pair
  /*******************************************************************************************************************/

  for (uint16_t row = 1; row < QH; row++) {
    rho_p      = rholine.get() + 1;
    E_p        = Eline.get() + 1;
    mp0        = block->sample_buf.get() + (row * 2U) * block->blksampl_stride;
    mp1        = block->sample_buf.get() + (row * 2U + 1U) * block->blksampl_stride;
    sp0        = block->block_states.get() + (row * 2U + 1U) * block->blkstate_stride + 1U;
    sp1        = block->block_states.get() + (row * 2U + 2U) * block->blkstate_stride + 1U;
    int32_t qx = 0;
    rho1       = 0;

    // Calculate Emax for the next two quads
    int32_t Emax0, Emax1;
    Emax0 = find_max(E_p[-1], E_p[0], E_p[1], E_p[2]);
    Emax1 = find_max(E_p[1], E_p[2], E_p[3], E_p[4]);

    // calculate context for the next quad
    context = ((rho1 & 0x4) << 6) | ((rho1 & 0x8) << 5);            // (w | sw) << 8
    context |= ((rho_p[-1] & 0x8) << 4) | ((rho_p[0] & 0x2) << 6);  // (nw | n) << 7
    context |= ((rho_p[0] & 0x8) << 6) | ((rho_p[1] & 0x2) << 8);   // (ne | nf) << 9

    for (qx = 0; qx < QW - 1; qx += 2) {
      // Decoding of significance and EMB patterns and unsigned residual offsets
      vlcval       = VLC_dec.fetch();
      uint16_t tv0 = dec_table1[(vlcval & 0x7F) + (static_cast<unsigned int>(context))];
      if (context == 0) {
        mel_run -= 2;
        tv0 = (mel_run == -1) ? tv0 : 0;
        if (mel_run < 0) {
          mel_run = MEL.get_run();
        }
      }
      rho0    = static_cast<uint8_t>(_pext_u32(tv0, 0x00F0U));
      emb_k_0 = static_cast<uint8_t>(_pext_u32(tv0, 0x0F00U));
      emb_1_0 = static_cast<uint8_t>(_pext_u32(tv0, 0xF000U));

      vlcval = VLC_dec.advance(static_cast<uint8_t>((tv0 & 0x000F) >> 1));

      // calculate context for the next quad
      context = ((rho0 & 0x4) << 6) | ((rho0 & 0x8) << 5);           // (w | sw) << 8
      context |= ((rho_p[0] & 0x8) << 4) | ((rho_p[1] & 0x2) << 6);  // (nw | n) << 7
      context |= ((rho_p[1] & 0x8) << 6) | ((rho_p[2] & 0x2) << 8);  // (ne | nf) << 9

      // Decoding of significance and EMB patterns and unsigned residual offsets
      uint16_t tv1 = dec_table1[(vlcval & 0x7F) + (static_cast<unsigned int>(context))];
      if (context == 0) {
        mel_run -= 2;
        tv1 = (mel_run == -1) ? tv1 : 0;
        if (mel_run < 0) {
          mel_run = MEL.get_run();
        }
      }

      rho1    = static_cast<uint8_t>(_pext_u32(tv1, 0x00F0U));
      emb_k_1 = static_cast<uint8_t>(_pext_u32(tv1, 0x0F00U));
      emb_1_1 = static_cast<uint8_t>(_pext_u32(tv1, 0xF000U));

      // calculate context for the next quad
      context = ((rho1 & 0x4) << 6) | ((rho1 & 0x8) << 5);           // (w | sw) << 8
      context |= ((rho_p[1] & 0x8) << 4) | ((rho_p[2] & 0x2) << 6);  // (nw | n) << 7
      context |= ((rho_p[2] & 0x8) << 6) | ((rho_p[3] & 0x2) << 8);  // (ne | nf) << 9

      vlcval = VLC_dec.advance(static_cast<uint8_t>((tv1 & 0x000F) >> 1));

      // UVLC decoding
      u_off0       = tv0 & 1;
      u_off1       = tv1 & 1;
      uint32_t idx = (vlcval & 0x3F) + (u_off0 << 6U) + (u_off1 << 7U);

      int32_t uvlc_result = uvlc_dec_1[idx];
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
      u1 = static_cast<uint32_t>(uvlc_result >> 3) + (tmp >> len);

      gamma0 = (popcount32(static_cast<uint32_t>(rho0)) < 2) ? 0 : 1;
      gamma1 = (popcount32(static_cast<uint32_t>(rho1)) < 2) ? 0 : 1;
      kappa0 = (1 > gamma0 * (Emax0 - 1)) ? 1U : static_cast<uint8_t>(gamma0 * (Emax0 - 1));
      kappa1 = (1 > gamma1 * (Emax1 - 1)) ? 1U : static_cast<uint8_t>(gamma1 * (Emax1 - 1));
      U0     = kappa0 + u0;
      U1     = kappa1 + u1;

      auto vemb_k = _mm256_inserti128_si256(_mm256_set1_epi32(emb_k_0), _mm_set1_epi32(emb_k_1), 1);
      vemb_k      = _mm256_and_si256(_mm256_srav_epi32(vemb_k, _mm256_setr_epi32(0, 1, 2, 3, 0, 1, 2, 3)),
                                     _mm256_set1_epi32(1));
      auto v_m_quads = _mm256_inserti128_si256(_mm256_set1_epi32(static_cast<int32_t>(U0)),
                                               _mm_set1_epi32(static_cast<int32_t>(U1)), 1);
      auto vrho      = _mm256_inserti128_si256(_mm256_set1_epi32(rho0), _mm_set1_epi32(rho1), 1);
      auto vsigma    = _mm256_and_si256(_mm256_srav_epi32(vrho, _mm256_setr_epi32(0, 1, 2, 3, 0, 1, 2, 3)),
                                        _mm256_set1_epi32(1));
      v_m_quads      = _mm256_sub_epi32(_mm256_mullo_epi32(vsigma, v_m_quads), vemb_k);
      uint32_t mmqq0 = static_cast<uint32_t>(
          _mm256_extract_epi32(_mm256_hadd_epi32(_mm256_hadd_epi32(v_m_quads, v_m_quads),
                                                 _mm256_hadd_epi32(v_m_quads, v_m_quads)),
                               0));
      uint32_t mmqq1 = static_cast<uint32_t>(
          _mm256_extract_epi32(_mm256_hadd_epi32(_mm256_hadd_epi32(v_m_quads, v_m_quads),
                                                 _mm256_hadd_epi32(v_m_quads, v_m_quads)),
                               4));

      // recoverMagSgnValue
      auto mmm0 = MagSgn.fetch(_mm256_extracti128_si256(v_m_quads, 0));
      MagSgn.advance(mmqq0);
      auto mmm1 = MagSgn.fetch(_mm256_extracti128_si256(v_m_quads, 1));
      MagSgn.advance(mmqq1);

      auto vknown_1 = _mm256_and_si256(
          _mm256_srav_epi32(_mm256_inserti128_si256(_mm256_set1_epi32(emb_1_0), _mm_set1_epi32(emb_1_1), 1),
                            _mm256_setr_epi32(0, 1, 2, 3, 0, 1, 2, 3)),
          _mm256_set1_epi32(1));
      auto vmsval = _mm256_inserti128_si256(_mm256_broadcastsi128_si256(mmm0), mmm1, 1);
      auto vmask =
          _mm256_sub_epi32(_mm256_sllv_epi32(_mm256_set1_epi32(1), v_m_quads), _mm256_set1_epi32(1));
      auto v_v_quads = _mm256_and_si256(vmsval, vmask);
      v_v_quads      = _mm256_or_si256(v_v_quads, _mm256_sllv_epi32(vknown_1, v_m_quads));
      vmask =
          _mm256_xor_si256(_mm256_cmpeq_epi32(v_m_quads, _mm256_setzero_si256()), _mm256_set1_epi32(-1));
      auto v_mu = _mm256_add_epi32(_mm256_srai_epi32(v_v_quads, 1), _mm256_set1_epi32(1));
      v_mu      = _mm256_slli_epi32(v_mu, pLSB);
      v_mu =
          _mm256_or_si256(v_mu, _mm256_slli_epi32(_mm256_and_si256(v_v_quads, _mm256_set1_epi32(1)), 31));
      v_mu = _mm256_and_si256(v_mu, vmask);

      // store mu
      // 0, 2, 4, 6, 1, 3, 5, 7
      v_mu = _mm256_permutevar8x32_epi32(v_mu, _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7));
      _mm256_storeu2_m128i((__m128i *)mp1, (__m128i *)mp0, v_mu);
      mp0 += 4;
      mp1 += 4;

      // store sigma
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

      // Update Exponent
      Emax0     = find_max(E_p[3], E_p[4], E_p[5], E_p[6]);
      Emax1     = find_max(E_p[5], E_p[6], E_p[7], E_p[8]);
      v_v_quads = _mm256_sub_epi32(_mm256_set1_epi32(32), avx2_lzcnt2_epi32(v_v_quads));
      v_v_quads = _mm256_permutevar8x32_epi32(v_v_quads, _mm256_setr_epi32(1, 3, 5, 7, 0, 2, 4, 6));
      vExp      = _mm256_extracti128_si256(v_v_quads, 0);
      _mm256_zeroupper();
      _mm_storeu_si128((__m128i *)E_p, vExp);
      E_p += 4;
    }

    // process the last block (left over)
    if (QW % 2) {
      // Decoding of significance and EMB patterns and unsigned residual offsets
      vlcval      = VLC_dec.fetch();
      int32_t tv0 = dec_table1[(vlcval & 0x7F) + (static_cast<unsigned int>(context))];
      if (context == 0) {
        mel_run -= 2;
        tv0 = (mel_run == -1) ? tv0 : 0;
        if (mel_run < 0) {
          mel_run = MEL.get_run();
        }
      }
      rho0    = static_cast<uint8_t>((tv0 & 0x00F0) >> 4);
      emb_k_0 = static_cast<uint8_t>((tv0 & 0x0F00) >> 8);
      emb_1_0 = static_cast<uint8_t>((tv0 & 0xF000) >> 12);

      // UVLC decoding
      vlcval = VLC_dec.advance(static_cast<uint8_t>((tv0 & 0x000F) >> 1));
      u_off0 = tv0 & 1;

      uint32_t idx        = (vlcval & 0x3F) + (u_off0 << 6U);
      int32_t uvlc_result = uvlc_dec_0[idx];
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

      gamma0 = (popcount32(static_cast<uint32_t>(rho0)) < 2) ? 0 : 1;
      kappa0 = (1 > gamma0 * (Emax0 - 1)) ? 1U : static_cast<uint8_t>(gamma0 * (Emax0 - 1));
      U0     = kappa0 + u0;

      auto vemb_k = _mm256_inserti128_si256(_mm256_set1_epi32(emb_k_0), _mm_setzero_si128(), 1);
      vemb_k      = _mm256_and_si256(_mm256_srav_epi32(vemb_k, _mm256_setr_epi32(0, 1, 2, 3, 0, 1, 2, 3)),
                                     _mm256_set1_epi32(1));
      auto vrho   = _mm256_inserti128_si256(_mm256_set1_epi32(rho0), _mm_setzero_si128(), 1);
      auto vsigma = _mm256_and_si256(_mm256_srav_epi32(vrho, _mm256_setr_epi32(0, 1, 2, 3, 0, 1, 2, 3)),
                                     _mm256_set1_epi32(1));

      auto v_m_quads =
          _mm256_inserti128_si256(_mm256_set1_epi32(static_cast<int32_t>(U0)), _mm_setzero_si128(), 1);
      v_m_quads      = _mm256_sub_epi32(_mm256_mullo_epi32(vsigma, v_m_quads), vemb_k);
      uint32_t mmqq0 = static_cast<uint32_t>(
          _mm256_extract_epi32(_mm256_hadd_epi32(_mm256_hadd_epi32(v_m_quads, v_m_quads),
                                                 _mm256_hadd_epi32(v_m_quads, v_m_quads)),
                               0));

      // recoverMagSgnValue
      auto mmm = MagSgn.fetch(_mm256_extracti128_si256(v_m_quads, 0));
      MagSgn.advance(mmqq0);

      auto vknown_1 = _mm256_and_si256(
          _mm256_srav_epi32(_mm256_inserti128_si256(_mm256_set1_epi32(emb_1_0), _mm_set1_epi32(emb_1_1), 1),
                            _mm256_setr_epi32(0, 1, 2, 3, 0, 1, 2, 3)),
          _mm256_set1_epi32(1));
      auto vmsval = _mm256_inserti128_si256(_mm256_broadcastsi128_si256(mmm), _mm_setzero_si128(), 1);
      auto vmask =
          _mm256_sub_epi32(_mm256_sllv_epi32(_mm256_set1_epi32(1), v_m_quads), _mm256_set1_epi32(1));
      auto v_v_quads = _mm256_and_si256(vmsval, vmask);
      v_v_quads      = _mm256_or_si256(v_v_quads, _mm256_sllv_epi32(vknown_1, v_m_quads));
      vmask =
          _mm256_xor_si256(_mm256_cmpeq_epi32(v_m_quads, _mm256_setzero_si256()), _mm256_set1_epi32(-1));
      auto v_mu = _mm256_add_epi32(_mm256_srai_epi32(v_v_quads, 1), _mm256_set1_epi32(1));
      v_mu      = _mm256_slli_epi32(v_mu, pLSB);
      v_mu =
          _mm256_or_si256(v_mu, _mm256_slli_epi32(_mm256_and_si256(v_v_quads, _mm256_set1_epi32(1)), 31));
      v_mu = _mm256_and_si256(v_mu, vmask);

      // store mu
      *mp0++ = _mm256_extract_epi32(v_mu, 0);
      *mp0++ = _mm256_extract_epi32(v_mu, 2);
      *mp1++ = _mm256_extract_epi32(v_mu, 1);
      *mp1++ = _mm256_extract_epi32(v_mu, 3);

      // store sigma
      *sp0++ = (rho0 >> 0) & 1;
      *sp0++ = (rho0 >> 2) & 1;
      *sp1++ = (rho0 >> 1) & 1;
      *sp1++ = (rho0 >> 3) & 1;

      // Update Exponent
      E_p[0] = static_cast<int32_t>(
          32 - count_leading_zeros(static_cast<uint32_t>(_mm256_extract_epi32(v_v_quads, 1))));
      E_p[1] = static_cast<int32_t>(
          32 - count_leading_zeros(static_cast<uint32_t>(_mm256_extract_epi32(v_v_quads, 3))));
      *rho_p++ = rho0;
      E_p += 2;
    }
  }  // Non-Initial line-pair end
}

auto process_stripes_block_dec = [](SP_dec &SigProp, j2k_codeblock *block, const int32_t i_start,
                                    const int32_t j_start, const uint16_t width, const uint16_t height,
                                    const uint8_t &pLSB) {
  int32_t *sp;
  uint8_t causal_cond = 0;
  uint8_t bit;
  uint8_t mbr;
  const auto block_width  = static_cast<uint16_t>(j_start + width);
  const auto block_height = static_cast<uint16_t>(i_start + height);

  for (int16_t j = (int16_t)j_start; j < block_width; j++) {
    for (int16_t i = (int16_t)i_start; i < block_height; i++) {
      sp = &block->sample_buf[static_cast<size_t>(j) + static_cast<size_t>(i) * block->blksampl_stride];
      causal_cond = (((block->Cmodes & CAUSAL) == 0) || (i != block_height - 1));
      mbr         = 0;
      if (block->get_state(Sigma, i, j) == 0) {
        mbr = block->calc_mbr(i, j, causal_cond);
      }
      if (mbr != 0) {
        block->modify_state(refinement_indicator, 1, i, j);
        bit = SigProp.importSigPropBit();
        block->modify_state(refinement_value, bit, i, j);
        *sp |= bit << pLSB;
      }
      block->modify_state(scan, 1, i, j);
    }
  }
  for (int16_t j = (int16_t)j_start; j < block_width; j++) {
    for (int16_t i = (int16_t)i_start; i < block_height; i++) {
      sp = &block->sample_buf[static_cast<size_t>(j) + static_cast<size_t>(i) * block->blksampl_stride];
      // decode sign
      if ((*sp & (1 << pLSB)) != 0) {
        *sp = (*sp & 0x7FFFFFFF) | (SigProp.importSigPropBit() << 31);
      }
    }
  }
};

void ht_sigprop_decode(j2k_codeblock *block, uint8_t *HT_magref_segment, uint32_t magref_length,
                       const uint8_t &pLSB) {
  SP_dec SigProp(HT_magref_segment, magref_length);
  const uint16_t num_v_stripe = static_cast<uint16_t>(block->size.y / 4);
  const uint16_t num_h_stripe = static_cast<uint16_t>(block->size.x / 4);
  int32_t i_start             = 0, j_start;
  uint16_t width              = 4;
  uint16_t width_last;
  uint16_t height = 4;

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
  MR_dec MagRef(HT_magref_segment, magref_length);
  const uint16_t blk_height   = static_cast<uint16_t>(block->size.y);
  const uint16_t blk_width    = static_cast<uint16_t>(block->size.x);
  const uint16_t num_v_stripe = static_cast<uint16_t>(block->size.y / 4);
  int16_t i_start             = 0;
  int16_t height              = 4;
  int32_t *sp;

  for (int16_t n1 = 0; n1 < num_v_stripe; n1++) {
    for (int16_t j = 0; j < blk_width; j++) {
      for (int16_t i = i_start; i < i_start + height; i++) {
        sp = &block->sample_buf[static_cast<size_t>(j) + static_cast<size_t>(i) * block->blksampl_stride];
        if (block->get_state(Sigma, i, j) != 0) {
          block->modify_state(refinement_indicator, 1, i, j);
          sp[0] |= MagRef.importMagRefBit() << pLSB;
        }
      }
    }
    i_start = static_cast<int16_t>(i_start + 4);
  }
  height = static_cast<int16_t>(blk_height % 4);
  for (int16_t j = 0; j < blk_width; j++) {
    for (int16_t i = i_start; i < i_start + height; i++) {
      sp = &block->sample_buf[static_cast<size_t>(j) + static_cast<size_t>(i) * block->blksampl_stride];
      if (block->get_state(Sigma, i, j) != 0) {
        block->modify_state(refinement_indicator, 1, i, j);
        sp[0] |= MagRef.importMagRefBit() << pLSB;
      }
    }
  }
}

void j2k_codeblock::dequantize(uint8_t S_blk, uint8_t ROIshift) const {
  /* ready for ROI adjustment and dequantization */

  // number of decoded magnitude bit‐planes
  const int32_t pLSB = 31 - M_b;  // indicates binary point;

  // bit mask for ROI detection
  const uint32_t mask = UINT32_MAX >> (M_b + 1);
  // reconstruction parameter defined in E.1.1.2 of the spec

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
  if (this->transformation) {
    // lossless path
    for (size_t i = 0; i < static_cast<size_t>(this->size.y); i++) {
      int32_t *val      = this->sample_buf.get() + i * this->blksampl_stride;
      sprec_t *dst      = this->i_samples + i * this->band_stride;
      uint8_t *blkstate = this->block_states.get() + (i + 1) * this->blkstate_stride + 1;
      size_t simdlen    = static_cast<size_t>(this->size.x) - static_cast<size_t>(this->size.x) % 16;
      for (size_t j = 0; j < simdlen; j += 16) {
        auto vsrc0 = _mm256_loadu_si256((__m256i *)val);
        auto vsrc1 = _mm256_loadu_si256((__m256i *)(val + 8));
        auto vsign0 =
            _mm256_or_si256(_mm256_and_si256(vsrc0, _mm256_set1_epi32(INT32_MIN)), _mm256_set1_epi32(1));
        auto vsign1 =
            _mm256_or_si256(_mm256_and_si256(vsrc1, _mm256_set1_epi32(INT32_MIN)), _mm256_set1_epi32(1));
        vsrc0 = _mm256_and_si256(vsrc0, _mm256_set1_epi32(0x7FFFFFFF));
        vsrc1 = _mm256_and_si256(vsrc1, _mm256_set1_epi32(0x7FFFFFFF));
        // upshift background region, if necessary
        auto vROImask = _mm256_and_si256(vsrc0, _mm256_set1_epi32(static_cast<int32_t>(~mask)));
        vROImask      = _mm256_cmpeq_epi32(vROImask, _mm256_setzero_si256());
        vROImask      = _mm256_and_si256(vROImask, _mm256_set1_epi32(ROIshift));
        vsrc0         = _mm256_sllv_epi32(vsrc0, vROImask);
        vROImask      = _mm256_and_si256(vsrc1, _mm256_set1_epi32(static_cast<int32_t>(~mask)));
        vROImask      = _mm256_cmpeq_epi32(vROImask, _mm256_setzero_si256());
        vROImask      = _mm256_and_si256(vROImask, _mm256_set1_epi32(ROIshift));
        vsrc1         = _mm256_sllv_epi32(vsrc1, vROImask);

        // retrieve number of decoded magnitude bit-planes
        auto vstate      = _mm_loadu_si128((__m128i *)blkstate);
        auto vstate_low  = _mm256_cvtepi8_epi32(vstate);
        auto vstate_high = _mm256_cvtepi8_epi32(_mm_srli_si128(vstate, 8));
        vstate_low       = _mm256_and_si256(_mm256_srai_epi32(vstate_low, 2), _mm256_set1_epi32(1));
        vstate_high      = _mm256_and_si256(_mm256_srai_epi32(vstate_high, 2), _mm256_set1_epi32(1));
        auto vNb0        = _mm256_add_epi32(_mm256_set1_epi32(S_blk + 1), vstate_low);
        auto vNb1        = _mm256_add_epi32(_mm256_set1_epi32(S_blk + 1), vstate_high);

        // add reconstruction value, if necessary (it will happen for a truncated codestream)
        auto vMb           = _mm256_set1_epi32(M_b);
        auto v_recval_mask = _mm256_cmpgt_epi32(vMb, vNb0);
        v_recval_mask = _mm256_and_si256(v_recval_mask, _mm256_cmpgt_epi32(vsrc0, _mm256_setzero_si256()));
        auto vrecval0 = _mm256_and_si256(
            _mm256_sllv_epi32(_mm256_set1_epi32(1), _mm256_sub_epi32(_mm256_set1_epi32(30), vNb0)),
            v_recval_mask);
        v_recval_mask = _mm256_cmpgt_epi32(vMb, vNb1);
        v_recval_mask = _mm256_and_si256(v_recval_mask, _mm256_cmpgt_epi32(vsrc1, _mm256_setzero_si256()));
        auto vrecval1 = _mm256_and_si256(
            _mm256_sllv_epi32(_mm256_set1_epi32(1), _mm256_sub_epi32(_mm256_set1_epi32(30), vNb1)),
            v_recval_mask);
        vsrc0 = _mm256_or_si256(vsrc0, vrecval0);
        vsrc1 = _mm256_or_si256(vsrc1, vrecval1);

        // convert values from sign-magnitude form to two's complement one
        auto vdst0 = _mm256_srai_epi32(_mm256_sign_epi32(vsrc0, vsign0), pLSB);
        auto vdst1 = _mm256_srai_epi32(_mm256_sign_epi32(vsrc1, vsign1), pLSB);
        _mm256_storeu_si256((__m256i *)dst,
                            _mm256_permute4x64_epi64(_mm256_packs_epi32(vdst0, vdst1), 0xD8));
        val += 16;
        dst += 16;
        blkstate += 16;
      }
      for (size_t j = static_cast<size_t>(this->size.x) - static_cast<size_t>(this->size.x) % 16;
           j < static_cast<size_t>(this->size.x); j++) {
        int32_t sign = *val & INT32_MIN;
        *val &= INT32_MAX;
        // detect background region and upshift it
        if (ROIshift && (((uint32_t)*val & ~mask) == 0)) {
          *val <<= ROIshift;
        }
        // do adjustment of the position indicating 0.5
        int32_t N_b = S_blk + 1 + ((*blkstate >> 2) & 1);
        if (ROIshift) {
          N_b = M_b;
        }
        if (N_b < M_b && *val) {
          *val |= 1 << (31 - N_b - 1);
        }
        // bring sign back
        *val |= sign;
        // convert sign-magnitude to two's complement form
        if (*val < 0) {
          *val = -(*val & INT32_MAX);
        }

        assert(pLSB >= 0);  // assure downshift is not negative
        *dst = static_cast<int16_t>(*val >> pLSB);
        val++;
        dst++;
        blkstate++;
      }
    }
  } else {
    // lossy path
    [[maybe_unused]] int32_t ROImask = 0;
    if (ROIshift) {
      ROImask = static_cast<int32_t>(0xFFFFFFFF);
    }
    //    auto vROIshift = vdupq_n_s32(ROImask);
    for (size_t i = 0; i < static_cast<size_t>(this->size.y); i++) {
      int32_t *val      = this->sample_buf.get() + i * this->blksampl_stride;
      sprec_t *dst      = this->i_samples + i * this->band_stride;
      uint8_t *blkstate = this->block_states.get() + (i + 1) * this->blkstate_stride + 1;
      size_t len        = this->size.x;
      for (; len >= 16; len -= 16) {
        auto vsrc0 = _mm256_loadu_si256((__m256i *)val);
        auto vsrc1 = _mm256_loadu_si256((__m256i *)(val + 8));
        auto vsign0 =
            _mm256_or_si256(_mm256_and_si256(vsrc0, _mm256_set1_epi32(INT32_MIN)), _mm256_set1_epi32(1));
        auto vsign1 =
            _mm256_or_si256(_mm256_and_si256(vsrc1, _mm256_set1_epi32(INT32_MIN)), _mm256_set1_epi32(1));
        vsrc0 = _mm256_and_si256(vsrc0, _mm256_set1_epi32(0x7FFFFFFF));
        vsrc1 = _mm256_and_si256(vsrc1, _mm256_set1_epi32(0x7FFFFFFF));
        // upshift background region, if necessary
        auto vROImask = _mm256_and_si256(vsrc0, _mm256_set1_epi32(static_cast<int32_t>(~mask)));
        vROImask      = _mm256_cmpeq_epi32(vROImask, _mm256_setzero_si256());
        vROImask      = _mm256_and_si256(vROImask, _mm256_set1_epi32(ROIshift));
        vsrc0         = _mm256_sllv_epi32(vsrc0, vROImask);
        vROImask      = _mm256_and_si256(vsrc1, _mm256_set1_epi32(static_cast<int32_t>(~mask)));
        vROImask      = _mm256_cmpeq_epi32(vROImask, _mm256_setzero_si256());
        vROImask      = _mm256_and_si256(vROImask, _mm256_set1_epi32(ROIshift));
        vsrc1         = _mm256_sllv_epi32(vsrc1, vROImask);

        // retrieve number of decoded magnitude bit-planes
        auto vstate      = _mm_loadu_si128((__m128i *)blkstate);
        auto vstate_low  = _mm256_cvtepi8_epi32(vstate);
        auto vstate_high = _mm256_cvtepi8_epi32(_mm_srli_si128(vstate, 8));
        vstate_low       = _mm256_and_si256(_mm256_srai_epi32(vstate_low, 2), _mm256_set1_epi32(1));
        vstate_high      = _mm256_and_si256(_mm256_srai_epi32(vstate_high, 2), _mm256_set1_epi32(1));
        auto vNb0        = _mm256_add_epi32(_mm256_set1_epi32(S_blk + 1), vstate_low);
        auto vNb1        = _mm256_add_epi32(_mm256_set1_epi32(S_blk + 1), vstate_high);
        if (ROIshift) {
          vNb0 = _mm256_set1_epi32(M_b);
          vNb1 = _mm256_set1_epi32(M_b);
        }

        // add reconstruction value, if necessary (it will happen for a truncated codestream)
        auto v_recval_mask = _mm256_cmpgt_epi32(vsrc0, _mm256_setzero_si256());
        auto vrecval0      = _mm256_and_si256(
                 _mm256_sllv_epi32(_mm256_set1_epi32(1), _mm256_sub_epi32(_mm256_set1_epi32(30), vNb0)),
                 v_recval_mask);
        v_recval_mask = _mm256_cmpgt_epi32(vsrc1, _mm256_setzero_si256());
        auto vrecval1 = _mm256_and_si256(
            _mm256_sllv_epi32(_mm256_set1_epi32(1), _mm256_sub_epi32(_mm256_set1_epi32(30), vNb1)),
            v_recval_mask);
        vsrc0 = _mm256_or_si256(vsrc0, vrecval0);
        vsrc1 = _mm256_or_si256(vsrc1, vrecval1);

        // to prevent overflow, truncate to int16_t range
        vsrc0 = _mm256_srai_epi32(_mm256_add_epi32(vsrc0, _mm256_set1_epi32(1 << 15)), 16);
        vsrc1 = _mm256_srai_epi32(_mm256_add_epi32(vsrc1, _mm256_set1_epi32(1 << 15)), 16);

        // dequantization
        vsrc0 = _mm256_mullo_epi32(vsrc0, _mm256_set1_epi32(scale));
        vsrc1 = _mm256_mullo_epi32(vsrc1, _mm256_set1_epi32(scale));

        // downshift and convert values from sign-magnitude form to two's complement one
        vsrc0 =
            _mm256_srai_epi32(_mm256_add_epi32(vsrc0, _mm256_set1_epi32(1 << (downshift - 1))), downshift);
        vsrc1 =
            _mm256_srai_epi32(_mm256_add_epi32(vsrc1, _mm256_set1_epi32(1 << (downshift - 1))), downshift);

        vsrc0 = _mm256_sign_epi32(vsrc0, vsign0);
        vsrc1 = _mm256_sign_epi32(vsrc1, vsign1);

        _mm256_storeu_si256((__m256i *)dst,
                            _mm256_permute4x64_epi64(_mm256_packs_epi32(vsrc0, vsrc1), 0xD8));

        val += 16;
        dst += 16;
        blkstate += 16;
      }
      for (; len > 0; --len) {
        int32_t sign = *val & INT32_MIN;
        *val &= INT32_MAX;
        // detect background region and upshift it
        if (ROIshift && (((uint32_t)*val & ~mask) == 0)) {
          *val <<= ROIshift;
        }
        // do adjustment of the position indicating 0.5
        int32_t N_b = S_blk + 1 + ((*blkstate >> 2) & 1);
        if (ROIshift) {
          N_b = M_b;
        }
        if (*val) {
          *val |= 1 << (31 - N_b - 1);
        }

        // to prevent overflow, truncate to int16_t
        *val = (*val + (1 << 15)) >> 16;
        //  dequantization
        *val *= scale;
        // downshift
        *dst = (int16_t)((*val + (1 << (downshift - 1))) >> downshift);
        // convert sign-magnitude to two's complement form
        if (sign) {
          *dst = static_cast<int16_t>(-(*dst));
        }
        val++;
        dst++;
        blkstate++;
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
    for (uint32_t i = 0; i < block->pass_length.size(); i++) {
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
    std::vector<uint8_t> all_segments;
    all_segments.reserve(3);
    for (uint32_t i = 0; i < block->pass_length.size(); i++) {
      if (block->pass_length[i] != 0) {
        all_segments.push_back(static_cast<uint8_t>(i));
      }
    }
    Lcup += static_cast<int32_t>(block->pass_length[all_segments[0]]);
    if (Lcup < 2) {
      printf("WARNING: Cleanup pass length must be at least 2 bytes in length.\n");
      return false;
    }
    for (uint32_t i = 1; i < all_segments.size(); i++) {
      Lref += block->pass_length[all_segments[i]];
    }
    Dcup = block->get_compressed_data();

    if (block->num_passes > 1 && all_segments.size() > 1) {
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
    //    state_MS_dec MS     = state_MS_dec(Dcup, Pcup);
    fwd_buf<0xFF> MagSgn(Dcup, Pcup);
    //    state_MEL_unPacker MEL_unPacker = state_MEL_unPacker(Dcup, Lcup, Pcup);
    //    state_MEL_decoder MEL_decoder   = state_MEL_decoder(MEL_unPacker);
    //    state_VLC_dec VLC               = state_VLC_dec(Dcup, Lcup, Pcup);
    MEL_dec MEL(Dcup, Lcup, Scup);
    rev_buf VLCdec(Dcup, Lcup, Scup);
    ht_cleanup_decode(block, static_cast<uint8_t>(30 - S_blk), MagSgn, MEL, VLCdec);
    if (num_ht_passes > 1) {
      ht_sigprop_decode(block, Dref, Lref, static_cast<uint8_t>(30 - (S_blk + 1)));
    }
    if (num_ht_passes > 2) {
      ht_magref_decode(block, Dref, Lref, static_cast<uint8_t>(30 - (S_blk + 1)));
    }

    // dequantization
    block->dequantize(S_blk, ROIshift);

  }  // end

  return true;
}
#endif