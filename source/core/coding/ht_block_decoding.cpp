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

#if !defined(OPENHTJ2K_ENABLE_ARM_NEON) && (!defined(__AVX2__) || !defined(OPENHTJ2K_TRY_AVX2)) \
    && !defined(OPENHTJ2K_ENABLE_WASM_SIMD)
  #include "coding_units.hpp"
  #include "dec_CxtVLC_tables.hpp"
  #include "ht_block_decoding.hpp"
  #include "block_decoding.hpp"
  #include "coding_local.hpp"
  #include "utils.hpp"

  #define Q0 0
  #define Q1 1

uint8_t j2k_codeblock::calc_mbr(const uint32_t i, const uint32_t j, const uint8_t causal_cond) const {
  uint8_t *state_p0 = block_states + static_cast<size_t>(i) * blkstate_stride + j;
  uint8_t *state_p1 = block_states + static_cast<size_t>(i + 1) * blkstate_stride + j;
  uint8_t *state_p2 = block_states + static_cast<size_t>(i + 2) * blkstate_stride + j;

  uint8_t mbr0 = state_p0[0] | state_p0[1] | state_p0[2];
  uint8_t mbr1 = state_p1[0] | state_p1[2];
  uint8_t mbr2 = state_p2[0] | state_p2[1] | state_p2[2];
  uint8_t mbr  = mbr0 | mbr1 | (mbr2 & causal_cond);
  mbr |= (mbr0 >> SHIFT_REF) & (mbr0 >> SHIFT_SCAN);
  mbr |= (mbr1 >> SHIFT_REF) & (mbr1 >> SHIFT_SCAN);
  mbr |= (mbr2 >> SHIFT_REF) & (mbr2 >> SHIFT_SCAN) & causal_cond;
  return mbr & 1;
}

// Scalar fused dequantize-and-store: convert sign-magnitude int32 to dequantized float.
// Branchless in the sign, which is as good as random: the negation is (x ^ s) - s for an
// integer and a sign-bit flip for a float (the same bits as -f, also for f == 0).
template <bool StoreI32 = false>
static FORCE_INLINE void dequant_store_scalar(void *dst, int32_t val, uint8_t transformation,
                                              int32_t pLSB_dq, float fscale_direct) {
  if (transformation == 1) {
    const int32_t s   = val >> 31;  // 0 or -1
    const int32_t mag = (val & INT32_MAX) >> pLSB_dq;
    const int32_t v   = (mag ^ s) - s;
    OPENHTJ2K_IF_CONSTEXPR (StoreI32)
      memcpy(dst, &v, sizeof(v));
    else {
      const float f = static_cast<float>(v);
      memcpy(dst, &f, sizeof(f));
    }
  } else {
    const float f = static_cast<float>(val & INT32_MAX) * fscale_direct;
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    bits |= static_cast<uint32_t>(val) & 0x80000000U;
    memcpy(dst, &bits, sizeof(bits));
  }
}

// MagSgn decoding of sample i (0..3) of a quad, branchless: an insignificant sample
// consumes no bits and decodes to 0, and so does its v_n (for E of the next line-pair).
static FORCE_INLINE uint32_t ms_decode_sample(fwd_buf<0xFF> &MagSgn, const uint32_t inf, const uint32_t U_q,
                                              const uint32_t i, const uint8_t pLSB, uint32_t &v_n) {
  const uint32_t sig    = 0U - ((inf >> (4 + i)) & 1U);            // all ones when significant
  const uint32_t m_n    = (U_q - ((inf >> (12 + i)) & 1U)) & sig;  // remove e_k
  const uint32_t ms_val = MagSgn.fetch();                          // 32 bits of MagSgn data
  MagSgn.advance(m_n);                                             // consume m_n
  uint32_t v = ms_val & ((1U << m_n) - 1U);                        // keep only m_n bits
  v |= ((inf >> (8 + i)) & 1U) << m_n;                             // add EMB e_1 as MSB
  v |= 1;                                                          // add center of bin
  v_n = v & sig;
  // v is 2 * (mu - 1) + 0.5 with the sign bit in its LSB; add 2 to make it 2 * mu + 0.5,
  // shift it up to the missing MSBs and put the sign at bit 31
  return ((ms_val << 31) | ((v + 2) << (pLSB - 1))) & sig;
}

  #if defined(OPENHTJ2K_ENABLE_SSE2)
// The four ms_decode_sample calls of a quad in SSE2 lanes (x86-64 builds without AVX2):
// significance, m_n and the bit offsets in vectors, the four 32-bit windows read with
// scalar loads (SSE2 has no byte shuffle to gather them from one fetch), then the value
// assembly in vectors.  Lane i is sample i.  Same arithmetic as ms_decode_sample, so the
// output is bit-identical.
static FORCE_INLINE __m128i ms_decode_quad_sse2(fwd_buf<0xFF> &MagSgn, const uint32_t inf,
                                                const uint32_t U_q, const __m128i shift_pLSB,
                                                __m128i &v_n) {
  const __m128i one = _mm_set1_epi32(1);
  // lane i: inf bits 4+i (rho), 8+i (e_1), 12+i (e_k), moved to bits 7, 11, 15
  const __m128i flags = _mm_mullo_epi16(_mm_and_si128(_mm_set1_epi32(static_cast<int32_t>(inf)),
                                                      _mm_set_epi32(0x8880, 0x4440, 0x2220, 0x1110)),
                                        _mm_set_epi16(0, 1, 0, 2, 0, 4, 0, 8));
  const __m128i insig = _mm_cmpeq_epi32(_mm_and_si128(flags, _mm_set1_epi32(0x80)), _mm_setzero_si128());
  const __m128i emb_k = _mm_srli_epi32(flags, 15);
  const __m128i emb_1 = _mm_and_si128(_mm_srli_epi32(flags, 11), one);
  const __m128i m_n =
      _mm_andnot_si128(insig, _mm_sub_epi32(_mm_set1_epi32(static_cast<int32_t>(U_q)), emb_k));
  // bit offset of each sample from the current position: exclusive prefix sum of m_n
  __m128i inc = _mm_add_epi32(m_n, _mm_slli_si128(m_n, 4));
  inc         = _mm_add_epi32(inc, _mm_slli_si128(inc, 8));
  alignas(16) uint32_t ex[4];
  _mm_store_si128(reinterpret_cast<__m128i *>(ex), _mm_slli_si128(inc, 4));
  const uint32_t pos   = MagSgn.position();
  const uint32_t total = static_cast<uint32_t>(_mm_cvtsi128_si32(_mm_shuffle_epi32(inc, 0xFF)));
  uint32_t ms0, ms1, ms2, ms3;
  if (total <= 56) {
    // one fetch holds all four: sample i needs bits ex[i] .. ex[i] + max(m_n, 1) - 1
    const uint64_t win = MagSgn.fetch64_at(pos);
    ms0                = static_cast<uint32_t>(win);
    ms1                = static_cast<uint32_t>(win >> ex[1]);
    ms2                = static_cast<uint32_t>(win >> ex[2]);
    ms3                = static_cast<uint32_t>(win >> ex[3]);
  } else {
    ms0 = MagSgn.fetch_at(pos);
    ms1 = MagSgn.fetch_at(pos + ex[1]);
    ms2 = MagSgn.fetch_at(pos + ex[2]);
    ms3 = MagSgn.fetch_at(pos + ex[3]);
  }
  const __m128i ms = _mm_unpacklo_epi64(_mm_unpacklo_epi32(_mm_cvtsi32_si128(static_cast<int32_t>(ms0)),
                                                           _mm_cvtsi32_si128(static_cast<int32_t>(ms1))),
                                        _mm_unpacklo_epi32(_mm_cvtsi32_si128(static_cast<int32_t>(ms2)),
                                                           _mm_cvtsi32_si128(static_cast<int32_t>(ms3))));
  MagSgn.advance(total);
  // 2^m_n through the float exponent (the float 2^31 converts to 0x80000000 = 1 << 31)
  const __m128i pow2 =
      _mm_cvttps_epi32(_mm_castsi128_ps(_mm_add_epi32(_mm_slli_epi32(m_n, 23), _mm_set1_epi32(127 << 23))));
  __m128i v = _mm_and_si128(ms, _mm_sub_epi32(pow2, one));                                // keep m_n bits
  v   = _mm_or_si128(v, _mm_and_si128(pow2, _mm_sub_epi32(_mm_setzero_si128(), emb_1)));  // e_1 as MSB
  v   = _mm_or_si128(v, one);                                                             // center of bin
  v_n = _mm_andnot_si128(insig, v);
  const __m128i val =
      _mm_or_si128(_mm_slli_epi32(ms, 31), _mm_sll_epi32(_mm_add_epi32(v, _mm_set1_epi32(2)), shift_pLSB));
  return _mm_andnot_si128(insig, val);
}
  #endif

// Step-2 of the HT cleanup pass: MagSgn decoding over the (tv, u) scratch
// written by ht_cleanup_step1_nway (the former phase 1 of this function).
// Kept per-block: the fwd_buf destuff scratch is thread-local (constructing a
// second fwd_buf on the same thread invalidates the first), and step-2 is
// throughput-bound — there is nothing to gain from interleaving it.
// Significance is data-dependent, so a quad decodes its samples without branching
// on it (ms_decode_sample); only a quad with no significant sample is skipped.
template <bool fuse_dequant = false, bool store_i32 = false>
static void ht_cleanup_step2(j2k_codeblock *block, const uint8_t pLSB, const int32_t Pcup,
                             uint16_t *scratch, const int32_t sstr) {
  uint8_t *compressed_data = block->get_compressed_data();
  // Fused dequantize setup
  int32_t pLSB_dq     = 0;
  float fscale_direct = 0.0f;
  uint32_t out_stride = block->blksampl_stride;
  OPENHTJ2K_IF_CONSTEXPR (fuse_dequant) {
    const int32_t M_b_val = block->get_Mb();
    pLSB_dq               = 31 - M_b_val;
    out_stride            = block->band_stride;
    if (block->transformation != 1) {
      // lossy path (transformation==0 for irrev97, transformation>=2 for ATK irrev)
      fscale_direct = block->stepsize;
      fscale_direct *= static_cast<float>(1 << FRACBITS);
      if (M_b_val <= 31)
        fscale_direct /= static_cast<float>(1 << (31 - M_b_val));
      else
        fscale_direct *= static_cast<float>(1 << (M_b_val - 31));
    }
  }
  const uint8_t transformation = block->transformation;
  auto store_sample            = [&](int32_t *dst, uint32_t ival) {
    OPENHTJ2K_IF_CONSTEXPR (fuse_dequant) {
      dequant_store_scalar<store_i32>(dst, static_cast<int32_t>(ival), transformation, pLSB_dq,
                                                 fscale_direct);
    } else {
      *dst = static_cast<int32_t>(ival);
    }
  };

  // v_n of the row-1 samples of the previous line-pair, one entry per quad:
  // v_n_scratch[q] = v_n(column 2q) | v_n(column 2q - 1).  512 quads at most, plus one
  // for vp[1] at the last quad; 4 extra keep the size a multiple of 16 bytes.
  const int v_n_size             = 512 + 4;
  uint32_t v_n_scratch[v_n_size] = {0};  // 2+ kB

  fwd_buf<0xFF> MagSgn(compressed_data, Pcup);
  #if defined(OPENHTJ2K_ENABLE_SSE2)
  const __m128i shift_pLSB = _mm_cvtsi32_si128(pLSB - 1);
  const __m128i shift_dq   = _mm_cvtsi32_si128(pLSB_dq);
  const __m128i vmag       = _mm_set1_epi32(INT32_MAX);
  const __m128 vfscale     = _mm_set1_ps(fscale_direct);
  #endif

  const uint32_t width = block->size.x;
  for (uint32_t y = 0; y < block->size.y; y += 2) {
    const uint16_t *sp = scratch + (y >> 1) * static_cast<uint32_t>(sstr);
    uint32_t *vp       = v_n_scratch;
    int32_t *dp       = fuse_dequant ? reinterpret_cast<int32_t *>(block->band_buf) + y * block->band_stride
                                     : block->sample_buf + y * block->blksampl_stride;
    uint32_t prev_v_n = 0;
    for (uint32_t x = 0; x < width; x += 2, sp += 2, ++vp, dp += 2) {
      const uint32_t inf = sp[0];
      uint32_t U_q       = sp[1];
      if (y > 0) {
        uint32_t gamma = inf & 0xF0;
        gamma &= gamma - 0x10;  // is gamma_q 1?
        uint32_t emax = vp[0] | vp[1];
        emax          = 31 - count_leading_zeros(emax | 2);  // emax - 1
        U_q += gamma ? emax : 1;                             // kappa
      }
      if (U_q > ((30U - pLSB) + 2U)) {
        printf("ERROR\n");
      }
      const bool two_columns = x + 1 < width;
      uint32_t v1 = 0, v3 = 0;
  #if defined(OPENHTJ2K_ENABLE_SSE2)
      if (inf & 0xF0) {
        __m128i vn;
        // a one-column quad decodes only samples 0 and 1
        const __m128i val =
            ms_decode_quad_sse2(MagSgn, two_columns ? inf : (inf & 0x3333U), U_q, shift_pLSB, vn);
        __m128i res = val;
        OPENHTJ2K_IF_CONSTEXPR (fuse_dequant) {
          if (transformation == 1) {
            const __m128i s   = _mm_srai_epi32(val, 31);
            const __m128i mag = _mm_srl_epi32(_mm_and_si128(val, vmag), shift_dq);
            res               = _mm_sub_epi32(_mm_xor_si128(mag, s), s);
            OPENHTJ2K_IF_CONSTEXPR (!store_i32) res = _mm_castps_si128(_mm_cvtepi32_ps(res));
          } else {
            const __m128 f = _mm_mul_ps(_mm_cvtepi32_ps(_mm_and_si128(val, vmag)), vfscale);
            res            = _mm_or_si128(_mm_castps_si128(f), _mm_andnot_si128(vmag, val));
          }
        }
        if (two_columns) {
          const __m128i o = _mm_shuffle_epi32(res, _MM_SHUFFLE(3, 1, 2, 0));  // row 0, then row 1
          _mm_storel_epi64(reinterpret_cast<__m128i *>(dp), o);
          _mm_storel_epi64(reinterpret_cast<__m128i *>(dp + out_stride), _mm_unpackhi_epi64(o, o));
          v3 = static_cast<uint32_t>(_mm_cvtsi128_si32(_mm_shuffle_epi32(vn, 0xFF)));
        } else {
          dp[0]          = _mm_cvtsi128_si32(res);
          dp[out_stride] = _mm_cvtsi128_si32(_mm_shuffle_epi32(res, 0x55));
        }
        v1 = static_cast<uint32_t>(_mm_cvtsi128_si32(_mm_shuffle_epi32(vn, 0x55)));
      } else {
  #else
      uint32_t vx;
      if (inf & 0xF0) {
        store_sample(&dp[0], ms_decode_sample(MagSgn, inf, U_q, 0, pLSB, vx));
        store_sample(&dp[out_stride], ms_decode_sample(MagSgn, inf, U_q, 1, pLSB, v1));
        if (two_columns) {
          store_sample(&dp[1], ms_decode_sample(MagSgn, inf, U_q, 2, pLSB, vx));
          store_sample(&dp[1 + out_stride], ms_decode_sample(MagSgn, inf, U_q, 3, pLSB, v3));
        }
      } else {
  #endif
        store_sample(&dp[0], 0);
        store_sample(&dp[out_stride], 0);
        if (two_columns) {
          store_sample(&dp[1], 0);
          store_sample(&dp[1 + out_stride], 0);
        }
      }
      vp[0]    = prev_v_n | v1;
      prev_v_n = v3;
    }
    vp[0] = prev_v_n;
  }
}

void ht_cleanup_decode2(j2k_codeblock *block, const uint8_t &pLSB, const int32_t Lcup, const int32_t Pcup,
                        const int32_t Scup) {
  fwd_buf<0xFF> MagSgn(block->get_compressed_data(), Pcup);
  MEL_dec MEL(block->get_compressed_data(), Lcup, Scup);
  rev_buf VLC_dec(block->get_compressed_data(), Lcup, Scup);
  const uint16_t QW = static_cast<uint16_t>(ceil_int(static_cast<int16_t>(block->size.x), 2));
  const uint16_t QH = static_cast<uint16_t>(ceil_int(static_cast<int16_t>(block->size.y), 2));

  alignas(32) uint32_t m_quads[8];
  alignas(32) uint32_t msval[8];
  alignas(32) uint32_t sigma_quads[8];
  alignas(32) uint32_t mu_quads[8];
  alignas(32) uint32_t v_quads[8];
  alignas(32) uint32_t known_1[2];

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
  uint32_t kappa0 = 1, kappa1 = 1;  // kappa is always 1 for initial line-pair

  const uint16_t *dec_table0, *dec_table1;
  dec_table0 = dec_CxtVLC_table0_fast_16;
  dec_table1 = dec_CxtVLC_table1_fast_16;

  alignas(32) uint32_t rholine[516];  // QW_max + 4, QW_max = 512
  std::memset(rholine, 0, (QW + 4U) * sizeof(uint32_t));
  uint32_t *rho_p = rholine + 1;
  alignas(32) int32_t Eline[1032];  // 2 * QW_max + 8, QW_max = 512
  std::memset(Eline, 0, (2U * QW + 8U) * sizeof(int32_t));
  int32_t *E_p = Eline + 1;

  uint32_t context = 0;
  uint32_t vlcval;
  int32_t mel_run = MEL.get_run();

  int32_t qx;
  // Initial line-pair
  for (qx = QW; qx > 0; qx -= 2) {
    // Decoding of significance and EMB patterns and unsigned residual offsets
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

    *rho_p++ = rho0;
    // calculate context for the next quad
    context = ((tv0 & 0xE0U) << 2) | ((tv0 & 0x10U) << 3);

    // Decoding of significance and EMB patterns and unsigned residual offsets
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

    *rho_p++ = rho1;

    for (uint32_t i = 0; i < 4; i++) {
      sigma_quads[i] = (rho0 >> i) & 1;
    }
    for (uint32_t i = 0; i < 4; i++) {
      sigma_quads[i + 4] = (rho1 >> i) & 1;
    }

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
    vlcval       = VLC_dec.advance(len);
    uvlc_result >>= 4;
    // quad 0 length
    len = uvlc_result & 0x7;  // quad 0 suffix length
    uvlc_result >>= 3;
    u0 = (uvlc_result & 7) + (tmp & ~(0xFFU << len));
    u1 = (uvlc_result >> 3) + (tmp >> len);

    U0 = kappa0 + u0;
    U1 = kappa1 + u1;

    for (uint32_t i = 0; i < 4; i++) {
      m_quads[i]     = sigma_quads[i] * U0 - ((emb_k_0 >> i) & 1);
      m_quads[i + 4] = sigma_quads[i + 4] * U1 - ((emb_k_1 >> i) & 1);
    }

    // recoverMagSgnValue
    for (uint32_t i = 0; i < 8; i++) {
      msval[i] = MagSgn.fetch();
      MagSgn.advance(m_quads[i]);
    }

    for (uint32_t i = 0; i < 4; i++) {
      known_1[Q0] = (emb_1_0 >> i) & 1;
      v_quads[i]  = msval[i] & ((1 << m_quads[i]) - 1U);
      v_quads[i] |= known_1[Q0] << m_quads[i];
      if (m_quads[i] != 0) {
        mu_quads[i] = v_quads[i] + 2;
        mu_quads[i] |= 1;
        mu_quads[i] <<= pLSB - 1;
        mu_quads[i] |= (v_quads[i] & 1) << 31;  // sign bit
      } else {
        mu_quads[i] = 0;
      }
    }
    for (uint32_t i = 0; i < 4; i++) {
      known_1[Q1]    = (emb_1_1 >> i) & 1;
      v_quads[i + 4] = msval[i + 4] & ((1 << m_quads[i + 4]) - 1U);
      v_quads[i + 4] |= known_1[Q1] << m_quads[i + 4];
      if (m_quads[i + 4] != 0) {
        mu_quads[i + 4] = v_quads[i + 4] + 2;
        mu_quads[i + 4] |= 1;
        mu_quads[i + 4] <<= pLSB - 1;
        mu_quads[i + 4] |= (v_quads[i + 4] & 1) << 31;  // sign bit
      } else {
        mu_quads[i + 4] = 0;
      }
    }
    *mp0++ = static_cast<int>(mu_quads[0]);
    *mp0++ = static_cast<int>(mu_quads[2]);
    *mp0++ = static_cast<int>(mu_quads[0 + 4]);
    *mp0++ = static_cast<int>(mu_quads[2 + 4]);
    *mp1++ = static_cast<int>(mu_quads[1]);
    *mp1++ = static_cast<int>(mu_quads[3]);
    *mp1++ = static_cast<int>(mu_quads[1 + 4]);
    *mp1++ = static_cast<int>(mu_quads[3 + 4]);

    *sp0++ = (rho0 >> 0) & 1;
    *sp0++ = (rho0 >> 2) & 1;
    *sp0++ = (rho1 >> 0) & 1;
    *sp0++ = (rho1 >> 2) & 1;
    *sp1++ = (rho0 >> 1) & 1;
    *sp1++ = (rho0 >> 3) & 1;
    *sp1++ = (rho1 >> 1) & 1;
    *sp1++ = (rho1 >> 3) & 1;

    *E_p++ = static_cast<int32_t>(32 - count_leading_zeros(v_quads[1]));
    *E_p++ = static_cast<int32_t>(32 - count_leading_zeros(v_quads[3]));
    *E_p++ = static_cast<int32_t>(32 - count_leading_zeros(v_quads[5]));
    *E_p++ = static_cast<int32_t>(32 - count_leading_zeros(v_quads[7]));
  }  // Initial line-pair end

  /*******************************************************************************************************************/
  // Non-initial line-pair
  /*******************************************************************************************************************/

  for (uint16_t row = 1; row < QH; row++) {
    rho_p = rholine + 1;
    E_p   = Eline + 1;
    mp0   = block->sample_buf + (row * 2U) * block->blksampl_stride;
    mp1   = block->sample_buf + (row * 2U + 1U) * block->blksampl_stride;
    sp0   = block->block_states + (row * 2U + 1U) * block->blkstate_stride + 1U;
    sp1   = block->block_states + (row * 2U + 2U) * block->blkstate_stride + 1U;
    rho1  = 0;

    int32_t Emax0, Emax1;
    // Calculate Emax for the next two quads
    Emax0 = find_max(E_p[-1], E_p[0], E_p[1], E_p[2]);
    Emax1 = find_max(E_p[1], E_p[2], E_p[3], E_p[4]);

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

      vlcval = VLC_dec.advance((tv1 & 0x000F) >> 1);

      for (uint32_t i = 0; i < 4; i++) {
        sigma_quads[i] = (rho0 >> i) & 1;
      }
      for (uint32_t i = 0; i < 4; i++) {
        sigma_quads[i + 4] = (rho1 >> i) & 1;
      }

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
      vlcval       = VLC_dec.advance(len);
      uvlc_result >>= 4;
      // quad 0 length
      len = uvlc_result & 0x7;  // quad 0 suffix length
      uvlc_result >>= 3;
      u0 = (uvlc_result & 7) + (tmp & ~(0xFFU << len));
      u1 = (uvlc_result >> 3) + (tmp >> len);

      gamma0 = ((rho0 & (rho0 - 1)) == 0) ? 0 : 1;  // (popcount32(rho0) < 2) ? 0 : 1;
      gamma1 = ((rho1 & (rho1 - 1)) == 0) ? 0 : 1;  // (popcount32(rho1) < 2) ? 0 : 1;
      kappa0 = (1 > gamma0 * (Emax0 - 1)) ? 1U : static_cast<uint8_t>(gamma0 * (Emax0 - 1));
      kappa1 = (1 > gamma1 * (Emax1 - 1)) ? 1U : static_cast<uint8_t>(gamma1 * (Emax1 - 1));
      U0     = kappa0 + u0;
      U1     = kappa1 + u1;

      for (uint32_t i = 0; i < 4; i++) {
        m_quads[i]     = sigma_quads[i] * U0 - ((emb_k_0 >> i) & 1);
        m_quads[i + 4] = sigma_quads[i + 4] * U1 - ((emb_k_1 >> i) & 1);
      }

      // recoverMagSgnValue
      for (uint32_t i = 0; i < 8; i++) {
        msval[i] = MagSgn.fetch();
        MagSgn.advance(m_quads[i]);
      }

      for (uint32_t i = 0; i < 4; i++) {
        known_1[Q0] = (emb_1_0 >> i) & 1;
        v_quads[i]  = msval[i] & ((1 << m_quads[i]) - 1U);
        v_quads[i] |= known_1[Q0] << m_quads[i];
        if (m_quads[i] != 0) {
          mu_quads[i] = v_quads[i] + 2;
          mu_quads[i] |= 1;
          mu_quads[i] <<= pLSB - 1;
          mu_quads[i] |= (v_quads[i] & 1) << 31;  // sign bit
        } else {
          mu_quads[i] = 0;
        }
      }
      for (uint32_t i = 0; i < 4; i++) {
        known_1[Q1]    = (emb_1_1 >> i) & 1;
        v_quads[i + 4] = msval[i + 4] & ((1 << m_quads[i + 4]) - 1U);
        v_quads[i + 4] |= known_1[Q1] << m_quads[i + 4];
        if (m_quads[i + 4] != 0) {
          mu_quads[i + 4] = v_quads[i + 4] + 2;
          mu_quads[i + 4] |= 1;
          mu_quads[i + 4] <<= pLSB - 1;
          mu_quads[i + 4] |= (v_quads[i + 4] & 1) << 31;  // sign bit
        } else {
          mu_quads[i + 4] = 0;
        }
      }
      *mp0++ = static_cast<int>(mu_quads[0]);
      *mp0++ = static_cast<int>(mu_quads[2]);
      *mp0++ = static_cast<int>(mu_quads[0 + 4]);
      *mp0++ = static_cast<int>(mu_quads[2 + 4]);
      *mp1++ = static_cast<int>(mu_quads[1]);
      *mp1++ = static_cast<int>(mu_quads[3]);
      *mp1++ = static_cast<int>(mu_quads[1 + 4]);
      *mp1++ = static_cast<int>(mu_quads[3 + 4]);

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

      Emax0  = find_max(E_p[3], E_p[4], E_p[5], E_p[6]);
      Emax1  = find_max(E_p[5], E_p[6], E_p[7], E_p[8]);
      E_p[0] = static_cast<int32_t>(32 - count_leading_zeros(v_quads[1]));
      E_p[1] = static_cast<int32_t>(32 - count_leading_zeros(v_quads[3]));
      E_p[2] = static_cast<int32_t>(32 - count_leading_zeros(v_quads[5]));
      E_p[3] = static_cast<int32_t>(32 - count_leading_zeros(v_quads[7]));
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
        state_p[0] |= bit << SHIFT_REF;
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
  if (pLSB == 0) return;  // no plane below the LSB; mirrors ht_magref_decode (avoids 1 << (pLSB-1) UB)
  SP_dec SigProp(HT_magref_segment, magref_length);
  const uint32_t num_v_stripe = block->size.y / 4;
  const uint32_t num_h_stripe = block->size.x / 4;
  uint32_t i_start            = 0, j_start;
  uint32_t width              = 4;
  uint32_t width_last;
  uint32_t height = 4;

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
  if (pLSB == 0) return;
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
          tmp = static_cast<int32_t>((0xFFFFFFFEU | static_cast<unsigned int>(bit)) << pLSB);
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
        tmp = static_cast<int32_t>((0xFFFFFFFEU | static_cast<unsigned int>(bit)) << pLSB);
        sp[0] &= tmp;
        sp[0] |= 1 << (pLSB - 1);  // new bin center ( = 0.5)
      }
    }
  }
}

// Reversible-path dequantization: undo the ROI up-shift and the binary point,
// restore the sign, and write each sample through the store callables (int32
// for the integer 5/3 IDWT, float otherwise).  A function template rather than
// a generic lambda so that the library builds as C++11.
template <class Store>
static inline void rev_dequant_scalar(const j2k_codeblock *cb, uint8_t ROIshift, int32_t pLSB,
                                      uint32_t mask, Store store_fn) {
  for (size_t i = 0; i < static_cast<size_t>(cb->size.y); i++) {
    int32_t *val = cb->sample_buf + i * cb->blksampl_stride;
    sprec_t *dst = cb->band_buf + i * cb->band_stride;
    size_t len   = cb->size.x;
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
      store_fn(dst, *val);
      val++;
      dst++;
    }
  }
}

void j2k_codeblock::dequantize(uint8_t ROIshift) const {
  /* ready for ROI adjustment and dequantization */

  // number of decoded magnitude bit‐planes
  const int32_t pLSB = 31 - M_b;  // indicates binary point;

  // bit mask for ROI detection
  const uint32_t mask = UINT32_MAX >> (M_b + 1);
  // reconstruction parameter defined in E.1.1.2 of the spec

  // Direct float scale factor: decoded magnitude is in Q(31-M_b) fixed-point;
  // the result must be in Q(FRACBITS).
  float fscale_direct = this->stepsize;
  fscale_direct *= static_cast<float>(1 << FRACBITS);
  if (M_b <= 31) {
    fscale_direct /= (static_cast<float>(1 << (31 - M_b)));
  } else {
    fscale_direct *= (static_cast<float>(1 << (M_b - 31)));
  }
  if (this->transformation == 1) {
    if (this->dequant_i32)
      rev_dequant_scalar(this, ROIshift, pLSB, mask,
                         [](sprec_t *d, int32_t v) { *reinterpret_cast<int32_t *>(d) = v; });
    else
      rev_dequant_scalar(this, ROIshift, pLSB, mask,
                         [](sprec_t *d, int32_t v) { *d = static_cast<sprec_t>(v); });
  } else if (ROIshift == 0) {
    // Lossy, no ROI (common case): direct float multiply, matching the fused
    // dequant path (dequant_store_scalar) and the AVX2/NEON/WASM variants of
    // this function bit-exactly.  The previous integer fixed-point pipeline
    // here rounded every sample to an integer in the Q(FRACBITS) domain, so
    // blocks taking this fallback (multiple HT passes, odd height) were
    // dequantized with different precision than fused blocks and than SIMD
    // builds, causing up to ±1 LSB divergence in decoded images.
    for (size_t i = 0; i < static_cast<size_t>(this->size.y); i++) {
      int32_t *val = this->sample_buf + i * this->blksampl_stride;
      sprec_t *dst = this->band_buf + i * this->band_stride;
      size_t len   = this->size.x;
      for (; len > 0; --len) {
        const int32_t sign = *val & INT32_MIN;
        float f            = static_cast<float>(*val & INT32_MAX) * fscale_direct;
        if (sign) f = -f;
        *dst = f;
        val++;
        dst++;
      }
    }
  } else {
    // Lossy ROI path — rarely used; keep the integer-arithmetic approach.
    constexpr int32_t downshift = 15;
    float fscale                = fscale_direct;
    fscale *= static_cast<float>(1 << 16) * static_cast<float>(1 << downshift);
    const auto scale = static_cast<int32_t>(fscale + 0.5f);
    for (size_t i = 0; i < static_cast<size_t>(this->size.y); i++) {
      int32_t *val = this->sample_buf + i * this->blksampl_stride;
      sprec_t *dst = this->band_buf + i * this->band_stride;
      size_t len   = this->size.x;

      for (; len > 0; --len) {
        int32_t sign = *val & INT32_MIN;
        *val &= INT32_MAX;
        // detect background region and upshift it
        if (((uint32_t)*val & ~mask) == 0) {
          *val <<= ROIshift;
        }
        // to prevent overflow, truncate to int16_t
        *val = (*val + (1 << 15)) >> 16;
        //  dequantization
        *val *= scale;
        // downshift
        *val = (int32_t)((*val + (1 << (downshift - 1))) >> downshift);
        // convert sign-magnitude to two's complement form
        if (sign) {
          *val = -(*val & INT32_MAX);
        }
        *dst = static_cast<sprec_t>(*val);
        val++;
        dst++;
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

  // HT block decoding
  bool dequant_done = false;
  // The fused-dequant scalar/NEON/AVX2/WASM kernels process samples in
  // 2-row quad pairs and write both rows of every pair unconditionally to
  // `block->band_buf`.  When `block->size.y` is odd the last pair's
  // second-row write lands `band_stride` bytes past the block's final
  // row — into the NEXT block's band_buf region when codeblocks tile
  // vertically inside the subband (e.g. 1×1 blocks stacked in a narrow
  // subband from a horizontally-subsampled component).  Under single-
  // threaded decode the overflow is harmlessly overwritten by the next
  // block's legitimate row-0 write, but under multi-threaded dispatch
  // the blocks finish out-of-order and the stale overflow clobbers
  // adjacent blocks' output.  Fall back to the non-fused path (which
  // writes into the block-local sample_buf scratch, not band_buf) when
  // height is odd; the separate dequant pass below handles bounds
  // correctly.
  const bool fuseable = (su.num_ht_passes == 1) && (ROIshift == 0) && ((block->size.y & 1u) == 0u);
  if (fuseable) {
    if (block->dequant_i32)
      ht_cleanup_step2<true, true>(block, pLSB, su.Pcup, scratch, sstr);
    else
      ht_cleanup_step2<true>(block, pLSB, su.Pcup, scratch, sstr);
    dequant_done = true;
  } else {
    ht_cleanup_step2<>(block, pLSB, su.Pcup, scratch, sstr);
  }
  if (su.num_ht_passes > 1) {
    ht_sigprop_decode(block, su.Dref, su.Lref, static_cast<uint8_t>(30 - (su.S_blk + 1)));
  }
  if (su.num_ht_passes > 2) {
    ht_magref_decode(block, su.Dref, su.Lref, static_cast<uint8_t>(30 - (su.S_blk + 1)));
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

bool htj2k_decode_cleanup_raw(j2k_codeblock *block) {
  const ht_dec_setup su = htj2k_dec_setup(block);
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
  ht_cleanup_step1_nway<1, true>(&ln, QW, QH, sstr);
  ht_cleanup_step2<>(block, static_cast<uint8_t>(30 - su.S_blk), su.Pcup, scratch, sstr);
  return true;
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