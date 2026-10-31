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

// HT block encoder for x86-64 builds without AVX2 (OPENHTJ2K_ENABLE_SSE2, utils.hpp).
// It follows the portable encoder (ht_block_encoding.cpp) step for step and writes the
// same bytes; the per-sample work of each quad, its exponents, significance pattern,
// EMB pattern and MagSgn codewords, is done in the four lanes of an SSE2 register.
// SSE2 has no count-leading-zeros or per-lane shift: the bit length of a sample comes
// from the exponent of its conversion to float, and 2^m from building the float 2^m.

#include "utils.hpp"  // OPENHTJ2K_ENABLE_SSE2
#if defined(OPENHTJ2K_ENABLE_SSE2)
  #include <algorithm>
  #include <cmath>
  #include <emmintrin.h>
  #include "coding_units.hpp"
  #include "ht_block_encoding.hpp"
  #include "coding_local.hpp"
  #include "enc_CxtVLC_tables.hpp"

void j2k_codeblock::quantize(uint32_t &or_val) {
  const uint32_t height = this->size.y;
  const uint32_t width  = this->size.x;
  const uint32_t stride = this->band_stride;
  const bool lossless   = (this->transformation != 0);

  float fscale = 1.0f;
  if (!lossless) {
    fscale = 1.0f / this->stepsize;
    fscale /= (1 << (FRACBITS));
  }

  const __m128 vscale = _mm_set1_ps(fscale);
  __m128i vor         = _mm_setzero_si128();
  uint32_t or_acc     = 0;
  for (uint32_t i = 0; i < height; ++i) {
    const sprec_t *sp = this->band_buf + i * stride;
    int32_t *dp       = this->sample_buf + i * blksampl_stride;
    uint32_t x        = 0;
    for (; x + 4 <= width; x += 4) {
      const __m128 f = _mm_loadu_ps(sp + x);
      _mm_storeu_si128(reinterpret_cast<__m128i *>(dp + x),
                       htj2k_magsgn_sse2(_mm_cvttps_epi32(lossless ? f : _mm_mul_ps(f, vscale)), vor));
    }
    if (lossless) {
      for (; x < width; ++x) dp[x] = htj2k_magsgn(static_cast<int32_t>(sp[x]), or_acc);
    } else {
      for (; x < width; ++x)
        dp[x] = htj2k_magsgn(static_cast<int32_t>(static_cast<float>(sp[x]) * fscale), or_acc);
    }
    if (blksampl_stride > width) memset(dp + width, 0, (blksampl_stride - width) * sizeof(int32_t));
  }
  if (any_nonzero_sse2(vor)) or_acc |= 1;
  if (or_acc) or_val |= 1;
  const uint32_t QHx2 = (height + 7U) & ~7U;
  for (uint32_t i = height; i < QHx2; ++i)
    memset(this->sample_buf + i * blksampl_stride, 0, blksampl_stride * sizeof(int32_t));
}

/********************************************************************************
 * HT cleanup encoding: helper functions
 *******************************************************************************/
namespace {
struct ht_cleanup_scratch {
  alignas(16) uint32_t flat_v[512 * 4];  // MagSgn codewords of one line pair
  alignas(16) uint32_t flat_m[512 * 4];  // and their lengths
  int32_t Eline[2][2 * 512 + 6];
  int32_t rholine[2][512 + 3];
  int32_t ctx[512];
  int32_t rho[512];
  int32_t uq[512];
  uint32_t cwd[512];
  uint32_t lw[512];
};

// One quad: its four samples s0 = sp0[0], s1 = sp1[0], s2 = sp0[1], s3 = sp1[1] in the
// lanes, in the encoder's sample form (bit 31 set if significant).
struct quad {
  __m128i raw;     // the samples
  __m128i v;       // their MagSgn values (bit 31 cleared)
  __m128i E;       // bit lengths of v (0 for 0)
  __m128i Emax_v;  // their largest, in every lane
  int32_t rho;     // significance pattern: bit i = lane i significant
  int32_t Emax;
};

FORCE_INLINE void load_quad(const int32_t *sp0, const int32_t *sp1, quad &qd) {
  const __m128i t0 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(sp0));
  const __m128i t1 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(sp1));
  qd.raw           = _mm_unpacklo_epi32(t0, t1);
  qd.rho           = _mm_movemask_ps(_mm_castsi128_ps(qd.raw));
  qd.v             = _mm_and_si128(qd.raw, _mm_set1_epi32(0x7FFFFFFF));
  // Bit length from the float exponent.  Clearing every bit 8 places below a set one keeps
  // the conversion from rounding up to the next power of two; 0 converts to exponent 0.
  const __m128i k = _mm_andnot_si128(_mm_srli_epi32(qd.v, 8), qd.v);
  const __m128i e = _mm_srli_epi32(_mm_castps_si128(_mm_cvtepi32_ps(k)), 23);
  qd.E            = _mm_subs_epu16(e, _mm_set1_epi32(126));  // 32-bit lanes below 2^16
  __m128i mx      = _mm_max_epi16(qd.E, _mm_shuffle_epi32(qd.E, _MM_SHUFFLE(2, 3, 0, 1)));
  qd.Emax_v       = _mm_max_epi16(mx, _mm_shuffle_epi32(mx, _MM_SHUFFLE(1, 0, 3, 2)));
  qd.Emax         = _mm_cvtsi128_si32(qd.Emax_v);
}

// Bit i set where the exponent of sample i is Emax.
FORCE_INLINE int32_t emb_pattern(const quad &qd) {
  return _mm_movemask_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(qd.E, qd.Emax_v)));
}

// Lanes all ones where bit i of x (0..15) is set.
alignas(16) const int32_t spread_table[16][4] = {
    {0, 0, 0, 0},   {-1, 0, 0, 0},   {0, -1, 0, 0},   {-1, -1, 0, 0},  {0, 0, -1, 0},  {-1, 0, -1, 0},
    {0, -1, -1, 0}, {-1, -1, -1, 0}, {0, 0, 0, -1},   {-1, 0, 0, -1},  {0, -1, 0, -1}, {-1, -1, 0, -1},
    {0, 0, -1, -1}, {-1, 0, -1, -1}, {0, -1, -1, -1}, {-1, -1, -1, -1}};
FORCE_INLINE __m128i spread_bits(int32_t x) {
  return _mm_load_si128(reinterpret_cast<const __m128i *>(spread_table[x]));
}

// The quad's MagSgn codewords and lengths: m = U for a significant sample, less 1 where the
// EMB pattern says its top bit is implied (embk), and v less that top bit where it is 1
// (emb1).  2^m, m in [0, 31], is the float 2^m converted back (2^31 converts to 1 << 31).
FORCE_INLINE void append_quad(uint32_t *fv, uint32_t *fm, const quad &qd, int32_t U, int32_t embk,
                              int32_t emb1) {
  const __m128i sig = _mm_srai_epi32(qd.raw, 31);
  const __m128i m   = _mm_add_epi32(_mm_and_si128(sig, _mm_set1_epi32(U)), spread_bits(embk));
  const __m128i pow2 =
      _mm_cvttps_epi32(_mm_castsi128_ps(_mm_slli_epi32(_mm_add_epi32(m, _mm_set1_epi32(127)), 23)));
  _mm_storeu_si128(reinterpret_cast<__m128i *>(fv),
                   _mm_sub_epi32(qd.v, _mm_and_si128(pow2, spread_bits(emb1))));
  _mm_storeu_si128(reinterpret_cast<__m128i *>(fm), m);
}

// The exponents of the quad's lower row (lanes 1 and 3), the line's next E values.
FORCE_INLINE void store_lower_E(int32_t *E, const quad &qd) {
  E[0] = _mm_cvtsi128_si32(_mm_shuffle_epi32(qd.E, _MM_SHUFFLE(1, 1, 1, 1)));
  E[1] = _mm_cvtsi128_si32(_mm_shuffle_epi32(qd.E, _MM_SHUFFLE(3, 3, 3, 3)));
}
}  // namespace

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

/********************************************************************************
 * HT cleanup encoding
 *******************************************************************************/
int32_t htj2k_cleanup_encode(j2k_codeblock *const block, const uint8_t ROIshift) noexcept {
  int32_t Lcup;  // length of HT cleanup pass
  int32_t Pcup;  // length of MagSgn buffer
  int32_t Scup;  // length of MEL buffer + VLC buffer
  uint32_t or_val = 0;
  if (ROIshift) {
    printf("WARNING: Encoding with ROI is not supported.\n");
  }

  const int32_t QW = static_cast<int32_t>(ceil_int(static_cast<int16_t>(block->size.x), 2));
  const int32_t QH = static_cast<int32_t>(ceil_int(static_cast<int16_t>(block->size.y), 2));

  if (!block->pre_quantized)
    block->quantize(or_val);
  else
    or_val = block->pre_or_val;

  if (!or_val) {
    // nothing to code: the code-block is empty
    block->length          = 0;
    block->pass_length[0]  = 0;
    block->num_passes      = 0;
    block->layer_passes[0] = 0;
    block->layer_start[0]  = 0;
    block->num_ZBP         = static_cast<uint8_t>(block->get_Mb() - 1);
    return static_cast<int32_t>(block->length);
  }

  alignas(4) static thread_local uint8_t fwd_buf[MAX_Lcup];
  alignas(4) static thread_local uint8_t rev_buf[MAX_Scup];
  static thread_local ht_cleanup_scratch scratch;
  ht_cleanup_scratch &sc = scratch;

  state_MS_enc MagSgn_encoder(fwd_buf);
  state_MEL_enc MEL_encoder(rev_buf);
  state_VLC_enc VLC_encoder(rev_buf);

  const size_t stride = block->blksampl_stride;
  // Exponents and significance of the line pair above (old) and of this one (new), with a
  // zero border on each side.
  std::fill_n(sc.Eline[0], 2 * QW + 6, int32_t{0});
  std::fill_n(sc.Eline[1], 2 * QW + 6, int32_t{0});
  std::fill_n(sc.rholine[0], QW + 3, int32_t{0});
  std::fill_n(sc.rholine[1], QW + 3, int32_t{0});
  int32_t *E_old   = sc.Eline[0] + 1;
  int32_t *E_new   = sc.Eline[1] + 1;
  int32_t *rho_old = sc.rholine[0] + 1;
  int32_t *rho_new = sc.rholine[1] + 1;

  quad q0, q1;
  /*******************************************************************************************************************/
  // Initial line-pair: quads in pairs, as the UVLC code of the first line pair joins them.
  /*******************************************************************************************************************/
  {
    const int32_t *sp0 = block->sample_buf;
    const int32_t *sp1 = sp0 + stride;
    int32_t context    = 0;
    int32_t fc         = 0;
    int32_t q          = 0;
    for (; q + 1 < QW; q += 2) {
      load_quad(sp0 + 2 * q, sp1 + 2 * q, q0);
      load_quad(sp0 + 2 * q + 2, sp1 + 2 * q + 2, q1);
      if (context == 0) MEL_encoder.encodeMEL(q0.rho != 0);
      const int32_t U0       = std::max(q0.Emax, 1);
      const int32_t u0       = U0 - 1;
      const int32_t emb0     = u0 ? emb_pattern(q0) : 0;
      const uint32_t cv0     = enc_CxtVLC_table0[emb0 + (q0.rho << 4) + (context << 8)];
      const int32_t embk0    = static_cast<int32_t>(cv0 & 0xF);
      const uint32_t lw0     = (cv0 >> 4) & 0x07;
      context                = (q0.rho >> 1) | (q0.rho & 0x1);
      const int32_t U1       = std::max(q1.Emax, 1);
      const int32_t u1       = U1 - 1;
      const int32_t emb1     = u1 ? emb_pattern(q1) : 0;
      const uint32_t cv1     = enc_CxtVLC_table0[emb1 + (q1.rho << 4) + (context << 8)];
      const int32_t embk1    = static_cast<int32_t>(cv1 & 0xF);
      const uint32_t lw1     = (cv1 >> 4) & 0x07;
      const uint32_t uvlc    = enc_UVLC_table0[u0 + (u1 << 5)];
      const uint32_t uvlc_lw = uvlc & 0xFF;
      VLC_encoder.emitVLCBits((cv0 >> 7) | (static_cast<uint64_t>(cv1 >> 7) << lw0)
                                  | (static_cast<uint64_t>(uvlc >> 8) << (lw0 + lw1)),
                              lw0 + lw1 + uvlc_lw);
      const int32_t u_min = std::min(u0, u1);
      if (context == 0) {
        MEL_encoder.encodeMEL((q1.rho != 0 || u_min > 2) ? 1 : 0);
      } else if (u0 && u1) {
        MEL_encoder.encodeMEL(u_min > 2 ? 1 : 0);
      }
      append_quad(sc.flat_v + fc, sc.flat_m + fc, q0, U0, embk0, emb0 & embk0);
      append_quad(sc.flat_v + fc + 4, sc.flat_m + fc + 4, q1, U1, embk1, emb1 & embk1);
      fc += 8;
      context        = (q1.rho >> 1) | (q1.rho & 0x1);
      rho_old[q]     = q0.rho;
      rho_old[q + 1] = q1.rho;
      store_lower_E(E_old + 2 * q, q0);
      store_lower_E(E_old + 2 * q + 2, q1);
    }
    if (q < QW) {
      load_quad(sp0 + 2 * q, sp1 + 2 * q, q0);
      if (context == 0) MEL_encoder.encodeMEL(q0.rho != 0);
      const int32_t U0       = std::max(q0.Emax, 1);
      const int32_t u0       = U0 - 1;
      const int32_t emb0     = u0 ? emb_pattern(q0) : 0;
      const uint32_t cv0     = enc_CxtVLC_table0[emb0 + (q0.rho << 4) + (context << 8)];
      const int32_t embk0    = static_cast<int32_t>(cv0 & 0xF);
      const uint32_t lw0     = (cv0 >> 4) & 0x07;
      const uint32_t uvlc    = enc_UVLC_table0[u0];
      const uint32_t uvlc_lw = uvlc & 0xFF;
      VLC_encoder.emitVLCBits((cv0 >> 7) | (static_cast<uint64_t>(uvlc >> 8) << lw0), lw0 + uvlc_lw);
      append_quad(sc.flat_v + fc, sc.flat_m + fc, q0, U0, embk0, emb0 & embk0);
      fc += 4;
      rho_old[q] = q0.rho;
      store_lower_E(E_old + 2 * q, q0);
    }
    MagSgn_encoder.emitFlat(sc.flat_v, sc.flat_m, fc);
  }

  /*******************************************************************************************************************/
  // Non-initial line-pairs: compute every quad's context, kappa, VLC codeword and MagSgn
  // codewords first (no serial dependency), then run the serial MEL/VLC pass.
  /*******************************************************************************************************************/
  for (int32_t qy = 1; qy < QH; qy++) {
    const int32_t *sp0 = block->sample_buf + 2U * static_cast<uint32_t>(qy) * stride;
    const int32_t *sp1 = sp0 + stride;
    int32_t rho_west   = 0;
    for (int32_t q = 0; q < QW; ++q) {
      load_quad(sp0 + 2 * q, sp1 + 2 * q, q0);
      const int32_t rho = q0.rho;
      // (w | sw) << 9, (nw | n) << 8, (ne | nf) << 10
      const int32_t ctx = ((rho_west & 0x4) << 7) | ((rho_west & 0x8) << 6) | ((rho_old[q - 1] & 0x8) << 5)
                          | ((rho_old[q] & 0xa) << 7) | ((rho_old[q + 1] & 0x2) << 9);
      const int32_t Emax_line =
          find_max(E_old[2 * q - 1], E_old[2 * q], E_old[2 * q + 1], E_old[2 * q + 2]);
      // gamma = (popcount(rho) > 1)
      const int32_t kappa = (rho & (rho - 1)) ? std::max(Emax_line - 1, 1) : 1;
      const int32_t U     = std::max(q0.Emax, kappa);
      const int32_t uq    = U - kappa;
      const int32_t emb   = uq ? emb_pattern(q0) : 0;
      const uint32_t cv   = enc_CxtVLC_table1[emb + (rho << 4) + ctx];
      const int32_t embk  = static_cast<int32_t>(cv & 0xF);
      append_quad(sc.flat_v + 4 * q, sc.flat_m + 4 * q, q0, U, embk, emb & embk);
      sc.ctx[q] = ctx;
      sc.rho[q] = rho;
      sc.uq[q]  = uq;
      sc.cwd[q] = cv >> 7;
      sc.lw[q]  = (cv >> 4) & 0x07;
      store_lower_E(E_new + 2 * q, q0);
      rho_new[q] = rho;
      rho_west   = rho;
    }

    int32_t q = 0;
    for (; q + 1 < QW; q += 2) {
      if (sc.ctx[q] == 0) MEL_encoder.encodeMEL(sc.rho[q] != 0);
      const uint32_t uvlc = enc_UVLC_table1[sc.uq[q] + (sc.uq[q + 1] << 5)];
      const uint32_t lw0  = sc.lw[q];
      const uint32_t lw1  = sc.lw[q + 1];
      VLC_encoder.emitVLCBits(sc.cwd[q] | (static_cast<uint64_t>(sc.cwd[q + 1]) << lw0)
                                  | (static_cast<uint64_t>(uvlc >> 8) << (lw0 + lw1)),
                              lw0 + lw1 + (uvlc & 0xFF));
      if (sc.ctx[q + 1] == 0) MEL_encoder.encodeMEL(sc.rho[q + 1] != 0);
    }
    if (q < QW) {
      if (sc.ctx[q] == 0) MEL_encoder.encodeMEL(sc.rho[q] != 0);
      const uint32_t uvlc = enc_UVLC_table1[sc.uq[q]];
      VLC_encoder.emitVLCBits(sc.cwd[q] | (static_cast<uint64_t>(uvlc >> 8) << sc.lw[q]),
                              sc.lw[q] + (uvlc & 0xFF));
    }
    MagSgn_encoder.emitFlat(sc.flat_v, sc.flat_m, 4 * QW);
    std::swap(E_old, E_new);
    std::swap(rho_old, rho_new);
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
  block->set_compressed_data(fwd_buf, static_cast<uint16_t>(Lcup), static_cast<uint16_t>(HT_MAX_LREF));
  block->length          = static_cast<uint32_t>(Lcup);
  block->pass_length[0]  = static_cast<unsigned int>(Lcup);
  block->num_passes      = 1;
  block->layer_passes[0] = 1;
  block->layer_start[0]  = 0;
  block->num_ZBP         = static_cast<uint8_t>(block->get_Mb() - 1);
  return static_cast<int32_t>(block->length);
}
/********************************************************************************
 * HT encoding
 *******************************************************************************/
int32_t htj2k_encode(j2k_codeblock *block, uint8_t ROIshift) noexcept {
  htj2k_cleanup_encode(block, ROIshift);
  return EXIT_SUCCESS;
}
#endif
