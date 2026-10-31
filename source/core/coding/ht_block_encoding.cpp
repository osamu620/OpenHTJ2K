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

// Portable HT block encoder (no SIMD; x86-64 builds without AVX2 use the SSE2 one,
// ht_block_encoding_sse2.cpp).  It mirrors the AVX2 version
// (ht_block_encoding_avx2.cpp) in plain C++ and writes the same bytes:
// VLC and UVLC codewords of a quad pair are emitted together, MagSgn
// codewords are collected per line pair and written in one batch, and in
// line pairs after the first every per-quad value is computed before the
// serial MEL/VLC pass.

#include "utils.hpp"  // OPENHTJ2K_ENABLE_SSE2
#if !defined(OPENHTJ2K_ENABLE_ARM_NEON) && (!defined(__AVX2__) || !defined(OPENHTJ2K_TRY_AVX2)) \
    && !defined(OPENHTJ2K_ENABLE_WASM_SIMD) && !defined(OPENHTJ2K_ENABLE_SSE2)
  #include <algorithm>
  #include <cmath>
  #include "coding_units.hpp"
  #include "ht_block_encoding.hpp"
  #include "coding_local.hpp"
  #include "enc_CxtVLC_tables.hpp"
  #include "utils.hpp"

// Quantize DWT coefficients and transfer them to codeblock buffer in a form of MagSgn value
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

  uint32_t or_acc = 0;
  for (uint32_t i = 0; i < height; ++i) {
    const sprec_t *sp = this->band_buf + i * stride;
    int32_t *dp       = this->sample_buf + i * blksampl_stride;
    if (lossless) {
      for (uint32_t x = 0; x < width; ++x) dp[x] = htj2k_magsgn(static_cast<int32_t>(sp[x]), or_acc);
    } else {
      for (uint32_t x = 0; x < width; ++x)
        dp[x] = htj2k_magsgn(static_cast<int32_t>(static_cast<float>(sp[x]) * fscale), or_acc);
    }
    if (blksampl_stride > width) memset(dp + width, 0, (blksampl_stride - width) * sizeof(int32_t));
  }
  if (or_acc) or_val |= 1;
  const uint32_t QHx2 = (height + 7U) & ~7U;
  for (uint32_t i = height; i < QHx2; ++i)
    memset(this->sample_buf + i * blksampl_stride, 0, blksampl_stride * sizeof(int32_t));
}

/********************************************************************************
 * HT cleanup encoding: helper functions
 *******************************************************************************/
namespace {
// Per-thread scratch for one codeblock (width <= 1024, so at most 512 quads
// per line pair).  Kept off the stack: some targets have small thread stacks.
struct ht_cleanup_scratch {
  uint32_t flat_v[512 * 4];  // MagSgn codewords of one line pair
  uint32_t flat_m[512 * 4];  // and their lengths
  int32_t Eline[2][2 * 512 + 6];
  int32_t rholine[2][512 + 3];
  int32_t ctx[512];
  int32_t rho[512];
  int32_t uq[512];
  uint32_t cwd[512];
  uint32_t lw[512];
};

// Read one quad (2x2 samples, scan order (r0,c0) (r1,c0) (r0,c1) (r1,c1)).
// v: magnitude-sign values, E: their bit lengths, rho: significance bits.
FORCE_INLINE void load_quad(const int32_t *sp0, const int32_t *sp1, uint32_t *v, int32_t *E, int32_t &rho,
                            int32_t &Emax) {
  const uint32_t s0 = static_cast<uint32_t>(sp0[0]);
  const uint32_t s1 = static_cast<uint32_t>(sp1[0]);
  const uint32_t s2 = static_cast<uint32_t>(sp0[1]);
  const uint32_t s3 = static_cast<uint32_t>(sp1[1]);
  rho  = static_cast<int32_t>((s0 >> 31) | ((s1 >> 31) << 1) | ((s2 >> 31) << 2) | ((s3 >> 31) << 3));
  v[0] = s0 & 0x7FFFFFFFU;
  v[1] = s1 & 0x7FFFFFFFU;
  v[2] = s2 & 0x7FFFFFFFU;
  v[3] = s3 & 0x7FFFFFFFU;
  // An insignificant sample is stored as 0, so its bit length is 0 too.
  E[0] = static_cast<int32_t>(32 - count_leading_zeros(v[0]));
  E[1] = static_cast<int32_t>(32 - count_leading_zeros(v[1]));
  E[2] = static_cast<int32_t>(32 - count_leading_zeros(v[2]));
  E[3] = static_cast<int32_t>(32 - count_leading_zeros(v[3]));
  Emax = find_max(E[0], E[1], E[2], E[3]);
}

// Samples whose exponent equals the quad maximum.
FORCE_INLINE int32_t emb_pattern(const int32_t *E, int32_t Emax) {
  return (E[0] == Emax) | ((E[1] == Emax) << 1) | ((E[2] == Emax) << 2) | ((E[3] == Emax) << 3);
}

// MagSgn codewords of one quad: m = U for a significant sample, less one
// when its MSB is implied by the EMB pattern (embk); a known 1 (emb1) is
// removed from the value.
FORCE_INLINE void append_quad(uint32_t *fv, uint32_t *fm, const uint32_t *v, int32_t rho, int32_t U,
                              int32_t embk, int32_t emb1) {
  for (int32_t i = 0; i < 4; ++i) {
    const uint32_t m = static_cast<uint32_t>(((rho >> i) & 1) * U - ((embk >> i) & 1));
    fv[i]            = v[i] - (static_cast<uint32_t>((emb1 >> i) & 1) << (m & 31U));
    fm[i]            = m;
  }
}
}  // namespace

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

  const int32_t QW = static_cast<int32_t>(ceil_int(static_cast<int16_t>(block->size.x), 2));
  const int32_t QH = static_cast<int32_t>(ceil_int(static_cast<int16_t>(block->size.y), 2));

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
  // All positions written before they are read; no zeroing needed between codeblocks.
  alignas(4) static thread_local uint8_t fwd_buf[MAX_Lcup];
  alignas(4) static thread_local uint8_t rev_buf[MAX_Scup];
  static thread_local ht_cleanup_scratch scratch;
  ht_cleanup_scratch &sc = scratch;

  state_MS_enc MagSgn_encoder(fwd_buf);
  state_MEL_enc MEL_encoder(rev_buf);
  state_VLC_enc VLC_encoder(rev_buf);

  const size_t stride = block->blksampl_stride;
  // Bit lengths of the bottom row of the previous line pair (two per quad) and
  // its rho values, double-buffered, with zero entries past both ends.
  std::fill_n(sc.Eline[0], 2 * QW + 6, int32_t{0});
  std::fill_n(sc.Eline[1], 2 * QW + 6, int32_t{0});
  std::fill_n(sc.rholine[0], QW + 3, int32_t{0});
  std::fill_n(sc.rholine[1], QW + 3, int32_t{0});
  int32_t *E_old   = sc.Eline[0] + 1;
  int32_t *E_new   = sc.Eline[1] + 1;
  int32_t *rho_old = sc.rholine[0] + 1;
  int32_t *rho_new = sc.rholine[1] + 1;

  uint32_t v0[4], v1[4];
  int32_t E0[4], E1[4];
  /*******************************************************************************************************************/
  // Initial line-pair
  /*******************************************************************************************************************/
  {
    const int32_t *sp0 = block->sample_buf;
    const int32_t *sp1 = sp0 + stride;
    int32_t context    = 0;
    int32_t fc         = 0;
    int32_t q          = 0;
    for (; q + 1 < QW; q += 2) {
      int32_t rho0, rho1, Emax0, Emax1;
      load_quad(sp0 + 2 * q, sp1 + 2 * q, v0, E0, rho0, Emax0);
      load_quad(sp0 + 2 * q + 2, sp1 + 2 * q + 2, v1, E1, rho1, Emax1);
      // MEL encoding for the first quad
      if (context == 0) MEL_encoder.encodeMEL(rho0 != 0);
      // kappa is 1 in the initial line pair
      const int32_t U0       = std::max(Emax0, 1);
      const int32_t u0       = U0 - 1;
      const int32_t emb0     = u0 ? emb_pattern(E0, Emax0) : 0;
      const uint32_t cv0     = enc_CxtVLC_table0[emb0 + (rho0 << 4) + (context << 8)];
      const int32_t embk0    = static_cast<int32_t>(cv0 & 0xF);
      const uint32_t lw0     = (cv0 >> 4) & 0x07;
      context                = (rho0 >> 1) | (rho0 & 0x1);
      const int32_t U1       = std::max(Emax1, 1);
      const int32_t u1       = U1 - 1;
      const int32_t emb1     = u1 ? emb_pattern(E1, Emax1) : 0;
      const uint32_t cv1     = enc_CxtVLC_table0[emb1 + (rho1 << 4) + (context << 8)];
      const int32_t embk1    = static_cast<int32_t>(cv1 & 0xF);
      const uint32_t lw1     = (cv1 >> 4) & 0x07;
      const uint32_t uvlc    = enc_UVLC_table0[u0 + (u1 << 5)];
      const uint32_t uvlc_lw = uvlc & 0xFF;
      // VLC codewords of both quads, then their UVLC codeword
      VLC_encoder.emitVLCBits((cv0 >> 7) | (static_cast<uint64_t>(cv1 >> 7) << lw0)
                                  | (static_cast<uint64_t>(uvlc >> 8) << (lw0 + lw1)),
                              lw0 + lw1 + uvlc_lw);
      // MEL encoding of the second quad
      const int32_t u_min = std::min(u0, u1);
      if (context == 0) {
        MEL_encoder.encodeMEL((rho1 != 0 || u_min > 2) ? 1 : 0);
      } else if (u0 && u1) {
        MEL_encoder.encodeMEL(u_min > 2 ? 1 : 0);
      }
      append_quad(sc.flat_v + fc, sc.flat_m + fc, v0, rho0, U0, embk0, emb0 & embk0);
      append_quad(sc.flat_v + fc + 4, sc.flat_m + fc + 4, v1, rho1, U1, embk1, emb1 & embk1);
      fc += 8;
      // context for the next quad
      context          = (rho1 >> 1) | (rho1 & 0x1);
      rho_old[q]       = rho0;
      rho_old[q + 1]   = rho1;
      E_old[2 * q]     = E0[1];
      E_old[2 * q + 1] = E0[3];
      E_old[2 * q + 2] = E1[1];
      E_old[2 * q + 3] = E1[3];
    }
    if (q < QW) {
      int32_t rho0, Emax0;
      load_quad(sp0 + 2 * q, sp1 + 2 * q, v0, E0, rho0, Emax0);
      if (context == 0) MEL_encoder.encodeMEL(rho0 != 0);
      const int32_t U0       = std::max(Emax0, 1);
      const int32_t u0       = U0 - 1;
      const int32_t emb0     = u0 ? emb_pattern(E0, Emax0) : 0;
      const uint32_t cv0     = enc_CxtVLC_table0[emb0 + (rho0 << 4) + (context << 8)];
      const int32_t embk0    = static_cast<int32_t>(cv0 & 0xF);
      const uint32_t lw0     = (cv0 >> 4) & 0x07;
      const uint32_t uvlc    = enc_UVLC_table0[u0];
      const uint32_t uvlc_lw = uvlc & 0xFF;
      VLC_encoder.emitVLCBits((cv0 >> 7) | (static_cast<uint64_t>(uvlc >> 8) << lw0), lw0 + uvlc_lw);
      append_quad(sc.flat_v + fc, sc.flat_m + fc, v0, rho0, U0, embk0, emb0 & embk0);
      fc += 4;
      rho_old[q]       = rho0;
      E_old[2 * q]     = E0[1];
      E_old[2 * q + 1] = E0[3];
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
      int32_t rho, Emax_q;
      load_quad(sp0 + 2 * q, sp1 + 2 * q, v0, E0, rho, Emax_q);
      // (w | sw) << 9, (nw | n) << 8, (ne | nf) << 10
      const int32_t ctx = ((rho_west & 0x4) << 7) | ((rho_west & 0x8) << 6) | ((rho_old[q - 1] & 0x8) << 5)
                          | ((rho_old[q] & 0xa) << 7) | ((rho_old[q + 1] & 0x2) << 9);
      const int32_t Emax_line =
          find_max(E_old[2 * q - 1], E_old[2 * q], E_old[2 * q + 1], E_old[2 * q + 2]);
      // gamma = (popcount(rho) > 1)
      const int32_t kappa = (rho & (rho - 1)) ? std::max(Emax_line - 1, 1) : 1;
      const int32_t U     = std::max(Emax_q, kappa);
      const int32_t uq    = U - kappa;
      const int32_t emb   = uq ? emb_pattern(E0, Emax_q) : 0;
      const uint32_t cv   = enc_CxtVLC_table1[emb + (rho << 4) + ctx];
      const int32_t embk  = static_cast<int32_t>(cv & 0xF);
      append_quad(sc.flat_v + 4 * q, sc.flat_m + 4 * q, v0, rho, U, embk, emb & embk);
      sc.ctx[q]        = ctx;
      sc.rho[q]        = rho;
      sc.uq[q]         = uq;
      sc.cwd[q]        = cv >> 7;
      sc.lw[q]         = (cv >> 4) & 0x07;
      E_new[2 * q]     = E0[1];
      E_new[2 * q + 1] = E0[3];
      rho_new[q]       = rho;
      rho_west         = rho;
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
 * HT encoding
 *******************************************************************************/
int32_t htj2k_encode(j2k_codeblock *block, uint8_t ROIshift) noexcept {
  htj2k_cleanup_encode(block, ROIshift);
  return EXIT_SUCCESS;
}
#endif
