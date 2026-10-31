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

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>
#include "coding_units.hpp"
#include "coding_local.hpp"
#include "mq_encoder.hpp"
#include "ht_rate_control.hpp"
#include "EBCOTtables.hpp"
#include "ebcot_context.hpp"
#include "ebcot_sig_lut9.hpp"

// Part 1 (EBCOT) block encoder: the significance propagation, magnitude
// refinement and cleanup passes of ISO/IEC 15444-1 Annex D on the packed
// stripe-column state of ebcot_context.hpp, mirroring block_decoding.cpp pass
// for pass.  All six code-block mode switches of Table A.19 are supported:
// BYPASS (raw significance propagation and magnitude refinement passes from the
// fifth bit-plane, D.6), RESET (context states reset at every pass, D.4),
// RESTART (every pass terminated, D.4), CAUSAL (vertically causal contexts,
// D.7), ERTERM (predictable termination, D.4.2 and D.6) and SEGMARK (a
// segmentation symbol after every cleanup pass, D.5).
//
// Input: block->sample_buf in the HT encoder's sample format produced by
// j2k_codeblock::quantize() (bit 31 set for a non-zero sample, whose quantised
// magnitude m is stored as ((m - 1) << 1) | sign).  It is rewritten in place as
// m | sign << 31 before coding.
#if defined(_MSC_VER)
  #define EBCOT_ENC_NOINLINE __declspec(noinline)
  #define EBCOT_ENC_INLINE __forceinline
#else
  #define EBCOT_ENC_NOINLINE __attribute__((noinline))
  #define EBCOT_ENC_INLINE inline __attribute__((always_inline))
#endif

namespace {

constexpr uint8_t LABEL_RUN = 17;
constexpr uint8_t LABEL_UNI = 18;
// Most bytes a code-block's codeword segments may total: set_compressed_data()
// takes a 16-bit length.  A 4096-sample block with 31 magnitude bit-planes
// codes about 17 KB; this leaves ample room for the MQ coder's worst case and
// the extra bytes of per-pass terminations.
constexpr uint32_t MAX_SEGMENT_BYTES = 65535;
// With BYPASS, passes from this index on (the fifth bit-plane's significance
// propagation pass) code their significance propagation and magnitude
// refinement passes raw.
constexpr uint32_t BYPASS_FIRST_RAW_PASS = 10;

struct enc_setup {
  uint32_t *ctx;
  size_t ctx_stride;
  const int32_t *samples;
  size_t sample_stride;
  uint32_t width;
  uint32_t height;
  const uint8_t *sig_lut;
};

// Causal selects vertically causal context formation (the CAUSAL mode switch).
// It is a template parameter so that the context lookups of the other modes
// carry no run-time test for it.
//
// The coded symbols of the significance propagation and magnitude refinement
// passes, through the MQ coder or, in the raw passes of BYPASS, straight into
// the bit stream (the sign is coded as is, D-2).
EBCOT_ENC_INLINE void code_significance(mq_encoder &mq, const enc_setup &st, uint32_t nb, uint32_t bit) {
  mq.encode(st.sig_lut[nb], bit);
}
EBCOT_ENC_INLINE void code_significance(raw_encoder &raw, const enc_setup &, uint32_t, uint32_t bit) {
  raw.put(bit);
}
template <bool Causal>
EBCOT_ENC_INLINE void code_sign(mq_encoder &mq, const uint32_t *wp, uint32_t w, uint32_t i, uint32_t sign) {
  const uint32_t idx = ebcot::sign_context_index(w, wp[-1], wp[1], i, Causal);
  mq.encode(sign_LUT[0][idx], sign ^ sign_LUT[1][idx]);
}
template <bool Causal>
EBCOT_ENC_INLINE void code_sign(raw_encoder &raw, const uint32_t *, uint32_t, uint32_t, uint32_t sign) {
  raw.put(sign);
}
template <bool Causal>
EBCOT_ENC_INLINE void code_refinement(mq_encoder &mq, uint32_t w, uint32_t i, uint32_t bit) {
  uint8_t label = 16;  // later refinements
  if ((w & ebcot::sigma_bar_bit(i)) == 0) {
    label = (ebcot::sigma_neighbourhood(w, i, Causal) & ~0x10u) ? 15 : 14;
  }
  mq.encode(label, bit);
}
template <bool Causal>
EBCOT_ENC_INLINE void code_refinement(raw_encoder &raw, uint32_t, uint32_t, uint32_t bit) {
  raw.put(bit);
}

// Rate control (the Rd template parameter): the decrease of a sample's squared error, in
// doubled units ((2m + 1) is twice the true value m + 1/2), when it becomes significant at
// bit-plane p (from 0 to the mid-point 3 * 2^p of its interval) and when its bit p is
// refined (from the mid-point at p + 1 to the one at p).  m < 2^30, so the terms fit.
EBCOT_ENC_INLINE double gain_significant(uint32_t m, uint32_t p) {
  const int64_t t = 2 * static_cast<int64_t>(m) + 1;
  const int64_t e = t - (static_cast<int64_t>(3) << p);
  return static_cast<double>(t * t - e * e);
}
EBCOT_ENC_INLINE double gain_refined(uint32_t m, uint32_t p) {
  const int64_t t  = 2 * static_cast<int64_t>(m) + 1;
  const int64_t eb = t - ((2 * static_cast<int64_t>(m >> (p + 1)) + 1) << (p + 1));
  const int64_t ea = t - ((2 * static_cast<int64_t>(m >> p) + 1) << p);
  return static_cast<double>(eb * eb - ea * ea);
}

// Stripe row i of the column at wp becomes significant in this bit-plane: codes
// its sign, sets σ and χ in the column's register word w and updates the
// neighbouring words.
template <bool Causal, class Coder>
EBCOT_ENC_INLINE void encode_new_significance(Coder &coder, const enc_setup &st, uint32_t *wp, uint32_t &w,
                                              uint32_t i, uint32_t sign) {
  code_sign<Causal>(coder, wp, w, i, sign);
  w |= ebcot::sigma_bit(i) | (sign << (3 * i + 19));
  ebcot::broadcast_significance(wp, st.ctx_stride, i, sign);
}

// One sample of the significance propagation pass: coded if insignificant with a
// significant neighbour.
template <bool Causal, bool Rd, class Coder>
EBCOT_ENC_INLINE void sigprop_sample(Coder &coder, const enc_setup &st, uint32_t *wp, uint32_t &w,
                                     const int32_t *col, uint32_t i, uint32_t p, double &dd) {
  if (w & ebcot::sigma_bit(i)) return;
  const uint32_t nb = ebcot::sigma_neighbourhood(w, i, Causal);
  if (nb == 0) return;
  w |= ebcot::pi_bit(i);
  const uint32_t s   = static_cast<uint32_t>(col[i * st.sample_stride]);
  const uint32_t bit = (s >> p) & 1u;
  code_significance(coder, st, nb, bit);
  if (bit) {
    encode_new_significance<Causal>(coder, st, wp, w, i, s >> 31);
    if (Rd) dd += gain_significant(s & 0x7FFFFFFFu, p);
  }
}

// One sample of the magnitude refinement pass: coded if it was significant before
// this bit-plane (σ = 1, π = 0).
template <bool Causal, bool Rd, class Coder>
EBCOT_ENC_INLINE void magref_sample(Coder &coder, const enc_setup &st, uint32_t &w, uint32_t todo,
                                    const int32_t *col, uint32_t i, uint32_t p, double &dd) {
  if ((todo & ebcot::sigma_bit(i)) == 0) return;
  const uint32_t s = static_cast<uint32_t>(col[i * st.sample_stride]);
  code_refinement<Causal>(coder, w, i, (s >> p) & 1u);
  w |= ebcot::sigma_bar_bit(i);
  if (Rd) dd += gain_refined(s & 0x7FFFFFFFu, p);
}

// One sample of the cleanup pass: coded if neither significant nor coded by this
// bit-plane's significance propagation pass.
template <bool Causal, bool Rd>
EBCOT_ENC_INLINE void cleanup_sample(mq_encoder &mq, const enc_setup &st, uint32_t *wp, uint32_t &w,
                                     const int32_t *col, uint32_t i, uint32_t p, double &dd) {
  if (w & (ebcot::sigma_bit(i) | ebcot::pi_bit(i))) return;
  const uint32_t s   = static_cast<uint32_t>(col[i * st.sample_stride]);
  const uint32_t bit = (s >> p) & 1u;
  mq.encode(st.sig_lut[ebcot::sigma_neighbourhood(w, i, Causal)], bit);
  if (bit) {
    encode_new_significance<Causal>(mq, st, wp, w, i, s >> 31);
    if (Rd) dd += gain_significant(s & 0x7FFFFFFFu, p);
  }
}

// Each pass walks the stripes; a full stripe (4 rows) is unrolled so that every
// row's bit positions are constants, and a final partial stripe takes a loop.
template <bool Causal, bool Rd, class Coder, bool Full>
EBCOT_ENC_INLINE void sigprop_stripe(Coder &coder, const enc_setup &st, uint32_t *wp, const int32_t *sp,
                                     uint32_t rows, uint32_t p, double &dd) {
  for (uint32_t x = 0; x < st.width; ++x, ++wp) {
    uint32_t w = *wp;
    if ((w & ebcot::SIGMA_WINDOW) == 0) continue;  // no significant neighbour: nothing to code
    const int32_t *col = sp + x;
    if (Full) {
      sigprop_sample<Causal, Rd>(coder, st, wp, w, col, 0, p, dd);
      sigprop_sample<Causal, Rd>(coder, st, wp, w, col, 1, p, dd);
      sigprop_sample<Causal, Rd>(coder, st, wp, w, col, 2, p, dd);
      sigprop_sample<Causal, Rd>(coder, st, wp, w, col, 3, p, dd);
    } else {
      for (uint32_t i = 0; i < rows; ++i) sigprop_sample<Causal, Rd>(coder, st, wp, w, col, i, p, dd);
    }
    *wp = w;
  }
}

// The passes return the decrease of the squared error they make with Rd, 0 otherwise.
template <bool Causal, bool Rd, class Coder>
EBCOT_ENC_NOINLINE double encode_sigprop(const enc_setup &st, uint32_t p, Coder &coder_io) {
  Coder coder = coder_io;  // the coder state stays in registers for the whole pass
  double dd   = 0;
  for (uint32_t y0 = 0, s = 0; y0 < st.height; y0 += 4, ++s) {
    uint32_t *wp      = st.ctx + (s + 1) * st.ctx_stride + 1;
    const int32_t *sp = st.samples + y0 * st.sample_stride;
    if (st.height - y0 >= 4)
      sigprop_stripe<Causal, Rd, Coder, true>(coder, st, wp, sp, 4, p, dd);
    else
      sigprop_stripe<Causal, Rd, Coder, false>(coder, st, wp, sp, st.height - y0, p, dd);
  }
  coder_io = coder;
  return dd;
}

template <bool Causal, bool Rd, class Coder, bool Full>
EBCOT_ENC_INLINE void magref_stripe(Coder &coder, const enc_setup &st, uint32_t *wp, const int32_t *sp,
                                    uint32_t rows, uint32_t p, double &dd) {
  for (uint32_t x = 0; x < st.width; ++x, ++wp) {
    uint32_t w = *wp;
    // Samples that were significant before this bit-plane: σ = 1 and π = 0.
    const uint32_t todo = w & ebcot::SIGMA_SELF_ALL & ~(w >> ebcot::PI_TO_SIGMA_SHIFT);
    if (todo == 0) continue;
    const int32_t *col = sp + x;
    if (Full) {
      magref_sample<Causal, Rd>(coder, st, w, todo, col, 0, p, dd);
      magref_sample<Causal, Rd>(coder, st, w, todo, col, 1, p, dd);
      magref_sample<Causal, Rd>(coder, st, w, todo, col, 2, p, dd);
      magref_sample<Causal, Rd>(coder, st, w, todo, col, 3, p, dd);
    } else {
      for (uint32_t i = 0; i < rows; ++i) magref_sample<Causal, Rd>(coder, st, w, todo, col, i, p, dd);
    }
    *wp = w;
  }
}

template <bool Causal, bool Rd, class Coder>
EBCOT_ENC_NOINLINE double encode_magref(const enc_setup &st, uint32_t p, Coder &coder_io) {
  Coder coder = coder_io;
  double dd   = 0;
  for (uint32_t y0 = 0, s = 0; y0 < st.height; y0 += 4, ++s) {
    uint32_t *wp      = st.ctx + (s + 1) * st.ctx_stride + 1;
    const int32_t *sp = st.samples + y0 * st.sample_stride;
    if (st.height - y0 >= 4)
      magref_stripe<Causal, Rd, Coder, true>(coder, st, wp, sp, 4, p, dd);
    else
      magref_stripe<Causal, Rd, Coder, false>(coder, st, wp, sp, st.height - y0, p, dd);
  }
  coder_io = coder;
  return dd;
}

template <bool Causal, bool Rd>
EBCOT_ENC_NOINLINE double encode_cleanup(const enc_setup &st, uint32_t p, mq_encoder &mq_io) {
  // Run mode needs all four samples' neighbourhoods insignificant; a vertically
  // causal neighbourhood does not see the row below the stripe.
  constexpr uint32_t run_mask = Causal ? ebcot::SIGMA_RUN_CAUSAL : ebcot::SIGMA_WINDOW;
  mq_encoder mq               = mq_io;
  double dd                   = 0;
  for (uint32_t y0 = 0, s = 0; y0 < st.height; y0 += 4, ++s) {
    uint32_t *wp      = st.ctx + (s + 1) * st.ctx_stride + 1;
    const int32_t *sp = st.samples + y0 * st.sample_stride;
    if (st.height - y0 >= 4) {
      for (uint32_t x = 0; x < st.width; ++x, ++wp) {
        uint32_t w         = *wp;
        const int32_t *col = sp + x;
        uint32_t r         = 0;  // first row still to code
        if ((w & run_mask) == 0) {
          // Run mode: all four samples are insignificant with insignificant
          // neighbours.  Code whether any of them becomes significant and, if so,
          // the index of the first one (two UNIFORM bits, MSB first).
          uint32_t first = 4;
          for (uint32_t i = 0; i < 4; ++i) {
            if ((static_cast<uint32_t>(col[i * st.sample_stride]) >> p) & 1u) {
              first = i;
              break;
            }
          }
          if (first == 4) {
            mq.encode(LABEL_RUN, 0);
            continue;  // all four stay insignificant; the word is unchanged
          }
          mq.encode(LABEL_RUN, 1);
          mq.encode(LABEL_UNI, first >> 1);
          mq.encode(LABEL_UNI, first & 1u);
          const uint32_t sf = static_cast<uint32_t>(col[first * st.sample_stride]);
          encode_new_significance<Causal>(mq, st, wp, w, first, sf >> 31);
          if (Rd) dd += gain_significant(sf & 0x7FFFFFFFu, p);
          r = first + 1;
        }
        if (r == 0) cleanup_sample<Causal, Rd>(mq, st, wp, w, col, 0, p, dd);
        if (r <= 1) cleanup_sample<Causal, Rd>(mq, st, wp, w, col, 1, p, dd);
        if (r <= 2) cleanup_sample<Causal, Rd>(mq, st, wp, w, col, 2, p, dd);
        if (r <= 3) cleanup_sample<Causal, Rd>(mq, st, wp, w, col, 3, p, dd);
        *wp = w & ~ebcot::PI_ALL;
      }
    } else {
      const uint32_t rows = st.height - y0;
      for (uint32_t x = 0; x < st.width; ++x, ++wp) {
        uint32_t w         = *wp;
        const int32_t *col = sp + x;
        for (uint32_t i = 0; i < rows; ++i) cleanup_sample<Causal, Rd>(mq, st, wp, w, col, i, p, dd);
        *wp = w & ~ebcot::PI_ALL;
      }
    }
  }
  mq_io = mq;
  return dd;
}

// Codes pass k (0 SPP, 1 MRP, 2 cleanup) of bit-plane p; returns the decrease of the
// squared error with Rd.
template <bool Causal, bool Rd>
double encode_pass(const enc_setup &st, uint32_t k, uint32_t p, bool is_raw, mq_encoder &mq,
                   raw_encoder &raw) {
  if (k == 0)
    return is_raw ? encode_sigprop<Causal, Rd>(st, p, raw) : encode_sigprop<Causal, Rd>(st, p, mq);
  if (k == 1) return is_raw ? encode_magref<Causal, Rd>(st, p, raw) : encode_magref<Causal, Rd>(st, p, mq);
  return encode_cleanup<Causal, Rd>(st, p, mq);
}

}  // namespace

int32_t ebcot_encode(j2k_codeblock *const block, const uint8_t ROIshift) noexcept {
  if (ROIshift) {
    printf("WARNING: Encoding with ROI is not supported.\n");
  }
  uint32_t or_val = 0;
  if (!block->pre_quantized)
    block->quantize(or_val);
  else
    or_val = block->pre_or_val;

  if (!or_val) {
    // Empty code-block: no passes, not included in any layer.
    block->length          = 0;
    block->pass_length[0]  = 0;
    block->num_passes      = 0;
    block->layer_passes[0] = 0;
    block->layer_start[0]  = 0;
    block->num_ZBP         = static_cast<uint8_t>(block->get_Mb() - 1);
    return 0;
  }

  // Rewrite the HT sample format as magnitude | sign << 31 and find the number of
  // magnitude bit-planes actually used.
  const uint32_t width  = block->size.x;
  const uint32_t height = block->size.y;
  const size_t stride   = block->blksampl_stride;
  ht_rd_info *const rd  = block->rd;  // rate control: record every pass as a truncation point
  uint32_t or_mag       = 0;
  double D              = 0;  // with rd: the squared error left (doubled units, as ht_rd_plane)
  for (uint32_t y = 0; y < height; ++y) {
    int32_t *sp = block->sample_buf + y * stride;
    for (uint32_t x = 0; x < width; ++x) {
      const uint32_t s = static_cast<uint32_t>(sp[x]);
      uint32_t out     = 0;
      if (s & 0x80000000u) {
        const uint32_t v = s & 0x7FFFFFFFu;
        const uint32_t m = (v >> 1) + 1;
        out              = m | ((v & 1u) << 31);
        or_mag |= m;
        if (rd != nullptr) {
          const int64_t t = 2 * static_cast<int64_t>(m) + 1;
          D += static_cast<double>(t * t);
        }
      }
      sp[x] = static_cast<int32_t>(out);
    }
  }
  uint32_t msb_len  = 32 - count_leading_zeros(or_mag);
  const uint32_t Mb = block->get_Mb();
  if (msb_len > Mb) {
    // Not enough guard bits for this block's magnitudes; the stream would be
    // undecodable.  Clamp rather than index past M_b (same assumption as HT).
    printf("WARNING: code-block magnitudes need %u bit-planes but M_b is %u; increase Qguard.\n", msb_len,
           Mb);
    msb_len = Mb;
  }

  // Packed stripe-column state: one word per column with a one-word border left
  // and right and a one-stripe border above and below, all zero.
  const size_t ctx_stride = block->block_contexts_stride;
  const size_t nwords     = (static_cast<size_t>((height + 3) / 4) + 2) * ctx_stride;
  static thread_local std::vector<uint32_t> ctx_words;
  if (ctx_words.size() < nwords) ctx_words.resize(nwords);
  std::memset(ctx_words.data(), 0, nwords * sizeof(uint32_t));
  static thread_local uint8_t out_buf[MAX_SEGMENT_BYTES + 16];

  const enc_setup st   = {ctx_words.data(),
                          ctx_stride,
                          block->sample_buf,
                          stride,
                          width,
                          height,
                          sig_LUT9_table.data[block->get_orientation()]};
  const uint32_t modes = block->Cmodes;
  const bool causal    = (modes & CAUSAL) != 0;
  const bool bypass    = (modes & BYPASS) != 0;
  const bool reset     = (modes & RESET) != 0;
  const bool restart   = (modes & RESTART) != 0;
  const bool erterm    = (modes & ERTERM) != 0;
  const bool segmark   = (modes & SEGMARK) != 0;

  // The most significant bit-plane has a cleanup pass only.  With rate control the
  // passes stop at the cleanup pass of bit-plane rd->p_coded: the cut never reaches below.
  const uint32_t p_low      = (rd != nullptr) ? std::min<uint32_t>(rd->p_coded, msb_len - 1) : 0;
  const uint32_t num_passes = 3 * (msb_len - p_low) - 2;
  block->ensure_pass_capacity(static_cast<uint8_t>(num_passes));
  if (rd != nullptr) {
    rd->D2zero    = D;
    rd->p1        = MAKE_UNIQUE<p1_rd_pass[]>(num_passes);
    rd->p1_passes = static_cast<uint8_t>(num_passes);
    rd->p1_zbp    = static_cast<uint8_t>(Mb - msb_len);
  }
  p1_prefix_probe *const probes = (rd != nullptr && rd->p1_prefix) ? p1_probe_buffer(num_passes) : nullptr;
  mq_encoder mq(out_buf);
  mq.init_states_for_all_contexts();
  raw_encoder raw;
  uint32_t len      = 0;     // bytes of the segments terminated so far
  bool segment_open = true;  // the MQ coder above has started the first segment
  uint32_t z        = 0;     // pass index
  for (uint32_t p = msb_len; p-- > p_low;) {
    for (uint32_t k = (p == msb_len - 1) ? 2 : 0; k < 3; ++k, ++z) {  // 0 SPP, 1 MRP, 2 cleanup
      const bool is_raw = bypass && k < 2 && z >= BYPASS_FIRST_RAW_PASS;
      if (!segment_open) {
        if (is_raw)
          raw.init(out_buf + len);
        else
          mq.init(out_buf + len);
        segment_open = true;
      }
      if (reset && z > 0) mq.init_states_for_all_contexts();
      if (rd != nullptr)
        D -= causal ? encode_pass<true, true>(st, k, p, is_raw, mq, raw)
                    : encode_pass<false, true>(st, k, p, is_raw, mq, raw);
      else if (causal)
        encode_pass<true, false>(st, k, p, is_raw, mq, raw);
      else
        encode_pass<false, false>(st, k, p, is_raw, mq, raw);
      if (k == 2 && segmark) {  // the segmentation symbol 1010 (D.5)
        mq.encode(LABEL_UNI, 1);
        mq.encode(LABEL_UNI, 0);
        mq.encode(LABEL_UNI, 1);
        mq.encode(LABEL_UNI, 0);
      }

      // Codeword segment boundaries (Tables D.8 and D.9): every pass with
      // RESTART; with BYPASS, the fourth cleanup pass, each raw magnitude
      // refinement pass and each later cleanup pass; and always the last pass.
      const bool terminate =
          restart || z == num_passes - 1
          || (bypass && (z == BYPASS_FIRST_RAW_PASS - 1 || (z >= BYPASS_FIRST_RAW_PASS && k != 0)));
      uint32_t seg_len = 0;
      if (terminate) {
        if (is_raw)
          seg_len = raw.terminate(erterm);
        else
          seg_len = erterm ? mq.erterm() : mq.flush();
        len += seg_len;
        segment_open = false;
        if (len > MAX_SEGMENT_BYTES) {
          printf("ERROR: Part 1 code-block of %u bytes exceeds the 16-bit limit.\n", len);
          return -1;
        }
      }
      // A segment's length sits on its last pass; the packet header signals the
      // sum over each segment's passes.
      block->pass_length[z] = seg_len;

      // Rate control: where a cut after this pass ends.  A terminated pass ends its
      // segment; otherwise a copy of the coder terminates the segment here, into the
      // bytes the real coder has yet to write, and the bytes it changes are kept.
      if (rd != nullptr) {
        p1_rd_pass &pt  = rd->p1[z];
        pt.D            = D;
        pt.ends_segment = terminate;
        if (terminate) {
          pt.keep     = len;
          pt.tail_len = 0;
        } else {
          uint8_t *seg     = out_buf + len;
          uint32_t settled = 0, end = 0;
          if (is_raw) {
            raw_encoder t = raw;
            settled       = raw.settled_bytes();
            end           = t.terminate(erterm);
          } else {
            mq_encoder t = mq;
            settled      = mq.settled_bytes();
            end          = erterm ? t.erterm() : t.flush();
          }
          settled     = std::min(settled, end);
          pt.keep     = len + settled;
          pt.tail_len = static_cast<uint8_t>(end - settled);
          if (pt.tail_len > sizeof(pt.tail)) {
            printf("ERROR: Part 1 rate control: a %u-byte termination.\n", pt.tail_len);
            return -1;
          }
          std::memcpy(pt.tail, seg + settled, pt.tail_len);
          if (probes != nullptr && !p1_take_probe(probes[z], len, seg, is_raw, mq, raw)) return -1;
        }
      }
    }
  }

  if (probes != nullptr) p1_find_prefixes(rd->p1.get(), probes, num_passes, out_buf, len);

  block->pass_length_count = static_cast<uint8_t>(num_passes);
  block->set_compressed_data(out_buf, static_cast<uint16_t>(len), 0);
  block->length          = len;
  block->num_passes      = static_cast<uint8_t>(num_passes);
  block->layer_passes[0] = static_cast<uint8_t>(num_passes);
  block->layer_start[0]  = 0;
  block->num_ZBP         = static_cast<uint8_t>(Mb - msb_len);
  return static_cast<int32_t>(len);
}
