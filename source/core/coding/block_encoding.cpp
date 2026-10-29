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

#include <cstdio>
#include <cstring>
#include <vector>
#include "coding_units.hpp"
#include "coding_local.hpp"
#include "mq_encoder.hpp"
#include "EBCOTtables.hpp"
#include "ebcot_context.hpp"
#include "ebcot_sig_lut9.hpp"

// Part 1 (EBCOT) block encoder: the significance propagation, magnitude
// refinement and cleanup passes of ISO/IEC 15444-1 Annex D on the packed
// stripe-column state of ebcot_context.hpp, mirroring block_decoding.cpp pass
// for pass.  Default mode only: one MQ codeword segment for all passes, no
// BYPASS, RESET, RESTART, CAUSAL, ERTERM or SEGMARK.
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
// Largest MQ codeword segment a code-block may produce: set_compressed_data()
// takes a 16-bit length.  A 4096-sample block with 31 magnitude bit-planes
// codes about 17 KB; this leaves ample room for the MQ coder's worst case.
constexpr uint32_t MAX_SEGMENT_BYTES = 65535;

struct enc_setup {
  uint32_t *ctx;
  size_t ctx_stride;
  const int32_t *samples;
  size_t sample_stride;
  uint32_t width;
  uint32_t height;
  const uint8_t *sig_lut;
};

// Stripe row i of the column at wp becomes significant in this bit-plane: codes
// its sign, sets σ and χ in the column's register word w and updates the
// neighbouring words.
EBCOT_ENC_INLINE void encode_new_significance(mq_encoder &mq, const enc_setup &st, uint32_t *wp,
                                              uint32_t &w, uint32_t i, uint32_t sign) {
  const uint32_t idx = ebcot::sign_context_index(w, wp[-1], wp[1], i, false);
  mq.encode(sign_LUT[0][idx], sign ^ sign_LUT[1][idx]);
  w |= ebcot::sigma_bit(i) | (sign << (3 * i + 19));
  ebcot::broadcast_significance(wp, st.ctx_stride, i, sign);
}

// One sample of the significance propagation pass: coded if insignificant with a
// significant neighbour.
EBCOT_ENC_INLINE void sigprop_sample(mq_encoder &mq, const enc_setup &st, uint32_t *wp, uint32_t &w,
                                     const int32_t *col, uint32_t i, uint32_t p) {
  if (w & ebcot::sigma_bit(i)) return;
  const uint32_t nb = ebcot::sigma_neighbourhood(w, i, false);
  if (nb == 0) return;
  w |= ebcot::pi_bit(i);
  const uint32_t s   = static_cast<uint32_t>(col[i * st.sample_stride]);
  const uint32_t bit = (s >> p) & 1u;
  mq.encode(st.sig_lut[nb], bit);
  if (bit) encode_new_significance(mq, st, wp, w, i, s >> 31);
}

// One sample of the magnitude refinement pass: coded if it was significant before
// this bit-plane (σ = 1, π = 0).
EBCOT_ENC_INLINE void magref_sample(mq_encoder &mq, const enc_setup &st, uint32_t &w, uint32_t todo,
                                    const int32_t *col, uint32_t i, uint32_t p) {
  if ((todo & ebcot::sigma_bit(i)) == 0) return;
  uint8_t label = 16;  // later refinements
  if ((w & ebcot::sigma_bar_bit(i)) == 0) {
    label = (ebcot::sigma_neighbourhood(w, i, false) & ~0x10u) ? 15 : 14;
  }
  mq.encode(label, (static_cast<uint32_t>(col[i * st.sample_stride]) >> p) & 1u);
  w |= ebcot::sigma_bar_bit(i);
}

// One sample of the cleanup pass: coded if neither significant nor coded by this
// bit-plane's significance propagation pass.
EBCOT_ENC_INLINE void cleanup_sample(mq_encoder &mq, const enc_setup &st, uint32_t *wp, uint32_t &w,
                                     const int32_t *col, uint32_t i, uint32_t p) {
  if (w & (ebcot::sigma_bit(i) | ebcot::pi_bit(i))) return;
  const uint32_t s   = static_cast<uint32_t>(col[i * st.sample_stride]);
  const uint32_t bit = (s >> p) & 1u;
  mq.encode(st.sig_lut[ebcot::sigma_neighbourhood(w, i, false)], bit);
  if (bit) encode_new_significance(mq, st, wp, w, i, s >> 31);
}

// Each pass walks the stripes; a full stripe (4 rows) is unrolled so that every
// row's bit positions are constants, and a final partial stripe takes a loop.
template <bool Full>
EBCOT_ENC_INLINE void sigprop_stripe(mq_encoder &mq, const enc_setup &st, uint32_t *wp, const int32_t *sp,
                                     uint32_t rows, uint32_t p) {
  for (uint32_t x = 0; x < st.width; ++x, ++wp) {
    uint32_t w = *wp;
    if ((w & ebcot::SIGMA_WINDOW) == 0) continue;  // no significant neighbour: nothing to code
    const int32_t *col = sp + x;
    if (Full) {
      sigprop_sample(mq, st, wp, w, col, 0, p);
      sigprop_sample(mq, st, wp, w, col, 1, p);
      sigprop_sample(mq, st, wp, w, col, 2, p);
      sigprop_sample(mq, st, wp, w, col, 3, p);
    } else {
      for (uint32_t i = 0; i < rows; ++i) sigprop_sample(mq, st, wp, w, col, i, p);
    }
    *wp = w;
  }
}

EBCOT_ENC_NOINLINE void encode_sigprop(const enc_setup &st, uint32_t p, mq_encoder &mq_io) {
  mq_encoder mq = mq_io;  // the coder state stays in registers for the whole pass
  for (uint32_t y0 = 0, s = 0; y0 < st.height; y0 += 4, ++s) {
    uint32_t *wp      = st.ctx + (s + 1) * st.ctx_stride + 1;
    const int32_t *sp = st.samples + y0 * st.sample_stride;
    if (st.height - y0 >= 4)
      sigprop_stripe<true>(mq, st, wp, sp, 4, p);
    else
      sigprop_stripe<false>(mq, st, wp, sp, st.height - y0, p);
  }
  mq_io = mq;
}

template <bool Full>
EBCOT_ENC_INLINE void magref_stripe(mq_encoder &mq, const enc_setup &st, uint32_t *wp, const int32_t *sp,
                                    uint32_t rows, uint32_t p) {
  for (uint32_t x = 0; x < st.width; ++x, ++wp) {
    uint32_t w = *wp;
    // Samples that were significant before this bit-plane: σ = 1 and π = 0.
    const uint32_t todo = w & ebcot::SIGMA_SELF_ALL & ~(w >> ebcot::PI_TO_SIGMA_SHIFT);
    if (todo == 0) continue;
    const int32_t *col = sp + x;
    if (Full) {
      magref_sample(mq, st, w, todo, col, 0, p);
      magref_sample(mq, st, w, todo, col, 1, p);
      magref_sample(mq, st, w, todo, col, 2, p);
      magref_sample(mq, st, w, todo, col, 3, p);
    } else {
      for (uint32_t i = 0; i < rows; ++i) magref_sample(mq, st, w, todo, col, i, p);
    }
    *wp = w;
  }
}

EBCOT_ENC_NOINLINE void encode_magref(const enc_setup &st, uint32_t p, mq_encoder &mq_io) {
  mq_encoder mq = mq_io;
  for (uint32_t y0 = 0, s = 0; y0 < st.height; y0 += 4, ++s) {
    uint32_t *wp      = st.ctx + (s + 1) * st.ctx_stride + 1;
    const int32_t *sp = st.samples + y0 * st.sample_stride;
    if (st.height - y0 >= 4)
      magref_stripe<true>(mq, st, wp, sp, 4, p);
    else
      magref_stripe<false>(mq, st, wp, sp, st.height - y0, p);
  }
  mq_io = mq;
}

EBCOT_ENC_NOINLINE void encode_cleanup(const enc_setup &st, uint32_t p, mq_encoder &mq_io) {
  mq_encoder mq = mq_io;
  for (uint32_t y0 = 0, s = 0; y0 < st.height; y0 += 4, ++s) {
    uint32_t *wp      = st.ctx + (s + 1) * st.ctx_stride + 1;
    const int32_t *sp = st.samples + y0 * st.sample_stride;
    if (st.height - y0 >= 4) {
      for (uint32_t x = 0; x < st.width; ++x, ++wp) {
        uint32_t w         = *wp;
        const int32_t *col = sp + x;
        uint32_t r         = 0;  // first row still to code
        if ((w & ebcot::SIGMA_WINDOW) == 0) {
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
          encode_new_significance(mq, st, wp, w, first,
                                  static_cast<uint32_t>(col[first * st.sample_stride]) >> 31);
          r = first + 1;
        }
        if (r == 0) cleanup_sample(mq, st, wp, w, col, 0, p);
        if (r <= 1) cleanup_sample(mq, st, wp, w, col, 1, p);
        if (r <= 2) cleanup_sample(mq, st, wp, w, col, 2, p);
        if (r <= 3) cleanup_sample(mq, st, wp, w, col, 3, p);
        *wp = w & ~ebcot::PI_ALL;
      }
    } else {
      const uint32_t rows = st.height - y0;
      for (uint32_t x = 0; x < st.width; ++x, ++wp) {
        uint32_t w         = *wp;
        const int32_t *col = sp + x;
        for (uint32_t i = 0; i < rows; ++i) cleanup_sample(mq, st, wp, w, col, i, p);
        *wp = w & ~ebcot::PI_ALL;
      }
    }
  }
  mq_io = mq;
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
  uint32_t or_mag       = 0;
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

  const enc_setup st = {ctx_words.data(),
                        ctx_stride,
                        block->sample_buf,
                        stride,
                        width,
                        height,
                        sig_LUT9_table.data[block->get_orientation()]};
  mq_encoder mq(out_buf);
  mq.init_states_for_all_contexts();
  // The most significant bit-plane has a cleanup pass only.
  for (uint32_t p = msb_len; p-- > 0;) {
    if (p != msb_len - 1) {
      encode_sigprop(st, p, mq);
      encode_magref(st, p, mq);
    }
    encode_cleanup(st, p, mq);
  }
  const uint32_t len = mq.flush();
  if (len > MAX_SEGMENT_BYTES) {
    printf("ERROR: Part 1 code-block codeword segment of %u bytes exceeds the 16-bit limit.\n", len);
    return -1;
  }
  const uint8_t num_passes = static_cast<uint8_t>(3 * msb_len - 2);

  // One codeword segment for every pass: the packet header signals their total
  // length, so only pass_length[0] carries bytes.
  block->ensure_pass_capacity(num_passes);
  block->pass_length[0] = len;
  for (uint8_t i = 1; i < num_passes; ++i) block->pass_length[i] = 0;
  block->pass_length_count = num_passes;
  block->set_compressed_data(out_buf, static_cast<uint16_t>(len), 0);
  block->length          = len;
  block->num_passes      = num_passes;
  block->layer_passes[0] = num_passes;
  block->layer_start[0]  = 0;
  block->num_ZBP         = static_cast<uint8_t>(Mb - msb_len);
  return static_cast<int32_t>(len);
}
