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

#include "coding_units.hpp"
#include "mq_decoder.hpp"
#include "coding_local.hpp"
#include "EBCOTtables.hpp"
#include "ebcot_context.hpp"
#include "block_dequant.hpp"

// 9-bit-indexed σ-context LUT for the packed stripe-column word (book §17.1.2 figure 17.2).
// Built once at startup from the 8-bit sig_LUT by remapping the neighbourhood-bit layout:
//   8-bit sig_LUT:  TL=0 TR=1 BL=2 BR=3 T=4 B=5 L=6 R=7
//   9-bit window:   TL=0 T=1 TR=2 L=3 self=4 R=5 BL=6 B=7 BR=8
// The self bit (bit 4) is a don't-care since κ^sig is only queried when σ=0; we still
// populate all 512 entries so the index never needs masking.
namespace {
// clang-format off
struct SigLUT9Init {
  uint8_t data[4][512];
  SigLUT9Init() {
    for (uint32_t idx9 = 0; idx9 < 512; ++idx9) {
      const uint8_t idx8 = static_cast<uint8_t>(
          ((idx9 >> 0) & 1u)                 // TL bit 0 → bit 0
          | (((idx9 >> 2) & 1u) << 1)        // TR bit 2 → bit 1
          | (((idx9 >> 6) & 1u) << 2)        // BL bit 6 → bit 2
          | (((idx9 >> 8) & 1u) << 3)        // BR bit 8 → bit 3
          | (((idx9 >> 1) & 1u) << 4)        // T  bit 1 → bit 4
          | (((idx9 >> 7) & 1u) << 5)        // B  bit 7 → bit 5
          | (((idx9 >> 3) & 1u) << 6)        // L  bit 3 → bit 6
          | (((idx9 >> 5) & 1u) << 7));      // R  bit 5 → bit 7
      for (int o = 0; o < 4; ++o) data[o][idx9] = sig_LUT[o][idx8];
    }
  }
};
// clang-format on
const SigLUT9Init sig_LUT9_table;
}  // namespace

// The coding passes below keep all per-sample state in the packed stripe-column
// words (see ebcot_context.hpp).  Each pass loads a column's word once, works on a
// register copy, and stores it back after the column; neighbouring words are
// updated in memory as samples become significant.  Each pass is its own function
// with the MQ decoder copied into a local, so the coder state stays in registers.
//
// Reconstruction: when a sample is coded at bit-plane p, bit p-1 of its magnitude
// is set as a marker (half = 1 << (p - 1)); the next refinement moves it down.  For
// a significant sample this is exactly the mid-point that j2k_dequant adds, so no
// per-sample bit-plane index needs to be stored.  With an ROI shift, j2k_dequant
// does not use the decoded bit-plane, so half is 0 and no marker is set.
#if defined(_MSC_VER)
  #define EBCOT_PASS_NOINLINE __declspec(noinline)
  #define EBCOT_FORCE_INLINE __forceinline
#else
  #define EBCOT_PASS_NOINLINE __attribute__((noinline))
  #define EBCOT_FORCE_INLINE inline __attribute__((always_inline))
#endif

namespace {

struct pass_setup {
  uint32_t *ctx;
  size_t ctx_stride;
  int32_t *samples;
  size_t sample_stride;
  uint32_t width;
  uint32_t height;
  const uint8_t *sig_lut;
  bool causal;
  explicit pass_setup(const j2k_codeblock *block)
      : ctx(block->block_contexts),
        ctx_stride(block->block_contexts_stride),
        samples(block->sample_buf),
        sample_stride(block->blksampl_stride),
        width(block->size.x),
        height(block->size.y),
        sig_lut(sig_LUT9_table.data[block->get_orientation()]),
        causal((block->Cmodes & CAUSAL) != 0) {}
};

// Decodes the sign of stripe row i of the column at wp and records the sample as
// significant: sets bit p, the marker and the sign in the sample (one | half), and
// updates the column's register word w and the neighbouring words.
template <bool Raw>
EBCOT_FORCE_INLINE void decode_new_significance(mq_decoder &mq, const pass_setup &st, uint32_t *wp,
                                                uint32_t &w, int32_t *col, uint32_t i, int32_t one_half) {
  uint32_t sign;
  if (Raw) {
    sign = mq.get_raw_symbol();
  } else {
    const uint32_t idx = ebcot::sign_context_index(w, wp[-1], wp[1], i, st.causal);
    sign               = (mq.decode(sign_LUT[0][idx]) ^ sign_LUT[1][idx]) & 1u;
  }
  col[i * st.sample_stride] |= one_half | static_cast<int32_t>(sign << 31);
  w |= ebcot::sigma_bit(i) | (sign << (3 * i + 19));
  ebcot::broadcast_significance(wp, st.ctx_stride, i, sign);
}

// One sample of the significance propagation pass: coded if insignificant with a
// significant neighbour.
template <bool Raw>
EBCOT_FORCE_INLINE void sigprop_sample(mq_decoder &mq, const pass_setup &st, uint32_t *wp, uint32_t &w,
                                       int32_t *col, uint32_t i, int32_t one_half) {
  if (w & ebcot::sigma_bit(i)) return;
  const uint32_t nb = ebcot::sigma_neighbourhood(w, i, st.causal);
  if (nb == 0) return;
  w |= ebcot::pi_bit(i);
  const uint32_t bit = Raw ? mq.get_raw_symbol() : mq.decode(st.sig_lut[nb]);
  if (bit) decode_new_significance<Raw>(mq, st, wp, w, col, i, one_half);
}

// One sample of the magnitude refinement pass: coded if it was significant before
// this bit-plane (σ = 1, π = 0).  The marker from the previous bit-plane sits at
// bit p (old_marker); it is replaced by the decoded bit and the new marker goes to
// bit p - 1.  With half == 0 this is a plain OR of the decoded bit.
template <bool Raw>
EBCOT_FORCE_INLINE void magref_sample(mq_decoder &mq, const pass_setup &st, uint32_t &w, uint32_t todo,
                                      int32_t *col, uint32_t i, uint8_t p, int32_t half,
                                      int32_t old_marker) {
  if ((todo & ebcot::sigma_bit(i)) == 0) return;
  uint32_t bit;
  if (Raw) {
    bit = mq.get_raw_symbol();
  } else {
    uint8_t label = 16;  // later refinements
    if ((w & ebcot::sigma_bar_bit(i)) == 0) {
      label = (ebcot::sigma_neighbourhood(w, i, st.causal) & ~0x10u) ? 15 : 14;
    }
    bit = mq.decode(label);
  }
  int32_t &v = col[i * st.sample_stride];
  v          = (v & ~old_marker) | static_cast<int32_t>(bit << p) | half;
  w |= ebcot::sigma_bar_bit(i);
}

// One sample of the cleanup pass: coded if neither significant nor coded by this
// bit-plane's significance propagation pass.
EBCOT_FORCE_INLINE void cleanup_sample(mq_decoder &mq, const pass_setup &st, uint32_t *wp, uint32_t &w,
                                       int32_t *col, uint32_t i, int32_t one_half) {
  if (w & (ebcot::sigma_bit(i) | ebcot::pi_bit(i))) return;
  if (mq.decode(st.sig_lut[ebcot::sigma_neighbourhood(w, i, st.causal)]))
    decode_new_significance<false>(mq, st, wp, w, col, i, one_half);
}

// Each pass walks the stripes; a full stripe (4 rows) is unrolled so that every
// row's bit positions are constants, and a final partial stripe takes a loop.
template <bool Raw, bool Full>
EBCOT_FORCE_INLINE void sigprop_stripe(mq_decoder &mq, const pass_setup &st, uint32_t *wp, int32_t *sp,
                                       uint32_t rows, int32_t one_half) {
  for (uint32_t x = 0; x < st.width; ++x, ++wp) {
    uint32_t w   = *wp;
    int32_t *col = sp + x;
    if (Full) {
      sigprop_sample<Raw>(mq, st, wp, w, col, 0, one_half);
      sigprop_sample<Raw>(mq, st, wp, w, col, 1, one_half);
      sigprop_sample<Raw>(mq, st, wp, w, col, 2, one_half);
      sigprop_sample<Raw>(mq, st, wp, w, col, 3, one_half);
    } else {
      for (uint32_t i = 0; i < rows; ++i) sigprop_sample<Raw>(mq, st, wp, w, col, i, one_half);
    }
    *wp = w;
  }
}

template <bool Raw>
EBCOT_PASS_NOINLINE void decode_sigprop(j2k_codeblock *block, uint8_t p, int32_t half, mq_decoder &mq_io) {
  mq_decoder mq = mq_io;
  const pass_setup st(block);
  const int32_t one_half = (1 << p) | half;
  for (uint32_t y0 = 0, s = 0; y0 < st.height; y0 += 4, ++s) {
    uint32_t *wp = st.ctx + (s + 1) * st.ctx_stride + 1;
    int32_t *sp  = st.samples + y0 * st.sample_stride;
    if (st.height - y0 >= 4)
      sigprop_stripe<Raw, true>(mq, st, wp, sp, 4, one_half);
    else
      sigprop_stripe<Raw, false>(mq, st, wp, sp, st.height - y0, one_half);
  }
  mq_io = mq;
}

template <bool Raw, bool Full>
EBCOT_FORCE_INLINE void magref_stripe(mq_decoder &mq, const pass_setup &st, uint32_t *wp, int32_t *sp,
                                      uint32_t rows, uint8_t p, int32_t half) {
  const int32_t old_marker = half << 1;
  for (uint32_t x = 0; x < st.width; ++x, ++wp) {
    uint32_t w = *wp;
    // Samples that were significant before this bit-plane: σ = 1 and π = 0.
    const uint32_t todo = w & ebcot::SIGMA_SELF_ALL & ~(w >> ebcot::PI_TO_SIGMA_SHIFT);
    int32_t *col        = sp + x;
    if (Full) {
      magref_sample<Raw>(mq, st, w, todo, col, 0, p, half, old_marker);
      magref_sample<Raw>(mq, st, w, todo, col, 1, p, half, old_marker);
      magref_sample<Raw>(mq, st, w, todo, col, 2, p, half, old_marker);
      magref_sample<Raw>(mq, st, w, todo, col, 3, p, half, old_marker);
    } else {
      for (uint32_t i = 0; i < rows; ++i) magref_sample<Raw>(mq, st, w, todo, col, i, p, half, old_marker);
    }
    *wp = w;
  }
}

template <bool Raw>
EBCOT_PASS_NOINLINE void decode_magref(j2k_codeblock *block, uint8_t p, int32_t half, mq_decoder &mq_io) {
  mq_decoder mq = mq_io;
  const pass_setup st(block);
  for (uint32_t y0 = 0, s = 0; y0 < st.height; y0 += 4, ++s) {
    uint32_t *wp = st.ctx + (s + 1) * st.ctx_stride + 1;
    int32_t *sp  = st.samples + y0 * st.sample_stride;
    if (st.height - y0 >= 4)
      magref_stripe<Raw, true>(mq, st, wp, sp, 4, p, half);
    else
      magref_stripe<Raw, false>(mq, st, wp, sp, st.height - y0, p, half);
  }
  mq_io = mq;
}

EBCOT_PASS_NOINLINE void decode_cleanup(j2k_codeblock *block, uint8_t p, int32_t half, mq_decoder &mq_io) {
  constexpr uint8_t label_run = 17;
  constexpr uint8_t label_uni = 18;
  mq_decoder mq               = mq_io;
  const pass_setup st(block);
  const int32_t one_half  = (1 << p) | half;
  const uint32_t run_mask = st.causal ? ebcot::SIGMA_RUN_CAUSAL : ebcot::SIGMA_WINDOW;
  for (uint32_t y0 = 0, s = 0; y0 < st.height; y0 += 4, ++s) {
    uint32_t *wp = st.ctx + (s + 1) * st.ctx_stride + 1;
    int32_t *sp  = st.samples + y0 * st.sample_stride;
    if (st.height - y0 >= 4) {
      for (uint32_t x = 0; x < st.width; ++x, ++wp) {
        uint32_t w   = *wp;
        int32_t *col = sp + x;
        uint32_t r   = 0;  // first row still to code
        if ((w & run_mask) == 0) {
          // Run mode: all four samples are insignificant with insignificant neighbours.
          if (mq.decode(label_run) == 0) continue;  // all four stay insignificant; no π is set
          r = static_cast<uint32_t>(mq.decode(label_uni)) << 1;
          r |= mq.decode(label_uni);
          decode_new_significance<false>(mq, st, wp, w, col, r, one_half);
          ++r;
        }
        if (r == 0) cleanup_sample(mq, st, wp, w, col, 0, one_half);
        if (r <= 1) cleanup_sample(mq, st, wp, w, col, 1, one_half);
        if (r <= 2) cleanup_sample(mq, st, wp, w, col, 2, one_half);
        if (r <= 3) cleanup_sample(mq, st, wp, w, col, 3, one_half);
        *wp = w & ~ebcot::PI_ALL;
      }
    } else {
      const uint32_t rows = st.height - y0;
      for (uint32_t x = 0; x < st.width; ++x, ++wp) {
        uint32_t w   = *wp;
        int32_t *col = sp + x;
        for (uint32_t i = 0; i < rows; ++i) cleanup_sample(mq, st, wp, w, col, i, one_half);
        *wp = w & ~ebcot::PI_ALL;
      }
    }
  }
  mq_io = mq;
}

}  // namespace

void j2k_decode(j2k_codeblock *block, const uint8_t ROIshift) {
  constexpr uint8_t label_uni = 18;

  uint8_t num_decode_pass = 0;
  for (uint16_t i = 0; i < block->num_layers; i++) {
    num_decode_pass = static_cast<uint8_t>(num_decode_pass + block->layer_passes[i]);
  }
  mq_decoder mq_dec(block->get_compressed_data());
  // mq_dec.set_dynamic_table();

  const auto M_b           = static_cast<int32_t>(block->get_Mb());
  const int32_t K          = M_b + static_cast<int32_t>(ROIshift) - static_cast<int32_t>(block->num_ZBP);
  const int32_t max_passes = 3 * (K)-2;

  uint8_t z                    = 0;  // pass index
  uint8_t k                    = 2;  // pass category (0 = sig, 1 = mag, 2 = cleanup)
  uint8_t pmsb                 = static_cast<uint8_t>(30 - block->num_ZBP);  // index of the MSB
  uint8_t p                    = pmsb;                                       // index of current bitplane
  uint8_t current_segment_pass = 0;  // number of passes in a current codeword segment
  uint32_t segment_bytes       = 0;
  uint32_t segment_pos         = 0;  // position of a current codeword segment
  uint8_t bypass_threshold     = 0;  // threshold of pass index in BYPASS mode

  if (block->Cmodes & BYPASS) {
    bypass_threshold = 10;
  }
  bool is_bypass = false;  // flag for BYPASS mode
  while (z < num_decode_pass) {
    if (k == 3) {
      k = 0;
      p--;  // move down to the next bitplane
    }
    if (current_segment_pass == 0) {
      // segment_start = z;
      current_segment_pass = (uint8_t)max_passes;

      // BYPASS mode
      if (bypass_threshold > 0) {
        if (z < bypass_threshold) {
          current_segment_pass = static_cast<uint8_t>(bypass_threshold - z);
        } else if (k == 2) {  // Cleanup pass
          current_segment_pass = 1;
          is_bypass            = false;
        } else {  // Sigprop or Magref pass
          current_segment_pass = 2;
          is_bypass            = true;
        }
      }

      // RESTART mode
      if (block->Cmodes & RESTART) {
        current_segment_pass = 1;
      }

      if ((z + current_segment_pass) > num_decode_pass) {
        current_segment_pass = static_cast<uint8_t>(num_decode_pass - z);
        if (num_decode_pass < max_passes) {
          // TODO: ?truncated?
        }
      }

      segment_bytes = 0;
      for (uint8_t n = 0; n < current_segment_pass; n++) {
        segment_bytes += block->pass_length[static_cast<size_t>(z + n)];
      }
      // pass_length[] is attacker-controlled; keep the MQ segment inside the
      // (already clamped) codeblock buffer to avoid an over-read.
      if (static_cast<uint64_t>(segment_pos) + segment_bytes > block->length) {
        printf("WARNING: MQ segment length exceeds codeblock bytes — malformed input.\n");
        throw std::exception();
      }
      mq_dec.init(segment_pos, segment_bytes, is_bypass);
      segment_pos += segment_bytes;
    }

    // Reconstruction marker for this bit-plane (see the comment above the passes).
    const int32_t half = (ROIshift == 0 && p > 0) ? (1 << p) >> 1 : 0;

    if (z == 0 || block->Cmodes & RESET) {
      mq_dec.init_states_for_all_contexts();
    }

    if (k == 0) {
      if (is_bypass) {
        decode_sigprop<true>(block, p, half, mq_dec);
      } else {
        decode_sigprop<false>(block, p, half, mq_dec);
      }
    } else if (k == 1) {
      if (is_bypass) {
        decode_magref<true>(block, p, half, mq_dec);
      } else {
        decode_magref<false>(block, p, half, mq_dec);
      }
    } else {
      decode_cleanup(block, p, half, mq_dec);
      if (block->Cmodes & SEGMARK) {
        int32_t r = 0;
        for (uint8_t i = 0; i < 4; i++) {
          r <<= 1;
          r += mq_dec.decode(label_uni);
        }
        if (r != 10) {
          printf("ERROR: SEGMARK test failed.\n");
          throw std::exception();
        }
      }
    }
    current_segment_pass--;
    if (current_segment_pass == 0) {
      mq_dec.finish();
    }
    z++;
    k++;
  }  // end of while

  j2k_dequant(block->sample_buf, block->blksampl_stride, block->band_buf, block->band_stride, block->size.x,
              block->size.y, M_b, ROIshift, block->transformation, block->stepsize, block->dequant_i32);
  // TODO: if k !=0
}
