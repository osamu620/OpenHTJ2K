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
#include "block_dequant.hpp"

// 9-bit-indexed σ-context LUT for the packed stripe-column word (book §17.1.2 figure 17.2).
// Built once at startup from the 8-bit sig_LUT by remapping the neighbourhood-bit layout:
//   8-bit sig_LUT:  TL=0 TR=1 BL=2 BR=3 T=4 B=5 L=6 R=7
//   9-bit window:   TL=0 T=1 TR=2 L=3 self=4 R=5 BL=6 B=7 BR=8
// The self bit (bit 4) is a don't-care since κ^sig is only queried when σ=0; we still
// populate all 512 entries so the index never needs masking.
namespace {
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
const SigLUT9Init sig_LUT9_table;
}  // namespace

// When sample (j1, j2) becomes significant, broadcast its σ bit into every context
// word that references the 3×3 neighbourhood around it: 3 adjacent columns in the
// owning stripe, plus up to 3 more words in the previous/next stripe if j1 is on
// the stripe edge (book §17.1.2 state broadcasting).
static inline void broadcast_sigma_context(j2k_codeblock *block, uint32_t j1, uint32_t j2) {
  uint32_t *ctx     = block->block_contexts;
  const size_t cs   = block->block_contexts_stride;
  const uint32_t s0 = j1 >> 2;
  const uint32_t sr = j1 & 3u;
  const size_t xb   = static_cast<size_t>(j2) + 1;  // +1 for left border column
  const size_t sb   = static_cast<size_t>(s0) + 1;  // +1 for top border stripe
  uint32_t *row     = ctx + sb * cs;
  const uint32_t bp = 3u * (sr + 1);
  row[xb - 1] |= 1u << (bp + 2);    // σ(j1, j2) as right neighbour of column j2-1
  row[xb    ] |= 1u << (bp + 1);    // σ(j1, j2) as self
  row[xb + 1] |= 1u << (bp + 0);    // σ(j1, j2) as left neighbour of column j2+1
  if (sr == 0) {
    uint32_t *prev = row - cs;
    prev[xb - 1] |= 1u << 17;        // "row below previous stripe"
    prev[xb    ] |= 1u << 16;
    prev[xb + 1] |= 1u << 15;
  } else if (sr == 3) {
    uint32_t *next = row + cs;
    next[xb - 1] |= 1u << 2;         // "row above next stripe"
    next[xb    ] |= 1u << 1;
    next[xb + 1] |= 1u << 0;
  }
}

// void j2k_codeblock::update_sample(const uint8_t &symbol, const uint8_t &p, const int16_t &j1,
//                                   const int16_t &j2) const {
//   sample_buf[static_cast<size_t>(j2) + static_cast<size_t>(j1) * blksampl_stride] |=
//       static_cast<int32_t>(symbol) << p;
// }
//
// void j2k_codeblock::update_sign(const int8_t &val, const uint32_t &j1, const uint32_t &j2) const {
//   sample_buf[static_cast<size_t>(j2) + static_cast<size_t>(j1) * blksampl_stride] |= val << 31;
// }
// static inline uint8_t get_sign(j2k_codeblock *block, const uint32_t &j1, const uint32_t &j2) {
//  return static_cast<uint8_t>(((uint32_t)(block->sample_buf[j2 + j1 * block->blksampl_stride] | 0x8000))
//                              >> 31);
//}

static inline uint8_t get_context_label_sig(j2k_codeblock *block, const uint32_t &j1, const uint32_t &j2) {
  // Packed stripe-column word (book §17.1.2): one 32-bit load replaces 8 neighbour state loads.
  const uint32_t *ctx = block->block_contexts;
  const size_t cs     = block->block_contexts_stride;
  const uint32_t s    = j1 >> 2;
  const uint32_t sr   = j1 & 3u;
  const uint32_t word = ctx[(static_cast<size_t>(s) + 1) * cs + (static_cast<size_t>(j2) + 1)];
  uint32_t idx9       = (word >> (3u * sr)) & 0x1FFu;
  if ((block->Cmodes & CAUSAL) && sr == 3u) {
    // Clear the 3 "row below" bits (BL, B, BR at 9-bit positions 6,7,8) — same semantics
    // as the old 0xD3 mask on the 8-bit index.
    idx9 &= 0x03Fu;
  }
  return sig_LUT9_table.data[block->get_orientation()][idx9];
}
static inline uint8_t get_signLUT_index(j2k_codeblock *block, const uint32_t &j1, const uint32_t &j2) {
  int32_t idx          = 0;
  uint8_t *p           = block->block_states;
  const size_t stride  = block->blkstate_stride;
  int32_t *sp          = block->sample_buf + j2 + j1 * block->blksampl_stride;
  const size_t sstride = block->blksampl_stride;
  uint8_t *sigma_p0    = p + j1 * stride + j2;
  uint8_t *sigma_p1    = p + (j1 + 1) * stride + j2;
  uint8_t *sigma_p2    = p + (j1 + 2) * stride + j2;

  idx += sigma_p0[1] & 1;                                                // top
  idx += (sigma_p1[0] & 1) << 2;                                         // left
  idx += (sigma_p1[2] & 1) << 3;                                         // right
  idx += (sigma_p2[1] & 1) << 1;                                         // bottom
  idx += (j1 > 0) ? ((sp[-sstride] >> 31) & 1) << 4 : 0;                 // top sign
  idx += (j2 > 0) ? ((sp[-1] >> 31) & 1) << 6 : 0;                       // left sign
  idx += (j2 < block->size.x - 1) ? ((sp[1] >> 31) & 1) << 7 : 0;        // right sign
  idx += (j1 < block->size.y - 1) ? ((sp[sstride] >> 31) & 1) << 5 : 0;  // bottom sign

  //  idx += get_state(Sigma, j1m1, j2);  // top
  //  idx += (j1 > 0) ? get_sign(block, j1 - 1, j2) << 4 : 0;  // top sign
  //  idx += get_state(Sigma, j1, j2m1) << 2;                                        // left
  //  idx += (j2 > 0) ? get_sign(block, j1, j2 - 1) << 6 : 0;  // left sign
  //  idx += get_state(Sigma, j1, j2p1) << 3;                                        // right
  //  idx += (j2 < block->size.x - 1) ? get_sign(block, j1, j2 + 1) << 7 : 0;  // right sign
  //  idx += get_state(Sigma, j1p1, j2) << 1;                                        // bottom
  //  idx += (j1 < block->size.y - 1) ? get_sign(block, j1 + 1, j2) << 5 : 0;  // bottom sign

  return static_cast<uint8_t>(idx);
}

static inline void decode_j2k_sign_raw(j2k_codeblock *block, mq_decoder &mq_dec, const uint32_t &j1,
                                       const uint32_t &j2) {
  uint8_t symbol = mq_dec.get_raw_symbol();
  //  block->update_sign(static_cast<int8_t>(symbol), j1, j2);
  block->sample_buf[j1 * block->blksampl_stride + j2] |= symbol << 31;
}
static inline void decode_j2k_sign(j2k_codeblock *block, mq_decoder &mq_dec, const uint32_t &j1,
                                   const uint32_t &j2) {
  uint8_t idx = get_signLUT_index(block, j1, j2);
  if ((block->Cmodes & CAUSAL) && j1 % 4 == 3) {
    idx &= 0xDD;
  }
  const uint8_t x      = mq_dec.decode(sign_LUT[0][idx]);
  const uint8_t XORbit = sign_LUT[1][idx];
  //  block->update_sign((x ^ XORbit) & 1, j1, j2);
  block->sample_buf[j1 * block->blksampl_stride + j2] |= ((x ^ XORbit) & 1) << 31;
}

inline void decode_sigprop_pass_raw(j2k_codeblock *block, const uint8_t &p, mq_decoder &mq_dec) {
  uint16_t num_v_stripe = static_cast<uint16_t>(block->size.y / 4);
  uint32_t j1, j2, j1_start = 0;
  uint8_t label_sig;
  uint8_t symbol;
  for (uint16_t n = 0; n < num_v_stripe; n++) {
    for (j2 = 0; j2 < block->size.x; j2++) {
      for (j1 = j1_start; j1 < j1_start + 4; j1++) {
        uint8_t *state_p = block->block_states + (j1 + 1) * block->blkstate_stride + (j2 + 1);
        if ((state_p[0] >> SHIFT_SIGMA & 1) == 0
            && (label_sig = get_context_label_sig(block, j1, j2)) > 0) {
          //            block->modify_state(decoded_bitplane_index, p, j1, j2);
          state_p[0] &= 0x7;
          state_p[0] |= static_cast<uint8_t>(p << SHIFT_P);
          symbol = mq_dec.get_raw_symbol();
          //          block->update_sample(symbol, p, j1, j2);
          if (symbol) {
            block->sample_buf[j1 * block->blksampl_stride + j2] |= 1 << p;
            //            block->modify_state(sigma, symbol, j1, j2);  // symbol shall be 1
            state_p[0] |= symbol;
            broadcast_sigma_context(block, j1, j2);
            decode_j2k_sign_raw(block, mq_dec, j1, j2);
          }
          //          block->modify_state(pi_, 1, j1, j2);
          state_p[0] |= static_cast<uint8_t>(1 << SHIFT_PI_);
        } else {
          //          block->modify_state(pi_, 0, j1, j2);
          state_p[0] &= static_cast<uint8_t>(~(1 << SHIFT_PI_));
        }
      }
    }
    j1_start += 4;
  }

  if (block->size.y % 4) {
    for (j2 = 0; j2 < block->size.x; j2++) {
      for (j1 = j1_start; j1 < j1_start + block->size.y % 4; j1++) {
        uint8_t *state_p =
            block->block_states + (static_cast<unsigned long>(j1 + 1)) * block->blkstate_stride + (j2 + 1);
        if ((state_p[0] >> SHIFT_SIGMA & 1) == 0
            && (label_sig = get_context_label_sig(block, j1, j2)) > 0) {
          //            block->modify_state(decoded_bitplane_index, p, j1, j2);
          state_p[0] &= 0x7;
          state_p[0] |= static_cast<uint8_t>(p << SHIFT_P);
          symbol = mq_dec.get_raw_symbol();
          //          block->update_sample(symbol, p, j1, j2);
          if (symbol) {
            block->sample_buf[j1 * block->blksampl_stride + j2] |= 1 << p;
            //            block->modify_state(sigma, symbol, j1, j2);  // symbol shall be 1
            state_p[0] |= symbol;
            broadcast_sigma_context(block, j1, j2);
            decode_j2k_sign_raw(block, mq_dec, j1, j2);
          }
          //          block->modify_state(pi_, 1, j1, j2);
          state_p[0] |= static_cast<uint8_t>(1 << SHIFT_PI_);
        } else {
          //          block->modify_state(pi_, 0, j1, j2);
          state_p[0] &= static_cast<uint8_t>(~(1 << SHIFT_PI_));
        }
      }
    }
  }
}

inline void decode_sigprop_pass(j2k_codeblock *block, const uint8_t &p, mq_decoder &mq_dec) {
  const uint32_t width        = block->size.x;
  const uint32_t height       = block->size.y;
  const uint16_t num_v_stripe = static_cast<uint16_t>(height / 4);
  const size_t stride         = block->blkstate_stride;
  const size_t sstride        = block->blksampl_stride;
  uint8_t *const states       = block->block_states;
  int32_t *const samples      = block->sample_buf;
  uint32_t j1_start = 0;
  uint8_t label_sig, symbol;

  for (uint16_t n = 0; n < num_v_stripe; n++) {
    // Precompute row base pointers for this stripe
    uint8_t *row[6];
    for (int r = 0; r < 6; r++) {
      row[r] = states + (j1_start + static_cast<uint32_t>(r)) * stride;
    }
    for (uint32_t j2 = 0; j2 < width; j2++) {
      for (int ri = 0; ri < 4; ri++) {
        uint32_t j1      = j1_start + static_cast<uint32_t>(ri);
        uint8_t *state_p = row[ri + 1] + (j2 + 1);
        if ((state_p[0] >> SHIFT_SIGMA & 1) == 0
            && (label_sig = get_context_label_sig(block, j1, j2)) > 0) {
          state_p[0] &= 0x7;
          state_p[0] |= static_cast<uint8_t>(p << SHIFT_P);
          symbol = mq_dec.decode(label_sig);
          if (symbol) {
            samples[j1 * sstride + j2] |= 1 << p;
            state_p[0] |= symbol;
            broadcast_sigma_context(block, j1, j2);
            decode_j2k_sign(block, mq_dec, j1, j2);
          }
          state_p[0] |= static_cast<uint8_t>(1 << SHIFT_PI_);
        } else {
          state_p[0] &= static_cast<uint8_t>(~(1 << SHIFT_PI_));
        }
      }
    }
    j1_start += 4;
  }

  if (height % 4) {
    for (uint32_t j2 = 0; j2 < width; j2++) {
      for (uint32_t j1 = j1_start; j1 < j1_start + height % 4; j1++) {
        uint8_t *state_p = states + (j1 + 1) * stride + (j2 + 1);
        if ((state_p[0] >> SHIFT_SIGMA & 1) == 0
            && (label_sig = get_context_label_sig(block, j1, j2)) > 0) {
          state_p[0] &= 0x7;
          state_p[0] |= static_cast<uint8_t>(p << SHIFT_P);
          symbol = mq_dec.decode(label_sig);
          if (symbol) {
            samples[j1 * sstride + j2] |= 1 << p;
            state_p[0] |= symbol;
            broadcast_sigma_context(block, j1, j2);
            decode_j2k_sign(block, mq_dec, j1, j2);
          }
          state_p[0] |= static_cast<uint8_t>(1 << SHIFT_PI_);
        } else {
          state_p[0] &= static_cast<uint8_t>(~(1 << SHIFT_PI_));
        }
      }
    }
  }
}

inline void decode_magref_pass_raw(j2k_codeblock *block, const uint8_t &p, mq_decoder &mq_dec) {
  uint16_t num_v_stripe = static_cast<uint16_t>(block->size.y / 4);
  uint32_t j1, j2, j1_start = 0;
  uint8_t symbol;
  for (uint16_t n = 0; n < num_v_stripe; n++) {
    for (j2 = 0; j2 < block->size.x; j2++) {
      for (j1 = j1_start; j1 < j1_start + 4; j1++) {
        uint8_t *state_p = block->block_states + (j1 + 1) * block->blkstate_stride + (j2 + 1);
        if ((state_p[0] & 1 << SHIFT_SIGMA) == 1 && (state_p[0] & 1 << SHIFT_PI_) == 0) {
          //          block->modify_state(decoded_bitplane_index, p, j1, j2);
          state_p[0] &= 0x7;
          state_p[0] |= static_cast<uint8_t>(p << SHIFT_P);
          symbol = mq_dec.get_raw_symbol();
          //          block->update_sample(symbol, p, j1, j2);
          block->sample_buf[j1 * block->blksampl_stride + j2] |= symbol << p;
          //          block->modify_state(sigma_, 1, j1, j2);
          state_p[0] |= 1 << SHIFT_SIGMA_;
        }
      }
    }
    j1_start += 4;
  }

  if (block->size.y % 4 != 0) {
    for (j2 = 0; j2 < block->size.x; j2++) {
      for (j1 = j1_start; j1 < j1_start + block->size.y % 4; j1++) {
        uint8_t *state_p = block->block_states + (j1 + 1) * block->blkstate_stride + (j2 + 1);
        if ((state_p[0] & 1 << SHIFT_SIGMA) == 1 && (state_p[0] & 1 << SHIFT_PI_) == 0) {
          //          block->modify_state(decoded_bitplane_index, p, j1, j2);
          state_p[0] &= 0x7;
          state_p[0] |= static_cast<uint8_t>(p << SHIFT_P);
          symbol = mq_dec.get_raw_symbol();
          //          block->update_sample(symbol, p, j1, j2);
          block->sample_buf[j1 * block->blksampl_stride + j2] |= symbol << p;
          //          block->modify_state(sigma_, 1, j1, j2);
          state_p[0] |= 1 << SHIFT_SIGMA_;
        }
      }
    }
  }
}

inline void decode_magref_pass(j2k_codeblock *block, const uint8_t &p, mq_decoder &mq_dec) {
  const uint32_t width        = block->size.x;
  const uint32_t height       = block->size.y;
  const uint16_t num_v_stripe = static_cast<uint16_t>(height / 4);
  const size_t stride         = block->blkstate_stride;
  const size_t sstride        = block->blksampl_stride;
  uint8_t *const states       = block->block_states;
  int32_t *const samples      = block->sample_buf;
  uint32_t j1_start = 0;
  uint8_t label_sig, label_mag;
  uint8_t symbol;
  constexpr uint8_t mmm[4] = {14, 15, 16, 16};

  for (uint16_t n = 0; n < num_v_stripe; n++) {
    uint8_t *row[6];
    for (int r = 0; r < 6; r++) {
      row[r] = states + (j1_start + static_cast<uint32_t>(r)) * stride;
    }
    for (uint32_t j2 = 0; j2 < width; j2++) {
      for (int ri = 0; ri < 4; ri++) {
        uint8_t *state_p = row[ri + 1] + (j2 + 1);
        if ((state_p[0] & 1 << SHIFT_SIGMA) == 1 && (state_p[0] & 1 << SHIFT_PI_) == 0) {
          state_p[0] &= 0x7;
          state_p[0] |= static_cast<uint8_t>(p << SHIFT_P);
          label_sig = get_context_label_sig(block, j1_start + static_cast<uint32_t>(ri), j2);
          label_mag = mmm[(state_p[0] & 0x2) | (label_sig > 0)];
          symbol    = mq_dec.decode(label_mag);
          samples[(j1_start + static_cast<uint32_t>(ri)) * sstride + j2] |= symbol << p;
          state_p[0] |= 1 << SHIFT_SIGMA_;
        }
      }
    }
    j1_start += 4;
  }

  if (height % 4 != 0) {
    for (uint32_t j2 = 0; j2 < width; j2++) {
      for (uint32_t j1 = j1_start; j1 < j1_start + height % 4; j1++) {
        uint8_t *state_p = states + (j1 + 1) * stride + (j2 + 1);
        if ((state_p[0] & 1 << SHIFT_SIGMA) == 1 && (state_p[0] & 1 << SHIFT_PI_) == 0) {
          state_p[0] &= 0x7;
          state_p[0] |= static_cast<uint8_t>(p << SHIFT_P);
          label_sig = get_context_label_sig(block, j1, j2);
          label_mag = mmm[(state_p[0] & 0x2) | (label_sig > 0)];
          symbol    = mq_dec.decode(label_mag);
          samples[j1 * sstride + j2] |= symbol << p;
          state_p[0] |= 1 << SHIFT_SIGMA_;
        }
      }
    }
  }
}

inline void decode_cleanup_pass(j2k_codeblock *block, const uint8_t &p, mq_decoder &mq_dec) {
  const uint32_t width        = block->size.x;
  const uint32_t height       = block->size.y;
  const uint16_t num_v_stripe = static_cast<uint16_t>(height / 4);
  const size_t stride         = block->blkstate_stride;
  const size_t sstride        = block->blksampl_stride;
  uint8_t *const states       = block->block_states;
  int32_t *const samples      = block->sample_buf;
  uint32_t j1_start           = 0;
  uint8_t label_sig;
  const uint8_t label_run = 17;
  const uint8_t label_uni = 18;
  uint8_t symbol          = 0;
  int32_t k;
  int32_t r = 0;

  for (uint16_t n = 0; n < num_v_stripe; n++) {
    for (uint32_t j2 = 0; j2 < width; j2++) {
      k = 4;
      while (k > 0) {
        uint32_t j1      = j1_start + 4 - static_cast<uint32_t>(k);
        uint8_t *state_p = states + (j1 + 1) * stride + (j2 + 1);
        r                = -1;
        if (j1 % 4 == 0 && j1 <= height - 4) {
          // Run-mode trigger: all 4 samples' κ^sig == 0 ⟺ every σ bit in the 6×3
          // neighbourhood spanning the stripe column is zero — i.e. bits 0..17 of c[j].
          // For CAUSAL mode the 4th sample's "row below" bits (c[j] bits 15..17) are
          // don't-care; mask accordingly (book §17.1.2, last paragraph).
          const uint32_t cword =
              block->block_contexts[(static_cast<size_t>(j1_start >> 2) + 1)
                                        * block->block_contexts_stride
                                    + (static_cast<size_t>(j2) + 1)];
          const uint32_t sig_mask = (block->Cmodes & CAUSAL) ? 0x07FFFu : 0x3FFFFu;
          if ((cword & sig_mask) == 0) {
            symbol = mq_dec.decode(label_run);
            if (symbol == 0) {
              r = 4;
            } else {
              r = mq_dec.decode(label_uni);
              r <<= 1;
              r += mq_dec.decode(label_uni);
              samples[(j1 + static_cast<uint32_t>(r)) * sstride + j2] |= symbol << p;
            }
            k -= r;
          }
          if (k != 0) {
            j1      = j1_start + 4 - static_cast<uint32_t>(k);
            state_p = states + (j1 + 1) * stride + (j2 + 1);
          }
        }
        if ((state_p[0] & 1 << SHIFT_SIGMA) == 0 && (state_p[0] & 1 << SHIFT_PI_) == 0) {
          state_p[0] &= 0x7;
          state_p[0] |= static_cast<uint8_t>(p << SHIFT_P);
          if (r >= 0) {
            r = r - 1;
          } else {
            label_sig = get_context_label_sig(block, j1, j2);
            symbol    = mq_dec.decode(label_sig);
            samples[j1 * sstride + j2] |= symbol << p;
          }
          if (samples[j2 + j1 * sstride] == static_cast<int32_t>(1) << p) {
            state_p[0] |= 1;
            broadcast_sigma_context(block, j1, j2);
            decode_j2k_sign(block, mq_dec, j1, j2);
          }
        }
        k--;
      }
    }
    j1_start += 4;
  }

  if (height % 4 != 0) {
    for (uint32_t j2 = 0; j2 < width; j2++) {
      for (uint32_t j1 = j1_start; j1 < j1_start + height % 4; j1++) {
        uint8_t *state_p = states + (j1 + 1) * stride + (j2 + 1);
        if ((state_p[0] & 1 << SHIFT_SIGMA) == 0 && (state_p[0] & 1 << SHIFT_PI_) == 0) {
          state_p[0] &= 0x7;
          state_p[0] |= static_cast<uint8_t>(p << SHIFT_P);
          label_sig = get_context_label_sig(block, j1, j2);
          symbol    = mq_dec.decode(label_sig);
          samples[j1 * sstride + j2] |= symbol << p;
          if (symbol) {
            state_p[0] |= 1;
            broadcast_sigma_context(block, j1, j2);
            decode_j2k_sign(block, mq_dec, j1, j2);
          }
        }
      }
    }
  }
}

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
      mq_dec.init(segment_pos, segment_bytes, is_bypass);
      segment_pos += segment_bytes;
    }

    if (z == 0 || block->Cmodes & RESET) {
      mq_dec.init_states_for_all_contexts();
    }

    if (k == 0) {
      if (is_bypass) {
        decode_sigprop_pass_raw(block, p, mq_dec);
      } else {
        decode_sigprop_pass(block, p, mq_dec);
      }
    } else if (k == 1) {
      if (is_bypass) {
        decode_magref_pass_raw(block, p, mq_dec);
      } else {
        decode_magref_pass(block, p, mq_dec);
      }
    } else {
      decode_cleanup_pass(block, p, mq_dec);
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

  j2k_dequant(block->sample_buf, block->blksampl_stride, block->block_states, block->blkstate_stride,
              block->i_samples, block->band_stride, block->size.x, block->size.y, M_b, ROIshift,
              block->transformation, block->stepsize);
  // TODO: if k !=0
}
