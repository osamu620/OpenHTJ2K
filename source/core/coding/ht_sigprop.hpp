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

#pragma once

#include <cstdint>

// Which samples the HT SigProp pass (ISO/IEC 15444-15, 7.4) codes and which it makes
// significant, a stripe at a time.  Shared by the SigProp encoder and the rate control,
// which counts them at every bit-plane without coding.
namespace ht_sigprop {

// Stripe words: bit 4c + r of word k is the sample in row r and column 16k + c of a
// stripe, so each 16-bit field is one coding group of four columns in scan order.
constexpr uint64_t ROW0      = 0x1111111111111111ull;  // row 0 of every column
constexpr uint64_t ROW3      = ROW0 << 3;
constexpr uint32_t MAX_WORDS = 1024 / 16;

// The samples that a newly significant sample n makes SigProp candidates, other than
// the next one along its own row: the sample below it, and the samples above and
// below that one in the next column.  left is the word before n.
inline uint64_t reached(uint64_t n, uint64_t left) {
  return (((n << 1) | (n << 5) | (left >> 59)) & ~ROW0) | (((n << 3) | (left >> 61)) & ~ROW3);
}

// Solves the SigProp pass of one stripe of `rows` rows (1 to 4) and nk words (at most
// MAX_WORDS).  sig: the samples significant after the cleanup pass; bit: the magnitude bit
// SigProp codes (zero past the code-block); next_sig: the next stripe's sig (zero after the
// last stripe; only its first row counts); above: at row 3, the stripe above's last row
// after SigProp (zero for the first stripe), which this updates for the next stripe;
// last_cols: the columns of word nk - 1 inside the code-block.  Gives newly, the samples
// that become significant, and coded, the samples SigProp codes a bit for.
//
// A sample is coded when it is not significant and one of its eight neighbours is
// significant or became significant earlier in the pass: in the stripe above, the column
// to the left, or the rows above in the same column.  With CAUSAL the last row of a stripe
// does not look at the row below.
//
// NK, when not 0, is nk known at compile time (the loops then unroll; nk is ignored).
template <uint32_t NK = 0>
inline void solve_stripe(const uint64_t *sig, const uint64_t *bit, const uint64_t *next_sig,
                         uint64_t *above, uint32_t nk_, uint32_t rows, uint64_t last_cols, bool causal,
                         uint64_t *newly, uint64_t *coded) {
  const uint32_t nk = NK ? NK : nk_;
  uint64_t vert[MAX_WORDS + 1];  // a significant sample in the same column, rows -1..+1
  uint64_t near_sig[MAX_WORDS];  // a significant sample among the eight neighbours
  uint64_t grow[MAX_WORDS];      // not significant, with the bit set: significant once coded
  const uint64_t live = ROW0 * ((1u << rows) - 1);
  for (uint32_t k = 0; k < nk; ++k) {
    const uint64_t g = sig[k];
    uint64_t v       = g | ((g << 1) & ~ROW0) | ((g >> 1) & ~ROW3) | ((above[k] >> 3) & ROW0);
    if (!causal) v |= (next_sig[k] & ROW0) << 3;
    vert[k] = v;
  }
  vert[nk] = 0;
  // The newly significant samples solve, in scan order,
  //   newly(r, j) = grow(r, j) & (near_sig(r, j) | newly(r, j - 1) | reached(r, j)).
  // Most of them are next to a sample that the cleanup pass made significant; the
  // rest are solved only where those can spread.
  uint64_t left = 0, left_new = 0, spread = 0;
  for (uint32_t k = 0; k < nk; ++k) {
    const uint64_t v = vert[k];
    near_sig[k]      = v | (v << 4) | (left >> 60) | (v >> 4) | (vert[k + 1] << 60);
    left             = v;
    grow[k]          = ~sig[k] & live & bit[k];
    const uint64_t n = grow[k] & near_sig[k];
    spread |= grow[k] & ~n & (reached(n, left_new) | (n << 4) | (left_new >> 60));
    newly[k] = n;
    left_new = n;
  }

  // Short spreads settle after a step or two over all rows at once.
  for (uint32_t step = 0; step < 2 && spread; ++step) {
    spread = 0;
    left   = 0;
    for (uint32_t k = 0; k < nk; ++k) {
      const uint64_t n   = newly[k];
      const uint64_t add = grow[k] & ~n & (reached(n, left) | (n << 4) | (left >> 60));
      newly[k]           = n | add;
      spread |= add;
      left = n | add;
    }
  }
  // Otherwise, row by row, newly(r, j - 1) is a run along the row: adding the seeds to
  // the row's grow bits (other bits set, so carries pass them) clears each run from
  // its first seed on.  The row below is known only from the previous round, so the
  // rows are repeated while it adds a seed to the row above.
  while (spread) {
    for (uint32_t r = 0; r < rows; ++r) {
      const uint64_t rr = ROW0 << r;
      uint64_t carry    = 0;
      left              = 0;
      for (uint32_t k = 0; k < nk; ++k) {
        const uint64_t n    = newly[k];
        const uint64_t gr   = grow[k] & rr;
        const uint64_t seed = (near_sig[k] | reached(n, left)) & gr;
        const uint64_t run  = gr | ~rr;
        const uint64_t t    = run + seed;
        const uint64_t sum  = t + carry;
        carry               = static_cast<uint64_t>(t < run) | static_cast<uint64_t>(sum < t);
        newly[k]            = (n & ~rr) | (((run & ~sum) | seed) & gr);
        left                = n;
      }
    }
    uint64_t more = 0;
    left          = 0;
    for (uint32_t k = 0; k < nk; ++k) {
      const uint64_t n = newly[k];
      more |= (((n << 3) | (left >> 61)) & ~ROW3) & grow[k] & ~n;
      left = n;
    }
    if (more == 0) break;
  }

  left = 0;
  for (uint32_t k = 0; k < nk; ++k) {
    const uint64_t n    = newly[k];
    const uint64_t open = ~sig[k] & live & ((k + 1 == nk) ? last_cols : ~0ull);
    coded[k]            = open & (near_sig[k] | reached(n, left) | (n << 4) | (left >> 60));
    left                = n;
    above[k]            = (sig[k] | n) & ROW3;
  }
}

}  // namespace ht_sigprop
