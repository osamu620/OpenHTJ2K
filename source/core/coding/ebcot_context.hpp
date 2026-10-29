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

#include <cstddef>
#include <cstdint>

// Part 1 (EBCOT) coding-pass state, packed into one 32-bit word per stripe column
// (Taubman & Marcellin, "JPEG2000", §17.1.2, figure 17.2).  A stripe column is the
// 4 samples of one column inside a 4-row stripe.  The words form a 2-D array with
// a one-word border on the left and right and a one-stripe border above and below,
// so neighbour updates never need a bounds check.  The state is the same for
// encoding and decoding; nothing here depends on the decoder.
//
//   bits 0..17   σ (significance) of the 6x3 neighbourhood: row r at bits 3r..3r+2,
//                with r = 0 the row above the stripe, r = 1..4 the stripe rows and
//                r = 5 the row below; within a row, bit 0 = left column, bit 1 =
//                this column, bit 2 = right column.
//   bit  18      χ (sign) of the row above the stripe.
//   bit  19+3i   χ of stripe row i (i = 0..3).
//   bit  20+3i   σ̄ of stripe row i: the sample has had its first magnitude refinement.
//   bit  21+3i   π of stripe row i: the sample was coded in this bit-plane's
//                significance propagation pass.  The cleanup pass clears it.
//   bit  31      χ of the row below the stripe.
namespace ebcot {

constexpr uint32_t SIGMA_WINDOW     = 0x3FFFFu;  // all 18 σ bits
constexpr uint32_t SIGMA_RUN_CAUSAL = 0x07FFFu;  // σ bits a vertically causal run may see
constexpr uint32_t SIGMA_SELF_ALL   = 0x2490u;   // σ of stripe rows 0..3 (bits 4, 7, 10, 13)
constexpr uint32_t CHI_ABOVE        = 1u << 18;
constexpr uint32_t CHI_BELOW        = 1u << 31;
constexpr uint32_t PI_ALL           = (1u << 21) | (1u << 24) | (1u << 27) | (1u << 30);
constexpr int PI_TO_SIGMA_SHIFT     = 17;  // π of row i >> 17 lands on σ of row i

constexpr uint32_t sigma_bit(uint32_t i) { return 1u << (3 * i + 4); }
constexpr uint32_t chi_bit(uint32_t i) { return 1u << (3 * i + 19); }
constexpr uint32_t sigma_bar_bit(uint32_t i) { return 1u << (3 * i + 20); }
constexpr uint32_t pi_bit(uint32_t i) { return 1u << (3 * i + 21); }

// The 3x3 σ neighbourhood of stripe row i as a 9-bit index: TL T TR L self R BL B BR
// at bits 0..8.  In vertically causal mode the row below stripe row 3 is not visible,
// so its three bits are cleared.
inline uint32_t sigma_neighbourhood(uint32_t w, uint32_t i, bool causal) {
  uint32_t nb = (w >> (3 * i)) & 0x1FFu;
  if (causal && i == 3) nb &= 0x03Fu;
  return nb;
}

// Index into sign_LUT for stripe row i of the column whose word is w, given the words
// of the columns to its left (wl) and right (wr):
//   bit 0 σ above, bit 1 σ below, bit 2 σ left, bit 3 σ right,
//   bit 4 χ above, bit 5 χ below, bit 6 χ left, bit 7 χ right.
// χ is only ever set together with σ, so an insignificant neighbour contributes 0.
inline uint32_t sign_context_index(uint32_t w, uint32_t wl, uint32_t wr, uint32_t i, bool causal) {
  const uint32_t s = 3 * i;
  uint32_t idx     = ((w >> (s + 1)) & 1u) | (((w >> (s + 7)) & 1u) << 1) | (((w >> (s + 3)) & 1u) << 2)
                 | (((w >> (s + 5)) & 1u) << 3);
  const uint32_t chi_above = (i == 0) ? (w >> 18) & 1u : (w >> (s + 16)) & 1u;
  idx |= chi_above << 4;
  idx |= ((w >> (s + 22)) & 1u) << 5;
  idx |= ((wl >> (s + 19)) & 1u) << 6;
  idx |= ((wr >> (s + 19)) & 1u) << 7;
  if (causal && i == 3) idx &= 0xDDu;  // the row below is not visible
  return idx;
}

// Stripe row i of the column at wp became significant with sign `sign` (0 or 1).
// Updates the neighbouring words only; the caller keeps the column's own word in a
// register and adds sigma_bit(i) and, for a negative sample, chi_bit(i) to it.
inline void broadcast_significance(uint32_t *wp, size_t stride, uint32_t i, uint32_t sign) {
  const uint32_t r = 3 * (i + 1);
  wp[-1] |= 1u << (r + 2);  // the right neighbour of the column to the left
  wp[1] |= 1u << r;         // the left neighbour of the column to the right
  if (i == 0) {             // the row below the stripe above
    uint32_t *up = wp - stride;
    up[-1] |= 1u << 17;
    up[0] |= (1u << 16) | (sign << 31);
    up[1] |= 1u << 15;
  } else if (i == 3) {  // the row above the stripe below
    uint32_t *down = wp + stride;
    down[-1] |= 1u << 2;
    down[0] |= (1u << 1) | (sign << 18);
    down[1] |= 1u << 0;
  }
}

}  // namespace ebcot
