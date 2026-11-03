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
#include <vector>

namespace open_htj2k {

// Vertical layout of the rows that the streaming decoders hand to their row callback
// (openhtj2k_decoder::invoke_line_based_stream and the entry points built on it).
//
// Positions are rows of the reference grid reduced by 2^reduce_NL.  The image covers
// grid rows [ceil(YOsiz / 2^r), ceil(Ysiz / 2^r)), and row k of component c, whose
// vertical subsampling factor is yr[c] (SIZ YRsiz), sits at grid row k * yr[c] and covers
// grid rows [k * yr[c], (k + 1) * yr[c]) of the image (ISO/IEC 15444-1, B.2 and B.5:
// reducing the resolution and then subsampling gives the same component rows as the
// other way round).
//
// Output row y covers the grid rows from start(y) = s + y * g, where g is the greatest
// common divisor of the yr[c] and s the first grid row that any component covers, to
// start(y + 1).  Every multiple of every yr[c] is an output row boundary, so a
// component has a single row in each output row: row abs_row(c, y) =
// floor(start(y) / yr[c]).  Grid rows above a component's first sample are covered by
// none of its rows (abs_row(c, y) < first[c]).  When every yr[c] is a multiple of
// yr[0], the output rows are exactly the rows of component 0.
struct stream_row_map {
  uint32_t g = 1;               // output-row pitch on the reduced grid
  uint64_t s = 0;               // reduced grid row where output row 0 starts
  uint32_t n = 0;               // number of output rows
  std::vector<uint32_t> yr;     // vertical subsampling factor of each component
  std::vector<uint32_t> first;  // absolute index of each component's first row
  std::vector<uint32_t> end;    // one past the absolute index of its last row

  // ceil(a / 2^r) for a < 2^32 and any r.
  static uint64_t ceil_shift(uint64_t a, uint8_t r) {
    return (r >= 32) ? (a != 0) : (a + (uint64_t{1} << r) - 1) >> r;
  }
  static uint32_t gcd(uint32_t a, uint32_t b) {
    while (b != 0) {
      const uint32_t t = a % b;
      a                = b;
      b                = t;
    }
    return a;
  }

  // YOsiz, Ysiz: SIZ image origin and extent; YRsiz: one factor (>= 1) per component.
  void init(uint32_t YOsiz, uint32_t Ysiz, const std::vector<uint32_t> &YRsiz, uint8_t reduce_NL) {
    yr = YRsiz;
    first.assign(yr.size(), 0);
    end.assign(yr.size(), 0);
    const uint64_t a = ceil_shift(YOsiz, reduce_NL);
    const uint64_t b = ceil_shift(Ysiz, reduce_NL);
    g                = 0;
    s                = UINT64_MAX;
    for (size_t c = 0; c < yr.size(); ++c) {
      first[c]           = static_cast<uint32_t>((a + yr[c] - 1) / yr[c]);
      end[c]             = static_cast<uint32_t>((b + yr[c] - 1) / yr[c]);
      g                  = gcd(yr[c], g);
      const uint64_t s_c = static_cast<uint64_t>(first[c]) * yr[c];
      if (s_c < s) s = s_c;
    }
    if (g == 0) g = 1;
    n = (s < b) ? static_cast<uint32_t>((b - s + g - 1) / g) : 0;
  }
  uint64_t start(uint32_t y) const { return s + static_cast<uint64_t>(y) * g; }
  // Absolute row of component c whose cell starts at or above output row y.
  uint32_t abs_row(uint16_t c, uint32_t y) const { return static_cast<uint32_t>(start(y) / yr[c]); }
  // Row of component c (0 = its first row) covering output row y; UINT32_MAX for none.
  uint32_t row(uint16_t c, uint32_t y) const {
    if (c >= yr.size() || y >= n) return UINT32_MAX;
    const uint32_t k = abs_row(c, y);
    return (k < first[c] || k >= end[c]) ? UINT32_MAX : k - first[c];
  }
  // First output row that starts at reduced grid row r or later (n when none does).
  uint32_t first_out_at(uint64_t r) const {
    if (r <= s) return 0;
    const uint64_t y = (r - s + g - 1) / g;
    return (y < n) ? static_cast<uint32_t>(y) : n;
  }
};

}  // namespace open_htj2k
