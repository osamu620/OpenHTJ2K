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

class j2k_tile;
class j2k_codeblock;

// Rate control (EXPERIMENTAL) for HT code-blocks: the codestream is cut to a target size at
// the least squared error.  See ht_rate_control.cpp for the method.
//
// Each code-block takes two steps.  ht_rd_stats() measures it (counts and squared errors at
// every bit-plane that matters); ht_rd_code() then codes the HT set chosen for the slope
// threshold estimated from every code-block measured so far.  The encoder runs the first
// as soon as a code-block's samples are ready and the second once the estimate includes
// it; between them, on the encoding thread and in a fixed order, it folds the measurements
// into the estimate (ht_rc_state), so the output does not depend on the number of threads.
// ht_rate_control() finally picks every code-block's truncation point and transcodes the
// code-blocks whose point lies above their coded set.

// The slope threshold estimate: the measured code-blocks' rate-distortion curves, by
// subband class, extrapolated to the code-blocks not yet measured.  For Part 1 code-blocks
// it is given P1_ESTIMATE_BUDGET times their budget: its models are HT's, whose rates are
// 8-30% above Part 1's on the images measured.
constexpr double P1_ESTIMATE_BUDGET = 1.4;
class ht_rc_state {
 public:
  // budget: bytes the code-blocks may take (the target less an estimate of everything else).
  explicit ht_rc_state(uint64_t budget);
  // Registers the HT code-blocks of a tile (after enc_init()) and sets their weights.
  void add_tile(j2k_tile &tile);
  // Adds a measured code-block (ht_rd_stats() has returned) to the estimate.
  void fold(const j2k_codeblock *block);
  // Re-solves the threshold from everything folded.  0 until something is folded.
  void solve();
  double lambda() const { return lambda_; }

 private:
  struct sb_class {
    uint8_t comp, level, band;  // level 0 = the lowest resolution's LL band, 1 its detail bands, ...
    uint64_t total = 0, seen = 0;
    std::vector<double> rate;  // bytes of the folded hull steps, by slope bin
  };
  uint64_t budget_;
  double lambda_ = 0;
  std::vector<sb_class> classes_;
};

// Measures a code-block (its samples quantised in block->sample_buf, or quantised here).
// lambda: the current threshold estimate (0: none yet), which bounds the planes measured.
void ht_rd_stats(j2k_codeblock *block, double lambda);
// Codes the HT set of a measured code-block for block->rd->lambda.
int32_t ht_rd_code(j2k_codeblock *block, uint8_t ROIshift);
// Codes a measured Part 1 code-block's passes down to the bit-plane that the cut can need
// at block->rd->lambda, recording every pass as a truncation point.
int32_t p1_rd_code(j2k_codeblock *block, uint8_t ROIshift);

// Picks every HT code-block's truncation point so that the codestream takes at most
// target_bytes, transcodes the code-blocks whose point lies above their coded set, and
// builds the packet headers of every tile.  fixed_bytes is what the codestream takes outside
// the tile-parts (SOC, main header, EOC).  Returns the codestream length; it exceeds
// target_bytes only when even empty code-blocks do not fit.
uint64_t ht_rate_control(j2k_tile *tiles, uint32_t num_tiles, uint64_t fixed_bytes, uint64_t target_bytes);

// The same for Part 1 (EBCOT) code-blocks, whose coded passes ebcot_encode() records as
// truncation points (p1_rd_code()): picks every code-block's last pass so that the
// codestream takes at most target_bytes, cuts its data there, terminated, and builds the
// packet headers of every tile.  Returns the codestream length.
uint64_t p1_rate_control(j2k_tile *tiles, uint32_t num_tiles, uint64_t fixed_bytes, uint64_t target_bytes);
