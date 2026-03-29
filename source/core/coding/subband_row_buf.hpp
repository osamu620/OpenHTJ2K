// Copyright (c) 2019 - 2021, Osamu Watanabe
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
#include <cstddef>
#include "coding_units.hpp"

// ─────────────────────────────────────────────────────────────────────────────
// j2k_subband_row_buf — lazy per-strip codeblock decoder for one subband
//
// Wraps a j2k_subband and its parent j2k_resolution.  Codeblocks are decoded
// strip-by-strip on demand: the first call to row_ptr() or get_row() whose
// abs_row falls outside the currently-decoded strip triggers the decode of all
// codeblocks in that codeblock-row strip (across every precinct).
//
// After decode, samples reside in j2k_subband::i_samples so no extra copy is
// needed.  Scratch buffers (sample_buf / block_states) for the single-codeblock
// serial decode are owned here and reused across strips.
//
// Usage pattern (sequential top-to-bottom access expected):
//   j2k_subband_row_buf rb;
//   rb.init(res, band_idx, cb_h, ROIshift);
//   const sprec_t *p = rb.row_ptr(abs_row);   // decode strip if needed
// ─────────────────────────────────────────────────────────────────────────────
struct j2k_subband_row_buf {
  j2k_subband    *sb;         // geometry, i_samples, decode params
  j2k_resolution *res;        // to enumerate precincts
  uint8_t         band_idx;   // index within resolution's subbands (0=HL,1=LH,2=HH)
  uint8_t         ROIshift;

  int32_t cb_h;       // codeblock height for this resolution (max across precincts)
  int32_t strip_y0;   // y-start of the currently-decoded codeblock strip (-1 = none)
  int32_t strip_y1;   // y-end  of the currently-decoded codeblock strip (exclusive)

  // When true, skip decode_strip() in row_ptr() — caller has pre-populated sb->i_samples.
  bool    bypass_decode;

  // Ring buffer for line-based mode.
  // When ring_mode=true, decoded samples go here instead of sb->i_samples.
  bool     ring_mode;   // use ring buffer instead of sb->i_samples
  sprec_t *ring_buf;    // cb_h × sb->stride floats (one strip wide)
  int32_t  ring_y0;     // first row of current strip in ring_buf (= strip_y0)

  // Scratch buffers reused across codeblocks (serial decode; one block at a time).
  int32_t *cb_sample_buf;
  uint8_t *cb_state_buf;
  size_t   cb_sample_cap;  // current capacity in elements
  size_t   cb_state_cap;

  // Initialise. cb_h is the maximum codeblock height for this resolution level.
  // When use_ring=true, allocates a ring buffer (cb_h rows) instead of using sb->i_samples.
  void init(j2k_resolution *res, uint8_t band_idx, int32_t cb_h, uint8_t ROIshift,
            bool use_ring = false);

  // Release scratch buffers.
  void free_resources();

  // Return pointer into sb->i_samples for abs_row.
  // Decodes the containing codeblock strip if not yet decoded.
  const sprec_t *row_ptr(int32_t abs_row);

  // Copy abs_row into out[0 .. sb->pos1.x - sb->pos0.x - 1].
  void get_row(int32_t abs_row, sprec_t *out);

 private:
  // Decode all codeblocks in the strip covering abs_row.
  void decode_strip(int32_t abs_row);
};
