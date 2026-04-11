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
#include <cstddef>
#ifdef OPENHTJ2K_THREAD
  #include <atomic>
  #include <memory>
#endif
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

#ifdef OPENHTJ2K_THREAD
  // Double-buffer for strip prefetch: while IDWT consumes ring_buf (current strip),
  // background tasks decode the next strip into prefetch_buf.
  // ring_buf and prefetch_buf are the two halves of a single combined aligned allocation
  // (combined_buf).  std::swap(ring_buf, prefetch_buf) happens on prefetch hit, so after
  // a swap ring_buf may point to the upper half.  Always free combined_buf, not ring_buf.
  sprec_t *prefetch_buf;   // decode target for next strip (upper half of combined alloc)
  sprec_t *combined_buf;   // base pointer of the ring+prefetch combined allocation
  int32_t  prefetch_y0;    // strip bounds of pending prefetch task (-1 = none)
  int32_t  prefetch_y1;

  // Unified codeblock task descriptor used by both decode_strip_core() (parallel path)
  // and trigger_prefetch().  The two paths are mutually exclusive — decode_strip_core
  // waits synchronously for all tasks before returning, and trigger_prefetch is called
  // only after that — so a single set of scratch resources serves both.
  struct CblkTask {
    j2k_codeblock *block;
    uint32_t       QWx2, QHx2;
    size_t         sample_off, state_off;
    ptrdiff_t      row_off, col_off;  // ring/prefetch target offset; 0 in non-ring mode
  };

  // Grow-only scratch pools (never freed until free_resources()).
  // Sized at init() from a per-subband strip pre-scan; only realloc'd on growth.
  int32_t *par_spool;
  uint8_t *par_stpool;
  size_t   par_spool_cap;
  size_t   par_stpool_cap;

  // Task list pre-reserved to max codeblocks per strip (from init() pre-scan).
  std::vector<CblkTask> par_tasks;

  // In-flight counter shared by decode_strip_core and trigger_prefetch.
  // Replaces both the local 'remaining' atomic and the old shared_ptr<atomic> prefetch_cnt.
  std::atomic<int> par_cnt;

  // Per-strip cached codeblock enumeration.  Populated lazily on the first
  // trigger_prefetch() call for each strip; reused on every subsequent frame
  // under single-tile reuse because the codeblock *tree* (positions, sizes,
  // count) is stable across frames once the main-header fingerprint is
  // unchanged.  Retires the access_precinct / access_pband / overlap-filter
  // walk that perf showed as ~14% of total cycles on 4K HT.
  //
  // Key subtlety: a codeblock's empty-vs-nonempty status (block->num_passes)
  // is re-parsed per frame from the packet stream and is NOT stable across
  // frames.  The cache therefore stores EVERY block in the strip regardless
  // of emptiness, and the hot path branches on num_passes at dispatch time.
  // The per-block branch is cheap relative to the tree walk we avoid.
  //
  // Indexed by strip_idx = (next_y0 - sb_y0) / cb_h.
  struct CachedBlock {
    j2k_codeblock *block;
    uint32_t       QWx2, QHx2;      // round_up(size.x, 8), round_up(size.y, 8)
    ptrdiff_t      row_off, col_off; // strip-relative offsets into prefetch_buf
    uint32_t       size_x, size_y;   // block->size, captured for empty memset
  };
  struct StripCacheEntry {
    bool                      built = false;
    std::vector<CachedBlock>  blocks;
  };
  std::vector<StripCacheEntry> strip_cache_;
#endif

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
  // Core per-strip codeblock decode. Writes decoded samples to target_buf
  // (when ring_mode && target_buf != nullptr); otherwise leaves block->i_samples
  // pointing into sb->i_samples (non-ring / bypass path).
  void decode_strip_core(sprec_t *target_buf, int32_t y0, int32_t y1);

  // Decode-on-demand wrapper: computes strip bounds, zeros ring_buf, calls core,
  // and (when OPENHTJ2K_THREAD) triggers a prefetch for the next strip.
  void decode_strip(int32_t abs_row);

#ifdef OPENHTJ2K_THREAD
  // Submit a background task to decode the strip starting at next_y0 into prefetch_buf.
  void trigger_prefetch(int32_t next_y0);
#endif
};
