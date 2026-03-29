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

#include <cstring>
#include <cstdlib>
#include <algorithm>
#include "subband_row_buf.hpp"
#include "block_decoding.hpp"
#include "../common/open_htj2k_typedef.hpp"

// ─── init / free ──────────────────────────────────────────────────────────────

void j2k_subband_row_buf::init(j2k_resolution *resolution, uint8_t b_idx,
                               int32_t codeblock_height, uint8_t roi_shift) {
  res            = resolution;
  band_idx       = b_idx;
  ROIshift       = roi_shift;
  cb_h           = codeblock_height;
  strip_y0       = -1;
  strip_y1       = -1;
  bypass_decode  = false;

  sb = res->access_subband(band_idx);

  // Pre-allocate scratch for a 64×64 codeblock (grow-on-demand).
  cb_sample_cap  = static_cast<size_t>(round_up(64, 8) * round_up(64, 8));
  cb_state_cap   = static_cast<size_t>((round_up(64, 8) + 2) * (round_up(64, 8) + 2));
  cb_sample_buf  = static_cast<int32_t *>(std::malloc(cb_sample_cap * sizeof(int32_t)));
  cb_state_buf   = static_cast<uint8_t *>(std::malloc(cb_state_cap));
}

void j2k_subband_row_buf::free_resources() {
  std::free(cb_sample_buf); cb_sample_buf = nullptr; cb_sample_cap = 0;
  std::free(cb_state_buf);  cb_state_buf  = nullptr; cb_state_cap  = 0;
  strip_y0 = strip_y1 = -1;
}

// ─── decode_strip ─────────────────────────────────────────────────────────────

void j2k_subband_row_buf::decode_strip(int32_t abs_row) {
  // Compute strip bounds in subband coordinate space.
  const int32_t sb_y0 = static_cast<int32_t>(sb->get_pos0().y);
  const int32_t sb_y1 = static_cast<int32_t>(sb->get_pos1().y);
  const int32_t rel   = abs_row - sb_y0;
  const int32_t s_y0  = sb_y0 + (rel / cb_h) * cb_h;
  const int32_t s_y1  = std::min(s_y0 + cb_h, sb_y1);

  strip_y0 = s_y0;
  strip_y1 = s_y1;

  // Iterate over all precincts; find codeblocks in this band that overlap the strip.
  const uint32_t np = res->npw * res->nph;
  for (uint32_t p = 0; p < np; ++p) {
    j2k_precinct         *cp  = res->access_precinct(p);
    j2k_precinct_subband *cpb = cp->access_pband(band_idx);

    const uint32_t num_cblks = cpb->num_codeblock_x * cpb->num_codeblock_y;
    for (uint32_t bi = 0; bi < num_cblks; ++bi) {
      j2k_codeblock *block = cpb->access_codeblock(bi);

      // Skip codeblocks outside this strip.
      if (static_cast<int32_t>(block->get_pos1().y) <= s_y0) continue;
      if (static_cast<int32_t>(block->get_pos0().y) >= s_y1) continue;
      // Skip empty codeblocks (no coding passes).
      if (!block->num_passes) continue;

      // Grow scratch if needed for this codeblock size.
      const uint32_t QWx2 = round_up(block->size.x, 8U);
      const uint32_t QHx2 = round_up(block->size.y, 8U);
      const size_t need_s  = static_cast<size_t>(QWx2 * QHx2);
      const size_t need_st = static_cast<size_t>((QWx2 + 2) * (QHx2 + 2));

      if (need_s > cb_sample_cap) {
        std::free(cb_sample_buf);
        cb_sample_buf = static_cast<int32_t *>(std::malloc(need_s * sizeof(int32_t)));
        cb_sample_cap = need_s;
      }
      if (need_st > cb_state_cap) {
        std::free(cb_state_buf);
        cb_state_buf = static_cast<uint8_t *>(std::malloc(need_st));
        cb_state_cap = need_st;
      }

      std::memset(cb_sample_buf, 0, need_s * sizeof(int32_t));
      std::memset(cb_state_buf,  0, need_st);

      block->sample_buf    = cb_sample_buf;
      block->blksampl_stride = QWx2;
      block->block_states  = cb_state_buf;
      block->blkstate_stride = QWx2 + 2;

      if ((block->Cmodes & HT) >> 6)
        htj2k_decode(block, ROIshift);
      else
        j2k_decode(block, ROIshift);
    }
  }
}

// ─── public API ──────────────────────────────────────────────────────────────

const sprec_t *j2k_subband_row_buf::row_ptr(int32_t abs_row) {
  // Guard: empty subband (zero-height or zero-width tile boundary case).
  // i_samples is null when pos1.x==pos0.x or pos1.y==pos0.y (num_samples==0).
  // Return a pointer into a static zero buffer; the caller memcpy's width bytes
  // which are all zero — correct since an empty subband has no HF content.
  if (sb->i_samples == nullptr) {
    static const sprec_t zero_row[4096] = {};
    return zero_row;
  }
  if (!bypass_decode && (abs_row < strip_y0 || abs_row >= strip_y1)) decode_strip(abs_row);
  const int32_t rel = abs_row - static_cast<int32_t>(sb->get_pos0().y);
  return sb->i_samples + static_cast<ptrdiff_t>(rel) * sb->stride;
}

void j2k_subband_row_buf::get_row(int32_t abs_row, sprec_t *out) {
  const sprec_t *p = row_ptr(abs_row);
  std::memcpy(out, p, sizeof(sprec_t) * static_cast<size_t>(sb->get_pos1().x - sb->get_pos0().x));
}
