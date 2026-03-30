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
#include <memory>
#include <vector>
#include "subband_row_buf.hpp"
#include "block_decoding.hpp"
#include "../common/open_htj2k_typedef.hpp"
#ifdef OPENHTJ2K_THREAD
  #include <thread>
  #include "ThreadPool.hpp"
#endif

// ─── init / free ──────────────────────────────────────────────────────────────

void j2k_subband_row_buf::init(j2k_resolution *resolution, uint8_t b_idx,
                               int32_t codeblock_height, uint8_t roi_shift, bool use_ring) {
  res            = resolution;
  band_idx       = b_idx;
  ROIshift       = roi_shift;
  cb_h           = codeblock_height;
  strip_y0       = -1;
  strip_y1       = -1;
  bypass_decode  = false;
  ring_mode      = use_ring;
  ring_buf       = nullptr;
  ring_y0        = -1;

#ifdef OPENHTJ2K_THREAD
  prefetch_buf    = nullptr;
  prefetch_y0     = -1;
  prefetch_y1     = -1;
  prefetch_active = false;
#endif

  sb = res->access_subband(band_idx);

  if (use_ring) {
    const int32_t sb_w = static_cast<int32_t>(sb->get_pos1().x - sb->get_pos0().x);
    const int32_t sb_h = static_cast<int32_t>(sb->get_pos1().y - sb->get_pos0().y);
    if (sb_w > 0 && sb_h > 0) {
      const size_t ring_rows = static_cast<size_t>(codeblock_height);
      const size_t buf_bytes = sizeof(sprec_t) * ring_rows * static_cast<size_t>(sb->stride);
      ring_buf = static_cast<sprec_t *>(aligned_mem_alloc(buf_bytes, 32));
#ifdef OPENHTJ2K_THREAD
      prefetch_buf = static_cast<sprec_t *>(aligned_mem_alloc(buf_bytes, 32));
#endif
    }
  }

  // Pre-allocate scratch for a 64×64 codeblock (grow-on-demand).
  cb_sample_cap  = static_cast<size_t>(round_up(64, 8) * round_up(64, 8));
  cb_state_cap   = static_cast<size_t>((round_up(64, 8) + 2) * (round_up(64, 8) + 2));
  cb_sample_buf  = static_cast<int32_t *>(std::malloc(cb_sample_cap * sizeof(int32_t)));
  cb_state_buf   = static_cast<uint8_t *>(std::malloc(cb_state_cap));
}

void j2k_subband_row_buf::free_resources() {
#ifdef OPENHTJ2K_THREAD
  if (prefetch_active) { prefetch_future.wait(); prefetch_active = false; }
  aligned_mem_free(prefetch_buf); prefetch_buf = nullptr;
#endif
  aligned_mem_free(ring_buf); ring_buf = nullptr; ring_y0 = -1;
  std::free(cb_sample_buf); cb_sample_buf = nullptr; cb_sample_cap = 0;
  std::free(cb_state_buf);  cb_state_buf  = nullptr; cb_state_cap  = 0;
  strip_y0 = strip_y1 = -1;
}

// ─── decode_strip_core ────────────────────────────────────────────────────────
// Decodes all non-empty codeblocks whose rows intersect [y0, y1) and writes
// their samples into target_buf (ring/prefetch mode) or directly into
// block->i_samples (non-ring mode, target_buf == nullptr).

void j2k_subband_row_buf::decode_strip_core(sprec_t *target_buf, int32_t y0, int32_t y1) {
  const uint32_t np = res->npw * res->nph;

#ifdef OPENHTJ2K_THREAD
  {
    auto *pool = ThreadPool::get();
    // Avoid nested dispatch: if this call is already running inside a pool worker
    // (e.g. triggered by component-parallel pull_line() or a prefetch task),
    // fall through to serial.
    const bool in_worker = pool && (pool->thread_number(std::this_thread::get_id()) >= 0);
    if (pool && pool->num_threads() > 1 && !in_worker) {
      // ── Parallel path ──────────────────────────────────────────────────────
      struct BlockTask {
        j2k_codeblock *block;
        uint32_t       QWx2, QHx2;
        size_t         sample_off, state_off;
        ptrdiff_t      ring_row_off, ring_col_off;
      };

      std::vector<BlockTask> tasks;
      size_t total_s = 0, total_st = 0;

      for (uint32_t p = 0; p < np; ++p) {
        j2k_precinct         *cp  = res->access_precinct(p);
        j2k_precinct_subband *cpb = cp->access_pband(band_idx);
        const uint32_t num_cblks  = cpb->num_codeblock_x * cpb->num_codeblock_y;
        for (uint32_t bi = 0; bi < num_cblks; ++bi) {
          j2k_codeblock *block = cpb->access_codeblock(bi);
          if (static_cast<int32_t>(block->get_pos1().y) <= y0) continue;
          if (static_cast<int32_t>(block->get_pos0().y) >= y1) continue;
          if (!block->num_passes) continue;
          const uint32_t QWx2 = round_up(block->size.x, 8U);
          const uint32_t QHx2 = round_up(block->size.y, 8U);
          BlockTask bt;
          bt.block      = block;
          bt.QWx2       = QWx2;
          bt.QHx2       = QHx2;
          bt.sample_off = total_s;
          bt.state_off  = total_st;
          bt.ring_row_off = (ring_mode && target_buf)
              ? (static_cast<int32_t>(block->get_pos0().y) - y0) * static_cast<ptrdiff_t>(sb->stride)
              : 0;
          bt.ring_col_off = (ring_mode && target_buf)
              ? static_cast<int32_t>(block->get_pos0().x) - static_cast<int32_t>(sb->get_pos0().x)
              : 0;
          total_s  += static_cast<size_t>(QWx2 * QHx2);
          total_st += static_cast<size_t>((QWx2 + 2) * (QHx2 + 2));
          tasks.push_back(bt);
        }
      }

      if (!tasks.empty()) {
        std::vector<int32_t> spool(total_s, 0);
        std::vector<uint8_t> stpool(total_st, 0);

        std::atomic<int> remaining{static_cast<int>(tasks.size())};
        for (auto &bt : tasks) {
          bt.block->sample_buf      = spool.data() + bt.sample_off;
          bt.block->blksampl_stride = bt.QWx2;
          bt.block->block_states    = stpool.data() + bt.state_off;
          bt.block->blkstate_stride = bt.QWx2 + 2;
          if (ring_mode && target_buf)
            bt.block->i_samples = target_buf + bt.ring_row_off + bt.ring_col_off;

          auto *blk = bt.block;
          auto  roi = ROIshift;
          pool->push([blk, roi, &remaining]() {
            if ((blk->Cmodes & HT) >> 6)
              htj2k_decode(blk, roi);
            else
              j2k_decode(blk, roi);
            remaining.fetch_sub(1, std::memory_order_release);
          });
        }
        while (remaining.load(std::memory_order_acquire) > 0)
          std::this_thread::yield();
      }
      return;
    }
  }
#endif

  // ── Serial path ────────────────────────────────────────────────────────────
  for (uint32_t p = 0; p < np; ++p) {
    j2k_precinct         *cp  = res->access_precinct(p);
    j2k_precinct_subband *cpb = cp->access_pband(band_idx);

    const uint32_t num_cblks = cpb->num_codeblock_x * cpb->num_codeblock_y;
    for (uint32_t bi = 0; bi < num_cblks; ++bi) {
      j2k_codeblock *block = cpb->access_codeblock(bi);

      if (static_cast<int32_t>(block->get_pos1().y) <= y0) continue;
      if (static_cast<int32_t>(block->get_pos0().y) >= y1) continue;
      if (!block->num_passes) continue;

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

      block->sample_buf      = cb_sample_buf;
      block->blksampl_stride = QWx2;
      block->block_states    = cb_state_buf;
      block->blkstate_stride = QWx2 + 2;

      if (ring_mode && target_buf != nullptr) {
        const ptrdiff_t row_off = (static_cast<int32_t>(block->get_pos0().y) - y0)
                                  * static_cast<ptrdiff_t>(sb->stride);
        const ptrdiff_t col_off = static_cast<int32_t>(block->get_pos0().x)
                                  - static_cast<int32_t>(sb->get_pos0().x);
        block->i_samples = target_buf + row_off + col_off;
      }

      if ((block->Cmodes & HT) >> 6)
        htj2k_decode(block, ROIshift);
      else
        j2k_decode(block, ROIshift);
    }
  }
}

// ─── decode_strip (on-demand wrapper) ────────────────────────────────────────

void j2k_subband_row_buf::decode_strip(int32_t abs_row) {
  const int32_t sb_y0 = static_cast<int32_t>(sb->get_pos0().y);
  const int32_t sb_y1 = static_cast<int32_t>(sb->get_pos1().y);
  const int32_t rel   = abs_row - sb_y0;
  const int32_t s_y0  = sb_y0 + (rel / cb_h) * cb_h;
  const int32_t s_y1  = std::min(s_y0 + cb_h, sb_y1);

  strip_y0 = s_y0;
  strip_y1 = s_y1;

  if (ring_mode) {
    ring_y0 = s_y0;
    if (ring_buf != nullptr) {
      const int32_t rows = s_y1 - s_y0;
      std::memset(ring_buf, 0, sizeof(sprec_t) * static_cast<size_t>(rows) * static_cast<size_t>(sb->stride));
    }
  }

  decode_strip_core(ring_mode ? ring_buf : nullptr, s_y0, s_y1);

#ifdef OPENHTJ2K_THREAD
  if (ring_mode) trigger_prefetch(s_y1);
#endif
}

#ifdef OPENHTJ2K_THREAD
// ─── trigger_prefetch ────────────────────────────────────────────────────────
// Submit a background task to decode the strip starting at next_y0 into
// prefetch_buf, so it is ready before row_ptr() is called for those rows.

void j2k_subband_row_buf::trigger_prefetch(int32_t next_y0) {
  if (prefetch_buf == nullptr || prefetch_active) return;

  const int32_t sb_y1 = static_cast<int32_t>(sb->get_pos1().y);
  if (next_y0 >= sb_y1) return;  // no more strips to prefetch

  auto *pool = ThreadPool::get();
  if (!pool || pool->num_threads() <= 1) return;
  // Don't enqueue from inside a pool worker to avoid pool exhaustion.
  if (pool->thread_number(std::this_thread::get_id()) >= 0) return;

  prefetch_y0     = next_y0;
  prefetch_y1     = std::min(next_y0 + cb_h, sb_y1);
  prefetch_active = true;

  // Zero target buffer before decode (same invariant as ring_buf in decode_strip).
  const int32_t rows = prefetch_y1 - prefetch_y0;
  std::memset(prefetch_buf, 0,
              sizeof(sprec_t) * static_cast<size_t>(rows) * static_cast<size_t>(sb->stride));

  // Capture by value everything the task needs; 'this' is stable for the subband lifetime.
  prefetch_future = pool->enqueue([this]() {
    decode_strip_core(prefetch_buf, prefetch_y0, prefetch_y1);
  });
}
#endif

// ─── public API ──────────────────────────────────────────────────────────────

const sprec_t *j2k_subband_row_buf::row_ptr(int32_t abs_row) {
  if (ring_mode) {
    if (ring_buf == nullptr) {
      static const sprec_t zero_row[4096] = {};
      return zero_row;
    }
    if (abs_row < strip_y0 || abs_row >= strip_y1) {
#ifdef OPENHTJ2K_THREAD
      if (prefetch_active && abs_row >= prefetch_y0 && abs_row < prefetch_y1) {
        // Prefetch hit: wait for background decode, then swap buffers.
        prefetch_future.wait();
        prefetch_active = false;
        std::swap(ring_buf, prefetch_buf);
        strip_y0 = ring_y0 = prefetch_y0;
        strip_y1            = prefetch_y1;
        trigger_prefetch(strip_y1);
      } else {
        // Prefetch miss or stale: cancel stale prefetch and decode synchronously.
        if (prefetch_active) { prefetch_future.wait(); prefetch_active = false; }
        decode_strip(abs_row);
      }
#else
      decode_strip(abs_row);
#endif
    }
    return ring_buf + static_cast<ptrdiff_t>(abs_row - ring_y0) * sb->stride;
  }
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
  const int32_t width = static_cast<int32_t>(sb->get_pos1().x - sb->get_pos0().x);
  if (width <= 0) {
    return;
  }
  // If there is no backing buffer for this subband row, return zeros without
  // reading from the small static zero_row used by row_ptr().
  if ((ring_mode && ring_buf == nullptr) || (!ring_mode && sb->i_samples == nullptr)) {
    std::memset(out, 0, sizeof(sprec_t) * static_cast<size_t>(width));
    return;
  }
  const sprec_t *p = row_ptr(abs_row);
  std::memcpy(out, p, sizeof(sprec_t) * static_cast<size_t>(width));
}
