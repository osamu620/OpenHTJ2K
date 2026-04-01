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
  #if defined(__x86_64__) || defined(_M_X64)
    #include <immintrin.h>
    #define HW_PAUSE() _mm_pause()
  #elif defined(__aarch64__) || defined(_M_ARM64)
    #include <arm_acle.h>
    #define HW_PAUSE() __isb(0xf)
  #else
    #define HW_PAUSE() ((void)0)
  #endif
#endif

#ifdef OPENHTJ2K_THREAD
// Spin-wait with hardware pause hints before falling back to OS yield.
// Avoids kernel scheduler spinlock contention for short waits.
static inline void spin_wait(std::atomic_int &cnt) {
  for (int spin = 0; spin < 1000; ++spin) {
    if (cnt.load(std::memory_order_acquire) == 0) return;
    HW_PAUSE();
  }
  while (cnt.load(std::memory_order_acquire) > 0)
    std::this_thread::yield();
}
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
  combined_buf    = nullptr;
  prefetch_y0     = -1;
  prefetch_y1     = -1;
  par_spool       = nullptr;
  par_stpool      = nullptr;
  par_spool_cap   = 0;
  par_stpool_cap  = 0;
  par_cnt.store(0, std::memory_order_relaxed);
#endif

  sb = res->access_subband(band_idx);

  if (use_ring) {
    const int32_t sb_w = static_cast<int32_t>(sb->get_pos1().x - sb->get_pos0().x);
    const int32_t sb_h = static_cast<int32_t>(sb->get_pos1().y - sb->get_pos0().y);
    if (sb_w > 0 && sb_h > 0) {
      const size_t ring_rows  = static_cast<size_t>(codeblock_height);
      const size_t buf_floats = ring_rows * static_cast<size_t>(sb->stride);
      const size_t buf_bytes  = sizeof(sprec_t) * buf_floats;
#ifdef OPENHTJ2K_THREAD
      // Allocate ring_buf and prefetch_buf as a single contiguous block:
      // [ring_buf | prefetch_buf], each buf_floats elements.
      // combined_buf holds the base pointer (ring_buf/prefetch_buf swap on prefetch hit,
      // so ring_buf may become an interior pointer — always free combined_buf).
      sprec_t *combined = static_cast<sprec_t *>(aligned_mem_alloc(buf_bytes * 2, 32));
      combined_buf = combined;
      ring_buf     = combined;
      prefetch_buf = combined + buf_floats;
      // Pre-fault all pages once at allocation time so that dequantize's first
      // write to i_samples doesn't trigger repeated page-table lock acquisitions.
      std::memset(combined, 0, buf_bytes * 2);
#else
      ring_buf = static_cast<sprec_t *>(aligned_mem_alloc(buf_bytes, 32));
      std::memset(ring_buf, 0, buf_bytes);
#endif
    }
  }

  // Pre-allocate scratch for a 64×64 codeblock (grow-on-demand).
  cb_sample_cap  = static_cast<size_t>(round_up(64, 8) * round_up(64, 8));
  cb_state_cap   = static_cast<size_t>((round_up(64, 8) + 2) * (round_up(64, 8) + 2));
  cb_sample_buf  = static_cast<int32_t *>(std::malloc(cb_sample_cap * sizeof(int32_t)));
  cb_state_buf   = static_cast<uint8_t *>(std::malloc(cb_state_cap));

#ifdef OPENHTJ2K_THREAD
  // Pre-scan all codeblock strips so par_spool / par_stpool / par_tasks can be
  // pre-allocated to the worst-case strip size, eliminating per-strip malloc.
  {
    const int32_t sb_y0_v = static_cast<int32_t>(sb->get_pos0().y);
    const int32_t sb_y1_v = static_cast<int32_t>(sb->get_pos1().y);
    size_t   max_s = 0, max_st = 0;
    uint32_t max_cblks = 0;
    const uint32_t np = resolution->npw * resolution->nph;
    for (int32_t sy = sb_y0_v; sy < sb_y1_v; sy += codeblock_height) {
      const int32_t ey = std::min(sy + codeblock_height, sb_y1_v);
      size_t   strip_s = 0, strip_st = 0;
      uint32_t strip_cblks = 0;
      for (uint32_t p = 0; p < np; ++p) {
        j2k_precinct_subband *cpb = resolution->access_precinct(p)->access_pband(b_idx);
        const uint32_t nc = cpb->num_codeblock_x * cpb->num_codeblock_y;
        for (uint32_t bi = 0; bi < nc; ++bi) {
          j2k_codeblock *blk = cpb->access_codeblock(bi);
          if (static_cast<int32_t>(blk->get_pos1().y) <= sy) continue;
          if (static_cast<int32_t>(blk->get_pos0().y) >= ey) continue;
          const uint32_t QWx2 = round_up(blk->size.x, 8U);
          const uint32_t QHx2 = round_up(blk->size.y, 8U);
          strip_s  += static_cast<size_t>(QWx2 * QHx2);
          strip_st += static_cast<size_t>((QWx2 + 2) * (QHx2 + 2));
          ++strip_cblks;
        }
      }
      if (strip_s  > max_s)     max_s     = strip_s;
      if (strip_st > max_st)    max_st    = strip_st;
      if (strip_cblks > max_cblks) max_cblks = strip_cblks;
    }
    // Ensure at least room for a single 64×64 codeblock.
    const size_t min_s  = static_cast<size_t>(round_up(64, 8) * round_up(64, 8));
    const size_t min_st = static_cast<size_t>((round_up(64, 8) + 2) * (round_up(64, 8) + 2));
    par_spool_cap  = std::max(max_s,  min_s);
    par_stpool_cap = std::max(max_st, min_st);
    par_spool  = static_cast<int32_t *>(std::malloc(par_spool_cap * sizeof(int32_t)));
    par_stpool = static_cast<uint8_t *>(std::malloc(par_stpool_cap));
    par_tasks.reserve(std::max(max_cblks, 16U));
  }
#endif
}

void j2k_subband_row_buf::free_resources() {
#ifdef OPENHTJ2K_THREAD
  // Drain any in-flight tasks before touching the scratch buffers.
  spin_wait(par_cnt);
  prefetch_y0 = prefetch_y1 = -1;
  // Free combined allocation via its stable base pointer — ring_buf may have been
  // swapped with prefetch_buf and could be an interior pointer after prefetch hits.
  aligned_mem_free(combined_buf);
  combined_buf = ring_buf = prefetch_buf = nullptr;
  ring_y0 = -1;
  std::free(par_spool);  par_spool  = nullptr;  par_spool_cap  = 0;
  std::free(par_stpool); par_stpool = nullptr;  par_stpool_cap = 0;
  par_tasks.clear();
  par_tasks.shrink_to_fit();
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
      par_tasks.clear();
      size_t total_s = 0, total_st = 0;

      for (uint32_t p = 0; p < np; ++p) {
        j2k_precinct         *cp  = res->access_precinct(p);
        j2k_precinct_subband *cpb = cp->access_pband(band_idx);
        const uint32_t num_cblks  = cpb->num_codeblock_x * cpb->num_codeblock_y;
        for (uint32_t bi = 0; bi < num_cblks; ++bi) {
          j2k_codeblock *block = cpb->access_codeblock(bi);
          if (static_cast<int32_t>(block->get_pos1().y) <= y0) continue;
          if (static_cast<int32_t>(block->get_pos0().y) >= y1) continue;
          if (!block->num_passes) {
            // Empty block: dequantize never runs, so zero its region in target_buf
            // explicitly (replaces the bulk ring_buf pre-zero in decode_strip).
            if (ring_mode && target_buf) {
              const ptrdiff_t roff =
                  (static_cast<int32_t>(block->get_pos0().y) - y0) * static_cast<ptrdiff_t>(sb->stride);
              const ptrdiff_t coff = static_cast<int32_t>(block->get_pos0().x)
                                     - static_cast<int32_t>(sb->get_pos0().x);
              sprec_t *dst = target_buf + roff + coff;
              for (uint32_t r = 0; r < block->size.y; r++)
                std::memset(dst + r * sb->stride, 0, block->size.x * sizeof(sprec_t));
            }
            continue;
          }
          const uint32_t QWx2 = round_up(block->size.x, 8U);
          const uint32_t QHx2 = round_up(block->size.y, 8U);
          CblkTask bt;
          bt.block      = block;
          bt.QWx2       = QWx2;
          bt.QHx2       = QHx2;
          bt.sample_off = total_s;
          bt.state_off  = total_st;
          bt.row_off    = (ring_mode && target_buf)
              ? (static_cast<int32_t>(block->get_pos0().y) - y0) * static_cast<ptrdiff_t>(sb->stride)
              : 0;
          bt.col_off    = (ring_mode && target_buf)
              ? static_cast<int32_t>(block->get_pos0().x) - static_cast<int32_t>(sb->get_pos0().x)
              : 0;
          total_s  += static_cast<size_t>(QWx2 * QHx2);
          total_st += static_cast<size_t>((QWx2 + 2) * (QHx2 + 2));
          par_tasks.push_back(bt);
        }
      }

      if (!par_tasks.empty()) {
        // Grow-only: realloc only when capacity is insufficient.
        if (total_s > par_spool_cap) {
          std::free(par_spool);
          par_spool     = static_cast<int32_t *>(std::malloc(total_s * sizeof(int32_t)));
          par_spool_cap = total_s;
        }
        if (total_st > par_stpool_cap) {
          std::free(par_stpool);
          par_stpool     = static_cast<uint8_t *>(std::malloc(total_st));
          par_stpool_cap = total_st;
        }
        // Setup pass: assign scratch pointers and selectively zero buffers.
        // ht_cleanup_decode writes every sample_buf/block_states position before
        // reading, so HT single-pass blocks need no pre-zeroing. EBCOT and HT
        // multi-pass blocks still require it (EBCOT reads before writing; HT
        // multi-pass sigprop/magref read the block_states border written only by
        // the cleanup interior pass, leaving the border uninitialised).
        par_cnt.store(static_cast<int>(par_tasks.size()), std::memory_order_relaxed);
        for (auto &bt : par_tasks) {
          bt.block->sample_buf      = par_spool  + bt.sample_off;
          bt.block->blksampl_stride = bt.QWx2;
          bt.block->block_states    = par_stpool + bt.state_off;
          bt.block->blkstate_stride = bt.QWx2 + 2;
          if (ring_mode && target_buf)
            bt.block->i_samples = target_buf + bt.row_off + bt.col_off;
          const bool is_ht = (bt.block->Cmodes & HT) >> 6;
          if (!is_ht) {
            std::memset(bt.block->sample_buf, 0, bt.QWx2 * bt.QHx2 * sizeof(int32_t));
            std::memset(bt.block->block_states, 0, (bt.QWx2 + 2) * (bt.QHx2 + 2));
          } else if (bt.block->num_passes > 1) {
            std::memset(bt.block->block_states, 0, (bt.QWx2 + 2) * (bt.QHx2 + 2));
          }
        }
        // Batch-push all tasks under a single mutex lock + notify_all.
        const uint8_t roi = ROIshift;
        pool->push_batch(par_tasks, [roi, this](const CblkTask &bt) {
          auto *blk = bt.block;
          return [blk, roi, this]() {
            if ((blk->Cmodes & HT) >> 6)
              htj2k_decode(blk, roi);
            else
              j2k_decode(blk, roi);
            par_cnt.fetch_sub(1, std::memory_order_release);
          };
        });
        spin_wait(par_cnt);
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
      if (!block->num_passes) {
        // Empty block: zero its region in target_buf (replaces bulk pre-zero).
        if (ring_mode && target_buf) {
          const ptrdiff_t roff =
              (static_cast<int32_t>(block->get_pos0().y) - y0) * static_cast<ptrdiff_t>(sb->stride);
          const ptrdiff_t coff = static_cast<int32_t>(block->get_pos0().x)
                                 - static_cast<int32_t>(sb->get_pos0().x);
          sprec_t *dst = target_buf + roff + coff;
          for (uint32_t r = 0; r < block->size.y; r++)
            std::memset(dst + r * sb->stride, 0, block->size.x * sizeof(sprec_t));
        }
        continue;
      }

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

      // ht_cleanup_decode writes every position before reading, so no
      // pre-zeroing is needed for single-pass HT blocks. For EBCOT and
      // multi-pass HT blocks, zero the necessary regions.
      const bool is_ht = (block->Cmodes & HT) >> 6;
      if (!is_ht) {
        std::memset(cb_sample_buf, 0, need_s * sizeof(int32_t));
        std::memset(cb_state_buf,  0, need_st);
      } else if (block->num_passes > 1) {
        std::memset(cb_state_buf,  0, need_st);
      }

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
    // No bulk pre-zero: decode_strip_core zeroes empty-block regions selectively.
  }

  decode_strip_core(ring_mode ? ring_buf : nullptr, s_y0, s_y1);

#ifdef OPENHTJ2K_THREAD
  if (ring_mode) trigger_prefetch(s_y1);
#endif
}

#ifdef OPENHTJ2K_THREAD
// ─── trigger_prefetch ────────────────────────────────────────────────────────
// Enumerate all codeblocks for the strip starting at next_y0 and dispatch each
// as an independent pool task.  A shared atomic counter tracks outstanding tasks;
// row_ptr() spins on it before swapping buffers.

void j2k_subband_row_buf::trigger_prefetch(int32_t next_y0) {
  if (prefetch_buf == nullptr || prefetch_y0 != -1) return;  // no buf or already pending

  const int32_t sb_y1 = static_cast<int32_t>(sb->get_pos1().y);
  if (next_y0 >= sb_y1) return;

  auto *pool = ThreadPool::get();
  if (!pool || pool->num_threads() <= 1) return;

  prefetch_y0 = next_y0;
  prefetch_y1 = std::min(next_y0 + cb_h, sb_y1);

  // ── Enumerate codeblocks for [prefetch_y0, prefetch_y1) ──────────────────
  // No bulk pre-zero of prefetch_buf: empty blocks are zeroed selectively below.
  par_tasks.clear();
  const uint32_t np = res->npw * res->nph;
  size_t total_s = 0, total_st = 0;

  for (uint32_t p = 0; p < np; ++p) {
    j2k_precinct         *cp  = res->access_precinct(p);
    j2k_precinct_subband *cpb = cp->access_pband(band_idx);
    const uint32_t num_cblks  = cpb->num_codeblock_x * cpb->num_codeblock_y;
    for (uint32_t bi = 0; bi < num_cblks; ++bi) {
      j2k_codeblock *block = cpb->access_codeblock(bi);
      if (static_cast<int32_t>(block->get_pos1().y) <= prefetch_y0) continue;
      if (static_cast<int32_t>(block->get_pos0().y) >= prefetch_y1) continue;
      if (!block->num_passes) {
        // Empty block: zero its region in prefetch_buf now (before workers start).
        if (prefetch_buf) {
          const ptrdiff_t roff = (static_cast<int32_t>(block->get_pos0().y) - prefetch_y0)
                                 * static_cast<ptrdiff_t>(sb->stride);
          const ptrdiff_t coff = static_cast<int32_t>(block->get_pos0().x)
                                 - static_cast<int32_t>(sb->get_pos0().x);
          sprec_t *dst = prefetch_buf + roff + coff;
          for (uint32_t r = 0; r < block->size.y; r++)
            std::memset(dst + r * sb->stride, 0, block->size.x * sizeof(sprec_t));
        }
        continue;
      }
      CblkTask pb;
      pb.block      = block;
      pb.QWx2       = round_up(block->size.x, 8U);
      pb.QHx2       = round_up(block->size.y, 8U);
      pb.sample_off = total_s;
      pb.state_off  = total_st;
      pb.row_off    = (static_cast<int32_t>(block->get_pos0().y) - prefetch_y0)
                      * static_cast<ptrdiff_t>(sb->stride);
      pb.col_off    = static_cast<int32_t>(block->get_pos0().x)
                      - static_cast<int32_t>(sb->get_pos0().x);
      total_s  += static_cast<size_t>(pb.QWx2 * pb.QHx2);
      total_st += static_cast<size_t>((pb.QWx2 + 2) * (pb.QHx2 + 2));
      par_tasks.push_back(pb);
    }
  }

  if (par_tasks.empty()) {
    // No codeblocks to decode — prefetch is trivially complete.
    par_cnt.store(0, std::memory_order_relaxed);
    return;
  }

  // ── Grow-only scratch pools; shared with decode_strip_core (never concurrent) ──
  if (total_s > par_spool_cap) {
    std::free(par_spool);
    par_spool     = static_cast<int32_t *>(std::malloc(total_s * sizeof(int32_t)));
    par_spool_cap = total_s;
  }
  if (total_st > par_stpool_cap) {
    std::free(par_stpool);
    par_stpool     = static_cast<uint8_t *>(std::malloc(total_st));
    par_stpool_cap = total_st;
  }

  par_cnt.store(static_cast<int>(par_tasks.size()), std::memory_order_relaxed);

  // Setup pass: assign scratch/output pointers and selectively zero buffers.
  // HT single-pass blocks need no pre-zeroing (ht_cleanup_decode writes all
  // positions before reading). EBCOT and multi-pass HT blocks still need it.
  sprec_t *pbuf     = prefetch_buf;
  const uint8_t roi = ROIshift;
  for (auto &pb : par_tasks) {
    pb.block->sample_buf      = par_spool  + pb.sample_off;
    pb.block->blksampl_stride = pb.QWx2;
    pb.block->block_states    = par_stpool + pb.state_off;
    pb.block->blkstate_stride = pb.QWx2 + 2;
    pb.block->i_samples       = pbuf + pb.row_off + pb.col_off;
    const bool is_ht = (pb.block->Cmodes & HT) >> 6;
    if (!is_ht) {
      std::memset(pb.block->sample_buf,   0, pb.QWx2 * pb.QHx2 * sizeof(int32_t));
      std::memset(pb.block->block_states, 0, (pb.QWx2 + 2) * (pb.QHx2 + 2));
    } else if (pb.block->num_passes > 1) {
      std::memset(pb.block->block_states, 0, (pb.QWx2 + 2) * (pb.QHx2 + 2));
    }
  }
  // Batch-push all tasks under a single mutex lock + notify_all.
  pool->push_batch(par_tasks, [roi, this](const CblkTask &pb) {
    auto *blk = pb.block;
    return [blk, roi, this]() {
      if ((blk->Cmodes & HT) >> 6)
        htj2k_decode(blk, roi);
      else
        j2k_decode(blk, roi);
      par_cnt.fetch_sub(1, std::memory_order_release);
    };
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
      if (prefetch_y0 != -1 && abs_row >= prefetch_y0 && abs_row < prefetch_y1) {
        // Prefetch hit: wait for all in-flight tasks to finish, then swap buffers.
        spin_wait(par_cnt);
        std::swap(ring_buf, prefetch_buf);
        strip_y0 = ring_y0 = prefetch_y0;
        strip_y1            = prefetch_y1;
        prefetch_y0 = prefetch_y1 = -1;
        trigger_prefetch(strip_y1);
      } else {
        // Prefetch miss or stale: drain any in-flight tasks, then decode synchronously.
        spin_wait(par_cnt);
        prefetch_y0 = prefetch_y1 = -1;
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
