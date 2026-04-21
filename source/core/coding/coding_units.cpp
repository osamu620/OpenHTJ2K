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

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#ifdef OPENHTJ2K_THREAD
  #include <thread>
#endif
#include "coding_units.hpp"
#include "block_decoding.hpp"
#include "decode_timing.hpp"
#include "dwt.hpp"
#include "color.hpp"
#include "finalize_narrow.hpp"
#include "subband_row_buf.hpp"
#if defined(OPENHTJ2K_ENABLE_WASM_SIMD)
  #include <wasm_simd128.h>
#endif

namespace {
// RAII guard for placement-new array construction loops.
//
// Pattern:
//   T *arr = static_cast<T *>(operator new[](sizeof(T) * n));
//   placement_new_array_guard<T> guard(arr);
//   for (uint32_t i = 0; i < n; ++i) {
//     new (&arr[i]) T(...);
//     guard.commit_one();
//   }
//   guard.release();          // success — leave the array intact for the owner
//
// If any T(...) throws, ~placement_new_array_guard destroys the elements
// already constructed (0..constructed-1), frees the raw storage, and the
// exception propagates to the caller.  This keeps the placement-new pattern
// (used for non-default-constructible types like j2k_subband, j2k_precinct,
// j2k_codeblock) exception-safe without requiring extra member fields on the
// owning class.
template <typename T>
struct placement_new_array_guard {
  T *base;
  uint32_t constructed = 0;
  explicit placement_new_array_guard(T *b) : base(b) {}
  ~placement_new_array_guard() {
    if (base) {
      for (uint32_t i = 0; i < constructed; ++i) base[i].~T();
      operator delete[](base);
    }
  }
  placement_new_array_guard(const placement_new_array_guard &)            = delete;
  placement_new_array_guard &operator=(const placement_new_array_guard &) = delete;
  void commit_one() noexcept { ++constructed; }
  void release() noexcept { base = nullptr; }
};
}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// Line-decode supporting types (file-scope, internal to this TU)
// ─────────────────────────────────────────────────────────────────────────────

// Source context for one idwt_2d_state level.
struct idwt_level_src_ctx {
  int32_t v0;           // first row of this resolution in full-tile space
  int32_t u0, u1;       // horizontal extent of this resolution
  uint8_t transformation;
  dwt_type dir;         // DWT type for this level (BIDIR, HORZ, NO)

  // LL row source: either pull from child idwt_2d_state (levels > 1) or from ll0_buf (level 1).
  bool             has_child;
  idwt_2d_state   *child_state;        // non-null when has_child
  j2k_subband_row_buf *ll0_buf;        // non-null when !has_child

  // HL / LH / HH subband row buffers for this resolution level.
  j2k_subband_row_buf *hl_buf;
  j2k_subband_row_buf *lh_buf;         // null for HORZ and NO levels
  j2k_subband_row_buf *hh_buf;         // null for HORZ and NO levels

  // Scratch rows for interleaving (allocated per level).
  // lp_tmp: used when has_child=true to receive output of idwt_2d_state_pull_row.
  //         nullptr (not allocated) when has_child=false (direct pointer to ll0_buf used).
  // hp_tmp: no longer allocated — HP data is read directly from subband row buffers.
  sprec_t *lp_tmp;   // LL row scratch (non-null only when has_child=true)
  // ext_buf removed: ring buffer slots now include PSE prefix/suffix (IDWT_RING_PSE_LEFT),
  // so in-place horizontal PSE fill + filter is applied directly on the slot data area.

  // Subband dimensions (set once at init).
  int32_t lp_width;    // width of LL / LH subband
  int32_t hp_width;    // width of HL / HH subband
  int32_t ll_y0;       // pos0.y of LL (used only when !has_child)
  int32_t ll0_height;  // row count of LL0 subband; PSE needed when sub_idx >= ll0_height
  int32_t hl_y0, lh_y0, hh_y0;
  // PSE counts for in-place horizontal filter (precomputed at init).
  int32_t h_pse_left;   // left PSE samples for this level (function of u0%2, transformation)
  int32_t h_pse_right;  // right PSE samples for this level (function of u1%2, transformation)

  // Horizontal IDWT output range (Phase 4B spatial-region decode).  Default
  // = [u0, u1] → full-width horizontal lifting, byte-identical to pre-patch
  // behaviour.  set_line_decode_col_range() can narrow this to the viewport
  // range widened by per-level filter support, which routes the callback
  // into a sub-range horizontal lifter that skips columns whose subband
  // values are zero.
  int32_t col_lo;
  int32_t col_hi;
};

// Whole-sample symmetric extension: reflect idx into [0, len).
// Handles len==1 (single element) and len==0 (empty → returns 0 safely).
static inline int32_t pse_row_idx(int32_t idx, int32_t len) {
  if (len <= 1) return 0;
  const int32_t period = 2 * (len - 1);
  idx = ((idx % period) + period) % period;
  return (idx < len) ? idx : period - idx;
}

// Early-exit check for LP+HP both-zero.  Scans 16 uint32 words at a time and
// returns false the moment any non-zero bit pattern (including -0.0f) is seen.
// For dense data the first block fails the test, so cost is O(64 B) per row.
// For JPIP sparse frames the scan runs to completion and the caller skips the
// full interleave + horizontal-IDWT path for this row.
static inline bool lp_hp_both_zero(const sprec_t *lp, int32_t lp_w,
                                   const sprec_t *hp, int32_t hp_w) {
  auto scan = [](const sprec_t *row, int32_t w) -> bool {
    const uint32_t *p = reinterpret_cast<const uint32_t *>(row);
    int32_t i = 0;
    for (; i + 16 <= w; i += 16) {
      uint32_t acc = 0;
      for (int32_t j = 0; j < 16; ++j) acc |= p[i + j];
      if (acc) return false;
    }
    uint32_t acc = 0;
    for (; i < w; ++i) acc |= p[i];
    return acc == 0;
  };
  return scan(lp, lp_w) && scan(hp, hp_w);
}

// Callback invoked by idwt_2d_state::fetch_one() for each source row.
static void idwt_level_src_fn(void *ctx, int32_t abs_row, sprec_t *out) {
  auto *c = static_cast<idwt_level_src_ctx *>(ctx);

  // ── DWT_NO: passthrough — copy LL row directly ────────────────────────────
  if (c->dir == DWT_NO) {
    const int32_t sub_idx = abs_row - c->v0;
    const sprec_t *lp_ptr;
    if (c->has_child) {
      lp_ptr = idwt_2d_state_pull_row_ref(c->child_state);
      if (lp_ptr == nullptr) { memset(out, 0, sizeof(sprec_t) * static_cast<size_t>(c->lp_width)); return; }
    } else {
      const int32_t clamped = pse_row_idx(sub_idx, c->ll0_height);
      lp_ptr = c->ll0_buf->row_ptr(c->ll_y0 + clamped);
    }
    memcpy(out, lp_ptr, sizeof(sprec_t) * static_cast<size_t>(c->lp_width));
    return;
  }

  // ── DWT_HORZ: all rows are LP-style — interleave LL+H then horizontal IDWT ─
  // For HORZ levels, there is no vertical split: abs_row maps 1:1 to subband rows.
  if (c->dir == DWT_HORZ) {
    const int32_t sub_idx = abs_row - c->v0;  // direct row index (no /2)
    const sprec_t *lp_ptr;
    if (c->has_child) {
      lp_ptr = idwt_2d_state_pull_row_ref(c->child_state);
      if (lp_ptr == nullptr) {
        memset(c->lp_tmp, 0, sizeof(sprec_t) * static_cast<size_t>(c->lp_width));
        lp_ptr = c->lp_tmp;
      }
    } else {
      const int32_t clamped = pse_row_idx(sub_idx, c->ll0_height);
      lp_ptr = c->ll0_buf->row_ptr(c->ll_y0 + clamped);
    }
    const sprec_t *hp_ptr = c->hl_buf->row_ptr(c->hl_y0 + sub_idx);
    const int32_t width = c->u1 - c->u0;
    if (width <= 0) return;
    // Fast path: both subbands all-zero → horizontal IDWT output is all-zero.
    if (lp_hp_both_zero(lp_ptr, c->lp_width, hp_ptr, c->hp_width)) {
      memset(out, 0, sizeof(sprec_t) * static_cast<size_t>(width));
      return;
    }
    // Fall through to shared interleave + horizontal IDWT code below.
    // Interleave: LP at u0%2, HP at 1-u0%2.
    const int32_t u_off  = c->u0 & 1;
    const int32_t min_w  = std::min(c->lp_width, c->hp_width);
    if (u_off == 0) {
      for (int32_t j = 0; j < c->lp_width; ++j)  out[2 * j]     = lp_ptr[j];
      for (int32_t j = 0; j < c->hp_width; ++j)  out[2 * j + 1] = hp_ptr[j];
    } else {
      for (int32_t j = 0; j < c->hp_width; ++j)  out[2 * j]     = hp_ptr[j];
      for (int32_t j = 0; j < c->lp_width; ++j)  out[2 * j + 1] = lp_ptr[j];
    }
    (void)min_w;  // suppress unused-variable warning (used in BIDIR path below)
    if (width == 1) {
      if ((c->u0 % 2 != 0) && (c->transformation == 1)) out[0] /= 2.0f;
      return;
    }
    idwt_1d_row_inplace_range(out, c->h_pse_left, c->h_pse_right, c->u0, c->u1, c->transformation,
                              c->col_lo, c->col_hi);
    return;
  }

  // ── DWT_BIDIR: LP rows (even) carry LL+HL, HP rows (odd) carry LH+HH ─────
  // LP rows are at even absolute positions; HP at odd.
  const bool lp = (abs_row & 1) == 0;
  // LP sub_idx = floor(abs_row/2) - ll_y0,  where ll_y0 = ceil(v0/2) = (v0+1)>>1
  // HP sub_idx = floor(abs_row/2) - lh_y0,  where lh_y0 = floor(v0/2) = v0>>1
  const int32_t sub_idx = lp ? (abs_row >> 1) - ((c->v0 + 1) >> 1)
                             : (abs_row >> 1) - (c->v0 >> 1);

  // Obtain direct const-pointers to the LP and HP subband data, avoiding
  // unnecessary memcpy to scratch buffers. lp_tmp is only needed when
  // has_child=true (the child state writes its output into lp_tmp).
  const sprec_t *lp_ptr;
  const sprec_t *hp_ptr;

  if (lp) {
    if (c->has_child) {
      // Zero-copy: use pointer directly into child state's ring buffer.
      lp_ptr = idwt_2d_state_pull_row_ref(c->child_state);
      if (lp_ptr == nullptr) {
        memset(c->lp_tmp, 0, sizeof(sprec_t) * static_cast<size_t>(c->lp_width));
        lp_ptr = c->lp_tmp;
      }
    } else {
      const int32_t clamped = pse_row_idx(sub_idx, c->ll0_height);
      lp_ptr = c->ll0_buf->row_ptr(c->ll_y0 + clamped);  // no copy
    }
    hp_ptr = c->hl_buf->row_ptr(c->hl_y0 + sub_idx);     // no copy
  } else {
    lp_ptr = c->lh_buf->row_ptr(c->lh_y0 + sub_idx);     // no copy
    hp_ptr = c->hh_buf->row_ptr(c->hh_y0 + sub_idx);     // no copy
  }

  // Fast path: both subbands all-zero → horizontal IDWT output is all-zero.
  // For dense codestreams the early-exit scan returns false after ~1 cache
  // line; for sparse JPIP frames this eliminates the per-row interleave +
  // 1D-IDWT cost for absent precincts.
  {
    const int32_t width_z = c->u1 - c->u0;
    if (width_z > 0 && lp_hp_both_zero(lp_ptr, c->lp_width, hp_ptr, c->hp_width)) {
      memset(out, 0, sizeof(sprec_t) * static_cast<size_t>(width_z));
      return;
    }
  }

  // Interleave: LP always at u0%2 column-offset, HP at 1-u0%2.
  const int32_t u_off = c->u0 & 1;
  const int32_t min_w = std::min(c->lp_width, c->hp_width);
#if defined(OPENHTJ2K_TRY_AVX2) && defined(__AVX2__)
  {
    // AVX2: process 8 LP + 8 HP → 16 interleaved floats per iteration.
    // _mm256_unpacklo/hi_ps interleave lanes, permute2f128 reorders 128-bit halves.
    const sprec_t *a_ptr = (u_off == 0) ? lp_ptr : hp_ptr;  // → even slots
    const sprec_t *b_ptr = (u_off == 0) ? hp_ptr : lp_ptr;  // → odd slots
    int32_t i = 0;
    for (; i + 8 <= min_w; i += 8) {
      __m256 va  = _mm256_loadu_ps(a_ptr + i);
      __m256 vb  = _mm256_loadu_ps(b_ptr + i);
      __m256 lo  = _mm256_unpacklo_ps(va, vb);  // a0,b0,a1,b1, a4,b4,a5,b5
      __m256 hi  = _mm256_unpackhi_ps(va, vb);  // a2,b2,a3,b3, a6,b6,a7,b7
      __m256 r0  = _mm256_permute2f128_ps(lo, hi, 0x20);  // a0..b3
      __m256 r1  = _mm256_permute2f128_ps(lo, hi, 0x31);  // a4..b7
      _mm256_storeu_ps(out + 2 * i,     r0);
      _mm256_storeu_ps(out + 2 * i + 8, r1);
    }
    // Two independent scalar tails starting from i (independent loop variables).
    if (u_off == 0) {
      for (int32_t j = i; j < c->lp_width; ++j)  out[2 * j]     = lp_ptr[j];
      for (int32_t j = i; j < c->hp_width; ++j)  out[2 * j + 1] = hp_ptr[j];
    } else {
      for (int32_t j = i; j < c->hp_width; ++j)  out[2 * j]     = hp_ptr[j];
      for (int32_t j = i; j < c->lp_width; ++j)  out[2 * j + 1] = lp_ptr[j];
    }
  }
#elif defined(OPENHTJ2K_ENABLE_WASM_SIMD)
  {
    const sprec_t *a_ptr = (u_off == 0) ? lp_ptr : hp_ptr;
    const sprec_t *b_ptr = (u_off == 0) ? hp_ptr : lp_ptr;
    int32_t i = 0;
    for (; i + 4 <= min_w; i += 4) {
      v128_t va = wasm_v128_load(a_ptr + i);
      v128_t vb = wasm_v128_load(b_ptr + i);
      // Interleave 4 floats: lo=a0,b0,a1,b1  hi=a2,b2,a3,b3
      v128_t lo = wasm_i8x16_shuffle(va, vb, 0,1,2,3, 16,17,18,19, 4,5,6,7, 20,21,22,23);
      v128_t hi = wasm_i8x16_shuffle(va, vb, 8,9,10,11, 24,25,26,27, 12,13,14,15, 28,29,30,31);
      wasm_v128_store(out + 2 * i,     lo);
      wasm_v128_store(out + 2 * i + 4, hi);
    }
    if (u_off == 0) {
      for (int32_t j = i; j < c->lp_width; ++j)  out[2 * j]     = lp_ptr[j];
      for (int32_t j = i; j < c->hp_width; ++j)  out[2 * j + 1] = hp_ptr[j];
    } else {
      for (int32_t j = i; j < c->hp_width; ++j)  out[2 * j]     = hp_ptr[j];
      for (int32_t j = i; j < c->lp_width; ++j)  out[2 * j + 1] = lp_ptr[j];
    }
  }
#elif defined(OPENHTJ2K_ENABLE_ARM_NEON)
  {
    // NEON: vzipq_f32 interleaves two float32x4 vectors. 4× unrolled to process 16 pairs/iter.
    const sprec_t *a_ptr = (u_off == 0) ? lp_ptr : hp_ptr;
    const sprec_t *b_ptr = (u_off == 0) ? hp_ptr : lp_ptr;
    int32_t i = 0;
    for (; i + 16 <= min_w; i += 16) {
      float32x4x2_t z0 = vzipq_f32(vld1q_f32(a_ptr + i),      vld1q_f32(b_ptr + i));
      float32x4x2_t z1 = vzipq_f32(vld1q_f32(a_ptr + i + 4),  vld1q_f32(b_ptr + i + 4));
      float32x4x2_t z2 = vzipq_f32(vld1q_f32(a_ptr + i + 8),  vld1q_f32(b_ptr + i + 8));
      float32x4x2_t z3 = vzipq_f32(vld1q_f32(a_ptr + i + 12), vld1q_f32(b_ptr + i + 12));
      vst1q_f32(out + 2 * i,      z0.val[0]); vst1q_f32(out + 2 * i + 4,  z0.val[1]);
      vst1q_f32(out + 2 * i + 8,  z1.val[0]); vst1q_f32(out + 2 * i + 12, z1.val[1]);
      vst1q_f32(out + 2 * i + 16, z2.val[0]); vst1q_f32(out + 2 * i + 20, z2.val[1]);
      vst1q_f32(out + 2 * i + 24, z3.val[0]); vst1q_f32(out + 2 * i + 28, z3.val[1]);
    }
    for (; i + 8 <= min_w; i += 8) {
      float32x4x2_t z0 = vzipq_f32(vld1q_f32(a_ptr + i),     vld1q_f32(b_ptr + i));
      float32x4x2_t z1 = vzipq_f32(vld1q_f32(a_ptr + i + 4), vld1q_f32(b_ptr + i + 4));
      vst1q_f32(out + 2 * i,      z0.val[0]);
      vst1q_f32(out + 2 * i + 4,  z0.val[1]);
      vst1q_f32(out + 2 * i + 8,  z1.val[0]);
      vst1q_f32(out + 2 * i + 12, z1.val[1]);
    }
    for (; i + 4 <= min_w; i += 4) {
      float32x4x2_t zipped = vzipq_f32(vld1q_f32(a_ptr + i), vld1q_f32(b_ptr + i));
      vst1q_f32(out + 2 * i,     zipped.val[0]);
      vst1q_f32(out + 2 * i + 4, zipped.val[1]);
    }
    if (u_off == 0) {
      for (int32_t j = i; j < c->lp_width; ++j)  out[2 * j]     = lp_ptr[j];
      for (int32_t j = i; j < c->hp_width; ++j)  out[2 * j + 1] = hp_ptr[j];
    } else {
      for (int32_t j = i; j < c->hp_width; ++j)  out[2 * j]     = hp_ptr[j];
      for (int32_t j = i; j < c->lp_width; ++j)  out[2 * j + 1] = lp_ptr[j];
    }
  }
#else
  (void)min_w;
  if (u_off == 0) {
    for (int32_t i = 0; i < c->lp_width; ++i) out[2 * i]     = lp_ptr[i];
    for (int32_t i = 0; i < c->hp_width; ++i) out[2 * i + 1] = hp_ptr[i];
  } else {
    for (int32_t i = 0; i < c->hp_width; ++i) out[2 * i]     = hp_ptr[i];
    for (int32_t i = 0; i < c->lp_width; ++i) out[2 * i + 1] = lp_ptr[i];
  }
#endif

  // In-place horizontal IDWT: the ring buffer slot has IDWT_RING_PSE_LEFT scratch floats
  // before out[0], so we can fill PSE directly into out[-left..-1] and out[width..width+right-1]
  // then filter in-place — no ext_buf copy needed.
  const int32_t width = c->u1 - c->u0;
  if (width <= 0) return;
  if (width == 1) {
    // Single-sample edge case (same as idwt_1d_row_fixed).
    if ((c->u0 % 2 != 0) && (c->transformation == 1)) out[0] /= 2.0f;
    return;
  }
  idwt_1d_row_inplace_range(out, c->h_pse_left, c->h_pse_right, c->u0, c->u1, c->transformation,
                            c->col_lo, c->col_hi);
}

// ─────────────────────────────────────────────────────────────────────────────
// Line-encode supporting types
// ─────────────────────────────────────────────────────────────────────────────

// Forward declaration so fdwt_level_sink_ctx can hold a pointer without the
// full ThreadPool class being visible here (included later under OPENHTJ2K_THREAD).
#ifdef OPENHTJ2K_THREAD
class ThreadPool;
#endif

// Sink context for one fdwt_2d_state level.
struct fdwt_level_sink_ctx {
  int32_t u0;
  int32_t lp_width;    // width of LP (LL/LH) output coefficients
  int32_t hp_width;    // width of HP (HL/HH) output coefficients
  sprec_t *lp_tmp;     // scratch for LP deinterleave
  sprec_t *hp_tmp;     // scratch for HP deinterleave

  // HL subband (LP-vert, HP-horiz)
  sprec_t  *hl_samples;
  uint32_t  hl_stride;
  int32_t   hl_y0;
  int32_t   hl_h;      // HL band height in rows

  // LH subband (HP-vert, LP-horiz)
  sprec_t  *lh_samples;
  uint32_t  lh_stride;
  int32_t   lh_y0;
  int32_t   lh_h;      // LH band height in rows (same as HH band height)

  // HH subband (HP-vert, HP-horiz)
  sprec_t  *hh_samples;
  uint32_t  hh_stride;
  int32_t   hh_y0;

  // LP rows are forwarded to the coarser FDWT state (has_child) or the LL0 buffer.
  bool           has_child;
  fdwt_2d_state *child_state;  // non-null when has_child

  // LL0 destination (coarsest level only, when !has_child)
  sprec_t  *ll0_samples;
  uint32_t  ll0_stride;
  int32_t   ll0_y0;

  // --- Overlap HT block encoding (populated by encode_line_based_stream) ---
  // cb_h == 0 means overlap is disabled for this level.
  int32_t   cb_h          = 0;
  int32_t   num_cblk_rows = 0;
  // Per-codeblock-row completion flags: bit0 = HL done, bit1 = LH+HH done.
  // Dispatch fires when both bits are set (flags == 3).
  std::unique_ptr<std::atomic<uint8_t>[]> cblk_row_done;

  j2k_resolution *enc_cr      = nullptr;
  uint8_t         enc_ROIshift = 0;
  j2k_tile::EncodePoolCtx *enc_epc      = nullptr;
#ifdef OPENHTJ2K_THREAD
  ThreadPool        *enc_pool      = nullptr;
  std::atomic<int>  *enc_remaining = nullptr;
#endif
};

// Dispatch HT block encoding for all codeblocks in logical codeblock row `br`
// (0-indexed relative to each band's subband-row origin) at the resolution
// Forward declaration; full definition is after the ThreadPool include below.
static void enc_overlap_dispatch(fdwt_level_sink_ctx *c, int32_t br);

// Sink callback invoked by fdwt_2d_state after each row has been through H-DWT.
// The interleaved_row has u1-u0 samples: LP at u0%2 offsets, HP at (1-u0%2) offsets.
static void fdwt_level_sink_fn(void *ctx, bool is_hp, int32_t abs_row,
                               const sprec_t *interleaved_row) {
  auto *c = static_cast<fdwt_level_sink_ctx *>(ctx);

  const int32_t u_off = c->u0 & 1;
  for (int32_t i = 0; i < c->lp_width; ++i) c->lp_tmp[i] = interleaved_row[2 * i + u_off];
  for (int32_t i = 0; i < c->hp_width; ++i) c->hp_tmp[i] = interleaved_row[2 * i + (1 - u_off)];

  if (!is_hp) {
    // LP vertical row: HP-horiz → HL, LP-horiz → child/LL0
    const int32_t hl_sub = (abs_row >> 1) - c->hl_y0;
    memcpy(c->hl_samples + static_cast<ptrdiff_t>(hl_sub) * static_cast<ptrdiff_t>(c->hl_stride), c->hp_tmp,
           static_cast<size_t>(c->hp_width) * sizeof(sprec_t));
    if (c->has_child) {
      fdwt_2d_state_push_row(c->child_state, c->lp_tmp);
    } else {
      const int32_t ll_sub = (abs_row >> 1) - c->ll0_y0;
      memcpy(c->ll0_samples + static_cast<ptrdiff_t>(ll_sub) * static_cast<ptrdiff_t>(c->ll0_stride), c->lp_tmp,
             static_cast<size_t>(c->lp_width) * sizeof(sprec_t));
    }
    // Overlap: dispatch HT block encoding if this HL row completes codeblock row br.
    if (c->cb_h > 0) {
      const int32_t br   = hl_sub / c->cb_h;
      const int32_t last = std::min((br + 1) * c->cb_h, c->hl_h) - 1;
      if (hl_sub == last) {
        const uint8_t old = c->cblk_row_done[static_cast<size_t>(br)].fetch_or(1, std::memory_order_acq_rel);
        if ((old | 1) == 3) enc_overlap_dispatch(c, br);
      }
    }
  } else {
    // HP vertical row: LP-horiz → LH, HP-horiz → HH
    const int32_t hp_sub = (abs_row >> 1) - c->lh_y0;
    memcpy(c->lh_samples + static_cast<ptrdiff_t>(hp_sub) * static_cast<ptrdiff_t>(c->lh_stride), c->lp_tmp,
           static_cast<size_t>(c->lp_width) * sizeof(sprec_t));
    memcpy(c->hh_samples + static_cast<ptrdiff_t>(hp_sub) * static_cast<ptrdiff_t>(c->hh_stride), c->hp_tmp,
           static_cast<size_t>(c->hp_width) * sizeof(sprec_t));
    // Overlap: dispatch HT block encoding if this LH+HH row completes codeblock row br.
    if (c->cb_h > 0) {
      const int32_t br   = hp_sub / c->cb_h;
      const int32_t last = std::min((br + 1) * c->cb_h, c->lh_h) - 1;
      if (hp_sub == last) {
        const uint8_t old = c->cblk_row_done[static_cast<size_t>(br)].fetch_or(2, std::memory_order_acq_rel);
        if ((old | 2) == 3) enc_overlap_dispatch(c, br);
      }
    }
  }
}

// Per-component line-encode state.
struct j2k_tcomp_line_enc {
  int32_t NL_active;                              // number of FDWT levels (== component NL for encoder)
  std::unique_ptr<fdwt_2d_state[]>       states;  // [NL_active]; states[NL_active-1] = finest
  std::unique_ptr<fdwt_level_sink_ctx[]> ctxs;    // [NL_active]
};

// Per-component line-decode state (definition; forward-declared in coding_units.hpp as opaque ptr).
struct j2k_tcomp_line_dec {
  int32_t NL_active;   // NL - reduce_NL
  int32_t next_row;    // abs row cursor (used only when NL_active==0)

  j2k_subband_row_buf  ll0_buf;   // LL0 at the coarsest active resolution

  // Per active IDWT level arrays (length NL_active each, heap-allocated).
  std::unique_ptr<idwt_2d_state[]>       states;
  std::unique_ptr<idwt_level_src_ctx[]>  ctxs;
  std::unique_ptr<j2k_subband_row_buf[]> hl_bufs;
  std::unique_ptr<j2k_subband_row_buf[]> lh_bufs;
  std::unique_ptr<j2k_subband_row_buf[]> hh_bufs;

  // Grow-only scratch buffer holding one outer strip worth of finalized IDWT
  // rows for this component.  Filled by pull_strip_into_buf() via
  // pull_line_ref() + memcpy, consumed by the finalize + callback inner loop
  // in j2k_tile::decode_line_based_stream.  The memcpy is necessary because
  // idwt_2d_state's internal ring is only IDWT_STATE_RING_DEPTH (8) slots
  // deep — too shallow to keep a whole strip of rows pinned.
  //
  // 32-byte aligned so the AVX2/AVX-512 finalize inner loops can use
  // unaligned loads without penalty.  Persistent across frames under
  // single-tile reuse; freed in finalize_line_decode() on teardown.
  sprec_t *strip_buf       = nullptr;
  size_t   strip_buf_floats = 0;   // current capacity, in sprec_t elements
};



// Thread-local pointer to the active cblk_data_pool for HTJ2K encode.
// Set on each thread (main or worker) before calling htj2k_encode(),
// cleared after. When null, set_compressed_data falls back to malloc().
static thread_local cblk_data_pool *g_cblk_pool = nullptr;

// Per-thread pool slot for encoder pool assignment.
// gen tracks the tile-encode generation; when it differs from EncodePoolCtx::gen,
// the thread claims a new slot (one atomic fetch_add per thread per tile encode).
#ifdef OPENHTJ2K_THREAD
namespace {
struct TlPoolSlot {
  int slot       = -1;
  uint32_t gen   = ~uint32_t{0};
};
}  // namespace
static thread_local TlPoolSlot g_tl_pool_slot;
#endif

#ifdef OPENHTJ2K_THREAD
namespace {
// Lightweight task descriptor for decoder: holds everything needed by a worker thread.
struct DecTaskArgs {
  j2k_codeblock *block;
  uint8_t ROIshift;
  std::atomic<int> *remaining;
};
}  // namespace
#endif

#if defined(OPENHTJ2K_TRY_AVX2) && defined(__AVX512F__)
OPENHTJ2K_MAYBE_UNUSED static cvt_color_func cvt_rgb_to_ycbcr[2] = {cvt_rgb_to_ycbcr_irrev_avx512, cvt_rgb_to_ycbcr_rev_avx512};
static cvt_color_float_func cvt_ycbcr_to_rgb_float[2] = {cvt_ycbcr_to_rgb_irrev_float_avx512,
                                                          cvt_ycbcr_to_rgb_rev_float_avx512};
static cvt_color_i32_to_f_func cvt_rgb_to_ycbcr_float[2] = {cvt_rgb_to_ycbcr_irrev_float_avx512,
                                                              cvt_rgb_to_ycbcr_rev_float_avx512};
static fused_mct_finalize_func fused_mct_finalize[2] = {fused_ycbcr_irrev_to_rgb_i32_avx512,
                                                         fused_ycbcr_rev_to_rgb_i32_avx512};
#elif defined(OPENHTJ2K_TRY_AVX2) && defined(__AVX2__)
OPENHTJ2K_MAYBE_UNUSED static cvt_color_func cvt_rgb_to_ycbcr[2] = {cvt_rgb_to_ycbcr_irrev_avx2, cvt_rgb_to_ycbcr_rev_avx2};
static cvt_color_float_func cvt_ycbcr_to_rgb_float[2] = {cvt_ycbcr_to_rgb_irrev_float_avx2,
                                                          cvt_ycbcr_to_rgb_rev_float_avx2};
static cvt_color_i32_to_f_func cvt_rgb_to_ycbcr_float[2] = {cvt_rgb_to_ycbcr_irrev_float_avx2,
                                                              cvt_rgb_to_ycbcr_rev_float_avx2};
// Fused inverse MCT + float→int32 finalize dispatch table.
// Index 0: irrev (ICT, lossy 9/7); Index 1: rev (RCT, lossless 5/3).
static fused_mct_finalize_func fused_mct_finalize[2] = {fused_ycbcr_irrev_to_rgb_i32_avx2,
                                                         fused_ycbcr_rev_to_rgb_i32_avx2};
#elif defined(OPENHTJ2K_ENABLE_WASM_SIMD)
OPENHTJ2K_MAYBE_UNUSED static cvt_color_func cvt_rgb_to_ycbcr[2] = {cvt_rgb_to_ycbcr_irrev_wasm, cvt_rgb_to_ycbcr_rev_wasm};
static cvt_color_float_func cvt_ycbcr_to_rgb_float[2] = {cvt_ycbcr_to_rgb_irrev_float_wasm,
                                                          cvt_ycbcr_to_rgb_rev_float_wasm};
static cvt_color_i32_to_f_func cvt_rgb_to_ycbcr_float[2] = {cvt_rgb_to_ycbcr_irrev_float_wasm,
                                                              cvt_rgb_to_ycbcr_rev_float_wasm};
static fused_mct_finalize_func fused_mct_finalize[2] = {fused_ycbcr_irrev_to_rgb_i32_wasm,
                                                         fused_ycbcr_rev_to_rgb_i32_wasm};
#elif defined(OPENHTJ2K_ENABLE_ARM_NEON)
OPENHTJ2K_MAYBE_UNUSED static cvt_color_func cvt_rgb_to_ycbcr[2] = {cvt_rgb_to_ycbcr_irrev_neon, cvt_rgb_to_ycbcr_rev_neon};
static cvt_color_float_func cvt_ycbcr_to_rgb_float[2] = {cvt_ycbcr_to_rgb_irrev_float_neon,
                                                          cvt_ycbcr_to_rgb_rev_float_neon};
static cvt_color_i32_to_f_func cvt_rgb_to_ycbcr_float[2] = {cvt_rgb_to_ycbcr_irrev_float_neon,
                                                              cvt_rgb_to_ycbcr_rev_float_neon};
static fused_mct_finalize_func fused_mct_finalize[2] = {fused_ycbcr_irrev_to_rgb_i32_neon,
                                                         fused_ycbcr_rev_to_rgb_i32_neon};
#else
OPENHTJ2K_MAYBE_UNUSED static cvt_color_func cvt_rgb_to_ycbcr[2] = {cvt_rgb_to_ycbcr_irrev, cvt_rgb_to_ycbcr_rev};
static cvt_color_float_func cvt_ycbcr_to_rgb_float[2] = {cvt_ycbcr_to_rgb_irrev_float,
                                                          cvt_ycbcr_to_rgb_rev_float};
static cvt_color_i32_to_f_func cvt_rgb_to_ycbcr_float[2] = {cvt_rgb_to_ycbcr_irrev_float,
                                                              cvt_rgb_to_ycbcr_rev_float};
static fused_mct_finalize_func fused_mct_finalize[2] = {fused_ycbcr_irrev_to_rgb_i32,
                                                         fused_ycbcr_rev_to_rgb_i32};
#endif

#ifdef OPENHTJ2K_THREAD
  #include "ThreadPool.hpp"
std::atomic<ThreadPool *> ThreadPool::singleton_{nullptr};
std::mutex ThreadPool::singleton_mutex;

// Local "wait for N outstanding worker tasks" helper used by
// j2k_tile::decode_line_based_stream to barrier the per-component
// strip-pull tasks.  Mirrors subband_row_buf.cpp's spin_wait but is
// private to this translation unit.  The calling thread drains work
// from the shared pool queue via try_run_one() while blocked, so it
// contributes to draining its own tasks' nested HT subtasks.
namespace {
inline void dec_strip_barrier_wait(std::atomic<int> &cnt) {
  auto *p = ThreadPool::get();
  while (cnt.load(std::memory_order_acquire) > 0) {
    if (p && p->try_run_one()) continue;
    std::this_thread::yield();
  }
}
}  // namespace
#endif
// #include <hwy/highway.h>

// Full definition of enc_overlap_dispatch (declared earlier).
// Placed here so ThreadPool, g_cblk_pool, TlPoolSlot, and htj2k_encode are visible.
static void enc_overlap_dispatch(fdwt_level_sink_ctx *c, int32_t br) {
  j2k_resolution *cr = c->enc_cr;
  const int32_t cb_h = c->cb_h;
  for (uint32_t p = 0; p < cr->npw * cr->nph; ++p) {
    j2k_precinct *cp = cr->access_precinct(p);
    for (uint8_t b = 0; b < cr->num_bands; ++b) {
      j2k_precinct_subband *cpb = cp->access_pband(b);
      if (!cpb->num_codeblock_x || !cpb->num_codeblock_y) continue;
      // LH and HH share the HP-vert origin (lh_y0); HL uses hl_y0.
      const int32_t band_y0    = (b == 0) ? c->hl_y0 : c->lh_y0;
      const int32_t abs_cblk_y = (band_y0 + br * cb_h) / cb_h;
      const int32_t base_y     = static_cast<int32_t>(cpb->pos0.y) / cb_h;
      const int32_t cy_local   = abs_cblk_y - base_y;
      if (cy_local < 0 || cy_local >= static_cast<int32_t>(cpb->num_codeblock_y)) continue;
      for (uint32_t cx = 0; cx < cpb->num_codeblock_x; ++cx) {
        j2k_codeblock *block =
            cpb->access_codeblock(cx + static_cast<uint32_t>(cy_local) * cpb->num_codeblock_x);
#ifdef OPENHTJ2K_THREAD
        if (c->enc_pool && c->enc_pool->num_threads() > 1) {
          using EPC = j2k_tile::EncodePoolCtx;
          c->enc_remaining->fetch_add(1, std::memory_order_relaxed);
          EPC              *epc       = c->enc_epc;
          uint8_t           ROIshift  = c->enc_ROIshift;
          std::atomic<int> *remaining = c->enc_remaining;
          c->enc_pool->push([epc, block, ROIshift, remaining]() {
            TlPoolSlot &ts = g_tl_pool_slot;
            if (ts.gen != epc->gen) {
              const int slot = epc->slot_cnt.fetch_add(1, std::memory_order_relaxed);
              ts.slot        = std::min(slot, static_cast<int>(epc->pools.size()) - 1);
              ts.gen         = epc->gen;
            }
            g_cblk_pool = epc->pools[static_cast<size_t>(ts.slot)].get();
            htj2k_encode(block, ROIshift);
            g_cblk_pool = nullptr;
            remaining->fetch_sub(1, std::memory_order_release);
          });
        } else {
          g_cblk_pool = c->enc_epc->pools[0].get();
          htj2k_encode(block, c->enc_ROIshift);
          g_cblk_pool = nullptr;
        }
#else
        g_cblk_pool = c->enc_epc->pools[0].get();
        htj2k_encode(block, c->enc_ROIshift);
        g_cblk_pool = nullptr;
#endif
      }
    }
  }
}

float bibo_step_gains[32][5] = {{1.00000000F, 4.17226868F, 1.44209458F, 2.10966980F, 1.69807026F},
                                {1.38034954F, 4.58473765F, 1.83866981F, 2.13405021F, 1.63956779F},
                                {1.33279329F, 4.58985327F, 1.75793599F, 2.07403081F, 1.60751898F},
                                {1.30674103F, 4.48819441F, 1.74087517F, 2.00811395F, 1.60270904F},
                                {1.30283106F, 4.44564235F, 1.72542071F, 2.00171155F, 1.59940161F},
                                {1.30014247F, 4.43925026F, 1.72264700F, 1.99727052F, 1.59832420F},
                                {1.29926666F, 4.43776733F, 1.72157554F, 1.99642626F, 1.59828968F},
                                {1.29923860F, 4.43704105F, 1.72132351F, 1.99619334F, 1.59826880F},
                                {1.29922163F, 4.43682858F, 1.72125886F, 1.99616484F, 1.59826245F},
                                {1.29921646F, 4.43680359F, 1.72124892F, 1.99615185F, 1.59826037F},
                                {1.29921477F, 4.43679132F, 1.72124493F, 1.99614775F, 1.59825980F},
                                {1.29921431F, 4.43678921F, 1.72124414F, 1.99614684F, 1.59825953F},
                                {1.29921409F, 4.43678858F, 1.72124384F, 1.99614656F, 1.59825948F},
                                {1.29921405F, 4.43678831F, 1.72124381F, 1.99614653F, 1.59825947F},
                                {1.29921404F, 4.43678829F, 1.72124381F, 1.99614652F, 1.59825947F},
                                {1.29921404F, 4.43678829F, 1.72124381F, 1.99614652F, 1.59825947F},
                                {1.29921404F, 4.43678829F, 1.72124381F, 1.99614652F, 1.59825947F},
                                {1.29921404F, 4.43678829F, 1.72124381F, 1.99614652F, 1.59825947F},
                                {1.29921404F, 4.43678829F, 1.72124381F, 1.99614652F, 1.59825947F},
                                {1.29921404F, 4.43678829F, 1.72124381F, 1.99614652F, 1.59825947F},
                                {1.29921404F, 4.43678829F, 1.72124381F, 1.99614652F, 1.59825947F},
                                {1.29921404F, 4.43678829F, 1.72124381F, 1.99614652F, 1.59825947F},
                                {1.29921404F, 4.43678829F, 1.72124381F, 1.99614652F, 1.59825947F},
                                {1.29921404F, 4.43678829F, 1.72124381F, 1.99614652F, 1.59825947F},
                                {1.29921404F, 4.43678829F, 1.72124381F, 1.99614652F, 1.59825947F},
                                {1.29921404F, 4.43678829F, 1.72124381F, 1.99614652F, 1.59825947F},
                                {1.29921404F, 4.43678829F, 1.72124381F, 1.99614652F, 1.59825947F},
                                {1.29921404F, 4.43678829F, 1.72124381F, 1.99614652F, 1.59825947F},
                                {1.29921404F, 4.43678829F, 1.72124381F, 1.99614652F, 1.59825947F},
                                {1.29921404F, 4.43678829F, 1.72124381F, 1.99614652F, 1.59825947F},
                                {1.29921404F, 4.43678829F, 1.72124381F, 1.99614652F, 1.59825947F},
                                {1.29921404F, 4.43678829F, 1.72124381F, 1.99614652F, 1.59825947F}};

static void find_child_ranges(float *child_ranges, uint8_t &normalizing_upshift, float &normalization,
                              uint8_t lev, uint32_t u0, uint32_t u1, uint32_t v0, uint32_t v1,
                              dwt_type dir = DWT_BIDIR, float K = 1.230174104914001F) {
  if (u0 == u1 || v0 == v1) {
    return;
  }
  const float low_gain  = 1.0f / K;
  const float high_gain = K / 2.0f;

  // initialization
  const bool unit_width  = (u0 == u1 - 1);
  const bool unit_height = (v0 == v1 - 1);
  float bibo_max         = normalization;
  normalizing_upshift    = 0;
  for (uint8_t b = 0; b < 4; ++b) {
    child_ranges[b] = normalization;
  }

  // DFS direction selects which axes are transformed
  const bool do_vert = (dir == DWT_BIDIR || dir == DWT_VERT) && !unit_height;
  const bool do_horz = (dir == DWT_BIDIR || dir == DWT_HORZ) && !unit_width;

  // vertical analysis gain, if any
  if (do_vert) {
    child_ranges[BAND_LL] /= low_gain;
    child_ranges[BAND_HL] /= low_gain;
    child_ranges[BAND_LH] /= high_gain;
    child_ranges[BAND_HH] /= high_gain;
    float bibo_prev, bibo_in, bibo_out;
    bibo_prev = bibo_step_gains[lev][0] * normalization;
    bibo_in   = bibo_prev * bibo_step_gains[lev][0];
    for (uint8_t n = 0; n < 4; ++n) {
      bibo_out = bibo_prev * bibo_step_gains[lev][n + 1];
      bibo_max = std::max(bibo_max, bibo_out);
      bibo_max = std::max(bibo_max, bibo_in);
      bibo_in  = bibo_out;
    }
  }
  // horizontal analysis gain, if any
  if (do_horz) {
    child_ranges[BAND_LL] /= low_gain;
    child_ranges[BAND_HL] /= high_gain;
    child_ranges[BAND_LH] /= low_gain;
    child_ranges[BAND_HH] /= high_gain;
    float bibo_prev, bibo_in, bibo_out;
    bibo_prev = std::max(bibo_step_gains[lev][4], bibo_step_gains[lev][3]);
    bibo_prev *= normalization;
    bibo_in = bibo_prev * bibo_step_gains[lev][0];
    for (uint8_t n = 0; n < 4; ++n) {
      bibo_out = bibo_prev * bibo_step_gains[lev][n + 1];
      bibo_max = std::max(bibo_max, bibo_out);
      bibo_max = std::max(bibo_max, bibo_in);
      bibo_in  = bibo_out;
    }
  }

  // float overflow_limit = 1.0f * (1 << (16 - FRACBITS));
  // while (bibo_max > 0.95f * overflow_limit) {
  //   normalizing_upshift++;
  //   for (uint8_t b = 0; b < 4; ++b) {
  //     child_ranges[b] *= 0.5f;
  //   }
  //   bibo_max *= 0.5f;
  // }
  normalization = child_ranges[BAND_LL];
}

/********************************************************************************
 * j2k_codeblock
 *******************************************************************************/

j2k_codeblock::j2k_codeblock(const uint32_t &idx, uint8_t orientation, uint8_t M_b, uint8_t R_b,
                             uint8_t transformation, float stepsize, uint32_t band_stride, sprec_t *ibuf,
                             uint32_t offset, const uint16_t &numlayers, const uint8_t &codeblock_style,
                             const element_siz &p0, const element_siz &p1, const element_siz &s)
    : j2k_region(p0, p1),
      // public
      size(s),
      // private
      compressed_data(nullptr),
      current_address(nullptr),
      band(orientation),
      M_b(M_b),
      index(idx),
      //  public
      i_samples(ibuf ? ibuf + offset : nullptr),
      band_stride(band_stride),
      R_b(R_b),
      transformation(transformation),
      stepsize(stepsize),
      num_layers(numlayers),
      length(0),
      Cmodes(codeblock_style),
      num_passes(0),
      num_ZBP(0),
      fast_skip_passes(0),
      Lblock(0),
      already_included(false),
      refsegment(false) {
  const uint32_t QWx2 = round_up(size.x, 8U);
  blksampl_stride    = QWx2;
  blkstate_stride    = QWx2 + 2;
  memset(this->pass_length, 0, sizeof(this->pass_length));
  this->pass_length_count = 0;
  // layer_start and layer_passes are set by j2k_precinct_subband after construction
}

uint8_t j2k_codeblock::get_Mb() const { return this->M_b; }

uint8_t *j2k_codeblock::get_compressed_data() { return this->compressed_data; }

void j2k_codeblock::set_compressed_data(uint8_t *const buf, const uint16_t bufsize, const uint16_t Lref) {
  if (this->compressed_data != nullptr) {
    if (!refsegment) {
      printf(
          "ERROR: illegal attempt to allocate codeblock's compressed data but the data is not "
          "null.\n");
      throw std::exception();
    } else {
      // if we are here, this function has been called to copy Dref[]
      memcpy(this->current_address + this->pass_length[0], buf, bufsize);
      return;
    }
  }
  const size_t n = static_cast<size_t>(bufsize) + static_cast<size_t>(Lref) * refsegment;
  if (g_cblk_pool != nullptr) {
    this->compressed_data   = g_cblk_pool->bump(n);
    this->compressed_is_pooled = true;
  } else {
    this->compressed_data = static_cast<uint8_t *>(malloc(n));
  }
  memcpy(this->compressed_data, buf, bufsize);
  this->current_address = this->compressed_data;
}

void j2k_codeblock::create_compressed_buffer(buf_chain *tile_buf, int32_t buf_limit,
                                             const uint16_t &layer) {
  if (this->layer_passes[layer] == 0) {
    return;
  }

  // Compute total bytes contributed by this layer's passes.
  int32_t l0       = this->layer_start[layer];
  int32_t l1       = l0 + this->layer_passes[layer];
  uint32_t layer_length = 0;
  for (int32_t i = l0; i < l1; i++) {
    layer_length += this->pass_length[static_cast<size_t>(i)];
  }

  // Clamp layer_length to bytes remaining in the tile-part codestream.
  // The packet-header-decoded pass_length[] is attacker-controllable on
  // malformed inputs; without this clamp, a later copy_N_bytes/borrow_N_bytes
  // call can read past the end of the tile_buf node (UB on Linux release
  // where asserts are compiled out, or a CRT abort on Windows MSVC release
  // where asserts remain enabled).
  {
    const uint32_t avail = tile_buf->get_remaining_bytes();
    if (layer_length > avail) {
      printf("WARNING: codeblock layer length %u exceeds %u remaining bytes — malformed input.\n",
             (unsigned)layer_length, (unsigned)avail);
      layer_length = avail;
    }
  }

  if (this->compressed_data == nullptr) {
    // First contributing layer for this codeblock.
    if (layer_length == 0) {
      // Passes exist but contribute zero bytes (valid in J2K spec).
      // Allocate a minimal placeholder so later layers can append.
      this->compressed_data = static_cast<uint8_t *>(malloc(static_cast<size_t>(buf_limit)));
      this->current_address = this->compressed_data;
    } else {
      // Zero-copy: borrow a direct pointer into the codestream buffer.
      // The codestream buffer outlives all codeblocks, and all decoders
      // (HT fwd/rev/MEL, and MQ) treat compressed_data as read-only.
      // compressed_is_pooled = true suppresses free() in the destructor.
      this->compressed_data    = tile_buf->borrow_N_bytes(layer_length);
      this->current_address    = this->compressed_data + layer_length;
      this->length             = layer_length;
      this->compressed_is_pooled = true;
    }
    return;
  }

  // Second or later layer: nothing to append if this layer is empty.
  if (layer_length == 0) {
    return;
  }

  // If compressed_data is a borrowed (zero-copy) pointer we must convert it
  // to an owned buffer before appending, since the layers are not contiguous
  // in the codestream.
  if (this->compressed_is_pooled) {
    // Allocate at least buf_limit bytes (same as the original first-layer malloc)
    // so that subsequent layer appends do not overflow the owned buffer.
    uint32_t alloc_size = std::max(static_cast<uint32_t>(buf_limit), this->length + layer_length);
    uint8_t *owned      = static_cast<uint8_t *>(malloc(alloc_size));
    memcpy(owned, this->compressed_data, this->length);
    this->compressed_data      = owned;
    this->current_address      = owned + this->length;
    this->compressed_is_pooled = false;
    buf_limit                  = static_cast<int32_t>(alloc_size);
  } else {
    // Already an owned buffer — extend if needed.
    while (this->length + layer_length > static_cast<uint32_t>(buf_limit)) {
      uint8_t *newbuf =
          static_cast<uint8_t *>(realloc(this->compressed_data, this->length + layer_length));
      if (newbuf == nullptr) {
        // realloc failed: original buffer is still valid; abort to avoid leaking it.
        throw std::bad_alloc();
      }
      this->compressed_data = newbuf;
      this->current_address = this->compressed_data + this->length;
      buf_limit             = static_cast<int32_t>(this->length + layer_length);
    }
  }

  // we assume that the size of the compressed data is less than or equal to that of buf_chain node.
  tile_buf->copy_N_bytes(this->current_address, layer_length);
  this->length += layer_length;
}

void j2k_codeblock::skip_compressed_buffer(buf_chain *tile_buf, const uint16_t &layer) {
  if (this->layer_passes[layer] == 0) {
    return;
  }
  // Mirror create_compressed_buffer's layer_length computation so the byte
  // stream advances by exactly the same amount as the attach path.
  int32_t l0 = this->layer_start[layer];
  int32_t l1 = l0 + this->layer_passes[layer];
  uint32_t layer_length = 0;
  for (int32_t i = l0; i < l1; i++) {
    layer_length += this->pass_length[static_cast<size_t>(i)];
  }
  if (layer_length == 0) return;

  const uint32_t avail = tile_buf->get_remaining_bytes();
  if (layer_length > avail) layer_length = avail;
  if (layer_length == 0) return;

  // Advance tile_buf's read cursor without attaching the borrowed pointer
  // anywhere — borrow_N_bytes just moves `pos` forward by N within the
  // current node.
  (void)tile_buf->borrow_N_bytes(layer_length);
}

void j2k_codeblock::reset_for_next_frame() {
  // Release an owned compressed buffer before nulling the pointer.  Pooled
  // buffers are freed by their owning cblk_data_pool (decoder-side pools
  // are per-tile and are reset by the surrounding code; encoder-side pools
  // live in EncodePoolCtx and are not relevant here).
  if (compressed_data != nullptr && !compressed_is_pooled) {
    free(compressed_data);
  }
  compressed_data       = nullptr;
  current_address       = nullptr;
  compressed_is_pooled  = false;
  length                = 0;
  num_passes            = 0;
  num_ZBP               = 0;
  fast_skip_passes      = 0;
  Lblock                = 0;
  already_included      = false;
  refsegment            = false;
  pass_length_count     = 0;
  memset(pass_length, 0, sizeof(pass_length));
  // parse_packet_header ORs HT_PHLD and HT_MIXED onto Cmodes during parse;
  // strip those back off so the codeblock starts from the COD-declared
  // base codeblock style.  (The base bits, e.g. HT itself, stay set.)
  Cmodes &= static_cast<uint16_t>(~(HT_PHLD | HT_MIXED));
  // layer_start / layer_passes are pooled by the owning j2k_precinct_subband
  // (cb_layer_pool) and must be zeroed so parse_packet_header sees no
  // stale contributions from the previous frame.
  if (layer_start != nullptr) memset(layer_start, 0, num_layers);
  if (layer_passes != nullptr) memset(layer_passes, 0, num_layers);
}

/********************************************************************************
 * j2k_precinct_subband
 *******************************************************************************/
j2k_precinct_subband::j2k_precinct_subband(uint8_t orientation, uint8_t M_b, uint8_t R_b,
                                           uint8_t transformation, float stepsize, sprec_t *ibuf,
                                           const element_siz &bp0, const element_siz &p0,
                                           const element_siz &p1, const uint32_t band_stride,
                                           const uint16_t &num_layers, const element_siz &codeblock_size,
                                           const uint8_t &Cmodes)
    : j2k_region(p0, p1),
      orientation(orientation),
      codeblocks(nullptr) {
  if (this->pos1.x > this->pos0.x) {
    this->num_codeblock_x =
        ceil_int(this->pos1.x - 0, codeblock_size.x) - (this->pos0.x - 0) / codeblock_size.x;
  } else {
    this->num_codeblock_x = 0;
  }
  if (this->pos1.y > this->pos0.y) {
    this->num_codeblock_y =
        ceil_int(this->pos1.y - 0, codeblock_size.y) - (this->pos0.y - 0) / codeblock_size.y;
  } else {
    this->num_codeblock_y = 0;
  }

  const uint32_t num_codeblocks = this->num_codeblock_x * this->num_codeblock_y;
  if (num_codeblocks != 0) {
    inclusion_info = tagtree(this->num_codeblock_x, this->num_codeblock_y);
    ZBP_info       = tagtree(this->num_codeblock_x, this->num_codeblock_y);

    // Single flat buffer for all layer_start + layer_passes arrays
    cb_layer_pool = MAKE_UNIQUE<uint8_t[]>(static_cast<size_t>(num_codeblocks) * 2 * num_layers);
    memset(cb_layer_pool.get(), 0, static_cast<size_t>(num_codeblocks) * 2 * num_layers);

    // Single allocation for all codeblock objects (placement new).  The guard
    // unwinds partial construction if any j2k_codeblock(...) throws.
    auto *raw = static_cast<j2k_codeblock *>(operator new[](sizeof(j2k_codeblock) * num_codeblocks));
    placement_new_array_guard<j2k_codeblock> guard(raw);
    uint8_t *layer_pool_ptr = cb_layer_pool.get();
    for (uint32_t cb = 0; cb < num_codeblocks; cb++) {
      const uint32_t x = cb % this->num_codeblock_x;
      const uint32_t y = cb / this->num_codeblock_x;

      const element_siz cblkpos0(std::max(pos0.x, codeblock_size.x * (x + pos0.x / codeblock_size.x)),
                                 std::max(pos0.y, codeblock_size.y * (y + pos0.y / codeblock_size.y)));
      const element_siz cblkpos1(std::min(pos1.x, codeblock_size.x * (x + 1 + pos0.x / codeblock_size.x)),
                                 std::min(pos1.y, codeblock_size.y * (y + 1 + pos0.y / codeblock_size.y)));
      const element_siz cblksize(cblkpos1.x - cblkpos0.x, cblkpos1.y - cblkpos0.y);
      const uint32_t offset = cblkpos0.x - bp0.x + (cblkpos0.y - bp0.y) * band_stride;
      j2k_codeblock *blk =
          new (&raw[cb]) j2k_codeblock(cb, orientation, M_b, R_b, transformation, stepsize,
                                       band_stride, ibuf, offset, num_layers, Cmodes,
                                       cblkpos0, cblkpos1, cblksize);
      guard.commit_one();
      // Hand out slices of the layer pool (layer_start then layer_passes)
      blk->layer_start  = layer_pool_ptr;
      blk->layer_passes = layer_pool_ptr + num_layers;
      layer_pool_ptr += 2 * num_layers;
    }
    guard.release();
    this->codeblocks = raw;
  }
}

tagtree_node *j2k_precinct_subband::get_inclusion_node(uint32_t i) {
  return &this->inclusion_info.node[i];
}
tagtree_node *j2k_precinct_subband::get_ZBP_node(uint32_t i) { return &this->ZBP_info.node[i]; }

j2k_codeblock *j2k_precinct_subband::access_codeblock(uint32_t i) { return &this->codeblocks[i]; }

void j2k_precinct_subband::reset_for_next_frame() {
  // Reset the two tagtrees node-by-node.  tagtree::build() is not used here
  // because it skips nodes whose set_flag is true (a ctor-time encoder path
  // we don't want to trigger), and it recomputes value from children —
  // which for the decoder means propagating stale values from the previous
  // frame.  reset_for_next_frame() on the node clears all four state fields.
  for (uint32_t i = 0; i < inclusion_info.num_nodes; ++i) {
    inclusion_info.node[i].reset_for_next_frame();
  }
  for (uint32_t i = 0; i < ZBP_info.num_nodes; ++i) {
    ZBP_info.node[i].reset_for_next_frame();
  }
  const uint32_t N = num_codeblock_x * num_codeblock_y;
  for (uint32_t i = 0; i < N; ++i) {
    codeblocks[i].reset_for_next_frame();
  }
}

void j2k_precinct_subband::parse_packet_header(buf_chain *packet_header, uint16_t layer_idx,
                                               uint16_t Ccap15) {
  // if no codeblock exists, nothing to do is left
  if (this->num_codeblock_x * this->num_codeblock_y == 0) {
    return;
  }
  uint8_t bit;
  bool is_included = false;
  uint16_t threshold;
  std::vector<uint32_t> tree_path;

  for (uint32_t idx = 0; idx < this->num_codeblock_x * this->num_codeblock_y; ++idx) {
    j2k_codeblock *block  = this->access_codeblock(idx);
    uint8_t cumsum_layers = 0;
    for (uint32_t i = 0; i < block->num_layers; ++i) {
      cumsum_layers = static_cast<uint8_t>(cumsum_layers + block->layer_passes[i]);
    }
    // uint32_t number_of_bytes      = 0;  // initialize to zero in case of `not included`.
    block->layer_start[layer_idx] = cumsum_layers;

    tagtree_node *current_node, *parent_node;

    if (!block->already_included) {
      // Flags for placeholder passes and mixed mode
      if (block->Cmodes >= HT) {
        // adding HT_PHLD flag because an HT codeblock may include placeholder passes
        block->Cmodes |= HT_PHLD;
        // If both Bits14 and 15 of Ccap15 is true, Mixed mode.
        if (Ccap15 & 0xC000) {
          block->Cmodes |= HT_MIXED;
        }
      }
      // Retrieve codeblock inclusion
      assert(block->fast_skip_passes == 0);

      // build tagtree search path
      tree_path.clear();
      current_node           = this->get_inclusion_node(idx);
      uint8_t max_tree_level = current_node->get_level();
      if (max_tree_level != 0xFF) {
        max_tree_level++;
      }
      tree_path.reserve(max_tree_level);

      tree_path.push_back(static_cast<uint32_t>(current_node->get_index()));
      while (current_node->get_parent_index() >= 0) {
        current_node = this->get_inclusion_node(static_cast<uint32_t>(current_node->get_parent_index()));
        tree_path.push_back(static_cast<uint32_t>(current_node->get_index()));
      }
      is_included = false;

      if (layer_idx > 0) {
        // Special case; A codeblock is not included in the first layer (layer 0).
        // Inclusion information of layer 0 (i.e. before the first contribution) shall be decoded.
        threshold = 0;

        for (size_t i = tree_path.size(); i > 0; --i) {
          current_node = this->get_inclusion_node(tree_path[i - 1]);
          if (current_node->get_state() == 0) {
            if (current_node->get_parent_index() < 0) {
              parent_node = nullptr;
            } else {
              parent_node =
                  this->get_inclusion_node(static_cast<uint32_t>(current_node->get_parent_index()));
            }
            if (current_node->get_level() > 0 && parent_node != nullptr) {
              if (current_node->get_current_value() < parent_node->get_current_value()) {
                current_node->set_current_value(parent_node->get_current_value());
              }
            }
            if (current_node->get_current_value() <= threshold) {
              bit = packet_header->get_bit();
              if (bit == 1) {
                current_node->set_value(current_node->get_current_value());
                current_node->set_state(1);
                is_included = true;
              } else {
                current_node->set_current_value(
                    static_cast<uint16_t>(current_node->get_current_value() + 1));
                is_included = false;
              }
            }
          }
        }
      }

      // Normal case of inclusion information
      threshold = layer_idx;

      for (size_t i = tree_path.size(); i > 0; --i) {
        current_node = this->get_inclusion_node(tree_path[i - 1]);
        if (current_node->get_state() == 0) {
          if (current_node->get_parent_index() < 0) {
            parent_node = nullptr;
          } else {
            parent_node = this->get_inclusion_node(static_cast<uint32_t>(current_node->get_parent_index()));
          }
          if (current_node->get_level() > 0 && parent_node != nullptr) {
            if (current_node->get_current_value() < parent_node->get_current_value()) {
              current_node->set_current_value(parent_node->get_current_value());
            }
          }
          if (current_node->get_current_value() <= threshold) {
            bit = packet_header->get_bit();
            if (bit == 1) {
              current_node->set_value(current_node->get_current_value());
              current_node->set_state(1);
              is_included = true;
            } else {
              current_node->set_current_value(static_cast<uint16_t>(current_node->get_current_value() + 1));
              is_included = false;
            }
          }
        }
      }

      // Retrieve number of zero bit planes
      if (is_included) {
        block->already_included = true;
        for (size_t i = tree_path.size(); i > 0; --i) {
          current_node = this->get_ZBP_node(tree_path[i - 1]);
          if (current_node->get_state() == 0) {
            if (current_node->get_parent_index() < 0) {
              parent_node = nullptr;
            } else {
              parent_node = this->get_ZBP_node(static_cast<uint32_t>(current_node->get_parent_index()));
            }
            if (current_node->get_level() > 0) {
              if (current_node->get_current_value() < parent_node->get_current_value()) {
                current_node->set_current_value(parent_node->get_current_value());
              }
            }
            while (current_node->get_state() == 0) {
              bit = packet_header->get_bit();
              if (bit == 0) {
                current_node->set_current_value(
                    static_cast<uint16_t>(current_node->get_current_value() + 1));
              } else {
                current_node->set_value(current_node->get_current_value());
                current_node->set_state(1);
              }
            }
          }
        }
        block->num_ZBP = static_cast<uint8_t>(current_node->get_value());
        block->Lblock  = 3;
      }
    } else {
      // this codeblock has been already included in previous packets
      bit = packet_header->get_bit();
      if (bit) {
        is_included = true;
      } else {
        is_included = false;
      }
    }

    if (is_included) {
      // Retrieve number of coding passes in this layer
      int32_t new_passes = 1;
      bit                = packet_header->get_bit();
      new_passes += bit;
      if (new_passes >= 2) {
        bit = packet_header->get_bit();
        new_passes += bit;
        if (new_passes >= 3) {
          new_passes += static_cast<uint8_t>(packet_header->get_N_bits(2));
          if (new_passes >= 6) {
            new_passes += static_cast<uint8_t>(packet_header->get_N_bits(5));
            if (new_passes >= 37) {
              new_passes += static_cast<uint8_t>(packet_header->get_N_bits(7));
            }
          }
        }
      }
      block->layer_passes[layer_idx] = static_cast<uint8_t>(new_passes);
      // Retrieve Lblock
      while ((bit = packet_header->get_bit()) == 1) {
        block->Lblock++;
      }
      uint8_t bypass_term_threshold = 0;
      uint8_t bits_to_read          = 0;
      uint8_t pass_index            = block->num_passes;
      uint32_t segment_bytes        = 0;
      int32_t segment_passes        = 0;
      uint8_t next_segment_passes   = 0;
      int32_t href_passes, pass_bound;
      if (block->Cmodes & HT_PHLD) {
        href_passes    = (pass_index + new_passes - 1) % 3;
        segment_passes = new_passes - href_passes;
        pass_bound     = 2;
        bits_to_read   = static_cast<uint8_t>(block->Lblock);
        if (segment_passes < 1) {
          // No possible HT Cleanup pass here; may have placeholder passes
          // or an original J2K block bit-stream (in MIXED mode).
          segment_passes = new_passes;
          while (pass_bound <= segment_passes) {
            bits_to_read++;
            pass_bound += pass_bound;
          }
          segment_bytes = packet_header->get_N_bits(bits_to_read);
          if (segment_bytes) {
            if (block->Cmodes & HT_MIXED) {
              block->Cmodes &= static_cast<uint16_t>(~(HT_PHLD | HT));
            } else {
              printf("ERROR: Length information for a HT-codeblock is invalid\n");
              throw std::exception();
            }
          }
        } else {
          while (pass_bound <= segment_passes) {
            bits_to_read++;
            pass_bound += pass_bound;
          }
          segment_bytes = packet_header->get_N_bits(bits_to_read);
          if (segment_bytes) {
            // No more placeholder passes
            if (!(block->Cmodes & HT_MIXED)) {
              // Must be the first HT Cleanup pass
              if (segment_bytes < 2) {
                printf("ERROR: Length information for a HT-codeblock is invalid\n");
                throw std::exception();
              }
              next_segment_passes = 2;
              block->Cmodes &= static_cast<uint16_t>(~(HT_PHLD));
            } else if (block->Lblock > 3 && segment_bytes > 1
                       && (segment_bytes >> (bits_to_read - 1)) == 0) {
              // Must be the first HT Cleanup pass, since length MSB is 0
              next_segment_passes = 2;
              block->Cmodes &= static_cast<uint16_t>(~(HT_PHLD));
            } else {
              // Must have an original (non-HT) block coding pass
              block->Cmodes &= static_cast<uint16_t>(~(HT_PHLD | HT));
              segment_passes = new_passes;
              while (pass_bound <= segment_passes) {
                bits_to_read++;
                pass_bound += pass_bound;
                segment_bytes <<= 1;
                segment_bytes += packet_header->get_bit();
              }
            }
          } else {
            // Probably parsing placeholder passes, but we need to read an
            // extra length bit to verify this, since prior to the first
            // HT Cleanup pass, the number of length bits read for a
            // contributing code-block is dependent on the number of passes
            // being included, as if it were a non-HT code-block.
            segment_passes = new_passes;
            if (pass_bound <= segment_passes) {
              while (true) {
                bits_to_read++;
                pass_bound += pass_bound;
                segment_bytes <<= 1;
                segment_bytes += packet_header->get_bit();
                if (pass_bound > segment_passes) {
                  break;
                }
              }
              if (segment_bytes) {
                if (block->Cmodes & HT_MIXED) {
                  block->Cmodes &= static_cast<uint16_t>(~(HT_PHLD | HT));
                } else {
                  printf("ERROR: Length information for a HT-codeblock is invalid\n");
                  throw std::exception();
                }
              }
            }
          }
        }
      } else if (block->Cmodes & HT) {
        // Quality layer commences with a non-initial HT coding pass
        assert(bits_to_read == 0);
        segment_passes = block->num_passes % 3;
        if (segment_passes == 0) {
          // num_passes is a HT Cleanup pass; next segment has refinement passes
          segment_passes      = 1;
          next_segment_passes = 2;
          if (segment_bytes == 1) {
            printf("ERROR: something wrong 943.\n");
            throw std::exception();
          }
        } else {
          // new pass = 1 means num_passes is HT SigProp; 2 means num_passes is
          // HT MagRef pass
          if (new_passes > 1) {
            segment_passes = 3 - segment_passes;
          } else {
            segment_passes = 1;
          }
          next_segment_passes = 1;
          bits_to_read        = static_cast<uint8_t>(segment_passes - 1);
        }
        bits_to_read  = static_cast<uint8_t>(bits_to_read + block->Lblock);
        segment_bytes = packet_header->get_N_bits(bits_to_read);
      } else if (!(block->Cmodes & (RESTART | BYPASS))) {
        // Common case for non-HT code-blocks; we have only one segment
        bits_to_read   = static_cast<uint8_t>(block->Lblock + int_log2(static_cast<uint8_t>(new_passes)));
        segment_bytes  = packet_header->get_N_bits(bits_to_read);
        segment_passes = new_passes;
      } else if (block->Cmodes & RESTART) {
        // RESTART MODE
        bits_to_read        = static_cast<uint8_t>(block->Lblock);
        segment_bytes       = packet_header->get_N_bits(bits_to_read);
        segment_passes      = 1;
        next_segment_passes = 1;
      } else {
        // BYPASS MODE
        bypass_term_threshold = 10;
        assert(bits_to_read == 0);
        if (block->num_passes < bypass_term_threshold) {
          // May have from 1 to 10 uninterrupted passes before 1st RAW SigProp
          segment_passes = bypass_term_threshold - block->num_passes;
          if (segment_passes > new_passes) {
            segment_passes = new_passes;
          }
          while ((2 << bits_to_read) <= segment_passes) {
            bits_to_read++;
          }
          next_segment_passes = 2;
        } else if ((block->num_passes - bypass_term_threshold) % 3 < 2) {
          // new_passes = 0 means `num_passes' is a RAW SigProp; 1 means
          // `num_passes' is a RAW MagRef pass
          if (new_passes > 1) {
            segment_passes = 2 - (block->num_passes - bypass_term_threshold) % 3;
          } else {
            segment_passes = 1;
          }
          bits_to_read        = static_cast<uint8_t>(segment_passes - 1);
          next_segment_passes = 1;
        } else {
          // `num_passes' is an isolated Cleanup pass that precedes a RAW
          // SigProp pass
          segment_passes      = 1;
          next_segment_passes = 2;
        }
        bits_to_read  = static_cast<uint8_t>(bits_to_read + block->Lblock);
        segment_bytes = packet_header->get_N_bits(bits_to_read);
      }

      block->num_passes = static_cast<uint8_t>(block->num_passes + segment_passes);
      while (block->pass_length_count < block->num_passes) {
        block->pass_length[block->pass_length_count++] = 0;
      }
      block->pass_length[static_cast<size_t>(block->num_passes - 1)] = segment_bytes;
      // number_of_bytes += segment_bytes;

      uint8_t primary_passes, secondary_passes;
      uint32_t primary_bytes, secondary_bytes;
      OPENHTJ2K_MAYBE_UNUSED uint32_t fast_skip_bytes = 0;
      bool empty_set;
      if ((block->Cmodes & (HT | HT_PHLD)) == HT) {
        new_passes -= static_cast<uint8_t>(segment_passes);
        primary_passes          = static_cast<uint8_t>(segment_passes + block->fast_skip_passes);
        block->fast_skip_passes = 0;
        primary_bytes           = segment_bytes;
        secondary_passes        = 0;
        secondary_bytes         = 0;
        empty_set               = false;
        if (next_segment_passes == 2 && segment_bytes == 0) {
          empty_set = true;
        }
        while (new_passes > 0) {
          if (new_passes > 1) {
            segment_passes = next_segment_passes;
          } else {
            segment_passes = 1;
          }
          next_segment_passes = static_cast<uint8_t>(3 - next_segment_passes);
          bits_to_read =
              static_cast<uint8_t>(block->Lblock + static_cast<unsigned int>(segment_passes) - 1);
          segment_bytes = packet_header->get_N_bits(bits_to_read);
          new_passes -= static_cast<uint8_t>(segment_passes);
          if (next_segment_passes == 2) {
            // This is a FAST Cleanup pass
            assert(segment_passes == 1);
            if (segment_bytes != 0) {
              // This will have to be the new primary
              if (segment_bytes < 2) {
                printf("ERROR: Something wrong 1037\n");
                throw std::exception();
              }
              fast_skip_bytes += primary_bytes + secondary_bytes;
              primary_passes++;
              primary_passes          = static_cast<uint8_t>(primary_passes + secondary_passes);
              primary_bytes           = segment_bytes;
              secondary_bytes         = 0;
              secondary_passes        = 0;
              primary_passes          = static_cast<uint8_t>(primary_passes + block->fast_skip_passes);
              block->fast_skip_passes = 0;
              empty_set               = false;
            } else {
              // Starting a new empty set
              block->fast_skip_passes++;
              empty_set = true;
            }
          } else {
            // This is a FAST Refinement pass
            if (empty_set) {
              if (segment_bytes != 0) {
                printf("ERROR: Something wrong 1225\n");
                throw std::exception();
              }
              block->fast_skip_passes = static_cast<uint8_t>(block->fast_skip_passes + segment_passes);
            } else {
              secondary_passes = static_cast<uint8_t>(segment_passes);
              secondary_bytes  = segment_bytes;
            }
          }

          block->num_passes = static_cast<uint8_t>(block->num_passes + segment_passes);
          while (block->pass_length_count < block->num_passes) {
            block->pass_length[block->pass_length_count++] = 0;
          }
          block->pass_length[static_cast<size_t>(block->num_passes - 1)] = segment_bytes;
          // number_of_bytes += segment_bytes;
        }
      } else {
        new_passes -= static_cast<uint8_t>(segment_passes);
        while (new_passes > 0) {
          if (bypass_term_threshold != 0) {
            if (new_passes > 1) {
              segment_passes = next_segment_passes;
            } else {
              segment_passes = 1;
            }
            next_segment_passes = static_cast<uint8_t>(3 - next_segment_passes);
            bits_to_read =
                static_cast<uint8_t>(block->Lblock + static_cast<unsigned int>(segment_passes) - 1);
          } else {
            assert((block->Cmodes & RESTART) != 0);
            segment_passes = 1;
            bits_to_read   = static_cast<uint8_t>(block->Lblock);
          }
          segment_bytes = packet_header->get_N_bits(bits_to_read);
          new_passes -= static_cast<uint8_t>(segment_passes);
          block->num_passes = static_cast<uint8_t>(block->num_passes + segment_passes);
          while (block->pass_length_count < block->num_passes) {
            block->pass_length[block->pass_length_count++] = 0;
          }
          block->pass_length[static_cast<size_t>(block->num_passes - 1)] = segment_bytes;
          // number_of_bytes += segment_bytes;
        }
      }
    } else {
      // this layer has no contribution from this codeblock
      block->layer_passes[layer_idx] = 0;
    }
  }
}

void j2k_precinct_subband::generate_packet_header(packet_header_writer &header, uint16_t layer_idx) {
  // if no codeblock exists, nothing to do is left
  if (this->num_codeblock_x * this->num_codeblock_y == 0) {
    return;
  }

  uint16_t threshold;
  j2k_codeblock *blk;
  std::vector<int32_t> tree_path;

  // set value for each leaf node
  for (uint32_t idx = 0; idx < this->num_codeblock_x * this->num_codeblock_y; ++idx) {
    blk = this->access_codeblock(idx);
    if (blk->length) {
      this->inclusion_info.node[idx].set_value(blk->layer_start[0]);
    } else {
      this->inclusion_info.node[idx].set_value(1);
    }
    this->ZBP_info.node[idx].set_value(blk->num_ZBP);
  }
  // Building tagtree structures
  this->inclusion_info.build();
  this->ZBP_info.build();

  tagtree_node *current_node, *parent_node;

  for (uint32_t idx = 0; idx < this->num_codeblock_x * this->num_codeblock_y; ++idx) {
    blk                            = this->access_codeblock(idx);
    uint8_t preceding_layer_passes = 0;
    for (size_t i = 0; i < layer_idx; ++i) {
      preceding_layer_passes = static_cast<uint8_t>(preceding_layer_passes + blk->layer_passes[i]);
    }

    if (preceding_layer_passes == 0) {
      // this is the first contribution
      current_node = this->get_inclusion_node(idx);

      // build tagtree search path
      tree_path.clear();
      uint8_t max_tree_level = current_node->get_level();
      if (max_tree_level != 0xFF) {
        max_tree_level++;
      }
      tree_path.reserve(max_tree_level);
      tree_path.push_back(current_node->get_index());
      while (current_node->get_parent_index() >= 0) {
        current_node = this->get_inclusion_node(static_cast<uint32_t>(current_node->get_parent_index()));
        tree_path.push_back(current_node->get_index());
      }

      // inclusion tagtree coding
      threshold = layer_idx;
      for (size_t i = tree_path.size(); i > 0; --i) {
        current_node = this->get_inclusion_node(static_cast<uint32_t>(tree_path[i - 1]));
        if (current_node->get_state() == 0) {
          if (current_node->get_parent_index() < 0) {
            parent_node = nullptr;
          } else {
            parent_node = this->get_inclusion_node(static_cast<uint32_t>(current_node->get_parent_index()));
          }
          if (current_node->get_level() > 0 && parent_node != nullptr) {
            if (current_node->get_current_value() < parent_node->get_current_value()) {
              current_node->set_current_value(parent_node->get_current_value());
            }
          }
          if (current_node->get_current_value() <= threshold) {
            if (current_node->get_value() <= threshold) {
              header.put_bit(1);
              current_node->set_state(1);
            } else {
              header.put_bit(0);
              current_node->set_current_value(static_cast<uint8_t>(current_node->get_current_value() + 1));
            }
          }
        }
      }

      // number of zero bit plane tagtree coding
      if (blk->layer_passes[layer_idx] > 0) {
        blk->already_included = true;
        blk->Lblock           = 3;

        for (size_t i = tree_path.size(); i > 0; --i) {
          current_node = this->get_ZBP_node(static_cast<uint32_t>(tree_path[i - 1]));
          if (current_node->get_parent_index() < 0) {
            threshold = 0;
          } else {
            threshold =
                this->get_ZBP_node(static_cast<uint32_t>(current_node->get_parent_index()))->get_value();
          }
          while (current_node->get_state() == 0) {
            while (threshold < current_node->get_value()) {
              header.put_bit(0);
              threshold++;
            }
            current_node->set_state(1);
            header.put_bit(1);
          }
        }
      }
    } else {
      // if we get here, this codeblock has been included in at least one of preceding layers
      header.put_bit(std::min((uint8_t)1, blk->layer_passes[layer_idx]));
    }

    const uint8_t num_passes = blk->layer_passes[layer_idx];
    if (num_passes) {
      // number of coding passes encoding
      if (blk->layer_passes[layer_idx] > 0) {
        assert(num_passes < 165);
        if (num_passes == 1) {
          header.put_bit(0);
        } else if (num_passes == 2) {
          header.put_Nbits(0x2, 2);
        } else if (num_passes < 6) {
          header.put_Nbits(0x3, 2);
          header.put_Nbits(num_passes - 3U, 2U);
        } else if (num_passes < 37) {
          header.put_Nbits(0xF, 4);
          header.put_Nbits(num_passes - 6U, 5U);
        } else {
          header.put_Nbits(0x1FF, 9);
          header.put_Nbits(num_passes - 37U, 7U);
        }
      }

      // compute number of coded bytes in this layer_idx
      uint8_t l0 = blk->layer_start[layer_idx];
      uint8_t l1 = blk->layer_passes[layer_idx];

      OPENHTJ2K_MAYBE_UNUSED uint32_t buf_start = 0, buf_end = 0;
      // NOTE: the following code to derive number_of_bytes shall be improved
      if (l0) {
        for (size_t i = 0; i < l0; ++i) {
          buf_start += blk->pass_length[i];
        }
      }
      for (size_t i = 0; i < l0 + l1; ++i) {
        buf_end += blk->pass_length[i];
      }
      // uint32_t number_of_bytes = buf_end - buf_start;

      // length coding: currently only for HT Cleanup pass
      int new_passes = static_cast<int32_t>(num_passes);
      // uint8_t bits_to_write  = 0;
      uint8_t pass_idx                      = l0;
      uint32_t segment_bytes                = 0;
      uint8_t segment_passes                = 0;
      OPENHTJ2K_MAYBE_UNUSED uint32_t total_bytes = 0;
      uint8_t length_bits                   = 0;

      while (new_passes > 0) {
        assert(blk->Cmodes & HT);
        segment_passes = (pass_idx == 0) ? 1 : static_cast<uint8_t>(new_passes);

        length_bits = 0;
        // length_bits = floor(log2(segment_passes))
        while ((2 << length_bits) <= segment_passes) {
          length_bits++;
        }
        length_bits = static_cast<uint8_t>(length_bits + blk->Lblock);

        segment_bytes = 0;
        auto val      = static_cast<uint32_t>(segment_passes);
        while (val > 0) {
          segment_bytes += blk->pass_length[pass_idx + val - 1];
          val--;
        }

        while (segment_bytes >= static_cast<uint32_t>(1 << length_bits)) {
          header.put_bit(1);
          length_bits++;
          blk->Lblock++;
        }
        new_passes -= segment_passes;
        pass_idx = static_cast<uint8_t>(pass_idx + segment_passes);
        total_bytes += segment_bytes;
      }
      header.put_bit(0);

      // bits_to_write  = 0;
      pass_idx       = l0;
      segment_bytes  = 0;
      segment_passes = 0;
      new_passes     = num_passes;
      total_bytes    = 0;

      while (new_passes > 0) {
        assert(blk->Cmodes & HT);
        segment_passes = (pass_idx == 0) ? 1 : static_cast<uint8_t>(new_passes);

        length_bits = 0;
        // length_bits = floor(log2(segment_passes))
        while ((2 << length_bits) <= segment_passes) {
          length_bits++;
        }
        length_bits = static_cast<uint8_t>(length_bits + blk->Lblock);

        segment_bytes = 0;
        auto val      = static_cast<uint32_t>(segment_passes);
        while (val > 0) {
          segment_bytes += blk->pass_length[pass_idx + val - 1];
          val--;
        }

        for (int i = length_bits - 1; i >= 0; --i) {
          header.put_bit(static_cast<uint8_t>((segment_bytes & static_cast<uint32_t>(1 << i)) >> i));
        }
        new_passes -= segment_passes;
        pass_idx = static_cast<uint8_t>(pass_idx + segment_passes);
        total_bytes += segment_bytes;
      }
    }
  }  // end of outer for loop
}

/********************************************************************************
 * j2k_precinct
 *******************************************************************************/
j2k_precinct::j2k_precinct(const uint8_t &r, const uint32_t &idx, const element_siz &p0,
                           const element_siz &p1,
                           const j2k_subband *subband,
                           const uint16_t &num_layers, const element_siz &codeblock_size,
                           const uint8_t &Cmodes, uint8_t nb, dwt_type dfs_dir)
    : j2k_region(p0, p1),
      index(idx),
      resolution(r),
      num_bands((nb != 0) ? nb : ((resolution == 0) ? 1 : 3)),
      length(0),
      pband(nullptr),
      packet_header(nullptr),
      packet_header_length(0) {
  length = 0;  // for encoder only

  // Single allocation for all precinct-subbands (placement new) — replaces the
  // double-indirection unique_ptr<unique_ptr<T>[]> pattern.  The guard unwinds
  // partial construction if any j2k_precinct_subband(...) throws.
  auto *raw = static_cast<j2k_precinct_subband *>(
      operator new[](sizeof(j2k_precinct_subband) * num_bands));
  placement_new_array_guard<j2k_precinct_subband> guard(raw);
  const uint8_t xob[4] = {0, 1, 0, 1};
  const uint8_t yob[4] = {0, 0, 1, 1};
  for (unsigned long i = 0; i < num_bands; ++i) {
    const j2k_subband &sb = subband[i];
    const uint8_t ori = sb.orientation;
    uint32_t sr_x     = (ori == BAND_LL) ? 1U : 2U;
    uint32_t sr_y     = (ori == BAND_LL) ? 1U : 2U;
    // DFS HORZ: H band splits only horizontally — no vertical halving in precinct subband
    if (dfs_dir == DWT_HORZ) sr_y = 1U;
    // DFS VERT: H band splits only vertically — no horizontal halving
    if (dfs_dir == DWT_VERT) sr_x = 1U;
    const element_siz pbpos0(ceil_int(pos0.x - xob[ori], sr_x), ceil_int(pos0.y - yob[ori], sr_y));
    const element_siz pbpos1(ceil_int(pos1.x - xob[ori], sr_x), ceil_int(pos1.y - yob[ori], sr_y));
    new (&raw[i]) j2k_precinct_subband(
        sb.orientation, sb.M_b, sb.R_b, sb.transformation,
        sb.delta, sb.i_samples, sb.pos0, pbpos0, pbpos1, sb.stride,
        num_layers, codeblock_size, Cmodes);
    guard.commit_one();
  }
  guard.release();
  this->pband = raw;
}

j2k_precinct_subband *j2k_precinct::access_pband(uint8_t b) {
  assert(b < num_bands);
  return &this->pband[b];
}

/********************************************************************************
 * j2k_subband
 *******************************************************************************/
j2k_subband::j2k_subband(element_siz p0, element_siz p1, uint8_t orientation, uint8_t transformation,
                         uint8_t R_b, uint8_t epsilon_b, uint16_t mantissa_b, uint8_t M_b, float delta,
                         float nominal_range, sprec_t *ibuf, bool no_alloc)
    : j2k_region(p0, p1),
      orientation(orientation),
      transformation(transformation),
      R_b(R_b),
      epsilon_b(epsilon_b),
      mantissa_b(mantissa_b),
      M_b(M_b),
      delta(delta),
      nominal_range(nominal_range),
      i_samples(nullptr) {
  // TODO: consider reduce_NL value
  const uint32_t num_samples = (pos1.x - pos0.x) * (pos1.y - pos0.y);
  if (num_samples) {
    if (orientation != BAND_LL) {
      if (!no_alloc) {
        // Batch decode path: allocate and zero the full subband sample buffer.
        // One extra stride of padding allows the fused dequantize path to write
        // mp1 (the second row of a pair) safely even for the last row of a subband.
        // Additional DWT_LEFT_SLACK + DWT_RIGHT_SLACK floats are allocated as
        // border slack so that the in-place horizontal DWT can safely process
        // the first row's in[-LEFT..-1] and the last row's in[width..width+RIGHT-1]
        // without falling outside the allocation. The user-visible i_samples
        // pointer is offset by DWT_LEFT_SLACK from the allocator base.
        const size_t alloc_samples = sizeof(sprec_t) * this->stride * (pos1.y - pos0.y + 1)
                                     + sizeof(sprec_t) * (DWT_LEFT_SLACK + DWT_RIGHT_SLACK);
        sprec_t *base = static_cast<sprec_t *>(aligned_mem_alloc(alloc_samples, 32));
        memset(base, 0, alloc_samples);
        i_samples = base + DWT_LEFT_SLACK;
      }
      // When no_alloc=true (ring-mode line-based decode), i_samples stays nullptr.
      // decode_strip() will redirect block->i_samples to the ring buffer before decoding.
      // When no_alloc=true (ring-mode line-based decode), i_samples stays nullptr.
      // decode_strip() will redirect block->i_samples to the ring buffer before decoding.
    } else {
      i_samples = ibuf;
    }
  }
}

j2k_subband::~j2k_subband() {
  // printf("INFO: destructor of j2k_subband %d is called\n", orientation);
  if (orientation != BAND_LL) {
    if (i_samples != nullptr) aligned_mem_free(i_samples - DWT_LEFT_SLACK);
  }
}

/********************************************************************************
 * j2k_resolution
 *******************************************************************************/
j2k_resolution::j2k_resolution(const uint8_t &r, const element_siz &p0, const element_siz &p1,
                               const uint32_t &w, const uint32_t &h, bool no_alloc, uint8_t nb,
                               dwt_type dir)
    : j2k_region(p0, p1),
      index(r),
      precincts(nullptr),
      num_precincts(0),
      subbands(nullptr),
      // public
      num_bands((nb != 0) ? nb : ((index == 0) ? 1 : 3)),
      transform_direction(dir),
      npw(w),
      nph(h),
      is_empty((npw * nph == 0)),
      normalizing_upshift(0),
      normalizing_downshift(0),
      i_samples(nullptr) {
  // create buffer of LL band
  i_samples = nullptr;
  // In line-based decode mode (no_alloc=true), skip the large i_samples buffer.
  // init_line_decode() uses ring buffers instead; it would free these pages anyway,
  // but on macOS free() does not return pages to the OS. Skipping the allocation
  // avoids the malloc+memset entirely, keeping these pages out of RSS.
  if (!is_empty && !no_alloc) {
    // For resolution 0 (LL band), add one extra stride row of padding so the
    // fused dequantize path can safely write mp1 for the last line-pair of a
    // codeblock with odd height (same padding that j2k_subband already adds).
    // Additional DWT_LEFT_SLACK + DWT_RIGHT_SLACK floats are allocated as
    // border slack so that the in-place horizontal DWT can safely process
    // the first row's in[-LEFT..-1] and the last row's in[width..width+RIGHT-1]
    // without falling outside the allocation. The user-visible i_samples
    // pointer is offset by DWT_LEFT_SLACK from the allocator base.
    const uint32_t pad  = (index == 0) ? 1U : 0U;
    const size_t alloc_samples = sizeof(sprec_t) * this->stride * (pos1.y - pos0.y + pad)
                                 + sizeof(sprec_t) * (DWT_LEFT_SLACK + DWT_RIGHT_SLACK);
    sprec_t *base = static_cast<sprec_t *>(aligned_mem_alloc(alloc_samples, 32));
    if (index == 0) {
      memset(base, 0, alloc_samples);
    }
    i_samples = base + DWT_LEFT_SLACK;
  }
}

j2k_resolution::~j2k_resolution() {
  // Destroy precincts (placement-new'd flat array).
  if (precincts != nullptr) {
    for (uint32_t i = 0; i < num_precincts; ++i) {
      precincts[i].~j2k_precinct();
    }
    operator delete[](precincts);
  }
  // Destroy subbands (placement-new'd flat array).
  if (subbands != nullptr) {
    for (uint8_t b = 0; b < num_bands; ++b) {
      subbands[b].~j2k_subband();
    }
    operator delete[](subbands);
  }
  if (i_samples != nullptr) aligned_mem_free(i_samples - DWT_LEFT_SLACK);
}

void j2k_resolution::create_subbands(element_siz &p0, element_siz &p1, uint8_t NL, uint8_t transformation,
                                     std::vector<uint8_t> &exponents, std::vector<uint16_t> &mantissas,
                                     uint8_t num_guard_bits, uint8_t qstyle, uint8_t bitdepth,
                                     bool line_based, const DFS_marker *dfs) {
  // Flat array of j2k_subband objects (single allocation, placement-new'd).
  // Build into a local pointer under a guard so partial construction unwinds
  // cleanly if any j2k_subband(...) throws; subbands stays nullptr on failure.
  auto *raw = static_cast<j2k_subband *>(operator new[](sizeof(j2k_subband) * num_bands));
  placement_new_array_guard<j2k_subband> sb_guard(raw);
  uint8_t i;
  uint8_t b;
  uint8_t xob[4]    = {0, 1, 0, 1};
  uint8_t yob[4]    = {0, 0, 1, 1};
  uint8_t gain_b[4] = {0, 1, 1, 2};
  uint8_t bstart    = (index == 0) ? 0 : 1;
  uint8_t bstop     = (index == 0) ? 0 : 3;

  // Determine DFS type and QCD offset for this resolution.
  dwt_type dfs_type = transform_direction;  // already set by create_resolutions
  uint8_t qcd_base  = 0;  // base index into exponents/mantissas for this resolution
  if (dfs != nullptr && index > 0) {
    qcd_base = dfs->qcd_offset[index];
    // Adjust bstart/bstop based on DFS type.
    if (dfs_type == DWT_HORZ) {
      bstart = bstop = BAND_HL;  // only horizontal high-freq band
    } else if (dfs_type == DWT_VERT) {
      bstart = bstop = BAND_LH;  // only vertical high-freq band
    } else if (dfs_type == DWT_NO) {
      // NO_DWT: no high-freq bands; subbands array is empty (num_bands was set to 0).
      return;
    }
    // BIDIR: bstart=1, bstop=3 as normal.
  }

  uint8_t nb = static_cast<uint8_t>(NL - index);
  if (index != 0) {
    nb++;
  }
  uint8_t nb_1 = 0;
  if (nb > 0) {
    nb_1 = static_cast<uint8_t>(nb - 1);
  }
  uint8_t epsilon_b, R_b = 0, M_b = 0;
  uint16_t mantissa_b = 0;
  float delta, nominal_range;

  for (i = 0, b = bstart; b <= bstop; b++, i++) {
    // For DFS HORZ or VERT, the subband divides only one axis.
    // xob/yob control which origin offset to subtract before halving.
    element_siz spos0, spos1;
    if (dfs != nullptr && index > 0 && dfs_type != DWT_BIDIR) {
      if (dfs_type == DWT_HORZ) {
        // Split only horizontally: x is halved, y spans full resolution extent.
        spos0 = element_siz(ceil_int(p0.x - (1U << nb_1) * xob[BAND_HL], 1U << nb), pos0.y);
        spos1 = element_siz(ceil_int(p1.x - (1U << nb_1) * xob[BAND_HL], 1U << nb), pos1.y);
      } else {
        // VERT: split only vertically: y is halved, x spans full resolution extent.
        spos0 = element_siz(pos0.x, ceil_int(p0.y - (1U << nb_1) * yob[BAND_LH], 1U << nb));
        spos1 = element_siz(pos1.x, ceil_int(p1.y - (1U << nb_1) * yob[BAND_LH], 1U << nb));
      }
    } else if (b == BAND_LL) {
      // LL0 (index==0): use the resolution's own pos0/pos1.
      // create_resolutions already accounts for DFS (only BIDIR/VERT levels split vertically),
      // so pos0/pos1 are correct for both standard and DFS codestreams.
      spos0 = pos0;
      spos1 = pos1;
    } else {
      spos0 = element_siz(ceil_int(p0.x - (1U << (nb_1)) * xob[b], 1U << nb),
                          ceil_int(p0.y - (1U << (nb_1)) * yob[b], 1U << nb));
      spos1 = element_siz(ceil_int(p1.x - (1U << (nb_1)) * xob[b], 1U << nb),
                          ceil_int(p1.y - (1U << (nb_1)) * yob[b], 1U << nb));
    }

    // nominal range does not have any effect to lossless path
    nominal_range = this->child_ranges[b];

    // Compute QCD flat index for this subband.
    uint8_t qcd_idx;
    if (dfs != nullptr && index > 0) {
      // DFS: use precomputed qcd_offset[r] + local band index within this resolution.
      qcd_idx = static_cast<uint8_t>(qcd_base + i);
    } else {
      // Standard: 3*(NL-nb)+b, equivalent to 3*(r-1)+b for r > 0 (b 1-indexed).
      qcd_idx = static_cast<uint8_t>(3 * (NL - nb) + b);
    }

    if (transformation == 1) {
      // lossless
      epsilon_b = exponents[qcd_idx];
      M_b       = static_cast<uint8_t>(epsilon_b + num_guard_bits - 1);
      delta     = 1.0;
    } else {
      // lossy compression (irrev97 or ATK irrev)
      assert(transformation == 0 || transformation >= 2);
      if (qstyle == 1) {
        // derived
        epsilon_b  = static_cast<uint8_t>(exponents[0] - NL + nb);
        mantissa_b = mantissas[0];
      } else {
        // expounded
        assert(qstyle == 2);
        epsilon_b  = exponents[qcd_idx];
        mantissa_b = mantissas[qcd_idx];
      }
      M_b   = static_cast<uint8_t>(epsilon_b + num_guard_bits - 1);
      R_b   = static_cast<uint8_t>(bitdepth + gain_b[b]);
      delta = (1.0f / (static_cast<float>(1 << epsilon_b)))
              * (1.0f + (static_cast<float>(mantissa_b)) / (static_cast<float>(1 << 11)));
      // delta, which is quantization step-size, is scaled by nominal-range of this band
      delta *= nominal_range;
    }
    new (&raw[i]) j2k_subband(spos0, spos1, b, transformation, R_b, epsilon_b, mantissa_b,
                              M_b, delta, nominal_range, i_samples, line_based);
    sb_guard.commit_one();
  }
  sb_guard.release();
  subbands = raw;
}

void j2k_resolution::create_precincts(element_siz log2PP, uint16_t numlayers, element_siz codeblock_size,
                                      uint8_t Cmodes) {
  // precinct size signalled in header
  const element_siz PP(1U << log2PP.x, 1U << log2PP.y);
  // offset of horizontal precinct index
  const uint32_t idxoff_x = (pos0.x - 0) / PP.x;
  // offset of vertical precinct index
  const uint32_t idxoff_y = (pos0.y - 0) / PP.y;

  if (!is_empty) {
    const uint32_t total = npw * nph;
    // Single allocation for all precincts (placement new) — replaces double-indirection.
    // Build into a local pointer under a guard so partial construction unwinds
    // cleanly if any j2k_precinct(...) throws; precincts stays nullptr on failure
    // and num_precincts only commits to the constructed count.
    auto *raw = static_cast<j2k_precinct *>(operator new[](sizeof(j2k_precinct) * total));
    placement_new_array_guard<j2k_precinct> guard(raw);
    for (uint32_t i = 0; i < total; i++) {
      uint32_t x, y;
      x = i % npw;
      y = i / npw;
      const element_siz prcpos0(std::max(pos0.x, 0 + PP.x * (x + idxoff_x)),
                                std::max(pos0.y, 0 + PP.y * (y + idxoff_y)));
      const element_siz prcpos1(std::min(pos1.x, 0 + PP.x * (x + 1 + idxoff_x)),
                                std::min(pos1.y, 0 + PP.y * (y + 1 + idxoff_y)));
      new (&raw[i]) j2k_precinct(index, i, prcpos0, prcpos1, subbands, numlayers,
                                 codeblock_size, Cmodes, num_bands, transform_direction);
      guard.commit_one();
    }
    guard.release();
    precincts     = raw;
    num_precincts = total;
  }
}

j2k_precinct *j2k_resolution::access_precinct(uint32_t p) {
  if (p > npw * nph) {
    printf("ERROR: attempt to access precinct whose index is out of the valid range.\n");
    throw std::exception();
  }
  return &this->precincts[p];
}

j2c_packet::j2c_packet(const uint16_t l, const uint8_t r, const uint16_t c, const uint32_t p,
                       j2k_precinct *const cp, uint8_t num_bands)
    : layer(l), resolution(r), component(c), precinct(p), header(nullptr), body(nullptr) {
  // get length of the corresponding precinct
  length = cp->get_length();
  // create buffer to accommodate packet header and body
  buf        = MAKE_UNIQUE<uint8_t[]>(static_cast<size_t>(length));
  size_t pos = cp->packet_header_length;
  // copy packet header to packet buffer
  for (size_t i = 0; i < pos; ++i) {
    buf[i] = cp->packet_header[i];
  }
  // copy packet body to packet buffer
  for (uint8_t b = 0; b < num_bands; ++b) {
    j2k_precinct_subband *cpb = cp->access_pband(b);
    const uint32_t num_cblks  = cpb->num_codeblock_x * cpb->num_codeblock_y;
    for (uint32_t block_index = 0; block_index < num_cblks; ++block_index) {
      j2k_codeblock *block = cpb->access_codeblock(block_index);
      memcpy(&buf[pos], block->get_compressed_data(), sizeof(uint8_t) * block->length);
      pos += block->length;
    }
  }
}

j2k_subband *j2k_resolution::access_subband(uint8_t b) { return &this->subbands[b]; }

/********************************************************************************
 * j2k_tile_part
 *******************************************************************************/
j2k_tile_part::j2k_tile_part(uint16_t num_components) {
  tile_index      = 0;
  tile_part_index = 0;
  body            = nullptr;
  length          = 0;
  header          = MAKE_UNIQUE<j2k_tilepart_header>(num_components);
}

void j2k_tile_part::set_SOT(SOT_marker &tmpSOT) {
  this->tile_index      = tmpSOT.get_tile_index();
  this->tile_part_index = tmpSOT.get_tile_part_index();
  // this->header->SOT     = SOT_marker;
  this->header->SOT = tmpSOT;
}

int j2k_tile_part::read(j2c_src_memory &in) {
  uint32_t length_of_tilepart_markers = this->header->read(in);
  const uint32_t Psot                 = this->header->SOT.get_tile_part_length();
  if (Psot == 0) {
    // Psot=0: tile-part extends to EOC. Scan backward to find the last
    // 0xFF 0xD9 marker; some encoders append padding bytes after EOC, so
    // don't assume it is the final 2 bytes. The scan steps by 1 byte
    // because EOC can land at any byte offset within the tile-part body
    // (e.g., when the body length is odd) — stepping by 2 misses every
    // EOC at an odd offset.
    const uint32_t remaining = in.get_remaining();
    const uint8_t *p         = in.get_buf_pos();
    uint32_t eoc_offset      = (remaining >= 2) ? remaining - 2 : 0;
    for (uint32_t i = remaining; i >= 2; --i) {
      if (p[i - 2] == 0xFF && p[i - 1] == 0xD9) {
        eoc_offset = i - 2;
        break;
      }
    }
    this->length = eoc_offset;
  } else {
    this->length += Psot - length_of_tilepart_markers;
    // Clamp to available data: handles truncated codestreams where the encoder
    // pre-declared a Psot larger than the bytes actually written to file.
    const uint32_t remaining = in.get_remaining();
    if (this->length > remaining) {
      this->length = remaining;
    }
  }
  this->body = in.get_buf_pos();

  try {
    in.forward_Nbytes(this->length);
  } catch (std::exception &exc) {
    printf("ERROR: forward_Nbytes exceeds the size of buffer.\n");
    throw;
  }
  return EXIT_SUCCESS;
}

uint16_t j2k_tile_part::get_tile_index() const { return tile_index; }

uint8_t j2k_tile_part::get_tile_part_index() const { return tile_part_index; }

uint32_t j2k_tile_part::get_length() const { return this->length; }

uint8_t *j2k_tile_part::get_buf() { return body; }

void j2k_tile_part::set_tile_index(uint16_t t) { tile_index = t; }
void j2k_tile_part::set_tile_part_index(uint8_t tp) { tile_part_index = tp; }

/********************************************************************************
 * j2k_tile_component
 *******************************************************************************/
j2k_tile_component::j2k_tile_component() {
  index              = 0;
  bitdepth           = 0;
  NL                 = 0;
  Cmodes             = 0;
  codeblock_size.x   = 0;
  codeblock_size.y   = 0;
  precinct_size      = {};
  transformation     = 0;
  quantization_style = 0;
  exponents          = {};
  mantissas          = {};
  num_guard_bits     = 0;
  ROIshift           = 0;
  samples            = nullptr;
  resolution         = nullptr;
  num_resolutions    = 0;
  line_dec           = nullptr;
  line_enc           = nullptr;
}

j2k_tile_component::~j2k_tile_component() {
  // Ensure aligned_mem_alloc'd sub-resources inside line_dec/line_enc (lp_tmp, hp_tmp,
  // ring buffers, row buffers) are freed even if finalize_*() was never called
  // explicitly (e.g. due to an exception during decode/encode).  Clear the
  // single-tile reuse flag so finalize_line_decode does a real teardown.
  line_dec_persistent_ = false;
  finalize_line_decode();
  finalize_line_encode();
  // Destroy resolutions (placement-new'd flat array).
  if (resolution != nullptr) {
    for (uint8_t r = 0; r < num_resolutions; ++r) {
      resolution[r].~j2k_resolution();
    }
    operator delete[](resolution);
  }
  aligned_mem_free(samples);
}

// ─────────────────────────────────────────────────────────────────────────────
// Line-based decode: init / pull / finalize
// ─────────────────────────────────────────────────────────────────────────────

void j2k_tile_component::init_line_decode(bool ring_mode) {
  // Single-tile reuse fast path: the line_dec tree built on the first frame
  // already has the right geometry — the underlying j2k_subband_row_buf
  // ring allocations, idwt_2d_state, and level contexts are all reusable.
  // All we need is to return each cursor to its pristine position.
  if (line_dec != nullptr && line_dec_persistent_) {
    reset_line_decode_cursors();
    return;
  }

  const int32_t NL_act   = static_cast<int32_t>(NL) - static_cast<int32_t>(reduce_NL);
  const int32_t cb_h_val = static_cast<int32_t>(codeblock_size.y);

  line_dec      = std::make_unique<j2k_tcomp_line_dec>();
  auto *ld      = line_dec.get();
  ld->NL_active = NL_act;
  ld->next_row  = 0;

  // Coarsest active resolution: resolution[0] (always LL0 regardless of reduce_NL).
  j2k_resolution *r0 = access_resolution(0);
  ld->ll0_buf.init(r0, 0, cb_h_val, ROIshift, ring_mode);
  ld->next_row = static_cast<int32_t>(r0->get_pos0().y);

  if (NL_act == 0) {
    // Free full-tile sample buffers when ring mode is active (no IDWT needed).
    if (ring_mode) {
      if (r0->i_samples != nullptr) aligned_mem_free(r0->i_samples - DWT_LEFT_SLACK);
      r0->i_samples = nullptr;
    }
    return;  // no IDWT needed; pull directly from LL0
  }

  // Validate DWT directions before allocating any state — fail fast so partial
  // resources don't need to be unwound on error.
  for (int32_t i = 0; i < NL_act; ++i) {
    const dwt_type level_dir = access_resolution(static_cast<uint8_t>(i + 1))->transform_direction;
    if (level_dir == DWT_VERT) {
      throw std::runtime_error(
          "Line-based streaming decode does not support DWT_VERT DFS levels.");
    }
  }

  ld->states  = std::make_unique<idwt_2d_state[]>(static_cast<size_t>(NL_act));
  ld->ctxs    = std::make_unique<idwt_level_src_ctx[]>(static_cast<size_t>(NL_act));
  ld->hl_bufs = std::make_unique<j2k_subband_row_buf[]>(static_cast<size_t>(NL_act));
  ld->lh_bufs = std::make_unique<j2k_subband_row_buf[]>(static_cast<size_t>(NL_act));
  ld->hh_bufs = std::make_unique<j2k_subband_row_buf[]>(static_cast<size_t>(NL_act));

  // Cache raw pointers for index-by-int loops below (avoids size_t conversion warnings).
  idwt_2d_state       *states  = ld->states.get();
  idwt_level_src_ctx  *ctxs    = ld->ctxs.get();
  j2k_subband_row_buf *hl_bufs = ld->hl_bufs.get();
  j2k_subband_row_buf *lh_bufs = ld->lh_bufs.get();
  j2k_subband_row_buf *hh_bufs = ld->hh_bufs.get();

  // Initialise states from coarsest (i=0) to finest (i=NL_act-1).
  // resolution[i+1] corresponds to IDWT level i+1 (resolution index 1-based).
  for (int32_t i = 0; i < NL_act; ++i) {
    j2k_resolution *cr         = access_resolution(static_cast<uint8_t>(i + 1));
    const dwt_type  level_dir  = cr->transform_direction;

    // Initialise subband row buffers based on DWT type.
    // BIDIR: HL (0), LH (1), HH (2).  HORZ: H band in slot 0 only.  NO: no high-freq bands.
    j2k_subband *sb_HL = nullptr;
    j2k_subband *sb_LH = nullptr;
    j2k_subband *sb_HH = nullptr;

    if (level_dir == DWT_BIDIR) {
      sb_HL = cr->access_subband(0);
      sb_LH = cr->access_subband(1);
      sb_HH = cr->access_subband(2);
      hl_bufs[i].init(cr, 0, cb_h_val, ROIshift, ring_mode);
      lh_bufs[i].init(cr, 1, cb_h_val, ROIshift, ring_mode);
      hh_bufs[i].init(cr, 2, cb_h_val, ROIshift, ring_mode);
    } else if (level_dir == DWT_HORZ) {
      sb_HL = cr->access_subband(0);  // H band (horizontal high-pass)
      hl_bufs[i].init(cr, 0, cb_h_val, ROIshift, ring_mode);
      // lh_bufs[i] and hh_bufs[i] left default-constructed (no high-freq vertical bands).
    }
    // DWT_NO: no high-freq subbands at all.

    // Resolution geometry (output space of IDWT at this level).
    const int32_t u0 = static_cast<int32_t>(cr->get_pos0().x);
    const int32_t u1 = static_cast<int32_t>(cr->get_pos1().x);
    const int32_t v0 = static_cast<int32_t>(cr->get_pos0().y);
    const int32_t v1 = static_cast<int32_t>(cr->get_pos1().y);

    // LL-subband geometry for this level = resolution[i] (the child LL).
    // For i==0: child LL lives in ld->ll0_buf (resolution[0]).
    // For i>0:  child is the idwt_2d_state at level i-1.
    j2k_resolution *cr_ll = access_resolution(static_cast<uint8_t>(i));

    // lp_width = width of the child LL output = full width of resolution[i].
    const int32_t lp_width = static_cast<int32_t>(cr_ll->get_pos1().x - cr_ll->get_pos0().x);
    const int32_t hp_width = (sb_HL != nullptr)
                             ? static_cast<int32_t>(sb_HL->get_pos1().x - sb_HL->get_pos0().x)
                             : 0;
    // PSE counts for in-place horizontal filter (precomputed; indexed [u%2][eff]).
    // ATK (transformation>=2) uses same PSE lengths as rev53 (2-step filter).
    static constexpr int32_t kHPseLeft[2][2]  = {{3, 1}, {4, 2}};  // [u0%2][eff]
    static constexpr int32_t kHPseRight[2][2] = {{4, 2}, {3, 1}};  // [u1%2][eff]
    const uint8_t eff = (transformation < 2) ? transformation : 1;

    idwt_level_src_ctx &c = ctxs[i];
    c.v0             = v0;
    c.u0             = u0;
    c.u1             = u1;
    c.transformation = transformation;
    c.dir            = level_dir;
    c.has_child      = (i > 0);
    c.child_state    = (i > 0) ? &states[i - 1] : nullptr;
    c.ll0_buf        = (i == 0) ? &ld->ll0_buf : nullptr;
    c.hl_buf         = (sb_HL != nullptr) ? &hl_bufs[i] : nullptr;
    c.lh_buf         = (sb_LH != nullptr) ? &lh_bufs[i] : nullptr;
    c.hh_buf         = (sb_HH != nullptr) ? &hh_bufs[i] : nullptr;
    c.lp_width       = lp_width;
    c.hp_width       = hp_width;
    c.ll_y0          = static_cast<int32_t>(cr_ll->get_pos0().y);
    c.ll0_height     = (i == 0) ? static_cast<int32_t>(cr_ll->get_pos1().y - cr_ll->get_pos0().y) : 0;
    c.hl_y0          = (sb_HL != nullptr) ? static_cast<int32_t>(sb_HL->get_pos0().y) : 0;
    c.lh_y0          = (sb_LH != nullptr) ? static_cast<int32_t>(sb_LH->get_pos0().y) : 0;
    c.hh_y0          = (sb_HH != nullptr) ? static_cast<int32_t>(sb_HH->get_pos0().y) : 0;
    c.h_pse_left     = (u1 - u0 > 1) ? kHPseLeft[u0 % 2][eff]  : 0;
    c.h_pse_right    = (u1 - u0 > 1) ? kHPseRight[u1 % 2][eff] : 0;
    c.col_lo         = u0;
    c.col_hi         = u1;

    // lp_tmp only needed when has_child (child state writes output into it as fallback).
    // hp_tmp eliminated: HP data is read directly from subband row buffers.
    // ext_buf eliminated: in-place horizontal IDWT uses ring buffer PSE prefix.
    c.lp_tmp  = (i > 0) ? static_cast<sprec_t *>(aligned_mem_alloc(sizeof(sprec_t) * static_cast<size_t>(lp_width + SIMD_PADDING), 32)) : nullptr;

    idwt_2d_state_init(&states[i], u0, u1, v0, v1, transformation, level_dir,
                       idwt_level_src_fn, &ctxs[i]);
  }

  // In ring mode: free full-tile sample buffers — ring bufs are used instead.
  // The destructors of j2k_resolution/j2k_subband call aligned_mem_free(i_samples - DWT_LEFT_SLACK);
  // with nullptr set here that becomes free(nullptr) which is a safe no-op.
  if (ring_mode) {
    const int32_t NL_all = static_cast<int32_t>(NL);
    for (int32_t lv = 0; lv <= NL_all; ++lv) {
      j2k_resolution *cr = access_resolution(static_cast<uint8_t>(lv));
      if (cr->i_samples != nullptr) aligned_mem_free(cr->i_samples - DWT_LEFT_SLACK);
      cr->i_samples = nullptr;
      for (uint8_t b = 0; b < cr->num_bands; ++b) {
        j2k_subband *sb = cr->access_subband(b);
        if (sb->orientation != BAND_LL) {
          if (sb->i_samples != nullptr) aligned_mem_free(sb->i_samples - DWT_LEFT_SLACK);
          sb->i_samples = nullptr;
        }
      }
    }
  }
}

void j2k_tile_component::set_line_decode_col_range(uint32_t col_lo, uint32_t col_hi) {
  if (line_dec == nullptr) return;
  j2k_tcomp_line_dec *ld = line_dec.get();
  if (ld->NL_active <= 0) return;
  idwt_2d_state       *states = ld->states.get();
  idwt_level_src_ctx  *ctxs   = ld->ctxs.get();
  const int32_t NL_act        = ld->NL_active;
  // Seed with caller's finest-level range.  Coarser levels are derived by
  // halving and widening by the 9/7 filter support (4 samples each side,
  // conservative over-estimate that also covers 5/3 where support is 2).
  int32_t a = static_cast<int32_t>(col_lo);
  int32_t b = static_cast<int32_t>(col_hi);
  for (int32_t i = NL_act - 1; i >= 0; --i) {
    idwt_2d_state_set_col_range(&states[i], a, b);
    // Mirror the same range onto the level's source ctx so the horizontal
    // IDWT producer narrows its processing window to match.  Clamp to the
    // level's [u0, u1].
    int32_t ca = a, cb = b;
    if (ca < ctxs[i].u0) ca = ctxs[i].u0;
    if (cb > ctxs[i].u1) cb = ctxs[i].u1;
    if (ca > cb) ca = cb;
    ctxs[i].col_lo = ca;
    ctxs[i].col_hi = cb;
    // Next coarser level's output feeds this level's horizontal IDWT, which
    // reads up to 4 samples beyond each output column.  In subband coords
    // (half-width) that maps to ±2 samples; we round up to 4 for safety.
    const int32_t halved_lo = (a >> 1) - 4;
    const int32_t halved_hi = ((b + 1) >> 1) + 4;
    a = halved_lo;
    b = halved_hi;
  }
}

bool j2k_tile_component::pull_line(sprec_t *out) {
  if (line_dec == nullptr) return false;
  j2k_tcomp_line_dec *ld = line_dec.get();

  if (ld->NL_active == 0) {
    // No IDWT: output directly from LL0 subband row buffer.
    j2k_subband *sb = ld->ll0_buf.sb;
    if (ld->next_row >= static_cast<int32_t>(sb->get_pos1().y)) return false;
    const sprec_t *src = ld->ll0_buf.row_ptr(ld->next_row++);
    const int32_t  w   = static_cast<int32_t>(sb->get_pos1().x - sb->get_pos0().x);
    memcpy(out, src, sizeof(sprec_t) * static_cast<size_t>(w));
    return true;
  }

  return idwt_2d_state_pull_row(&ld->states.get()[ld->NL_active - 1], out);
}

sprec_t *j2k_tile_component::pull_line_ref() {
  if (line_dec == nullptr) return nullptr;
  j2k_tcomp_line_dec *ld = line_dec.get();

  if (ld->NL_active == 0) {
    j2k_subband *sb = ld->ll0_buf.sb;
    if (ld->next_row >= static_cast<int32_t>(sb->get_pos1().y)) return nullptr;
    return const_cast<sprec_t *>(ld->ll0_buf.row_ptr(ld->next_row++));
  }

  return idwt_2d_state_pull_row_ref(&ld->states.get()[ld->NL_active - 1]);
}

sprec_t *j2k_tile_component::pull_strip_into_buf(uint32_t count, uint32_t stride_floats) {
  if (line_dec == nullptr) return nullptr;
  j2k_tcomp_line_dec *ld = line_dec.get();

  // Grow-on-demand: the largest strip seen over the component's lifetime
  // sets the capacity, so steady-state decode allocates once and reuses.
  const size_t needed = static_cast<size_t>(count) * static_cast<size_t>(stride_floats);
  if (needed > ld->strip_buf_floats) {
    aligned_mem_free(ld->strip_buf);
    ld->strip_buf = static_cast<sprec_t *>(
        aligned_mem_alloc(needed * sizeof(sprec_t), 32));
    ld->strip_buf_floats = needed;
  }

  return pull_strip_into_buf(count, stride_floats, ld->strip_buf);
}

sprec_t *j2k_tile_component::pull_strip_into_buf(uint32_t count, uint32_t stride_floats,
                                                 sprec_t *dst) {
  if (line_dec == nullptr || dst == nullptr) return nullptr;
  if (count == 0) return dst;

  for (uint32_t r = 0; r < count; ++r) {
    const sprec_t *src = pull_line_ref();
    if (src == nullptr) {
      // Component exhausted earlier than the caller expected — zero the
      // tail so the finalize inner loop does not read uninitialised memory.
      const size_t tail_rows = count - r;
      std::memset(dst + static_cast<size_t>(r) * stride_floats, 0,
                  sizeof(sprec_t) * stride_floats * tail_rows);
      break;
    }
    std::memcpy(dst + static_cast<size_t>(r) * stride_floats, src,
                sizeof(sprec_t) * stride_floats);
  }
  return dst;
}

void j2k_tile_component::finalize_line_decode() {
  if (line_dec == nullptr) return;
  // Single-tile reuse: skip teardown so init_line_decode on the next frame
  // can reuse every allocation.  The j2k_tile_component destructor clears
  // line_dec_persistent_ before calling us, so final cleanup still happens
  // when the tile_component itself is destroyed.
  if (line_dec_persistent_) return;
  j2k_tcomp_line_dec *ld = line_dec.get();

  idwt_2d_state       *states  = ld->states.get();
  idwt_level_src_ctx  *ctxs    = ld->ctxs.get();
  j2k_subband_row_buf *hl_bufs = ld->hl_bufs.get();
  j2k_subband_row_buf *lh_bufs = ld->lh_bufs.get();
  j2k_subband_row_buf *hh_bufs = ld->hh_bufs.get();

  const int32_t n = ld->NL_active;
  for (int32_t i = 0; i < n; ++i) {
    idwt_2d_state_free(&states[i]);
    aligned_mem_free(ctxs[i].lp_tmp);
    hl_bufs[i].free_resources();
    // lh_bufs and hh_bufs are only initialised for BIDIR levels.
    const dwt_type dir = access_resolution(static_cast<uint8_t>(i + 1))->transform_direction;
    if (dir == DWT_BIDIR) {
      lh_bufs[i].free_resources();
      hh_bufs[i].free_resources();
    }
  }
  ld->ll0_buf.free_resources();
  aligned_mem_free(ld->strip_buf);
  ld->strip_buf        = nullptr;
  ld->strip_buf_floats = 0;
  line_dec.reset();
}

// Reset every cursor in the persistent line_dec so the next decode starts
// from a clean state.  Ring buffers, prefetch buffers, scratch allocations,
// idwt_2d_state contents, and all idwt_level_src_ctx fields stay alive —
// this function only walks the subband_row_buf objects and rewinds their
// strip_y0/strip_y1/ring_y0/prefetch_y0 cursors plus par_cnt.
//
// Must be kept in sync with j2k_subband_row_buf::init — any cursor field
// init sets to a "nothing decoded yet" value must be reset here too.
void j2k_tile_component::reset_line_decode_cursors() {
  if (line_dec == nullptr) return;
  j2k_tcomp_line_dec *ld = line_dec.get();

  auto rewind = [](j2k_subband_row_buf &rb) {
    rb.strip_y0 = -1;
    rb.strip_y1 = -1;
    rb.ring_y0  = -1;
    rb.bypass_decode = false;
#ifdef OPENHTJ2K_THREAD
    rb.prefetch_y0 = -1;
    rb.prefetch_y1 = -1;
    // After free_resources has NOT been called, the combined allocation is
    // still live, but ring_buf may have been swapped with prefetch_buf on a
    // prefetch hit — restore the canonical layout (ring_buf = low half,
    // prefetch_buf = high half) so decode_strip writes to the expected
    // location.  combined_buf is the stable base pointer.
    if (rb.combined_buf != nullptr && rb.ring_mode) {
      const int32_t sb_w =
          rb.sb ? static_cast<int32_t>(rb.sb->get_pos1().x - rb.sb->get_pos0().x) : 0;
      const int32_t sb_h =
          rb.sb ? static_cast<int32_t>(rb.sb->get_pos1().y - rb.sb->get_pos0().y) : 0;
      if (sb_w > 0 && sb_h > 0) {
        const size_t ring_rows  = static_cast<size_t>(rb.cb_h) + 1;
        const size_t buf_floats = ring_rows * static_cast<size_t>(rb.sb->stride);
        rb.ring_buf     = rb.combined_buf;
        rb.prefetch_buf = rb.combined_buf + buf_floats;
      }
    }
    rb.par_cnt.store(0, std::memory_order_relaxed);
    rb.par_tasks.clear();
#endif
  };

  // LL0 is always present.
  rewind(ld->ll0_buf);

  // Per active IDWT level: HL is always populated for BIDIR/HORZ; LH and HH
  // only for BIDIR levels.  Mirror the init loop's selective path so we
  // don't touch default-constructed slots for HORZ/VERT/NO levels.
  const int32_t n = ld->NL_active;
  if (n > 0) {
    j2k_subband_row_buf *hl_bufs = ld->hl_bufs.get();
    j2k_subband_row_buf *lh_bufs = ld->lh_bufs.get();
    j2k_subband_row_buf *hh_bufs = ld->hh_bufs.get();
    for (int32_t i = 0; i < n; ++i) {
      const dwt_type dir =
          access_resolution(static_cast<uint8_t>(i + 1))->transform_direction;
      if (dir == DWT_BIDIR) {
        rewind(hl_bufs[i]);
        rewind(lh_bufs[i]);
        rewind(hh_bufs[i]);
      } else if (dir == DWT_HORZ) {
        rewind(hl_bufs[i]);
      }
      // DWT_VERT is rejected by init_line_decode; DWT_NO has no HP bufs.
    }
    // Rewind idwt_2d_state cursors so the pull loop restarts at the top.
    idwt_2d_state *states = ld->states.get();
    for (int32_t i = 0; i < n; ++i) {
      idwt_2d_state_rewind(&states[i]);
    }
  }

  // next_row cursor: tied to LL0 geometry when NL_active == 0, else unused.
  j2k_resolution *r0 = access_resolution(0);
  ld->next_row       = static_cast<int32_t>(r0->get_pos0().y);
}

void j2k_tile_component::mark_line_dec_predecoded() {
  if (line_dec == nullptr) return;
  j2k_tcomp_line_dec *ld = line_dec.get();
  ld->ll0_buf.bypass_decode = true;
  j2k_subband_row_buf *hl_bufs = ld->hl_bufs.get();
  j2k_subband_row_buf *lh_bufs = ld->lh_bufs.get();
  j2k_subband_row_buf *hh_bufs = ld->hh_bufs.get();
  const int32_t n = ld->NL_active;
  for (int32_t i = 0; i < n; ++i) {
    hl_bufs[i].bypass_decode = true;
    const dwt_type dir = access_resolution(static_cast<uint8_t>(i + 1))->transform_direction;
    if (dir == DWT_BIDIR) {
      lh_bufs[i].bypass_decode = true;
      hh_bufs[i].bypass_decode = true;
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Line-based encode: init / push / finalize
// ─────────────────────────────────────────────────────────────────────────────

void j2k_tile_component::init_line_encode() {
  const uint8_t NL_enc = get_dwt_levels();
  if (NL_enc == 0) {
    line_enc = nullptr;
    return;
  }
  const uint8_t xform = get_transformation();

  line_enc      = std::make_unique<j2k_tcomp_line_enc>();
  auto *le      = line_enc.get();
  le->NL_active = static_cast<int32_t>(NL_enc);
  le->states    = std::make_unique<fdwt_2d_state[]>(NL_enc);
  le->ctxs      = std::make_unique<fdwt_level_sink_ctx[]>(NL_enc);

  // Cache raw pointers for index-by-int loops below.
  fdwt_2d_state       *states = le->states.get();
  fdwt_level_sink_ctx *ctxs   = le->ctxs.get();

  // state[i] corresponds to batch-encoder level r = i+1:
  //   input:  access_resolution(i+1)  (dimensions of cr)
  //   output: HL/LH/HH at access_resolution(i+1)
  //   LP out: access_resolution(i)    (LL0 or child state)
  for (int32_t i = 0; i < static_cast<int32_t>(NL_enc); ++i) {
    j2k_resolution *cr  = access_resolution(static_cast<uint8_t>(i + 1));
    j2k_resolution *ncr = access_resolution(static_cast<uint8_t>(i));

    const int32_t u0 = static_cast<int32_t>(cr->get_pos0().x);
    const int32_t u1 = static_cast<int32_t>(cr->get_pos1().x);
    const int32_t v0 = static_cast<int32_t>(cr->get_pos0().y);
    const int32_t v1 = static_cast<int32_t>(cr->get_pos1().y);

    j2k_subband *HL = cr->access_subband(0);
    j2k_subband *LH = cr->access_subband(1);
    j2k_subband *HH = cr->access_subband(2);

    auto *cx     = &ctxs[i];
    cx->u0       = u0;
    cx->lp_width = static_cast<int32_t>(ncr->get_pos1().x - ncr->get_pos0().x);
    cx->hp_width = static_cast<int32_t>(HL->get_pos1().x - HL->get_pos0().x);

    cx->lp_tmp = static_cast<sprec_t *>(
        aligned_mem_alloc(sizeof(sprec_t) * round_up(static_cast<uint32_t>(cx->lp_width), 32U), 32));
    cx->hp_tmp = static_cast<sprec_t *>(
        aligned_mem_alloc(sizeof(sprec_t) * round_up(static_cast<uint32_t>(cx->hp_width), 32U), 32));

    cx->hl_samples = HL->i_samples;
    cx->hl_stride  = HL->stride;
    cx->hl_y0      = static_cast<int32_t>(HL->get_pos0().y);
    cx->hl_h       = static_cast<int32_t>(HL->get_pos1().y - HL->get_pos0().y);

    cx->lh_samples = LH->i_samples;
    cx->lh_stride  = LH->stride;
    cx->lh_y0      = static_cast<int32_t>(LH->get_pos0().y);
    cx->lh_h       = static_cast<int32_t>(LH->get_pos1().y - LH->get_pos0().y);

    cx->hh_samples = HH->i_samples;
    cx->hh_stride  = HH->stride;
    cx->hh_y0      = static_cast<int32_t>(HH->get_pos0().y);

    cx->has_child   = (i > 0);
    cx->child_state = (i > 0) ? &states[i - 1] : nullptr;

    cx->ll0_samples = ncr->i_samples;
    cx->ll0_stride  = ncr->stride;
    cx->ll0_y0      = static_cast<int32_t>(ncr->get_pos0().y);

    fdwt_2d_state_init(&states[i], u0, u1, v0, v1, xform, fdwt_level_sink_fn, cx);
  }
}

void j2k_tile_component::push_line_enc(const sprec_t *in) {
  if (!line_enc) return;
  fdwt_2d_state_push_row(&line_enc->states.get()[line_enc->NL_active - 1], in);
}

void j2k_tile_component::finalize_line_encode() {
  if (!line_enc) return;
  auto *le = line_enc.get();

  fdwt_2d_state       *states = le->states.get();
  fdwt_level_sink_ctx *ctxs   = le->ctxs.get();

  // Flush from finest to coarsest: each flush may push LP rows into the next
  // coarser state, which in turn needs its own flush to drain remaining rows.
  for (int32_t i = le->NL_active - 1; i >= 0; --i) {
    fdwt_2d_state_flush(&states[i]);
  }

  // Free all allocated resources.
  for (int32_t i = 0; i < le->NL_active; ++i) {
    fdwt_2d_state_free(&states[i]);
    aligned_mem_free(ctxs[i].lp_tmp);
    aligned_mem_free(ctxs[i].hp_tmp);
  }
  line_enc.reset();
}

void j2k_tile_component::init(j2k_main_header *hdr, j2k_tilepart_header *tphdr, j2k_tile_base *tile,
                              uint16_t c, std::vector<int32_t *> img, bool lb_enc) {
  index = c;
  // copy both coding and quantization styles from COD or tile-part COD
  NL                 = tile->NL;
  reduce_NL          = tile->reduce_NL;
  codeblock_size     = tile->codeblock_size;
  Cmodes             = tile->Cmodes;
  transformation     = tile->transformation;
  precinct_size      = tile->precinct_size;
  quantization_style = tile->quantization_style;
  exponents          = tile->exponents;
  mantissas          = tile->mantissas;
  num_guard_bits     = tile->num_guard_bits;

  // set bitdepth from main header
  bitdepth = hdr->SIZ->get_bitdepth(c);
  element_siz subsampling;
  hdr->SIZ->get_subsampling_factor(subsampling, c);

  pos0.x = ceil_int(tile->pos0.x, subsampling.x);
  pos0.y = ceil_int(tile->pos0.y, subsampling.y);
  pos1.x = ceil_int(tile->pos1.x, subsampling.x);
  pos1.y = ceil_int(tile->pos1.y, subsampling.y);

  // apply COC, if any
  if (!tphdr->COC.empty()) {
    for (auto &i : tphdr->COC) {
      if (i->get_component_index() == c) {
        setCOCparams(i.get());
      }
    }
  } else {
    for (auto &i : hdr->COC) {
      if (i->get_component_index() == c) {
        setCOCparams(i.get());
      }
    }
  }

  // apply QCC, if any
  if (!tphdr->QCC.empty()) {
    for (auto &i : tphdr->QCC) {
      if (i->get_component_index() == c) {
        setQCCparams(i.get());
      }
    }
  } else {
    for (auto &i : hdr->QCC) {
      if (i->get_component_index() == c) {
        setQCCparams(i.get());
      }
    }
  }

  // apply RGN, if any
  if (!tphdr->RGN.empty()) {
    for (auto &i : tphdr->RGN) {
      if (i->get_component_index() == c) {
        setRGNparams(i.get());
      }
    }
  } else {
    for (auto &i : hdr->RGN) {
      if (i->get_component_index() == c) {
        setRGNparams(i.get());
      }
    }
  }

  // We consider "-reduce" parameter value to determine necessary buffer size.
  const uint32_t aligned_stride =
      round_up((ceil_int(pos1.x, 1U << tile->reduce_NL) - ceil_int(pos0.x, 1U << tile->reduce_NL)), 32U);
  const auto height             = static_cast<uint32_t>(ceil_int(pos1.y, 1U << tile->reduce_NL)
                                                        - ceil_int(pos0.y, 1U << tile->reduce_NL));

  element_siz Osiz;
  hdr->SIZ->get_image_origin(Osiz);
  // create tile samples, only for encoding;
  if (!img.empty()) {
    const auto width = static_cast<uint32_t>(pos1.x - pos0.x);
    // stride may differ from width with non-zero origin
    const uint32_t stride = hdr->SIZ->get_component_stride(this->index);
    if (lb_enc) {
      // In line-based encode mode, skip the full W×H int32 copy.
      // Store the raw input pointer; DC offset is applied row-by-row during encode.
      lb_src_ptr    = img[this->index] + (pos0.y - Osiz.y) * stride + pos0.x - Osiz.x;
      lb_src_stride = stride;
    } else {
      const uint32_t num_bufsamples = aligned_stride * height;
      samples = static_cast<int32_t *>(aligned_mem_alloc(sizeof(int32_t) * num_bufsamples, 32));
      int32_t *src = img[this->index] + (pos0.y - Osiz.y) * stride + pos0.x - Osiz.x;
      int32_t *dst = samples;
      for (uint32_t i = 0; i < height; ++i) {
        memcpy(dst, src, sizeof(int32_t) * width);
        src += stride;
        dst += aligned_stride;
      }
    }
  }
  lb_enc_mode = lb_enc;

  // Resolve DFS and ATK marker pointers for this component.
  if (dfs_index != 0) {
    dfs_info = hdr->get_dfs_marker(dfs_index);
  }
  if (transformation >= 2) {
    atk_info = hdr->get_atk_marker(transformation);
  }
}

void j2k_tile_component::setCOCparams(COC_marker *COC) {
  // coding style related properties
  // When DFS is active, SPcoc[0] encodes the DFS index, not the level count.
  // Keep NL from COD in that case.
  if (!COC->is_dfs_defined()) {
    NL = COC->get_dwt_levels();
  } else {
    dfs_index = COC->get_dfs_index();
  }
  COC->get_codeblock_size(codeblock_size);
  Cmodes         = COC->get_Cmodes();
  transformation = COC->get_transformation();
  precinct_size.clear();
  precinct_size.reserve(NL + 1U);
  element_siz tmp;
  for (uint8_t r = 0; r <= NL; r++) {
    COC->get_precinct_size(tmp, r);
    precinct_size.emplace_back(tmp);
  }
}

void j2k_tile_component::setQCCparams(QCC_marker *QCC) {
  quantization_style = QCC->get_quantization_style();
  exponents.clear();
  mantissas.clear();
  if (quantization_style != 1) {
    // lossless or lossy expounded
    const uint8_t max_entries = static_cast<uint8_t>(3 * NL + 1);
    const uint8_t actual      = std::min(max_entries, QCC->get_num_entries());
    for (uint8_t nb = 0; nb < actual; nb++) {
      exponents.push_back(QCC->get_exponents(nb));
      if (quantization_style == 2) {
        // lossy expounded
        mantissas.push_back(QCC->get_mantissas(nb));
      }
    }
  } else {
    // lossy derived
    exponents.push_back(QCC->get_exponents(0));
    mantissas.push_back(QCC->get_mantissas(0));
  }
  num_guard_bits = QCC->get_number_of_guardbits();
}

void j2k_tile_component::setRGNparams(RGN_marker *RGN) { this->ROIshift = RGN->get_ROIshift(); }

int32_t *j2k_tile_component::get_sample_address(uint32_t x, uint32_t y) {
  return this->samples + x + y * (this->pos1.x - this->pos0.x);
}

uint8_t j2k_tile_component::get_dwt_levels() { return this->NL; }

uint8_t j2k_tile_component::get_transformation() { return this->transformation; }

uint8_t j2k_tile_component::get_Cmodes() const { return this->Cmodes; }

uint8_t j2k_tile_component::get_bitdepth() const { return this->bitdepth; }

element_siz j2k_tile_component::get_precinct_size(uint8_t r) { return this->precinct_size[r]; }

element_siz j2k_tile_component::get_codeblock_size() { return this->codeblock_size; }

uint8_t j2k_tile_component::get_ROIshift() const { return this->ROIshift; }

j2k_resolution *j2k_tile_component::access_resolution(uint8_t r) { return &this->resolution[r]; }

void j2k_tile_component::create_resolutions(uint16_t numlayers, bool line_based, bool enc_lb) {
  const uint8_t total_res = static_cast<uint8_t>(NL + 1U);
  // Single allocation for all j2k_resolution objects (placement new) — replaces double-indirection.
  // Build into a local pointer under a guard so that if any j2k_resolution(...)
  // placement-new throws partway through the loop, the previously-constructed
  // resolutions are destroyed and the raw storage is freed.  num_resolutions
  // commits to the full count only on success.
  //
  // NOTE: methods called after each placement-new (set_nominal_ranges,
  // create_subbands, create_precincts) are individually exception-safe via their
  // own guards, so if one of them throws the resolution at index r is left
  // in a partially-initialized state and its destructor (run during this
  // function's unwind via the guard) will correctly clean up whatever
  // sub-resources WERE constructed.  The thread-pool dispatch path below
  // does NOT wait for in-flight futures on exception — that is a separate
  // pre-existing concern.
  auto *raw = static_cast<j2k_resolution *>(operator new[](sizeof(j2k_resolution) * total_res));
  placement_new_array_guard<j2k_resolution> res_guard(raw);

  float tmp_ranges[4]       = {1.0, 1.0, 1.0, 1.0};
  float child_ranges[32][4] = {{0}};
  float normalization       = 1.0;
  uint8_t normalizing_shift = 0;
  uint8_t nb, r, b;
  uint8_t nshift[32] = {0};
  uint32_t dx, dy;
  element_siz log2PP, PP, respos0, respos1;
  for (r = static_cast<uint8_t>(NL - reduce_NL); r > 0; --r) {
    if (dfs_info != nullptr) {
      dx = 1U << dfs_info->hor_depth[NL - r];
      dy = 1U << dfs_info->ver_depth[NL - r];
    } else {
      dx = dy = static_cast<uint32_t>(1 << (NL - r));
    }
    respos0.x = static_cast<uint32_t>(ceil_int(pos0.x, dx));
    respos0.y = static_cast<uint32_t>(ceil_int(pos0.y, dy));
    respos1.x = static_cast<uint32_t>(ceil_int(pos1.x, dx));
    respos1.y = static_cast<uint32_t>(ceil_int(pos1.y, dy));
    nb                = static_cast<uint8_t>(NL - r + 1);
    const dwt_type dir = (dfs_info != nullptr) ? dfs_info->get_dwt_type(nb) : DWT_BIDIR;
    const float K      = (transformation >= 2 && atk_info != nullptr) ? atk_info->get_Katk()
                                                                       : 1.230174104914001F;
    find_child_ranges(tmp_ranges, normalizing_shift, normalization, nb, respos0.x, respos1.x, respos0.y,
                      respos1.y, dir, K);
    nshift[r] = normalizing_shift;
    for (b = 0; b < 4; ++b) {
      child_ranges[r][b] = tmp_ranges[b];
    }
  }
  nshift[0]          = 0;
  child_ranges[0][0] = tmp_ranges[0];

#ifdef OPENHTJ2K_THREAD
  auto pool = ThreadPool::get();
  std::vector<std::future<int>> results;
#endif
  for (r = 0; r <= NL; r++) {
    if (dfs_info != nullptr) {
      dx = 1U << dfs_info->hor_depth[NL - r];
      dy = 1U << dfs_info->ver_depth[NL - r];
    } else {
      dx = dy = static_cast<uint32_t>(1 << (NL - r));
    }
    respos0.x          = static_cast<uint32_t>(ceil_int(pos0.x, dx));
    respos0.y          = static_cast<uint32_t>(ceil_int(pos0.y, dy));
    respos1.x          = static_cast<uint32_t>(ceil_int(pos1.x, dx));
    respos1.y          = static_cast<uint32_t>(ceil_int(pos1.y, dy));
    log2PP             = get_precinct_size(r);
    PP.x               = 1U << log2PP.x;
    PP.y               = 1U << log2PP.y;
    const uint32_t npw = (respos1.x > respos0.x) ? ceil_int(respos1.x, PP.x) - respos0.x / PP.x : 0;
    const uint32_t nph = (respos1.y > respos0.y) ? ceil_int(respos1.y, PP.y) - respos0.y / PP.y : 0;

    // Determine DFS type and band count for this resolution.
    dwt_type dir  = DWT_BIDIR;
    uint8_t dfs_nb = 0;  // 0 = use default (1 for r=0, 3 otherwise)
    if (dfs_info != nullptr && r > 0) {
      const uint8_t dfs_lev = static_cast<uint8_t>(NL - r + 1);  // 1=finest, NL=coarsest
      dir    = dfs_info->get_dwt_type(dfs_lev);
      dfs_nb = dfs_info->get_num_bands(r, NL);
    } else if (r == 0) {
      dfs_nb = 1;
    }

    new (&raw[r]) j2k_resolution(r, respos0, respos1, npw, nph,
                                 line_based || (enc_lb && r == this->NL && this->NL > 0),
                                 dfs_nb, dir);
    res_guard.commit_one();
    raw[r].set_nominal_ranges(child_ranges[r]);
    raw[r].normalizing_downshift = nshift[r];
    raw[r].normalizing_upshift   = nshift[r + 1];
    raw[r].create_subbands(this->pos0, this->pos1, this->NL, this->transformation, this->exponents,
                           this->mantissas, this->num_guard_bits, this->quantization_style,
                           this->bitdepth, line_based, this->dfs_info);
#ifdef OPENHTJ2K_THREAD
    if (pool && pool->num_threads() > 1) {
      results.emplace_back(pool->enqueue([r, raw, numlayers, this] {
        raw[r].create_precincts(precinct_size[r], numlayers, codeblock_size, Cmodes);
        return 0;
      }));
    } else {
      raw[r].create_precincts(precinct_size[r], numlayers, codeblock_size, Cmodes);
    }
  }
  for (auto &result : results) {
    result.get();
  }
#else
    raw[r].create_precincts(precinct_size[r], numlayers, codeblock_size, Cmodes);
  }
#endif
  res_guard.release();
  resolution      = raw;
  num_resolutions = total_res;
}

void j2k_tile_component::perform_dc_offset(const uint8_t transformation, const bool is_signed) {
  const int32_t shiftup = (transformation == 1) ? 0 : FRACBITS - this->bitdepth;
  if (shiftup < 0) {
    printf("WARNING: Over 13 bpp precision will be down-shifted to 12 bpp.\n");
  }
  // DC offset for signed input shall be 0
  const int32_t DC_OFFSET = (is_signed) ? 0 : 1 << (this->bitdepth - 1 + shiftup);

  // In LB encode mode (lb_enc_mode), samples was not allocated.
  // Store parameters for row-by-row application during encode_line_based().
  if (lb_enc_mode) {
    lb_dc_offset  = DC_OFFSET;
    lb_dc_shiftup = shiftup;
    return;
  }

  const int32_t stride    = round_up(static_cast<int32_t>(this->pos1.x - this->pos0.x), 32);
  const int32_t width     = static_cast<int32_t>(this->pos1.x - this->pos0.x);
  const int32_t height    = static_cast<int32_t>(this->pos1.y - this->pos0.y);
  int32_t *src            = this->samples;
#if defined(OPENHTJ2K_ENABLE_WASM_SIMD)
  {
    const v128_t vdoff = wasm_i32x4_splat(DC_OFFSET);
    if (shiftup < 0) {
      for (int32_t y = 0; y < height; ++y) {
        int32_t *sp = src + y * stride;
        int32_t len = width;
        for (int32_t i = 0; i < round_down(len, 8); i += 8) {
          v128_t v0 = wasm_v128_load(sp + i);
          v128_t v1 = wasm_v128_load(sp + i + 4);
          wasm_v128_store(sp + i,     wasm_i32x4_sub(wasm_i32x4_shr(v0, -shiftup), vdoff));
          wasm_v128_store(sp + i + 4, wasm_i32x4_sub(wasm_i32x4_shr(v1, -shiftup), vdoff));
        }
        for (int32_t i = round_down(len, 8); i < len; ++i) {
          sp[i] >>= -shiftup;
          sp[i] -= DC_OFFSET;
        }
      }
    } else {
      for (int32_t y = 0; y < height; ++y) {
        int32_t *sp = src + y * stride;
        int32_t len = width;
        for (int32_t i = 0; i < round_down(len, 8); i += 8) {
          v128_t v0 = wasm_v128_load(sp + i);
          v128_t v1 = wasm_v128_load(sp + i + 4);
          wasm_v128_store(sp + i,     wasm_i32x4_sub(wasm_i32x4_shl(v0, shiftup), vdoff));
          wasm_v128_store(sp + i + 4, wasm_i32x4_sub(wasm_i32x4_shl(v1, shiftup), vdoff));
        }
        for (int32_t i = round_down(len, 8); i < len; ++i) {
          sp[i] <<= shiftup;
          sp[i] -= DC_OFFSET;
        }
      }
    }
  }
#elif defined(OPENHTJ2K_ENABLE_ARM_NEON)
  const int32x4_t doff     = vdupq_n_s32(DC_OFFSET);
  const int32x4_t vshiftup = vdupq_n_s32(shiftup);
  for (int32_t y = 0; y < height; ++y) {
    int32_t *sp = src + y * stride;
    int32_t len = width;
    for (int32_t i = 0; i < round_down(len, 8); i += 8) {
      int32x4_t v0 = vld1q_s32(sp + i);
      int32x4_t v1 = vld1q_s32(sp + i + 4);
      vst1q_s32(sp + i, vsubq_s32(vshlq_s32(v0, vshiftup), doff));
      vst1q_s32(sp + i + 4, vsubq_s32(vshlq_s32(v1, vshiftup), doff));
    }
    for (int32_t i = round_down(len, 8); i < len; ++i) {
      sp[i] <<= shiftup;
      sp[i] -= DC_OFFSET;
    }
  }
#elif defined(OPENHTJ2K_TRY_AVX2) && defined(__AVX2__)
  if (shiftup < 0) {
    const __m256i doff = _mm256_set1_epi32(DC_OFFSET);
    for (int32_t y = 0; y < height; ++y) {
      int32_t *sp = src + y * stride;
      int32_t len = width;
      for (int32_t i = 0; i < round_down(len, 8); i += 8) {
        __m256i v            = *(__m256i *)(sp + i);
        *(__m256i *)(sp + i) = _mm256_sub_epi32(_mm256_srai_epi32(v, -shiftup), doff);
      }
      for (int32_t i = round_down(len, 8); i < len; ++i) {
        sp[i] >>= -shiftup;
        sp[i] -= DC_OFFSET;
      }
    }
  } else {
    const __m256i doff = _mm256_set1_epi32(DC_OFFSET);
    for (int32_t y = 0; y < height; ++y) {
      int32_t *sp = src + y * stride;
      int32_t len = width;
      for (int32_t i = 0; i < round_down(len, 8); i += 8) {
        __m256i v            = *(__m256i *)(sp + i);
        *(__m256i *)(sp + i) = _mm256_sub_epi32(_mm256_slli_epi32(v, shiftup), doff);
      }
      for (int32_t i = round_down(len, 8); i < len; ++i) {
        sp[i] <<= shiftup;
        sp[i] -= DC_OFFSET;
      }
    }
  }
#else
    if (shiftup < 0) {
      for (int32_t y = 0; y < height; ++y) {
        int32_t *sp = src + y * stride;
        int32_t len = width;
        for (int32_t i = 0; i < len; ++i) {
          sp[i] >>= -shiftup;
          sp[i] -= DC_OFFSET;
        }
      }
    } else {
      for (int32_t y = 0; y < height; ++y) {
        int32_t *sp = src + y * stride;
        int32_t len = width;
        for (int32_t i = 0; i < len; ++i) {
          sp[i] <<= shiftup;
          sp[i] -= DC_OFFSET;
        }
      }
    }
#endif
}

/********************************************************************************
 * j2k_tile
 *******************************************************************************/
j2k_tile::j2k_tile()
    : tile_part(),
      index(0),
      num_components(0),
      use_SOP(false),
      use_EPH(false),
      progression_order(0),
      numlayers(0),
      MCT(0),
      length(0),
      tile_buf(nullptr),
      packet_header(nullptr),
      num_tile_part(0),
      current_tile_part_pos(-1),
      tcomp(nullptr),
      ppt_header(nullptr),
      num_packets(0),
      packet(nullptr),
      // reduce_NL(0),
      Ccap15(0) {}

void j2k_tile::setCODparams(COD_marker *COD) {
  // coding style related properties
  use_SOP           = COD->is_use_SOP();
  use_EPH           = COD->is_use_EPH();
  progression_order = COD->get_progression_order();
  numlayers         = COD->get_number_of_layers();
  MCT               = COD->use_color_trafo();
  NL                = COD->get_dwt_levels();
  COD->get_codeblock_size(codeblock_size);
  Cmodes         = COD->get_Cmodes();
  transformation = COD->get_transformation();
  precinct_size.clear();
  precinct_size.reserve(NL + 1U);
  element_siz tmp;
  for (uint8_t r = 0; r <= NL; r++) {
    COD->get_precinct_size(tmp, r);
    precinct_size.emplace_back(tmp);
  }
}

void j2k_tile::setQCDparams(QCD_marker *QCD) {
  quantization_style = QCD->get_quantization_style();
  exponents.clear();
  mantissas.clear();
  if (quantization_style != 1) {
    // lossless or lossy expounded
    const uint8_t max_entries = static_cast<uint8_t>(3 * NL + 1);
    const uint8_t actual      = std::min(max_entries, QCD->get_num_entries());
    for (uint8_t nb = 0; nb < actual; nb++) {
      exponents.push_back(QCD->get_exponents(nb));
      if (quantization_style == 2) {
        // lossy expounded
        mantissas.push_back(QCD->get_mantissas(nb));
      }
    }
  } else {
    // lossy derived
    exponents.push_back(QCD->get_exponents(0));
    mantissas.push_back(QCD->get_mantissas(0));
  }
  num_guard_bits = QCD->get_number_of_guardbits();
}

void j2k_tile::dec_init(uint16_t idx, j2k_main_header &main_header, uint8_t reduce_levels) {
  index          = idx;
  num_components = main_header.SIZ->get_num_components();
  // set coding style related properties from main header
  setCODparams(main_header.COD.get());
  // set quantization style related properties from main header
  setQCDparams(main_header.QCD.get());
  // set Ccap15(HTJ2K only or mixed)
  Ccap15 = (main_header.CAP != nullptr) ? main_header.CAP->get_Ccap(15) : 0;
  // set resolution reduction, if any
  reduce_NL = reduce_levels;
}

void j2k_tile::add_tile_part(SOT_marker &tmpSOT, j2c_src_memory &in, j2k_main_header &main_header) {
  this->length += tmpSOT.get_tile_part_length();
  // this->tile_part.push_back(move(make_unique<j2k_tile_part>(num_components)));
  this->tile_part.push_back(MAKE_UNIQUE<j2k_tile_part>(num_components));
  this->num_tile_part++;
  this->current_tile_part_pos++;
  this->tile_part[static_cast<size_t>(current_tile_part_pos)]->set_SOT(tmpSOT);
  this->tile_part[static_cast<size_t>(current_tile_part_pos)]->read(in);
  j2k_tilepart_header *tphdr = this->tile_part[static_cast<size_t>(current_tile_part_pos)]->header.get();

  uint8_t tile_part_index = tmpSOT.get_tile_part_index();
  if (tile_part_index == 0) {
    // this->set_index(tmpSOT->get_tile_index(), main_header);
    element_siz numTiles;
    element_siz Siz, Osiz, Tsiz, TOsiz;
    main_header.get_number_of_tiles(numTiles.x, numTiles.y);
    uint16_t p = static_cast<uint16_t>(this->index % numTiles.x);
    uint16_t q = static_cast<uint16_t>(this->index / numTiles.x);
    main_header.SIZ->get_image_size(Siz);
    main_header.SIZ->get_image_origin(Osiz);
    main_header.SIZ->get_tile_size(Tsiz);
    main_header.SIZ->get_tile_origin(TOsiz);

    this->pos0.x = std::max(TOsiz.x + p * Tsiz.x, Osiz.x);
    this->pos0.y = std::max(TOsiz.y + q * Tsiz.y, Osiz.y);
    this->pos1.x = std::min(TOsiz.x + (p + 1U) * Tsiz.x, Siz.x);
    this->pos1.y = std::min(TOsiz.y + (q + 1U) * Tsiz.y, Siz.y);

    // set coding style related properties from tile-part header
    if (tphdr->COD != nullptr) {
      setCODparams(tphdr->COD.get());
    }
    // set quantization style related properties from tile-part header
    if (tphdr->QCD != nullptr) {
      setQCDparams(tphdr->QCD.get());
    }

    // create tile components — skipped on the single-tile reuse path, where
    // prepare_for_next_frame() has cleared mutable state but left the
    // tile_component array (and all of its line_dec / resolution / precinct
    // / codeblock sub-allocations) alive.  tcomp[c].init() is still called
    // unconditionally because it only writes scalar fields and vectors that
    // are idempotent under identical main-header bytes.
    if (!structure_built_) {
      this->tcomp = MAKE_UNIQUE<j2k_tile_component[]>(num_components);
    }
    for (uint16_t c = 0; c < num_components; c++) {
      this->tcomp[c].init(&main_header, tphdr, this, c);
    }

    // apply POC, if any
    if (tphdr->POC != nullptr) {
      for (unsigned long i = 0; i < tphdr->POC->nPOC; ++i) {
        porder_info.add(tphdr->POC->RSpoc[i], tphdr->POC->CSpoc[i], tphdr->POC->LYEpoc[i],
                        tphdr->POC->REpoc[i], tphdr->POC->CEpoc[i], tphdr->POC->Ppoc[i]);
      }
    } else if (main_header.POC != nullptr) {
      for (unsigned long i = 0; i < main_header.POC->nPOC; ++i) {
        porder_info.add(main_header.POC->RSpoc[i], main_header.POC->CSpoc[i], main_header.POC->LYEpoc[i],
                        main_header.POC->REpoc[i], main_header.POC->CEpoc[i], main_header.POC->Ppoc[i]);
      }
    }
  }
}

void j2k_tile::create_tile_buf(j2k_main_header &main_header) {
  // On the single-tile reuse path, num_packets and porder_info accumulate
  // across calls (see lines that `+=` and `.add`) unless we reset them
  // here.  The packet[] array is allocated below from num_packets, so it
  // must start from zero on every call — the reuse-path guard further
  // down skips the actual allocation when the shape is unchanged.
  if (structure_built_) {
    num_packets = 0;
    porder_info.nPOC = 0;
    porder_info.RSpoc.clear();
    porder_info.CSpoc.clear();
    porder_info.LYEpoc.clear();
    porder_info.REpoc.clear();
    porder_info.CEpoc.clear();
    porder_info.Ppoc.clear();
  }

  // Concatenate tile-parts into a tile.  On the reuse path, reset the
  // existing buf_chain instead of heap-allocating a new one.
  uint8_t t = 0;
  if (structure_built_ && tile_buf) {
    tile_buf->reset(num_tile_part);
  } else {
    this->tile_buf = MAKE_UNIQUE<buf_chain>(num_tile_part);
  }
  for (unsigned long i = 0; i < num_tile_part; i++) {
    // If a length of a tile-part is 0, buf number 't' should not be
    // incremented!!
    if (this->tile_part[i]->get_length() != 0) {
      this->tile_buf->set_buf_node(t, this->tile_part[i]->get_buf(), this->tile_part[i]->get_length());
      t++;
    }
  }
  this->tile_buf->activate();
  // If PPT exits, create PPT buf chain
  if (!this->tile_part[0]->header->PPT.empty()) {
    ppt_header = MAKE_UNIQUE<buf_chain>();
    for (unsigned long i = 0; i < num_tile_part; i++) {
      for (auto &ppt : this->tile_part[i]->header->PPT) {
        ppt_header->add_buf_node(ppt->pptbuf, ppt->pptlen);
      }
    }
    ppt_header->activate();
  }

  // determine the location of the packet header
  this->packet_header = nullptr;
  buf_chain *ppp;
  if (main_header.get_ppm_header() != nullptr) {
    assert(ppt_header == nullptr);
    ppp = main_header.get_ppm_header();
    // TODO: this implementation may not be enough because this does not
    // consider "tile-part". MARK: find beginning of the packet header for a
    // tile!
    ppp->activate(this->index);
    packet_header = ppp;  // main_header.get_ppm_header();
  } else if (ppt_header != nullptr) {
    assert(main_header.get_ppm_header() == nullptr);
    packet_header = ppt_header.get();
  } else {
    packet_header = this->tile_buf.get();
  }

  // create resolution levels, subbands, precincts, precinct subbands and codeblocks
  uint32_t max_res_precincts = 0;
  uint8_t c_NL;
  uint8_t max_c_NL = 0;
  for (uint16_t c = 0; c < num_components; c++) {
    c_NL = this->tcomp[c].NL;
    if (c_NL < this->reduce_NL) {
      throw std::runtime_error("Resolution level reduction exceeds the DWT level");
    }
    // Single-tile reuse: skip create_resolutions (structural allocations),
    // but keep the counting loop — it warms resolution/precinct objects
    // into L1/L2 cache before the progression-order packet-parse loop.
    if (!structure_built_) {
      this->tcomp[c].create_resolutions(numlayers, this->line_based_decode);
    }
    max_c_NL = std::max(c_NL, max_c_NL);
    j2k_resolution *cr;
    for (uint8_t r = 0; r <= c_NL; r++) {
      cr = this->tcomp[c].access_resolution(r);
      num_packets += cr->npw * cr->nph;
      max_res_precincts = std::max(cr->npw * cr->nph, max_res_precincts);
    }
  }
  num_packets *= numlayers;
  // TODO: create packets with progression order
  // The packet[] array is a scratch "log of packets seen during parse" used
  // only within create_tile_buf (j2c_packet's decode ctor is the only
  // call site).  On the reuse path its size is unchanged, so we reallocate
  // only when num_packets differs — which shouldn't happen in practice for
  // same-geometry streams, but the guard keeps the contract simple.
  if (packet == nullptr || !structure_built_) {
    this->packet = MAKE_UNIQUE<j2c_packet[]>(static_cast<size_t>(num_packets));
  }
  // ── Packet parsing ─────────────────────────────────────────────────────
  // After the first frame, the progression order never changes, so we
  // cache the (component, resolution, precinct) traversal order and replay
  // it directly — skipping the progression-order switch, the 4D
  // is_packet_read vector, and all per-frame p_x/p_y/x_examin/y_examin
  // heap allocations.
  if (crp_cached_) {
    // Fast path: replay the cached traversal order.
    size_t packet_count = 0;
    for (const auto &e : cached_crp_) {
      j2k_resolution *cr = this->tcomp[e.c].access_resolution(e.r);
      j2k_precinct   *cp = cr->access_precinct(e.p);
      const bool skip = precinct_filter_ && !precinct_filter_(e.c, e.r, e.p);
      this->packet[packet_count++] = j2c_packet(0, e.r, e.c, e.p, packet_header, tile_buf.get());
      const uint64_t _pk_off =
          packet_observer_ ? tile_buf->get_total_position() : 0u;
      this->read_packet(cp, 0, cr->num_bands, skip);
      if (packet_observer_) {
        packet_observer_(static_cast<uint16_t>(e.c), e.r, e.p, /*layer=*/0, _pk_off,
                         tile_buf->get_total_position() - _pk_off);
      }
    }
  } else {
    // First frame: run the full progression-order traversal and record
    // every (c, r, p) tuple into cached_crp_.
    porder_info.add(0, 0, this->numlayers, static_cast<uint8_t>(max_c_NL + 1), this->num_components,
                    this->progression_order);
    uint8_t PO, RS, RE, r, local_RE;
    uint16_t LYE, CS, CE, c, l;
    uint32_t p;
    bool x_cond, y_cond;
    std::vector<std::vector<std::vector<std::vector<bool>>>> is_packet_read(
        numlayers, std::vector<std::vector<std::vector<bool>>>(
                       max_c_NL + 1U, std::vector<std::vector<bool>>(
                                          num_components, std::vector<bool>(max_res_precincts, false))));
    j2k_resolution *cr  = nullptr;
    j2k_precinct *cp    = nullptr;
    size_t packet_count = 0;
    cached_crp_.clear();
    cached_crp_.reserve(num_packets);
    for (unsigned long i = 0; i < porder_info.nPOC; ++i) {
      RS  = porder_info.RSpoc[i];
      CS  = porder_info.CSpoc[i];
      LYE = std::min(porder_info.LYEpoc[i], numlayers);
      RE  = porder_info.REpoc[i];
      CE  = std::min(porder_info.CEpoc[i], num_components);
      PO  = porder_info.Ppoc[i];
      std::vector<std::vector<uint32_t>> p_x(static_cast<uint32_t>(num_components),
                                             std::vector<uint32_t>(static_cast<uint32_t>(max_c_NL + 1), 0));
      std::vector<std::vector<uint32_t>> p_y(static_cast<uint32_t>(num_components),
                                             std::vector<uint32_t>(static_cast<uint32_t>(max_c_NL + 1), 0));

      element_siz PP, cPP, csub;
      std::vector<uint32_t> x_examin;
      std::vector<uint32_t> y_examin;

      switch (PO) {
        case 0:  // LRCP
          for (l = 0; l < LYE; l++) {
            for (r = RS; r < RE; r++) {
              for (c = CS; c < CE; c++) {
                c_NL = this->tcomp[c].NL;
                if (r <= c_NL) {
                  cr = this->tcomp[c].access_resolution(r);
                  if (!cr->is_empty) {
                    for (p = 0; p < cr->npw * cr->nph; p++) {
                      cp = cr->access_precinct(p);
                      if (!is_packet_read[l][r][c][p]) {
                        cached_crp_.push_back({static_cast<uint8_t>(c), r, static_cast<uint16_t>(p)});
                        this->packet[packet_count++] = j2c_packet(l, r, c, p, packet_header, tile_buf.get());
                        {
                          const uint64_t _pk_off =
                              packet_observer_ ? tile_buf->get_total_position() : 0u;
                          this->read_packet(cp, l, cr->num_bands,
                                            precinct_filter_
                                                && !precinct_filter_(static_cast<uint16_t>(c), r, p));
                          if (packet_observer_) {
                            packet_observer_(static_cast<uint16_t>(c), r, p, l, _pk_off,
                                             tile_buf->get_total_position() - _pk_off);
                          }
                        }
                        is_packet_read[l][r][c][p] = true;
                      }
                    }
                  }
                }
              }
            }
          }
          break;
        case 1:  // RLCP
          for (r = RS; r < RE; r++) {
            for (l = 0; l < LYE; l++) {
              for (c = CS; c < CE; c++) {
                c_NL = this->tcomp[c].NL;
                if (r <= c_NL) {
                  cr = this->tcomp[c].access_resolution(r);
                  if (!cr->is_empty) {
                    for (p = 0; p < cr->npw * cr->nph; p++) {
                      cp = cr->access_precinct(p);
                      if (!is_packet_read[l][r][c][p]) {
                        cached_crp_.push_back({static_cast<uint8_t>(c), r, static_cast<uint16_t>(p)});
                        this->packet[packet_count++] = j2c_packet(l, r, c, p, packet_header, tile_buf.get());
                        {
                          const uint64_t _pk_off =
                              packet_observer_ ? tile_buf->get_total_position() : 0u;
                          this->read_packet(cp, l, cr->num_bands,
                                            precinct_filter_
                                                && !precinct_filter_(static_cast<uint16_t>(c), r, p));
                          if (packet_observer_) {
                            packet_observer_(static_cast<uint16_t>(c), r, p, l, _pk_off,
                                             tile_buf->get_total_position() - _pk_off);
                          }
                        }
                        is_packet_read[l][r][c][p] = true;
                      }
                    }
                  }
                }
              }
            }
          }
          break;
        case 2:  // RPCL
          this->find_gcd_of_precinct_size(PP);
          x_examin.push_back(pos0.x);
          for (uint32_t x = 0; x < this->pos1.x; x += (1U << PP.x)) {
            if (x > pos0.x) {
              x_examin.push_back(x);
            }
          }
          y_examin.push_back(pos0.y);
          for (uint32_t y = 0; y < this->pos1.y; y += (1U << PP.y)) {
            if (y > pos0.y) {
              y_examin.push_back(y);
            }
          }
          for (r = RS; r < RE; r++) {
            for (uint32_t y : y_examin) {
              for (uint32_t x : x_examin) {
                for (c = CS; c < CE; c++) {
                  c_NL = this->tcomp[c].NL;
                  if (r <= c_NL) {
                    cPP = this->tcomp[c].get_precinct_size(r);
                    cr  = this->tcomp[c].access_resolution(r);
                    if (!cr->is_empty) {
                      element_siz tr0 = cr->get_pos0();
                      x_cond          = false;
                      y_cond          = false;
                      main_header.SIZ->get_subsampling_factor(csub, c);
                      {
                        const DFS_marker *cdfs = this->tcomp[c].dfs_info;
                        const uint8_t hd = cdfs ? cdfs->hor_depth[c_NL - r] : static_cast<uint8_t>(c_NL - r);
                        const uint8_t vd = cdfs ? cdfs->ver_depth[c_NL - r] : static_cast<uint8_t>(c_NL - r);
                        x_cond = (x % (csub.x * (1U << (cPP.x + hd))) == 0)
                                 || ((x == pos0.x)
                                     && ((tr0.x * (1U << hd)) % (1U << (cPP.x + hd)) != 0));
                        y_cond = (y % (csub.y * (1U << (cPP.y + vd))) == 0)
                                 || ((y == pos0.y)
                                     && ((tr0.y * (1U << vd)) % (1U << (cPP.y + vd)) != 0));
                      }
                      if (x_cond && y_cond) {
                        p  = p_x[c][r] + p_y[c][r] * cr->npw;
                        cp = cr->access_precinct(p);
                        for (l = 0; l < LYE; l++) {
                          if (!is_packet_read[l][r][c][p]) {
                            cached_crp_.push_back({static_cast<uint8_t>(c), r, static_cast<uint16_t>(p)});
                            this->packet[packet_count++] =
                                j2c_packet(l, r, c, p, packet_header, tile_buf.get());
                            {
                          const uint64_t _pk_off =
                              packet_observer_ ? tile_buf->get_total_position() : 0u;
                          this->read_packet(cp, l, cr->num_bands,
                                            precinct_filter_
                                                && !precinct_filter_(static_cast<uint16_t>(c), r, p));
                          if (packet_observer_) {
                            packet_observer_(static_cast<uint16_t>(c), r, p, l, _pk_off,
                                             tile_buf->get_total_position() - _pk_off);
                          }
                        }
                            is_packet_read[l][r][c][p] = true;
                          }
                        }
                        p_x[c][r] += 1;
                        if (p_x[c][r] == cr->npw) {
                          p_x[c][r] = 0;
                          p_y[c][r] += 1;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
          break;
        case 3:  // PCRL
          this->find_gcd_of_precinct_size(PP);
          x_examin.push_back(pos0.x);
          for (uint32_t x = 0; x < this->pos1.x; x += (1U << PP.x)) {
            if (x > pos0.x) {
              x_examin.push_back(x);
            }
          }
          y_examin.push_back(pos0.y);
          for (uint32_t y = 0; y < this->pos1.y; y += (1U << PP.y)) {
            if (y > pos0.y) {
              y_examin.push_back(y);
            }
          }
          for (uint32_t y : y_examin) {
            for (uint32_t x : x_examin) {
              for (c = CS; c < CE; c++) {
                c_NL     = this->tcomp[c].NL;
                local_RE = ((c_NL + 1) < RE) ? static_cast<uint8_t>(c_NL + 1U) : RE;
                for (r = RS; r < local_RE; r++) {
                  cPP = this->tcomp[c].get_precinct_size(r);
                  cr  = this->tcomp[c].access_resolution(r);
                  if (!cr->is_empty) {
                    element_siz tr0 = cr->get_pos0();
                    x_cond          = false;
                    y_cond          = false;
                    main_header.SIZ->get_subsampling_factor(csub, c);
                    {
                      const DFS_marker *cdfs = this->tcomp[c].dfs_info;
                      const uint8_t hd = cdfs ? cdfs->hor_depth[c_NL - r] : static_cast<uint8_t>(c_NL - r);
                      const uint8_t vd = cdfs ? cdfs->ver_depth[c_NL - r] : static_cast<uint8_t>(c_NL - r);
                      x_cond = (x % (csub.x * (1U << (cPP.x + hd))) == 0)
                               || ((x == pos0.x)
                                   && ((tr0.x * (1U << hd)) % (1U << (cPP.x + hd)) != 0));
                      y_cond = (y % (csub.y * (1U << (cPP.y + vd))) == 0)
                               || ((y == pos0.y)
                                   && ((tr0.y * (1U << vd)) % (1U << (cPP.y + vd)) != 0));
                    }
                    if (x_cond && y_cond) {
                      p  = p_x[c][r] + p_y[c][r] * cr->npw;
                      cp = cr->access_precinct(p);
                      for (l = 0; l < LYE; l++) {
                        if (!is_packet_read[l][r][c][p]) {
                          cached_crp_.push_back({static_cast<uint8_t>(c), r, static_cast<uint16_t>(p)});
                          this->packet[packet_count++] =
                              j2c_packet(l, r, c, p, packet_header, tile_buf.get());
                          {
                          const uint64_t _pk_off =
                              packet_observer_ ? tile_buf->get_total_position() : 0u;
                          this->read_packet(cp, l, cr->num_bands,
                                            precinct_filter_
                                                && !precinct_filter_(static_cast<uint16_t>(c), r, p));
                          if (packet_observer_) {
                            packet_observer_(static_cast<uint16_t>(c), r, p, l, _pk_off,
                                             tile_buf->get_total_position() - _pk_off);
                          }
                        }
                          is_packet_read[l][r][c][p] = true;
                        }
                      }
                      p_x[c][r] += 1;
                      if (p_x[c][r] == cr->npw) {
                        p_x[c][r] = 0;
                        p_y[c][r] += 1;
                      }
                    }
                  }
                }
              }
            }
          }
          break;
        case 4:  // CPRL
          this->find_gcd_of_precinct_size(PP);
          x_examin.push_back(pos0.x);
          for (uint32_t x = 0; x < this->pos1.x; x += (1U << PP.x)) {
            if (x > pos0.x) {
              x_examin.push_back(x);
            }
          }
          y_examin.push_back(pos0.y);
          for (uint32_t y = 0; y < this->pos1.y; y += (1U << PP.y)) {
            if (y > pos0.y) {
              y_examin.push_back(y);
            }
          }
          for (c = CS; c < CE; c++) {
            c_NL     = this->tcomp[c].NL;
            local_RE = ((c_NL + 1) < RE) ? static_cast<uint8_t>(c_NL + 1U) : RE;
            for (uint32_t y : y_examin) {
              for (uint32_t x : x_examin) {
                for (r = RS; r < local_RE; r++) {
                  cPP = this->tcomp[c].get_precinct_size(r);
                  cr  = this->tcomp[c].access_resolution(r);
                  if (!cr->is_empty) {
                    element_siz tr0 = cr->get_pos0();
                    x_cond          = false;
                    y_cond          = false;
                    main_header.SIZ->get_subsampling_factor(csub, c);
                    {
                      const DFS_marker *cdfs = this->tcomp[c].dfs_info;
                      const uint8_t hd = cdfs ? cdfs->hor_depth[c_NL - r] : static_cast<uint8_t>(c_NL - r);
                      const uint8_t vd = cdfs ? cdfs->ver_depth[c_NL - r] : static_cast<uint8_t>(c_NL - r);
                      x_cond = (x % (csub.x * (1U << (cPP.x + hd))) == 0)
                               || ((x == pos0.x)
                                   && ((tr0.x * (1U << hd)) % (1U << (cPP.x + hd)) != 0));
                      y_cond = (y % (csub.y * (1U << (cPP.y + vd))) == 0)
                               || ((y == pos0.y)
                                   && ((tr0.y * (1U << vd)) % (1U << (cPP.y + vd)) != 0));
                    }
                    if (x_cond && y_cond) {
                      p  = p_x[c][r] + p_y[c][r] * cr->npw;
                      cp = cr->access_precinct(p);
                      for (l = 0; l < LYE; l++) {
                        if (!is_packet_read[l][r][c][p]) {
                          cached_crp_.push_back({static_cast<uint8_t>(c), r, static_cast<uint16_t>(p)});
                          this->packet[packet_count++] =
                              j2c_packet(l, r, c, p, packet_header, tile_buf.get());
                          {
                          const uint64_t _pk_off =
                              packet_observer_ ? tile_buf->get_total_position() : 0u;
                          this->read_packet(cp, l, cr->num_bands,
                                            precinct_filter_
                                                && !precinct_filter_(static_cast<uint16_t>(c), r, p));
                          if (packet_observer_) {
                            packet_observer_(static_cast<uint16_t>(c), r, p, l, _pk_off,
                                             tile_buf->get_total_position() - _pk_off);
                          }
                        }
                          is_packet_read[l][r][c][p] = true;
                        }
                      }
                      p_x[c][r] += 1;
                      if (p_x[c][r] == cr->npw) {
                        p_x[c][r] = 0;
                        p_y[c][r] += 1;
                      }
                    }
                  }
                }
              }
            }
          }
          break;

        default:
          printf(
              "ERROR: Progression order number shall be in the range from 0 "
              "to 4\n");
          throw std::exception();
          // break;
      }
    }
    crp_cached_ = true;
  }
  // Mark the tree as built so the next create_tile_buf call on this tile
  // skips structural allocations.  The decoder_impl cache path reads this
  // flag via the public accessor before deciding whether to call
  // prepare_for_next_frame() (reuse) or dec_init() (fresh build).
  structure_built_ = true;
}

void j2k_tile::prepare_for_next_frame() {
  // Reset per-frame packet-parsing state across every codeblock and
  // tagtree, reset per-frame tile-level accumulators, and drop the
  // last frame's tile_buf / ppt_header / tile_part owning pointers so
  // the next add_tile_part loop repopulates from scratch.  Does NOT
  // touch tcomp[] (resolution/precinct/codeblock tree) or line_dec —
  // those are what we're trying to keep alive.
  if (!structure_built_) return;
  // Tile-level mutable fields.
  tile_part.clear();
  length                = 0;
  num_tile_part         = 0;
  current_tile_part_pos = -1;
  tile_buf.reset();
  ppt_header.reset();
  packet_header        = nullptr;
  // POC accumulator: cleared inside create_tile_buf via the same branch,
  // but do it here as well for robustness if a caller skips create_tile_buf.
  porder_info.nPOC = 0;
  porder_info.RSpoc.clear();
  porder_info.CSpoc.clear();
  porder_info.LYEpoc.clear();
  porder_info.REpoc.clear();
  porder_info.CEpoc.clear();
  porder_info.Ppoc.clear();
  num_packets = 0;

  // Walk every precinct_subband to reset its tagtrees + codeblocks.
  for (uint16_t c = 0; c < num_components; ++c) {
    const uint8_t c_NL = tcomp[c].NL;
    for (uint8_t r = 0; r <= c_NL; ++r) {
      j2k_resolution *cr = tcomp[c].access_resolution(r);
      if (cr == nullptr) continue;
      const uint32_t np = cr->npw * cr->nph;
      for (uint32_t p = 0; p < np; ++p) {
        j2k_precinct *cp = cr->access_precinct(p);
        if (cp == nullptr) continue;
        const uint8_t nb = cp->get_num_bands();
        for (uint8_t b = 0; b < nb; ++b) {
          j2k_precinct_subband *cpb = cp->access_pband(b);
          if (cpb != nullptr) cpb->reset_for_next_frame();
        }
      }
    }
    // Keep line_dec alive and let init_line_decode on the next frame see
    // the persistence flag so it short-circuits to a cursor reset.
    tcomp[c].set_line_decode_persistent(true);
  }
}

void j2k_tile::set_line_decode_persistent_all(bool on) {
  if (tcomp == nullptr) return;
  for (uint16_t c = 0; c < num_components; ++c) {
    tcomp[c].set_line_decode_persistent(on);
  }
}

void j2k_tile::construct_packets(j2k_main_header &main_header) {
  // derive number of packets needed to be created
  num_packets                = 0;
  uint32_t max_res_precincts = 0;
  uint8_t c_NL;
  uint8_t max_c_NL = 0;

  for (uint16_t c = 0; c < num_components; c++) {
    c_NL     = this->tcomp[c].NL;
    max_c_NL = std::max(c_NL, max_c_NL);
    j2k_resolution *cr;
    for (uint8_t r = 0; r <= c_NL; r++) {
      cr = this->tcomp[c].access_resolution(r);
      num_packets += cr->npw * cr->nph;
      max_res_precincts = std::max(cr->npw * cr->nph, max_res_precincts);
    }
  }
  num_packets *= this->numlayers;
  this->packet = MAKE_UNIQUE<j2c_packet[]>(static_cast<size_t>(num_packets));

  // need to construct a POC marker from progression order value in COD marker
  porder_info.add(0, 0, this->numlayers, static_cast<uint8_t>(max_c_NL + 1), this->num_components,
                  this->progression_order);
  uint8_t PO, RS, RE, r, local_RE;
  uint16_t LYE, CS, CE, c, l;
  uint32_t p;
  bool x_cond, y_cond;
  std::vector<std::vector<std::vector<std::vector<bool>>>> is_packet_created(
      numlayers, std::vector<std::vector<std::vector<bool>>>(
                     max_c_NL + 1U, std::vector<std::vector<bool>>(
                                        num_components, std::vector<bool>(max_res_precincts, false))));

  j2k_resolution *cr  = nullptr;
  j2k_precinct *cp    = nullptr;
  size_t packet_count = 0;
  for (unsigned long i = 0; i < porder_info.nPOC; ++i) {
    RS  = porder_info.RSpoc[i];
    CS  = porder_info.CSpoc[i];
    LYE = std::min(porder_info.LYEpoc[i], numlayers);
    RE  = porder_info.REpoc[i];
    CE  = std::min(porder_info.CEpoc[i], num_components);
    PO  = porder_info.Ppoc[i];
    std::vector<std::vector<uint32_t>> p_x(static_cast<uint32_t>(num_components),
                                           std::vector<uint32_t>(static_cast<uint32_t>(max_c_NL + 1), 0));
    std::vector<std::vector<uint32_t>> p_y(static_cast<uint32_t>(num_components),
                                           std::vector<uint32_t>(static_cast<uint32_t>(max_c_NL + 1), 0));

    element_siz PP, cPP, csub;
    std::vector<uint32_t> x_examin;
    std::vector<uint32_t> y_examin;

    switch (PO) {
      case 0:  // LRCP
        for (l = 0; l < LYE; l++) {
          for (r = RS; r < RE; r++) {
            for (c = CS; c < CE; c++) {
              c_NL = this->tcomp[c].NL;
              if (r <= c_NL) {
                cr = this->tcomp[c].access_resolution(r);
                if (!cr->is_empty) {
                  for (p = 0; p < cr->npw * cr->nph; p++) {
                    cp = cr->access_precinct(p);
                    if (!is_packet_created[l][r][c][p]) {
                      this->packet[packet_count++]  = j2c_packet(l, r, c, p, cp, cr->num_bands);
                      is_packet_created[l][r][c][p] = true;
                    }
                  }
                }
              }
            }
          }
        }
        break;
      case 1:  // RLCP
        for (r = RS; r < RE; r++) {
          for (l = 0; l < LYE; l++) {
            for (c = CS; c < CE; c++) {
              c_NL = this->tcomp[c].NL;
              if (r <= c_NL) {
                cr = this->tcomp[c].access_resolution(r);
                if (!cr->is_empty) {
                  for (p = 0; p < cr->npw * cr->nph; p++) {
                    cp = cr->access_precinct(p);
                    if (!is_packet_created[l][r][c][p]) {
                      this->packet[packet_count++]  = j2c_packet(l, r, c, p, cp, cr->num_bands);
                      is_packet_created[l][r][c][p] = true;
                    }
                  }
                }
              }
            }
          }
        }
        break;
      case 2:  // RPCL
        this->find_gcd_of_precinct_size(PP);
        x_examin.push_back(pos0.x);
        for (uint32_t x = 0; x < this->pos1.x; x += (1U << PP.x)) {
          if (x > pos0.x) {
            x_examin.push_back(x);
          }
        }
        y_examin.push_back(pos0.y);
        for (uint32_t y = 0; y < this->pos1.y; y += (1U << PP.y)) {
          if (y > pos0.y) {
            y_examin.push_back(y);
          }
        }
        for (r = RS; r < RE; r++) {
          for (uint32_t y : y_examin) {
            for (uint32_t x : x_examin) {
              for (c = CS; c < CE; c++) {
                c_NL = this->tcomp[c].NL;
                if (r <= c_NL) {
                  cPP = this->tcomp[c].get_precinct_size(r);
                  cr  = this->tcomp[c].access_resolution(r);
                  if (!cr->is_empty) {
                    element_siz tr0 = cr->get_pos0();
                    x_cond          = false;
                    y_cond          = false;
                    main_header.SIZ->get_subsampling_factor(csub, c);
                    {
                      const DFS_marker *cdfs = this->tcomp[c].dfs_info;
                      const uint8_t hd = cdfs ? cdfs->hor_depth[c_NL - r] : static_cast<uint8_t>(c_NL - r);
                      const uint8_t vd = cdfs ? cdfs->ver_depth[c_NL - r] : static_cast<uint8_t>(c_NL - r);
                      x_cond = (x % (csub.x * (1U << (cPP.x + hd))) == 0)
                               || ((x == pos0.x)
                                   && ((tr0.x * (1U << hd)) % (1U << (cPP.x + hd)) != 0));
                      y_cond = (y % (csub.y * (1U << (cPP.y + vd))) == 0)
                               || ((y == pos0.y)
                                   && ((tr0.y * (1U << vd)) % (1U << (cPP.y + vd)) != 0));
                    }
                    if (x_cond && y_cond) {
                      p  = p_x[c][r] + p_y[c][r] * cr->npw;
                      cp = cr->access_precinct(p);
                      for (l = 0; l < LYE; l++) {
                        if (!is_packet_created[l][r][c][p]) {
                          this->packet[packet_count++]  = j2c_packet(l, r, c, p, cp, cr->num_bands);
                          is_packet_created[l][r][c][p] = true;
                        }
                      }
                      p_x[c][r] += 1;
                      if (p_x[c][r] == cr->npw) {
                        p_x[c][r] = 0;
                        p_y[c][r] += 1;
                      }
                    }
                  }
                }
              }
            }
          }
        }
        break;
      case 3:  // PCRL
        this->find_gcd_of_precinct_size(PP);
        x_examin.push_back(pos0.x);
        for (uint32_t x = 0; x < this->pos1.x; x += (1U << PP.x)) {
          if (x > pos0.x) {
            x_examin.push_back(x);
          }
        }
        y_examin.push_back(pos0.y);
        for (uint32_t y = 0; y < this->pos1.y; y += (1U << PP.y)) {
          if (y > pos0.y) {
            y_examin.push_back(y);
          }
        }
        for (uint32_t y : y_examin) {
          for (uint32_t x : x_examin) {
            for (c = CS; c < CE; c++) {
              c_NL     = this->tcomp[c].NL;
              local_RE = ((c_NL + 1) < RE) ? static_cast<uint8_t>(c_NL + 1U) : RE;
              for (r = RS; r < local_RE; r++) {
                cPP = this->tcomp[c].get_precinct_size(r);
                cr  = this->tcomp[c].access_resolution(r);
                if (!cr->is_empty) {
                  element_siz tr0 = cr->get_pos0();
                  x_cond          = false;
                  y_cond          = false;
                  main_header.SIZ->get_subsampling_factor(csub, c);
                  {
                    const DFS_marker *cdfs = this->tcomp[c].dfs_info;
                    const uint8_t hd = cdfs ? cdfs->hor_depth[c_NL - r] : static_cast<uint8_t>(c_NL - r);
                    const uint8_t vd = cdfs ? cdfs->ver_depth[c_NL - r] : static_cast<uint8_t>(c_NL - r);
                    x_cond = (x % (csub.x * (1U << (cPP.x + hd))) == 0)
                             || ((x == pos0.x)
                                 && ((tr0.x * (1U << hd)) % (1U << (cPP.x + hd)) != 0));
                    y_cond = (y % (csub.y * (1U << (cPP.y + vd))) == 0)
                             || ((y == pos0.y)
                                 && ((tr0.y * (1U << vd)) % (1U << (cPP.y + vd)) != 0));
                  }
                  if (x_cond && y_cond) {
                    p  = p_x[c][r] + p_y[c][r] * cr->npw;
                    cp = cr->access_precinct(p);
                    for (l = 0; l < LYE; l++) {
                      if (!is_packet_created[l][r][c][p]) {
                        this->packet[packet_count++]  = j2c_packet(l, r, c, p, cp, cr->num_bands);
                        is_packet_created[l][r][c][p] = true;
                      }
                    }
                    p_x[c][r] += 1;
                    if (p_x[c][r] == cr->npw) {
                      p_x[c][r] = 0;
                      p_y[c][r] += 1;
                    }
                  }
                }
              }
            }
          }
        }
        break;
      case 4:  // CPRL
        this->find_gcd_of_precinct_size(PP);
        x_examin.push_back(pos0.x);
        for (uint32_t x = 0; x < this->pos1.x; x += (1U << PP.x)) {
          if (x > pos0.x) {
            x_examin.push_back(x);
          }
        }
        y_examin.push_back(pos0.y);
        for (uint32_t y = 0; y < this->pos1.y; y += (1U << PP.y)) {
          if (y > pos0.y) {
            y_examin.push_back(y);
          }
        }
        for (c = CS; c < CE; c++) {
          c_NL     = this->tcomp[c].NL;
          local_RE = ((c_NL + 1) < RE) ? static_cast<uint8_t>(c_NL + 1U) : RE;
          for (uint32_t y : y_examin) {
            for (uint32_t x : x_examin) {
              for (r = RS; r < local_RE; r++) {
                cPP = this->tcomp[c].get_precinct_size(r);
                cr  = this->tcomp[c].access_resolution(r);
                if (!cr->is_empty) {
                  element_siz tr0 = cr->get_pos0();
                  x_cond          = false;
                  y_cond          = false;
                  main_header.SIZ->get_subsampling_factor(csub, c);
                  {
                    const DFS_marker *cdfs = this->tcomp[c].dfs_info;
                    const uint8_t hd = cdfs ? cdfs->hor_depth[c_NL - r] : static_cast<uint8_t>(c_NL - r);
                    const uint8_t vd = cdfs ? cdfs->ver_depth[c_NL - r] : static_cast<uint8_t>(c_NL - r);
                    x_cond = (x % (csub.x * (1U << (cPP.x + hd))) == 0)
                             || ((x == pos0.x)
                                 && ((tr0.x * (1U << hd)) % (1U << (cPP.x + hd)) != 0));
                    y_cond = (y % (csub.y * (1U << (cPP.y + vd))) == 0)
                             || ((y == pos0.y)
                                 && ((tr0.y * (1U << vd)) % (1U << (cPP.y + vd)) != 0));
                  }
                  if (x_cond && y_cond) {
                    p  = p_x[c][r] + p_y[c][r] * cr->npw;
                    cp = cr->access_precinct(p);
                    for (l = 0; l < LYE; l++) {
                      if (!is_packet_created[l][r][c][p]) {
                        this->packet[packet_count++]  = j2c_packet(l, r, c, p, cp, cr->num_bands);
                        is_packet_created[l][r][c][p] = true;
                      }
                    }
                    p_x[c][r] += 1;
                    if (p_x[c][r] == cr->npw) {
                      p_x[c][r] = 0;
                      p_y[c][r] += 1;
                    }
                  }
                }
              }
            }
          }
        }
        break;
      default:
        printf(
            "ERROR: Progression order number shall be in the range from 0 "
            "to 4\n");
        throw std::exception();
    }
  }
}

void j2k_tile::write_packets(j2c_dst_memory &outbuf) {
  for (size_t i = 0; i < this->num_tile_part; ++i) {
    j2k_tile_part *tp = this->tile_part[i].get();
    // set tile-part length
    this->tile_part[0]->header->SOT.set_tile_part_length(
        this->length + static_cast<uint32_t>(6 * this->num_packets * this->is_use_SOP()));
    tp->header->SOT.write(outbuf);
    // write packets
    for (size_t n = 0; n < static_cast<size_t>(this->num_packets); ++n) {
      if (this->is_use_SOP()) {
        outbuf.put_word(_SOP);
        outbuf.put_word(0x0004);
        outbuf.put_word(static_cast<uint16_t>(n % 65536));
      }
      outbuf.put_N_bytes(this->packet[n].buf.get(), static_cast<uint32_t>(this->packet[n].length));
    }
  }
}

void j2k_tile::decode() {
#ifdef OPENHTJ2K_THREAD
  auto pool = ThreadPool::get();
  // std::vector<std::future<int>> results;
#endif
  OPENHTJ2K_TIME_REGION_BEGIN(BlockDecode)
  for (uint16_t c = 0; c < num_components; c++) {
    const uint8_t ROIshift = this->tcomp[c].get_ROIshift();
    const uint8_t NL       = this->tcomp[c].get_dwt_levels();

    // Pre-scan: compute the max precinct codeblock count per DWT level.
    // Stored per level so the decode pool can be grown incrementally (coarsest→finest),
    // keeping the working set small for coarse levels and improving cache behaviour.
    const int num_dec_levels = static_cast<int>(NL) - static_cast<int>(this->reduce_NL) + 1;
    std::vector<uint32_t> level_max_cblks(static_cast<size_t>(num_dec_levels), 0);
    for (int8_t lev = (int8_t)NL; lev >= this->reduce_NL; --lev) {
      j2k_resolution *cr           = this->tcomp[c].access_resolution(static_cast<uint8_t>(NL - lev));
      const uint32_t num_precincts = cr->npw * cr->nph;
      for (uint32_t p = 0; p < num_precincts; p++) {
        j2k_precinct *cp     = cr->access_precinct(p);
        uint32_t total_cblks = 0;
        for (uint8_t b = 0; b < cr->num_bands; b++) {
          j2k_precinct_subband *cpb = cp->access_pband(b);
          total_cblks += cpb->num_codeblock_x * cpb->num_codeblock_y;
        }
        const size_t idx                = static_cast<size_t>(NL - lev);
        level_max_cblks[idx] = std::max(level_max_cblks[idx], total_cblks);
      }
    }

    // Grow-only pool: start empty, expand only when this level needs more than current capacity.
    // Frees and re-allocates only on growth (never shrinks), so warm pages are preserved
    // for precincts within the same level. Coarse levels start with a tiny allocation
    // (better L2/L3 cache fit); the pool grows to full size only at the finest level.
    size_t alloc_samples_bytes = 0;
    size_t alloc_states_bytes  = 0;
    int32_t *buf_for_samples   = nullptr;
    uint8_t *buf_for_states    = nullptr;
#ifdef OPENHTJ2K_THREAD
    // Hoist dec_task_args outside the level loop: clear()+reserve() per iteration
    // avoids one heap allocation per level (15 allocs saved for a 5-level 3-component tile).
    std::vector<DecTaskArgs> dec_task_args;
    std::atomic<int> dec_remaining{0};
#endif

    for (int8_t lev = (int8_t)NL; lev >= this->reduce_NL; --lev) {
      const uint32_t lev_max = level_max_cblks[static_cast<size_t>(NL - lev)];
      if (lev_max == 0) continue;

      const size_t need_samples = sizeof(int32_t) * lev_max * 4096;
      const size_t need_states  = sizeof(uint8_t) * lev_max * 6156;
      if (need_samples > alloc_samples_bytes) {
        aligned_mem_free(buf_for_samples);
        buf_for_samples     = static_cast<int32_t *>(aligned_mem_alloc(need_samples, 32));
        alloc_samples_bytes = need_samples;
      }
      if (need_states > alloc_states_bytes) {
        aligned_mem_free(buf_for_states);
        buf_for_states     = static_cast<uint8_t *>(aligned_mem_alloc(need_states, 32));
        alloc_states_bytes = need_states;
      }

      j2k_resolution *cr           = this->tcomp[c].access_resolution(static_cast<uint8_t>(NL - lev));
      const uint32_t num_precincts = cr->npw * cr->nph;
#ifdef OPENHTJ2K_THREAD
      dec_task_args.clear();
      dec_task_args.reserve(lev_max);
      dec_remaining.store(0, std::memory_order_relaxed);
#endif
      for (uint32_t p = 0; p < num_precincts; p++) {
        j2k_precinct *cp = cr->access_precinct(p);

        int32_t *pbuf  = buf_for_samples;
        uint8_t *spbuf = buf_for_states;

        // Pass 1: assign sample/state buffer pointers to all codeblocks in this precinct.
        for (uint8_t b = 0; b < cr->num_bands; b++) {
          j2k_precinct_subband *cpb = cp->access_pband(b);
          const uint32_t num_cblks  = cpb->num_codeblock_x * cpb->num_codeblock_y;
          for (uint32_t block_index = 0; block_index < num_cblks; ++block_index) {
            j2k_codeblock *block = cpb->access_codeblock(block_index);
            const uint32_t QWx2  = round_up(block->size.x, 8U);
            const uint32_t QHx2  = round_up(block->size.y, 8U);
            block->sample_buf    = pbuf;
            pbuf += QWx2 * QHx2;
            block->block_states = spbuf;
            spbuf += (QWx2 + 2) * (QHx2 + 2);
          }
        }

        // Single bulk zero of the entire used pool region for this precinct.
        // Replaces N per-codeblock memsets with 2 sequential writes for better cache streaming.
        memset(buf_for_samples, 0, static_cast<size_t>(pbuf - buf_for_samples) * sizeof(int32_t));
        memset(buf_for_states, 0, static_cast<size_t>(spbuf - buf_for_states));

        // Pass 2: decode all non-empty codeblocks (buffers are already zeroed above).
#ifdef OPENHTJ2K_THREAD
        dec_task_args.clear();
        dec_remaining.store(0, std::memory_order_relaxed);
#endif
        for (uint8_t b = 0; b < cr->num_bands; b++) {
          j2k_precinct_subband *cpb = cp->access_pband(b);
          const uint32_t num_cblks  = cpb->num_codeblock_x * cpb->num_codeblock_y;
          for (uint32_t block_index = 0; block_index < num_cblks; ++block_index) {
            j2k_codeblock *block = cpb->access_codeblock(block_index);
            // only decode a codeblock having non-zero coding passes AND
            // attached compressed data.  The JPIP precinct filter installs a
            // state where num_passes > 0 (packet header was parsed) but
            // compressed_data == nullptr (body was dropped); treat that as
            // "no passes to decode" so the codeblock's samples remain zero.
            if (block->num_passes && block->get_compressed_data() != nullptr) {
#ifdef OPENHTJ2K_THREAD
              if (pool && pool->num_threads() > 1) {
                dec_task_args.push_back({block, ROIshift, &dec_remaining});
                auto *da = &dec_task_args.back();
                dec_remaining.fetch_add(1, std::memory_order_relaxed);
                pool->push([da]() {
                  if ((da->block->Cmodes & HT) >> 6)
                    htj2k_decode(da->block, da->ROIshift);
                  else
                    j2k_decode(da->block, da->ROIshift);
                  da->remaining->fetch_sub(1, std::memory_order_release);
                });
              } else {
                if ((block->Cmodes & HT) >> 6)
                  htj2k_decode(block, ROIshift);
                else
                  j2k_decode(block, ROIshift);
              }
#else
              if ((block->Cmodes & HT) >> 6)
                htj2k_decode(block, ROIshift);
              else
                j2k_decode(block, ROIshift);
#endif
            }
          }  // end of codeblock loop
        }  // end of subband loop
#ifdef OPENHTJ2K_THREAD
        while (dec_remaining.load(std::memory_order_acquire) > 0)
          std::this_thread::yield();
#endif
      }  // end of precinct loop
    }
    aligned_mem_free(buf_for_states);
    aligned_mem_free(buf_for_samples);
  }
  OPENHTJ2K_TIME_REGION_END

  OPENHTJ2K_TIME_REGION_BEGIN(IDWT)
  for (uint16_t c = 0; c < num_components; c++) {
    // const uint8_t ROIshift       = this->tcomp[c].get_ROIshift();
    const uint8_t NL             = this->tcomp[c].get_dwt_levels();
    const uint8_t transformation = this->tcomp[c].get_transformation();

    // Allocate pse_scratch once for all idwt_2d_sr_fixed calls in this component.
    // Sized for the finest (widest) resolution level; coarser levels use a sub-region.
    const int32_t max_idwt_pse_len =
        (NL > this->reduce_NL)
            ? round_up(
                  static_cast<int32_t>(
                      this->tcomp[c].access_resolution(NL - this->reduce_NL)->get_pos1().x
                      - this->tcomp[c].access_resolution(NL - this->reduce_NL)->get_pos0().x),
                  32)
            : 0;
    sprec_t *idwt_pse_scratch =
        (max_idwt_pse_len > 0)
            ? static_cast<sprec_t *>(
                  aligned_mem_alloc(sizeof(sprec_t) * 8 * static_cast<size_t>(max_idwt_pse_len), 32))
            : nullptr;
    // Allocate buf_scratch once: pointer array for the row-pointer table used in vertical DWT.
    // Sized for the finest resolution height + 8 (covers max PSE extension top+bottom ≤ 8).
    const uint32_t max_idwt_height =
        (NL > this->reduce_NL)
            ? static_cast<uint32_t>(
                  this->tcomp[c].access_resolution(NL - this->reduce_NL)->get_pos1().y
                  - this->tcomp[c].access_resolution(NL - this->reduce_NL)->get_pos0().y)
            : 0u;
    sprec_t **idwt_buf_scratch = (max_idwt_height > 0)
                                     ? new sprec_t *[static_cast<size_t>(max_idwt_height + 8u)]
                                     : nullptr;

    for (int8_t lev = (int8_t)NL; lev >= this->reduce_NL; --lev) {
      j2k_resolution *cr = this->tcomp[c].access_resolution(static_cast<uint8_t>(NL - lev));
      // lowest resolution level (= LL0) does not have HL, LH, HH bands.
      if (lev != NL) {
        j2k_resolution *pcr        = this->tcomp[c].access_resolution(static_cast<uint8_t>(NL - lev - 1));
        const element_siz top_left = cr->get_pos0();
        const element_siz bottom_right = cr->get_pos1();
        const int32_t u0               = static_cast<int32_t>(top_left.x);
        const int32_t u1               = static_cast<int32_t>(bottom_right.x);
        const int32_t v0               = static_cast<int32_t>(top_left.y);
        const int32_t v1               = static_cast<int32_t>(bottom_right.y);

        j2k_subband *HL = cr->access_subband(0);

        if (u1 != u0 && v1 != v0) {
          switch (cr->transform_direction) {
            case DWT_BIDIR:
            default: {
              j2k_subband *LH = cr->access_subband(1);
              j2k_subband *HH = cr->access_subband(2);
              idwt_2d_sr_fixed(cr->i_samples, pcr->i_samples, HL->i_samples, LH->i_samples, HH->i_samples,
                               u0, u1, v0, v1, transformation, idwt_pse_scratch, idwt_buf_scratch);
              break;
            }
            case DWT_HORZ:
              idwt_horz_only_sr_fixed(cr->i_samples, pcr->i_samples, HL->i_samples, u0, u1, v0, v1,
                                      transformation);
              break;
            case DWT_VERT:
              idwt_vert_only_sr_fixed(cr->i_samples, pcr->i_samples, HL->i_samples, u0, u1, v0, v1,
                                      transformation, idwt_pse_scratch, idwt_buf_scratch);
              break;
            case DWT_NO:
              memcpy(cr->i_samples, pcr->i_samples,
                     sizeof(sprec_t) * static_cast<size_t>(cr->stride) * static_cast<size_t>(v1 - v0));
              break;
          }
        }
      }
    }  // end of resolution loop
    aligned_mem_free(idwt_pse_scratch);
    delete[] idwt_buf_scratch;
    j2k_resolution *cr = this->tcomp[c].access_resolution(static_cast<uint8_t>(NL - reduce_NL));

    // modify coordinates of tile component considering a value defined via "-reduce" parameter
    this->tcomp[c].set_pos0(cr->get_pos0());
    this->tcomp[c].set_pos1(cr->get_pos1());

  }  // end of component loop
  OPENHTJ2K_TIME_REGION_END
}
void j2k_tile::read_packet(j2k_precinct *current_precint, uint16_t layer, uint8_t num_band,
                           bool skip_body) {
  OPENHTJ2K_MAYBE_UNUSED uint16_t Nsop = 0;
  uint16_t Lsop;
  if (use_SOP) {
    uint16_t word = this->tile_buf->get_word();
    if (word != _SOP) {
      printf("ERROR: Expected SOP marker but %04X is found\n", word);
      throw std::exception();
    }
    Lsop = this->tile_buf->get_word();
    if (Lsop != 4) {
      printf("ERROR: illegal Lsop value %d is found\n", Lsop);
      throw std::exception();
    }
    Nsop = this->tile_buf->get_word();
  }

  uint8_t bit = this->packet_header->get_bit();
  if (bit == 0) {                       // if 0, empty packet
    this->packet_header->flush_bits();  // flushing remaining bits of packet header
    if (use_EPH) {
      uint16_t word = this->packet_header->get_word();
      if (word != _EPH) {
        printf("ERROR: Expected EPH marker but %04X is found\n", word);
        throw std::exception();
      }
    }
    return;
  }
  j2k_precinct_subband *cpb;
  // uint32_t num_bytes;
  for (uint8_t b = 0; b < num_band; b++) {
    cpb = current_precint->access_pband(b);  //&current_precint->pband[b];
    cpb->parse_packet_header(this->packet_header, layer, this->Ccap15);
  }
  // if the last byte of a packet header is 0xFF, one bit shall be read.
  this->packet_header->check_last_FF();
  this->packet_header->flush_bits();
  // check EPH
  if (use_EPH) {
    uint16_t word = this->packet_header->get_word();
    if (word != _EPH) {
      printf("ERROR: Expected EPH marker but %04X is found\n", word);
      throw std::exception();
    }
  }

  j2k_codeblock *block;
  uint16_t buf_limit = 8192;
  for (uint8_t b = 0; b < num_band; b++) {
    cpb                      = current_precint->access_pband(b);  //&current_precint->pband[b];
    const uint32_t num_cblks = cpb->num_codeblock_x * cpb->num_codeblock_y;
    if (num_cblks != 0) {
      for (uint32_t block_index = 0; block_index < num_cblks; block_index++) {
        block = cpb->access_codeblock(block_index);
        if (skip_body) {
          block->skip_compressed_buffer(this->tile_buf.get(), layer);
        } else {
          block->create_compressed_buffer(this->tile_buf.get(), buf_limit, layer);
        }
      }
    }
  }
}

void j2k_tile::find_gcd_of_precinct_size(element_siz &out) {
  element_siz PP;
  uint8_t PPx = 16, PPy = 16;
  for (uint16_t c = 0; c < num_components; c++) {
    for (uint8_t r = 0; r <= this->tcomp[c].get_dwt_levels(); r++) {
      PP  = this->tcomp[c].get_precinct_size(r);
      PPx = (PPx > PP.x) ? static_cast<uint8_t>(PP.x) : PPx;
      PPy = (PPy > PP.y) ? static_cast<uint8_t>(PP.y) : PPy;
    }
  }
  out.x = PPx;
  out.y = PPy;
}

void j2k_tile::ycbcr_to_rgb() {
  if (num_components < 3 || !MCT) {
    return;
  }
  uint8_t transformation;
  transformation = this->tcomp[0].get_transformation();
  assert(transformation == this->tcomp[1].get_transformation());
  assert(transformation == this->tcomp[2].get_transformation());

  element_siz tc0       = this->tcomp[0].get_pos0();
  element_siz tc1       = this->tcomp[0].get_pos1();
  const uint32_t width  = tc1.x - tc0.x;
  const uint32_t height = tc1.y - tc0.y;
  const uint32_t stride = round_up(width, 32U);

  // Access the float resolution buffers directly (avoiding the int32 intermediate copy)
  const uint8_t NL0     = tcomp[0].get_dwt_levels();
  const uint8_t NL1     = tcomp[1].get_dwt_levels();
  const uint8_t NL2     = tcomp[2].get_dwt_levels();
  sprec_t *sp0 = tcomp[0].access_resolution(static_cast<uint8_t>(NL0 - reduce_NL))->i_samples;
  sprec_t *sp1 = tcomp[1].access_resolution(static_cast<uint8_t>(NL1 - reduce_NL))->i_samples;
  sprec_t *sp2 = tcomp[2].access_resolution(static_cast<uint8_t>(NL2 - reduce_NL))->i_samples;

  // ATK (transformation>=2) is irreversible → use ICT (index 0), same as irrev97.
  // Dispatch table has only 2 entries (0=irrev, 1=rev); never index with raw transformation.
  const uint8_t ct_idx = (transformation == 1) ? 1 : 0;
  cvt_ycbcr_to_rgb_float[ct_idx](sp0, sp1, sp2, width, height, stride);
}

void j2k_tile::finalize(j2k_main_header &hdr, uint8_t reduce_NL, std::vector<int32_t *> &dst) {
  for (uint16_t c = 0; c < this->num_components; ++c) {
    const int32_t DC_OFFSET = (hdr.SIZ->is_signed(c)) ? 0 : 1 << (tcomp[c].bitdepth - 1);
    const int32_t MAXVAL =
        (hdr.SIZ->is_signed(c)) ? (1 << (tcomp[c].bitdepth - 1)) - 1 : (1 << tcomp[c].bitdepth) - 1;
    const int32_t MINVAL = (hdr.SIZ->is_signed(c)) ? -(1 << (tcomp[c].bitdepth - 1)) : 0;

    element_siz siz, Osiz, Rsiz, csize;
    hdr.SIZ->get_image_size(siz);
    hdr.SIZ->get_image_origin(Osiz);
    hdr.SIZ->get_subsampling_factor(Rsiz, c);
    const uint32_t x0     = ceil_int(Osiz.x, Rsiz.x);
    const uint32_t y0     = ceil_int(Osiz.y, Rsiz.y);
    const uint32_t x1     = ceil_int(siz.x, Rsiz.x);
    const element_siz tc0 = tcomp[c].get_pos0();
    tcomp[c].get_size(csize);
    const uint32_t in_stride = round_up(csize.x, 32U);
    const uint32_t in_height = csize.y;
    const uint32_t x_offset  = tc0.x - ceil_int(x0, (1U << reduce_NL));
    const uint32_t y_offset  = tc0.y - ceil_int(y0, (1U << reduce_NL));

    const uint32_t out_stride = ceil_int(x1 - x0, (1U << reduce_NL));

    // downshift value for lossy path
    // Reversible (transformation==1) uses downshift=0 (lossless, no scaling).
    // All irreversible paths (irrev97=0, ATK irrev>=2) use downshift = FRACBITS - bitdepth.
    int16_t downshift = (tcomp[c].transformation == 1) ? 0 : static_cast<int16_t>(FRACBITS - tcomp[c].bitdepth);
    // For bitdepth > FRACBITS (e.g., 16-bit): downshift < 0, meaning the internal representation
    // was right-shifted during encoding and must be left-shifted back during reconstruction.
    // No rounding offset is applied for a left shift (it would introduce a constant bias).
    // For bitdepth <= FRACBITS: downshift >= 0 (right shift), use (1<<downshift)>>1 for rounding.
    int16_t offset = (downshift <= 0) ? 0 : static_cast<int16_t>((1 << downshift) >> 1);
    // Read directly from the float resolution buffer (avoiding the int32 intermediate copy)
    const uint8_t NL_c    = tcomp[c].get_dwt_levels();
    j2k_resolution *cr    = tcomp[c].access_resolution(static_cast<uint8_t>(NL_c - reduce_NL));
    sprec_t *const src    = cr->i_samples;
    int32_t *const cdst   = dst[c];
    const sprec_t *spf;
    int32_t *dp;
#if defined(OPENHTJ2K_ENABLE_WASM_SIMD)
    {
      const v128_t vdco = wasm_i32x4_splat(DC_OFFSET);
      const v128_t vmx  = wasm_i32x4_splat(MAXVAL);
      const v128_t vmn  = wasm_i32x4_splat(MINVAL);
      if (downshift == 0) {
        for (uint32_t y = 0; y < in_height; ++y) {
          uint32_t len = csize.x;
          spf          = src + y * in_stride;
          dp           = cdst + x_offset + (y + y_offset) * out_stride;
          for (; len >= 8; len -= 8) {
            v128_t v0 = wasm_i32x4_trunc_sat_f32x4(wasm_v128_load(spf));
            v128_t v1 = wasm_i32x4_trunc_sat_f32x4(wasm_v128_load(spf + 4));
            v0 = wasm_i32x4_max(wasm_i32x4_min(wasm_i32x4_add(v0, vdco), vmx), vmn);
            v1 = wasm_i32x4_max(wasm_i32x4_min(wasm_i32x4_add(v1, vdco), vmx), vmn);
            wasm_v128_store(dp, v0);
            wasm_v128_store(dp + 4, v1);
            spf += 8;
            dp += 8;
          }
          for (; len > 0; --len) {
            int32_t ival = static_cast<int32_t>(*spf++) + DC_OFFSET;
            ival  = (ival > MAXVAL) ? MAXVAL : ival;
            ival  = (ival < MINVAL) ? MINVAL : ival;
            *dp++ = ival;
          }
        }
      } else if (downshift < 0) {
        const v128_t vo = wasm_i32x4_splat(offset);
        for (uint32_t y = 0; y < in_height; ++y) {
          uint32_t len = csize.x;
          spf          = src + y * in_stride;
          dp           = cdst + x_offset + (y + y_offset) * out_stride;
          for (; len >= 8; len -= 8) {
            v128_t v0 = wasm_i32x4_shl(wasm_i32x4_add(wasm_i32x4_trunc_sat_f32x4(wasm_v128_load(spf)), vo), (int)-downshift);
            v128_t v1 = wasm_i32x4_shl(wasm_i32x4_add(wasm_i32x4_trunc_sat_f32x4(wasm_v128_load(spf + 4)), vo), (int)-downshift);
            v0 = wasm_i32x4_max(wasm_i32x4_min(wasm_i32x4_add(v0, vdco), vmx), vmn);
            v1 = wasm_i32x4_max(wasm_i32x4_min(wasm_i32x4_add(v1, vdco), vmx), vmn);
            wasm_v128_store(dp, v0);
            wasm_v128_store(dp + 4, v1);
            spf += 8;
            dp += 8;
          }
          for (; len > 0; --len) {
            int32_t ival = (static_cast<int32_t>(*spf++) + offset) << -downshift;
            ival += DC_OFFSET;
            ival  = (ival > MAXVAL) ? MAXVAL : ival;
            ival  = (ival < MINVAL) ? MINVAL : ival;
            *dp++ = ival;
          }
        }
      } else {
        const v128_t vo = wasm_i32x4_splat(offset);
        for (uint32_t y = 0; y < in_height; ++y) {
          uint32_t len = csize.x;
          spf          = src + y * in_stride;
          dp           = cdst + x_offset + (y + y_offset) * out_stride;
          for (; len >= 8; len -= 8) {
            v128_t v0 = wasm_i32x4_shr(wasm_i32x4_add(wasm_i32x4_trunc_sat_f32x4(wasm_v128_load(spf)), vo), (int)downshift);
            v128_t v1 = wasm_i32x4_shr(wasm_i32x4_add(wasm_i32x4_trunc_sat_f32x4(wasm_v128_load(spf + 4)), vo), (int)downshift);
            v0 = wasm_i32x4_max(wasm_i32x4_min(wasm_i32x4_add(v0, vdco), vmx), vmn);
            v1 = wasm_i32x4_max(wasm_i32x4_min(wasm_i32x4_add(v1, vdco), vmx), vmn);
            wasm_v128_store(dp, v0);
            wasm_v128_store(dp + 4, v1);
            spf += 8;
            dp += 8;
          }
          for (; len > 0; --len) {
            int32_t ival = (static_cast<int32_t>(*spf++) + offset) >> downshift;
            ival += DC_OFFSET;
            ival  = (ival > MAXVAL) ? MAXVAL : ival;
            ival  = (ival < MINVAL) ? MINVAL : ival;
            *dp++ = ival;
          }
        }
      }
    }
#elif defined(OPENHTJ2K_ENABLE_ARM_NEON)
    {
      const int32x4_t dco  = vdupq_n_s32(DC_OFFSET);
      const int32x4_t vmax = vdupq_n_s32(MAXVAL);
      const int32x4_t vmin = vdupq_n_s32(MINVAL);
      if (downshift == 0) {
        // Lossless: no rounding offset (offset=0), no shift — just convert and clamp.
        for (uint32_t y = 0; y < in_height; ++y) {
          uint32_t len = csize.x;
          spf          = src + y * in_stride;
          dp           = cdst + x_offset + (y + y_offset) * out_stride;
          for (; len >= 8; len -= 8) {
            int32x4_t v0 = vcvtq_s32_f32(vld1q_f32(spf));
            int32x4_t v1 = vcvtq_s32_f32(vld1q_f32(spf + 4));
            v0 = vmaxq_s32(vminq_s32(vaddq_s32(v0, dco), vmax), vmin);
            v1 = vmaxq_s32(vminq_s32(vaddq_s32(v1, dco), vmax), vmin);
            vst1q_s32(dp, v0);
            vst1q_s32(dp + 4, v1);
            spf += 8;
            dp += 8;
          }
          for (; len > 0; --len) {
            int32_t ival = static_cast<int32_t>(*spf++) + DC_OFFSET;
            ival  = (ival > MAXVAL) ? MAXVAL : ival;
            ival  = (ival < MINVAL) ? MINVAL : ival;
            *dp++ = ival;
          }
        }
      } else if (downshift < 0) {
        const int32x4_t o      = vdupq_n_s32(offset);
        const int32x4_t vshift = vdupq_n_s32(-downshift);  // positive → left shift
        for (uint32_t y = 0; y < in_height; ++y) {
          uint32_t len = csize.x;
          spf          = src + y * in_stride;
          dp           = cdst + x_offset + (y + y_offset) * out_stride;
          for (; len >= 8; len -= 8) {
            int32x4_t v0 = vshlq_s32(vaddq_s32(vcvtq_s32_f32(vld1q_f32(spf)), o), vshift);
            int32x4_t v1 = vshlq_s32(vaddq_s32(vcvtq_s32_f32(vld1q_f32(spf + 4)), o), vshift);
            v0 = vmaxq_s32(vminq_s32(vaddq_s32(v0, dco), vmax), vmin);
            v1 = vmaxq_s32(vminq_s32(vaddq_s32(v1, dco), vmax), vmin);
            vst1q_s32(dp, v0);
            vst1q_s32(dp + 4, v1);
            spf += 8;
            dp += 8;
          }
          for (; len > 0; --len) {
            int32_t ival = (static_cast<int32_t>(*spf++) + offset) << -downshift;
            ival += DC_OFFSET;
            ival  = (ival > MAXVAL) ? MAXVAL : ival;
            ival  = (ival < MINVAL) ? MINVAL : ival;
            *dp++ = ival;
          }
        }
      } else {
        const int32x4_t o      = vdupq_n_s32(offset);
        const int32x4_t vshift = vdupq_n_s32(-downshift);  // negative → right shift
        for (uint32_t y = 0; y < in_height; ++y) {
          uint32_t len = csize.x;
          spf          = src + y * in_stride;
          dp           = cdst + x_offset + (y + y_offset) * out_stride;
          for (; len >= 8; len -= 8) {
            int32x4_t v0 = vshlq_s32(vaddq_s32(vcvtq_s32_f32(vld1q_f32(spf)), o), vshift);
            int32x4_t v1 = vshlq_s32(vaddq_s32(vcvtq_s32_f32(vld1q_f32(spf + 4)), o), vshift);
            v0 = vmaxq_s32(vminq_s32(vaddq_s32(v0, dco), vmax), vmin);
            v1 = vmaxq_s32(vminq_s32(vaddq_s32(v1, dco), vmax), vmin);
            vst1q_s32(dp, v0);
            vst1q_s32(dp + 4, v1);
            spf += 8;
            dp += 8;
          }
          for (; len > 0; --len) {
            int32_t ival = (static_cast<int32_t>(*spf++) + offset) >> downshift;
            ival += DC_OFFSET;
            ival  = (ival > MAXVAL) ? MAXVAL : ival;
            ival  = (ival < MINVAL) ? MINVAL : ival;
            *dp++ = ival;
          }
        }
      }
    }
#elif defined(OPENHTJ2K_TRY_AVX2) && defined(__AVX2__)
    if (downshift < 0) {
      __m256i v, o, dco, vmax, vmin;
      o    = _mm256_set1_epi32(offset);
      dco  = _mm256_set1_epi32(DC_OFFSET);
      vmax = _mm256_set1_epi32(MAXVAL);
      vmin = _mm256_set1_epi32(MINVAL);
      for (uint32_t y = 0; y < in_height; ++y) {
        uint32_t len = csize.x;
        spf          = src + y * in_stride;
        dp           = cdst + x_offset + (y + y_offset) * out_stride;
        for (; len >= 8; len -= 8) {
          v = _mm256_cvttps_epi32(_mm256_load_ps(spf));
          v = _mm256_slli_epi32(_mm256_add_epi32(v, o), -downshift);
          v = _mm256_add_epi32(v, dco);
          v = _mm256_min_epi32(v, vmax);
          v = _mm256_max_epi32(v, vmin);
          _mm256_storeu_si256((__m256i *)dp, v);
          spf += 8;
          dp += 8;
        }
        for (; len > 0; --len) {
          int32_t ival = static_cast<int32_t>(*spf++);
          ival         = (ival + offset) << -downshift;
          ival += DC_OFFSET;
          ival   = (ival > MAXVAL) ? MAXVAL : ival;
          ival   = (ival < MINVAL) ? MINVAL : ival;
          *dp++  = ival;
        }
      }
    } else {
      __m256i v, o, dco, vmax, vmin;
      o    = _mm256_set1_epi32(offset);
      dco  = _mm256_set1_epi32(DC_OFFSET);
      vmax = _mm256_set1_epi32(MAXVAL);
      vmin = _mm256_set1_epi32(MINVAL);
      for (uint32_t y = 0; y < in_height; ++y) {
        uint32_t len = csize.x;
        spf          = src + y * in_stride;
        dp           = cdst + x_offset + (y + y_offset) * out_stride;
        for (; len >= 8; len -= 8) {
          v = _mm256_cvttps_epi32(_mm256_load_ps(spf));
          v = _mm256_srai_epi32(_mm256_add_epi32(v, o), downshift);
          v = _mm256_add_epi32(v, dco);
          v = _mm256_min_epi32(v, vmax);
          v = _mm256_max_epi32(v, vmin);
          _mm256_storeu_si256((__m256i *)dp, v);
          spf += 8;
          dp += 8;
        }
        for (; len > 0; --len) {
          int32_t ival = static_cast<int32_t>(*spf++);
          ival         = (ival + offset) >> downshift;
          ival += DC_OFFSET;
          ival   = (ival > MAXVAL) ? MAXVAL : ival;
          ival   = (ival < MINVAL) ? MINVAL : ival;
          *dp++  = ival;
        }
      }
    }
#else
      if (downshift < 0) {
        for (uint32_t y = 0; y < in_height; ++y) {
          uint32_t len = csize.x;
          spf          = src + y * in_stride;
          dp           = cdst + x_offset + (y + y_offset) * out_stride;
          for (uint32_t n = 0; n < len; ++n) {
            int32_t ival = static_cast<int32_t>(spf[n]);
            ival         = (ival + offset) << -downshift;
            ival += DC_OFFSET;
            ival   = (ival > MAXVAL) ? MAXVAL : ival;
            ival   = (ival < MINVAL) ? MINVAL : ival;
            dp[n]  = ival;
          }
        }
      } else {
        for (uint32_t y = 0; y < in_height; ++y) {
          uint32_t len = csize.x;
          spf          = src + y * in_stride;
          dp           = cdst + x_offset + (y + y_offset) * out_stride;
          for (uint32_t n = 0; n < len; ++n) {
            int32_t ival = static_cast<int32_t>(spf[n]);
            ival         = (ival + offset) >> downshift;
            ival += DC_OFFSET;
            ival   = (ival > MAXVAL) ? MAXVAL : ival;
            ival   = (ival < MINVAL) ? MINVAL : ival;
            dp[n]  = ival;
          }
        }
      }
#endif
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Line-based decode: lazy IDWT + per-row YCbCr→RGB + float→int32 conversion.
// Must be called after create_tile_buf() (packets already parsed).
// Does NOT call decode() / ycbcr_to_rgb() / finalize().
// ─────────────────────────────────────────────────────────────────────────────
void j2k_tile::decode_line_based(j2k_main_header &hdr, uint8_t reduce_NL_val,
                                  std::vector<int32_t *> &dst) {
  const uint16_t NC = num_components;

  // Pre-compute per-component output mapping parameters (same logic as finalize()).
  struct CInfo {
    int32_t DC_OFFSET, MAXVAL, MINVAL;
    int16_t downshift, rnd;
    uint32_t csize_x, csize_y;
    uint32_t x_offset, y_offset, out_stride;
    int32_t *cdst;
  };
  std::vector<CInfo> ci(NC);
  for (uint16_t c = 0; c < NC; ++c) {
    CInfo &I = ci[c];
    const uint8_t bd = tcomp[c].bitdepth;
    I.DC_OFFSET       = (hdr.SIZ->is_signed(c)) ? 0 : 1 << (bd - 1);
    I.MAXVAL          = (hdr.SIZ->is_signed(c)) ? (1 << (bd - 1)) - 1 : (1 << bd) - 1;
    I.MINVAL          = (hdr.SIZ->is_signed(c)) ? -(1 << (bd - 1)) : 0;

    element_siz siz, Osiz, Rsiz;
    hdr.SIZ->get_image_size(siz);
    hdr.SIZ->get_image_origin(Osiz);
    hdr.SIZ->get_subsampling_factor(Rsiz, c);
    const uint32_t x0 = ceil_int(Osiz.x, Rsiz.x);
    const uint32_t y0 = ceil_int(Osiz.y, Rsiz.y);
    const uint32_t x1 = ceil_int(siz.x, Rsiz.x);
    // Use the active (reduced) resolution geometry, matching what decode() does when
    // it updates tcomp[c].pos0/pos1 to the active resolution after IDWT.
    const uint8_t NL_c = tcomp[c].get_dwt_levels();
    j2k_resolution *cr_act =
        tcomp[c].access_resolution(static_cast<uint8_t>(NL_c - reduce_NL_val));
    const element_siz tc0 = cr_act->get_pos0();
    const element_siz tc1 = cr_act->get_pos1();
    I.csize_x   = tc1.x - tc0.x;
    I.csize_y   = tc1.y - tc0.y;
    I.x_offset  = tc0.x - ceil_int(x0, (1U << reduce_NL_val));
    I.y_offset  = tc0.y - ceil_int(y0, (1U << reduce_NL_val));
    I.out_stride = ceil_int(x1 - x0, (1U << reduce_NL_val));
    I.downshift = (tcomp[c].transformation == 1) ? 0
                                                 : static_cast<int16_t>(FRACBITS - bd);
    I.rnd  = (I.downshift <= 0) ? 0 : static_cast<int16_t>((1 << I.downshift) >> 1);
    I.cdst = dst[c];
  }

  // Per-component float row scratch buffers (each sized with SIMD headroom).
  // Only allocated when needed (non-MCT extra components or fallback path).
  std::vector<std::vector<sprec_t>> rows;

  const bool   do_mct    = (NC >= 3 && MCT != 0);
  const uint8_t xform    = tcomp[0].get_transformation();
  // ATK (xform>=2) is irreversible; dispatch table has only 2 entries (0=irrev, 1=rev).
  const uint8_t xform_ct = (xform == 1) ? 1 : 0;
  const uint32_t mct_w   = ci[0].csize_x;

  // Pre-build FinalizeParams for the fused MCT+finalize path.
  FinalizeParams fp[3] = {};
  for (uint16_t c = 0; c < std::min(NC, static_cast<uint16_t>(3)); ++c) {
    fp[c].ds     = ci[c].downshift;
    fp[c].rnd    = ci[c].rnd;
    fp[c].dc     = ci[c].DC_OFFSET;
    fp[c].maxval = ci[c].MAXVAL;
    fp[c].minval = ci[c].MINVAL;
  }

  // Allocate scratch rows only for components not covered by the fused MCT path.
  const uint16_t first_non_mct = do_mct ? 3 : 0;
  rows.resize(NC);
  for (uint16_t c = first_non_mct; c < NC; ++c) {
    const size_t sz = static_cast<size_t>(round_up(
        static_cast<int32_t>(ci[c].csize_x) + SIMD_PADDING, SIMD_PADDING));
    rows[c].assign(sz, 0.0f);
  }

  // Init line-based decoder state on all components (ring mode: use per-strip ring buffers).
  for (uint16_t c = 0; c < NC; ++c)
    tcomp[c].init_line_decode(/*ring_mode=*/true);

  // Pull rows from the stateful IDWT, apply per-row color transform and
  // float→int32 conversion, then write to the output buffer.
  const uint32_t H = ci[0].csize_y;
  for (uint32_t y = 0; y < H; ++y) {
    if (do_mct) {
      // Fused: get read-only ring buffer pointers for the 3 MCT components, apply inverse
      // MCT + float→int32 finalize in a single pass (no scratch buffer, no memcpy).
      const sprec_t *p0 = tcomp[0].pull_line_ref();
      const sprec_t *p1 = tcomp[1].pull_line_ref();
      const sprec_t *p2 = tcomp[2].pull_line_ref();
      int32_t *dp0 = ci[0].cdst + ci[0].x_offset + (y + ci[0].y_offset) * ci[0].out_stride;
      int32_t *dp1 = ci[1].cdst + ci[1].x_offset + (y + ci[1].y_offset) * ci[1].out_stride;
      int32_t *dp2 = ci[2].cdst + ci[2].x_offset + (y + ci[2].y_offset) * ci[2].out_stride;
      fused_mct_finalize[xform_ct](p0, p1, p2, dp0, dp1, dp2, mct_w, fp);
      // Extra components beyond 3 (no MCT applied); finalize individually.
      for (uint16_t c = 3; c < NC; ++c) {
        if (y >= ci[c].csize_y) continue;
        tcomp[c].pull_line(rows[c].data());
        const CInfo   &I   = ci[c];
        const sprec_t *spf = rows[c].data();
        int32_t       *dp  = I.cdst + I.x_offset + (y + I.y_offset) * I.out_stride;
        for (uint32_t n = 0; n < I.csize_x; ++n) {
          int32_t v = static_cast<int32_t>(spf[n]);
          v = (I.downshift < 0) ? (v + I.rnd) << -I.downshift
                                : (I.downshift > 0) ? (v + I.rnd) >> I.downshift : v;
          v += I.DC_OFFSET;
          if (v > I.MAXVAL) v = I.MAXVAL;
          if (v < I.MINVAL) v = I.MINVAL;
          dp[n] = v;
        }
      }
    } else {
      // No MCT: each component is independent. Use pull_line_ref + per-component finalize.
      for (uint16_t c = 0; c < NC; ++c) {
        if (y >= ci[c].csize_y) continue;
        const sprec_t *spf = tcomp[c].pull_line_ref();
        const CInfo   &I   = ci[c];
        int32_t       *dp  = I.cdst + I.x_offset + (y + I.y_offset) * I.out_stride;
        const int16_t  ds  = I.downshift;
        const int16_t  ro  = I.rnd;
#if defined(OPENHTJ2K_TRY_AVX2) && defined(__AVX2__)
        {
          const __m256i vdco = _mm256_set1_epi32(I.DC_OFFSET);
          const __m256i vmx  = _mm256_set1_epi32(I.MAXVAL);
          const __m256i vmn  = _mm256_set1_epi32(I.MINVAL);
          uint32_t n = 0;
          if (ds < 0) {
            const __m128i vsh  = _mm_cvtsi32_si128(-ds);
            const __m256i vrnd = _mm256_set1_epi32(ro);
            for (; n + 16 <= I.csize_x; n += 16) {
              __m256i v0 = _mm256_sll_epi32(_mm256_add_epi32(_mm256_cvttps_epi32(_mm256_loadu_ps(spf + n)), vrnd), vsh);
              __m256i v1 = _mm256_sll_epi32(_mm256_add_epi32(_mm256_cvttps_epi32(_mm256_loadu_ps(spf + n + 8)), vrnd), vsh);
              _mm256_storeu_si256((__m256i *)(dp + n),
                                  _mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(v0, vdco), vmn), vmx));
              _mm256_storeu_si256((__m256i *)(dp + n + 8),
                                  _mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(v1, vdco), vmn), vmx));
            }
          } else if (ds > 0) {
            const __m128i vsh  = _mm_cvtsi32_si128(ds);
            const __m256i vrnd = _mm256_set1_epi32(ro);
            for (; n + 16 <= I.csize_x; n += 16) {
              __m256i v0 = _mm256_sra_epi32(_mm256_add_epi32(_mm256_cvttps_epi32(_mm256_loadu_ps(spf + n)), vrnd), vsh);
              __m256i v1 = _mm256_sra_epi32(_mm256_add_epi32(_mm256_cvttps_epi32(_mm256_loadu_ps(spf + n + 8)), vrnd), vsh);
              _mm256_storeu_si256((__m256i *)(dp + n),
                                  _mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(v0, vdco), vmn), vmx));
              _mm256_storeu_si256((__m256i *)(dp + n + 8),
                                  _mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(v1, vdco), vmn), vmx));
            }
          } else {
            for (; n + 16 <= I.csize_x; n += 16) {
              __m256i v0 = _mm256_cvttps_epi32(_mm256_loadu_ps(spf + n));
              __m256i v1 = _mm256_cvttps_epi32(_mm256_loadu_ps(spf + n + 8));
              _mm256_storeu_si256((__m256i *)(dp + n),
                                  _mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(v0, vdco), vmn), vmx));
              _mm256_storeu_si256((__m256i *)(dp + n + 8),
                                  _mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(v1, vdco), vmn), vmx));
            }
          }
          for (; n < I.csize_x; ++n) {
            int32_t v = static_cast<int32_t>(spf[n]);
            v = (ds < 0) ? (v + ro) << -ds : (ds > 0) ? (v + ro) >> ds : v;
            v += I.DC_OFFSET;
            if (v > I.MAXVAL) v = I.MAXVAL;
            if (v < I.MINVAL) v = I.MINVAL;
            dp[n] = v;
          }
        }
#else
        for (uint32_t n = 0; n < I.csize_x; ++n) {
          int32_t v = static_cast<int32_t>(spf[n]);
          v = (ds < 0) ? (v + ro) << -ds : (ds > 0) ? (v + ro) >> ds : v;
          v += I.DC_OFFSET;
          if (v > I.MAXVAL) v = I.MAXVAL;
          if (v < I.MINVAL) v = I.MINVAL;
          dp[n] = v;
        }
#endif
      }
    }
  }

  for (uint16_t c = 0; c < NC; ++c)
    tcomp[c].finalize_line_decode();
}

void j2k_tile::decode_line_based_stream(j2k_main_header &hdr, uint8_t reduce_NL_val,
                                        const std::function<void(uint32_t, int32_t *const *, uint16_t)> &cb,
                                        uint32_t row_limit,
                                        uint32_t col_lo_in, uint32_t col_hi_in) {
  const uint16_t NC = num_components;

  // Apply per-level column range to each component's IDWT state chain.
  // col_lo_in / col_hi_in are in the finest active-level output coord space
  // (= subsampled-tcomp coords after reduce).  When the caller uses the
  // defaults [0, UINT32_MAX) every level's col range resolves to the full
  // [u0, u1] → idwt kernel loops unchanged from pre-patch behaviour.
  if (col_lo_in != 0 || col_hi_in != UINT32_MAX) {
    for (uint16_t c = 0; c < NC; ++c) {
      tcomp[c].set_line_decode_col_range(col_lo_in, col_hi_in);
    }
  }

  struct CInfo {
    int32_t DC_OFFSET, MAXVAL, MINVAL;
    int16_t downshift, rnd;
    uint32_t csize_x, csize_y;
    uint32_t yr;  // SIZ YRsiz: vertical subsampling factor (1 = no subsampling)
  };
  std::vector<CInfo> ci(NC);
  for (uint16_t c = 0; c < NC; ++c) {
    CInfo &I        = ci[c];
    const uint8_t bd = tcomp[c].bitdepth;
    I.DC_OFFSET      = (hdr.SIZ->is_signed(c)) ? 0 : 1 << (bd - 1);
    I.MAXVAL         = (hdr.SIZ->is_signed(c)) ? (1 << (bd - 1)) - 1 : (1 << bd) - 1;
    I.MINVAL         = (hdr.SIZ->is_signed(c)) ? -(1 << (bd - 1)) : 0;

    element_siz siz, Osiz, Rsiz;
    hdr.SIZ->get_image_size(siz);
    hdr.SIZ->get_image_origin(Osiz);
    hdr.SIZ->get_subsampling_factor(Rsiz, c);
    I.yr             = Rsiz.y;
    const uint8_t NL_c = tcomp[c].get_dwt_levels();
    j2k_resolution *cr_act =
        tcomp[c].access_resolution(static_cast<uint8_t>(NL_c - reduce_NL_val));
    const element_siz tc1 = cr_act->get_pos1();
    const element_siz tc0 = cr_act->get_pos0();
    I.csize_x  = tc1.x - tc0.x;
    I.csize_y  = tc1.y - tc0.y;
    I.downshift = (tcomp[c].transformation == 1) ? 0 : static_cast<int16_t>(FRACBITS - bd);
    I.rnd       = (I.downshift <= 0) ? 0 : static_cast<int16_t>((1 << I.downshift) >> 1);
  }

  // Per-component int32 output row scratch (one row each, reused per y).
  std::vector<std::vector<int32_t>> out_rows(NC);
  for (uint16_t c = 0; c < NC; ++c)
    out_rows[c].assign(ci[c].csize_x, 0);

  // Pointers passed to the callback.
  std::vector<int32_t *> out_ptrs(NC);
  for (uint16_t c = 0; c < NC; ++c)
    out_ptrs[c] = out_rows[c].data();

  const bool    do_mct = (NC >= 3 && MCT != 0);
  const uint8_t xform  = tcomp[0].get_transformation();
  // ATK (xform>=2) is irreversible; dispatch table has only 2 entries (0=irrev, 1=rev).
  const uint8_t xform_ct = (xform == 1) ? 1 : 0;
  const uint32_t mct_w  = ci[0].csize_x;

  // Pre-build FinalizeParams for the fused MCT+finalize path.
  FinalizeParams fp[3] = {};
  for (uint16_t c = 0; c < std::min(NC, static_cast<uint16_t>(3)); ++c) {
    fp[c].ds     = ci[c].downshift;
    fp[c].rnd    = ci[c].rnd;
    fp[c].dc     = ci[c].DC_OFFSET;
    fp[c].maxval = ci[c].MAXVAL;
    fp[c].minval = ci[c].MINVAL;
  }

  for (uint16_t c = 0; c < NC; ++c)
    tcomp[c].init_line_decode(/*ring_mode=*/true);

  const uint32_t H = ci[0].csize_y;
  const uint32_t effective_H = std::min(H, row_limit);

  // ── Strip-granular pull driver ────────────────────────────────────────────
  // Batch row pulls for one outer strip per component at a time, then run
  // the existing finalize + callback inner loop against the strip scratch.
  // Per-component pulls for the same strip are mutually independent (each
  // tcomp's line_dec state is disjoint), so they dispatch as worker tasks
  // while the main thread pitches in via try_run_one.
  //
  // Outer strip height = tcomp[0].codeblock_size.y rounded up to the max
  // vertical subsampling factor — guarantees every strip boundary is on an
  // integer chroma row.  Per-component pull counts:
  //   • MCT main (c=0..2 when do_mct): strip_y1 - strip_y0.  The existing
  //     driver iterates over ci[0].csize_y and pulls unconditionally for
  //     c=0..2, ignoring SIZ.YRsiz, because MCT demands 4:4:4 and the
  //     three components share geometry.  Matching this semantics is how
  //     conformance test p0_10 (YRsiz=4 on all 3 MCT comps) stays green.
  //   • Everything else: the subsampling-aware formula below, the same
  //     cadence the old non-MCT / MCT-extras path used.
  uint32_t max_yr = 1;
  for (uint16_t c = 0; c < NC; ++c) {
    if (ci[c].yr > max_yr) max_yr = ci[c].yr;
  }
  // Pick an outer strip height large enough to amortise the per-strip pull
  // dispatch + barrier, but not so large that the strip scratch outgrows L3.
  // Broadcast HTJ2K fixtures use cb_h as small as 8 rows for sub-codestream
  // latency; dispatching 3 tasks every 8 luma rows would spend more time on
  // mutex + spin than on actual IDWT.  Bump the strip up to at least 64 rows.
  uint32_t strip_h_luma = static_cast<uint32_t>(tcomp[0].codeblock_size.y);
  if (strip_h_luma == 0) strip_h_luma = 64;
  if (strip_h_luma < 64) strip_h_luma = 64;
  strip_h_luma = ((strip_h_luma + max_yr - 1) / max_yr) * max_yr;

  // Per-component base pointers into the strip scratch buffers; refreshed
  // at the top of every strip.  Row stride is ci[c].csize_x floats.
  std::vector<sprec_t *> strip_ptrs(NC, nullptr);

#ifdef OPENHTJ2K_THREAD
  // Scratch records for the per-component worker tasks.  The array lives
  // across the whole decode; spin-wait at the end of each strip guarantees
  // no task from strip N can still be reading these after strip N+1 begins
  // populating them.
  struct StripPullCtx {
    j2k_tile_component *tc;
    sprec_t           **slot;
    std::atomic<int>   *cnt;
    uint32_t            count;
    uint32_t            stride;
  };
  std::vector<StripPullCtx> strip_tasks(NC);
  std::atomic<int> pull_cnt(0);
  auto *pool = ThreadPool::get();
  const bool can_parallel_pull =
      (pool != nullptr) && (pool->num_threads() > 1) && (NC > 1);
#endif

  for (uint32_t strip_y0 = 0; strip_y0 < effective_H; strip_y0 += strip_h_luma) {
    const uint32_t strip_y1 = std::min(strip_y0 + strip_h_luma, effective_H);

    // Pre-compute per-component pull counts.
    uint32_t counts[16] = {};
    for (uint16_t c = 0; c < NC && c < 16; ++c) {
      if (do_mct && c < 3) {
        counts[c] = strip_y1 - strip_y0;
      } else {
        const uint32_t yr_c = ci[c].yr;
        const uint32_t c_y0 = strip_y0 / yr_c;
        const uint32_t c_y1 = std::min((strip_y1 + yr_c - 1) / yr_c, ci[c].csize_y);
        counts[c] = (c_y1 > c_y0) ? (c_y1 - c_y0) : 0u;
      }
    }

    // Phase 1: pull per-component strip rows into scratch.  Parallel when
    // a pool is available, serial otherwise.
#ifdef OPENHTJ2K_THREAD
    if (can_parallel_pull) {
      for (uint16_t c = 0; c < NC; ++c) {
        strip_tasks[c] = {&tcomp[c], &strip_ptrs[c], &pull_cnt,
                          counts[c], ci[c].csize_x};
      }
      pull_cnt.store(static_cast<int>(NC), std::memory_order_relaxed);
      // Batch-push all NC pull tasks in one mutex acquire to minimise
      // contention with the nested codeblock dispatch running inside
      // the tasks themselves.
      pool->push_batch(strip_tasks, [](const StripPullCtx &ctx) {
        const StripPullCtx *t = &ctx;
        return [t]() {
          *t->slot = t->tc->pull_strip_into_buf(t->count, t->stride);
          t->cnt->fetch_sub(1, std::memory_order_release);
        };
      });
      dec_strip_barrier_wait(pull_cnt);
    } else
#endif
    {
      for (uint16_t c = 0; c < NC; ++c) {
        strip_ptrs[c] = tcomp[c].pull_strip_into_buf(counts[c], ci[c].csize_x);
      }
    }

    // Phase 2: finalize + callback inner loop — reads strip_ptrs[c] at the
    // subsampling-aware row offset.  Inner SIMD kernels and control flow
    // are unchanged from the pre-strip driver.
    for (uint32_t y = strip_y0; y < strip_y1; ++y) {
    if (do_mct) {
      // Fused path: read 3 MCT component rows from strip scratch, then
      // apply inverse MCT + float→int32 finalize in a single pass.
      const size_t rs = static_cast<size_t>(y - strip_y0);
      const sprec_t *p0 = strip_ptrs[0] + rs * ci[0].csize_x;
      const sprec_t *p1 = strip_ptrs[1] + rs * ci[1].csize_x;
      const sprec_t *p2 = strip_ptrs[2] + rs * ci[2].csize_x;
      fused_mct_finalize[xform_ct](p0, p1, p2,
                                out_rows[0].data(), out_rows[1].data(), out_rows[2].data(),
                                mct_w, fp);
      // Extra components beyond 3 (no MCT applied).
      for (uint16_t c = 3; c < NC; ++c) {
        const uint32_t yr_c = ci[c].yr;
        if (y % yr_c != 0) continue;
        if (y / yr_c >= ci[c].csize_y) continue;
        const size_t rs_c = static_cast<size_t>((y - strip_y0) / yr_c);
        const sprec_t *spf = strip_ptrs[c] + rs_c * ci[c].csize_x;
        int32_t       *dp  = out_rows[c].data();
        const CInfo   &I   = ci[c];
        const int16_t  ds  = I.downshift;
        const int16_t  ro  = I.rnd;
#if defined(OPENHTJ2K_TRY_AVX2) && defined(__AVX2__)
        {
          const __m256i vdco = _mm256_set1_epi32(I.DC_OFFSET);
          const __m256i vmx  = _mm256_set1_epi32(I.MAXVAL);
          const __m256i vmn  = _mm256_set1_epi32(I.MINVAL);
          uint32_t n = 0;
          if (ds < 0) {
            const __m128i vsh = _mm_cvtsi32_si128(-ds);
            for (; n + 16 <= I.csize_x; n += 16) {
              __m256i v0 = _mm256_sll_epi32(_mm256_cvttps_epi32(_mm256_loadu_ps(spf + n)), vsh);
              __m256i v1 = _mm256_sll_epi32(_mm256_cvttps_epi32(_mm256_loadu_ps(spf + n + 8)), vsh);
              _mm256_storeu_si256((__m256i *)(dp + n),
                                  _mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(v0, vdco), vmn), vmx));
              _mm256_storeu_si256((__m256i *)(dp + n + 8),
                                  _mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(v1, vdco), vmn), vmx));
            }
          } else if (ds > 0) {
            const __m128i vsh = _mm_cvtsi32_si128(ds);
            const __m256i vrnd = _mm256_set1_epi32(ro);
            for (; n + 16 <= I.csize_x; n += 16) {
              __m256i v0 = _mm256_sra_epi32(_mm256_add_epi32(_mm256_cvttps_epi32(_mm256_loadu_ps(spf + n)), vrnd), vsh);
              __m256i v1 = _mm256_sra_epi32(_mm256_add_epi32(_mm256_cvttps_epi32(_mm256_loadu_ps(spf + n + 8)), vrnd), vsh);
              _mm256_storeu_si256((__m256i *)(dp + n),
                                  _mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(v0, vdco), vmn), vmx));
              _mm256_storeu_si256((__m256i *)(dp + n + 8),
                                  _mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(v1, vdco), vmn), vmx));
            }
          } else {
            for (; n + 16 <= I.csize_x; n += 16) {
              __m256i v0 = _mm256_cvttps_epi32(_mm256_loadu_ps(spf + n));
              __m256i v1 = _mm256_cvttps_epi32(_mm256_loadu_ps(spf + n + 8));
              _mm256_storeu_si256((__m256i *)(dp + n),
                                  _mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(v0, vdco), vmn), vmx));
              _mm256_storeu_si256((__m256i *)(dp + n + 8),
                                  _mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(v1, vdco), vmn), vmx));
            }
          }
          for (; n < I.csize_x; ++n) {
            int32_t v = static_cast<int32_t>(spf[n]);
            v = (ds < 0) ? (v + ro) << -ds : (ds > 0) ? (v + ro) >> ds : v;
            v += I.DC_OFFSET;
            if (v > I.MAXVAL) v = I.MAXVAL;
            if (v < I.MINVAL) v = I.MINVAL;
            dp[n] = v;
          }
        }
#elif defined(OPENHTJ2K_ENABLE_ARM_NEON)
        {
          const int32x4_t vdco = vdupq_n_s32(I.DC_OFFSET);
          const int32x4_t vmx  = vdupq_n_s32(I.MAXVAL);
          const int32x4_t vmn  = vdupq_n_s32(I.MINVAL);
          uint32_t n = 0;
          if (ds < 0) {
            const int32x4_t vsh = vdupq_n_s32(-ds);
            for (; n + 8 <= I.csize_x; n += 8) {
              int32x4_t v0 = vshlq_s32(vcvtq_s32_f32(vld1q_f32(spf + n)), vsh);
              int32x4_t v1 = vshlq_s32(vcvtq_s32_f32(vld1q_f32(spf + n + 4)), vsh);
              vst1q_s32(dp + n,     vmaxq_s32(vminq_s32(vaddq_s32(v0, vdco), vmx), vmn));
              vst1q_s32(dp + n + 4, vmaxq_s32(vminq_s32(vaddq_s32(v1, vdco), vmx), vmn));
            }
          } else if (ds > 0) {
            const int32x4_t vsh  = vdupq_n_s32(-ds);  // negative = right-shift
            const int32x4_t vrnd = vdupq_n_s32(ro);
            for (; n + 8 <= I.csize_x; n += 8) {
              int32x4_t v0 = vshlq_s32(vaddq_s32(vcvtq_s32_f32(vld1q_f32(spf + n)), vrnd), vsh);
              int32x4_t v1 = vshlq_s32(vaddq_s32(vcvtq_s32_f32(vld1q_f32(spf + n + 4)), vrnd), vsh);
              vst1q_s32(dp + n,     vmaxq_s32(vminq_s32(vaddq_s32(v0, vdco), vmx), vmn));
              vst1q_s32(dp + n + 4, vmaxq_s32(vminq_s32(vaddq_s32(v1, vdco), vmx), vmn));
            }
          } else {
            for (; n + 8 <= I.csize_x; n += 8) {
              int32x4_t v0 = vcvtq_s32_f32(vld1q_f32(spf + n));
              int32x4_t v1 = vcvtq_s32_f32(vld1q_f32(spf + n + 4));
              vst1q_s32(dp + n,     vmaxq_s32(vminq_s32(vaddq_s32(v0, vdco), vmx), vmn));
              vst1q_s32(dp + n + 4, vmaxq_s32(vminq_s32(vaddq_s32(v1, vdco), vmx), vmn));
            }
          }
          for (; n < I.csize_x; ++n) {
            int32_t v = static_cast<int32_t>(spf[n]);
            v = (ds < 0) ? (v + ro) << -ds : (ds > 0) ? (v + ro) >> ds : v;
            v += I.DC_OFFSET;
            if (v > I.MAXVAL) v = I.MAXVAL;
            if (v < I.MINVAL) v = I.MINVAL;
            dp[n] = v;
          }
        }
#else
        for (uint32_t n = 0; n < I.csize_x; ++n) {
          int32_t v = static_cast<int32_t>(spf[n]);
          v = (ds < 0) ? (v + ro) << -ds : (ds > 0) ? (v + ro) >> ds : v;
          v += I.DC_OFFSET;
          if (v > I.MAXVAL) v = I.MAXVAL;
          if (v < I.MINVAL) v = I.MINVAL;
          dp[n] = v;
        }
#endif
      }
    } else {
      // No MCT: each component is independent.  Read from its strip scratch
      // at the subsampling-aware offset; skip intermediate luma rows.
      for (uint16_t c = 0; c < NC; ++c) {
        const uint32_t yr_c = ci[c].yr;
        if (y % yr_c != 0) continue;           // reuse previous row for intermediate luma rows
        if (y / yr_c >= ci[c].csize_y) continue;  // past last row of this component
        const size_t rs_c = static_cast<size_t>((y - strip_y0) / yr_c);
        const sprec_t *spf = strip_ptrs[c] + rs_c * ci[c].csize_x;
        int32_t       *dp  = out_rows[c].data();
        const CInfo   &I   = ci[c];
        const int16_t  ds  = I.downshift;
        const int16_t  ro  = I.rnd;
#if defined(OPENHTJ2K_TRY_AVX2) && defined(__AVX2__)
        {
          const __m256i vdco = _mm256_set1_epi32(I.DC_OFFSET);
          const __m256i vmx  = _mm256_set1_epi32(I.MAXVAL);
          const __m256i vmn  = _mm256_set1_epi32(I.MINVAL);
          uint32_t n = 0;
          if (ds < 0) {
            const __m128i vsh = _mm_cvtsi32_si128(-ds);
            for (; n + 16 <= I.csize_x; n += 16) {
              __m256i v0 = _mm256_sll_epi32(_mm256_cvttps_epi32(_mm256_loadu_ps(spf + n)), vsh);
              __m256i v1 = _mm256_sll_epi32(_mm256_cvttps_epi32(_mm256_loadu_ps(spf + n + 8)), vsh);
              _mm256_storeu_si256((__m256i *)(dp + n),
                                  _mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(v0, vdco), vmn), vmx));
              _mm256_storeu_si256((__m256i *)(dp + n + 8),
                                  _mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(v1, vdco), vmn), vmx));
            }
          } else if (ds > 0) {
            const __m128i vsh = _mm_cvtsi32_si128(ds);
            const __m256i vrnd = _mm256_set1_epi32(ro);
            for (; n + 16 <= I.csize_x; n += 16) {
              __m256i v0 = _mm256_sra_epi32(_mm256_add_epi32(_mm256_cvttps_epi32(_mm256_loadu_ps(spf + n)), vrnd), vsh);
              __m256i v1 = _mm256_sra_epi32(_mm256_add_epi32(_mm256_cvttps_epi32(_mm256_loadu_ps(spf + n + 8)), vrnd), vsh);
              _mm256_storeu_si256((__m256i *)(dp + n),
                                  _mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(v0, vdco), vmn), vmx));
              _mm256_storeu_si256((__m256i *)(dp + n + 8),
                                  _mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(v1, vdco), vmn), vmx));
            }
          } else {
            for (; n + 16 <= I.csize_x; n += 16) {
              __m256i v0 = _mm256_cvttps_epi32(_mm256_loadu_ps(spf + n));
              __m256i v1 = _mm256_cvttps_epi32(_mm256_loadu_ps(spf + n + 8));
              _mm256_storeu_si256((__m256i *)(dp + n),
                                  _mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(v0, vdco), vmn), vmx));
              _mm256_storeu_si256((__m256i *)(dp + n + 8),
                                  _mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(v1, vdco), vmn), vmx));
            }
          }
          for (; n < I.csize_x; ++n) {
            int32_t v = static_cast<int32_t>(spf[n]);
            v = (ds < 0) ? (v + ro) << -ds : (ds > 0) ? (v + ro) >> ds : v;
            v += I.DC_OFFSET;
            if (v > I.MAXVAL) v = I.MAXVAL;
            if (v < I.MINVAL) v = I.MINVAL;
            dp[n] = v;
          }
        }
#elif defined(OPENHTJ2K_ENABLE_ARM_NEON)
        {
          const int32x4_t vdco = vdupq_n_s32(I.DC_OFFSET);
          const int32x4_t vmx  = vdupq_n_s32(I.MAXVAL);
          const int32x4_t vmn  = vdupq_n_s32(I.MINVAL);
          uint32_t n = 0;
          if (ds < 0) {
            const int32x4_t vsh = vdupq_n_s32(-ds);
            for (; n + 8 <= I.csize_x; n += 8) {
              int32x4_t v0 = vshlq_s32(vcvtq_s32_f32(vld1q_f32(spf + n)), vsh);
              int32x4_t v1 = vshlq_s32(vcvtq_s32_f32(vld1q_f32(spf + n + 4)), vsh);
              vst1q_s32(dp + n,     vmaxq_s32(vminq_s32(vaddq_s32(v0, vdco), vmx), vmn));
              vst1q_s32(dp + n + 4, vmaxq_s32(vminq_s32(vaddq_s32(v1, vdco), vmx), vmn));
            }
          } else if (ds > 0) {
            const int32x4_t vsh  = vdupq_n_s32(-ds);  // negative = right-shift
            const int32x4_t vrnd = vdupq_n_s32(ro);
            for (; n + 8 <= I.csize_x; n += 8) {
              int32x4_t v0 = vshlq_s32(vaddq_s32(vcvtq_s32_f32(vld1q_f32(spf + n)), vrnd), vsh);
              int32x4_t v1 = vshlq_s32(vaddq_s32(vcvtq_s32_f32(vld1q_f32(spf + n + 4)), vrnd), vsh);
              vst1q_s32(dp + n,     vmaxq_s32(vminq_s32(vaddq_s32(v0, vdco), vmx), vmn));
              vst1q_s32(dp + n + 4, vmaxq_s32(vminq_s32(vaddq_s32(v1, vdco), vmx), vmn));
            }
          } else {
            for (; n + 8 <= I.csize_x; n += 8) {
              int32x4_t v0 = vcvtq_s32_f32(vld1q_f32(spf + n));
              int32x4_t v1 = vcvtq_s32_f32(vld1q_f32(spf + n + 4));
              vst1q_s32(dp + n,     vmaxq_s32(vminq_s32(vaddq_s32(v0, vdco), vmx), vmn));
              vst1q_s32(dp + n + 4, vmaxq_s32(vminq_s32(vaddq_s32(v1, vdco), vmx), vmn));
            }
          }
          for (; n < I.csize_x; ++n) {
            int32_t v = static_cast<int32_t>(spf[n]);
            v = (ds < 0) ? (v + ro) << -ds : (ds > 0) ? (v + ro) >> ds : v;
            v += I.DC_OFFSET;
            if (v > I.MAXVAL) v = I.MAXVAL;
            if (v < I.MINVAL) v = I.MINVAL;
            dp[n] = v;
          }
        }
#else
        for (uint32_t n = 0; n < I.csize_x; ++n) {
          int32_t v = static_cast<int32_t>(spf[n]);
          v = (ds < 0) ? (v + ro) << -ds : (ds > 0) ? (v + ro) >> ds : v;
          v += I.DC_OFFSET;
          if (v > I.MAXVAL) v = I.MAXVAL;
          if (v < I.MINVAL) v = I.MINVAL;
          dp[n] = v;
        }
#endif
      }
    }

    cb(y, out_ptrs.data(), NC);
    }  // end of inner y-loop
  }    // end of outer strip loop

  for (uint16_t c = 0; c < NC; ++c)
    tcomp[c].finalize_line_decode();
}

// ── Direct-to-planar streaming decode ─────────────────────────────────────
// Row-by-row pull from IDWT ring → fused finalize+narrow → direct write to
// caller-provided uint8/uint16 plane buffers.  No strip scratch, no int32
// intermediate, no callback overhead.
void j2k_tile::decode_line_based_stream_planar(j2k_main_header &hdr, uint8_t reduce_NL_val,
                                               open_htj2k::PlanarOutputDesc *descs, uint16_t nc) {
  const uint16_t NC  = num_components;
  const uint8_t  MCT = hdr.COD->use_color_trafo();

  // MCT fallback: synthesize a callback that does the two-stage path.  This
  // keeps the initial implementation simple; a fused MCT+narrow NEON kernel
  // can be added later for 4:4:4 content.
  if (NC >= 3 && MCT != 0) {
    decode_line_based_stream(
        hdr, reduce_NL_val,
        [&](uint32_t y, int32_t *const *rows, uint16_t nc_cb) {
          for (uint16_t c = 0; c < nc_cb && c < nc; ++c) {
            const open_htj2k::PlanarOutputDesc &d = descs[c];
            const uint32_t yr_c = d.yr;
            if (y % yr_c != 0) continue;
            const uint32_t cy = y / yr_c;
            if (cy >= d.height) continue;
            if (d.is_16bit) {
              auto *dst = static_cast<uint16_t *>(d.base) + static_cast<size_t>(cy) * d.stride;
              for (uint32_t x = 0; x < d.width; ++x) {
                int32_t v = rows[c][x];
                if (v < d.minval) v = d.minval;
                if (v > d.maxval) v = d.maxval;
                dst[x] = static_cast<uint16_t>(v);
              }
            } else {
              auto *dst = static_cast<uint8_t *>(d.base) + static_cast<size_t>(cy) * d.stride;
              const int32_t ds = d.depth_shift;
              for (uint32_t x = 0; x < d.width; ++x) {
                int32_t v = rows[c][x];
                if (v < d.minval) v = d.minval;
                if (v > d.maxval) v = d.maxval;
                dst[x] = static_cast<uint8_t>(ds > 0 ? (v >> ds) : v);
              }
            }
          }
        });
    return;
  }

  // Non-MCT path: use the same strip-granular pull driver as
  // decode_line_based_stream (parallel per-component IDWT batch) but
  // replace Phase 2 with fused finalize+narrow → direct plane write.
  // This keeps the parallel IDWT throughput while eliminating the
  // out_rows int32 intermediate and callback overhead.

  // Compute CInfo (same as decode_line_based_stream).
  struct CInfo {
    uint32_t csize_x, csize_y, yr;
    int16_t downshift, rnd;
  };
  CInfo ci[16] = {};
  for (uint16_t c = 0; c < NC && c < 16; ++c) {
    element_siz Rsiz;
    hdr.SIZ->get_subsampling_factor(Rsiz, c);
    ci[c].yr = Rsiz.y;
    const uint8_t NL_c = tcomp[c].get_dwt_levels();
    j2k_resolution *cr_act =
        tcomp[c].access_resolution(static_cast<uint8_t>(NL_c - reduce_NL_val));
    ci[c].csize_x  = cr_act->get_pos1().x - cr_act->get_pos0().x;
    ci[c].csize_y  = cr_act->get_pos1().y - cr_act->get_pos0().y;
    ci[c].downshift = (tcomp[c].transformation == 1)
                          ? 0
                          : static_cast<int16_t>(FRACBITS - tcomp[c].bitdepth);
    ci[c].rnd = (ci[c].downshift <= 0) ? 0 : static_cast<int16_t>((1 << ci[c].downshift) >> 1);
  }

  for (uint16_t c = 0; c < NC; ++c)
    tcomp[c].init_line_decode(/*ring_mode=*/true);

  const uint32_t H = ci[0].csize_y;

  // Strip height — same logic as decode_line_based_stream.
  uint32_t max_yr = 1;
  for (uint16_t c = 0; c < NC; ++c) {
    if (ci[c].yr > max_yr) max_yr = ci[c].yr;
  }
  uint32_t strip_h_luma = static_cast<uint32_t>(tcomp[0].codeblock_size.y);
  if (strip_h_luma == 0) strip_h_luma = 64;
  if (strip_h_luma < 64) strip_h_luma = 64;
  strip_h_luma = ((strip_h_luma + max_yr - 1) / max_yr) * max_yr;

  std::vector<sprec_t *> strip_ptrs(NC, nullptr);

#ifdef OPENHTJ2K_THREAD
  struct StripPullCtx {
    j2k_tile_component *tc;
    sprec_t           **slot;
    std::atomic<int>   *cnt;
    uint32_t            count;
    uint32_t            stride;
  };
  std::vector<StripPullCtx> strip_tasks(NC);
  std::atomic<int> pull_cnt(0);
  auto *pool = ThreadPool::get();
  const bool can_parallel_pull =
      (pool != nullptr) && (pool->num_threads() > 1) && (NC > 1);
#endif

  for (uint32_t strip_y0 = 0; strip_y0 < H; strip_y0 += strip_h_luma) {
    const uint32_t strip_y1 = std::min(strip_y0 + strip_h_luma, H);

    // Pre-compute per-component pull counts.
    uint32_t counts[16] = {};
    for (uint16_t c = 0; c < NC && c < 16; ++c) {
      const uint32_t yr_c = ci[c].yr;
      const uint32_t c_y0 = strip_y0 / yr_c;
      const uint32_t c_y1 = std::min((strip_y1 + yr_c - 1) / yr_c, ci[c].csize_y);
      counts[c] = (c_y1 > c_y0) ? (c_y1 - c_y0) : 0u;
    }

    // Phase 1: pull per-component strip rows into scratch (parallel).
#ifdef OPENHTJ2K_THREAD
    if (can_parallel_pull) {
      for (uint16_t c = 0; c < NC; ++c) {
        strip_tasks[c] = {&tcomp[c], &strip_ptrs[c], &pull_cnt,
                          counts[c], ci[c].csize_x};
      }
      pull_cnt.store(static_cast<int>(NC), std::memory_order_relaxed);
      pool->push_batch(strip_tasks, [](const StripPullCtx &ctx) {
        const StripPullCtx *t = &ctx;
        return [t]() {
          *t->slot = t->tc->pull_strip_into_buf(t->count, t->stride);
          t->cnt->fetch_sub(1, std::memory_order_release);
        };
      });
      dec_strip_barrier_wait(pull_cnt);
    } else
#endif
    {
      for (uint16_t c = 0; c < NC; ++c) {
        strip_ptrs[c] = tcomp[c].pull_strip_into_buf(counts[c], ci[c].csize_x);
      }
    }

    // Phase 2: fused finalize+narrow directly to caller's plane buffers.
    // No out_rows int32 intermediate, no callback.
    for (uint32_t y = strip_y0; y < strip_y1; ++y) {
      for (uint16_t c = 0; c < NC && c < nc; ++c) {
        const uint32_t yr_c = ci[c].yr;
        if (y % yr_c != 0) continue;
        if (y / yr_c >= ci[c].csize_y) continue;
        const size_t rs_c = static_cast<size_t>((y - strip_y0) / yr_c);
        const sprec_t *spf = strip_ptrs[c] + rs_c * ci[c].csize_x;
        const open_htj2k::PlanarOutputDesc &d = descs[c];
        const uint32_t cy = y / yr_c;

        if (d.is_16bit) {
          auto *dst = static_cast<uint16_t *>(d.base) + static_cast<size_t>(cy) * d.stride;
          open_htj2k::finalize_f32_to_u16(spf, dst, d.width, ci[c].downshift, ci[c].rnd,
                                          d.dc, d.maxval, d.minval);
        } else {
          auto *dst = static_cast<uint8_t *>(d.base) + static_cast<size_t>(cy) * d.stride;
          open_htj2k::finalize_f32_to_u8(spf, dst, d.width, ci[c].downshift, ci[c].rnd,
                                         d.dc, d.maxval, d.minval, d.depth_shift);
        }
      }
    }
  }

  for (uint16_t c = 0; c < NC; ++c)
    tcomp[c].finalize_line_decode();
}

// Helper: after init_line_decode(), mark all subband row bufs as bypass (sb->i_samples is
// already populated by a prior decode_cblks call).
// Diagnostic: decode codeblocks only (no IDWT), then run line-based IDWT from pre-decoded data.
void j2k_tile::decode_line_based_predecoded(j2k_main_header &hdr, uint8_t reduce_NL_val,
                                            std::vector<int32_t *> &dst) {
  // Step 1: decode all codeblocks into sb->i_samples (same as decode() first loop).
  // Grow-only scratch buffers shared across all components and resolution levels —
  // avoids one heap allocation per level (previously std::vector per level).
  size_t   dl_sample_cap = 0, dl_state_cap = 0;
  int32_t *dl_sample_buf = nullptr;
  uint8_t *dl_state_buf  = nullptr;

  for (uint16_t c = 0; c < num_components; c++) {
    const uint8_t ROIshift = this->tcomp[c].get_ROIshift();
    const uint8_t NL       = this->tcomp[c].get_dwt_levels();

    for (int8_t lev = (int8_t)NL; lev >= this->reduce_NL; --lev) {
      j2k_resolution *cr           = this->tcomp[c].access_resolution(static_cast<uint8_t>(NL - lev));
      const uint32_t num_precincts = cr->npw * cr->nph;

      // Compute max codeblocks per precinct for pool sizing.
      uint32_t lev_max = 0;
      for (uint32_t p = 0; p < num_precincts; p++) {
        j2k_precinct *cp = cr->access_precinct(p);
        uint32_t total   = 0;
        for (uint8_t b = 0; b < cr->num_bands; b++)
          total += cp->access_pband(b)->num_codeblock_x * cp->access_pband(b)->num_codeblock_y;
        lev_max = std::max(lev_max, total);
      }
      if (lev_max == 0) continue;

      // Grow-only: realloc only when capacity is insufficient.
      const size_t need_s  = static_cast<size_t>(lev_max) * 4096;
      const size_t need_st = static_cast<size_t>(lev_max) * 6156;
      if (need_s > dl_sample_cap) {
        aligned_mem_free(dl_sample_buf);
        dl_sample_buf = static_cast<int32_t *>(aligned_mem_alloc(need_s * sizeof(int32_t), 32));
        dl_sample_cap = need_s;
      }
      if (need_st > dl_state_cap) {
        std::free(dl_state_buf);
        dl_state_buf = static_cast<uint8_t *>(std::malloc(need_st));
        dl_state_cap = need_st;
      }

      for (uint32_t p = 0; p < num_precincts; p++) {
        j2k_precinct *cp = cr->access_precinct(p);
        int32_t *pbuf    = dl_sample_buf;
        uint8_t *spbuf   = dl_state_buf;

        // Assign buffer pointers and decode in one pass.
        // ht_cleanup_decode writes every sample_buf and block_states position
        // before reading them, so pre-zeroing is unnecessary for single-pass
        // HT blocks (the common case for lossless HTJ2K). For EBCOT and
        // multi-pass HT blocks, zero only what is actually needed.
        for (uint8_t b = 0; b < cr->num_bands; b++) {
          j2k_precinct_subband *cpb = cp->access_pband(b);
          const uint32_t nc        = cpb->num_codeblock_x * cpb->num_codeblock_y;
          for (uint32_t bi = 0; bi < nc; ++bi) {
            j2k_codeblock *block = cpb->access_codeblock(bi);
            const uint32_t QWx2  = round_up(block->size.x, 8U);
            const uint32_t QHx2  = round_up(block->size.y, 8U);
            block->sample_buf    = pbuf; pbuf += QWx2 * QHx2;
            block->block_states  = spbuf; spbuf += (QWx2 + 2) * (QHx2 + 2);
            // Same JPIP precinct-filter guard as the tile decode path: a
            // masked codeblock has its packet header parsed (num_passes > 0)
            // but no compressed body attached.
            if (!block->num_passes || block->get_compressed_data() == nullptr) continue;
            const bool is_ht = (block->Cmodes & HT) >> 6;
            if (!is_ht) {
              // EBCOT: both buffers must be pre-zeroed.
              memset(block->sample_buf, 0, QWx2 * QHx2 * sizeof(int32_t));
              memset(block->block_states, 0, (QWx2 + 2) * (QHx2 + 2));
            } else if (block->num_passes > 1) {
              // HT multi-pass: sigprop/magref read the block_states border
              // (written by cleanup only for the interior). Zero block_states;
              // sample_buf is fully written by cleanup before sigprop reads it.
              memset(block->block_states, 0, (QWx2 + 2) * (QHx2 + 2));
            }
            // HT single-pass: ht_cleanup_decode initialises all positions
            // before reading — no pre-zeroing needed.
            if (is_ht)
              htj2k_decode(block, ROIshift);
            else
              j2k_decode(block, ROIshift);
          }
        }
      }
    }
  }
  aligned_mem_free(dl_sample_buf);
  std::free(dl_state_buf);

  // Step 2: run line-based path but bypass decode_strip (use pre-decoded sb->i_samples).
  const uint16_t NC = num_components;
  struct CInfo {
    int32_t DC_OFFSET, MAXVAL, MINVAL;
    int16_t downshift, rnd;
    uint32_t csize_x, csize_y;
    uint32_t x_offset, y_offset, out_stride;
    int32_t *cdst;
  };
  std::vector<CInfo> ci(NC);
  for (uint16_t c = 0; c < NC; ++c) {
    CInfo &I = ci[c];
    const uint8_t bd = tcomp[c].bitdepth;
    I.DC_OFFSET = (hdr.SIZ->is_signed(c)) ? 0 : 1 << (bd - 1);
    I.MAXVAL    = (hdr.SIZ->is_signed(c)) ? (1 << (bd - 1)) - 1 : (1 << bd) - 1;
    I.MINVAL    = (hdr.SIZ->is_signed(c)) ? -(1 << (bd - 1)) : 0;

    element_siz siz, Osiz, Rsiz;
    hdr.SIZ->get_image_size(siz);
    hdr.SIZ->get_image_origin(Osiz);
    hdr.SIZ->get_subsampling_factor(Rsiz, c);
    const uint32_t x0 = ceil_int(Osiz.x, Rsiz.x);
    const uint32_t y0 = ceil_int(Osiz.y, Rsiz.y);
    const uint32_t x1 = ceil_int(siz.x, Rsiz.x);
    const uint8_t NL_c = tcomp[c].get_dwt_levels();
    j2k_resolution *cr_act =
        tcomp[c].access_resolution(static_cast<uint8_t>(NL_c - reduce_NL_val));
    const element_siz tc0 = cr_act->get_pos0();
    const element_siz tc1 = cr_act->get_pos1();
    I.csize_x   = tc1.x - tc0.x;
    I.csize_y   = tc1.y - tc0.y;
    I.x_offset  = tc0.x - ceil_int(x0, (1U << reduce_NL_val));
    I.y_offset  = tc0.y - ceil_int(y0, (1U << reduce_NL_val));
    I.out_stride = ceil_int(x1 - x0, (1U << reduce_NL_val));
    I.downshift = (tcomp[c].transformation == 1) ? 0 : static_cast<int16_t>(FRACBITS - bd);
    I.rnd       = (I.downshift <= 0) ? 0 : static_cast<int16_t>((1 << I.downshift) >> 1);
    I.cdst      = dst[c];
  }

  // Per-component float row scratch buffers (each sized with SIMD headroom).
  std::vector<std::vector<sprec_t>> rows(NC);
  for (uint16_t c = 0; c < NC; ++c) {
    const size_t sz = static_cast<size_t>(round_up(static_cast<int32_t>(ci[c].csize_x) + SIMD_PADDING, SIMD_PADDING));
    rows[c].assign(sz, 0.0f);
  }

  const bool    do_mct = (NC >= 3 && MCT != 0);
  const uint8_t xform  = tcomp[0].get_transformation();
  // ATK (xform>=2) is irreversible; dispatch table has only 2 entries (0=irrev, 1=rev).
  const uint8_t xform_ct = (xform == 1) ? 1 : 0;
  const uint32_t mct_w = ci[0].csize_x;
  const uint32_t mct_str = round_up(mct_w, 32U);

  for (uint16_t c = 0; c < NC; ++c) {
    tcomp[c].init_line_decode();
    tcomp[c].mark_line_dec_predecoded();
  }

  const uint32_t H = ci[0].csize_y;
  for (uint32_t y = 0; y < H; ++y) {
    for (uint16_t c = 0; c < NC; ++c)
      if (y < ci[c].csize_y) tcomp[c].pull_line(rows[c].data());

    if (do_mct)
      cvt_ycbcr_to_rgb_float[xform_ct](rows[0].data(), rows[1].data(), rows[2].data(), mct_w, 1, mct_str);

    for (uint16_t c = 0; c < NC; ++c) {
      if (y >= ci[c].csize_y) continue;
      const CInfo &I   = ci[c];
      const sprec_t *spf = rows[c].data();
      int32_t *dp      = I.cdst + I.x_offset + (y + I.y_offset) * I.out_stride;
      const int16_t ds = I.downshift;
      const int16_t ro = I.rnd;
#if defined(OPENHTJ2K_ENABLE_WASM_SIMD)
      {
        const v128_t vdco = wasm_i32x4_splat(I.DC_OFFSET);
        const v128_t vmx  = wasm_i32x4_splat(I.MAXVAL);
        const v128_t vmn  = wasm_i32x4_splat(I.MINVAL);
        uint32_t n = 0;
        if (ds == 0) {
          for (; n + 16 <= I.csize_x; n += 16) {
            v128_t v0 = wasm_i32x4_trunc_sat_f32x4(wasm_v128_load(spf + n));
            v128_t v1 = wasm_i32x4_trunc_sat_f32x4(wasm_v128_load(spf + n + 4));
            v128_t v2 = wasm_i32x4_trunc_sat_f32x4(wasm_v128_load(spf + n + 8));
            v128_t v3 = wasm_i32x4_trunc_sat_f32x4(wasm_v128_load(spf + n + 12));
            v0 = wasm_i32x4_max(wasm_i32x4_min(wasm_i32x4_add(v0, vdco), vmx), vmn);
            v1 = wasm_i32x4_max(wasm_i32x4_min(wasm_i32x4_add(v1, vdco), vmx), vmn);
            v2 = wasm_i32x4_max(wasm_i32x4_min(wasm_i32x4_add(v2, vdco), vmx), vmn);
            v3 = wasm_i32x4_max(wasm_i32x4_min(wasm_i32x4_add(v3, vdco), vmx), vmn);
            wasm_v128_store(dp + n, v0);
            wasm_v128_store(dp + n + 4, v1);
            wasm_v128_store(dp + n + 8, v2);
            wasm_v128_store(dp + n + 12, v3);
          }
        } else if (ds < 0) {
          const v128_t vro = wasm_i32x4_splat(ro);
          for (; n + 16 <= I.csize_x; n += 16) {
            v128_t v0 = wasm_i32x4_shl(wasm_i32x4_add(wasm_i32x4_trunc_sat_f32x4(wasm_v128_load(spf + n)), vro), (int)-ds);
            v128_t v1 = wasm_i32x4_shl(wasm_i32x4_add(wasm_i32x4_trunc_sat_f32x4(wasm_v128_load(spf + n + 4)), vro), (int)-ds);
            v128_t v2 = wasm_i32x4_shl(wasm_i32x4_add(wasm_i32x4_trunc_sat_f32x4(wasm_v128_load(spf + n + 8)), vro), (int)-ds);
            v128_t v3 = wasm_i32x4_shl(wasm_i32x4_add(wasm_i32x4_trunc_sat_f32x4(wasm_v128_load(spf + n + 12)), vro), (int)-ds);
            v0 = wasm_i32x4_max(wasm_i32x4_min(wasm_i32x4_add(v0, vdco), vmx), vmn);
            v1 = wasm_i32x4_max(wasm_i32x4_min(wasm_i32x4_add(v1, vdco), vmx), vmn);
            v2 = wasm_i32x4_max(wasm_i32x4_min(wasm_i32x4_add(v2, vdco), vmx), vmn);
            v3 = wasm_i32x4_max(wasm_i32x4_min(wasm_i32x4_add(v3, vdco), vmx), vmn);
            wasm_v128_store(dp + n, v0);
            wasm_v128_store(dp + n + 4, v1);
            wasm_v128_store(dp + n + 8, v2);
            wasm_v128_store(dp + n + 12, v3);
          }
        } else {
          const v128_t vro = wasm_i32x4_splat(ro);
          for (; n + 16 <= I.csize_x; n += 16) {
            v128_t v0 = wasm_i32x4_shr(wasm_i32x4_add(wasm_i32x4_trunc_sat_f32x4(wasm_v128_load(spf + n)), vro), (int)ds);
            v128_t v1 = wasm_i32x4_shr(wasm_i32x4_add(wasm_i32x4_trunc_sat_f32x4(wasm_v128_load(spf + n + 4)), vro), (int)ds);
            v128_t v2 = wasm_i32x4_shr(wasm_i32x4_add(wasm_i32x4_trunc_sat_f32x4(wasm_v128_load(spf + n + 8)), vro), (int)ds);
            v128_t v3 = wasm_i32x4_shr(wasm_i32x4_add(wasm_i32x4_trunc_sat_f32x4(wasm_v128_load(spf + n + 12)), vro), (int)ds);
            v0 = wasm_i32x4_max(wasm_i32x4_min(wasm_i32x4_add(v0, vdco), vmx), vmn);
            v1 = wasm_i32x4_max(wasm_i32x4_min(wasm_i32x4_add(v1, vdco), vmx), vmn);
            v2 = wasm_i32x4_max(wasm_i32x4_min(wasm_i32x4_add(v2, vdco), vmx), vmn);
            v3 = wasm_i32x4_max(wasm_i32x4_min(wasm_i32x4_add(v3, vdco), vmx), vmn);
            wasm_v128_store(dp + n, v0);
            wasm_v128_store(dp + n + 4, v1);
            wasm_v128_store(dp + n + 8, v2);
            wasm_v128_store(dp + n + 12, v3);
          }
        }
        for (; n < I.csize_x; ++n) {
          int32_t v = static_cast<int32_t>(spf[n]);
          v = (ds < 0) ? (v + ro) << -ds : (ds > 0) ? (v + ro) >> ds : v;
          v += I.DC_OFFSET;
          if (v > I.MAXVAL) v = I.MAXVAL;
          if (v < I.MINVAL) v = I.MINVAL;
          dp[n] = v;
        }
      }
#elif defined(OPENHTJ2K_ENABLE_ARM_NEON)
      {
        const int32x4_t vdco = vdupq_n_s32(I.DC_OFFSET);
        const int32x4_t vmx  = vdupq_n_s32(I.MAXVAL);
        const int32x4_t vmn  = vdupq_n_s32(I.MINVAL);
        uint32_t n = 0;
        if (ds == 0) {
          // Lossless: ro=0, ds=0 — skip add+shift entirely.
          for (; n + 16 <= I.csize_x; n += 16) {
            int32x4_t v0 = vcvtq_s32_f32(vld1q_f32(spf + n));
            int32x4_t v1 = vcvtq_s32_f32(vld1q_f32(spf + n + 4));
            int32x4_t v2 = vcvtq_s32_f32(vld1q_f32(spf + n + 8));
            int32x4_t v3 = vcvtq_s32_f32(vld1q_f32(spf + n + 12));
            v0 = vmaxq_s32(vminq_s32(vaddq_s32(v0, vdco), vmx), vmn);
            v1 = vmaxq_s32(vminq_s32(vaddq_s32(v1, vdco), vmx), vmn);
            v2 = vmaxq_s32(vminq_s32(vaddq_s32(v2, vdco), vmx), vmn);
            v3 = vmaxq_s32(vminq_s32(vaddq_s32(v3, vdco), vmx), vmn);
            vst1q_s32(dp + n, v0);
            vst1q_s32(dp + n + 4, v1);
            vst1q_s32(dp + n + 8, v2);
            vst1q_s32(dp + n + 12, v3);
          }
        } else if (ds < 0) {
          const int32x4_t vro = vdupq_n_s32(ro);
          const int32x4_t vs  = vdupq_n_s32(-ds);  // positive → left shift
          for (; n + 16 <= I.csize_x; n += 16) {
            int32x4_t v0 = vshlq_s32(vaddq_s32(vcvtq_s32_f32(vld1q_f32(spf + n)), vro), vs);
            int32x4_t v1 = vshlq_s32(vaddq_s32(vcvtq_s32_f32(vld1q_f32(spf + n + 4)), vro), vs);
            int32x4_t v2 = vshlq_s32(vaddq_s32(vcvtq_s32_f32(vld1q_f32(spf + n + 8)), vro), vs);
            int32x4_t v3 = vshlq_s32(vaddq_s32(vcvtq_s32_f32(vld1q_f32(spf + n + 12)), vro), vs);
            v0 = vmaxq_s32(vminq_s32(vaddq_s32(v0, vdco), vmx), vmn);
            v1 = vmaxq_s32(vminq_s32(vaddq_s32(v1, vdco), vmx), vmn);
            v2 = vmaxq_s32(vminq_s32(vaddq_s32(v2, vdco), vmx), vmn);
            v3 = vmaxq_s32(vminq_s32(vaddq_s32(v3, vdco), vmx), vmn);
            vst1q_s32(dp + n, v0);
            vst1q_s32(dp + n + 4, v1);
            vst1q_s32(dp + n + 8, v2);
            vst1q_s32(dp + n + 12, v3);
          }
        } else {
          const int32x4_t vro = vdupq_n_s32(ro);
          const int32x4_t vs  = vdupq_n_s32(-ds);  // negative → right shift
          for (; n + 16 <= I.csize_x; n += 16) {
            int32x4_t v0 = vshlq_s32(vaddq_s32(vcvtq_s32_f32(vld1q_f32(spf + n)), vro), vs);
            int32x4_t v1 = vshlq_s32(vaddq_s32(vcvtq_s32_f32(vld1q_f32(spf + n + 4)), vro), vs);
            int32x4_t v2 = vshlq_s32(vaddq_s32(vcvtq_s32_f32(vld1q_f32(spf + n + 8)), vro), vs);
            int32x4_t v3 = vshlq_s32(vaddq_s32(vcvtq_s32_f32(vld1q_f32(spf + n + 12)), vro), vs);
            v0 = vmaxq_s32(vminq_s32(vaddq_s32(v0, vdco), vmx), vmn);
            v1 = vmaxq_s32(vminq_s32(vaddq_s32(v1, vdco), vmx), vmn);
            v2 = vmaxq_s32(vminq_s32(vaddq_s32(v2, vdco), vmx), vmn);
            v3 = vmaxq_s32(vminq_s32(vaddq_s32(v3, vdco), vmx), vmn);
            vst1q_s32(dp + n, v0);
            vst1q_s32(dp + n + 4, v1);
            vst1q_s32(dp + n + 8, v2);
            vst1q_s32(dp + n + 12, v3);
          }
        }
        for (; n < I.csize_x; ++n) {
          int32_t v = static_cast<int32_t>(spf[n]);
          v = (ds < 0) ? (v + ro) << -ds : (ds > 0) ? (v + ro) >> ds : v;
          v += I.DC_OFFSET;
          if (v > I.MAXVAL) v = I.MAXVAL;
          if (v < I.MINVAL) v = I.MINVAL;
          dp[n] = v;
        }
      }
#else
      for (uint32_t n = 0; n < I.csize_x; ++n) {
        int32_t v = static_cast<int32_t>(spf[n]);
        v = (ds < 0) ? (v + ro) << -ds : (ds > 0) ? (v + ro) >> ds : v;
        v += I.DC_OFFSET;
        if (v > I.MAXVAL) v = I.MAXVAL;
        if (v < I.MINVAL) v = I.MINVAL;
        dp[n] = v;
      }
#endif
    }
  }

  for (uint16_t c = 0; c < NC; ++c)
    tcomp[c].finalize_line_decode();
}

void j2k_tile::enc_init(uint16_t idx, j2k_main_header &main_header, std::vector<int32_t *> img,
                        bool line_based, bool streaming) {
  if (img.empty() && !streaming) {
    printf("ERROR: input image is empty.\n");
    throw std::exception();
  }
  index          = idx;
  num_components = main_header.SIZ->get_num_components();
  // set coding style related properties from main header
  setCODparams(main_header.COD.get());
  // set quantization style related properties from main header
  setQCDparams(main_header.QCD.get());
  // set Ccap15(HTJ2K only or mixed)
  Ccap15 = (main_header.CAP != nullptr) ? main_header.CAP->get_Ccap(15) : 0;
  // create tile-part(s)
  this->tile_part.push_back(MAKE_UNIQUE<j2k_tile_part>(num_components));
  this->num_tile_part++;
  this->current_tile_part_pos++;
  SOT_marker tmpSOT;
  tmpSOT.set_SOT_marker(index, 0, 1);  // only one tile-part is supported
  this->tile_part[static_cast<size_t>(current_tile_part_pos)]->set_SOT(tmpSOT);
  j2k_tilepart_header *tphdr = this->tile_part[static_cast<size_t>(current_tile_part_pos)]->header.get();

  element_siz numTiles;
  element_siz Siz, Osiz, Tsiz, TOsiz;
  main_header.get_number_of_tiles(numTiles.x, numTiles.y);
  uint16_t p = static_cast<uint16_t>(this->index % numTiles.x);
  uint16_t q = static_cast<uint16_t>(this->index / numTiles.x);
  main_header.SIZ->get_image_size(Siz);
  main_header.SIZ->get_image_origin(Osiz);
  main_header.SIZ->get_tile_size(Tsiz);
  main_header.SIZ->get_tile_origin(TOsiz);

  this->pos0.x = std::max(TOsiz.x + p * Tsiz.x, Osiz.x);
  this->pos0.y = std::max(TOsiz.y + q * Tsiz.y, Osiz.y);
  this->pos1.x = std::min(TOsiz.x + (p + 1U) * Tsiz.x, Siz.x);
  this->pos1.y = std::min(TOsiz.y + (q + 1U) * Tsiz.y, Siz.y);

  // set coding style related properties from tile-part header
  if (tphdr->COD != nullptr) {
    setCODparams(tphdr->COD.get());
  }
  // set quantization style related properties from tile-part header
  if (tphdr->QCD != nullptr) {
    setQCDparams(tphdr->QCD.get());
  }

  // create tile components
  this->tcomp = MAKE_UNIQUE<j2k_tile_component[]>(num_components);

  const bool lb_enc = line_based && (streaming || !(MCT && num_components >= 3));

  for (uint16_t c = 0; c < num_components; c++) {
    this->tcomp[c].init(&main_header, tphdr, this, c, img, lb_enc);
    this->tcomp[c].create_resolutions(1, false, lb_enc);  // enc_lb skips resolution[NL].i_samples
  }

  // apply POC, if any
  if (tphdr->POC != nullptr) {
    for (unsigned long i = 0; i < tphdr->POC->nPOC; ++i) {
      porder_info.add(tphdr->POC->RSpoc[i], tphdr->POC->CSpoc[i], tphdr->POC->LYEpoc[i],
                      tphdr->POC->REpoc[i], tphdr->POC->CEpoc[i], tphdr->POC->Ppoc[i]);
    }
  } else if (main_header.POC != nullptr) {
    for (unsigned long i = 0; i < main_header.POC->nPOC; ++i) {
      porder_info.add(main_header.POC->RSpoc[i], main_header.POC->CSpoc[i], main_header.POC->LYEpoc[i],
                      main_header.POC->REpoc[i], main_header.POC->CEpoc[i], main_header.POC->Ppoc[i]);
    }
  }
}

int j2k_tile::perform_dc_offset(j2k_main_header &hdr) {
  int done = 0;
  for (uint16_t c = 0; c < this->num_components; ++c) {
    this->tcomp[c].perform_dc_offset(this->transformation, hdr.SIZ->is_signed(c));
    done += 1;
  }
  return done;
}

void j2k_tile::rgb_to_ycbcr() {
  if (num_components < 3) {
    return;
  }
  const uint8_t transformation = this->tcomp[0].get_transformation();
  assert(transformation == this->tcomp[1].get_transformation());
  assert(transformation == this->tcomp[2].get_transformation());

  const element_siz tc0 = this->tcomp[0].get_pos0();
  const element_siz tc1 = this->tcomp[0].get_pos1();
  const uint32_t width  = tc1.x - tc0.x;
  const uint32_t height = tc1.y - tc0.y;
  const uint32_t stride = round_up(width, 32U);

  int32_t *const sp0 = this->tcomp[0].get_sample_address(0, 0);  // assume that comp0 is red
  int32_t *const sp1 = this->tcomp[1].get_sample_address(0, 0);  // assume that comp1 is green
  int32_t *const sp2 = this->tcomp[2].get_sample_address(0, 0);  // assume that comp2 is blue
  // ATK (transformation>=2) is irreversible; dispatch table has only 2 entries (0=irrev, 1=rev).
  const uint8_t ct_idx = (transformation == 1) ? 1 : 0;
  if (MCT) {
    j2k_resolution *cr0 = this->tcomp[0].access_resolution(this->tcomp[0].get_dwt_levels());
    j2k_resolution *cr1 = this->tcomp[1].access_resolution(this->tcomp[1].get_dwt_levels());
    j2k_resolution *cr2 = this->tcomp[2].access_resolution(this->tcomp[2].get_dwt_levels());
    cvt_rgb_to_ycbcr_float[ct_idx](sp0, sp1, sp2, cr0->i_samples, cr1->i_samples,
                                   cr2->i_samples, width, height, stride);
  }
}

uint8_t *j2k_tile::encode() {
#ifdef OPENHTJ2K_THREAD
  auto pool = ThreadPool::get();
  // std::vector<std::future<int>> results;
#endif
  // Set up per-thread bump allocators once for the entire tile encode.
  // All components share these pools; data must remain valid until packet writing is done.
  if (!this->encode_pool_ctx) this->encode_pool_ctx = std::make_unique<EncodePoolCtx>();
  {
    auto *epc = this->encode_pool_ctx.get();
    ++epc->gen;
    epc->slot_cnt.store(0, std::memory_order_relaxed);
#ifdef OPENHTJ2K_THREAD
    const int nslots = (pool ? static_cast<int>(pool->num_threads()) : 0) + 1;
#else
    const int nslots = 1;
#endif
    while (static_cast<int>(epc->pools.size()) < nslots)
      epc->pools.emplace_back(std::make_unique<cblk_data_pool>());
    for (auto &ep : epc->pools) ep->reset();
  }
  auto *epc = this->encode_pool_ctx.get();

  // Copy pixel data (dword) to the root resolution buffer (word)
  for (uint16_t c = 0; c < num_components; c++) {
    const uint8_t ROIshift       = tcomp[c].get_ROIshift();
    const uint8_t NL             = tcomp[c].get_dwt_levels();
    const uint8_t transformation = tcomp[c].get_transformation();
    element_siz top_left         = tcomp[c].get_pos0();
    element_siz bottom_right     = tcomp[c].get_pos1();
    j2k_resolution *cr           = tcomp[c].access_resolution(NL);

    int32_t *const src = tcomp[c].get_sample_address(0, 0);

    uint32_t stride = round_up(bottom_right.x - top_left.x, 32U);
    uint32_t height = (bottom_right.y - top_left.y);

    // convert int32_t pixel values into float
    // (skipped for MCT components 0/1/2: already written by rgb_to_ycbcr())
    if (!(MCT && num_components >= 3 && c < 3)) {
#if defined(OPENHTJ2K_TRY_AVX2) && defined(__AVX2__)
      for (uint32_t y = 0; y < height; ++y) {
        int32_t *sp             = src + y * stride;
        sprec_t *dp             = cr->i_samples + y * stride;
        uint32_t num_tc_samples = bottom_right.x - top_left.x;
        for (; num_tc_samples >= 8; num_tc_samples -= 8) {
          auto v0 = _mm256_load_si256((__m256i *)sp);
          auto t0 = _mm256_cvtepi32_ps(v0);
          _mm256_store_ps(dp, t0);
          sp += 8;
          dp += 8;
        }
        for (; num_tc_samples > 0; --num_tc_samples) {
          *dp++ = static_cast<sprec_t>(*sp++);
        }
      }
#elif defined(OPENHTJ2K_ENABLE_WASM_SIMD)
      for (uint32_t y = 0; y < height; ++y) {
        int32_t *sp             = src + y * stride;
        sprec_t *dp             = cr->i_samples + y * stride;
        uint32_t num_tc_samples = bottom_right.x - top_left.x;
        for (; num_tc_samples >= 8; num_tc_samples -= 8) {
          wasm_v128_store(dp,     wasm_f32x4_convert_i32x4(wasm_v128_load(sp)));
          wasm_v128_store(dp + 4, wasm_f32x4_convert_i32x4(wasm_v128_load(sp + 4)));
          sp += 8;
          dp += 8;
        }
        for (; num_tc_samples > 0; --num_tc_samples) {
          *dp++ = static_cast<sprec_t>(*sp++);
        }
      }
#elif defined(OPENHTJ2K_ENABLE_ARM_NEON)
      for (uint32_t y = 0; y < height; ++y) {
        int32_t *sp             = src + y * stride;
        sprec_t *dp             = cr->i_samples + y * stride;
        uint32_t num_tc_samples = bottom_right.x - top_left.x;
        for (; num_tc_samples >= 8; num_tc_samples -= 8) {
          auto vsrc0 = vld1q_s32(sp);
          auto vsrc1 = vld1q_s32(sp + 4);
          vst1q_f32(dp, vcvtq_f32_s32(vsrc0));
          vst1q_f32(dp + 4, vcvtq_f32_s32(vsrc1));
          sp += 8;
          dp += 8;
        }
        for (; num_tc_samples > 0; --num_tc_samples) {
          *dp++ = static_cast<sprec_t>(*sp++);
        }
      }
#else
        for (uint32_t y = 0; y < height; ++y) {
          int32_t *sp             = src + y * stride;
          sprec_t *dp             = cr->i_samples + y * round_up(bottom_right.x - top_left.x, 32U);
          uint32_t num_tc_samples = bottom_right.x - top_left.x;
          for (; num_tc_samples > 0; --num_tc_samples) {
            *dp++ = static_cast<sprec_t>(*sp++);
          }
        }
#endif
    }  // end skip for MCT components

    // Lambda function of block-endocing
#ifdef OPENHTJ2K_THREAD
    auto t1_encode = [epc, pool](j2k_resolution *cr, uint8_t ROIshift) {
      // Per-task argument struct for the encoder: defined here to access private EncodePoolCtx.
      struct EncTaskArgs {
        EncodePoolCtx *epc;
        j2k_codeblock *block;
        uint8_t ROIshift;
        std::atomic<int> *remaining;
      };
      // Pre-scan: find the largest codeblock count across all precincts at this resolution.
      // Allocate once and reuse to avoid per-precinct mmap/munmap page-fault cycles on Linux.
      uint32_t max_total_cblks = 0;
      for (uint32_t p = 0; p < cr->npw * cr->nph; ++p) {
        j2k_precinct *cp     = cr->access_precinct(p);
        uint32_t total_cblks = 0;
        for (uint8_t b = 0; b < cr->num_bands; b++) {
          j2k_precinct_subband *cpb = cp->access_pband(b);
          total_cblks += cpb->num_codeblock_x * cpb->num_codeblock_y;
        }
        max_total_cblks = std::max(max_total_cblks, total_cblks);
      }
      if (max_total_cblks == 0) return;
      // Use the tile-level grow-only scratch buffers to avoid per-resolution malloc/free.
      epc->reserve_scratch(static_cast<size_t>(max_total_cblks) * 4096,
                           static_cast<size_t>(max_total_cblks) * 6156);
      int32_t *gbuf  = epc->gbuf;
      uint8_t *sgbuf = epc->sgbuf;
      // Pre-allocate task-arg array once (sized to max codeblocks per precinct).
      // Reused across precincts; pointer stability guaranteed since size never exceeds max_total_cblks.
      auto enc_task_args = std::make_unique<EncTaskArgs[]>(max_total_cblks);
      std::atomic<int> enc_remaining{0};

      for (uint32_t p = 0; p < cr->npw * cr->nph; ++p) {
        j2k_precinct *cp = cr->access_precinct(p);
        int32_t *pbuf    = gbuf;
        uint8_t *spbuf   = sgbuf;

        // Pass 1: assign buffer pointers to all codeblocks in this precinct.
        for (uint8_t b = 0; b < cr->num_bands; b++) {
          j2k_precinct_subband *cpb = cp->access_pband(b);
          const uint32_t num_cblks  = cpb->num_codeblock_x * cpb->num_codeblock_y;
          for (uint32_t block_index = 0; block_index < num_cblks; ++block_index) {
            auto block          = cpb->access_codeblock(block_index);
            const uint32_t QWx2 = round_up(block->size.x, 8U);
            const uint32_t QHx2 = round_up(block->size.y, 8U);
            block->sample_buf   = pbuf;
            pbuf += QWx2 * QHx2;
            block->block_states = spbuf;
            spbuf += (QWx2 + 2) * (QHx2 + 2);
          }
        }

        // Bulk zero of the used pool region for this precinct.
        // Reuses already-faulted pages after the first precinct, avoiding repeated mmap faults.
        memset(gbuf, 0, static_cast<size_t>(pbuf - gbuf) * sizeof(int32_t));
        memset(sgbuf, 0, static_cast<size_t>(spbuf - sgbuf));

        // Pass 2: encode all codeblocks (buffers are zeroed above).
        enc_remaining.store(0, std::memory_order_relaxed);
        uint32_t task_idx = 0;
        for (uint8_t b = 0; b < cr->num_bands; ++b) {
          j2k_precinct_subband *cpb = cp->access_pband(b);
          const uint32_t num_cblks  = cpb->num_codeblock_x * cpb->num_codeblock_y;
          for (uint32_t block_index = 0; block_index < num_cblks; ++block_index) {
            auto block = cpb->access_codeblock(block_index);
            if (pool && pool->num_threads() > 1) {
              auto *ea = &enc_task_args[task_idx++];
              *ea      = {epc, block, ROIshift, &enc_remaining};
              enc_remaining.fetch_add(1, std::memory_order_relaxed);
              pool->push([ea]() {
                // Claim a per-thread pool slot once per tile encode (generation-guarded).
                TlPoolSlot &ts = g_tl_pool_slot;
                if (ts.gen != ea->epc->gen) {
                  const int slot = ea->epc->slot_cnt.fetch_add(1, std::memory_order_relaxed);
                  ts.slot        = std::min(slot, static_cast<int>(ea->epc->pools.size()) - 1);
                  ts.gen         = ea->epc->gen;
                }
                g_cblk_pool = ea->epc->pools[static_cast<size_t>(ts.slot)].get();
                htj2k_encode(ea->block, ea->ROIshift);
                g_cblk_pool = nullptr;
                ea->remaining->fetch_sub(1, std::memory_order_release);
              });
            } else {
              g_cblk_pool = epc->pools[0].get();
              htj2k_encode(block, ROIshift);
              g_cblk_pool = nullptr;
            }
          }
        }
        while (enc_remaining.load(std::memory_order_acquire) > 0)
          std::this_thread::yield();
      }
    };
#else
    auto t1_encode = [epc](j2k_resolution *cr, uint8_t ROIshift) {
      // Pre-scan: find the largest codeblock count across all precincts at this resolution.
      // Allocate once and reuse to avoid per-precinct mmap/munmap page-fault cycles on Linux.
      uint32_t max_total_cblks = 0;
      for (uint32_t p = 0; p < cr->npw * cr->nph; ++p) {
        j2k_precinct *cp     = cr->access_precinct(p);
        uint32_t total_cblks = 0;
        for (uint8_t b = 0; b < cr->num_bands; b++) {
          j2k_precinct_subband *cpb = cp->access_pband(b);
          total_cblks += cpb->num_codeblock_x * cpb->num_codeblock_y;
        }
        max_total_cblks = std::max(max_total_cblks, total_cblks);
      }
      if (max_total_cblks == 0) return;
      // Use the tile-level grow-only scratch buffers to avoid per-resolution malloc/free.
      epc->reserve_scratch(static_cast<size_t>(max_total_cblks) * 4096,
                           static_cast<size_t>(max_total_cblks) * 6156);
      int32_t *gbuf  = epc->gbuf;
      uint8_t *sgbuf = epc->sgbuf;

      for (uint32_t p = 0; p < cr->npw * cr->nph; ++p) {
        j2k_precinct *cp = cr->access_precinct(p);
        int32_t *pbuf    = gbuf;
        uint8_t *spbuf   = sgbuf;

        // Pass 1: assign buffer pointers to all codeblocks in this precinct.
        for (uint8_t b = 0; b < cr->num_bands; b++) {
          j2k_precinct_subband *cpb = cp->access_pband(b);
          const uint32_t num_cblks  = cpb->num_codeblock_x * cpb->num_codeblock_y;
          for (uint32_t block_index = 0; block_index < num_cblks; ++block_index) {
            auto block          = cpb->access_codeblock(block_index);
            const uint32_t QWx2 = round_up(block->size.x, 8U);
            const uint32_t QHx2 = round_up(block->size.y, 8U);
            block->sample_buf   = pbuf;
            pbuf += QWx2 * QHx2;
            block->block_states = spbuf;
            spbuf += (QWx2 + 2) * (QHx2 + 2);
          }
        }

        // Bulk zero of the used pool region for this precinct.
        // Reuses already-faulted pages after the first precinct, avoiding repeated mmap faults.
        memset(gbuf, 0, static_cast<size_t>(pbuf - gbuf) * sizeof(int32_t));
        memset(sgbuf, 0, static_cast<size_t>(spbuf - sgbuf));

        // Pass 2: encode all codeblocks (buffers are zeroed above).
        g_cblk_pool = epc->pools[0].get();
        for (uint8_t b = 0; b < cr->num_bands; ++b) {
          j2k_precinct_subband *cpb = cp->access_pband(b);
          const uint32_t num_cblks  = cpb->num_codeblock_x * cpb->num_codeblock_y;
          for (uint32_t block_index = 0; block_index < num_cblks; ++block_index) {
            auto block = cpb->access_codeblock(block_index);
            htj2k_encode(block, ROIshift);
          }
        }
        g_cblk_pool = nullptr;
      }
    };
#endif
    // Allocate pse_scratch once for all fdwt_2d_sr_fixed calls in this component.
    // stride (= round_up(finest_width, 32)) is already computed above; sized for 8 PSE rows.
    sprec_t *fdwt_pse_scratch =
        (NL > 0 && stride > 0)
            ? static_cast<sprec_t *>(aligned_mem_alloc(sizeof(sprec_t) * 8 * stride, 32))
            : nullptr;
    // Allocate buf_scratch: pointer array for the vertical DWT row-pointer table.
    // Height at the finest level (before the DWT loop shrinks top_left/bottom_right).
    const uint32_t fdwt_finest_height = static_cast<uint32_t>(bottom_right.y - top_left.y);
    sprec_t **fdwt_buf_scratch        = (NL > 0 && fdwt_finest_height > 0)
                                            ? new sprec_t *[static_cast<size_t>(fdwt_finest_height + 8u)]
                                            : nullptr;
    // Forward DWT
    for (uint8_t r = NL; r > 0; --r) {
      j2k_resolution *ncr = tcomp[c].access_resolution(static_cast<uint8_t>(r - 1));
      const int32_t u0    = static_cast<int32_t>(top_left.x);
      const int32_t u1    = static_cast<int32_t>(bottom_right.x);
      const int32_t v0    = static_cast<int32_t>(top_left.y);
      const int32_t v1    = static_cast<int32_t>(bottom_right.y);
      j2k_subband *HL     = cr->access_subband(0);
      j2k_subband *LH     = cr->access_subband(1);
      j2k_subband *HH     = cr->access_subband(2);

      // wavelet
      if (u1 != u0 && v1 != v0) {
        // cr->scale();
        fdwt_2d_sr_fixed(cr->i_samples, ncr->i_samples, HL->i_samples, LH->i_samples, HH->i_samples, u0, u1,
                         v0, v1, transformation, fdwt_pse_scratch, fdwt_buf_scratch);
      }
      // encode codeblocks in HL, LH, and HH
      t1_encode(cr, ROIshift);
      cr           = tcomp[c].access_resolution(static_cast<uint8_t>(r - 1));
      top_left     = cr->get_pos0();
      bottom_right = cr->get_pos1();
    }
    aligned_mem_free(fdwt_pse_scratch);
    delete[] fdwt_buf_scratch;
    // encode codeblocks in LL
    t1_encode(cr, ROIshift);
  }  // end of component loop

  // #ifdef OPENHTJ2K_THREAD
  //   for (auto &result : results) {
  //     result.get();
  //   }
  // #endif

  // Encode packets
  for (uint16_t c = 0; c < num_components; c++) {
    OPENHTJ2K_MAYBE_UNUSED const uint8_t ROIshift = tcomp[c].get_ROIshift();
    const uint8_t NL                        = tcomp[c].get_dwt_levels();
    // const uint8_t transformation = tcomp[c].get_transformation();
    //    element_siz top_left     = tcomp[c].get_pos0();
    //    element_siz bottom_right = tcomp[c].get_pos1();
    j2k_resolution *cr = tcomp[c].access_resolution(NL);

    auto t1_encode_packet = [](uint16_t numlayers_local, bool use_EPH_local, j2k_resolution *cr) {
      uint32_t length = 0;
      for (uint32_t p = 0; p < cr->npw * cr->nph; ++p) {
        uint32_t packet_length = 0;
        j2k_precinct *cp       = cr->access_precinct(p);
        packet_header_writer pckt_hdr;
        for (uint8_t b = 0; b < cr->num_bands; ++b) {
          j2k_precinct_subband *cpb = cp->access_pband(b);
          const uint32_t num_cblks  = cpb->num_codeblock_x * cpb->num_codeblock_y;
          // #pragma omp parallel for reduction(+ : packet_length)
          for (uint32_t block_index = 0; block_index < num_cblks; ++block_index) {
            auto block = cpb->access_codeblock(block_index);
            packet_length += block->length;
          }
          // construct packet header
          cpb->generate_packet_header(pckt_hdr, static_cast<uint16_t>(numlayers_local - 1));
        }
        // emit_qword packet header
        pckt_hdr.flush(use_EPH_local);
        cp->packet_header_length = static_cast<uint32_t>(pckt_hdr.get_length());
        cp->packet_header        = MAKE_UNIQUE<uint8_t[]>(cp->packet_header_length);

        pckt_hdr.copy_buf(cp->packet_header.get());
        packet_length += pckt_hdr.get_length();
        cp->set_length(packet_length);
        length += packet_length;
      }
      return length;
    };
    for (uint8_t r = NL; r > 0; --r) {
      // encode codeblocks in HL or LH or HH
      length += static_cast<uint32_t>(t1_encode_packet(numlayers, use_EPH, cr));
      cr = tcomp[c].access_resolution(static_cast<uint8_t>(r - 1));
      //      top_left     = cr->get_pos0();
      //      bottom_right = cr->get_pos1();
    }
    // encode codeblocks in LL
    length += static_cast<uint32_t>(t1_encode_packet(numlayers, use_EPH, cr));
  }  // end of component loop

  tile_part[0]->set_tile_index(this->index);
  tile_part[0]->set_tile_part_index(0);  // currently, only a single tile-part is supported
  // Length of tile-part will be written in j2k_tile::write_packets()

  return nullptr;  // fake
}

uint8_t *j2k_tile::encode_line_based() {
#ifdef OPENHTJ2K_THREAD
  auto pool = ThreadPool::get();
#endif
  if (!this->encode_pool_ctx) this->encode_pool_ctx = std::make_unique<EncodePoolCtx>();
  {
    auto *epc = this->encode_pool_ctx.get();
    ++epc->gen;
    epc->slot_cnt.store(0, std::memory_order_relaxed);
#ifdef OPENHTJ2K_THREAD
    const int nslots = (pool ? static_cast<int>(pool->num_threads()) : 0) + 1;
#else
    const int nslots = 1;
#endif
    while (static_cast<int>(epc->pools.size()) < nslots)
      epc->pools.emplace_back(std::make_unique<cblk_data_pool>());
    for (auto &ep : epc->pools) ep->reset();
  }
  auto *epc = this->encode_pool_ctx.get();

#ifdef OPENHTJ2K_THREAD
  auto t1_encode = [epc, pool](j2k_resolution *cr, uint8_t ROIshift) {
    struct EncTaskArgs {
      EncodePoolCtx *epc;
      j2k_codeblock *block;
      uint8_t ROIshift;
      std::atomic<int> *remaining;
    };
    uint32_t max_total_cblks = 0;
    for (uint32_t p = 0; p < cr->npw * cr->nph; ++p) {
      j2k_precinct *cp     = cr->access_precinct(p);
      uint32_t total_cblks = 0;
      for (uint8_t b = 0; b < cr->num_bands; b++) {
        j2k_precinct_subband *cpb = cp->access_pband(b);
        total_cblks += cpb->num_codeblock_x * cpb->num_codeblock_y;
      }
      max_total_cblks = std::max(max_total_cblks, total_cblks);
    }
    if (max_total_cblks == 0) return;
    epc->reserve_scratch(static_cast<size_t>(max_total_cblks) * 4096,
                         static_cast<size_t>(max_total_cblks) * 6156);
    int32_t *gbuf  = epc->gbuf;
    uint8_t *sgbuf = epc->sgbuf;
    auto enc_task_args = std::make_unique<EncTaskArgs[]>(max_total_cblks);
    std::atomic<int> enc_remaining{0};
    for (uint32_t p = 0; p < cr->npw * cr->nph; ++p) {
      j2k_precinct *cp = cr->access_precinct(p);
      int32_t *pbuf    = gbuf;
      uint8_t *spbuf   = sgbuf;
      for (uint8_t b = 0; b < cr->num_bands; b++) {
        j2k_precinct_subband *cpb = cp->access_pband(b);
        const uint32_t num_cblks  = cpb->num_codeblock_x * cpb->num_codeblock_y;
        for (uint32_t block_index = 0; block_index < num_cblks; ++block_index) {
          auto block          = cpb->access_codeblock(block_index);
          const uint32_t QWx2 = round_up(block->size.x, 8U);
          const uint32_t QHx2 = round_up(block->size.y, 8U);
          block->sample_buf   = pbuf;
          pbuf += QWx2 * QHx2;
          block->block_states = spbuf;
          spbuf += (QWx2 + 2) * (QHx2 + 2);
        }
      }
      memset(gbuf, 0, static_cast<size_t>(pbuf - gbuf) * sizeof(int32_t));
      memset(sgbuf, 0, static_cast<size_t>(spbuf - sgbuf));
      enc_remaining.store(0, std::memory_order_relaxed);
      uint32_t task_idx = 0;
      for (uint8_t b = 0; b < cr->num_bands; ++b) {
        j2k_precinct_subband *cpb = cp->access_pband(b);
        const uint32_t num_cblks  = cpb->num_codeblock_x * cpb->num_codeblock_y;
        for (uint32_t block_index = 0; block_index < num_cblks; ++block_index) {
          auto block = cpb->access_codeblock(block_index);
          if (pool && pool->num_threads() > 1) {
            auto *ea = &enc_task_args[task_idx++];
            *ea      = {epc, block, ROIshift, &enc_remaining};
            enc_remaining.fetch_add(1, std::memory_order_relaxed);
            pool->push([ea]() {
              TlPoolSlot &ts = g_tl_pool_slot;
              if (ts.gen != ea->epc->gen) {
                const int slot = ea->epc->slot_cnt.fetch_add(1, std::memory_order_relaxed);
                ts.slot        = std::min(slot, static_cast<int>(ea->epc->pools.size()) - 1);
                ts.gen         = ea->epc->gen;
              }
              g_cblk_pool = ea->epc->pools[static_cast<size_t>(ts.slot)].get();
              htj2k_encode(ea->block, ea->ROIshift);
              g_cblk_pool = nullptr;
              ea->remaining->fetch_sub(1, std::memory_order_release);
            });
          } else {
            g_cblk_pool = epc->pools[0].get();
            htj2k_encode(block, ROIshift);
            g_cblk_pool = nullptr;
          }
        }
      }
      while (enc_remaining.load(std::memory_order_acquire) > 0)
        std::this_thread::yield();
    }
  };
#else
  auto t1_encode = [epc](j2k_resolution *cr, uint8_t ROIshift) {
    uint32_t max_total_cblks = 0;
    for (uint32_t p = 0; p < cr->npw * cr->nph; ++p) {
      j2k_precinct *cp     = cr->access_precinct(p);
      uint32_t total_cblks = 0;
      for (uint8_t b = 0; b < cr->num_bands; b++) {
        j2k_precinct_subband *cpb = cp->access_pband(b);
        total_cblks += cpb->num_codeblock_x * cpb->num_codeblock_y;
      }
      max_total_cblks = std::max(max_total_cblks, total_cblks);
    }
    if (max_total_cblks == 0) return;
    epc->reserve_scratch(static_cast<size_t>(max_total_cblks) * 4096,
                         static_cast<size_t>(max_total_cblks) * 6156);
    int32_t *gbuf  = epc->gbuf;
    uint8_t *sgbuf = epc->sgbuf;
    for (uint32_t p = 0; p < cr->npw * cr->nph; ++p) {
      j2k_precinct *cp = cr->access_precinct(p);
      int32_t *pbuf    = gbuf;
      uint8_t *spbuf   = sgbuf;
      for (uint8_t b = 0; b < cr->num_bands; b++) {
        j2k_precinct_subband *cpb = cp->access_pband(b);
        const uint32_t num_cblks  = cpb->num_codeblock_x * cpb->num_codeblock_y;
        for (uint32_t block_index = 0; block_index < num_cblks; ++block_index) {
          auto block          = cpb->access_codeblock(block_index);
          const uint32_t QWx2 = round_up(block->size.x, 8U);
          const uint32_t QHx2 = round_up(block->size.y, 8U);
          block->sample_buf   = pbuf;
          pbuf += QWx2 * QHx2;
          block->block_states = spbuf;
          spbuf += (QWx2 + 2) * (QHx2 + 2);
        }
      }
      memset(gbuf, 0, static_cast<size_t>(pbuf - gbuf) * sizeof(int32_t));
      memset(sgbuf, 0, static_cast<size_t>(spbuf - sgbuf));
      g_cblk_pool = epc->pools[0].get();
      for (uint8_t b = 0; b < cr->num_bands; ++b) {
        j2k_precinct_subband *cpb = cp->access_pband(b);
        const uint32_t num_cblks  = cpb->num_codeblock_x * cpb->num_codeblock_y;
        for (uint32_t block_index = 0; block_index < num_cblks; ++block_index) {
          auto block = cpb->access_codeblock(block_index);
          htj2k_encode(block, ROIshift);
        }
      }
      g_cblk_pool = nullptr;
    }
  };
#endif

  for (uint16_t c = 0; c < num_components; ++c) {
    const uint8_t ROIshift   = tcomp[c].get_ROIshift();
    const uint8_t NL         = tcomp[c].get_dwt_levels();
    element_siz top_left     = tcomp[c].get_pos0();
    element_siz bottom_right = tcomp[c].get_pos1();
    const uint32_t width     = bottom_right.x - top_left.x;
    const uint32_t height    = bottom_right.y - top_left.y;
    const uint32_t stride    = round_up(width, 32U);
    j2k_resolution *cr       = tcomp[c].access_resolution(NL);
    if (NL == 0) {
      // No FDWT: copy int32 → float to res[0]->i_samples (if not already done by MCT).
      if (!(MCT && num_components >= 3 && c < 3)) {
        int32_t *src = tcomp[c].get_sample_address(0, 0);
        for (uint32_t y = 0; y < height; ++y) {
          const int32_t *sp = src + y * stride;
          sprec_t *dp       = cr->i_samples + y * stride;
          for (uint32_t x = 0; x < width; ++x) dp[x] = static_cast<sprec_t>(sp[x]);
        }
      }
      t1_encode(cr, ROIshift);
      continue;
    }

    // Stateful FDWT path.
    tcomp[c].init_line_encode();

    if (MCT && num_components >= 3 && c < 3) {
      // rgb_to_ycbcr() already wrote float rows into res[NL]->i_samples.
      for (uint32_t y = 0; y < height; ++y)
        tcomp[c].push_line_enc(cr->i_samples + y * stride);
    } else {
      auto *scratch    = static_cast<sprec_t *>(aligned_mem_alloc(sizeof(sprec_t) * stride, 32));
      if (tcomp[c].lb_src_ptr != nullptr) {
        // LB encode mode: read directly from raw input, applying DC offset on-the-fly.
        const int32_t *src    = tcomp[c].lb_src_ptr;
        const uint32_t isrc   = tcomp[c].lb_src_stride;
        const int32_t dco     = tcomp[c].lb_dc_offset;
        const int32_t shiftup = tcomp[c].lb_dc_shiftup;
        if (shiftup >= 0) {
          for (uint32_t y = 0; y < height; ++y) {
            const int32_t *sp = src + y * isrc;
            for (uint32_t x = 0; x < width; ++x)
              scratch[x] = static_cast<sprec_t>((sp[x] << shiftup) - dco);
            tcomp[c].push_line_enc(scratch);
          }
        } else {
          for (uint32_t y = 0; y < height; ++y) {
            const int32_t *sp = src + y * isrc;
            for (uint32_t x = 0; x < width; ++x)
              scratch[x] = static_cast<sprec_t>((sp[x] >> (-shiftup)) - dco);
            tcomp[c].push_line_enc(scratch);
          }
        }
      } else {
        int32_t *src = tcomp[c].get_sample_address(0, 0);
        for (uint32_t y = 0; y < height; ++y) {
          const int32_t *sp = src + y * stride;
          for (uint32_t x = 0; x < width; ++x) scratch[x] = static_cast<sprec_t>(sp[x]);
          tcomp[c].push_line_enc(scratch);
        }
      }
      aligned_mem_free(scratch);
    }

    // Flush FDWT states and free the line_enc structures.
    tcomp[c].finalize_line_encode();

    // Encode codeblocks for all resolution levels.
    cr = tcomp[c].access_resolution(NL);
    for (uint8_t r = NL; r > 0; --r) {
      t1_encode(cr, ROIshift);
      cr = tcomp[c].access_resolution(static_cast<uint8_t>(r - 1));
    }
    t1_encode(cr, ROIshift);  // LL subband
  }

  // Build packet headers (identical to encode()).
  for (uint16_t c = 0; c < num_components; c++) {
    const uint8_t NL   = tcomp[c].get_dwt_levels();
    j2k_resolution *cr = tcomp[c].access_resolution(NL);

    auto t1_encode_packet = [](uint16_t numlayers_local, bool use_EPH_local, j2k_resolution *cr) {
      uint32_t length = 0;
      for (uint32_t p = 0; p < cr->npw * cr->nph; ++p) {
        uint32_t packet_length = 0;
        j2k_precinct *cp       = cr->access_precinct(p);
        packet_header_writer pckt_hdr;
        for (uint8_t b = 0; b < cr->num_bands; ++b) {
          j2k_precinct_subband *cpb = cp->access_pband(b);
          const uint32_t num_cblks  = cpb->num_codeblock_x * cpb->num_codeblock_y;
          for (uint32_t block_index = 0; block_index < num_cblks; ++block_index) {
            auto block = cpb->access_codeblock(block_index);
            packet_length += block->length;
          }
          cpb->generate_packet_header(pckt_hdr, static_cast<uint16_t>(numlayers_local - 1));
        }
        pckt_hdr.flush(use_EPH_local);
        cp->packet_header_length = static_cast<uint32_t>(pckt_hdr.get_length());
        cp->packet_header        = MAKE_UNIQUE<uint8_t[]>(cp->packet_header_length);
        pckt_hdr.copy_buf(cp->packet_header.get());
        packet_length += pckt_hdr.get_length();
        cp->set_length(packet_length);
        length += packet_length;
      }
      return length;
    };
    for (uint8_t r = NL; r > 0; --r) {
      length += static_cast<uint32_t>(t1_encode_packet(numlayers, use_EPH, cr));
      cr = tcomp[c].access_resolution(static_cast<uint8_t>(r - 1));
    }
    length += static_cast<uint32_t>(t1_encode_packet(numlayers, use_EPH, cr));
  }

  tile_part[0]->set_tile_index(this->index);
  tile_part[0]->set_tile_part_index(0);

  return nullptr;  // fake
}

uint8_t *j2k_tile::encode_line_based_stream(
    std::function<void(uint32_t y, int32_t **rows, uint16_t nc)> src_fn,
    const std::vector<uint32_t> &img_comp_widths) {
#ifdef OPENHTJ2K_THREAD
  auto pool = ThreadPool::get();
#endif
  if (!this->encode_pool_ctx) this->encode_pool_ctx = std::make_unique<EncodePoolCtx>();
  {
    auto *epc = this->encode_pool_ctx.get();
    ++epc->gen;
    epc->slot_cnt.store(0, std::memory_order_relaxed);
#ifdef OPENHTJ2K_THREAD
    const int nslots = (pool ? static_cast<int>(pool->num_threads()) : 0) + 1;
#else
    const int nslots = 1;
#endif
    while (static_cast<int>(epc->pools.size()) < nslots)
      epc->pools.emplace_back(std::make_unique<cblk_data_pool>());
    for (auto &ep : epc->pools) ep->reset();
  }
  auto *epc = this->encode_pool_ctx.get();

#ifdef OPENHTJ2K_THREAD
  auto t1_encode = [epc, pool](j2k_resolution *cr, uint8_t ROIshift) {
    struct EncTaskArgs {
      EncodePoolCtx *epc;
      j2k_codeblock *block;
      uint8_t ROIshift;
      std::atomic<int> *remaining;
    };
    uint32_t max_total_cblks = 0;
    for (uint32_t p = 0; p < cr->npw * cr->nph; ++p) {
      j2k_precinct *cp     = cr->access_precinct(p);
      uint32_t total_cblks = 0;
      for (uint8_t b = 0; b < cr->num_bands; b++) {
        j2k_precinct_subband *cpb = cp->access_pband(b);
        total_cblks += cpb->num_codeblock_x * cpb->num_codeblock_y;
      }
      max_total_cblks = std::max(max_total_cblks, total_cblks);
    }
    if (max_total_cblks == 0) return;
    epc->reserve_scratch(static_cast<size_t>(max_total_cblks) * 4096,
                         static_cast<size_t>(max_total_cblks) * 6156);
    int32_t *gbuf  = epc->gbuf;
    uint8_t *sgbuf = epc->sgbuf;
    auto enc_task_args = std::make_unique<EncTaskArgs[]>(max_total_cblks);
    std::atomic<int> enc_remaining{0};
    for (uint32_t p = 0; p < cr->npw * cr->nph; ++p) {
      j2k_precinct *cp = cr->access_precinct(p);
      int32_t *pbuf    = gbuf;
      uint8_t *spbuf   = sgbuf;
      for (uint8_t b = 0; b < cr->num_bands; b++) {
        j2k_precinct_subband *cpb = cp->access_pband(b);
        const uint32_t num_cblks  = cpb->num_codeblock_x * cpb->num_codeblock_y;
        for (uint32_t block_index = 0; block_index < num_cblks; ++block_index) {
          auto block          = cpb->access_codeblock(block_index);
          const uint32_t QWx2 = round_up(block->size.x, 8U);
          const uint32_t QHx2 = round_up(block->size.y, 8U);
          block->sample_buf   = pbuf;
          pbuf += QWx2 * QHx2;
          block->block_states = spbuf;
          spbuf += (QWx2 + 2) * (QHx2 + 2);
        }
      }
      memset(gbuf, 0, static_cast<size_t>(pbuf - gbuf) * sizeof(int32_t));
      memset(sgbuf, 0, static_cast<size_t>(spbuf - sgbuf));
      enc_remaining.store(0, std::memory_order_relaxed);
      uint32_t task_idx = 0;
      for (uint8_t b = 0; b < cr->num_bands; ++b) {
        j2k_precinct_subband *cpb = cp->access_pband(b);
        const uint32_t num_cblks  = cpb->num_codeblock_x * cpb->num_codeblock_y;
        for (uint32_t block_index = 0; block_index < num_cblks; ++block_index) {
          auto block = cpb->access_codeblock(block_index);
          if (pool && pool->num_threads() > 1) {
            auto *ea = &enc_task_args[task_idx++];
            *ea      = {epc, block, ROIshift, &enc_remaining};
            enc_remaining.fetch_add(1, std::memory_order_relaxed);
            pool->push([ea]() {
              TlPoolSlot &ts = g_tl_pool_slot;
              if (ts.gen != ea->epc->gen) {
                const int slot = ea->epc->slot_cnt.fetch_add(1, std::memory_order_relaxed);
                ts.slot        = std::min(slot, static_cast<int>(ea->epc->pools.size()) - 1);
                ts.gen         = ea->epc->gen;
              }
              g_cblk_pool = ea->epc->pools[static_cast<size_t>(ts.slot)].get();
              htj2k_encode(ea->block, ea->ROIshift);
              g_cblk_pool = nullptr;
              ea->remaining->fetch_sub(1, std::memory_order_release);
            });
          } else {
            g_cblk_pool = epc->pools[0].get();
            htj2k_encode(block, ROIshift);
            g_cblk_pool = nullptr;
          }
        }
      }
      while (enc_remaining.load(std::memory_order_acquire) > 0)
        std::this_thread::yield();
    }
  };
#else
  auto t1_encode = [epc](j2k_resolution *cr, uint8_t ROIshift) {
    uint32_t max_total_cblks = 0;
    for (uint32_t p = 0; p < cr->npw * cr->nph; ++p) {
      j2k_precinct *cp     = cr->access_precinct(p);
      uint32_t total_cblks = 0;
      for (uint8_t b = 0; b < cr->num_bands; b++) {
        j2k_precinct_subband *cpb = cp->access_pband(b);
        total_cblks += cpb->num_codeblock_x * cpb->num_codeblock_y;
      }
      max_total_cblks = std::max(max_total_cblks, total_cblks);
    }
    if (max_total_cblks == 0) return;
    epc->reserve_scratch(static_cast<size_t>(max_total_cblks) * 4096,
                         static_cast<size_t>(max_total_cblks) * 6156);
    int32_t *gbuf  = epc->gbuf;
    uint8_t *sgbuf = epc->sgbuf;
    for (uint32_t p = 0; p < cr->npw * cr->nph; ++p) {
      j2k_precinct *cp = cr->access_precinct(p);
      int32_t *pbuf    = gbuf;
      uint8_t *spbuf   = sgbuf;
      for (uint8_t b = 0; b < cr->num_bands; b++) {
        j2k_precinct_subband *cpb = cp->access_pband(b);
        const uint32_t num_cblks  = cpb->num_codeblock_x * cpb->num_codeblock_y;
        for (uint32_t block_index = 0; block_index < num_cblks; ++block_index) {
          auto block          = cpb->access_codeblock(block_index);
          const uint32_t QWx2 = round_up(block->size.x, 8U);
          const uint32_t QHx2 = round_up(block->size.y, 8U);
          block->sample_buf   = pbuf;
          pbuf += QWx2 * QHx2;
          block->block_states = spbuf;
          spbuf += (QWx2 + 2) * (QHx2 + 2);
        }
      }
      memset(gbuf, 0, static_cast<size_t>(pbuf - gbuf) * sizeof(int32_t));
      memset(sgbuf, 0, static_cast<size_t>(spbuf - sgbuf));
      g_cblk_pool = epc->pools[0].get();
      for (uint8_t b = 0; b < cr->num_bands; ++b) {
        j2k_precinct_subband *cpb = cp->access_pband(b);
        const uint32_t num_cblks  = cpb->num_codeblock_x * cpb->num_codeblock_y;
        for (uint32_t block_index = 0; block_index < num_cblks; ++block_index) {
          auto block = cpb->access_codeblock(block_index);
          htj2k_encode(block, ROIshift);
        }
      }
      g_cblk_pool = nullptr;
    }
  };
#endif

  // Geometry from component 0
  const uint8_t NL0        = tcomp[0].get_dwt_levels();
  const element_siz top0   = tcomp[0].get_pos0();
  const element_siz bot0   = tcomp[0].get_pos1();
  const uint32_t width     = bot0.x - top0.x;
  const uint32_t height    = bot0.y - top0.y;
  const uint32_t stride    = round_up(width, 32U);
  const uint8_t transformation = tcomp[0].get_transformation();
  // ATK (transformation>=2) is irreversible; dispatch table has only 2 entries (0=irrev, 1=rev).
  const uint8_t ct_idx = (transformation == 1) ? 1 : 0;

  // Per-component aligned row scratch buffers.
  // int_rows[c] must hold the full image row so that src_fn (which provides
  // the full-width image row) can write into it without overflow.
  // float_rows[c] only needs tile width.
  std::vector<int32_t *> int_rows(num_components, nullptr);
  std::vector<sprec_t *> float_rows(num_components, nullptr);
  for (uint16_t c = 0; c < num_components; ++c) {
    const uint32_t alloc_w = round_up(img_comp_widths[c], 32U);
    int_rows[c]            = static_cast<int32_t *>(aligned_mem_alloc(sizeof(int32_t) * alloc_w, 32));
    float_rows[c]          = static_cast<sprec_t *>(aligned_mem_alloc(sizeof(sprec_t) * stride, 32));
  }

  auto cleanup_rows = [&]() {
    for (uint16_t c = 0; c < num_components; ++c) {
      aligned_mem_free(int_rows[c]);
      aligned_mem_free(float_rows[c]);
    }
  };

  // Packet-header builder (identical to encode_line_based)
  auto build_packet_headers = [&]() {
    auto t1_encode_packet = [](uint16_t numlayers_local, bool use_EPH_local, j2k_resolution *cr) {
      uint32_t length = 0;
      for (uint32_t p = 0; p < cr->npw * cr->nph; ++p) {
        uint32_t packet_length = 0;
        j2k_precinct *cp       = cr->access_precinct(p);
        packet_header_writer pckt_hdr;
        for (uint8_t b = 0; b < cr->num_bands; ++b) {
          j2k_precinct_subband *cpb = cp->access_pband(b);
          const uint32_t num_cblks  = cpb->num_codeblock_x * cpb->num_codeblock_y;
          for (uint32_t block_index = 0; block_index < num_cblks; ++block_index) {
            auto block = cpb->access_codeblock(block_index);
            packet_length += block->length;
          }
          cpb->generate_packet_header(pckt_hdr, static_cast<uint16_t>(numlayers_local - 1));
        }
        pckt_hdr.flush(use_EPH_local);
        cp->packet_header_length = static_cast<uint32_t>(pckt_hdr.get_length());
        cp->packet_header        = MAKE_UNIQUE<uint8_t[]>(cp->packet_header_length);
        pckt_hdr.copy_buf(cp->packet_header.get());
        packet_length += pckt_hdr.get_length();
        cp->set_length(packet_length);
        length += packet_length;
      }
      return length;
    };
    for (uint16_t c = 0; c < num_components; ++c) {
      const uint8_t NL   = tcomp[c].get_dwt_levels();
      j2k_resolution *cr = tcomp[c].access_resolution(NL);
      for (uint8_t r = NL; r > 0; --r) {
        this->length += static_cast<uint32_t>(t1_encode_packet(numlayers, use_EPH, cr));
        cr = tcomp[c].access_resolution(static_cast<uint8_t>(r - 1));
      }
      this->length += static_cast<uint32_t>(t1_encode_packet(numlayers, use_EPH, cr));
    }
  };

  if (NL0 == 0) {
    // No DWT: fill i_samples row by row then encode.
    for (uint32_t y = 0; y < height; ++y) {
      src_fn(top0.y + y, int_rows.data(), num_components);
      if (MCT && num_components >= 3) {
        // Apply DC offset in-place for tile-local portion of each MCT component.
        const uint32_t x_off0 = static_cast<uint32_t>(tcomp[0].get_pos0().x);
        for (uint32_t c = 0; c < 3; ++c) {
          const int32_t dco = tcomp[c].lb_dc_offset;
          const int32_t shu = tcomp[c].lb_dc_shiftup;
          int32_t *row      = int_rows[c] + x_off0;
          if (shu >= 0)
            for (uint32_t x = 0; x < width; ++x) row[x] = (row[x] << shu) - dco;
          else
            for (uint32_t x = 0; x < width; ++x) row[x] = (row[x] >> (-shu)) - dco;
        }
        // Convert RGB→YCbCr float directly into i_samples
        j2k_resolution *cr0 = tcomp[0].access_resolution(0);
        j2k_resolution *cr1 = tcomp[1].access_resolution(0);
        j2k_resolution *cr2 = tcomp[2].access_resolution(0);
        cvt_rgb_to_ycbcr_float[ct_idx](int_rows[0] + x_off0, int_rows[1] + x_off0, int_rows[2] + x_off0,
                                               cr0->i_samples + y * stride, cr1->i_samples + y * stride,
                                               cr2->i_samples + y * stride, width, 1, stride);
        // Extra components (c >= 3): non-MCT
        for (uint16_t c = 3; c < num_components; ++c) {
          const int32_t dco     = tcomp[c].lb_dc_offset;
          const int32_t shu     = tcomp[c].lb_dc_shiftup;
          const uint32_t wc     = tcomp[c].get_pos1().x - tcomp[c].get_pos0().x;
          const uint32_t x_off  = static_cast<uint32_t>(tcomp[c].get_pos0().x);
          const int32_t *sp     = int_rows[c] + x_off;
          const uint32_t sc     = round_up(wc, 32U);
          sprec_t *dp           = tcomp[c].access_resolution(0)->i_samples + y * sc;
          if (shu >= 0)
            for (uint32_t x = 0; x < wc; ++x) dp[x] = static_cast<sprec_t>((sp[x] << shu) - dco);
          else
            for (uint32_t x = 0; x < wc; ++x) dp[x] = static_cast<sprec_t>((sp[x] >> (-shu)) - dco);
        }
      } else {
        for (uint16_t c = 0; c < num_components; ++c) {
          const uint32_t h_c = tcomp[c].get_pos1().y - tcomp[c].get_pos0().y;
          const uint32_t yr  = (h_c > 0 && h_c < height) ? height / h_c : 1;
          if (yr > 1 && (y % yr) != 0) continue;
          const uint32_t cy    = (yr > 1) ? (y / yr) : y;
          const int32_t dco    = tcomp[c].lb_dc_offset;
          const int32_t shu    = tcomp[c].lb_dc_shiftup;
          const uint32_t wc    = tcomp[c].get_pos1().x - tcomp[c].get_pos0().x;
          const uint32_t x_off = static_cast<uint32_t>(tcomp[c].get_pos0().x);
          const int32_t *sp    = int_rows[c] + x_off;
          const uint32_t sc    = round_up(wc, 32U);
          sprec_t *dp          = tcomp[c].access_resolution(0)->i_samples + cy * sc;
          if (shu >= 0)
            for (uint32_t x = 0; x < wc; ++x) dp[x] = static_cast<sprec_t>((sp[x] << shu) - dco);
          else
            for (uint32_t x = 0; x < wc; ++x) dp[x] = static_cast<sprec_t>((sp[x] >> (-shu)) - dco);
        }
      }
    }
    for (uint16_t c = 0; c < num_components; ++c)
      t1_encode(tcomp[c].access_resolution(0), tcomp[c].get_ROIshift());
    build_packet_headers();
    cleanup_rows();
    tile_part[0]->set_tile_index(this->index);
    tile_part[0]->set_tile_part_index(0);
    return nullptr;
  }

  // NL > 0: stateful FDWT with streaming input.
  for (uint16_t c = 0; c < num_components; ++c)
    tcomp[c].init_line_encode();

  // Set up DWT–HT block encoder overlap.
  // For each component, if hl_y0 == lh_y0 for every FDWT level (true for tile
  // origins aligned to 2^NL), pre-allocate one flat sample/state slab for all
  // HL/LH/HH codeblocks and wire up the per-level dispatch contexts.
  // Otherwise the component falls back to sequential encode after finalize.
#ifdef OPENHTJ2K_THREAD
  std::atomic<int> enc_remaining{0};
#endif
  std::vector<int32_t *> comp_gbuf(num_components, nullptr);
  std::vector<uint8_t *> comp_sgbuf(num_components, nullptr);
  std::vector<bool>      comp_overlap(num_components, false);

  for (uint16_t c = 0; c < num_components; ++c) {
    const uint8_t NL = tcomp[c].get_dwt_levels();
    auto *le = tcomp[c].get_line_enc();
    if (!le || le->NL_active == 0) continue;

    // Overlap requires hl_y0 == lh_y0 for all levels (true for y0=0 tiles).
    fdwt_level_sink_ctx *le_ctxs = le->ctxs.get();
    bool ok = true;
    for (int32_t i = 0; i < le->NL_active && ok; ++i)
      if (le_ctxs[i].hl_y0 != le_ctxs[i].lh_y0 || le_ctxs[i].hl_h == 0) ok = false;
    if (!ok) continue;

    // Compute total flat buffer sizes for levels 1..NL (HL/LH/HH codeblocks).
    uint32_t total_cbuf = 0, total_sbuf = 0;
    for (uint8_t r = 1; r <= NL; ++r) {
      j2k_resolution *cr = tcomp[c].access_resolution(r);
      for (uint32_t p = 0; p < cr->npw * cr->nph; ++p) {
        j2k_precinct *cp = cr->access_precinct(p);
        for (uint8_t b = 0; b < cr->num_bands; ++b) {
          j2k_precinct_subband *cpb = cp->access_pband(b);
          for (uint32_t cb = 0; cb < cpb->num_codeblock_x * cpb->num_codeblock_y; ++cb) {
            j2k_codeblock *block = cpb->access_codeblock(cb);
            const uint32_t QWx2 = round_up(block->size.x, 8U);
            const uint32_t QHx2 = round_up(block->size.y, 8U);
            total_cbuf += QWx2 * QHx2;
            total_sbuf += (QWx2 + 2) * (QHx2 + 2);
          }
        }
      }
    }
    if (total_cbuf == 0) continue;

    int32_t *gbuf  = static_cast<int32_t *>(malloc(total_cbuf * sizeof(int32_t)));
    uint8_t *sgbuf = static_cast<uint8_t *>(malloc(total_sbuf));
    memset(gbuf, 0, total_cbuf * sizeof(int32_t));
    memset(sgbuf, 0, total_sbuf);

    // Assign sample_buf/block_states pointers and set up per-level overlap state.
    int32_t *pbuf  = gbuf;
    uint8_t *spbuf = sgbuf;
    const int32_t cb_h_val = static_cast<int32_t>(tcomp[c].get_codeblock_size().y);

    for (int32_t i = 0; i < le->NL_active; ++i) {
      j2k_resolution *cr = tcomp[c].access_resolution(static_cast<uint8_t>(i + 1));
      auto &cx = le_ctxs[i];

      for (uint32_t p = 0; p < cr->npw * cr->nph; ++p) {
        j2k_precinct *cp = cr->access_precinct(p);
        for (uint8_t b = 0; b < cr->num_bands; ++b) {
          j2k_precinct_subband *cpb = cp->access_pband(b);
          for (uint32_t cb = 0; cb < cpb->num_codeblock_x * cpb->num_codeblock_y; ++cb) {
            j2k_codeblock *block = cpb->access_codeblock(cb);
            const uint32_t QWx2  = round_up(block->size.x, 8U);
            const uint32_t QHx2  = round_up(block->size.y, 8U);
            block->sample_buf    = pbuf;
            block->block_states  = spbuf;
            pbuf  += QWx2 * QHx2;
            spbuf += (QWx2 + 2) * (QHx2 + 2);
          }
        }
      }

      cx.cb_h         = cb_h_val;
      cx.num_cblk_rows = (std::max(cx.hl_h, cx.lh_h) + cb_h_val - 1) / cb_h_val;
      cx.cblk_row_done = std::make_unique<std::atomic<uint8_t>[]>(static_cast<size_t>(cx.num_cblk_rows));
      for (int32_t br = 0; br < cx.num_cblk_rows; ++br) {
        uint8_t init_flags = 0;
        if (br * cb_h_val >= cx.hl_h) init_flags |= 1;  // no HL codeblocks in this row
        if (br * cb_h_val >= cx.lh_h) init_flags |= 2;  // no LH/HH codeblocks in this row
        cx.cblk_row_done[static_cast<size_t>(br)].store(init_flags, std::memory_order_relaxed);
      }
      cx.enc_cr       = cr;
      cx.enc_ROIshift = tcomp[c].get_ROIshift();
      cx.enc_epc      = epc;
#ifdef OPENHTJ2K_THREAD
      cx.enc_pool      = pool;
      cx.enc_remaining = &enc_remaining;
#endif
    }

    comp_gbuf[c]    = gbuf;
    comp_sgbuf[c]   = sgbuf;
    comp_overlap[c] = true;
  }

  for (uint32_t y = 0; y < height; ++y) {
    src_fn(top0.y + y, int_rows.data(), num_components);
    if (MCT && num_components >= 3) {
      // Apply DC offset to tile-local portion of components 0,1,2 in-place.
      const uint32_t x_off0 = static_cast<uint32_t>(tcomp[0].get_pos0().x);
      for (uint32_t c = 0; c < 3; ++c) {
        const int32_t dco = tcomp[c].lb_dc_offset;
        const int32_t shu = tcomp[c].lb_dc_shiftup;
        int32_t *row      = int_rows[c] + x_off0;
        if (shu >= 0)
          for (uint32_t x = 0; x < width; ++x) row[x] = (row[x] << shu) - dco;
        else
          for (uint32_t x = 0; x < width; ++x) row[x] = (row[x] >> (-shu)) - dco;
      }
      // Convert RGB→YCbCr float into scratch float rows
      cvt_rgb_to_ycbcr_float[ct_idx](int_rows[0] + x_off0, int_rows[1] + x_off0, int_rows[2] + x_off0,
                                             float_rows[0], float_rows[1], float_rows[2], width, 1, stride);
      for (uint32_t c = 0; c < 3; ++c)
        tcomp[c].push_line_enc(float_rows[c]);
      // Extra components without MCT
      for (uint16_t c = 3; c < num_components; ++c) {
        const int32_t dco    = tcomp[c].lb_dc_offset;
        const int32_t shu    = tcomp[c].lb_dc_shiftup;
        const uint32_t wc    = tcomp[c].get_pos1().x - tcomp[c].get_pos0().x;
        const uint32_t x_off = static_cast<uint32_t>(tcomp[c].get_pos0().x);
        const int32_t *sp    = int_rows[c] + x_off;
        if (shu >= 0)
          for (uint32_t x = 0; x < wc; ++x) float_rows[c][x] = static_cast<sprec_t>((sp[x] << shu) - dco);
        else
          for (uint32_t x = 0; x < wc; ++x)
            float_rows[c][x] = static_cast<sprec_t>((sp[x] >> (-shu)) - dco);
        tcomp[c].push_line_enc(float_rows[c]);
      }
    } else {
      for (uint16_t c = 0; c < num_components; ++c) {
        const uint32_t h_c = tcomp[c].get_pos1().y - tcomp[c].get_pos0().y;
        const uint32_t yr  = (h_c > 0 && h_c < height) ? height / h_c : 1;
        if (yr > 1 && (y % yr) != 0) continue;
        const int32_t dco    = tcomp[c].lb_dc_offset;
        const int32_t shu    = tcomp[c].lb_dc_shiftup;
        const uint32_t wc    = tcomp[c].get_pos1().x - tcomp[c].get_pos0().x;
        const uint32_t x_off = static_cast<uint32_t>(tcomp[c].get_pos0().x);
        const int32_t *sp    = int_rows[c] + x_off;
        if (shu >= 0)
          for (uint32_t x = 0; x < wc; ++x) float_rows[c][x] = static_cast<sprec_t>((sp[x] << shu) - dco);
        else
          for (uint32_t x = 0; x < wc; ++x)
            float_rows[c][x] = static_cast<sprec_t>((sp[x] >> (-shu)) - dco);
        tcomp[c].push_line_enc(float_rows[c]);
      }
    }
  }

  for (uint16_t c = 0; c < num_components; ++c)
    tcomp[c].finalize_line_encode();

  // Wait for all overlapped HT block encoder tasks dispatched from the DWT sink.
#ifdef OPENHTJ2K_THREAD
  while (enc_remaining.load(std::memory_order_acquire) > 0)
    std::this_thread::yield();
#endif

  // For components that used the overlap path, HL/LH/HH are already encoded.
  // For fallback components, encode HL/LH/HH sequentially now.
  // In both cases, encode LL0 (r=0) sequentially — it's complete only after
  // finalize_line_encode() flushes the coarsest DWT stage.
  for (uint16_t c = 0; c < num_components; ++c) {
    const uint8_t ROIshift = tcomp[c].get_ROIshift();
    const uint8_t NL       = tcomp[c].get_dwt_levels();
    if (!comp_overlap[c]) {
      j2k_resolution *cr = tcomp[c].access_resolution(NL);
      for (uint8_t r = NL; r > 0; --r) {
        t1_encode(cr, ROIshift);
        cr = tcomp[c].access_resolution(static_cast<uint8_t>(r - 1));
      }
      // cr is now access_resolution(0) — LL0 encoded below
    }
    t1_encode(tcomp[c].access_resolution(0), ROIshift);
  }

  // Release pre-allocated codeblock slabs.
  for (uint16_t c = 0; c < num_components; ++c) {
    free(comp_gbuf[c]);
    free(comp_sgbuf[c]);
  }

  build_packet_headers();
  cleanup_rows();

  tile_part[0]->set_tile_index(this->index);
  tile_part[0]->set_tile_part_index(0);
  return nullptr;
}

OPENHTJ2K_MAYBE_UNUSED uint8_t j2k_tile::get_byte_from_tile_buf() { return this->tile_buf->get_byte(); }

OPENHTJ2K_MAYBE_UNUSED uint8_t j2k_tile::get_bit_from_tile_buf() { return this->tile_buf->get_bit(); }

OPENHTJ2K_MAYBE_UNUSED uint32_t j2k_tile::get_length() const { return length; }

OPENHTJ2K_MAYBE_UNUSED uint32_t j2k_tile::get_buf_length() { return tile_buf->get_total_length(); }
