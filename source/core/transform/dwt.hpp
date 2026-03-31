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
#include "open_htj2k_typedef.hpp"
#if defined(OPENHTJ2K_TRY_AVX2) && defined(__AVX2__)
  #define OPENHTJ2K_ENABLE_AVX2
#endif
#define SIMD_PADDING 32
constexpr int32_t DWT_VERT_STRIP = 64;  // column-strip width for vertical DWT (multiple of 8)

constexpr float fA = -1.586134342059924f;
constexpr float fB = -0.052980118572961f;
constexpr float fC = 0.882911075530934f;
constexpr float fD = 0.443506852043971f;

constexpr int16_t Acoeff_simd      = -19206;  // need to -1
constexpr int16_t Bcoeff_simd      = -3472;   // need to >> 1
constexpr int16_t Bcoeff_simd_avx2 = -13888;  // need to (out+4) >> 3
constexpr int16_t Ccoeff_simd      = 28931;
constexpr int16_t Dcoeff_simd      = 14533;

constexpr int32_t Acoeff = -25987;
constexpr int32_t Bcoeff = -3472;
constexpr int32_t Ccoeff = 28931;
constexpr int32_t Dcoeff = 29066;

constexpr int32_t Aoffset = 8192;
constexpr int32_t Boffset = 32767;
constexpr int32_t Coffset = 16384;
constexpr int32_t Doffset = 32767;

constexpr int32_t Ashift = 14;
constexpr int32_t Bshift = 16;
constexpr int32_t Cshift = 15;
constexpr int32_t Dshift = 16;

// define pointer to FDWT functions
typedef void (*fdwt_1d_filtr_func_fixed)(sprec_t *, const int32_t, const int32_t, const int32_t);
typedef void (*fdwt_ver_filtr_func_fixed)(sprec_t *, const int32_t, const int32_t, const int32_t,
                                          const int32_t, const int32_t stride, sprec_t *pse_scratch,
                                          sprec_t **buf_scratch);
// define pointer to IDWT functions
typedef void (*idwt_1d_filtd_func_fixed)(sprec_t *, const int32_t, const int32_t, const int32_t);
typedef void (*idwt_ver_filtd_func_fixed)(sprec_t *, const int32_t, const int32_t, const int32_t,
                                          const int32_t, const int32_t stride, sprec_t *pse_scratch,
                                          sprec_t **buf_scratch);

// symmetric extension
static inline int32_t PSEo(const int32_t i, const int32_t i0, const int32_t i1) {
  const int32_t tmp0    = 2 * (i1 - i0 - 1);
  const int32_t tmp1    = ((i - i0) < 0) ? i0 - i : i - i0;
  const int32_t mod_val = tmp1 % tmp0;
  const int32_t min_val = mod_val < tmp0 - mod_val ? mod_val : tmp0 - mod_val;
  return min_val;
}
template <class T>
static inline void dwt_1d_extr_fixed(T *extbuf, T *buf, const int32_t left, const int32_t right,
                                     const int32_t i0, const int32_t i1) {
  memcpy(extbuf + left, buf, sizeof(T) * static_cast<size_t>((i1 - i0)));
  for (int32_t i = 1; i <= left; ++i) {
    extbuf[left - i] = buf[PSEo(i0 - i, i0, i1)];
  }
  for (int32_t i = 1; i <= right; ++i) {
    extbuf[left + (i1 - i0) + i - 1] = buf[PSEo(i1 - i0 + i - 1 + i0, i0, i1)];
  }
}

// FDWT
#if defined(OPENHTJ2K_ENABLE_ARM_NEON)
void fdwt_1d_filtr_irrev97_fixed_neon(sprec_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
void fdwt_1d_filtr_rev53_fixed_neon(sprec_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
void fdwt_irrev_ver_sr_fixed_neon(sprec_t *in, int32_t u0, int32_t u1, int32_t v0, int32_t v1,
                                  int32_t stride, sprec_t *pse_scratch, sprec_t **buf_scratch);
void fdwt_rev_ver_sr_fixed_neon(sprec_t *in, int32_t u0, int32_t u1, int32_t v0, int32_t v1,
                                int32_t stride, sprec_t *pse_scratch, sprec_t **buf_scratch);

#elif defined(OPENHTJ2K_ENABLE_AVX2)
void fdwt_1d_filtr_irrev97_fixed_avx2(sprec_t *X, const int32_t left, const int32_t u_i0,
                                      const int32_t u_i1);
void fdwt_1d_filtr_rev53_fixed_avx2(sprec_t *X, const int32_t left, const int32_t u_i0, const int32_t u_i1);
void fdwt_irrev_ver_sr_fixed_avx2(sprec_t *in, const int32_t u0, const int32_t u1, const int32_t v0,
                                  const int32_t v1, const int32_t stride, sprec_t *pse_scratch,
                                  sprec_t **buf_scratch);
void fdwt_rev_ver_sr_fixed_avx2(sprec_t *in, const int32_t u0, const int32_t u1, const int32_t v0,
                                const int32_t v1, const int32_t stride, sprec_t *pse_scratch,
                                sprec_t **buf_scratch);
#else
void fdwt_1d_filtr_irrev97_fixed(sprec_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
void fdwt_1d_filtr_rev53_fixed(sprec_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
void fdwt_irrev_ver_sr_fixed(sprec_t *in, int32_t u0, int32_t u1, int32_t v0, int32_t v1, int32_t stride, sprec_t *pse_scratch, sprec_t **buf_scratch);
void fdwt_rev_ver_sr_fixed(sprec_t *in, int32_t u0, int32_t u1, int32_t v0, int32_t v1, int32_t stride, sprec_t *pse_scratch, sprec_t **buf_scratch);
#endif

void fdwt_2d_sr_fixed(sprec_t *previousLL, sprec_t *LL, sprec_t *HL, sprec_t *LH, sprec_t *HH, int32_t u0,
                      int32_t u1, int32_t v0, int32_t v1, uint8_t transformation, sprec_t *pse_scratch,
                      sprec_t **buf_scratch);

// IDWT
#if defined(OPENHTJ2K_ENABLE_ARM_NEON)
void idwt_1d_filtr_rev53_fixed_neon(sprec_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
void idwt_1d_filtr_irrev97_fixed_neon(sprec_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
void idwt_irrev_ver_sr_fixed_neon(sprec_t *in, int32_t u0, int32_t u1, int32_t v0, int32_t v1,
                                  int32_t stride, sprec_t *pse_scratch, sprec_t **buf_scratch);
void idwt_rev_ver_sr_fixed_neon(sprec_t *in, int32_t u0, int32_t u1, int32_t v0, int32_t v1,
                                int32_t stride, sprec_t *pse_scratch, sprec_t **buf_scratch);
// Single-row irreversible vertical lifting step: tgt[i] -= coeff*(prev[i]+next[i]) using FMA.
// Uses SIMD for multiples of 4 elements, scalar for the tail.
void idwt_irrev_ver_step_fixed_neon(int32_t n, float *prev, float *next, float *tgt, float coeff);
// Single-row reversible (5/3) vertical lifting steps.
void idwt_rev_ver_lp_step_neon(int32_t n, const float *prev, const float *next, float *tgt);
void idwt_rev_ver_hp_step_neon(int32_t n, const float *prev, const float *next, float *tgt);
#elif defined(OPENHTJ2K_ENABLE_AVX2)
void idwt_1d_filtr_rev53_fixed_avx2(sprec_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
void idwt_1d_filtr_irrev97_fixed_avx2(sprec_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
void idwt_irrev_ver_sr_fixed_avx2(sprec_t *in, int32_t u0, int32_t u1, int32_t v0, int32_t v1,
                                  int32_t stride, sprec_t *pse_scratch, sprec_t **buf_scratch);
void idwt_rev_ver_sr_fixed_avx2(sprec_t *in, int32_t u0, int32_t u1, int32_t v0, int32_t v1,
                                int32_t stride, sprec_t *pse_scratch, sprec_t **buf_scratch);
// Single-row irreversible vertical lifting step: tgt[i] -= coeff*(prev[i]+next[i]) using FMA.
// Uses SIMD for multiples of 8 elements, scalar for the tail.
void idwt_irrev_ver_step_fixed_avx2(int32_t n, float *prev, float *next, float *tgt, float coeff);
// Single-row reversible (5/3) vertical lifting steps.
void idwt_rev_ver_lp_step_avx2(int32_t n, const float *prev, const float *next, float *tgt);
void idwt_rev_ver_hp_step_avx2(int32_t n, const float *prev, const float *next, float *tgt);
#else
void idwt_1d_filtr_rev53_fixed(sprec_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
void idwt_1d_filtr_irrev97_fixed(sprec_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
void idwt_irrev_ver_sr_fixed(sprec_t *in, int32_t u0, int32_t u1, int32_t v0, int32_t v1, int32_t stride, sprec_t *pse_scratch, sprec_t **buf_scratch);
void idwt_rev_ver_sr_fixed(sprec_t *in, int32_t u0, int32_t u1, int32_t v0, int32_t v1, int32_t stride, sprec_t *pse_scratch, sprec_t **buf_scratch);
#endif
void idwt_2d_sr_fixed(sprec_t *nextLL, sprec_t *LL, sprec_t *HL, sprec_t *LH, sprec_t *HH, int32_t u0,
                      int32_t u1, int32_t v0, int32_t v1, uint8_t transformation, sprec_t *pse_scratch,
                      sprec_t **buf_scratch);

// Apply 1-D horizontal IDWT synthesis in-place on row[0..u1-u0-1].
// ext_buf must hold at least round_up(u1-u0+8+SIMD_PADDING, SIMD_PADDING) sprec_t elements.
void idwt_1d_row_fixed(sprec_t *ext_buf, sprec_t *row, int32_t u0, int32_t u1, uint8_t transformation);

// ─────────────────────────────────────────────────────────────────────────────
// Streaming 2D IDWT — produces one output row per call via pull_row().
//
// The caller supplies a row-source callback that returns one horizontally-
// synthesised interleaved row (LL+HL interleaved for LP rows, LH+HH for HP)
// for absolute row indices v0..v1-1.  Vertical lifting is driven internally
// using a sliding ring buffer and delay-line d_level tracking.
//
// Memory per level  ≈  (RING_DEPTH + top_pse + bottom_pse) × stride × 4 B
// ─────────────────────────────────────────────────────────────────────────────

// Callback: write the horizontally-synthesised interleaved row at absolute
// index abs_row ∈ [v0, v1) into out_row[0..u1-u0-1].
typedef void (*idwt_row_src_fn)(void *ctx, int32_t abs_row, sprec_t *out_row);

// Ring depth: must hold enough real rows for steady-state output.
// PSE rows use separate top_pse_buf/bot_pse_buf and never occupy ring slots.
// For 9/7 (max_dl=2): need rows r..r+4 simultaneously → 5 ring slots; RING_DEPTH=8 is safe.
// For 5/3 (max_dl=1): need rows r..r+2 simultaneously → 3 ring slots; RING_DEPTH=8 is more than enough.
// Reducing from 12 → 8 cuts level-1 ring buffer (4K: 3840 floats/row) from 180KB to 120KB,
// improving L2 cache utilization for vertical lifting steps.
constexpr int32_t IDWT_STATE_RING_DEPTH = 8;

struct idwt_2d_state {
  // ── geometry ──────────────────────────────────────────────────────────────
  int32_t u0, u1, v0, v1;
  int32_t stride;          // round_up(u1-u0, SIMD_PADDING)
  uint8_t transformation;  // 0 = irrev 9/7, 1 = rev 5/3
  int8_t  top_pse;         // PSE rows above v0  (3 or 4 for 9/7; 1 or 2 for 5/3)
  int8_t  bottom_pse;      // PSE rows below v1-1

  // ── PSE scratch (separate from the ring) ──────────────────────────────────
  // top_pse_buf[0] ↔ physical row v0-1, [1] ↔ v0-2, …
  // bot_pse_buf[0] ↔ physical row v1,   [1] ↔ v1+1, …
  sprec_t *top_pse_buf;        // top_pse    × stride sprec_t (SIMD-aligned)
  sprec_t *bot_pse_buf;        // bottom_pse × stride sprec_t
  int8_t   top_dlevel[4];      // d_level per top-PSE slot (-1 = unfilled)
  int8_t   bot_dlevel[4];      // d_level per bot-PSE slot (-1 = unfilled)

  // ── sliding ring for real rows [v0, v1) ───────────────────────────────────
  // Slot for absolute row r : (r - ring_origin) % IDWT_STATE_RING_DEPTH
  sprec_t *ring_buf;                           // IDWT_STATE_RING_DEPTH × stride
  int32_t  ring_origin;                         // abs row mapped to slot 0
  int8_t   d_level[IDWT_STATE_RING_DEPTH];     // 0=raw, 1=step1, 2=step2, -1=unused

  // ── cursors ───────────────────────────────────────────────────────────────
  int32_t next_out;    // next output row (v0 ≤ next_out < v1)
  int32_t next_fetch;  // next real row to fetch from source (v0 ≤ next_fetch ≤ v1)

  // ── source ────────────────────────────────────────────────────────────────
  idwt_row_src_fn get_src_row;
  void           *src_ctx;
};

// Initialise (allocates ring_buf, top_pse_buf, bot_pse_buf).
void idwt_2d_state_init(idwt_2d_state *s,
                        int32_t u0, int32_t u1, int32_t v0, int32_t v1,
                        uint8_t transformation,
                        idwt_row_src_fn src_fn, void *src_ctx);

// Free buffers allocated by idwt_2d_state_init.
void idwt_2d_state_free(idwt_2d_state *s);

// Pull the next output row into out[0..u1-u0-1].
// Returns true while rows remain; false when all v1-v0 rows have been produced.
bool idwt_2d_state_pull_row(idwt_2d_state *s, sprec_t *out);

// ─────────────────────────────────────────────────────────────────────────────
// Streaming 2D FDWT — consumes one input row per push_row() call.
//
// The caller pushes image rows one at a time (v0..v1-1).  When vertical
// lifting completes a row, horizontal analysis is applied and the result is
// delivered to the sink callback (is_hp=false → LP/LL+HL row; true → HP/LH+HH
// row).  The sink receives a pointer to an interleaved row that is valid only
// for the duration of the callback.
//
// Call fdwt_2d_state_flush() after the last push_row() to drain any remaining
// rows that depend on the bottom-PSE extension.
// ─────────────────────────────────────────────────────────────────────────────

// Callback: delivers a completed row after both V and H FDWT.
// is_hp=false → LP row (even abs_phys_row), is_hp=true → HP row (odd).
// interleaved_row: u1-u0 samples at stride alignment; valid only in callback.
typedef void (*fdwt_row_sink_fn)(void *ctx, bool is_hp, int32_t abs_phys_row,
                                 const sprec_t *interleaved_row);

constexpr int32_t FDWT_STATE_RING_DEPTH = 12;

struct fdwt_2d_state {
  // ── geometry ──────────────────────────────────────────────────────────────
  int32_t u0, u1, v0, v1;
  int32_t stride;            // round_up(u1-u0, SIMD_PADDING)
  int32_t horiz_left;        // horizontal-DWT left PSE length
  int32_t horiz_right;       // horizontal-DWT right PSE length
  uint8_t transformation;    // 0 = irrev 9/7, 1 = rev 5/3
  int8_t  top_pse;           // PSE rows above v0
  int8_t  bottom_pse;        // PSE rows below v1-1

  // ── PSE scratch ───────────────────────────────────────────────────────────
  sprec_t *top_pse_buf;      // top_pse    × stride sprec_t
  sprec_t *bot_pse_buf;      // bottom_pse × stride sprec_t
  int8_t   top_dlevel[4];    // d_level per top-PSE slot (-1 = unfilled)
  int8_t   bot_dlevel[4];    // d_level per bot-PSE slot (-1 = unfilled)

  // ── sliding ring ──────────────────────────────────────────────────────────
  sprec_t *ring_buf;                          // FDWT_STATE_RING_DEPTH × stride
  int32_t  ring_origin;
  int8_t   d_level[FDWT_STATE_RING_DEPTH];   // 0=raw, 1=step1, 2=step2, -1=unused

  // ── horizontal-DWT temp buffer ────────────────────────────────────────────
  // Size: horiz_left + stride + horiz_right + SIMD_PADDING
  sprec_t *horiz_tmp;

  // ── cursors ───────────────────────────────────────────────────────────────
  int32_t next_in;    // next row to accept via push_row() [v0, v1]
  int32_t next_emit;  // next completed row waiting to be emitted [v0, v1)

  // ── sink ──────────────────────────────────────────────────────────────────
  fdwt_row_sink_fn put_row;
  void            *sink_ctx;
};

// Initialise (allocates ring_buf, PSE buffers, horiz_tmp).
void fdwt_2d_state_init(fdwt_2d_state *s,
                        int32_t u0, int32_t u1, int32_t v0, int32_t v1,
                        uint8_t transformation,
                        fdwt_row_sink_fn sink_fn, void *sink_ctx);

// Free buffers allocated by fdwt_2d_state_init.
void fdwt_2d_state_free(fdwt_2d_state *s);

// Push one input row in[0..u1-u0-1].  May trigger sink callbacks.
void fdwt_2d_state_push_row(fdwt_2d_state *s, const sprec_t *in);

// Finalise: fill bottom PSE, run remaining cascade, emit all pending rows.
// Must be called after the last push_row().
void fdwt_2d_state_flush(fdwt_2d_state *s);