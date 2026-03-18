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
                                          const int32_t, const int32_t stride);
// define pointer to IDWT functions
typedef void (*idwt_1d_filtd_func_fixed)(sprec_t *, const int32_t, const int32_t, const int32_t);
typedef void (*idwt_ver_filtd_func_fixed)(sprec_t *, const int32_t, const int32_t, const int32_t,
                                          const int32_t, const int32_t stride);

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
                                  int32_t stride);
void fdwt_rev_ver_sr_fixed_neon(sprec_t *in, int32_t u0, int32_t u1, int32_t v0, int32_t v1,
                                int32_t stride);

#elif defined(OPENHTJ2K_ENABLE_AVX2)
void fdwt_1d_filtr_irrev97_fixed_avx2(sprec_t *X, const int32_t left, const int32_t u_i0,
                                      const int32_t u_i1);
void fdwt_1d_filtr_rev53_fixed_avx2(sprec_t *X, const int32_t left, const int32_t u_i0, const int32_t u_i1);
void fdwt_irrev_ver_sr_fixed_avx2(sprec_t *in, const int32_t u0, const int32_t u1, const int32_t v0,
                                  const int32_t v1, const int32_t stride);
void fdwt_rev_ver_sr_fixed_avx2(sprec_t *in, const int32_t u0, const int32_t u1, const int32_t v0,
                                const int32_t v1, const int32_t stride);
#else
void fdwt_1d_filtr_irrev97_fixed(sprec_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
void fdwt_1d_filtr_rev53_fixed(sprec_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
void fdwt_irrev_ver_sr_fixed(sprec_t *in, int32_t u0, int32_t u1, int32_t v0, int32_t v1, int32_t stride);
void fdwt_rev_ver_sr_fixed(sprec_t *in, int32_t u0, int32_t u1, int32_t v0, int32_t v1, int32_t stride);
#endif

void fdwt_2d_sr_fixed(sprec_t *previousLL, sprec_t *LL, sprec_t *HL, sprec_t *LH, sprec_t *HH, int32_t u0,
                      int32_t u1, int32_t v0, int32_t v1, uint8_t transformation);

// ---------------------------------------------------------------------------
// Line-based stateful forward DWT.
// Accepts image rows one at a time, maintaining internal state across calls.
// Rows are stored in a compact flat extended buffer as they arrive.  The
// vertical DWT is applied via the cache-friendly column-strip algorithm and
// the horizontal DWT runs row-by-row immediately after, so each row is
// processed while it is still cache-hot from the vertical pass.
// ---------------------------------------------------------------------------
struct fdwt_line_state_t {
  int32_t u0, u1, v0, v1;    // tile coordinate bounds
  int32_t width;              // u1 - u0
  int32_t height;             // v1 - v0
  uint8_t transformation;     // 0 = 9/7 irreversible, 1 = 5/3 reversible

  int32_t top;                // PSE extension rows at the top
  int32_t bottom;             // PSE extension rows at the bottom
  int32_t total_rows;         // top + height + bottom

  // Flat 2D extended row buffer: row r at buf[r * buf_stride .. +width-1].
  // Rows 0..top-1          : PSE-top (filled as tile rows arrive).
  // Rows top..top+height-1 : tile rows after horizontal DWT.
  // Rows top+height..total_rows-1 : PSE-bottom (filled in finalize).
  int32_t buf_stride;         // round_up(width, SIMD_PADDING)
  sprec_t *buf;               // total_rows * buf_stride samples

  int32_t rows_received;      // tile rows pushed so far (0-indexed)

  // Workspace for the per-row horizontal 1D DWT.
  int32_t h_left;             // PSE samples on the left
  int32_t h_right;            // PSE samples on the right
  sprec_t *h_work;            // (width + h_left + h_right + SIMD_PADDING) samples
};

// Initialise a line-based forward-DWT state for one tile level.
void fdwt_line_state_init(fdwt_line_state_t *state, int32_t u0, int32_t u1, int32_t v0, int32_t v1,
                          uint8_t transformation);

// Push one tile row (width = u1-u0 samples at in_stride spacing from row_ptr).
// Stores the raw row in the internal flat buffer.  Must be called height times
// before fdwt_line_state_finalize.
void fdwt_line_state_push_row(fdwt_line_state_t *state, const sprec_t *row_ptr, int32_t in_stride);

// Finish the transform: apply column-strip vertical DWT, then horizontal DWT
// row-by-row (each row is processed while cache-hot from the vertical pass),
// then deinterleave into the four output subband buffers.
// The subband strides are round_up(subband_width,32).
void fdwt_line_state_finalize(fdwt_line_state_t *state, sprec_t *LL, sprec_t *HL, sprec_t *LH,
                              sprec_t *HH);

// Release all resources owned by the state.
void fdwt_line_state_free(fdwt_line_state_t *state);

// IDWT
#if defined(OPENHTJ2K_ENABLE_ARM_NEON)
void idwt_1d_filtr_rev53_fixed_neon(sprec_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
void idwt_1d_filtr_irrev97_fixed_neon(sprec_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
void idwt_irrev_ver_sr_fixed_neon(sprec_t *in, int32_t u0, int32_t u1, int32_t v0, int32_t v1,
                                  int32_t stride);
void idwt_rev_ver_sr_fixed_neon(sprec_t *in, int32_t u0, int32_t u1, int32_t v0, int32_t v1,
                                int32_t stride);
#elif defined(OPENHTJ2K_ENABLE_AVX2)
void idwt_1d_filtr_rev53_fixed_avx2(sprec_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
void idwt_1d_filtr_irrev97_fixed_avx2(sprec_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
void idwt_irrev_ver_sr_fixed_avx2(sprec_t *in, int32_t u0, int32_t u1, int32_t v0, int32_t v1,
                                  int32_t stride);
void idwt_rev_ver_sr_fixed_avx2(sprec_t *in, int32_t u0, int32_t u1, int32_t v0, int32_t v1,
                                int32_t stride);
#else
void idwt_1d_filtr_rev53_fixed(sprec_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
void idwt_1d_filtr_irrev97_fixed(sprec_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
void idwt_irrev_ver_sr_fixed(sprec_t *in, int32_t u0, int32_t u1, int32_t v0, int32_t v1, int32_t stride);
void idwt_rev_ver_sr_fixed(sprec_t *in, int32_t u0, int32_t u1, int32_t v0, int32_t v1, int32_t stride);
#endif
void idwt_2d_sr_fixed(sprec_t *nextLL, sprec_t *LL, sprec_t *HL, sprec_t *LH, sprec_t *HH, int32_t u0,
                      int32_t u1, int32_t v0, int32_t v1, uint8_t transformation,
                      uint8_t normalizing_upshift);

// ---------------------------------------------------------------------------
// Line-based stateful inverse DWT.
// Accepts pairs of subband rows (LL+HL and LH+HH) one pair at a time.
// Horizontal IDWT is applied immediately (cache-hot); vertical IDWT uses the
// column-strip algorithm for improved cache locality.
// ---------------------------------------------------------------------------
struct idwt_line_state_t {
  int32_t u0, u1, v0, v1;
  int32_t width;
  int32_t height;
  uint8_t transformation;
  uint8_t normalizing_upshift;

  int32_t top;
  int32_t bottom;
  int32_t total_rows;

  // Flat 2D extended row buffer (same layout as fdwt_line_state_t).
  int32_t buf_stride;
  sprec_t *buf;

  int32_t rows_received;

  int32_t h_left;
  int32_t h_right;
  sprec_t *h_work;
};

void idwt_line_state_init(idwt_line_state_t *state, int32_t u0, int32_t u1, int32_t v0, int32_t v1,
                          uint8_t transformation, uint8_t normalizing_upshift);

// Push one interleaved row (already combined from subband pair by caller, or
// built internally from the LL/HL and LH/HH sub-rows).  Must be called height
// times before idwt_line_state_finalize.
void idwt_line_state_push_row(idwt_line_state_t *state, const sprec_t *row_ptr, int32_t in_stride);

// Finish: fill PSE-bottom rows, apply column-strip vertical IDWT, apply the
// normalising up-shift for the irreversible path, and write to nextLL.
void idwt_line_state_finalize(idwt_line_state_t *state, sprec_t *nextLL);

void idwt_line_state_free(idwt_line_state_t *state);