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

#include <algorithm>
#include <cstring>
#include "dwt.hpp"
#include "utils.hpp"

// Number of columns processed per strip in the vertical lifting pass.
// A smaller value keeps the working set in L1 cache; 64 balances overhead.
static constexpr int32_t DWT_VERT_STRIP = 64;

// ---------------------------------------------------------------------------
// 1-D filter helpers (unchanged from original)
// ---------------------------------------------------------------------------

// irreversible FDWT
void fdwt_1d_filtr_irrev97_fixed(sprec_t *X, const int32_t left, const int32_t u_i0,
                                  const int32_t u_i1) {
  const int32_t i0    = static_cast<int32_t>(u_i0);
  const int32_t i1    = static_cast<int32_t>(u_i1);
  const int32_t start = ceil_int(i0, 2);
  const int32_t stop  = ceil_int(i1, 2);

  const int32_t offset = left + i0 % 2;
  for (int32_t n = -4 + offset, i = start - 2; i < stop + 1; i++, n += 2) {
    int32_t sum = X[n] + X[n + 2];
    X[n + 1]    = static_cast<sprec_t>(X[n + 1] + ((Acoeff * sum + Aoffset) >> Ashift));
  }
  for (int32_t n = -2 + offset, i = start - 1; i < stop + 1; i++, n += 2) {
    int32_t sum = X[n - 1] + X[n + 1];
    X[n]        = static_cast<sprec_t>(X[n] + ((Bcoeff * sum + Boffset) >> Bshift));
  }
  for (int32_t n = -2 + offset, i = start - 1; i < stop; i++, n += 2) {
    int32_t sum = X[n] + X[n + 2];
    X[n + 1]    = static_cast<sprec_t>(X[n + 1] + ((Ccoeff * sum + Coffset) >> Cshift));
  }
  for (int32_t n = 0 + offset, i = start; i < stop; i++, n += 2) {
    int32_t sum = X[n - 1] + X[n + 1];
    X[n]        = static_cast<sprec_t>(X[n] + ((Dcoeff * sum + Doffset) >> Dshift));
  }
}

// reversible FDWT
void fdwt_1d_filtr_rev53_fixed(sprec_t *X, const int32_t left, const int32_t u_i0,
                                const int32_t u_i1) {
  const int32_t i0     = static_cast<int32_t>(u_i0);
  const int32_t i1     = static_cast<int32_t>(u_i1);
  const int32_t start  = ceil_int(i0, 2);
  const int32_t stop   = ceil_int(i1, 2);
  const int32_t offset = left + i0 % 2;
  for (int32_t n = -2 + offset, i = start - 1; i < stop; ++i, n += 2) {
    int32_t sum = X[n] + X[n + 2];
    X[n + 1]    = static_cast<sprec_t>(X[n + 1] - (sum >> 1));
  }
  for (int32_t n = 0 + offset, i = start; i < stop; ++i, n += 2) {
    int32_t sum = X[n - 1] + X[n + 1];
    X[n]        = static_cast<sprec_t>(X[n] + ((sum + 2) >> 2));
  }
}

// 1-dimensional FDWT applied in-place to a single row
static inline void fdwt_1d_sr_fixed(sprec_t *buf, sprec_t *in, const int32_t left, const int32_t right,
                                    const int32_t i0, const int32_t i1, const uint8_t transformation) {
  dwt_1d_extr_fixed(buf, in, left, right, i0, i1);
  if (transformation == 0)
    fdwt_1d_filtr_irrev97_fixed(buf, left, i0, i1);
  else
    fdwt_1d_filtr_rev53_fixed(buf, left, i0, i1);
  memcpy(in, buf + left, sizeof(sprec_t) * static_cast<size_t>(i1 - i0));
}

// ---------------------------------------------------------------------------
// Horizontal FDWT – processes one row at a time (already cache-friendly)
// ---------------------------------------------------------------------------
static void fdwt_hor_sr_fixed(sprec_t *in, const int32_t u0, const int32_t u1, const int32_t v0,
                               const int32_t v1, const uint8_t transformation, const int32_t stride) {
  constexpr int32_t num_pse_i0[2][2] = {{4, 2}, {3, 1}};
  constexpr int32_t num_pse_i1[2][2] = {{3, 1}, {4, 2}};
  const int32_t left                 = num_pse_i0[u0 % 2][transformation];
  const int32_t right                = num_pse_i1[u1 % 2][transformation];

  if (u0 == u1 - 1) {
    for (int32_t row = 0; row < v1 - v0; ++row) {
      if (u0 % 2 == 0) {
        in[row * stride] = in[row * stride];
      } else {
        in[row * stride] =
            transformation ? static_cast<sprec_t>(in[row * stride] << 1) : in[row * stride];
      }
    }
  } else {
    const int32_t len = u1 - u0 + left + right;
    auto *Xext        = static_cast<sprec_t *>(aligned_mem_alloc(
        sizeof(sprec_t) * static_cast<size_t>(round_up(len + SIMD_PADDING, SIMD_PADDING)), 32));
    for (int32_t row = 0; row < v1 - v0; ++row) {
      fdwt_1d_sr_fixed(Xext, in, left, right, u0, u1, transformation);
      in += stride;
    }
    aligned_mem_free(Xext);
  }
}

// ---------------------------------------------------------------------------
// Vertical FDWT – column-strip (line-based) implementation.
//
// The lifting steps are driven by a compact flat local buffer of width
// DWT_VERT_STRIP samples.  For each column strip the relevant rows are
// copied into a contiguous local array, all lifting passes are applied,
// and the results are written back.  This keeps the working set small
// enough to reside in L1/L2 cache regardless of tile height, reducing
// page-fault pressure and TLB thrashing on large tiles.
// ---------------------------------------------------------------------------

// Apply the irreversible (9/7) vertical lifting steps to an already-built
// extended flat buffer.  buf_stride is the row stride of extbuf.
static void fdwt_irrev97_ver_lifting(sprec_t *extbuf, const int32_t buf_stride,
                                     const int32_t total_rows, const int32_t width, const int32_t top,
                                     const int32_t v0, const int32_t v1) {
  const int32_t start  = ceil_int(v0, 2);
  const int32_t stop   = ceil_int(v1, 2);
  const int32_t offset = top + v0 % 2;

  const int32_t ls = round_up(std::min(DWT_VERT_STRIP, width), static_cast<int32_t>(SIMD_PADDING));
  auto *strip =
      static_cast<sprec_t *>(aligned_mem_alloc(sizeof(sprec_t) * static_cast<size_t>(total_rows * ls), 32));

  for (int32_t cs = 0; cs < width; cs += DWT_VERT_STRIP) {
    const int32_t ce = (cs + DWT_VERT_STRIP < width) ? cs + DWT_VERT_STRIP : width;
    const int32_t sw = ce - cs;

    // Copy column strip from extbuf into compact local buffer.
    for (int32_t r = 0; r < total_rows; ++r)
      memcpy(strip + r * ls, extbuf + r * buf_stride + cs, sizeof(sprec_t) * static_cast<size_t>(sw));

    // Step A: update odd rows using even neighbours
    for (int32_t n = -4 + offset, i = start - 2; i < stop + 1; i++, n += 2) {
      sprec_t *rn  = strip + n * ls;
      sprec_t *rn1 = strip + (n + 1) * ls;
      sprec_t *rn2 = strip + (n + 2) * ls;
      for (int32_t c = 0; c < sw; ++c) {
        int32_t sum = rn[c] + rn2[c];
        rn1[c]      = static_cast<sprec_t>(rn1[c] + ((Acoeff * sum + Aoffset) >> Ashift));
      }
    }
    // Step B: update even rows using odd neighbours
    for (int32_t n = -2 + offset, i = start - 1; i < stop + 1; i++, n += 2) {
      sprec_t *rm1 = strip + (n - 1) * ls;
      sprec_t *rn  = strip + n * ls;
      sprec_t *rp1 = strip + (n + 1) * ls;
      for (int32_t c = 0; c < sw; ++c) {
        int32_t sum = rm1[c] + rp1[c];
        rn[c]       = static_cast<sprec_t>(rn[c] + ((Bcoeff * sum + Boffset) >> Bshift));
      }
    }
    // Step C: update odd rows using even neighbours
    for (int32_t n = -2 + offset, i = start - 1; i < stop; i++, n += 2) {
      sprec_t *rn  = strip + n * ls;
      sprec_t *rn1 = strip + (n + 1) * ls;
      sprec_t *rn2 = strip + (n + 2) * ls;
      for (int32_t c = 0; c < sw; ++c) {
        int32_t sum = rn[c] + rn2[c];
        rn1[c]      = static_cast<sprec_t>(rn1[c] + ((Ccoeff * sum + Coffset) >> Cshift));
      }
    }
    // Step D: update even rows using odd neighbours
    for (int32_t n = 0 + offset, i = start; i < stop; i++, n += 2) {
      sprec_t *rm1 = strip + (n - 1) * ls;
      sprec_t *rn  = strip + n * ls;
      sprec_t *rp1 = strip + (n + 1) * ls;
      for (int32_t c = 0; c < sw; ++c) {
        int32_t sum = rm1[c] + rp1[c];
        rn[c]       = static_cast<sprec_t>(rn[c] + ((Dcoeff * sum + Doffset) >> Dshift));
      }
    }

    // Write strip back.
    for (int32_t r = 0; r < total_rows; ++r)
      memcpy(extbuf + r * buf_stride + cs, strip + r * ls, sizeof(sprec_t) * static_cast<size_t>(sw));
  }

  aligned_mem_free(strip);
}

// Apply the reversible (5/3) vertical lifting steps.
static void fdwt_rev53_ver_lifting(sprec_t *extbuf, const int32_t buf_stride, const int32_t total_rows,
                                   const int32_t width, const int32_t top, const int32_t v0,
                                   const int32_t v1) {
  const int32_t start  = ceil_int(v0, 2);
  const int32_t stop   = ceil_int(v1, 2);
  const int32_t offset = top + v0 % 2;

  const int32_t ls = round_up(std::min(DWT_VERT_STRIP, width), static_cast<int32_t>(SIMD_PADDING));
  auto *strip =
      static_cast<sprec_t *>(aligned_mem_alloc(sizeof(sprec_t) * static_cast<size_t>(total_rows * ls), 32));

  for (int32_t cs = 0; cs < width; cs += DWT_VERT_STRIP) {
    const int32_t ce = (cs + DWT_VERT_STRIP < width) ? cs + DWT_VERT_STRIP : width;
    const int32_t sw = ce - cs;

    for (int32_t r = 0; r < total_rows; ++r)
      memcpy(strip + r * ls, extbuf + r * buf_stride + cs, sizeof(sprec_t) * static_cast<size_t>(sw));

    // Step 1: update odd rows (predict)
    for (int32_t n = -2 + offset, i = start - 1; i < stop; ++i, n += 2) {
      sprec_t *rn  = strip + n * ls;
      sprec_t *rn1 = strip + (n + 1) * ls;
      sprec_t *rn2 = strip + (n + 2) * ls;
      for (int32_t c = 0; c < sw; ++c)
        rn1[c] = static_cast<sprec_t>(rn1[c] - ((rn[c] + rn2[c]) >> 1));
    }
    // Step 2: update even rows (update)
    for (int32_t n = 0 + offset, i = start; i < stop; ++i, n += 2) {
      sprec_t *rm1 = strip + (n - 1) * ls;
      sprec_t *rn  = strip + n * ls;
      sprec_t *rp1 = strip + (n + 1) * ls;
      for (int32_t c = 0; c < sw; ++c)
        rn[c] = static_cast<sprec_t>(rn[c] + ((rm1[c] + rp1[c] + 2) >> 2));
    }

    for (int32_t r = 0; r < total_rows; ++r)
      memcpy(extbuf + r * buf_stride + cs, strip + r * ls, sizeof(sprec_t) * static_cast<size_t>(sw));
  }

  aligned_mem_free(strip);
}

// Build the extended buffer (PSE-top + tile rows + PSE-bottom), run the
// column-strip lifting in-place, and copy the tile rows back.
void fdwt_irrev_ver_sr_fixed(sprec_t *in, const int32_t u0, const int32_t u1, const int32_t v0,
                              const int32_t v1, const int32_t stride) {
  constexpr int32_t num_pse_i0[2] = {4, 3};
  constexpr int32_t num_pse_i1[2] = {3, 4};
  const int32_t top               = num_pse_i0[v0 % 2];
  const int32_t bottom            = num_pse_i1[v1 % 2];
  const int32_t height            = v1 - v0;
  const int32_t width             = u1 - u0;

  if (v0 == v1 - 1) {
    // Single-sample height: no-op for the 9/7 filter.
    return;
  }

  const int32_t total_rows = top + height + bottom;
  const int32_t bs         = round_up(width, static_cast<int32_t>(SIMD_PADDING));
  auto *extbuf             = static_cast<sprec_t *>(
      aligned_mem_alloc(sizeof(sprec_t) * static_cast<size_t>(total_rows * bs), 32));

  for (int32_t i = 1; i <= top; ++i)
    memcpy(extbuf + (top - i) * bs, &in[PSEo(v0 - i, v0, v1) * stride],
           sizeof(sprec_t) * static_cast<size_t>(width));
  for (int32_t row = 0; row < height; ++row)
    memcpy(extbuf + (top + row) * bs, &in[row * stride], sizeof(sprec_t) * static_cast<size_t>(width));
  for (int32_t i = 1; i <= bottom; ++i)
    memcpy(extbuf + (top + height + i - 1) * bs, &in[PSEo(v1 - v0 + i - 1 + v0, v0, v1) * stride],
           sizeof(sprec_t) * static_cast<size_t>(width));

  fdwt_irrev97_ver_lifting(extbuf, bs, total_rows, width, top, v0, v1);

  for (int32_t row = 0; row < height; ++row)
    memcpy(&in[row * stride], extbuf + (top + row) * bs, sizeof(sprec_t) * static_cast<size_t>(width));

  aligned_mem_free(extbuf);
}

void fdwt_rev_ver_sr_fixed(sprec_t *in, const int32_t u0, const int32_t u1, const int32_t v0,
                            const int32_t v1, const int32_t stride) {
  constexpr int32_t num_pse_i0[2] = {2, 1};
  constexpr int32_t num_pse_i1[2] = {1, 2};
  const int32_t top               = num_pse_i0[v0 % 2];
  const int32_t bottom            = num_pse_i1[v1 % 2];
  const int32_t height            = v1 - v0;
  const int32_t width             = u1 - u0;

  if (v0 == v1 - 1) {
    for (int32_t col = 0; col < width; ++col) {
      if (v0 % 2) in[col] = static_cast<sprec_t>(in[col] << 1);
    }
    return;
  }

  const int32_t total_rows = top + height + bottom;
  const int32_t bs         = round_up(width, static_cast<int32_t>(SIMD_PADDING));
  auto *extbuf             = static_cast<sprec_t *>(
      aligned_mem_alloc(sizeof(sprec_t) * static_cast<size_t>(total_rows * bs), 32));

  for (int32_t i = 1; i <= top; ++i)
    memcpy(extbuf + (top - i) * bs, &in[PSEo(v0 - i, v0, v1) * stride],
           sizeof(sprec_t) * static_cast<size_t>(width));
  for (int32_t row = 0; row < height; ++row)
    memcpy(extbuf + (top + row) * bs, &in[row * stride], sizeof(sprec_t) * static_cast<size_t>(width));
  for (int32_t i = 1; i <= bottom; ++i)
    memcpy(extbuf + (top + height + i - 1) * bs, &in[PSEo(v1 - v0 + i - 1 + v0, v0, v1) * stride],
           sizeof(sprec_t) * static_cast<size_t>(width));

  fdwt_rev53_ver_lifting(extbuf, bs, total_rows, width, top, v0, v1);

  for (int32_t row = 0; row < height; ++row)
    memcpy(&in[row * stride], extbuf + (top + row) * bs, sizeof(sprec_t) * static_cast<size_t>(width));

  aligned_mem_free(extbuf);
}

// ---------------------------------------------------------------------------
// Deinterleave the 2-D interleaved buffer into the four subband arrays.
// (Kept identical to the original implementation.)
// ---------------------------------------------------------------------------
static void fdwt_2d_deinterleave_fixed(sprec_t *buf, sprec_t *const LL, sprec_t *const HL,
                                       sprec_t *const LH, sprec_t *const HH, const int32_t u0,
                                       const int32_t u1, const int32_t v0, const int32_t v1,
                                       const int32_t stride) {
  const int32_t v_offset   = v0 % 2;
  const int32_t u_offset   = u0 % 2;
  sprec_t *dp[4]           = {LL, HL, LH, HH};
  const int32_t vstart[4]  = {ceil_int(v0, 2), ceil_int(v0, 2), v0 / 2, v0 / 2};
  const int32_t vstop[4]   = {ceil_int(v1, 2), ceil_int(v1, 2), v1 / 2, v1 / 2};
  const int32_t ustart[4]  = {ceil_int(u0, 2), u0 / 2, ceil_int(u0, 2), u0 / 2};
  const int32_t ustop[4]   = {ceil_int(u1, 2), u1 / 2, ceil_int(u1, 2), u1 / 2};
  const int32_t voffset[4] = {v_offset, v_offset, 1 - v_offset, 1 - v_offset};
  const int32_t uoffset[4] = {u_offset, 1 - u_offset, u_offset, 1 - u_offset};
  const int32_t stride2[4] = {round_up(ustop[0] - ustart[0], 32), round_up(ustop[1] - ustart[1], 32),
                               round_up(ustop[2] - ustart[2], 32), round_up(ustop[3] - ustart[3], 32)};

#if defined(OPENHTJ2K_ENABLE_ARM_NEON)
  if ((ustop[0] - ustart[0]) != (ustop[1] - ustart[1])) {
    for (uint8_t b = 0; b < 2; ++b) {
      for (int32_t v = 0, vb = vstart[b]; vb < vstop[b]; ++vb, ++v) {
        sprec_t *line = dp[b] + v * stride2[b];
        for (int32_t u = 0, ub = ustart[b]; ub < ustop[b]; ++ub, ++u)
          *(line++) = buf[2 * u + uoffset[b] + (2 * v + voffset[b]) * stride];
      }
    }
  } else {
    sprec_t *first = dp[0], *second = dp[1];
    if (uoffset[0] > uoffset[1]) { first = dp[1]; second = dp[0]; }
    for (int32_t v = 0, vb = vstart[0]; vb < vstop[0]; ++vb, ++v) {
      sprec_t *sp    = buf + (2 * v + voffset[0]) * stride;
      size_t len     = static_cast<size_t>(ustop[0] - ustart[0]);
      sprec_t *line0 = first + v * stride2[0];
      sprec_t *line1 = second + v * stride2[0];
      for (; len >= 8; len -= 8) {
        auto vline = vld2q_s16(sp);
        vst1q_s16(line0, vline.val[0]); vst1q_s16(line1, vline.val[1]);
        line0 += 8; line1 += 8; sp += 16;
      }
      for (; len > 0; --len) { *line0++ = *sp++; *line1++ = *sp++; }
    }
  }
  if ((ustop[2] - ustart[2]) != (ustop[3] - ustart[3])) {
    for (uint8_t b = 2; b < 4; ++b) {
      for (int32_t v = 0, vb = vstart[b]; vb < vstop[b]; ++vb, ++v) {
        sprec_t *line = dp[b] + v * stride2[b];
        for (int32_t u = 0, ub = ustart[b]; ub < ustop[b]; ++ub, ++u)
          *(line++) = buf[2 * u + uoffset[b] + (2 * v + voffset[b]) * stride];
      }
    }
  } else {
    sprec_t *first = dp[2], *second = dp[3];
    if (uoffset[2] > uoffset[3]) { first = dp[3]; second = dp[2]; }
    for (int32_t v = 0, vb = vstart[2]; vb < vstop[2]; ++vb, ++v) {
      sprec_t *sp    = buf + (2 * v + voffset[2]) * stride;
      size_t len     = static_cast<size_t>(ustop[2] - ustart[2]);
      sprec_t *line0 = first + v * stride2[2];
      sprec_t *line1 = second + v * stride2[2];
      for (; len >= 8; len -= 8) {
        auto vline = vld2q_s16(sp);
        vst1q_s16(line0, vline.val[0]); vst1q_s16(line1, vline.val[1]);
        line0 += 8; line1 += 8; sp += 16;
      }
      for (; len > 0; --len) { *line0++ = *sp++; *line1++ = *sp++; }
    }
  }
#elif defined(OPENHTJ2K_TRY_AVX2) && defined(__AVX2__)
  const __m256i vshmask = _mm256_set_epi8(15, 14, 11, 10, 7, 6, 3, 2, 13, 12, 9, 8, 5, 4, 1, 0,
                                          15, 14, 11, 10, 7, 6, 3, 2, 13, 12, 9, 8, 5, 4, 1, 0);
  for (int32_t pair = 0; pair < 2; ++pair) {
    uint8_t b0 = static_cast<uint8_t>(pair * 2), b1 = static_cast<uint8_t>(pair * 2 + 1);
    if ((ustop[b0] - ustart[b0]) != (ustop[b1] - ustart[b1])) {
      for (uint8_t b = b0; b <= b1; ++b) {
        for (int32_t v = 0, vb = vstart[b]; vb < vstop[b]; ++vb, ++v) {
          sprec_t *line = dp[b] + v * stride2[b];
          for (int32_t u = 0, ub = ustart[b]; ub < ustop[b]; ++ub, ++u)
            *(line++) = buf[2 * u + uoffset[b] + (2 * v + voffset[b]) * stride];
        }
      }
    } else {
      sprec_t *first = dp[b0], *second = dp[b1];
      if (uoffset[b0] > uoffset[b1]) { first = dp[b1]; second = dp[b0]; }
      for (int32_t v = 0, vb = vstart[b0]; vb < vstop[b0]; ++vb, ++v) {
        sprec_t *sp    = buf + (2 * v + voffset[b0]) * stride;
        size_t len     = static_cast<size_t>(ustop[b0] - ustart[b0]);
        sprec_t *line0 = first + v * stride2[b0];
        sprec_t *line1 = second + v * stride2[b0];
        for (; len >= 8; len -= 8) {
          __m256i vline = _mm256_loadu_si256((__m256i *)sp);
          vline         = _mm256_shuffle_epi8(vline, vshmask);
          vline         = _mm256_permute4x64_epi64(vline, 0xD8);
          _mm256_storeu2_m128i((__m128i *)line1, (__m128i *)line0, vline);
          line0 += 8; line1 += 8; sp += 16;
        }
        for (; len > 0; --len) { *line0++ = *sp++; *line1++ = *sp++; }
      }
    }
  }
#else
  for (uint8_t b = 0; b < 4; ++b) {
    for (int32_t v = 0, vb = vstart[b]; vb < vstop[b]; ++vb, ++v) {
      sprec_t *line = dp[b] + v * stride2[b];
      for (int32_t u = 0, ub = ustart[b]; ub < ustop[b]; ++ub, ++u)
        *(line++) = buf[2 * u + uoffset[b] + (2 * v + voffset[b]) * stride];
    }
  }
#endif
}

// ---------------------------------------------------------------------------
// 2-D FDWT
//
// Preserves the original vertical-first, horizontal-second ordering required
// for exact lossless (integer 5/3) round-trip recovery with the existing
// idwt_2d_sr_fixed (which applies H^-1 then V^-1).  The vertical pass now
// uses the cache-friendly column-strip lifting helper above instead of the
// original scattered row-pointer allocation.
// ---------------------------------------------------------------------------
void fdwt_2d_sr_fixed(sprec_t *previousLL, sprec_t *LL, sprec_t *HL, sprec_t *LH, sprec_t *HH,
                      const int32_t u0, const int32_t u1, const int32_t v0, const int32_t v1,
                      const uint8_t transformation) {
  const int32_t stride = round_up(u1 - u0, 32);
  sprec_t *src         = previousLL;

  // Step 1: apply vertical DWT via the cache-friendly column-strip path.
  if (transformation == 0)
    fdwt_irrev_ver_sr_fixed(src, u0, u1, v0, v1, stride);
  else
    fdwt_rev_ver_sr_fixed(src, u0, u1, v0, v1, stride);

  // Step 2: apply horizontal DWT row-by-row.
  fdwt_hor_sr_fixed(src, u0, u1, v0, v1, transformation, stride);

  // Step 3: separate the interleaved result into the four subband buffers.
  fdwt_2d_deinterleave_fixed(src, LL, HL, LH, HH, u0, u1, v0, v1, stride);
}

// ---------------------------------------------------------------------------
// Line-based stateful forward DWT implementation
// ---------------------------------------------------------------------------

void fdwt_line_state_init(fdwt_line_state_t *state, const int32_t u0, const int32_t u1,
                          const int32_t v0, const int32_t v1, const uint8_t transformation) {
  state->u0             = u0;
  state->u1             = u1;
  state->v0             = v0;
  state->v1             = v1;
  state->width          = u1 - u0;
  state->height         = v1 - v0;
  state->transformation = transformation;
  state->rows_received  = 0;

  // Vertical PSE extension sizes (must match fdwt_irrev/rev_ver_sr_fixed).
  constexpr int32_t ver_pse_i0_irrev[2] = {4, 3};
  constexpr int32_t ver_pse_i1_irrev[2] = {3, 4};
  constexpr int32_t ver_pse_i0_rev[2]   = {2, 1};
  constexpr int32_t ver_pse_i1_rev[2]   = {1, 2};
  if (transformation == 0) {
    state->top    = ver_pse_i0_irrev[v0 % 2];
    state->bottom = ver_pse_i1_irrev[v1 % 2];
  } else {
    state->top    = ver_pse_i0_rev[v0 % 2];
    state->bottom = ver_pse_i1_rev[v1 % 2];
  }
  state->total_rows = state->top + state->height + state->bottom;
  state->buf_stride = round_up(state->width, static_cast<int32_t>(SIMD_PADDING));
  state->buf        = static_cast<sprec_t *>(aligned_mem_alloc(
      sizeof(sprec_t) * static_cast<size_t>(state->total_rows * state->buf_stride), 32));

  // Pre-allocate state for future use; horizontal DWT workspace managed by
  // fdwt_hor_sr_fixed internally.
  state->h_left  = 0;
  state->h_right = 0;
  state->h_work  = nullptr;
}

// Push one tile row.  The caller passes a pointer to the first sample and the
// stride (in samples) between columns – normally 1 for a packed row.
// The raw row is stored in the flat extended buffer; horizontal DWT will be
// applied in fdwt_line_state_finalize (after vertical DWT) to preserve the
// original vertical-first ordering required by idwt_2d_sr_fixed.
void fdwt_line_state_push_row(fdwt_line_state_t *state, const sprec_t *row_ptr,
                               const int32_t in_stride) {
  const int32_t r   = state->rows_received;
  const int32_t top = state->top;
  const int32_t w   = state->width;
  const int32_t bs  = state->buf_stride;
  const int32_t v0  = state->v0;
  const int32_t v1  = state->v1;

  // Write raw row into the flat extended buffer at the tile-row position.
  sprec_t *dst = state->buf + static_cast<size_t>((top + r) * bs);
  if (in_stride == 1) {
    memcpy(dst, row_ptr, sizeof(sprec_t) * static_cast<size_t>(w));
  } else {
    for (int32_t c = 0; c < w; ++c) dst[c] = row_ptr[c * in_stride];
  }

  // If this tile row also provides a PSE-top sample, copy the raw row there now.
  // PSE-top slot (top-i) holds tile row PSEo(v0-i, v0, v1) for i = 1..top.
  for (int32_t i = 1; i <= top; ++i) {
    if (PSEo(v0 - i, v0, v1) == r) {
      memcpy(state->buf + static_cast<size_t>((top - i) * bs), dst,
             sizeof(sprec_t) * static_cast<size_t>(w));
    }
  }

  state->rows_received = r + 1;
}

// Finish the forward DWT: fill PSE-bottom, apply column-strip vertical DWT,
// then horizontal DWT, then deinterleave into the four output subband buffers.
// This matches the vertical-first ordering of fdwt_2d_sr_fixed so that
// idwt_2d_sr_fixed recovers the original image exactly (including lossless
// 5/3 round-trips where integer rounding makes H∘V ≠ V∘H).
void fdwt_line_state_finalize(fdwt_line_state_t *state, sprec_t *LL, sprec_t *HL, sprec_t *LH,
                              sprec_t *HH) {
  const int32_t top    = state->top;
  const int32_t bottom = state->bottom;
  const int32_t height = state->height;
  const int32_t w      = state->width;
  const int32_t bs     = state->buf_stride;
  const int32_t v0     = state->v0;
  const int32_t v1     = state->v1;
  const int32_t u0     = state->u0;
  const int32_t u1     = state->u1;
  const uint8_t tr     = state->transformation;

  // Fill PSE-bottom by copying raw tile rows (same as fdwt_irrev/rev_ver_sr_fixed).
  for (int32_t i = 1; i <= bottom; ++i) {
    const int32_t src_row = PSEo(v1 - v0 + i - 1 + v0, v0, v1);
    memcpy(state->buf + static_cast<size_t>((top + height + i - 1) * bs),
           state->buf + static_cast<size_t>((top + src_row) * bs),
           sizeof(sprec_t) * static_cast<size_t>(w));
  }

  // Step 1: apply column-strip vertical DWT on the compact flat buffer.
  if (tr == 0)
    fdwt_irrev97_ver_lifting(state->buf, bs, state->total_rows, w, top, v0, v1);
  else
    fdwt_rev53_ver_lifting(state->buf, bs, state->total_rows, w, top, v0, v1);

  // Step 2: apply horizontal DWT to each tile row (now vertically transformed).
  // Each row is visited sequentially so the DWT runs while the row is cache-hot.
  sprec_t *tile_start = state->buf + static_cast<size_t>(top * bs);
  fdwt_hor_sr_fixed(tile_start, u0, u1, v0, v1, tr, bs);

  // Step 3: deinterleave into the four subband buffers.
  fdwt_2d_deinterleave_fixed(tile_start, LL, HL, LH, HH, u0, u1, v0, v1, bs);
}

void fdwt_line_state_free(fdwt_line_state_t *state) {
  aligned_mem_free(state->buf);
  aligned_mem_free(state->h_work);
  state->buf    = nullptr;
  state->h_work = nullptr;
}
