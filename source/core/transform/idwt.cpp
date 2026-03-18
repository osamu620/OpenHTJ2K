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
static constexpr int32_t DWT_VERT_STRIP = 64;

// ---------------------------------------------------------------------------
// 1-D filter helpers (unchanged from original)
// ---------------------------------------------------------------------------

void idwt_1d_filtr_irrev97_fixed(sprec_t *X, const int32_t left, const int32_t u_i0,
                                  const int32_t u_i1) {
  const int32_t i0     = static_cast<int32_t>(u_i0);
  const int32_t i1     = static_cast<int32_t>(u_i1);
  const int32_t start  = i0 / 2;
  const int32_t stop   = i1 / 2;
  const int32_t offset = left - i0 % 2;

  for (int32_t n = -2 + offset, i = start - 1; i < stop + 2; i++, n += 2) {
    int32_t sum = X[n - 1] + X[n + 1];
    X[n]        = static_cast<sprec_t>(X[n] - ((Dcoeff * sum + Doffset) >> Dshift));
  }
  for (int32_t n = -2 + offset, i = start - 1; i < stop + 1; i++, n += 2) {
    int32_t sum = X[n] + X[n + 2];
    X[n + 1]    = static_cast<sprec_t>(X[n + 1] - ((Ccoeff * sum + Coffset) >> Cshift));
  }
  for (int32_t n = 0 + offset, i = start; i < stop + 1; i++, n += 2) {
    int32_t sum = X[n - 1] + X[n + 1];
    X[n]        = static_cast<sprec_t>(X[n] - ((Bcoeff * sum + Boffset) >> Bshift));
  }
  for (int32_t n = 0 + offset, i = start; i < stop; i++, n += 2) {
    int32_t sum = X[n] + X[n + 2];
    X[n + 1]    = static_cast<sprec_t>(X[n + 1] - ((Acoeff * sum + Aoffset) >> Ashift));
  }
}

void idwt_1d_filtr_rev53_fixed(sprec_t *X, const int32_t left, const int32_t u_i0,
                                const int32_t u_i1) {
  const int32_t i0     = static_cast<int32_t>(u_i0);
  const int32_t i1     = static_cast<int32_t>(u_i1);
  const int32_t start  = i0 / 2;
  const int32_t stop   = i1 / 2;
  const int32_t offset = left - i0 % 2;

  for (int32_t n = 0 + offset, i = start; i < stop + 1; ++i, n += 2)
    X[n] = static_cast<sprec_t>(X[n] - ((X[n - 1] + X[n + 1] + 2) >> 2));
  for (int32_t n = 0 + offset, i = start; i < stop; ++i, n += 2)
    X[n + 1] = static_cast<sprec_t>(X[n + 1] + ((X[n] + X[n + 2]) >> 1));
}

static void idwt_1d_sr_fixed(sprec_t *buf, sprec_t *in, const int32_t left, const int32_t right,
                              const int32_t i0, const int32_t i1, const uint8_t transformation) {
  dwt_1d_extr_fixed(buf, in, left, right, i0, i1);
  if (transformation == 0)
    idwt_1d_filtr_irrev97_fixed(buf, left, i0, i1);
  else
    idwt_1d_filtr_rev53_fixed(buf, left, i0, i1);
  memcpy(in, buf + left, sizeof(sprec_t) * static_cast<size_t>(i1 - i0));
}

// ---------------------------------------------------------------------------
// Horizontal IDWT – processes one row at a time
// ---------------------------------------------------------------------------
static void idwt_hor_sr_fixed(sprec_t *in, const int32_t u0, const int32_t u1, const int32_t v0,
                               const int32_t v1, const uint8_t transformation, const int32_t stride) {
  constexpr int32_t num_pse_i0[2][2] = {{3, 1}, {4, 2}};
  constexpr int32_t num_pse_i1[2][2] = {{4, 2}, {3, 1}};
  const int32_t left                 = num_pse_i0[u0 % 2][transformation];
  const int32_t right                = num_pse_i1[u1 % 2][transformation];

  if (u0 == u1 - 1) {
    for (int32_t row = 0; row < v1 - v0; ++row) {
      if (u0 % 2 != 0 && transformation)
        in[row * stride] = static_cast<sprec_t>(in[row * stride] >> 1);
    }
  } else {
    const int32_t len = u1 - u0 + left + right;
    auto *Yext        = static_cast<sprec_t *>(aligned_mem_alloc(
        sizeof(sprec_t) * static_cast<size_t>(round_up(len + SIMD_PADDING, SIMD_PADDING)), 32));
    for (int32_t row = 0; row < v1 - v0; ++row) {
      idwt_1d_sr_fixed(Yext, in, left, right, u0, u1, transformation);
      in += stride;
    }
    aligned_mem_free(Yext);
  }
}

// ---------------------------------------------------------------------------
// Vertical IDWT – column-strip (line-based) implementation.
// Same cache-friendly design as the forward path: lifting passes are applied
// to a compact local buffer of DWT_VERT_STRIP columns to reduce page-fault
// pressure and TLB thrashing on large tiles.
// ---------------------------------------------------------------------------

static void idwt_irrev97_ver_lifting(sprec_t *extbuf, const int32_t buf_stride,
                                     const int32_t total_rows, const int32_t width, const int32_t top,
                                     const int32_t v0, const int32_t v1) {
  const int32_t start  = v0 / 2;
  const int32_t stop   = v1 / 2;
  const int32_t offset = top - v0 % 2;

  const int32_t ls = round_up(std::min(DWT_VERT_STRIP, width), static_cast<int32_t>(SIMD_PADDING));
  auto *strip =
      static_cast<sprec_t *>(aligned_mem_alloc(sizeof(sprec_t) * static_cast<size_t>(total_rows * ls), 32));

  for (int32_t cs = 0; cs < width; cs += DWT_VERT_STRIP) {
    const int32_t ce = (cs + DWT_VERT_STRIP < width) ? cs + DWT_VERT_STRIP : width;
    const int32_t sw = ce - cs;

    for (int32_t r = 0; r < total_rows; ++r)
      memcpy(strip + r * ls, extbuf + r * buf_stride + cs, sizeof(sprec_t) * static_cast<size_t>(sw));

    // Step D-inverse: even rows
    for (int32_t n = -2 + offset, i = start - 1; i < stop + 2; i++, n += 2) {
      sprec_t *rm1 = strip + (n - 1) * ls;
      sprec_t *rn  = strip + n * ls;
      sprec_t *rp1 = strip + (n + 1) * ls;
      for (int32_t c = 0; c < sw; ++c) {
        int32_t sum = rm1[c] + rp1[c];
        rn[c]       = static_cast<sprec_t>(rn[c] - ((Dcoeff * sum + Doffset) >> Dshift));
      }
    }
    // Step C-inverse: odd rows
    for (int32_t n = -2 + offset, i = start - 1; i < stop + 1; i++, n += 2) {
      sprec_t *rn  = strip + n * ls;
      sprec_t *rn1 = strip + (n + 1) * ls;
      sprec_t *rn2 = strip + (n + 2) * ls;
      for (int32_t c = 0; c < sw; ++c) {
        int32_t sum = rn[c] + rn2[c];
        rn1[c]      = static_cast<sprec_t>(rn1[c] - ((Ccoeff * sum + Coffset) >> Cshift));
      }
    }
    // Step B-inverse: even rows
    for (int32_t n = 0 + offset, i = start; i < stop + 1; i++, n += 2) {
      sprec_t *rm1 = strip + (n - 1) * ls;
      sprec_t *rn  = strip + n * ls;
      sprec_t *rp1 = strip + (n + 1) * ls;
      for (int32_t c = 0; c < sw; ++c) {
        int32_t sum = rm1[c] + rp1[c];
        rn[c]       = static_cast<sprec_t>(rn[c] - ((Bcoeff * sum + Boffset) >> Bshift));
      }
    }
    // Step A-inverse: odd rows
    for (int32_t n = 0 + offset, i = start; i < stop; i++, n += 2) {
      sprec_t *rn  = strip + n * ls;
      sprec_t *rn1 = strip + (n + 1) * ls;
      sprec_t *rn2 = strip + (n + 2) * ls;
      for (int32_t c = 0; c < sw; ++c) {
        int32_t sum = rn[c] + rn2[c];
        rn1[c]      = static_cast<sprec_t>(rn1[c] - ((Acoeff * sum + Aoffset) >> Ashift));
      }
    }

    for (int32_t r = 0; r < total_rows; ++r)
      memcpy(extbuf + r * buf_stride + cs, strip + r * ls, sizeof(sprec_t) * static_cast<size_t>(sw));
  }

  aligned_mem_free(strip);
}

static void idwt_rev53_ver_lifting(sprec_t *extbuf, const int32_t buf_stride, const int32_t total_rows,
                                   const int32_t width, const int32_t top, const int32_t v0,
                                   const int32_t v1) {
  const int32_t start  = v0 / 2;
  const int32_t stop   = v1 / 2;
  const int32_t offset = top - v0 % 2;

  const int32_t ls = round_up(std::min(DWT_VERT_STRIP, width), static_cast<int32_t>(SIMD_PADDING));
  auto *strip =
      static_cast<sprec_t *>(aligned_mem_alloc(sizeof(sprec_t) * static_cast<size_t>(total_rows * ls), 32));

  for (int32_t cs = 0; cs < width; cs += DWT_VERT_STRIP) {
    const int32_t ce = (cs + DWT_VERT_STRIP < width) ? cs + DWT_VERT_STRIP : width;
    const int32_t sw = ce - cs;

    for (int32_t r = 0; r < total_rows; ++r)
      memcpy(strip + r * ls, extbuf + r * buf_stride + cs, sizeof(sprec_t) * static_cast<size_t>(sw));

    // Step 1-inverse: even rows
    for (int32_t n = 0 + offset, i = start; i < stop + 1; ++i, n += 2) {
      sprec_t *rm1 = strip + (n - 1) * ls;
      sprec_t *rn  = strip + n * ls;
      sprec_t *rp1 = strip + (n + 1) * ls;
      for (int32_t c = 0; c < sw; ++c)
        rn[c] = static_cast<sprec_t>(rn[c] - ((rm1[c] + rp1[c] + 2) >> 2));
    }
    // Step 2-inverse: odd rows
    for (int32_t n = 0 + offset, i = start; i < stop; ++i, n += 2) {
      sprec_t *rn  = strip + n * ls;
      sprec_t *rn1 = strip + (n + 1) * ls;
      sprec_t *rn2 = strip + (n + 2) * ls;
      for (int32_t c = 0; c < sw; ++c)
        rn1[c] = static_cast<sprec_t>(rn1[c] + ((rn[c] + rn2[c]) >> 1));
    }

    for (int32_t r = 0; r < total_rows; ++r)
      memcpy(extbuf + r * buf_stride + cs, strip + r * ls, sizeof(sprec_t) * static_cast<size_t>(sw));
  }

  aligned_mem_free(strip);
}

void idwt_irrev_ver_sr_fixed(sprec_t *in, const int32_t u0, const int32_t u1, const int32_t v0,
                              const int32_t v1, const int32_t stride) {
  constexpr int32_t num_pse_i0[2] = {3, 4};
  constexpr int32_t num_pse_i1[2] = {4, 3};
  const int32_t top               = num_pse_i0[v0 % 2];
  const int32_t bottom            = num_pse_i1[v1 % 2];
  const int32_t height            = v1 - v0;
  const int32_t width             = u1 - u0;

  if (v0 == v1 - 1) {
    // Single-sample height: no-op for the 9/7 inverse filter.
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

  idwt_irrev97_ver_lifting(extbuf, bs, total_rows, width, top, v0, v1);

  for (int32_t row = 0; row < height; ++row)
    memcpy(&in[row * stride], extbuf + (top + row) * bs, sizeof(sprec_t) * static_cast<size_t>(width));

  aligned_mem_free(extbuf);
}

void idwt_rev_ver_sr_fixed(sprec_t *in, const int32_t u0, const int32_t u1, const int32_t v0,
                            const int32_t v1, const int32_t stride) {
  constexpr int32_t num_pse_i0[2] = {1, 2};
  constexpr int32_t num_pse_i1[2] = {2, 1};
  const int32_t top               = num_pse_i0[v0 % 2];
  const int32_t bottom            = num_pse_i1[v1 % 2];
  const int32_t height            = v1 - v0;
  const int32_t width             = u1 - u0;

  if (v0 == v1 - 1 && (v0 % 2)) {
    for (int32_t col = 0; col < width; ++col)
      in[col] = static_cast<sprec_t>(in[col] >> 1);
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

  idwt_rev53_ver_lifting(extbuf, bs, total_rows, width, top, v0, v1);

  for (int32_t row = 0; row < height; ++row)
    memcpy(&in[row * stride], extbuf + (top + row) * bs, sizeof(sprec_t) * static_cast<size_t>(width));

  aligned_mem_free(extbuf);
}

// ---------------------------------------------------------------------------
// 2-D interleave helper (unchanged from original)
// ---------------------------------------------------------------------------
static void idwt_2d_interleave_fixed(sprec_t *buf, sprec_t *LL, sprec_t *HL, sprec_t *LH, sprec_t *HH,
                                     int32_t u0, int32_t u1, int32_t v0, int32_t v1,
                                     const int32_t stride) {
  const int32_t v_offset   = v0 % 2;
  const int32_t u_offset   = u0 % 2;
  sprec_t *sp[4]           = {LL, HL, LH, HH};
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
        sprec_t *line = sp[b] + v * stride2[b];
        for (int32_t u = 0, ub = ustart[b]; ub < ustop[b]; ++ub, ++u)
          buf[2 * u + uoffset[b] + (2 * v + voffset[b]) * stride] = *(line++);
      }
    }
  } else {
    sprec_t *first = sp[0], *second = sp[1];
    if (uoffset[0] > uoffset[1]) { first = sp[1]; second = sp[0]; }
    int16x8_t vf0, vf1, vs0, vs1;
    for (int32_t v = 0, vb = vstart[0]; vb < vstop[0]; ++vb, ++v) {
      sprec_t *dp    = buf + (2 * v + voffset[0]) * stride;
      size_t len     = static_cast<size_t>(ustop[0] - ustart[0]);
      sprec_t *line0 = first + v * stride2[0];
      sprec_t *line1 = second + v * stride2[0];
      for (; len >= 16; len -= 16) {
        vf0 = vld1q_s16(line0); vs0 = vld1q_s16(line1);
        vst1q_s16(dp,     vzip1q_s16(vf0, vs0));
        vst1q_s16(dp + 8, vzip2q_s16(vf0, vs0));
        vf1 = vld1q_s16(line0 + 8); vs1 = vld1q_s16(line1 + 8);
        vst1q_s16(dp + 16, vzip1q_s16(vf1, vs1));
        vst1q_s16(dp + 24, vzip2q_s16(vf1, vs1));
        line0 += 16; line1 += 16; dp += 32;
      }
      for (; len > 0; --len) { *dp++ = *line0++; *dp++ = *line1++; }
    }
  }
  if ((ustop[2] - ustart[2]) != (ustop[3] - ustart[3])) {
    for (uint8_t b = 2; b < 4; ++b) {
      for (int32_t v = 0, vb = vstart[b]; vb < vstop[b]; ++vb, ++v) {
        sprec_t *line = sp[b] + v * stride2[b];
        for (int32_t u = 0, ub = ustart[b]; ub < ustop[b]; ++ub, ++u)
          buf[2 * u + uoffset[b] + (2 * v + voffset[b]) * stride] = *(line++);
      }
    }
  } else {
    sprec_t *first = sp[2], *second = sp[3];
    if (uoffset[2] > uoffset[3]) { first = sp[3]; second = sp[2]; }
    int16x8_t vf0, vf1, vs0, vs1;
    for (int32_t v = 0, vb = vstart[2]; vb < vstop[2]; ++vb, ++v) {
      sprec_t *dp    = buf + (2 * v + voffset[2]) * stride;
      size_t len     = static_cast<size_t>(ustop[2] - ustart[2]);
      sprec_t *line0 = first + v * stride2[2];
      sprec_t *line1 = second + v * stride2[2];
      for (; len >= 16; len -= 16) {
        vf0 = vld1q_s16(line0); vs0 = vld1q_s16(line1);
        vst1q_s16(dp,     vzip1q_s16(vf0, vs0));
        vst1q_s16(dp + 8, vzip2q_s16(vf0, vs0));
        vf1 = vld1q_s16(line0 + 8); vs1 = vld1q_s16(line1 + 8);
        vst1q_s16(dp + 16, vzip1q_s16(vf1, vs1));
        vst1q_s16(dp + 24, vzip2q_s16(vf1, vs1));
        line0 += 16; line1 += 16; dp += 32;
      }
      for (; len > 0; --len) { *dp++ = *line0++; *dp++ = *line1++; }
    }
  }
#elif defined(OPENHTJ2K_TRY_AVX2) && defined(__AVX2__)
  for (int32_t pair = 0; pair < 2; ++pair) {
    uint8_t b0 = static_cast<uint8_t>(pair * 2), b1 = static_cast<uint8_t>(pair * 2 + 1);
    if ((ustop[b0] - ustart[b0]) != (ustop[b1] - ustart[b1])) {
      for (uint8_t b = b0; b <= b1; ++b) {
        for (int32_t v = 0, vb = vstart[b]; vb < vstop[b]; ++vb, ++v) {
          sprec_t *line = sp[b] + v * stride2[b];
          for (int32_t u = 0, ub = ustart[b]; ub < ustop[b]; ++ub, ++u)
            buf[2 * u + uoffset[b] + (2 * v + voffset[b]) * stride] = *(line++);
        }
      }
    } else {
      sprec_t *first = sp[b0], *second = sp[b1];
      if (uoffset[b0] > uoffset[b1]) { first = sp[b1]; second = sp[b0]; }
      __m256i vfirst, vsecond;
      for (int32_t v = 0, vb = vstart[b0]; vb < vstop[b0]; ++vb, ++v) {
        sprec_t *dp    = buf + (2 * v + voffset[b0]) * stride;
        size_t len     = static_cast<size_t>(ustop[b0] - ustart[b0]);
        sprec_t *line0 = first + v * stride2[b0];
        sprec_t *line1 = second + v * stride2[b0];
        for (; len >= 16; len -= 16) {
          vfirst  = _mm256_loadu_si256((__m256i *)line0);
          vsecond = _mm256_loadu_si256((__m256i *)line1);
          auto vtmp0 = _mm256_unpacklo_epi16(vfirst, vsecond);
          auto vtmp1 = _mm256_unpackhi_epi16(vfirst, vsecond);
          _mm256_storeu_si256((__m256i *)dp, _mm256_permute2x128_si256(vtmp0, vtmp1, 0x20));
          _mm256_storeu_si256((__m256i *)dp + 1, _mm256_permute2x128_si256(vtmp0, vtmp1, 0x31));
          line0 += 16; line1 += 16; dp += 32;
        }
        for (; len > 0; --len) { *dp++ = *line0++; *dp++ = *line1++; }
      }
    }
  }
#else
  for (uint8_t b = 0; b < 4; ++b) {
    for (int32_t v = 0, vb = vstart[b]; vb < vstop[b]; ++vb, ++v) {
      sprec_t *line = sp[b] + v * stride2[b];
      for (int32_t u = 0, ub = ustart[b]; ub < ustop[b]; ++ub, ++u)
        buf[2 * u + uoffset[b] + (2 * v + voffset[b]) * stride] = *(line++);
    }
  }
#endif
}

// ---------------------------------------------------------------------------
// 2-D IDWT (order kept: interleave → horizontal → vertical)
// ---------------------------------------------------------------------------
void idwt_2d_sr_fixed(sprec_t *nextLL, sprec_t *LL, sprec_t *HL, sprec_t *LH, sprec_t *HH,
                      const int32_t u0, const int32_t u1, const int32_t v0, const int32_t v1,
                      const uint8_t transformation, uint8_t normalizing_upshift) {
  const int32_t stride     = round_up(u1 - u0, 32);
  const int32_t buf_length = stride * (v1 - v0);
  sprec_t *src             = nextLL;

  idwt_2d_interleave_fixed(src, LL, HL, LH, HH, u0, u1, v0, v1, stride);
  idwt_hor_sr_fixed(src, u0, u1, v0, v1, transformation, stride);

  if (transformation == 0)
    idwt_irrev_ver_sr_fixed(src, u0, u1, v0, v1, stride);
  else
    idwt_rev_ver_sr_fixed(src, u0, u1, v0, v1, stride);

  // Scaling for 16-bit fixed-point representation (irreversible path only).
  if (transformation != 1 && normalizing_upshift) {
    int32_t len = buf_length;
#if defined(OPENHTJ2K_ENABLE_ARM_NEON)
    int16x8_t vshift = vdupq_n_s16(normalizing_upshift);
    int16x8_t in0, in1;
    for (; len >= 16; len -= 16) {
      in0 = vld1q_s16(src); in1 = vld1q_s16(src + 8);
      in0 = vshlq_s16(in0, vshift); in1 = vshlq_s16(in1, vshift);
      vst1q_s16(src, in0); vst1q_s16(src + 8, in1);
      src += 16;
    }
    for (; len > 0; --len) {
      *src = static_cast<sprec_t>(*src << normalizing_upshift); src++;
    }
#elif defined(OPENHTJ2K_TRY_AVX2) && defined(__AVX2__)
    for (; len >= 16; len -= 16) {
      __m256i tmp0 = _mm256_load_si256((__m256i *)src);
      __m256i tmp1 = _mm256_slli_epi16(tmp0, static_cast<int32_t>(normalizing_upshift));
      _mm256_store_si256((__m256i *)src, tmp1);
      src += 16;
    }
    for (; len > 0; --len) {
      *src = static_cast<sprec_t>(static_cast<usprec_t>(*src) << normalizing_upshift); src++;
    }
#else
    for (; len > 0; --len) {
      *src = static_cast<sprec_t>(static_cast<usprec_t>(*src) << normalizing_upshift); src++;
    }
#endif
  }
}

// ---------------------------------------------------------------------------
// Line-based stateful inverse DWT implementation
// ---------------------------------------------------------------------------

void idwt_line_state_init(idwt_line_state_t *state, const int32_t u0, const int32_t u1,
                          const int32_t v0, const int32_t v1, const uint8_t transformation,
                          const uint8_t normalizing_upshift) {
  state->u0                  = u0;
  state->u1                  = u1;
  state->v0                  = v0;
  state->v1                  = v1;
  state->width               = u1 - u0;
  state->height              = v1 - v0;
  state->transformation      = transformation;
  state->normalizing_upshift = normalizing_upshift;
  state->rows_received       = 0;

  constexpr int32_t ver_pse_i0_irrev[2] = {3, 4};
  constexpr int32_t ver_pse_i1_irrev[2] = {4, 3};
  constexpr int32_t ver_pse_i0_rev[2]   = {1, 2};
  constexpr int32_t ver_pse_i1_rev[2]   = {2, 1};
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

  // Horizontal IDWT workspace is managed by idwt_hor_sr_fixed internally.
  state->h_left  = 0;
  state->h_right = 0;
  state->h_work  = nullptr;
}

// Push one already-interleaved row (LL/HL or LH/HH merged into the nextLL
// interleaved layout) into the flat buffer.  Horizontal IDWT is deferred to
// idwt_line_state_finalize to preserve the original H^-1-then-V^-1 ordering
// required for exact lossless round-trip recovery.
void idwt_line_state_push_row(idwt_line_state_t *state, const sprec_t *row_ptr,
                               const int32_t in_stride) {
  const int32_t r   = state->rows_received;
  const int32_t top = state->top;
  const int32_t w   = state->width;
  const int32_t bs  = state->buf_stride;

  sprec_t *dst = state->buf + static_cast<size_t>((top + r) * bs);
  if (in_stride == 1) {
    memcpy(dst, row_ptr, sizeof(sprec_t) * static_cast<size_t>(w));
  } else {
    for (int32_t c = 0; c < w; ++c) dst[c] = row_ptr[c * in_stride];
  }

  state->rows_received = r + 1;
}

// Finish: apply horizontal IDWT, fill PSE rows, apply column-strip vertical
// IDWT, apply normalising up-shift (irreversible path), then copy to nextLL.
// The H^-1-then-V^-1 ordering matches idwt_2d_sr_fixed so that the streaming
// state can be used as a drop-in replacement for fdwt_line_state_t /
// idwt_line_state_t pairs.
void idwt_line_state_finalize(idwt_line_state_t *state, sprec_t *nextLL) {
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
  const int32_t stride = round_up(w, 32);

  // Step 1: apply horizontal IDWT to every tile row in the flat buffer.
  sprec_t *tile_start = state->buf + static_cast<size_t>(top * bs);
  idwt_hor_sr_fixed(tile_start, u0, u1, v0, v1, tr, bs);

  // Step 2: fill PSE-top and PSE-bottom from the now-H^-1-processed tile rows.
  // This mirrors what idwt_irrev/rev_ver_sr_fixed does after idwt_hor_sr_fixed.
  for (int32_t i = 1; i <= top; ++i) {
    const int32_t src = PSEo(v0 - i, v0, v1);
    memcpy(state->buf + static_cast<size_t>((top - i) * bs),
           state->buf + static_cast<size_t>((top + src) * bs),
           sizeof(sprec_t) * static_cast<size_t>(w));
  }
  for (int32_t i = 1; i <= bottom; ++i) {
    const int32_t src = PSEo(v1 - v0 + i - 1 + v0, v0, v1);
    memcpy(state->buf + static_cast<size_t>((top + height + i - 1) * bs),
           state->buf + static_cast<size_t>((top + src) * bs),
           sizeof(sprec_t) * static_cast<size_t>(w));
  }

  // Step 3: apply column-strip vertical IDWT.
  if (tr == 0)
    idwt_irrev97_ver_lifting(state->buf, bs, state->total_rows, w, top, v0, v1);
  else
    idwt_rev53_ver_lifting(state->buf, bs, state->total_rows, w, top, v0, v1);

  // Apply normalising up-shift (irreversible path).
  if (tr != 1 && state->normalizing_upshift) {
    const uint8_t shift = state->normalizing_upshift;
    for (int32_t row = 0; row < height; ++row) {
      sprec_t *p = state->buf + static_cast<size_t>((top + row) * bs);
      for (int32_t c = 0; c < w; ++c)
        p[c] = static_cast<sprec_t>(static_cast<usprec_t>(p[c]) << shift);
    }
  }

  // Copy tile rows to nextLL (which may have a different stride).
  for (int32_t row = 0; row < height; ++row)
    memcpy(nextLL + row * stride, state->buf + static_cast<size_t>((top + row) * bs),
           sizeof(sprec_t) * static_cast<size_t>(w));
}

void idwt_line_state_free(idwt_line_state_t *state) {
  aligned_mem_free(state->buf);
  aligned_mem_free(state->h_work);
  state->buf    = nullptr;
  state->h_work = nullptr;
}
