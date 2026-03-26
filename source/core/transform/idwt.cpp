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
#include <cmath>
#include "dwt.hpp"
#include "utils.hpp"
#if defined(OPENHTJ2K_ENABLE_ARM_NEON)
static idwt_1d_filtd_func_fixed idwt_1d_filtr_fixed[2] = {idwt_1d_filtr_irrev97_fixed_neon,
                                                          idwt_1d_filtr_rev53_fixed_neon};
static idwt_ver_filtd_func_fixed idwt_ver_sr_fixed[2]  = {idwt_irrev_ver_sr_fixed_neon,
                                                          idwt_rev_ver_sr_fixed_neon};
#elif defined(OPENHTJ2K_ENABLE_AVX2)
static idwt_1d_filtd_func_fixed idwt_1d_filtr_fixed[2] = {idwt_1d_filtr_irrev97_fixed_avx2,
                                                          idwt_1d_filtr_rev53_fixed_avx2};
static idwt_ver_filtd_func_fixed idwt_ver_sr_fixed[2]  = {idwt_irrev_ver_sr_fixed_avx2,
                                                          idwt_rev_ver_sr_fixed_avx2};
#else
static idwt_1d_filtd_func_fixed idwt_1d_filtr_fixed[2] = {idwt_1d_filtr_irrev97_fixed,
                                                          idwt_1d_filtr_rev53_fixed};
static idwt_ver_filtd_func_fixed idwt_ver_sr_fixed[2]  = {idwt_irrev_ver_sr_fixed, idwt_rev_ver_sr_fixed};
#endif

void idwt_1d_filtr_irrev97_fixed(sprec_t *X, const int32_t left, const int32_t u_i0, const int32_t u_i1) {
  const auto i0        = static_cast<int32_t>(u_i0);
  const auto i1        = static_cast<int32_t>(u_i1);
  const int32_t start  = i0 / 2;
  const int32_t stop   = i1 / 2;
  const int32_t offset = left - i0 % 2;

  float sum;
  /* K and 1/K have been already done by dequantization */
  for (int32_t n = -2 + offset, i = start - 1; i < stop + 2; i++, n += 2) {
    sum = X[n - 1];
    sum += X[n + 1];
    X[n] = X[n] - fD * sum;
  }
  int16_t a[16];
  memcpy(a, X - 2 + offset, sizeof(int16_t) * 16);
  for (int32_t n = -2 + offset, i = start - 1; i < stop + 1; i++, n += 2) {
    sum = X[n];
    sum += X[n + 2];
    X[n + 1] = X[n + 1] - fC * sum;
  }
  for (int32_t n = 0 + offset, i = start; i < stop + 1; i++, n += 2) {
    sum = X[n - 1];
    sum += X[n + 1];
    X[n] = X[n] - fB * sum;
  }
  for (int32_t n = 0 + offset, i = start; i < stop; i++, n += 2) {
    sum = X[n];
    sum += X[n + 2];
    X[n + 1] = X[n + 1] - fA * sum;
  }
}

void idwt_1d_filtr_rev53_fixed(sprec_t *X, const int32_t left, const int32_t u_i0, const int32_t u_i1) {
  const auto i0        = static_cast<int32_t>(u_i0);
  const auto i1        = static_cast<int32_t>(u_i1);
  const int32_t start  = i0 / 2;
  const int32_t stop   = i1 / 2;
  const int32_t offset = left - i0 % 2;

  for (int32_t n = 0 + offset, i = start; i < stop + 1; ++i, n += 2) {
    X[n] -= floorf((X[n - 1] + X[n + 1] + 2) * 0.25f);
  }

  for (int32_t n = 0 + offset, i = start; i < stop; ++i, n += 2) {
    X[n + 1] += floorf((X[n] + X[n + 2]) * 0.5f);
  }
}

static void idwt_1d_sr_fixed(sprec_t *buf, sprec_t *in, const int32_t left, const int32_t right,
                             const int32_t i0, const int32_t i1, const uint8_t transformation) {
  dwt_1d_extr_fixed(buf, in, left, right, i0, i1);
  idwt_1d_filtr_fixed[transformation](buf, left, i0, i1);
  memcpy(in, buf + left, sizeof(sprec_t) * (static_cast<size_t>(i1 - i0)));
}

// In-place 1-D IDWT for interior rows (not first or last).
// Operates directly on in[-left..width+SIMD_LEN_I32-1] without copying to/from an external buffer.
// Precondition: those memory locations are within the tile allocation (guaranteed for interior rows).
static inline void idwt_1d_sr_inplace(sprec_t *in, const int32_t left, const int32_t right,
                                      const int32_t i0, const int32_t i1,
                                      const uint8_t transformation) {
  const int32_t width = i1 - i0;
  // Save regions that the filter will temporarily overwrite with PSE data or SIMD tail writes.
  sprec_t left_save[4];
  sprec_t right_save[SIMD_LEN_I32];
  for (int32_t i = 0; i < left; ++i) left_save[i] = in[-left + i];
  for (int32_t i = 0; i < SIMD_LEN_I32; ++i) right_save[i] = in[width + i];
  // Fill left PSE into in[-left..-1] and right PSE into in[width..width+right-1].
  for (int32_t i = 1; i <= left; ++i)
    in[-i] = in[PSEo(i0 - i, i0, i1)];
  for (int32_t i = 1; i <= right; ++i)
    in[width + i - 1] = in[PSEo(i1 - i0 + i - 1 + i0, i0, i1)];
  // Filter in-place: in-left is the extended buffer (left PSE | data | right PSE).
  idwt_1d_filtr_fixed[transformation](in - left, left, i0, i1);
  // Restore the saved regions (IDWT output is in in[0..width-1], boundary regions are scratch).
  for (int32_t i = 0; i < left; ++i) in[-left + i] = left_save[i];
  for (int32_t i = 0; i < SIMD_LEN_I32; ++i) in[width + i] = right_save[i];
}

static void idwt_hor_sr_fixed(sprec_t *in, const int32_t u0, const int32_t u1, const int32_t v0,
                              const int32_t v1, const uint8_t transformation, const int32_t stride) {
  constexpr int32_t num_pse_i0[2][2] = {{3, 1}, {4, 2}};
  constexpr int32_t num_pse_i1[2][2] = {{4, 2}, {3, 1}};
  const int32_t left                 = num_pse_i0[u0 % 2][transformation];
  const int32_t right                = num_pse_i1[u1 % 2][transformation];

  if (u0 == u1 - 1) {
    // one sample case
    for (int32_t row = 0; row < v1 - v0; ++row) {
      if (u0 % 2 != 0 && transformation) {
        in[row * stride] = static_cast<sprec_t>(in[row * stride] / 2.0f);
      }
    }
  } else {
    // need to perform symmetric extension
    const int32_t nrows = v1 - v0;
    const int32_t len   = u1 - u0 + left + right;
    // Yext is used only for the first and last rows; interior rows use in-place transform.
    auto *Yext = static_cast<sprec_t *>(aligned_mem_alloc(
        sizeof(sprec_t) * static_cast<size_t>(round_up(len + SIMD_PADDING, SIMD_PADDING)), 32));
    for (int32_t row = 0; row < nrows; ++row) {
      if (row == 0 || row == nrows - 1) {
        // First/last rows: use copy-based path (in[-left] or in[width+SIMD_LEN_I32-1] may be
        // outside the tile allocation for the first and last rows respectively).
        idwt_1d_sr_fixed(Yext, in, left, right, u0, u1, transformation);
      } else {
        idwt_1d_sr_inplace(in, left, right, u0, u1, transformation);
      }
      in += stride;
    }
    aligned_mem_free(Yext);
  }
}

void idwt_irrev_ver_sr_fixed(sprec_t *in, const int32_t u0, const int32_t u1, const int32_t v0,
                             const int32_t v1, const int32_t stride, sprec_t *pse_scratch,
                             sprec_t **buf_scratch) {
  constexpr int32_t num_pse_i0[2] = {3, 4};
  constexpr int32_t num_pse_i1[2] = {4, 3};
  const int32_t top               = num_pse_i0[v0 % 2];
  const int32_t bottom            = num_pse_i1[v1 % 2];
  if (v0 == v1 - 1) {
    // one sample case
    for (int32_t col = 0; col < u1 - u0; ++col) {
      // in[col] >>= (v0 % 2 == 0) ? 0 : 0;
    }
  } else {
    const int32_t len = round_up(stride, SIMD_PADDING);
    sprec_t **buf     = buf_scratch;
    for (int32_t i = 1; i <= top; ++i) {
      // buf[top - i] =
      //     static_cast<sprec_t *>(aligned_mem_alloc(sizeof(sprec_t) * static_cast<size_t>(len), 32));
      buf[top - i] = pse_scratch + (i - 1) * len;
      memcpy(buf[top - i], &in[PSEo(v0 - i, v0, v1) * stride],
             sizeof(sprec_t) * static_cast<size_t>(stride));
    }
    for (int32_t row = 0; row < v1 - v0; ++row) {
      buf[top + row] = &in[row * stride];
    }
    for (int32_t i = 1; i <= bottom; i++) {
      // buf[top + (v1 - v0) + i - 1] =
      //     static_cast<sprec_t *>(aligned_mem_alloc(sizeof(sprec_t) * static_cast<size_t>(len), 32));
      buf[top + (v1 - v0) + i - 1] = pse_scratch + (top + i - 1) * len;
      memcpy(buf[top + (v1 - v0) + i - 1], &in[PSEo(v1 - v0 + i - 1 + v0, v0, v1) * stride],
             sizeof(sprec_t) * static_cast<size_t>(stride));
    }
    const int32_t start  = v0 / 2;
    const int32_t stop   = v1 / 2;
    const int32_t offset = top - v0 % 2;

    const int32_t width = u1 - u0;
    for (int32_t cs = 0; cs < width; cs += DWT_VERT_STRIP) {
      const int32_t ce = (cs + DWT_VERT_STRIP < width) ? cs + DWT_VERT_STRIP : width;
      for (int32_t n = -2 + offset, i = start - 1; i < stop + 2; i++, n += 2) {
        for (int32_t col = cs; col < ce; ++col) {
          buf[n][col] -= fD * (buf[n - 1][col] + buf[n + 1][col]);
        }
      }
      for (int32_t n = -2 + offset, i = start - 1; i < stop + 1; i++, n += 2) {
        for (int32_t col = cs; col < ce; ++col) {
          buf[n + 1][col] -= fC * (buf[n][col] + buf[n + 2][col]);
        }
      }
      for (int32_t n = 0 + offset, i = start; i < stop + 1; i++, n += 2) {
        for (int32_t col = cs; col < ce; ++col) {
          buf[n][col] -= fB * (buf[n - 1][col] + buf[n + 1][col]);
        }
      }
      for (int32_t n = 0 + offset, i = start; i < stop; i++, n += 2) {
        for (int32_t col = cs; col < ce; ++col) {
          buf[n + 1][col] -= fA * (buf[n][col] + buf[n + 2][col]);
        }
      }
    }

    // for (int32_t i = 1; i <= top; ++i) {
    //   aligned_mem_free(buf[top - i]);
    // }
    // for (int32_t i = 1; i <= bottom; i++) {
    //   aligned_mem_free(buf[top + (v1 - v0) + i - 1]);
    // }
  }
}

void idwt_rev_ver_sr_fixed(sprec_t *in, const int32_t u0, const int32_t u1, const int32_t v0,
                           const int32_t v1, const int32_t stride, sprec_t *pse_scratch,
                           sprec_t **buf_scratch) {
  constexpr int32_t num_pse_i0[2] = {1, 2};
  constexpr int32_t num_pse_i1[2] = {2, 1};
  const int32_t top               = num_pse_i0[v0 % 2];
  const int32_t bottom            = num_pse_i1[v1 % 2];
  if (v0 == v1 - 1 && (v0 % 2)) {
    // one sample case
    for (int32_t col = 0; col < u1 - u0; ++col) {
      in[col] = floorf(in[col] * 0.5f);
    }
  } else {
    const int32_t len = round_up(stride, SIMD_PADDING);
    sprec_t **buf     = buf_scratch;
    for (int32_t i = 1; i <= top; ++i) {
      // buf[top - i] =
      //     static_cast<sprec_t *>(aligned_mem_alloc(sizeof(sprec_t) * static_cast<size_t>(len), 32));
      buf[top - i] = pse_scratch + (i - 1) * len;
      memcpy(buf[top - i], &in[PSEo(v0 - i, v0, v1) * stride],
             sizeof(sprec_t) * static_cast<size_t>(stride));
    }
    for (int32_t row = 0; row < v1 - v0; ++row) {
      buf[top + row] = &in[row * stride];
    }
    for (int32_t i = 1; i <= bottom; i++) {
      // buf[top + (v1 - v0) + i - 1] =
      //     static_cast<sprec_t *>(aligned_mem_alloc(sizeof(sprec_t) * static_cast<size_t>(len), 32));
      buf[top + (v1 - v0) + i - 1] = pse_scratch + (top + i - 1) * len;
      memcpy(buf[top + (v1 - v0) + i - 1], &in[PSEo(v1 - v0 + i - 1 + v0, v0, v1) * stride],
             sizeof(sprec_t) * static_cast<size_t>(stride));
    }
    const int32_t start  = v0 / 2;
    const int32_t stop   = v1 / 2;
    const int32_t offset = top - v0 % 2;

    const int32_t width = u1 - u0;
    for (int32_t cs = 0; cs < width; cs += DWT_VERT_STRIP) {
      const int32_t ce = (cs + DWT_VERT_STRIP < width) ? cs + DWT_VERT_STRIP : width;
      for (int32_t n = 0 + offset, i = start; i < stop + 1; ++i, n += 2) {
        for (int32_t col = cs; col < ce; ++col) {
          buf[n][col] -= floorf((buf[n - 1][col] + buf[n + 1][col] + 2.0f) * 0.25f);
        }
      }
      for (int32_t n = 0 + offset, i = start; i < stop; ++i, n += 2) {
        for (int32_t col = cs; col < ce; ++col) {
          buf[n + 1][col] += floorf((buf[n][col] + buf[n + 2][col]) * 0.5f);
        }
      }
    }

    // for (int32_t i = 1; i <= top; ++i) {
    //   aligned_mem_free(buf[top - i]);
    // }
    // for (int32_t i = 1; i <= bottom; i++) {
    //   aligned_mem_free(buf[top + (v1 - v0) + i - 1]);
    // }
  }
}

static void idwt_2d_interleave_fixed(sprec_t *buf, sprec_t *LL, sprec_t *HL, sprec_t *LH, sprec_t *HH,
                                     int32_t u0, int32_t u1, int32_t v0, int32_t v1, const int32_t stride) {
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
        for (int32_t u = 0, ub = ustart[b]; ub < ustop[b]; ++ub, ++u) {
          buf[2 * u + uoffset[b] + (2 * v + voffset[b]) * stride] = *(line++);
        }
      }
    }
  } else {
    sprec_t *first, *second;
    first  = sp[0];
    second = sp[1];
    if (uoffset[0] > uoffset[1]) {
      first  = sp[1];
      second = sp[0];
    }
    float32x4_t vfirst0, vfirst1, vsecond0, vsecond1;
    //    int16x8x2_t vdst0, vdst1;
    for (int32_t v = 0, vb = vstart[0]; vb < vstop[0]; ++vb, ++v) {
      sprec_t *dp    = buf + (2 * v + voffset[0]) * stride;
      size_t len     = static_cast<size_t>(ustop[0] - ustart[0]);
      sprec_t *line0 = first + v * stride2[0];
      sprec_t *line1 = second + v * stride2[0];
      for (; len >= 8; len -= 8) {
        vfirst0  = vld1q_f32(line0);
        vsecond0 = vld1q_f32(line1);
        vst1q_f32(dp, vzip1q_f32(vfirst0, vsecond0));
        vst1q_f32(dp + 4, vzip2q_f32(vfirst0, vsecond0));
        vfirst1  = vld1q_f32(line0 + 4);
        vsecond1 = vld1q_f32(line1 + 4);
        vst1q_f32(dp + 8, vzip1q_f32(vfirst1, vsecond1));
        vst1q_f32(dp + 12, vzip2q_f32(vfirst1, vsecond1));
        line0 += 8;
        line1 += 8;
        dp += 16;
      }
      for (; len > 0; --len) {
        *dp++ = *line0++;
        *dp++ = *line1++;
      }
    }
  }

  if ((ustop[2] - ustart[2]) != (ustop[3] - ustart[3])) {
    for (uint8_t b = 2; b < 4; ++b) {
      for (int32_t v = 0, vb = vstart[b]; vb < vstop[b]; ++vb, ++v) {
        sprec_t *line = sp[b] + v * stride2[b];
        for (int32_t u = 0, ub = ustart[b]; ub < ustop[b]; ++ub, ++u) {
          buf[2 * u + uoffset[b] + (2 * v + voffset[b]) * stride] = *(line++);
        }
      }
    }
  } else {
    sprec_t *first, *second;
    first  = sp[2];
    second = sp[3];
    if (uoffset[2] > uoffset[3]) {
      first  = sp[3];
      second = sp[2];
    }
    float32x4_t vfirst0, vfirst1, vsecond0, vsecond1;
    //    int16x8x2_t vdst0, vdst1;
    for (int32_t v = 0, vb = vstart[2]; vb < vstop[2]; ++vb, ++v) {
      sprec_t *dp    = buf + (2 * v + voffset[2]) * stride;
      size_t len     = static_cast<size_t>(ustop[2] - ustart[2]);
      sprec_t *line0 = first + v * stride2[2];
      sprec_t *line1 = second + v * stride2[2];
      for (; len >= 8; len -= 8) {
        vfirst0  = vld1q_f32(line0);
        vsecond0 = vld1q_f32(line1);
        vst1q_f32(dp, vzip1q_f32(vfirst0, vsecond0));
        vst1q_f32(dp + 4, vzip2q_f32(vfirst0, vsecond0));
        vfirst1  = vld1q_f32(line0 + 4);
        vsecond1 = vld1q_f32(line1 + 4);
        vst1q_f32(dp + 8, vzip1q_f32(vfirst1, vsecond1));
        vst1q_f32(dp + 12, vzip2q_f32(vfirst1, vsecond1));
        line0 += 8;
        line1 += 8;
        dp += 16;
      }
      for (; len > 0; --len) {
        *dp++ = *line0++;
        *dp++ = *line1++;
      }
    }
  }
#elif defined(OPENHTJ2K_TRY_AVX2) && defined(__AVX2__)
  if ((ustop[0] - ustart[0]) != (ustop[1] - ustart[1])) {
    for (uint8_t b = 0; b < 2; ++b) {
      for (int32_t v = 0, vb = vstart[b]; vb < vstop[b]; ++vb, ++v) {
        sprec_t *line = sp[b] + v * stride2[b];
        for (int32_t u = 0, ub = ustart[b]; ub < ustop[b]; ++ub, ++u) {
          buf[2 * u + uoffset[b] + (2 * v + voffset[b]) * stride] = *(line++);
        }
      }
    }
  } else {
    sprec_t *first, *second;
    first  = sp[0];
    second = sp[1];
    if (uoffset[0] > uoffset[1]) {
      first  = sp[1];
      second = sp[0];
    }
    for (int32_t v = 0, vb = vstart[0]; vb < vstop[0]; ++vb, ++v) {
      sprec_t *dp    = buf + (2 * v + voffset[0]) * stride;
      size_t len     = static_cast<size_t>(ustop[0] - ustart[0]);
      sprec_t *line0 = first + v * stride2[0];
      sprec_t *line1 = second + v * stride2[0];
      // SSE version
      //  for (; len >= 8; len -= 8) {
      //    auto vfirst  = _mm_loadu_si128((__m128i *)line0);
      //    auto vsecond = _mm_loadu_si128((__m128i *)line1);
      //    auto vtmp0   = _mm_unpacklo_epi16(vfirst, vsecond);
      //    auto vtmp1   = _mm_unpackhi_epi16(vfirst, vsecond);
      //    _mm_storeu_si128((__m128i *)dp, vtmp0);
      //    _mm_storeu_si128((__m128i *)(dp + 8), vtmp1);
      //    line0 += 8;
      //    line1 += 8;
      //    dp += 16;
      // }

      // AVX2 version
      __m256i vfirst, vsecond;
      for (; len >= 8; len -= 8) {
        vfirst     = _mm256_loadu_si256((__m256i *)line0);
        vsecond    = _mm256_loadu_si256((__m256i *)line1);
        auto vtmp0 = _mm256_unpacklo_epi32(vfirst, vsecond);
        auto vtmp1 = _mm256_unpackhi_epi32(vfirst, vsecond);

        _mm256_storeu_si256((__m256i *)dp, _mm256_permute2x128_si256(vtmp0, vtmp1, 0x20));
        _mm256_storeu_si256((__m256i *)dp + 1, _mm256_permute2x128_si256(vtmp0, vtmp1, 0x31));
        line0 += 8;
        line1 += 8;
        dp += 16;
      }
      for (; len > 0; --len) {
        *dp++ = *line0++;
        *dp++ = *line1++;
      }
    }
  }

  if ((ustop[2] - ustart[2]) != (ustop[3] - ustart[3])) {
    for (uint8_t b = 2; b < 4; ++b) {
      for (int32_t v = 0, vb = vstart[b]; vb < vstop[b]; ++vb, ++v) {
        sprec_t *line = sp[b] + v * stride2[b];
        for (int32_t u = 0, ub = ustart[b]; ub < ustop[b]; ++ub, ++u) {
          buf[2 * u + uoffset[b] + (2 * v + voffset[b]) * stride] = *(line++);
        }
      }
    }
  } else {
    sprec_t *first, *second;
    first  = sp[2];
    second = sp[3];
    if (uoffset[2] > uoffset[3]) {
      first  = sp[3];
      second = sp[2];
    }
    for (int32_t v = 0, vb = vstart[2]; vb < vstop[2]; ++vb, ++v) {
      sprec_t *dp    = buf + (2 * v + voffset[2]) * stride;
      size_t len     = static_cast<size_t>(ustop[2] - ustart[2]);
      sprec_t *line0 = first + v * stride2[2];
      sprec_t *line1 = second + v * stride2[2];
      // SSE version
      //  for (; len >= 8; len -= 8) {
      //    auto vfirst  = _mm_loadu_si128((__m128i *)line0);
      //    auto vsecond = _mm_loadu_si128((__m128i *)line1);
      //    auto vtmp0   = _mm_unpacklo_epi16(vfirst, vsecond);
      //    auto vtmp1   = _mm_unpackhi_epi16(vfirst, vsecond);
      //    _mm_storeu_si128((__m128i *)dp, vtmp0);
      //    _mm_storeu_si128((__m128i *)(dp + 8), vtmp1);
      //    line0 += 8;
      //    line1 += 8;
      //    dp += 16;
      // }

      // AVX2 version
      __m256i vfirst, vsecond;
      for (; len >= 8; len -= 8) {
        vfirst     = _mm256_loadu_si256((__m256i *)line0);
        vsecond    = _mm256_loadu_si256((__m256i *)line1);
        auto vtmp0 = _mm256_unpacklo_epi32(vfirst, vsecond);
        auto vtmp1 = _mm256_unpackhi_epi32(vfirst, vsecond);

        _mm256_storeu_si256((__m256i *)dp, _mm256_permute2x128_si256(vtmp0, vtmp1, 0x20));
        _mm256_storeu_si256((__m256i *)dp + 1, _mm256_permute2x128_si256(vtmp0, vtmp1, 0x31));
        line0 += 8;
        line1 += 8;
        dp += 16;
      }
      for (; len > 0; --len) {
        *dp++ = *line0++;
        *dp++ = *line1++;
      }
    }
  }
#else
  for (uint8_t b = 0; b < 4; ++b) {
    for (int32_t v = 0, vb = vstart[b]; vb < vstop[b]; ++vb, ++v) {
      sprec_t *line = sp[b] + v * stride2[b];
      for (int32_t u = 0, ub = ustart[b]; ub < ustop[b]; ++ub, ++u) {
        buf[2 * u + uoffset[b] + (2 * v + voffset[b]) * stride] = *(line++);
      }
    }
  }
#endif
}

void idwt_2d_sr_fixed(sprec_t *nextLL, sprec_t *LL, sprec_t *HL, sprec_t *LH, sprec_t *HH, const int32_t u0,
                      const int32_t u1, const int32_t v0, const int32_t v1, const uint8_t transformation,
                      sprec_t *pse_scratch, sprec_t **buf_scratch) {
  const int32_t stride     = round_up(u1 - u0, 32);
  sprec_t *src             = nextLL;
  idwt_2d_interleave_fixed(src, LL, HL, LH, HH, u0, u1, v0, v1, stride);
  idwt_hor_sr_fixed(src, u0, u1, v0, v1, transformation, stride);

  // Vertical DWT (pse_scratch provided by caller, sized for 8 * round_up(stride, SIMD_LEN_I32))
  idwt_ver_sr_fixed[transformation](src, u0, u1, v0, v1, stride, pse_scratch, buf_scratch);
}
