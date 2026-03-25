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
static fdwt_1d_filtr_func_fixed fdwt_1d_filtr_fixed[2] = {fdwt_1d_filtr_irrev97_fixed_neon,
                                                          fdwt_1d_filtr_rev53_fixed_neon};
static fdwt_ver_filtr_func_fixed fdwt_ver_sr_fixed[2]  = {fdwt_irrev_ver_sr_fixed_neon,
                                                          fdwt_rev_ver_sr_fixed_neon};
#elif defined(OPENHTJ2K_ENABLE_AVX2)
static fdwt_1d_filtr_func_fixed fdwt_1d_filtr_fixed[2] = {fdwt_1d_filtr_irrev97_fixed_avx2,
                                                          fdwt_1d_filtr_rev53_fixed_avx2};
static fdwt_ver_filtr_func_fixed fdwt_ver_sr_fixed[2]  = {fdwt_irrev_ver_sr_fixed_avx2,
                                                          fdwt_rev_ver_sr_fixed_avx2};
#else
static fdwt_1d_filtr_func_fixed fdwt_1d_filtr_fixed[2] = {fdwt_1d_filtr_irrev97_fixed,
                                                          fdwt_1d_filtr_rev53_fixed};
static fdwt_ver_filtr_func_fixed fdwt_ver_sr_fixed[2]  = {fdwt_irrev_ver_sr_fixed, fdwt_rev_ver_sr_fixed};
#endif
// irreversible FDWT
void fdwt_1d_filtr_irrev97_fixed(sprec_t *X, const int32_t left, const int32_t u_i0, const int32_t u_i1) {
  const auto i0       = static_cast<int32_t>(u_i0);
  const auto i1       = static_cast<int32_t>(u_i1);
  const int32_t start = ceil_int(i0, 2);
  const int32_t stop  = ceil_int(i1, 2);

  const int32_t offset = left + i0 % 2;
  for (int32_t n = -4 + offset, i = start - 2; i < stop + 1; i++, n += 2) {
    float sum = X[n];
    sum += X[n + 2];
    // X[n + 1] = static_cast<sprec_t>(X[n + 1] + ((Acoeff * sum + Aoffset) >> Ashift));
    X[n + 1] += fA * sum;
  }
  for (int32_t n = -2 + offset, i = start - 1; i < stop + 1; i++, n += 2) {
    float sum = X[n - 1];
    sum += X[n + 1];
    // X[n] = static_cast<sprec_t>(X[n] + ((Bcoeff * sum + Boffset) >> Bshift));
    X[n] += fB * sum;
  }
  for (int32_t n = -2 + offset, i = start - 1; i < stop; i++, n += 2) {
    float sum = X[n];
    sum += X[n + 2];
    // X[n + 1] = static_cast<sprec_t>(X[n + 1] + ((Ccoeff * sum + Coffset) >> Cshift));
    X[n + 1] += fC * sum;
  }
  for (int32_t n = 0 + offset, i = start; i < stop; i++, n += 2) {
    float sum = X[n - 1];
    sum += X[n + 1];
    // X[n] = static_cast<sprec_t>(X[n] + ((Dcoeff * sum + Doffset) >> Dshift));
    X[n] += fD * sum;
  }
};

// reversible FDWT
void fdwt_1d_filtr_rev53_fixed(sprec_t *X, const int32_t left, const int32_t u_i0, const int32_t u_i1) {
  const auto i0       = static_cast<int32_t>(u_i0);
  const auto i1       = static_cast<int32_t>(u_i1);
  const int32_t start = ceil_int(i0, 2);
  const int32_t stop  = ceil_int(i1, 2);
  // X += left - i0 % 2;
  const int32_t offset = left + i0 % 2;
  for (int32_t n = -2 + offset, i = start - 1; i < stop; ++i, n += 2) {
    float sum = X[n] + X[n + 2];
    X[n + 1] -= floorf(sum * 0.5f);
  }
  for (int32_t n = 0 + offset, i = start; i < stop; ++i, n += 2) {
    float sum = X[n - 1] + X[n + 1];
    X[n] += floorf((sum + 2) * 0.25f);
  }
};

// 1-dimensional FDWT
static inline void fdwt_1d_sr_fixed(sprec_t *buf, sprec_t *in, const int32_t left, const int32_t right,
                                    const int32_t i0, const int32_t i1, const uint8_t transformation) {
  dwt_1d_extr_fixed(buf, in, left, right, i0, i1);
  fdwt_1d_filtr_fixed[transformation](buf, left, i0, i1);
  memcpy(in, buf + left, sizeof(sprec_t) * (static_cast<size_t>(i1 - i0)));
}

// In-place 1-D FDWT for interior rows (not first or last).
// Operates directly on in[-left..width+SIMD_LEN_I32-1] without copying to/from an external buffer.
// Precondition: those memory locations are within the tile allocation (guaranteed for interior rows).
static inline void fdwt_1d_sr_inplace(sprec_t *in, const int32_t left, const int32_t right,
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
  fdwt_1d_filtr_fixed[transformation](in - left, left, i0, i1);
  // Restore the saved regions (DWT output is in in[0..width-1], boundary regions are scratch).
  for (int32_t i = 0; i < left; ++i) in[-left + i] = left_save[i];
  for (int32_t i = 0; i < SIMD_LEN_I32; ++i) in[width + i] = right_save[i];
}

// FDWT for horizontal direction
static void fdwt_hor_sr_fixed(sprec_t *in, const int32_t u0, const int32_t u1, const int32_t v0,
                              const int32_t v1, const uint8_t transformation, const int32_t stride) {
  constexpr int32_t num_pse_i0[2][2] = {{4, 2}, {3, 1}};
  constexpr int32_t num_pse_i1[2][2] = {{3, 1}, {4, 2}};
  const int32_t left                 = num_pse_i0[u0 % 2][transformation];
  const int32_t right                = num_pse_i1[u1 % 2][transformation];

  if (u0 == u1 - 1) {
    // one sample case
    for (int32_t row = 0; row < v1 - v0; ++row) {
      if (u0 % 2 == 0) {
        in[row * stride] = (transformation) ? in[row * stride] : in[row * stride];
      } else {
        in[row * stride] =
            (transformation) ? floorf(in[row * stride] * 2.0f) : in[row * stride];
      }
    }
  } else {
    // need to perform symmetric extension
    const int32_t nrows = v1 - v0;
    const int32_t len   = u1 - u0 + left + right;
    // Xext is used only for the first and last rows; interior rows use in-place transform.
    auto *Xext = static_cast<sprec_t *>(aligned_mem_alloc(
        sizeof(sprec_t) * static_cast<size_t>(round_up(len + SIMD_PADDING, SIMD_PADDING)), 32));
    for (int32_t row = 0; row < nrows; ++row) {
      if (row == 0 || row == nrows - 1) {
        // First/last rows: use copy-based path (in[-left] or in[width+SIMD_LEN_I32-1] may be
        // outside the tile allocation for the first and last rows respectively).
        fdwt_1d_sr_fixed(Xext, in, left, right, u0, u1, transformation);
      } else {
        fdwt_1d_sr_inplace(in, left, right, u0, u1, transformation);
      }
      in += stride;
    }
    aligned_mem_free(Xext);
  }
}

void fdwt_irrev_ver_sr_fixed(sprec_t *in, const int32_t u0, const int32_t u1, const int32_t v0,
                             const int32_t v1, const int32_t stride, sprec_t *pse_scratch) {
  constexpr int32_t num_pse_i0[2] = {4, 3};
  constexpr int32_t num_pse_i1[2] = {3, 4};
  const int32_t top               = num_pse_i0[v0 % 2];
  const int32_t bottom            = num_pse_i1[v1 % 2];

  if (v0 == v1 - 1) {
    // one sample case
    for (int32_t col = 0; col < u1 - u0; ++col) {
      if (v0 % 2) {
        // in[col] <<= 0;
      }
    }
  } else {
    const int32_t len = round_up(stride, SIMD_LEN_I32);
    auto **buf        = new sprec_t *[static_cast<size_t>(top + v1 - v0 + bottom)];
    for (int32_t i = 1; i <= top; ++i) {
      // buf[top - i] =
      //     static_cast<sprec_t *>(aligned_mem_alloc(sizeof(sprec_t) * static_cast<size_t>(len), 32));
      buf[top - i] = pse_scratch + (i - 1) * len;
      memcpy(buf[top - i], &in[PSEo(v0 - i, v0, v1) * stride],
             sizeof(sprec_t) * static_cast<size_t>(stride));
      // buf[top - i] = &in[(PSEo(v0 - i, v0, v1) - v0) * stride];
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
    const int32_t start  = ceil_int(v0, 2);
    const int32_t stop   = ceil_int(v1, 2);
    const int32_t offset = top + v0 % 2;

    const int32_t width = u1 - u0;
    for (int32_t cs = 0; cs < width; cs += DWT_VERT_STRIP) {
      const int32_t ce = (cs + DWT_VERT_STRIP < width) ? cs + DWT_VERT_STRIP : width;
      for (int32_t n = -4 + offset, i = start - 2; i < stop + 1; i++, n += 2) {
        for (int32_t col = cs; col < ce; ++col) {
          buf[n + 1][col] += fA * (buf[n][col] + buf[n + 2][col]);
        }
      }
      for (int32_t n = -2 + offset, i = start - 1; i < stop + 1; i++, n += 2) {
        for (int32_t col = cs; col < ce; ++col) {
          buf[n][col] += fB * (buf[n - 1][col] + buf[n + 1][col]);
        }
      }
      for (int32_t n = -2 + offset, i = start - 1; i < stop; i++, n += 2) {
        for (int32_t col = cs; col < ce; ++col) {
          buf[n + 1][col] += fC * (buf[n][col] + buf[n + 2][col]);
        }
      }
      for (int32_t n = 0 + offset, i = start; i < stop; i++, n += 2) {
        for (int32_t col = cs; col < ce; ++col) {
          buf[n][col] += fD * (buf[n - 1][col] + buf[n + 1][col]);
        }
      }
    }

    // for (int32_t i = 1; i <= top; ++i) {
    //   aligned_mem_free(buf[top - i]);
    // }
    // for (int32_t i = 1; i <= bottom; i++) {
    //   aligned_mem_free(buf[top + (v1 - v0) + i - 1]);
    // }
    delete[] buf;
  }
}

void fdwt_rev_ver_sr_fixed(sprec_t *in, const int32_t u0, const int32_t u1, const int32_t v0,
                           const int32_t v1, const int32_t stride, sprec_t *pse_scratch) {
  constexpr int32_t num_pse_i0[2] = {2, 1};
  constexpr int32_t num_pse_i1[2] = {1, 2};
  const int32_t top               = num_pse_i0[v0 % 2];
  const int32_t bottom            = num_pse_i1[v1 % 2];

  if (v0 == v1 - 1) {
    // one sample case
    for (int32_t col = 0; col < u1 - u0; ++col) {
      if (v0 % 2) {
        in[col] = floorf(in[col] * 2.0f);
      }
    }
  } else {
    const int32_t len = round_up(stride, SIMD_PADDING);
    auto **buf        = new sprec_t *[static_cast<size_t>(top + v1 - v0 + bottom)];
    for (int32_t i = 1; i <= top; ++i) {
      // buf[top - i] =
      //     static_cast<sprec_t *>(aligned_mem_alloc(sizeof(sprec_t) * static_cast<size_t>(len), 32));
      buf[top - i] = pse_scratch + (i - 1) * len;
      memcpy(buf[top - i], &in[PSEo(v0 - i, v0, v1) * stride],
             sizeof(sprec_t) * static_cast<size_t>(stride));
      // buf[top - i] = &in[(PSEo(v0 - i, v0, v1) - v0) * stride];
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
    const int32_t start  = ceil_int(v0, 2);
    const int32_t stop   = ceil_int(v1, 2);
    const int32_t offset = top + v0 % 2;

    const int32_t width = u1 - u0;
    for (int32_t cs = 0; cs < width; cs += DWT_VERT_STRIP) {
      const int32_t ce = (cs + DWT_VERT_STRIP < width) ? cs + DWT_VERT_STRIP : width;
      for (int32_t n = -2 + offset, i = start - 1; i < stop; ++i, n += 2) {
        for (int32_t col = cs; col < ce; ++col) {
          buf[n + 1][col] -= floorf((buf[n][col] + buf[n + 2][col]) * 0.5f);
        }
      }
      for (int32_t n = 0 + offset, i = start; i < stop; ++i, n += 2) {
        for (int32_t col = cs; col < ce; ++col) {
          buf[n][col] += floorf((buf[n - 1][col] + buf[n + 1][col] + 2) * 0.25f);
        }
      }
    }

    // for (int32_t i = 1; i <= top; ++i) {
    //   aligned_mem_free(buf[top - i]);
    // }
    // for (int32_t i = 1; i <= bottom; i++) {
    //   aligned_mem_free(buf[top + (v1 - v0) + i - 1]);
    // }
    delete[] buf;
  }
}

// Deinterleaving to devide coefficients into subbands
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
        for (int32_t u = 0, ub = ustart[b]; ub < ustop[b]; ++ub, ++u) {
          *(line++) = buf[2 * u + uoffset[b] + (2 * v + voffset[b]) * stride];
        }
      }
    }
  } else {
    sprec_t *first, *second;
    first  = dp[0];
    second = dp[1];
    if (uoffset[0] > uoffset[1]) {
      first  = dp[1];
      second = dp[0];
    }
    for (int32_t v = 0, vb = vstart[0]; vb < vstop[0]; ++vb, ++v) {
      sprec_t *sp    = buf + (2 * v + voffset[0]) * stride;
      size_t len     = static_cast<size_t>(ustop[0] - ustart[0]);
      sprec_t *line0 = first + v * stride2[0];
      sprec_t *line1 = second + v * stride2[0];
      for (; len >= 4; len -= 4) {
        auto vline = vld2q_f32(sp);
        vst1q_f32(line0, vline.val[0]);
        vst1q_f32(line1, vline.val[1]);
        line0 += 4;
        line1 += 4;
        sp += 8;
      }
      for (; len > 0; --len) {
        *line0++ = *sp++;
        *line1++ = *sp++;
      }
    }
  }

  if ((ustop[2] - ustart[2]) != (ustop[3] - ustart[3])) {
    for (uint8_t b = 2; b < 4; ++b) {
      for (int32_t v = 0, vb = vstart[b]; vb < vstop[b]; ++vb, ++v) {
        sprec_t *line = dp[b] + v * stride2[b];
        for (int32_t u = 0, ub = ustart[b]; ub < ustop[b]; ++ub, ++u) {
          *(line++) = buf[2 * u + uoffset[b] + (2 * v + voffset[b]) * stride];
        }
      }
    }
  } else {
    sprec_t *first, *second;
    first  = dp[2];
    second = dp[3];
    if (uoffset[2] > uoffset[3]) {
      first  = dp[3];
      second = dp[2];
    }
    for (int32_t v = 0, vb = vstart[2]; vb < vstop[2]; ++vb, ++v) {
      sprec_t *sp    = buf + (2 * v + voffset[2]) * stride;
      size_t len     = static_cast<size_t>(ustop[2] - ustart[2]);
      sprec_t *line0 = first + v * stride2[2];
      sprec_t *line1 = second + v * stride2[2];
      for (; len >= 4; len -= 4) {
        auto vline = vld2q_f32(sp);
        vst1q_f32(line0, vline.val[0]);
        vst1q_f32(line1, vline.val[1]);
        line0 += 4;
        line1 += 4;
        sp += 8;
      }
      for (; len > 0; --len) {
        *line0++ = *sp++;
        *line1++ = *sp++;
      }
    }
  }
#elif defined(OPENHTJ2K_TRY_AVX2) && defined(__AVX2__)
  if ((ustop[0] - ustart[0]) != (ustop[1] - ustart[1])) {
    for (uint8_t b = 0; b < 2; ++b) {
      for (int32_t v = 0, vb = vstart[b]; vb < vstop[b]; ++vb, ++v) {
        sprec_t *line = dp[b] + v * stride2[b];
        for (int32_t u = 0, ub = ustart[b]; ub < ustop[b]; ++ub, ++u) {
          *(line++) = buf[2 * u + uoffset[b] + (2 * v + voffset[b]) * stride];
        }
      }
    }
  } else {
    sprec_t *first, *second;
    first  = dp[0];
    second = dp[1];
    if (uoffset[0] > uoffset[1]) {
      first  = dp[1];
      second = dp[0];
    }
    for (int32_t v = 0, vb = vstart[0]; vb < vstop[0]; ++vb, ++v) {
      sprec_t *sp    = buf + (2 * v + voffset[0]) * stride;
      size_t len     = static_cast<size_t>(ustop[0] - ustart[0]);
      sprec_t *line0 = first + v * stride2[0];
      sprec_t *line1 = second + v * stride2[0];
      for (; len >= 4; len -= 4) {
        __m256i vline = _mm256_loadu_si256((__m256i *)sp);
        vline         = _mm256_shuffle_epi32(vline, 0xD8);
        vline         = _mm256_permute4x64_epi64(vline, 0xD8);
        _mm256_storeu2_m128i((__m128i *)line1, (__m128i *)line0, vline);
        line0 += 4;
        line1 += 4;
        sp += 8;
      }
      for (; len > 0; --len) {
        *line0++ = *sp++;
        *line1++ = *sp++;
      }
    }
  }

  if ((ustop[2] - ustart[2]) != (ustop[3] - ustart[3])) {
    for (uint8_t b = 2; b < 4; ++b) {
      for (int32_t v = 0, vb = vstart[b]; vb < vstop[b]; ++vb, ++v) {
        sprec_t *line = dp[b] + v * stride2[b];
        for (int32_t u = 0, ub = ustart[b]; ub < ustop[b]; ++ub, ++u) {
          *(line++) = buf[2 * u + uoffset[b] + (2 * v + voffset[b]) * stride];
        }
      }
    }
  } else {
    sprec_t *first, *second;
    first  = dp[2];
    second = dp[3];
    if (uoffset[2] > uoffset[3]) {
      first  = dp[3];
      second = dp[2];
    }
    for (int32_t v = 0, vb = vstart[2]; vb < vstop[2]; ++vb, ++v) {
      sprec_t *sp    = buf + (2 * v + voffset[2]) * stride;
      size_t len     = static_cast<size_t>(ustop[2] - ustart[2]);
      sprec_t *line0 = first + v * stride2[2];
      sprec_t *line1 = second + v * stride2[2];
      for (; len >= 4; len -= 4) {
        __m256i vline = _mm256_loadu_si256((__m256i *)sp);
        vline         = _mm256_shuffle_epi32(vline, 0xD8);
        vline         = _mm256_permute4x64_epi64(vline, 0xD8);
        _mm256_storeu2_m128i((__m128i *)line1, (__m128i *)line0, vline);
        line0 += 4;
        line1 += 4;
        sp += 8;
      }
      for (; len > 0; --len) {
        *line0++ = *sp++;
        *line1++ = *sp++;
      }
    }
  }
#else
  for (uint8_t b = 0; b < 4; ++b) {
    for (int32_t v = 0, vb = vstart[b]; vb < vstop[b]; ++vb, ++v) {
      sprec_t *line = dp[b] + v * stride2[b];
      for (int32_t u = 0, ub = ustart[b]; ub < ustop[b]; ++ub, ++u) {
        *(line++) = buf[2 * u + uoffset[b] + (2 * v + voffset[b]) * stride];
      }
    }
  }
#endif
}

// 2D FDWT function
void fdwt_2d_sr_fixed(sprec_t *previousLL, sprec_t *LL, sprec_t *HL, sprec_t *LH, sprec_t *HH,
                      const int32_t u0, const int32_t u1, const int32_t v0, const int32_t v1,
                      const uint8_t transformation) {
  const int32_t stride = round_up(u1 - u0, 32);
  sprec_t *src         = previousLL;

  // Vertical DWT
  // scratch buffer for symmetric extension — allocate once:
  const int32_t pse_rows = 8;  // max for irrev97: top=4 + bottom=4
  const int32_t pse_len  = round_up(stride, SIMD_LEN_I32);
  auto *pse_scratch = static_cast<sprec_t*>(aligned_mem_alloc(sizeof(sprec_t) * pse_rows * static_cast<size_t>(pse_len), 32));
  fdwt_ver_sr_fixed[transformation](src, u0, u1, v0, v1, stride, pse_scratch);
  aligned_mem_free(pse_scratch);

  // Horizontal DWT
  fdwt_hor_sr_fixed(src, u0, u1, v0, v1, transformation, stride);

  fdwt_2d_deinterleave_fixed(src, LL, HL, LH, HH, u0, u1, v0, v1, stride);
}