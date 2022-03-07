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
auto fdwt_1d_filtr_irrev97_fixed_neon = [](sprec_t *X, const int32_t left, const int32_t right,
                                           const uint32_t u_i0, const uint32_t u_i1) {
  const auto i0       = static_cast<const int32_t>(u_i0);
  const auto i1       = static_cast<const int32_t>(u_i1);
  const int32_t start = ceil_int(i0, 2);
  const int32_t stop  = ceil_int(i1, 2);

  const int32_t offset = left + i0 % 2;
  int32_t simdlen      = stop + 1 - (start - 2);
  // step 1: simd
  for (int32_t n = -4 + offset, i = 0; i < simdlen - simdlen % 16; i += 8, n += 16) {
    auto xl0     = vld2q_s16(X + n);
    auto xl1     = vld2q_s16(X + n + 2);
    auto vcoeff  = vdupq_n_s32(Acoeff);
    auto voffset = vdupq_n_s32(Aoffset);
    auto x0      = vreinterpretq_s32_s16(xl0.val[0]);
    auto x0l     = vmovl_s16(vreinterpret_s16_s32(vget_low_s32(x0)));
    auto x0h     = vmovl_s16(vreinterpret_s16_s32(vget_high_s32(x0)));
    auto x2      = vreinterpretq_s32_s16(xl1.val[0]);
    auto x2l     = vmovl_s16(vreinterpret_s16_s32(vget_low_s32(x2)));
    auto x2h     = vmovl_s16(vreinterpret_s16_s32(vget_high_s32(x2)));
    auto xoutl   = ((x0l + x2l) * vcoeff + voffset) >> Ashift;
    auto xouth   = ((x0h + x2h) * vcoeff + voffset) >> Ashift;
    xl0.val[1] += vcombine_s16(vmovn_s32(xoutl), vmovn_s32(xouth));
    vst2q_s16(X + n, xl0);
  }
  // step 1: remaining
  for (int32_t n = -4 + offset + (simdlen - simdlen % 16) * 2, i = 0; i < simdlen % 16; i++, n += 2) {
    int32_t sum = X[n];
    sum += X[n + 2];
    X[n + 1] += (sprec_t)((Acoeff * sum + Aoffset) >> Ashift);
  }
  // step 2: simd
  simdlen = stop + 1 - (start - 1);
  for (int32_t n = -2 + offset, i = 0; i < simdlen - simdlen % 16; i += 8, n += 16) {
    auto xl0     = vld2q_s16(X + n - 1);
    auto xl1     = vld2q_s16(X + n + 1);
    auto vcoeff  = vdupq_n_s32(Bcoeff);
    auto voffset = vdupq_n_s32(Boffset);
    auto x0      = vreinterpretq_s32_s16(xl0.val[0]);
    auto x0l     = vmovl_s16(vreinterpret_s16_s32(vget_low_s32(x0)));
    auto x0h     = vmovl_s16(vreinterpret_s16_s32(vget_high_s32(x0)));
    auto x2      = vreinterpretq_s32_s16(xl1.val[0]);
    auto x2l     = vmovl_s16(vreinterpret_s16_s32(vget_low_s32(x2)));
    auto x2h     = vmovl_s16(vreinterpret_s16_s32(vget_high_s32(x2)));
    auto xoutl   = ((x0l + x2l) * vcoeff + voffset) >> Bshift;
    auto xouth   = ((x0h + x2h) * vcoeff + voffset) >> Bshift;
    xl0.val[1] += vcombine_s16(vmovn_s32(xoutl), vmovn_s32(xouth));
    vst2q_s16(X + n - 1, xl0);
  }
  // step 2: remaining
  for (int32_t n = -2 + offset + (simdlen - simdlen % 16) * 2, i = 0; i < simdlen % 16; i++, n += 2) {
    int32_t sum = X[n - 1];
    sum += X[n + 1];
    X[n] += (sprec_t)((Bcoeff * sum + Boffset) >> Bshift);
  }
  // step 3: simd
  simdlen = stop - (start - 1);
  for (int32_t n = -2 + offset, i = 0; i < simdlen - simdlen % 16; i += 8, n += 16) {
    auto xl0     = vld2q_s16(X + n);
    auto xl1     = vld2q_s16(X + n + 2);
    auto vcoeff  = vdupq_n_s32(Ccoeff);
    auto voffset = vdupq_n_s32(Coffset);
    auto x0      = vreinterpretq_s32_s16(xl0.val[0]);
    auto x0l     = vmovl_s16(vreinterpret_s16_s32(vget_low_s32(x0)));
    auto x0h     = vmovl_s16(vreinterpret_s16_s32(vget_high_s32(x0)));
    auto x2      = vreinterpretq_s32_s16(xl1.val[0]);
    auto x2l     = vmovl_s16(vreinterpret_s16_s32(vget_low_s32(x2)));
    auto x2h     = vmovl_s16(vreinterpret_s16_s32(vget_high_s32(x2)));
    auto xoutl   = ((x0l + x2l) * vcoeff + voffset) >> Cshift;
    auto xouth   = ((x0h + x2h) * vcoeff + voffset) >> Cshift;
    xl0.val[1] += vcombine_s16(vmovn_s32(xoutl), vmovn_s32(xouth));
    vst2q_s16(X + n, xl0);
  }
  // step 3: remaining
  for (int32_t n = -2 + offset + (simdlen - simdlen % 16) * 2, i = 0; i < simdlen % 16; i++, n += 2) {
    int32_t sum = X[n];
    sum += X[n + 2];
    X[n + 1] += (sprec_t)((Ccoeff * sum + Coffset) >> Cshift);
  }
  // step 4: simd
  simdlen = stop - start;
  for (int32_t n = 0 + offset, i = 0; i < simdlen - simdlen % 16; i += 8, n += 16) {
    auto xl0     = vld2q_s16(X + n - 1);
    auto xl1     = vld2q_s16(X + n + 1);
    auto vcoeff  = vdupq_n_s32(Dcoeff);
    auto voffset = vdupq_n_s32(Doffset);
    auto x0      = vreinterpretq_s32_s16(xl0.val[0]);
    auto x0l     = vmovl_s16(vreinterpret_s16_s32(vget_low_s32(x0)));
    auto x0h     = vmovl_s16(vreinterpret_s16_s32(vget_high_s32(x0)));
    auto x2      = vreinterpretq_s32_s16(xl1.val[0]);
    auto x2l     = vmovl_s16(vreinterpret_s16_s32(vget_low_s32(x2)));
    auto x2h     = vmovl_s16(vreinterpret_s16_s32(vget_high_s32(x2)));
    auto xoutl   = ((x0l + x2l) * vcoeff + voffset) >> Dshift;
    auto xouth   = ((x0h + x2h) * vcoeff + voffset) >> Dshift;
    xl0.val[1] += vcombine_s16(vmovn_s32(xoutl), vmovn_s32(xouth));
    vst2q_s16(X + n - 1, xl0);
  }
  // step 4: remaining
  for (int32_t n = 0 + offset + (simdlen - simdlen % 16) * 2, i = 0; i < simdlen % 16; i++, n += 2) {
    int32_t sum = X[n - 1];
    sum += X[n + 1];
    X[n] += (sprec_t)((Dcoeff * sum + Doffset) >> Dshift);
  }
};
#endif
// irreversible FDWT
auto fdwt_1d_filtr_irrev97_fixed = [](sprec_t *X, const int32_t left, const int32_t right,
                                      const uint32_t u_i0, const uint32_t u_i1) {
  const auto i0       = static_cast<const int32_t>(u_i0);
  const auto i1       = static_cast<const int32_t>(u_i1);
  const int32_t start = ceil_int(i0, 2);
  const int32_t stop  = ceil_int(i1, 2);

  const int32_t offset = left + i0 % 2;
  for (int32_t n = -4 + offset, i = start - 2; i < stop + 1; i++, n += 2) {
    int32_t sum = X[n];
    sum += X[n + 2];
    X[n + 1] += (sprec_t)((Acoeff * sum + Aoffset) >> Ashift);
  }
  for (int32_t n = -2 + offset, i = start - 1; i < stop + 1; i++, n += 2) {
    int32_t sum = X[n - 1];
    sum += X[n + 1];
    X[n] += (sprec_t)((Bcoeff * sum + Boffset) >> Bshift);
  }
  for (int32_t n = -2 + offset, i = start - 1; i < stop; i++, n += 2) {
    int32_t sum = X[n];
    sum += X[n + 2];
    X[n + 1] += (sprec_t)((Ccoeff * sum + Coffset) >> Cshift);
  }
  for (int32_t n = 0 + offset, i = start; i < stop; i++, n += 2) {
    int32_t sum = X[n - 1];
    sum += X[n + 1];
    X[n] += (sprec_t)((Dcoeff * sum + Doffset) >> Dshift);
  }
};

// reversible FDWT
auto fdwt_1d_filtr_rev53_fixed = [](sprec_t *X, const int32_t left, const int32_t right,
                                    const uint32_t u_i0, const uint32_t u_i1) {
  const auto i0       = static_cast<const int32_t>(u_i0);
  const auto i1       = static_cast<const int32_t>(u_i1);
  const int32_t start = ceil_int(i0, 2);
  const int32_t stop  = ceil_int(i1, 2);
  // X += left - i0 % 2;
  const int32_t offset = left + i0 % 2;
  for (int32_t n = -2 + offset, i = start - 1; i < stop; ++i, n += 2) {
    int32_t sum = X[n];
    sum += X[n + 2];
    X[n + 1] -= (sum >> 1);
  }
  for (int32_t n = 0 + offset, i = start; i < stop; ++i, n += 2) {
    int32_t sum = X[n - 1];
    sum += X[n + 1];
    X[n] += ((sum + 2) >> 2);
  }
};
#if defined(OPENHTJ2K_ENABLE_ARM_NEON)
static fdwt_1d_filtr_func_fixed fdwt_1d_filtr_fixed[2] = {fdwt_1d_filtr_irrev97_fixed_neon,
                                                          fdwt_1d_filtr_rev53_fixed};
#else
static fdwt_1d_filtr_func_fixed fdwt_1d_filtr_fixed[2] = {fdwt_1d_filtr_irrev97_fixed,
                                                          fdwt_1d_filtr_rev53_fixed};
#endif

// 1-dimensional FDWT
static inline void fdwt_1d_sr_fixed(sprec_t *buf, sprec_t *in, sprec_t *out, const int32_t left,
                                    const int32_t right, const uint32_t i0, const uint32_t i1,
                                    const uint8_t transformation) {
  //  const uint32_t len = round_up(i1 - i0 + left + right, SIMD_LEN_I16);
  //  auto *Xext         = static_cast<int16_t *>(aligned_mem_alloc(sizeof(int16_t) * len, 32));
  dwt_1d_extr_fixed(buf, in, left, right, i0, i1);
  fdwt_1d_filtr_fixed[transformation](buf, left, right, i0, i1);
  memcpy(out, buf + left, sizeof(sprec_t) * (i1 - i0));
  //  aligned_mem_free(Xext);
}

// FDWT for horizontal direction
static void fdwt_hor_sr_fixed(sprec_t *out, sprec_t *in, const uint32_t u0, const uint32_t u1,
                              const uint32_t v0, const uint32_t v1, const uint8_t transformation) {
  const uint32_t stride              = u1 - u0;
  constexpr int32_t num_pse_i0[2][2] = {{4, 2}, {3, 1}};
  constexpr int32_t num_pse_i1[2][2] = {{3, 1}, {4, 2}};
  const int32_t left                 = num_pse_i0[u0 % 2][transformation];
  const int32_t right                = num_pse_i1[u1 % 2][transformation];

  if (u0 == u1 - 1) {
    // one sample case
    const float K  = (transformation) ? 1 : 1.2301741 / 2;  // 04914001;
    const float K1 = (transformation) ? 1 : 0.8128931;      // 066115961;
    for (uint32_t row = 0; row < v1 - v0; ++row) {
      if (u0 % 2 == 0) {
        out[row] = (transformation) ? in[row] : (sprec_t)roundf(static_cast<float>(in[row]) * K1);
      } else {
        out[row] = (transformation) ? in[row] << 1 : (sprec_t)roundf(static_cast<float>(in[row]) * 2 * K);
      }
    }
  } else {
    // need to perform symmetric extension
    const uint32_t len = round_up(u1 - u0 + left + right, SIMD_LEN_I32);
    auto *Xext         = static_cast<sprec_t *>(aligned_mem_alloc(sizeof(sprec_t) * len, 32));
    //#pragma omp parallel for
    for (uint32_t row = 0; row < v1 - v0; ++row) {
      fdwt_1d_sr_fixed(Xext, &in[row * stride], &out[row * stride], left, right, u0, u1, transformation);
    }
    aligned_mem_free(Xext);
  }
}

auto fdwt_irrev_ver_sr_fixed = [](sprec_t *in, const uint32_t u0, const uint32_t u1, const uint32_t v0,
                                  const uint32_t v1) {
  const uint32_t stride           = u1 - u0;
  constexpr int32_t num_pse_i0[2] = {4, 3};
  constexpr int32_t num_pse_i1[2] = {3, 4};
  const int32_t top               = num_pse_i0[v0 % 2];
  const int32_t bottom            = num_pse_i1[v1 % 2];

  if (v0 == v1 - 1) {
    // one sample case
    constexpr float K  = 1.2301741 / 2;  // 04914001;
    constexpr float K1 = 0.8128931;      // 066115961;
    for (uint32_t col = 0; col < u1 - u0; ++col) {
      if (v0 % 2 == 0) {
        in[col] = (sprec_t)roundf(static_cast<float>(in[col]) * K1);
      } else {
        in[col] = (sprec_t)roundf(static_cast<float>(in[col]) * 2 * K);
      }
    }
  } else {
    const uint32_t len = round_up(stride, SIMD_LEN_I32);
    auto **buf         = new sprec_t *[top + v1 - v0 + bottom];
    for (uint32_t i = 1; i <= top; ++i) {
      buf[top - i] = static_cast<sprec_t *>(aligned_mem_alloc(sizeof(sprec_t) * len, 32));
      memcpy(buf[top - i], &in[(PSEo(v0 - i, v0, v1) - v0) * stride], sizeof(sprec_t) * stride);
      // buf[top - i] = &in[(PSEo(v0 - i, v0, v1) - v0) * stride];
    }
    for (uint32_t row = 0; row < v1 - v0; ++row) {
      buf[top + row] = &in[row * stride];
    }
    for (uint32_t i = 1; i <= bottom; i++) {
      buf[top + (v1 - v0) + i - 1] = static_cast<sprec_t *>(aligned_mem_alloc(sizeof(sprec_t) * len, 32));
      memcpy(buf[top + (v1 - v0) + i - 1], &in[(PSEo(v1 - v0 + i - 1 + v0, v0, v1) - v0) * stride],
             sizeof(sprec_t) * stride);
    }
    const int32_t start  = ceil_int(v0, 2);
    const int32_t stop   = ceil_int(v1, 2);
    const int32_t offset = top + v0 % 2;

    for (int32_t n = -4 + offset, i = start - 2; i < stop + 1; i++, n += 2) {
      for (uint32_t col = 0; col < u1 - u0; ++col) {
        int32_t sum = buf[n][col];
        sum += buf[n + 2][col];
        buf[n + 1][col] += (sprec_t)((Acoeff * sum + Aoffset) >> Ashift);
      }
    }
    for (int32_t n = -2 + offset, i = start - 1; i < stop + 1; i++, n += 2) {
      for (uint32_t col = 0; col < u1 - u0; ++col) {
        int32_t sum = buf[n - 1][col];
        sum += buf[n + 1][col];
        buf[n][col] += (sprec_t)((Bcoeff * sum + Boffset) >> Bshift);
      }
    }
    for (int32_t n = -2 + offset, i = start - 1; i < stop; i++, n += 2) {
      for (uint32_t col = 0; col < u1 - u0; ++col) {
        int32_t sum = buf[n][col];
        sum += buf[n + 2][col];
        buf[n + 1][col] += (sprec_t)((Ccoeff * sum + Coffset) >> Cshift);
      }
    }
    for (int32_t n = 0 + offset, i = start; i < stop; i++, n += 2) {
      for (uint32_t col = 0; col < u1 - u0; ++col) {
        int32_t sum = buf[n - 1][col];
        sum += buf[n + 1][col];
        buf[n][col] += (sprec_t)((Dcoeff * sum + Doffset) >> Dshift);
      }
    }

    for (uint32_t i = 1; i <= top; ++i) {
      aligned_mem_free(buf[top - i]);
    }
    for (uint32_t i = 1; i <= bottom; i++) {
      aligned_mem_free(buf[top + (v1 - v0) + i - 1]);
    }
    delete[] buf;
  }
};

auto fdwt_rev_ver_sr_fixed = [](sprec_t *in, const uint32_t u0, const uint32_t u1, const uint32_t v0,
                                const uint32_t v1) {
  const uint32_t stride           = u1 - u0;
  constexpr int32_t num_pse_i0[2] = {2, 1};
  constexpr int32_t num_pse_i1[2] = {1, 2};
  const int32_t top               = num_pse_i0[v0 % 2];
  const int32_t bottom            = num_pse_i1[v1 % 2];

  if (v0 == v1 - 1) {
    // one sample case
    for (uint32_t col = 0; col < u1 - u0; ++col) {
      if (v0 % 2 == 0) {
        in[col] = in[col];
      } else {
        in[col] = in[col] << 1;
      }
    }
  } else {
    const uint32_t len = round_up(stride, SIMD_LEN_I16);
    auto **buf         = new sprec_t *[top + v1 - v0 + bottom];
    for (uint32_t i = 1; i <= top; ++i) {
      buf[top - i] = static_cast<sprec_t *>(aligned_mem_alloc(sizeof(sprec_t) * len, 32));
      memcpy(buf[top - i], &in[(PSEo(v0 - i, v0, v1) - v0) * stride], sizeof(sprec_t) * stride);
      // buf[top - i] = &in[(PSEo(v0 - i, v0, v1) - v0) * stride];
    }
    for (uint32_t row = 0; row < v1 - v0; ++row) {
      buf[top + row] = &in[row * stride];
    }
    for (uint32_t i = 1; i <= bottom; i++) {
      buf[top + (v1 - v0) + i - 1] = static_cast<sprec_t *>(aligned_mem_alloc(sizeof(sprec_t) * len, 32));
      memcpy(buf[top + (v1 - v0) + i - 1], &in[(PSEo(v1 - v0 + i - 1 + v0, v0, v1) - v0) * stride],
             sizeof(sprec_t) * stride);
    }
    const int32_t start  = ceil_int(v0, 2);
    const int32_t stop   = ceil_int(v1, 2);
    const int32_t offset = top + v0 % 2;

    for (int32_t n = -2 + offset, i = start - 1; i < stop; ++i, n += 2) {
      for (uint32_t col = 0; col < u1 - u0; ++col) {
        int32_t sum = buf[n][col];
        sum += buf[n + 2][col];
        buf[n + 1][col] -= (sum >> 1);
      }
    }
    for (int32_t n = 0 + offset, i = start; i < stop; ++i, n += 2) {
      for (uint32_t col = 0; col < u1 - u0; ++col) {
        int32_t sum = buf[n - 1][col];
        sum += buf[n + 1][col];
        buf[n][col] += ((sum + 2) >> 2);
      }
    }

    for (uint32_t i = 1; i <= top; ++i) {
      aligned_mem_free(buf[top - i]);
    }
    for (uint32_t i = 1; i <= bottom; i++) {
      aligned_mem_free(buf[top + (v1 - v0) + i - 1]);
    }
    delete[] buf;
  }
};

static fdwt_ver_filtr_func_fixed fdwt_ver_sr_fixed[2] = {fdwt_irrev_ver_sr_fixed, fdwt_rev_ver_sr_fixed};

// Deinterleaving to devide coefficients into subbands
static void fdwt_2d_deinterleave_fixed(const sprec_t *buf, sprec_t *const LL, sprec_t *const HL,
                                       sprec_t *const LH, sprec_t *const HH, const uint32_t u0,
                                       const uint32_t u1, const uint32_t v0, const uint32_t v1,
                                       const uint8_t transformation) {
  const uint32_t stride     = u1 - u0;
  const uint32_t v_offset   = v0 % 2;
  const uint32_t u_offset   = u0 % 2;
  sprec_t *dp[4]            = {LL, HL, LH, HH};
  const uint32_t vstart[4]  = {ceil_int(v0, 2), ceil_int(v0, 2), v0 / 2, v0 / 2};
  const uint32_t vstop[4]   = {ceil_int(v1, 2), ceil_int(v1, 2), v1 / 2, v1 / 2};
  const uint32_t ustart[4]  = {ceil_int(u0, 2), u0 / 2, ceil_int(u0, 2), u0 / 2};
  const uint32_t ustop[4]   = {ceil_int(u1, 2), u1 / 2, ceil_int(u1, 2), u1 / 2};
  const uint32_t voffset[4] = {v_offset, v_offset, 1 - v_offset, 1 - v_offset};
  const uint32_t uoffset[4] = {u_offset, 1 - u_offset, u_offset, 1 - u_offset};

  for (uint8_t b = 0; b < 4; ++b) {
    for (uint32_t v = 0, vb = vstart[b]; vb < vstop[b]; ++vb, ++v) {
      for (uint32_t u = 0, ub = ustart[b]; ub < ustop[b]; ++ub, ++u) {
        *(dp[b]++) = buf[2 * u + uoffset[b] + (2 * v + voffset[b]) * stride];
      }
    }
  }
#ifdef BETTERQUANT
  // TODO: One sample case shall be considered. Currently implementation is not correct for 1xn or nx1 or
  // 1x1..
  constexpr float K     = 1.2301741 / 2;
  constexpr float K1    = 0.8128931;
  constexpr float KK[4] = {K1 * K1, K * K1, K1 * K, K * K};
  if (transformation) {
  #pragma omp parallel for  //default(none) \
    shared(dp, vstart, ustart, vstop, ustop, uoffset, voffset, stride, buf)
    for (uint8_t b = 0; b < 4; ++b) {
      for (uint32_t v = 0, vb = vstart[b]; vb < vstop[b]; ++vb, ++v) {
        for (uint32_t u = 0, ub = ustart[b]; ub < ustop[b]; ++ub, ++u) {
          *(dp[b]++) = buf[2 * u + uoffset[b] + (2 * v + voffset[b]) * stride];
        }
      }
    }
  } else {
  #pragma omp parallel for  //default(none) \
    shared(dp, vstart, ustart, vstop, ustop, uoffset, voffset, stride, buf)
    for (uint8_t b = 0; b < 4; ++b) {
      for (uint32_t v = 0, vb = vstart[b]; vb < vstop[b]; ++vb, ++v) {
        for (uint32_t u = 0, ub = ustart[b]; ub < ustop[b]; ++ub, ++u) {
          int16_t val  = buf[2 * u + uoffset[b] + (2 * v + voffset[b]) * stride];
          int16_t sign = val & 0x8000;
          val          = (val < 0) ? -val & 0x7FFF : val;
          val          = static_cast<int16_t>(val * KK[b] + 0.5);
          if (sign) {
            val = -val;
          }
          *(dp[b]++) = val;
          // TODO: LL band shall be scaled for 16bit fixed-point implementation
        }
      }
    }
  }
#endif
}

// 2D FDWT function
void fdwt_2d_sr_fixed(sprec_t *previousLL, sprec_t *LL, sprec_t *HL, sprec_t *LH, sprec_t *HH,
                      const uint32_t u0, const uint32_t u1, const uint32_t v0, const uint32_t v1,
                      const uint8_t transformation) {
  const uint32_t buf_length = (u1 - u0) * (v1 - v0);
  sprec_t *src              = previousLL;
  auto *dst                 = static_cast<sprec_t *>(aligned_mem_alloc(sizeof(sprec_t) * buf_length, 32));
  fdwt_ver_sr_fixed[transformation](src, u0, u1, v0, v1);
  fdwt_hor_sr_fixed(dst, src, u0, u1, v0, v1, transformation);
  fdwt_2d_deinterleave_fixed(dst, LL, HL, LH, HH, u0, u1, v0, v1, transformation);
  aligned_mem_free(dst);
}