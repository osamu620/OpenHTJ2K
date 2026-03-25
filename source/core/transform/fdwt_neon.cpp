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

#if defined(OPENHTJ2K_ENABLE_ARM_NEON)
#include <cmath>

  #include "dwt.hpp"
  #include "utils.hpp"
  #include <cstring>

/********************************************************************************
 * horizontal transforms
 *******************************************************************************/
// irreversible FDWT
auto fdwt_irrev97_fixed_neon_hor_step = [](const int32_t init_pos, const int32_t simdlen, float *X,
                                            const int32_t n0, const int32_t n1, float coeff) {
  auto vvv = vdupq_n_f32(coeff);
  X += init_pos;
  for (int32_t i = simdlen; i > 0; i -= 4) {
    auto x0   = vld2q_f32(X + n0);
    auto x1   = vld2q_f32(X + n1);
    auto tmp  = vaddq_f32(x0.val[0], x1.val[0]);
    // tmp       = vmulq_f32(tmp, vvv);
    // x0.val[1] = vaddq_f32(x0.val[1], tmp);
    x0.val[1] = vfmaq_f32(x0.val[1], tmp, vvv);
    vst2q_f32(X + n0, x0);
    X += 8;
  }
};

void fdwt_1d_filtr_irrev97_fixed_neon(sprec_t *X, const int32_t left, const int32_t u_i0,
                                      const int32_t u_i1) {
  const auto i0       = static_cast<int32_t>(u_i0);
  const auto i1       = static_cast<int32_t>(u_i1);
  const int32_t start = ceil_int(i0, 2);
  const int32_t stop  = ceil_int(i1, 2);

  const int32_t offset = left + i0 % 2;

  // step 1
  int32_t simdlen = stop + 1 - (start - 2);
  fdwt_irrev97_fixed_neon_hor_step(offset - 4, simdlen, X, 0, 2, fA);

  // step 2
  simdlen = stop + 1 - (start - 1);
  fdwt_irrev97_fixed_neon_hor_step(offset - 2, simdlen, X, -1, 1, fB);

  // step 3
  simdlen = stop - (start - 1);
  fdwt_irrev97_fixed_neon_hor_step(offset - 2, simdlen, X, 0, 2, fC);

  // step 4
  simdlen = stop - start;
  fdwt_irrev97_fixed_neon_hor_step(offset, simdlen, X, -1, 1, fD);
}

// reversible FDWT
void fdwt_1d_filtr_rev53_fixed_neon(sprec_t *X, const int32_t left, const int32_t u_i0,
                                    const int32_t u_i1) {
  const auto i0       = static_cast<int32_t>(u_i0);
  const auto i1       = static_cast<int32_t>(u_i1);
  const int32_t start = ceil_int(i0, 2);
  const int32_t stop  = ceil_int(i1, 2);

  const int32_t offset = left + i0 % 2;
  int32_t simdlen      = stop - (start - 1);
  for (int32_t n = -2 + offset, i = 0; i < simdlen; i += 4, n += 8) {
    auto xl0   = vld2q_f32(X + n);
    auto xl1   = vld2q_f32(X + n + 2);
    // xl0.val[1] = vcvtq_f32_s32(vsubq_s32(vcvtq_s32_f32(xl0.val[1]), vhaddq_s32(vcvtq_s32_f32(xl0.val[0]), vcvtq_s32_f32(xl1.val[0]))));
    auto xfloor = vrndmq_f32(vmulq_n_f32(vaddq_f32(xl0.val[0], xl1.val[0]), 0.5f));
    xl0.val[1] = vsubq_f32(xl0.val[1], xfloor);
    vst2q_f32(X + n, xl0);
  }

  simdlen = stop - start;
  const auto vtwo = vdupq_n_f32(2.0f);
  for (int32_t n = 0 + offset, i = 0; i < simdlen; i += 4, n += 8) {
    auto xl0 = vld2q_f32(X + n - 1);
    auto xl1 = vld2q_f32(X + n + 1);
    // (xl0.val[0] + xl1.val[0] + 2) >> 2;
    auto xfloor = vrndmq_f32(vmulq_n_f32(vaddq_f32(vaddq_f32(xl0.val[0], xl1.val[0]), vtwo), 0.25f));
    // xl0.val[1] = vcvtq_f32_s32(vaddq_s32(vcvtq_s32_f32(xl0.val[1]), vrshrq_n_s32(vhaddq_s32(vcvtq_s32_f32(xl0.val[0]), vcvtq_s32_f32(xl1.val[0])), 1)));
    xl0.val[1] = vaddq_f32(xl0.val[1], xfloor);
    vst2q_f32(X + n - 1, xl0);
  }
}

/********************************************************************************
 * vertical transforms
 *******************************************************************************/
// irreversible FDWT
auto fdwt_irrev97_fixed_neon_ver_step = [](const int32_t simdlen, float *const Xin0, float *const Xin1,
                                            float *const Xout, float coeff) {
  auto vvv = vdupq_n_f32(coeff);
  for (int32_t n = 0; n < simdlen; n += 4) {
    auto x0  = vld1q_f32(Xin0 + n);
    auto x2  = vld1q_f32(Xin1 + n);
    auto x1  = vld1q_f32(Xout + n);
    auto tmp = vaddq_f32(x0, x2);
    x1 = vfmaq_f32(x1, vvv, tmp);
    // tmp      = vmulq_f32(tmp, vvv);
    // x1       = vaddq_f32(x1, tmp);
    vst1q_f32(Xout + n, x1);
  }
};

void fdwt_irrev_ver_sr_fixed_neon(sprec_t *in, const int32_t u0, const int32_t u1, const int32_t v0,
                                  const int32_t v1, const int32_t stride) {
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
      buf[top - i] =
          static_cast<sprec_t *>(aligned_mem_alloc(sizeof(sprec_t) * static_cast<size_t>(len), 32));
      memcpy(buf[top - i], &in[PSEo(v0 - i, v0, v1) * stride],
             sizeof(sprec_t) * static_cast<size_t>(stride));
      // buf[top - i] = &in[(PSEo(v0 - i, v0, v1) - v0) * stride];
    }
    for (int32_t row = 0; row < v1 - v0; ++row) {
      buf[top + row] = &in[row * stride];
    }
    for (int32_t i = 1; i <= bottom; i++) {
      buf[top + (v1 - v0) + i - 1] =
          static_cast<sprec_t *>(aligned_mem_alloc(sizeof(sprec_t) * static_cast<size_t>(len), 32));
      memcpy(buf[top + (v1 - v0) + i - 1], &in[PSEo(v1 - v0 + i - 1 + v0, v0, v1) * stride],
             sizeof(sprec_t) * static_cast<size_t>(stride));
    }
    const int32_t start  = ceil_int(v0, 2);
    const int32_t stop   = ceil_int(v1, 2);
    const int32_t offset = top + v0 % 2;

    const int32_t simdlen = (u1 - u0) - (u1 - u0) % 4;
    for (int32_t n = -4 + offset, i = start - 2; i < stop + 1; i++, n += 2) {
      fdwt_irrev97_fixed_neon_ver_step(simdlen, buf[n], buf[n + 2], buf[n + 1], fA);
      for (int32_t col = simdlen; col < u1 - u0; ++col) {
        float sum = buf[n][col];
        sum += buf[n + 2][col];
        buf[n + 1][col] += fA * sum;
      }
    }
    for (int32_t n = -2 + offset, i = start - 1; i < stop + 1; i++, n += 2) {
      fdwt_irrev97_fixed_neon_ver_step(simdlen, buf[n - 1], buf[n + 1], buf[n], fB);
      for (int32_t col = simdlen; col < u1 - u0; ++col) {
        float sum = buf[n - 1][col];
        sum += buf[n + 1][col];
        buf[n][col] += fB * sum;
      }
    }
    for (int32_t n = -2 + offset, i = start - 1; i < stop; i++, n += 2) {
      fdwt_irrev97_fixed_neon_ver_step(simdlen, buf[n], buf[n + 2], buf[n + 1], fC);
      for (int32_t col = simdlen; col < u1 - u0; ++col) {
        float sum = buf[n][col];
        sum += buf[n + 2][col];
        buf[n + 1][col] += fC * sum;
      }
    }
    for (int32_t n = 0 + offset, i = start; i < stop; i++, n += 2) {
      fdwt_irrev97_fixed_neon_ver_step(simdlen, buf[n - 1], buf[n + 1], buf[n], fD);
      for (int32_t col = simdlen; col < u1 - u0; ++col) {
        float sum = buf[n - 1][col];
        sum += buf[n + 1][col];
        buf[n][col] += fD * sum;
      }
    }

    for (int32_t i = 1; i <= top; ++i) {
      aligned_mem_free(buf[top - i]);
    }
    for (int32_t i = 1; i <= bottom; i++) {
      aligned_mem_free(buf[top + (v1 - v0) + i - 1]);
    }
    delete[] buf;
  }
}

// reversible FDWT
void fdwt_rev_ver_sr_fixed_neon(sprec_t *in, const int32_t u0, const int32_t u1, const int32_t v0,
                                const int32_t v1, const int32_t stride) {
  constexpr int32_t num_pse_i0[2] = {2, 1};
  constexpr int32_t num_pse_i1[2] = {1, 2};
  const int32_t top               = num_pse_i0[v0 % 2];
  const int32_t bottom            = num_pse_i1[v1 % 2];

  if (v0 == v1 - 1) {
    // one sample case
    for (int32_t col = 0; col < u1 - u0; ++col) {
      if (v0 % 2) {
        in[col] *= 2;
      }
    }
  } else {
    const int32_t len = round_up(stride, SIMD_PADDING);
    auto **buf        = new sprec_t *[static_cast<size_t>(top + v1 - v0 + bottom)];
    for (int32_t i = 1; i <= top; ++i) {
      buf[top - i] =
          static_cast<sprec_t *>(aligned_mem_alloc(sizeof(sprec_t) * static_cast<size_t>(len), 32));
      memcpy(buf[top - i], &in[PSEo(v0 - i, v0, v1) * stride],
             sizeof(sprec_t) * static_cast<size_t>(stride));
      // buf[top - i] = &in[(PSEo(v0 - i, v0, v1) - v0) * stride];
    }
    for (int32_t row = 0; row < v1 - v0; ++row) {
      buf[top + row] = &in[row * stride];
    }
    for (int32_t i = 1; i <= bottom; i++) {
      buf[top + (v1 - v0) + i - 1] =
          static_cast<sprec_t *>(aligned_mem_alloc(sizeof(sprec_t) * static_cast<size_t>(len), 32));
      memcpy(buf[top + (v1 - v0) + i - 1], &in[PSEo(v1 - v0 + i - 1 + v0, v0, v1) * stride],
             sizeof(sprec_t) * static_cast<size_t>(stride));
    }
    const int32_t start  = ceil_int(v0, 2);
    const int32_t stop   = ceil_int(v1, 2);
    const int32_t offset = top + v0 % 2;

    int32_t simdlen = (u1 - u0) - (u1 - u0) % 4;
    for (int32_t n = -2 + offset, i = start - 1; i < stop; ++i, n += 2) {
      for (int32_t col = 0; col < simdlen; col += 4) {
        auto X0 = vld1q_f32(&buf[n][col]);
        auto X2 = vld1q_f32(&buf[n + 2][col]);
        auto X1 = vld1q_f32(&buf[n + 1][col]);
        auto xfloor = vrndmq_f32(vmulq_n_f32(vaddq_f32(X0, X2), 0.5f));
        X1 = vsubq_f32(X1, xfloor);
        vst1q_f32(&buf[n + 1][col], X1);
      }
      for (int32_t col = simdlen; col < u1 - u0; ++col) {
        float sum = buf[n][col];
        sum += buf[n + 2][col];
        buf[n + 1][col] -= floorf(sum * 0.5f);
      }
    }
    const auto vtwo = vdupq_n_f32(2.0f);
    for (int32_t n = 0 + offset, i = start; i < stop; ++i, n += 2) {
      for (int32_t col = 0; col < simdlen; col += 4) {
        auto X0 = vld1q_f32(&buf[n - 1][col]);
        auto X2 = vld1q_f32(&buf[n + 1][col]);
        auto X1 = vld1q_f32(&buf[n][col]);
        auto xfloor = vrndmq_f32(vmulq_n_f32(vaddq_f32(vaddq_f32(X0, X2), vtwo), 0.25f));
        X1 = vaddq_f32(X1, xfloor);
        vst1q_f32(&buf[n][col], X1);
      }
      for (int32_t col = simdlen; col < u1 - u0; ++col) {
        float sum = buf[n - 1][col];
        sum += buf[n + 1][col];
        sum += 2.0f;
        buf[n][col] += floorf(sum * 0.25f);
      }
    }

    for (int32_t i = 1; i <= top; ++i) {
      aligned_mem_free(buf[top - i]);
    }
    for (int32_t i = 1; i <= bottom; i++) {
      aligned_mem_free(buf[top + (v1 - v0) + i - 1]);
    }
    delete[] buf;
  }
}
#endif