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
  int32_t i = simdlen;
  // 2× unrolled main loop: two independent 4-pair groups per iteration for better ILP.
  for (; i > 4; i -= 8) {
    auto x0a  = vld2q_f32(X + n0);
    auto x1a  = vld2q_f32(X + n1);
    auto x0b  = vld2q_f32(X + 8 + n0);
    auto x1b  = vld2q_f32(X + 8 + n1);
    x0a.val[1] = vfmaq_f32(x0a.val[1], vaddq_f32(x0a.val[0], x1a.val[0]), vvv);
    x0b.val[1] = vfmaq_f32(x0b.val[1], vaddq_f32(x0b.val[0], x1b.val[0]), vvv);
    vst2q_f32(X + n0, x0a);
    vst2q_f32(X + 8 + n0, x0b);
    X += 16;
  }
  for (; i > 0; i -= 4) {
    auto x0   = vld2q_f32(X + n0);
    auto x1   = vld2q_f32(X + n1);
    x0.val[1] = vfmaq_f32(x0.val[1], vaddq_f32(x0.val[0], x1.val[0]), vvv);
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
  // step 1
  int32_t simdlen = stop - (start - 1);
  int32_t n = -2 + offset, i = 0;
  for (; i + 4 < simdlen; i += 8, n += 16) {
    auto xl0a  = vld2q_f32(X + n);
    auto xl1a  = vld2q_f32(X + n + 2);
    auto xl0b  = vld2q_f32(X + n + 8);
    auto xl1b  = vld2q_f32(X + n + 10);
    xl0a.val[1] = vsubq_f32(xl0a.val[1], vrndmq_f32(vmulq_n_f32(vaddq_f32(xl0a.val[0], xl1a.val[0]), 0.5f)));
    xl0b.val[1] = vsubq_f32(xl0b.val[1], vrndmq_f32(vmulq_n_f32(vaddq_f32(xl0b.val[0], xl1b.val[0]), 0.5f)));
    vst2q_f32(X + n, xl0a);
    vst2q_f32(X + n + 8, xl0b);
  }
  for (; i < simdlen; i += 4, n += 8) {
    auto xl0   = vld2q_f32(X + n);
    auto xl1   = vld2q_f32(X + n + 2);
    xl0.val[1] = vsubq_f32(xl0.val[1], vrndmq_f32(vmulq_n_f32(vaddq_f32(xl0.val[0], xl1.val[0]), 0.5f)));
    vst2q_f32(X + n, xl0);
  }

  // step 2
  simdlen = stop - start;
  const auto vtwo = vdupq_n_f32(2.0f);
  n = 0 + offset; i = 0;
  for (; i + 4 < simdlen; i += 8, n += 16) {
    auto xl0a  = vld2q_f32(X + n - 1);
    auto xl1a  = vld2q_f32(X + n + 1);
    auto xl0b  = vld2q_f32(X + n + 7);
    auto xl1b  = vld2q_f32(X + n + 9);
    xl0a.val[1] = vaddq_f32(xl0a.val[1], vrndmq_f32(vmulq_n_f32(vaddq_f32(vaddq_f32(xl0a.val[0], xl1a.val[0]), vtwo), 0.25f)));
    xl0b.val[1] = vaddq_f32(xl0b.val[1], vrndmq_f32(vmulq_n_f32(vaddq_f32(vaddq_f32(xl0b.val[0], xl1b.val[0]), vtwo), 0.25f)));
    vst2q_f32(X + n - 1, xl0a);
    vst2q_f32(X + n + 7, xl0b);
  }
  for (; i < simdlen; i += 4, n += 8) {
    auto xl0 = vld2q_f32(X + n - 1);
    auto xl1 = vld2q_f32(X + n + 1);
    xl0.val[1] = vaddq_f32(xl0.val[1], vrndmq_f32(vmulq_n_f32(vaddq_f32(vaddq_f32(xl0.val[0], xl1.val[0]), vtwo), 0.25f)));
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
  int32_t n = 0;
  // 2× unrolled: two independent FMA chains per iteration to hide 4-cycle FMA latency.
  for (; n + 4 < simdlen; n += 8) {
    auto x0a = vld1q_f32(Xin0 + n);     auto x2a = vld1q_f32(Xin1 + n);     auto x1a = vld1q_f32(Xout + n);
    auto x0b = vld1q_f32(Xin0 + n + 4); auto x2b = vld1q_f32(Xin1 + n + 4); auto x1b = vld1q_f32(Xout + n + 4);
    x1a = vfmaq_f32(x1a, vvv, vaddq_f32(x0a, x2a));
    x1b = vfmaq_f32(x1b, vvv, vaddq_f32(x0b, x2b));
    vst1q_f32(Xout + n, x1a);
    vst1q_f32(Xout + n + 4, x1b);
  }
  for (; n < simdlen; n += 4) {
    auto x0 = vld1q_f32(Xin0 + n);
    auto x2 = vld1q_f32(Xin1 + n);
    auto x1 = vld1q_f32(Xout + n);
    x1 = vfmaq_f32(x1, vvv, vaddq_f32(x0, x2));
    vst1q_f32(Xout + n, x1);
  }
};

void fdwt_irrev_ver_sr_fixed_neon(sprec_t *in, const int32_t u0, const int32_t u1, const int32_t v0,
                                  const int32_t v1, const int32_t stride, sprec_t *pse_scratch, sprec_t **buf_scratch) {
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
    sprec_t **buf     = buf_scratch;
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
      const int32_t ce        = (cs + DWT_VERT_STRIP < width) ? cs + DWT_VERT_STRIP : width;
      const int32_t simdlen_s = (ce - cs) - (ce - cs) % 4;
      for (int32_t n = -4 + offset, i = start - 2; i < stop + 1; i++, n += 2) {
        fdwt_irrev97_fixed_neon_ver_step(simdlen_s, buf[n] + cs, buf[n + 2] + cs, buf[n + 1] + cs, fA);
        for (int32_t col = cs + simdlen_s; col < ce; ++col) {
          buf[n + 1][col] += fA * (buf[n][col] + buf[n + 2][col]);
        }
      }
      for (int32_t n = -2 + offset, i = start - 1; i < stop + 1; i++, n += 2) {
        fdwt_irrev97_fixed_neon_ver_step(simdlen_s, buf[n - 1] + cs, buf[n + 1] + cs, buf[n] + cs, fB);
        for (int32_t col = cs + simdlen_s; col < ce; ++col) {
          buf[n][col] += fB * (buf[n - 1][col] + buf[n + 1][col]);
        }
      }
      for (int32_t n = -2 + offset, i = start - 1; i < stop; i++, n += 2) {
        fdwt_irrev97_fixed_neon_ver_step(simdlen_s, buf[n] + cs, buf[n + 2] + cs, buf[n + 1] + cs, fC);
        for (int32_t col = cs + simdlen_s; col < ce; ++col) {
          buf[n + 1][col] += fC * (buf[n][col] + buf[n + 2][col]);
        }
      }
      for (int32_t n = 0 + offset, i = start; i < stop; i++, n += 2) {
        fdwt_irrev97_fixed_neon_ver_step(simdlen_s, buf[n - 1] + cs, buf[n + 1] + cs, buf[n] + cs, fD);
        for (int32_t col = cs + simdlen_s; col < ce; ++col) {
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
  }
}

// reversible FDWT
void fdwt_rev_ver_sr_fixed_neon(sprec_t *in, const int32_t u0, const int32_t u1, const int32_t v0,
                                const int32_t v1, const int32_t stride, sprec_t *pse_scratch, sprec_t **buf_scratch) {
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
    sprec_t **buf     = buf_scratch;
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
      const int32_t ce        = (cs + DWT_VERT_STRIP < width) ? cs + DWT_VERT_STRIP : width;
      const int32_t simdlen_s = (ce - cs) - (ce - cs) % 4;
      for (int32_t n = -2 + offset, i = start - 1; i < stop; ++i, n += 2) {
        int32_t col = 0;
        for (; col + 4 < simdlen_s; col += 8) {
          auto X0a    = vld1q_f32(buf[n] + cs + col);
          auto X2a    = vld1q_f32(buf[n + 2] + cs + col);
          auto X1a    = vld1q_f32(buf[n + 1] + cs + col);
          auto X0b    = vld1q_f32(buf[n] + cs + col + 4);
          auto X2b    = vld1q_f32(buf[n + 2] + cs + col + 4);
          auto X1b    = vld1q_f32(buf[n + 1] + cs + col + 4);
          X1a         = vsubq_f32(X1a, vrndmq_f32(vmulq_n_f32(vaddq_f32(X0a, X2a), 0.5f)));
          X1b         = vsubq_f32(X1b, vrndmq_f32(vmulq_n_f32(vaddq_f32(X0b, X2b), 0.5f)));
          vst1q_f32(buf[n + 1] + cs + col, X1a);
          vst1q_f32(buf[n + 1] + cs + col + 4, X1b);
        }
        for (; col < simdlen_s; col += 4) {
          auto X0     = vld1q_f32(buf[n] + cs + col);
          auto X2     = vld1q_f32(buf[n + 2] + cs + col);
          auto X1     = vld1q_f32(buf[n + 1] + cs + col);
          auto xfloor = vrndmq_f32(vmulq_n_f32(vaddq_f32(X0, X2), 0.5f));
          X1          = vsubq_f32(X1, xfloor);
          vst1q_f32(buf[n + 1] + cs + col, X1);
        }
        for (int32_t col = cs + simdlen_s; col < ce; ++col) {
          buf[n + 1][col] -= floorf((buf[n][col] + buf[n + 2][col]) * 0.5f);
        }
      }
      const auto vtwo = vdupq_n_f32(2.0f);
      for (int32_t n = 0 + offset, i = start; i < stop; ++i, n += 2) {
        int32_t col = 0;
        for (; col + 4 < simdlen_s; col += 8) {
          auto X0a    = vld1q_f32(buf[n - 1] + cs + col);
          auto X2a    = vld1q_f32(buf[n + 1] + cs + col);
          auto X1a    = vld1q_f32(buf[n] + cs + col);
          auto X0b    = vld1q_f32(buf[n - 1] + cs + col + 4);
          auto X2b    = vld1q_f32(buf[n + 1] + cs + col + 4);
          auto X1b    = vld1q_f32(buf[n] + cs + col + 4);
          X1a         = vaddq_f32(X1a, vrndmq_f32(vmulq_n_f32(vaddq_f32(vaddq_f32(X0a, X2a), vtwo), 0.25f)));
          X1b         = vaddq_f32(X1b, vrndmq_f32(vmulq_n_f32(vaddq_f32(vaddq_f32(X0b, X2b), vtwo), 0.25f)));
          vst1q_f32(buf[n] + cs + col, X1a);
          vst1q_f32(buf[n] + cs + col + 4, X1b);
        }
        for (; col < simdlen_s; col += 4) {
          auto X0     = vld1q_f32(buf[n - 1] + cs + col);
          auto X2     = vld1q_f32(buf[n + 1] + cs + col);
          auto X1     = vld1q_f32(buf[n] + cs + col);
          auto xfloor = vrndmq_f32(vmulq_n_f32(vaddq_f32(vaddq_f32(X0, X2), vtwo), 0.25f));
          X1          = vaddq_f32(X1, xfloor);
          vst1q_f32(buf[n] + cs + col, X1);
        }
        for (int32_t col = cs + simdlen_s; col < ce; ++col) {
          buf[n][col] += floorf((buf[n - 1][col] + buf[n + 1][col] + 2.0f) * 0.25f);
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

// Single-row reversible (5/3) FDWT HP vertical lifting: tgt[i] -= floor((prev[i]+next[i])*0.5)
// 2× unrolled to hide vrndmq latency.
void fdwt_rev_ver_hp_step_neon(int32_t n, const float *prev, const float *next, float *tgt) {
  const float32x4_t k05 = vdupq_n_f32(0.5f);
  int32_t i = 0;
  for (; i + 4 < n; i += 8) {
    float32x4_t a0 = vld1q_f32(prev + i);     float32x4_t b0 = vld1q_f32(next + i);     float32x4_t t0 = vld1q_f32(tgt + i);
    float32x4_t a1 = vld1q_f32(prev + i + 4); float32x4_t b1 = vld1q_f32(next + i + 4); float32x4_t t1 = vld1q_f32(tgt + i + 4);
    t0 = vsubq_f32(t0, vrndmq_f32(vmulq_f32(vaddq_f32(a0, b0), k05)));
    t1 = vsubq_f32(t1, vrndmq_f32(vmulq_f32(vaddq_f32(a1, b1), k05)));
    vst1q_f32(tgt + i, t0);
    vst1q_f32(tgt + i + 4, t1);
  }
  for (; i + 4 <= n; i += 4) {
    float32x4_t a = vld1q_f32(prev + i);
    float32x4_t b = vld1q_f32(next + i);
    float32x4_t t = vld1q_f32(tgt  + i);
    t = vsubq_f32(t, vrndmq_f32(vmulq_f32(vaddq_f32(a, b), k05)));
    vst1q_f32(tgt + i, t);
  }
  for (; i < n; ++i)
    tgt[i] -= floorf((prev[i] + next[i]) * 0.5f);
}

// Single-row reversible (5/3) FDWT LP vertical lifting: tgt[i] += floor((prev[i]+next[i]+2)*0.25)
// 2× unrolled to hide vrndmq latency.
void fdwt_rev_ver_lp_step_neon(int32_t n, const float *prev, const float *next, float *tgt) {
  const float32x4_t k025 = vdupq_n_f32(0.25f);
  const float32x4_t k2   = vdupq_n_f32(2.0f);
  int32_t i = 0;
  for (; i + 4 < n; i += 8) {
    float32x4_t a0 = vld1q_f32(prev + i);     float32x4_t b0 = vld1q_f32(next + i);     float32x4_t t0 = vld1q_f32(tgt + i);
    float32x4_t a1 = vld1q_f32(prev + i + 4); float32x4_t b1 = vld1q_f32(next + i + 4); float32x4_t t1 = vld1q_f32(tgt + i + 4);
    t0 = vaddq_f32(t0, vrndmq_f32(vmulq_f32(vaddq_f32(vaddq_f32(a0, b0), k2), k025)));
    t1 = vaddq_f32(t1, vrndmq_f32(vmulq_f32(vaddq_f32(vaddq_f32(a1, b1), k2), k025)));
    vst1q_f32(tgt + i, t0);
    vst1q_f32(tgt + i + 4, t1);
  }
  for (; i + 4 <= n; i += 4) {
    float32x4_t a = vld1q_f32(prev + i);
    float32x4_t b = vld1q_f32(next + i);
    float32x4_t t = vld1q_f32(tgt  + i);
    t = vaddq_f32(t, vrndmq_f32(vmulq_f32(vaddq_f32(vaddq_f32(a, b), k2), k025)));
    vst1q_f32(tgt + i, t);
  }
  for (; i < n; ++i)
    tgt[i] += floorf((prev[i] + next[i] + 2.0f) * 0.25f);
}
#endif