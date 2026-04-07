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
  #include "dwt.hpp"
  #include "utils.hpp"
  #include <cstring>
#include <arm_neon.h>
#include <cmath>

/********************************************************************************
 * horizontal transforms
 *******************************************************************************/
// irreversible IDWT
auto idwt_irrev97_fixed_neon_hor_step = [](const int32_t init_pos, const int32_t simdlen, float *const X,
                                            const int32_t n0, const int32_t n1, float coeff) {
  auto vvv = vdupq_n_f32(coeff);
  int32_t n = init_pos, i = simdlen;
  // 2× unrolled main loop. n1 == n0+2 always (all 4 irrev97 steps), so:
  //   x1a.val[0] = vextq_f32(x0a.val[0], x0b.val[0], 1)
  //   x1b.val[0] = vextq_f32(x0b.val[0], x0c.val[0], 1)
  // x0c is a one-group look-ahead (safe: within SIMD_PADDING). Carrying x0c → x0a
  // across iterations cuts LD2 count from 4 to 2 per 8-pair group.
  if (i > 4) {
    auto x0a = vld2q_f32(X + n + n0);  // prologue: preload first group
    for (; i > 4; i -= 8, n += 16) {
      auto x0b         = vld2q_f32(X + n + 8 + n0);
      auto x0c         = vld2q_f32(X + n + 16 + n0);  // look-ahead for x1b and carry
      float32x4_t x1a0 = vextq_f32(x0a.val[0], x0b.val[0], 1);
      float32x4_t x1b0 = vextq_f32(x0b.val[0], x0c.val[0], 1);
      x0a.val[1]        = vfmsq_f32(x0a.val[1], vaddq_f32(x0a.val[0], x1a0), vvv);
      x0b.val[1]        = vfmsq_f32(x0b.val[1], vaddq_f32(x0b.val[0], x1b0), vvv);
      vst2q_f32(X + n + n0, x0a);
      vst2q_f32(X + n + 8 + n0, x0b);
      x0a = x0c;  // carry: x0c is the next iteration's x0a (no re-load needed)
    }
  }
  for (; i > 0; i -= 4, n += 8) {
    auto x0   = vld2q_f32(X + n + n0);
    auto x1   = vld2q_f32(X + n + n1);
    x0.val[1] = vfmsq_f32(x0.val[1], vaddq_f32(x0.val[0], x1.val[0]), vvv);
    vst2q_f32(X + n + n0, x0);
  }
};

void idwt_1d_filtr_irrev97_fixed_neon(sprec_t *X, const int32_t left, const int32_t u_i0,
                                      const int32_t u_i1) {
  const auto i0        = static_cast<int32_t>(u_i0);
  const auto i1        = static_cast<int32_t>(u_i1);
  const int32_t start  = i0 / 2;
  const int32_t stop   = i1 / 2;
  const int32_t offset = left - i0 % 2;

  // step 1
  int32_t simdlen = stop + 2 - (start - 1);
  idwt_irrev97_fixed_neon_hor_step(offset - 2, simdlen, X, -1, 1, fD);

  // step 2
  simdlen = stop + 1 - (start - 1);
  idwt_irrev97_fixed_neon_hor_step(offset - 2, simdlen, X, 0, 2, fC);

  // step 3
  simdlen = stop + 1 - start;
  idwt_irrev97_fixed_neon_hor_step(offset, simdlen, X, -1, 1, fB);

  // step 4
  simdlen = stop - start;
  idwt_irrev97_fixed_neon_hor_step(offset, simdlen, X, 0, 2, fA);
}

// reversible IDWT
void idwt_1d_filtr_rev53_fixed_neon(sprec_t *X, const int32_t left, const int32_t i0, const int32_t i1) {
  const int32_t start  = i0 / 2;
  const int32_t stop   = i1 / 2;
  const int32_t offset = left - i0 % 2;

  // step 1: 2× unrolled; xl1*.val[0] derived via vextq (n1 = n0+2 = -1+2 = +1) with carry.
  const int32_t base1 = offset;
  int32_t simdlen     = stop + 1 - start;
  const auto vtwo     = vdupq_n_f32(2.0f);
  int32_t k           = 0;
  if (k + 4 < simdlen) {
    auto xl0a = vld2q_f32(X + base1 - 1);  // prologue: preload first group (sp-1)
    for (; k + 4 < simdlen; k += 8) {
      sprec_t *sp       = X + base1 + k * 2;
      auto xl0b         = vld2q_f32(sp + 7);     // next group at sp+8-1
      auto xl0c         = vld2q_f32(sp + 15);    // look-ahead for xl1b and carry
      float32x4_t xl1a0 = vextq_f32(xl0a.val[0], xl0b.val[0], 1);
      float32x4_t xl1b0 = vextq_f32(xl0b.val[0], xl0c.val[0], 1);
      xl0a.val[1]        = vsubq_f32(xl0a.val[1], vrndmq_f32(vmulq_n_f32(vaddq_f32(vaddq_f32(xl0a.val[0], xl1a0), vtwo), 0.25f)));
      xl0b.val[1]        = vsubq_f32(xl0b.val[1], vrndmq_f32(vmulq_n_f32(vaddq_f32(vaddq_f32(xl0b.val[0], xl1b0), vtwo), 0.25f)));
      vst2q_f32(sp - 1, xl0a);
      vst2q_f32(sp + 7, xl0b);
      xl0a = xl0c;  // carry: xl0c becomes xl0a for next iteration
    }
  }
  for (; k < simdlen; k += 4) {
    sprec_t *sp = X + base1 + k * 2;
    auto xl0    = vld2q_f32(sp - 1);
    auto xl1    = vld2q_f32(sp + 1);
    xl0.val[1]  = vsubq_f32(xl0.val[1], vrndmq_f32(vmulq_n_f32(vaddq_f32(vaddq_f32(xl0.val[0], xl1.val[0]), vtwo), 0.25f)));
    vst2q_f32(sp - 1, xl0);
  }

  // step 2: 2× unrolled; xl1*.val[0] derived via vextq (n1 = n0+2 = 0+2 = +2) with carry.
  const int32_t base2 = offset;
  simdlen             = stop - start;
  k                   = 0;
  if (k + 4 < simdlen) {
    auto xl0a = vld2q_f32(X + base2);  // prologue: preload first group (sp+0)
    for (; k + 4 < simdlen; k += 8) {
      sprec_t *sp       = X + base2 + k * 2;
      auto xl0b         = vld2q_f32(sp + 8);     // next group at sp+8
      auto xl0c         = vld2q_f32(sp + 16);    // look-ahead
      float32x4_t xl1a0 = vextq_f32(xl0a.val[0], xl0b.val[0], 1);
      float32x4_t xl1b0 = vextq_f32(xl0b.val[0], xl0c.val[0], 1);
      xl0a.val[1]        = vaddq_f32(xl0a.val[1], vrndmq_f32(vmulq_n_f32(vaddq_f32(xl0a.val[0], xl1a0), 0.5f)));
      xl0b.val[1]        = vaddq_f32(xl0b.val[1], vrndmq_f32(vmulq_n_f32(vaddq_f32(xl0b.val[0], xl1b0), 0.5f)));
      vst2q_f32(sp, xl0a);
      vst2q_f32(sp + 8, xl0b);
      xl0a = xl0c;  // carry
    }
  }
  for (; k < simdlen; k += 4) {
    sprec_t *sp = X + base2 + k * 2;
    auto xl0    = vld2q_f32(sp);
    auto xl1    = vld2q_f32(sp + 2);
    xl0.val[1]  = vaddq_f32(xl0.val[1], vrndmq_f32(vmulq_n_f32(vaddq_f32(xl0.val[0], xl1.val[0]), 0.5f)));
    vst2q_f32(sp, xl0);
  }
}

// ATK irreversible 5/3 IDWT (horizontal)
void idwt_1d_filtr_irrev53_fixed_neon(sprec_t *X, const int32_t left, const int32_t u_i0,
                                      const int32_t u_i1) {
  const int32_t start    = u_i0 / 2;
  const int32_t stop     = u_i1 / 2;
  const int32_t offset   = left - u_i0 % 2;
  const int32_t lp_offset = offset + (u_i0 % 2) * 2;

  // Step 1: LP[k] -= 0.25*(HP[k-1]+HP[k])
  int32_t simdlen = stop + 1 - start;
  const auto x025 = vdupq_n_f32(0.25f);
  sprec_t *sp     = X + lp_offset;
  int32_t k       = 0;
  if (k + 4 < simdlen) {
    auto x0a = vld2q_f32(sp - 1);
    for (; k + 4 < simdlen; k += 8, sp += 16) {
      auto x0b         = vld2q_f32(sp + 7);
      auto x0c         = vld2q_f32(sp + 15);
      float32x4_t x1a0 = vextq_f32(x0a.val[0], x0b.val[0], 1);
      float32x4_t x1b0 = vextq_f32(x0b.val[0], x0c.val[0], 1);
      x0a.val[1]        = vfmsq_f32(x0a.val[1], vaddq_f32(x0a.val[0], x1a0), x025);
      x0b.val[1]        = vfmsq_f32(x0b.val[1], vaddq_f32(x0b.val[0], x1b0), x025);
      vst2q_f32(sp - 1, x0a);
      vst2q_f32(sp + 7, x0b);
      x0a = x0c;
    }
  }
  for (; k < simdlen; k += 4, sp += 8) {
    auto x0   = vld2q_f32(sp - 1);
    auto x1   = vld2q_f32(sp + 1);
    x0.val[1] = vfmsq_f32(x0.val[1], vaddq_f32(x0.val[0], x1.val[0]), x025);
    vst2q_f32(sp - 1, x0);
  }

  // Step 2: HP[k] += 0.5*(LP_mod[k]+LP_mod[k+1])
  simdlen = stop - start;
  const auto x05 = vdupq_n_f32(0.5f);
  sp = X + offset;
  k  = 0;
  if (k + 4 < simdlen) {
    auto x0a = vld2q_f32(sp);
    for (; k + 4 < simdlen; k += 8, sp += 16) {
      auto x0b         = vld2q_f32(sp + 8);
      auto x0c         = vld2q_f32(sp + 16);
      float32x4_t x1a0 = vextq_f32(x0a.val[0], x0b.val[0], 1);
      float32x4_t x1b0 = vextq_f32(x0b.val[0], x0c.val[0], 1);
      x0a.val[1]        = vfmaq_f32(x0a.val[1], vaddq_f32(x0a.val[0], x1a0), x05);
      x0b.val[1]        = vfmaq_f32(x0b.val[1], vaddq_f32(x0b.val[0], x1b0), x05);
      vst2q_f32(sp, x0a);
      vst2q_f32(sp + 8, x0b);
      x0a = x0c;
    }
  }
  for (; k < simdlen; k += 4, sp += 8) {
    auto x0   = vld2q_f32(sp);
    auto x1   = vld2q_f32(sp + 2);
    x0.val[1] = vfmaq_f32(x0.val[1], vaddq_f32(x0.val[0], x1.val[0]), x05);
    vst2q_f32(sp, x0);
  }
}

/********************************************************************************
 * vertical transform
 *******************************************************************************/
// irreversible IDWT
auto idwt_irrev97_fixed_neon_ver_step = [](const int32_t simdlen, float *const Xin0, float *const Xin1,
                                            float *const Xout, float coeff) {
  auto vvv = vdupq_n_f32(coeff);
  int32_t n = 0;
  // 2× unrolled: two independent FMS chains per iteration to hide 4-cycle FMA latency.
  for (; n + 4 < simdlen; n += 8) {
    auto x0a = vld1q_f32(Xin0 + n);     auto x2a = vld1q_f32(Xin1 + n);     auto x1a = vld1q_f32(Xout + n);
    auto x0b = vld1q_f32(Xin0 + n + 4); auto x2b = vld1q_f32(Xin1 + n + 4); auto x1b = vld1q_f32(Xout + n + 4);
    x1a = vfmsq_f32(x1a, vaddq_f32(x0a, x2a), vvv);
    x1b = vfmsq_f32(x1b, vaddq_f32(x0b, x2b), vvv);
    vst1q_f32(Xout + n, x1a);
    vst1q_f32(Xout + n + 4, x1b);
  }
  for (; n < simdlen; n += 4) {
    auto x0 = vld1q_f32(Xin0 + n);
    auto x2 = vld1q_f32(Xin1 + n);
    auto x1 = vld1q_f32(Xout + n);
    x1 = vfmsq_f32(x1, vaddq_f32(x0, x2), vvv);
    vst1q_f32(Xout + n, x1);
  }
};

// Single-row irreversible vertical lifting step for idwt_2d_state::adv_step().
// Applies tgt[i] -= coeff*(prev[i]+next[i]) using FMS, matching the batch path exactly.
// n is the row width; ring-buffer rows are always sufficiently aligned.
// Single-row reversible (5/3) LP vertical lifting: tgt[i] -= floor((prev[i]+next[i]+2)*0.25)
// 2× unrolled to match idwt_irrev_ver_step_fixed_neon and hide vrndmq latency.
void idwt_rev_ver_lp_step_neon(int32_t n, const float *prev, const float *next, float *tgt) {
  const float32x4_t k025 = vdupq_n_f32(0.25f);
  const float32x4_t k2   = vdupq_n_f32(2.0f);
  int32_t i = 0;
  for (; i + 4 < n; i += 8) {
    float32x4_t a0 = vld1q_f32(prev + i);     float32x4_t b0 = vld1q_f32(next + i);     float32x4_t t0 = vld1q_f32(tgt + i);
    float32x4_t a1 = vld1q_f32(prev + i + 4); float32x4_t b1 = vld1q_f32(next + i + 4); float32x4_t t1 = vld1q_f32(tgt + i + 4);
    t0 = vsubq_f32(t0, vrndmq_f32(vmulq_f32(vaddq_f32(vaddq_f32(a0, b0), k2), k025)));
    t1 = vsubq_f32(t1, vrndmq_f32(vmulq_f32(vaddq_f32(vaddq_f32(a1, b1), k2), k025)));
    vst1q_f32(tgt + i, t0);
    vst1q_f32(tgt + i + 4, t1);
  }
  for (; i + 4 <= n; i += 4) {
    float32x4_t a = vld1q_f32(prev + i);
    float32x4_t b = vld1q_f32(next + i);
    float32x4_t t = vld1q_f32(tgt  + i);
    t = vsubq_f32(t, vrndmq_f32(vmulq_f32(vaddq_f32(vaddq_f32(a, b), k2), k025)));
    vst1q_f32(tgt + i, t);
  }
  for (; i < n; ++i)
    tgt[i] -= floorf((prev[i] + next[i] + 2.0f) * 0.25f);
}

// Single-row reversible (5/3) HP vertical lifting: tgt[i] += floor((prev[i]+next[i])*0.5)
// 2× unrolled to match idwt_irrev_ver_step_fixed_neon and hide vrndmq latency.
void idwt_rev_ver_hp_step_neon(int32_t n, const float *prev, const float *next, float *tgt) {
  const float32x4_t k05 = vdupq_n_f32(0.5f);
  int32_t i = 0;
  for (; i + 4 < n; i += 8) {
    float32x4_t a0 = vld1q_f32(prev + i);     float32x4_t b0 = vld1q_f32(next + i);     float32x4_t t0 = vld1q_f32(tgt + i);
    float32x4_t a1 = vld1q_f32(prev + i + 4); float32x4_t b1 = vld1q_f32(next + i + 4); float32x4_t t1 = vld1q_f32(tgt + i + 4);
    t0 = vaddq_f32(t0, vrndmq_f32(vmulq_f32(vaddq_f32(a0, b0), k05)));
    t1 = vaddq_f32(t1, vrndmq_f32(vmulq_f32(vaddq_f32(a1, b1), k05)));
    vst1q_f32(tgt + i, t0);
    vst1q_f32(tgt + i + 4, t1);
  }
  for (; i + 4 <= n; i += 4) {
    float32x4_t a = vld1q_f32(prev + i);
    float32x4_t b = vld1q_f32(next + i);
    float32x4_t t = vld1q_f32(tgt  + i);
    t = vaddq_f32(t, vrndmq_f32(vmulq_f32(vaddq_f32(a, b), k05)));
    vst1q_f32(tgt + i, t);
  }
  for (; i < n; ++i)
    tgt[i] += floorf((prev[i] + next[i]) * 0.5f);
}

void idwt_irrev_ver_step_fixed_neon(int32_t n, float *prev, float *next, float *tgt, float coeff) {
  auto vvv  = vdupq_n_f32(coeff);
  int32_t i = 0;
  for (; i + 4 < n; i += 8) {
    auto x0a = vld1q_f32(prev + i);     auto x2a = vld1q_f32(next + i);     auto x1a = vld1q_f32(tgt + i);
    auto x0b = vld1q_f32(prev + i + 4); auto x2b = vld1q_f32(next + i + 4); auto x1b = vld1q_f32(tgt + i + 4);
    x1a = vfmsq_f32(x1a, vaddq_f32(x0a, x2a), vvv);
    x1b = vfmsq_f32(x1b, vaddq_f32(x0b, x2b), vvv);
    vst1q_f32(tgt + i, x1a);
    vst1q_f32(tgt + i + 4, x1b);
  }
  for (; i + 4 <= n; i += 4) {
    auto x0 = vld1q_f32(prev + i);
    auto x2 = vld1q_f32(next + i);
    auto x1 = vld1q_f32(tgt  + i);
    x1 = vfmsq_f32(x1, vaddq_f32(x0, x2), vvv);
    vst1q_f32(tgt + i, x1);
  }
  for (; i < n; ++i)
    tgt[i] -= coeff * (prev[i] + next[i]);
}

void idwt_irrev_ver_sr_fixed_neon(sprec_t *in, const int32_t u0, const int32_t u1, const int32_t v0,
                                  const int32_t v1, const int32_t stride, sprec_t *pse_scratch, sprec_t **buf_scratch) {
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
      const int32_t ce        = (cs + DWT_VERT_STRIP < width) ? cs + DWT_VERT_STRIP : width;
      const int32_t simdlen_s = (ce - cs) - (ce - cs) % 4;
      for (int32_t n = -2 + offset, i = start - 1; i < stop + 2; i++, n += 2) {
        idwt_irrev97_fixed_neon_ver_step(simdlen_s, buf[n - 1] + cs, buf[n + 1] + cs, buf[n] + cs, fD);
        for (int32_t col = cs + simdlen_s; col < ce; ++col) {
          buf[n][col] -= (buf[n - 1][col] + buf[n + 1][col]) * fD;
        }
      }
      for (int32_t n = -2 + offset, i = start - 1; i < stop + 1; i++, n += 2) {
        idwt_irrev97_fixed_neon_ver_step(simdlen_s, buf[n] + cs, buf[n + 2] + cs, buf[n + 1] + cs, fC);
        for (int32_t col = cs + simdlen_s; col < ce; ++col) {
          buf[n + 1][col] -= (buf[n][col] + buf[n + 2][col]) * fC;
        }
      }
      for (int32_t n = 0 + offset, i = start; i < stop + 1; i++, n += 2) {
        idwt_irrev97_fixed_neon_ver_step(simdlen_s, buf[n - 1] + cs, buf[n + 1] + cs, buf[n] + cs, fB);
        for (int32_t col = cs + simdlen_s; col < ce; ++col) {
          buf[n][col] -= (buf[n - 1][col] + buf[n + 1][col]) * fB;
        }
      }
      for (int32_t n = 0 + offset, i = start; i < stop; i++, n += 2) {
        idwt_irrev97_fixed_neon_ver_step(simdlen_s, buf[n] + cs, buf[n + 2] + cs, buf[n + 1] + cs, fA);
        for (int32_t col = cs + simdlen_s; col < ce; ++col) {
          buf[n + 1][col] -= (buf[n][col] + buf[n + 2][col]) * fA;
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

// reversible IDWT
void idwt_rev_ver_sr_fixed_neon(sprec_t *in, const int32_t u0, const int32_t u1, const int32_t v0,
                                const int32_t v1, const int32_t stride, sprec_t *pse_scratch, sprec_t **buf_scratch) {
  constexpr int32_t num_pse_i0[2] = {1, 2};
  constexpr int32_t num_pse_i1[2] = {2, 1};
  const int32_t top               = num_pse_i0[v0 % 2];
  const int32_t bottom            = num_pse_i1[v1 % 2];
  if (v0 == v1 - 1) {
    // one sample case
    for (int32_t col = 0; col < u1 - u0; ++col) {
      in[col] = (v0 % 2 == 0) ? in[col] : floorf(in[col] * 0.5f);
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
      const int32_t ce        = (cs + DWT_VERT_STRIP < width) ? cs + DWT_VERT_STRIP : width;
      const int32_t simdlen_s = (ce - cs) - (ce - cs) % 4;
      const auto vtwo         = vdupq_n_f32(2.0f);
      for (int32_t n = 0 + offset, i = start; i < stop + 1; ++i, n += 2) {
        int32_t col = 0;
        for (; col + 4 < simdlen_s; col += 8) {
          auto vin0a   = vld1q_f32(buf[n - 1] + cs + col);
          auto vouta   = vld1q_f32(buf[n] + cs + col);
          auto vin1a   = vld1q_f32(buf[n + 1] + cs + col);
          auto vin0b   = vld1q_f32(buf[n - 1] + cs + col + 4);
          auto voutb   = vld1q_f32(buf[n] + cs + col + 4);
          auto vin1b   = vld1q_f32(buf[n + 1] + cs + col + 4);
          vouta        = vsubq_f32(vouta, vrndmq_f32(vmulq_n_f32(vaddq_f32(vaddq_f32(vin0a, vin1a), vtwo), 0.25f)));
          voutb        = vsubq_f32(voutb, vrndmq_f32(vmulq_n_f32(vaddq_f32(vaddq_f32(vin0b, vin1b), vtwo), 0.25f)));
          vst1q_f32(buf[n] + cs + col, vouta);
          vst1q_f32(buf[n] + cs + col + 4, voutb);
        }
        for (; col < simdlen_s; col += 4) {
          auto vin0    = vld1q_f32(buf[n - 1] + cs + col);
          auto vout    = vld1q_f32(buf[n] + cs + col);
          auto vin1    = vld1q_f32(buf[n + 1] + cs + col);
          auto xfloor0 = vrndmq_f32(vmulq_n_f32(vaddq_f32(vaddq_f32(vin0, vin1), vtwo), 0.25f));
          vout         = vsubq_f32(vout, xfloor0);
          vst1q_f32(buf[n] + cs + col, vout);
        }
        for (int32_t col = cs + simdlen_s; col < ce; ++col) {
          buf[n][col] -= floorf((buf[n - 1][col] + buf[n + 1][col] + 2.0f) * 0.25f);
        }
      }
      for (int32_t n = 0 + offset, i = start; i < stop; ++i, n += 2) {
        int32_t col = 0;
        for (; col + 4 < simdlen_s; col += 8) {
          auto vin0a   = vld1q_f32(buf[n] + cs + col);
          auto vouta   = vld1q_f32(buf[n + 1] + cs + col);
          auto vin1a   = vld1q_f32(buf[n + 2] + cs + col);
          auto vin0b   = vld1q_f32(buf[n] + cs + col + 4);
          auto voutb   = vld1q_f32(buf[n + 1] + cs + col + 4);
          auto vin1b   = vld1q_f32(buf[n + 2] + cs + col + 4);
          vouta        = vaddq_f32(vouta, vrndmq_f32(vmulq_n_f32(vaddq_f32(vin0a, vin1a), 0.5f)));
          voutb        = vaddq_f32(voutb, vrndmq_f32(vmulq_n_f32(vaddq_f32(vin0b, vin1b), 0.5f)));
          vst1q_f32(buf[n + 1] + cs + col, vouta);
          vst1q_f32(buf[n + 1] + cs + col + 4, voutb);
        }
        for (; col < simdlen_s; col += 4) {
          auto vin0    = vld1q_f32(buf[n] + cs + col);
          auto vout    = vld1q_f32(buf[n + 1] + cs + col);
          auto vin1    = vld1q_f32(buf[n + 2] + cs + col);
          auto xfloor0 = vrndmq_f32(vmulq_n_f32(vaddq_f32(vin0, vin1), 0.5f));
          vout         = vaddq_f32(vout, xfloor0);
          vst1q_f32(buf[n + 1] + cs + col, vout);
        }
        for (int32_t col = cs + simdlen_s; col < ce; ++col) {
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

// ATK irreversible 5/3 IDWT (vertical)
void idwt_irrev53_ver_sr_fixed_neon(sprec_t *in, const int32_t u0, const int32_t u1, const int32_t v0,
                                    const int32_t v1, const int32_t stride, sprec_t *pse_scratch,
                                    sprec_t **buf_scratch) {
  constexpr int32_t num_pse_i0[2] = {1, 2};
  constexpr int32_t num_pse_i1[2] = {2, 1};
  const int32_t top    = num_pse_i0[v0 % 2];
  const int32_t bottom = num_pse_i1[v1 % 2];
  if (v0 == v1 - 1) {
    // single row: nothing to do
  } else {
    const int32_t len = round_up(stride, SIMD_PADDING);
    sprec_t **buf     = buf_scratch;
    for (int32_t i = 1; i <= top; ++i) {
      buf[top - i] = pse_scratch + (i - 1) * len;
      memcpy(buf[top - i], &in[PSEo(v0 - i, v0, v1) * stride],
             sizeof(sprec_t) * static_cast<size_t>(stride));
    }
    for (int32_t row = 0; row < v1 - v0; ++row) buf[top + row] = &in[row * stride];
    for (int32_t i = 1; i <= bottom; i++) {
      buf[top + (v1 - v0) + i - 1] = pse_scratch + (top + i - 1) * len;
      memcpy(buf[top + (v1 - v0) + i - 1], &in[PSEo(v1 - v0 + i - 1 + v0, v0, v1) * stride],
             sizeof(sprec_t) * static_cast<size_t>(stride));
    }
    const int32_t lp_count = ceil_int(v1, 2) - ceil_int(v0, 2);
    const int32_t hp_count = v1 / 2 - v0 / 2;
    const int32_t offset   = top - v0 % 2;
    const int32_t lp_n0    = top + v0 % 2;
    const int32_t width    = u1 - u0;
    for (int32_t cs = 0; cs < width; cs += DWT_VERT_STRIP) {
      const int32_t ce        = (cs + DWT_VERT_STRIP < width) ? cs + DWT_VERT_STRIP : width;
      const int32_t simdlen_s = (ce - cs) - (ce - cs) % 4;
      // Step 1: LP[k] -= 0.25*(HP[k-1]+HP[k])
      const auto x025 = vdupq_n_f32(0.25f);
      for (int32_t k = 0, n = lp_n0; k < lp_count; ++k, n += 2) {
        int32_t col = 0;
        for (; col + 4 < simdlen_s; col += 8) {
          auto x0a = vld1q_f32(buf[n - 1] + cs + col);
          auto x2a = vld1q_f32(buf[n + 1] + cs + col);
          auto x1a = vld1q_f32(buf[n] + cs + col);
          auto x0b = vld1q_f32(buf[n - 1] + cs + col + 4);
          auto x2b = vld1q_f32(buf[n + 1] + cs + col + 4);
          auto x1b = vld1q_f32(buf[n] + cs + col + 4);
          x1a = vfmsq_f32(x1a, vaddq_f32(x0a, x2a), x025);
          x1b = vfmsq_f32(x1b, vaddq_f32(x0b, x2b), x025);
          vst1q_f32(buf[n] + cs + col, x1a);
          vst1q_f32(buf[n] + cs + col + 4, x1b);
        }
        for (; col < simdlen_s; col += 4) {
          auto x0 = vld1q_f32(buf[n - 1] + cs + col);
          auto x2 = vld1q_f32(buf[n + 1] + cs + col);
          auto x1 = vld1q_f32(buf[n] + cs + col);
          x1 = vfmsq_f32(x1, vaddq_f32(x0, x2), x025);
          vst1q_f32(buf[n] + cs + col, x1);
        }
        for (int32_t c = cs + simdlen_s; c < ce; ++c)
          buf[n][c] -= 0.25f * (buf[n - 1][c] + buf[n + 1][c]);
      }
      // Step 2: HP[k] += 0.5*(LP_mod[k]+LP_mod[k+1])
      const auto x05 = vdupq_n_f32(0.5f);
      for (int32_t k = 0, n = offset; k < hp_count; ++k, n += 2) {
        int32_t col = 0;
        for (; col + 4 < simdlen_s; col += 8) {
          auto x0a = vld1q_f32(buf[n] + cs + col);
          auto x2a = vld1q_f32(buf[n + 2] + cs + col);
          auto x1a = vld1q_f32(buf[n + 1] + cs + col);
          auto x0b = vld1q_f32(buf[n] + cs + col + 4);
          auto x2b = vld1q_f32(buf[n + 2] + cs + col + 4);
          auto x1b = vld1q_f32(buf[n + 1] + cs + col + 4);
          x1a = vfmaq_f32(x1a, vaddq_f32(x0a, x2a), x05);
          x1b = vfmaq_f32(x1b, vaddq_f32(x0b, x2b), x05);
          vst1q_f32(buf[n + 1] + cs + col, x1a);
          vst1q_f32(buf[n + 1] + cs + col + 4, x1b);
        }
        for (; col < simdlen_s; col += 4) {
          auto x0 = vld1q_f32(buf[n] + cs + col);
          auto x2 = vld1q_f32(buf[n + 2] + cs + col);
          auto x1 = vld1q_f32(buf[n + 1] + cs + col);
          x1 = vfmaq_f32(x1, vaddq_f32(x0, x2), x05);
          vst1q_f32(buf[n + 1] + cs + col, x1);
        }
        for (int32_t c = cs + simdlen_s; c < ce; ++c)
          buf[n + 1][c] += 0.5f * (buf[n][c] + buf[n + 2][c]);
      }
    }
  }
}

#endif