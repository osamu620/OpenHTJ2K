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
  #include <cstdint>
  #include <cstdio>
  #include <cstdlib>
  #include <cstring>
#include <arm_neon.h>
#include <cmath>
  #include <vector>

  // MSVC ARM64 perturbs IDWT FP codegen depending on the inlining context — the
  // same source compiles to slightly different machine code when inlined into a
  // caller (batch path) vs. invoked through a function pointer (stream path),
  // producing 7-9 ULP divergence on the post-IDWT floats for some pixels.
  // Forcing the lifting helpers to live as a single, non-inlined compiled
  // instance gives both paths identical machine code and resolves the lbs
  // failure on lbs_p1_ht_05_11 / lbs_p1_05.  Other compilers/platforms aren't
  // affected by the perturbation; keep them inlinable for performance.
  #if defined(_MSC_VER) && defined(_M_ARM64)
    #define OPENHTJ2K_MSVC_ARM64_NOINLINE __declspec(noinline)
  #else
    #define OPENHTJ2K_MSVC_ARM64_NOINLINE
  #endif

/********************************************************************************
 * horizontal transforms
 *******************************************************************************/
// irreversible IDWT
OPENHTJ2K_MSVC_ARM64_NOINLINE void idwt_1d_filtr_irrev97_fixed_neon(sprec_t *X, const int32_t left,
                                                                    const int32_t u_i0,
                                                                    const int32_t u_i1) {
  const auto i0        = static_cast<int32_t>(u_i0);
  const auto i1        = static_cast<int32_t>(u_i1);
  const int32_t start  = i0 / 2;
  const int32_t stop   = i1 / 2;
  const int32_t offset = left - i0 % 2;

  // Fused single pass over the interleaved row (was four passes, each
  // re-loading and re-storing the full row through vld2q/vst2q).  With
  // E[j] = X[offset+2j] and O[j] = X[offset+2j+1] and N = stop - start:
  //   S1[j] = E[j]  - fD*(O[j-1]  + O[j])     j in [-1, N+1]
  //   S2[j] = O[j]  - fC*(S1[j]   + S1[j+1])  j in [-1, N]
  //   S3[j] = S1[j] - fB*(S2[j-1] + S2[j])    j in [ 0, N]
  //   S4[j] = S2[j] - fA*(S3[j]   + S3[j+1])  j in [ 0, N-1]
  // Final row: E[-1]=S1[-1], O[-1]=S2[-1], E[j]=S3[j], O[j]=S4[j],
  // O[N]=S2[N], E[N+1]=S1[N+1].  Each stage is the same single-rounded FMA on
  // the same inputs as the four-pass version (vfmsq in vectors, fmaf at the
  // boundaries), so the result is bit-identical.
  const int32_t N = stop - start;
  float *const B  = X + offset;

  if (N < 12) {
    // Short rows: four in-place scalar passes (identical formulas/rounding).
    for (int32_t j = -1; j <= N + 1; ++j) B[2 * j] = std::fmaf(-fD, B[2 * j - 1] + B[2 * j + 1], B[2 * j]);
    for (int32_t j = -1; j <= N; ++j) B[2 * j + 1] = std::fmaf(-fC, B[2 * j] + B[2 * j + 2], B[2 * j + 1]);
    for (int32_t j = 0; j <= N; ++j) B[2 * j] = std::fmaf(-fB, B[2 * j - 1] + B[2 * j + 1], B[2 * j]);
    for (int32_t j = 0; j < N; ++j) B[2 * j + 1] = std::fmaf(-fA, B[2 * j] + B[2 * j + 2], B[2 * j + 1]);
    return;
  }

  const float32x4_t vA = vdupq_n_f32(fA), vB = vdupq_n_f32(fB);
  const float32x4_t vC = vdupq_n_f32(fC), vD = vdupq_n_f32(fD);

  // Warmup: j = -1 scalars, then S1 of blocks 0 and 1, S2/S3 of block 0.
  const float s1m1 = std::fmaf(-fD, B[-3] + B[-1], B[-2]);  // S1[-1]
  float32x4x2_t x0 = vld2q_f32(B);
  float32x4_t O_b0 = x0.val[1];
  float32x4_t S1_b0 =
      vfmsq_f32(x0.val[0], vaddq_f32(vextq_f32(vdupq_n_f32(B[-1]), O_b0, 3), O_b0), vD);
  const float s2m1 = std::fmaf(-fC, s1m1 + vgetq_lane_f32(S1_b0, 0), B[-1]);  // S2[-1]
  B[-2]            = s1m1;                                               // final E[-1]
  B[-1]            = s2m1;                                               // final O[-1]

  float32x4x2_t x1 = vld2q_f32(B + 8);
  float32x4_t O_b1 = x1.val[1];
  float32x4_t S1_b1 =
      vfmsq_f32(x1.val[0], vaddq_f32(vextq_f32(O_b0, O_b1, 3), O_b1), vD);
  float32x4_t S2_b0 = vfmsq_f32(O_b0, vaddq_f32(S1_b0, vextq_f32(S1_b0, S1_b1, 1)), vC);
  float32x4_t S3_b0 =
      vfmsq_f32(S1_b0, vaddq_f32(vextq_f32(vdupq_n_f32(s2m1), S2_b0, 3), S2_b0), vB);

  // Steady state: iteration n loads input block n (j = 4n..4n+3) and emits
  // finished block n-2 with one vld2q + one vst2q.
  float32x4_t O_nm1 = O_b1, S1_nm1 = S1_b1, S2_nm2 = S2_b0, S3_nm2 = S3_b0;
  int32_t n = 2;
  for (; 4 * n + 3 <= N + 1; ++n) {
    float32x4x2_t xn = vld2q_f32(B + 8 * n);
    float32x4_t O_n  = xn.val[1];
    float32x4_t S1_n = vfmsq_f32(xn.val[0], vaddq_f32(vextq_f32(O_nm1, O_n, 3), O_n), vD);
    float32x4_t S2_nm1 = vfmsq_f32(O_nm1, vaddq_f32(S1_nm1, vextq_f32(S1_nm1, S1_n, 1)), vC);
    float32x4_t S3_nm1 =
        vfmsq_f32(S1_nm1, vaddq_f32(vextq_f32(S2_nm2, S2_nm1, 3), S2_nm1), vB);
    float32x4x2_t out;
    out.val[0] = S3_nm2;
    out.val[1] = vfmsq_f32(S2_nm2, vaddq_f32(S3_nm2, vextq_f32(S3_nm2, S3_nm1, 1)), vA);
    vst2q_f32(B + 8 * (n - 2), out);
    O_nm1  = O_n;
    S1_nm1 = S1_n;
    S2_nm2 = S2_nm1;
    S3_nm2 = S3_nm1;
  }

  // Drain: scalar finish for blocks n-2, n-1 (still in registers) and the
  // ragged tail.  Loop exit bounds m = 4n to [N-1, N+2], so with base = m-8
  // every stage index j-base fits comfortably in 16.  s1t/s2t/s3t[i] hold
  // S1/S2/S3[base+i]; s1t[0..3] are unused (block n-2's S1 was consumed).
  {
    const int32_t m    = 4 * n;
    const int32_t base = m - 8;
    float s1t[16], s2t[16], s3t[16];
    vst1q_f32(s1t + 4, S1_nm1);  // S1[m-4..m-1]
    vst1q_f32(s2t, S2_nm2);      // S2[m-8..m-5]
    vst1q_f32(s3t, S3_nm2);      // S3[m-8..m-5]
    // Remaining S1 from raw memory (positions >= block n-1 are unwritten)
    for (int32_t j = m; j <= N + 1; ++j)
      s1t[j - base] = std::fmaf(-fD, B[2 * j - 1] + B[2 * j + 1], B[2 * j]);
    for (int32_t j = m - 4; j <= N; ++j)
      s2t[j - base] = std::fmaf(-fC, s1t[j - base] + s1t[j - base + 1], B[2 * j + 1]);
    for (int32_t j = m - 4; j <= N; ++j)
      s3t[j - base] = std::fmaf(-fB, s2t[j - base - 1] + s2t[j - base], s1t[j - base]);
    // Final row: E[j] = S3[j], O[j] = S4[j]; then O[N] = S2[N], E[N+1] = S1[N+1].
    for (int32_t j = base; j <= N - 1; ++j) {
      B[2 * j]     = s3t[j - base];
      B[2 * j + 1] = std::fmaf(-fA, s3t[j - base] + s3t[j - base + 1], s2t[j - base]);
    }
    B[2 * N]       = s3t[N - base];
    B[2 * N + 1]   = s2t[N - base];
    B[2 * (N + 1)] = s1t[N + 1 - base];
  }
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
  const int32_t start  = u_i0 / 2;
  const int32_t stop   = u_i1 / 2;
  const int32_t offset = left - u_i0 % 2;

  // Step 1: LP[k] -= 0.25*(HP[k-1]+HP[k])
  // Starts at offset (not the first real LP sample): per-step extension semantics require the
  // LP pass to lift the even PSE positions adjacent to the data, same bounds as the rev53 kernel.
  int32_t simdlen = stop + 1 - start;
  const auto x025 = vdupq_n_f32(0.25f);
  sprec_t *sp     = X + offset;
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
// The batch path's vertical lifting now routes through the same standalone
// function that the streaming path calls (via idwt_irrev_ver_step_fixed_neon
// below).  The previous file-scope lambda has been removed because it inlined
// into idwt_irrev_ver_sr_fixed_neon under MSVC ARM64, producing different FMS
// machine code than the function-pointer-dispatched stream call — a 7-9 ULP
// post-IDWT divergence on lbs_p1_ht_05_11 / lbs_p1_05.  Sharing the noinline
// function (defined below) keeps both paths bit-identical.

// Single-row irreversible vertical lifting step for idwt_2d_state::adv_step().
// Applies tgt[i] -= coeff*(prev[i]+next[i]) using FMS, matching the batch path exactly.
// n is the row width; ring-buffer rows are always sufficiently aligned.
// Single-row reversible (5/3) LP vertical lifting: tgt[i] -= floor((prev[i]+next[i]+2)*0.25)
// 2× unrolled to match idwt_irrev_ver_step_fixed_neon and hide vrndmq latency.
void idwt_rev_ver_lp_step_neon(int32_t n, const float *prev, const float *next, float *tgt) {
  const float32x4_t k025 = vdupq_n_f32(0.25f);
  const float32x4_t k2   = vdupq_n_f32(2.0f);
  int32_t i = 0;
#if defined(__APPLE__) && defined(__aarch64__)
  for (; i + 12 < n; i += 16) {
    float32x4_t a0 = vld1q_f32(prev + i);      float32x4_t b0 = vld1q_f32(next + i);      float32x4_t t0 = vld1q_f32(tgt + i);
    float32x4_t a1 = vld1q_f32(prev + i + 4);  float32x4_t b1 = vld1q_f32(next + i + 4);  float32x4_t t1 = vld1q_f32(tgt + i + 4);
    float32x4_t a2 = vld1q_f32(prev + i + 8);  float32x4_t b2 = vld1q_f32(next + i + 8);  float32x4_t t2 = vld1q_f32(tgt + i + 8);
    float32x4_t a3 = vld1q_f32(prev + i + 12); float32x4_t b3 = vld1q_f32(next + i + 12); float32x4_t t3 = vld1q_f32(tgt + i + 12);
    t0 = vsubq_f32(t0, vrndmq_f32(vmulq_f32(vaddq_f32(vaddq_f32(a0, b0), k2), k025)));
    t1 = vsubq_f32(t1, vrndmq_f32(vmulq_f32(vaddq_f32(vaddq_f32(a1, b1), k2), k025)));
    t2 = vsubq_f32(t2, vrndmq_f32(vmulq_f32(vaddq_f32(vaddq_f32(a2, b2), k2), k025)));
    t3 = vsubq_f32(t3, vrndmq_f32(vmulq_f32(vaddq_f32(vaddq_f32(a3, b3), k2), k025)));
    vst1q_f32(tgt + i, t0);
    vst1q_f32(tgt + i + 4, t1);
    vst1q_f32(tgt + i + 8, t2);
    vst1q_f32(tgt + i + 12, t3);
  }
#else
  for (; i + 4 < n; i += 8) {
    float32x4_t a0 = vld1q_f32(prev + i);     float32x4_t b0 = vld1q_f32(next + i);     float32x4_t t0 = vld1q_f32(tgt + i);
    float32x4_t a1 = vld1q_f32(prev + i + 4); float32x4_t b1 = vld1q_f32(next + i + 4); float32x4_t t1 = vld1q_f32(tgt + i + 4);
    t0 = vsubq_f32(t0, vrndmq_f32(vmulq_f32(vaddq_f32(vaddq_f32(a0, b0), k2), k025)));
    t1 = vsubq_f32(t1, vrndmq_f32(vmulq_f32(vaddq_f32(vaddq_f32(a1, b1), k2), k025)));
    vst1q_f32(tgt + i, t0);
    vst1q_f32(tgt + i + 4, t1);
  }
#endif
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
#if defined(__APPLE__) && defined(__aarch64__)
  for (; i + 12 < n; i += 16) {
    float32x4_t a0 = vld1q_f32(prev + i);      float32x4_t b0 = vld1q_f32(next + i);      float32x4_t t0 = vld1q_f32(tgt + i);
    float32x4_t a1 = vld1q_f32(prev + i + 4);  float32x4_t b1 = vld1q_f32(next + i + 4);  float32x4_t t1 = vld1q_f32(tgt + i + 4);
    float32x4_t a2 = vld1q_f32(prev + i + 8);  float32x4_t b2 = vld1q_f32(next + i + 8);  float32x4_t t2 = vld1q_f32(tgt + i + 8);
    float32x4_t a3 = vld1q_f32(prev + i + 12); float32x4_t b3 = vld1q_f32(next + i + 12); float32x4_t t3 = vld1q_f32(tgt + i + 12);
    t0 = vaddq_f32(t0, vrndmq_f32(vmulq_f32(vaddq_f32(a0, b0), k05)));
    t1 = vaddq_f32(t1, vrndmq_f32(vmulq_f32(vaddq_f32(a1, b1), k05)));
    t2 = vaddq_f32(t2, vrndmq_f32(vmulq_f32(vaddq_f32(a2, b2), k05)));
    t3 = vaddq_f32(t3, vrndmq_f32(vmulq_f32(vaddq_f32(a3, b3), k05)));
    vst1q_f32(tgt + i, t0);
    vst1q_f32(tgt + i + 4, t1);
    vst1q_f32(tgt + i + 8, t2);
    vst1q_f32(tgt + i + 12, t3);
  }
#else
  for (; i + 4 < n; i += 8) {
    float32x4_t a0 = vld1q_f32(prev + i);     float32x4_t b0 = vld1q_f32(next + i);     float32x4_t t0 = vld1q_f32(tgt + i);
    float32x4_t a1 = vld1q_f32(prev + i + 4); float32x4_t b1 = vld1q_f32(next + i + 4); float32x4_t t1 = vld1q_f32(tgt + i + 4);
    t0 = vaddq_f32(t0, vrndmq_f32(vmulq_f32(vaddq_f32(a0, b0), k05)));
    t1 = vaddq_f32(t1, vrndmq_f32(vmulq_f32(vaddq_f32(a1, b1), k05)));
    vst1q_f32(tgt + i, t0);
    vst1q_f32(tgt + i + 4, t1);
  }
#endif
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

OPENHTJ2K_MSVC_ARM64_NOINLINE void idwt_irrev_ver_step_fixed_neon(int32_t n, float *prev, float *next,
                                                                  float *tgt, float coeff) {
  auto vvv  = vdupq_n_f32(coeff);
  int32_t i = 0;
#if defined(__APPLE__) && defined(__aarch64__)
  for (; i + 12 < n; i += 16) {
    auto x0a = vld1q_f32(prev + i);      auto x2a = vld1q_f32(next + i);      auto x1a = vld1q_f32(tgt + i);
    auto x0b = vld1q_f32(prev + i + 4);  auto x2b = vld1q_f32(next + i + 4);  auto x1b = vld1q_f32(tgt + i + 4);
    auto x0c = vld1q_f32(prev + i + 8);  auto x2c = vld1q_f32(next + i + 8);  auto x1c = vld1q_f32(tgt + i + 8);
    auto x0d = vld1q_f32(prev + i + 12); auto x2d = vld1q_f32(next + i + 12); auto x1d = vld1q_f32(tgt + i + 12);
    x1a = vfmsq_f32(x1a, vaddq_f32(x0a, x2a), vvv);
    x1b = vfmsq_f32(x1b, vaddq_f32(x0b, x2b), vvv);
    x1c = vfmsq_f32(x1c, vaddq_f32(x0c, x2c), vvv);
    x1d = vfmsq_f32(x1d, vaddq_f32(x0d, x2d), vvv);
    vst1q_f32(tgt + i, x1a);
    vst1q_f32(tgt + i + 4, x1b);
    vst1q_f32(tgt + i + 8, x1c);
    vst1q_f32(tgt + i + 12, x1d);
  }
#else
  for (; i + 4 < n; i += 8) {
    auto x0a = vld1q_f32(prev + i);     auto x2a = vld1q_f32(next + i);     auto x1a = vld1q_f32(tgt + i);
    auto x0b = vld1q_f32(prev + i + 4); auto x2b = vld1q_f32(next + i + 4); auto x1b = vld1q_f32(tgt + i + 4);
    x1a = vfmsq_f32(x1a, vaddq_f32(x0a, x2a), vvv);
    x1b = vfmsq_f32(x1b, vaddq_f32(x0b, x2b), vvv);
    vst1q_f32(tgt + i, x1a);
    vst1q_f32(tgt + i + 4, x1b);
  }
#endif
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
      const int32_t ce = (cs + DWT_VERT_STRIP < width) ? cs + DWT_VERT_STRIP : width;
      // Pass the full strip width (ce - cs) to the lifting helper; it handles
      // multiples of 4 via NEON and any odd-tail columns via its scalar branch.
      // Previously this caller had its own scalar tail using a C++
      // `tgt -= a * (prev + next)` form, which MSVC ARM64 compiled with
      // different FMA-contraction behaviour than the streaming path's call to
      // the same helper — producing the 7-9 ULP post-IDWT drift on
      // lbs_p1_ht_05_11 / lbs_p1_05.  Routing every column (including the odd
      // tail) through the single noinline'd helper yields identical bits in
      // batch and stream.  The helper's NEON main loop over-reads up to 3
      // floats past `n`, but every caller (here and in cascade) supplies
      // buffers with at least SIMD_PADDING float scratch beyond the data.
      const int32_t w = ce - cs;
      for (int32_t n = -2 + offset, i = start - 1; i < stop + 2; i++, n += 2) {
        idwt_irrev_ver_step_fixed_neon(w, buf[n - 1] + cs, buf[n + 1] + cs, buf[n] + cs, fD);
      }
      for (int32_t n = -2 + offset, i = start - 1; i < stop + 1; i++, n += 2) {
        idwt_irrev_ver_step_fixed_neon(w, buf[n] + cs, buf[n + 2] + cs, buf[n + 1] + cs, fC);
      }
      for (int32_t n = 0 + offset, i = start; i < stop + 1; i++, n += 2) {
        idwt_irrev_ver_step_fixed_neon(w, buf[n - 1] + cs, buf[n + 1] + cs, buf[n] + cs, fB);
      }
      for (int32_t n = 0 + offset, i = start; i < stop; i++, n += 2) {
        idwt_irrev_ver_step_fixed_neon(w, buf[n] + cs, buf[n + 2] + cs, buf[n + 1] + cs, fA);
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
    // Per-step extension semantics: the LP pass also lifts the even PSE rows adjacent to the
    // data (same bounds as the rev53 vertical kernel) so the HP pass reads post-LP values.
    const int32_t lp_count = v1 / 2 - v0 / 2 + 1;  // LP rows incl. even PSE rows at the edges
    const int32_t hp_count = v1 / 2 - v0 / 2;      // HP row count
    const int32_t offset   = top - v0 % 2;         // first LP row (may be a PSE row)
    const int32_t width    = u1 - u0;
    for (int32_t cs = 0; cs < width; cs += DWT_VERT_STRIP) {
      const int32_t ce        = (cs + DWT_VERT_STRIP < width) ? cs + DWT_VERT_STRIP : width;
      const int32_t simdlen_s = (ce - cs) - (ce - cs) % 4;
      // Step 1: LP[k] -= 0.25*(HP[k-1]+HP[k])
      const auto x025 = vdupq_n_f32(0.25f);
      for (int32_t k = 0, n = offset; k < lp_count; ++k, n += 2) {
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

// ─────────────────────────────────────────────────────────────────────────────
// Int32 reversible 5/3 IDWT primitives — exact inverse of the FDWT int32 path.
// Uses arithmetic right shift instead of float floor/multiply.
// ─────────────────────────────────────────────────────────────────────────────

void idwt_1d_filtr_rev53_i32_neon(int32_t *X, const int32_t left, const int32_t i0, const int32_t i1) {
  const int32_t start  = i0 / 2;
  const int32_t stop   = i1 / 2;
  const int32_t offset = left - i0 % 2;

  // Fused single pass over the interleaved row (was two passes, each re-loading
  // and re-storing the full row through vld2q/vst2q).  With E[j] = X[offset+2j]
  // (LP) and O[j] = X[offset+2j+1] (HP):
  //   S1[j] = E[j] - ((O[j-1] + O[j] + 2) >> 2)   for j in [0, N]   (undo LP update)
  //   S2[j] = O[j] + ((S1[j] + S1[j+1]) >> 1)     for j in [0, N)   (undo HP predict)
  // S2 of a block needs S1[j+4] of the next block; that one lane is computed
  // scalar from the carried raw O[j+3] so the pipeline has no depth.  Integer
  // ops are exact, so the fused pass is bit-identical to the two-pass version.
  const int32_t N      = stop - start;
  int32_t *const B     = X + offset;
  const int32x4_t vtwo = vdupq_n_s32(2);

  int32_t j       = 0;
  int32_t o_carry = B[-1];  // raw O[j-1] entering the current position
  for (; j + 4 <= N; j += 4) {
    int32x4x2_t xb   = vld2q_s32(B + 2 * j);                    // E[j..j+3], O[j..j+3]
    int32x4_t Ojm1   = vextq_s32(vdupq_n_s32(o_carry), xb.val[1], 3);  // O[j-1..j+2]
    int32x4_t S1b    = vsubq_s32(xb.val[0],
                                 vshrq_n_s32(vaddq_s32(vaddq_s32(Ojm1, xb.val[1]), vtwo), 2));
    // S1[j+4] from raw memory (those positions are not yet written this pass)
    const int32_t o3 = vgetq_lane_s32(xb.val[1], 3);            // raw O[j+3]
    const int32_t s1_next = B[2 * (j + 4)] - ((o3 + B[2 * (j + 4) + 1] + 2) >> 2);
    int32x4_t S1n    = vextq_s32(S1b, vdupq_n_s32(s1_next), 1);  // S1[j+1..j+4]
    int32x4x2_t out;
    out.val[0] = S1b;
    out.val[1] = vaddq_s32(xb.val[1], vshrq_n_s32(vaddq_s32(S1b, S1n), 1));
    vst2q_s32(B + 2 * j, out);
    o_carry = o3;
  }
  // Scalar tail: S1 for j..N (odd positions still hold raw O), then S2 for j..N-1.
  {
    int32_t o_prev = o_carry;
    for (int32_t t = j; t <= N; ++t) {
      const int32_t o = B[2 * t + 1];
      B[2 * t] -= (o_prev + o + 2) >> 2;
      o_prev = o;
    }
    for (int32_t t = j; t < N; ++t) {
      B[2 * t + 1] += (B[2 * t] + B[2 * t + 2]) >> 1;
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Planar-input horizontal synthesis — the LP and HP subband rows are read
// directly (E[j] = lp[j], O[j] = hp[j]) and the synthesised natural-domain row
// is written to out[].  Same fused pipelines as the interleaved kernels above,
// but every vld2q on the input side becomes a vld1q per plane and the caller's
// interleave pass disappears.  Boundary taps outside the planes use WSSE
// mirroring: positions u0-k and u0+k have equal parity, so each plane extends
// within itself; PSEo() yields the mirrored in-range position, whose half is
// the plane index.  These mirrored reads return exactly the values the
// in-place kernels found in their PSE-filled margins, and every lifting stage
// is the same single-rounded op on the same inputs — output is bit-identical.
//
// Contract (enforced by idwt_1d_row_from_planar): u0 even, N = u1/2 - u0/2
// >= 12, lp has ceil(u1/2) - u0/2 valid samples, hp has N, out has >= 2
// writable floats before index 0 and >= 8 after index u1-u0-1.
// ─────────────────────────────────────────────────────────────────────────────

OPENHTJ2K_MSVC_ARM64_NOINLINE void idwt_1d_filtr_irrev97_planar_neon(sprec_t *out, const sprec_t *lp,
                                                                     const sprec_t *hp,
                                                                     const int32_t u0,
                                                                     const int32_t u1) {
  const int32_t N = u1 / 2 - u0 / 2;
  // Mirrored raw-plane accessors (used only for warmup/drain boundary taps).
  auto E = [&](int32_t j) -> float { return lp[PSEo(u0 + 2 * j, u0, u1) >> 1]; };
  auto O = [&](int32_t j) -> float { return hp[PSEo(u0 + 2 * j + 1, u0, u1) >> 1]; };

  const float32x4_t vA = vdupq_n_f32(fA), vB = vdupq_n_f32(fB);
  const float32x4_t vC = vdupq_n_f32(fC), vD = vdupq_n_f32(fD);

  // Warmup: j = -1 scalars, then S1 of blocks 0 and 1, S2/S3 of block 0.
  const float om1  = O(-1);
  const float s1m1 = std::fmaf(-fD, O(-2) + om1, E(-1));  // S1[-1]
  float32x4_t O_b0 = vld1q_f32(hp);
  float32x4_t S1_b0 =
      vfmsq_f32(vld1q_f32(lp), vaddq_f32(vextq_f32(vdupq_n_f32(om1), O_b0, 3), O_b0), vD);
  const float s2m1 = std::fmaf(-fC, s1m1 + vgetq_lane_f32(S1_b0, 0), om1);  // S2[-1]
  out[-2]          = s1m1;                                                  // final E[-1]
  out[-1]          = s2m1;                                                  // final O[-1]

  float32x4_t O_b1 = vld1q_f32(hp + 4);
  float32x4_t S1_b1 =
      vfmsq_f32(vld1q_f32(lp + 4), vaddq_f32(vextq_f32(O_b0, O_b1, 3), O_b1), vD);
  float32x4_t S2_b0 = vfmsq_f32(O_b0, vaddq_f32(S1_b0, vextq_f32(S1_b0, S1_b1, 1)), vC);
  float32x4_t S3_b0 =
      vfmsq_f32(S1_b0, vaddq_f32(vextq_f32(vdupq_n_f32(s2m1), S2_b0, 3), S2_b0), vB);

  // Steady state: iteration n loads input block n (j = 4n..4n+3) with one
  // vld1q per plane and emits finished block n-2 with one vst2q.  The bound
  // is 4n+3 <= N-1 (vs N+1 for the in-place kernel) because the planes have
  // no PSE-filled margins to read past — the drain covers the rest scalar.
  float32x4_t O_nm1 = O_b1, S1_nm1 = S1_b1, S2_nm2 = S2_b0, S3_nm2 = S3_b0;
  int32_t n = 2;
  for (; 4 * n + 3 <= N - 1; ++n) {
    float32x4_t O_n  = vld1q_f32(hp + 4 * n);
    float32x4_t S1_n = vfmsq_f32(vld1q_f32(lp + 4 * n), vaddq_f32(vextq_f32(O_nm1, O_n, 3), O_n), vD);
    float32x4_t S2_nm1 = vfmsq_f32(O_nm1, vaddq_f32(S1_nm1, vextq_f32(S1_nm1, S1_n, 1)), vC);
    float32x4_t S3_nm1 =
        vfmsq_f32(S1_nm1, vaddq_f32(vextq_f32(S2_nm2, S2_nm1, 3), S2_nm1), vB);
    float32x4x2_t o;
    o.val[0] = S3_nm2;
    o.val[1] = vfmsq_f32(S2_nm2, vaddq_f32(S3_nm2, vextq_f32(S3_nm2, S3_nm1, 1)), vA);
    vst2q_f32(out + 8 * (n - 2), o);
    O_nm1  = O_n;
    S1_nm1 = S1_n;
    S2_nm2 = S2_nm1;
    S3_nm2 = S3_nm1;
  }

  // Drain: scalar finish for blocks n-2, n-1 (still in registers) and the
  // ragged tail.  Loop exit bounds m = 4n to [N-3, N], so with base = m-8
  // every stage index j-base fits in 16.  s1t/s2t/s3t[i] hold S1/S2/S3[base+i];
  // s1t[0..3] are unused (block n-2's S1 was consumed).
  {
    const int32_t m    = 4 * n;
    const int32_t base = m - 8;
    float s1t[16], s2t[16], s3t[16];
    vst1q_f32(s1t + 4, S1_nm1);  // S1[m-4..m-1]
    vst1q_f32(s2t, S2_nm2);      // S2[m-8..m-5]
    vst1q_f32(s3t, S3_nm2);      // S3[m-8..m-5]
    for (int32_t j = m; j <= N + 1; ++j)
      s1t[j - base] = std::fmaf(-fD, O(j - 1) + O(j), E(j));
    for (int32_t j = m - 4; j <= N; ++j)
      s2t[j - base] = std::fmaf(-fC, s1t[j - base] + s1t[j - base + 1], O(j));
    for (int32_t j = m - 4; j <= N; ++j)
      s3t[j - base] = std::fmaf(-fB, s2t[j - base - 1] + s2t[j - base], s1t[j - base]);
    // Final row: E[j] = S3[j], O[j] = S4[j]; then O[N] = S2[N], E[N+1] = S1[N+1].
    for (int32_t j = base; j <= N - 1; ++j) {
      out[2 * j]     = s3t[j - base];
      out[2 * j + 1] = std::fmaf(-fA, s3t[j - base] + s3t[j - base + 1], s2t[j - base]);
    }
    out[2 * N]       = s3t[N - base];
    out[2 * N + 1]   = s2t[N - base];
    out[2 * (N + 1)] = s1t[N + 1 - base];
  }
}

// Sub-range 9/7 planar synthesis — 4-lane NEON port of the AVX2
// idwt_1d_filtr_irrev97_planar_sr_avx2 (see it for the rationale and the
// bit-exactness argument).  vfmsq_f32 is single-rounded and equal to
// std::fmaf(-coeff, sum, x), so the vector lanes, the scalar drains and the
// other platforms' kernels all produce identical output.
void idwt_1d_filtr_irrev97_planar_sr_neon(sprec_t *out, const sprec_t *lp, const sprec_t *hp,
                                          const int32_t u0, const int32_t u1, const int32_t col_lo,
                                          const int32_t col_hi) {
  const int32_t N     = u1 / 2 - u0 / 2;
  const int32_t width = u1 - u0;
  int32_t row_lo      = col_lo - u0 - 4;
  int32_t row_hi      = col_hi - u0 - 1 + 4;
  if (row_lo < 0) row_lo = 0;
  if (row_hi > width - 1) row_hi = width - 1;
  if (row_lo > row_hi) return;
  const int32_t J0 = row_lo >> 1;
  const int32_t J1 = row_hi >> 1;

  const int32_t base = J0 - 1;
  const int32_t M    = (J1 + 2) - base + 1;
  thread_local std::vector<float> s1v, s2v, s3v;
  if (static_cast<int32_t>(s1v.size()) < M) {
    s1v.resize(M);
    s2v.resize(M);
    s3v.resize(M);
  }
  float *const s1 = s1v.data() - base;
  float *const s2 = s2v.data() - base;
  float *const s3 = s3v.data() - base;

  if (J0 >= 2 && J1 <= N - 3) {
    const float32x4_t vA = vdupq_n_f32(fA), vB = vdupq_n_f32(fB);
    const float32x4_t vC = vdupq_n_f32(fC), vD = vdupq_n_f32(fD);
    int32_t j;
    // Pass 1: S1[j] = E[j] - fD*(O[j-1] + O[j])
    for (j = J0 - 1; j + 4 <= J1 + 3; j += 4)
      vst1q_f32(s1 + j,
                vfmsq_f32(vld1q_f32(lp + j), vaddq_f32(vld1q_f32(hp + j - 1), vld1q_f32(hp + j)), vD));
    for (; j <= J1 + 2; ++j) s1[j] = std::fmaf(-fD, hp[j - 1] + hp[j], lp[j]);
    // Pass 2: S2[j] = O[j] - fC*(S1[j] + S1[j+1])
    for (j = J0 - 1; j + 4 <= J1 + 2; j += 4)
      vst1q_f32(s2 + j,
                vfmsq_f32(vld1q_f32(hp + j), vaddq_f32(vld1q_f32(s1 + j), vld1q_f32(s1 + j + 1)), vC));
    for (; j <= J1 + 1; ++j) s2[j] = std::fmaf(-fC, s1[j] + s1[j + 1], hp[j]);
    // Pass 3: S3[j] = S1[j] - fB*(S2[j-1] + S2[j])
    for (j = J0; j + 4 <= J1 + 2; j += 4)
      vst1q_f32(s3 + j,
                vfmsq_f32(vld1q_f32(s1 + j), vaddq_f32(vld1q_f32(s2 + j - 1), vld1q_f32(s2 + j)), vB));
    for (; j <= J1 + 1; ++j) s3[j] = std::fmaf(-fB, s2[j - 1] + s2[j], s1[j]);
    // Pass 4 + interleaved store: out[2j]=S3[j], out[2j+1]=S2[j]-fA*(S3[j]+S3[j+1])
    for (j = J0; j + 4 <= J1 + 1; j += 4) {
      float32x4_t s3j = vld1q_f32(s3 + j);
      float32x4x2_t st;
      st.val[0] = s3j;
      st.val[1] = vfmsq_f32(vld1q_f32(s2 + j), vaddq_f32(s3j, vld1q_f32(s3 + j + 1)), vA);
      vst2q_f32(out + 2 * j, st);
    }
    for (; j <= J1; ++j) {
      out[2 * j]     = s3[j];
      out[2 * j + 1] = std::fmaf(-fA, s3[j] + s3[j + 1], s2[j]);
    }
  } else {
    auto E = [&](int32_t j) -> float { return lp[PSEo(u0 + 2 * j, u0, u1) >> 1]; };
    auto O = [&](int32_t j) -> float { return hp[PSEo(u0 + 2 * j + 1, u0, u1) >> 1]; };
    for (int32_t j = J0 - 1; j <= J1 + 2; ++j) s1[j] = std::fmaf(-fD, O(j - 1) + O(j), E(j));
    for (int32_t j = J0 - 1; j <= J1 + 1; ++j) s2[j] = std::fmaf(-fC, s1[j] + s1[j + 1], O(j));
    for (int32_t j = J0; j <= J1 + 1; ++j) s3[j] = std::fmaf(-fB, s2[j - 1] + s2[j], s1[j]);
    for (int32_t j = J0; j <= J1; ++j) {
      out[2 * j]     = s3[j];
      out[2 * j + 1] = std::fmaf(-fA, s3[j] + s3[j + 1], s2[j]);
    }
  }
}

void idwt_1d_filtr_rev53_planar_i32_neon(int32_t *out, const int32_t *lp, const int32_t *hp,
                                         const int32_t u0, const int32_t u1) {
  const int32_t N = u1 / 2 - u0 / 2;
  auto E = [&](int32_t j) -> int32_t { return lp[PSEo(u0 + 2 * j, u0, u1) >> 1]; };
  auto O = [&](int32_t j) -> int32_t { return hp[PSEo(u0 + 2 * j + 1, u0, u1) >> 1]; };

  //   S1[j] = E[j] - ((O[j-1] + O[j] + 2) >> 2)   for j in [0, N]   (undo LP update)
  //   S2[j] = O[j] + ((S1[j] + S1[j+1]) >> 1)     for j in [0, N)   (undo HP predict)
  // Integer ops are exact, so this matches idwt_1d_filtr_rev53_i32_neon bit
  // for bit.  Loop bound j+4 <= N-1 keeps every plane read in range (the
  // s1_next lookahead reads lp[j+4] / hp[j+4]); the scalar tail mirrors.
  const int32x4_t vtwo = vdupq_n_s32(2);
  int32_t j       = 0;
  int32_t o_carry = O(-1);  // raw O[j-1] entering the current position
  for (; j + 4 <= N - 1; j += 4) {
    int32x4_t Ob   = vld1q_s32(hp + j);
    int32x4_t Ojm1 = vextq_s32(vdupq_n_s32(o_carry), Ob, 3);  // O[j-1..j+2]
    int32x4_t S1b  = vsubq_s32(vld1q_s32(lp + j),
                               vshrq_n_s32(vaddq_s32(vaddq_s32(Ojm1, Ob), vtwo), 2));
    const int32_t o3      = vgetq_lane_s32(Ob, 3);  // raw O[j+3]
    const int32_t s1_next = lp[j + 4] - ((o3 + hp[j + 4] + 2) >> 2);
    int32x4_t S1n         = vextq_s32(S1b, vdupq_n_s32(s1_next), 1);  // S1[j+1..j+4]
    int32x4x2_t o;
    o.val[0] = S1b;
    o.val[1] = vaddq_s32(Ob, vshrq_n_s32(vaddq_s32(S1b, S1n), 1));
    vst2q_s32(out + 2 * j, o);
    o_carry = o3;
  }
  // Scalar tail (at most 5 S1 values: loop exits with N - j <= 4).
  {
    int32_t s1t[8];
    int32_t o_prev = o_carry;
    for (int32_t t = j; t <= N; ++t) {
      const int32_t o = O(t);
      s1t[t - j]      = E(t) - ((o_prev + o + 2) >> 2);
      o_prev          = o;
    }
    for (int32_t t = j; t <= N; ++t) out[2 * t] = s1t[t - j];
    for (int32_t t = j; t < N; ++t) out[2 * t + 1] = O(t) + ((s1t[t - j] + s1t[t - j + 1]) >> 1);
  }
}

void idwt_rev_ver_lp_step_i32_neon(int32_t n, const int32_t *prev, const int32_t *next, int32_t *tgt) {
  const int32x4_t vtwo = vdupq_n_s32(2);
  int32_t i = 0;
  for (; i + 8 <= n; i += 8) {
    int32x4_t a0 = vld1q_s32(prev + i);     int32x4_t b0 = vld1q_s32(next + i);     int32x4_t t0 = vld1q_s32(tgt + i);
    int32x4_t a1 = vld1q_s32(prev + i + 4); int32x4_t b1 = vld1q_s32(next + i + 4); int32x4_t t1 = vld1q_s32(tgt + i + 4);
    t0 = vsubq_s32(t0, vshrq_n_s32(vaddq_s32(vaddq_s32(a0, b0), vtwo), 2));
    t1 = vsubq_s32(t1, vshrq_n_s32(vaddq_s32(vaddq_s32(a1, b1), vtwo), 2));
    vst1q_s32(tgt + i, t0);
    vst1q_s32(tgt + i + 4, t1);
  }
  for (; i + 4 <= n; i += 4) {
    int32x4_t a = vld1q_s32(prev + i);
    int32x4_t b = vld1q_s32(next + i);
    int32x4_t t = vld1q_s32(tgt  + i);
    t = vsubq_s32(t, vshrq_n_s32(vaddq_s32(vaddq_s32(a, b), vtwo), 2));
    vst1q_s32(tgt + i, t);
  }
  for (; i < n; ++i) tgt[i] -= (prev[i] + next[i] + 2) >> 2;
}

void idwt_rev_ver_hp_step_i32_neon(int32_t n, const int32_t *prev, const int32_t *next, int32_t *tgt) {
  int32_t i = 0;
  for (; i + 8 <= n; i += 8) {
    int32x4_t a0 = vld1q_s32(prev + i);     int32x4_t b0 = vld1q_s32(next + i);     int32x4_t t0 = vld1q_s32(tgt + i);
    int32x4_t a1 = vld1q_s32(prev + i + 4); int32x4_t b1 = vld1q_s32(next + i + 4); int32x4_t t1 = vld1q_s32(tgt + i + 4);
    t0 = vaddq_s32(t0, vshrq_n_s32(vaddq_s32(a0, b0), 1));
    t1 = vaddq_s32(t1, vshrq_n_s32(vaddq_s32(a1, b1), 1));
    vst1q_s32(tgt + i, t0);
    vst1q_s32(tgt + i + 4, t1);
  }
  for (; i + 4 <= n; i += 4) {
    int32x4_t a = vld1q_s32(prev + i);
    int32x4_t b = vld1q_s32(next + i);
    int32x4_t t = vld1q_s32(tgt  + i);
    t = vaddq_s32(t, vshrq_n_s32(vaddq_s32(a, b), 1));
    vst1q_s32(tgt + i, t);
  }
  for (; i < n; ++i) tgt[i] += (prev[i] + next[i]) >> 1;
}

#endif