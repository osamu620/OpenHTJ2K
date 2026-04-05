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

#if defined(OPENHTJ2K_ENABLE_WASM_SIMD)
  #include <wasm_simd128.h>
  #include <cmath>
  #include <cstring>
  #include "dwt.hpp"
  #include "utils.hpp"

/********************************************************************************
 * WASM-SIMD helpers: emulate NEON vld2q_f32 / vst2q_f32
 *
 * vld2q_f32 deinterleaves 8 consecutive floats into two vectors of 4:
 *   input:  [a0,b0,a1,b1,a2,b2,a3,b3]
 *   val[0]: [a0,a1,a2,a3]  (even positions — LP)
 *   val[1]: [b0,b1,b2,b3]  (odd  positions — HP)
 *
 * WASM-SIMD has no interleaved load, so we emulate with two wasm_v128_load +
 * two wasm_i32x4_shuffle.  On x86 these lower to movups + unpcklps/unpckhps.
 *******************************************************************************/
struct f32x4x2 {
  v128_t val[2];
};

static inline f32x4x2 vld2q(const float *ptr) {
  v128_t lo   = wasm_v128_load(ptr);
  v128_t hi   = wasm_v128_load(ptr + 4);
  f32x4x2 r;
  r.val[0] = wasm_i32x4_shuffle(lo, hi, 0, 2, 4, 6); // [a0,a1,a2,a3]
  r.val[1] = wasm_i32x4_shuffle(lo, hi, 1, 3, 5, 7); // [b0,b1,b2,b3]
  return r;
}

static inline void vst2q(float *ptr, f32x4x2 x) {
  v128_t lo = wasm_i32x4_shuffle(x.val[0], x.val[1], 0, 4, 1, 5); // [a0,b0,a1,b1]
  v128_t hi = wasm_i32x4_shuffle(x.val[0], x.val[1], 2, 6, 3, 7); // [a2,b2,a3,b3]
  wasm_v128_store(ptr, lo);
  wasm_v128_store(ptr + 4, hi);
}

/********************************************************************************
 * horizontal transforms
 *******************************************************************************/
// irreversible FDWT: x0.HP += coeff * (x0.LP + x1.LP)
static auto fdwt_irrev97_wasm_hor_step = [](const int32_t init_pos, const int32_t simdlen,
                                             float *X, const int32_t n0, const int32_t n1,
                                             float coeff) {
  v128_t vvv = wasm_f32x4_splat(coeff);
  X += init_pos;
  int32_t i = simdlen;
  for (; i > 4; i -= 8) {
    auto x0a   = vld2q(X + n0);
    auto x1a   = vld2q(X + n1);
    auto x0b   = vld2q(X + 8 + n0);
    auto x1b   = vld2q(X + 8 + n1);
    x0a.val[1] = wasm_f32x4_add(x0a.val[1], wasm_f32x4_mul(wasm_f32x4_add(x0a.val[0], x1a.val[0]), vvv));
    x0b.val[1] = wasm_f32x4_add(x0b.val[1], wasm_f32x4_mul(wasm_f32x4_add(x0b.val[0], x1b.val[0]), vvv));
    vst2q(X + n0, x0a);
    vst2q(X + 8 + n0, x0b);
    X += 16;
  }
  for (; i > 0; i -= 4) {
    auto x0   = vld2q(X + n0);
    auto x1   = vld2q(X + n1);
    x0.val[1] = wasm_f32x4_add(x0.val[1], wasm_f32x4_mul(wasm_f32x4_add(x0.val[0], x1.val[0]), vvv));
    vst2q(X + n0, x0);
    X += 8;
  }
};

void fdwt_1d_filtr_irrev97_fixed_wasm(sprec_t *X, const int32_t left, const int32_t u_i0,
                                      const int32_t u_i1) {
  const auto i0        = static_cast<int32_t>(u_i0);
  const auto i1        = static_cast<int32_t>(u_i1);
  const int32_t start  = ceil_int(i0, 2);
  const int32_t stop   = ceil_int(i1, 2);
  const int32_t offset = left + i0 % 2;

  int32_t simdlen = stop + 1 - (start - 2);
  fdwt_irrev97_wasm_hor_step(offset - 4, simdlen, X, 0, 2, fA);
  simdlen = stop + 1 - (start - 1);
  fdwt_irrev97_wasm_hor_step(offset - 2, simdlen, X, -1, 1, fB);
  simdlen = stop - (start - 1);
  fdwt_irrev97_wasm_hor_step(offset - 2, simdlen, X, 0, 2, fC);
  simdlen = stop - start;
  fdwt_irrev97_wasm_hor_step(offset, simdlen, X, -1, 1, fD);
}

// reversible FDWT 5/3
void fdwt_1d_filtr_rev53_fixed_wasm(sprec_t *X, const int32_t left, const int32_t u_i0,
                                    const int32_t u_i1) {
  const auto i0        = static_cast<int32_t>(u_i0);
  const auto i1        = static_cast<int32_t>(u_i1);
  const int32_t start  = ceil_int(i0, 2);
  const int32_t stop   = ceil_int(i1, 2);
  const int32_t offset = left + i0 % 2;

  // step 1: HP[i] -= floor((LP[i] + LP[i+1]) * 0.5)
  int32_t simdlen    = stop - (start - 1);
  const v128_t vhalf = wasm_f32x4_splat(0.5f);
  int32_t n = -2 + offset, i = 0;
  for (; i + 4 < simdlen; i += 8, n += 16) {
    auto xl0a  = vld2q(X + n);
    auto xl1a  = vld2q(X + n + 2);
    auto xl0b  = vld2q(X + n + 8);
    auto xl1b  = vld2q(X + n + 10);
    xl0a.val[1] = wasm_f32x4_sub(xl0a.val[1],
                    wasm_f32x4_floor(wasm_f32x4_mul(wasm_f32x4_add(xl0a.val[0], xl1a.val[0]), vhalf)));
    xl0b.val[1] = wasm_f32x4_sub(xl0b.val[1],
                    wasm_f32x4_floor(wasm_f32x4_mul(wasm_f32x4_add(xl0b.val[0], xl1b.val[0]), vhalf)));
    vst2q(X + n, xl0a);
    vst2q(X + n + 8, xl0b);
  }
  for (; i < simdlen; i += 4, n += 8) {
    auto xl0   = vld2q(X + n);
    auto xl1   = vld2q(X + n + 2);
    xl0.val[1] = wasm_f32x4_sub(xl0.val[1],
                   wasm_f32x4_floor(wasm_f32x4_mul(wasm_f32x4_add(xl0.val[0], xl1.val[0]), vhalf)));
    vst2q(X + n, xl0);
  }

  // step 2: LP[i] += floor((HP[i-1] + HP[i] + 2) * 0.25)
  simdlen           = stop - start;
  const v128_t vqrt = wasm_f32x4_splat(0.25f);
  const v128_t vtwo = wasm_f32x4_splat(2.0f);
  n = 0 + offset; i = 0;
  for (; i + 4 < simdlen; i += 8, n += 16) {
    auto xl0a  = vld2q(X + n - 1);
    auto xl1a  = vld2q(X + n + 1);
    auto xl0b  = vld2q(X + n + 7);
    auto xl1b  = vld2q(X + n + 9);
    xl0a.val[1] = wasm_f32x4_add(xl0a.val[1],
                    wasm_f32x4_floor(wasm_f32x4_mul(
                      wasm_f32x4_add(wasm_f32x4_add(xl0a.val[0], xl1a.val[0]), vtwo), vqrt)));
    xl0b.val[1] = wasm_f32x4_add(xl0b.val[1],
                    wasm_f32x4_floor(wasm_f32x4_mul(
                      wasm_f32x4_add(wasm_f32x4_add(xl0b.val[0], xl1b.val[0]), vtwo), vqrt)));
    vst2q(X + n - 1, xl0a);
    vst2q(X + n + 7, xl0b);
  }
  for (; i < simdlen; i += 4, n += 8) {
    auto xl0   = vld2q(X + n - 1);
    auto xl1   = vld2q(X + n + 1);
    xl0.val[1] = wasm_f32x4_add(xl0.val[1],
                   wasm_f32x4_floor(wasm_f32x4_mul(
                     wasm_f32x4_add(wasm_f32x4_add(xl0.val[0], xl1.val[0]), vtwo), vqrt)));
    vst2q(X + n - 1, xl0);
  }
}

/********************************************************************************
 * vertical transforms
 *******************************************************************************/
// irreversible FDWT vertical step: Xout[i] += coeff * (Xin0[i] + Xin1[i])
static auto fdwt_irrev97_wasm_ver_step = [](const int32_t simdlen, float *const Xin0,
                                             float *const Xin1, float *const Xout, float coeff) {
  v128_t vvv = wasm_f32x4_splat(coeff);
  int32_t n  = 0;
  for (; n + 4 < simdlen; n += 8) {
    v128_t x0a = wasm_v128_load(Xin0 + n);     v128_t x2a = wasm_v128_load(Xin1 + n);     v128_t x1a = wasm_v128_load(Xout + n);
    v128_t x0b = wasm_v128_load(Xin0 + n + 4); v128_t x2b = wasm_v128_load(Xin1 + n + 4); v128_t x1b = wasm_v128_load(Xout + n + 4);
    x1a = wasm_f32x4_add(x1a, wasm_f32x4_mul(vvv, wasm_f32x4_add(x0a, x2a)));
    x1b = wasm_f32x4_add(x1b, wasm_f32x4_mul(vvv, wasm_f32x4_add(x0b, x2b)));
    wasm_v128_store(Xout + n,     x1a);
    wasm_v128_store(Xout + n + 4, x1b);
  }
  for (; n < simdlen; n += 4) {
    v128_t x0 = wasm_v128_load(Xin0 + n);
    v128_t x2 = wasm_v128_load(Xin1 + n);
    v128_t x1 = wasm_v128_load(Xout + n);
    x1 = wasm_f32x4_add(x1, wasm_f32x4_mul(vvv, wasm_f32x4_add(x0, x2)));
    wasm_v128_store(Xout + n, x1);
  }
};

void fdwt_irrev_ver_sr_fixed_wasm(sprec_t *in, const int32_t u0, const int32_t u1, const int32_t v0,
                                  const int32_t v1, const int32_t stride, sprec_t *pse_scratch,
                                  sprec_t **buf_scratch) {
  constexpr int32_t num_pse_i0[2] = {4, 3};
  constexpr int32_t num_pse_i1[2] = {3, 4};
  const int32_t top               = num_pse_i0[v0 % 2];
  const int32_t bottom            = num_pse_i1[v1 % 2];

  if (v0 == v1 - 1) {
    // one sample case
  } else {
    const int32_t len = round_up(stride, SIMD_LEN_I32);
    sprec_t **buf     = buf_scratch;
    for (int32_t i = 1; i <= top; ++i) {
      buf[top - i] = pse_scratch + (i - 1) * len;
      memcpy(buf[top - i], &in[PSEo(v0 - i, v0, v1) * stride],
             sizeof(sprec_t) * static_cast<size_t>(stride));
    }
    for (int32_t row = 0; row < v1 - v0; ++row) {
      buf[top + row] = &in[row * stride];
    }
    for (int32_t i = 1; i <= bottom; i++) {
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
        fdwt_irrev97_wasm_ver_step(simdlen_s, buf[n] + cs, buf[n + 2] + cs, buf[n + 1] + cs, fA);
        for (int32_t col = cs + simdlen_s; col < ce; ++col) {
          buf[n + 1][col] += fA * (buf[n][col] + buf[n + 2][col]);
        }
      }
      for (int32_t n = -2 + offset, i = start - 1; i < stop + 1; i++, n += 2) {
        fdwt_irrev97_wasm_ver_step(simdlen_s, buf[n - 1] + cs, buf[n + 1] + cs, buf[n] + cs, fB);
        for (int32_t col = cs + simdlen_s; col < ce; ++col) {
          buf[n][col] += fB * (buf[n - 1][col] + buf[n + 1][col]);
        }
      }
      for (int32_t n = -2 + offset, i = start - 1; i < stop; i++, n += 2) {
        fdwt_irrev97_wasm_ver_step(simdlen_s, buf[n] + cs, buf[n + 2] + cs, buf[n + 1] + cs, fC);
        for (int32_t col = cs + simdlen_s; col < ce; ++col) {
          buf[n + 1][col] += fC * (buf[n][col] + buf[n + 2][col]);
        }
      }
      for (int32_t n = 0 + offset, i = start; i < stop; i++, n += 2) {
        fdwt_irrev97_wasm_ver_step(simdlen_s, buf[n - 1] + cs, buf[n + 1] + cs, buf[n] + cs, fD);
        for (int32_t col = cs + simdlen_s; col < ce; ++col) {
          buf[n][col] += fD * (buf[n - 1][col] + buf[n + 1][col]);
        }
      }
    }
  }
}

// reversible FDWT 5/3 vertical
void fdwt_rev_ver_sr_fixed_wasm(sprec_t *in, const int32_t u0, const int32_t u1, const int32_t v0,
                                const int32_t v1, const int32_t stride, sprec_t *pse_scratch,
                                sprec_t **buf_scratch) {
  constexpr int32_t num_pse_i0[2] = {2, 1};
  constexpr int32_t num_pse_i1[2] = {1, 2};
  const int32_t top               = num_pse_i0[v0 % 2];
  const int32_t bottom            = num_pse_i1[v1 % 2];

  if (v0 == v1 - 1) {
    for (int32_t col = 0; col < u1 - u0; ++col) {
      if (v0 % 2) in[col] *= 2;
    }
  } else {
    const int32_t len = round_up(stride, SIMD_PADDING);
    sprec_t **buf     = buf_scratch;
    for (int32_t i = 1; i <= top; ++i) {
      buf[top - i] = pse_scratch + (i - 1) * len;
      memcpy(buf[top - i], &in[PSEo(v0 - i, v0, v1) * stride],
             sizeof(sprec_t) * static_cast<size_t>(stride));
    }
    for (int32_t row = 0; row < v1 - v0; ++row) {
      buf[top + row] = &in[row * stride];
    }
    for (int32_t i = 1; i <= bottom; i++) {
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
      // LP step: HP[i] -= floor((LP[i] + LP[i+1]) * 0.5)
      for (int32_t n = -2 + offset, i = start - 1; i < stop; ++i, n += 2) {
        int32_t col = 0;
        for (; col + 4 < simdlen_s; col += 8) {
          v128_t X0a = wasm_v128_load(buf[n] + cs + col);
          v128_t X2a = wasm_v128_load(buf[n + 2] + cs + col);
          v128_t X1a = wasm_v128_load(buf[n + 1] + cs + col);
          v128_t X0b = wasm_v128_load(buf[n] + cs + col + 4);
          v128_t X2b = wasm_v128_load(buf[n + 2] + cs + col + 4);
          v128_t X1b = wasm_v128_load(buf[n + 1] + cs + col + 4);
          X1a = wasm_f32x4_sub(X1a, wasm_f32x4_floor(wasm_f32x4_mul(
                  wasm_f32x4_add(X0a, X2a), wasm_f32x4_splat(0.5f))));
          X1b = wasm_f32x4_sub(X1b, wasm_f32x4_floor(wasm_f32x4_mul(
                  wasm_f32x4_add(X0b, X2b), wasm_f32x4_splat(0.5f))));
          wasm_v128_store(buf[n + 1] + cs + col,     X1a);
          wasm_v128_store(buf[n + 1] + cs + col + 4, X1b);
        }
        for (; col < simdlen_s; col += 4) {
          v128_t X0 = wasm_v128_load(buf[n] + cs + col);
          v128_t X2 = wasm_v128_load(buf[n + 2] + cs + col);
          v128_t X1 = wasm_v128_load(buf[n + 1] + cs + col);
          X1 = wasm_f32x4_sub(X1, wasm_f32x4_floor(wasm_f32x4_mul(
                 wasm_f32x4_add(X0, X2), wasm_f32x4_splat(0.5f))));
          wasm_v128_store(buf[n + 1] + cs + col, X1);
        }
        for (int32_t col2 = cs + simdlen_s; col2 < ce; ++col2) {
          buf[n + 1][col2] -= floorf((buf[n][col2] + buf[n + 2][col2]) * 0.5f);
        }
      }
      // HP step: LP[i] += floor((HP[i-1] + HP[i] + 2) * 0.25)
      const v128_t vtwo = wasm_f32x4_splat(2.0f);
      for (int32_t n = 0 + offset, i = start; i < stop; ++i, n += 2) {
        int32_t col = 0;
        for (; col + 4 < simdlen_s; col += 8) {
          v128_t X0a = wasm_v128_load(buf[n - 1] + cs + col);
          v128_t X2a = wasm_v128_load(buf[n + 1] + cs + col);
          v128_t X1a = wasm_v128_load(buf[n] + cs + col);
          v128_t X0b = wasm_v128_load(buf[n - 1] + cs + col + 4);
          v128_t X2b = wasm_v128_load(buf[n + 1] + cs + col + 4);
          v128_t X1b = wasm_v128_load(buf[n] + cs + col + 4);
          X1a = wasm_f32x4_add(X1a, wasm_f32x4_floor(wasm_f32x4_mul(
                  wasm_f32x4_add(wasm_f32x4_add(X0a, X2a), vtwo), wasm_f32x4_splat(0.25f))));
          X1b = wasm_f32x4_add(X1b, wasm_f32x4_floor(wasm_f32x4_mul(
                  wasm_f32x4_add(wasm_f32x4_add(X0b, X2b), vtwo), wasm_f32x4_splat(0.25f))));
          wasm_v128_store(buf[n] + cs + col,     X1a);
          wasm_v128_store(buf[n] + cs + col + 4, X1b);
        }
        for (; col < simdlen_s; col += 4) {
          v128_t X0 = wasm_v128_load(buf[n - 1] + cs + col);
          v128_t X2 = wasm_v128_load(buf[n + 1] + cs + col);
          v128_t X1 = wasm_v128_load(buf[n] + cs + col);
          X1 = wasm_f32x4_add(X1, wasm_f32x4_floor(wasm_f32x4_mul(
                 wasm_f32x4_add(wasm_f32x4_add(X0, X2), vtwo), wasm_f32x4_splat(0.25f))));
          wasm_v128_store(buf[n] + cs + col, X1);
        }
        for (int32_t col2 = cs + simdlen_s; col2 < ce; ++col2) {
          buf[n][col2] += floorf((buf[n - 1][col2] + buf[n + 1][col2] + 2.0f) * 0.25f);
        }
      }
    }
  }
}

// Single-row rev53 FDWT HP vertical lifting: tgt[i] -= floor((prev[i]+next[i])*0.5).
void fdwt_rev_ver_hp_step_wasm(int32_t n, const float *prev, const float *next, float *tgt) {
  const v128_t k05 = wasm_f32x4_splat(0.5f);
  int32_t i = 0;
  for (; i + 4 < n; i += 8) {
    v128_t a0 = wasm_v128_load(prev + i);     v128_t b0 = wasm_v128_load(next + i);     v128_t t0 = wasm_v128_load(tgt + i);
    v128_t a1 = wasm_v128_load(prev + i + 4); v128_t b1 = wasm_v128_load(next + i + 4); v128_t t1 = wasm_v128_load(tgt + i + 4);
    t0 = wasm_f32x4_sub(t0, wasm_f32x4_floor(wasm_f32x4_mul(wasm_f32x4_add(a0, b0), k05)));
    t1 = wasm_f32x4_sub(t1, wasm_f32x4_floor(wasm_f32x4_mul(wasm_f32x4_add(a1, b1), k05)));
    wasm_v128_store(tgt + i,     t0);
    wasm_v128_store(tgt + i + 4, t1);
  }
  for (; i + 4 <= n; i += 4) {
    v128_t a = wasm_v128_load(prev + i);
    v128_t b = wasm_v128_load(next + i);
    v128_t t = wasm_v128_load(tgt  + i);
    t = wasm_f32x4_sub(t, wasm_f32x4_floor(wasm_f32x4_mul(wasm_f32x4_add(a, b), k05)));
    wasm_v128_store(tgt + i, t);
  }
  for (; i < n; ++i)
    tgt[i] -= floorf((prev[i] + next[i]) * 0.5f);
}

// Single-row rev53 FDWT LP vertical lifting: tgt[i] += floor((prev[i]+next[i]+2)*0.25).
void fdwt_rev_ver_lp_step_wasm(int32_t n, const float *prev, const float *next, float *tgt) {
  const v128_t k025 = wasm_f32x4_splat(0.25f);
  const v128_t k2   = wasm_f32x4_splat(2.0f);
  int32_t i = 0;
  for (; i + 4 < n; i += 8) {
    v128_t a0 = wasm_v128_load(prev + i);     v128_t b0 = wasm_v128_load(next + i);     v128_t t0 = wasm_v128_load(tgt + i);
    v128_t a1 = wasm_v128_load(prev + i + 4); v128_t b1 = wasm_v128_load(next + i + 4); v128_t t1 = wasm_v128_load(tgt + i + 4);
    t0 = wasm_f32x4_add(t0, wasm_f32x4_floor(wasm_f32x4_mul(
           wasm_f32x4_add(wasm_f32x4_add(a0, b0), k2), k025)));
    t1 = wasm_f32x4_add(t1, wasm_f32x4_floor(wasm_f32x4_mul(
           wasm_f32x4_add(wasm_f32x4_add(a1, b1), k2), k025)));
    wasm_v128_store(tgt + i,     t0);
    wasm_v128_store(tgt + i + 4, t1);
  }
  for (; i + 4 <= n; i += 4) {
    v128_t a = wasm_v128_load(prev + i);
    v128_t b = wasm_v128_load(next + i);
    v128_t t = wasm_v128_load(tgt  + i);
    t = wasm_f32x4_add(t, wasm_f32x4_floor(wasm_f32x4_mul(
          wasm_f32x4_add(wasm_f32x4_add(a, b), k2), k025)));
    wasm_v128_store(tgt + i, t);
  }
  for (; i < n; ++i)
    tgt[i] += floorf((prev[i] + next[i] + 2.0f) * 0.25f);
}

#endif  // OPENHTJ2K_ENABLE_WASM_SIMD
