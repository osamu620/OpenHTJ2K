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
 * WASM-SIMD helpers (local to this TU)
 *******************************************************************************/
struct f32x4x2 {
  v128_t val[2];
};

static inline f32x4x2 vld2q(const float *ptr) {
  v128_t lo = wasm_v128_load(ptr);
  v128_t hi = wasm_v128_load(ptr + 4);
  f32x4x2 r;
  r.val[0] = wasm_i32x4_shuffle(lo, hi, 0, 2, 4, 6);
  r.val[1] = wasm_i32x4_shuffle(lo, hi, 1, 3, 5, 7);
  return r;
}

static inline void vst2q(float *ptr, f32x4x2 x) {
  v128_t lo = wasm_i32x4_shuffle(x.val[0], x.val[1], 0, 4, 1, 5);
  v128_t hi = wasm_i32x4_shuffle(x.val[0], x.val[1], 2, 6, 3, 7);
  wasm_v128_store(ptr, lo);
  wasm_v128_store(ptr + 4, hi);
}

/********************************************************************************
 * horizontal transforms
 *******************************************************************************/
// irreversible IDWT: x0.HP -= coeff * (x0.LP + x1.LP)
static auto idwt_irrev97_wasm_hor_step = [](const int32_t init_pos, const int32_t simdlen,
                                             float *const X, const int32_t n0, const int32_t n1,
                                             float coeff) {
  v128_t vvv = wasm_f32x4_splat(coeff);
  int32_t n = init_pos, i = simdlen;
  for (; i > 4; i -= 8, n += 16) {
    auto x0a   = vld2q(X + n + n0);
    auto x1a   = vld2q(X + n + n1);
    auto x0b   = vld2q(X + n + 8 + n0);
    auto x1b   = vld2q(X + n + 8 + n1);
    x0a.val[1] = wasm_f32x4_sub(x0a.val[1], wasm_f32x4_mul(wasm_f32x4_add(x0a.val[0], x1a.val[0]), vvv));
    x0b.val[1] = wasm_f32x4_sub(x0b.val[1], wasm_f32x4_mul(wasm_f32x4_add(x0b.val[0], x1b.val[0]), vvv));
    vst2q(X + n + n0, x0a);
    vst2q(X + n + 8 + n0, x0b);
  }
  for (; i > 0; i -= 4, n += 8) {
    auto x0   = vld2q(X + n + n0);
    auto x1   = vld2q(X + n + n1);
    x0.val[1] = wasm_f32x4_sub(x0.val[1], wasm_f32x4_mul(wasm_f32x4_add(x0.val[0], x1.val[0]), vvv));
    vst2q(X + n + n0, x0);
  }
};

void idwt_1d_filtr_irrev97_fixed_wasm(sprec_t *X, const int32_t left, const int32_t u_i0,
                                      const int32_t u_i1) {
  const auto i0        = static_cast<int32_t>(u_i0);
  const auto i1        = static_cast<int32_t>(u_i1);
  const int32_t start  = i0 / 2;
  const int32_t stop   = i1 / 2;
  const int32_t offset = left - i0 % 2;

  int32_t simdlen = stop + 2 - (start - 1);
  idwt_irrev97_wasm_hor_step(offset - 2, simdlen, X, -1, 1, fD);
  simdlen = stop + 1 - (start - 1);
  idwt_irrev97_wasm_hor_step(offset - 2, simdlen, X, 0, 2, fC);
  simdlen = stop + 1 - start;
  idwt_irrev97_wasm_hor_step(offset, simdlen, X, -1, 1, fB);
  simdlen = stop - start;
  idwt_irrev97_wasm_hor_step(offset, simdlen, X, 0, 2, fA);
}

// reversible IDWT 5/3
void idwt_1d_filtr_rev53_fixed_wasm(sprec_t *X, const int32_t left, const int32_t i0,
                                    const int32_t i1) {
  const int32_t start  = i0 / 2;
  const int32_t stop   = i1 / 2;
  const int32_t offset = left - i0 % 2;

  // step 1: LP[i] -= floor((HP[i-1] + HP[i] + 2) * 0.25)
  const int32_t base1 = offset;
  int32_t simdlen     = stop + 1 - start;
  const v128_t vtwo   = wasm_f32x4_splat(2.0f);
  const v128_t vqrt   = wasm_f32x4_splat(0.25f);
  int32_t k = 0;
  for (; k + 4 < simdlen; k += 8) {
    sprec_t *sp = X + base1 + k * 2;
    auto xl0a   = vld2q(sp - 1);
    auto xl1a   = vld2q(sp + 1);
    auto xl0b   = vld2q(sp + 7);
    auto xl1b   = vld2q(sp + 9);
    xl0a.val[1] = wasm_f32x4_sub(xl0a.val[1],
                    wasm_f32x4_floor(wasm_f32x4_mul(
                      wasm_f32x4_add(wasm_f32x4_add(xl0a.val[0], xl1a.val[0]), vtwo), vqrt)));
    xl0b.val[1] = wasm_f32x4_sub(xl0b.val[1],
                    wasm_f32x4_floor(wasm_f32x4_mul(
                      wasm_f32x4_add(wasm_f32x4_add(xl0b.val[0], xl1b.val[0]), vtwo), vqrt)));
    vst2q(sp - 1, xl0a);
    vst2q(sp + 7, xl0b);
  }
  for (; k < simdlen; k += 4) {
    sprec_t *sp = X + base1 + k * 2;
    auto xl0    = vld2q(sp - 1);
    auto xl1    = vld2q(sp + 1);
    xl0.val[1]  = wasm_f32x4_sub(xl0.val[1],
                    wasm_f32x4_floor(wasm_f32x4_mul(
                      wasm_f32x4_add(wasm_f32x4_add(xl0.val[0], xl1.val[0]), vtwo), vqrt)));
    vst2q(sp - 1, xl0);
  }

  // step 2: HP[i] += floor((LP[i] + LP[i+1]) * 0.5)
  const int32_t base2 = offset;
  simdlen             = stop - start;
  const v128_t vhalf  = wasm_f32x4_splat(0.5f);
  k                   = 0;
  for (; k + 4 < simdlen; k += 8) {
    sprec_t *sp = X + base2 + k * 2;
    auto xl0a   = vld2q(sp);
    auto xl1a   = vld2q(sp + 2);
    auto xl0b   = vld2q(sp + 8);
    auto xl1b   = vld2q(sp + 10);
    xl0a.val[1] = wasm_f32x4_add(xl0a.val[1],
                    wasm_f32x4_floor(wasm_f32x4_mul(wasm_f32x4_add(xl0a.val[0], xl1a.val[0]), vhalf)));
    xl0b.val[1] = wasm_f32x4_add(xl0b.val[1],
                    wasm_f32x4_floor(wasm_f32x4_mul(wasm_f32x4_add(xl0b.val[0], xl1b.val[0]), vhalf)));
    vst2q(sp, xl0a);
    vst2q(sp + 8, xl0b);
  }
  for (; k < simdlen; k += 4) {
    sprec_t *sp = X + base2 + k * 2;
    auto xl0    = vld2q(sp);
    auto xl1    = vld2q(sp + 2);
    xl0.val[1]  = wasm_f32x4_add(xl0.val[1],
                    wasm_f32x4_floor(wasm_f32x4_mul(wasm_f32x4_add(xl0.val[0], xl1.val[0]), vhalf)));
    vst2q(sp, xl0);
  }
}

/********************************************************************************
 * vertical transform — single-row step functions (used by idwt_2d_state::adv_step)
 *******************************************************************************/
// tgt[i] -= coeff * (prev[i] + next[i])
void idwt_irrev_ver_step_fixed_wasm(int32_t n, float *prev, float *next, float *tgt, float coeff) {
  v128_t vvv = wasm_f32x4_splat(coeff);
  int32_t i  = 0;
  for (; i + 4 < n; i += 8) {
    v128_t x0a = wasm_v128_load(prev + i);     v128_t x2a = wasm_v128_load(next + i);     v128_t x1a = wasm_v128_load(tgt + i);
    v128_t x0b = wasm_v128_load(prev + i + 4); v128_t x2b = wasm_v128_load(next + i + 4); v128_t x1b = wasm_v128_load(tgt + i + 4);
    x1a = wasm_f32x4_sub(x1a, wasm_f32x4_mul(wasm_f32x4_add(x0a, x2a), vvv));
    x1b = wasm_f32x4_sub(x1b, wasm_f32x4_mul(wasm_f32x4_add(x0b, x2b), vvv));
    wasm_v128_store(tgt + i,     x1a);
    wasm_v128_store(tgt + i + 4, x1b);
  }
  for (; i + 4 <= n; i += 4) {
    v128_t x0 = wasm_v128_load(prev + i);
    v128_t x2 = wasm_v128_load(next + i);
    v128_t x1 = wasm_v128_load(tgt  + i);
    x1 = wasm_f32x4_sub(x1, wasm_f32x4_mul(wasm_f32x4_add(x0, x2), vvv));
    wasm_v128_store(tgt + i, x1);
  }
  for (; i < n; ++i)
    tgt[i] -= coeff * (prev[i] + next[i]);
}

// tgt[i] -= floor((prev[i] + next[i] + 2) * 0.25)
void idwt_rev_ver_lp_step_wasm(int32_t n, const float *prev, const float *next, float *tgt) {
  const v128_t k025 = wasm_f32x4_splat(0.25f);
  const v128_t k2   = wasm_f32x4_splat(2.0f);
  int32_t i = 0;
  for (; i + 4 < n; i += 8) {
    v128_t a0 = wasm_v128_load(prev + i);     v128_t b0 = wasm_v128_load(next + i);     v128_t t0 = wasm_v128_load(tgt + i);
    v128_t a1 = wasm_v128_load(prev + i + 4); v128_t b1 = wasm_v128_load(next + i + 4); v128_t t1 = wasm_v128_load(tgt + i + 4);
    t0 = wasm_f32x4_sub(t0, wasm_f32x4_floor(wasm_f32x4_mul(
           wasm_f32x4_add(wasm_f32x4_add(a0, b0), k2), k025)));
    t1 = wasm_f32x4_sub(t1, wasm_f32x4_floor(wasm_f32x4_mul(
           wasm_f32x4_add(wasm_f32x4_add(a1, b1), k2), k025)));
    wasm_v128_store(tgt + i,     t0);
    wasm_v128_store(tgt + i + 4, t1);
  }
  for (; i + 4 <= n; i += 4) {
    v128_t a = wasm_v128_load(prev + i);
    v128_t b = wasm_v128_load(next + i);
    v128_t t = wasm_v128_load(tgt  + i);
    t = wasm_f32x4_sub(t, wasm_f32x4_floor(wasm_f32x4_mul(
          wasm_f32x4_add(wasm_f32x4_add(a, b), k2), k025)));
    wasm_v128_store(tgt + i, t);
  }
  for (; i < n; ++i)
    tgt[i] -= floorf((prev[i] + next[i] + 2.0f) * 0.25f);
}

// tgt[i] += floor((prev[i] + next[i]) * 0.5)
void idwt_rev_ver_hp_step_wasm(int32_t n, const float *prev, const float *next, float *tgt) {
  const v128_t k05 = wasm_f32x4_splat(0.5f);
  int32_t i = 0;
  for (; i + 4 < n; i += 8) {
    v128_t a0 = wasm_v128_load(prev + i);     v128_t b0 = wasm_v128_load(next + i);     v128_t t0 = wasm_v128_load(tgt + i);
    v128_t a1 = wasm_v128_load(prev + i + 4); v128_t b1 = wasm_v128_load(next + i + 4); v128_t t1 = wasm_v128_load(tgt + i + 4);
    t0 = wasm_f32x4_add(t0, wasm_f32x4_floor(wasm_f32x4_mul(wasm_f32x4_add(a0, b0), k05)));
    t1 = wasm_f32x4_add(t1, wasm_f32x4_floor(wasm_f32x4_mul(wasm_f32x4_add(a1, b1), k05)));
    wasm_v128_store(tgt + i,     t0);
    wasm_v128_store(tgt + i + 4, t1);
  }
  for (; i + 4 <= n; i += 4) {
    v128_t a = wasm_v128_load(prev + i);
    v128_t b = wasm_v128_load(next + i);
    v128_t t = wasm_v128_load(tgt  + i);
    t = wasm_f32x4_add(t, wasm_f32x4_floor(wasm_f32x4_mul(wasm_f32x4_add(a, b), k05)));
    wasm_v128_store(tgt + i, t);
  }
  for (; i < n; ++i)
    tgt[i] += floorf((prev[i] + next[i]) * 0.5f);
}

/********************************************************************************
 * vertical transforms — batch (full-tile)
 *******************************************************************************/
// irreversible IDWT 9/7 vertical step (internal lambda)
static auto idwt_irrev97_wasm_ver_step = [](const int32_t simdlen, float *const Xin0,
                                             float *const Xin1, float *const Xout, float coeff) {
  v128_t vvv = wasm_f32x4_splat(coeff);
  int32_t n  = 0;
  for (; n + 4 < simdlen; n += 8) {
    v128_t x0a = wasm_v128_load(Xin0 + n);     v128_t x2a = wasm_v128_load(Xin1 + n);     v128_t x1a = wasm_v128_load(Xout + n);
    v128_t x0b = wasm_v128_load(Xin0 + n + 4); v128_t x2b = wasm_v128_load(Xin1 + n + 4); v128_t x1b = wasm_v128_load(Xout + n + 4);
    x1a = wasm_f32x4_sub(x1a, wasm_f32x4_mul(wasm_f32x4_add(x0a, x2a), vvv));
    x1b = wasm_f32x4_sub(x1b, wasm_f32x4_mul(wasm_f32x4_add(x0b, x2b), vvv));
    wasm_v128_store(Xout + n,     x1a);
    wasm_v128_store(Xout + n + 4, x1b);
  }
  for (; n < simdlen; n += 4) {
    v128_t x0 = wasm_v128_load(Xin0 + n);
    v128_t x2 = wasm_v128_load(Xin1 + n);
    v128_t x1 = wasm_v128_load(Xout + n);
    x1 = wasm_f32x4_sub(x1, wasm_f32x4_mul(wasm_f32x4_add(x0, x2), vvv));
    wasm_v128_store(Xout + n, x1);
  }
};

void idwt_irrev_ver_sr_fixed_wasm(sprec_t *in, const int32_t u0, const int32_t u1, const int32_t v0,
                                  const int32_t v1, const int32_t stride, sprec_t *pse_scratch,
                                  sprec_t **buf_scratch) {
  constexpr int32_t num_pse_i0[2] = {3, 4};
  constexpr int32_t num_pse_i1[2] = {4, 3};
  const int32_t top               = num_pse_i0[v0 % 2];
  const int32_t bottom            = num_pse_i1[v1 % 2];
  if (v0 == v1 - 1) {
    // one sample case
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
    const int32_t start  = v0 / 2;
    const int32_t stop   = v1 / 2;
    const int32_t offset = top - v0 % 2;

    const int32_t width = u1 - u0;
    for (int32_t cs = 0; cs < width; cs += DWT_VERT_STRIP) {
      const int32_t ce        = (cs + DWT_VERT_STRIP < width) ? cs + DWT_VERT_STRIP : width;
      const int32_t simdlen_s = (ce - cs) - (ce - cs) % 4;
      for (int32_t n = -2 + offset, i = start - 1; i < stop + 2; i++, n += 2) {
        idwt_irrev97_wasm_ver_step(simdlen_s, buf[n - 1] + cs, buf[n + 1] + cs, buf[n] + cs, fD);
        for (int32_t col = cs + simdlen_s; col < ce; ++col) {
          buf[n][col] -= (buf[n - 1][col] + buf[n + 1][col]) * fD;
        }
      }
      for (int32_t n = -2 + offset, i = start - 1; i < stop + 1; i++, n += 2) {
        idwt_irrev97_wasm_ver_step(simdlen_s, buf[n] + cs, buf[n + 2] + cs, buf[n + 1] + cs, fC);
        for (int32_t col = cs + simdlen_s; col < ce; ++col) {
          buf[n + 1][col] -= (buf[n][col] + buf[n + 2][col]) * fC;
        }
      }
      for (int32_t n = 0 + offset, i = start; i < stop + 1; i++, n += 2) {
        idwt_irrev97_wasm_ver_step(simdlen_s, buf[n - 1] + cs, buf[n + 1] + cs, buf[n] + cs, fB);
        for (int32_t col = cs + simdlen_s; col < ce; ++col) {
          buf[n][col] -= (buf[n - 1][col] + buf[n + 1][col]) * fB;
        }
      }
      for (int32_t n = 0 + offset, i = start; i < stop; i++, n += 2) {
        idwt_irrev97_wasm_ver_step(simdlen_s, buf[n] + cs, buf[n + 2] + cs, buf[n + 1] + cs, fA);
        for (int32_t col = cs + simdlen_s; col < ce; ++col) {
          buf[n + 1][col] -= (buf[n][col] + buf[n + 2][col]) * fA;
        }
      }
    }
  }
}

// reversible IDWT 5/3 vertical
void idwt_rev_ver_sr_fixed_wasm(sprec_t *in, const int32_t u0, const int32_t u1, const int32_t v0,
                                const int32_t v1, const int32_t stride, sprec_t *pse_scratch,
                                sprec_t **buf_scratch) {
  constexpr int32_t num_pse_i0[2] = {1, 2};
  constexpr int32_t num_pse_i1[2] = {2, 1};
  const int32_t top               = num_pse_i0[v0 % 2];
  const int32_t bottom            = num_pse_i1[v1 % 2];
  if (v0 == v1 - 1) {
    for (int32_t col = 0; col < u1 - u0; ++col) {
      in[col] = (v0 % 2 == 0) ? in[col] : floorf(in[col] * 0.5f);
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
    const int32_t start  = v0 / 2;
    const int32_t stop   = v1 / 2;
    const int32_t offset = top - v0 % 2;

    const int32_t width = u1 - u0;
    for (int32_t cs = 0; cs < width; cs += DWT_VERT_STRIP) {
      const int32_t ce        = (cs + DWT_VERT_STRIP < width) ? cs + DWT_VERT_STRIP : width;
      const int32_t simdlen_s = (ce - cs) - (ce - cs) % 4;
      const v128_t vtwo       = wasm_f32x4_splat(2.0f);
      // LP step: HP[i] -= floor((LP[i-1] + LP[i+1] + 2) * 0.25)
      for (int32_t n = 0 + offset, i = start; i < stop + 1; ++i, n += 2) {
        int32_t col = 0;
        for (; col + 4 < simdlen_s; col += 8) {
          v128_t vin0a = wasm_v128_load(buf[n - 1] + cs + col);
          v128_t vouta = wasm_v128_load(buf[n] + cs + col);
          v128_t vin1a = wasm_v128_load(buf[n + 1] + cs + col);
          v128_t vin0b = wasm_v128_load(buf[n - 1] + cs + col + 4);
          v128_t voutb = wasm_v128_load(buf[n] + cs + col + 4);
          v128_t vin1b = wasm_v128_load(buf[n + 1] + cs + col + 4);
          vouta = wasm_f32x4_sub(vouta, wasm_f32x4_floor(wasm_f32x4_mul(
                    wasm_f32x4_add(wasm_f32x4_add(vin0a, vin1a), vtwo), wasm_f32x4_splat(0.25f))));
          voutb = wasm_f32x4_sub(voutb, wasm_f32x4_floor(wasm_f32x4_mul(
                    wasm_f32x4_add(wasm_f32x4_add(vin0b, vin1b), vtwo), wasm_f32x4_splat(0.25f))));
          wasm_v128_store(buf[n] + cs + col,     vouta);
          wasm_v128_store(buf[n] + cs + col + 4, voutb);
        }
        for (; col < simdlen_s; col += 4) {
          v128_t vin0 = wasm_v128_load(buf[n - 1] + cs + col);
          v128_t vout = wasm_v128_load(buf[n] + cs + col);
          v128_t vin1 = wasm_v128_load(buf[n + 1] + cs + col);
          vout = wasm_f32x4_sub(vout, wasm_f32x4_floor(wasm_f32x4_mul(
                   wasm_f32x4_add(wasm_f32x4_add(vin0, vin1), vtwo), wasm_f32x4_splat(0.25f))));
          wasm_v128_store(buf[n] + cs + col, vout);
        }
        for (int32_t col2 = cs + simdlen_s; col2 < ce; ++col2) {
          buf[n][col2] -= floorf((buf[n - 1][col2] + buf[n + 1][col2] + 2.0f) * 0.25f);
        }
      }
      // HP step: LP[i] += floor((HP[i] + HP[i+2]) * 0.5)
      for (int32_t n = 0 + offset, i = start; i < stop; ++i, n += 2) {
        int32_t col = 0;
        for (; col + 4 < simdlen_s; col += 8) {
          v128_t vin0a = wasm_v128_load(buf[n] + cs + col);
          v128_t vouta = wasm_v128_load(buf[n + 1] + cs + col);
          v128_t vin1a = wasm_v128_load(buf[n + 2] + cs + col);
          v128_t vin0b = wasm_v128_load(buf[n] + cs + col + 4);
          v128_t voutb = wasm_v128_load(buf[n + 1] + cs + col + 4);
          v128_t vin1b = wasm_v128_load(buf[n + 2] + cs + col + 4);
          vouta = wasm_f32x4_add(vouta, wasm_f32x4_floor(wasm_f32x4_mul(
                    wasm_f32x4_add(vin0a, vin1a), wasm_f32x4_splat(0.5f))));
          voutb = wasm_f32x4_add(voutb, wasm_f32x4_floor(wasm_f32x4_mul(
                    wasm_f32x4_add(vin0b, vin1b), wasm_f32x4_splat(0.5f))));
          wasm_v128_store(buf[n + 1] + cs + col,     vouta);
          wasm_v128_store(buf[n + 1] + cs + col + 4, voutb);
        }
        for (; col < simdlen_s; col += 4) {
          v128_t vin0 = wasm_v128_load(buf[n] + cs + col);
          v128_t vout = wasm_v128_load(buf[n + 1] + cs + col);
          v128_t vin1 = wasm_v128_load(buf[n + 2] + cs + col);
          vout = wasm_f32x4_add(vout, wasm_f32x4_floor(wasm_f32x4_mul(
                   wasm_f32x4_add(vin0, vin1), wasm_f32x4_splat(0.5f))));
          wasm_v128_store(buf[n + 1] + cs + col, vout);
        }
        for (int32_t col2 = cs + simdlen_s; col2 < ce; ++col2) {
          buf[n + 1][col2] += floorf((buf[n][col2] + buf[n + 2][col2]) * 0.5f);
        }
      }
    }
  }
}

#endif  // OPENHTJ2K_ENABLE_WASM_SIMD
