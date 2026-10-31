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

// FDWT kernels for x86-64 builds without AVX2 (OPENHTJ2K_ENABLE_SSE2, utils.hpp): the
// 4-lane WASM SIMD kernels (fdwt_wasm.cpp) in SSE2.  As there, every multiply and add is
// rounded separately (no FMA), in the order of the scalar kernels, so the output matches
// the scalar build bit for bit.  SSE2 has no floor: floor_ps() truncates, converts back
// and steps down where that went up (exact for the magnitudes a DWT sees, below 2^31).

#include "utils.hpp"  // OPENHTJ2K_ENABLE_SSE2
#if defined(OPENHTJ2K_ENABLE_SSE2)
  #include <emmintrin.h>
  #include <cmath>
  #include <cstring>
  #include "dwt.hpp"

namespace {
struct f32x4x2 {
  __m128 val[2];
};
// [a0,b0,a1,b1,a2,b2,a3,b3] -> [a0..a3], [b0..b3]
inline f32x4x2 vld2q(const float *ptr) {
  const __m128 lo = _mm_loadu_ps(ptr);
  const __m128 hi = _mm_loadu_ps(ptr + 4);
  f32x4x2 r;
  r.val[0] = _mm_shuffle_ps(lo, hi, _MM_SHUFFLE(2, 0, 2, 0));
  r.val[1] = _mm_shuffle_ps(lo, hi, _MM_SHUFFLE(3, 1, 3, 1));
  return r;
}
inline void vst2q(float *ptr, const f32x4x2 &x) {
  _mm_storeu_ps(ptr, _mm_unpacklo_ps(x.val[0], x.val[1]));
  _mm_storeu_ps(ptr + 4, _mm_unpackhi_ps(x.val[0], x.val[1]));
}
inline __m128 floor_ps(__m128 x) {
  const __m128 t = _mm_cvtepi32_ps(_mm_cvttps_epi32(x));
  return _mm_sub_ps(t, _mm_and_ps(_mm_cmpgt_ps(t, x), _mm_set1_ps(1.0f)));
}
// [a1, a2, a3, b0] and [a3, b0, b1, b2]
inline __m128 next1(__m128 a, __m128 b) {
  const __m128 m = _mm_move_ss(a, b);  // [b0, a1, a2, a3]
  return _mm_shuffle_ps(m, m, _MM_SHUFFLE(0, 3, 2, 1));
}
inline __m128 prev1(__m128 a, __m128 b) {
  const __m128 t = _mm_shuffle_ps(a, b, _MM_SHUFFLE(0, 0, 3, 3));  // [a3, a3, b0, b0]
  return _mm_shuffle_ps(t, b, _MM_SHUFFLE(2, 1, 2, 0));
}
inline __m128i next1(__m128i a, __m128i b) {
  return _mm_castps_si128(next1(_mm_castsi128_ps(a), _mm_castsi128_ps(b)));
}
inline __m128i prev1(__m128i a, __m128i b) {
  return _mm_castps_si128(prev1(_mm_castsi128_ps(a), _mm_castsi128_ps(b)));
}

struct i32x4x2 {
  __m128i val[2];
};
inline i32x4x2 vld2q_i32(const int32_t *ptr) {
  const f32x4x2 f = vld2q(reinterpret_cast<const float *>(ptr));
  i32x4x2 r;
  r.val[0] = _mm_castps_si128(f.val[0]);
  r.val[1] = _mm_castps_si128(f.val[1]);
  return r;
}
inline void vst2q_i32(int32_t *ptr, const i32x4x2 &x) {
  _mm_storeu_si128(reinterpret_cast<__m128i *>(ptr), _mm_unpacklo_epi32(x.val[0], x.val[1]));
  _mm_storeu_si128(reinterpret_cast<__m128i *>(ptr + 4), _mm_unpackhi_epi32(x.val[0], x.val[1]));
}
inline __m128i ld(const int32_t *p) { return _mm_loadu_si128(reinterpret_cast<const __m128i *>(p)); }
inline void st(int32_t *p, __m128i v) { _mm_storeu_si128(reinterpret_cast<__m128i *>(p), v); }

/********************************************************************************
 * horizontal transforms
 *******************************************************************************/
// irreversible FDWT: x0.HP += coeff * (x0.LP + x1.LP)
void fdwt_irrev97_hor_step(const int32_t init_pos, const int32_t simdlen, float *X, const int32_t n0,
                           const int32_t n1, float coeff) {
  const __m128 vvv = _mm_set1_ps(coeff);
  X += init_pos;
  int32_t i = simdlen;
  for (; i > 4; i -= 8) {
    auto x0a   = vld2q(X + n0);
    auto x1a   = vld2q(X + n1);
    auto x0b   = vld2q(X + 8 + n0);
    auto x1b   = vld2q(X + 8 + n1);
    x0a.val[1] = _mm_add_ps(x0a.val[1], _mm_mul_ps(_mm_add_ps(x0a.val[0], x1a.val[0]), vvv));
    x0b.val[1] = _mm_add_ps(x0b.val[1], _mm_mul_ps(_mm_add_ps(x0b.val[0], x1b.val[0]), vvv));
    vst2q(X + n0, x0a);
    vst2q(X + 8 + n0, x0b);
    X += 16;
  }
  for (; i > 0; i -= 4) {
    auto x0   = vld2q(X + n0);
    auto x1   = vld2q(X + n1);
    x0.val[1] = _mm_add_ps(x0.val[1], _mm_mul_ps(_mm_add_ps(x0.val[0], x1.val[0]), vvv));
    vst2q(X + n0, x0);
    X += 8;
  }
}

// irreversible FDWT vertical step: Xout[i] += coeff * (Xin0[i] + Xin1[i])
void fdwt_irrev97_ver_step(const int32_t simdlen, float *const Xin0, float *const Xin1, float *const Xout,
                           float coeff) {
  const __m128 vvv = _mm_set1_ps(coeff);
  int32_t n        = 0;
  for (; n + 4 < simdlen; n += 8) {
    __m128 x0a = _mm_loadu_ps(Xin0 + n), x2a = _mm_loadu_ps(Xin1 + n), x1a = _mm_loadu_ps(Xout + n);
    __m128 x0b = _mm_loadu_ps(Xin0 + n + 4), x2b = _mm_loadu_ps(Xin1 + n + 4);
    __m128 x1b = _mm_loadu_ps(Xout + n + 4);
    x1a        = _mm_add_ps(x1a, _mm_mul_ps(vvv, _mm_add_ps(x0a, x2a)));
    x1b        = _mm_add_ps(x1b, _mm_mul_ps(vvv, _mm_add_ps(x0b, x2b)));
    _mm_storeu_ps(Xout + n, x1a);
    _mm_storeu_ps(Xout + n + 4, x1b);
  }
  for (; n < simdlen; n += 4) {
    const __m128 x0 = _mm_loadu_ps(Xin0 + n);
    const __m128 x2 = _mm_loadu_ps(Xin1 + n);
    __m128 x1       = _mm_loadu_ps(Xout + n);
    x1              = _mm_add_ps(x1, _mm_mul_ps(vvv, _mm_add_ps(x0, x2)));
    _mm_storeu_ps(Xout + n, x1);
  }
}
}  // namespace

void fdwt_1d_filtr_irrev97_fixed_sse2(sprec_t *X, const int32_t left, const int32_t u_i0,
                                      const int32_t u_i1) {
  const auto i0        = static_cast<int32_t>(u_i0);
  const auto i1        = static_cast<int32_t>(u_i1);
  const int32_t start  = ceil_int(i0, 2);
  const int32_t stop   = ceil_int(i1, 2);
  const int32_t offset = left + i0 % 2;

  int32_t simdlen = stop + 1 - (start - 2);
  fdwt_irrev97_hor_step(offset - 4, simdlen, X, 0, 2, fA);
  simdlen = stop + 1 - (start - 1);
  fdwt_irrev97_hor_step(offset - 2, simdlen, X, -1, 1, fB);
  simdlen = stop - (start - 1);
  fdwt_irrev97_hor_step(offset - 2, simdlen, X, 0, 2, fC);
  simdlen = stop - start;
  fdwt_irrev97_hor_step(offset, simdlen, X, -1, 1, fD);
}

// reversible FDWT 5/3
void fdwt_1d_filtr_rev53_fixed_sse2(sprec_t *X, const int32_t left, const int32_t u_i0,
                                    const int32_t u_i1) {
  const auto i0        = static_cast<int32_t>(u_i0);
  const auto i1        = static_cast<int32_t>(u_i1);
  const int32_t start  = ceil_int(i0, 2);
  const int32_t stop   = ceil_int(i1, 2);
  const int32_t offset = left + i0 % 2;

  // step 1: HP[i] -= floor((LP[i] + LP[i+1]) * 0.5)
  int32_t simdlen    = stop - (start - 1);
  const __m128 vhalf = _mm_set1_ps(0.5f);
  int32_t n = -2 + offset, i = 0;
  for (; i + 4 < simdlen; i += 8, n += 16) {
    auto xl0a = vld2q(X + n);
    auto xl1a = vld2q(X + n + 2);
    auto xl0b = vld2q(X + n + 8);
    auto xl1b = vld2q(X + n + 10);
    xl0a.val[1] =
        _mm_sub_ps(xl0a.val[1], floor_ps(_mm_mul_ps(_mm_add_ps(xl0a.val[0], xl1a.val[0]), vhalf)));
    xl0b.val[1] =
        _mm_sub_ps(xl0b.val[1], floor_ps(_mm_mul_ps(_mm_add_ps(xl0b.val[0], xl1b.val[0]), vhalf)));
    vst2q(X + n, xl0a);
    vst2q(X + n + 8, xl0b);
  }
  for (; i < simdlen; i += 4, n += 8) {
    auto xl0   = vld2q(X + n);
    auto xl1   = vld2q(X + n + 2);
    xl0.val[1] = _mm_sub_ps(xl0.val[1], floor_ps(_mm_mul_ps(_mm_add_ps(xl0.val[0], xl1.val[0]), vhalf)));
    vst2q(X + n, xl0);
  }

  // step 2: LP[i] += floor((HP[i-1] + HP[i] + 2) * 0.25)
  simdlen           = stop - start;
  const __m128 vqrt = _mm_set1_ps(0.25f);
  const __m128 vtwo = _mm_set1_ps(2.0f);
  n                 = 0 + offset;
  i                 = 0;
  for (; i + 4 < simdlen; i += 8, n += 16) {
    auto xl0a   = vld2q(X + n - 1);
    auto xl1a   = vld2q(X + n + 1);
    auto xl0b   = vld2q(X + n + 7);
    auto xl1b   = vld2q(X + n + 9);
    xl0a.val[1] = _mm_add_ps(
        xl0a.val[1], floor_ps(_mm_mul_ps(_mm_add_ps(_mm_add_ps(xl0a.val[0], xl1a.val[0]), vtwo), vqrt)));
    xl0b.val[1] = _mm_add_ps(
        xl0b.val[1], floor_ps(_mm_mul_ps(_mm_add_ps(_mm_add_ps(xl0b.val[0], xl1b.val[0]), vtwo), vqrt)));
    vst2q(X + n - 1, xl0a);
    vst2q(X + n + 7, xl0b);
  }
  for (; i < simdlen; i += 4, n += 8) {
    auto xl0   = vld2q(X + n - 1);
    auto xl1   = vld2q(X + n + 1);
    xl0.val[1] = _mm_add_ps(
        xl0.val[1], floor_ps(_mm_mul_ps(_mm_add_ps(_mm_add_ps(xl0.val[0], xl1.val[0]), vtwo), vqrt)));
    vst2q(X + n - 1, xl0);
  }
}

/********************************************************************************
 * vertical transforms
 *******************************************************************************/
void fdwt_irrev_ver_sr_fixed_sse2(sprec_t *in, const int32_t u0, const int32_t u1, const int32_t v0,
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
        fdwt_irrev97_ver_step(simdlen_s, buf[n] + cs, buf[n + 2] + cs, buf[n + 1] + cs, fA);
        for (int32_t col = cs + simdlen_s; col < ce; ++col) {
          buf[n + 1][col] += fA * (buf[n][col] + buf[n + 2][col]);
        }
      }
      for (int32_t n = -2 + offset, i = start - 1; i < stop + 1; i++, n += 2) {
        fdwt_irrev97_ver_step(simdlen_s, buf[n - 1] + cs, buf[n + 1] + cs, buf[n] + cs, fB);
        for (int32_t col = cs + simdlen_s; col < ce; ++col) {
          buf[n][col] += fB * (buf[n - 1][col] + buf[n + 1][col]);
        }
      }
      for (int32_t n = -2 + offset, i = start - 1; i < stop; i++, n += 2) {
        fdwt_irrev97_ver_step(simdlen_s, buf[n] + cs, buf[n + 2] + cs, buf[n + 1] + cs, fC);
        for (int32_t col = cs + simdlen_s; col < ce; ++col) {
          buf[n + 1][col] += fC * (buf[n][col] + buf[n + 2][col]);
        }
      }
      for (int32_t n = 0 + offset, i = start; i < stop; i++, n += 2) {
        fdwt_irrev97_ver_step(simdlen_s, buf[n - 1] + cs, buf[n + 1] + cs, buf[n] + cs, fD);
        for (int32_t col = cs + simdlen_s; col < ce; ++col) {
          buf[n][col] += fD * (buf[n - 1][col] + buf[n + 1][col]);
        }
      }
    }
  }
}

// reversible FDWT 5/3 vertical
void fdwt_rev_ver_sr_fixed_sse2(sprec_t *in, const int32_t u0, const int32_t u1, const int32_t v0,
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
        fdwt_rev_ver_hp_step_sse2(simdlen_s, buf[n] + cs, buf[n + 2] + cs, buf[n + 1] + cs);
        for (int32_t col2 = cs + simdlen_s; col2 < ce; ++col2) {
          buf[n + 1][col2] -= floorf((buf[n][col2] + buf[n + 2][col2]) * 0.5f);
        }
      }
      // HP step: LP[i] += floor((HP[i-1] + HP[i] + 2) * 0.25)
      for (int32_t n = 0 + offset, i = start; i < stop; ++i, n += 2) {
        fdwt_rev_ver_lp_step_sse2(simdlen_s, buf[n - 1] + cs, buf[n + 1] + cs, buf[n] + cs);
        for (int32_t col2 = cs + simdlen_s; col2 < ce; ++col2) {
          buf[n][col2] += floorf((buf[n - 1][col2] + buf[n + 1][col2] + 2.0f) * 0.25f);
        }
      }
    }
  }
}

// Single-row 9/7 lifting step: tgt[i] -= coeff * (prev[i] + next[i]).
void fdwt_irrev_ver_step_sse2(int32_t n, float *prev, float *next, float *tgt, float coeff) {
  const __m128 vvv = _mm_set1_ps(coeff);
  int32_t i        = 0;
  for (; i + 4 < n; i += 8) {
    __m128 x0a = _mm_loadu_ps(prev + i), x2a = _mm_loadu_ps(next + i), x1a = _mm_loadu_ps(tgt + i);
    __m128 x0b = _mm_loadu_ps(prev + i + 4), x2b = _mm_loadu_ps(next + i + 4);
    __m128 x1b = _mm_loadu_ps(tgt + i + 4);
    x1a        = _mm_sub_ps(x1a, _mm_mul_ps(_mm_add_ps(x0a, x2a), vvv));
    x1b        = _mm_sub_ps(x1b, _mm_mul_ps(_mm_add_ps(x0b, x2b), vvv));
    _mm_storeu_ps(tgt + i, x1a);
    _mm_storeu_ps(tgt + i + 4, x1b);
  }
  for (; i + 4 <= n; i += 4) {
    const __m128 x0 = _mm_loadu_ps(prev + i);
    const __m128 x2 = _mm_loadu_ps(next + i);
    __m128 x1       = _mm_loadu_ps(tgt + i);
    x1              = _mm_sub_ps(x1, _mm_mul_ps(_mm_add_ps(x0, x2), vvv));
    _mm_storeu_ps(tgt + i, x1);
  }
  for (; i < n; ++i) tgt[i] -= coeff * (prev[i] + next[i]);
}

// Single-row rev53 FDWT HP vertical lifting: tgt[i] -= floor((prev[i]+next[i])*0.5).
void fdwt_rev_ver_hp_step_sse2(int32_t n, const float *prev, const float *next, float *tgt) {
  const __m128 k05 = _mm_set1_ps(0.5f);
  int32_t i        = 0;
  for (; i + 4 <= n; i += 4) {
    const __m128 a = _mm_loadu_ps(prev + i);
    const __m128 b = _mm_loadu_ps(next + i);
    _mm_storeu_ps(tgt + i, _mm_sub_ps(_mm_loadu_ps(tgt + i), floor_ps(_mm_mul_ps(_mm_add_ps(a, b), k05))));
  }
  for (; i < n; ++i) tgt[i] -= floorf((prev[i] + next[i]) * 0.5f);
}

// Single-row rev53 FDWT LP vertical lifting: tgt[i] += floor((prev[i]+next[i]+2)*0.25).
void fdwt_rev_ver_lp_step_sse2(int32_t n, const float *prev, const float *next, float *tgt) {
  const __m128 k025 = _mm_set1_ps(0.25f);
  const __m128 k2   = _mm_set1_ps(2.0f);
  int32_t i         = 0;
  for (; i + 4 <= n; i += 4) {
    const __m128 a = _mm_loadu_ps(prev + i);
    const __m128 b = _mm_loadu_ps(next + i);
    _mm_storeu_ps(tgt + i, _mm_add_ps(_mm_loadu_ps(tgt + i),
                                      floor_ps(_mm_mul_ps(_mm_add_ps(_mm_add_ps(a, b), k2), k025))));
  }
  for (; i < n; ++i) tgt[i] += floorf((prev[i] + next[i] + 2.0f) * 0.25f);
}

/********************************************************************************
 * int32 5/3 reversible DWT (lossless path): arithmetic shifts, exact
 *******************************************************************************/
void fdwt_1d_filtr_rev53_i32_sse2(int32_t *X, const int32_t left, const int32_t u_i0, const int32_t u_i1) {
  const int32_t i0     = u_i0;
  const int32_t i1     = u_i1;
  const int32_t start  = ceil_int(i0, 2);
  const int32_t stop   = ceil_int(i1, 2);
  const int32_t offset = left + i0 % 2;

  // step 1: H[k] -= (L[k] + L[k+1]) >> 1
  int32_t simdlen = stop - (start - 1);
  int32_t n = -2 + offset, i = 0;
  for (; i + 4 < simdlen; i += 8, n += 16) {
    i32x4x2 xl0a = vld2q_i32(X + n);
    i32x4x2 xl1a = vld2q_i32(X + n + 2);
    i32x4x2 xl0b = vld2q_i32(X + n + 8);
    i32x4x2 xl1b = vld2q_i32(X + n + 10);
    xl0a.val[1]  = _mm_sub_epi32(xl0a.val[1], _mm_srai_epi32(_mm_add_epi32(xl0a.val[0], xl1a.val[0]), 1));
    xl0b.val[1]  = _mm_sub_epi32(xl0b.val[1], _mm_srai_epi32(_mm_add_epi32(xl0b.val[0], xl1b.val[0]), 1));
    vst2q_i32(X + n, xl0a);
    vst2q_i32(X + n + 8, xl0b);
  }
  for (; i < simdlen; i += 4, n += 8) {
    i32x4x2 xl0 = vld2q_i32(X + n);
    i32x4x2 xl1 = vld2q_i32(X + n + 2);
    xl0.val[1]  = _mm_sub_epi32(xl0.val[1], _mm_srai_epi32(_mm_add_epi32(xl0.val[0], xl1.val[0]), 1));
    vst2q_i32(X + n, xl0);
  }

  // step 2: L[k] += (H[k-1] + H[k] + 2) >> 2
  simdlen            = stop - start;
  const __m128i vtwo = _mm_set1_epi32(2);
  n                  = 0 + offset;
  i                  = 0;
  for (; i + 4 < simdlen; i += 8, n += 16) {
    i32x4x2 xl0a = vld2q_i32(X + n - 1);
    i32x4x2 xl1a = vld2q_i32(X + n + 1);
    i32x4x2 xl0b = vld2q_i32(X + n + 7);
    i32x4x2 xl1b = vld2q_i32(X + n + 9);
    xl0a.val[1]  = _mm_add_epi32(
        xl0a.val[1], _mm_srai_epi32(_mm_add_epi32(_mm_add_epi32(xl0a.val[0], xl1a.val[0]), vtwo), 2));
    xl0b.val[1] = _mm_add_epi32(
        xl0b.val[1], _mm_srai_epi32(_mm_add_epi32(_mm_add_epi32(xl0b.val[0], xl1b.val[0]), vtwo), 2));
    vst2q_i32(X + n - 1, xl0a);
    vst2q_i32(X + n + 7, xl0b);
  }
  for (; i < simdlen; i += 4, n += 8) {
    i32x4x2 xl0 = vld2q_i32(X + n - 1);
    i32x4x2 xl1 = vld2q_i32(X + n + 1);
    xl0.val[1]  = _mm_add_epi32(
        xl0.val[1], _mm_srai_epi32(_mm_add_epi32(_mm_add_epi32(xl0.val[0], xl1.val[0]), vtwo), 2));
    vst2q_i32(X + n - 1, xl0);
  }
}

void fdwt_rev_ver_hp_step_i32_sse2(int32_t n, const int32_t *prev, const int32_t *next, int32_t *tgt) {
  int32_t i = 0;
  for (; i + 4 <= n; i += 4)
    st(tgt + i, _mm_sub_epi32(ld(tgt + i), _mm_srai_epi32(_mm_add_epi32(ld(prev + i), ld(next + i)), 1)));
  for (; i < n; ++i) tgt[i] -= (prev[i] + next[i]) >> 1;
}

void fdwt_rev_ver_lp_step_i32_sse2(int32_t n, const int32_t *prev, const int32_t *next, int32_t *tgt) {
  const __m128i vtwo = _mm_set1_epi32(2);
  int32_t i          = 0;
  for (; i + 4 <= n; i += 4)
    st(tgt + i,
       _mm_add_epi32(ld(tgt + i),
                     _mm_srai_epi32(_mm_add_epi32(_mm_add_epi32(ld(prev + i), ld(next + i)), vtwo), 2)));
  for (; i < n; ++i) tgt[i] += (prev[i] + next[i] + 2) >> 2;
}

/********************************************************************************
 * Planar horizontal FDWT: the ring row is read deinterleaved (E[j] = in[2j],
 * O[j] = in[2j+1]) and the LP/HP planes are written directly.  Same recurrences
 * and rounding as fdwt_1d_filtr_irrev97_planar_wasm / _rev53_planar_i32_wasm.
 *******************************************************************************/
//   T1[j] = O[j]  + fA*(E[j]    + E[j+1])   j in [-2, NL]
//   T2[j] = E[j]  + fB*(T1[j-1] + T1[j])    j in [-1, NL]
//   T3[j] = T1[j] + fC*(T2[j]   + T2[j+1])  j in [-1, NL-1]
//   T4[j] = T2[j] + fD*(T3[j-1] + T3[j])    j in [ 0, NL-1]
//   hp[j] = T3[j] (j < NH),  lp[j] = T4[j] (j < NL)
void fdwt_1d_filtr_irrev97_planar_sse2(sprec_t *lp, sprec_t *hp, const sprec_t *in, const int32_t u0,
                                       const int32_t u1) {
  const int32_t NH = u1 / 2 - u0 / 2;
  const int32_t NL = ceil_int(u1, 2) - u0 / 2;
  auto E           = [&](int32_t j) -> float { return in[PSEo(u0 + 2 * j, u0, u1)]; };
  auto O           = [&](int32_t j) -> float { return in[PSEo(u0 + 2 * j + 1, u0, u1)]; };

  const __m128 vA = _mm_set1_ps(fA), vB = _mm_set1_ps(fB);
  const __m128 vC = _mm_set1_ps(fC), vD = _mm_set1_ps(fD);

  // Warmup: boundary scalars, blocks 0/1 loads, T1/T2 of block 0.
  const float t1m2 = O(-2) + fA * (E(-2) + E(-1));  // T1[-2]
  const float t1m1 = O(-1) + fA * (E(-1) + E(0));   // T1[-1]
  f32x4x2 b0       = vld2q(in);                     // E/O block 0
  f32x4x2 b1       = vld2q(in + 8);                 // E/O block 1
  __m128 T1_0 = _mm_add_ps(b0.val[1], _mm_mul_ps(_mm_add_ps(b0.val[0], next1(b0.val[0], b1.val[0])), vA));
  __m128 T2_0 = _mm_add_ps(b0.val[0], _mm_mul_ps(_mm_add_ps(prev1(_mm_set1_ps(t1m1), T1_0), T1_0), vB));
  const float t2m1 = E(-1) + fB * (t1m2 + t1m1);                // T2[-1]
  const float t3m1 = t1m1 + fC * (t2m1 + _mm_cvtss_f32(T2_0));  // T3[-1]

  // Steady state: iteration n loads input block n and stores finished plane block n-2.
  __m128 E_nm1 = b1.val[0], O_nm1 = b1.val[1], T1_nm2 = T1_0, T2_nm2 = T2_0;
  __m128 T3_nm3 = _mm_set1_ps(t3m1);  // only its top lane is used
  int32_t n     = 2;
  for (; 4 * n + 3 <= NH - 1; ++n) {
    f32x4x2 bn    = vld2q(in + 8 * n);
    __m128 T1_nm1 = _mm_add_ps(O_nm1, _mm_mul_ps(_mm_add_ps(E_nm1, next1(E_nm1, bn.val[0])), vA));
    __m128 T2_nm1 = _mm_add_ps(E_nm1, _mm_mul_ps(_mm_add_ps(prev1(T1_nm2, T1_nm1), T1_nm1), vB));
    __m128 T3_nm2 = _mm_add_ps(T1_nm2, _mm_mul_ps(_mm_add_ps(T2_nm2, next1(T2_nm2, T2_nm1)), vC));
    __m128 T4_nm2 = _mm_add_ps(T2_nm2, _mm_mul_ps(_mm_add_ps(prev1(T3_nm3, T3_nm2), T3_nm2), vD));
    _mm_storeu_ps(hp + 4 * (n - 2), T3_nm2);
    _mm_storeu_ps(lp + 4 * (n - 2), T4_nm2);
    E_nm1  = bn.val[0];
    O_nm1  = bn.val[1];
    T1_nm2 = T1_nm1;
    T2_nm2 = T2_nm1;
    T3_nm3 = T3_nm2;
  }

  // Drain: scalar finish from the carried registers.  P = 4*(n-2) is the first plane
  // index not yet stored; NL - P <= 12, so with base = P - 4 every index fits in 24.
  {
    const int32_t P    = 4 * (n - 2);
    const int32_t base = P - 4;
    float t1[24], t2[24], t3[24];
    _mm_storeu_ps(t1 + (P - base), T1_nm2);  // T1[P..P+3]
    _mm_storeu_ps(t2 + (P - base), T2_nm2);  // T2[P..P+3]
    _mm_storeu_ps(t3, T3_nm3);               // T3[P-4..P-1] (top lane = T3[P-1])
    for (int32_t j = P + 4; j <= NL; ++j) t1[j - base] = O(j) + fA * (E(j) + E(j + 1));
    for (int32_t j = P + 4; j <= NL; ++j) t2[j - base] = E(j) + fB * (t1[j - base - 1] + t1[j - base]);
    for (int32_t j = P; j <= NL - 1; ++j)
      t3[j - base] = t1[j - base] + fC * (t2[j - base] + t2[j - base + 1]);
    for (int32_t j = P; j <= NL - 1; ++j) lp[j] = t2[j - base] + fD * (t3[j - base - 1] + t3[j - base]);
    for (int32_t j = P; j <= NH - 1; ++j) hp[j] = t3[j - base];
  }
}

//   T1[j] = O[j] - ((E[j]    + E[j+1]) >> 1)      j in [-1, NL-1]
//   T2[j] = E[j] + ((T1[j-1] + T1[j] + 2) >> 2)   j in [ 0, NL-1]
//   hp[j] = T1[j] (j < NH),  lp[j] = T2[j] (j < NL)
void fdwt_1d_filtr_rev53_planar_i32_sse2(int32_t *lp, int32_t *hp, const int32_t *in, const int32_t u0,
                                         const int32_t u1) {
  const int32_t NH = u1 / 2 - u0 / 2;
  const int32_t NL = ceil_int(u1, 2) - u0 / 2;
  auto E           = [&](int32_t j) -> int32_t { return in[PSEo(u0 + 2 * j, u0, u1)]; };
  auto O           = [&](int32_t j) -> int32_t { return in[PSEo(u0 + 2 * j + 1, u0, u1)]; };

  const __m128i vtwo = _mm_set1_epi32(2);

  // Warmup: boundary scalar T1[-1], block 0 load.
  const int32_t t1m1 = O(-1) - ((E(-1) + E(0)) >> 1);
  i32x4x2 b0         = vld2q_i32(in);  // E/O block 0
  __m128i E_nm1 = b0.val[0], O_nm1 = b0.val[1];
  __m128i T1_nm2 = _mm_set1_epi32(t1m1);  // only its top lane is used

  // Steady state: iteration n loads input block n and stores finished plane block n-1.
  int32_t n = 1;
  for (; 4 * n + 3 <= NH - 1; ++n) {
    i32x4x2 bn     = vld2q_i32(in + 8 * n);
    __m128i T1_nm1 = _mm_sub_epi32(O_nm1, _mm_srai_epi32(_mm_add_epi32(E_nm1, next1(E_nm1, bn.val[0])), 1));
    __m128i T2_nm1 = _mm_add_epi32(
        E_nm1, _mm_srai_epi32(_mm_add_epi32(_mm_add_epi32(prev1(T1_nm2, T1_nm1), T1_nm1), vtwo), 2));
    st(hp + 4 * (n - 1), T1_nm1);
    st(lp + 4 * (n - 1), T2_nm1);
    E_nm1  = bn.val[0];
    O_nm1  = bn.val[1];
    T1_nm2 = T1_nm1;
  }

  // Drain: scalar finish.  P = 4*(n-1) is the first plane index not yet stored;
  // NH - P <= 7, so NL - P <= 8.
  {
    const int32_t P = 4 * (n - 1);
    const int32_t t1_prev =
        _mm_cvtsi128_si32(_mm_shuffle_epi32(T1_nm2, _MM_SHUFFLE(3, 3, 3, 3)));  // T1[P-1]
    int32_t t1buf[8];
    for (int32_t j = P; j <= NL - 1; ++j) t1buf[j - P] = O(j) - ((E(j) + E(j + 1)) >> 1);
    for (int32_t j = P; j <= NH - 1; ++j) hp[j] = t1buf[j - P];
    for (int32_t j = P; j <= NL - 1; ++j) {
      const int32_t t1l = (j == P) ? t1_prev : t1buf[j - P - 1];
      lp[j]             = E(j) + ((t1l + t1buf[j - P] + 2) >> 2);
    }
  }
}

#endif  // OPENHTJ2K_ENABLE_SSE2
