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

// Rate control for HT code-blocks (EXPERIMENTAL).
//
// An HT code-block is coded as one HT set: the cleanup pass at a bit-plane p and the
// SigProp and MagRef passes of plane p - 1, which give three truncation points.  The plane
// has to be chosen while the code-block's samples are at hand, before the slope threshold
// of the whole image is known, so:
//
// 1. ht_rd_stats() measures each code-block over the planes that can matter: how many
//    samples and quads are significant at each plane, the exact squared error after a
//    cleanup pass there, and which samples the SigProp pass below it would code and make
//    significant.  A linear model of the cleanup segment length in those counts gives every
//    plane's rate; the SigProp and MagRef passes write raw bits, so their lengths follow
//    from the counts.
// 2. ht_rc_state folds the measured code-blocks' curves into a threshold estimate for the
//    whole image: each subband class's measured code-blocks stand for its others, and a
//    class not reached yet borrows the next finer level's curve.
// 3. ht_rd_code() codes the set that holds the code-block's best point at the estimate,
//    and keeps its three points' exact lengths and their distortions.  A code-block left out even at a
//    quarter of the estimate is not coded at all.
// 4. Once every code-block is coded, ht_rate_control() picks every code-block's point at
//    one threshold over all of them (PCRD optimisation, with the packet headers measured).
//    A point above the coded set is reached by transcoding: the cleanup pass at p holds
//    every magnitude bit at or above p, so decoding it and coding again from a higher plane
//    gives exactly the set the encoder would have coded there.  The choice is made once
//    more with the transcoded sets' exact lengths.
//
// Distortion is the squared error of the reconstructed image: a code-block's error in
// steps of its base step, weighted by the step size squared, by the energy of its subband's
// synthesis basis and, with the irreversible colour transform, by that of the inverse
// transform's column.  The decoder puts a coefficient at the centre of its last coded bin,
// so a magnitude m truncated at plane p comes back as (floor(m / 2^p) + 1/2) 2^p for
// floor(m / 2^p) > 0; the true value is taken as m + 1/2 (the centre of its bin at the base
// step).  The measurements keep errors doubled, to stay in integers.

#include "ht_rate_control.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <tuple>
#include <vector>
#ifdef OPENHTJ2K_THREAD
  #include <thread>
  #include "ThreadPool.hpp"
#endif

#include "block_decoding.hpp"
#include "coding_units.hpp"
#include "ht_sigprop.hpp"
#include "utils.hpp"

#if defined(OPENHTJ2K_TRY_AVX2) && (defined(__AVX512F__) || defined(__AVX2__))
  #include <immintrin.h>
#elif defined(OPENHTJ2K_ENABLE_ARM_NEON)
  #include <arm_neon.h>
#endif

namespace {

// Cleanup segment length (bytes) at plane p, linear in: the sum over samples of their
// bit-planes above p, the significant samples, the significant quads, the sum over quads of
// their exponent above p, the insignificant quads, and 1.  Fitted (least squares, weighted
// toward relative error) on 14 images coded with this encoder: median error 1.5-2.5%.
const double LCUP_COEF[6] = {0.020573, 0.086332, 0.352, 0.47361, 0.0070397, 3.9884};

// The threshold estimate keeps rates in slope bins of a quarter octave.
constexpr int BINS_PER_OCTAVE = 4;
constexpr int BIN_OFFSET      = 96 * BINS_PER_OCTAVE;  // slopes from 2^-96 up
constexpr int NUM_BINS        = 160 * BINS_PER_OCTAVE;
// A class not reached yet takes the next finer level's rate per sample times this, per level.
constexpr double KAPPA = 2.0;

inline int slope_bin(double s) {
  if (!(s > 0)) return 0;
  const int b = static_cast<int>(std::floor(std::log2(s) * BINS_PER_OCTAVE)) + BIN_OFFSET;
  return std::min(std::max(b, 0), NUM_BINS - 1);
}
inline double bin_slope(int b) { return std::exp2(static_cast<double>(b - BIN_OFFSET) / BINS_PER_OCTAVE); }

inline uint32_t bit_length(uint32_t m) { return m ? 32 - count_leading_zeros(m) : 0; }

// The measuring sums run over 16 lanes: lane l takes the values i with i mod 16 == l, in
// order, with no fused multiply-add, so every instruction set gives the same floats (and
// the output does not depend on it).  n is a multiple of 16; values are below 2^31, and
// t is in [1, 2^30].
constexpr int LANES = 16;

// e2[l] = sum of sq[i] and cnt[l] = number of i with m[i] < t.
void below_plane(const uint32_t *m, const float *sq, size_t n, uint32_t t, float *e2, uint32_t *cnt) {
#if defined(OPENHTJ2K_TRY_AVX2) && defined(__AVX512F__)
  __m512 a        = _mm512_setzero_ps();
  __m512i c       = _mm512_setzero_si512();
  const __m512i T = _mm512_set1_epi32(static_cast<int32_t>(t)), one = _mm512_set1_epi32(1);
  for (size_t i = 0; i < n; i += 16) {
    const __mmask16 k = _mm512_cmplt_epu32_mask(_mm512_loadu_si512(m + i), T);
    a                 = _mm512_mask_add_ps(a, k, a, _mm512_loadu_ps(sq + i));
    c                 = _mm512_mask_add_epi32(c, k, c, one);
  }
  _mm512_storeu_ps(e2, a);
  _mm512_storeu_si512(cnt, c);
#elif defined(OPENHTJ2K_TRY_AVX2) && defined(__AVX2__)
  __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
  __m256i c0 = _mm256_setzero_si256(), c1 = _mm256_setzero_si256();
  const __m256i T = _mm256_set1_epi32(static_cast<int32_t>(t));
  for (size_t i = 0; i < n; i += 16) {
    const __m256i k0 = _mm256_cmpgt_epi32(T, _mm256_loadu_si256(reinterpret_cast<const __m256i *>(m + i)));
    const __m256i k1 =
        _mm256_cmpgt_epi32(T, _mm256_loadu_si256(reinterpret_cast<const __m256i *>(m + i + 8)));
    a0 = _mm256_add_ps(a0, _mm256_and_ps(_mm256_castsi256_ps(k0), _mm256_loadu_ps(sq + i)));
    a1 = _mm256_add_ps(a1, _mm256_and_ps(_mm256_castsi256_ps(k1), _mm256_loadu_ps(sq + i + 8)));
    c0 = _mm256_sub_epi32(c0, k0);
    c1 = _mm256_sub_epi32(c1, k1);
  }
  _mm256_storeu_ps(e2, a0);
  _mm256_storeu_ps(e2 + 8, a1);
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(cnt), c0);
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(cnt + 8), c1);
#elif defined(OPENHTJ2K_ENABLE_ARM_NEON)
  float32x4_t a[4];
  uint32x4_t c[4];
  for (int q = 0; q < 4; ++q) {
    a[q] = vdupq_n_f32(0.0f);
    c[q] = vdupq_n_u32(0);
  }
  const uint32x4_t T = vdupq_n_u32(t);
  for (size_t i = 0; i < n; i += 16)
    for (int q = 0; q < 4; ++q) {
      const uint32x4_t k = vcltq_u32(vld1q_u32(m + i + 4 * q), T);
      a[q]               = vaddq_f32(
          a[q], vreinterpretq_f32_u32(vandq_u32(k, vreinterpretq_u32_f32(vld1q_f32(sq + i + 4 * q)))));
      c[q] = vsubq_u32(c[q], k);
    }
  for (int q = 0; q < 4; ++q) {
    vst1q_f32(e2 + 4 * q, a[q]);
    vst1q_u32(cnt + 4 * q, c[q]);
  }
#else
  for (int l = 0; l < LANES; ++l) {
    e2[l]  = 0.0f;
    cnt[l] = 0;
  }
  for (size_t i = 0; i < n; i += LANES)
    for (int l = 0; l < LANES; ++l) {
      const bool b = m[i + l] < t;
      e2[l] += b ? sq[i + l] : 0.0f;
      cnt[l] += b;
    }
#endif
}

// cnt[l] = number of i with m[i] < t.
void below_count(const uint32_t *m, size_t n, uint32_t t, uint32_t *cnt) {
#if defined(OPENHTJ2K_TRY_AVX2) && defined(__AVX512F__)
  __m512i c       = _mm512_setzero_si512();
  const __m512i T = _mm512_set1_epi32(static_cast<int32_t>(t)), one = _mm512_set1_epi32(1);
  for (size_t i = 0; i < n; i += 16)
    c = _mm512_mask_add_epi32(c, _mm512_cmplt_epu32_mask(_mm512_loadu_si512(m + i), T), c, one);
  _mm512_storeu_si512(cnt, c);
#elif defined(OPENHTJ2K_TRY_AVX2) && defined(__AVX2__)
  __m256i c0 = _mm256_setzero_si256(), c1 = _mm256_setzero_si256();
  const __m256i T = _mm256_set1_epi32(static_cast<int32_t>(t));
  for (size_t i = 0; i < n; i += 16) {
    c0 = _mm256_sub_epi32(
        c0, _mm256_cmpgt_epi32(T, _mm256_loadu_si256(reinterpret_cast<const __m256i *>(m + i))));
    c1 = _mm256_sub_epi32(
        c1, _mm256_cmpgt_epi32(T, _mm256_loadu_si256(reinterpret_cast<const __m256i *>(m + i + 8))));
  }
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(cnt), c0);
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(cnt + 8), c1);
#elif defined(OPENHTJ2K_ENABLE_ARM_NEON)
  uint32x4_t c[4];
  for (int q = 0; q < 4; ++q) c[q] = vdupq_n_u32(0);
  const uint32x4_t T = vdupq_n_u32(t);
  for (size_t i = 0; i < n; i += 16)
    for (int q = 0; q < 4; ++q) c[q] = vsubq_u32(c[q], vcltq_u32(vld1q_u32(m + i + 4 * q), T));
  for (int q = 0; q < 4; ++q) vst1q_u32(cnt + 4 * q, c[q]);
#else
  for (int l = 0; l < LANES; ++l) cnt[l] = 0;
  for (size_t i = 0; i < n; i += LANES)
    for (int l = 0; l < LANES; ++l) cnt[l] += m[i + l] < t;
#endif
}

// With e = 2(m[i] mod t) + 1 - t (t a power of two) and ee = float(e) * float(e):
// r2[l] = sum of ee over m[i] >= t, r2u[l] = sum of ee over m[i] >= 2t.
void residual_plane(const uint32_t *m, size_t n, uint32_t t, float *r2, float *r2u) {
#if defined(OPENHTJ2K_TRY_AVX2) && defined(__AVX512F__)
  __m512 a = _mm512_setzero_ps(), b = _mm512_setzero_ps();
  const __m512i T  = _mm512_set1_epi32(static_cast<int32_t>(t)),
                T2 = _mm512_set1_epi32(static_cast<int32_t>(2 * t));
  const __m512i M  = _mm512_set1_epi32(static_cast<int32_t>(t - 1)),
                C  = _mm512_set1_epi32(static_cast<int32_t>(t - 1));
  for (size_t i = 0; i < n; i += 16) {
    const __m512i v = _mm512_loadu_si512(m + i);
    const __m512 e  = _mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_slli_epi32(_mm512_and_si512(v, M), 1), C));
    const __m512 ee = _mm512_mul_ps(e, e);
    a               = _mm512_mask_add_ps(a, _mm512_cmpge_epu32_mask(v, T), a, ee);
    b               = _mm512_mask_add_ps(b, _mm512_cmpge_epu32_mask(v, T2), b, ee);
  }
  _mm512_storeu_ps(r2, a);
  _mm512_storeu_ps(r2u, b);
#elif defined(OPENHTJ2K_TRY_AVX2) && defined(__AVX2__)
  __m256 a[2]       = {_mm256_setzero_ps(), _mm256_setzero_ps()},
         b[2]       = {_mm256_setzero_ps(), _mm256_setzero_ps()};
  const __m256i T1  = _mm256_set1_epi32(static_cast<int32_t>(t - 1)),
                T21 = _mm256_set1_epi32(static_cast<int32_t>(2 * t - 1));
  const __m256i M   = _mm256_set1_epi32(static_cast<int32_t>(t - 1));
  for (size_t i = 0; i < n; i += 16)
    for (int h = 0; h < 2; ++h) {
      const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(m + i + 8 * h));
      const __m256 e =
          _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_slli_epi32(_mm256_and_si256(v, M), 1), M));
      const __m256 ee = _mm256_mul_ps(e, e);
      a[h] = _mm256_add_ps(a[h], _mm256_and_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(v, T1)), ee));
      b[h] = _mm256_add_ps(b[h], _mm256_and_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(v, T21)), ee));
    }
  for (int h = 0; h < 2; ++h) {
    _mm256_storeu_ps(r2 + 8 * h, a[h]);
    _mm256_storeu_ps(r2u + 8 * h, b[h]);
  }
#elif defined(OPENHTJ2K_ENABLE_ARM_NEON)
  float32x4_t a[4], b[4];
  for (int q = 0; q < 4; ++q) a[q] = b[q] = vdupq_n_f32(0.0f);
  const uint32x4_t T = vdupq_n_u32(t), T2 = vdupq_n_u32(2 * t), M = vdupq_n_u32(t - 1);
  for (size_t i = 0; i < n; i += 16)
    for (int q = 0; q < 4; ++q) {
      const uint32x4_t v = vld1q_u32(m + i + 4 * q);
      const int32x4_t ei =
          vsubq_s32(vreinterpretq_s32_u32(vshlq_n_u32(vandq_u32(v, M), 1)), vreinterpretq_s32_u32(M));
      const float32x4_t e  = vcvtq_f32_s32(ei);
      const float32x4_t ee = vmulq_f32(e, e);
      a[q] = vaddq_f32(a[q], vreinterpretq_f32_u32(vandq_u32(vcgeq_u32(v, T), vreinterpretq_u32_f32(ee))));
      b[q] = vaddq_f32(b[q], vreinterpretq_f32_u32(vandq_u32(vcgeq_u32(v, T2), vreinterpretq_u32_f32(ee))));
    }
  for (int q = 0; q < 4; ++q) {
    vst1q_f32(r2 + 4 * q, a[q]);
    vst1q_f32(r2u + 4 * q, b[q]);
  }
#else
  for (int l = 0; l < LANES; ++l) r2[l] = r2u[l] = 0.0f;
  for (size_t i = 0; i < n; i += LANES)
    for (int l = 0; l < LANES; ++l) {
      const uint32_t v = m[i + l];
      const float e =
          static_cast<float>(static_cast<int32_t>(2 * (v & (t - 1))) - static_cast<int32_t>(t - 1));
      const float ee = e * e;
      r2[l] += (v >= t) ? ee : 0.0f;
      r2u[l] += (v >= 2 * t) ? ee : 0.0f;
    }
#endif
}

// Bit i of the result is set when v[i] >= t (i < 64; v[i] < 2^31, t >= 1).
inline uint64_t mask_ge64(const uint32_t *v, uint32_t t) {
  uint64_t r = 0;
#if defined(OPENHTJ2K_TRY_AVX2) && defined(__AVX512F__)
  const __m512i T = _mm512_set1_epi32(static_cast<int32_t>(t));
  for (int q = 0; q < 4; ++q)
    r |= static_cast<uint64_t>(_mm512_cmpge_epu32_mask(_mm512_loadu_si512(v + 16 * q), T)) << (16 * q);
#elif defined(OPENHTJ2K_TRY_AVX2) && defined(__AVX2__)
  const __m256i T1 = _mm256_set1_epi32(static_cast<int32_t>(t - 1));
  for (int q = 0; q < 8; ++q) {
    const __m256i c =
        _mm256_cmpgt_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(v + 8 * q)), T1);
    r |= static_cast<uint64_t>(static_cast<uint32_t>(_mm256_movemask_ps(_mm256_castsi256_ps(c))))
         << (8 * q);
  }
#elif defined(OPENHTJ2K_ENABLE_ARM_NEON)
  // Eight compares at a time narrowed to bytes, weighted by their bit and summed in pairs
  // (no AArch64-only instructions: this also builds for 32-bit ARM).
  const uint32x4_t T     = vdupq_n_u32(t);
  const uint8_t wts[8]   = {1, 2, 4, 8, 16, 32, 64, 128};
  const uint8x8_t weight = vld1_u8(wts);
  for (int q = 0; q < 8; ++q) {
    const uint16x4_t a = vmovn_u32(vcgeq_u32(vld1q_u32(v + 8 * q), T));
    const uint16x4_t b = vmovn_u32(vcgeq_u32(vld1q_u32(v + 8 * q + 4), T));
    uint8x8_t x        = vand_u8(vmovn_u16(vcombine_u16(a, b)), weight);
    x                  = vpadd_u8(x, x);
    x                  = vpadd_u8(x, x);
    x                  = vpadd_u8(x, x);
    r |= static_cast<uint64_t>(vget_lane_u8(x, 0)) << (8 * q);
  }
#else
  for (int i = 0; i < 64; ++i) r |= static_cast<uint64_t>(v[i] >= t) << i;
#endif
  return r;
}

// Four rows of 16 magnitudes (r[0..3], each from its own row) into one stripe word's 64
// values: dst[4c + r] = rows[r][c].
inline void stripe_word(const uint32_t *r0, const uint32_t *r1, const uint32_t *r2, const uint32_t *r3,
                        uint32_t *dst) {
#if defined(OPENHTJ2K_TRY_AVX2) && defined(__AVX512F__)
  const __m512i a0 = _mm512_loadu_si512(r0), a1 = _mm512_loadu_si512(r1);
  const __m512i a2 = _mm512_loadu_si512(r2), a3 = _mm512_loadu_si512(r3);
  const __m512i t0 = _mm512_unpacklo_epi32(a0, a1), t1 = _mm512_unpackhi_epi32(a0, a1);
  const __m512i t2 = _mm512_unpacklo_epi32(a2, a3), t3 = _mm512_unpackhi_epi32(a2, a3);
  // Per 128-bit lane L: column 4L + i, rows 0..3, in u[i].
  const __m512i u0 = _mm512_unpacklo_epi64(t0, t2), u1 = _mm512_unpackhi_epi64(t0, t2);
  const __m512i u2 = _mm512_unpacklo_epi64(t1, t3), u3 = _mm512_unpackhi_epi64(t1, t3);
  const __m512i x0 = _mm512_shuffle_i32x4(u0, u1, 0x44), x1 = _mm512_shuffle_i32x4(u2, u3, 0x44);
  const __m512i x2 = _mm512_shuffle_i32x4(u0, u1, 0xEE), x3 = _mm512_shuffle_i32x4(u2, u3, 0xEE);
  _mm512_storeu_si512(dst, _mm512_shuffle_i32x4(x0, x1, 0x88));
  _mm512_storeu_si512(dst + 16, _mm512_shuffle_i32x4(x0, x1, 0xDD));
  _mm512_storeu_si512(dst + 32, _mm512_shuffle_i32x4(x2, x3, 0x88));
  _mm512_storeu_si512(dst + 48, _mm512_shuffle_i32x4(x2, x3, 0xDD));
#elif defined(OPENHTJ2K_TRY_AVX2) && defined(__AVX2__)
  for (int h = 0; h < 2; ++h) {  // eight columns at a time
    const __m256i a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(r0 + 8 * h));
    const __m256i a1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(r1 + 8 * h));
    const __m256i a2 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(r2 + 8 * h));
    const __m256i a3 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(r3 + 8 * h));
    const __m256i t0 = _mm256_unpacklo_epi32(a0, a1), t1 = _mm256_unpackhi_epi32(a0, a1);
    const __m256i t2 = _mm256_unpacklo_epi32(a2, a3), t3 = _mm256_unpackhi_epi32(a2, a3);
    const __m256i u0 = _mm256_unpacklo_epi64(t0, t2), u1 = _mm256_unpackhi_epi64(t0, t2);
    const __m256i u2 = _mm256_unpacklo_epi64(t1, t3), u3 = _mm256_unpackhi_epi64(t1, t3);
    __m256i *d = reinterpret_cast<__m256i *>(dst + 32 * h);
    _mm256_storeu_si256(d, _mm256_permute2x128_si256(u0, u1, 0x20));
    _mm256_storeu_si256(d + 1, _mm256_permute2x128_si256(u2, u3, 0x20));
    _mm256_storeu_si256(d + 2, _mm256_permute2x128_si256(u0, u1, 0x31));
    _mm256_storeu_si256(d + 3, _mm256_permute2x128_si256(u2, u3, 0x31));
  }
#elif defined(OPENHTJ2K_ENABLE_ARM_NEON)
  for (int c = 0; c < 16; c += 4) {
    uint32x4x4_t v;
    v.val[0] = vld1q_u32(r0 + c);
    v.val[1] = vld1q_u32(r1 + c);
    v.val[2] = vld1q_u32(r2 + c);
    v.val[3] = vld1q_u32(r3 + c);
    vst4q_u32(dst + 4 * c, v);
  }
#else
  for (int c = 0; c < 16; ++c) {
    dst[4 * c]     = r0[c];
    dst[4 * c + 1] = r1[c];
    dst[4 * c + 2] = r2[c];
    dst[4 * c + 3] = r3[c];
  }
#endif
}

inline uint32_t popcount64(uint64_t x) {
  return static_cast<uint32_t>(popcount32(static_cast<uint32_t>(x))
                               + popcount32(static_cast<uint32_t>(x >> 32)));
}

// The magnitudes of a code-block's W x H samples, row after row, into m (then zeros up to
// a multiple of 16); returns their OR.
uint32_t magnitudes(const uint32_t *smp, uint32_t W, uint32_t H, size_t stride, std::vector<uint32_t> &m) {
  const size_t N  = static_cast<size_t>(W) * H;
  const size_t NP = (N + LANES - 1) / LANES * LANES;
  if (m.size() < NP) m.resize(NP);  // grow only: no zero fill per code-block
  uint32_t o = 0;
  for (uint32_t y = 0; y < H; ++y) {
    const uint32_t *sp = smp + y * stride;
    uint32_t *dp       = m.data() + static_cast<size_t>(y) * W;
    for (uint32_t x = 0; x < W; ++x) {
      const uint32_t v = ((sp[x] & 0x7FFFFFFFu) >> 1) + (sp[x] >> 31);
      dp[x]            = v;
      o |= v;
    }
  }
  for (size_t i = N; i < NP; ++i) m[i] = 0;
  return o;
}

// Sum of 16 lanes in order.
template <class T>
double lane_sum(const T *v) {
  double s = 0;
  for (int l = 0; l < LANES; ++l) s += static_cast<double>(v[l]);
  return s;
}

// Magnitude of a sample in the HT encoder's form (htj2k_magsgn).
inline uint32_t sample_magnitude(uint32_t u) { return ((u & 0x7FFFFFFFu) >> 1) + (u >> 31); }

inline const ht_rd_plane &plane_of(const ht_rd_info &rd, uint32_t j) { return rd.planes[j - rd.j0]; }

// Model cleanup length at plane p (rd.j0 <= p < rd.msb).
double model_lcup(const ht_rd_info &rd, uint32_t p) {
  double SE = 0, SU = 0;
  for (uint32_t j = p; j < rd.msb; ++j) {
    SE += plane_of(rd, j).C;
    SU += plane_of(rd, j).CQ;
  }
  const ht_rd_plane &pl = plane_of(rd, p);
  const double f[6]     = {
      SE, static_cast<double>(pl.C), static_cast<double>(pl.CQ), SU, static_cast<double>(rd.nquads - pl.CQ),
      1.0};
  double L = 0;
  for (int i = 0; i < 6; ++i) L += LCUP_COEF[i] * f[i];
  return std::max(L, 2.0);
}

// Squared error (steps) after a cleanup pass at plane p (rd.j0 <= p < rd.msb).
inline double cleanup_distortion(const ht_rd_info &rd, uint32_t p) {
  const ht_rd_plane &pl = plane_of(rd, p);
  return (pl.E2 + pl.R2) * 0.25;
}

// Distortions of an HT set at plane p >= max(1, rd.j0 + 1) after SigProp and after MagRef,
// given how many samples SigProp made significant.  SigProp reaches some of the samples
// with m in [2^(p-1), 2^p); their errors are taken as that band's average.
void refined_distortion(const ht_rd_info &rd, uint32_t p, double newly, double &Dsp, double &Dmr) {
  const ht_rd_plane &hi = plane_of(rd, p);
  const ht_rd_plane &lo = plane_of(rd, p - 1);
  const uint32_t band   = lo.C - hi.C;
  const double share    = band ? newly / band : 0.0;
  const double gain     = share * ((hi.E2 - lo.E2) - (lo.R2 - lo.R2up));
  Dsp                   = (hi.E2 - gain + hi.R2) * 0.25;
  Dmr                   = (hi.E2 - gain + lo.R2up) * 0.25;
}

// Marks a plane whose SigProp pass the measuring did not count.
constexpr uint32_t NO_COUNT = 0xFFFFFFFFu;

// Model of an HT set at plane p (max(1, rd.j0 + 1) <= p < rd.msb): its three points' rates
// (bytes) and distortions (steps).  Only the cleanup length is a model: the refinement
// passes write raw bits, SigProp one per sample it codes and a sign per sample it makes
// significant, MagRef one per significant sample, which the measuring counts.
struct set_model {
  double R[3], D[3];
};
set_model model_set(const ht_rd_info &rd, uint32_t p) {
  set_model s;
  const double Lc       = model_lcup(rd, p);
  const ht_rd_plane &hi = plane_of(rd, p);
  if (hi.Nsp == NO_COUNT) {  // not counted: the cleanup point alone
    s.R[0] = s.R[1] = s.R[2] = Lc;
    s.D[0] = s.D[1] = s.D[2] = cleanup_distortion(rd, p);
    return s;
  }
  const double sp_bits = static_cast<double>(hi.Nsp) + hi.Nnew;
  s.R[0]               = Lc;
  s.D[0]               = cleanup_distortion(rd, p);
  s.R[1]               = Lc + std::ceil(sp_bits / 8);
  s.R[2]               = Lc + std::ceil((sp_bits + hi.C) / 8);
  refined_distortion(rd, p, hi.Nnew, s.D[1], s.D[2]);
  return s;
}

// Per-thread scratch of the measuring, coding and transcoding.
struct rd_scratch {
  std::vector<uint32_t> qmax;    // quad maxima
  std::vector<uint32_t> all;     // all magnitudes, W x H
  std::vector<uint32_t> zs;      // all magnitudes in stripe-word order (ht_sigprop.hpp)
  std::vector<uint64_t> sig[2];  // stripe words: the samples significant at a plane
  std::vector<float> sq;         // (2m + 1)^2
  std::vector<uint64_t> newly;   // SigProp's newly significant samples
  std::vector<int32_t> dec;      // decoded cleanup samples
  std::vector<uint8_t> states;   // decoder states
  std::vector<int32_t> enc;      // samples for the re-coding
  std::vector<uint8_t> data;     // a copy of the coded data, with room past it
};
rd_scratch &thread_scratch() {
  static thread_local rd_scratch s;
  return s;
}

uint32_t count_newly(const std::vector<uint64_t> &newly, size_t words) {
  size_t n = 0;
  for (size_t k = 0; k < words; ++k)
    n += popcount32(static_cast<uint32_t>(newly[k])) + popcount32(static_cast<uint32_t>(newly[k] >> 32));
  return static_cast<uint32_t>(n);
}

}  // namespace

/********************************************************************************
 * Measurement
 *******************************************************************************/
void ht_rd_stats(j2k_codeblock *block, double lambda) {
  uint32_t or_val = 0;
  if (!block->pre_quantized)
    block->quantize(or_val);
  else
    or_val = block->pre_or_val;
  block->pre_quantized = true;
  block->pre_or_val    = or_val;

  ht_rd_info &rd      = *block->rd;
  const uint32_t W    = block->size.x;
  const uint32_t H    = block->size.y;
  const size_t sstr   = block->blksampl_stride;
  const uint32_t *smp = reinterpret_cast<const uint32_t *>(block->sample_buf);
  const uint32_t QW   = (W + 1) / 2;
  const uint32_t QH   = (H + 1) / 2;
  rd_scratch &sc      = thread_scratch();
  rd.nquads           = QW * QH;
  rd.planes.reset();
  rd.steps.reset();
  rd.nsteps = 0;
  rd.msb    = 0;
  rd.j0     = 0;
  rd.D2zero = 0;
  if (!or_val) return;

  // Magnitudes, their largest, and (2m + 1)^2 as float (0 for m = 0).
  const uint32_t msb = bit_length(magnitudes(smp, W, H, sstr, sc.all));
  if (msb == 0) return;
  const size_t NP = (static_cast<size_t>(W) * H + LANES - 1) / LANES * LANES;  // padded with zeros
  if (sc.sq.size() < NP) sc.sq.resize(NP);
  {
    const uint32_t *mg = sc.all.data();
    float *sq          = sc.sq.data();
    for (size_t i = 0; i < NP; ++i) {  // m < 2^30
      const float tm = static_cast<float>(static_cast<int32_t>(2 * mg[i] + 1));
      sq[i]          = (tm * tm) * static_cast<float>(mg[i] != 0);
    }
  }
  // Quad maxima (a quad's significance at a plane is its maximum's).
  const size_t NQ  = static_cast<size_t>(QW) * QH;
  const size_t NQP = (NQ + LANES - 1) / LANES * LANES;
  if (sc.qmax.size() < NQP) sc.qmax.resize(NQP);
  for (size_t i = NQ; i < NQP; ++i) sc.qmax[i] = 0;
  for (uint32_t qy = 0; qy < QH; ++qy) {
    const uint32_t *r0 = sc.all.data() + static_cast<size_t>(2 * qy) * W;
    const uint32_t *r1 = (2 * qy + 1 < H) ? r0 + W : r0;
    uint32_t *qp       = sc.qmax.data() + static_cast<size_t>(qy) * QW;
    for (uint32_t qx = 0; qx < W / 2; ++qx)
      qp[qx] = std::max(std::max(r0[2 * qx], r0[2 * qx + 1]), std::max(r1[2 * qx], r1[2 * qx + 1]));
    if (W & 1) qp[QW - 1] = std::max(r0[W - 1], r1[W - 1]);  // a last quad one column wide
  }

  // The planes measured: from a few below the plane the estimate points to (an HT set
  // at plane p has a slope near its weight times 4^p), so that the set chosen later and
  // the plane under it are in range.
  uint32_t j0 = 0;
  if (lambda > 0 && rd.w > 0) {
    const double p_hint = 0.5 * std::log2(lambda / rd.w);
    const double lo     = std::floor(p_hint) - 3;
    j0                  = (lo <= 0) ? 0 : static_cast<uint32_t>(std::min(lo, 31.0));
  }
  // A set at plane p needs plane p - 1 measured too, the coarsest set included.
  j0 = std::min(j0, msb >= 2 ? msb - 2 : 0u);

  rd.msb             = static_cast<uint8_t>(msb);
  rd.j0              = static_cast<uint8_t>(j0);
  rd.planes          = MAKE_UNIQUE<ht_rd_plane[]>(msb - j0);
  const uint32_t *mg = sc.all.data();
  const float *sq    = sc.sq.data();
  const uint32_t *qm = sc.qmax.data();
  for (uint32_t j = j0; j <= msb; ++j) {
    const uint32_t t = 1u << j;  // msb <= 30: magnitudes stay below 2^30
    float e2[LANES];
    uint32_t lt[LANES], lq[LANES];  // samples and quads below 2^j
    below_plane(mg, sq, NP, t, e2, lt);
    below_count(qm, NQP, t, lq);
    const double E2     = lane_sum(e2);
    const size_t below  = static_cast<size_t>(lane_sum(lt));
    const size_t belowq = static_cast<size_t>(lane_sum(lq));
    if (j == msb) {  // every sample below 2^msb: leaving the code-block out
      rd.D2zero = E2;
      break;
    }
    ht_rd_plane &pl = rd.planes[j - j0];
    pl.C            = static_cast<uint32_t>(NP - below);
    pl.CQ           = static_cast<uint32_t>(NQP - belowq);
    pl.E2           = E2;
  }
  // The residuals after a cleanup pass at each plane.  (Modelling them as uniform costs
  // little less and misleads badly on smooth content: 14 dB at 2 bpp on a 16-bit
  // animation frame.)
  for (uint32_t j = j0; j < msb; ++j) {
    float r2[LANES], r2u[LANES];
    residual_plane(mg, NP, 1u << j, r2, r2u);
    ht_rd_plane &pl = rd.planes[j - j0];
    pl.R2           = lane_sum(r2);
    pl.R2up         = lane_sum(r2u);
    pl.Nsp = pl.Nnew = 0;
  }

  // The SigProp pass of the set at each plane j > j0: the samples it codes and those it
  // makes significant.  A sample below 2^j has bit j - 1 set when it is significant at
  // plane j - 1, so the significance at the two planes is all it needs.
  {
    using ht_sigprop::MAX_WORDS;
    const uint32_t ns        = (H + 3) / 4;
    const uint32_t nk        = (W + 15) / 16;
    const size_t nw          = static_cast<size_t>(ns) * nk;
    const uint64_t last_cols = (W & 15) ? (1ull << (4 * (W & 15))) - 1 : ~0ull;
    const bool causal        = (block->Cmodes & CAUSAL) != 0;
    if (sc.zs.size() < 64 * nw) sc.zs.resize(64 * nw);
    uint32_t *zs = sc.zs.data();
    // Whole words four at a time; the rest (a last stripe or word only partly inside the
    // code-block) sample by sample, over zeros.
    const uint32_t full_s = H / 4, full_k = W / 16;
    if (full_s < ns || full_k < nk) std::fill(zs, zs + 64 * nw, 0u);
    for (uint32_t st = 0; st < ns; ++st) {
      const uint32_t *row = mg + static_cast<size_t>(4 * st) * W;
      uint32_t *dst       = zs + static_cast<size_t>(st) * nk * 64;
      if (st < full_s)
        for (uint32_t k = 0; k < full_k; ++k)
          stripe_word(row + 16 * k, row + W + 16 * k, row + 2 * W + 16 * k, row + 3 * W + 16 * k,
                      dst + 64 * k);
      for (uint32_t y = 4 * st; y < std::min(H, 4 * st + 4); ++y)
        for (uint32_t x = (st < full_s) ? 16 * full_k : 0; x < W; ++x)
          dst[(x >> 4) * 64 + 4 * (x & 15) + (y & 3)] = mg[static_cast<size_t>(y) * W + x];
    }
    for (auto &v : sc.sig)
      if (v.size() < nw) v.resize(nw);
    // Below the plane under the estimate's (j0 + 2), a set's refinement points matter only
    // when the estimate is 16 times too large: those sets keep their cleanup point alone.
    const uint32_t jc = (lambda > 0 && j0 > 0) ? std::min(j0 + 2, msb) : j0 + 1;
    for (uint32_t j = j0 + 1; j < jc; ++j) rd.planes[j - j0].Nsp = NO_COUNT;
    uint64_t *lo = sc.sig[0].data(), *hi = sc.sig[1].data();
    for (size_t w = 0; w < nw; ++w) lo[w] = mask_ge64(sc.zs.data() + 64 * w, 1u << (jc - 1));
    static const uint64_t none[MAX_WORDS] = {};
    for (uint32_t j = jc; j < msb; ++j) {
      for (size_t w = 0; w < nw; ++w) hi[w] = mask_ge64(sc.zs.data() + 64 * w, 1u << j);
      uint64_t above[MAX_WORDS] = {}, newly[MAX_WORDS], coded[MAX_WORDS];
      uint32_t nsp = 0, nnew = 0;
      bool above_any = false;  // a sample in above[] is set
      for (uint32_t st = 0; st < ns; ++st) {
        const uint64_t *sig  = hi + static_cast<size_t>(st) * nk;
        const uint64_t *next = (st + 1 < ns) ? hi + static_cast<size_t>(st + 1) * nk : none;
        const uint32_t rows  = std::min(4u, H - 4 * st);
        const uint64_t live  = ht_sigprop::ROW0 * ((1u << rows) - 1);
        // SigProp codes nothing in a stripe with nothing significant in or next to it, nor
        // in one where every sample is significant already.
        uint64_t any_sig = 0, any_open = 0, next_row0 = 0;
        for (uint32_t k = 0; k < nk; ++k) {
          any_sig |= sig[k];
          any_open |= ~sig[k] & live & ((k + 1 == nk) ? last_cols : ~0ull);
          next_row0 |= next[k] & ht_sigprop::ROW0;
        }
        if (any_sig == 0 && !above_any && (causal || next_row0 == 0)) continue;  // above[] stays zero
        if (any_open == 0) {
          above_any = false;
          for (uint32_t k = 0; k < nk; ++k) {
            above[k] = sig[k] & ht_sigprop::ROW3;
            above_any |= above[k] != 0;
          }
          continue;
        }
        if (nk == 4)  // 64 samples wide, the usual
          ht_sigprop::solve_stripe<4>(sig, lo + static_cast<size_t>(st) * nk, next, above, nk, rows,
                                      last_cols, causal, newly, coded);
        else
          ht_sigprop::solve_stripe<>(sig, lo + static_cast<size_t>(st) * nk, next, above, nk, rows,
                                     last_cols, causal, newly, coded);
        above_any = false;
        for (uint32_t k = 0; k < nk; ++k) {
          nsp += popcount64(coded[k]);
          nnew += popcount64(newly[k]);
          above_any |= above[k] != 0;
        }
      }
      rd.planes[j - j0].Nsp  = nsp;
      rd.planes[j - j0].Nnew = nnew;
      std::swap(lo, hi);
    }
  }

  // The steps of the convex hull of the model points, for the threshold estimate: rate by
  // slope bin.
  struct pt {
    double R, D;
  };
  pt cand[3 * 32];
  int nc = 0;
  for (uint32_t j = j0; j < msb; ++j) {
    if (j == j0) {
      cand[nc++] = {model_lcup(rd, j), rd.w * cleanup_distortion(rd, j)};
      continue;
    }
    const set_model sm = model_set(rd, j);
    for (int k = 0; k < 3; ++k) cand[nc++] = {sm.R[k], rd.w * sm.D[k]};
  }
  std::sort(cand, cand + nc,
            [](const pt &a, const pt &b) { return a.R < b.R || (a.R == b.R && a.D < b.D); });
  pt hull[3 * 32 + 1];
  hull[0] = {0, rd.w * rd.D2zero * 0.25};
  int n   = 1;
  for (int c = 0; c < nc; ++c) {  // increasing rate
    const pt q = cand[c];
    if (q.D >= hull[n - 1].D || q.R <= hull[n - 1].R) continue;
    while (n >= 2
           && (hull[n - 1].D - q.D) * (hull[n - 1].R - hull[n - 2].R)
                  >= (hull[n - 2].D - hull[n - 1].D) * (q.R - hull[n - 1].R))
      --n;
    hull[n++] = q;
  }
  rd.steps  = MAKE_UNIQUE<std::pair<uint16_t, double>[]>(static_cast<size_t>(n - 1));
  rd.nsteps = static_cast<uint8_t>(n - 1);
  for (int i = 1; i < n; ++i) {
    const double dR = hull[i].R - hull[i - 1].R;
    rd.steps[static_cast<size_t>(i - 1)] =
        std::make_pair(static_cast<uint16_t>(slope_bin((hull[i - 1].D - hull[i].D) / dR)), dR);
  }
}

/********************************************************************************
 * Coding
 *******************************************************************************/
// The plane of the HT set holding a measured code-block's best model point at slope L
// (cleanup, SigProp or MagRef); false when leaving the code-block out is best.  A set's
// SigProp and MagRef points need the plane under it measured.
static bool best_set(const ht_rd_info &rd, double L, uint32_t &p) {
  double best = rd.w * rd.D2zero * 0.25;
  bool found  = false;
  for (uint32_t j = rd.j0; j < rd.msb; ++j) {
    if (j == 0) {
      const double c = rd.w * cleanup_distortion(rd, 0) + L * model_lcup(rd, 0);
      if (c < best) {
        best  = c;
        p     = 0;
        found = true;
      }
      continue;
    }
    if (j == rd.j0) continue;
    const set_model m = model_set(rd, j);
    for (int k = 0; k < 3; ++k) {
      const double c = rd.w * m.D[k] + L * m.R[k];
      if (c < best) {
        best  = c;
        p     = j;
        found = true;
      }
    }
  }
  return found;
}
int32_t ht_rd_code(j2k_codeblock *block, const uint8_t ROIshift) {
  ht_rd_info &rd    = *block->rd;
  block->refsegment = false;
  if (rd.msb == 0) {
    block->ht_refine_planes = 0;
    htj2k_encode(block, ROIshift);  // nothing significant: an empty code-block
    rd.p_coded = 0;
    rd.Lcup = rd.Lsp = rd.Lref = 0;
    rd.Dcup = rd.Dsp = rd.Dmr = 0;
    return EXIT_SUCCESS;
  }

  // The set holding the best point at the estimate (cleanup, SigProp or MagRef).
  uint32_t p           = 0;
  const bool pred_none = !best_set(rd, rd.lambda, p);
  if (pred_none) {
    // Left out at the estimate.  Unless it stays out at a quarter of it (two planes'
    // worth), code the coarsest set, which costs little and may still be needed;
    // otherwise code nothing: a code-block with something significant still takes a full
    // pass of the cleanup coder, and most of these would be coded for nothing.
    uint32_t q = 0;
    if (!best_set(rd, 0.25 * rd.lambda, q)) {
      rd.p_coded = 0;
      rd.refined = false;
      rd.skipped = true;
      rd.Lcup = rd.Lsp = rd.Lref = 0;
      block->length              = 0;
      block->num_passes          = 0;
      block->layer_passes[0]     = 0;
      block->pass_length[0]      = 0;
      return EXIT_SUCCESS;
    }
    p = rd.msb - 1;
  }

  rd.p_coded     = static_cast<uint8_t>(p);
  rd_scratch &sc = thread_scratch();

  sc.newly.assign(static_cast<size_t>((block->size.y + 3) / 4) * ((block->size.x + 15) / 16), 0);
  ht_refinement_info info;
  info.newly              = sc.newly.data();
  block->ht_refine_planes = static_cast<uint8_t>(p);
  htj2k_encode_refined(block, ROIshift, &info);
  block->ht_refine_planes = 0;
  block->refsegment       = false;

  rd.refined = p >= 1;
  rd.Lcup    = block->pass_length[0];
  rd.Dcup    = cleanup_distortion(rd, p);
  if (p >= 1) {
    rd.Lsp     = info.Lsp;
    rd.Lref    = block->pass_length[2];
    rd.sp_tail = info.sp_tail;
    refined_distortion(rd, p, static_cast<double>(count_newly(sc.newly, sc.newly.size())), rd.Dsp, rd.Dmr);
  } else {
    rd.Lsp = rd.Lref = 0;
    rd.Dsp = rd.Dmr = rd.Dcup;
  }
  return EXIT_SUCCESS;
}

/********************************************************************************
 * Threshold estimate
 *******************************************************************************/
namespace {

// Unscaled 9/7 synthesis filters: the irreversible lifting without its K step, which the
// encoder folds into the step sizes.
const double SYN97_L[7] = {-0.091271763114250, -0.057543526228500, 0.591271763114250, 1.115087052457000,
                           0.591271763114250,  -0.057543526228500, -0.091271763114250};
const double SYN97_H[9] = {0.053497514821622,  0.033728236885750, -0.156446533057980,
                           -0.533728236885750, 1.205898036472720, -0.533728236885750,
                           -0.156446533057980, 0.033728236885750, 0.053497514821622};
constexpr double K97    = 1.230174104914001;

// Energy of the 1-D synthesis basis of the low- and high-pass band at each level (0 is the
// finest) for NL levels: iterated upsampling and low-pass filtering.
void synthesis_energies(uint8_t NL, std::vector<double> &gl, std::vector<double> &gh) {
  std::vector<double> L(SYN97_L, SYN97_L + 7), H(SYN97_H, SYN97_H + 9);
  for (auto &v : L) v /= K97;
  for (auto &v : H) v *= K97 / 2;
  std::vector<double> bl = L, bh = H;
  gl.assign(NL, 0);
  gh.assign(NL, 0);
  auto next = [&L](const std::vector<double> &b) {
    std::vector<double> out(L.size() + 2 * b.size() - 1, 0.0);
    for (size_t i = 0; i < b.size(); ++i)
      for (size_t j = 0; j < L.size(); ++j) out[2 * i + j] += b[i] * L[j];
    return out;
  };
  for (uint8_t lev = 0; lev < NL; ++lev) {
    for (double v : bl) gl[lev] += v * v;
    for (double v : bh) gh[lev] += v * v;
    if (lev + 1 == NL) break;
    bl = next(bl);
    bh = next(bh);
  }
}

// Calls fn(block, comp, resolution, NL) for every HT code-block of a tile with a record.
template <class F>
void for_each_rd_block(j2k_tile &tile, F &&fn) {
  for (uint16_t c = 0; c < tile.get_num_components(); ++c) {
    j2k_tile_component *tc = tile.get_tile_component(c);
    const uint8_t NL       = tc->get_dwt_levels();
    for (uint8_t r = 0; r <= NL; ++r) {
      j2k_resolution *cr = tc->access_resolution(r);
      if (cr->is_empty) continue;
      for (uint32_t pi = 0; pi < cr->npw * cr->nph; ++pi) {
        j2k_precinct *cp = cr->access_precinct(pi);
        for (uint8_t b = 0; b < cr->num_bands; ++b) {
          j2k_precinct_subband *cpb = cp->access_pband(b);
          const uint32_t n          = cpb->num_codeblock_x * cpb->num_codeblock_y;
          for (uint32_t i = 0; i < n; ++i) {
            j2k_codeblock *block = cpb->access_codeblock(i);
            if (block->rd != nullptr) fn(block, c, r, NL);
          }
        }
      }
    }
  }
}

}  // namespace

ht_rc_state::ht_rc_state(uint64_t budget) : budget_(budget) {}

void ht_rc_state::add_tile(j2k_tile &tile) {
  const uint16_t ncomp = tile.get_num_components();
  const bool ict       = tile.uses_MCT() && ncomp >= 3;
  // Inverse irreversible colour transform: energy of each component's column.
  const double gc_ict[3] = {3.0, 0.344136 * 0.344136 + 1.772 * 1.772, 1.402 * 1.402 + 0.714136 * 0.714136};
  std::map<std::tuple<uint16_t, uint8_t, uint8_t>, uint16_t> index;
  for (size_t i = 0; i < classes_.size(); ++i)
    index[std::make_tuple(classes_[i].comp, classes_[i].level, classes_[i].band)] =
        static_cast<uint16_t>(i);
  std::vector<double> gl, gh;
  uint8_t gains_for = 0xFF;
  for_each_rd_block(tile, [&](j2k_codeblock *block, uint16_t c, uint8_t r, uint8_t NL) {
    if (gains_for != NL) {
      synthesis_energies(NL, gl, gh);
      gains_for = NL;
    }
    const uint8_t band = block->get_orientation();
    double gb          = 1.0;
    if (NL > 0) {
      const uint8_t lev = static_cast<uint8_t>(r ? NL - r : NL - 1);
      gb = (band == 0) ? gl[lev] * gl[lev] : (band == 3) ? gh[lev] * gh[lev] : gl[lev] * gh[lev];
    }
    const double step = block->stepsize;
    const double gc   = (ict && c < 3) ? gc_ict[c] : 1.0;
    block->rd->w      = step * step * gb * gc;
    // Classes by component, resolution and band, across tiles.
    const auto key = std::make_tuple(c, r, band);
    auto it        = index.find(key);
    if (it == index.end()) {
      sb_class k;
      k.comp  = static_cast<uint8_t>(c);
      k.level = r;
      k.band  = band;
      k.rate.assign(NUM_BINS, 0.0);
      classes_.push_back(k);
      it = index.insert(std::make_pair(key, static_cast<uint16_t>(classes_.size() - 1))).first;
    }
    block->rd->cls = it->second;
    classes_[it->second].total += static_cast<uint64_t>(block->size.x) * block->size.y;
  });
}

void ht_rc_state::fold(const j2k_codeblock *block) {
  const ht_rd_info &rd = *block->rd;
  sb_class &k          = classes_[rd.cls];
  k.seen += static_cast<uint64_t>(block->size.x) * block->size.y;
  for (uint32_t i = 0; i < rd.nsteps; ++i) k.rate[rd.steps[i].first] += rd.steps[i].second;
}

void ht_rc_state::solve() {
  // Each measured class stands for its whole sample count; a class not measured yet takes
  // the next finer measured resolution of its component and orientation (LL: HL), its rate
  // per sample KAPPA times larger per level.
  std::vector<double> mult(classes_.size(), 0.0);
  bool any = false;
  for (size_t i = 0; i < classes_.size(); ++i) {
    const sb_class &k = classes_[i];
    if (k.seen) {
      mult[i] += static_cast<double>(k.total) / static_cast<double>(k.seen);
      any = true;
      continue;
    }
    const uint8_t band = k.band ? k.band : 1;  // LL takes HL
    double f           = KAPPA;
    for (int lev = k.level + 1; lev < 64; ++lev, f *= KAPPA) {
      size_t found = classes_.size();
      bool exists  = false;
      for (size_t m = 0; m < classes_.size(); ++m) {
        if (classes_[m].comp != k.comp || classes_[m].level != lev) continue;
        exists = true;
        if (classes_[m].band == band && classes_[m].seen) found = m;
      }
      if (found < classes_.size()) {
        mult[found] += f * static_cast<double>(k.total) / static_cast<double>(classes_[found].seen);
        break;
      }
      if (!exists) break;
    }
  }
  if (!any) return;
  double R = 0;
  for (int b = NUM_BINS - 1; b >= 0; --b) {
    for (size_t i = 0; i < classes_.size(); ++i) R += mult[i] * classes_[i].rate[static_cast<size_t>(b)];
    if (R >= static_cast<double>(budget_)) {
      lambda_ = bin_slope(b);
      return;
    }
  }
  lambda_ = bin_slope(0);  // everything fits
}

/********************************************************************************
 * Final cut
 *******************************************************************************/
namespace {

// A truncation point of a code-block: rate in bytes, weighted distortion, the cleanup
// plane of its set and its number of passes (0: the code-block is left out).
struct rd_point {
  double R;
  double D;
  uint8_t p;
  uint8_t passes;
};

struct cblk_ref {
  j2k_codeblock *block;
  uint32_t first;  // its hull points: hull[first .. first + n)
  uint32_t n;
};

// A step along a code-block's hull, from point k - 1 to point k.
struct hull_step {
  double slope;  // distortion decrease per byte
  double dR;
  uint32_t cblk;
  uint32_t k;
};

// The code-block's passes and lengths for a point of the HT set at cleanup plane p.
void set_point(j2k_codeblock *b, uint8_t p, uint8_t passes, uint32_t Lcup, uint32_t Lsp, uint32_t Lref) {
  b->ensure_pass_capacity(3);
  b->layer_start[0]    = 0;
  b->layer_passes[0]   = passes;
  b->num_passes        = passes;
  b->pass_length[0]    = (passes >= 1) ? Lcup : 0;
  b->pass_length[1]    = (passes == 2) ? Lsp : 0;
  b->pass_length[2]    = (passes == 3) ? Lref : 0;
  b->pass_length_count = passes;
  b->length            = (passes == 0) ? 0 : Lcup + (passes == 2 ? Lsp : 0) + (passes == 3 ? Lref : 0);
  // A code-block left out keeps the largest value, so that its tag-tree parents do not drop.
  b->num_ZBP = static_cast<uint8_t>(b->get_Mb() - 1 - (passes ? p : 0));
}

// Runs fn(0) .. fn(n - 1), on the thread pool when there is one.
void parallel_for(size_t n, const std::function<void(size_t)> &fn) {
#ifdef OPENHTJ2K_THREAD
  ThreadPool *pool = ThreadPool::get();
  if (pool != nullptr && pool->num_threads() > 1 && n > 1) {
    struct ctx_t {
      std::atomic<size_t> next{0};
      std::atomic<int> running{0};
      size_t n;
      const std::function<void(size_t)> *fn;
    } ctx;
    ctx.n             = n;
    ctx.fn            = &fn;
    const int workers = static_cast<int>(std::min<size_t>(pool->num_threads(), n));
    ctx_t *c          = &ctx;
    auto drain        = [c]() {
      for (size_t i = c->next.fetch_add(1); i < c->n; i = c->next.fetch_add(1)) (*c->fn)(i);
    };
    ctx.running.store(workers, std::memory_order_relaxed);
    for (int w = 0; w < workers; ++w) {
      pool->push([c, drain]() {
        drain();
        c->running.fetch_sub(1, std::memory_order_release);
      });
    }
    drain();
    while (ctx.running.load(std::memory_order_acquire) > 0) {
      if (!pool->try_run_one()) std::this_thread::yield();
    }
    return;
  }
#endif
  for (size_t i = 0; i < n; ++i) fn(i);
}

// Re-codes a code-block whose data holds the HT set at cleanup plane rd.p_coded as the set
// at plane p (rd.p_coded < p < rd.msb), and makes that the coded set.
bool transcode(j2k_codeblock *block, uint32_t p) {
  ht_rd_info &rd        = *block->rd;
  const uint32_t W      = block->size.x;
  const uint32_t H      = block->size.y;
  const uint32_t stride = round_up(W, 8U);
  const uint32_t QH     = round_up(H, 8U);
  const uint32_t Lcup   = rd.Lcup;
  const uint32_t d      = p - rd.p_coded;
  rd_scratch &sc        = thread_scratch();

  // Decode the coded cleanup pass from a copy: the decoder modifies its input, and may read
  // a few bytes past it.
  sc.data.assign(static_cast<size_t>(Lcup) + 64, 0);
  std::memcpy(sc.data.data(), block->get_compressed_data(), Lcup);
  sc.dec.assign(static_cast<size_t>(stride) * (QH + 2), 0);
  sc.states.assign(static_cast<size_t>(stride + 2) * (QH + 2), 0);
  bool pooled    = false;
  uint8_t *coded = block->detach_compressed_data(pooled);
  block->attach_compressed_data(sc.data.data(), true);  // not freed with the code-block
  block->length            = Lcup;
  block->num_passes        = 1;
  block->pass_length[0]    = Lcup;
  block->pass_length_count = 1;
  block->num_ZBP           = static_cast<uint8_t>(block->get_Mb() - 1 - rd.p_coded);
  block->sample_buf        = sc.dec.data();
  block->blksampl_stride   = stride;
  block->block_states      = sc.states.data();
  block->blkstate_stride   = stride + 2;
  const bool ok            = htj2k_decode_cleanup_raw(block);
  bool unused              = false;
  block->detach_compressed_data(unused);
  block->block_states = nullptr;
  if (!ok) {
    block->attach_compressed_data(coded, pooled);
    block->sample_buf = nullptr;
    return false;
  }

  // The decoded magnitudes, at the coded plane, in the encoder's sample form.
  const uint32_t pLSB = 30u - block->num_ZBP;
  sc.enc.assign(static_cast<size_t>(stride) * QH, 0);
  uint32_t or_val = 0;
  for (uint32_t y = 0; y < H; ++y) {
    const uint32_t *sp =
        reinterpret_cast<const uint32_t *>(sc.dec.data()) + static_cast<size_t>(y) * stride;
    int32_t *dp = sc.enc.data() + static_cast<size_t>(y) * stride;
    for (uint32_t x = 0; x < W; ++x) {
      const uint32_t mu = (sp[x] & 0x7FFFFFFFu) >> pLSB;
      or_val |= mu;
      dp[x] = mu ? static_cast<int32_t>(0x80000000u | ((mu - 1) << 1) | (sp[x] >> 31)) : 0;
    }
  }

  // Code the set d planes up from the coded one, into an exact-size buffer.
  sc.newly.assign(static_cast<size_t>((H + 3) / 4) * ((W + 15) / 16), 0);
  ht_refinement_info info;
  info.newly               = sc.newly.data();
  block->sample_buf        = sc.enc.data();
  block->pre_quantized     = true;
  block->pre_or_val        = or_val ? 1 : 0;
  block->refsegment        = false;
  block->ht_refine_planes  = static_cast<uint8_t>(d);
  cblk_data_pool *own_pool = swap_thread_cblk_pool(nullptr);
  htj2k_encode_refined(block, 0, &info);
  swap_thread_cblk_pool(own_pool);
  block->ht_refine_planes = 0;
  block->refsegment       = false;
  block->sample_buf       = nullptr;
  block->num_ZBP          = static_cast<uint8_t>(block->num_ZBP - rd.p_coded);
  bool trial_pooled       = false;
  uint8_t *trial          = block->detach_compressed_data(trial_pooled);
  auto *exact             = static_cast<uint8_t *>(malloc(block->length ? block->length : 1));
  std::memcpy(exact, trial, block->length);
  free(trial);
  block->attach_compressed_data(exact, false);
  if (!pooled) free(coded);
  if (block->num_passes != 3) return false;

  rd.p_coded = static_cast<uint8_t>(p);
  rd.Lcup    = block->pass_length[0];
  rd.Lsp     = info.Lsp;
  rd.Lref    = block->pass_length[2];
  rd.sp_tail = info.sp_tail;
  rd.Dcup    = cleanup_distortion(rd, p);
  refined_distortion(rd, p, static_cast<double>(count_newly(sc.newly, sc.newly.size())), rd.Dsp, rd.Dmr);
  return true;
}

}  // namespace

uint64_t ht_rate_control(j2k_tile *tiles, uint32_t num_tiles, uint64_t fixed_bytes, uint64_t target_bytes) {
  std::vector<j2k_codeblock *> blocks;
  for (uint32_t t = 0; t < num_tiles; ++t)
    for_each_rd_block(tiles[t], [&](j2k_codeblock *b, uint16_t, uint8_t, uint8_t) { blocks.push_back(b); });

  std::vector<cblk_ref> cblks(blocks.size());
  std::vector<rd_point> hull, pts;
  std::vector<hull_step> steps;
  std::vector<uint32_t> choice(blocks.size(), 0);
  auto point_of = [&](uint32_t i) -> const rd_point & { return hull[cblks[i].first + choice[i]]; };

  // Every code-block's convex hull: leaving it out, the three points of its coded set, and
  // the model points of the sets above.
  bool with_model = true;
  auto build      = [&]() {
    hull.clear();
    steps.clear();
    for (uint32_t i = 0; i < blocks.size(); ++i) {
      const ht_rd_info &rd = *blocks[i]->rd;
      const double w       = rd.w;
      pts.clear();
      pts.push_back({0, w * rd.D2zero * 0.25, 0, 0});
      if (rd.msb > 0 && !rd.skipped) {
        pts.push_back({static_cast<double>(rd.Lcup), w * rd.Dcup, rd.p_coded, 1});
        if (rd.refined) {
          if (rd.Lsp) pts.push_back({static_cast<double>(rd.Lcup + rd.Lsp), w * rd.Dsp, rd.p_coded, 2});
          pts.push_back({static_cast<double>(rd.Lcup + rd.Lref), w * rd.Dmr, rd.p_coded, 3});
        }
        if (with_model)
          for (uint32_t p = rd.p_coded + 1u; p < rd.msb; ++p) {
            const set_model sm = model_set(rd, p);
            for (uint8_t k = 0; k < 3; ++k)
              pts.push_back({sm.R[k], w * sm.D[k], static_cast<uint8_t>(p), static_cast<uint8_t>(k + 1)});
          }
      }
      // Ties go to the plane nearest the coded one (fewer transcodes).
      std::sort(pts.begin(), pts.end(), [](const rd_point &a, const rd_point &b) {
        if (a.R != b.R) return a.R < b.R;
        if (a.D != b.D) return a.D < b.D;
        if (a.p != b.p) return a.p < b.p;
        return a.passes < b.passes;
      });
      cblk_ref ref{blocks[i], static_cast<uint32_t>(hull.size()), 0};
      hull.push_back(pts[0]);
      for (size_t k = 1; k < pts.size(); ++k) {
        const rd_point &q = pts[k];
        if (q.D >= hull.back().D || q.R <= hull.back().R) continue;
        while (hull.size() - ref.first >= 2) {
          const rd_point &a = hull[hull.size() - 2];
          const rd_point &b = hull.back();
          if ((b.D - q.D) * (b.R - a.R) >= (a.D - b.D) * (q.R - b.R))
            hull.pop_back();
          else
            break;
        }
        hull.push_back(q);
      }
      ref.n    = static_cast<uint32_t>(hull.size()) - ref.first;
      cblks[i] = ref;
      for (uint32_t k = 1; k < ref.n; ++k) {
        const rd_point &a = hull[ref.first + k - 1];
        const rd_point &b = hull[ref.first + k];
        steps.push_back({(a.D - b.D) / (b.R - a.R), b.R - a.R, i, k});
      }
    }
    std::sort(steps.begin(), steps.end(), [](const hull_step &a, const hull_step &b) {
      if (a.slope != b.slope) return a.slope > b.slope;
      if (a.cblk != b.cblk) return a.cblk < b.cblk;
      return a.k < b.k;
    });
  };
  // The points for a byte budget of the code-blocks: the steps in decreasing slope while
  // they fit, then any later step that still fits and continues its code-block's hull.
  auto cut = [&](double budget) {
    std::fill(choice.begin(), choice.end(), 0);
    double used = 0;
    for (const hull_step &s : steps) {
      if (choice[s.cblk] + 1 != s.k || used + s.dR > budget) continue;
      choice[s.cblk] = s.k;
      used += s.dR;
    }
  };
  // Sets every code-block to its chosen point (model lengths above the coded set), builds
  // the packet headers and returns the codestream length.
  auto apply = [&]() {
    for (uint32_t i = 0; i < blocks.size(); ++i) {
      const rd_point &pt   = point_of(i);
      const ht_rd_info &rd = *blocks[i]->rd;
      if (pt.passes == 0)
        set_point(blocks[i], 0, 0, 0, 0, 0);
      else if (pt.p == rd.p_coded)
        set_point(blocks[i], pt.p, pt.passes, rd.Lcup, rd.Lsp, rd.Lref);
      else
        set_point(blocks[i], pt.p, 1, static_cast<uint32_t>(std::ceil(pt.R)), 0, 0);
    }
    uint64_t total = fixed_bytes;
    for (uint32_t t = 0; t < num_tiles; ++t) {
      tiles[t].build_packet_headers();
      total += tiles[t].get_tile_part_size();
    }
    return total;
  };
  // The packet headers depend on the points: search the budget that fills the target.
  double start = (target_bytes > fixed_bytes) ? static_cast<double>(target_bytes - fixed_bytes) : 0;
  auto search  = [&]() {
    double budget       = start;
    double best         = -1;
    uint64_t best_total = 0, total = 0;
    for (int iter = 0; iter < 8; ++iter) {
      cut(budget);
      total = apply();
      if (total <= target_bytes) {
        if (budget > best) {
          best       = budget;
          best_total = total;
        }
        if (target_bytes - total < 16) break;
        budget += static_cast<double>(target_bytes - total);
      } else {
        budget -= static_cast<double>(total - target_bytes) * (iter < 4 ? 1 : 2);
        if (budget < 0) budget = 0;
      }
      if (best >= 0 && budget <= best) break;
    }
    if (best < 0) best = 0;
    if (total != best_total || budget != best) {
      cut(best);
      total = apply();
    }
    start = best;
    return total;
  };

  // Cut, transcode the code-blocks whose point lies above their coded set, and cut again
  // with their exact lengths.
  uint64_t total = 0;
  for (;;) {
    build();
    total = search();
    std::vector<uint32_t> moved;
    for (uint32_t i = 0; i < blocks.size(); ++i) {
      const rd_point &pt = point_of(i);
      if (pt.passes != 0 && pt.p != blocks[i]->rd->p_coded) moved.push_back(i);
    }
    if (moved.empty()) break;
    std::atomic<bool> failed{false};
    parallel_for(moved.size(), [&](size_t j) {
      if (!transcode(blocks[moved[j]], point_of(moved[j]).p)) failed.store(true);
    });
    if (failed.load()) {
      printf("ERROR: rate control could not transcode an HT code-block.\n");
      throw std::exception();
    }
    // Once more with the transcoded sets' exact lengths, and no more transcoding: the few
    // code-blocks that would still move stay with the points they have.
    with_model = false;
  }

  // Final points from the data held: the cut after SigProp ends in its own byte.
  for (uint32_t i = 0; i < blocks.size(); ++i) {
    const rd_point &pt   = point_of(i);
    j2k_codeblock *block = blocks[i];
    ht_rd_info &rd       = *block->rd;
    if (pt.passes == 0) {
      set_point(block, 0, 0, 0, 0, 0);
    } else {
      set_point(block, pt.p, pt.passes, rd.Lcup, rd.Lsp, rd.Lref);
      if (pt.passes == 2 && rd.Lsp) block->get_compressed_data()[rd.Lcup + rd.Lsp - 1] = rd.sp_tail;
    }
    rd.planes.reset();
  }
  total = fixed_bytes;
  for (uint32_t t = 0; t < num_tiles; ++t) {
    tiles[t].build_packet_headers();
    total += tiles[t].get_tile_part_size();
  }
  return total;
}
