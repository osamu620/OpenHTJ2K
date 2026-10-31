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

// Planar-input horizontal IDWT kernels for x86-64 builds without AVX2
// (OPENHTJ2K_ENABLE_SSE2, utils.hpp): the 4-lane WASM SIMD planar kernels
// (idwt_wasm.cpp) in SSE2.  They lift straight from the LP/HP subband planes, so a
// row needs no interleave pass.  Every multiply and add is rounded separately (no
// FMA), in the order of the scalar in-place kernels (idwt.cpp), so the output matches
// the scalar build bit for bit; the 5/3 kernel is integer arithmetic and exact.
// Dispatched from idwt_1d_row_from_planar, which enforces the contract documented
// with the WASM kernels: u0 even and N = u1/2 - u0/2 >= 12.

#include "utils.hpp"  // OPENHTJ2K_ENABLE_SSE2
#if defined(OPENHTJ2K_ENABLE_SSE2)
  #include <emmintrin.h>
  #include <vector>
  #include "dwt.hpp"

namespace {
// [a1, a2, a3, b0]
inline __m128 next1(__m128 a, __m128 b) {
  const __m128 m = _mm_move_ss(a, b);  // [b0, a1, a2, a3]
  return _mm_shuffle_ps(m, m, _MM_SHUFFLE(0, 3, 2, 1));
}
// [a3, b0, b1, b2]
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
// x - c * (a + b), rounded as the scalar expression is
inline __m128 lift(__m128 x, __m128 a, __m128 b, __m128 c) {
  return _mm_sub_ps(x, _mm_mul_ps(_mm_add_ps(a, b), c));
}
// [e0, o0, e1, o1, ...] for 4 + 4 samples
inline void store_interleaved(float *p, __m128 e, __m128 o) {
  _mm_storeu_ps(p, _mm_unpacklo_ps(e, o));
  _mm_storeu_ps(p + 4, _mm_unpackhi_ps(e, o));
}
inline void store_interleaved(int32_t *p, __m128i e, __m128i o) {
  _mm_storeu_si128(reinterpret_cast<__m128i *>(p), _mm_unpacklo_epi32(e, o));
  _mm_storeu_si128(reinterpret_cast<__m128i *>(p + 4), _mm_unpackhi_epi32(e, o));
}
}  // namespace

// 9/7 irreversible synthesis of one row from its planes.  Lifting stages, with
// E[j] = lp[j] and O[j] = hp[j] mirrored at the row ends (PSEo):
//   S1[j] = E[j] - fD (O[j-1] + O[j]),   S2[j] = O[j] - fC (S1[j] + S1[j+1]),
//   S3[j] = S1[j] - fB (S2[j-1] + S2[j]), out[2j] = S3[j],
//   out[2j+1] = S2[j] - fA (S3[j] + S3[j+1]).
// Iteration n of the steady state loads input block n (j = 4n..4n+3) and stores
// finished block n-2; the scalar warmup and drain cover the mirrored ends.
void idwt_1d_filtr_irrev97_planar_sse2(sprec_t *out, const sprec_t *lp, const sprec_t *hp, const int32_t u0,
                                       const int32_t u1) {
  const int32_t N = u1 / 2 - u0 / 2;
  auto E          = [&](int32_t j) -> float { return lp[PSEo(u0 + 2 * j, u0, u1) >> 1]; };
  auto O          = [&](int32_t j) -> float { return hp[PSEo(u0 + 2 * j + 1, u0, u1) >> 1]; };

  const __m128 vA = _mm_set1_ps(fA), vB = _mm_set1_ps(fB);
  const __m128 vC = _mm_set1_ps(fC), vD = _mm_set1_ps(fD);

  // Warmup: j = -1 scalars, then S1 of blocks 0 and 1, S2/S3 of block 0.
  const float om1  = O(-1);
  const float s1m1 = E(-1) - fD * (O(-2) + om1);  // S1[-1]
  __m128 O_b0      = _mm_loadu_ps(hp);
  __m128 S1_b0     = lift(_mm_loadu_ps(lp), prev1(_mm_set1_ps(om1), O_b0), O_b0, vD);
  const float s2m1 = om1 - fC * (s1m1 + _mm_cvtss_f32(S1_b0));  // S2[-1]
  out[-2]          = s1m1;                                      // final E[-1]
  out[-1]          = s2m1;                                      // final O[-1]

  __m128 O_b1  = _mm_loadu_ps(hp + 4);
  __m128 S1_b1 = lift(_mm_loadu_ps(lp + 4), prev1(O_b0, O_b1), O_b1, vD);
  __m128 S2_b0 = lift(O_b0, S1_b0, next1(S1_b0, S1_b1), vC);
  __m128 S3_b0 = lift(S1_b0, prev1(_mm_set1_ps(s2m1), S2_b0), S2_b0, vB);

  // Steady state.  The bound is 4n+3 <= N-1 because the planes have no PSE-filled
  // margins to read past; the drain covers the rest.
  __m128 O_nm1 = O_b1, S1_nm1 = S1_b1, S2_nm2 = S2_b0, S3_nm2 = S3_b0;
  int32_t n = 2;
  for (; 4 * n + 3 <= N - 1; ++n) {
    const __m128 O_n    = _mm_loadu_ps(hp + 4 * n);
    const __m128 S1_n   = lift(_mm_loadu_ps(lp + 4 * n), prev1(O_nm1, O_n), O_n, vD);
    const __m128 S2_nm1 = lift(O_nm1, S1_nm1, next1(S1_nm1, S1_n), vC);
    const __m128 S3_nm1 = lift(S1_nm1, prev1(S2_nm2, S2_nm1), S2_nm1, vB);
    store_interleaved(out + 8 * (n - 2), S3_nm2, lift(S2_nm2, S3_nm2, next1(S3_nm2, S3_nm1), vA));
    O_nm1  = O_n;
    S1_nm1 = S1_n;
    S2_nm2 = S2_nm1;
    S3_nm2 = S3_nm1;
  }

  // Drain: scalar finish for blocks n-2, n-1 (still in registers) and the ragged
  // tail.  Loop exit bounds m = 4n to [N-3, N], so with base = m-8 every stage index
  // j-base fits in 16.  s1t/s2t/s3t[i] hold S1/S2/S3[base+i]; s1t[0..3] are unused.
  {
    const int32_t m    = 4 * n;
    const int32_t base = m - 8;
    float s1t[16], s2t[16], s3t[16];
    _mm_storeu_ps(s1t + 4, S1_nm1);  // S1[m-4..m-1]
    _mm_storeu_ps(s2t, S2_nm2);      // S2[m-8..m-5]
    _mm_storeu_ps(s3t, S3_nm2);      // S3[m-8..m-5]
    for (int32_t j = m; j <= N + 1; ++j) s1t[j - base] = E(j) - fD * (O(j - 1) + O(j));
    for (int32_t j = m - 4; j <= N; ++j) s2t[j - base] = O(j) - fC * (s1t[j - base] + s1t[j - base + 1]);
    for (int32_t j = m - 4; j <= N; ++j)
      s3t[j - base] = s1t[j - base] - fB * (s2t[j - base - 1] + s2t[j - base]);
    // Final row: E[j] = S3[j], O[j] = S4[j]; then O[N] = S2[N], E[N+1] = S1[N+1].
    for (int32_t j = base; j <= N - 1; ++j) {
      out[2 * j]     = s3t[j - base];
      out[2 * j + 1] = s2t[j - base] - fA * (s3t[j - base] + s3t[j - base + 1]);
    }
    out[2 * N]       = s3t[N - base];
    out[2 * N + 1]   = s2t[N - base];
    out[2 * (N + 1)] = s1t[N + 1 - base];
  }
}

// Sub-range (JPIP column window) variant: lifts only the window [col_lo, col_hi)
// plus the 4-sample filter reach, from the planes.  Bit-identical to the full-width
// kernels for the columns the caller reads.
void idwt_1d_filtr_irrev97_planar_sr_sse2(sprec_t *out, const sprec_t *lp, const sprec_t *hp,
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
    s1v.resize(static_cast<size_t>(M));
    s2v.resize(static_cast<size_t>(M));
    s3v.resize(static_cast<size_t>(M));
  }
  float *const s1 = s1v.data() - base;
  float *const s2 = s2v.data() - base;
  float *const s3 = s3v.data() - base;

  if (J0 >= 2 && J1 <= N - 3) {
    const __m128 vA = _mm_set1_ps(fA), vB = _mm_set1_ps(fB);
    const __m128 vC = _mm_set1_ps(fC), vD = _mm_set1_ps(fD);
    int32_t j;
    // Pass 1: S1[j] = E[j] - fD*(O[j-1] + O[j])
    for (j = J0 - 1; j + 4 <= J1 + 3; j += 4)
      _mm_storeu_ps(s1 + j, lift(_mm_loadu_ps(lp + j), _mm_loadu_ps(hp + j - 1), _mm_loadu_ps(hp + j), vD));
    for (; j <= J1 + 2; ++j) s1[j] = lp[j] - fD * (hp[j - 1] + hp[j]);
    // Pass 2: S2[j] = O[j] - fC*(S1[j] + S1[j+1])
    for (j = J0 - 1; j + 4 <= J1 + 2; j += 4)
      _mm_storeu_ps(s2 + j, lift(_mm_loadu_ps(hp + j), _mm_loadu_ps(s1 + j), _mm_loadu_ps(s1 + j + 1), vC));
    for (; j <= J1 + 1; ++j) s2[j] = hp[j] - fC * (s1[j] + s1[j + 1]);
    // Pass 3: S3[j] = S1[j] - fB*(S2[j-1] + S2[j])
    for (j = J0; j + 4 <= J1 + 2; j += 4)
      _mm_storeu_ps(s3 + j, lift(_mm_loadu_ps(s1 + j), _mm_loadu_ps(s2 + j - 1), _mm_loadu_ps(s2 + j), vB));
    for (; j <= J1 + 1; ++j) s3[j] = s1[j] - fB * (s2[j - 1] + s2[j]);
    // Pass 4 + interleaved store: out[2j]=S3[j], out[2j+1]=S2[j]-fA*(S3[j]+S3[j+1])
    for (j = J0; j + 4 <= J1 + 1; j += 4) {
      const __m128 s3j = _mm_loadu_ps(s3 + j);
      store_interleaved(out + 2 * j, s3j, lift(_mm_loadu_ps(s2 + j), s3j, _mm_loadu_ps(s3 + j + 1), vA));
    }
    for (; j <= J1; ++j) {
      out[2 * j]     = s3[j];
      out[2 * j + 1] = s2[j] - fA * (s3[j] + s3[j + 1]);
    }
  } else {
    auto E = [&](int32_t j) -> float { return lp[PSEo(u0 + 2 * j, u0, u1) >> 1]; };
    auto O = [&](int32_t j) -> float { return hp[PSEo(u0 + 2 * j + 1, u0, u1) >> 1]; };
    for (int32_t j = J0 - 1; j <= J1 + 2; ++j) s1[j] = E(j) - fD * (O(j - 1) + O(j));
    for (int32_t j = J0 - 1; j <= J1 + 1; ++j) s2[j] = O(j) - fC * (s1[j] + s1[j + 1]);
    for (int32_t j = J0; j <= J1 + 1; ++j) s3[j] = s1[j] - fB * (s2[j - 1] + s2[j]);
    for (int32_t j = J0; j <= J1; ++j) {
      out[2 * j]     = s3[j];
      out[2 * j + 1] = s2[j] - fA * (s3[j] + s3[j + 1]);
    }
  }
}

// Reversible 5/3 synthesis of one row from its int32 planes:
//   S1[j] = E[j] - ((O[j-1] + O[j] + 2) >> 2)   for j in [0, N]   (undo LP update)
//   S2[j] = O[j] + ((S1[j] + S1[j+1]) >> 1)     for j in [0, N)   (undo HP predict)
// The loop bound j+4 <= N-1 keeps every plane read in range (the S1[j+4] lookahead
// reads lp[j+4] / hp[j+4]); the scalar tail mirrors.
void idwt_1d_filtr_rev53_planar_i32_sse2(int32_t *out, const int32_t *lp, const int32_t *hp,
                                         const int32_t u0, const int32_t u1) {
  const int32_t N = u1 / 2 - u0 / 2;
  auto E          = [&](int32_t j) -> int32_t { return lp[PSEo(u0 + 2 * j, u0, u1) >> 1]; };
  auto O          = [&](int32_t j) -> int32_t { return hp[PSEo(u0 + 2 * j + 1, u0, u1) >> 1]; };

  const __m128i vtwo = _mm_set1_epi32(2);
  int32_t j          = 0;
  int32_t o_carry    = O(-1);  // raw O[j-1] entering the current position
  for (; j + 4 <= N - 1; j += 4) {
    const __m128i Ob   = _mm_loadu_si128(reinterpret_cast<const __m128i *>(hp + j));
    const __m128i Ojm1 = prev1(_mm_set1_epi32(o_carry), Ob);  // O[j-1..j+2]
    const __m128i S1b  = _mm_sub_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(lp + j)),
                                       _mm_srai_epi32(_mm_add_epi32(_mm_add_epi32(Ojm1, Ob), vtwo), 2));
    // S1[j+4] from the planes (those positions are not lifted yet)
    const int32_t o3      = hp[j + 3];
    const int32_t s1_next = lp[j + 4] - ((o3 + hp[j + 4] + 2) >> 2);
    const __m128i S1n     = next1(S1b, _mm_set1_epi32(s1_next));  // S1[j+1..j+4]
    store_interleaved(out + 2 * j, S1b, _mm_add_epi32(Ob, _mm_srai_epi32(_mm_add_epi32(S1b, S1n), 1)));
    o_carry = o3;
  }
  // Scalar tail (at most 5 S1 values: the loop exits with N - j <= 4).
  {
    int32_t s1t[8] = {0};
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
#endif  // OPENHTJ2K_ENABLE_SSE2
