// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.
//
// Licensed under the same BSD 3-Clause terms as the rest of OpenHTJ2K.

#pragma once

// int32 -> u8 planar sample shift for the shader-path decode callback.
//
// invoke_line_based_stream hands the receiver per-row planar int32_t
// samples; the shader-path code writes them into R8 planar textures, so
// each sample needs a min/max clamp and a (depth - 8)-bit right shift.
// At 4K 4:2:2 the three planes are ~16.6 MSamples/frame, and the scalar
// loop accounts for ~4.6 ms of the ~24 ms decode budget on a Ryzen 9
// 9950X — 9% of the perf profile.
//
// This header provides a single `shift_i32_plane_to_u8` function that is
// SIMD-dispatched at compile time.  When __AVX2__ is defined we process
// 8 samples per iteration with one loaded vector, two 32-bit clamps,
// one arithmetic right shift, and a two-step pack (packus int32->u16,
// packus u16->u8) into an 8-byte store.  Otherwise we fall through to
// a portable scalar loop.  Tail pixels (width % 8) always use the
// scalar loop.
//
// Behavior must match the scalar reference exactly for every input —
// tested in main_rtp_recv --smoke-test via plane_shift_smoke_test().

#include <cstddef>
#include <cstdint>

#if defined(__AVX2__)
#  include <immintrin.h>
#elif defined(__ARM_NEON)
#  include <arm_neon.h>
#endif

namespace open_htj2k::rtp_recv {

// Scalar reference — the semantics the SIMD path must reproduce exactly.
// `shift` is the per-sample right-shift amount (depth - 8; >= 0 for any
// supported bit depth).  `maxval` is (1 << depth) - 1 for clamping.
inline void shift_i32_plane_to_u8_scalar(const int32_t* in, uint8_t* out, uint32_t width,
                                         int32_t shift, int32_t maxval) {
  for (uint32_t x = 0; x < width; ++x) {
    int32_t v = in[x];
    if (v < 0) v = 0;
    if (v > maxval) v = maxval;
    out[x] = static_cast<uint8_t>(shift > 0 ? (v >> shift) : v);
  }
}

// Public entry point.  Dispatches to the AVX2 fast path when available
// and falls through to the scalar loop for the tail in either case.
inline void shift_i32_plane_to_u8(const int32_t* in, uint8_t* out, uint32_t width,
                                  int32_t shift, int32_t maxval) {
  uint32_t x = 0;

#if defined(__AVX2__)
  // AVX2: 8 int32 per iteration.  Clamp to [0, maxval], arithmetic-shift
  // right by `shift`, pack down to u8 via packus_epi32 → u16 → packus_epi16.
  //
  // packus_epi32(v, v) in AVX2 is lane-local: each 128-bit lane produces
  // four u16 values in its low 64 bits, and duplicates them in its high
  // 64 bits (since both operands are the same vector).  So lane 0 low 64
  // holds samples 0-3 and lane 1 low 64 holds samples 4-7.  We reassemble
  // those two 64-bit halves into a single __m128i with unpacklo_epi64,
  // then one more packus_epi16 collapses 16 u16 → 16 u8 (we only need
  // the low 8).
  //
  // `shift` can legally be 0 (depth == 8) — srai_epi32 by 0 is a 1-uop
  // no-op and we rely on that rather than branching inside the loop.
  const __m256i vzero   = _mm256_setzero_si256();
  const __m256i vmaxval = _mm256_set1_epi32(maxval);
  for (; x + 8 <= width; x += 8) {
    __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in + x));
    v         = _mm256_max_epi32(v, vzero);
    v         = _mm256_min_epi32(v, vmaxval);
    v         = _mm256_srai_epi32(v, shift);

    const __m256i p16   = _mm256_packus_epi32(v, v);
    const __m128i lane0 = _mm256_castsi256_si128(p16);
    const __m128i lane1 = _mm256_extracti128_si256(p16, 1);
    const __m128i merged16 = _mm_unpacklo_epi64(lane0, lane1);
    const __m128i packed8  = _mm_packus_epi16(merged16, merged16);

    _mm_storel_epi64(reinterpret_cast<__m128i*>(out + x), packed8);
  }
#elif defined(__ARM_NEON)
  // NEON: 16 int32 per iteration (four int32x4 groups).  Clamp to [0, maxval],
  // arithmetic right-shift, narrow int32→int16→uint8, store 16 bytes.
  // Widened to 16-wide to match Apple Clang's auto-vectorized 16-wide loop.
  const int32x4_t vzero   = vdupq_n_s32(0);
  const int32x4_t vmaxval = vdupq_n_s32(maxval);
  const int32x4_t vshift  = vdupq_n_s32(-shift);  // negative = right-shift
  for (; x + 16 <= width; x += 16) {
    int32x4_t a = vld1q_s32(in + x);
    int32x4_t b = vld1q_s32(in + x + 4);
    int32x4_t c = vld1q_s32(in + x + 8);
    int32x4_t d = vld1q_s32(in + x + 12);
    a = vminq_s32(vmaxq_s32(a, vzero), vmaxval);
    b = vminq_s32(vmaxq_s32(b, vzero), vmaxval);
    c = vminq_s32(vmaxq_s32(c, vzero), vmaxval);
    d = vminq_s32(vmaxq_s32(d, vzero), vmaxval);
    a = vshlq_s32(a, vshift);
    b = vshlq_s32(b, vshift);
    c = vshlq_s32(c, vshift);
    d = vshlq_s32(d, vshift);
    // Narrow: int32→int16 (saturating), then int16→uint8 (unsigned saturating).
    int16x8_t ab = vcombine_s16(vqmovn_s32(a), vqmovn_s32(b));
    int16x8_t cd = vcombine_s16(vqmovn_s32(c), vqmovn_s32(d));
    uint8x16_t packed = vcombine_u8(vqmovun_s16(ab), vqmovun_s16(cd));
    vst1q_u8(out + x, packed);
  }
  // 8-wide tail for residual 8..15 samples.
  for (; x + 8 <= width; x += 8) {
    int32x4_t a = vld1q_s32(in + x);
    int32x4_t b = vld1q_s32(in + x + 4);
    a = vminq_s32(vmaxq_s32(a, vzero), vmaxval);
    b = vminq_s32(vmaxq_s32(b, vzero), vmaxval);
    a = vshlq_s32(a, vshift);
    b = vshlq_s32(b, vshift);
    int16x8_t merged = vcombine_s16(vqmovn_s32(a), vqmovn_s32(b));
    vst1_u8(out + x, vqmovun_s16(merged));
  }
#endif

  // Tail (and the whole row on non-AVX2 builds).
  for (; x < width; ++x) {
    int32_t v = in[x];
    if (v < 0) v = 0;
    if (v > maxval) v = maxval;
    out[x] = static_cast<uint8_t>(shift > 0 ? (v >> shift) : v);
  }
}

// ── 16-bit variant ──────────────────────────────────────────────────────────
// Clamp int32 samples to [0, maxval] and pack into the top `depth` bits of a
// uint16_t, where `depth` is the source bit depth.  Used by the shader path
// when the source is >8 bit: the GL_R16 texture is unsigned-normalized, so
// writing the sample value directly produces a texture that reads back as
// (sample / 65535) in the shader; the fragment program then renormalizes
// via a uNormScale uniform set to (65535.0 / ((1<<depth)-1)) to restore the
// sample's original [0, 1] range before bias/scale/matrix math.  No right
// shift: the whole point of the 16-bit path is to preserve the LSBs the u8
// path truncated.
inline void clamp_i32_plane_to_u16_scalar(const int32_t* in, uint16_t* out, uint32_t width,
                                          int32_t maxval) {
  for (uint32_t x = 0; x < width; ++x) {
    int32_t v = in[x];
    if (v < 0) v = 0;
    if (v > maxval) v = maxval;
    out[x] = static_cast<uint16_t>(v);
  }
}

inline void clamp_i32_plane_to_u16(const int32_t* in, uint16_t* out, uint32_t width,
                                   int32_t maxval) {
  uint32_t x = 0;

#if defined(__AVX2__)
  // AVX2: 8 int32 per iteration.  Clamp to [0, maxval], pack down to 8 u16
  // via packus_epi32, reassemble the two lanes into a single __m128i, and
  // store the eight u16 values.
  //
  // packus_epi32 is lane-local: the low 64 bits of each 128-bit lane hold
  // four u16 values, and the high 64 bits duplicate them.  We
  // unpacklo_epi64 the two lanes to get eight contiguous u16s.
  //
  // No srai here: the u16 preserves the source's full dynamic range, and
  // the fragment shader renormalizes via uNormScale at sample time.
  const __m256i vzero   = _mm256_setzero_si256();
  const __m256i vmaxval = _mm256_set1_epi32(maxval);
  for (; x + 8 <= width; x += 8) {
    __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in + x));
    v         = _mm256_max_epi32(v, vzero);
    v         = _mm256_min_epi32(v, vmaxval);

    const __m256i p16      = _mm256_packus_epi32(v, v);
    const __m128i lane0    = _mm256_castsi256_si128(p16);
    const __m128i lane1    = _mm256_extracti128_si256(p16, 1);
    const __m128i merged16 = _mm_unpacklo_epi64(lane0, lane1);

    _mm_storeu_si128(reinterpret_cast<__m128i*>(out + x), merged16);
  }
#elif defined(__ARM_NEON)
  // NEON: 16 int32 per iteration.  Clamp to [0, maxval], narrow int32→uint16,
  // store 16 u16 values.
  const int32x4_t vzero   = vdupq_n_s32(0);
  const int32x4_t vmaxval = vdupq_n_s32(maxval);
  for (; x + 16 <= width; x += 16) {
    int32x4_t a = vld1q_s32(in + x);
    int32x4_t b = vld1q_s32(in + x + 4);
    int32x4_t c = vld1q_s32(in + x + 8);
    int32x4_t d = vld1q_s32(in + x + 12);
    a = vminq_s32(vmaxq_s32(a, vzero), vmaxval);
    b = vminq_s32(vmaxq_s32(b, vzero), vmaxval);
    c = vminq_s32(vmaxq_s32(c, vzero), vmaxval);
    d = vminq_s32(vmaxq_s32(d, vzero), vmaxval);
    uint16x8_t ab = vcombine_u16(vqmovun_s32(a), vqmovun_s32(b));
    uint16x8_t cd = vcombine_u16(vqmovun_s32(c), vqmovun_s32(d));
    vst1q_u16(out + x, ab);
    vst1q_u16(out + x + 8, cd);
  }
  // 8-wide tail.
  for (; x + 8 <= width; x += 8) {
    int32x4_t a = vld1q_s32(in + x);
    int32x4_t b = vld1q_s32(in + x + 4);
    a = vminq_s32(vmaxq_s32(a, vzero), vmaxval);
    b = vminq_s32(vmaxq_s32(b, vzero), vmaxval);
    vst1q_u16(out + x, vcombine_u16(vqmovun_s32(a), vqmovun_s32(b)));
  }
#endif

  // Tail (and the whole row on non-AVX2 builds).
  for (; x < width; ++x) {
    int32_t v = in[x];
    if (v < 0) v = 0;
    if (v > maxval) v = maxval;
    out[x] = static_cast<uint16_t>(v);
  }
}

// Byte-equality smoke test used from main_rtp_recv --smoke-test.  Drives
// the AVX2 and scalar paths with a hand-crafted input that exercises the
// clamp low/high edges, every residue of width % 8, and the shift == 0
// / shift > 0 branches.  Returns true on pass, false on any mismatch.
inline bool plane_shift_smoke_test() {
  constexpr uint32_t kMax = 96;  // exercises a few residues past a multiple of 8
  alignas(32) int32_t input[kMax];
  alignas(32) uint8_t  out_simd[kMax];
  alignas(32) uint8_t  out_ref[kMax];
  alignas(32) uint16_t out16_simd[kMax];
  alignas(32) uint16_t out16_ref[kMax];

  // Mix: negatives, zeros, small positives, near-max, over-max.
  for (uint32_t i = 0; i < kMax; ++i) {
    switch (i % 8) {
      case 0: input[i] = -1234;              break;
      case 1: input[i] = 0;                  break;
      case 2: input[i] = 1;                  break;
      case 3: input[i] = 127;                break;
      case 4: input[i] = 255;                break;
      case 5: input[i] = 256;                break;
      case 6: input[i] = 1023;               break;
      case 7: input[i] = (1 << 20) - 1;      break;
    }
  }

  struct Case {
    int32_t shift;
    int32_t maxval;
  };
  const Case cases[] = {
      {0, 255},      // 8-bit input, no shift
      {2, 1023},     // 10-bit -> 8-bit
      {4, 4095},     // 12-bit -> 8-bit
      {8, 65535},    // 16-bit -> 8-bit
  };

  for (const Case& c : cases) {
    for (uint32_t w = 1; w <= kMax; ++w) {
      // Reset outputs so a skipped write is visible as a mismatch.
      for (uint32_t i = 0; i < kMax; ++i) out_simd[i] = 0xAB;
      for (uint32_t i = 0; i < kMax; ++i) out_ref[i]  = 0xAB;

      shift_i32_plane_to_u8(input, out_simd, w, c.shift, c.maxval);
      shift_i32_plane_to_u8_scalar(input, out_ref, w, c.shift, c.maxval);

      for (uint32_t i = 0; i < w; ++i) {
        if (out_simd[i] != out_ref[i]) return false;
      }
      // Make sure we did not clobber beyond `w`.
      for (uint32_t i = w; i < kMax; ++i) {
        if (out_simd[i] != 0xAB) return false;
      }
    }
  }

  // Same matrix against the u16 clamp helper.
  const int32_t u16_cases[] = {255, 1023, 4095, 65535};
  for (int32_t maxval : u16_cases) {
    for (uint32_t w = 1; w <= kMax; ++w) {
      for (uint32_t i = 0; i < kMax; ++i) out16_simd[i] = 0xABCD;
      for (uint32_t i = 0; i < kMax; ++i) out16_ref[i]  = 0xABCD;

      clamp_i32_plane_to_u16(input, out16_simd, w, maxval);
      clamp_i32_plane_to_u16_scalar(input, out16_ref, w, maxval);

      for (uint32_t i = 0; i < w; ++i) {
        if (out16_simd[i] != out16_ref[i]) return false;
      }
      for (uint32_t i = w; i < kMax; ++i) {
        if (out16_simd[i] != 0xABCD) return false;
      }
    }
  }

  return true;
}

}  // namespace open_htj2k::rtp_recv
