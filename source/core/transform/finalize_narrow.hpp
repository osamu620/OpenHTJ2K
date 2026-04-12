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

#pragma once

// Fused float→uint8 / float→uint16 finalize+narrow kernels.
//
// These combine two stages that are normally separate:
//   Stage 1 (finalize): float → truncate to int32, shift/round, add DC, clamp
//   Stage 2 (narrow):   int32 → right-shift by (depth-8), pack to uint8
//
// By fusing them, the int32 intermediate never touches memory.  At 4K 4:2:2
// this eliminates ~132 MB of memory traffic per frame (two full-width int32
// read+write round-trips).  NEON inner loops process 16 samples per iteration.

#include <cstdint>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#  include <arm_neon.h>
#endif

#if defined(OPENHTJ2K_TRY_AVX2) && defined(__AVX2__)
#  include <immintrin.h>
#endif

namespace open_htj2k {

// ── Scalar reference ────────────────────────────────────────────────────────
// float → finalize (shift/round + DC + clamp) → right-shift by depth_shift → uint8.
// `downshift`: finalize downshift (FRACBITS - bd for irreversible, 0 for lossless).
// `rnd`:       rounding offset ((1<<downshift)>>1, or 0 when downshift <= 0).
// `dc`:        DC offset (128 for 8-bit unsigned, 0 for signed).
// `maxval`:    clamp upper bound ((1<<bd)-1 for unsigned).
// `minval`:    clamp lower bound (0 for unsigned).
// `depth_shift`: additional right-shift for 8-bit packing (bd - 8; >= 0).
inline void finalize_f32_to_u8_scalar(const float *src, uint8_t *dst, uint32_t width,
                                      int16_t downshift, int16_t rnd, int32_t dc,
                                      int32_t maxval, int32_t minval, int32_t depth_shift) {
  for (uint32_t x = 0; x < width; ++x) {
    int32_t v = static_cast<int32_t>(src[x]);
    if (downshift < 0)
      v = (v + rnd) << -downshift;
    else if (downshift > 0)
      v = (v + rnd) >> downshift;
    v += dc;
    if (v > maxval) v = maxval;
    if (v < minval) v = minval;
    dst[x] = static_cast<uint8_t>(depth_shift > 0 ? (v >> depth_shift) : v);
  }
}

// float → finalize → uint16 (no depth_shift; preserves full dynamic range).
inline void finalize_f32_to_u16_scalar(const float *src, uint16_t *dst, uint32_t width,
                                       int16_t downshift, int16_t rnd, int32_t dc,
                                       int32_t maxval, int32_t minval) {
  for (uint32_t x = 0; x < width; ++x) {
    int32_t v = static_cast<int32_t>(src[x]);
    if (downshift < 0)
      v = (v + rnd) << -downshift;
    else if (downshift > 0)
      v = (v + rnd) >> downshift;
    v += dc;
    if (v > maxval) v = maxval;
    if (v < minval) v = minval;
    dst[x] = static_cast<uint16_t>(v);
  }
}

// ── Public entry points ─────────────────────────────────────────────────────

inline void finalize_f32_to_u8(const float *src, uint8_t *dst, uint32_t width,
                                int16_t downshift, int16_t rnd, int32_t dc,
                                int32_t maxval, int32_t minval, int32_t depth_shift) {
  uint32_t x = 0;

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  const int32x4_t vdc  = vdupq_n_s32(dc);
  const int32x4_t vmx  = vdupq_n_s32(maxval);
  const int32x4_t vmn  = vdupq_n_s32(minval);
  const int32x4_t vdsh = vdupq_n_s32(-depth_shift);  // negative = right-shift

  // 16-wide main loop: 4x float32x4 → 1x uint8x16.
  if (downshift > 0) {
    const int32x4_t vfsh = vdupq_n_s32(-downshift);
    const int32x4_t vrnd = vdupq_n_s32(rnd);
    for (; x + 16 <= width; x += 16) {
      int32x4_t a = vshlq_s32(vaddq_s32(vcvtq_s32_f32(vld1q_f32(src + x)),      vrnd), vfsh);
      int32x4_t b = vshlq_s32(vaddq_s32(vcvtq_s32_f32(vld1q_f32(src + x + 4)),  vrnd), vfsh);
      int32x4_t c = vshlq_s32(vaddq_s32(vcvtq_s32_f32(vld1q_f32(src + x + 8)),  vrnd), vfsh);
      int32x4_t d = vshlq_s32(vaddq_s32(vcvtq_s32_f32(vld1q_f32(src + x + 12)), vrnd), vfsh);
      a = vshlq_s32(vmaxq_s32(vminq_s32(vaddq_s32(a, vdc), vmx), vmn), vdsh);
      b = vshlq_s32(vmaxq_s32(vminq_s32(vaddq_s32(b, vdc), vmx), vmn), vdsh);
      c = vshlq_s32(vmaxq_s32(vminq_s32(vaddq_s32(c, vdc), vmx), vmn), vdsh);
      d = vshlq_s32(vmaxq_s32(vminq_s32(vaddq_s32(d, vdc), vmx), vmn), vdsh);
      int16x8_t ab = vcombine_s16(vqmovn_s32(a), vqmovn_s32(b));
      int16x8_t cd = vcombine_s16(vqmovn_s32(c), vqmovn_s32(d));
      vst1q_u8(dst + x, vcombine_u8(vqmovun_s16(ab), vqmovun_s16(cd)));
    }
  } else if (downshift < 0) {
    const int32x4_t vfsh = vdupq_n_s32(-downshift);  // positive = left-shift
    for (; x + 16 <= width; x += 16) {
      int32x4_t a = vshlq_s32(vcvtq_s32_f32(vld1q_f32(src + x)),      vfsh);
      int32x4_t b = vshlq_s32(vcvtq_s32_f32(vld1q_f32(src + x + 4)),  vfsh);
      int32x4_t c = vshlq_s32(vcvtq_s32_f32(vld1q_f32(src + x + 8)),  vfsh);
      int32x4_t d = vshlq_s32(vcvtq_s32_f32(vld1q_f32(src + x + 12)), vfsh);
      a = vshlq_s32(vmaxq_s32(vminq_s32(vaddq_s32(a, vdc), vmx), vmn), vdsh);
      b = vshlq_s32(vmaxq_s32(vminq_s32(vaddq_s32(b, vdc), vmx), vmn), vdsh);
      c = vshlq_s32(vmaxq_s32(vminq_s32(vaddq_s32(c, vdc), vmx), vmn), vdsh);
      d = vshlq_s32(vmaxq_s32(vminq_s32(vaddq_s32(d, vdc), vmx), vmn), vdsh);
      int16x8_t ab = vcombine_s16(vqmovn_s32(a), vqmovn_s32(b));
      int16x8_t cd = vcombine_s16(vqmovn_s32(c), vqmovn_s32(d));
      vst1q_u8(dst + x, vcombine_u8(vqmovun_s16(ab), vqmovun_s16(cd)));
    }
  } else {
    // downshift == 0 (lossless 5/3)
    for (; x + 16 <= width; x += 16) {
      int32x4_t a = vcvtq_s32_f32(vld1q_f32(src + x));
      int32x4_t b = vcvtq_s32_f32(vld1q_f32(src + x + 4));
      int32x4_t c = vcvtq_s32_f32(vld1q_f32(src + x + 8));
      int32x4_t d = vcvtq_s32_f32(vld1q_f32(src + x + 12));
      a = vshlq_s32(vmaxq_s32(vminq_s32(vaddq_s32(a, vdc), vmx), vmn), vdsh);
      b = vshlq_s32(vmaxq_s32(vminq_s32(vaddq_s32(b, vdc), vmx), vmn), vdsh);
      c = vshlq_s32(vmaxq_s32(vminq_s32(vaddq_s32(c, vdc), vmx), vmn), vdsh);
      d = vshlq_s32(vmaxq_s32(vminq_s32(vaddq_s32(d, vdc), vmx), vmn), vdsh);
      int16x8_t ab = vcombine_s16(vqmovn_s32(a), vqmovn_s32(b));
      int16x8_t cd = vcombine_s16(vqmovn_s32(c), vqmovn_s32(d));
      vst1q_u8(dst + x, vcombine_u8(vqmovun_s16(ab), vqmovun_s16(cd)));
    }
  }
  // 8-wide tail.
  if (x + 8 <= width) {
    auto finalize4 = [&](const float *p) -> int32x4_t {
      int32x4_t v = vcvtq_s32_f32(vld1q_f32(p));
      if (downshift > 0)
        v = vshlq_s32(vaddq_s32(v, vdupq_n_s32(rnd)), vdupq_n_s32(-downshift));
      else if (downshift < 0)
        v = vshlq_s32(v, vdupq_n_s32(-downshift));
      v = vmaxq_s32(vminq_s32(vaddq_s32(v, vdc), vmx), vmn);
      return vshlq_s32(v, vdsh);
    };
    int32x4_t a = finalize4(src + x);
    int32x4_t b = finalize4(src + x + 4);
    int16x8_t merged = vcombine_s16(vqmovn_s32(a), vqmovn_s32(b));
    vst1_u8(dst + x, vqmovun_s16(merged));
    x += 8;
  }
#elif defined(OPENHTJ2K_TRY_AVX2) && defined(__AVX2__)
  const __m256i vdc  = _mm256_set1_epi32(dc);
  const __m256i vmx  = _mm256_set1_epi32(maxval);
  const __m256i vmn  = _mm256_set1_epi32(minval);

  if (downshift > 0) {
    const __m128i vfsh = _mm_cvtsi32_si128(downshift);
    const __m256i vrnd = _mm256_set1_epi32(rnd);
    for (; x + 16 <= width; x += 16) {
      __m256i v0 = _mm256_sra_epi32(_mm256_add_epi32(_mm256_cvttps_epi32(_mm256_loadu_ps(src + x)), vrnd), vfsh);
      __m256i v1 = _mm256_sra_epi32(_mm256_add_epi32(_mm256_cvttps_epi32(_mm256_loadu_ps(src + x + 8)), vrnd), vfsh);
      v0 = _mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(v0, vdc), vmn), vmx);
      v1 = _mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(v1, vdc), vmn), vmx);
      if (depth_shift > 0) {
        const __m128i vdsh = _mm_cvtsi32_si128(depth_shift);
        v0 = _mm256_sra_epi32(v0, vdsh);
        v1 = _mm256_sra_epi32(v1, vdsh);
      }
      // Pack int32→u16→u8: 16 samples → 16 bytes.
      __m256i p16_0 = _mm256_packus_epi32(v0, v1);
      // packus_epi32 is lane-local; permute to get contiguous order.
      p16_0 = _mm256_permute4x64_epi64(p16_0, 0xD8);  // [0,2,1,3]
      __m128i lo128 = _mm256_castsi256_si128(p16_0);
      __m128i packed8 = _mm_packus_epi16(lo128, _mm256_extracti128_si256(p16_0, 1));
      _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + x), packed8);
    }
  } else if (downshift == 0) {
    for (; x + 16 <= width; x += 16) {
      __m256i v0 = _mm256_cvttps_epi32(_mm256_loadu_ps(src + x));
      __m256i v1 = _mm256_cvttps_epi32(_mm256_loadu_ps(src + x + 8));
      v0 = _mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(v0, vdc), vmn), vmx);
      v1 = _mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(v1, vdc), vmn), vmx);
      if (depth_shift > 0) {
        const __m128i vdsh = _mm_cvtsi32_si128(depth_shift);
        v0 = _mm256_sra_epi32(v0, vdsh);
        v1 = _mm256_sra_epi32(v1, vdsh);
      }
      __m256i p16_0 = _mm256_packus_epi32(v0, v1);
      p16_0 = _mm256_permute4x64_epi64(p16_0, 0xD8);
      __m128i lo128 = _mm256_castsi256_si128(p16_0);
      __m128i packed8 = _mm_packus_epi16(lo128, _mm256_extracti128_si256(p16_0, 1));
      _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + x), packed8);
    }
  }
  // AVX2: no ds<0 fast path; falls through to scalar (rare case).
#endif

  // Scalar tail (and the whole row on non-SIMD builds).
  for (; x < width; ++x) {
    int32_t v = static_cast<int32_t>(src[x]);
    if (downshift < 0)
      v = (v + rnd) << -downshift;
    else if (downshift > 0)
      v = (v + rnd) >> downshift;
    v += dc;
    if (v > maxval) v = maxval;
    if (v < minval) v = minval;
    dst[x] = static_cast<uint8_t>(depth_shift > 0 ? (v >> depth_shift) : v);
  }
}

inline void finalize_f32_to_u16(const float *src, uint16_t *dst, uint32_t width,
                                 int16_t downshift, int16_t rnd, int32_t dc,
                                 int32_t maxval, int32_t minval) {
  uint32_t x = 0;

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  const int32x4_t vdc = vdupq_n_s32(dc);
  const int32x4_t vmx = vdupq_n_s32(maxval);
  const int32x4_t vmn = vdupq_n_s32(minval);

  if (downshift > 0) {
    const int32x4_t vfsh = vdupq_n_s32(-downshift);
    const int32x4_t vrnd = vdupq_n_s32(rnd);
    for (; x + 16 <= width; x += 16) {
      int32x4_t a = vshlq_s32(vaddq_s32(vcvtq_s32_f32(vld1q_f32(src + x)),      vrnd), vfsh);
      int32x4_t b = vshlq_s32(vaddq_s32(vcvtq_s32_f32(vld1q_f32(src + x + 4)),  vrnd), vfsh);
      int32x4_t c = vshlq_s32(vaddq_s32(vcvtq_s32_f32(vld1q_f32(src + x + 8)),  vrnd), vfsh);
      int32x4_t d = vshlq_s32(vaddq_s32(vcvtq_s32_f32(vld1q_f32(src + x + 12)), vrnd), vfsh);
      a = vmaxq_s32(vminq_s32(vaddq_s32(a, vdc), vmx), vmn);
      b = vmaxq_s32(vminq_s32(vaddq_s32(b, vdc), vmx), vmn);
      c = vmaxq_s32(vminq_s32(vaddq_s32(c, vdc), vmx), vmn);
      d = vmaxq_s32(vminq_s32(vaddq_s32(d, vdc), vmx), vmn);
      vst1q_u16(dst + x,     vcombine_u16(vqmovun_s32(a), vqmovun_s32(b)));
      vst1q_u16(dst + x + 8, vcombine_u16(vqmovun_s32(c), vqmovun_s32(d)));
    }
  } else if (downshift < 0) {
    const int32x4_t vfsh = vdupq_n_s32(-downshift);
    for (; x + 16 <= width; x += 16) {
      int32x4_t a = vshlq_s32(vcvtq_s32_f32(vld1q_f32(src + x)),      vfsh);
      int32x4_t b = vshlq_s32(vcvtq_s32_f32(vld1q_f32(src + x + 4)),  vfsh);
      int32x4_t c = vshlq_s32(vcvtq_s32_f32(vld1q_f32(src + x + 8)),  vfsh);
      int32x4_t d = vshlq_s32(vcvtq_s32_f32(vld1q_f32(src + x + 12)), vfsh);
      a = vmaxq_s32(vminq_s32(vaddq_s32(a, vdc), vmx), vmn);
      b = vmaxq_s32(vminq_s32(vaddq_s32(b, vdc), vmx), vmn);
      c = vmaxq_s32(vminq_s32(vaddq_s32(c, vdc), vmx), vmn);
      d = vmaxq_s32(vminq_s32(vaddq_s32(d, vdc), vmx), vmn);
      vst1q_u16(dst + x,     vcombine_u16(vqmovun_s32(a), vqmovun_s32(b)));
      vst1q_u16(dst + x + 8, vcombine_u16(vqmovun_s32(c), vqmovun_s32(d)));
    }
  } else {
    for (; x + 16 <= width; x += 16) {
      int32x4_t a = vcvtq_s32_f32(vld1q_f32(src + x));
      int32x4_t b = vcvtq_s32_f32(vld1q_f32(src + x + 4));
      int32x4_t c = vcvtq_s32_f32(vld1q_f32(src + x + 8));
      int32x4_t d = vcvtq_s32_f32(vld1q_f32(src + x + 12));
      a = vmaxq_s32(vminq_s32(vaddq_s32(a, vdc), vmx), vmn);
      b = vmaxq_s32(vminq_s32(vaddq_s32(b, vdc), vmx), vmn);
      c = vmaxq_s32(vminq_s32(vaddq_s32(c, vdc), vmx), vmn);
      d = vmaxq_s32(vminq_s32(vaddq_s32(d, vdc), vmx), vmn);
      vst1q_u16(dst + x,     vcombine_u16(vqmovun_s32(a), vqmovun_s32(b)));
      vst1q_u16(dst + x + 8, vcombine_u16(vqmovun_s32(c), vqmovun_s32(d)));
    }
  }
#endif

  // Scalar tail (and the whole row on non-SIMD builds).
  for (; x < width; ++x) {
    int32_t v = static_cast<int32_t>(src[x]);
    if (downshift < 0)
      v = (v + rnd) << -downshift;
    else if (downshift > 0)
      v = (v + rnd) >> downshift;
    v += dc;
    if (v > maxval) v = maxval;
    if (v < minval) v = minval;
    dst[x] = static_cast<uint16_t>(v);
  }
}

}  // namespace open_htj2k
