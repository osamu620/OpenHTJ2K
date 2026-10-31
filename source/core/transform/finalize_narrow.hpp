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

// Fused float→int32 / float→uint8 / float→uint16 finalize kernels.
//
// Every decoder output path ends here.  An IDWT output row is float: Q(FRACBITS)
// units for an irreversible transform, integer-valued for a reversible one.  Each
// sample is rounded to the nearest output LSB (finalize_round in color.hpp: round
// half up of v * 2^-ds, one formula for every bit depth), offset by DC, clamped and
// written as int32, or narrowed to uint8 / uint16 for the planar API.  Fusing the
// narrowing into the finalize keeps the int32 intermediate out of memory; at 4K
// 4:2:2 that is ~132 MB of traffic per frame.
//
// `scale`:       2^-ds, ds = FRACBITS - bd (finalize_scale(ds)); 1 for lossless.
// `dc`:          DC offset (128 for 8-bit unsigned, 0 for signed).
// `maxval`:      clamp upper bound ((1<<bd)-1 for unsigned).
// `minval`:      clamp lower bound (0 for unsigned).
// `depth_shift`: additional right-shift for 8-bit packing (bd - 8; >= 0).

#include <cstdint>
#include "utils.hpp"
#include "color.hpp"

namespace open_htj2k {

inline int32_t finalize_sample(float v, float scale, int32_t dc, int32_t maxval, int32_t minval) {
  int32_t x = finalize_round(v, scale) + dc;
  if (x > maxval) x = maxval;
  if (x < minval) x = minval;
  return x;
}

// ── Scalar reference ────────────────────────────────────────────────────────
inline void finalize_f32_to_u8_scalar(const float *src, uint8_t *dst, uint32_t width, float scale,
                                      int32_t dc, int32_t maxval, int32_t minval, int32_t depth_shift) {
  for (uint32_t x = 0; x < width; ++x) {
    const int32_t v = finalize_sample(src[x], scale, dc, maxval, minval);
    dst[x]          = static_cast<uint8_t>(depth_shift > 0 ? (v >> depth_shift) : v);
  }
}

inline void finalize_f32_to_u16_scalar(const float *src, uint16_t *dst, uint32_t width, float scale,
                                       int32_t dc, int32_t maxval, int32_t minval) {
  for (uint32_t x = 0; x < width; ++x) {
    dst[x] = static_cast<uint16_t>(finalize_sample(src[x], scale, dc, maxval, minval));
  }
}

// ── Public entry points ─────────────────────────────────────────────────────

// float → finalize → int32.  The row kernel behind every int32 output path.
inline void finalize_f32_to_i32(const float *src, int32_t *dst, uint32_t width, float scale, int32_t dc,
                                int32_t maxval, int32_t minval) {
  uint32_t x = 0;
#if defined(OPENHTJ2K_TRY_AVX2) && defined(__AVX2__)
  {
    const __m256 vsc   = _mm256_set1_ps(scale);
    const __m256 vhalf = _mm256_set1_ps(0.5f);
    const __m256i vdc  = _mm256_set1_epi32(dc);
    const __m256i vmx  = _mm256_set1_epi32(maxval);
    const __m256i vmn  = _mm256_set1_epi32(minval);
    for (; x + 16 <= width; x += 16) {
      __m256i v0 = finalize_round_avx2(_mm256_loadu_ps(src + x), vsc, vhalf);
      __m256i v1 = finalize_round_avx2(_mm256_loadu_ps(src + x + 8), vsc, vhalf);
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst + x),
                          _mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(v0, vdc), vmn), vmx));
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst + x + 8),
                          _mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(v1, vdc), vmn), vmx));
    }
  }
#elif defined(OPENHTJ2K_ENABLE_ARM_NEON)
  {
    const float32x4_t vsc   = vdupq_n_f32(scale);
    const float32x4_t vhalf = vdupq_n_f32(0.5f);
    const int32x4_t vdc     = vdupq_n_s32(dc);
    const int32x4_t vmx     = vdupq_n_s32(maxval);
    const int32x4_t vmn     = vdupq_n_s32(minval);
    for (; x + 8 <= width; x += 8) {
      int32x4_t v0 = finalize_round_neon(vld1q_f32(src + x), vsc, vhalf);
      int32x4_t v1 = finalize_round_neon(vld1q_f32(src + x + 4), vsc, vhalf);
      vst1q_s32(dst + x, vmaxq_s32(vminq_s32(vaddq_s32(v0, vdc), vmx), vmn));
      vst1q_s32(dst + x + 4, vmaxq_s32(vminq_s32(vaddq_s32(v1, vdc), vmx), vmn));
    }
  }
#elif defined(OPENHTJ2K_ENABLE_WASM_SIMD)
  {
    const v128_t vsc   = wasm_f32x4_splat(scale);
    const v128_t vhalf = wasm_f32x4_splat(0.5f);
    const v128_t vdc   = wasm_i32x4_splat(dc);
    const v128_t vmx   = wasm_i32x4_splat(maxval);
    const v128_t vmn   = wasm_i32x4_splat(minval);
    for (; x + 8 <= width; x += 8) {
      v128_t v0 = finalize_round_wasm(wasm_v128_load(src + x), vsc, vhalf);
      v128_t v1 = finalize_round_wasm(wasm_v128_load(src + x + 4), vsc, vhalf);
      wasm_v128_store(dst + x, wasm_i32x4_min(wasm_i32x4_max(wasm_i32x4_add(v0, vdc), vmn), vmx));
      wasm_v128_store(dst + x + 4, wasm_i32x4_min(wasm_i32x4_max(wasm_i32x4_add(v1, vdc), vmn), vmx));
    }
  }
#elif defined(OPENHTJ2K_ENABLE_SSE2)
  {
    const __m128 vsc   = _mm_set1_ps(scale);
    const __m128 vhalf = _mm_set1_ps(0.5f);
    const __m128i vdc  = _mm_set1_epi32(dc);
    const __m128i vmx  = _mm_set1_epi32(maxval);
    const __m128i vmn  = _mm_set1_epi32(minval);
    for (; x + 8 <= width; x += 8) {
      __m128i v0 = finalize_round_sse2(_mm_loadu_ps(src + x), vsc, vhalf);
      __m128i v1 = finalize_round_sse2(_mm_loadu_ps(src + x + 4), vsc, vhalf);
      _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + x),
                       clamp_epi32_sse2(_mm_add_epi32(v0, vdc), vmn, vmx));
      _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + x + 4),
                       clamp_epi32_sse2(_mm_add_epi32(v1, vdc), vmn, vmx));
    }
  }
#endif
  for (; x < width; ++x) dst[x] = finalize_sample(src[x], scale, dc, maxval, minval);
}

// float → finalize → right-shift by depth_shift → uint8.
inline void finalize_f32_to_u8(const float *src, uint8_t *dst, uint32_t width, float scale, int32_t dc,
                               int32_t maxval, int32_t minval, int32_t depth_shift) {
  uint32_t x = 0;

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  const float32x4_t vsc   = vdupq_n_f32(scale);
  const float32x4_t vhalf = vdupq_n_f32(0.5f);
  const int32x4_t vdc     = vdupq_n_s32(dc);
  const int32x4_t vmx     = vdupq_n_s32(maxval);
  const int32x4_t vmn     = vdupq_n_s32(minval);
  const int32x4_t vdsh    = vdupq_n_s32(-depth_shift);  // negative = right-shift
  auto finalize4          = [&](const float *p) -> int32x4_t {
    int32x4_t v = finalize_round_neon(vld1q_f32(p), vsc, vhalf);
    v           = vmaxq_s32(vminq_s32(vaddq_s32(v, vdc), vmx), vmn);
    return vshlq_s32(v, vdsh);
  };
  // 16-wide main loop: 4x float32x4 → 1x uint8x16.
  for (; x + 16 <= width; x += 16) {
    int32x4_t a  = finalize4(src + x);
    int32x4_t b  = finalize4(src + x + 4);
    int32x4_t c  = finalize4(src + x + 8);
    int32x4_t d  = finalize4(src + x + 12);
    int16x8_t ab = vcombine_s16(vqmovn_s32(a), vqmovn_s32(b));
    int16x8_t cd = vcombine_s16(vqmovn_s32(c), vqmovn_s32(d));
    vst1q_u8(dst + x, vcombine_u8(vqmovun_s16(ab), vqmovun_s16(cd)));
  }
  // 8-wide tail.
  if (x + 8 <= width) {
    int32x4_t a      = finalize4(src + x);
    int32x4_t b      = finalize4(src + x + 4);
    int16x8_t merged = vcombine_s16(vqmovn_s32(a), vqmovn_s32(b));
    vst1_u8(dst + x, vqmovun_s16(merged));
    x += 8;
  }
#elif defined(OPENHTJ2K_TRY_AVX2) && defined(__AVX2__)
  const __m256 vsc   = _mm256_set1_ps(scale);
  const __m256 vhalf = _mm256_set1_ps(0.5f);
  const __m256i vdc  = _mm256_set1_epi32(dc);
  const __m256i vmx  = _mm256_set1_epi32(maxval);
  const __m256i vmn  = _mm256_set1_epi32(minval);
  const __m128i vdsh = _mm_cvtsi32_si128(depth_shift);
  for (; x + 16 <= width; x += 16) {
    __m256i v0 = finalize_round_avx2(_mm256_loadu_ps(src + x), vsc, vhalf);
    __m256i v1 = finalize_round_avx2(_mm256_loadu_ps(src + x + 8), vsc, vhalf);
    v0         = _mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(v0, vdc), vmn), vmx);
    v1         = _mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(v1, vdc), vmn), vmx);
    if (depth_shift > 0) {
      v0 = _mm256_sra_epi32(v0, vdsh);
      v1 = _mm256_sra_epi32(v1, vdsh);
    }
    // Pack int32→u16→u8: 16 samples → 16 bytes.  packus_epi32 is lane-local;
    // permute to get contiguous order.
    __m256i p16     = _mm256_permute4x64_epi64(_mm256_packus_epi32(v0, v1), 0xD8);  // [0,2,1,3]
    __m128i packed8 = _mm_packus_epi16(_mm256_castsi256_si128(p16), _mm256_extracti128_si256(p16, 1));
    _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + x), packed8);
  }
#endif

  // Scalar tail (and the whole row on non-SIMD builds).
  for (; x < width; ++x) {
    const int32_t v = finalize_sample(src[x], scale, dc, maxval, minval);
    dst[x]          = static_cast<uint8_t>(depth_shift > 0 ? (v >> depth_shift) : v);
  }
}

// float → finalize → uint16 (no depth_shift; preserves full dynamic range).
inline void finalize_f32_to_u16(const float *src, uint16_t *dst, uint32_t width, float scale, int32_t dc,
                                int32_t maxval, int32_t minval) {
  uint32_t x = 0;

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  const float32x4_t vsc   = vdupq_n_f32(scale);
  const float32x4_t vhalf = vdupq_n_f32(0.5f);
  const int32x4_t vdc     = vdupq_n_s32(dc);
  const int32x4_t vmx     = vdupq_n_s32(maxval);
  const int32x4_t vmn     = vdupq_n_s32(minval);
  auto finalize4          = [&](const float *p) -> int32x4_t {
    int32x4_t v = finalize_round_neon(vld1q_f32(p), vsc, vhalf);
    return vmaxq_s32(vminq_s32(vaddq_s32(v, vdc), vmx), vmn);
  };
  for (; x + 16 <= width; x += 16) {
    int32x4_t a = finalize4(src + x);
    int32x4_t b = finalize4(src + x + 4);
    int32x4_t c = finalize4(src + x + 8);
    int32x4_t d = finalize4(src + x + 12);
    vst1q_u16(dst + x, vcombine_u16(vqmovun_s32(a), vqmovun_s32(b)));
    vst1q_u16(dst + x + 8, vcombine_u16(vqmovun_s32(c), vqmovun_s32(d)));
  }
#endif

  // Scalar tail (and the whole row on non-SIMD builds).
  for (; x < width; ++x) {
    dst[x] = static_cast<uint16_t>(finalize_sample(src[x], scale, dc, maxval, minval));
  }
}

}  // namespace open_htj2k
