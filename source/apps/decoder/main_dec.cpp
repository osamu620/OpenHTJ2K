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

// open_htj2k_dec: A decoder implementation for JPEG 2000 Part 1 and 15
// (ITU-T Rec. 814 | ISO/IEC 15444-15 and ITU-T Rec. 814 | ISO/IEC 15444-15)
//
// (c) 2019 - 2021 Osamu Watanabe, Takushoku University, Vrije Universiteit Brussels

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _OPENMP
  #include <omp.h>
#endif
#if defined(__AVX2__)
  #include <immintrin.h>
#elif defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(_M_ARM64)
  #include <arm_neon.h>
#endif

#include "decoder.hpp"
#include "dec_utils.hpp"

// Vectorized int32 → big-endian byte packing for PGM/PGX output
namespace {

// bpp==2, big-endian: int32 + offset → {hi, lo} byte pairs
inline void pack_i32_to_be16(const int32_t *src, uint8_t *dst, uint32_t width, int32_t offset) {
#if defined(__AVX2__)
  const __m256i voff = _mm256_set1_epi32(offset);
  const __m256i bswap =
      _mm256_setr_epi8(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 1, 0, 3, 2, 5, 4,
                       7, 6, 9, 8, 11, 10, 13, 12, 15, 14);
  uint32_t n = 0;
  for (; n + 16 <= width; n += 16) {
    __m256i v0 = _mm256_add_epi32(_mm256_loadu_si256((const __m256i *)(src + n)), voff);
    __m256i v1 = _mm256_add_epi32(_mm256_loadu_si256((const __m256i *)(src + n + 8)), voff);
    __m256i packed = _mm256_packs_epi32(v0, v1);
    packed         = _mm256_permute4x64_epi64(packed, 0xD8);
    packed         = _mm256_shuffle_epi8(packed, bswap);
    _mm256_storeu_si256((__m256i *)(dst + n * 2), packed);
  }
  for (; n < width; ++n) {
    int32_t v            = src[n] + offset;
    dst[n * 2]           = static_cast<uint8_t>(v >> 8);
    dst[n * 2 + 1]       = static_cast<uint8_t>(v);
  }
#elif defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(_M_ARM64)
  const int32x4_t voff = vdupq_n_s32(offset);
  uint32_t n = 0;
  for (; n + 8 <= width; n += 8) {
    int32x4_t v0 = vaddq_s32(vld1q_s32(src + n), voff);
    int32x4_t v1 = vaddq_s32(vld1q_s32(src + n + 4), voff);
    int16x4_t lo = vmovn_s32(v0);
    int16x4_t hi = vmovn_s32(v1);
    int16x8_t combined = vcombine_s16(lo, hi);
    // Byte-swap for big-endian
    uint8x16_t bytes = vreinterpretq_u8_s16(combined);
    uint8x16_t swapped = vrev16q_u8(bytes);
    vst1q_u8(dst + n * 2, swapped);
  }
  for (; n < width; ++n) {
    int32_t v            = src[n] + offset;
    dst[n * 2]           = static_cast<uint8_t>(v >> 8);
    dst[n * 2 + 1]       = static_cast<uint8_t>(v);
  }
#else
  for (uint32_t n = 0; n < width; ++n) {
    int32_t v            = src[n] + offset;
    dst[n * 2]           = static_cast<uint8_t>(v >> 8);
    dst[n * 2 + 1]       = static_cast<uint8_t>(v);
  }
#endif
}

// bpp==2, little-endian (PGX): int32 → {lo, hi} byte pairs (no offset)
inline void pack_i32_to_le16(const int32_t *src, uint8_t *dst, uint32_t width) {
#if defined(__AVX2__)
  uint32_t n = 0;
  for (; n + 16 <= width; n += 16) {
    __m256i v0     = _mm256_loadu_si256((const __m256i *)(src + n));
    __m256i v1     = _mm256_loadu_si256((const __m256i *)(src + n + 8));
    __m256i packed = _mm256_packs_epi32(v0, v1);
    packed         = _mm256_permute4x64_epi64(packed, 0xD8);
    // Already little-endian on x86
    _mm256_storeu_si256((__m256i *)(dst + n * 2), packed);
  }
  for (; n < width; ++n) {
    int32_t v            = src[n];
    dst[n * 2]           = static_cast<uint8_t>(v);
    dst[n * 2 + 1]       = static_cast<uint8_t>(v >> 8);
  }
#elif defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(_M_ARM64)
  uint32_t n = 0;
  for (; n + 8 <= width; n += 8) {
    int32x4_t v0 = vld1q_s32(src + n);
    int32x4_t v1 = vld1q_s32(src + n + 4);
    int16x4_t lo = vmovn_s32(v0);
    int16x4_t hi = vmovn_s32(v1);
    int16x8_t combined = vcombine_s16(lo, hi);
    vst1q_s16((int16_t *)(dst + n * 2), combined);
  }
  for (; n < width; ++n) {
    int32_t v            = src[n];
    dst[n * 2]           = static_cast<uint8_t>(v);
    dst[n * 2 + 1]       = static_cast<uint8_t>(v >> 8);
  }
#else
  for (uint32_t n = 0; n < width; ++n) {
    int32_t v            = src[n];
    dst[n * 2]           = static_cast<uint8_t>(v);
    dst[n * 2 + 1]       = static_cast<uint8_t>(v >> 8);
  }
#endif
}

// bpp==1: int32 + offset → uint8
inline void pack_i32_to_u8(const int32_t *src, uint8_t *dst, uint32_t width, int32_t offset) {
#if defined(__AVX2__)
  const __m256i voff  = _mm256_set1_epi32(offset);
  const __m256i vmask = _mm256_set1_epi32(0xFF);
  // Permutation to fix the lane-crossing interleave of successive packs
  const __m256i fix = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
  uint32_t n = 0;
  for (; n + 32 <= width; n += 32) {
    // Mask to low byte so packus works correctly for both signed and unsigned data
    __m256i v0 = _mm256_and_si256(_mm256_add_epi32(_mm256_loadu_si256((const __m256i *)(src + n)), voff), vmask);
    __m256i v1 = _mm256_and_si256(_mm256_add_epi32(_mm256_loadu_si256((const __m256i *)(src + n + 8)), voff), vmask);
    __m256i v2 = _mm256_and_si256(_mm256_add_epi32(_mm256_loadu_si256((const __m256i *)(src + n + 16)), voff), vmask);
    __m256i v3 = _mm256_and_si256(_mm256_add_epi32(_mm256_loadu_si256((const __m256i *)(src + n + 24)), voff), vmask);
    __m256i p01 = _mm256_packus_epi32(v0, v1);
    __m256i p23 = _mm256_packus_epi32(v2, v3);
    __m256i p8  = _mm256_packus_epi16(p01, p23);
    p8          = _mm256_permutevar8x32_epi32(p8, fix);
    _mm256_storeu_si256((__m256i *)(dst + n), p8);
  }
  for (; n < width; ++n)
    dst[n] = static_cast<uint8_t>(src[n] + offset);
#elif defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(_M_ARM64)
  const int32x4_t voff = vdupq_n_s32(offset);
  uint32_t n = 0;
  for (; n + 16 <= width; n += 16) {
    int32x4_t v0 = vaddq_s32(vld1q_s32(src + n), voff);
    int32x4_t v1 = vaddq_s32(vld1q_s32(src + n + 4), voff);
    int32x4_t v2 = vaddq_s32(vld1q_s32(src + n + 8), voff);
    int32x4_t v3 = vaddq_s32(vld1q_s32(src + n + 12), voff);
    int16x4_t h0 = vmovn_s32(v0);
    int16x4_t h1 = vmovn_s32(v1);
    int16x4_t h2 = vmovn_s32(v2);
    int16x4_t h3 = vmovn_s32(v3);
    uint8x8_t b0 = vmovn_u16(vreinterpretq_u16_s16(vcombine_s16(h0, h1)));
    uint8x8_t b1 = vmovn_u16(vreinterpretq_u16_s16(vcombine_s16(h2, h3)));
    vst1_u8(dst + n, b0);
    vst1_u8(dst + n + 8, b1);
  }
  for (; n < width; ++n)
    dst[n] = static_cast<uint8_t>(src[n] + offset);
#else
  for (uint32_t n = 0; n < width; ++n)
    dst[n] = static_cast<uint8_t>(src[n] + offset);
#endif
}

// 3-component int32 → interleaved 8-bit PPM output (R,G,B per pixel).
// Clamps each sample to [0, max_val], then packs to uint8.
inline void ppm_interleave_8(const int32_t *sp0, const int32_t *sp1, const int32_t *sp2,
                             uint8_t *dp, uint32_t width, uint32_t bit_depth) {
#if defined(__SSE4_1__) || defined(__AVX2__)
  const __m128i max_val = _mm_set1_epi32((1 << bit_depth) - 1);
  const __m128i zero    = _mm_setzero_si128();
  // Shuffle: pack 4 pixels of [R,G,B,0] → 12 bytes of [R,G,B,R,G,B,...].
  // Each 32-bit word has R in byte 0, G in byte 1, B in byte 2, 0 in byte 3.
  // Output: first 12 of 16 bytes (3 pixels = 12 bytes, 4th pixel in bytes 12-14).
  const __m128i m0 = _mm_setr_epi8(0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, -1, -1, -1, -1);
  uint32_t n       = 0;
  for (; n + 16 <= width; n += 16, sp0 += 16, sp1 += 16, sp2 += 16, dp += 48) {
    __m128i t, u, v, w, a;
    // Pixels 0-3
    a = _mm_loadu_si128((const __m128i *)sp0);
    a = _mm_max_epi32(a, zero);
    t = _mm_min_epi32(a, max_val);
    a = _mm_loadu_si128((const __m128i *)sp1);
    a = _mm_max_epi32(a, zero);
    a = _mm_min_epi32(a, max_val);
    t = _mm_or_si128(t, _mm_slli_epi32(a, 8));
    a = _mm_loadu_si128((const __m128i *)sp2);
    a = _mm_max_epi32(a, zero);
    a = _mm_min_epi32(a, max_val);
    t = _mm_or_si128(t, _mm_slli_epi32(a, 16));
    t = _mm_shuffle_epi8(t, m0);  // 12 bytes of output
    // Pixels 4-7
    a = _mm_loadu_si128((const __m128i *)(sp0 + 4));
    a = _mm_max_epi32(a, zero);
    u = _mm_min_epi32(a, max_val);
    a = _mm_loadu_si128((const __m128i *)(sp1 + 4));
    a = _mm_max_epi32(a, zero);
    a = _mm_min_epi32(a, max_val);
    u = _mm_or_si128(u, _mm_slli_epi32(a, 8));
    a = _mm_loadu_si128((const __m128i *)(sp2 + 4));
    a = _mm_max_epi32(a, zero);
    a = _mm_min_epi32(a, max_val);
    u = _mm_or_si128(u, _mm_slli_epi32(a, 16));
    u = _mm_shuffle_epi8(u, m0);
    // Pixels 8-11
    a = _mm_loadu_si128((const __m128i *)(sp0 + 8));
    a = _mm_max_epi32(a, zero);
    v = _mm_min_epi32(a, max_val);
    a = _mm_loadu_si128((const __m128i *)(sp1 + 8));
    a = _mm_max_epi32(a, zero);
    a = _mm_min_epi32(a, max_val);
    v = _mm_or_si128(v, _mm_slli_epi32(a, 8));
    a = _mm_loadu_si128((const __m128i *)(sp2 + 8));
    a = _mm_max_epi32(a, zero);
    a = _mm_min_epi32(a, max_val);
    v = _mm_or_si128(v, _mm_slli_epi32(a, 16));
    v = _mm_shuffle_epi8(v, m0);
    // Pixels 12-15
    a = _mm_loadu_si128((const __m128i *)(sp0 + 12));
    a = _mm_max_epi32(a, zero);
    w = _mm_min_epi32(a, max_val);
    a = _mm_loadu_si128((const __m128i *)(sp1 + 12));
    a = _mm_max_epi32(a, zero);
    a = _mm_min_epi32(a, max_val);
    w = _mm_or_si128(w, _mm_slli_epi32(a, 8));
    a = _mm_loadu_si128((const __m128i *)(sp2 + 12));
    a = _mm_max_epi32(a, zero);
    a = _mm_min_epi32(a, max_val);
    w = _mm_or_si128(w, _mm_slli_epi32(a, 16));
    w = _mm_shuffle_epi8(w, m0);
    // Combine 4 groups of 12 bytes into 3 × 16-byte stores
    _mm_storeu_si128((__m128i *)dp, _mm_or_si128(t, _mm_bslli_si128(u, 12)));
    _mm_storeu_si128((__m128i *)(dp + 16),
                     _mm_or_si128(_mm_bsrli_si128(u, 4), _mm_bslli_si128(v, 8)));
    _mm_storeu_si128((__m128i *)(dp + 32),
                     _mm_or_si128(_mm_bsrli_si128(v, 8), _mm_bslli_si128(w, 4)));
  }
  int mv = (1 << bit_depth) - 1;
  for (; n < width; ++n, ++sp0, ++sp1, ++sp2) {
    int v;
    v     = *sp0;
    *dp++ = static_cast<uint8_t>(v < 0 ? 0 : v > mv ? mv : v);
    v     = *sp1;
    *dp++ = static_cast<uint8_t>(v < 0 ? 0 : v > mv ? mv : v);
    v     = *sp2;
    *dp++ = static_cast<uint8_t>(v < 0 ? 0 : v > mv ? mv : v);
  }
#elif defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(_M_ARM64)
  const int32x4_t vmax = vdupq_n_s32((1 << bit_depth) - 1);
  const int32x4_t zero = vdupq_n_s32(0);
  uint32_t n           = 0;
  for (; n + 8 <= width; n += 8, sp0 += 8, sp1 += 8, sp2 += 8, dp += 24) {
    // Process 8 pixels → 24 bytes
    int32x4_t r0 = vminq_s32(vmaxq_s32(vld1q_s32(sp0), zero), vmax);
    int32x4_t r1 = vminq_s32(vmaxq_s32(vld1q_s32(sp0 + 4), zero), vmax);
    int32x4_t g0 = vminq_s32(vmaxq_s32(vld1q_s32(sp1), zero), vmax);
    int32x4_t g1 = vminq_s32(vmaxq_s32(vld1q_s32(sp1 + 4), zero), vmax);
    int32x4_t b0 = vminq_s32(vmaxq_s32(vld1q_s32(sp2), zero), vmax);
    int32x4_t b1 = vminq_s32(vmaxq_s32(vld1q_s32(sp2 + 4), zero), vmax);
    uint16x4_t r16_lo = vqmovun_s32(r0), r16_hi = vqmovun_s32(r1);
    uint16x4_t g16_lo = vqmovun_s32(g0), g16_hi = vqmovun_s32(g1);
    uint16x4_t b16_lo = vqmovun_s32(b0), b16_hi = vqmovun_s32(b1);
    uint8x8_t r8 = vqmovn_u16(vcombine_u16(r16_lo, r16_hi));
    uint8x8_t g8 = vqmovn_u16(vcombine_u16(g16_lo, g16_hi));
    uint8x8_t b8 = vqmovn_u16(vcombine_u16(b16_lo, b16_hi));
    uint8x8x3_t rgb;
    rgb.val[0] = r8;
    rgb.val[1] = g8;
    rgb.val[2] = b8;
    vst3_u8(dp, rgb);
  }
  int mv = (1 << bit_depth) - 1;
  for (; n < width; ++n, ++sp0, ++sp1, ++sp2) {
    int v;
    v     = *sp0;
    *dp++ = static_cast<uint8_t>(v < 0 ? 0 : v > mv ? mv : v);
    v     = *sp1;
    *dp++ = static_cast<uint8_t>(v < 0 ? 0 : v > mv ? mv : v);
    v     = *sp2;
    *dp++ = static_cast<uint8_t>(v < 0 ? 0 : v > mv ? mv : v);
  }
#else
  int mv = (1 << bit_depth) - 1;
  for (uint32_t n = 0; n < width; ++n, ++sp0, ++sp1, ++sp2) {
    int v;
    v     = *sp0;
    *dp++ = static_cast<uint8_t>(v < 0 ? 0 : v > mv ? mv : v);
    v     = *sp1;
    *dp++ = static_cast<uint8_t>(v < 0 ? 0 : v > mv ? mv : v);
    v     = *sp2;
    *dp++ = static_cast<uint8_t>(v < 0 ? 0 : v > mv ? mv : v);
  }
#endif
}

// 3-component int32 → interleaved 16-bit big-endian PPM output.
// Clamps each sample to [0, max_val], byte-swaps to big-endian, and interleaves R,G,B.
inline void ppm_interleave_16be(const int32_t *sp0, const int32_t *sp1, const int32_t *sp2,
                                uint8_t *dp, uint32_t width, uint32_t bit_depth) {
#if defined(__SSE4_1__) || defined(__AVX2__)
  const __m128i max_val = _mm_set1_epi32((1 << bit_depth) - 1);
  const __m128i zero    = _mm_setzero_si128();
  uint16_t *p           = reinterpret_cast<uint16_t *>(dp);

  // Shuffle masks for 3-way 16-bit big-endian interleave (8 pixels → 48 bytes).
  // t = [r|g<<16] per 32-bit word. u = [b|r'<<16]. v = [g'|b'<<16].
  // Masks produce big-endian byte order and interleave R,G,B across 3 stores.
  const __m128i m0 = _mm_set_epi64x((long long)0x0A0B0809FFFF0607ULL, (long long)0x0405FFFF02030001ULL);
  const __m128i m1 = _mm_set_epi64x((long long)0xFFFFFFFF0405FFFFULL, (long long)0xFFFF0001FFFFFFFFULL);
  const __m128i m2 = _mm_set_epi64x((long long)0xFFFFFFFFFFFFFFFFULL, (long long)0xFFFF0E0F0C0DFFFFULL);
  const __m128i m3 = _mm_set_epi64x((long long)0x0607FFFFFFFF0203ULL, (long long)0x0C0DFFFFFFFF0809ULL);
  const __m128i m4 = _mm_set_epi64x((long long)0xFFFF02030001FFFFULL, (long long)0xFFFFFFFFFFFFFFFFULL);
  const __m128i m5 = _mm_set_epi64x((long long)0xFFFFFFFF0E0FFFFFULL, (long long)0xFFFF0A0BFFFFFFFFULL);
  const __m128i m6 = _mm_set_epi64x((long long)0x0E0F0C0DFFFF0A0BULL, (long long)0x0809FFFF06070405ULL);

  uint32_t n = 0;
  for (; n + 8 <= width; n += 8, sp0 += 8, sp1 += 8, sp2 += 8, p += 24) {
    __m128i a, t, u, v;
    // t = [r|g<<16] for pixels 0-3
    a = _mm_loadu_si128((const __m128i *)sp0);
    a = _mm_max_epi32(a, zero);
    t = _mm_min_epi32(a, max_val);
    a = _mm_loadu_si128((const __m128i *)sp1);
    a = _mm_max_epi32(a, zero);
    a = _mm_min_epi32(a, max_val);
    t = _mm_or_si128(t, _mm_slli_epi32(a, 16));
    // u = [b|r'<<16]: b for pixels 0-3, r for pixels 4-7
    a = _mm_loadu_si128((const __m128i *)sp2);
    a = _mm_max_epi32(a, zero);
    u = _mm_min_epi32(a, max_val);
    a = _mm_loadu_si128((const __m128i *)(sp0 + 4));
    a = _mm_max_epi32(a, zero);
    a = _mm_min_epi32(a, max_val);
    u = _mm_or_si128(u, _mm_slli_epi32(a, 16));
    // v = [g'|b'<<16]: g for pixels 4-7, b for pixels 4-7
    a = _mm_loadu_si128((const __m128i *)(sp1 + 4));
    a = _mm_max_epi32(a, zero);
    v = _mm_min_epi32(a, max_val);
    a = _mm_loadu_si128((const __m128i *)(sp2 + 4));
    a = _mm_max_epi32(a, zero);
    a = _mm_min_epi32(a, max_val);
    v = _mm_or_si128(v, _mm_slli_epi32(a, 16));
    // Shuffle + interleave → 3 × 16 bytes
    _mm_storeu_si128((__m128i *)p,
                     _mm_or_si128(_mm_shuffle_epi8(t, m0), _mm_shuffle_epi8(u, m1)));
    _mm_storeu_si128(
        (__m128i *)(p + 8),
        _mm_or_si128(_mm_shuffle_epi8(t, m2),
                     _mm_or_si128(_mm_shuffle_epi8(u, m3), _mm_shuffle_epi8(v, m4))));
    _mm_storeu_si128((__m128i *)(p + 16),
                     _mm_or_si128(_mm_shuffle_epi8(u, m5), _mm_shuffle_epi8(v, m6)));
  }
  int mv = (1 << bit_depth) - 1;
  for (; n < width; ++n, ++sp0, ++sp1, ++sp2) {
    int v;
    v    = *sp0;
    v    = v < 0 ? 0 : v > mv ? mv : v;
    *p++ = static_cast<uint16_t>((v >> 8) | (v << 8));
    v    = *sp1;
    v    = v < 0 ? 0 : v > mv ? mv : v;
    *p++ = static_cast<uint16_t>((v >> 8) | (v << 8));
    v    = *sp2;
    v    = v < 0 ? 0 : v > mv ? mv : v;
    *p++ = static_cast<uint16_t>((v >> 8) | (v << 8));
  }
#elif defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(_M_ARM64)
  const int32x4_t vmax = vdupq_n_s32((1 << bit_depth) - 1);
  const int32x4_t vzero = vdupq_n_s32(0);
  uint16_t *p           = reinterpret_cast<uint16_t *>(dp);
  uint32_t n            = 0;
  for (; n + 4 <= width; n += 4, sp0 += 4, sp1 += 4, sp2 += 4, p += 12) {
    int32x4_t rv = vminq_s32(vmaxq_s32(vld1q_s32(sp0), vzero), vmax);
    int32x4_t gv = vminq_s32(vmaxq_s32(vld1q_s32(sp1), vzero), vmax);
    int32x4_t bv = vminq_s32(vmaxq_s32(vld1q_s32(sp2), vzero), vmax);
    uint16x4_t r16 = vqmovun_s32(rv);
    uint16x4_t g16 = vqmovun_s32(gv);
    uint16x4_t b16 = vqmovun_s32(bv);
    // Byte-swap for big-endian
    uint8x8_t rs = vrev16_u8(vreinterpret_u8_u16(r16));
    uint8x8_t gs = vrev16_u8(vreinterpret_u8_u16(g16));
    uint8x8_t bs = vrev16_u8(vreinterpret_u8_u16(b16));
    uint16x4x3_t rgb;
    rgb.val[0] = vreinterpret_u16_u8(rs);
    rgb.val[1] = vreinterpret_u16_u8(gs);
    rgb.val[2] = vreinterpret_u16_u8(bs);
    vst3_u16(p, rgb);
  }
  int mv = (1 << bit_depth) - 1;
  for (; n < width; ++n, ++sp0, ++sp1, ++sp2) {
    int v;
    v    = *sp0;
    v    = v < 0 ? 0 : v > mv ? mv : v;
    *p++ = static_cast<uint16_t>((v >> 8) | (v << 8));
    v    = *sp1;
    v    = v < 0 ? 0 : v > mv ? mv : v;
    *p++ = static_cast<uint16_t>((v >> 8) | (v << 8));
    v    = *sp2;
    v    = v < 0 ? 0 : v > mv ? mv : v;
    *p++ = static_cast<uint16_t>((v >> 8) | (v << 8));
  }
#else
  int mv = (1 << bit_depth) - 1;
  for (uint32_t n = 0; n < width; ++n, ++sp0, ++sp1, ++sp2) {
    int v;
    v    = *sp0;
    v    = v < 0 ? 0 : v > mv ? mv : v;
    *p++ = static_cast<uint16_t>((v >> 8) | (v << 8));
    v    = *sp1;
    v    = v < 0 ? 0 : v > mv ? mv : v;
    *p++ = static_cast<uint16_t>((v >> 8) | (v << 8));
    v    = *sp2;
    v    = v < 0 ? 0 : v > mv ? mv : v;
    *p++ = static_cast<uint16_t>((v >> 8) | (v << 8));
  }
#endif
}

}  // namespace

void print_help(char *cmd) {
  printf("JPEG 2000 Part 1 and Part 15 decoder\n");
  printf("USAGE: %s [options]\n\n", cmd);
  printf("OPTIONS:\n");
  printf("-i: Input file. .j2k, .j2c, .jhc, and .jph are supported.\n");
  printf("-o: Output file. Supported formats are PPM, PGM, PGX and RAW.\n");
  printf("-reduce n: Number of DWT resolution reduction.\n");
  printf("-iter n: Repeat decoding n times (for benchmarking). Output is written once.\n");
  printf("-num_threads n: Number of threads (0 = auto).\n");
  printf("-batch: Use batch (full-image buffer) decode path instead of the default streaming path.\n");
  printf("-ycbcr bt601|bt709: [EXPERIMENTAL] Convert YCbCr to RGB (PPM output only).\n");
}

int main(int argc, char *argv[]) {
  // parse input args
  char *infile_name, *infile_ext_name;
  char *outfile_name, *outfile_ext_name;
  if (command_option_exists(argc, argv, "-h") || argc < 2) {
    print_help(argv[0]);
    exit(EXIT_SUCCESS);
  }
  if (nullptr == (infile_name = get_command_option(argc, argv, "-i"))) {
    printf("ERROR: Input file is missing. Use -i to specify input file.\n");
    exit(EXIT_FAILURE);
  }
  infile_ext_name = strrchr(infile_name, '.');
  if (infile_ext_name == nullptr) {
    printf("ERROR: Supported extensions are .j2k, .j2c, .jhc, .jph\n");
    exit(EXIT_FAILURE);
  }
  const bool is_jph = (strcmp(infile_ext_name, ".jph") == 0 || strcmp(infile_ext_name, ".JPH") == 0);
  if (!is_jph && strcmp(infile_ext_name, ".j2k") != 0 && strcmp(infile_ext_name, ".j2c") != 0
      && strcmp(infile_ext_name, ".jhc") != 0) {
    printf("ERROR: Supported extensions are .j2k, .j2c, .jhc, .jph\n");
    exit(EXIT_FAILURE);
  }
  if (nullptr == (outfile_name = get_command_option(argc, argv, "-o"))) {
    printf(
        "ERROR: Output files are missing. Use -o to specify output file "
        "names.\n");
    exit(EXIT_FAILURE);
  }
  outfile_ext_name        = strrchr(outfile_name, '.');
  bool discard_output     = (outfile_ext_name == nullptr);
  if (!discard_output
      && strcmp(outfile_ext_name, ".pgm") != 0 && strcmp(outfile_ext_name, ".ppm") != 0
      && strcmp(outfile_ext_name, ".raw") != 0 && strcmp(outfile_ext_name, ".pgx") != 0) {
    printf("ERROR: Unsupported output file type.\n");
    exit(EXIT_FAILURE);
  }
  char *tmp_param, *endptr;
  long tmp_val;
  uint8_t reduce_NL;
  if (nullptr == (tmp_param = get_command_option(argc, argv, "-reduce"))) {
    reduce_NL = 0;
  } else {
    tmp_val = strtol(tmp_param, &endptr, 10);
    if (tmp_val >= 0 && tmp_val <= 32 && tmp_param != endptr) {
      reduce_NL = static_cast<uint8_t>(tmp_val);
    } else {
      printf("ERROR: -reduce takes non-negative integer in the range from 0 to 32.\n");
      exit(EXIT_FAILURE);
    }
  }
  int32_t num_iterations;
  if (nullptr == (tmp_param = get_command_option(argc, argv, "-iter"))) {
    num_iterations = 1;
  } else {
    tmp_val = strtol(tmp_param, &endptr, 10);
    if (tmp_param == endptr) {
      printf("ERROR: -iter takes positive integer.\n");
      exit(EXIT_FAILURE);
    }
    if (tmp_val < 1 || tmp_val > INT32_MAX) {
      printf("ERROR: -iter takes positive integer ( < INT32_MAX).\n");
      exit(EXIT_FAILURE);
    }
    num_iterations = static_cast<int32_t>(tmp_val);
  }

  uint32_t num_threads;
  if (nullptr == (tmp_param = get_command_option(argc, argv, "-num_threads"))) {
    num_threads = 0;
  } else {
    tmp_val = strtol(tmp_param, &endptr, 10);
    if (tmp_param == endptr) {
      printf("ERROR: -num_threads takes non-negative integer.\n");
      exit(EXIT_FAILURE);
    }
    if (tmp_val < 0 || tmp_val > UINT32_MAX) {
      printf("ERROR: -num_threads takes non-negative integer ( < UINT32_MAX).\n");
      exit(EXIT_FAILURE);
    }
    //    num_iterations = static_cast<int32_t>(tmp_val);
    num_threads = static_cast<uint32_t>(tmp_val);  // strtoul(tmp_param, nullptr, 10);
  }
  // Reject any unrecognised flags.
  {
    static const char *const known[] = {
        "-h", "-i", "-o", "-reduce", "-iter", "-num_threads", "-batch", "-ycbcr", nullptr};
    for (int i = 1; i < argc; ++i) {
      if (argv[i][0] != '-') continue;
      bool recognised = false;
      for (int k = 0; known[k]; ++k) {
        if (strcmp(argv[i], known[k]) == 0) { recognised = true; break; }
      }
      if (!recognised) {
        printf("ERROR: unknown option %s\n", argv[i]);
        exit(EXIT_FAILURE);
      }
    }
  }

  // For JPH inputs: probe the colorspace from the library (no app-level box parsing).
  uint32_t detected_cs = 0;
  if (is_jph) {
    open_htj2k::openhtj2k_decoder probe(infile_name, 0, 0);
    detected_cs = probe.get_colorspace();
    if (detected_cs == open_htj2k::ENUMCS_SRGB)
      printf("INFO: JPH colorspace: sRGB\n");
    else if (detected_cs == open_htj2k::ENUMCS_GRAYSCALE)
      printf("INFO: JPH colorspace: Grayscale\n");
    else if (detected_cs == open_htj2k::ENUMCS_YCBCR)
      printf("INFO: JPH colorspace: YCbCr\n");
    else if (detected_cs != 0)
      printf("INFO: JPH colorspace: unknown EnumCS %u\n", detected_cs);
  }

  // Parse experimental -ycbcr flag (PPM output only).
  bool               do_ycbcr  = false;
  ycbcr_coefficients ycbcr_coeff{};
  // Auto-enable for JPH files with YCbCr colorspace (BT.601 by default).
  if (detected_cs == open_htj2k::ENUMCS_YCBCR && strcmp(outfile_ext_name, ".ppm") == 0) {
    do_ycbcr    = true;
    ycbcr_coeff = YCBCR_BT601;
  }
  if (nullptr != (tmp_param = get_command_option(argc, argv, "-ycbcr"))) {
    if (strcmp(tmp_param, "bt601") == 0) {
      do_ycbcr    = true;
      ycbcr_coeff = YCBCR_BT601;
    } else if (strcmp(tmp_param, "bt709") == 0) {
      do_ycbcr    = true;
      ycbcr_coeff = YCBCR_BT709;
    } else {
      printf("ERROR: -ycbcr takes 'bt601' or 'bt709'.\n");
      exit(EXIT_FAILURE);
    }
    if (strcmp(outfile_ext_name, ".ppm") != 0) {
      printf("WARNING: -ycbcr has no effect for non-PPM output.\n");
    }
  }
  if (do_ycbcr && strcmp(outfile_ext_name, ".ppm") == 0) {
    const bool is601 = (ycbcr_coeff.cr_to_r == YCBCR_BT601.cr_to_r);
    if (detected_cs == open_htj2k::ENUMCS_YCBCR && get_command_option(argc, argv, "-ycbcr") == nullptr)
      printf("INFO: YCbCr→RGB conversion auto-enabled (BT.601). Use -ycbcr bt709 to override.\n");
    else
      printf("INFO: YCbCr→RGB conversion enabled (%s).\n", is601 ? "BT.601" : "BT.709");
  }

  const bool use_batch = command_option_exists(argc, argv, "-batch");

  std::vector<uint32_t> img_width;
  std::vector<uint32_t> img_height;
  std::vector<uint8_t> img_depth;
  std::vector<bool> img_signed;

  auto start = std::chrono::high_resolution_clock::now();

  // Batch path: decode entire image into full-image buffers, then write.
  if (use_batch) {
    std::vector<int32_t *> buf;
    for (int32_t i = 0; i < num_iterations; ++i) {
      open_htj2k::openhtj2k_decoder decoder(infile_name, reduce_NL, num_threads);
      for (auto &p : buf) delete[] p;
      buf.clear();
      img_width.clear();
      img_height.clear();
      img_depth.clear();
      img_signed.clear();
      try {
        decoder.parse();
        decoder.invoke(buf, img_width, img_height, img_depth, img_signed);
      } catch (std::exception &exc) {
        printf("ERROR: %s\n", exc.what());
        return EXIT_FAILURE;
      }
    }
    auto duration       = std::chrono::high_resolution_clock::now() - start;
    auto num_components = static_cast<uint16_t>(img_depth.size());
    if (!discard_output) {
      if (strcmp(outfile_ext_name, ".ppm") == 0) {
        write_ppm(outfile_name, outfile_ext_name, buf, img_width, img_height, img_depth, img_signed,
                  do_ycbcr ? &ycbcr_coeff : nullptr);
      } else {
        write_components(outfile_name, outfile_ext_name, buf, img_width, img_height, img_depth,
                         img_signed);
      }
    }
    uint32_t total_samples = 0;
    for (uint16_t c = 0; c < num_components; ++c) {
      total_samples += img_width[c] * img_height[c];
      delete[] buf[c];
    }
    auto count = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
    printf("elapsed time %-15.3lf[ms]\n",
           static_cast<double>(count) / 1000.0 / static_cast<double>(num_iterations));
    printf("throughput %lf [Msamples/s]\n",
           total_samples * static_cast<double>(num_iterations) / static_cast<double>(count));
    printf("throughput %lf [usec/sample]\n",
           static_cast<double>(count) / static_cast<double>(num_iterations) / total_samples);
    return EXIT_SUCCESS;
  }

  const bool want_ppm = !discard_output && (strcmp(outfile_ext_name, ".ppm") == 0);
  const bool want_pgm = !discard_output && (strcmp(outfile_ext_name, ".pgm") == 0);
  const bool want_pgx = !discard_output && (strcmp(outfile_ext_name, ".pgx") == 0);

  // Output file handles (opened lazily on the last iteration's first row).
  std::vector<FILE *> fps;
  std::vector<uint8_t> row_buf;
  uint8_t bpp            = 0;
  int32_t pnm_offset     = 0;
  uint32_t total_samples = 0;
  // State for experimental YCbCr→RGB streaming conversion.
  int32_t ycbcr_cb_center = 0, ycbcr_cr_center = 0, ycbcr_maxval = 0;

  for (int32_t i = 0; i < num_iterations; ++i) {
    const bool is_last = (i == num_iterations - 1);

    open_htj2k::openhtj2k_decoder decoder(infile_name, reduce_NL, num_threads);
    try {
      decoder.parse();
    } catch (std::exception &exc) {
      printf("ERROR: %s\n", exc.what());
      return EXIT_FAILURE;
    }

    try {
      decoder.invoke_line_based_stream(
          [&](uint32_t y, int32_t *const *rows, uint16_t nc) {
            if (!is_last) return;  // warm-up iterations: decode but discard output
            if (discard_output) return;  // no extension → discard output (e.g. /dev/null)

            if (y == 0) {
              total_samples = 0;
              bpp           = static_cast<uint8_t>(ceil_int(static_cast<int32_t>(img_depth[0]), 8));
              pnm_offset    = (want_pgm && img_signed[0]) ? (1 << (img_depth[0] - 1)) : 0;
              if (do_ycbcr && nc >= 3) {
                ycbcr_maxval    = (1 << img_depth[0]) - 1;
                ycbcr_cb_center = img_signed[1] ? 0 : (1 << (img_depth[1] - 1));
                ycbcr_cr_center = img_signed[2] ? 0 : (1 << (img_depth[2] - 1));
              }
              if (want_ppm && nc == 3) {
                // Single PPM file: chroma is nearest-neighbour upsampled during row write.
                char fname[256], base[256];
                memcpy(base, outfile_name, static_cast<size_t>(outfile_ext_name - outfile_name));
                base[outfile_ext_name - outfile_name] = '\0';
                snprintf(fname, sizeof(fname), "%s%s", base, outfile_ext_name);
                FILE *fp = fopen(fname, "wb");
                if (fp == nullptr) {
                  printf("ERROR: Failed to open output file: %s\n", fname);
                  throw std::runtime_error("fopen failed");
                }
                fprintf(fp, "P6 %d %d %d\n", img_width[0], img_height[0],
                        (1 << img_depth[0]) - 1);
                fps.push_back(fp);
                fps.push_back(nullptr);
                fps.push_back(nullptr);
                row_buf.resize(static_cast<size_t>(img_width[0]) * 3 * bpp);
              } else {
                // One file per component (PGM, PGX, or RAW)
                fps.resize(nc, nullptr);
                for (uint16_t c = 0; c < nc; ++c) {
                  char fname[256], base[256];
                  memcpy(base, outfile_name,
                         static_cast<size_t>(outfile_ext_name - outfile_name));
                  base[outfile_ext_name - outfile_name] = '\0';
                  snprintf(fname, sizeof(fname), "%s_%02d%s", base, c, outfile_ext_name);
                  fps[c] = fopen(fname, "wb");
                  if (fps[c] == nullptr) {
                    printf("ERROR: Failed to open output file: %s\n", fname);
                    throw std::runtime_error("fopen failed");
                  }
                  if (want_pgm)
                    fprintf(fps[c], "P5 %d %d %d\n", img_width[c], img_height[c],
                            (1 << img_depth[c]) - 1);
                  if (want_pgx) {
                    char sign = img_signed[c] ? '-' : '+';
                    fprintf(fps[c], "PG LM %c %d %d %d\n", sign, img_depth[c], img_width[c],
                            img_height[c]);
                  }
                }
                uint32_t max_w = 0;
                for (uint16_t c = 0; c < nc; ++c)
                  max_w = std::max(max_w, img_width[c]);
                row_buf.resize(static_cast<size_t>(max_w) * bpp);
              }
              for (uint16_t c = 0; c < nc; ++c)
                total_samples += img_width[c] * img_height[c];
            }

            const uint16_t nc_all = static_cast<uint16_t>(fps.size());
            if (want_ppm && nc_all >= 3 && fps[0] != nullptr && fps[1] == nullptr) {
              // Interleaved PPM row
              uint8_t *out = row_buf.data();
              if (do_ycbcr) {
                // YCbCr→RGB conversion (experimental); chroma upsampled as needed.
                if (bpp == 1) {
                  for (uint32_t n = 0; n < img_width[0]; ++n) {
                    const uint32_t n1 = n * img_width[1] / img_width[0];
                    const uint32_t n2 = n * img_width[2] / img_width[0];
                    const int32_t Y   = rows[0][n];
                    const int32_t Cb  = rows[1][n1] - ycbcr_cb_center;
                    const int32_t Cr  = rows[2][n2] - ycbcr_cr_center;
                    int32_t r = Y + ((ycbcr_coeff.cr_to_r * Cr + 8192) >> 14);
                    int32_t g = Y - ((ycbcr_coeff.cb_to_g * Cb + ycbcr_coeff.cr_to_g * Cr + 8192) >> 14);
                    int32_t b = Y + ((ycbcr_coeff.cb_to_b * Cb + 8192) >> 14);
                    *out++ = static_cast<uint8_t>(r < 0 ? 0 : r > ycbcr_maxval ? ycbcr_maxval : r);
                    *out++ = static_cast<uint8_t>(g < 0 ? 0 : g > ycbcr_maxval ? ycbcr_maxval : g);
                    *out++ = static_cast<uint8_t>(b < 0 ? 0 : b > ycbcr_maxval ? ycbcr_maxval : b);
                  }
                } else {
                  for (uint32_t n = 0; n < img_width[0]; ++n) {
                    const uint32_t n1 = n * img_width[1] / img_width[0];
                    const uint32_t n2 = n * img_width[2] / img_width[0];
                    const int32_t Y   = rows[0][n];
                    const int32_t Cb  = rows[1][n1] - ycbcr_cb_center;
                    const int32_t Cr  = rows[2][n2] - ycbcr_cr_center;
                    int32_t r = Y + ((ycbcr_coeff.cr_to_r * Cr + 8192) >> 14);
                    int32_t g = Y - ((ycbcr_coeff.cb_to_g * Cb + ycbcr_coeff.cr_to_g * Cr + 8192) >> 14);
                    int32_t b = Y + ((ycbcr_coeff.cb_to_b * Cb + 8192) >> 14);
                    r         = r < 0 ? 0 : r > ycbcr_maxval ? ycbcr_maxval : r;
                    g         = g < 0 ? 0 : g > ycbcr_maxval ? ycbcr_maxval : g;
                    b         = b < 0 ? 0 : b > ycbcr_maxval ? ycbcr_maxval : b;
                    *out++    = static_cast<uint8_t>(r >> 8); *out++ = static_cast<uint8_t>(r);
                    *out++    = static_cast<uint8_t>(g >> 8); *out++ = static_cast<uint8_t>(g);
                    *out++    = static_cast<uint8_t>(b >> 8); *out++ = static_cast<uint8_t>(b);
                  }
                }
              } else if (bpp == 1) {
                if (img_width[0] == img_width[1] && img_width[0] == img_width[2]) {
                  ppm_interleave_8(rows[0], rows[1], rows[2], out, img_width[0], img_depth[0]);
                } else {
                  for (uint32_t n = 0; n < img_width[0]; ++n) {
                    const uint32_t n1 = n * img_width[1] / img_width[0];
                    const uint32_t n2 = n * img_width[2] / img_width[0];
                    *out++ = static_cast<uint8_t>(rows[0][n] + pnm_offset);
                    *out++ = static_cast<uint8_t>(rows[1][n1] + pnm_offset);
                    *out++ = static_cast<uint8_t>(rows[2][n2] + pnm_offset);
                  }
                }
              } else {
                if (img_width[0] == img_width[1] && img_width[0] == img_width[2]) {
                  ppm_interleave_16be(rows[0], rows[1], rows[2], out, img_width[0], img_depth[0]);
                } else {
                  for (uint32_t n = 0; n < img_width[0]; ++n) {
                    const uint32_t n1 = n * img_width[1] / img_width[0];
                    const uint32_t n2 = n * img_width[2] / img_width[0];
                    int32_t r = rows[0][n] + pnm_offset;
                    int32_t g = rows[1][n1] + pnm_offset;
                    int32_t b = rows[2][n2] + pnm_offset;
                    *out++    = static_cast<uint8_t>(r >> 8);
                    *out++    = static_cast<uint8_t>(r);
                    *out++    = static_cast<uint8_t>(g >> 8);
                    *out++    = static_cast<uint8_t>(g);
                    *out++    = static_cast<uint8_t>(b >> 8);
                    *out++    = static_cast<uint8_t>(b);
                  }
                }
              }
              fwrite(row_buf.data(), 1, row_buf.size(), fps[0]);
            } else {
              // Per-component files.
              // For vertically subsampled components (yr_c > 1, e.g. 4:2:0 chroma):
              // decode_line_based_stream holds the component row stable across yr_c luma
              // rows and only advances the ring buffer every yr_c luma rows.  So we write
              // once per yr_c luma rows (at y%yr_c==0) and use y/yr_c as the component-
              // space row boundary.  For non-subsampled components (yr_c==1) the
              // behaviour is identical to the original.
              // Use ceiling division to recover YRsiz: ceil(H0/Hc), which is exact for
              // any valid JPEG 2000 subsampling factor (including odd-height images where
              // integer division would give the wrong answer).
              for (uint16_t c = 0; c < nc_all; ++c) {
                if (fps[c] == nullptr) continue;
                const uint32_t h0 = img_height[0], hc = img_height[c];
                const uint32_t yr_c = (hc > 0 && hc < h0) ? (h0 + hc - 1) / hc : 1u;
                if (y % yr_c != 0) continue;         // duplicate luma row: skip
                if (y / yr_c >= hc) continue;         // past end of component
                uint8_t *out = row_buf.data();
                if (bpp == 1) {
                  pack_i32_to_u8(rows[c], out, img_width[c], pnm_offset);
                } else if (want_pgx) {
                  pack_i32_to_le16(rows[c], out, img_width[c]);
                } else {
                  pack_i32_to_be16(rows[c], out, img_width[c], pnm_offset);
                }
                fwrite(row_buf.data(), 1, static_cast<size_t>(img_width[c]) * bpp, fps[c]);
              }
            }
          },
          img_width, img_height, img_depth, img_signed);
    } catch (std::exception &exc) {
      printf("ERROR: %s\n", exc.what());
    }
  }

  for (FILE *fp : fps)
    if (fp) fclose(fp);

  auto duration = std::chrono::high_resolution_clock::now() - start;
  auto count    = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
  printf("elapsed time %-15.3lf[ms]\n",
         static_cast<double>(count) / 1000.0 / static_cast<double>(num_iterations));
  printf("throughput %lf [Msamples/s]\n",
         total_samples * static_cast<double>(num_iterations) / static_cast<double>(count));
  printf("throughput %lf [usec/sample]\n",
         static_cast<double>(count) / static_cast<double>(num_iterations) / total_samples);
  return EXIT_SUCCESS;
}
