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

#include <cstdint>
#include <cstdlib>
#include "open_htj2k_typedef.hpp"

#define round_up(x, n) (((x) + (n) - 1) & (-n))
// #define round_down(x, n) ((x) & (-n))
#define round_down(x, n) ((x) - ((x) % (n)))
#define ceil_int(a, b) ((a) + ((b) - 1)) / (b)

// ─────────────────────────────────────────────────────────────────────────────
// DWT buffer border slack
//
// j2k_resolution and j2k_subband sample buffers are allocated with
// DWT_LEFT_SLACK floats of slack before the first row and DWT_RIGHT_SLACK
// floats of slack after the last row. The user-visible i_samples pointer is
// offset by DWT_LEFT_SLACK from the allocator base, so consumers continue to
// index as i_samples + y * stride + x as before — the stride is unchanged.
//
// This border slack lets the in-place horizontal DWT process the *first* and
// *last* rows of a tile without an external copy buffer (Xext/Yext): the
// first row's negative-index slack and the last row's past-the-end slack
// both land in valid memory inside the allocation. Interior rows already
// worked via save/restore in fdwt_1d_sr_inplace / idwt_1d_sr_inplace.
//
// DWT_LEFT_SLACK is 8 floats (32 bytes) so that adding it to a 32-byte-aligned
// base preserves 32-byte alignment of i_samples. The horizontal filter only
// needs MAX_PSE_LEFT = 4 (9/7), but rounding up to 8 avoids alignment
// surprises in the SIMD loads.
//
// DWT_RIGHT_SLACK matches SIMD_PADDING (= 32 floats) so that AVX-512 tail
// writes (up to 15 floats past width) and the right PSE prefix both fit.
// ─────────────────────────────────────────────────────────────────────────────
constexpr int32_t DWT_LEFT_SLACK  = 8;
constexpr int32_t DWT_RIGHT_SLACK = 32;

#if defined(__INTEL_LLVM_COMPILER)
  #define __INTEL_COMPILER
#endif

#if defined(OPENHTJ2K_ENABLE_ARM_NEON)
  // #include <arm_acle.h>
  #include <arm_neon.h>
  #if defined(_MSC_VER)
    #define openhtj2k_arm_prefetch(x) __prefetch((x))
    #define openhtj2k_arm_prefetch2(x, y) __prefetch2((x), (y))
  #else
    #define openhtj2k_arm_prefetch(x) __builtin_prefetch((x))
    #define openhtj2k_arm_prefetch2(x, y) __builtin_prefetch((x), (y))
  #endif
#elif defined(_MSC_VER) || defined(__MINGW64__)
  #include <intrin.h>
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
  #include <x86intrin.h>
#endif

#if defined(_M_IX86) || defined(_M_X64) || defined(__x86_64__) || defined(_M_X64) || defined(__i386__) \
    || defined(_M_IX86)
  #ifndef __clang__
    #ifndef __INTEL_COMPILER
      #if defined(__GNUC__) && (__GNUC__ < 10)
// _mm256_storeu2_m128i and _mm256_set_m128i were added to GCC's <avxintrin.h>
// in GCC 10 (October 2019).  Older GCCs (e.g. GCC 7.5.0 still found in some
// long-LTS distros and embedded toolchains) lack them entirely, which manifests
// in template-heavy headers as a two-phase-lookup error of the form
//   "there are no arguments to _mm256_set_m128i that depend on a template
//    parameter, so a declaration of _mm256_set_m128i must be available"
// Provide static-inline polyfills here so call sites can keep using the
// readable Intel-style names.
static inline void _mm256_storeu2_m128i(__m128i_u* __addr_hi, __m128i_u* __addr_lo, __m256i __a) {
  __m128i __v128;

  __v128 = _mm256_castsi256_si128(__a);
  _mm_storeu_si128(__addr_lo, __v128);
  __v128 = _mm256_extractf128_si256(__a, 1);
  _mm_storeu_si128(__addr_hi, __v128);
}

static inline __m256i _mm256_set_m128i(__m128i __H, __m128i __L) {
  return _mm256_insertf128_si256(_mm256_castsi128_si256(__L), __H, 1);
}
      #endif
    #endif
  #endif
#endif

template <class T>
static inline T find_max(T x0, T x1, T x2, T x3) {
  T v0 = ((x0 > x1) ? x0 : x1);
  T v1 = ((x2 > x3) ? x2 : x3);
  return (v0 > v1) ? v0 : v1;
}

#if (defined(OPENHTJ2K_TRY_AVX2) && defined(__AVX2__)) \
    || (defined(_MSC_VER) && defined(_M_X64))
// Horizontal max of 4 int32 lanes using SSE4.1 (implied by AVX2).
static inline int32_t hMax(__m128i v) {
  v = _mm_max_epi32(v, _mm_shuffle_epi32(v, _MM_SHUFFLE(2, 3, 0, 1)));
  v = _mm_max_epi32(v, _mm_shuffle_epi32(v, _MM_SHUFFLE(1, 0, 3, 2)));
  return _mm_cvtsi128_si32(v);
}
#endif

#ifdef _MSC_VER
#include <stdlib.h>
#define openhtj2k_bswap32(x) _byteswap_ulong(x)
#else
#define openhtj2k_bswap32(x) __builtin_bswap32(x)
#endif

// Portable count-trailing-zeros for uint32_t (input must be non-zero).
static inline uint32_t openhtj2k_ctz32(uint32_t x) {
#if defined(_MSC_VER)
  unsigned long idx;
  _BitScanForward(&idx, static_cast<unsigned long>(x));
  return static_cast<uint32_t>(idx);
#else
  return static_cast<uint32_t>(__builtin_ctz(x));
#endif
}

static inline size_t popcount32(uint32_t num) {
  size_t precision = 0;
#if defined(_MSC_VER) && !defined(_M_ARM64)
  precision = __popcnt(static_cast<uint32_t>(num));
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
  precision = static_cast<size_t>(_popcnt32(num));
#elif defined(OPENHTJ2K_ENABLE_ARM_NEON)
  uint32x2_t val = vld1_dup_u32(static_cast<const uint32_t*>(&num));
  uint8_t a      = vaddv_u8(vcnt_u8(vreinterpret_u8_u32(val)));
  precision      = a >> 1;
#else
  while (num != 0) {
    if (1 == (num & 1)) {
      precision++;
    }
    num >>= 1;
  }
#endif
  return precision;
}

static inline uint32_t int_log2(const uint32_t x) {
  uint32_t y;
#if defined(_MSC_VER)
  unsigned long tmp;
  _BitScanReverse(&tmp, x);
  y = tmp;
#else
  y = static_cast<uint32_t>(31 - __builtin_clz(x));
#endif
  return (x == 0) ? 0 : y;
}

static inline uint32_t count_leading_zeros(const uint32_t x) {
  uint32_t y;
#if defined(_MSC_VER) && !defined(_M_ARM64)
  y = __lzcnt(x);
#elif defined(OPENHTJ2K_TRY_AVX2) && defined(__AVX2__) && defined(_MSC_VER) && defined(_WIN32)
  y = _lzcnt_u32(x);
#elif defined(OPENHTJ2K_TRY_AVX2) && defined(__AVX2__) && defined(__MINGW32__)
  y = __builtin_ia32_lzcnt_u32(x);
#elif defined(__MINGW32__) || defined(__MINGW64__)
  y = __builtin_clz(x);
#elif defined(OPENHTJ2K_ENABLE_ARM_NEON) && !defined(_M_ARM64)
  y = static_cast<uint32_t>(__builtin_clz(x));
#elif defined(OPENHTJ2K_ENABLE_ARM_NEON) && defined(_M_ARM64)
  unsigned long tmp;
  _BitScanReverse(&tmp, x);
  y = 31 - tmp;
#else
  y = 31 - int_log2(x);
#endif
  return (x == 0) ? 32 : y;
}

// Large-buffer reuse pool (Linux/glibc and Apple platforms).
// Buffers >= THRESHOLD bytes freed via aligned_mem_free are cached
// (physical pages kept mapped) and returned on the next aligned_mem_alloc of a
// compatible size, eliminating repeated mmap/munmap page-fault cycles on
// successive decode calls with the same image geometry.
#if !defined(__INTEL_COMPILER) && !defined(_MSC_VER) \
    && !defined(__MINGW32__) && !defined(__MINGW64__)
  #if defined(__linux__)
    #include <malloc.h>          // malloc_usable_size
    #define OPENHTJ2K_POOL_SIZE_FN(p) malloc_usable_size(p)
    #define OPENHTJ2K_LARGE_POOL 1
  #elif defined(__APPLE__)
    #include <malloc/malloc.h>   // malloc_size
    #define OPENHTJ2K_POOL_SIZE_FN(p) malloc_size(p)
    #define OPENHTJ2K_LARGE_POOL 1
  #endif
#endif

#ifdef OPENHTJ2K_LARGE_POOL
struct AlignedLargePool {
  static constexpr size_t THRESHOLD = 16384;  // 16 KB: cache buffers >= this size
  static constexpr int    MAX_SLOTS = 128;    // max cached entries per thread
  struct Slot { void* ptr; size_t usable; };
  Slot   slots[MAX_SLOTS];
  int    count = 0;

  void* alloc(size_t bytes, size_t align) {
    if (bytes >= THRESHOLD) {
      // Best-fit: find smallest cached buffer with usable >= bytes AND alignment compatible.
      int    best_i = -1;
      size_t best_u = SIZE_MAX;
      for (int i = 0; i < count; ++i) {
        const uintptr_t addr = reinterpret_cast<uintptr_t>(slots[i].ptr);
        if (slots[i].usable >= bytes && slots[i].usable < best_u
            && (addr & (align - 1)) == 0) {
          best_i = i;
          best_u = slots[i].usable;
        }
      }
      if (best_i >= 0) {
        void* p        = slots[best_i].ptr;
        slots[best_i]  = slots[--count];
        return p;
      }
    }
    void* result;
    if (posix_memalign(&result, align, bytes)) result = nullptr;
    return result;
  }

  void release(void* ptr) {
    if (!ptr) return;
    size_t sz = OPENHTJ2K_POOL_SIZE_FN(ptr);
    if (sz >= THRESHOLD && count < MAX_SLOTS) {
      slots[count++] = {ptr, sz};
    } else {
      free(ptr);
    }
  }

  ~AlignedLargePool() {
    for (int i = 0; i < count; ++i) free(slots[i].ptr);
    count = 0;
  }
};
static thread_local AlignedLargePool tl_aligned_pool;
#endif  // OPENHTJ2K_LARGE_POOL

static inline void* aligned_mem_alloc(size_t size, size_t align) {
  void* result;
#if defined(__INTEL_COMPILER)
  result = _mm_malloc(size, align);
#elif defined(_MSC_VER)
  result = _aligned_malloc(size, align);
#elif defined(__MINGW32__) || defined(__MINGW64__)
  result = __mingw_aligned_malloc(size, align);
#elif defined(OPENHTJ2K_LARGE_POOL)
  result = tl_aligned_pool.alloc(size, align);
#else
  if (posix_memalign(&result, align, size)) {
    result = nullptr;
  }
#endif
  return result;
}

static inline void aligned_mem_free(void* ptr) {
#if defined(__INTEL_COMPILER)
  _mm_free(ptr);
#elif defined(_MSC_VER)
  _aligned_free(ptr);
#elif defined(__MINGW32__) || defined(__MINGW64__)
  __mingw_aligned_free(ptr);
#elif defined(OPENHTJ2K_LARGE_POOL)
  tl_aligned_pool.release(ptr);
#else
  free(ptr);
#endif
}
#if ((defined(_MSVC_LANG) && _MSVC_LANG > 201103L) || __cplusplus > 201103L)
  #define MAKE_UNIQUE std::make_unique
#else
  #define MAKE_UNIQUE open_htj2k::make_unique
#endif

#if ((defined(_MSVC_LANG) && _MSVC_LANG <= 201103L) || __cplusplus <= 201103L)
  #include <cstddef>
  #include <memory>
  #include <type_traits>
  #include <utility>
namespace open_htj2k {
template <class T>
struct _Unique_if {
  typedef std::unique_ptr<T> _Single_object;
};

template <class T>
struct _Unique_if<T[]> {
  typedef std::unique_ptr<T[]> _Unknown_bound;
};

template <class T, size_t N>
struct _Unique_if<T[N]> {
  typedef void _Known_bound;
};

template <class T, class... Args>
typename _Unique_if<T>::_Single_object make_unique(Args&&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

template <class T>
typename _Unique_if<T>::_Unknown_bound make_unique(size_t n) {
  typedef typename std::remove_extent<T>::type U;
  return std::unique_ptr<T>(new U[n]());
}

template <class T, class... Args>
typename _Unique_if<T>::_Known_bound make_unique(Args&&...) = delete;
}  // namespace open_htj2k
#endif
