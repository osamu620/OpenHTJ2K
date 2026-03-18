// Copyright (c) 2019 - 2021, Osamu Watanabe
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
#include <cstddef>
#include "open_htj2k_typedef.hpp"
#if defined(OPENHTJ2K_TRY_AVX2) && defined(__AVX2__)
  #define OPENHTJ2K_ENABLE_AVX2
#endif
#define SIMD_PADDING 32
// Number of columns per vertical-DWT strip for cache-friendly processing
#define DWT_VERT_STRIP 64

constexpr int16_t Acoeff_simd      = -19206;  // need to -1
constexpr int16_t Bcoeff_simd      = -3472;   // need to >> 1
constexpr int16_t Bcoeff_simd_avx2 = -13888;  // need to (out+4) >> 3
constexpr int16_t Ccoeff_simd      = 28931;
constexpr int16_t Dcoeff_simd      = 14533;

constexpr int32_t Acoeff = -25987;
constexpr int32_t Bcoeff = -3472;
constexpr int32_t Ccoeff = 28931;
constexpr int32_t Dcoeff = 29066;

constexpr int32_t Aoffset = 8192;
constexpr int32_t Boffset = 32767;
constexpr int32_t Coffset = 16384;
constexpr int32_t Doffset = 32767;

constexpr int32_t Ashift = 14;
constexpr int32_t Bshift = 16;
constexpr int32_t Cshift = 15;
constexpr int32_t Dshift = 16;

// define pointer to FDWT functions
typedef void (*fdwt_1d_filtr_func_fixed)(sprec_t *, const int32_t, const int32_t, const int32_t);
typedef void (*fdwt_ver_filtr_func_fixed)(sprec_t *, const int32_t, const int32_t, const int32_t,
                                          const int32_t, const int32_t stride);
// define pointer to IDWT functions
typedef void (*idwt_1d_filtd_func_fixed)(sprec_t *, const int32_t, const int32_t, const int32_t);
typedef void (*idwt_ver_filtd_func_fixed)(sprec_t *, const int32_t, const int32_t, const int32_t,
                                          const int32_t, const int32_t stride);

// symmetric extension
static inline int32_t PSEo(const int32_t i, const int32_t i0, const int32_t i1) {
  const int32_t tmp0    = 2 * (i1 - i0 - 1);
  const int32_t tmp1    = ((i - i0) < 0) ? i0 - i : i - i0;
  const int32_t mod_val = tmp1 % tmp0;
  const int32_t min_val = mod_val < tmp0 - mod_val ? mod_val : tmp0 - mod_val;
  return min_val;
}
template <class T>
static inline void dwt_1d_extr_fixed(T *extbuf, T *buf, const int32_t left, const int32_t right,
                                     const int32_t i0, const int32_t i1) {
  memcpy(extbuf + left, buf, sizeof(T) * static_cast<size_t>((i1 - i0)));
  for (int32_t i = 1; i <= left; ++i) {
    extbuf[left - i] = buf[PSEo(i0 - i, i0, i1)];
  }
  for (int32_t i = 1; i <= right; ++i) {
    extbuf[left + (i1 - i0) + i - 1] = buf[PSEo(i1 - i0 + i - 1 + i0, i0, i1)];
  }
}

// FDWT
#if defined(OPENHTJ2K_ENABLE_ARM_NEON)
void fdwt_1d_filtr_irrev97_fixed_neon(sprec_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
void fdwt_1d_filtr_rev53_fixed_neon(sprec_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
void fdwt_irrev_ver_sr_fixed_neon(sprec_t *in, int32_t u0, int32_t u1, int32_t v0, int32_t v1,
                                  int32_t stride);
void fdwt_rev_ver_sr_fixed_neon(sprec_t *in, int32_t u0, int32_t u1, int32_t v0, int32_t v1,
                                int32_t stride);

#elif defined(OPENHTJ2K_ENABLE_AVX2)
void fdwt_1d_filtr_irrev97_fixed_avx2(sprec_t *X, const int32_t left, const int32_t u_i0,
                                      const int32_t u_i1);
void fdwt_1d_filtr_rev53_fixed_avx2(sprec_t *X, const int32_t left, const int32_t u_i0, const int32_t u_i1);
void fdwt_irrev_ver_sr_fixed_avx2(sprec_t *in, const int32_t u0, const int32_t u1, const int32_t v0,
                                  const int32_t v1, const int32_t stride);
void fdwt_rev_ver_sr_fixed_avx2(sprec_t *in, const int32_t u0, const int32_t u1, const int32_t v0,
                                const int32_t v1, const int32_t stride);
#else
void fdwt_1d_filtr_irrev97_fixed(sprec_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
void fdwt_1d_filtr_rev53_fixed(sprec_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
void fdwt_irrev_ver_sr_fixed(sprec_t *in, int32_t u0, int32_t u1, int32_t v0, int32_t v1, int32_t stride);
void fdwt_rev_ver_sr_fixed(sprec_t *in, int32_t u0, int32_t u1, int32_t v0, int32_t v1, int32_t stride);
#endif

void fdwt_2d_sr_fixed(sprec_t *previousLL, sprec_t *LL, sprec_t *HL, sprec_t *LH, sprec_t *HH, int32_t u0,
                      int32_t u1, int32_t v0, int32_t v1, uint8_t transformation);

/**
 * @brief Stateful line-based 2-D forward DWT (push model).
 *
 * Feed image rows one at a time via push_row().  After all rows have been
 * pushed call produce_subbands() to retrieve the four output sub-bands.
 * Internally the vertical lifting steps are executed on narrow column strips
 * (DWT_VERT_STRIP wide) so the working set always fits in L2 cache.
 */
class fdwt_2d_state {
 public:
  fdwt_2d_state(int32_t u0, int32_t u1, int32_t v0, int32_t v1, uint8_t transformation);
  ~fdwt_2d_state();

  // Push one input row (length u1-u0).  Must be called exactly (v1-v0) times.
  void push_row(const sprec_t *row);

  // After all rows have been pushed, compute the 2-D DWT and write results
  // into the four sub-band buffers.
  void produce_subbands(sprec_t *LL, sprec_t *HL, sprec_t *LH, sprec_t *HH);

 private:
  int32_t u0_, u1_, v0_, v1_;
  uint8_t transformation_;
  int32_t width_;   // u1 - u0
  int32_t height_;  // v1 - v0
  int32_t stride_;  // round_up(width, 32)
  sprec_t *buf_;    // internal buffer: height_ × stride_ (actual rows only)
  int32_t n_pushed_;
};

// IDWT
#if defined(OPENHTJ2K_ENABLE_ARM_NEON)
void idwt_1d_filtr_rev53_fixed_neon(sprec_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
void idwt_1d_filtr_irrev97_fixed_neon(sprec_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
void idwt_irrev_ver_sr_fixed_neon(sprec_t *in, int32_t u0, int32_t u1, int32_t v0, int32_t v1,
                                  int32_t stride);
void idwt_rev_ver_sr_fixed_neon(sprec_t *in, int32_t u0, int32_t u1, int32_t v0, int32_t v1,
                                int32_t stride);
#elif defined(OPENHTJ2K_ENABLE_AVX2)
void idwt_1d_filtr_rev53_fixed_avx2(sprec_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
void idwt_1d_filtr_irrev97_fixed_avx2(sprec_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
void idwt_irrev_ver_sr_fixed_avx2(sprec_t *in, int32_t u0, int32_t u1, int32_t v0, int32_t v1,
                                  int32_t stride);
void idwt_rev_ver_sr_fixed_avx2(sprec_t *in, int32_t u0, int32_t u1, int32_t v0, int32_t v1,
                                int32_t stride);
#else
void idwt_1d_filtr_rev53_fixed(sprec_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
void idwt_1d_filtr_irrev97_fixed(sprec_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
void idwt_irrev_ver_sr_fixed(sprec_t *in, int32_t u0, int32_t u1, int32_t v0, int32_t v1, int32_t stride);
void idwt_rev_ver_sr_fixed(sprec_t *in, int32_t u0, int32_t u1, int32_t v0, int32_t v1, int32_t stride);
#endif
void idwt_2d_sr_fixed(sprec_t *nextLL, sprec_t *LL, sprec_t *HL, sprec_t *LH, sprec_t *HH, int32_t u0,
                      int32_t u1, int32_t v0, int32_t v1, uint8_t transformation,
                      uint8_t normalizing_upshift);

/**
 * @brief Stateful pull-based 2-D inverse DWT (pull model).
 *
 * Provide the decoded sub-bands via set_subbands(), then call pull_row() for
 * each output row index in order.  The inverse DWT is computed lazily on the
 * first pull_row() call using narrow column strips (DWT_VERT_STRIP wide) so
 * the working set fits in L2 cache.
 */
class idwt_2d_state {
 public:
  idwt_2d_state(int32_t u0, int32_t u1, int32_t v0, int32_t v1, uint8_t transformation,
                uint8_t normalizing_upshift);
  ~idwt_2d_state();

  // Provide pointers to the four decoded sub-band buffers.
  void set_subbands(sprec_t *LL, sprec_t *HL, sprec_t *LH, sprec_t *HH);

  // Return a pointer to the reconstructed output row row_idx (0-based).
  // The IDWT is computed lazily on the first call and cached.
  // Returns nullptr if row_idx is out of bounds [0, v1-v0).
  // The pointer is valid until the destructor is called.
  const sprec_t *pull_row(int32_t row_idx);

 private:
  int32_t u0_, u1_, v0_, v1_;
  uint8_t transformation_;
  uint8_t normalizing_upshift_;
  int32_t width_;   // u1 - u0
  int32_t height_;  // v1 - v0
  int32_t stride_;  // round_up(width, 32)
  sprec_t *buf_;    // output buffer: height_ × stride_
  bool computed_;
  sprec_t *LL_, *HL_, *LH_, *HH_;
};