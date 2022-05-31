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

#define SIMD_PADDING 32

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
                                          const int32_t);
// define pointer to IDWT functions
typedef void (*idwt_1d_filtd_func_fixed)(sprec_t *, const int32_t, const int32_t, const int32_t);
typedef void (*idwt_ver_filtd_func_fixed)(sprec_t *, const int32_t, const int32_t, const int32_t,
                                          const int32_t);

// symmetric extension
static inline int32_t PSEo(const int32_t i, const int32_t i0, const int32_t i1) {
  const int32_t tmp0    = 2 * (i1 - i0 - 1);
  const int32_t tmp1    = ((i - i0) < 0) ? i0 - i : i - i0;
  const int32_t mod_val = tmp1 % tmp0;
  const int32_t min_val = mod_val < tmp0 - mod_val ? mod_val : tmp0 - mod_val;
  return i0 + min_val;
}
template <class T>
static inline void dwt_1d_extr_fixed(T *extbuf, T *buf, const int32_t left, const int32_t right,
                                     const int32_t i0, const int32_t i1) {
  memcpy(extbuf + left, buf, sizeof(T) * static_cast<size_t>((i1 - i0)));
  for (int32_t i = 1; i <= left; ++i) {
    extbuf[left - i] = buf[PSEo(i0 - i, i0, i1) - i0];
  }
  for (int32_t i = 1; i <= right; ++i) {
    extbuf[left + (i1 - i0) + i - 1] = buf[PSEo(i1 - i0 + i - 1 + i0, i0, i1) - i0];
  }
}

// FDWT
#if defined(OPENHTJ2K_ENABLE_ARM_NEON)
void fdwt_1d_filtr_irrev97_fixed_neon(sprec_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
void fdwt_1d_filtr_rev53_fixed_neon(sprec_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
static fdwt_1d_filtr_func_fixed fdwt_1d_filtr_fixed[2] = {fdwt_1d_filtr_irrev97_fixed_neon,
                                                          fdwt_1d_filtr_rev53_fixed_neon};
void fdwt_irrev_ver_sr_fixed_neon(sprec_t *in, int32_t u0, int32_t u1, int32_t v0, int32_t v1);
void fdwt_rev_ver_sr_fixed_neon(sprec_t *in, int32_t u0, int32_t u1, int32_t v0, int32_t v1);
static fdwt_ver_filtr_func_fixed fdwt_ver_sr_fixed[2] = {fdwt_irrev_ver_sr_fixed_neon,
                                                         fdwt_rev_ver_sr_fixed_neon};
#elif defined(OPENHTJ2K_TRY_AVX2) && defined(__AVX2__)
void fdwt_1d_filtr_irrev97_fixed_avx2(sprec_t *X, const int32_t left, const int32_t u_i0,
                                      const int32_t u_i1);
void fdwt_1d_filtr_rev53_fixed_avx2(sprec_t *X, const int32_t left, const int32_t u_i0, const int32_t u_i1);
static fdwt_1d_filtr_func_fixed fdwt_1d_filtr_fixed[2] = {fdwt_1d_filtr_irrev97_fixed_avx2,
                                                          fdwt_1d_filtr_rev53_fixed_avx2};
void fdwt_irrev_ver_sr_fixed_avx2(sprec_t *in, const int32_t u0, const int32_t u1, const int32_t v0,
                                  const int32_t v1);
void fdwt_rev_ver_sr_fixed_avx2(sprec_t *in, const int32_t u0, const int32_t u1, const int32_t v0,
                                const int32_t v1);
static fdwt_ver_filtr_func_fixed fdwt_ver_sr_fixed[2] = {fdwt_irrev_ver_sr_fixed_avx2,
                                                         fdwt_rev_ver_sr_fixed_avx2};
#else
void fdwt_1d_filtr_irrev97_fixed(sprec_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
void fdwt_1d_filtr_rev53_fixed(sprec_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
static fdwt_1d_filtr_func_fixed fdwt_1d_filtr_fixed[2] = {fdwt_1d_filtr_irrev97_fixed,
                                                          fdwt_1d_filtr_rev53_fixed};
void fdwt_irrev_ver_sr_fixed(sprec_t *in, int32_t u0, int32_t u1, int32_t v0, int32_t v1);
void fdwt_rev_ver_sr_fixed(sprec_t *in, int32_t u0, int32_t u1, int32_t v0, int32_t v1);
static fdwt_ver_filtr_func_fixed fdwt_ver_sr_fixed[2] = {fdwt_irrev_ver_sr_fixed, fdwt_rev_ver_sr_fixed};
#endif

void fdwt_2d_sr_fixed(sprec_t *previousLL, sprec_t *LL, sprec_t *HL, sprec_t *LH, sprec_t *HH, int32_t u0,
                      int32_t u1, int32_t v0, int32_t v1, uint8_t transformation);

// IDWT
#if defined(OPENHTJ2K_ENABLE_ARM_NEON)
void idwt_1d_filtr_rev53_fixed_neon(sprec_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
void idwt_1d_filtr_irrev97_fixed_neon(sprec_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
static idwt_1d_filtd_func_fixed idwt_1d_filtr_fixed[2] = {idwt_1d_filtr_irrev97_fixed_neon,
                                                          idwt_1d_filtr_rev53_fixed_neon};
void idwt_irrev_ver_sr_fixed_neon(sprec_t *in, int32_t u0, int32_t u1, int32_t v0, int32_t v1);
void idwt_rev_ver_sr_fixed_neon(sprec_t *in, int32_t u0, int32_t u1, int32_t v0, int32_t v1);
static idwt_ver_filtd_func_fixed idwt_ver_sr_fixed[2] = {idwt_irrev_ver_sr_fixed_neon,
                                                         idwt_rev_ver_sr_fixed_neon};
#elif defined(OPENHTJ2K_TRY_AVX22) && defined(__AVX2__)
void idwt_1d_filtr_rev53_fixed_avx2(sprec_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
void idwt_1d_filtr_irrev97_fixed_avx2(sprec_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
static idwt_1d_filtd_func_fixed idwt_1d_filtr_fixed[2] = {idwt_1d_filtr_irrev97_fixed_avx2,
                                                          idwt_1d_filtr_rev53_fixed_avx2};
void idwt_irrev_ver_sr_fixed_avx2(sprec_t *in, int32_t u0, int32_t u1, int32_t v0, int32_t v1);
void idwt_rev_ver_sr_fixed_avx2(sprec_t *in, int32_t u0, int32_t u1, int32_t v0, int32_t v1);
static idwt_ver_filtd_func_fixed idwt_ver_sr_fixed[2] = {idwt_irrev_ver_sr_fixed_avx2,
                                                         idwt_rev_ver_sr_fixed_avx2};
#else
void idwt_1d_filtr_rev53_fixed(sprec_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
void idwt_1d_filtr_irrev97_fixed(sprec_t *X, int32_t left, int32_t u_i0, int32_t u_i1);
static idwt_1d_filtd_func_fixed idwt_1d_filtr_fixed[2] = {idwt_1d_filtr_irrev97_fixed,
                                                          idwt_1d_filtr_rev53_fixed};
void idwt_irrev_ver_sr_fixed(sprec_t *in, int32_t u0, int32_t u1, int32_t v0, int32_t v1);
void idwt_rev_ver_sr_fixed(sprec_t *in, int32_t u0, int32_t u1, int32_t v0, int32_t v1);
static idwt_ver_filtd_func_fixed idwt_ver_sr_fixed[2] = {idwt_irrev_ver_sr_fixed, idwt_rev_ver_sr_fixed};
#endif
void idwt_2d_sr_fixed(sprec_t *nextLL, sprec_t *LL, sprec_t *HL, sprec_t *LH, sprec_t *HH, int32_t u0,
                      int32_t u1, int32_t v0, int32_t v1, uint8_t transformation,
                      uint8_t normalizing_upshift);