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

#if defined(OPENHTJ2K_ENABLE_WASM_SIMD)
  #include <wasm_simd128.h>
  #include "dwt.hpp"
  #include "utils.hpp"

/********************************************************************************
 * WASM-SIMD helpers: emulate NEON vld2q_f32 / vst2q_f32
 * (identical to fdwt_wasm.cpp — kept local to avoid cross-TU linkage)
 *******************************************************************************/
struct f32x4x2 {
  v128_t val[2];
};

static inline f32x4x2 vld2q(const float *ptr) {
  v128_t lo   = wasm_v128_load(ptr);
  v128_t hi   = wasm_v128_load(ptr + 4);
  f32x4x2 r;
  r.val[0] = wasm_i32x4_shuffle(lo, hi, 0, 2, 4, 6);
  r.val[1] = wasm_i32x4_shuffle(lo, hi, 1, 3, 5, 7);
  return r;
}

static inline void vst2q(float *ptr, f32x4x2 x) {
  v128_t lo = wasm_i32x4_shuffle(x.val[0], x.val[1], 0, 4, 1, 5);
  v128_t hi = wasm_i32x4_shuffle(x.val[0], x.val[1], 2, 6, 3, 7);
  wasm_v128_store(ptr, lo);
  wasm_v128_store(ptr + 4, hi);
}

/********************************************************************************
 * horizontal transforms
 *******************************************************************************/
// irreversible IDWT: x0.HP -= coeff * (x0.LP + x1.LP)
// Mirrors idwt_irrev97_fixed_neon_hor_step with vfmsq_f32 → mul+sub.
static auto idwt_irrev97_fixed_wasm_hor_step = [](const int32_t init_pos, const int32_t simdlen,
                                                   float *const X, const int32_t n0, const int32_t n1,
                                                   float coeff) {
  v128_t vvv = wasm_f32x4_splat(coeff);
  int32_t n = init_pos, i = simdlen;
  // 2× unrolled: two independent groups per iteration for better ILP.
  for (; i > 4; i -= 8, n += 16) {
    auto x0a       = vld2q(X + n + n0);
    auto x1a       = vld2q(X + n + n1);
    auto x0b       = vld2q(X + n + 8 + n0);
    auto x1b       = vld2q(X + n + 8 + n1);
    x0a.val[1]     = wasm_f32x4_sub(x0a.val[1], wasm_f32x4_mul(wasm_f32x4_add(x0a.val[0], x1a.val[0]), vvv));
    x0b.val[1]     = wasm_f32x4_sub(x0b.val[1], wasm_f32x4_mul(wasm_f32x4_add(x0b.val[0], x1b.val[0]), vvv));
    vst2q(X + n + n0, x0a);
    vst2q(X + n + 8 + n0, x0b);
  }
  for (; i > 0; i -= 4, n += 8) {
    auto x0   = vld2q(X + n + n0);
    auto x1   = vld2q(X + n + n1);
    x0.val[1] = wasm_f32x4_sub(x0.val[1], wasm_f32x4_mul(wasm_f32x4_add(x0.val[0], x1.val[0]), vvv));
    vst2q(X + n + n0, x0);
  }
};

void idwt_1d_filtr_irrev97_fixed_wasm(sprec_t *X, const int32_t left, const int32_t u_i0,
                                      const int32_t u_i1) {
  const auto i0        = static_cast<int32_t>(u_i0);
  const auto i1        = static_cast<int32_t>(u_i1);
  const int32_t start  = i0 / 2;
  const int32_t stop   = i1 / 2;
  const int32_t offset = left - i0 % 2;

  int32_t simdlen = stop + 2 - (start - 1);
  idwt_irrev97_fixed_wasm_hor_step(offset - 2, simdlen, X, -1, 1, fD);

  simdlen = stop + 1 - (start - 1);
  idwt_irrev97_fixed_wasm_hor_step(offset - 2, simdlen, X, 0, 2, fC);

  simdlen = stop + 1 - start;
  idwt_irrev97_fixed_wasm_hor_step(offset, simdlen, X, -1, 1, fB);

  simdlen = stop - start;
  idwt_irrev97_fixed_wasm_hor_step(offset, simdlen, X, 0, 2, fA);
}

// reversible IDWT 5/3
void idwt_1d_filtr_rev53_fixed_wasm(sprec_t *X, const int32_t left, const int32_t i0,
                                    const int32_t i1) {
  const int32_t start  = i0 / 2;
  const int32_t stop   = i1 / 2;
  const int32_t offset = left - i0 % 2;

  // step 1: LP[i] -= floor((HP[i-1] + HP[i] + 2) * 0.25)
  const int32_t base1 = offset;
  int32_t simdlen     = stop + 1 - start;
  const v128_t vtwo   = wasm_f32x4_splat(2.0f);
  const v128_t vqrt   = wasm_f32x4_splat(0.25f);
  int32_t k = 0;
  for (; k + 4 < simdlen; k += 8) {
    sprec_t *sp = X + base1 + k * 2;
    auto xl0a   = vld2q(sp - 1);
    auto xl1a   = vld2q(sp + 1);
    auto xl0b   = vld2q(sp + 7);
    auto xl1b   = vld2q(sp + 9);
    xl0a.val[1] = wasm_f32x4_sub(xl0a.val[1],
                    wasm_f32x4_floor(wasm_f32x4_mul(
                      wasm_f32x4_add(wasm_f32x4_add(xl0a.val[0], xl1a.val[0]), vtwo), vqrt)));
    xl0b.val[1] = wasm_f32x4_sub(xl0b.val[1],
                    wasm_f32x4_floor(wasm_f32x4_mul(
                      wasm_f32x4_add(wasm_f32x4_add(xl0b.val[0], xl1b.val[0]), vtwo), vqrt)));
    vst2q(sp - 1, xl0a);
    vst2q(sp + 7, xl0b);
  }
  for (; k < simdlen; k += 4) {
    sprec_t *sp = X + base1 + k * 2;
    auto xl0    = vld2q(sp - 1);
    auto xl1    = vld2q(sp + 1);
    xl0.val[1]  = wasm_f32x4_sub(xl0.val[1],
                    wasm_f32x4_floor(wasm_f32x4_mul(
                      wasm_f32x4_add(wasm_f32x4_add(xl0.val[0], xl1.val[0]), vtwo), vqrt)));
    vst2q(sp - 1, xl0);
  }

  // step 2: HP[i] += floor((LP[i] + LP[i+1]) * 0.5)
  const int32_t base2 = offset;
  simdlen             = stop - start;
  const v128_t vhalf  = wasm_f32x4_splat(0.5f);
  k                   = 0;
  for (; k + 4 < simdlen; k += 8) {
    sprec_t *sp = X + base2 + k * 2;
    auto xl0a   = vld2q(sp);
    auto xl1a   = vld2q(sp + 2);
    auto xl0b   = vld2q(sp + 8);
    auto xl1b   = vld2q(sp + 10);
    xl0a.val[1] = wasm_f32x4_add(xl0a.val[1],
                    wasm_f32x4_floor(wasm_f32x4_mul(wasm_f32x4_add(xl0a.val[0], xl1a.val[0]), vhalf)));
    xl0b.val[1] = wasm_f32x4_add(xl0b.val[1],
                    wasm_f32x4_floor(wasm_f32x4_mul(wasm_f32x4_add(xl0b.val[0], xl1b.val[0]), vhalf)));
    vst2q(sp, xl0a);
    vst2q(sp + 8, xl0b);
  }
  for (; k < simdlen; k += 4) {
    sprec_t *sp = X + base2 + k * 2;
    auto xl0    = vld2q(sp);
    auto xl1    = vld2q(sp + 2);
    xl0.val[1]  = wasm_f32x4_add(xl0.val[1],
                    wasm_f32x4_floor(wasm_f32x4_mul(wasm_f32x4_add(xl0.val[0], xl1.val[0]), vhalf)));
    vst2q(sp, xl0);
  }
}

#endif  // OPENHTJ2K_ENABLE_WASM_SIMD
