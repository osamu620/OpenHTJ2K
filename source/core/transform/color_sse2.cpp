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

// Fused inverse ICT + finalize for x86-64 builds without AVX2 (OPENHTJ2K_ENABLE_SSE2,
// utils.hpp).  Every multiply and add is rounded separately (no FMA), in the order of the
// scalar code in color.cpp, so the output matches the scalar build bit for bit.  The
// scalar loop cannot vectorize on its own: SSE2 has no floor for finalize_round.  (The
// reversible fused_ycbcr_rev_to_rgb_i32 has no floor, and GCC vectorizes it as is.)

#include "utils.hpp"  // OPENHTJ2K_ENABLE_SSE2
#if defined(OPENHTJ2K_ENABLE_SSE2)
  #include "color.hpp"

namespace {
int32_t finalize_one(float v, const FinalizeParams &p) {
  int32_t x = finalize_round(v, p.scale) + p.dc;
  if (x > p.maxval) x = p.maxval;
  if (x < p.minval) x = p.minval;
  return x;
}
}  // namespace

// Inverse ICT (lossy) + finalize: round to the nearest output LSB, add DC, clamp.
void fused_ycbcr_irrev_to_rgb_i32_sse2(const float *y, const float *cb, const float *cr, int32_t *r,
                                       int32_t *g, int32_t *b, uint32_t width, const FinalizeParams *fp) {
  const __m128 kCrR  = _mm_set1_ps(static_cast<float>(CR_FACT_R));
  const __m128 kCrG  = _mm_set1_ps(static_cast<float>(CR_FACT_G));
  const __m128 kCbG  = _mm_set1_ps(static_cast<float>(CB_FACT_G));
  const __m128 kCbB  = _mm_set1_ps(static_cast<float>(CB_FACT_B));
  const __m128 vhalf = _mm_set1_ps(0.5f);
  const __m128 vsc0 = _mm_set1_ps(fp[0].scale), vsc1 = _mm_set1_ps(fp[1].scale),
               vsc2  = _mm_set1_ps(fp[2].scale);
  const __m128i vdc0 = _mm_set1_epi32(fp[0].dc), vdc1 = _mm_set1_epi32(fp[1].dc),
                vdc2 = _mm_set1_epi32(fp[2].dc);
  const __m128i vmn0 = _mm_set1_epi32(fp[0].minval), vmn1 = _mm_set1_epi32(fp[1].minval),
                vmn2 = _mm_set1_epi32(fp[2].minval);
  const __m128i vmx0 = _mm_set1_epi32(fp[0].maxval), vmx1 = _mm_set1_epi32(fp[1].maxval),
                vmx2 = _mm_set1_epi32(fp[2].maxval);
  uint32_t n         = 0;
  for (; n + 4 <= width; n += 4) {
    const __m128 vY  = _mm_loadu_ps(y + n);
    const __m128 vCb = _mm_loadu_ps(cb + n);
    const __m128 vCr = _mm_loadu_ps(cr + n);
    const __m128 vR  = _mm_add_ps(vY, _mm_mul_ps(kCrR, vCr));
    const __m128 vG  = _mm_sub_ps(_mm_sub_ps(vY, _mm_mul_ps(kCrG, vCr)), _mm_mul_ps(kCbG, vCb));
    const __m128 vB  = _mm_add_ps(vY, _mm_mul_ps(kCbB, vCb));
    const __m128i iR = _mm_add_epi32(finalize_round_sse2(vR, vsc0, vhalf), vdc0);
    const __m128i iG = _mm_add_epi32(finalize_round_sse2(vG, vsc1, vhalf), vdc1);
    const __m128i iB = _mm_add_epi32(finalize_round_sse2(vB, vsc2, vhalf), vdc2);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(r + n), clamp_epi32_sse2(iR, vmn0, vmx0));
    _mm_storeu_si128(reinterpret_cast<__m128i *>(g + n), clamp_epi32_sse2(iG, vmn1, vmx1));
    _mm_storeu_si128(reinterpret_cast<__m128i *>(b + n), clamp_epi32_sse2(iB, vmn2, vmx2));
  }
  for (; n < width; ++n) {
    const float Y = y[n], Cb = cb[n], Cr = cr[n];
    r[n] = finalize_one(Y + static_cast<float>(CR_FACT_R) * Cr, fp[0]);
    g[n] = finalize_one(Y - static_cast<float>(CR_FACT_G) * Cr - static_cast<float>(CB_FACT_G) * Cb, fp[1]);
    b[n] = finalize_one(Y + static_cast<float>(CB_FACT_B) * Cb, fp[2]);
  }
}
#endif
