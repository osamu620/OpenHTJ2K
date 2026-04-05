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

#if defined(OPENHTJ2K_ENABLE_WASM_SIMD)
  #include <wasm_simd128.h>
  #include "color.hpp"

// lossless: forward RCT
void cvt_rgb_to_ycbcr_rev_wasm(int32_t *sp0, int32_t *sp1, int32_t *sp2, uint32_t width,
                                uint32_t height) {
  for (uint32_t y = 0; y < height; ++y) {
    int32_t *p0 = sp0 + y * round_up(width, 32U);
    int32_t *p1 = sp1 + y * round_up(width, 32U);
    int32_t *p2 = sp2 + y * round_up(width, 32U);
    int32_t len = static_cast<int32_t>(width);
    for (; len > 0; len -= 4) {
      v128_t vR = wasm_v128_load(p0);
      v128_t vG = wasm_v128_load(p1);
      v128_t vB = wasm_v128_load(p2);
      // Y = (R + 2G + B) >> 2
      v128_t vY = wasm_i32x4_shr(wasm_i32x4_add(wasm_i32x4_add(vR, vB), wasm_i32x4_add(vG, vG)), 2);
      wasm_v128_store(p0, vY);
      wasm_v128_store(p1, wasm_i32x4_sub(vB, vG));  // Cb = B - G
      wasm_v128_store(p2, wasm_i32x4_sub(vR, vG));  // Cr = R - G
      p0 += 4;
      p1 += 4;
      p2 += 4;
    }
  }
}

// lossy: forward ICT
void cvt_rgb_to_ycbcr_irrev_wasm(int32_t *sp0, int32_t *sp1, int32_t *sp2, uint32_t width,
                                  uint32_t height) {
  const v128_t fALPHA_R = wasm_f32x4_splat(static_cast<float>(ALPHA_R));
  const v128_t fALPHA_G = wasm_f32x4_splat(static_cast<float>(ALPHA_G));
  const v128_t fALPHA_B = wasm_f32x4_splat(static_cast<float>(ALPHA_B));
  const v128_t fCB_FACT = wasm_f32x4_splat(static_cast<float>(1.0 / CB_FACT_B));
  const v128_t fCR_FACT = wasm_f32x4_splat(static_cast<float>(1.0 / CR_FACT_R));
  for (uint32_t y = 0; y < height; ++y) {
    int32_t *p0 = sp0 + y * round_up(width, 32U);
    int32_t *p1 = sp1 + y * round_up(width, 32U);
    int32_t *p2 = sp2 + y * round_up(width, 32U);
    int32_t len = static_cast<int32_t>(width);
    for (; len > 0; len -= 4) {
      v128_t fR = wasm_f32x4_convert_i32x4(wasm_v128_load(p0));
      v128_t fG = wasm_f32x4_convert_i32x4(wasm_v128_load(p1));
      v128_t fB = wasm_f32x4_convert_i32x4(wasm_v128_load(p2));
      v128_t fY = wasm_f32x4_add(wasm_f32x4_add(wasm_f32x4_mul(fR, fALPHA_R),
                                                 wasm_f32x4_mul(fG, fALPHA_G)),
                                 wasm_f32x4_mul(fB, fALPHA_B));
      // round to nearest integer
      wasm_v128_store(p0, wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_nearest(fY)));
      wasm_v128_store(p1, wasm_i32x4_trunc_sat_f32x4(
                              wasm_f32x4_nearest(wasm_f32x4_mul(fCB_FACT, wasm_f32x4_sub(fB, fY)))));
      wasm_v128_store(p2, wasm_i32x4_trunc_sat_f32x4(
                              wasm_f32x4_nearest(wasm_f32x4_mul(fCR_FACT, wasm_f32x4_sub(fR, fY)))));
      p0 += 4;
      p1 += 4;
      p2 += 4;
    }
  }
}

// lossless: inverse RCT
void cvt_ycbcr_to_rgb_rev_wasm(int32_t *sp0, int32_t *sp1, int32_t *sp2, uint32_t width,
                                uint32_t height) {
  for (uint32_t y = 0; y < height; ++y) {
    int32_t *p0 = sp0 + y * round_up(width, 32U);
    int32_t *p1 = sp1 + y * round_up(width, 32U);
    int32_t *p2 = sp2 + y * round_up(width, 32U);
    int32_t len = static_cast<int32_t>(width);
    for (; len > 0; len -= 4) {
      v128_t vY  = wasm_v128_load(p0);
      v128_t vCb = wasm_v128_load(p1);
      v128_t vCr = wasm_v128_load(p2);
      v128_t vG  = wasm_i32x4_sub(vY, wasm_i32x4_shr(wasm_i32x4_add(vCb, vCr), 2));
      wasm_v128_store(p0, wasm_i32x4_add(vCr, vG));  // R = Cr + G
      wasm_v128_store(p1, vG);                        // G
      wasm_v128_store(p2, wasm_i32x4_add(vCb, vG));  // B = Cb + G
      p0 += 4;
      p1 += 4;
      p2 += 4;
    }
  }
}

// lossy: inverse ICT
void cvt_ycbcr_to_rgb_irrev_wasm(int32_t *sp0, int32_t *sp1, int32_t *sp2, uint32_t width,
                                  uint32_t height) {
  const v128_t fCR_FACT_R = wasm_f32x4_splat(static_cast<float>(CR_FACT_R));
  const v128_t fCB_FACT_B = wasm_f32x4_splat(static_cast<float>(CB_FACT_B));
  const v128_t fCR_FACT_G = wasm_f32x4_splat(static_cast<float>(CR_FACT_G));
  const v128_t fCB_FACT_G = wasm_f32x4_splat(static_cast<float>(CB_FACT_G));
  for (uint32_t y = 0; y < height; ++y) {
    int32_t *p0 = sp0 + y * round_up(width, 32U);
    int32_t *p1 = sp1 + y * round_up(width, 32U);
    int32_t *p2 = sp2 + y * round_up(width, 32U);
    int32_t len = static_cast<int32_t>(width);
    for (; len > 0; len -= 4) {
      v128_t fY  = wasm_f32x4_convert_i32x4(wasm_v128_load(p0));
      v128_t fCb = wasm_f32x4_convert_i32x4(wasm_v128_load(p1));
      v128_t fCr = wasm_f32x4_convert_i32x4(wasm_v128_load(p2));
      v128_t fR  = wasm_f32x4_add(fY, wasm_f32x4_mul(fCr, fCR_FACT_R));
      v128_t fB  = wasm_f32x4_add(fY, wasm_f32x4_mul(fCb, fCB_FACT_B));
      v128_t fG  = wasm_f32x4_sub(wasm_f32x4_sub(fY, wasm_f32x4_mul(fCr, fCR_FACT_G)),
                                   wasm_f32x4_mul(fCb, fCB_FACT_G));
      wasm_v128_store(p0, wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_nearest(fR)));
      wasm_v128_store(p1, wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_nearest(fG)));
      wasm_v128_store(p2, wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_nearest(fB)));
      p0 += 4;
      p1 += 4;
      p2 += 4;
    }
  }
}

// lossless: inverse RCT on float buffers
void cvt_ycbcr_to_rgb_rev_float_wasm(float *sp0, float *sp1, float *sp2, uint32_t width,
                                     uint32_t height, uint32_t stride) {
  for (uint32_t y = 0; y < height; ++y) {
    float *p0   = sp0 + y * stride;
    float *p1   = sp1 + y * stride;
    float *p2   = sp2 + y * stride;
    int32_t len = static_cast<int32_t>(width);
    for (; len >= 4; len -= 4) {
      // floats hold exact integers; trunc == nearest
      v128_t iY  = wasm_i32x4_trunc_sat_f32x4(wasm_v128_load(p0));
      v128_t iCb = wasm_i32x4_trunc_sat_f32x4(wasm_v128_load(p1));
      v128_t iCr = wasm_i32x4_trunc_sat_f32x4(wasm_v128_load(p2));
      v128_t iG  = wasm_i32x4_sub(iY, wasm_i32x4_shr(wasm_i32x4_add(iCb, iCr), 2));
      wasm_v128_store(p0, wasm_f32x4_convert_i32x4(wasm_i32x4_add(iCr, iG)));
      wasm_v128_store(p1, wasm_f32x4_convert_i32x4(iG));
      wasm_v128_store(p2, wasm_f32x4_convert_i32x4(wasm_i32x4_add(iCb, iG)));
      p0 += 4;
      p1 += 4;
      p2 += 4;
    }
    for (; len > 0; --len) {
      int32_t Y  = static_cast<int32_t>(*p0);
      int32_t Cb = static_cast<int32_t>(*p1);
      int32_t Cr = static_cast<int32_t>(*p2);
      int32_t G  = Y - ((Cb + Cr) >> 2);
      *p0++      = static_cast<float>(Cr + G);
      *p1++      = static_cast<float>(G);
      *p2++      = static_cast<float>(Cb + G);
    }
  }
}

// lossy: inverse ICT on float buffers
void cvt_ycbcr_to_rgb_irrev_float_wasm(float *sp0, float *sp1, float *sp2, uint32_t width,
                                       uint32_t height, uint32_t stride) {
  const v128_t fCR_FACT_R = wasm_f32x4_splat(static_cast<float>(CR_FACT_R));
  const v128_t fCB_FACT_B = wasm_f32x4_splat(static_cast<float>(CB_FACT_B));
  const v128_t fCR_FACT_G = wasm_f32x4_splat(static_cast<float>(CR_FACT_G));
  const v128_t fCB_FACT_G = wasm_f32x4_splat(static_cast<float>(CB_FACT_G));
  for (uint32_t y = 0; y < height; ++y) {
    float *p0   = sp0 + y * stride;
    float *p1   = sp1 + y * stride;
    float *p2   = sp2 + y * stride;
    int32_t len = static_cast<int32_t>(width);
    for (; len >= 4; len -= 4) {
      v128_t fY  = wasm_v128_load(p0);
      v128_t fCb = wasm_v128_load(p1);
      v128_t fCr = wasm_v128_load(p2);
      wasm_v128_store(p0, wasm_f32x4_add(fY, wasm_f32x4_mul(fCr, fCR_FACT_R)));
      wasm_v128_store(p2, wasm_f32x4_add(fY, wasm_f32x4_mul(fCb, fCB_FACT_B)));
      fY = wasm_f32x4_sub(wasm_f32x4_sub(fY, wasm_f32x4_mul(fCr, fCR_FACT_G)),
                          wasm_f32x4_mul(fCb, fCB_FACT_G));
      wasm_v128_store(p1, fY);
      p0 += 4;
      p1 += 4;
      p2 += 4;
    }
    for (; len > 0; --len) {
      float Y  = *p0;
      float Cb = *p1;
      float Cr = *p2;
      *p0++    = Y + static_cast<float>(CR_FACT_R) * Cr;
      *p1++    = Y - static_cast<float>(CR_FACT_G) * Cr - static_cast<float>(CB_FACT_G) * Cb;
      *p2++    = Y + static_cast<float>(CB_FACT_B) * Cb;
    }
  }
}

// lossless: fused int32→float + forward RCT
void cvt_rgb_to_ycbcr_rev_float_wasm(const int32_t *sp0, const int32_t *sp1, const int32_t *sp2,
                                     float *dp0, float *dp1, float *dp2, uint32_t width,
                                     uint32_t height, uint32_t stride) {
  for (uint32_t y = 0; y < height; ++y) {
    const int32_t *p0 = sp0 + y * stride;
    const int32_t *p1 = sp1 + y * stride;
    const int32_t *p2 = sp2 + y * stride;
    float *d0         = dp0 + y * stride;
    float *d1         = dp1 + y * stride;
    float *d2         = dp2 + y * stride;
    int32_t len       = static_cast<int32_t>(width);
    for (; len >= 4; len -= 4) {
      v128_t mR = wasm_v128_load(p0);
      v128_t mG = wasm_v128_load(p1);
      v128_t mB = wasm_v128_load(p2);
      v128_t mY = wasm_i32x4_shr(wasm_i32x4_add(wasm_i32x4_add(mR, mB), wasm_i32x4_add(mG, mG)), 2);
      wasm_v128_store(d0, wasm_f32x4_convert_i32x4(mY));
      wasm_v128_store(d1, wasm_f32x4_convert_i32x4(wasm_i32x4_sub(mB, mG)));
      wasm_v128_store(d2, wasm_f32x4_convert_i32x4(wasm_i32x4_sub(mR, mG)));
      p0 += 4; p1 += 4; p2 += 4;
      d0 += 4; d1 += 4; d2 += 4;
    }
    for (; len > 0; --len) {
      int32_t R = *p0++, G = *p1++, B = *p2++;
      *d0++ = static_cast<float>((R + 2 * G + B) >> 2);
      *d1++ = static_cast<float>(B - G);
      *d2++ = static_cast<float>(R - G);
    }
  }
}

// lossy: fused int32→float + forward ICT
void cvt_rgb_to_ycbcr_irrev_float_wasm(const int32_t *sp0, const int32_t *sp1, const int32_t *sp2,
                                       float *dp0, float *dp1, float *dp2, uint32_t width,
                                       uint32_t height, uint32_t stride) {
  const v128_t fALPHA_R = wasm_f32x4_splat(static_cast<float>(ALPHA_R));
  const v128_t fALPHA_G = wasm_f32x4_splat(static_cast<float>(ALPHA_G));
  const v128_t fALPHA_B = wasm_f32x4_splat(static_cast<float>(ALPHA_B));
  const v128_t fCB_FACT = wasm_f32x4_splat(static_cast<float>(1.0 / CB_FACT_B));
  const v128_t fCR_FACT = wasm_f32x4_splat(static_cast<float>(1.0 / CR_FACT_R));
  for (uint32_t y = 0; y < height; ++y) {
    const int32_t *p0 = sp0 + y * stride;
    const int32_t *p1 = sp1 + y * stride;
    const int32_t *p2 = sp2 + y * stride;
    float *d0         = dp0 + y * stride;
    float *d1         = dp1 + y * stride;
    float *d2         = dp2 + y * stride;
    int32_t len       = static_cast<int32_t>(width);
    for (; len >= 4; len -= 4) {
      v128_t mR = wasm_f32x4_convert_i32x4(wasm_v128_load(p0));
      v128_t mG = wasm_f32x4_convert_i32x4(wasm_v128_load(p1));
      v128_t mB = wasm_f32x4_convert_i32x4(wasm_v128_load(p2));
      v128_t mY = wasm_f32x4_add(wasm_f32x4_add(wasm_f32x4_mul(mR, fALPHA_R),
                                                 wasm_f32x4_mul(mG, fALPHA_G)),
                                 wasm_f32x4_mul(mB, fALPHA_B));
      wasm_v128_store(d0, mY);
      wasm_v128_store(d1, wasm_f32x4_mul(fCB_FACT, wasm_f32x4_sub(mB, mY)));
      wasm_v128_store(d2, wasm_f32x4_mul(fCR_FACT, wasm_f32x4_sub(mR, mY)));
      p0 += 4; p1 += 4; p2 += 4;
      d0 += 4; d1 += 4; d2 += 4;
    }
    for (; len > 0; --len) {
      float R = static_cast<float>(*p0++);
      float G = static_cast<float>(*p1++);
      float B = static_cast<float>(*p2++);
      float Y = static_cast<float>(ALPHA_R) * R + static_cast<float>(ALPHA_G) * G
                + static_cast<float>(ALPHA_B) * B;
      *d0++ = Y;
      *d1++ = static_cast<float>(1.0 / CB_FACT_B) * (B - Y);
      *d2++ = static_cast<float>(1.0 / CR_FACT_R) * (R - Y);
    }
  }
}

// Fused inverse ICT + float→int32 finalize, WASM-SIMD vectorized.
// Mirrors the AVX2 implementation: two fast paths (ds>0 and ds==0), scalar tail for ds<0.
void fused_ycbcr_irrev_to_rgb_i32_wasm(const float *y, const float *cb, const float *cr,
                                        int32_t *r, int32_t *g, int32_t *b, uint32_t width,
                                        const FinalizeParams *fp) {
  const v128_t fCR_FACT_R = wasm_f32x4_splat(static_cast<float>(CR_FACT_R));
  const v128_t fCR_FACT_G = wasm_f32x4_splat(static_cast<float>(CR_FACT_G));
  const v128_t fCB_FACT_B = wasm_f32x4_splat(static_cast<float>(CB_FACT_B));
  const v128_t fCB_FACT_G = wasm_f32x4_splat(static_cast<float>(CB_FACT_G));

  const v128_t vrnd0 = wasm_i32x4_splat(fp[0].rnd);
  const v128_t vdc0  = wasm_i32x4_splat(fp[0].dc);
  const v128_t vmx0  = wasm_i32x4_splat(fp[0].maxval);
  const v128_t vmn0  = wasm_i32x4_splat(fp[0].minval);
  const v128_t vrnd1 = wasm_i32x4_splat(fp[1].rnd);
  const v128_t vdc1  = wasm_i32x4_splat(fp[1].dc);
  const v128_t vmx1  = wasm_i32x4_splat(fp[1].maxval);
  const v128_t vmn1  = wasm_i32x4_splat(fp[1].minval);
  const v128_t vrnd2 = wasm_i32x4_splat(fp[2].rnd);
  const v128_t vdc2  = wasm_i32x4_splat(fp[2].dc);
  const v128_t vmx2  = wasm_i32x4_splat(fp[2].maxval);
  const v128_t vmn2  = wasm_i32x4_splat(fp[2].minval);

  uint32_t n = 0;
  if (fp[0].ds > 0 && fp[1].ds > 0 && fp[2].ds > 0) {
    const int ds0 = fp[0].ds, ds1 = fp[1].ds, ds2 = fp[2].ds;
    for (; n + 4 <= width; n += 4) {
      v128_t fY  = wasm_v128_load(y + n);
      v128_t fCb = wasm_v128_load(cb + n);
      v128_t fCr = wasm_v128_load(cr + n);
      v128_t fR  = wasm_f32x4_add(fY, wasm_f32x4_mul(fCr, fCR_FACT_R));
      v128_t fB  = wasm_f32x4_add(fY, wasm_f32x4_mul(fCb, fCB_FACT_B));
      v128_t fG  = wasm_f32x4_sub(wasm_f32x4_sub(fY, wasm_f32x4_mul(fCr, fCR_FACT_G)),
                                   wasm_f32x4_mul(fCb, fCB_FACT_G));
      v128_t vR  = wasm_i32x4_shr(wasm_i32x4_add(wasm_i32x4_trunc_sat_f32x4(fR), vrnd0), ds0);
      v128_t vG  = wasm_i32x4_shr(wasm_i32x4_add(wasm_i32x4_trunc_sat_f32x4(fG), vrnd1), ds1);
      v128_t vB  = wasm_i32x4_shr(wasm_i32x4_add(wasm_i32x4_trunc_sat_f32x4(fB), vrnd2), ds2);
      wasm_v128_store(r + n, wasm_i32x4_min(wasm_i32x4_max(wasm_i32x4_add(vR, vdc0), vmn0), vmx0));
      wasm_v128_store(g + n, wasm_i32x4_min(wasm_i32x4_max(wasm_i32x4_add(vG, vdc1), vmn1), vmx1));
      wasm_v128_store(b + n, wasm_i32x4_min(wasm_i32x4_max(wasm_i32x4_add(vB, vdc2), vmn2), vmx2));
    }
  } else if (fp[0].ds == 0 && fp[1].ds == 0 && fp[2].ds == 0) {
    for (; n + 4 <= width; n += 4) {
      v128_t fY  = wasm_v128_load(y + n);
      v128_t fCb = wasm_v128_load(cb + n);
      v128_t fCr = wasm_v128_load(cr + n);
      v128_t vR  = wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_add(fY, wasm_f32x4_mul(fCr, fCR_FACT_R)));
      v128_t vB  = wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_add(fY, wasm_f32x4_mul(fCb, fCB_FACT_B)));
      v128_t vG  = wasm_i32x4_trunc_sat_f32x4(
          wasm_f32x4_sub(wasm_f32x4_sub(fY, wasm_f32x4_mul(fCr, fCR_FACT_G)),
                         wasm_f32x4_mul(fCb, fCB_FACT_G)));
      wasm_v128_store(r + n, wasm_i32x4_min(wasm_i32x4_max(wasm_i32x4_add(vR, vdc0), vmn0), vmx0));
      wasm_v128_store(g + n, wasm_i32x4_min(wasm_i32x4_max(wasm_i32x4_add(vG, vdc1), vmn1), vmx1));
      wasm_v128_store(b + n, wasm_i32x4_min(wasm_i32x4_max(wasm_i32x4_add(vB, vdc2), vmn2), vmx2));
    }
  }
  // Scalar tail (also handles ds < 0 edge case for high-bitdepth images).
  auto finalize_scalar = [](float v, const FinalizeParams &p) -> int32_t {
    int32_t x = static_cast<int32_t>(v);
    if (p.ds > 0) x = (x + p.rnd) >> p.ds;
    else if (p.ds < 0) x <<= -p.ds;
    x += p.dc;
    if (x > p.maxval) x = p.maxval;
    if (x < p.minval) x = p.minval;
    return x;
  };
  for (; n < width; ++n) {
    float Y = y[n], Cb = cb[n], Cr = cr[n];
    r[n] = finalize_scalar(Y + static_cast<float>(CR_FACT_R) * Cr, fp[0]);
    g[n] = finalize_scalar(
        Y - static_cast<float>(CR_FACT_G) * Cr - static_cast<float>(CB_FACT_G) * Cb, fp[1]);
    b[n] = finalize_scalar(Y + static_cast<float>(CB_FACT_B) * Cb, fp[2]);
  }
}

// Fused inverse RCT (lossless) + int32 finalize, WASM-SIMD vectorized.
// ds must be 0 for all components; only DC_OFFSET and clamp are applied.
void fused_ycbcr_rev_to_rgb_i32_wasm(const float *y, const float *cb, const float *cr,
                                      int32_t *r, int32_t *g, int32_t *b, uint32_t width,
                                      const FinalizeParams *fp) {
  const v128_t vdc0 = wasm_i32x4_splat(fp[0].dc);
  const v128_t vmx0 = wasm_i32x4_splat(fp[0].maxval);
  const v128_t vmn0 = wasm_i32x4_splat(fp[0].minval);
  const v128_t vdc1 = wasm_i32x4_splat(fp[1].dc);
  const v128_t vmx1 = wasm_i32x4_splat(fp[1].maxval);
  const v128_t vmn1 = wasm_i32x4_splat(fp[1].minval);
  const v128_t vdc2 = wasm_i32x4_splat(fp[2].dc);
  const v128_t vmx2 = wasm_i32x4_splat(fp[2].maxval);
  const v128_t vmn2 = wasm_i32x4_splat(fp[2].minval);

  uint32_t n = 0;
  for (; n + 4 <= width; n += 4) {
    // Float values are integer-valued after 5/3 IDWT; trunc == floor for exact integers.
    v128_t iY  = wasm_i32x4_trunc_sat_f32x4(wasm_v128_load(y + n));
    v128_t iCb = wasm_i32x4_trunc_sat_f32x4(wasm_v128_load(cb + n));
    v128_t iCr = wasm_i32x4_trunc_sat_f32x4(wasm_v128_load(cr + n));
    // RCT: G = Y - (Cb+Cr)>>2 (arithmetic shift); R = Cr+G; B = Cb+G
    v128_t iG  = wasm_i32x4_sub(iY, wasm_i32x4_shr(wasm_i32x4_add(iCb, iCr), 2));
    v128_t iR  = wasm_i32x4_add(iCr, iG);
    v128_t iB  = wasm_i32x4_add(iCb, iG);
    wasm_v128_store(r + n, wasm_i32x4_min(wasm_i32x4_max(wasm_i32x4_add(iR, vdc0), vmn0), vmx0));
    wasm_v128_store(g + n, wasm_i32x4_min(wasm_i32x4_max(wasm_i32x4_add(iG, vdc1), vmn1), vmx1));
    wasm_v128_store(b + n, wasm_i32x4_min(wasm_i32x4_max(wasm_i32x4_add(iB, vdc2), vmn2), vmx2));
  }
  for (; n < width; ++n) {
    int32_t Y  = static_cast<int32_t>(y[n]);
    int32_t Cb = static_cast<int32_t>(cb[n]);
    int32_t Cr = static_cast<int32_t>(cr[n]);
    int32_t G  = Y - ((Cb + Cr) >> 2);
    auto clamp = [](int32_t v, const FinalizeParams &p) -> int32_t {
      v += p.dc;
      if (v > p.maxval) v = p.maxval;
      if (v < p.minval) v = p.minval;
      return v;
    };
    r[n] = clamp(Cr + G, fp[0]);
    g[n] = clamp(G, fp[1]);
    b[n] = clamp(Cb + G, fp[2]);
  }
}
#endif
