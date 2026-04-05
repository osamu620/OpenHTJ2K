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

// Scalar fallback: active when neither AVX2 (both flag+ISA), WASM-SIMD, nor NEON are available.
// This mirrors the #else branch of the dispatch table in coding_units.cpp.
#if !(defined(OPENHTJ2K_TRY_AVX2) && defined(__AVX2__)) && \
    !defined(OPENHTJ2K_ENABLE_WASM_SIMD) && !defined(OPENHTJ2K_ENABLE_ARM_NEON)
  #include "color.hpp"

void cvt_rgb_to_ycbcr_rev(int32_t *sp0, int32_t *sp1, int32_t *sp2, uint32_t width, uint32_t height) {
  int32_t R, G, B;
  int32_t Y, Cb, Cr;
  for (uint32_t y = 0; y < height; ++y) {
    int32_t *p0 = sp0 + y * round_up(width, 32U);
    int32_t *p1 = sp1 + y * round_up(width, 32U);
    int32_t *p2 = sp2 + y * round_up(width, 32U);
    for (uint32_t n = 0; n < width; n++) {
      R     = p0[n];
      G     = p1[n];
      B     = p2[n];
      Y     = (R + 2 * G + B) >> 2;
      Cb    = B - G;
      Cr    = R - G;
      p0[n] = Y;
      p1[n] = Cb;
      p2[n] = Cr;
    }
  }
}

void cvt_rgb_to_ycbcr_irrev(int32_t *sp0, int32_t *sp1, int32_t *sp2, uint32_t width, uint32_t height) {
  double fR, fG, fB;
  double fY, fCb, fCr;
  for (uint32_t y = 0; y < height; ++y) {
    int32_t *p0 = sp0 + y * round_up(width, 32U);
    int32_t *p1 = sp1 + y * round_up(width, 32U);
    int32_t *p2 = sp2 + y * round_up(width, 32U);
    for (uint32_t n = 0; n < width; n++) {
      fR    = static_cast<double>(p0[n]);
      fG    = static_cast<double>(p1[n]);
      fB    = static_cast<double>(p2[n]);
      fY    = ALPHA_R * fR + ALPHA_G * fG + ALPHA_B * fB;
      fCb   = (1.0 / CB_FACT_B) * (fB - fY);
      fCr   = (1.0 / CR_FACT_R) * (fR - fY);
      p0[n] = round_d(fY);
      p1[n] = round_d(fCb);
      p2[n] = round_d(fCr);
    }
  }
}

void cvt_ycbcr_to_rgb_rev(int32_t *sp0, int32_t *sp1, int32_t *sp2, uint32_t width, uint32_t height) {
  int32_t R, G, B;
  int32_t Y, Cb, Cr;
  for (uint32_t y = 0; y < height; ++y) {
    int32_t *p0 = sp0 + y * round_up(width, 32U);
    int32_t *p1 = sp1 + y * round_up(width, 32U);
    int32_t *p2 = sp2 + y * round_up(width, 32U);
    for (uint32_t n = 0; n < width; n++) {
      Y     = p0[n];
      Cb    = p1[n];
      Cr    = p2[n];
      G     = Y - ((Cb + Cr) >> 2);
      R     = Cr + G;
      B     = Cb + G;
      p0[n] = R;
      p1[n] = G;
      p2[n] = B;
    }
  }
}

void cvt_ycbcr_to_rgb_irrev(int32_t *sp0, int32_t *sp1, int32_t *sp2, uint32_t width, uint32_t height) {
  int32_t R, G, B;
  double fY, fCb, fCr;
  for (uint32_t y = 0; y < height; ++y) {
    int32_t *p0 = sp0 + y * round_up(width, 32U);
    int32_t *p1 = sp1 + y * round_up(width, 32U);
    int32_t *p2 = sp2 + y * round_up(width, 32U);
    for (uint32_t n = 0; n < width; n++) {
      fY    = static_cast<double>(p0[n]);
      fCb   = static_cast<double>(p1[n]);
      fCr   = static_cast<double>(p2[n]);
      R     = static_cast<int32_t>(round_d(fY + CR_FACT_R * fCr));
      B     = static_cast<int32_t>(round_d(fY + CB_FACT_B * fCb));
      G     = static_cast<int32_t>(round_d(fY - CR_FACT_G * fCr - CB_FACT_G * fCb));
      p0[n] = R;
      p1[n] = G;
      p2[n] = B;
    }
  }
}

void cvt_ycbcr_to_rgb_rev_float(float *sp0, float *sp1, float *sp2, uint32_t width, uint32_t height,
                                uint32_t stride) {
  for (uint32_t y = 0; y < height; ++y) {
    float *p0 = sp0 + y * stride;
    float *p1 = sp1 + y * stride;
    float *p2 = sp2 + y * stride;
    for (uint32_t n = 0; n < width; ++n) {
      int32_t Y  = static_cast<int32_t>(p0[n]);
      int32_t Cb = static_cast<int32_t>(p1[n]);
      int32_t Cr = static_cast<int32_t>(p2[n]);
      int32_t G  = Y - ((Cb + Cr) >> 2);
      p0[n]      = static_cast<float>(Cr + G);
      p1[n]      = static_cast<float>(G);
      p2[n]      = static_cast<float>(Cb + G);
    }
  }
}

void cvt_ycbcr_to_rgb_irrev_float(float *sp0, float *sp1, float *sp2, uint32_t width, uint32_t height,
                                  uint32_t stride) {
  for (uint32_t y = 0; y < height; ++y) {
    float *p0 = sp0 + y * stride;
    float *p1 = sp1 + y * stride;
    float *p2 = sp2 + y * stride;
    for (uint32_t n = 0; n < width; ++n) {
      float Y  = p0[n];
      float Cb = p1[n];
      float Cr = p2[n];
      p0[n]    = Y + static_cast<float>(CR_FACT_R) * Cr;
      p1[n]    = Y - static_cast<float>(CR_FACT_G) * Cr - static_cast<float>(CB_FACT_G) * Cb;
      p2[n]    = Y + static_cast<float>(CB_FACT_B) * Cb;
    }
  }
}

void cvt_rgb_to_ycbcr_rev_float(const int32_t *sp0, const int32_t *sp1, const int32_t *sp2,
                                float *dp0, float *dp1, float *dp2,
                                uint32_t width, uint32_t height, uint32_t stride) {
  for (uint32_t y = 0; y < height; ++y) {
    const int32_t *p0 = sp0 + y * stride;
    const int32_t *p1 = sp1 + y * stride;
    const int32_t *p2 = sp2 + y * stride;
    float *d0         = dp0 + y * stride;
    float *d1         = dp1 + y * stride;
    float *d2         = dp2 + y * stride;
    for (uint32_t n = 0; n < width; ++n) {
      int32_t R = p0[n], G = p1[n], B = p2[n];
      d0[n] = static_cast<float>((R + 2 * G + B) >> 2);
      d1[n] = static_cast<float>(B - G);
      d2[n] = static_cast<float>(R - G);
    }
  }
}

void cvt_rgb_to_ycbcr_irrev_float(const int32_t *sp0, const int32_t *sp1, const int32_t *sp2,
                                  float *dp0, float *dp1, float *dp2,
                                  uint32_t width, uint32_t height, uint32_t stride) {
  for (uint32_t y = 0; y < height; ++y) {
    const int32_t *p0 = sp0 + y * stride;
    const int32_t *p1 = sp1 + y * stride;
    const int32_t *p2 = sp2 + y * stride;
    float *d0         = dp0 + y * stride;
    float *d1         = dp1 + y * stride;
    float *d2         = dp2 + y * stride;
    for (uint32_t n = 0; n < width; ++n) {
      float R = static_cast<float>(p0[n]);
      float G = static_cast<float>(p1[n]);
      float B = static_cast<float>(p2[n]);
      float Y = static_cast<float>(ALPHA_R) * R + static_cast<float>(ALPHA_G) * G
                + static_cast<float>(ALPHA_B) * B;
      d0[n] = Y;
      d1[n] = static_cast<float>(1.0 / CB_FACT_B) * (B - Y);
      d2[n] = static_cast<float>(1.0 / CR_FACT_R) * (R - Y);
    }
  }
}

static int32_t finalize_one(float v, const FinalizeParams &p) {
  int32_t x = static_cast<int32_t>(v);
  if (p.ds > 0) x = (x + p.rnd) >> p.ds;
  else if (p.ds < 0) x <<= -p.ds;
  x += p.dc;
  if (x > p.maxval) x = p.maxval;
  if (x < p.minval) x = p.minval;
  return x;
}

void fused_ycbcr_irrev_to_rgb_i32(const float *y, const float *cb, const float *cr, int32_t *r,
                                   int32_t *g, int32_t *b, uint32_t width,
                                   const FinalizeParams *fp) {
  for (uint32_t n = 0; n < width; ++n) {
    float Y = y[n], Cb = cb[n], Cr = cr[n];
    r[n] = finalize_one(Y + static_cast<float>(CR_FACT_R) * Cr, fp[0]);
    g[n] = finalize_one(
        Y - static_cast<float>(CR_FACT_G) * Cr - static_cast<float>(CB_FACT_G) * Cb, fp[1]);
    b[n] = finalize_one(Y + static_cast<float>(CB_FACT_B) * Cb, fp[2]);
  }
}

void fused_ycbcr_rev_to_rgb_i32(const float *y, const float *cb, const float *cr, int32_t *r,
                                 int32_t *g, int32_t *b, uint32_t width,
                                 const FinalizeParams *fp) {
  for (uint32_t n = 0; n < width; ++n) {
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