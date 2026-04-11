// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.
//
// Licensed under the same BSD 3-Clause terms as the rest of OpenHTJ2K.

#pragma once

// HDR-aware colour pipeline parameters for the RFC 9828 receiver's shader
// path.  One ColorPipelineParams is stamped onto every DecodedFrame by the
// decode thread and forwarded to draw_ycbcr_program(), which sets the
// matching uniforms before drawing.
//
// The pipeline stages (in shader order, after the existing YCbCr -> R'G'B'
// matrix) are:
//
//   1. Inverse transfer (EOTF) to linear light in source primaries
//      (uTransfer: gamma2.2 / PQ / HLG).
//   2. Gamut matrix to linear light in display primaries
//      (uGamutMatrix: identity for matched primaries, BT.2020 -> BT.709
//      otherwise).
//   3. Hard clipping to [0, 1] (tone mapping v1 — BT.2390 EETF is a
//      follow-up).
//   4. Display encoding to non-linear framebuffer values
//      (uDisplayEncoding: sRGB / gamma2.2 / linear).
//
// The CPU-side reference functions in this header mirror the GLSL stages
// bit-for-bit (within float epsilon) so a host-side smoke test can
// exercise the pipeline without a GL context.

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace open_htj2k::rtp_recv {

// Transfer / EOTF selector (uTransfer in the fragment shader).
constexpr int TRANSFER_GAMMA22 = 0;
constexpr int TRANSFER_PQ      = 1;
constexpr int TRANSFER_HLG     = 2;

// Display-encoding selector (uDisplayEncoding in the fragment shader).
constexpr int DISPLAY_ENCODING_SRGB    = 0;
constexpr int DISPLAY_ENCODING_GAMMA22 = 1;
constexpr int DISPLAY_ENCODING_LINEAR  = 2;

// 3x3 matrices in GL column-major layout so the same float[9] can be
// passed straight to glUniformMatrix3fv(.., GL_FALSE, ..).
//
// Identity — used when the source primaries already match the display.
constexpr float kIdentityMatrix3[9] = {
    1.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 1.0f,
};

// BT.2020 -> BT.709 linear-light gamut matrix (ITU-R BT.2087-0).
//
// Mathematical form:
//   [ 1.660491  -0.587641  -0.072850]
//   [-0.124550   1.132900  -0.008350]
//   [-0.018151  -0.100579   1.118730]
//
// Stored column-major below.  The values match the BT.2087-0 rounding.
constexpr float kBt2020ToBt709[9] = {
     1.660491f, -0.124550f, -0.018151f,   // column 0
    -0.587641f,  1.132900f, -0.100579f,   // column 1
    -0.072850f, -0.008350f,  1.118730f,   // column 2
};

// Per-frame colour-pipeline state pushed from the decode thread to the
// renderer.  `gamut_matrix` aliases immortal constexpr data (one of the
// two arrays above); never owns.
struct ColorPipelineParams {
  int          transfer         = TRANSFER_GAMMA22;
  int          display_encoding = DISPLAY_ENCODING_SRGB;
  const float* gamut_matrix     = kIdentityMatrix3;
};

// ----- Host-side reference mirrors of the GLSL stages -----
//
// Kept as inline free functions so the smoke test can verify each stage
// against hand-computed reference points without touching a GL context.
// Any change to a GLSL helper in gl_renderer.cpp must be mirrored here
// (and vice versa); the smoke test embeds the arithmetic identities and
// will fail loudly on drift.

// SMPTE ST 2084 PQ EOTF.  Input: PQ-encoded e in [0, 1].  Output: linear
// light in [0, 1] where 1.0 corresponds to the 10 000 nit reference peak.
//
// Constants per ITU-R BT.2100 Table 4 / SMPTE ST 2084:
//   m1 = 2610 / 16384            = 0.1593017578125
//   m2 = (2523 / 4096) * 128     = 78.84375
//   c1 = 3424 / 4096             = 0.8359375
//   c2 = (2413 / 4096) * 32      = 18.8515625
//   c3 = (2392 / 4096) * 32      = 18.6875
inline float pq_to_linear(float e) {
  constexpr float m1 = 0.1593017578125f;
  constexpr float m2 = 78.84375f;
  constexpr float c1 = 0.8359375f;
  constexpr float c2 = 18.8515625f;
  constexpr float c3 = 18.6875f;
  const float v = std::pow(std::max(e, 0.0f), 1.0f / m2);
  const float num = std::max(v - c1, 0.0f);
  const float den = c2 - c3 * v;
  if (den <= 0.0f) return 1.0f;
  return std::pow(num / den, 1.0f / m1);
}

// ITU-R BT.2100 HLG inverse OETF.  Input: HLG-encoded e in [0, 1].
// Output: scene-linear in [0, 1].  The system OOTF (luma exponent) is
// treated as the identity — the HLG signal is assumed to be already
// display-referred for SDR output, which is accurate to within a single
// tunable gamma that is a follow-up PR (see project_pq_hlg_gamut_plan).
inline float hlg_inverse(float e) {
  constexpr float a = 0.17883277f;
  constexpr float b = 0.28466892f;
  constexpr float c = 0.55991073f;
  if (e <= 0.5f) return (e * e) / 3.0f;
  return (std::exp((e - c) / a) + b) / 12.0f;
}

// Display-encoding helpers.  All operate per channel on [0, 1] linear
// input and return [0, 1] non-linear framebuffer values.
inline float linear_to_srgb(float l) {
  if (l <= 0.0031308f) return 12.92f * l;
  return 1.055f * std::pow(l, 1.0f / 2.4f) - 0.055f;
}

inline float linear_to_gamma22(float l) {
  return std::pow(std::max(l, 0.0f), 1.0f / 2.2f);
}

// Inverse-EOTF dispatcher mirroring the `uTransfer` selector.
inline float apply_transfer(int transfer, float e) {
  switch (transfer) {
    case TRANSFER_PQ:   return pq_to_linear(e);
    case TRANSFER_HLG:  return hlg_inverse(e);
    case TRANSFER_GAMMA22:
    default:            return std::pow(std::max(e, 0.0f), 2.2f);
  }
}

// Display-encoding dispatcher mirroring `uDisplayEncoding`.
inline float apply_display_encoding(int enc, float l) {
  switch (enc) {
    case DISPLAY_ENCODING_GAMMA22: return linear_to_gamma22(l);
    case DISPLAY_ENCODING_LINEAR:  return l;
    case DISPLAY_ENCODING_SRGB:
    default:                       return linear_to_srgb(std::max(0.0f, std::min(1.0f, l)));
  }
}

// Apply a 3x3 gamut matrix stored column-major (same layout as the
// uniform passed to glUniformMatrix3fv).
inline void apply_gamut_matrix(const float m[9], float in_r, float in_g,
                               float in_b, float& out_r, float& out_g,
                               float& out_b) {
  out_r = m[0] * in_r + m[3] * in_g + m[6] * in_b;
  out_g = m[1] * in_r + m[4] * in_g + m[7] * in_b;
  out_b = m[2] * in_r + m[5] * in_g + m[8] * in_b;
}

// End-to-end host-side mirror of the shader's post-matrix stages.  Takes
// non-linear R'G'B' in [0, 1] (i.e. the output of the existing YCbCr ->
// R'G'B' matrix) and returns the framebuffer values the shader would
// write for the same inputs.  Used by smoke_test_color_pipeline.
inline void apply_color_pipeline(const ColorPipelineParams& p, float r_in,
                                 float g_in, float b_in, float& r_out,
                                 float& g_out, float& b_out) {
  const float lr = apply_transfer(p.transfer, r_in);
  const float lg = apply_transfer(p.transfer, g_in);
  const float lb = apply_transfer(p.transfer, b_in);
  float mr, mg, mb;
  apply_gamut_matrix(p.gamut_matrix, lr, lg, lb, mr, mg, mb);
  // Stage 3 is hard-clipping.  sRGB encoding already clamps internally;
  // the other encodings need the explicit clamp so out-of-range PQ
  // highlights do not blow up in pow().
  mr = std::max(0.0f, std::min(1.0f, mr));
  mg = std::max(0.0f, std::min(1.0f, mg));
  mb = std::max(0.0f, std::min(1.0f, mb));
  r_out = apply_display_encoding(p.display_encoding, mr);
  g_out = apply_display_encoding(p.display_encoding, mg);
  b_out = apply_display_encoding(p.display_encoding, mb);
}

// Human-readable labels used by log_coefficients_choice_once.
inline const char* transfer_label(int t) {
  switch (t) {
    case TRANSFER_PQ:  return "pq";
    case TRANSFER_HLG: return "hlg";
    case TRANSFER_GAMMA22:
    default:           return "gamma2.2";
  }
}

inline const char* display_encoding_label(int e) {
  switch (e) {
    case DISPLAY_ENCODING_GAMMA22: return "gamma22";
    case DISPLAY_ENCODING_LINEAR:  return "linear";
    case DISPLAY_ENCODING_SRGB:
    default:                       return "srgb";
  }
}

inline const char* gamut_matrix_label(const float* m) {
  return (m == kBt2020ToBt709) ? "bt2020->bt709" : "identity";
}

}  // namespace open_htj2k::rtp_recv
