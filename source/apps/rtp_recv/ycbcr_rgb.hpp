// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.
//
// Licensed under the same BSD 3-Clause terms as the rest of OpenHTJ2K.

#pragma once

// YCbCr -> 8-bit RGB row conversion for the RFC 9828 receiver.
//
// Self-contained so the new app does not reach into source/apps/decoder/
// dec_utils.hpp.  Duplicates a handful of matrix constants at the cost of a
// clean module boundary.
//
// The conversion uses single-precision float — the decoder output is a
// handful of megapixels per frame, float matrix math is not the bottleneck,
// and float keeps the narrow/full-range and bit-depth arithmetic simple.
// Tighten to SIMD fixed-point later if profiling demands it.
//
// Supports:
//   - unsigned JPEG 2000 components (the broadcast case; signed is TODO)
//   - input depths 8..16
//   - chroma subsampling 4:4:4, 4:2:2, 4:2:0 via per-component stride ratios
//   - full-range and narrow-range BT.601 / BT.709
//
// NOT supported in v1:
//   - signed components (triggers an assert / std::abort)
//   - BT.2020 NCL (add in v2)

#include <algorithm>
#include <cassert>
#include <cstdint>

namespace open_htj2k::rtp_recv {

struct ycbcr_coefficients {
  bool  narrow_range;
  float cr_to_r;
  float cb_to_g;
  float cr_to_g;
  float cb_to_b;
};

// ITU-R BT.601 full-range (JPEG / JFIF convention)
constexpr ycbcr_coefficients YCBCR_BT601_FULL = {
    /*narrow_range*/ false,
    /*cr_to_r*/ 1.402f,
    /*cb_to_g*/ 0.344136f,
    /*cr_to_g*/ 0.714136f,
    /*cb_to_b*/ 1.772f,
};

// ITU-R BT.709 full-range
constexpr ycbcr_coefficients YCBCR_BT709_FULL = {
    /*narrow_range*/ false,
    /*cr_to_r*/ 1.5748f,
    /*cb_to_g*/ 0.1873f,
    /*cr_to_g*/ 0.4681f,
    /*cb_to_b*/ 1.8556f,
};

// ITU-R BT.601 narrow-range (SMPTE 170M, studio-range 16..235 luma, 16..240 chroma)
constexpr ycbcr_coefficients YCBCR_BT601_NARROW = {
    /*narrow_range*/ true,
    /*cr_to_r*/ 1.596f,
    /*cb_to_g*/ 0.391f,
    /*cr_to_g*/ 0.813f,
    /*cb_to_b*/ 2.017f,
};

// ITU-R BT.709 narrow-range
constexpr ycbcr_coefficients YCBCR_BT709_NARROW = {
    /*narrow_range*/ true,
    /*cr_to_r*/ 1.793f,
    /*cb_to_g*/ 0.213f,
    /*cr_to_g*/ 0.533f,
    /*cb_to_b*/ 2.112f,
};

// Select coefficients from RFC 9828 §5.3 MAT + RANGE (S=1 case).
// Returns nullptr for unsupported MAT values (e.g. BT.2020 NCL, log, etc.);
// the caller should log and drop the frame or fall back to CLI settings.
inline const ycbcr_coefficients* select_coefficients_from_mat(uint8_t mat, bool full_range) {
  // Values from ITU-T H.273 Table 4 (MatrixCoefficients).
  // 1 = BT.709; 5, 6 = BT.601 (625/525 lines); 9 = BT.2020 NCL (not yet).
  switch (mat) {
    case 1:
      return full_range ? &YCBCR_BT709_FULL : &YCBCR_BT709_NARROW;
    case 5:
    case 6:
      return full_range ? &YCBCR_BT601_FULL : &YCBCR_BT601_NARROW;
    default:
      return nullptr;
  }
}

// Convert one output row of YCbCr samples to 8-bit interleaved RGB.
//
//   y_row, cb_row, cr_row  — per-component row pointers from
//                            invoke_line_based_stream (planar int32_t samples).
//   out_rgb                — destination, 3*width bytes, row-contiguous.
//   width                  — number of luma samples in the row.
//   cb_stride_ratio        — ratio width[0]/width[1], i.e. how many luma
//                            pixels per Cb sample (1 for 4:4:4, 2 for 4:2:2/4:2:0).
//   cr_stride_ratio        — same for Cr.
//   coeffs                 — BT.601/709 full/narrow-range choice.
//   depth                  — JPEG 2000 bit depth of each component (8..16).
//                            All three components assumed same depth in v1.
//   is_signed              — must be false in v1; asserts otherwise.
inline void ycbcr_row_to_rgb8(const int32_t* y_row, const int32_t* cb_row, const int32_t* cr_row,
                              uint8_t* out_rgb, uint32_t width, uint32_t cb_stride_ratio,
                              uint32_t cr_stride_ratio, const ycbcr_coefficients& coeffs,
                              uint8_t depth, bool is_signed) {
  assert(!is_signed && "signed components not implemented in v1");
  assert(depth >= 8 && depth <= 16);
  assert(cb_stride_ratio >= 1 && cr_stride_ratio >= 1);
  (void)is_signed;  // only referenced by the assert, which vanishes in Release.

  // Bit-depth-dependent normalization constants.  All math in float because
  // the narrow-range scale factors differ for every bit depth and keeping
  // integer arithmetic correct is more error-prone than it is fast.
  const int32_t shift      = static_cast<int32_t>(depth) - 8;
  const float   maxval_f   = static_cast<float>((1 << depth) - 1);
  const float   c_center_f = static_cast<float>(1 << (depth - 1));

  // Narrow-range black level and gain.  For 8-bit these are 16 and 255/219;
  // for higher depths the levels scale left (16<<shift, 235<<shift) but the
  // normalized 0..1 output is the same.
  const float narrow_y_black  = static_cast<float>(16 << shift);
  const float narrow_y_range  = static_cast<float>((235 - 16) << shift);
  const float narrow_c_range  = static_cast<float>((240 - 16) << shift);  // for scaling Cb/Cr around center

  for (uint32_t x = 0; x < width; ++x) {
    const int32_t Y_i  = y_row[x];
    const int32_t Cb_i = cb_row[x / cb_stride_ratio];
    const int32_t Cr_i = cr_row[x / cr_stride_ratio];

    float y_n, cb_n, cr_n;
    if (coeffs.narrow_range) {
      y_n  = (static_cast<float>(Y_i) - narrow_y_black) / narrow_y_range;
      cb_n = (static_cast<float>(Cb_i) - c_center_f) / narrow_c_range;
      cr_n = (static_cast<float>(Cr_i) - c_center_f) / narrow_c_range;
    } else {
      y_n  = static_cast<float>(Y_i) / maxval_f;
      cb_n = (static_cast<float>(Cb_i) - c_center_f) / maxval_f;
      cr_n = (static_cast<float>(Cr_i) - c_center_f) / maxval_f;
    }

    const float r = y_n + coeffs.cr_to_r * cr_n;
    const float g = y_n - coeffs.cb_to_g * cb_n - coeffs.cr_to_g * cr_n;
    const float b = y_n + coeffs.cb_to_b * cb_n;

    out_rgb[3 * x + 0] = static_cast<uint8_t>(std::clamp(r * 255.0f + 0.5f, 0.0f, 255.0f));
    out_rgb[3 * x + 1] = static_cast<uint8_t>(std::clamp(g * 255.0f + 0.5f, 0.0f, 255.0f));
    out_rgb[3 * x + 2] = static_cast<uint8_t>(std::clamp(b * 255.0f + 0.5f, 0.0f, 255.0f));
  }
}

// Directly copy an RGB/planar-RGB row (Table 1 component mapping "RGB" or
// identity MAT with no color conversion needed).
inline void rgb_row_to_rgb8(const int32_t* r_row, const int32_t* g_row, const int32_t* b_row,
                            uint8_t* out_rgb, uint32_t width, uint8_t depth) {
  const int32_t shift  = static_cast<int32_t>(depth) - 8;
  const int32_t maxval = (1 << depth) - 1;
  for (uint32_t x = 0; x < width; ++x) {
    out_rgb[3 * x + 0] =
        static_cast<uint8_t>(std::clamp(r_row[x], int32_t{0}, maxval) >> (shift > 0 ? shift : 0));
    out_rgb[3 * x + 1] =
        static_cast<uint8_t>(std::clamp(g_row[x], int32_t{0}, maxval) >> (shift > 0 ? shift : 0));
    out_rgb[3 * x + 2] =
        static_cast<uint8_t>(std::clamp(b_row[x], int32_t{0}, maxval) >> (shift > 0 ? shift : 0));
  }
}

}  // namespace open_htj2k::rtp_recv
