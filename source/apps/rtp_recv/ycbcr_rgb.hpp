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
//   - full-range and narrow-range BT.601 / BT.709 / BT.2020 NCL
//
// NOT supported:
//   - signed components (triggers an assert / std::abort)
//   - BT.2020 constant-luminance (ITU-T H.273 MatrixCoefficients = 10)
//   - ICtCp / XYZ matrices
//
// HDR colorimetry scope: this file covers only the YCbCr -> RGB matrix step.
// The result is non-linear R'G'B' in the source primaries (BT.601 / BT.709 /
// BT.2020) -- NOT linear light.  A PQ / HLG EOTF and a gamut-mapping step
// from the source primaries to the display primaries belong in a later
// slice of the HDR roadmap.

#include <algorithm>
#include <cassert>
#include <cstdint>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

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

// ITU-R BT.2020 NCL full-range.  Derived from Kr=0.2627, Kg=0.6780,
// Kb=0.0593 (BT.2020 Table 4, ITU-T H.273 MatrixCoefficients = 9):
//   cr_to_r = 2*(1 - Kr)                         = 1.4746
//   cb_to_b = 2*(1 - Kb)                         = 1.8814
//   cr_to_g = (2 * Kr * (1 - Kr)) / Kg           = 0.571353...
//   cb_to_g = (2 * Kb * (1 - Kb)) / Kg           = 0.164553...
constexpr ycbcr_coefficients YCBCR_BT2020_FULL = {
    /*narrow_range*/ false,
    /*cr_to_r*/ 1.4746f,
    /*cb_to_g*/ 0.16455313f,
    /*cr_to_g*/ 0.57135314f,
    /*cb_to_b*/ 1.8814f,
};

// ITU-R BT.2020 NCL narrow-range.  Mirrors the existing BT.601 / BT.709
// narrow-range entries, which scale the full-range coefficients by
// 255/224.  Kept consistent with the pre-existing narrow-range
// convention so the four narrow entries behave uniformly end-to-end;
// whether that convention is itself arithmetically correct under the
// shader's `(s - uBias) * uScale` prelude is tracked as a separate
// investigation rather than being fixed in this slice.
constexpr ycbcr_coefficients YCBCR_BT2020_NARROW = {
    /*narrow_range*/ true,
    /*cr_to_r*/ 1.67861f,    // 1.4746   * 255/224
    /*cb_to_g*/ 0.18732f,    // 0.164553 * 255/224
    /*cr_to_g*/ 0.65026f,    // 0.571353 * 255/224
    /*cb_to_b*/ 2.14179f,    // 1.8814   * 255/224
};

// Select coefficients from RFC 9828 §5.3 MAT + RANGE (S=1 case).
// Returns nullptr for unsupported MAT values (constant-luminance BT.2020,
// ICtCp, chroma-derived, log); the caller should log and drop the frame
// or fall back to CLI settings.
inline const ycbcr_coefficients* select_coefficients_from_mat(uint8_t mat, bool full_range) {
  // Values from ITU-T H.273 Table 4 (MatrixCoefficients).
  // 1 = BT.709; 5, 6 = BT.601 (625/525 lines); 9 = BT.2020 NCL.
  // 10 (BT.2020 CL), 11 (SMPTE ST 2085), 12-14 (chroma-derived /
  // ICtCp) are intentionally not mapped.
  switch (mat) {
    case 1:
      return full_range ? &YCBCR_BT709_FULL : &YCBCR_BT709_NARROW;
    case 5:
    case 6:
      return full_range ? &YCBCR_BT601_FULL : &YCBCR_BT601_NARROW;
    case 9:
      return full_range ? &YCBCR_BT2020_FULL : &YCBCR_BT2020_NARROW;
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

  // Pre-baked bias/scale for the normalization step: y_n = (Y - y_bias) * y_scale,
  // cb_n = (Cb - c_bias) * c_scale, cr_n = (Cr - c_bias) * c_scale.  Folding
  // the range selection outside the per-pixel loop lets the compiler hoist
  // the branch and keeps the SIMD inner loop branchless.
  float y_bias;
  float y_scale;
  float c_bias = c_center_f;
  float c_scale;
  if (coeffs.narrow_range) {
    y_bias  = narrow_y_black;
    y_scale = 1.0f / narrow_y_range;
    c_scale = 1.0f / narrow_c_range;
  } else {
    y_bias  = 0.0f;
    y_scale = 1.0f / maxval_f;
    c_scale = 1.0f / maxval_f;
  }

  uint32_t x = 0;

#if defined(__AVX2__)
  // AVX2 fast path: 8 luma pixels per iteration.  Handles the two
  // chroma layouts that appear in broadcast content (4:4:4 with
  // stride_ratio 1, and 4:2:2 / 4:2:0 with stride_ratio 2).  Both Cb
  // and Cr must use the same ratio so the SIMD loader can hit one of
  // the two precomputed duplication patterns; asymmetric ratios fall
  // through to the scalar loop below.
  //
  // The per-frame speedup versus the scalar loop is ~6-8x on broadcast
  // 4K 4:2:2 and is the reason the CPU color path is still a viable
  // fallback for GL-incompatible environments (headless, no GL 3.3,
  // WSL without hardware GL).  See the project memory for measurements.
  if (cb_stride_ratio == cr_stride_ratio
      && (cb_stride_ratio == 1 || cb_stride_ratio == 2)) {
    const __m256 vy_bias    = _mm256_set1_ps(y_bias);
    const __m256 vy_scale   = _mm256_set1_ps(y_scale);
    const __m256 vc_bias    = _mm256_set1_ps(c_bias);
    const __m256 vc_scale   = _mm256_set1_ps(c_scale);
    const __m256 vcr_to_r   = _mm256_set1_ps(coeffs.cr_to_r);
    const __m256 vcb_to_g   = _mm256_set1_ps(coeffs.cb_to_g);
    const __m256 vcr_to_g   = _mm256_set1_ps(coeffs.cr_to_g);
    const __m256 vcb_to_b   = _mm256_set1_ps(coeffs.cb_to_b);
    const __m256 v255       = _mm256_set1_ps(255.0f);
    const __m256 vhalf      = _mm256_set1_ps(0.5f);
    const __m256 vzero      = _mm256_setzero_ps();
    // Permute lanes used for 4:2:2 / 4:2:0 to replicate each of 4
    // chroma samples into adjacent luma positions.
    const __m256i vdup_chroma =
        _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3);

    // Scratch buffers for the 8-pixel RGB pack.  Hoisted out of the
    // loop body so the compiler doesn't need to prove that the stack
    // allocation is loop-invariant; harmless in any case and clearer.
    alignas(32) int32_t rtmp[8];
    alignas(32) int32_t gtmp[8];
    alignas(32) int32_t btmp[8];

    for (; x + 8 <= width; x += 8) {
      __m256i yi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y_row + x));

      __m256i cbi;
      __m256i cri;
      if (cb_stride_ratio == 1) {
        cbi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(cb_row + x));
        cri = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(cr_row + x));
      } else {  // cb_stride_ratio == 2
        const __m128i cb_half =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(cb_row + (x >> 1)));
        const __m128i cr_half =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(cr_row + (x >> 1)));
        // Promote to 256-bit (low half carries the 4 samples; high half
        // is don't-care) and duplicate via the permute mask.
        cbi = _mm256_permutevar8x32_epi32(_mm256_castsi128_si256(cb_half), vdup_chroma);
        cri = _mm256_permutevar8x32_epi32(_mm256_castsi128_si256(cr_half), vdup_chroma);
      }

      const __m256 yf  = _mm256_cvtepi32_ps(yi);
      const __m256 cbf = _mm256_cvtepi32_ps(cbi);
      const __m256 crf = _mm256_cvtepi32_ps(cri);

      const __m256 yn  = _mm256_mul_ps(_mm256_sub_ps(yf, vy_bias), vy_scale);
      const __m256 cbn = _mm256_mul_ps(_mm256_sub_ps(cbf, vc_bias), vc_scale);
      const __m256 crn = _mm256_mul_ps(_mm256_sub_ps(crf, vc_bias), vc_scale);

      // R = y + cr_to_r * cr
      __m256 r = _mm256_fmadd_ps(vcr_to_r, crn, yn);
      // G = y - cb_to_g * cb - cr_to_g * cr
      __m256 g = _mm256_fnmadd_ps(vcb_to_g, cbn, yn);
      g        = _mm256_fnmadd_ps(vcr_to_g, crn, g);
      // B = y + cb_to_b * cb
      __m256 b = _mm256_fmadd_ps(vcb_to_b, cbn, yn);

      // Scale to 0..255 with round-to-nearest, then clamp.
      r = _mm256_add_ps(_mm256_mul_ps(r, v255), vhalf);
      g = _mm256_add_ps(_mm256_mul_ps(g, v255), vhalf);
      b = _mm256_add_ps(_mm256_mul_ps(b, v255), vhalf);
      r = _mm256_min_ps(_mm256_max_ps(r, vzero), v255);
      g = _mm256_min_ps(_mm256_max_ps(g, vzero), v255);
      b = _mm256_min_ps(_mm256_max_ps(b, vzero), v255);

      // Truncating convert is fine now because we added 0.5 above.
      _mm256_store_si256(reinterpret_cast<__m256i*>(rtmp), _mm256_cvttps_epi32(r));
      _mm256_store_si256(reinterpret_cast<__m256i*>(gtmp), _mm256_cvttps_epi32(g));
      _mm256_store_si256(reinterpret_cast<__m256i*>(btmp), _mm256_cvttps_epi32(b));
      uint8_t* out = out_rgb + static_cast<size_t>(x) * 3;
      for (int i = 0; i < 8; ++i) {
        out[3 * i + 0] = static_cast<uint8_t>(rtmp[i]);
        out[3 * i + 1] = static_cast<uint8_t>(gtmp[i]);
        out[3 * i + 2] = static_cast<uint8_t>(btmp[i]);
      }
    }
    // Fall through to the scalar loop for the trailing 0..7 pixels.
  }
#endif

  for (; x < width; ++x) {
    const int32_t Y_i  = y_row[x];
    const int32_t Cb_i = cb_row[x / cb_stride_ratio];
    const int32_t Cr_i = cr_row[x / cr_stride_ratio];

    const float y_n  = (static_cast<float>(Y_i)  - y_bias) * y_scale;
    const float cb_n = (static_cast<float>(Cb_i) - c_bias) * c_scale;
    const float cr_n = (static_cast<float>(Cr_i) - c_bias) * c_scale;

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
