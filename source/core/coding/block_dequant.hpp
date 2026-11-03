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

#include "open_htj2k_typedef.hpp"
#include <cstddef>
#include <cstdint>

namespace open_htj2k {

// x = step * 2^31 + 0.5, the integer step of the ROI dequantization paths, truncated to
// int32.  A malformed QCD/QCC (a step far above any real one) or ATK (a gain that is not
// finite) puts x outside the int32 range, where a plain conversion is undefined:
// saturate instead (NaN gives INT32_MAX).
static inline int32_t roi_scale_to_i32(double x) {
  if (!(x < 2147483648.0)) return INT32_MAX;
  if (x <= -2147483649.0) return INT32_MIN;
  return static_cast<int32_t>(x);
}

// Scalar steps of the ROI dequantization paths, in the wrapping 32-bit arithmetic of
// their SIMD lanes (which the scalar tails must match).  Nothing wraps on most valid
// data, but the integer step multiply can, and a malformed codestream (background
// samples with bits a valid one cannot have, moved past bit 31 by the upshift) would
// overflow int32 everywhere.  The upshift gives 0 for a count of 32 or more, like the
// SIMD shifts.
static inline int32_t roi_upshift(int32_t v, uint8_t s) {
  return (s < 32) ? static_cast<int32_t>(static_cast<uint32_t>(v) << s) : 0;
}
static inline int32_t wrap_add(int32_t a, int32_t b) {
  return static_cast<int32_t>(static_cast<uint32_t>(a) + static_cast<uint32_t>(b));
}
static inline int32_t wrap_sub(int32_t a, int32_t b) {
  return static_cast<int32_t>(static_cast<uint32_t>(a) - static_cast<uint32_t>(b));
}
static inline int32_t wrap_mul(int32_t a, int32_t b) {
  return static_cast<int32_t>(static_cast<uint32_t>(a) * static_cast<uint32_t>(b));
}

// Dequantizes a decoded Part 1 code-block.  Without an ROI shift, every significant
// sample carries a reconstruction marker at bit p - 1, where p is the lowest
// bit-plane decoded for it (set by j2k_decode); that marker is the mid-point
// reconstruction value.
void j2k_dequant(const int32_t *sample_buf, size_t blksampl_stride, sprec_t *band_buf, uint32_t band_stride,
                 uint32_t width, uint32_t height, int32_t M_b, uint8_t ROIshift, uint8_t transformation,
                 float stepsize, bool dequant_i32 = false);
}  // namespace open_htj2k
