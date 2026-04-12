// Copyright (c) 2019 - 2026, Osamu Watanabe
// All rights reserved.
//
// Licensed under the same BSD 3-Clause terms as the rest of OpenHTJ2K.

#pragma once
#include <cstdint>

namespace open_htj2k {

// Per-component output descriptor for the direct-to-plane decode path.
// Used by invoke_line_based_direct() and decode_line_based_stream_planar().
struct PlanarOutputDesc {
  void    *base;         // uint8_t* or uint16_t*
  uint32_t stride;       // row stride in samples
  uint32_t width;        // component width in samples
  uint32_t height;       // component height in samples
  uint32_t yr;           // vertical subsampling factor (SIZ YRsiz)
  int32_t  dc;           // DC offset (e.g. 128 for 8-bit unsigned)
  int32_t  maxval, minval;
  int32_t  depth_shift;  // additional right-shift for 8-bit packing (bd - 8; 0 for 16-bit)
  bool     is_16bit;     // false: base is uint8_t*, true: base is uint16_t*
};

}  // namespace open_htj2k
