// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.
//
// Lightweight codestream marker walker for JPIP wire-format work.
//
// Identifies the byte offsets of SOC, SOT, SOD, and EOC markers — enough to
// carve out main-header / tile-header / tile-part-body byte ranges without
// invoking the full j2k_main_header parser (which carries internal state
// the JPIP code does not need).
//
// Stops at the first malformed marker; reports what it managed to find.
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

#if defined(_MSC_VER) && !defined(OHTJ2K_STATIC)
  #define OPENHTJ2K_JPIP_EXPORT __declspec(dllexport)
#else
  #define OPENHTJ2K_JPIP_EXPORT
#endif

namespace open_htj2k {
namespace jpip {

struct TilePartLocation {
  std::size_t sot_offset    = 0;  // start of the SOT marker (FF 90)
  std::size_t header_end    = 0;  // start of the SOD marker (= one past last header byte)
  std::size_t body_offset   = 0;  // start of packet data (= header_end + 2 to skip SOD)
  std::size_t body_end      = 0;  // one past last byte of this tile-part's body
  uint16_t    tile_index    = 0;
  uint8_t     tile_part_idx = 0;  // TPsot field — 0 for the first tile-part of a tile
  uint8_t     tile_part_cnt = 0;  // TNsot field — total tile-parts for this tile, or 0 if unknown
};

struct CodestreamLayout {
  // SOC marker offset (= 0 in a well-formed codestream).
  std::size_t soc_offset = 0;
  // The byte just past the main-header — the offset of the first SOT.
  // The main-header data-bin payload is bytes [soc_offset, main_header_end).
  std::size_t main_header_end = 0;
  // EOC marker offset, or `len` if the codestream is truncated / lacks EOC.
  std::size_t eoc_offset = 0;
  // Tile-parts in declaration order (which is also their byte order).
  std::vector<TilePartLocation> tile_parts;
};

// Walk the codestream and return its layout.  Stops on the first malformed
// marker but still returns whatever was collected up to that point — the
// caller can validate by checking that `tile_parts` is non-empty and that
// `eoc_offset > soc_offset`.  Returns `false` on hard errors (e.g. no SOC
// at the start) and `true` otherwise.
OPENHTJ2K_JPIP_EXPORT bool walk_codestream(const uint8_t *bytes, std::size_t len,
                                           CodestreamLayout *out);

}  // namespace jpip
}  // namespace open_htj2k
