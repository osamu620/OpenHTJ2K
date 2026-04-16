// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.
//
// JPIP precinct geometry index — Phase 1, commit 1.
// Builds a (tile, component, resolution, intra-resolution-index) → JPIP
// sequence number `s` and in-class identifier `I` lookup from a parsed
// j2k_main_header.  No byte offsets yet — those land in commit 3 alongside
// the partial-decode mask.
#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace open_htj2k {
namespace jpip {

// 2-D size / coordinate pair used by the JPIP geometry API.  Kept as its own
// type rather than reusing the internal open_htj2k::element_siz so that this
// header does not leak library internals.
struct Point2 {
  uint32_t x = 0;
  uint32_t y = 0;
};

// Wire-level identity of a precinct: tile index, component index, resolution
// level (0 = LL, NL = finest), and intra-resolution raster index.
// Maps to JPIP sequence number `s` and in-class identifier `I` via the
// CodestreamIndex accessors.
struct PrecinctKey {
  uint16_t t    = 0;
  uint16_t c    = 0;
  uint8_t  r    = 0;
  uint32_t p_rc = 0;
};

// Per-(tile, component) precinct geometry, indexed by resolution r ∈ [0, NL].
// r = 0 is the LL subband; r = NL is the finest level.  npw/nph follow the
// same convention as j2k_resolution: the number of precincts on the subband
// reference grid for that resolution.
struct TileComponentInfo {
  uint8_t  NL = 0;
  std::vector<uint32_t> npw;
  std::vector<uint32_t> nph;
  // s_offset[r] = sum_{r' < r} npw[r'] * nph[r'] — prefix sum used to derive
  // the JPIP sequence number `s` from a (r, p_rc) pair.
  std::vector<uint32_t> s_offset;
  uint32_t total = 0;

  // Tile-component extent on the component's subsampled reference grid.
  // tc_x1 = tc_x0 + tile-component width, etc.
  Point2 tc_pos0;
  Point2 tc_pos1;

  // Per-resolution precinct partition parameters (size NL+1 each).  Mirrors
  // what j2k_resolution::create_precincts consumes.  PP = 2^log2PP.  respos0
  // is the precinct-grid origin on the subband reference grid of resolution r.
  std::vector<uint8_t>  log2PPx;
  std::vector<uint8_t>  log2PPy;
  std::vector<uint32_t> respos0_x;
  std::vector<uint32_t> respos0_y;
};

struct ImageGeometry {
  Point2 canvas_size;    // (Xsiz, Ysiz)
  Point2 canvas_origin;  // (XOsiz, YOsiz)
  Point2 tile_size;      // (XTsiz, YTsiz)
  Point2 tile_origin;    // (XTOsiz, YTOsiz)
};

class CodestreamIndex {
 public:
  // Build the index by parsing the main header of a raw J2C codestream
  // (no JP2/JPH container).  Throws std::runtime_error on parse failure.
  static std::unique_ptr<CodestreamIndex> build(const uint8_t *codestream, std::size_t len);

  uint32_t num_tiles_x()    const { return num_tiles_x_; }
  uint32_t num_tiles_y()    const { return num_tiles_y_; }
  uint32_t num_tiles()      const { return num_tiles_x_ * num_tiles_y_; }
  uint16_t num_components() const { return num_components_; }
  uint8_t  progression_order() const { return progression_; }
  const ImageGeometry &geometry() const { return geometry_; }
  // COD SPcod[4]: 0 = 9/7 (irreversible, lossy), 1 = 5/3 (reversible).
  bool     is_irreversible()   const { return is_irreversible_; }
  // Largest NL across all (t, c).  Used by resolve_view_window for picking
  // the discard level under round-direction "down".
  uint8_t  max_NL() const;
  // Component subsampling XRsiz/YRsiz (Point2.{x,y}).
  Point2   subsampling(uint16_t c) const;

  const TileComponentInfo &tile_component(uint16_t t, uint16_t c) const;

  // JPIP §A.3.2.1 sequence number within the tile-component.
  uint32_t s(uint16_t t, uint16_t c, uint8_t r, uint32_t p_rc) const;

  // JPIP §A.3.2.1 Eq A-1: I = t + (c + s · num_components) · num_tiles
  uint64_t I(uint16_t t, uint16_t c, uint8_t r, uint32_t p_rc) const;

  // Total precinct count summed over all (t, c).
  uint64_t total_precincts() const;

 private:
  CodestreamIndex() = default;

  uint16_t num_components_ = 0;
  uint32_t num_tiles_x_    = 0;
  uint32_t num_tiles_y_    = 0;
  uint8_t  progression_    = 0;
  bool     is_irreversible_ = false;
  ImageGeometry geometry_{};
  std::vector<Point2> subsampling_;  // size num_components_
  // Row-major [t * num_components + c].
  std::vector<TileComponentInfo> tcinfo_;
};

}  // namespace jpip
}  // namespace open_htj2k
