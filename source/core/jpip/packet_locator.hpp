// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.
//
// Packet locator — builds a per-(tile, component, resolution, precinct,
// layer) → absolute codestream byte-range map by driving the existing
// decoder with its packet observer hook.  The decoder already knows how
// to parse every packet header correctly (tagtrees, Lblock state, VLCs);
// we just tap the byte advances it produces and add the tile-part-body
// base offset so the ranges are absolute rather than tile_buf-relative.
//
// Used by the precinct data-bin emitter to carve per-precinct byte ranges
// out of the source codestream.  For codestreams whose progression order
// keeps the packets of a precinct contiguous (PCRL, RPCL, CPRL), each
// precinct resolves to a single byte range; other orders would yield a
// set of ranges that the emitter would have to concatenate.  The v1
// accessors expose both shapes.
#pragma once
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <tuple>
#include <vector>

#include "codestream_walker.hpp"
#include "precinct_index.hpp"

#if defined(_MSC_VER) && !defined(OHTJ2K_STATIC)
  #define OPENHTJ2K_JPIP_EXPORT __declspec(dllexport)
#else
  #define OPENHTJ2K_JPIP_EXPORT
#endif

namespace open_htj2k {
namespace jpip {

struct PacketByteRange {
  uint64_t offset = 0;  // absolute offset into the codestream
  uint64_t length = 0;
};

class OPENHTJ2K_JPIP_EXPORT PacketLocator {
 public:
  // Walk every packet in the codestream and record absolute byte ranges.
  // Takes the already-built CodestreamIndex + CodestreamLayout as context.
  // Returns nullptr on failure (decoder parse failed, or the codestream
  // has features this v1 locator does not support — e.g. PPM/PPT packet
  // headers stored outside the tile body).
  static std::unique_ptr<PacketLocator> build(const uint8_t *codestream,
                                              std::size_t len,
                                              const CodestreamIndex &idx,
                                              const CodestreamLayout &layout);

  // All byte ranges for the packets of this precinct, in the order the
  // decoder visited them (= layer 0, 1, …).  For a PCRL/RPCL/CPRL
  // codestream the ranges will be contiguous and in ascending offset
  // order; callers that care about that property are free to check.
  const std::vector<PacketByteRange> &packets_of(uint16_t t, uint16_t c, uint8_t r,
                                                 uint32_t p_rc) const;

  // Total number of packet byte ranges recorded.
  std::size_t size() const { return total_packets_; }

  // Precincts in the order the decoder first visited them within the
  // requested tile.  For layer-subordinate progression orders (PCRL,
  // RPCL, CPRL) this is also the order the precincts appear in the
  // tile-part body bytes — each precinct's packets (across every layer)
  // form a contiguous run in the codestream.  The returned keys all
  // share the requested tile index; an unknown tile returns an empty
  // vector.
  std::vector<PrecinctKey> precincts_of_tile(uint16_t t) const;

 private:
  PacketLocator() = default;

  using Key = std::tuple<uint16_t, uint16_t, uint8_t, uint32_t>;
  std::map<Key, std::vector<PacketByteRange>> packets_;
  // Precincts in first-appearance order, flat across all tiles.  The
  // tile index on each PrecinctKey distinguishes them; precincts_of_tile()
  // applies the obvious per-tile filter.
  std::vector<PrecinctKey> precinct_visit_order_;
  std::size_t total_packets_ = 0;
};

}  // namespace jpip
}  // namespace open_htj2k
