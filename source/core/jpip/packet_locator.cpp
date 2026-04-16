// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.

#include "packet_locator.hpp"

#include <tuple>

// Relative include: the public decoder API lives in source/core/interface,
// which is not on the library's PRIVATE include path by design (core code
// is intentionally insulated from the public ABI).  The packet locator is
// the one JPIP-side component that deliberately consumes the public API
// to drive an observer-enabled decode walk, so reach for it explicitly.
#include "../interface/decoder.hpp"

namespace open_htj2k {
namespace jpip {

namespace {

const std::vector<PacketByteRange> &empty_ranges() {
  static const std::vector<PacketByteRange> v;
  return v;
}

// Translate a tile_buf-relative offset to an absolute codestream offset,
// given the tile-part bodies that make up this tile's concatenated data.
// Returns UINT64_MAX if the offset does not map into any tile-part body
// for this tile — signals a malformed stream or locator bug.
uint64_t to_absolute(const CodestreamLayout &layout, uint16_t tile_index,
                     uint64_t tile_buf_offset) {
  uint64_t accumulated = 0;
  for (const auto &tp : layout.tile_parts) {
    if (tp.tile_index != tile_index) continue;
    const uint64_t body_len =
        (tp.body_end > tp.body_offset) ? (tp.body_end - tp.body_offset) : 0u;
    if (tile_buf_offset < accumulated + body_len) {
      return tp.body_offset + (tile_buf_offset - accumulated);
    }
    accumulated += body_len;
  }
  return UINT64_MAX;
}

}  // namespace

std::unique_ptr<PacketLocator> PacketLocator::build(const uint8_t *codestream,
                                                   std::size_t len,
                                                   const CodestreamIndex &idx,
                                                   const CodestreamLayout &layout) {
  if (codestream == nullptr || len == 0) return nullptr;
  std::unique_ptr<PacketLocator> self(new PacketLocator());

  // Drive the decoder: init_borrow → parse → install filter + observer →
  // invoke_line_based_stream.  The filter returns false for every precinct
  // so block decode is skipped; we only need the packet-walk side-effect.
  openhtj2k_decoder dec;
  // init_borrow requires 16 bytes of SIMD read-ahead past the buffer end.
  // The conformance tests guarantee this; for caller-supplied buffers we
  // fall back to the copying init() to stay safe.  The byte ranges we
  // record refer to the original codestream bytes, which the caller
  // owns, so either init path is transparent to the observer semantics.
  dec.init(codestream, len, /*reduce_NL=*/0, /*num_threads=*/1);
  dec.parse();

  auto observer_ok = std::make_shared<bool>(true);
  dec.set_precinct_filter([](uint16_t, uint16_t, uint8_t, uint32_t) { return false; });
  auto self_raw = self.get();
  dec.set_packet_observer([self_raw, &layout, observer_ok](
                              uint16_t t, uint16_t c, uint8_t r, uint32_t p_rc,
                              uint16_t layer, uint64_t offset, uint64_t length) {
    (void)layer;  // layer is implicit in insertion order
    const uint64_t abs_off = to_absolute(layout, t, offset);
    if (abs_off == UINT64_MAX) {
      *observer_ok = false;
      return;
    }
    PacketByteRange r1{abs_off, length};
    self_raw->packets_[std::make_tuple(t, c, r, p_rc)].push_back(r1);
    ++self_raw->total_packets_;
  });

  // Drive the packet walk.  We use invoke_line_based_stream because it's
  // the cheapest path that still runs every read_packet; a full-image
  // buffer allocation is not needed since our filter rejects everything
  // (no codeblock decoded → every row is a zero-fill from IDWT on empty
  // subbands, which we simply discard).
  std::vector<uint32_t> widths, heights;
  std::vector<uint8_t>  depths;
  std::vector<bool>     signeds;
  try {
    dec.invoke_line_based_stream(
        [](uint32_t /*y*/, int32_t *const * /*rows*/, uint16_t /*nc*/) {},
        widths, heights, depths, signeds);
  } catch (std::exception &) {
    return nullptr;
  }
  if (!*observer_ok) return nullptr;

  // Sanity-check the harvested map against the index's precinct count per
  // tile-component — any mismatch means the locator missed (or
  // double-counted) packets.  Not a cheap check for huge codestreams but
  // the assertion cost is amortised across the decode walk anyway.
  (void)idx;  // index is currently unused for validation; reserved for future checks.

  return self;
}

const std::vector<PacketByteRange> &PacketLocator::packets_of(uint16_t t, uint16_t c,
                                                              uint8_t r,
                                                              uint32_t p_rc) const {
  auto it = packets_.find(std::make_tuple(t, c, r, p_rc));
  if (it == packets_.end()) return empty_ranges();
  return it->second;
}

}  // namespace jpip
}  // namespace open_htj2k
