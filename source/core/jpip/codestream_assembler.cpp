// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.

#include "codestream_assembler.hpp"

#include "jpp_message.hpp"

namespace open_htj2k {
namespace jpip {

namespace {

// Marker codes (ITU-T T.800 Table A.1).
constexpr uint16_t kMarkerSOT = 0xFF90;
constexpr uint16_t kMarkerSOD = 0xFF93;
constexpr uint16_t kMarkerEOC = 0xFFD9;

inline void append_u16_be(std::vector<uint8_t> &out, uint16_t v) {
  out.push_back(static_cast<uint8_t>(v >> 8));
  out.push_back(static_cast<uint8_t>(v & 0xFF));
}

inline void append_u32_be(std::vector<uint8_t> &out, uint32_t v) {
  out.push_back(static_cast<uint8_t>((v >> 24) & 0xFF));
  out.push_back(static_cast<uint8_t>((v >> 16) & 0xFF));
  out.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
  out.push_back(static_cast<uint8_t>(v & 0xFF));
}

// Progression orders (COD SGcod byte 0 per Table A.16).
constexpr uint8_t kProgressionLRCP = 0;
constexpr uint8_t kProgressionRLCP = 1;
constexpr uint8_t kProgressionRPCL = 2;
constexpr uint8_t kProgressionPCRL = 3;
constexpr uint8_t kProgressionCPRL = 4;

}  // namespace

ReassembleStatus reassemble_codestream(const uint8_t *codestream, std::size_t len,
                                       const DataBinSet &set,
                                       const CodestreamIndex &idx,
                                       const CodestreamLayout &layout,
                                       const PacketLocator &locator,
                                       std::vector<uint8_t> &out) {
  (void)codestream;
  (void)len;

  // v1 only supports layer-subordinate progression orders where each
  // precinct's packets (all layers) are contiguous in the codestream.
  const uint8_t po = idx.progression_order();
  if (po == kProgressionLRCP || po == kProgressionRLCP) {
    return ReassembleStatus::UnsupportedProgression;
  }
  if (po != kProgressionPCRL && po != kProgressionRPCL && po != kProgressionCPRL) {
    return ReassembleStatus::UnsupportedProgression;
  }
  if (idx.use_SOP() || idx.use_EPH()) {
    return ReassembleStatus::UnsupportedFeature;
  }

  // The main header bin is required.
  if (!set.contains(kMsgClassMainHeader, 0)) return ReassembleStatus::MissingMainHeader;
  const std::vector<uint8_t> &main_header = set.get(kMsgClassMainHeader, 0);
  out.insert(out.end(), main_header.begin(), main_header.end());

  const uint16_t num_layers = idx.num_layers();

  // Walk tiles in declaration order — each tile's data goes into its own
  // tile-part.  v1 emits exactly one tile-part per tile, regardless of
  // how the source codestream was fragmented.
  const uint32_t nT = idx.num_tiles();
  for (uint32_t t = 0; t < nT; ++t) {
    // Build the tile-part body first so we know its length, then emit
    // SOT (with Psot) + tile-header-bin bytes + SOD + body.
    std::vector<uint8_t> body;

    // Walk precincts in the order the source codestream visited them.
    const auto order = locator.precincts_of_tile(static_cast<uint16_t>(t));
    for (const auto &pk : order) {
      const uint64_t I = idx.I(pk.t, pk.c, pk.r, pk.p_rc);
      if (set.contains(kMsgClassPrecinct, I)) {
        // Present: copy the data-bin bytes verbatim.  The emitter
        // guarantees these are the concatenation of every packet's
        // original bytes in layer order.
        const auto &bin = set.get(kMsgClassPrecinct, I);
        body.insert(body.end(), bin.begin(), bin.end());
      } else {
        // Absent: emit one "empty packet" header per layer.  Per spec
        // B.10, a packet header bit of 0 means "no code-blocks included",
        // and is padded to a byte boundary — 0x00 is the canonical
        // encoding when neither SOP nor EPH is in use.
        body.insert(body.end(), num_layers, 0x00);
      }
    }

    // Tile-header-bin: per §A.3.3 the decoder is told "may or may not
    // contain SOD", so we emit just the raw marker segments (no SOD;
    // we'll add it ourselves below).  Tile header must be present for
    // the tile to be decodable — even if the bin payload is empty.
    if (!set.contains(kMsgClassTileHeader, t)) return ReassembleStatus::LayoutMismatch;
    const auto &tile_hdr = set.get(kMsgClassTileHeader, t);

    // Psot = bytes from the start of the SOT marker to the end of the
    // tile-part data (inclusive of SOT, tile-header, SOD, and body).
    //   SOT = 12 bytes, SOD = 2 bytes.
    const uint64_t psot = 12u + tile_hdr.size() + 2u + body.size();
    if (psot > 0xFFFFFFFFull) return ReassembleStatus::UnsupportedFeature;  // Psot is 32-bit

    // Emit SOT marker (FF 90), Lsot (2 bytes, always 10), Isot (2),
    // Psot (4), TPsot (1), TNsot (1).  We emit a single tile-part per
    // tile, so TPsot = 0, TNsot = 1.
    append_u16_be(out, kMarkerSOT);
    append_u16_be(out, 10);                    // Lsot
    append_u16_be(out, static_cast<uint16_t>(t));  // Isot
    append_u32_be(out, static_cast<uint32_t>(psot));
    out.push_back(0);                          // TPsot
    out.push_back(1);                          // TNsot

    // Tile-part header marker segments.
    out.insert(out.end(), tile_hdr.begin(), tile_hdr.end());

    // SOD marker — no length field.
    append_u16_be(out, kMarkerSOD);

    // Tile-part body.
    out.insert(out.end(), body.begin(), body.end());
  }

  // EOC marker terminates the codestream.
  append_u16_be(out, kMarkerEOC);

  (void)layout;
  return ReassembleStatus::Ok;
}

}  // namespace jpip
}  // namespace open_htj2k
