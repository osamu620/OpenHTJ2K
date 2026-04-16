// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.
//
// Codestream reassembler — the inverse of the B3 emitters.
//
// Given a DataBinSet (from a JPIP client's cache), a CodestreamIndex (for
// precinct geometry + num_layers), a PacketLocator (for the precinct
// visit order the source codestream used), and the original
// CodestreamLayout, synthesise a sparse J2C codestream that the existing
// openhtj2k_decoder can decode without modification.
//
// "Sparse" here means: precincts whose class-0 data-bin is absent from
// the set get their packets replaced with one-bit "empty packet" headers
// (a single 0x00 byte per layer).  The decoder parses these as valid
// packets that include no code-blocks, and the resulting subband samples
// stay at zero — producing an image that's pixel-accurate inside the
// covered region and gradually-smoothed toward zero outside.  This
// matches the behaviour of openhtj2k_decoder::set_precinct_filter(), but
// uses the wire-format round-trip instead of a decoder-side filter.
//
// v1 scope: PCRL / RPCL / CPRL progression, no SOP/EPH, single
// tile-part per tile.  Rejects codestreams that use any feature outside
// this scope; the demo asset and conformance fixtures all fit.
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

#include "codestream_walker.hpp"
#include "jpp_parser.hpp"
#include "packet_locator.hpp"
#include "precinct_index.hpp"

#if defined(_MSC_VER) && !defined(OHTJ2K_STATIC)
  #define OPENHTJ2K_JPIP_EXPORT __declspec(dllexport)
#else
  #define OPENHTJ2K_JPIP_EXPORT
#endif

namespace open_htj2k {
namespace jpip {

// Status code returned by reassemble_codestream() — non-zero values are
// informational ("couldn't do the reassembly on this input"), not errors.
enum class ReassembleStatus : uint8_t {
  Ok                     = 0,
  MissingMainHeader      = 1,
  UnsupportedProgression = 2,  // LRCP / RLCP — packets not contiguous per precinct
  UnsupportedFeature     = 3,  // SOP / EPH / multi-tile-part / etc.
  LayoutMismatch         = 4,  // tile count / tile-part count mismatch
};

// Reassemble a sparse J2C codestream into `out`.
//
//   codestream, len : original codestream bytes — used only for the
//                     initial SOC/marker layout (the packet bytes are all
//                     sourced from the DataBinSet).
//   set             : client cache; must contain main-header and per-tile
//                     tile-header bins, plus the precinct bins the caller
//                     wants present.  Any precinct with no class-0 bin is
//                     replaced with empty packet placeholders.
//   idx, layout,
//   locator         : same objects the emitters were built from, so the
//                     reassembled codestream's packet order matches the
//                     source.
//   out             : destination buffer; the function appends to it.
//
// Returns ReassembleStatus::Ok when the reassembly completed.  Other
// values indicate the input did not satisfy v1's scope.
OPENHTJ2K_JPIP_EXPORT ReassembleStatus
reassemble_codestream(const uint8_t *codestream, std::size_t len,
                      const DataBinSet &set,
                      const CodestreamIndex &idx,
                      const CodestreamLayout &layout,
                      const PacketLocator &locator,
                      std::vector<uint8_t> &out);

}  // namespace jpip
}  // namespace open_htj2k
