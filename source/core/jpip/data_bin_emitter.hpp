// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.
//
// Data-bin emitters for the JPP-stream wire format (ISO/IEC 15444-9 §A.3).
//
// Each emit_*_databin() function appends one or more JPP-stream messages
// to the caller-supplied buffer.  A `MessageHeaderContext` is threaded
// through so dependent-form headers (Class/CSn omission) work across
// successive data-bins.
//
// The v1 emitters always pack each data-bin into a single message — i.e.
// msg_offset = 0, msg_length = (bin size), is_last = true.  Splitting a
// bin across multiple messages is allowed by the spec (and useful for
// flow-control and progressive delivery) but unnecessary for the local
// round-trip use case in PHASE2_PLAN.md.
//
// Precinct data-bins (class 0/1) are not emitted by this header — they
// require packet-header parsing and live in a follow-up commit; see
// PHASE2_PLAN.md item B3-precincts.
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

#include "codestream_walker.hpp"
#include "jpp_message.hpp"
#include "packet_locator.hpp"
#include "precinct_index.hpp"

#if defined(_MSC_VER) && !defined(OHTJ2K_STATIC)
  #define OPENHTJ2K_JPIP_EXPORT __declspec(dllexport)
#else
  #define OPENHTJ2K_JPIP_EXPORT
#endif

namespace open_htj2k {
namespace jpip {

// Emit the main-header data-bin (class 6, in-class id 0).  Payload is the
// codestream bytes from the SOC marker (inclusive) up to the first SOT
// marker (exclusive).  Returns the number of bytes appended to `out`, or 0
// on error (malformed codestream / no SOT found).
OPENHTJ2K_JPIP_EXPORT std::size_t
emit_main_header_databin(const uint8_t *codestream, std::size_t len,
                         const CodestreamLayout &layout,
                         MessageHeaderContext &ctx,
                         std::vector<uint8_t> &out);

// Emit the tile-header data-bin (class 2, in-class id = `tile_index`).
// Per §A.3.3, the bin contains all tile-part-header marker segments for
// the tile concatenated, with SOT excluded; SOD inclusion is optional and
// this implementation excludes them.  When a tile has multiple tile-parts
// (TPsot > 0 in any of them), all matching tile-parts contribute their
// header bytes in order.  Returns 0 if the tile has no tile-parts in the
// layout.
OPENHTJ2K_JPIP_EXPORT std::size_t
emit_tile_header_databin(const uint8_t *codestream, std::size_t len,
                         uint16_t tile_index,
                         const CodestreamLayout &layout,
                         MessageHeaderContext &ctx,
                         std::vector<uint8_t> &out);

// Emit metadata-bin 0 (class 8, in-class id 0).  Per §A.3.6.1 the server
// must always send this bin; for raw .j2c codestreams (no JP2/JPH boxes)
// it is empty.  This implementation emits a single zero-length, is_last
// message.
OPENHTJ2K_JPIP_EXPORT std::size_t
emit_metadata_bin_zero(MessageHeaderContext &ctx, std::vector<uint8_t> &out);

// Emit a precinct data-bin (class 0, in-class id = `I` per §A.3.2.1
// Eq. A-1) covering every packet of this precinct across every layer.
// The payload is the concatenation of the ranges reported by
// `locator.packets_of(t, c, r, p_rc)`.  For PCRL/RPCL/CPRL codestreams
// those ranges are contiguous in the source bytes; for LRCP/RLCP they
// are scattered — the emitter copies each range in the locator's order
// (= insertion order = layer order) so the v1 output still round-trips
// through the decoder when the receiver's reassembler stitches them
// back into a sparse codestream.
//
// Returns 0 if the precinct has no recorded packets (e.g. empty
// resolution or out-of-range index) — in that case the caller should
// emit a zero-length is_last message separately if the spec requires it.
OPENHTJ2K_JPIP_EXPORT std::size_t
emit_precinct_databin(const uint8_t *codestream, std::size_t len,
                      uint16_t t, uint16_t c, uint8_t r, uint32_t p_rc,
                      const CodestreamIndex &idx,
                      const PacketLocator &locator,
                      MessageHeaderContext &ctx,
                      std::vector<uint8_t> &out);

// Emit an End-of-Response (EOR) message (class 7, §A.3) with the given
// reason code.  The EOR message signals the end of a server response and
// carries a one-byte reason code as its body.
OPENHTJ2K_JPIP_EXPORT std::size_t
emit_eor(EorReason reason, MessageHeaderContext &ctx, std::vector<uint8_t> &out);

}  // namespace jpip
}  // namespace open_htj2k
