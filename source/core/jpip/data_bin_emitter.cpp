// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.

#include "data_bin_emitter.hpp"

namespace open_htj2k {
namespace jpip {

namespace {

// Append a complete JPP-stream message (header + payload) covering
// `payload_len` bytes from `payload`.  Reserves enough capacity in `out`
// up front so the encoded header bytes don't trigger a reallocation
// while we're still writing them.
std::size_t append_message(MessageHeader hdr, const uint8_t *payload,
                           std::size_t payload_len,
                           MessageHeaderContext &ctx,
                           std::vector<uint8_t> &out) {
  hdr.msg_offset = 0;
  hdr.msg_length = payload_len;
  hdr.is_last    = true;

  const std::size_t prev = out.size();
  out.resize(prev + kMessageHeaderMaxBytes);
  const std::size_t hdr_bytes = encode_header(hdr, ctx, out.data() + prev);
  // Truncate the over-reservation back to the actual header length, then
  // append the payload bytes.
  out.resize(prev + hdr_bytes);
  out.insert(out.end(), payload, payload + payload_len);
  return hdr_bytes + payload_len;
}

}  // namespace

std::size_t emit_main_header_databin(const uint8_t *codestream, std::size_t len,
                                     const CodestreamLayout &layout,
                                     MessageHeaderContext &ctx,
                                     std::vector<uint8_t> &out) {
  if (codestream == nullptr || layout.main_header_end == 0) return 0;
  if (layout.main_header_end > len) return 0;
  MessageHeader hdr{};
  hdr.class_id    = kMsgClassMainHeader;
  hdr.cs_n        = 0;
  hdr.in_class_id = 0;
  return append_message(hdr, codestream + layout.soc_offset,
                        layout.main_header_end - layout.soc_offset, ctx, out);
}

std::size_t emit_tile_header_databin(const uint8_t *codestream, std::size_t len,
                                     uint16_t tile_index,
                                     const CodestreamLayout &layout,
                                     MessageHeaderContext &ctx,
                                     std::vector<uint8_t> &out) {
  if (codestream == nullptr) return 0;
  // Per §A.3.3, the server "is required to send a tile header data bin for
  // a tile even if the tile header is empty" — so we emit the message
  // (zero-length, is_last = true) whenever the tile exists in the layout,
  // and only return 0 when no tile-parts match the requested index.
  bool tile_exists = false;
  std::vector<uint8_t> payload;
  for (const auto &tp : layout.tile_parts) {
    if (tp.tile_index != tile_index) continue;
    tile_exists = true;
    // SOT marker segment is fixed at 12 bytes (2 marker + 2 Lsot + 8 body).
    constexpr std::size_t kSotSize = 12;
    if (tp.sot_offset + kSotSize > len || tp.header_end > len ||
        tp.sot_offset + kSotSize > tp.header_end) {
      continue;  // malformed tile-part — skip its bytes but still emit the bin
    }
    const uint8_t *first = codestream + tp.sot_offset + kSotSize;
    const uint8_t *last  = codestream + tp.header_end;
    payload.insert(payload.end(), first, last);
  }
  if (!tile_exists) return 0;

  MessageHeader hdr{};
  hdr.class_id    = kMsgClassTileHeader;
  hdr.cs_n        = 0;
  hdr.in_class_id = tile_index;
  return append_message(hdr, payload.data(), payload.size(), ctx, out);
}

std::size_t emit_metadata_bin_zero(MessageHeaderContext &ctx,
                                   std::vector<uint8_t> &out) {
  MessageHeader hdr{};
  hdr.class_id    = kMsgClassMetadata;
  hdr.cs_n        = 0;
  hdr.in_class_id = 0;
  return append_message(hdr, /*payload=*/nullptr, 0, ctx, out);
}

std::size_t emit_precinct_databin(const uint8_t *codestream, std::size_t len,
                                  uint16_t t, uint16_t c, uint8_t r, uint32_t p_rc,
                                  const CodestreamIndex &idx,
                                  const PacketLocator &locator,
                                  MessageHeaderContext &ctx,
                                  std::vector<uint8_t> &out) {
  if (codestream == nullptr) return 0;
  const auto &ranges = locator.packets_of(t, c, r, p_rc);
  if (ranges.empty()) return 0;
  // Concatenate each packet's bytes into a contiguous payload.  For
  // precinct-subordinate progression orders (PCRL, RPCL, CPRL) the ranges
  // are already contiguous in the source, so this is effectively a single
  // memcpy; for LRCP/RLCP the copies stitch scattered packets together.
  std::vector<uint8_t> payload;
  std::size_t total = 0;
  for (const auto &rg : ranges) total += static_cast<std::size_t>(rg.length);
  payload.reserve(total);
  for (const auto &rg : ranges) {
    if (rg.offset + rg.length > len) return 0;  // malformed
    payload.insert(payload.end(), codestream + rg.offset,
                   codestream + rg.offset + rg.length);
  }

  MessageHeader hdr{};
  hdr.class_id    = kMsgClassPrecinct;
  hdr.cs_n        = 0;
  hdr.in_class_id = idx.I(t, c, r, p_rc);
  return append_message(hdr, payload.data(), payload.size(), ctx, out);
}

std::size_t emit_eor(EorReason reason, MessageHeaderContext &ctx, std::vector<uint8_t> &out) {
  MessageHeader hdr;
  hdr.class_id    = kMsgClassEOR;
  hdr.in_class_id = 0;
  hdr.msg_offset  = 0;
  hdr.msg_length  = 1;
  hdr.is_last     = true;
  const uint8_t reason_byte = static_cast<uint8_t>(reason);
  return append_message(hdr, &reason_byte, 1, ctx, out);
}

}  // namespace jpip
}  // namespace open_htj2k
