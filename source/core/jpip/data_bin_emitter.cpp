// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.

#include "data_bin_emitter.hpp"

#include <algorithm>

namespace open_htj2k {
namespace jpip {

namespace {

// Upper bound on per-message payload size.  Interop testing showed that
// some external JPIP clients trip their own cache-segment bookkeeping
// when a single message carries more than ~1 KB of payload.  987 bytes
// is the chunk size other conforming servers use and fits inside a 1 KB
// segment buffer together with the encoded message header.  Data-bin
// contributions that fit in one chunk still emit a single `is_last=1`
// message; larger contributions are split across multiple messages with
// monotonically increasing `msg_offset`.
constexpr std::size_t kMaxMessagePayload = 987;

// Append a data-bin contribution as one or more JPP-stream messages.  For
// bins whose payload fits in `kMaxMessagePayload` this is a single message
// with `is_last=1`.  Larger bins are split into multiple messages with
// monotonically increasing `msg_offset`; only the final message has
// `is_last=1`.  The split is on byte boundaries — JPP messages do not
// constrain split points within the payload.
//
// When `win` is non-null, emission starts at payload offset `win->skip`
// and appends at most `win->budget` bytes (headers + payload).  The
// header-size reservation is conservative (kMessageHeaderMaxBytes), so a
// few bytes of budget can go unused at the response tail; the §C.6.1 cap
// is an upper bound, never an entitlement, so under-filling is safe.
// Empty bins (and `skip == payload_len` resumptions) emit the zero-length
// `is_last=1` message that declares the bin complete.
std::size_t append_message(MessageHeader hdr, const uint8_t *payload,
                           std::size_t payload_len,
                           MessageHeaderContext &ctx,
                           std::vector<uint8_t> &out,
                           BinWindow *win = nullptr) {
  std::size_t written = 0;
  std::size_t offset  = win ? std::min(win->skip, payload_len) : 0;
  const std::size_t budget = win ? win->budget : SIZE_MAX;
  const std::size_t first_offset = offset;
  bool complete = false;

  // Always try to emit at least one message, even for empty bins (so
  // callers can declare "empty and complete" via msg_length=0, is_last=1).
  for (;;) {
    const std::size_t remaining = payload_len - offset;
    const std::size_t avail     = (budget > written) ? budget - written : 0;
    if (avail <= kMessageHeaderMaxBytes) {
      if (win) win->budget_blocked = true;
      break;
    }
    const std::size_t this_len =
        std::min(std::min(remaining, kMaxMessagePayload), avail - kMessageHeaderMaxBytes);
    if (remaining > 0 && this_len == 0) {
      if (win) win->budget_blocked = true;
      break;
    }
    const bool is_last = (offset + this_len == payload_len);

    hdr.msg_offset = offset;
    hdr.msg_length = this_len;
    hdr.is_last    = is_last;

    const std::size_t prev = out.size();
    out.resize(prev + kMessageHeaderMaxBytes);
    const std::size_t hdr_bytes = encode_header(hdr, ctx, out.data() + prev);
    out.resize(prev + hdr_bytes);
    if (this_len > 0) {
      out.insert(out.end(), payload + offset, payload + offset + this_len);
    }
    written += hdr_bytes + this_len;
    offset  += this_len;
    if (is_last) { complete = true; break; }
  }

  if (win) {
    win->payload_sent = offset - first_offset;
    win->complete     = complete;
  }
  return written;
}

}  // namespace

std::size_t emit_main_header_databin(const uint8_t *codestream, std::size_t len,
                                     const CodestreamLayout &layout,
                                     MessageHeaderContext &ctx,
                                     std::vector<uint8_t> &out,
                                     BinWindow *win) {
  if (codestream == nullptr || layout.main_header_end == 0) return 0;
  if (layout.main_header_end > len) return 0;
  MessageHeader hdr{};
  hdr.class_id    = kMsgClassMainHeader;
  hdr.cs_n        = 0;
  hdr.in_class_id = 0;
  return append_message(hdr, codestream + layout.soc_offset,
                        layout.main_header_end - layout.soc_offset, ctx, out, win);
}

std::size_t emit_tile_header_databin(const uint8_t *codestream, std::size_t len,
                                     uint16_t tile_index,
                                     const CodestreamLayout &layout,
                                     MessageHeaderContext &ctx,
                                     std::vector<uint8_t> &out,
                                     BinWindow *win) {
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
  return append_message(hdr, payload.data(), payload.size(), ctx, out, win);
}

std::size_t emit_metadata_bin_zero(MessageHeaderContext &ctx,
                                   std::vector<uint8_t> &out,
                                   BinWindow *win) {
  MessageHeader hdr{};
  hdr.class_id    = kMsgClassMetadata;
  hdr.cs_n        = 0;
  hdr.in_class_id = 0;
  return append_message(hdr, /*payload=*/nullptr, 0, ctx, out, win);
}

std::size_t emit_precinct_databin(const uint8_t *codestream, std::size_t len,
                                  uint16_t t, uint16_t c, uint8_t r, uint32_t p_rc,
                                  const CodestreamIndex &idx,
                                  const PacketLocator &locator,
                                  MessageHeaderContext &ctx,
                                  std::vector<uint8_t> &out,
                                  BinWindow *win) {
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
  return append_message(hdr, payload.data(), payload.size(), ctx, out, win);
}

std::size_t emit_eor(EorReason reason, MessageHeaderContext &ctx, std::vector<uint8_t> &out) {
  // §D.3: the EOR message is not an Annex A message and not a class.
  // Wire format is:
  //     0x00 (identifier) | reason (1 byte) | body-length (VBAS) | body
  // We never emit a body, so the VBAS length is 0 — three bytes total.
  // MessageHeaderContext is untouched because EOR does not participate in
  // the dependent-form inheritance that Annex A message headers use.
  (void)ctx;
  const std::size_t before = out.size();
  out.push_back(0x00);
  out.push_back(static_cast<uint8_t>(reason));
  out.push_back(0x00);
  return out.size() - before;
}

}  // namespace jpip
}  // namespace open_htj2k
