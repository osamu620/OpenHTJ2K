// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.

#include "jpp_message.hpp"

namespace open_htj2k {
namespace jpip {

namespace {

// Number of bytes required to encode `v` as a Bin-ID VBAS.  Per §A.2.1,
// the Bin-ID's first byte holds 4 payload bits (the rest is the Class/CSn
// indicator, the is-last flag, and the VBAS continuation bit) and each
// subsequent byte holds 7 payload bits.  So k bytes encode 4 + 7(k-1) = 7k-3
// bits of in-class identifier.
std::size_t bin_id_bytes(uint64_t v) {
  if (v < 16u) return 1;
  std::size_t bits = 0;
  for (uint64_t x = v; x != 0; x >>= 1) ++bits;
  // Need 7k - 3 ≥ bits → k ≥ ceil((bits + 3) / 7).
  return (bits + 3 + 6) / 7;
}

// Emit the Bin-ID VBAS for an in-class identifier value with the given
// indicator bits (Table A.1: 01 / 10 / 11) and is-last flag.  Returns the
// number of bytes written.
std::size_t encode_bin_id(uint64_t in_class_id, uint8_t indicator_bb,
                          bool is_last, uint8_t *dst) {
  const std::size_t k = bin_id_bytes(in_class_id);
  // Top 4 bits of the in-class id under the 4 + 7(k-1) layout.
  const uint8_t top4   = static_cast<uint8_t>((in_class_id >> (7 * (k - 1))) & 0xFu);
  const uint8_t cont0  = (k > 1) ? 0x80u : 0x00u;
  const uint8_t bb_fld = static_cast<uint8_t>((indicator_bb & 0x3u) << 5);
  const uint8_t c_fld  = is_last ? 0x10u : 0x00u;
  dst[0] = static_cast<uint8_t>(cont0 | bb_fld | c_fld | top4);
  for (std::size_t i = 1; i < k; ++i) {
    const std::size_t shift = 7u * (k - 1 - i);
    const uint8_t payload = static_cast<uint8_t>((in_class_id >> shift) & 0x7Fu);
    const uint8_t cont    = (i + 1 < k) ? 0x80u : 0x00u;
    dst[i] = static_cast<uint8_t>(payload | cont);
  }
  return k;
}

// Decode the Bin-ID VBAS.  Returns true on success and writes the in-class
// id, indicator bits (1, 2, or 3), is-last flag, and bytes consumed.
bool decode_bin_id(const uint8_t *src, std::size_t src_len, uint64_t *in_class_id,
                   uint8_t *bb_out, bool *is_last, std::size_t *advance) {
  if (src_len < 1) return false;
  const uint8_t b0 = src[0];
  const uint8_t bb = static_cast<uint8_t>((b0 >> 5) & 0x3u);
  if (bb == 0u) return false;  // Table A.1: 00 is prohibited.
  *bb_out  = bb;
  *is_last = (b0 & 0x10u) != 0u;
  uint64_t v = static_cast<uint64_t>(b0 & 0xFu);
  std::size_t i = 1;
  if (b0 & 0x80u) {
    for (;;) {
      if (i >= src_len) return false;
      if (i >= kVbasMaxBytes) return false;
      // Same overflow guard as the standalone VBAS decoder.
      if ((v >> (64 - 7)) != 0u) return false;
      const uint8_t b = src[i++];
      v = (v << 7) | static_cast<uint64_t>(b & 0x7Fu);
      if ((b & 0x80u) == 0u) break;
    }
  }
  *in_class_id = v;
  *advance     = i;
  return true;
}

std::size_t encode_with_indicator(const MessageHeader &hdr, bool emit_class,
                                  bool emit_cs_n, uint8_t *dst) {
  // Indicator bits: 01 = neither, 10 = Class only, 11 = both.  CSn-only is
  // not representable, so callers that need to send CSn must also send Class.
  uint8_t bb;
  if (!emit_class && !emit_cs_n) bb = 0x1u;
  else if (emit_class && !emit_cs_n) bb = 0x2u;
  else bb = 0x3u;

  std::size_t pos = 0;
  pos += encode_bin_id(hdr.in_class_id, bb, hdr.is_last, dst + pos);
  if (emit_class) pos += vbas_encode(hdr.class_id, dst + pos);
  if (emit_cs_n)  pos += vbas_encode(hdr.cs_n, dst + pos);
  pos += vbas_encode(hdr.msg_offset, dst + pos);
  pos += vbas_encode(hdr.msg_length, dst + pos);
  if (msg_class_has_aux(hdr.class_id)) {
    pos += vbas_encode(hdr.aux, dst + pos);
  }
  return pos;
}

}  // namespace

std::size_t encode_header(const MessageHeader &hdr, MessageHeaderContext &ctx,
                          uint8_t *dst) {
  const bool class_changed = (hdr.class_id != ctx.last_class_id);
  const bool cs_n_changed  = (hdr.cs_n     != ctx.last_cs_n);
  // CSn change forces Class to ride along (Table A.1 has no CSn-only combo).
  const bool emit_class = class_changed || cs_n_changed;
  const bool emit_cs_n  = cs_n_changed;
  const std::size_t n = encode_with_indicator(hdr, emit_class, emit_cs_n, dst);
  ctx.last_class_id = hdr.class_id;
  ctx.last_cs_n     = hdr.cs_n;
  return n;
}

std::size_t encode_header_independent(const MessageHeader &hdr, uint8_t *dst) {
  return encode_with_indicator(hdr, /*emit_class=*/true, /*emit_cs_n=*/true, dst);
}

bool decode_header(const uint8_t *src, std::size_t src_len,
                   MessageHeaderContext &ctx, MessageHeader *out,
                   std::size_t *advance) {
  std::size_t pos = 0;
  std::size_t step = 0;

  uint64_t in_class_id = 0;
  uint8_t  bb          = 0;
  bool     is_last     = false;
  if (!decode_bin_id(src + pos, src_len - pos, &in_class_id, &bb, &is_last, &step)) {
    return false;
  }
  pos += step;

  // Class / CSn handling, per Table A.1: bb ∈ {01, 10, 11}; we already
  // rejected 00 in decode_bin_id.
  uint8_t  class_id = ctx.last_class_id;
  uint16_t cs_n     = ctx.last_cs_n;
  if (bb == 0x2u || bb == 0x3u) {
    uint64_t v = 0;
    if (!vbas_decode(src + pos, src_len - pos, &v, &step)) return false;
    pos += step;
    if (v > 0xFFu) return false;        // class id field is one byte wide
    class_id = static_cast<uint8_t>(v);
  }
  if (bb == 0x3u) {
    uint64_t v = 0;
    if (!vbas_decode(src + pos, src_len - pos, &v, &step)) return false;
    pos += step;
    if (v > 0xFFFFu) return false;      // codestream sequence is uint16
    cs_n = static_cast<uint16_t>(v);
  }

  uint64_t msg_offset = 0;
  if (!vbas_decode(src + pos, src_len - pos, &msg_offset, &step)) return false;
  pos += step;

  uint64_t msg_length = 0;
  if (!vbas_decode(src + pos, src_len - pos, &msg_length, &step)) return false;
  pos += step;

  uint64_t aux = 0;
  if (msg_class_has_aux(class_id)) {
    if (!vbas_decode(src + pos, src_len - pos, &aux, &step)) return false;
    pos += step;
  }

  out->class_id    = class_id;
  out->cs_n        = cs_n;
  out->in_class_id = in_class_id;
  out->msg_offset  = msg_offset;
  out->msg_length  = msg_length;
  out->aux         = aux;
  out->is_last     = is_last;

  ctx.last_class_id = class_id;
  ctx.last_cs_n     = cs_n;

  *advance = pos;
  return true;
}

}  // namespace jpip
}  // namespace open_htj2k
