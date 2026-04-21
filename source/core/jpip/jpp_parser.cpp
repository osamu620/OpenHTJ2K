// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.

#include "jpp_parser.hpp"

#include "jpp_message.hpp"
#include "vbas.hpp"

namespace open_htj2k {
namespace jpip {

namespace {

const std::vector<uint8_t> &empty_bytes() {
  static const std::vector<uint8_t> v;
  return v;
}

}  // namespace

bool DataBinSet::contains(uint8_t class_id, uint64_t in_class_id) const {
  return bins_.find({class_id, in_class_id}) != bins_.end();
}

bool DataBinSet::is_complete(uint8_t class_id, uint64_t in_class_id) const {
  auto it = bins_.find({class_id, in_class_id});
  return it != bins_.end() && it->second.is_last;
}

const std::vector<uint8_t> &DataBinSet::get(uint8_t class_id, uint64_t in_class_id) const {
  auto it = bins_.find({class_id, in_class_id});
  if (it == bins_.end()) return empty_bytes();
  return it->second.bytes;
}

std::vector<std::pair<uint8_t, uint64_t>> DataBinSet::keys() const {
  std::vector<std::pair<uint8_t, uint64_t>> out;
  out.reserve(bins_.size());
  for (const auto &kv : bins_) out.push_back(kv.first);
  return out;
}

void DataBinSet::merge_from(const DataBinSet &other) {
  for (const auto &kv : other.bins_) {
    auto it = bins_.find(kv.first);
    if (it == bins_.end()) {
      bins_[kv.first] = kv.second;
    } else if (kv.second.bytes.size() > it->second.bytes.size()) {
      it->second = kv.second;
    } else if (kv.second.is_last && !it->second.is_last) {
      it->second.is_last = true;
    }
  }
}

bool DataBinSet::erase(uint8_t class_id, uint64_t in_class_id) {
  return bins_.erase({class_id, in_class_id}) > 0;
}

bool DataBinSet::append(uint8_t class_id, uint64_t in_class_id, uint64_t msg_offset,
                        const uint8_t *payload, std::size_t payload_len, bool is_last) {
  const Key key{class_id, in_class_id};
  Entry &e = bins_[key];
  // Reject any further bytes on a bin the server has already closed with
  // is_last — even if the new message also carries is_last=true and a
  // zero-length payload.  That's an encoder / protocol bug on our end.
  if (e.is_last && payload_len > 0) return false;
  // Strict in-order: msg_offset must equal whatever we've accumulated so
  // far.  Anything else is a non-contiguous (out-of-order or duplicate)
  // delivery which we reject in v1.
  if (msg_offset != e.bytes.size()) return false;
  if (payload_len > 0 && payload != nullptr) {
    e.bytes.insert(e.bytes.end(), payload, payload + payload_len);
  }
  if (is_last) e.is_last = true;
  return true;
}

// Try to decode exactly one JPP-stream message (Annex A message or an EOR)
// at the start of `src`.  On a clean decode, appends the payload to `*out`
// (or flags EOR on the set), writes the consumed byte count to `*consumed`,
// and returns Decoded::Ok.  Returns Decoded::NeedMore when the buffer is
// truncated but may still be a valid message once more bytes arrive, and
// Decoded::Malformed when no amount of additional bytes could make the
// current position parseable.  `ctx` is only advanced on Ok.
enum class Decoded { Ok, NeedMore, Malformed };
namespace {
Decoded try_decode_one_message(const uint8_t *src, std::size_t src_len,
                               MessageHeaderContext &ctx, DataBinSet *out,
                               std::size_t *consumed) {
  if (src_len == 0) return Decoded::NeedMore;

  // §D.3 EOR: identifier = 0x00, reason = 1 byte, body_length = VBAS, body.
  if (src[0] == 0x00) {
    if (src_len < 2) return Decoded::NeedMore;
    const uint8_t reason = src[1];
    uint64_t body_len = 0;
    std::size_t vbas_adv = 0;
    if (!vbas_decode(src + 2, src_len - 2, &body_len, &vbas_adv)) {
      // VBAS failure could be truncation or overflow.  Truncation is only
      // possible while fewer than kVbasMaxBytes bytes follow the reason
      // byte.  Beyond that, the VBAS is definitively malformed.
      return (src_len - 2 < kVbasMaxBytes) ? Decoded::NeedMore : Decoded::Malformed;
    }
    if (body_len > src_len - 2 - vbas_adv) return Decoded::NeedMore;
    out->set_eor(reason);
    *consumed = 2 + vbas_adv + static_cast<std::size_t>(body_len);
    return Decoded::Ok;
  }

  // Annex A message.  decode_header only commits its writes to `ctx` at
  // the very end (after all VBAS fields decode cleanly), so on failure
  // `ctx` is left untouched — safe to retry once more bytes arrive.  We
  // still snapshot before calling, because on the happy path we need to
  // roll ctx back if the header decodes but the payload is truncated
  // (otherwise the retry would see the wrong last_class_id / last_cs_n
  // and mis-decode any dependent-form successor).
  const MessageHeaderContext ctx_snap = ctx;
  MessageHeader hdr;
  std::size_t hdr_bytes = 0;
  if (!decode_header(src, src_len, ctx, &hdr, &hdr_bytes)) {
    // If we haven't yet seen a full worst-case header window we cannot
    // distinguish truncation from a real header-format violation; assume
    // truncation and wait for more bytes.
    if (src_len < kMessageHeaderMaxBytes) return Decoded::NeedMore;
    return Decoded::Malformed;
  }
  if (hdr.msg_length > src_len - hdr_bytes) {
    ctx = ctx_snap;
    return Decoded::NeedMore;
  }
  if (!out->append(hdr.class_id, hdr.in_class_id, hdr.msg_offset,
                   src + hdr_bytes, static_cast<std::size_t>(hdr.msg_length),
                   hdr.is_last)) {
    ctx = ctx_snap;
    return Decoded::Malformed;
  }
  *consumed = hdr_bytes + static_cast<std::size_t>(hdr.msg_length);
  return Decoded::Ok;
}
}  // namespace

bool parse_jpp_stream(const uint8_t *bytes, std::size_t len, DataBinSet *out) {
  if (out == nullptr) return false;
  MessageHeaderContext ctx;
  std::size_t pos = 0;
  while (pos < len) {
    std::size_t consumed = 0;
    const Decoded d = try_decode_one_message(bytes + pos, len - pos, ctx, out, &consumed);
    if (d != Decoded::Ok) return false;  // NeedMore on a whole-buffer caller == truncated
    pos += consumed;
    if (out->has_eor()) break;  // §D.3 terminates the stream
  }
  return true;
}

// ── StreamingJppParser ────────────────────────────────────────────────────

bool StreamingJppParser::feed(const uint8_t *bytes, std::size_t len, DataBinSet *out) {
  if (out == nullptr) return false;
  if (len == 0) return true;

  // If we have leftover bytes from the previous chunk, join them with the
  // new bytes into a scratch buffer and parse from there.  For the common
  // steady-state case (pending_ is empty because each chunk ends on a
  // message boundary) we parse straight from `bytes` with no copy.
  const uint8_t *p = bytes;
  std::size_t avail = len;
  std::vector<uint8_t> joined;
  if (!pending_.empty()) {
    joined.reserve(pending_.size() + len);
    joined.insert(joined.end(), pending_.begin(), pending_.end());
    joined.insert(joined.end(), bytes, bytes + len);
    p = joined.data();
    avail = joined.size();
    pending_.clear();
  }

  std::size_t pos = 0;
  while (pos < avail) {
    std::size_t consumed = 0;
    const Decoded d =
        try_decode_one_message(p + pos, avail - pos, ctx_, out, &consumed);
    if (d == Decoded::Malformed) return false;
    if (d == Decoded::NeedMore) break;
    pos += consumed;
    if (out->has_eor()) break;  // further bytes after EOR would be a protocol violation
  }

  if (pos < avail) {
    pending_.assign(p + pos, p + avail);
  }
  return true;
}

void StreamingJppParser::reset() {
  pending_.clear();
  ctx_.clear();
}

}  // namespace jpip
}  // namespace open_htj2k
