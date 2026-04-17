// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.

#include "jpp_parser.hpp"

#include "jpp_message.hpp"

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

bool parse_jpp_stream(const uint8_t *bytes, std::size_t len, DataBinSet *out) {
  if (out == nullptr) return false;
  MessageHeaderContext ctx;
  std::size_t pos = 0;
  while (pos < len) {
    MessageHeader hdr;
    std::size_t hdr_bytes = 0;
    if (!decode_header(bytes + pos, len - pos, ctx, &hdr, &hdr_bytes)) return false;
    pos += hdr_bytes;
    if (hdr.msg_length > len - pos) return false;  // truncated payload
    if (!out->append(hdr.class_id, hdr.in_class_id, hdr.msg_offset, bytes + pos,
                     static_cast<std::size_t>(hdr.msg_length), hdr.is_last)) {
      return false;
    }
    pos += static_cast<std::size_t>(hdr.msg_length);
  }
  return true;
}

}  // namespace jpip
}  // namespace open_htj2k
