// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.

#include "vbas.hpp"

namespace open_htj2k {
namespace jpip {

std::size_t vbas_encode(uint64_t value, uint8_t *dst) {
  if (value == 0) {
    dst[0] = 0;
    return 1;
  }
  // Number of 7-bit groups needed: floor(log2(value)/7) + 1.
  std::size_t n = 0;
  for (uint64_t v = value; v != 0; v >>= 7) ++n;
  // Big-endian: the most significant group first; continuation bit set on
  // every byte except the last.
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t shift = (n - 1 - i) * 7;
    const uint8_t group = static_cast<uint8_t>((value >> shift) & 0x7F);
    const bool more = (i + 1 < n);
    dst[i] = static_cast<uint8_t>(group | (more ? 0x80 : 0x00));
  }
  return n;
}

bool vbas_decode(const uint8_t *src, std::size_t src_len, uint64_t *out,
                 std::size_t *advance) {
  uint64_t v = 0;
  std::size_t i = 0;
  for (;;) {
    if (i >= src_len) return false;          // truncated input
    if (i >= kVbasMaxBytes) return false;    // implausibly long encoding
    // Before shifting the accumulated value left by 7, verify no payload
    // bits would be lost.  This rejects values that would overflow uint64_t
    // and accepts the legitimate full unsigned 64-bit range (which uses up
    // to 10 bytes).  At iteration 0 the check is trivially satisfied since
    // v == 0, so we skip it.
    if (i > 0 && (v >> (64 - 7)) != 0) return false;
    const uint8_t byte = src[i++];
    v = (v << 7) | static_cast<uint64_t>(byte & 0x7F);
    if ((byte & 0x80) == 0) {
      *out     = v;
      *advance = i;
      return true;
    }
  }
}

}  // namespace jpip
}  // namespace open_htj2k
