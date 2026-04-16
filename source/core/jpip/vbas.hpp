// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.
//
// VBAS (Variable Byte Aligned Segment) codec, ISO/IEC 15444-9 §A.2.1.
//
// A VBAS encodes a non-negative integer as a sequence of bytes.  Each byte
// carries 7 bits of payload (bits 0..6) and a continuation flag in bit 7
// (1 = "another byte follows", 0 = "this is the last byte").  The value is
// formed by concatenating the 7-bit payloads in big-endian order.
//
// The smallest VBAS is one byte; the value zero is encoded as the single
// byte 0x00.
#pragma once
#include <cstddef>
#include <cstdint>

#if defined(_MSC_VER) && !defined(OHTJ2K_STATIC)
  #define OPENHTJ2K_JPIP_EXPORT __declspec(dllexport)
#else
  #define OPENHTJ2K_JPIP_EXPORT
#endif

namespace open_htj2k {
namespace jpip {

// 10 = ceil(64 / 7) — the maximum number of bytes a uint64_t can require.
constexpr std::size_t kVbasMaxBytes = 10;

// Encode `value` as a VBAS into `dst`.  The caller must provide at least
// kVbasMaxBytes writable bytes at `dst`.  Returns the number of bytes
// written (1 .. kVbasMaxBytes).
OPENHTJ2K_JPIP_EXPORT std::size_t vbas_encode(uint64_t value, uint8_t *dst);

// Decode a VBAS starting at `src` (which has `src_len` bytes available).
// On success: writes the decoded value to *out, the number of consumed
// bytes to *advance, and returns true.  On failure (truncated input, or
// the encoded value would overflow uint64_t): returns false and leaves
// *out / *advance untouched.
OPENHTJ2K_JPIP_EXPORT bool vbas_decode(const uint8_t *src, std::size_t src_len,
                                       uint64_t *out, std::size_t *advance);

}  // namespace jpip
}  // namespace open_htj2k
