// jpip_vbas_check: ctest harness for the VBAS codec (ISO/IEC 15444-9 §A.2.1).
//
// Self-contained — runs every test internally, takes no input.  Exit 0 on
// every assertion passing, 1 on the first failure.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "vbas.hpp"

using open_htj2k::jpip::kVbasMaxBytes;
using open_htj2k::jpip::vbas_decode;
using open_htj2k::jpip::vbas_encode;

namespace {

int failures = 0;

#define CHECK(cond, ...)                                            \
  do {                                                              \
    if (!(cond)) {                                                  \
      std::fprintf(stderr, "FAIL [%s:%d] %s — ", __FILE__, __LINE__, #cond); \
      std::fprintf(stderr, __VA_ARGS__);                            \
      std::fprintf(stderr, "\n");                                   \
      ++failures;                                                   \
    }                                                               \
  } while (0)

// Encode v, then decode the result and check the round-trip yields v with
// the expected number of bytes consumed.
void roundtrip(uint64_t v, std::size_t expected_len) {
  uint8_t buf[kVbasMaxBytes] = {};
  const std::size_t enc = vbas_encode(v, buf);
  CHECK(enc == expected_len, "encode(%llu) wrote %zu bytes, expected %zu",
        static_cast<unsigned long long>(v), enc, expected_len);
  uint64_t out      = ~v;       // poison
  std::size_t adv   = 0;
  const bool ok     = vbas_decode(buf, enc, &out, &adv);
  CHECK(ok, "decode(encode(%llu)) returned false", static_cast<unsigned long long>(v));
  CHECK(out == v, "decode(encode(%llu)) yielded %llu",
        static_cast<unsigned long long>(v), static_cast<unsigned long long>(out));
  CHECK(adv == enc, "decode advance %zu != encoded length %zu", adv, enc);
}

// Decode `bytes` and check the result equals `expected`.  Useful for
// verifying byte-identical interop with reference encodings.
void decode_known(const std::vector<uint8_t> &bytes, uint64_t expected,
                  std::size_t expected_advance) {
  uint64_t out      = ~expected;
  std::size_t adv   = 0;
  const bool ok     = vbas_decode(bytes.data(), bytes.size(), &out, &adv);
  CHECK(ok, "decode_known returned false");
  CHECK(out == expected, "decode_known got %llu, expected %llu",
        static_cast<unsigned long long>(out),
        static_cast<unsigned long long>(expected));
  CHECK(adv == expected_advance, "decode_known advance %zu, expected %zu", adv,
        expected_advance);
}

// Decode should reject `bytes` (truncated, or value would overflow).
void reject(const std::vector<uint8_t> &bytes, const char *why) {
  uint64_t out    = 0;
  std::size_t adv = 0;
  const bool ok   = vbas_decode(bytes.data(), bytes.size(), &out, &adv);
  CHECK(!ok, "expected reject for %s, got value %llu (%zu bytes consumed)", why,
        static_cast<unsigned long long>(out), adv);
}

}  // namespace

int main() {
  // ── Round-trip values that exercise every byte-length boundary ─────────
  roundtrip(0,                          1);
  roundtrip(1,                          1);
  roundtrip(0x7F,                       1);
  roundtrip(0x80,                       2);
  roundtrip(0x3FFFu,                    2);
  roundtrip(0x4000u,                    3);
  roundtrip(0x1FFFFFu,                  3);
  roundtrip(0x200000u,                  4);
  roundtrip(0xFFFFFFFFu,                5);
  roundtrip(0x7FFFFFFFFFFFFFFFull,      9);
  // Full unsigned 64-bit range — needs the 10th byte.
  roundtrip(0xFFFFFFFFFFFFFFFFull,     10);
  roundtrip(0x8000000000000000ull,     10);

  // ── Byte-identical interop checks ──────────────────────────────────────
  // 0x80 → bytes 0x81 0x00 (cont, payload=1; last, payload=0 → 1<<7 == 128).
  decode_known({0x81, 0x00},                 128,   2);
  // 0x4000 → bytes 0x81 0x80 0x00 → 1<<14 == 16384.
  decode_known({0x81, 0x80, 0x00},           16384, 3);
  // Trailing garbage after a complete VBAS must not be consumed.
  decode_known({0x7F, 0xDE, 0xAD},           127,   1);

  // ── Rejection cases ────────────────────────────────────────────────────
  reject({},                                            "empty input");
  reject({0x80},                                        "single byte with continuation set, no follow-up");
  reject({0x81, 0x82},                                  "second byte also has continuation, no terminator");
  // 11-byte continuation chain — exceeds kVbasMaxBytes, even before considering value overflow.
  reject({0x81, 0x81, 0x81, 0x81, 0x81, 0x81, 0x81, 0x81, 0x81, 0x81, 0x00},
         "11-byte VBAS over limit");
  // 10-byte VBAS whose value would overflow uint64_t: payload 2 in the
  // first byte (so the implied bit 64 is set after concatenation).
  reject({0x82, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x00},
         "10-byte VBAS encoding > 2^64-1");

  if (failures == 0) {
    std::printf("OK vbas_check: all round-trip / interop / reject cases pass\n");
    return 0;
  }
  std::fprintf(stderr, "vbas_check: %d failures\n", failures);
  return 1;
}
