// Unit test for the decoder's 4 GiB codestream-length bound.
//
// j2c_src_memory copies an attacker-controlled codestream length into a buffer
// whose accounting (len/cap) is uint32_t and which adds a 16-byte SIMD
// over-read pad.  A length in (UINT32_MAX-16, UINT32_MAX] would wrap
// `length + 16` to a tiny allocation, and a size_t larger than UINT32_MAX would
// be truncated by the size_t->uint32_t cast at the call site — either way the
// following memcpy of the full length overflows the heap.  alloc_memory() now
// rejects such lengths by throwing.
//
// The test drives this through the public decoder constructor with a small
// raw-codestream buffer but a lied-about length: jph_is_signature rejects the
// SOC header after reading 12 bytes (so detection does not over-read the small
// buffer on the lied length), and the raw-codestream path then calls
// alloc_memory(length), which must throw before the copy runs.  This exercises
// only the exported API, so it links and runs on every platform and needs no
// ~4 GiB input.  A decoder missing the bound instead wraps/truncates the
// allocation and over-reads the buffer (crash), which the test harness flags.
#include "decoder.hpp"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {
bool ctor_rejects(size_t length) {
  std::vector<uint8_t> buf(64, 0);
  buf[0] = 0xFF;
  buf[1] = 0x4F;  // SOC — marks a raw J2K codestream (not a JP2/JPH container)
  buf[2] = 0xFF;
  buf[3] = 0x51;  // SIZ
  try {
    open_htj2k::openhtj2k_decoder dec(buf.data(), length, 0, 1);
    return false;
  } catch (...) {
    return true;
  }
}
}  // namespace

int main() {
  const size_t U32 = static_cast<size_t>(UINT32_MAX);
  int fail         = 0;

  // Length in the `+16` wrap window: a uint32_t allocation rounds down to a
  // tiny buffer.
  if (!ctor_rejects(U32)) {
    printf("FAIL: decoder(length=UINT32_MAX) was not rejected\n");
    fail = 1;
  }
  // Length larger than uint32_t: a size_t->uint32_t cast would truncate.
  if (!ctor_rejects(U32 + 4096)) {
    printf("FAIL: decoder(length=UINT32_MAX+4096) was not rejected\n");
    fail = 1;
  }
  // A legitimate length must still be accepted (no false positive).
  if (ctor_rejects(64)) {
    printf("FAIL: decoder(length=64) rejected a valid size\n");
    fail = 1;
  }

  if (!fail) printf("codestream_bounds_check: all cases passed\n");
  return fail;
}
