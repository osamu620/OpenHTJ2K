// Shared by the fuzz targets of this directory.
//
// A target that takes options reads them from the LAST bytes of its input, so that the file
// under test starts at byte 0: a codestream followed by zero bytes is a seed that runs with
// the default options, and what the fuzzing engine changes in the file stays in place.
#ifndef OPENHTJ2K_FUZZ_COMMON_HPP
#define OPENHTJ2K_FUZZ_COMMON_HPP

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <vector>
#include "decoder.hpp"

namespace fuzz {

// What a target decodes.  Larger images are well formed, and slow under the sanitizers.
const uint64_t kMaxSamples    = 1u << 22;
const uint64_t kMaxTiles      = 1024;
const uint16_t kMaxComponents = 64;

// The library reports warnings through the message handler: a fuzzing run wants none.
inline void quiet(open_htj2k::message_level, const char *, void *) {}

// Keeps a result alive, so that the calls behind it are not optimised away.
inline void use(uint64_t v) {
  static std::atomic<uint64_t> sink(0);
  sink.fetch_add(v, std::memory_order_relaxed);
}

inline uint64_t be32(const uint8_t *p) {
  return (static_cast<uint64_t>(p[0]) << 24) | (static_cast<uint64_t>(p[1]) << 16)
         | (static_cast<uint64_t>(p[2]) << 8) | p[3];
}

// Tiles and samples the SIZ marker segment declares, for the first codestream in
// [p, p + len): a raw one, or the one inside a JPH or JP2 file.  Read without the library.
// False when there is no SIZ or the library refuses it anyway.
inline bool siz_declares(const uint8_t *p, size_t len, uint64_t &tiles, uint64_t &samples) {
  size_t s   = 0;
  bool found = false;
  for (; s + 42 <= len; ++s) {
    if (p[s] == 0xFF && p[s + 1] == 0x4F && p[s + 2] == 0xFF && p[s + 3] == 0x51) {  // SOC, SIZ
      found = true;
      break;
    }
  }
  if (!found) return false;
  const uint64_t X = be32(p + s + 8), Y = be32(p + s + 12), XO = be32(p + s + 16), YO = be32(p + s + 20);
  const uint64_t XT = be32(p + s + 24), YT = be32(p + s + 28), XTO = be32(p + s + 32),
                 YTO = be32(p + s + 36);
  const size_t C     = (static_cast<size_t>(p[s + 40]) << 8) | p[s + 41];
  if (XT == 0 || YT == 0 || X <= XO || Y <= YO || XTO > XO || YTO > YO) return false;
  tiles   = ((X - XTO + XT - 1) / XT) * ((Y - YTO + YT - 1) / YT);
  samples = 0;
  for (size_t c = 0; c < C && s + 42 + 3 * c + 3 <= len; ++c) {
    const uint64_t xr = p[s + 42 + 3 * c + 1], yr = p[s + 42 + 3 * c + 2];
    if (xr == 0 || yr == 0) return false;
    const uint64_t w = (X + xr - 1) / xr - (XO + xr - 1) / xr;
    const uint64_t h = (Y + yr - 1) / yr - (YO + yr - 1) / yr;
    if (h != 0 && w > (UINT64_MAX - samples) / h) {
      samples = UINT64_MAX;
      break;
    }
    samples += w * h;
  }
  return true;
}

inline bool too_many_tiles(const uint8_t *p, size_t len) {
  uint64_t tiles = 0, samples = 0;
  return siz_declares(p, len, tiles, samples) && tiles > kMaxTiles;
}

inline bool too_large(const uint8_t *p, size_t len) {
  uint64_t tiles = 0, samples = 0;
  return siz_declares(p, len, tiles, samples) && (tiles > kMaxTiles || samples > kMaxSamples);
}

// Decodes a codestream a target produced (the transcoder's output, a JPIP reassembly) and
// reads its rows.
inline void decode_result(const uint8_t *cs, size_t len) {
  if (len < 4 || too_many_tiles(cs, len)) return;
  try {
    open_htj2k::openhtj2k_decoder dec;
    dec.init(cs, len, 0, 1);
    dec.parse();
    uint64_t samples  = 0;
    const uint16_t nc = dec.get_num_component();
    for (uint16_t i = 0; i < nc; ++i) {
      samples += static_cast<uint64_t>(dec.get_component_width(i)) * dec.get_component_height(i);
    }
    if (nc == 0 || nc > kMaxComponents || samples > kMaxSamples) return;
    std::vector<uint32_t> w, h;
    std::vector<uint8_t> depth;
    std::vector<bool> sgn;
    uint64_t acc = 0;
    dec.invoke_line_based_stream(
        [&](uint32_t, int32_t *const *rows, uint16_t n) {
          for (size_t i = 0; i < n && i < w.size(); ++i) {
            if (rows[i] != nullptr && w[i] != 0) acc += static_cast<uint32_t>(rows[i][w[i] - 1]);
          }
        },
        w, h, depth, sgn);
    use(acc);
  } catch (std::exception &) {
  }
}

}  // namespace fuzz

#endif  // OPENHTJ2K_FUZZ_COMMON_HPP
