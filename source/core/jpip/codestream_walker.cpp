// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.

#include "codestream_walker.hpp"

namespace open_htj2k {
namespace jpip {

namespace {

constexpr uint16_t kMarkerSOC = 0xFF4F;
constexpr uint16_t kMarkerSOT = 0xFF90;
constexpr uint16_t kMarkerSOD = 0xFF93;
constexpr uint16_t kMarkerEOC = 0xFFD9;

inline uint16_t read_u16_be(const uint8_t *p) {
  return static_cast<uint16_t>((static_cast<uint16_t>(p[0]) << 8) | p[1]);
}
inline uint32_t read_u32_be(const uint8_t *p) {
  return (static_cast<uint32_t>(p[0]) << 24) | (static_cast<uint32_t>(p[1]) << 16)
       | (static_cast<uint32_t>(p[2]) << 8)  |  static_cast<uint32_t>(p[3]);
}

// Markers without a length field (per ITU-T T.800 Table A.1).  All other
// 0xFFxx markers have a 2-byte big-endian length immediately following.
inline bool marker_has_no_length(uint16_t m) {
  // SOC, SOD, EOC, EPH, plus the reserved single-byte 0xFF{30..3F} markers.
  if (m == kMarkerSOC || m == kMarkerSOD || m == kMarkerEOC) return true;
  if (m == 0xFF92 /* EPH */) return true;
  const uint8_t low = static_cast<uint8_t>(m & 0xFF);
  return (low >= 0x30 && low <= 0x3F);
}

}  // namespace

bool walk_codestream(const uint8_t *bytes, std::size_t len, CodestreamLayout *out) {
  if (out == nullptr || bytes == nullptr || len < 2) return false;

  // Must start with SOC.  We're strict here — any prefix bytes the caller
  // wants to skip should be stripped before calling.
  if (read_u16_be(bytes) != kMarkerSOC) return false;

  out->soc_offset      = 0;
  out->main_header_end = 0;
  out->eoc_offset      = len;
  out->tile_parts.clear();

  std::size_t i = 2;       // past the SOC
  bool first_sot_seen = false;

  while (i + 2 <= len) {
    const uint16_t m = read_u16_be(bytes + i);

    if (m == kMarkerEOC) {
      out->eoc_offset = i;
      return true;
    }

    if (m == kMarkerSOT) {
      // SOT marker segment: 2 (marker) + 2 (Lsot) + Isot(2) + Psot(4)
      // + TPsot(1) + TNsot(1) = 12 bytes total.  Lsot is fixed at 10.
      if (i + 12 > len) return true;  // truncated SOT: stop, return what we have
      const uint16_t Lsot = read_u16_be(bytes + i + 2);
      if (Lsot != 10) return true;     // malformed
      const uint16_t Isot = read_u16_be(bytes + i + 4);
      const uint32_t Psot = read_u32_be(bytes + i + 6);
      const uint8_t  TPsot = bytes[i + 10];
      const uint8_t  TNsot = bytes[i + 11];

      if (!first_sot_seen) {
        out->main_header_end = i;
        first_sot_seen       = true;
      }

      // Walk forward from past-SOT looking for the SOD that closes this
      // tile-part header.
      std::size_t j = i + 12;
      while (j + 2 <= len) {
        const uint16_t m2 = read_u16_be(bytes + j);
        if (m2 == kMarkerSOD) {
          TilePartLocation tp{};
          tp.sot_offset    = i;
          tp.header_end    = j;
          tp.body_offset   = j + 2;  // skip SOD marker (2 bytes, no length field)
          // Psot is the total tile-part length from the start of the SOT
          // marker to the end of the tile-part data.  When Psot == 0 the
          // last tile-part of the codestream extends to EOC (or EOF).
          if (Psot != 0) {
            tp.body_end = i + Psot;
            if (tp.body_end > len) tp.body_end = len;  // truncated
          } else {
            tp.body_end = len;  // probe until EOC during next iteration
          }
          tp.tile_index    = Isot;
          tp.tile_part_idx = TPsot;
          tp.tile_part_cnt = TNsot;
          out->tile_parts.push_back(tp);
          // Advance to the next marker after this tile-part body.
          i = tp.body_end;
          break;
        }
        if (marker_has_no_length(m2)) {
          // Single-byte markers inside a tile-part header are unusual;
          // skip the 2-byte marker code.
          j += 2;
          continue;
        }
        if (j + 4 > len) return true;
        const uint16_t Lmar = read_u16_be(bytes + j + 2);
        if (Lmar < 2 || j + 2 + Lmar > len) return true;
        j += 2 + Lmar;
      }
      // If we ran off the end without finding SOD, drop out.
      if (j + 2 > len) return true;
      continue;
    }

    if (marker_has_no_length(m)) {
      // SOC/SOD/EOC are handled above; this catches reserved single-byte
      // markers.  Just skip past them.
      i += 2;
      continue;
    }

    // Generic length-bearing main-header marker (SIZ, COD, COC, QCD, QCC,
    // CAP, COM, PLM, PPM, TLM, RGN, POC, …).
    if (i + 4 > len) return true;
    const uint16_t Lmar = read_u16_be(bytes + i + 2);
    if (Lmar < 2 || i + 2 + Lmar > len) return true;
    i += 2 + Lmar;
  }

  return true;
}

}  // namespace jpip
}  // namespace open_htj2k
