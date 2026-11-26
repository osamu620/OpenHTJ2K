// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.
//
// Licensed under the same BSD 3-Clause terms as the rest of OpenHTJ2K.

// Payload header parsing of the RFC 9828 RTP receiver (source/apps/rtp_recv/rfc9828_parser.cpp,
// also built into the WebAssembly RTP module).  The receiver itself needs a window system, so
// this checks the parser on its own:
//   - the fixed Main Packet header fields;
//   - §8.4: a Main Packet with XTRAC != 0 is accepted, its XTRAB (XTRAC 4-byte words) is
//     skipped unread, and the codestream bytes start after it; an XTRAB running past the
//     payload is refused;
//   - the Body Packet header still starts its codestream at byte 8.
// Exits 0 when every case holds.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "rfc9828_parser.hpp"

using namespace open_htj2k::rtp_recv;

namespace {

int failures = 0;

void check(bool ok, const char *what) {
  if (!ok) {
    std::printf("FAIL: %s\n", what);
    ++failures;
  }
}

// An 8-byte Main Packet header: MH=3, ORDH=4 (PCRL), P=1, PTSTAMP=0x0ABC, ESEQ=0x42,
// R=1 S=1 RANGE=1, PRIMS/TRANS/MAT=1, and the given XTRAC.
std::vector<uint8_t> main_header(uint8_t xtrac) {
  std::vector<uint8_t> h(8, 0);
  h[0] = static_cast<uint8_t>((3u << 6) | 4u);
  h[1] = static_cast<uint8_t>(0x80 | ((xtrac & 0x07u) << 4) | ((0x0ABC >> 8) & 0x0F));
  h[2] = static_cast<uint8_t>(0x0ABC & 0xFF);
  h[3] = 0x42;
  h[4] = static_cast<uint8_t>(0x80 | 0x40 | 0x01);
  h[5] = 1;
  h[6] = 1;
  h[7] = 1;
  return h;
}

bool fields_ok(const MainPacketHeader &m) {
  return m.mh == 3 && m.ordh == ORDH_PCRL_RESYNC && m.p && m.ptstamp == 0x0ABC && m.eseq == 0x42 && m.r
         && m.s && m.range && m.prims == 1 && m.trans == 1 && m.mat == 1;
}

}  // namespace

int main() {
  std::string err;

  // XTRAC = 0: the codestream follows the 8-byte header.
  {
    std::vector<uint8_t> p = main_header(0);
    p.push_back(0xFF);
    p.push_back(0x4F);  // SOC
    MainPacketHeader m{};
    const bool ok = parse_main_packet_header(p.data(), p.size(), m, err);
    check(ok, "XTRAC = 0 accepted");
    check(ok && fields_ok(m) && m.xtrac == 0 && m.codestream_offset == 8, "XTRAC = 0: fields, offset 8");
  }

  // XTRAC = 1..7: accepted, XTRAB skipped whatever it holds, codestream after it.
  for (uint8_t x = 1; x <= 7; ++x) {
    std::vector<uint8_t> p = main_header(x);
    for (size_t i = 0; i < 4u * x; ++i) p.push_back(static_cast<uint8_t>(0xA5 ^ i));  // XTRAB
    p.push_back(0xFF);
    p.push_back(0x4F);  // SOC
    MainPacketHeader m{};
    const bool ok = parse_main_packet_header(p.data(), p.size(), m, err);
    char what[96];
    std::snprintf(what, sizeof(what), "XTRAC = %u accepted (%s)", x, ok ? "ok" : err.c_str());
    check(ok, what);
    std::snprintf(what, sizeof(what), "XTRAC = %u: fields, codestream at %u", x, 8 + 4 * x);
    check(ok && fields_ok(m) && m.xtrac == x && m.codestream_offset == 8u + 4u * x
              && p[m.codestream_offset] == 0xFF && p[m.codestream_offset + 1] == 0x4F,
          what);
  }

  // XTRAB filling the whole payload: accepted, no codestream bytes in this packet.
  {
    std::vector<uint8_t> p = main_header(2);
    p.resize(16, 0);
    MainPacketHeader m{};
    const bool ok = parse_main_packet_header(p.data(), p.size(), m, err);
    check(ok && m.codestream_offset == 16, "XTRAB filling the payload accepted, offset 16");
  }

  // XTRAB running past the payload: refused.
  {
    std::vector<uint8_t> p = main_header(3);
    p.resize(19, 0);  // 8 + 12 bytes needed
    MainPacketHeader m{};
    check(!parse_main_packet_header(p.data(), p.size(), m, err), "XTRAB past the payload refused");
  }

  // Body Packet: codestream at byte 8 (Body Packets have no XTRAB).
  {
    uint8_t b[10] = {0};
    b[0] = 2u;  // MH = 0 (Body), TP = 0, RES = 2 (low bits of byte 0)
    b[1] = static_cast<uint8_t>(0x80 | (3u << 4));
    b[8] = 0xAB;
    BodyPacketHeader body{};
    const bool ok = parse_body_packet_header(b, sizeof(b), body, err);
    check(ok && body.codestream_offset == 8, "Body Packet: codestream at byte 8");
  }

  if (failures == 0) std::printf("rfc9828_parser_check: all cases passed\n");
  return failures == 0 ? 0 : 1;
}
