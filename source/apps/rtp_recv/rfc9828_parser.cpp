// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.
//
// Licensed under the same BSD 3-Clause terms as the rest of OpenHTJ2K.

#include "rfc9828_parser.hpp"

namespace open_htj2k::rtp_recv {

namespace {
inline uint16_t rd_u16_be(const uint8_t* p) {
  return static_cast<uint16_t>((static_cast<uint16_t>(p[0]) << 8) | p[1]);
}
inline uint32_t rd_u32_be(const uint8_t* p) {
  return (static_cast<uint32_t>(p[0]) << 24) | (static_cast<uint32_t>(p[1]) << 16)
         | (static_cast<uint32_t>(p[2]) << 8) | static_cast<uint32_t>(p[3]);
}
}  // namespace

bool parse_rtp_header(const uint8_t* data, size_t len, RtpHeader& out, std::string& error) {
  // Fixed 12-byte header, plus 4*CC CSRC bytes, plus optional extension header.
  if (len < 12) {
    error = "rtp header: need at least 12 bytes";
    return false;
  }
  out.version      = static_cast<uint8_t>(data[0] >> 6);
  out.padding      = (data[0] & 0x20) != 0;
  out.extension    = (data[0] & 0x10) != 0;
  out.csrc_count   = static_cast<uint8_t>(data[0] & 0x0F);
  out.marker       = (data[1] & 0x80) != 0;
  out.payload_type = static_cast<uint8_t>(data[1] & 0x7F);
  out.sequence     = rd_u16_be(data + 2);
  out.timestamp    = rd_u32_be(data + 4);
  out.ssrc         = rd_u32_be(data + 8);

  if (out.version != 2) {
    error = "rtp header: version != 2";
    return false;
  }

  size_t offset = 12 + static_cast<size_t>(out.csrc_count) * 4;
  if (len < offset) {
    error = "rtp header: CSRC list overruns buffer";
    return false;
  }

  if (out.extension) {
    // Extension header: 16-bit profile-defined + 16-bit length (in 32-bit words, not counting
    // the 4-byte header itself).
    if (len < offset + 4) {
      error = "rtp header: extension header overruns buffer";
      return false;
    }
    const uint16_t ext_words = rd_u16_be(data + offset + 2);
    const size_t ext_bytes  = 4u + static_cast<size_t>(ext_words) * 4u;
    if (len < offset + ext_bytes) {
      error = "rtp header: extension body overruns buffer";
      return false;
    }
    offset += ext_bytes;
  }

  // We intentionally do not strip padding; RFC 9828 senders shouldn't use it
  // in steady-state, and the frame handler ignores tail padding bytes anyway.

  out.payload_offset = offset;
  return true;
}

bool parse_main_packet_header(const uint8_t* payload, size_t len, MainPacketHeader& out,
                              std::string& error) {
  if (len < 8) {
    error = "main packet: need at least 8 bytes";
    return false;
  }

  // Byte 0: MH(2) TP(3) ORDH(3)
  const uint8_t b0 = payload[0];
  out.mh   = static_cast<uint8_t>((b0 >> 6) & 0x03);
  out.tp   = static_cast<uint8_t>((b0 >> 3) & 0x07);
  out.ordh = static_cast<uint8_t>(b0 & 0x07);

  if (out.mh == MH_BODY) {
    error = "main packet: MH=0 is a Body packet, not Main";
    return false;
  }

  // Byte 1: P(1) XTRAC(3) PTSTAMP[11:8](4)
  const uint8_t b1 = payload[1];
  out.p     = (b1 & 0x80) != 0;
  out.xtrac = static_cast<uint8_t>((b1 >> 4) & 0x07);
  const uint16_t ptstamp_hi = static_cast<uint16_t>(b1 & 0x0F);

  // Byte 2: PTSTAMP[7:0]
  out.ptstamp = static_cast<uint16_t>((ptstamp_hi << 8) | payload[2]);

  // Byte 3: ESEQ
  out.eseq = payload[3];

  // Current RFC revision mandates XTRAC == 0.  Refuse anything else — we'd
  // be consuming bytes we don't know how to interpret.
  if (out.xtrac != 0) {
    error = "main packet: XTRAC != 0 (future revision, unsupported)";
    return false;
  }

  // Byte 4: R(1) S(1) C(1) RSVD(4) RANGE(1)
  const uint8_t b4 = payload[4];
  out.r     = (b4 & 0x80) != 0;
  out.s     = (b4 & 0x40) != 0;
  out.c     = (b4 & 0x20) != 0;
  const uint8_t rsvd = static_cast<uint8_t>((b4 >> 1) & 0x0F);
  out.range = (b4 & 0x01) != 0;

  if (rsvd != 0) {
    error = "main packet: reserved bits in byte 4 must be zero";
    return false;
  }

  // Byte 5/6/7: PRIMS, TRANS, MAT
  out.prims = payload[5];
  out.trans = payload[6];
  out.mat   = payload[7];

  // §5.3: when S = 0, PRIMS/TRANS/MAT/RANGE MUST be zero.  A sender that
  // violates this is nominally non-conformant; warn loudly by rejecting.
  if (!out.s && (out.prims != 0 || out.trans != 0 || out.mat != 0 || out.range)) {
    error = "main packet: S=0 requires PRIMS/TRANS/MAT/RANGE to be zero";
    return false;
  }

  // §7.9: C = 1 requires R = 1 (code-block caching needs a stable main header).
  if (out.c && !out.r) {
    error = "main packet: C=1 without R=1 (caching requires main-header reuse)";
    return false;
  }

  // XTRAB is zero-length in the current revision, so codestream starts
  // immediately after byte 7.
  out.codestream_offset = 8;

  return true;
}

bool parse_body_packet_header(const uint8_t* payload, size_t len, BodyPacketHeader& out,
                              std::string& error) {
  if (len < 8) {
    error = "body packet: need at least 8 bytes";
    return false;
  }

  // Byte 0: MH(2) TP(3) RES(3)
  const uint8_t b0 = payload[0];
  out.mh  = static_cast<uint8_t>((b0 >> 6) & 0x03);
  out.tp  = static_cast<uint8_t>((b0 >> 3) & 0x07);
  out.res = static_cast<uint8_t>(b0 & 0x07);

  if (out.mh != MH_BODY) {
    error = "body packet: MH != 0";
    return false;
  }

  // Byte 1: ORDB(1) QUAL(3) PTSTAMP[11:8](4)
  const uint8_t b1 = payload[1];
  out.ordb = (b1 & 0x80) != 0;
  out.qual = static_cast<uint8_t>((b1 >> 4) & 0x07);
  const uint16_t ptstamp_hi = static_cast<uint16_t>(b1 & 0x0F);

  // Byte 2: PTSTAMP[7:0]
  out.ptstamp = static_cast<uint16_t>((ptstamp_hi << 8) | payload[2]);

  // Byte 3: ESEQ
  out.eseq = payload[3];

  // Bytes 4-7: POS(12) then PID(20), packed MSB-first.
  //
  //   Byte:  4        5        6        7
  //          PPPPPPPP PPPPIIII IIIIIIII IIIIIIII
  //          \_POS(12)_/\______ PID(20) ______/
  out.pos = static_cast<uint16_t>((static_cast<uint16_t>(payload[4]) << 4)
                                  | static_cast<uint16_t>(payload[5] >> 4));
  out.pid = (static_cast<uint32_t>(payload[5] & 0x0F) << 16)
            | (static_cast<uint32_t>(payload[6]) << 8) | static_cast<uint32_t>(payload[7]);

  // §5.2: POS and PID MUST be zero when ORDB = 0.  Enforce for sanity.
  if (!out.ordb && (out.pos != 0 || out.pid != 0)) {
    error = "body packet: ORDB=0 requires POS=0 and PID=0";
    return false;
  }

  out.codestream_offset = 8;
  return true;
}

}  // namespace open_htj2k::rtp_recv
