// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.
//
// Licensed under the same BSD 3-Clause terms as the rest of OpenHTJ2K.

#pragma once

// Pure-function parsers for:
//   - the 12-byte RTP fixed header (RFC 3550 §5.1)
//   - the RFC 9828 §5 Main Packet payload header (8 bytes)
//   - the RFC 9828 §5 Body Packet payload header (8 bytes)
//
// No I/O, no allocation.  Parsers take a raw byte pointer + length and return
// true on success, filling the caller's POD struct; on failure they fill
// `error` with a short diagnostic.  The payload pointer/length returned from
// parse_rtp_header() is the slice the Main/Body parsers should be called on.

#include <cstddef>
#include <cstdint>
#include <string>

namespace open_htj2k::rtp_recv {

// -------- RTP fixed header --------

struct RtpHeader {
  uint8_t  version        = 0;  // must be 2
  bool     padding        = false;
  bool     extension      = false;
  uint8_t  csrc_count     = 0;
  bool     marker         = false;
  uint8_t  payload_type   = 0;
  uint16_t sequence       = 0;
  uint32_t timestamp      = 0;
  uint32_t ssrc           = 0;
  // Offset within the input buffer where the RTP payload begins
  // (past fixed header + CSRCs + optional extension).
  size_t payload_offset = 0;
};

bool parse_rtp_header(const uint8_t* data, size_t len, RtpHeader& out, std::string& error);

// -------- RFC 9828 Main Packet payload header (§5.1) --------
// 8-byte fixed field; XTRAB extension follows when XTRAC != 0 (current
// revision mandates XTRAC == 0 so we reject non-zero values).

struct MainPacketHeader {
  uint8_t  mh        = 0;    // 2 bits: 0=Body(shouldn't see), 1=Main+Main, 2=Main+Body, 3=Single Main
  uint8_t  tp        = 0;    // 3 bits: image type (0=progressive, 1..4=interlaced, 5..6=PsF, 7=ext)
  uint8_t  ordh      = 0;    // 3 bits: progression order + resync signaling
  bool     p         = false;// 1 bit: PTSTAMP valid
  uint8_t  xtrac     = 0;    // 3 bits: XTRAB length in 4-byte units (must be 0 in current revision)
  uint16_t ptstamp   = 0;    // 12 bits
  uint8_t  eseq      = 0;    // 8 bits: high byte of extended sequence number
  bool     r         = false;// 1 bit: main header reuse
  bool     s         = false;// 1 bit: parameterized colorspace present
  bool     c         = false;// 1 bit: code-block caching enabled (requires r)
  bool     range     = false;// 1 bit: VideoFullRangeFlag (H.273); 0=narrow, 1=full
  uint8_t  prims     = 0;    // 8 bits: H.273 ColourPrimaries
  uint8_t  trans     = 0;    // 8 bits: H.273 TransferCharacteristics
  uint8_t  mat       = 0;    // 8 bits: H.273 MatrixCoefficients

  // Offset (relative to the RTP payload start) where the codestream bytes
  // for this packet begin — past the 8-byte header and any XTRAB bytes.
  size_t codestream_offset = 0;
};

bool parse_main_packet_header(const uint8_t* payload, size_t len, MainPacketHeader& out,
                              std::string& error);

// -------- RFC 9828 Body Packet payload header (§5.2) --------
// 8-byte fixed field.

struct BodyPacketHeader {
  uint8_t  mh      = 0;   // 2 bits: 0 = Body packet (expected here)
  uint8_t  tp      = 0;   // 3 bits: image type
  uint8_t  res     = 0;   // 3 bits: resolution-level floor
  bool     ordb    = false; // 1 bit: resync-point flag
  uint8_t  qual    = 0;   // 3 bits: quality-layer floor
  uint16_t ptstamp = 0;   // 12 bits
  uint8_t  eseq    = 0;   // 8 bits: high byte of extended sequence number
  uint16_t pos     = 0;   // 12 bits: resync-point byte offset (0 if !ordb)
  uint32_t pid     = 0;   // 20 bits: precinct identifier (0 if !ordb)

  size_t codestream_offset = 0;  // always 8 for Body packets
};

bool parse_body_packet_header(const uint8_t* payload, size_t len, BodyPacketHeader& out,
                              std::string& error);

// -------- ORDH constants (Main Packet) --------
constexpr uint8_t ORDH_UNSPECIFIED = 0;
constexpr uint8_t ORDH_LRCP_RESYNC = 1;
constexpr uint8_t ORDH_RLCP_RESYNC = 2;
constexpr uint8_t ORDH_RPCL_RESYNC = 3;
constexpr uint8_t ORDH_PCRL_RESYNC = 4;  // sub-codestream latency
constexpr uint8_t ORDH_CPRL_RESYNC = 5;
constexpr uint8_t ORDH_PRCL_RESYNC = 6;  // sub-codestream latency (Part 2)

// -------- MH constants --------
constexpr uint8_t MH_BODY              = 0;
constexpr uint8_t MH_MAIN_MORE_MAIN    = 1;
constexpr uint8_t MH_MAIN_THEN_BODY    = 2;
constexpr uint8_t MH_MAIN_SINGLE       = 3;

}  // namespace open_htj2k::rtp_recv
