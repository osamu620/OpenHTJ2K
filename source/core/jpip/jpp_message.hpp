// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.
//
// JPP-stream message header codec, ISO/IEC 15444-9 §A.2 + Tables A.1, A.2.
//
// Each JPP-stream message starts with a header of the form:
//
//   Bin-ID  [, Class]  [, CSn]  ,  Msg-Offset  ,  Msg-Length  [, Aux]
//
// All fields are VBASs (see vbas.hpp), with the Bin-ID VBAS additionally
// carrying three control bits: a "Class/CSn presence" indicator (Table A.1),
// a "is last byte of data-bin" flag, and 4 bits of in-class identifier in
// the first byte (the remaining bits live in the subsequent VBAS bytes).
//
// Class identifies the data-bin type per Table A.2; CSn identifies the
// codestream index.  Both are optional in the wire format: when omitted,
// the decoder inherits the value from the previous message in the stream
// (defaulting to 0 at the start).  The dependent form is honoured by both
// the encoder and the decoder via a MessageHeaderContext carried across
// calls.
//
// The Aux field is present iff the class identifier is odd
// (kMsgClassExtPrecinct, kMsgClassExtTile).  Per §A.2.2 it carries
// auxiliary information whose meaning depends on the class.
#pragma once
#include <cstddef>
#include <cstdint>

#include "vbas.hpp"

#if defined(_MSC_VER) && !defined(OHTJ2K_STATIC)
  #define OPENHTJ2K_JPIP_EXPORT __declspec(dllexport)
#else
  #define OPENHTJ2K_JPIP_EXPORT
#endif

namespace open_htj2k {
namespace jpip {

// Class identifiers per Table A.2.  EOR is intentionally absent here: per
// §D.3, the EOR message is "not defined in Annex A and is not formally part
// of the JPP- or JPT-stream media types" — it uses a special sentinel byte
// (0x00) as its identifier, not a class.  See data_bin_emitter.cpp and
// jpp_parser.cpp for the wire-format handling.
constexpr uint8_t kMsgClassPrecinct    = 0;  // JPP-stream
constexpr uint8_t kMsgClassExtPrecinct = 1;  // JPP-stream, has Aux
constexpr uint8_t kMsgClassTileHeader  = 2;  // JPP-stream
constexpr uint8_t kMsgClassTile        = 4;  // JPT-stream
constexpr uint8_t kMsgClassExtTile     = 5;  // JPT-stream, has Aux
constexpr uint8_t kMsgClassMainHeader  = 6;  // JPP- and JPT-stream
constexpr uint8_t kMsgClassMetadata    = 8;  // JPP- and JPT-stream

// EOR reason codes (§D.3, Table D.2).
enum class EorReason : uint8_t {
  ImageDone     = 1,   // all available image information transferred
  WindowDone    = 2,   // all information relevant to the view-window sent
  WindowChange  = 3,   // server preempting to service a new request
  ByteLimit     = 4,   // Maximum Response Length reached
  QualityLimit  = 5,   // Quality request field limit reached
  SessionLimit  = 6,   // session resource limit reached
  ResponseLimit = 7,   // non-session response limit reached
  NonSpecified  = 0xFF
};

// Per §A.2.2: "Class identifiers are chosen such that an Aux VBAS is present
// if and only if the identifier is odd."
inline bool msg_class_has_aux(uint8_t class_id) {
  return (class_id & 1u) != 0u;
}

struct MessageHeader {
  uint8_t  class_id    = 0;
  uint16_t cs_n        = 0;
  uint64_t in_class_id = 0;
  uint64_t msg_offset  = 0;
  uint64_t msg_length  = 0;
  uint64_t aux         = 0;     // ignored unless msg_class_has_aux(class_id)
  bool     is_last     = false; // bit 4 of the first Bin-ID byte
};

// Carries the most recently emitted/decoded class and codestream index
// across calls so the dependent form of subsequent messages can omit the
// matching Class/CSn fields (per §A.2.1).  Default-constructed values match
// the spec's "no previous message" defaults: class 0, codestream 0.
struct MessageHeaderContext {
  uint8_t  last_class_id = 0;
  uint16_t last_cs_n     = 0;
  void clear() { *this = {}; }
};

// Worst-case encoded header size: Bin-ID can reach kVbasMaxBytes; each of
// Class, CSn, Msg-Offset, Msg-Length, Aux is also a VBAS so the same
// upper bound applies.  Six fields total.
constexpr std::size_t kMessageHeaderMaxBytes = kVbasMaxBytes * 6;

// Encode `hdr` into `dst` (must have ≥ kMessageHeaderMaxBytes writable).
// Class and CSn are emitted only if they differ from `ctx`'s last seen
// values; the spec also requires Class to accompany CSn when CSn is sent
// (Table A.1 has no "CSn-only" combination), so a CSn change always pulls
// Class along.  Updates `ctx` to reflect the encoded values.  Returns the
// number of bytes written.
OPENHTJ2K_JPIP_EXPORT std::size_t encode_header(const MessageHeader &hdr,
                                                MessageHeaderContext &ctx,
                                                uint8_t *dst);

// As above, but always emits the independent form (Class + CSn always
// present).  Useful for stand-alone test vectors and for the first message
// of a stream where the caller wants no implicit defaults.
OPENHTJ2K_JPIP_EXPORT std::size_t encode_header_independent(const MessageHeader &hdr,
                                                            uint8_t *dst);

// Decode a message header starting at `src`.  Uses `ctx` to fill in
// missing Class/CSn from the previous message and updates `ctx` with the
// decoded effective values.  On success, populates *out, writes the bytes
// consumed to *advance, and returns true.  On any malformed input
// (truncated, prohibited bb=00, value overflow) returns false and leaves
// *out / *advance untouched.
OPENHTJ2K_JPIP_EXPORT bool decode_header(const uint8_t *src, std::size_t src_len,
                                         MessageHeaderContext &ctx,
                                         MessageHeader *out, std::size_t *advance);

}  // namespace jpip
}  // namespace open_htj2k
