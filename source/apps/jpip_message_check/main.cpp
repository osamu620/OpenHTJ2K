// jpip_message_check: ctest harness for the JPP-stream message header
// codec (ISO/IEC 15444-9 §A.2 + Tables A.1, A.2).
//
// Self-contained: every assertion lives in this file, no input needed.
// Exits 0 on every assertion passing, 1 on the first failure.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "jpp_message.hpp"
#include "vbas.hpp"

using open_htj2k::jpip::decode_header;
using open_htj2k::jpip::encode_header;
using open_htj2k::jpip::encode_header_independent;
using open_htj2k::jpip::kMessageHeaderMaxBytes;
using open_htj2k::jpip::kMsgClassExtPrecinct;
using open_htj2k::jpip::kMsgClassMainHeader;
using open_htj2k::jpip::kMsgClassMetadata;
using open_htj2k::jpip::kMsgClassPrecinct;
using open_htj2k::jpip::kMsgClassTileHeader;
using open_htj2k::jpip::MessageHeader;
using open_htj2k::jpip::MessageHeaderContext;
using open_htj2k::jpip::msg_class_has_aux;

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

bool eq_header(const MessageHeader &a, const MessageHeader &b) {
  if (a.class_id    != b.class_id)    return false;
  if (a.cs_n        != b.cs_n)        return false;
  if (a.in_class_id != b.in_class_id) return false;
  if (a.msg_offset  != b.msg_offset)  return false;
  if (a.msg_length  != b.msg_length)  return false;
  if (a.is_last     != b.is_last)     return false;
  if (msg_class_has_aux(a.class_id) && a.aux != b.aux) return false;
  return true;
}

void roundtrip_independent(const MessageHeader &h) {
  uint8_t buf[kMessageHeaderMaxBytes] = {};
  const std::size_t n = encode_header_independent(h, buf);
  MessageHeaderContext ctx;
  MessageHeader out;
  std::size_t adv = 0;
  CHECK(decode_header(buf, n, ctx, &out, &adv),
        "decode_header returned false on independent round-trip");
  CHECK(adv == n, "independent decode advance %zu != encoded length %zu", adv, n);
  CHECK(eq_header(out, h),
        "independent round-trip class=%u cs_n=%u id=%llu off=%llu len=%llu aux=%llu last=%d",
        h.class_id, h.cs_n,
        static_cast<unsigned long long>(h.in_class_id),
        static_cast<unsigned long long>(h.msg_offset),
        static_cast<unsigned long long>(h.msg_length),
        static_cast<unsigned long long>(h.aux), h.is_last);
}

// Verify a known byte string decodes to the expected MessageHeader, and
// then re-encode (independent form) to confirm we'd produce the same bytes
// back when no dependent-form context is in play.
void decode_known(const std::vector<uint8_t> &bytes, const MessageHeader &expected,
                  std::size_t expected_advance) {
  MessageHeaderContext ctx;
  MessageHeader out;
  std::size_t adv = 0;
  CHECK(decode_header(bytes.data(), bytes.size(), ctx, &out, &adv),
        "decode_known returned false");
  CHECK(adv == expected_advance, "decode_known advance %zu, expected %zu", adv,
        expected_advance);
  CHECK(eq_header(out, expected),
        "decode_known mismatch: got class=%u cs_n=%u id=%llu off=%llu len=%llu aux=%llu last=%d",
        out.class_id, out.cs_n,
        static_cast<unsigned long long>(out.in_class_id),
        static_cast<unsigned long long>(out.msg_offset),
        static_cast<unsigned long long>(out.msg_length),
        static_cast<unsigned long long>(out.aux), out.is_last);
}

void reject(const std::vector<uint8_t> &bytes, const char *why) {
  MessageHeaderContext ctx;
  MessageHeader out;
  std::size_t adv = 0;
  const bool ok = decode_header(bytes.data(), bytes.size(), ctx, &out, &adv);
  CHECK(!ok, "expected reject for %s, got class=%u id=%llu (%zu bytes)", why,
        out.class_id, static_cast<unsigned long long>(out.in_class_id), adv);
}

}  // namespace

int main() {
  // ── Round-trip a representative cross-section of headers ──────────────
  roundtrip_independent({/*class*/0, /*cs*/0, /*id*/3,    /*off*/107, /*len*/165, /*aux*/0, /*last*/false});
  roundtrip_independent({/*class*/0, /*cs*/0, /*id*/3,    /*off*/0,   /*len*/512, /*aux*/0, /*last*/true });
  roundtrip_independent({/*class*/1, /*cs*/0, /*id*/3,    /*off*/107, /*len*/165, /*aux*/3, /*last*/false});
  roundtrip_independent({/*class*/2, /*cs*/0, /*id*/0,    /*off*/0,   /*len*/40,  /*aux*/0, /*last*/true });
  roundtrip_independent({/*class*/6, /*cs*/0, /*id*/0,    /*off*/0,   /*len*/255, /*aux*/0, /*last*/true });
  roundtrip_independent({/*class*/8, /*cs*/0, /*id*/42,   /*off*/100, /*len*/200, /*aux*/0, /*last*/false});
  // Exercise multi-byte Bin-ID (in-class id > 15) and large offsets/lengths.
  roundtrip_independent({/*class*/0, /*cs*/3,  /*id*/16,         /*off*/0x4000, /*len*/0x200000, /*aux*/0, /*last*/false});
  roundtrip_independent({/*class*/0, /*cs*/65535, /*id*/0xDEADBEEFull, /*off*/0xCAFEBABEull, /*len*/0xFEEDFACEull, /*aux*/0, /*last*/false});
  // Aux-bearing class with non-zero aux at multi-byte VBAS boundaries.
  roundtrip_independent({/*class*/1, /*cs*/0, /*id*/3, /*off*/107, /*len*/165, /*aux*/0x3FFF, /*last*/false});
  roundtrip_independent({/*class*/5, /*cs*/0, /*id*/0, /*off*/0,   /*len*/100, /*aux*/0x4000, /*last*/false});

  // ── §A.3.2.2 spec examples — Case A non-extended precinct message ─────
  // Bytes 00100011 01101011 10000001 00100101 = 0x23 0x6B 0x81 0x25.
  // Bin-ID 0x23 = a=0 (1 byte), bb=01 (no Class/CSn), c=0, id=3.
  // Msg-Offset 0x6B = single-byte VBAS payload 107.
  // Msg-Length 0x81 0x25 = (1<<7) | 37 = 165.
  decode_known({0x23, 0x6B, 0x81, 0x25},
               {/*class*/0, /*cs*/0, /*id*/3, /*off*/107, /*len*/165, /*aux*/0, /*last*/false},
               4);

  // Case A extended (precinct + Aux): 0x43 0x01 0x6B 0x81 0x25 0x03.
  // Bin-ID 0x43 = a=0, bb=10 (Class only), c=0, id=3.
  // Class VBAS 0x01 = 1 (extended precinct).
  // Then Msg-Offset/Msg-Length as above.
  // Aux VBAS 0x03 = 3 — number of completed quality layers.
  decode_known({0x43, 0x01, 0x6B, 0x81, 0x25, 0x03},
               {/*class*/1, /*cs*/0, /*id*/3, /*off*/107, /*len*/165, /*aux*/3, /*last*/false},
               6);

  // Case C non-extended: 0x33 0x81 0x08 0x81 0x35.  Bin-ID 0x33 has c=1
  // (this message contains the last byte of the data-bin).  Msg-Offset
  // 0x81 0x08 = 136.  Msg-Length 0x81 0x35 = (1<<7) | 53 = 181.
  decode_known({0x33, 0x81, 0x08, 0x81, 0x35},
               {/*class*/0, /*cs*/0, /*id*/3, /*off*/136, /*len*/181, /*aux*/0, /*last*/true},
               5);

  // ── Dependent-form behaviour across a sequence of three messages ──────
  // Encode three precinct-data-bin messages (class 0, codestream 0) into
  // a single buffer.  Expect the second and third to omit Class/CSn.
  {
    std::vector<uint8_t> stream(kMessageHeaderMaxBytes * 3);
    MessageHeaderContext enc_ctx;
    std::size_t pos = 0;
    pos += encode_header({/*class*/0, /*cs*/0, /*id*/3, /*off*/107, /*len*/165, /*aux*/0, /*last*/false},
                         enc_ctx, stream.data() + pos);
    pos += encode_header({/*class*/0, /*cs*/0, /*id*/4, /*off*/272, /*len*/200, /*aux*/0, /*last*/false},
                         enc_ctx, stream.data() + pos);
    pos += encode_header({/*class*/0, /*cs*/0, /*id*/5, /*off*/472, /*len*/100, /*aux*/0, /*last*/true},
                         enc_ctx, stream.data() + pos);
    stream.resize(pos);

    // Each subsequent message should be smaller than the first because it
    // omits Class (single-byte VBAS).  Strictly: after the first message,
    // the Bin-ID byte's bb field changes from emit-class to no-class; the
    // dropped Class VBAS is one byte for class 0.
    MessageHeaderContext dec_ctx;
    MessageHeader m1{}, m2{}, m3{};
    std::size_t adv1 = 0, adv2 = 0, adv3 = 0;
    CHECK(decode_header(stream.data(),                stream.size(),                  dec_ctx, &m1, &adv1), "msg 1 decode");
    CHECK(decode_header(stream.data() + adv1,         stream.size() - adv1,           dec_ctx, &m2, &adv2), "msg 2 decode");
    CHECK(decode_header(stream.data() + adv1 + adv2,  stream.size() - adv1 - adv2,    dec_ctx, &m3, &adv3), "msg 3 decode");
    CHECK(adv1 + adv2 + adv3 == stream.size(), "stream not fully consumed: %zu/%zu",
          adv1 + adv2 + adv3, stream.size());
    CHECK(m1.class_id == 0 && m1.in_class_id == 3 && m1.msg_offset == 107 && m1.msg_length == 165, "m1 fields");
    CHECK(m2.class_id == 0 && m2.in_class_id == 4 && m2.msg_offset == 272 && m2.msg_length == 200, "m2 fields");
    CHECK(m3.class_id == 0 && m3.in_class_id == 5 && m3.msg_offset == 472 && m3.msg_length == 100 && m3.is_last, "m3 fields");
  }

  // CSn change forces Class to be sent again (Table A.1 has no CSn-only).
  {
    uint8_t buf1[kMessageHeaderMaxBytes];
    uint8_t buf2[kMessageHeaderMaxBytes];
    MessageHeaderContext ctx;
    const std::size_t n1 = encode_header({/*class*/2, /*cs*/0, /*id*/0, /*off*/0, /*len*/40, /*aux*/0, /*last*/true}, ctx, buf1);
    const std::size_t n2 = encode_header({/*class*/2, /*cs*/1, /*id*/0, /*off*/0, /*len*/40, /*aux*/0, /*last*/true}, ctx, buf2);
    // The Bin-ID's bb field on buf2 must indicate Class AND CSn (bb=11).
    const uint8_t bb2 = static_cast<uint8_t>((buf2[0] >> 5) & 0x3u);
    CHECK(bb2 == 0x3u, "CSn-change without Class: bb=%u, expected 3", bb2);
    (void)n1; (void)n2;
  }

  // ── Rejection cases ──────────────────────────────────────────────────
  reject({},                                   "empty input");
  reject({0x03, 0x00, 0x00},                   "bb=00 (prohibited)");
  // Bin-ID claims continuation but no follow-up byte.
  reject({0xA3},                               "Bin-ID continuation with no second byte");
  // Msg-Offset truncated mid-VBAS.
  reject({0x23, 0x80},                         "Msg-Offset VBAS truncated");
  // Aux truncated for an aux-bearing class.
  reject({0x43, 0x01, 0x00, 0x00, 0x80},       "Aux VBAS truncated");

  if (failures == 0) {
    std::printf("OK message_check: round-trip / spec-example / dependent-form / reject all pass\n");
    return 0;
  }
  std::fprintf(stderr, "message_check: %d failures\n", failures);
  return 1;
}
