// jpip_http_check: ctest harness for the JPIP request parser + response
// formatter (Phase 3, S1).
//
// Self-contained — runs every test internally, takes no input.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "jpip_request.hpp"
#include "jpip_response.hpp"

using open_htj2k::jpip::decode_chunked_body;
using open_htj2k::jpip::format_chunk_header;
using open_htj2k::jpip::format_error_response;
using open_htj2k::jpip::format_jpp_response;
using open_htj2k::jpip::format_jpp_response_headers_chunked;
using open_htj2k::jpip::format_last_chunk;
using open_htj2k::jpip::JpipRequest;
using open_htj2k::jpip::parse_http_chunked;
using open_htj2k::jpip::parse_http_content_length;
using open_htj2k::jpip::parse_jpip_query;
using open_htj2k::jpip::RequestParseStatus;
using open_htj2k::jpip::split_http_get_line;
using open_htj2k::jpip::ViewWindow;

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

}  // namespace

int main() {
  // ── split_http_get_line ────────────────────────────────────────────────
  {
    std::string path, query;
    CHECK(split_http_get_line("GET /jpip?fsiz=1920,1080&type=jpp-stream HTTP/1.1", &path, &query),
          "split basic");
    CHECK(path == "/jpip", "path='%s'", path.c_str());
    CHECK(query == "fsiz=1920,1080&type=jpp-stream", "query='%s'", query.c_str());

    CHECK(split_http_get_line("GET /jpip HTTP/1.1", &path, &query), "split no query");
    CHECK(path == "/jpip", "path='%s'", path.c_str());
    CHECK(query.empty(), "query should be empty");

    CHECK(!split_http_get_line("POST /foo HTTP/1.1", &path, &query), "reject POST");
    CHECK(!split_http_get_line("", &path, &query), "reject empty");
  }

  // ── parse_jpip_query: full request ─────────────────────────────────────
  {
    JpipRequest req;
    auto s = parse_jpip_query(
        "target=image.j2c&fsiz=1920,1080,round-up&roff=400,300&rsiz=500,500"
        "&comps=0,1,2&type=jpp-stream", &req);
    CHECK(s == RequestParseStatus::Ok, "status=%d", static_cast<int>(s));
    CHECK(req.target == "image.j2c", "target='%s'", req.target.c_str());
    CHECK(req.has_fsiz, "has_fsiz");
    CHECK(req.view_window.fx == 1920, "fx=%u", req.view_window.fx);
    CHECK(req.view_window.fy == 1080, "fy=%u", req.view_window.fy);
    CHECK(req.view_window.round == ViewWindow::Round::Up, "round");
    CHECK(req.has_roff, "has_roff");
    CHECK(req.view_window.ox == 400, "ox=%u", req.view_window.ox);
    CHECK(req.view_window.oy == 300, "oy=%u", req.view_window.oy);
    CHECK(req.has_rsiz, "has_rsiz");
    CHECK(req.view_window.sx == 500, "sx=%u", req.view_window.sx);
    CHECK(req.view_window.sy == 500, "sy=%u", req.view_window.sy);
    CHECK(req.has_comps, "has_comps");
    CHECK(req.view_window.comps.size() == 3, "comps size=%zu", req.view_window.comps.size());
    CHECK(req.type == "jpp-stream", "type='%s'", req.type.c_str());
  }

  // ── parse_jpip_query: minimal (fsiz only) ──────────────────────────────
  {
    JpipRequest req;
    auto s = parse_jpip_query("?fsiz=960,960", &req);
    CHECK(s == RequestParseStatus::Ok, "status=%d", static_cast<int>(s));
    CHECK(req.view_window.fx == 960, "fx=%u", req.view_window.fx);
    CHECK(req.view_window.round == ViewWindow::Round::Down, "default round");
    CHECK(!req.has_roff && !req.has_rsiz && !req.has_comps, "no optional fields");
  }

  // ── parse_jpip_query: round-down explicit ──────────────────────────────
  {
    JpipRequest req;
    auto s = parse_jpip_query("fsiz=480,270,round-down", &req);
    CHECK(s == RequestParseStatus::Ok, "status=%d", static_cast<int>(s));
    CHECK(req.view_window.fx == 480, "fx");
    CHECK(req.view_window.fy == 270, "fy");
    CHECK(req.view_window.round == ViewWindow::Round::Down, "round-down");
  }

  // ── parse_jpip_query: closest ──────────────────────────────────────────
  {
    JpipRequest req;
    auto s = parse_jpip_query("fsiz=800,600,closest", &req);
    CHECK(s == RequestParseStatus::Ok, "status=%d", static_cast<int>(s));
    CHECK(req.view_window.round == ViewWindow::Round::Closest, "closest");
  }

  // ── parse_jpip_query: unsupported type ──────────────────────────────────
  {
    JpipRequest req;
    auto s = parse_jpip_query("fsiz=100,100&type=jpt-stream", &req);
    CHECK(s == RequestParseStatus::UnsupportedType, "type");
  }

  // ── parse_jpip_query: malformed fsiz ────────────────────────────────────
  {
    JpipRequest req;
    auto s = parse_jpip_query("fsiz=abc", &req);
    CHECK(s == RequestParseStatus::MalformedField, "malformed fsiz");
  }

  // ── parse_jpip_query: unknown fields silently ignored ──────────────────
  {
    JpipRequest req;
    auto s = parse_jpip_query("fsiz=100,200&foo=bar&baz=1", &req);
    CHECK(s == RequestParseStatus::Ok, "unknown fields accepted");
    CHECK(req.view_window.fx == 100, "fx after unknown");
  }

  // ── format + parse round-trip ──────────────────────────────────────────
  {
    const uint8_t body[] = {0xDE, 0xAD, 0xBE, 0xEF};
    auto resp = format_jpp_response(body, 4, "mytarget");
    // Parse Content-Length back.
    std::size_t hdr_end = 0;
    const std::size_t cl = parse_http_content_length(resp.data(), resp.size(), &hdr_end);
    CHECK(cl == 4, "Content-Length=%zu", cl);
    CHECK(hdr_end + 4 == resp.size(), "total=%zu hdr_end=%zu", resp.size(), hdr_end);
    CHECK(std::memcmp(resp.data() + hdr_end, body, 4) == 0, "body mismatch");
    // Check JPIP-tid header is present.
    std::string hdr_str(resp.begin(), resp.begin() + static_cast<std::ptrdiff_t>(hdr_end));
    CHECK(hdr_str.find("JPIP-tid: mytarget") != std::string::npos, "JPIP-tid missing");
    CHECK(hdr_str.find("image/jpp-stream") != std::string::npos, "Content-Type missing");
  }

  // ── error response ─────────────────────────────────────────────────────
  {
    auto resp = format_error_response(404, "Not Found");
    std::string s(resp.begin(), resp.end());
    CHECK(s.find("404 Not Found") != std::string::npos, "404");
    CHECK(s.find("Content-Length: 0") != std::string::npos, "empty body");
  }

  // ── parse_jpip_query: len= + quality= (§C.6.1) ─────────────────────────
  {
    JpipRequest req;
    auto s = parse_jpip_query("fsiz=1,1&type=jpp-stream&len=4096&quality=3", &req);
    CHECK(s == RequestParseStatus::Ok, "status=%d", static_cast<int>(s));
    CHECK(req.has_len, "has_len");
    CHECK(req.len == 4096, "len=%llu", static_cast<unsigned long long>(req.len));
    CHECK(req.has_quality, "has_quality");
    CHECK(req.quality == 3, "quality=%u", req.quality);
  }
  {
    JpipRequest req;
    auto s = parse_jpip_query("fsiz=1,1&type=jpp-stream", &req);
    CHECK(s == RequestParseStatus::Ok, "status");
    CHECK(!req.has_len && req.len == 0, "no len");
    CHECK(!req.has_quality && req.quality == 0, "no quality");
  }
  {
    // Malformed len (non-numeric) -> MalformedField.
    JpipRequest req;
    auto s = parse_jpip_query("fsiz=1,1&len=abc", &req);
    CHECK(s == RequestParseStatus::MalformedField, "reject non-numeric len");
  }
  {
    // Large len value round-trips intact.
    JpipRequest req;
    auto s = parse_jpip_query("len=4294967296", &req);
    CHECK(s == RequestParseStatus::Ok, "status");
    CHECK(req.len == 4294967296ULL, "len=%llu", static_cast<unsigned long long>(req.len));
  }

  // ── chunked headers + chunk framing round-trip ────────────────────────
  {
    auto hdrs = format_jpp_response_headers_chunked("tid-foo");
    std::string hs(hdrs.begin(), hdrs.end());
    CHECK(hs.find("Transfer-Encoding: chunked") != std::string::npos,
          "chunked header missing");
    CHECK(hs.find("Content-Length") == std::string::npos,
          "chunked response must not carry Content-Length");
    CHECK(hs.find("JPIP-tid: tid-foo") != std::string::npos, "JPIP-tid missing");
    CHECK(hs.size() >= 4 && std::memcmp(hs.data() + hs.size() - 4, "\r\n\r\n", 4) == 0,
          "chunked headers must end with CRLFCRLF");

    // Assemble a complete chunked response: headers, three chunks, last chunk.
    const uint8_t msg_a[] = {0x01, 0x02, 0x03};
    const uint8_t msg_b[] = {0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE};
    const uint8_t msg_c[] = {0x42};

    std::vector<uint8_t> resp = hdrs;
    for (const auto &m : {std::vector<uint8_t>(msg_a, msg_a + sizeof(msg_a)),
                          std::vector<uint8_t>(msg_b, msg_b + sizeof(msg_b)),
                          std::vector<uint8_t>(msg_c, msg_c + sizeof(msg_c))}) {
      auto ch = format_chunk_header(m.size());
      resp.insert(resp.end(), ch.begin(), ch.end());
      resp.insert(resp.end(), m.begin(), m.end());
      resp.push_back('\r');
      resp.push_back('\n');
    }
    auto last = format_last_chunk();
    resp.insert(resp.end(), last.begin(), last.end());

    std::size_t hdr_end = 0;
    CHECK(parse_http_chunked(resp.data(), resp.size(), &hdr_end),
          "parse_http_chunked should detect chunked");

    std::vector<uint8_t> decoded;
    CHECK(decode_chunked_body(resp.data() + hdr_end, resp.size() - hdr_end, &decoded),
          "decode_chunked_body");
    const std::vector<uint8_t> expect = {
        0x01, 0x02, 0x03, 0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0x42};
    CHECK(decoded == expect, "decoded body mismatch (size=%zu)", decoded.size());
  }

  // ── decode_chunked_body rejects malformed inputs ───────────────────────
  {
    std::vector<uint8_t> out;
    // Missing terminator.
    const char *no_term = "3\r\nABC\r\n";
    CHECK(!decode_chunked_body(reinterpret_cast<const uint8_t *>(no_term),
                               std::strlen(no_term), &out),
          "unterminated chunked body must fail");
    // Non-hex size.
    const char *bad_hex = "zz\r\n\r\n";
    CHECK(!decode_chunked_body(reinterpret_cast<const uint8_t *>(bad_hex),
                               std::strlen(bad_hex), &out),
          "non-hex chunk size must fail");
    // Terminating 0-chunk alone is a valid empty body.
    const char *zero_only = "0\r\n\r\n";
    std::vector<uint8_t> empty_out;
    CHECK(decode_chunked_body(reinterpret_cast<const uint8_t *>(zero_only),
                              std::strlen(zero_only), &empty_out),
          "`0\\r\\n\\r\\n` is a legal empty body");
    CHECK(empty_out.empty(), "empty body decodes to empty vector");
  }

  // ── Content-Length path unchanged by chunked additions ─────────────────
  {
    const uint8_t body[] = {0x11, 0x22};
    auto resp = format_jpp_response(body, sizeof(body), "");
    std::size_t hdr_end = 0;
    CHECK(!parse_http_chunked(resp.data(), resp.size(), &hdr_end),
          "Content-Length response must not be flagged as chunked");
    const std::size_t cl = parse_http_content_length(resp.data(), resp.size(), &hdr_end);
    CHECK(cl == 2, "Content-Length still parses on legacy path");
  }

  if (failures == 0) {
    std::printf("OK http_check: request/response formatting all pass\n");
    return 0;
  }
  std::fprintf(stderr, "http_check: %d failures\n", failures);
  return 1;
}
