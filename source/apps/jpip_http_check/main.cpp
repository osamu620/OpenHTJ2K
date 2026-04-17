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

using open_htj2k::jpip::format_error_response;
using open_htj2k::jpip::format_jpp_response;
using open_htj2k::jpip::JpipRequest;
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

  if (failures == 0) {
    std::printf("OK http_check: request/response formatting all pass\n");
    return 0;
  }
  std::fprintf(stderr, "http_check: %d failures\n", failures);
  return 1;
}
