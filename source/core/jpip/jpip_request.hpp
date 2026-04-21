// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.
//
// JPIP request parser — extracts view-window parameters from an HTTP GET
// query string per ISO/IEC 15444-9 §C.1–C.4.
//
// A typical JPIP request URL looks like:
//
//   GET /jpip?target=image.j2c&fsiz=1920,1080&roff=400,300&rsiz=500,500
//             &type=jpp-stream HTTP/1.1
//
// This module parses the query-string portion (after '?') into a
// ViewWindow plus an optional target filename.  Only the fields needed
// for Phase 3 v1 stateless delivery are supported.
#pragma once
#include <cstdint>
#include <string>

#include "view_window.hpp"

#if defined(_MSC_VER) && !defined(OHTJ2K_STATIC)
  #define OPENHTJ2K_JPIP_EXPORT __declspec(dllexport)
#else
  #define OPENHTJ2K_JPIP_EXPORT
#endif

namespace open_htj2k {
namespace jpip {

struct JpipRequest {
  std::string target;       // §C.2.1: filename / resource identifier
  ViewWindow  view_window;
  bool        has_fsiz   = false;
  bool        has_roff   = false;
  bool        has_rsiz   = false;
  bool        has_comps  = false;
  // §C.4 Table C.4: image return type.  Must be "jpp-stream" for our use.
  std::string type;
  // §C.9: client cache model — lists data-bins the client already has.
  std::string model;
  // §C.6.1 Maximum Response Length — byte cap on the response payload
  // (the EOR message itself does not count per §D.3).  When honoured,
  // the server emits messages up to the cap and terminates with
  // EOR reason=4 (ByteLimit).
  uint64_t    len         = 0;
  bool        has_len     = false;
  // §C.6.x Quality — cap on the number of quality layers in the
  // response.  Not yet enforced; accepted and ignored.
  uint32_t    quality     = 0;
  bool        has_quality = false;
  // §C.3.3 `cnew` — client requests a new session/channel.  Value is the
  // requested transport (typically "http").  The server reserves a
  // channel identifier and echoes it back in a `JPIP-cnew:` response
  // header so the client can bind subsequent precinct data into a
  // cacheable session.  Reference GUI clients refuse to commit precincts
  // to their cache without this binding — without the header they loop
  // re-requesting data and never advance the cache model.
  std::string cnew;
  bool        has_cnew    = false;
  // §C.3.3 `cid` — channel identifier for follow-up requests on an
  // existing session.  Our server is stateless (it re-derives the view
  // window and honours the client's cache model on every request), so
  // cid arrives for trace only; accept it without validation.
  std::string cid;
  bool        has_cid     = false;
};

enum class RequestParseStatus : uint8_t {
  Ok                  = 0,
  EmptyQuery          = 1,
  UnsupportedType     = 2,  // type != "jpp-stream"
  MalformedField      = 3,  // bad syntax in a known field
  UnknownField        = 4,  // unrecognised key (informational)
};

// Parse a query string ("key=val&key=val&...") into a JpipRequest.
// Leading '?' is accepted and stripped.  Unknown fields are silently
// ignored (per §C.1: "A server shall ignore request fields it does not
// recognize").  Returns Ok on success; on failure the request is
// partially populated and the status indicates the first problem.
OPENHTJ2K_JPIP_EXPORT RequestParseStatus
parse_jpip_query(const std::string &query, JpipRequest *out);

// Extract the query-string portion from a full HTTP request line
// ("GET /path?query HTTP/1.1") and also the path.  Returns false if
// the line is not a valid GET request.
OPENHTJ2K_JPIP_EXPORT bool
split_http_get_line(const std::string &request_line,
                    std::string *path, std::string *query);

}  // namespace jpip
}  // namespace open_htj2k
