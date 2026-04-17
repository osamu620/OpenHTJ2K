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
