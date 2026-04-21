// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.
//
// JPIP HTTP/1.1 response formatting per ISO/IEC 15444-9 §D.2 + Annex F.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#if defined(_MSC_VER) && !defined(OHTJ2K_STATIC)
  #define OPENHTJ2K_JPIP_EXPORT __declspec(dllexport)
#else
  #define OPENHTJ2K_JPIP_EXPORT
#endif

namespace open_htj2k {
namespace jpip {

// Format a complete HTTP/1.1 response (headers + body) for a JPP-stream
// payload.  `target_id` is the JPIP-tid response header (§D.2.3); an
// empty string omits the header.  The response uses Connection: close
// for v1 simplicity.
OPENHTJ2K_JPIP_EXPORT std::vector<uint8_t>
format_jpp_response(const uint8_t *body, std::size_t body_len,
                    const std::string &target_id = "",
                    const std::string &cnew_header = "");

// Format a simple error response (no body).
OPENHTJ2K_JPIP_EXPORT std::vector<uint8_t>
format_error_response(int http_status, const std::string &reason);

// Parse the Content-Length from an HTTP response header block.  Returns
// the body length if found, or SIZE_MAX if not parseable.  Also reports
// the offset of the first byte after the blank line ("\r\n\r\n") that
// separates headers from body.
OPENHTJ2K_JPIP_EXPORT std::size_t
parse_http_content_length(const uint8_t *data, std::size_t len,
                          std::size_t *header_end_out);

}  // namespace jpip
}  // namespace open_htj2k
