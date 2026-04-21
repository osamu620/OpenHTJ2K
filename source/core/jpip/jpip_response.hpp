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

// Format only the HTTP/1.1 response headers for a chunked (`Transfer-Encoding:
// chunked`) JPP-stream delivery.  The body bytes are written separately as
// HTTP/1.1 chunks (see `format_chunk_header` / `format_last_chunk` below),
// letting the server push each JPP message to the socket as soon as it is
// produced.  The returned vector ends with the blank line that separates
// headers from body.
OPENHTJ2K_JPIP_EXPORT std::vector<uint8_t>
format_jpp_response_headers_chunked(const std::string &target_id = "",
                                    const std::string &cnew_header = "");

// Build the HTTP/1.1 chunk header for a payload of `n` bytes: the hex length
// followed by CRLF.  The caller writes this, then the `n` bytes of payload,
// then a literal CRLF.  `n` must be non-zero — use `format_last_chunk` for
// the terminating chunk.
OPENHTJ2K_JPIP_EXPORT std::vector<uint8_t> format_chunk_header(std::size_t n);

// Build the terminating chunk (`0\r\n\r\n`) for a chunked response.
OPENHTJ2K_JPIP_EXPORT std::vector<uint8_t> format_last_chunk();

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

// True if the HTTP response header block `data..data+len` advertises
// `Transfer-Encoding: chunked`.  `*header_end_out` is set to the byte
// offset past the \r\n\r\n boundary (matching parse_http_content_length).
OPENHTJ2K_JPIP_EXPORT bool
parse_http_chunked(const uint8_t *data, std::size_t len,
                   std::size_t *header_end_out);

// Decode a chunked body buffered in `data..data+len` into `out`.  Expects
// a complete chunked stream (all chunks present, terminating `0\r\n\r\n`).
// Returns true on success.  Trailers are discarded.
OPENHTJ2K_JPIP_EXPORT bool
decode_chunked_body(const uint8_t *data, std::size_t len,
                    std::vector<uint8_t> *out);

}  // namespace jpip
}  // namespace open_htj2k
