// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.

#include "jpip_response.hpp"

#include <cstdio>
#include <cstring>

namespace open_htj2k {
namespace jpip {

std::vector<uint8_t> format_jpp_response(const uint8_t *body, std::size_t body_len,
                                         const std::string &target_id,
                                         const std::string &cnew_header) {
  char header[512];
  int n = std::snprintf(header, sizeof(header),
                        "HTTP/1.1 200 OK\r\n"
                        "Content-Type: image/jpp-stream\r\n"
                        "Content-Length: %zu\r\n"
                        "Access-Control-Allow-Origin: *\r\n"
                        "Connection: close\r\n",
                        body_len);
  std::vector<uint8_t> out;
  out.reserve(static_cast<std::size_t>(n) + target_id.size() + cnew_header.size() + 64 + body_len);
  out.insert(out.end(), reinterpret_cast<const uint8_t *>(header),
             reinterpret_cast<const uint8_t *>(header) + n);
  if (!target_id.empty()) {
    char tid[256];
    int tn = std::snprintf(tid, sizeof(tid), "JPIP-tid: %s\r\n", target_id.c_str());
    out.insert(out.end(), reinterpret_cast<const uint8_t *>(tid),
               reinterpret_cast<const uint8_t *>(tid) + tn);
  }
  if (!cnew_header.empty()) {
    char cn[512];
    int cnn = std::snprintf(cn, sizeof(cn), "JPIP-cnew: %s\r\n", cnew_header.c_str());
    out.insert(out.end(), reinterpret_cast<const uint8_t *>(cn),
               reinterpret_cast<const uint8_t *>(cn) + cnn);
  }
  out.push_back('\r');
  out.push_back('\n');
  if (body != nullptr && body_len > 0) {
    out.insert(out.end(), body, body + body_len);
  }
  return out;
}

std::vector<uint8_t> format_error_response(int http_status, const std::string &reason) {
  char buf[512];
  int n = std::snprintf(buf, sizeof(buf),
                        "HTTP/1.1 %d %s\r\n"
                        "Content-Length: 0\r\n"
                        "Access-Control-Allow-Origin: *\r\n"
                        "Connection: close\r\n"
                        "\r\n",
                        http_status, reason.c_str());
  return {reinterpret_cast<const uint8_t *>(buf),
          reinterpret_cast<const uint8_t *>(buf) + n};
}

std::size_t parse_http_content_length(const uint8_t *data, std::size_t len,
                                      std::size_t *header_end_out) {
  // Find "\r\n\r\n" boundary.
  const char *s = reinterpret_cast<const char *>(data);
  const char *end = s + len;
  const char *boundary = nullptr;
  for (const char *p = s; p + 4 <= end; ++p) {
    if (p[0] == '\r' && p[1] == '\n' && p[2] == '\r' && p[3] == '\n') {
      boundary = p;
      break;
    }
  }
  if (!boundary) return SIZE_MAX;
  if (header_end_out) *header_end_out = static_cast<std::size_t>(boundary - s) + 4;

  // Scan headers for "Content-Length: N".
  for (const char *line = s; line < boundary;) {
    const char *eol = static_cast<const char *>(std::memchr(line, '\n', static_cast<std::size_t>(boundary - line)));
    if (!eol) eol = boundary;
    const std::size_t line_len = static_cast<std::size_t>(eol - line);
    if (line_len > 16 &&
        (line[0] == 'C' || line[0] == 'c') &&
        (std::strncmp(line, "Content-Length:", 15) == 0 ||
         std::strncmp(line, "content-length:", 15) == 0)) {
      const char *val = line + 15;
      while (val < eol && (*val == ' ' || *val == '\t')) ++val;
      return static_cast<std::size_t>(std::strtoull(val, nullptr, 10));
    }
    line = eol + 1;
  }
  return SIZE_MAX;
}

}  // namespace jpip
}  // namespace open_htj2k
