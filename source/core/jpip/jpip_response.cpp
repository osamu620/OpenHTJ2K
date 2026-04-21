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

std::vector<uint8_t> format_jpp_response_headers_chunked(const std::string &target_id,
                                                         const std::string &cnew_header) {
  // Same status line + JPIP response headers as the buffered path but with
  // `Transfer-Encoding: chunked` in place of `Content-Length`.  HTTP/1.1
  // §4.1: a message MUST NOT carry both Content-Length and Transfer-Encoding.
  char header[512];
  int n = std::snprintf(header, sizeof(header),
                        "HTTP/1.1 200 OK\r\n"
                        "Content-Type: image/jpp-stream\r\n"
                        "Transfer-Encoding: chunked\r\n"
                        "Access-Control-Allow-Origin: *\r\n"
                        "Connection: close\r\n");
  std::vector<uint8_t> out;
  out.reserve(static_cast<std::size_t>(n) + target_id.size() + cnew_header.size() + 64);
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
  return out;
}

std::vector<uint8_t> format_chunk_header(std::size_t n) {
  char buf[24];
  int w = std::snprintf(buf, sizeof(buf), "%zx\r\n", n);
  return {reinterpret_cast<const uint8_t *>(buf),
          reinterpret_cast<const uint8_t *>(buf) + w};
}

std::vector<uint8_t> format_last_chunk() {
  static const uint8_t kLast[] = {'0', '\r', '\n', '\r', '\n'};
  return {kLast, kLast + sizeof(kLast)};
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

bool parse_http_chunked(const uint8_t *data, std::size_t len,
                        std::size_t *header_end_out) {
  const char *s = reinterpret_cast<const char *>(data);
  const char *end = s + len;
  const char *boundary = nullptr;
  for (const char *p = s; p + 4 <= end; ++p) {
    if (p[0] == '\r' && p[1] == '\n' && p[2] == '\r' && p[3] == '\n') {
      boundary = p;
      break;
    }
  }
  if (!boundary) return false;
  if (header_end_out) *header_end_out = static_cast<std::size_t>(boundary - s) + 4;

  for (const char *line = s; line < boundary;) {
    const char *eol = static_cast<const char *>(
        std::memchr(line, '\n', static_cast<std::size_t>(boundary - line)));
    if (!eol) eol = boundary;
    const std::size_t line_len = static_cast<std::size_t>(eol - line);
    if (line_len > 18 &&
        (std::strncmp(line, "Transfer-Encoding:", 18) == 0 ||
         std::strncmp(line, "transfer-encoding:", 18) == 0)) {
      const char *val = line + 18;
      while (val < eol && (*val == ' ' || *val == '\t')) ++val;
      // Only care whether "chunked" is present in the comma-separated list.
      for (const char *p = val; p + 7 <= eol; ++p) {
        if ((p[0] == 'c' || p[0] == 'C') && (p[1] == 'h' || p[1] == 'H')
            && (p[2] == 'u' || p[2] == 'U') && (p[3] == 'n' || p[3] == 'N')
            && (p[4] == 'k' || p[4] == 'K') && (p[5] == 'e' || p[5] == 'E')
            && (p[6] == 'd' || p[6] == 'D')) {
          return true;
        }
      }
    }
    line = eol + 1;
  }
  return false;
}

bool decode_chunked_body(const uint8_t *data, std::size_t len,
                         std::vector<uint8_t> *out) {
  if (!out) return false;
  out->clear();
  std::size_t i = 0;
  while (i < len) {
    // Parse the size line terminated by CRLF.  Per RFC 7230 §4.1 it may
    // carry chunk extensions after a `;`; we ignore them.
    std::size_t line_end = i;
    while (line_end + 1 < len && !(data[line_end] == '\r' && data[line_end + 1] == '\n')) {
      ++line_end;
    }
    if (line_end + 1 >= len) return false;
    std::size_t size_end = i;
    while (size_end < line_end && data[size_end] != ';') ++size_end;
    std::size_t chunk_size = 0;
    bool any_digit = false;
    for (std::size_t j = i; j < size_end; ++j) {
      const uint8_t c = data[j];
      int v;
      if (c >= '0' && c <= '9')      v = c - '0';
      else if (c >= 'a' && c <= 'f') v = 10 + (c - 'a');
      else if (c >= 'A' && c <= 'F') v = 10 + (c - 'A');
      else return false;
      chunk_size = (chunk_size << 4) | static_cast<std::size_t>(v);
      any_digit = true;
    }
    if (!any_digit) return false;
    i = line_end + 2;
    if (chunk_size == 0) {
      // Skip trailers (any `name: value\r\n` lines) up to the final CRLF.
      while (i + 1 < len) {
        if (data[i] == '\r' && data[i + 1] == '\n') { i += 2; return true; }
        std::size_t te = i;
        while (te + 1 < len && !(data[te] == '\r' && data[te + 1] == '\n')) ++te;
        if (te + 1 >= len) return false;
        i = te + 2;
      }
      return false;
    }
    if (i + chunk_size + 2 > len) return false;
    out->insert(out->end(), data + i, data + i + chunk_size);
    i += chunk_size;
    if (data[i] != '\r' || data[i + 1] != '\n') return false;
    i += 2;
  }
  return false;
}

}  // namespace jpip
}  // namespace open_htj2k
