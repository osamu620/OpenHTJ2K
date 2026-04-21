// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.

#include "jpip_client.hpp"

#include <cstdio>

#include "cache_model.hpp"
#include "jpip_response.hpp"
#include "tcp_socket.hpp"

namespace open_htj2k {
namespace jpip {

std::string format_view_window_query(const ViewWindow &vw) {
  std::string q;
  q += "fsiz=";
  q += std::to_string(vw.fx);
  q += ",";
  q += std::to_string(vw.fy);
  switch (vw.round) {
    case ViewWindow::Round::Up:      q += ",round-up"; break;
    case ViewWindow::Round::Closest: q += ",closest";  break;
    default: break;
  }
  if (vw.ox != 0 || vw.oy != 0) {
    q += "&roff=";
    q += std::to_string(vw.ox);
    q += ",";
    q += std::to_string(vw.oy);
  }
  if (vw.sx != 0 || vw.sy != 0) {
    q += "&rsiz=";
    q += std::to_string(vw.sx);
    q += ",";
    q += std::to_string(vw.sy);
  }
  if (!vw.comps.empty()) {
    q += "&comps=";
    for (std::size_t i = 0; i < vw.comps.size(); ++i) {
      if (i > 0) q += ",";
      q += std::to_string(vw.comps[i]);
    }
  }
  q += "&type=jpp-stream";
  return q;
}

bool JpipClient::fetch(const std::string &host, uint16_t port,
                       const ViewWindow &vw, DataBinSet *out,
                       const CacheModel *model) {
  return fetch_streaming(host, port, vw, out, model, /*on_progress=*/{});
}

namespace {

// Incremental HTTP/1.1 chunked-transfer decoder.  feed() accepts bytes
// off the wire and invokes `on_payload(p, n)` once per non-empty chunk
// payload.  Returns true until the terminating 0-chunk is seen, at
// which point `*done` is set and feed() must not be called again.
// Tolerates chunk-extensions (ignored) but expects the terminating
// 0-chunk to be followed by a simple `\r\n` (no trailer headers),
// which is what our server emits.
struct ChunkedStreamDecoder {
  std::vector<uint8_t> buf;
  std::size_t          consumed = 0;
  bool                 done     = false;
  // Payload remaining in the current chunk; 0 when at a size line.
  std::size_t          chunk_left     = 0;
  bool                 in_chunk       = false;
  bool                 awaiting_crlf  = false;  // after payload, before next size
  bool                 awaiting_final = false;  // after size=0 line, expect final CRLF

  template <typename Sink>
  bool feed(const uint8_t *data, std::size_t len, Sink &&on_payload,
            std::string *err) {
    buf.insert(buf.end(), data, data + len);
    for (;;) {
      if (in_chunk) {
        const std::size_t available = buf.size() - consumed;
        if (available == 0) return true;
        const std::size_t take = std::min(chunk_left, available);
        if (take > 0) {
          if (!on_payload(buf.data() + consumed, take)) {
            *err = "jpp parse failed mid-chunk";
            return false;
          }
          consumed   += take;
          chunk_left -= take;
        }
        if (chunk_left == 0) {
          in_chunk      = false;
          awaiting_crlf = true;
        } else {
          return true;  // need more bytes
        }
      }
      if (awaiting_crlf) {
        if (buf.size() - consumed < 2) return true;
        if (buf[consumed] != '\r' || buf[consumed + 1] != '\n') {
          *err = "chunked: missing CRLF after payload";
          return false;
        }
        consumed      += 2;
        awaiting_crlf  = false;
      }
      if (awaiting_final) {
        if (buf.size() - consumed < 2) return true;
        // Our server emits `0\r\n\r\n` (no trailers).  Accept either that
        // or an immediate `\r\n` (per RFC 7230 §4.1.2 an empty trailer).
        if (buf[consumed] != '\r' || buf[consumed + 1] != '\n') {
          *err = "chunked: unexpected bytes after terminating 0-chunk";
          return false;
        }
        consumed       += 2;
        awaiting_final  = false;
        done            = true;
        return true;
      }
      // Parse a size line.
      const uint8_t *line = buf.data() + consumed;
      const std::size_t rem = buf.size() - consumed;
      const uint8_t *eol = static_cast<const uint8_t *>(std::memchr(line, '\n', rem));
      if (!eol) return true;  // wait for more bytes
      if (eol == line || *(eol - 1) != '\r') {
        *err = "chunked: malformed size line";
        return false;
      }
      const std::size_t line_len = static_cast<std::size_t>(eol - line) - 1;
      // Stop at chunk-extension separator `;`.
      std::size_t size_end = 0;
      while (size_end < line_len && line[size_end] != ';') ++size_end;
      std::size_t size_val = 0;
      bool any_digit = false;
      for (std::size_t i = 0; i < size_end; ++i) {
        const uint8_t c = line[i];
        int v;
        if (c >= '0' && c <= '9')      v = c - '0';
        else if (c >= 'a' && c <= 'f') v = 10 + (c - 'a');
        else if (c >= 'A' && c <= 'F') v = 10 + (c - 'A');
        else { *err = "chunked: non-hex size digit"; return false; }
        size_val = (size_val << 4) | static_cast<std::size_t>(v);
        any_digit = true;
      }
      if (!any_digit) { *err = "chunked: empty size line"; return false; }
      consumed += static_cast<std::size_t>(eol - line) + 1;  // includes \r\n
      if (size_val == 0) {
        awaiting_final = true;
        continue;
      }
      in_chunk   = true;
      chunk_left = size_val;
    }
  }
};

}  // namespace

bool JpipClient::fetch_streaming(const std::string &host, uint16_t port,
                                 const ViewWindow &vw, DataBinSet *out,
                                 const CacheModel *model,
                                 const OnProgressCallback &on_progress) {
  if (!out) { err_ = "null DataBinSet"; return false; }
  *out = {};

  TcpStream conn;
  if (!conn.connect(host, port)) {
    err_ = "connect: " + conn.last_error();
    return false;
  }

  std::string query = format_view_window_query(vw);
  if (model && model->size() > 0) {
    query += "&model=";
    query += model->format();
  }
  std::string request = "GET /jpip?" + query + " HTTP/1.1\r\n"
                        "Host: " + host + "\r\n"
                        "Connection: close\r\n"
                        "\r\n";
  if (!conn.send_all(reinterpret_cast<const uint8_t *>(request.data()), request.size())) {
    err_ = "send: " + conn.last_error();
    return false;
  }

  // Read the response headers.
  std::vector<uint8_t> raw;
  const std::size_t hdr_bytes = conn.recv_until_header_end(raw, 65536);
  if (hdr_bytes == 0) {
    err_ = "recv headers: empty or error";
    return false;
  }
  std::size_t header_end = 0;
  const bool is_chunked = parse_http_chunked(raw.data(), raw.size(), &header_end);
  std::size_t content_length = 0;
  if (!is_chunked) {
    content_length = parse_http_content_length(raw.data(), raw.size(), &header_end);
    if (content_length == SIZE_MAX) {
      err_ = "missing Content-Length/Transfer-Encoding";
      return false;
    }
  }

  // Feed every decoded JPP byte into a StreamingJppParser so the caller's
  // DataBinSet grows as chunks land on the wire rather than after an
  // EOF-driven buffer-and-decode pass.  `on_progress` (if set) fires
  // after each fed chunk that produced at least one completed message.
  StreamingJppParser jpp;
  std::size_t last_bin_count = 0;
  bool        last_had_eor   = false;
  auto feed_jpp = [&](const uint8_t *p, std::size_t n) {
    if (n == 0) return true;
    if (!jpp.feed(p, n, out)) {
      err_ = "StreamingJppParser: malformed JPP-stream";
      return false;
    }
    if (on_progress && (out->size() != last_bin_count || out->has_eor() != last_had_eor)) {
      last_bin_count = out->size();
      last_had_eor   = out->has_eor();
      on_progress(*out);
    }
    return true;
  };

  // Feed the body bytes that came in with the header read.
  const uint8_t *body_start    = raw.data() + header_end;
  const std::size_t body_first = raw.size() - header_end;

  ChunkedStreamDecoder chunked;
  std::size_t cl_received = 0;

  auto push_wire_bytes = [&](const uint8_t *p, std::size_t n) {
    if (is_chunked) {
      return chunked.feed(p, n, feed_jpp, &err_);
    }
    const std::size_t want = std::min(n, content_length - cl_received);
    if (!feed_jpp(p, want)) return false;
    cl_received += want;
    return true;
  };
  if (body_first > 0 && !push_wire_bytes(body_start, body_first)) return false;

  // Read until the transport signals the response is complete.
  uint8_t tmp[16 * 1024];
  while (true) {
    if (is_chunked && chunked.done) break;
    if (!is_chunked && cl_received >= content_length) break;
    const std::size_t n = conn.recv_some(tmp, sizeof(tmp));
    if (n == SIZE_MAX) { err_ = "recv: " + conn.last_error(); return false; }
    if (n == 0) break;  // peer closed
    if (!push_wire_bytes(tmp, n)) return false;
  }

  if (!jpp.finish()) {
    err_ = "stream ended mid JPP-message";
    return false;
  }
  return true;
}

}  // namespace jpip
}  // namespace open_htj2k
