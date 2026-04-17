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

  // Parse Content-Length and find body start.
  std::size_t header_end = 0;
  const std::size_t content_length = parse_http_content_length(raw.data(), raw.size(), &header_end);
  if (content_length == SIZE_MAX) {
    err_ = "missing Content-Length header";
    return false;
  }

  // We may have already received part of the body in the header read.
  const std::size_t body_already = raw.size() - header_end;
  std::vector<uint8_t> body;
  body.reserve(content_length);
  body.insert(body.end(), raw.data() + header_end, raw.data() + raw.size());

  // Read the remaining body bytes.
  if (body.size() < content_length) {
    const std::size_t remaining = content_length - body.size();
    const std::size_t prev = body.size();
    body.resize(content_length);
    if (!conn.recv_all(body.data() + prev, remaining)) {
      err_ = "recv body: " + conn.last_error();
      return false;
    }
  }

  // Parse the JPP-stream.
  if (!parse_jpp_stream(body.data(), body.size(), out)) {
    err_ = "parse_jpp_stream failed";
    return false;
  }
  return true;
}

}  // namespace jpip
}  // namespace open_htj2k
