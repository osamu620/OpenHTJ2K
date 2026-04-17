// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.
//
// HTTP/3 JPIP client — wraps MsQuic (QUIC transport) + nghttp3 (HTTP/3 framing).

#ifndef OPENHTJ2K_H3_CLIENT_HPP
#define OPENHTJ2K_H3_CLIENT_HPP

#ifdef OPENHTJ2K_ENABLE_QUIC

#include <cstdint>
#include <string>
#include <vector>

namespace open_htj2k {
namespace jpip {

class H3Client {
 public:
  H3Client();
  ~H3Client();

  H3Client(const H3Client &) = delete;
  H3Client &operator=(const H3Client &) = delete;

  bool connect(const std::string &host, uint16_t port, bool validate_cert = false);
  void disconnect();

  // Sends an HTTP/3 GET request and blocks until the response body is received.
  std::vector<uint8_t> fetch(const std::string &path_and_query);

  std::string last_error() const;

 private:
  struct Impl;
  Impl *impl_;
};

}  // namespace jpip
}  // namespace open_htj2k

#endif  // OPENHTJ2K_ENABLE_QUIC
#endif  // OPENHTJ2K_H3_CLIENT_HPP
