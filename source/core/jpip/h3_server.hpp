// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.
//
// HTTP/3 JPIP server — wraps MsQuic (QUIC transport) + nghttp3 (HTTP/3 framing).

#ifndef OPENHTJ2K_H3_SERVER_HPP
#define OPENHTJ2K_H3_SERVER_HPP

#ifdef OPENHTJ2K_ENABLE_QUIC

#include <cstdint>
#include <string>

#include "h3_common.hpp"

namespace open_htj2k {
namespace jpip {

class H3Server {
 public:
  H3Server();
  ~H3Server();

  H3Server(const H3Server &) = delete;
  H3Server &operator=(const H3Server &) = delete;

  bool start(uint16_t port, const TlsCertConfig &tls, H3RequestHandler handler);
  void stop();
  std::string last_error() const;

  struct Impl;
 private:
  Impl *impl_;
};

}  // namespace jpip
}  // namespace open_htj2k

#endif  // OPENHTJ2K_ENABLE_QUIC
#endif  // OPENHTJ2K_H3_SERVER_HPP
