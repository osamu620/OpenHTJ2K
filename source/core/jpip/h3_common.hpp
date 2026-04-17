// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.
//
// Shared types for the HTTP/3 JPIP transport (Phase 5, Annex P).

#ifndef OPENHTJ2K_H3_COMMON_HPP
#define OPENHTJ2K_H3_COMMON_HPP

#ifdef OPENHTJ2K_ENABLE_QUIC

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace open_htj2k {
namespace jpip {

struct TlsCertConfig {
  std::string cert_file;
  std::string key_file;
};

struct H3Request {
  std::string method;
  std::string path;
  std::string query;
  int64_t     stream_id = -1;
};

using H3RequestHandler = std::function<std::vector<uint8_t>(const H3Request &req)>;

}  // namespace jpip
}  // namespace open_htj2k

#endif  // OPENHTJ2K_ENABLE_QUIC
#endif  // OPENHTJ2K_H3_COMMON_HPP
