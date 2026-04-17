// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.
//
// JPIP HTTP/1.1 client — sends a view-window request to a JPIP server
// and feeds the JPP-stream response into a DataBinSet.
#pragma once
#include <cstdint>
#include <string>

#include "cache_model.hpp"
#include "jpp_parser.hpp"
#include "view_window.hpp"

#if defined(_MSC_VER) && !defined(OHTJ2K_STATIC)
  #define OPENHTJ2K_JPIP_EXPORT __declspec(dllexport)
#else
  #define OPENHTJ2K_JPIP_EXPORT
#endif

namespace open_htj2k {
namespace jpip {

class OPENHTJ2K_JPIP_EXPORT JpipClient {
 public:
  JpipClient() = default;

  // Format the view-window as an HTTP GET query string, connect to the
  // server, receive the JPP-stream response, and parse it into `*out`.
  // Returns true on success, false on any network / parse error
  // (last_error() holds the reason).  The DataBinSet is cleared at the
  // start of each call (v1 stateless — no incremental caching).
  bool fetch(const std::string &host, uint16_t port,
             const ViewWindow &vw, DataBinSet *out,
             const CacheModel *model = nullptr);

  const std::string &last_error() const { return err_; }

 private:
  std::string err_;
};

// Format a ViewWindow as a JPIP query string suitable for an HTTP GET.
OPENHTJ2K_JPIP_EXPORT std::string format_view_window_query(const ViewWindow &vw);

}  // namespace jpip
}  // namespace open_htj2k
