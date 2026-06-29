// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.

#include "jpip_request.hpp"

#include <cstdlib>

namespace open_htj2k {
namespace jpip {

namespace {

// A codestream has at most 16384 components (Csiz range, ISO/IEC 15444-1
// Table A.9), so a component selection can never name more than that many.
// We cap the expanded `comps` list here so a tiny query such as
// "comps=0-4294967295" (or a long run of repeated ranges) cannot expand
// into a multi-billion-entry push loop — a memory/CPU DoS on the server.
constexpr std::size_t kJpipMaxComponents = 16384;

// Split "key=val" pairs from a '&'-delimited query string.  Calls fn
// with each (key, val) pair; val may be empty.
template <typename Fn>
void for_each_param(const std::string &query, Fn &&fn) {
  std::size_t pos = 0;
  while (pos < query.size()) {
    const std::size_t amp = query.find('&', pos);
    const std::string param = query.substr(pos, amp == std::string::npos ? std::string::npos : amp - pos);
    pos = (amp == std::string::npos) ? query.size() : amp + 1;
    if (param.empty()) continue;
    const std::size_t eq = param.find('=');
    if (eq == std::string::npos) {
      fn(param, std::string());
    } else {
      fn(param.substr(0, eq), param.substr(eq + 1));
    }
  }
}

// Parse "N,M" into two uint32_t values.  Returns false on any syntax error.
bool parse_pair(const std::string &val, uint32_t &a, uint32_t &b) {
  const std::size_t comma = val.find(',');
  if (comma == std::string::npos) return false;
  char *end = nullptr;
  a = static_cast<uint32_t>(std::strtoul(val.c_str(), &end, 10));
  if (end != val.c_str() + comma) return false;
  b = static_cast<uint32_t>(std::strtoul(val.c_str() + comma + 1, &end, 10));
  if (*end != '\0') return false;
  return true;
}

// §C.4: one image-return-type element, either the reserved short token
// ("jpp-stream") or the media-type form ("image/jpp-stream"), optionally
// followed by ";parameter" suffixes which we ignore.
bool is_acceptable_return_type(const std::string &item) {
  std::size_t b = 0, e = item.size();
  while (b < e && (item[b] == ' ' || item[b] == '\t')) ++b;
  while (e > b && (item[e - 1] == ' ' || item[e - 1] == '\t')) --e;
  const std::size_t semi = item.find(';', b);
  if (semi != std::string::npos && semi < e) e = semi;
  const std::size_t n = e - b;
  return (n == 10 && item.compare(b, n, "jpp-stream") == 0) ||
         (n == 16 && item.compare(b, n, "image/jpp-stream") == 0);
}

}  // namespace

RequestParseStatus parse_jpip_query(const std::string &query_in, JpipRequest *out) {
  if (!out) return RequestParseStatus::EmptyQuery;
  *out = {};
  std::string query = query_in;
  if (!query.empty() && query[0] == '?') query = query.substr(1);
  if (query.empty()) return RequestParseStatus::EmptyQuery;

  RequestParseStatus status = RequestParseStatus::Ok;
  for_each_param(query, [&](const std::string &key, const std::string &val) {
    if (status != RequestParseStatus::Ok) return;

    if (key == "target") {
      out->target = val;
    } else if (key == "fsiz") {
      // "fx,fy[,round-direction]"
      // Split on first comma to get fx, then check for a second comma.
      const std::size_t c1 = val.find(',');
      if (c1 == std::string::npos) { status = RequestParseStatus::MalformedField; return; }
      const std::size_t c2 = val.find(',', c1 + 1);
      char *end = nullptr;
      out->view_window.fx = static_cast<uint32_t>(std::strtoul(val.c_str(), &end, 10));
      if (end != val.c_str() + c1) { status = RequestParseStatus::MalformedField; return; }
      if (c2 == std::string::npos) {
        out->view_window.fy = static_cast<uint32_t>(std::strtoul(val.c_str() + c1 + 1, &end, 10));
        if (*end != '\0') { status = RequestParseStatus::MalformedField; return; }
      } else {
        out->view_window.fy = static_cast<uint32_t>(std::strtoul(val.c_str() + c1 + 1, &end, 10));
        if (end != val.c_str() + c2) { status = RequestParseStatus::MalformedField; return; }
        const std::string rd = val.substr(c2 + 1);
        if (rd == "round-up")        out->view_window.round = ViewWindow::Round::Up;
        else if (rd == "round-down")  out->view_window.round = ViewWindow::Round::Down;
        else if (rd == "closest")     out->view_window.round = ViewWindow::Round::Closest;
        else { status = RequestParseStatus::MalformedField; return; }
      }
      out->has_fsiz = true;
    } else if (key == "roff") {
      if (!parse_pair(val, out->view_window.ox, out->view_window.oy)) {
        status = RequestParseStatus::MalformedField; return;
      }
      out->has_roff = true;
    } else if (key == "rsiz") {
      if (!parse_pair(val, out->view_window.sx, out->view_window.sy)) {
        status = RequestParseStatus::MalformedField; return;
      }
      out->has_rsiz = true;
    } else if (key == "comps") {
      // §C.4.6: comma-separated list of indices or ranges, e.g. "0,2" or
      // "0-2" or "0,3-5".
      out->view_window.comps.clear();
      std::size_t p = 0;
      while (p < val.size()) {
        const std::size_t c = val.find(',', p);
        const std::string item = val.substr(p, c == std::string::npos ? std::string::npos : c - p);
        if (!item.empty()) {
          char *end = nullptr;
          const unsigned long lo = std::strtoul(item.c_str(), &end, 10);
          unsigned long hi = lo;
          if (end == item.c_str()) { status = RequestParseStatus::MalformedField; return; }
          if (*end == '-') {
            const char *hs = end + 1;
            hi = std::strtoul(hs, &end, 10);
            if (end == hs || *end != '\0' || hi < lo) {
              status = RequestParseStatus::MalformedField; return;
            }
          } else if (*end != '\0') {
            status = RequestParseStatus::MalformedField; return;
          }
          for (unsigned long v = lo; v <= hi; ++v) {
            if (out->view_window.comps.size() >= kJpipMaxComponents) {
              // More components named than can possibly exist — reject
              // rather than keep expanding an attacker-controlled range.
              status = RequestParseStatus::MalformedField;
              return;
            }
            out->view_window.comps.push_back(static_cast<uint16_t>(v));
          }
        }
        p = (c == std::string::npos) ? val.size() : c + 1;
      }
      out->has_comps = true;
    } else if (key == "model") {
      out->model = val;
    } else if (key == "len") {
      // §C.6.1 Maximum Response Length — non-negative decimal byte cap.
      char *end = nullptr;
      const unsigned long long n = std::strtoull(val.c_str(), &end, 10);
      if (end == val.c_str() || *end != '\0') {
        status = RequestParseStatus::MalformedField;
        return;
      }
      out->len     = static_cast<uint64_t>(n);
      out->has_len = true;
    } else if (key == "quality") {
      // §C.6.x Quality — non-negative decimal quality-layer cap.
      char *end = nullptr;
      const unsigned long n = std::strtoul(val.c_str(), &end, 10);
      if (end == val.c_str() || *end != '\0') {
        status = RequestParseStatus::MalformedField;
        return;
      }
      out->quality     = static_cast<uint32_t>(n);
      out->has_quality = true;
    } else if (key == "type") {
      // §C.4: comma-separated preference list; accept the request if ANY
      // element is a jpp-stream form.  We always serve jpp-stream, so the
      // stored type is the token we will put on the wire.
      bool acceptable = false;
      std::size_t p = 0;
      while (p <= val.size()) {
        std::size_t c = val.find(',', p);
        if (c == std::string::npos) c = val.size();
        if (is_acceptable_return_type(val.substr(p, c - p))) { acceptable = true; break; }
        p = c + 1;
      }
      if (!acceptable) {
        out->type = val;
        status = RequestParseStatus::UnsupportedType;
        return;
      }
      out->type = "jpp-stream";
    } else if (key == "cnew") {
      // §C.3.3: client requests a new JPIP channel; value is the preferred
      // transport (typically "http").  Server echoes a channel id back in
      // the JPIP-cnew response header so the client can commit received
      // data to a session-scoped cache.
      out->cnew     = val;
      out->has_cnew = true;
    } else if (key == "cid") {
      // §C.3.3: channel id from a previously issued JPIP-cnew.  Stateless
      // servers need only trace this — the cache model field carries all
      // information required for each response.
      out->cid     = val;
      out->has_cid = true;
    } else if (key == "cclose") {
      // §C.3.4: "*" or a comma-separated list of channel-ids.
      out->cclose     = val;
      out->has_cclose = true;
    } else if (key == "qid") {
      // §C.3.5: non-negative decimal request ID.
      char *end = nullptr;
      const unsigned long long n = std::strtoull(val.c_str(), &end, 10);
      if (end == val.c_str() || *end != '\0') {
        status = RequestParseStatus::MalformedField;
        return;
      }
      out->qid     = static_cast<uint64_t>(n);
      out->has_qid = true;
    }
    // Unknown keys are silently ignored per §C.1.
  });

  return status;
}

bool split_http_get_line(const std::string &line, std::string *path, std::string *query) {
  // Expect "GET /path?query HTTP/1.x"
  if (line.size() < 14) return false;  // "GET / HTTP/1.1" minimum
  if (line.substr(0, 4) != "GET ") return false;
  const std::size_t sp = line.find(' ', 4);
  if (sp == std::string::npos) return false;
  const std::string uri = line.substr(4, sp - 4);
  const std::size_t qm = uri.find('?');
  if (qm == std::string::npos) {
    if (path) *path = uri;
    if (query) query->clear();
  } else {
    if (path) *path = uri.substr(0, qm);
    if (query) *query = uri.substr(qm + 1);
  }
  return true;
}

}  // namespace jpip
}  // namespace open_htj2k
