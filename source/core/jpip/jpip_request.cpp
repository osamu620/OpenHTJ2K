// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.

#include "jpip_request.hpp"

#include <cstdlib>

namespace open_htj2k {
namespace jpip {

namespace {

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
      // Comma-separated uint16 list, e.g. "0,1,2" or "0-2".
      // v1: only comma-separated individual indices.
      out->view_window.comps.clear();
      std::size_t p = 0;
      while (p < val.size()) {
        const std::size_t c = val.find(',', p);
        const std::string item = val.substr(p, c == std::string::npos ? std::string::npos : c - p);
        if (!item.empty()) {
          out->view_window.comps.push_back(static_cast<uint16_t>(std::strtoul(item.c_str(), nullptr, 10)));
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
      out->type = val;
      if (val != "jpp-stream") {
        status = RequestParseStatus::UnsupportedType;
        return;
      }
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
