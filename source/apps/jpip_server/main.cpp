// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.
//
// open_htj2k_jpip_server: stateless JPIP server (HTTP/1.1 or HTTP/3).
//
// Loads a single JPEG 2000 codestream, builds the JPIP index + packet
// locator once, then serves view-window requests.
//
// Usage:
//   open_htj2k_jpip_server <input.j2c> [--port N=8080]
//       [--h3 --cert server.cert --key server.key]
//
// Each request is an HTTP GET with JPIP query parameters:
//   GET /jpip?fsiz=W,H&roff=X,Y&rsiz=W,H&type=jpp-stream HTTP/1.1
//
// The server responds with a complete JPP-stream containing the
// main-header, tile-header, metadata-bin-0, and every precinct data-bin
// selected by the view-window resolver.
//
// Requests are served statelessly (the client's `model=` field carries
// its cache state) unless the client establishes a session with
// `cnew=http`: the server then issues a channel-id (JPIP-cnew, §D.2.3)
// and keeps a per-channel cache model so bins already delivered on that
// channel are never re-sent (§B.2).

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "codestream_walker.hpp"
#include "data_bin_emitter.hpp"
#include "jpp_message.hpp"
#include "jpip_request.hpp"
#include "jpip_response.hpp"
#include "packet_locator.hpp"
#include "precinct_index.hpp"
#include "cache_model.hpp"
#include "channel_manager.hpp"
#include "tcp_socket.hpp"
#include "view_window.hpp"
#ifdef OPENHTJ2K_ENABLE_QUIC
#include "h3_server.hpp"
#endif

using namespace open_htj2k::jpip;

namespace {

std::vector<uint8_t> read_file(const char *path) {
  FILE *f = std::fopen(path, "rb");
  if (!f) { std::fprintf(stderr, "ERROR: cannot open %s\n", path); return {}; }
  std::fseek(f, 0, SEEK_END);
  auto sz = static_cast<std::size_t>(std::ftell(f));
  std::fseek(f, 0, SEEK_SET);
  std::vector<uint8_t> buf(sz);
  std::size_t rd = std::fread(buf.data(), 1, sz, f);
  std::fclose(f);
  if (rd != sz) buf.clear();
  return buf;
}

struct ServerState {
  std::vector<uint8_t>              codestream;
  std::unique_ptr<CodestreamIndex>  idx;
  CodestreamLayout                  layout;
  std::unique_ptr<PacketLocator>    locator;
  std::string                       target_id;
  // When true, HTTP/1.1 responses use `Transfer-Encoding: chunked` and
  // flush each JPP message to the socket as it is produced — time-to-first-
  // byte then equals the time to emit metadata-bin 0 rather than the time
  // to build the full response.  Opt-in via `--chunked` because some
  // interactive reference JPIP clients lack chunked-transfer support in
  // their HTTP/1.1 parsers and report "connection closed unexpectedly"
  // when the Content-Length header they were expecting is absent.  Our
  // browser demos + C++ JpipClient both accept either format, so the
  // conservative default is Content-Length.
  bool                              chunked_responses = false;
  // §B.2 sessions: per-channel cache models, keyed by the cid issued in
  // JPIP-cnew.  Granting a channel commits the server to never re-sending
  // bins already delivered on it; ChannelManager carries that state.
  ChannelManager                    channels;
};

// Callback invoked once per completed JPP-stream message (including EOR).
// Returning false aborts further emission; the caller is expected to have
// flushed what it already received.  `is_eor` lets the transport treat the
// final message differently (e.g. set FIN, stop accounting bytes against
// the §C.6.1 cap).
using JppMessageSink = std::function<bool(const uint8_t *data, std::size_t len, bool is_eor)>;

// Stream the JPP-stream for a given ViewWindow, skipping data-bins the
// client already has (per the cache model from §C.9).  Each completed
// message is handed to `sink` as soon as it is built, so a chunked HTTP
// transport can push it to the socket immediately.  When `n_keys_out` is
// non-null the caller gets the precinct count back without having to
// re-run resolve_view_window.  `total_bytes_out`, if non-null, receives
// the cumulative payload size (excluding EOR, matching the §C.6.1 cap
// accounting).
//
// `max_bytes` is the §C.6.1 "Maximum Response Length" cap: the cumulative
// response size excluding the EOR message (the EOR does not count per
// §D.3).  Bins are delivered through resumable byte windows (BinWindow):
// a bin larger than the remaining budget contributes a prefix now and
// resumes at that byte offset on the next request, so byte-limited
// clients always make forward progress.  EOR reason becomes `ByteLimit`
// (4) instead of `WindowDone` (2) when the cap interrupted delivery.
// Pass UINT64_MAX for "no cap"; `len=0` is a legal request meaning "EOR
// only", so 0 is a real cap value here.
//
// `sent_out`, when non-null, receives one SentBin per data-bin that
// contributed bytes to the response — what a session server must commit
// to the channel's cache model afterwards.
bool stream_jpp_response(const ServerState &st, const ViewWindow &vw,
                         const CacheModel &client_cache, uint64_t max_bytes,
                         const JppMessageSink &sink,
                         size_t *n_keys_out = nullptr,
                         std::size_t *total_bytes_out = nullptr,
                         std::vector<SentBin> *sent_out = nullptr) {
  auto keys = resolve_view_window(*st.idx, vw);
  if (n_keys_out) *n_keys_out = keys.size();

  MessageHeaderContext ctx;
  std::vector<uint8_t> scratch;
  scratch.reserve(64 * 1024);
  std::size_t total = 0;
  bool aborted   = false;
  bool truncated = false;

  // Deliver one data-bin (or the part of it the remaining budget allows)
  // to `sink`.  A bin the client holds in full is skipped; a partial
  // holding resumes at its recorded byte offset (§C.9.2).  Returns false
  // when emission of further bins must stop — budget exhausted
  // (`truncated`) or sink failure (`aborted`).  The emitter only encodes
  // message headers for messages it actually appends, so `ctx` needs no
  // rollback handling here.
  auto emit_bin = [&](uint8_t cls, uint64_t id, auto &&fn) -> bool {
    if (client_cache.has(cls, id)) return true;
    BinWindow win;
    win.skip   = static_cast<std::size_t>(client_cache.received_bytes(cls, id));
    win.budget = (max_bytes > total)
                     ? static_cast<std::size_t>(std::min<uint64_t>(max_bytes - total, SIZE_MAX))
                     : 0;
    scratch.clear();
    fn(&win);
    if (!scratch.empty()) {
      if (!sink(scratch.data(), scratch.size(), false)) {
        aborted = true;
        return false;
      }
      total += scratch.size();
      if (sent_out) {
        sent_out->push_back({cls, id, win.skip + win.payload_sent, win.complete});
      }
    }
    if (win.budget_blocked) {
      truncated = true;
      return false;
    }
    return true;
  };

  // §A.3.6.1: "servers should send metadata-bin 0 in advance of all other
  // bins."  Interactive clients use its arrival (even as an empty bin with
  // is_last=1) as the session-binding signal before they commit precinct
  // data into their cache, so emit it first — before main-header — even
  // for bare J2C codestreams that have no JP2/JPX box structure to ship.
  bool emitting = emit_bin(kMsgClassMetadata, 0,
                           [&](BinWindow *w) { emit_metadata_bin_zero(ctx, scratch, w); });
  if (emitting) {
    emitting = emit_bin(kMsgClassMainHeader, 0, [&](BinWindow *w) {
      emit_main_header_databin(st.codestream.data(), st.codestream.size(),
                               st.layout, ctx, scratch, w);
    });
  }
  // §A.3.3: deliver the tile-header data-bin for every tile whose index
  // appears in the view-window result — NOT for every tile in the image.
  // On large multi-tile codestreams (heic2501a has ~41k tiles) iterating
  // every tile also exploded the response with empty tile-headers for
  // tiles well outside the fovea, crowding out actual precinct payload.
  // Collect tile indices from `keys` in first-seen order so the resulting
  // tile-header messages are deterministic.
  if (emitting) {
    std::vector<uint16_t> window_tiles;
    {
      std::vector<bool> seen(st.idx->num_tiles(), false);
      for (const auto &k : keys) {
        if (!seen[k.t]) { seen[k.t] = true; window_tiles.push_back(k.t); }
      }
    }
    for (uint16_t t : window_tiles) {
      emitting = emit_bin(kMsgClassTileHeader, t, [&](BinWindow *w) {
        emit_tile_header_databin(st.codestream.data(), st.codestream.size(),
                                 t, st.layout, ctx, scratch, w);
      });
      if (!emitting) break;
    }
  }

  // Emit exactly the precincts resolve_view_window selected, in ascending
  // in-class-id (I) order.  resolve_view_window returns keys in
  // (t, c, r, p_rc) order, which for our I = t + (c + s·C)·T formula gives
  // IDs 0, 3, 6, … (step = C) before 1, 4, 7, … — i.e., component-major.
  // Interactive clients iterating their cache by ascending bin-id expect
  // id=1 and id=2 to arrive in the first response before id=3; if they see
  // gaps they may refuse to treat the view-window as partially delivered
  // and retry with the same cache model, never advertising the precincts
  // they DID receive (because from their perspective those have not been
  // "accepted" into the contiguous portion of the cache).  Sort by I
  // before emission so deliveries are gap-free.
  if (emitting) {
    struct KeyWithId { PrecinctKey k; uint64_t I; };
    std::vector<KeyWithId> ordered;
    ordered.reserve(keys.size());
    for (const auto &k : keys) {
      ordered.push_back({k, st.idx->I(k.t, k.c, k.r, k.p_rc)});
    }
    std::sort(ordered.begin(), ordered.end(),
              [](const KeyWithId &a, const KeyWithId &b) { return a.I < b.I; });
    for (const auto &e : ordered) {
      emitting = emit_bin(kMsgClassPrecinct, e.I, [&](BinWindow *w) {
        emit_precinct_databin(st.codestream.data(), st.codestream.size(),
                              e.k.t, e.k.c, e.k.r, e.k.p_rc,
                              *st.idx, *st.locator, ctx, scratch, w);
      });
      if (!emitting) break;
    }
  }

  if (!aborted) {
    scratch.clear();
    emit_eor(truncated ? EorReason::ByteLimit : EorReason::WindowDone, ctx, scratch);
    if (!sink(scratch.data(), scratch.size(), /*is_eor=*/true)) aborted = true;
  }

  if (total_bytes_out) *total_bytes_out = total;
  return !aborted;
}

// Legacy buffered path: collect every JPP message into a single vector
// (used by tests, benchmarks, and the `Content-Length:` transport).
std::vector<uint8_t> build_jpp_stream(const ServerState &st, const ViewWindow &vw,
                                      const CacheModel &client_cache = {},
                                      size_t *n_keys_out = nullptr,
                                      uint64_t max_bytes = UINT64_MAX,
                                      std::vector<SentBin> *sent_out = nullptr) {
  std::vector<uint8_t> stream;
  stream.reserve(64 * 1024);
  stream_jpp_response(
      st, vw, client_cache, max_bytes,
      [&](const uint8_t *data, std::size_t len, bool /*is_eor*/) {
        stream.insert(stream.end(), data, data + len);
        return true;
      },
      n_keys_out, /*total_bytes_out=*/nullptr, sent_out);
  return stream;
}

// Find the "<Name>: <value>" header line for `name` (case-insensitive) in a
// buffer holding the complete HTTP request headers up to the \r\n\r\n
// terminator.  Returns the value with surrounding whitespace trimmed, or an
// empty string if not found.  Used by the POST handler to read
// Content-Length.
std::string find_header(const uint8_t *buf, std::size_t len, const char *name) {
  const std::size_t nlen = std::strlen(name);
  const auto ieq = [](char a, char b) { return std::tolower(static_cast<unsigned char>(a))
                                            == std::tolower(static_cast<unsigned char>(b)); };
  // Skip the first line (request line).  Walk each subsequent CRLF-delimited
  // header and try to match `name`.
  std::size_t i = 0;
  while (i < len && buf[i] != '\n') ++i;
  if (i < len) ++i;  // past the \n of the first line
  while (i + nlen + 1 < len) {
    std::size_t end = i;
    while (end < len && buf[end] != '\r' && buf[end] != '\n') ++end;
    if (end - i >= nlen + 1 && buf[i + nlen] == ':') {
      bool match = true;
      for (std::size_t j = 0; j < nlen; ++j) {
        if (!ieq(static_cast<char>(buf[i + j]), name[j])) { match = false; break; }
      }
      if (match) {
        std::size_t v0 = i + nlen + 1;
        while (v0 < end && (buf[v0] == ' ' || buf[v0] == '\t')) ++v0;
        std::size_t v1 = end;
        while (v1 > v0 && (buf[v1 - 1] == ' ' || buf[v1 - 1] == '\t')) --v1;
        return std::string(reinterpret_cast<const char *>(buf + v0), v1 - v0);
      }
    }
    while (end < len && (buf[end] == '\r' || buf[end] == '\n')) ++end;
    if (end == i) break;
    i = end;
  }
  return {};
}

void handle_connection(TcpStream &conn, ServerState &st) {
  std::vector<uint8_t> raw;
  const std::size_t hdr_bytes = conn.recv_until_header_end(raw, 65536);
  if (hdr_bytes == 0) return;

  std::string request_line;
  std::size_t body_offset = 0;  // index in `raw` of first byte past "\r\n\r\n"
  {
    const char *s = reinterpret_cast<const char *>(raw.data());
    const char *eol = static_cast<const char *>(std::memchr(s, '\r', hdr_bytes));
    if (!eol) eol = static_cast<const char *>(std::memchr(s, '\n', hdr_bytes));
    if (!eol) { conn.send_all(format_error_response(400, "Bad Request")); return; }
    request_line.assign(s, eol);
    // Locate the \r\n\r\n boundary so we know where a POST body begins.
    for (std::size_t i = 0; i + 4 <= hdr_bytes; ++i) {
      if (std::memcmp(s + i, "\r\n\r\n", 4) == 0) { body_offset = i + 4; break; }
    }
  }

  // Handle CORS preflight (OPTIONS) — browsers send this before cross-origin
  // fetch() requests.
  if (request_line.substr(0, 8) == "OPTIONS ") {
    const char *cors =
        "HTTP/1.1 204 No Content\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Access-Control-Allow-Headers: *\r\n"
        "Access-Control-Max-Age: 86400\r\n"
        "Connection: close\r\n"
        "\r\n";
    conn.send_all(reinterpret_cast<const uint8_t *>(cors), std::strlen(cors));
    return;
  }

  std::string path, query;
  if (request_line.compare(0, 5, "POST ") == 0) {
    // §C.1: clients MAY POST with the query string as the body when the
    // request doesn't fit as a URL.  Parse path from the request line,
    // then read Content-Length bytes of body and use them as the query.
    const std::size_t sp1 = request_line.find(' ');
    const std::size_t sp2 = (sp1 == std::string::npos) ? std::string::npos
                                                       : request_line.find(' ', sp1 + 1);
    if (sp1 == std::string::npos || sp2 == std::string::npos) {
      conn.send_all(format_error_response(400, "Bad Request"));
      return;
    }
    path = request_line.substr(sp1 + 1, sp2 - sp1 - 1);

    const std::string clen_s = find_header(raw.data(), hdr_bytes, "Content-Length");
    if (clen_s.empty()) {
      conn.send_all(format_error_response(411, "Length Required"));
      return;
    }
    char *clen_end = nullptr;
    const unsigned long long clen = std::strtoull(clen_s.c_str(), &clen_end, 10);
    if (clen_end == clen_s.c_str() || *clen_end != '\0') {
      conn.send_all(format_error_response(400, "Bad Request"));
      return;
    }
    constexpr std::size_t kMaxPostBody = 16u * 1024u * 1024u;  // 16 MB cap
    if (clen > kMaxPostBody) {
      conn.send_all(format_error_response(413, "Payload Too Large"));
      return;
    }

    std::vector<uint8_t> body;
    body.reserve(static_cast<std::size_t>(clen));
    // Absorb any body bytes already captured alongside the headers.
    if (body_offset < hdr_bytes) {
      body.insert(body.end(), raw.begin() + static_cast<std::ptrdiff_t>(body_offset),
                  raw.begin() + static_cast<std::ptrdiff_t>(hdr_bytes));
    }
    if (body.size() < clen) {
      const std::size_t remaining = static_cast<std::size_t>(clen) - body.size();
      const std::size_t prev = body.size();
      body.resize(static_cast<std::size_t>(clen));
      if (!conn.recv_all(body.data() + prev, remaining)) {
        conn.send_all(format_error_response(400, "Bad Request"));
        return;
      }
    } else if (body.size() > clen) {
      body.resize(static_cast<std::size_t>(clen));
    }
    query.assign(reinterpret_cast<const char *>(body.data()), body.size());
  } else if (!split_http_get_line(request_line, &path, &query)) {
    conn.send_all(format_error_response(405, "Method Not Allowed"));
    return;
  }

  JpipRequest req;
  const auto ps = parse_jpip_query(query, &req);
  if (ps == RequestParseStatus::UnsupportedType) {
    conn.send_all(format_error_response(501, "Unsupported Type"));
    return;
  }
  if (ps == RequestParseStatus::MalformedField) {
    conn.send_all(format_error_response(400, "Malformed Field"));
    return;
  }

  // Default: full image if fsiz is omitted.
  if (!req.has_fsiz) {
    req.view_window.fx = st.idx->geometry().canvas_size.x;
    req.view_window.fy = st.idx->geometry().canvas_size.y;
  }
  if (!req.has_rsiz) {
    req.view_window.sx = req.view_window.fx;
    req.view_window.sy = req.view_window.fy;
  }

  using Clock = std::chrono::steady_clock;
  const auto t0 = Clock::now();

  // §D.2.4: when the request carries a qid, the response shall echo it.
  std::string extra_headers;
  if (req.has_qid) {
    extra_headers += "JPIP-qid: " + std::to_string(req.qid) + "\r\n";
  }

  // ── Session resolution (§B.2, §C.3) ──────────────────────────────────
  // Three modes:
  //   cid present  → request within an existing channel: serve against the
  //                  channel's server-side cache model (the client does NOT
  //                  send `model=` in sessions, §B.3); unknown cid → 404 so
  //                  the client re-establishes instead of silently looping
  //                  on stateless duplicates.
  //   cnew present → channel-establishment request: grant only when a
  //                  transport we actually support is in the client's list
  //                  (Table D.1: the granted transport shall be one the
  //                  client offered).  Otherwise §C.3.3: serve statelessly
  //                  with NO JPIP-cnew header.
  //   neither      → stateless: the `model=` field carries the client state.
  CacheModel  client_cache;
  std::string session_cid;
  std::string cnew_header;
  if (req.has_cid) {
    if (!st.channels.snapshot(req.cid, &client_cache)) {
      conn.send_all(format_error_response(404, "Channel Not Found"));
      return;
    }
    session_cid = req.cid;
    // Sessions still accept client-initiated model updates (e.g. the
    // client discarded cached bins): fold them into the channel's model
    // and into this response's view of it.
    if (!req.model.empty()) {
      st.channels.apply_model(session_cid, req.model);
      client_cache.apply(req.model);
    }
  } else {
    if (!req.model.empty()) client_cache = CacheModel::parse(req.model);
    if (req.has_cnew) {
      const std::string transport = ChannelManager::negotiate_transport(req.cnew);
      if (!transport.empty()) {
        session_cid = st.channels.open();
        if (!req.model.empty()) st.channels.apply_model(session_cid, req.model);
        cnew_header = "cid=" + session_cid + ",path=jpip,transport=" + transport;
      }
    }
  }
  // Session responses must not be cached by intermediaries (Annex F).
  if (!session_cid.empty()) extra_headers += "Cache-Control: no-cache\r\n";

  const uint64_t max_bytes = req.has_len ? req.len : UINT64_MAX;
  std::vector<SentBin> sent_bins;
  auto *sent_out = session_cid.empty() ? nullptr : &sent_bins;
  size_t n_keys = 0;
  std::size_t body_bytes = 0;
  if (st.chunked_responses) {
    // Chunked path: emit the headers first so the client can start
    // adjusting its UI (progress bar, fetch() stream), then flush each
    // JPP message as its own HTTP chunk.  `sink` returning false on
    // send_all failure aborts the emission early so we don't waste
    // CPU building messages that can't be delivered.
    auto hdrs = format_jpp_response_headers_chunked(st.target_id, cnew_header, extra_headers);
    if (!conn.send_all(hdrs)) return;
    bool send_ok = true;
    stream_jpp_response(
        st, req.view_window, client_cache, max_bytes,
        [&](const uint8_t *data, std::size_t len, bool /*is_eor*/) {
          if (len == 0) return true;  // empty chunk would terminate the stream
          auto ch = format_chunk_header(len);
          if (!conn.send_all(ch)) { send_ok = false; return false; }
          if (!conn.send_all(data, len)) { send_ok = false; return false; }
          static const uint8_t kCrLf[2] = {'\r', '\n'};
          if (!conn.send_all(kCrLf, 2)) { send_ok = false; return false; }
          body_bytes += len;
          return true;
        },
        &n_keys, /*total_bytes_out=*/nullptr, sent_out);
    if (send_ok) {
      auto last = format_last_chunk();
      conn.send_all(last);
    }
  } else {
    auto jpp = build_jpp_stream(st, req.view_window, client_cache, &n_keys, max_bytes, sent_out);
    body_bytes = jpp.size();
    auto resp = format_jpp_response(jpp.data(), jpp.size(), st.target_id, cnew_header, extra_headers);
    conn.send_all(resp);
  }

  // Bins handed to the transport are now part of the channel's history —
  // the next request on this cid must not receive them again.  Aborted
  // sends never reach `sent_bins`, so a dropped connection re-sends.
  if (!session_cid.empty()) st.channels.commit(session_cid, sent_bins);

  // §C.3.4 cclose: close channels only after the response completes.
  if (req.has_cclose) {
    if (req.cclose == "*") {
      if (!session_cid.empty()) st.channels.close(session_cid);
    } else {
      std::size_t p = 0;
      while (p <= req.cclose.size()) {
        std::size_t c = req.cclose.find(',', p);
        if (c == std::string::npos) c = req.cclose.size();
        const std::string one = req.cclose.substr(p, c - p);
        if (!one.empty()) st.channels.close(one);
        p = c + 1;
      }
    }
  }

  const auto t1 = Clock::now();
  const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  std::printf("  → %zu precincts (%.1f%%), %zu bytes JPP-stream, %.1f ms%s%s%s\n",
              n_keys,
              st.idx->total_precincts()
                  ? (100.0 * static_cast<double>(n_keys) / static_cast<double>(st.idx->total_precincts()))
                  : 0.0,
              body_bytes, ms,
              st.chunked_responses ? " [chunked]" : "",
              session_cid.empty() ? "" : " cid=",
              session_cid.c_str());
  std::fflush(stdout);
}

#ifdef OPENHTJ2K_ENABLE_QUIC
// H3 request handler — parses the JPIP query from the HTTP/3 :path pseudo-header
// and builds the same JPP-stream as the HTTP/1.1 path.
std::vector<uint8_t> handle_h3_request(const ServerState &st,
                                       const open_htj2k::jpip::H3Request &req) {
  JpipRequest jpip_req;
  const auto ps = parse_jpip_query(req.query, &jpip_req);
  if (ps != RequestParseStatus::Ok) return {};

  if (!jpip_req.has_fsiz) {
    jpip_req.view_window.fx = st.idx->geometry().canvas_size.x;
    jpip_req.view_window.fy = st.idx->geometry().canvas_size.y;
  }
  if (!jpip_req.has_rsiz) {
    jpip_req.view_window.sx = jpip_req.view_window.fx;
    jpip_req.view_window.sy = jpip_req.view_window.fy;
  }

  using Clock = std::chrono::steady_clock;
  const auto t0 = Clock::now();
  CacheModel h3_cache;
  if (!jpip_req.model.empty()) h3_cache = CacheModel::parse(jpip_req.model);
  size_t n_keys = 0;
  auto jpp = build_jpp_stream(st, jpip_req.view_window, h3_cache, &n_keys,
                              jpip_req.has_len ? jpip_req.len : UINT64_MAX);
  const auto t1 = Clock::now();
  const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  std::printf("  [H3] → %zu precincts (%.1f%%), %zu bytes, %.1f ms\n",
              n_keys,
              st.idx->total_precincts()
                  ? (100.0 * static_cast<double>(n_keys) / static_cast<double>(st.idx->total_precincts()))
                  : 0.0,
              jpp.size(), ms);
  std::fflush(stdout);
  return jpp;
}
#endif

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr,
        "Usage: open_htj2k_jpip_server <input.j2c> [--port N=8080]\n"
        "       [--h3 --cert server.cert --key server.key]\n"
        "       [--chunked]      # opt in to Transfer-Encoding: chunked\n"
        "       [--no-chunked]   # explicitly force Content-Length (default)\n");
    return EXIT_FAILURE;
  }
  std::string infile = argv[1];
  uint16_t port = 8080;
  bool use_h3 = false;
  // Default to Content-Length; chunked is opt-in because some reference
  // JPIP clients fail to parse chunked responses and report "connection
  // closed unexpectedly".
  bool chunked = false;
  std::string cert_file, key_file;
  for (int i = 2; i < argc; ++i) {
    if (std::strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
      port = static_cast<uint16_t>(std::atoi(argv[++i]));
    } else if (std::strcmp(argv[i], "--h3") == 0) {
      use_h3 = true;
    } else if (std::strcmp(argv[i], "--cert") == 0 && i + 1 < argc) {
      cert_file = argv[++i];
    } else if (std::strcmp(argv[i], "--key") == 0 && i + 1 < argc) {
      key_file = argv[++i];
    } else if (std::strcmp(argv[i], "--chunked") == 0) {
      chunked = true;
    } else if (std::strcmp(argv[i], "--no-chunked") == 0) {
      chunked = false;  // accepted for backward compatibility; now the default
    }
  }

  ServerState st;
  st.chunked_responses = chunked;
  st.codestream = read_file(infile.c_str());
  if (st.codestream.empty()) return EXIT_FAILURE;

  st.idx = CodestreamIndex::build(st.codestream.data(), st.codestream.size());
  if (!st.idx) { std::fprintf(stderr, "CodestreamIndex build failed\n"); return EXIT_FAILURE; }

  walk_codestream(st.codestream.data(), st.codestream.size(), &st.layout);

  st.locator = PacketLocator::build(st.codestream.data(), st.codestream.size(), *st.idx, st.layout);
  if (!st.locator) { std::fprintf(stderr, "PacketLocator build failed\n"); return EXIT_FAILURE; }

  // JPIP §D.4: the target identifier is an opaque string.  Emitting the
  // full server-side filesystem path (as we did before) is legal but
  // works poorly with clients that treat the TID as a token — some echo
  // it back verbatim in `&tid=` on every follow-up request, producing
  // unusually long, slash-laden query strings.  Strip down to the
  // basename so the TID is a short, path-free identifier.
  {
    const auto last_slash = infile.find_last_of("/\\");
    st.target_id = (last_slash == std::string::npos) ? infile
                                                     : infile.substr(last_slash + 1);
  }

  std::printf("JPIP server: %s (%u×%u, %llu precincts)\n",
              infile.c_str(),
              st.idx->geometry().canvas_size.x, st.idx->geometry().canvas_size.y,
              static_cast<unsigned long long>(st.idx->total_precincts()));

#ifdef OPENHTJ2K_ENABLE_QUIC
  if (use_h3) {
    if (cert_file.empty() || key_file.empty()) {
      std::fprintf(stderr, "ERROR: --h3 requires --cert and --key\n");
      return EXIT_FAILURE;
    }
    open_htj2k::jpip::TlsCertConfig tls{cert_file, key_file};
    open_htj2k::jpip::H3Server h3;
    if (!h3.start(port, tls, [&st](const open_htj2k::jpip::H3Request &req) {
          return handle_h3_request(st, req);
        })) {
      std::fprintf(stderr, "H3 server start failed: %s\n", h3.last_error().c_str());
      return EXIT_FAILURE;
    }
    std::printf("listening on https://localhost:%u/jpip (HTTP/3 over QUIC)\n", port);
    std::fflush(stdout);
    // Block until interrupted — MsQuic handles connections on its own threads.
    std::printf("Press Enter to stop.\n");
    std::fflush(stdout);
    std::getchar();
    h3.stop();
    return EXIT_SUCCESS;
  }
#else
  (void)use_h3; (void)cert_file; (void)key_file;
#endif

  tcp_wsa_init();

  TcpListener listener;
  if (!listener.bind(port)) {
    std::fprintf(stderr, "bind port %u: %s\n", port, listener.last_error().c_str());
    tcp_wsa_cleanup();
    return EXIT_FAILURE;
  }
  if (!listener.listen()) {
    std::fprintf(stderr, "listen: %s\n", listener.last_error().c_str());
    tcp_wsa_cleanup();
    return EXIT_FAILURE;
  }
  std::printf("listening on http://localhost:%u/jpip%s\n", port,
              st.chunked_responses ? " (Transfer-Encoding: chunked)" : " (Content-Length)");
  std::fflush(stdout);

  while (true) {
    TcpStream conn = listener.accept();
    if (!conn.is_open()) {
      std::fprintf(stderr, "accept: %s\n", listener.last_error().c_str());
      continue;
    }
    handle_connection(conn, st);
  }

  tcp_wsa_cleanup();
  return EXIT_SUCCESS;
}
