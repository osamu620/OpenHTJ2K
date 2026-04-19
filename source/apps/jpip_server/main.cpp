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
// selected by the view-window resolver.  Stateless — no session, no
// cache model, each request is self-contained.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "codestream_walker.hpp"
#include "data_bin_emitter.hpp"
#include "jpp_message.hpp"
#include "jpip_request.hpp"
#include "jpip_response.hpp"
#include "packet_locator.hpp"
#include "precinct_index.hpp"
#include "cache_model.hpp"
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
};

// Build the JPP-stream for a given ViewWindow, skipping data-bins
// the client already has (per the cache model from §C.9).  When
// `n_keys_out` is non-null the caller gets the precinct count back
// without having to re-run resolve_view_window.
std::vector<uint8_t> build_jpp_stream(const ServerState &st, const ViewWindow &vw,
                                      const CacheModel &client_cache = {},
                                      size_t *n_keys_out = nullptr) {
  auto keys = resolve_view_window(*st.idx, vw);
  if (n_keys_out) *n_keys_out = keys.size();

  std::vector<uint8_t> stream;
  MessageHeaderContext ctx;
  if (!client_cache.has(kMsgClassMainHeader, 0))
    emit_main_header_databin(st.codestream.data(), st.codestream.size(), st.layout, ctx, stream);
  for (uint32_t t = 0; t < st.idx->num_tiles(); ++t) {
    if (!client_cache.has(kMsgClassTileHeader, t))
      emit_tile_header_databin(st.codestream.data(), st.codestream.size(),
                               static_cast<uint16_t>(t), st.layout, ctx, stream);
  }
  if (!client_cache.has(kMsgClassMetadata, 0))
    emit_metadata_bin_zero(ctx, stream);

  // Emit exactly the precincts resolve_view_window selected.  The old code
  // walked the full T×C×R×P grid and tested every precinct ID against a
  // hash set built from `keys`, which is O(total_precincts) per request.
  // On heic2501a (103K precincts, ~500-precinct typical fovea), the walk
  // dominated — especially under the foveation demo's 3× amplification.
  // JPP-stream messages are order-agnostic (each carries class + id; the
  // client reassembler sorts them), so emitting in `keys` order is fine.
  for (const auto &k : keys) {
    const uint64_t I = st.idx->I(k.t, k.c, k.r, k.p_rc);
    if (!client_cache.has(kMsgClassPrecinct, I)) {
      emit_precinct_databin(st.codestream.data(), st.codestream.size(),
                            k.t, k.c, k.r, k.p_rc,
                            *st.idx, *st.locator, ctx, stream);
    }
  }
  emit_eor(EorReason::WindowDone, ctx, stream);
  return stream;
}

void handle_connection(TcpStream &conn, const ServerState &st) {
  std::vector<uint8_t> raw;
  const std::size_t hdr_bytes = conn.recv_until_header_end(raw, 65536);
  if (hdr_bytes == 0) return;

  std::string request_line;
  {
    const char *s = reinterpret_cast<const char *>(raw.data());
    const char *eol = static_cast<const char *>(std::memchr(s, '\r', hdr_bytes));
    if (!eol) eol = static_cast<const char *>(std::memchr(s, '\n', hdr_bytes));
    if (!eol) { conn.send_all(format_error_response(400, "Bad Request")); return; }
    request_line.assign(s, eol);
  }

  // Handle CORS preflight (OPTIONS) — browsers send this before cross-origin
  // fetch() requests.
  if (request_line.substr(0, 8) == "OPTIONS ") {
    const char *cors =
        "HTTP/1.1 204 No Content\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: GET, OPTIONS\r\n"
        "Access-Control-Allow-Headers: *\r\n"
        "Access-Control-Max-Age: 86400\r\n"
        "Connection: close\r\n"
        "\r\n";
    conn.send_all(reinterpret_cast<const uint8_t *>(cors), std::strlen(cors));
    return;
  }

  std::string path, query;
  if (!split_http_get_line(request_line, &path, &query)) {
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

  CacheModel client_cache;
  if (!req.model.empty()) client_cache = CacheModel::parse(req.model);
  size_t n_keys = 0;
  auto jpp = build_jpp_stream(st, req.view_window, client_cache, &n_keys);

  const auto t1 = Clock::now();
  const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  std::printf("  → %zu precincts (%.1f%%), %zu bytes JPP-stream, %.1f ms\n",
              n_keys,
              st.idx->total_precincts()
                  ? (100.0 * static_cast<double>(n_keys) / static_cast<double>(st.idx->total_precincts()))
                  : 0.0,
              jpp.size(), ms);
  std::fflush(stdout);

  auto resp = format_jpp_response(jpp.data(), jpp.size(), st.target_id);
  conn.send_all(resp);
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
  auto jpp = build_jpp_stream(st, jpip_req.view_window, h3_cache, &n_keys);
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
        "       [--h3 --cert server.cert --key server.key]\n");
    return EXIT_FAILURE;
  }
  std::string infile = argv[1];
  uint16_t port = 8080;
  bool use_h3 = false;
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
    }
  }

  ServerState st;
  st.codestream = read_file(infile.c_str());
  if (st.codestream.empty()) return EXIT_FAILURE;

  st.idx = CodestreamIndex::build(st.codestream.data(), st.codestream.size());
  if (!st.idx) { std::fprintf(stderr, "CodestreamIndex build failed\n"); return EXIT_FAILURE; }

  walk_codestream(st.codestream.data(), st.codestream.size(), &st.layout);

  st.locator = PacketLocator::build(st.codestream.data(), st.codestream.size(), *st.idx, st.layout);
  if (!st.locator) { std::fprintf(stderr, "PacketLocator build failed\n"); return EXIT_FAILURE; }

  st.target_id = infile;

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
  std::printf("listening on http://localhost:%u/jpip\n", port);
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
