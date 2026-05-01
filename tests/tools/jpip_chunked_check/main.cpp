// jpip_chunked_check: end-to-end round-trip of the `Transfer-Encoding:
// chunked` JPIP response path (issue #297).
//
// Spawns a mini HTTP/1.1 responder on a localhost port that hand-writes a
// chunked response containing a known JPP-stream body (metadata-bin 0 +
// main-header bin), then drives `JpipClient::fetch` against it and
// verifies the client's chunked decoder recovered the body and the JPP
// parser populated DataBinSet with the expected bins.
//
// This exercises:
//   - jpip_response:: format_jpp_response_headers_chunked / format_chunk_header
//     / format_last_chunk
//   - JpipClient's parse_http_chunked + drain-to-EOF + decode_chunked_body
//     path
//   - That messages written as multiple wire chunks reassemble into the
//     same JPP-stream the server would have produced in a single buffer.
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

#include "cache_model.hpp"
#include "data_bin_emitter.hpp"
#include "jpip_client.hpp"
#include "jpip_response.hpp"
#include "jpp_message.hpp"
#include "jpp_parser.hpp"
#include "tcp_socket.hpp"

using open_htj2k::jpip::DataBinSet;
using open_htj2k::jpip::emit_metadata_bin_zero;
using open_htj2k::jpip::format_chunk_header;
using open_htj2k::jpip::format_jpp_response_headers_chunked;
using open_htj2k::jpip::format_last_chunk;
using open_htj2k::jpip::JpipClient;
using open_htj2k::jpip::kMsgClassMainHeader;
using open_htj2k::jpip::kMsgClassMetadata;
using open_htj2k::jpip::MessageHeaderContext;
using open_htj2k::jpip::TcpListener;
using open_htj2k::jpip::TcpStream;
using open_htj2k::jpip::tcp_wsa_cleanup;
using open_htj2k::jpip::tcp_wsa_init;
using open_htj2k::jpip::ViewWindow;

namespace {
int failures = 0;
#define CHECK(cond, ...)                                            \
  do {                                                              \
    if (!(cond)) {                                                  \
      std::fprintf(stderr, "FAIL [%s:%d] %s — ", __FILE__, __LINE__, #cond); \
      std::fprintf(stderr, __VA_ARGS__);                            \
      std::fprintf(stderr, "\n");                                   \
      ++failures;                                                   \
    }                                                               \
  } while (0)

// Minimal JPP-stream for the test: metadata-bin 0 (empty) + main-header
// bin carrying a SOC + a fake SIZ-like payload.  The JpipClient's parser
// should happily accept any main-header payload because it only records
// byte ranges against in-class id 0.
std::vector<uint8_t> build_fake_main_header() {
  // SOC (FF 4F) + a fabricated 9-byte "marker" so there's something
  // non-trivial for the emitter to ship.  Real main-header emission
  // expects the codestream to contain a SOT, but here we're exercising
  // the HTTP wire path only — we write the bin body directly rather
  // than calling emit_main_header_databin().
  std::vector<uint8_t> mh = {0xFF, 0x4F, 'O', 'P', 'E', 'N', 'H', 'T', 'J'};
  return mh;
}

std::vector<uint8_t> build_jpp_body() {
  MessageHeaderContext ctx;
  std::vector<uint8_t> body;
  emit_metadata_bin_zero(ctx, body);
  // Hand-craft a main-header message.  We reuse the emitter-independent
  // wire encoding by building the JPP message manually: bin-class=6,
  // is_last=1, offset=0, body = fake_main_header bytes.
  const auto mh = build_fake_main_header();
  // Build a minimal JPP message header via encode_header_independent.
  open_htj2k::jpip::MessageHeader hdr{};
  hdr.class_id    = kMsgClassMainHeader;
  hdr.cs_n        = 0;
  hdr.in_class_id = 0;
  hdr.msg_offset  = 0;
  hdr.msg_length  = mh.size();
  hdr.is_last     = true;
  uint8_t hdr_buf[open_htj2k::jpip::kMessageHeaderMaxBytes];
  std::size_t hn = open_htj2k::jpip::encode_header(hdr, ctx, hdr_buf);
  body.insert(body.end(), hdr_buf, hdr_buf + hn);
  body.insert(body.end(), mh.begin(), mh.end());
  return body;
}

// Hand-written chunked server: for each entry in `chunks` emit one HTTP
// chunk; then terminate.  Connection is always `close`.
void serve_chunked(TcpStream &conn,
                   const std::vector<std::vector<uint8_t>> &chunks) {
  // Drain the request headers.
  std::vector<uint8_t> raw;
  conn.recv_until_header_end(raw, 65536);

  auto hdrs = format_jpp_response_headers_chunked("test-tid");
  conn.send_all(hdrs);

  for (const auto &c : chunks) {
    if (c.empty()) continue;
    auto ch = format_chunk_header(c.size());
    conn.send_all(ch);
    conn.send_all(c.data(), c.size());
    const uint8_t crlf[2] = {'\r', '\n'};
    conn.send_all(crlf, 2);
  }
  auto last = format_last_chunk();
  conn.send_all(last);
}

}  // namespace

int main() {
  tcp_wsa_init();

  constexpr uint16_t kTestPort = 19285;
  const auto jpp = build_jpp_body();

  // Pre-slice the JPP body into multiple chunks (10+10+remainder) so the
  // client's decoder actually has to merge several chunks — if we shipped
  // the whole body as a single chunk the test would not distinguish a
  // chunked from a Content-Length path.
  std::vector<std::vector<uint8_t>> wire_chunks;
  std::size_t cursor = 0;
  for (std::size_t target : {std::size_t{10}, std::size_t{10}}) {
    if (cursor + target > jpp.size()) break;
    wire_chunks.emplace_back(jpp.begin() + static_cast<std::ptrdiff_t>(cursor),
                             jpp.begin() + static_cast<std::ptrdiff_t>(cursor + target));
    cursor += target;
  }
  if (cursor < jpp.size()) {
    wire_chunks.emplace_back(jpp.begin() + static_cast<std::ptrdiff_t>(cursor), jpp.end());
  }
  CHECK(wire_chunks.size() >= 2, "test needs multiple chunks to be meaningful");

  std::atomic<bool> server_bound{false};
  std::thread server_thread([&]() {
    TcpListener listener;
    if (!listener.bind(kTestPort, "127.0.0.1")) {
      std::fprintf(stderr, "bind: %s\n", listener.last_error().c_str());
      server_bound = true;
      return;
    }
    if (!listener.listen()) {
      std::fprintf(stderr, "listen: %s\n", listener.last_error().c_str());
      server_bound = true;
      return;
    }
    server_bound = true;
    TcpStream conn = listener.accept();
    if (!conn.is_open()) return;
    serve_chunked(conn, wire_chunks);
  });

  while (!server_bound.load()) std::this_thread::sleep_for(std::chrono::milliseconds(5));
  std::this_thread::sleep_for(std::chrono::milliseconds(25));

  JpipClient client;
  ViewWindow vw;
  vw.fx = 64; vw.fy = 64;
  vw.sx = 64; vw.sy = 64;
  DataBinSet bins;
  // Track how many times the progress callback fires and the bin-count
  // sequence it reports.  The multi-chunk server above slices the JPP
  // body into 3 wire chunks, and the fake JPP body contains 2 bins
  // (metadata-bin 0 + main-header); the callback should observe the bin
  // count grow monotonically from 0 to 2 across the streamed chunks.
  std::size_t progress_fires = 0;
  std::size_t max_bins_seen = 0;
  auto on_progress = [&](const DataBinSet &s) {
    ++progress_fires;
    if (s.size() > max_bins_seen) max_bins_seen = s.size();
  };
  const bool ok = client.fetch_streaming("127.0.0.1", kTestPort, vw, &bins,
                                         /*model=*/nullptr, on_progress);
  CHECK(ok, "JpipClient::fetch_streaming failed: %s", client.last_error().c_str());
  CHECK(progress_fires >= 1, "progress callback should fire at least once (fires=%zu)",
        progress_fires);
  CHECK(max_bins_seen == 2, "progress should see both bins (max=%zu)", max_bins_seen);

  // metadata-bin 0 must be present (empty) and is_last.
  CHECK(bins.contains(kMsgClassMetadata, 0), "metadata-bin 0 missing");
  CHECK(bins.is_complete(kMsgClassMetadata, 0),
        "metadata-bin 0 should be complete (is_last)");
  CHECK(bins.get(kMsgClassMetadata, 0).empty(),
        "metadata-bin 0 must be empty for bare J2C");

  // main-header bin should carry the fake payload.
  CHECK(bins.contains(kMsgClassMainHeader, 0), "main-header bin missing");
  const auto &mh_bytes = bins.get(kMsgClassMainHeader, 0);
  const auto expect    = build_fake_main_header();
  CHECK(mh_bytes == expect,
        "main-header bin bytes mismatch (got %zu, expected %zu)",
        mh_bytes.size(), expect.size());

  server_thread.join();
  tcp_wsa_cleanup();

  if (failures == 0) {
    std::printf("OK chunked_check: Transfer-Encoding: chunked round-trip\n");
    return 0;
  }
  std::fprintf(stderr, "chunked_check: %d failures\n", failures);
  return 1;
}
