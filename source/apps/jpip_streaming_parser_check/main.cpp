// jpip_streaming_parser_check: stress test for StreamingJppParser.
//
// For the issue #297 browser + C++ streaming clients to work, the parser
// must tolerate chunk boundaries that fall anywhere inside a JPP-stream
// — mid-VBAS, mid-header, mid-payload.  This test builds a real JPP-stream
// from a conformance codestream, parses it once with the one-shot
// `parse_jpp_stream` to get a reference DataBinSet, then re-parses it via
// StreamingJppParser in two feed() calls for every possible split offset
// and confirms the reconstructed DataBinSet is identical to the reference.
//
// It also checks:
//   - Byte-at-a-time feeding (worst case: 1-byte chunks).
//   - finish() returns true after a clean stream ended.
//   - reset() lets the parser be reused across responses.
//
// Run-time budget: the pair-split pass is O(n²) in buffer size but the
// fold-into-set operations are cheap; for the largest conformance asset
// we test against (~90 KB) it comes in well under a second.
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "codestream_walker.hpp"
#include "data_bin_emitter.hpp"
#include "jpp_message.hpp"
#include "jpp_parser.hpp"
#include "packet_locator.hpp"
#include "precinct_index.hpp"
#include "view_window.hpp"

using namespace open_htj2k::jpip;

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

std::vector<uint8_t> read_file(const char *path) {
  FILE *f = std::fopen(path, "rb");
  if (!f) return {};
  std::fseek(f, 0, SEEK_END);
  const auto sz = static_cast<std::size_t>(std::ftell(f));
  std::fseek(f, 0, SEEK_SET);
  std::vector<uint8_t> buf(sz);
  const auto rd = std::fread(buf.data(), 1, sz, f);
  std::fclose(f);
  if (rd != sz) buf.clear();
  return buf;
}

// Build a full-image JPP-stream the same way the server does, but in-
// process so the test has nothing to do with the network.
std::vector<uint8_t> build_jpp_stream(const std::vector<uint8_t> &cs) {
  auto idx = CodestreamIndex::build(cs.data(), cs.size());
  CodestreamLayout layout;
  walk_codestream(cs.data(), cs.size(), &layout);
  auto locator = PacketLocator::build(cs.data(), cs.size(), *idx, layout);

  ViewWindow vw;
  vw.fx = idx->geometry().canvas_size.x;
  vw.fy = idx->geometry().canvas_size.y;
  vw.sx = vw.fx; vw.sy = vw.fy;
  auto keys = resolve_view_window(*idx, vw);

  std::vector<uint8_t> body;
  MessageHeaderContext ctx;
  emit_metadata_bin_zero(ctx, body);
  emit_main_header_databin(cs.data(), cs.size(), layout, ctx, body);
  std::vector<bool> tile_seen(idx->num_tiles(), false);
  for (const auto &k : keys) {
    if (!tile_seen[k.t]) {
      tile_seen[k.t] = true;
      emit_tile_header_databin(cs.data(), cs.size(), k.t, layout, ctx, body);
    }
  }
  for (const auto &k : keys) {
    emit_precinct_databin(cs.data(), cs.size(), k.t, k.c, k.r, k.p_rc,
                          *idx, *locator, ctx, body);
  }
  emit_eor(EorReason::WindowDone, ctx, body);
  return body;
}

// True iff two parsed sets have the same keys, bytes, is_last flags, and
// EOR state.  (DataBinSet has no operator==, so we walk keys() manually.)
bool bin_sets_equal(const DataBinSet &a, const DataBinSet &b) {
  if (a.size() != b.size()) return false;
  if (a.has_eor() != b.has_eor()) return false;
  if (a.has_eor() && a.eor_reason() != b.eor_reason()) return false;
  for (const auto &k : a.keys()) {
    if (!b.contains(k.first, k.second)) return false;
    if (a.get(k.first, k.second) != b.get(k.first, k.second)) return false;
    if (a.is_complete(k.first, k.second) != b.is_complete(k.first, k.second)) return false;
  }
  return true;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <path-to.j2c>\n", argv[0]);
    return 1;
  }
  const auto cs = read_file(argv[1]);
  if (cs.empty()) { std::fprintf(stderr, "cannot read %s\n", argv[1]); return 1; }
  const auto body = build_jpp_stream(cs);
  if (body.empty()) { std::fprintf(stderr, "empty JPP-stream\n"); return 1; }

  // Reference: one-shot parse.
  DataBinSet ref;
  CHECK(parse_jpp_stream(body.data(), body.size(), &ref), "one-shot parse");
  CHECK(ref.has_eor(), "reference must have EOR");

  // Single-chunk feed via the streaming parser should match the reference.
  {
    StreamingJppParser p;
    DataBinSet got;
    CHECK(p.feed(body.data(), body.size(), &got), "single-chunk feed");
    CHECK(p.finish(), "single-chunk finish");
    CHECK(bin_sets_equal(got, ref), "single-chunk bins match reference");
  }

  // Every pair split: two feed() calls, one at offset [1, body_size-1].
  std::size_t pair_fail = 0;
  for (std::size_t split = 1; split + 1 < body.size(); ++split) {
    StreamingJppParser p;
    DataBinSet got;
    const bool ok1 = p.feed(body.data(), split, &got);
    const bool ok2 = p.feed(body.data() + split, body.size() - split, &got);
    if (!ok1 || !ok2 || !p.finish() || !bin_sets_equal(got, ref)) {
      ++pair_fail;
      if (pair_fail <= 5) {
        std::fprintf(stderr, "FAIL split=%zu ok1=%d ok2=%d pending=%zu\n",
                     split, ok1, ok2, p.pending());
      }
    }
  }
  CHECK(pair_fail == 0, "pair-split failures: %zu / %zu offsets", pair_fail,
        body.size() - 2);

  // One-byte-at-a-time feed.  Worst case for the internal tail buffer.
  {
    StreamingJppParser p;
    DataBinSet got;
    bool ok = true;
    for (std::size_t i = 0; i < body.size() && ok; ++i) {
      ok = p.feed(body.data() + i, 1, &got);
    }
    CHECK(ok, "byte-at-a-time feed");
    CHECK(p.finish(), "byte-at-a-time finish");
    CHECK(bin_sets_equal(got, ref), "byte-at-a-time bins match reference");
  }

  // reset() returns the parser to a pristine state.
  {
    StreamingJppParser p;
    DataBinSet tmp;
    p.feed(body.data(), body.size() / 2, &tmp);  // intentionally partial
    CHECK(p.pending() != 0 || body.size() / 2 == 0, "partial feed should leave pending");
    p.reset();
    CHECK(p.pending() == 0, "pending cleared after reset");
    CHECK(p.finish(), "reset parser is at a clean boundary");

    DataBinSet got2;
    CHECK(p.feed(body.data(), body.size(), &got2), "feed after reset");
    CHECK(p.finish(), "finish after reset");
    CHECK(bin_sets_equal(got2, ref), "reused parser matches reference");
  }

  if (failures == 0) {
    std::printf("OK streaming_parser_check: %zu pair splits, byte-at-a-time, reset — all pass\n",
                body.size() - 2);
    return 0;
  }
  std::fprintf(stderr, "streaming_parser_check: %d failures\n", failures);
  return 1;
}
