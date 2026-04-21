// One-off probe for issue #297 follow-up: does parse_jpp_stream tolerate
// being fed a JPP-stream response in two halves, the way a browser using
// ReadableStream would feed it?
//
// Loads a real codestream, builds its JPP-stream via the same code path
// the server uses for a full-image response, and then:
//   1. Parses the whole buffer → baseline success, captures DataBinSet.
//   2. For every split offset in [1, body_len-1], calls parse_jpp_stream
//      twice (first half, second half) into a fresh DataBinSet each time
//      and tests whether the two parses (a) both succeed and (b) produce
//      the same DataBinSet as the baseline.
//
// A 20-line JS refactor is only safe if the answer is "every offset
// works"; anything else means the library needs a resumable parser.
//
// Build manually (not a ctest — one-off diagnostic):
//   c++ -std=c++17 -I source/core/jpip -I source/core/interface \
//       -I source/core/common tools/jpip_split_parse_probe.cpp \
//       build/bin/libopenhtj2k.dylib -o /tmp/jpip_split_probe

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "codestream_assembler.hpp"
#include "codestream_walker.hpp"
#include "data_bin_emitter.hpp"
#include "jpp_message.hpp"
#include "jpp_parser.hpp"
#include "packet_locator.hpp"
#include "precinct_index.hpp"
#include "view_window.hpp"

using namespace open_htj2k::jpip;

static std::vector<uint8_t> read_file(const char *path) {
  FILE *f = std::fopen(path, "rb");
  if (!f) return {};
  std::fseek(f, 0, SEEK_END);
  auto sz = static_cast<std::size_t>(std::ftell(f));
  std::fseek(f, 0, SEEK_SET);
  std::vector<uint8_t> buf(sz);
  std::fread(buf.data(), 1, sz, f);
  std::fclose(f);
  return buf;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <path-to.j2c>\n", argv[0]);
    return 1;
  }
  auto cs = read_file(argv[1]);
  if (cs.empty()) { std::fprintf(stderr, "cannot read %s\n", argv[1]); return 1; }

  auto idx = CodestreamIndex::build(cs.data(), cs.size());
  CodestreamLayout layout;
  walk_codestream(cs.data(), cs.size(), &layout);
  auto locator = PacketLocator::build(cs.data(), cs.size(), *idx, layout);
  if (!idx || !locator) { std::fprintf(stderr, "index/locator build failed\n"); return 1; }

  // Emit a full-image JPP-stream the same way the server does.
  ViewWindow vw;
  vw.fx = idx->geometry().canvas_size.x;
  vw.fy = idx->geometry().canvas_size.y;
  vw.sx = vw.fx; vw.sy = vw.fy;
  auto keys = resolve_view_window(*idx, vw);

  std::vector<uint8_t> body;
  MessageHeaderContext ctx;
  emit_metadata_bin_zero(ctx, body);
  emit_main_header_databin(cs.data(), cs.size(), layout, ctx, body);
  // One tile-header per tile touched.
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

  std::printf("JPP-stream: %zu bytes\n", body.size());

  DataBinSet baseline;
  if (!parse_jpp_stream(body.data(), body.size(), &baseline)) {
    std::fprintf(stderr, "baseline parse failed\n"); return 1;
  }
  std::printf("baseline OK: %zu bins, eor=%d reason=%u\n",
              baseline.size(), baseline.has_eor() ? 1 : 0, baseline.eor_reason());

  // Probe every split offset.
  std::size_t both_ok = 0, first_fail = 0, second_fail = 0, divergent = 0;
  std::size_t first_example = SIZE_MAX, last_example = SIZE_MAX;
  for (std::size_t split = 1; split + 1 < body.size(); ++split) {
    DataBinSet part1, part2;
    const bool ok1 = parse_jpp_stream(body.data(), split, &part1);
    const bool ok2 = parse_jpp_stream(body.data() + split, body.size() - split, &part2);
    if (!ok1) { ++first_fail; continue; }
    if (!ok2) { ++second_fail; continue; }

    // Merge and compare to baseline: is every (class, id) a match?
    DataBinSet merged;
    merged.merge_from(part1);
    merged.merge_from(part2);
    if (part2.has_eor()) merged.set_eor(part2.eor_reason());

    bool matches = merged.size() == baseline.size()
                   && merged.has_eor() == baseline.has_eor();
    if (matches) {
      for (const auto &kv : baseline.keys()) {
        if (!merged.contains(kv.first, kv.second)) { matches = false; break; }
        if (merged.get(kv.first, kv.second) != baseline.get(kv.first, kv.second)) {
          matches = false; break;
        }
        if (merged.is_complete(kv.first, kv.second)
            != baseline.is_complete(kv.first, kv.second)) { matches = false; break; }
      }
    }
    if (matches) {
      ++both_ok;
      if (first_example == SIZE_MAX) first_example = split;
      last_example = split;
    } else {
      ++divergent;
    }
  }

  const std::size_t total = body.size() - 2;
  std::printf("split results over %zu offsets:\n", total);
  std::printf("  both parses succeed AND bins match baseline: %zu  (%.2f%%)\n",
              both_ok, 100.0 * static_cast<double>(both_ok) / static_cast<double>(total));
  std::printf("  first-half parse fails (truncation):         %zu\n", first_fail);
  std::printf("  second-half parse fails (mid-msg / dep form): %zu\n", second_fail);
  std::printf("  both succeed but bins diverge from baseline:  %zu\n", divergent);
  if (both_ok) {
    std::printf("  lucky offsets: first=%zu, last=%zu\n", first_example, last_example);
  }
  return 0;
}
