// jpip_parser_check: ctest harness for the JPP-stream parser (B4).
//
// Round-trips through the emitter and the parser:
//   1. Walk the codestream and emit the three header data-bins (main,
//      tile-0 header, metadata-bin 0) into a single JPP-stream buffer.
//   2. parse_jpp_stream() the buffer into a DataBinSet.
//   3. Verify the set has exactly three bins, each with the expected
//      is_complete flag, and each bin's bytes are byte-identical to the
//      corresponding slice of the source codestream.
// Additionally exercises the in-order / is_last rejection paths.
//
// Usage:  jpip_parser_check <input.j2c>
// Exits 0 on every assertion passing, 1 on the first failure.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "codestream_walker.hpp"
#include "data_bin_emitter.hpp"
#include "jpp_message.hpp"
#include "jpp_parser.hpp"

using open_htj2k::jpip::CodestreamLayout;
using open_htj2k::jpip::DataBinSet;
using open_htj2k::jpip::emit_main_header_databin;
using open_htj2k::jpip::emit_metadata_bin_zero;
using open_htj2k::jpip::emit_tile_header_databin;
using open_htj2k::jpip::kMsgClassMainHeader;
using open_htj2k::jpip::kMsgClassMetadata;
using open_htj2k::jpip::kMsgClassTileHeader;
using open_htj2k::jpip::MessageHeaderContext;
using open_htj2k::jpip::parse_jpp_stream;
using open_htj2k::jpip::walk_codestream;

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
  if (!f) { std::fprintf(stderr, "ERROR: cannot open %s\n", path); return {}; }
  std::fseek(f, 0, SEEK_END);
  auto sz = static_cast<std::size_t>(std::ftell(f));
  std::fseek(f, 0, SEEK_SET);
  std::vector<uint8_t> buf(sz);
  std::size_t rd = std::fread(buf.data(), 1, sz, f);
  std::fclose(f);
  if (rd != sz) { std::fprintf(stderr, "ERROR: partial read\n"); buf.clear(); }
  return buf;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "Usage: jpip_parser_check <input.j2c>\n");
    return 1;
  }

  auto bytes = read_file(argv[1]);
  if (bytes.empty()) return 1;

  CodestreamLayout layout;
  CHECK(walk_codestream(bytes.data(), bytes.size(), &layout), "walk_codestream");
  if (failures) return 1;

  // ── Emit the three header data-bins into one stream buffer ───────────
  std::vector<uint8_t> stream;
  MessageHeaderContext enc_ctx;
  emit_main_header_databin(bytes.data(), bytes.size(), layout, enc_ctx, stream);
  emit_tile_header_databin(bytes.data(), bytes.size(), /*tile=*/0, layout, enc_ctx, stream);
  emit_metadata_bin_zero(enc_ctx, stream);

  // ── Parse it back ────────────────────────────────────────────────────
  DataBinSet set;
  CHECK(parse_jpp_stream(stream.data(), stream.size(), &set), "parse_jpp_stream");
  CHECK(set.size() == 3, "set.size()=%zu, expected 3", set.size());

  // Main header (class 6).
  {
    CHECK(set.contains(kMsgClassMainHeader, 0), "main bin present");
    CHECK(set.is_complete(kMsgClassMainHeader, 0), "main bin complete");
    const auto &got = set.get(kMsgClassMainHeader, 0);
    const std::size_t expected_len = layout.main_header_end - layout.soc_offset;
    CHECK(got.size() == expected_len, "main len=%zu expected %zu", got.size(), expected_len);
    CHECK(std::memcmp(got.data(), bytes.data() + layout.soc_offset, expected_len) == 0,
          "main header bytes not byte-identical");
  }

  // Tile-0 header (class 2).
  {
    CHECK(set.contains(kMsgClassTileHeader, 0), "tile-0 bin present");
    CHECK(set.is_complete(kMsgClassTileHeader, 0), "tile-0 bin complete");
    const auto &got = set.get(kMsgClassTileHeader, 0);
    // Reconstruct expected bytes — concatenation of each tile-0 tile-part's
    // marker bytes, SOT excluded.
    std::vector<uint8_t> expected;
    for (const auto &tp : layout.tile_parts) {
      if (tp.tile_index != 0) continue;
      constexpr std::size_t kSotSize = 12;
      const uint8_t *first = bytes.data() + tp.sot_offset + kSotSize;
      const uint8_t *last  = bytes.data() + tp.header_end;
      expected.insert(expected.end(), first, last);
    }
    CHECK(got.size() == expected.size(), "tile-0 len=%zu expected %zu",
          got.size(), expected.size());
    CHECK(std::memcmp(got.data(), expected.data(), expected.size()) == 0,
          "tile-0 header bytes not byte-identical");
  }

  // Metadata-bin 0 (class 8) — should be present, complete, empty.
  {
    CHECK(set.contains(kMsgClassMetadata, 0), "meta bin present");
    CHECK(set.is_complete(kMsgClassMetadata, 0), "meta bin complete");
    const auto &got = set.get(kMsgClassMetadata, 0);
    CHECK(got.empty(), "meta bin should be empty, got %zu bytes", got.size());
  }

  // Keys come back sorted.
  {
    const auto keys = set.keys();
    CHECK(keys.size() == 3, "keys size=%zu", keys.size());
    // class 2 (tile header), 6 (main header), 8 (metadata) — ascending by class id.
    CHECK(keys[0].first == kMsgClassTileHeader, "keys[0].class=%u", keys[0].first);
    CHECK(keys[1].first == kMsgClassMainHeader, "keys[1].class=%u", keys[1].first);
    CHECK(keys[2].first == kMsgClassMetadata,   "keys[2].class=%u", keys[2].first);
  }

  // ── Rejection cases on the DataBinSet's append() contract ────────────
  {
    DataBinSet s;
    // Non-contiguous offset (spec permits out-of-order; v1 parser rejects).
    CHECK(!s.append(0, 0, /*offset=*/5, nullptr, 0, false),
          "append with non-zero offset on empty bin should fail");
    // Append after is_last with non-empty payload.
    const uint8_t one = 0x42;
    CHECK(s.append(0, 0, 0, &one, 1, /*is_last=*/true), "initial append");
    CHECK(!s.append(0, 0, 1, &one, 1, /*is_last=*/false),
          "append after is_last with payload should fail");
    // Empty zero-offset append after is_last is a no-op but legal (no new bytes).
    CHECK(s.append(0, 0, 1, nullptr, 0, /*is_last=*/true),
          "zero-length repeat of is_last should succeed");
  }

  // Malformed stream: truncated mid-message.
  {
    std::vector<uint8_t> bad(stream.begin(), stream.begin() + stream.size() / 2);
    DataBinSet t;
    CHECK(!parse_jpp_stream(bad.data(), bad.size(), &t),
          "truncated stream should be rejected");
  }

  if (failures == 0) {
    std::printf("OK parser_check: emit→parse round-trip + rejection cases all pass\n");
    return 0;
  }
  std::fprintf(stderr, "parser_check: %d failures\n", failures);
  return 1;
}
