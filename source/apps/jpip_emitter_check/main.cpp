// jpip_emitter_check: ctest harness for the codestream walker + header
// data-bin emitters added in PHASE2_PLAN.md item B3 (header bins).
//
// Usage:  jpip_emitter_check <input.j2c>
//
// For the asset given on the command line, this:
//   - walks the codestream (codestream_walker) and prints the layout summary,
//   - emits the main-header data-bin and re-decodes its message header
//     to check class=6, in_class_id=0, payload byte-identical to the
//     codestream's [SOC..first SOT) range,
//   - emits the tile-header data-bin for tile 0 and re-decodes its header
//     to check class=2, in_class_id=0, payload byte-identical to the
//     concatenation of every contributing tile-part's marker segments
//     between SOT and SOD,
//   - emits metadata-bin 0, re-decodes, expects class=8, in_class_id=0,
//     payload empty, is_last=true.
//
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

using open_htj2k::jpip::CodestreamLayout;
using open_htj2k::jpip::decode_header;
using open_htj2k::jpip::emit_main_header_databin;
using open_htj2k::jpip::emit_metadata_bin_zero;
using open_htj2k::jpip::emit_tile_header_databin;
using open_htj2k::jpip::kMsgClassMainHeader;
using open_htj2k::jpip::kMsgClassMetadata;
using open_htj2k::jpip::kMsgClassTileHeader;
using open_htj2k::jpip::MessageHeader;
using open_htj2k::jpip::MessageHeaderContext;
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
    std::fprintf(stderr, "Usage: jpip_emitter_check <input.j2c>\n");
    return 1;
  }

  auto bytes = read_file(argv[1]);
  if (bytes.empty()) return 1;

  CodestreamLayout layout;
  CHECK(walk_codestream(bytes.data(), bytes.size(), &layout), "walk_codestream");
  if (failures) return 1;

  std::printf("layout: SOC=%zu  main_header_end=%zu  EOC=%zu  tile_parts=%zu\n",
              layout.soc_offset, layout.main_header_end, layout.eoc_offset,
              layout.tile_parts.size());
  for (const auto &tp : layout.tile_parts) {
    std::printf("  tile=%u part=%u/%u  SOT=%zu  header_end=%zu  body=[%zu, %zu)\n",
                tp.tile_index, tp.tile_part_idx, tp.tile_part_cnt,
                tp.sot_offset, tp.header_end, tp.body_offset, tp.body_end);
  }

  CHECK(layout.main_header_end > 0, "main_header_end must be > 0");
  CHECK(!layout.tile_parts.empty(), "must have at least one tile-part");

  // Emit + decode the three header data-bins back-to-back through one ctx
  // so dependent-form encoding gets exercised.
  std::vector<uint8_t> stream;
  MessageHeaderContext enc_ctx;

  const std::size_t main_bytes = emit_main_header_databin(
      bytes.data(), bytes.size(), layout, enc_ctx, stream);
  CHECK(main_bytes > 0, "emit_main_header_databin returned 0");

  // Tile 0 should always exist for our test assets.
  const std::size_t tile0_bytes = emit_tile_header_databin(
      bytes.data(), bytes.size(), /*tile_index=*/0, layout, enc_ctx, stream);
  CHECK(tile0_bytes > 0, "emit_tile_header_databin tile=0 returned 0");

  const std::size_t meta_bytes = emit_metadata_bin_zero(enc_ctx, stream);
  CHECK(meta_bytes > 0, "emit_metadata_bin_zero returned 0");

  std::printf("emitted: main=%zu  tile0=%zu  metadata0=%zu  total=%zu\n",
              main_bytes, tile0_bytes, meta_bytes, stream.size());

  // Decode and verify.
  MessageHeaderContext dec_ctx;
  std::size_t pos = 0;

  // Main header.
  {
    MessageHeader m;
    std::size_t adv = 0;
    CHECK(decode_header(stream.data() + pos, stream.size() - pos, dec_ctx, &m, &adv),
          "decode main header");
    CHECK(m.class_id == kMsgClassMainHeader, "main class=%u", m.class_id);
    CHECK(m.cs_n == 0, "main cs_n=%u", m.cs_n);
    CHECK(m.in_class_id == 0, "main in_class_id=%llu",
          static_cast<unsigned long long>(m.in_class_id));
    CHECK(m.is_last, "main is_last");
    const std::size_t expected_len = layout.main_header_end - layout.soc_offset;
    CHECK(m.msg_length == expected_len, "main msg_length=%llu expected %zu",
          static_cast<unsigned long long>(m.msg_length), expected_len);
    pos += adv;
    CHECK(std::memcmp(stream.data() + pos, bytes.data() + layout.soc_offset,
                      expected_len) == 0,
          "main header payload not byte-identical to codestream slice");
    pos += expected_len;
  }

  // Tile-0 header.
  {
    MessageHeader m;
    std::size_t adv = 0;
    CHECK(decode_header(stream.data() + pos, stream.size() - pos, dec_ctx, &m, &adv),
          "decode tile header");
    CHECK(m.class_id == kMsgClassTileHeader, "tile class=%u", m.class_id);
    CHECK(m.in_class_id == 0, "tile in_class_id=%llu",
          static_cast<unsigned long long>(m.in_class_id));
    CHECK(m.is_last, "tile is_last");
    pos += adv;

    // Reconstruct the expected tile-header payload by concatenating each
    // tile-part's marker bytes (excluding SOT) for tile 0.
    std::vector<uint8_t> expected;
    for (const auto &tp : layout.tile_parts) {
      if (tp.tile_index != 0) continue;
      constexpr std::size_t kSotSize = 12;
      const uint8_t *first = bytes.data() + tp.sot_offset + kSotSize;
      const uint8_t *last  = bytes.data() + tp.header_end;
      expected.insert(expected.end(), first, last);
    }
    CHECK(m.msg_length == expected.size(),
          "tile msg_length=%llu expected %zu",
          static_cast<unsigned long long>(m.msg_length), expected.size());
    CHECK(std::memcmp(stream.data() + pos, expected.data(), expected.size()) == 0,
          "tile header payload not byte-identical to expected concatenation");
    pos += expected.size();
  }

  // Metadata-bin 0.
  {
    MessageHeader m;
    std::size_t adv = 0;
    CHECK(decode_header(stream.data() + pos, stream.size() - pos, dec_ctx, &m, &adv),
          "decode metadata bin 0");
    CHECK(m.class_id == kMsgClassMetadata, "meta class=%u", m.class_id);
    CHECK(m.in_class_id == 0, "meta in_class_id=%llu",
          static_cast<unsigned long long>(m.in_class_id));
    CHECK(m.is_last, "meta is_last");
    CHECK(m.msg_length == 0, "meta msg_length=%llu", static_cast<unsigned long long>(m.msg_length));
    pos += adv;
  }

  CHECK(pos == stream.size(), "stream not fully consumed: %zu/%zu", pos, stream.size());

  if (failures == 0) {
    std::printf("OK emitter_check: main / tile0 / metadata round-trip cleanly\n");
    return 0;
  }
  std::fprintf(stderr, "emitter_check: %d failures\n", failures);
  return 1;
}
