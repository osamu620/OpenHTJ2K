// jpip_precinct_check: ctest harness for the packet locator + precinct
// data-bin emitter (the second half of B3 in PHASE2_PLAN.md).
//
// For the codestream on argv[1]:
//   1. Build the CodestreamIndex + CodestreamLayout.
//   2. Build the PacketLocator — drives the decoder with a reject-all
//      precinct filter and a packet observer that records per-packet
//      byte ranges.
//   3. Emit a precinct data-bin for EVERY precinct known to the index.
//   4. Parse the resulting JPP-stream back via B4 and verify each bin's
//      accumulated bytes match the concatenation of the locator's
//      recorded ranges byte-for-byte.
//   5. Verify the total payload across all precinct bins equals the sum
//      of (tile-part body size) across tile-parts — i.e. every packet byte
//      from the codestream is accounted for exactly once.
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
#include "jpp_parser.hpp"
#include "packet_locator.hpp"
#include "precinct_index.hpp"

using open_htj2k::jpip::CodestreamIndex;
using open_htj2k::jpip::CodestreamLayout;
using open_htj2k::jpip::DataBinSet;
using open_htj2k::jpip::emit_precinct_databin;
using open_htj2k::jpip::kMsgClassPrecinct;
using open_htj2k::jpip::MessageHeaderContext;
using open_htj2k::jpip::PacketLocator;
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
    std::fprintf(stderr, "Usage: jpip_precinct_check <input.j2c>\n");
    return 1;
  }
  auto bytes = read_file(argv[1]);
  if (bytes.empty()) return 1;

  auto idx = CodestreamIndex::build(bytes.data(), bytes.size());
  CHECK(idx != nullptr, "CodestreamIndex::build");
  if (!idx) return 1;

  CodestreamLayout layout;
  CHECK(walk_codestream(bytes.data(), bytes.size(), &layout), "walk_codestream");
  if (failures) return 1;

  auto locator = PacketLocator::build(bytes.data(), bytes.size(), *idx, layout);
  CHECK(locator != nullptr, "PacketLocator::build");
  if (!locator) return 1;

  std::printf("locator: %zu packet byte-ranges across %llu precincts\n",
              locator->size(),
              static_cast<unsigned long long>(idx->total_precincts()));

  // ── Emit every precinct data-bin into one stream buffer ───────────────
  std::vector<uint8_t> stream;
  MessageHeaderContext enc_ctx;
  std::size_t emitted = 0;
  std::size_t total_payload = 0;
  for (uint32_t t = 0; t < idx->num_tiles(); ++t) {
    for (uint16_t c = 0; c < idx->num_components(); ++c) {
      const auto &info = idx->tile_component(static_cast<uint16_t>(t), c);
      for (uint8_t r = 0; r <= info.NL; ++r) {
        const uint32_t n = info.npw[r] * info.nph[r];
        for (uint32_t p = 0; p < n; ++p) {
          const std::size_t before = stream.size();
          const std::size_t nbytes = emit_precinct_databin(
              bytes.data(), bytes.size(),
              static_cast<uint16_t>(t), c, r, p, *idx, *locator, enc_ctx, stream);
          if (nbytes > 0) {
            ++emitted;
            // Payload is the bytes after the header.  We can't split
            // exactly without re-decoding, but we know the total minus
            // any header bytes must match what was in the ranges.
            (void)before;
          }
        }
      }
    }
  }
  std::printf("emitted: %zu precinct bins  stream=%zu bytes\n", emitted, stream.size());

  // ── Parse the stream back via B4 ──────────────────────────────────────
  DataBinSet set;
  CHECK(parse_jpp_stream(stream.data(), stream.size(), &set),
        "parse_jpp_stream");
  if (failures) return 1;
  CHECK(set.size() == emitted, "set.size()=%zu, expected %zu", set.size(), emitted);

  // ── Every parsed bin's bytes must match the concatenation of the
  //    locator's recorded ranges for that precinct.  Also sum up the
  //    total to cross-check against tile-part body sizes. ──────────────
  std::size_t total_bin_bytes = 0;
  for (uint32_t t = 0; t < idx->num_tiles(); ++t) {
    for (uint16_t c = 0; c < idx->num_components(); ++c) {
      const auto &info = idx->tile_component(static_cast<uint16_t>(t), c);
      for (uint8_t r = 0; r <= info.NL; ++r) {
        const uint32_t n = info.npw[r] * info.nph[r];
        for (uint32_t p = 0; p < n; ++p) {
          const auto &ranges = locator->packets_of(static_cast<uint16_t>(t), c, r, p);
          if (ranges.empty()) continue;
          const uint64_t I = idx->I(static_cast<uint16_t>(t), c, r, p);
          CHECK(set.contains(kMsgClassPrecinct, I),
                "precinct bin I=%llu missing (t=%u c=%u r=%u p=%u)",
                static_cast<unsigned long long>(I), t, c, r, p);
          CHECK(set.is_complete(kMsgClassPrecinct, I),
                "precinct bin I=%llu not complete", static_cast<unsigned long long>(I));
          const auto &got = set.get(kMsgClassPrecinct, I);
          std::vector<uint8_t> expected;
          for (const auto &rg : ranges) {
            expected.insert(expected.end(), bytes.data() + rg.offset,
                            bytes.data() + rg.offset + rg.length);
          }
          CHECK(got.size() == expected.size(),
                "bin I=%llu size %zu expected %zu",
                static_cast<unsigned long long>(I), got.size(), expected.size());
          CHECK(std::memcmp(got.data(), expected.data(), expected.size()) == 0,
                "bin I=%llu bytes differ (t=%u c=%u r=%u p=%u)",
                static_cast<unsigned long long>(I), t, c, r, p);
          total_bin_bytes += got.size();
        }
      }
    }
  }

  // Sum of tile-part body sizes (packet data only, excluding headers).
  std::size_t expected_total = 0;
  for (const auto &tp : layout.tile_parts) {
    expected_total += (tp.body_end > tp.body_offset) ? (tp.body_end - tp.body_offset) : 0;
  }
  CHECK(total_bin_bytes == expected_total,
        "aggregated precinct bytes %zu != tile-part body bytes %zu",
        total_bin_bytes, expected_total);
  (void)total_payload;

  if (failures == 0) {
    std::printf("OK precinct_check: every precinct bin round-trips through emit→parse "
                "byte-identical, and accounts for all %zu tile-part body bytes\n",
                expected_total);
    return 0;
  }
  std::fprintf(stderr, "precinct_check: %d failures\n", failures);
  return 1;
}
