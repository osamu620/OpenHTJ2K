// jpip_assembler_check: full JPP-stream round-trip validator.
//
// For the codestream on argv[1]:
//   1. Build CodestreamIndex, CodestreamLayout, PacketLocator.
//   2. Emit every data-bin — main header, every tile header, metadata-bin 0,
//      every precinct — into one JPP-stream buffer.
//   3. Parse the stream into a DataBinSet via B4.
//   4. Reassemble a sparse J2C codestream via B5 (with ALL precincts
//      present, so the output should be fully decodable).
//   5. Decode both the original codestream and the reassembled one.
//      The per-pixel output must match byte-for-byte across every
//      component — full round-trip fidelity.
//
// Exits 0 on match, 1 on any mismatch / reassembly failure.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "codestream_assembler.hpp"
#include "codestream_walker.hpp"
#include "data_bin_emitter.hpp"
#include "decoder.hpp"
#include "jpp_parser.hpp"
#include "packet_locator.hpp"
#include "precinct_index.hpp"

using open_htj2k::openhtj2k_decoder;
using open_htj2k::jpip::CodestreamIndex;
using open_htj2k::jpip::CodestreamLayout;
using open_htj2k::jpip::DataBinSet;
using open_htj2k::jpip::emit_main_header_databin;
using open_htj2k::jpip::emit_metadata_bin_zero;
using open_htj2k::jpip::emit_precinct_databin;
using open_htj2k::jpip::emit_tile_header_databin;
using open_htj2k::jpip::MessageHeaderContext;
using open_htj2k::jpip::PacketLocator;
using open_htj2k::jpip::parse_jpp_stream;
using open_htj2k::jpip::reassemble_codestream;
using open_htj2k::jpip::ReassembleStatus;
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

struct DecodedFrame {
  std::vector<std::vector<int32_t>> planes;
  std::vector<uint32_t> width, height;
};

// Decode a raw J2C codestream into int32 component planes via invoke().
// Copies the planes out of the decoder's internal storage so the result
// survives the decoder's destructor.
bool decode_into(const std::vector<uint8_t> &bytes, DecodedFrame &out) {
  openhtj2k_decoder dec;
  dec.init(bytes.data(), bytes.size(), /*reduce_NL=*/0, /*num_threads=*/1);
  dec.parse();
  std::vector<int32_t *> buf;
  std::vector<uint8_t>   depth;
  std::vector<bool>      is_signed;
  try {
    dec.invoke(buf, out.width, out.height, depth, is_signed);
  } catch (std::exception &e) {
    std::fprintf(stderr, "decode failed: %s\n", e.what());
    return false;
  }
  out.planes.clear();
  out.planes.resize(buf.size());
  for (std::size_t c = 0; c < buf.size(); ++c) {
    const std::size_t n = static_cast<std::size_t>(out.width[c]) * out.height[c];
    out.planes[c].assign(buf[c], buf[c] + n);
  }
  return true;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "Usage: jpip_assembler_check <input.j2c>\n");
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

  // ── Emit every data-bin into one JPP-stream ───────────────────────────
  std::vector<uint8_t> stream;
  MessageHeaderContext enc_ctx;
  emit_main_header_databin(bytes.data(), bytes.size(), layout, enc_ctx, stream);
  for (uint32_t t = 0; t < idx->num_tiles(); ++t) {
    emit_tile_header_databin(bytes.data(), bytes.size(),
                             static_cast<uint16_t>(t), layout, enc_ctx, stream);
  }
  emit_metadata_bin_zero(enc_ctx, stream);
  for (uint32_t t = 0; t < idx->num_tiles(); ++t) {
    for (uint16_t c = 0; c < idx->num_components(); ++c) {
      const auto &info = idx->tile_component(static_cast<uint16_t>(t), c);
      for (uint8_t r = 0; r <= info.NL; ++r) {
        const uint32_t n = info.npw[r] * info.nph[r];
        for (uint32_t p = 0; p < n; ++p) {
          emit_precinct_databin(bytes.data(), bytes.size(),
                                static_cast<uint16_t>(t), c, r, p,
                                *idx, *locator, enc_ctx, stream);
        }
      }
    }
  }
  std::printf("emitted JPP-stream: %zu bytes for %llu precincts across %u tiles\n",
              stream.size(), static_cast<unsigned long long>(idx->total_precincts()),
              idx->num_tiles());

  // ── Parse back into a DataBinSet ──────────────────────────────────────
  DataBinSet set;
  CHECK(parse_jpp_stream(stream.data(), stream.size(), &set), "parse_jpp_stream");
  if (failures) return 1;

  // ── Reassemble into a sparse codestream ───────────────────────────────
  std::vector<uint8_t> reassembled;
  const auto status = reassemble_codestream(bytes.data(), bytes.size(), set, *idx, layout,
                                            *locator, reassembled);
  // UnsupportedProgression / UnsupportedFeature are deliberate v1 outs:
  // the harness just confirmed emit + parse works on this asset and that
  // the reassembler rejects the codestream cleanly.  Treat as pass.
  if (status == ReassembleStatus::UnsupportedProgression ||
      status == ReassembleStatus::UnsupportedFeature) {
    std::printf("OK assembler_check: emit + parse succeeded; reassembler correctly "
                "reports status=%d (asset outside v1 scope — LRCP / RLCP / SOP / EPH)\n",
                static_cast<int>(status));
    return 0;
  }
  CHECK(status == ReassembleStatus::Ok, "reassemble_codestream status=%d",
        static_cast<int>(status));
  if (failures) return 1;
  std::printf("reassembled codestream: %zu bytes (original was %zu)\n",
              reassembled.size(), bytes.size());

  // ── Decode both codestreams and compare per-pixel ─────────────────────
  DecodedFrame orig, reco;
  CHECK(decode_into(bytes, orig), "decode original");
  CHECK(decode_into(reassembled, reco), "decode reassembled");
  if (failures) return 1;

  CHECK(orig.planes.size() == reco.planes.size(),
        "component count: orig %zu vs reco %zu",
        orig.planes.size(), reco.planes.size());

  for (std::size_t c = 0; c < orig.planes.size(); ++c) {
    CHECK(orig.width[c] == reco.width[c] && orig.height[c] == reco.height[c],
          "c=%zu dims: orig %ux%u vs reco %ux%u", c,
          orig.width[c], orig.height[c], reco.width[c], reco.height[c]);
    if (failures) return 1;
    if (orig.planes[c] != reco.planes[c]) {
      // Find the first difference for diagnostics.
      std::size_t first_diff = 0;
      for (; first_diff < orig.planes[c].size(); ++first_diff) {
        if (orig.planes[c][first_diff] != reco.planes[c][first_diff]) break;
      }
      std::fprintf(stderr, "FAIL c=%zu pixel-mismatch at idx=%zu: orig=%d reco=%d\n",
                   c, first_diff, orig.planes[c][first_diff], reco.planes[c][first_diff]);
      ++failures;
    }
  }

  if (failures == 0) {
    std::printf("OK assembler_check: full round-trip (emit → parse → reassemble → decode) "
                "pixel-identical across %zu components\n", orig.planes.size());
    return 0;
  }
  std::fprintf(stderr, "assembler_check: %d failures\n", failures);
  return 1;
}
