// Fuzz target: a JPIP server on an untrusted codestream to serve (source/core/jpip): the
// codestream layout and index, the packet locator, a view-window resolved to precincts, the
// data-bins emitted for it; then the client's side of the same response: parser, reassembly
// (with and without the served codestream) and decode.
//
// Input = the codestream, then 8 option bytes (fuzz_common.hpp says why they come last):
//   o0 o1   view-window frame size, x (fsiz)
//   o2 o3   view-window frame size, y
//   o4 o5   region offset, x and y (roff)
//   o6 o7   region size, x and y, in units of 4 (rsiz)
#include <cstdint>
#include <exception>
#include <memory>
#include <vector>
#include "decoder.hpp"
#include "codestream_assembler.hpp"
#include "codestream_walker.hpp"
#include "data_bin_emitter.hpp"
#include "jpp_message.hpp"
#include "jpp_parser.hpp"
#include "packet_locator.hpp"
#include "precinct_index.hpp"
#include "view_window.hpp"
#include "fuzz_common.hpp"

using namespace open_htj2k;
using namespace open_htj2k::jpip;

namespace {
const size_t kOptions = 8;
// A server answers one request with the precincts of its window; these keep one run short.
const uint64_t kMaxPrecincts      = 1u << 16;
const size_t kMaxEmittedPrecincts = 4096;
const uint32_t kMaxTileHeaders    = 64;
}  // namespace

extern "C" int LLVMFuzzerInitialize(int *, char ***) {
  set_message_handler(fuzz::quiet, nullptr);
  return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  if (size < kOptions + 4) return 0;
  const size_t len   = size - kOptions;
  const uint8_t *opt = data + len;
  // The packet locator decodes the packet headers of the whole image: keep to the sizes the
  // other targets decode.
  if (fuzz::too_large(data, len)) return 0;
  ViewWindow vw;
  vw.fx = (static_cast<uint32_t>(opt[0]) << 8) | opt[1];
  vw.fy = (static_cast<uint32_t>(opt[2]) << 8) | opt[3];
  vw.ox = opt[4];
  vw.oy = opt[5];
  vw.sx = static_cast<uint32_t>(opt[6]) * 4;
  vw.sy = static_cast<uint32_t>(opt[7]) * 4;
  try {
    CodestreamLayout layout;
    if (!walk_codestream(data, len, &layout)) return 0;
    std::unique_ptr<CodestreamIndex> idx = CodestreamIndex::build(data, len);
    if (!idx || idx->total_precincts() > kMaxPrecincts) return 0;
    std::unique_ptr<PacketLocator> locator = PacketLocator::build(data, len, *idx, layout);
    if (!locator) return 0;
    const std::vector<PrecinctKey> keys = resolve_view_window(*idx, vw);
    fuzz::use(pick_discard_level(*idx, vw));

    std::vector<uint8_t> response;
    MessageHeaderContext ctx;
    emit_main_header_databin(data, len, layout, ctx, response);
    emit_metadata_bin_zero(ctx, response);
    const uint32_t tiles = idx->num_tiles_x() * idx->num_tiles_y();
    for (uint32_t t = 0; t < tiles && t < kMaxTileHeaders; ++t) {
      emit_tile_header_databin(data, len, static_cast<uint16_t>(t), layout, ctx, response);
    }
    size_t emitted = 0;
    for (const PrecinctKey &k : keys) {
      if (++emitted > kMaxEmittedPrecincts) break;
      emit_precinct_databin(data, len, k.t, k.c, k.r, k.p_rc, *idx, *locator, ctx, response);
    }
    emit_eor(EorReason::WindowDone, ctx, response);

    DataBinSet set;
    (void)parse_jpp_stream(response.data(), response.size(), &set);
    if (set.contains(6, 0)) {  // the main header data-bin
      try {
        std::unique_ptr<CodestreamIndex> client_idx =
            CodestreamIndex::build_from_main_header_bin(set.get(6, 0));
        std::vector<uint8_t> sparse;
        if (client_idx && reassemble_codestream_client(set, *client_idx, sparse) == ReassembleStatus::Ok) {
          fuzz::decode_result(sparse.data(), sparse.size());
        }
      } catch (std::exception &) {
      }
    }
    std::vector<uint8_t> full;
    if (reassemble_codestream(data, len, set, *idx, layout, *locator, full) == ReassembleStatus::Ok) {
      fuzz::decode_result(full.data(), full.size());
    }
  } catch (std::exception &) {
  }
  return 0;
}
