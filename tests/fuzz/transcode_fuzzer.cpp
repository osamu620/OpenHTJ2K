// Fuzz target: the transcoder (transcoder.hpp), Part 1 -> HTJ2K and HTJ2K -> Part 1, on an
// untrusted codestream.  The direction follows the input, as in open_htj2k_transcode.
//
// Input = the codestream, then 4 option bytes (fuzz_common.hpp says why they come last):
//   o0   bits0-1 partial plane (keep, drop, refine), bit2 layers keep (Part 1 -> HTJ2K),
//        bit3 JPH output, bits4-5 colour space of the JPH file, bit6 2 threads,
//        bit7 input given with padding
//   o1   HTJ2K -> Part 1: bit0 layers merge, bit1 take o2 as the code-block style
//   o2   code-block style for HTJ2K -> Part 1
//   o3   bit0 decode the output, bit1 transcode the output back, bit2 decode that too
#include <cstdint>
#include <exception>
#include <vector>
#include "decoder.hpp"
#include "transcoder.hpp"
#include "fuzz_common.hpp"

using namespace open_htj2k;

namespace {
const size_t kOptions = 4;
}  // namespace

extern "C" int LLVMFuzzerInitialize(int *, char ***) {
  set_message_handler(fuzz::quiet, nullptr);
  return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  if (size < kOptions + 2) return 0;
  const size_t len   = size - kOptions;
  const uint8_t *opt = data + len;
  const uint8_t o0 = opt[0], o1 = opt[1], o2 = opt[2], o3 = opt[3];
  if (fuzz::too_many_tiles(data, len)) return 0;
  const bool padded = (o0 & 0x80) != 0;
  std::vector<uint8_t> in(data, data + len);
  if (padded) in.resize(len + 16, 0);

  // The size the main header declares, as the decoder reads it.
  try {
    openhtj2k_decoder dec;
    dec.init(in.data(), len, 0, 1);
    dec.parse();
    uint64_t samples  = 0;
    const uint16_t nc = dec.get_num_component();
    for (uint16_t i = 0; i < nc; ++i) {
      samples += static_cast<uint64_t>(dec.get_component_width(i)) * dec.get_component_height(i);
    }
    if (samples > fuzz::kMaxSamples) return 0;
  } catch (std::exception &) {
    // the transcoder has checks of its own: go on
  }

  bool to_part1 = false;
  try {
    to_part1 = transcode_input_is_htj2k(in.data(), len);
  } catch (std::exception &) {
    return 0;
  }
  std::vector<uint8_t> out;
  try {
    if (to_part1) {
      transcode_to_part1_options o;
      o.input_padded = padded;
      o.layers       = (o1 & 1) ? transcode_layers::merge : transcode_layers::keep;
      if (o1 & 2) o.code_block_style = o2;
      o.num_threads = (o0 & 0x40) ? 2 : 1;
      transcode_htj2k_to_part1(in.data(), len, out, o);
    } else {
      transcode_options o;
      transcode_stats stats;
      o.input_padded     = padded;
      o.partial_plane    = static_cast<transcode_partial_plane>((o0 & 3) % 3);
      o.layers           = (o0 & 4) ? transcode_layers::keep : transcode_layers::merge;
      o.jph_output       = (o0 & 8) != 0;
      o.jph_colour_space = static_cast<transcode_colour_space>(((o0 >> 4) & 3) % 3);
      o.num_threads      = (o0 & 0x40) ? 2 : 1;
      o.stats            = &stats;
      transcode_part1_to_htj2k(in.data(), len, out, o);
      fuzz::use(stats.blocks + stats.full + stats.coarser + stats.partial + stats.refined);
    }
  } catch (std::exception &) {
    return 0;
  }
  if (out.empty()) return 0;
  if (o3 & 1) fuzz::decode_result(out.data(), out.size());
  if (o3 & 2) {
    // Back again.  (A JPH file from Part 1 -> HTJ2K reads as HTJ2K.)
    try {
      std::vector<uint8_t> back;
      if (transcode_input_is_htj2k(out.data(), out.size())) {
        transcode_htj2k_to_part1(out.data(), out.size(), back);
      } else {
        transcode_part1_to_htj2k(out.data(), out.size(), back);
      }
      if ((o3 & 4) && !back.empty()) fuzz::decode_result(back.data(), back.size());
    } catch (std::exception &) {
    }
  }
  return 0;
}
