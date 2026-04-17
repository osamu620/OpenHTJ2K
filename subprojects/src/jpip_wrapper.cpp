#ifdef __EMSCRIPTEN__
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

#include <emscripten.h>

#include "decoder.hpp"
#include "precinct_index.hpp"
#include "jpp_parser.hpp"
#include "codestream_assembler.hpp"
#include "jpp_message.hpp"

// ────────────────────────────────────────────────────────────────────────────
// JPIP WASM API — thin C-linkage entry points for the browser demo.
//
// Lifecycle:
//   1. JS fetches the main-header data-bin from the JPIP server.
//   2. jpip_create_context(bin, len) → opaque handle.
//   3. Per frame:
//      a. jpip_begin_frame(handle)              — clears the DataBinSet.
//      b. jpip_add_response(handle, data, len)  — parse a JPP-stream
//                                                  response and merge.
//         (called 1–3 times: fovea / parafovea / periphery)
//      c. jpip_end_frame(handle, rgb, w, h)     — reassemble + decode +
//                                                  write RGB8.
//   4. jpip_destroy_context(handle).
// ────────────────────────────────────────────────────────────────────────────

struct JpipContext {
  std::unique_ptr<open_htj2k::jpip::CodestreamIndex> idx;
  open_htj2k::jpip::DataBinSet set;
  uint32_t canvas_w  = 0;
  uint32_t canvas_h  = 0;
  uint8_t  reduce_NL = 0;
};

extern "C" {

// Create context from a JPP-stream response (as returned by the JPIP
// server).  Parses the response to extract the main-header data-bin
// (class 6, id 0), then builds the CodestreamIndex from it.
EMSCRIPTEN_KEEPALIVE
void *jpip_create_context(const uint8_t *jpp_stream, size_t len) {
  if (!jpp_stream || len == 0) return nullptr;
  open_htj2k::jpip::DataBinSet tmp;
  if (!open_htj2k::jpip::parse_jpp_stream(jpp_stream, len, &tmp)) return nullptr;
  const auto &mh_bin = tmp.get(open_htj2k::jpip::kMsgClassMainHeader, 0);
  if (mh_bin.empty()) return nullptr;
  auto idx = open_htj2k::jpip::CodestreamIndex::build_from_main_header_bin(mh_bin);
  if (!idx) return nullptr;
  auto *ctx     = new JpipContext();
  ctx->canvas_w = idx->geometry().canvas_size.x;
  ctx->canvas_h = idx->geometry().canvas_size.y;
  ctx->idx      = std::move(idx);
  return ctx;
}

EMSCRIPTEN_KEEPALIVE
int jpip_get_canvas_width(void *handle) {
  if (!handle) return 0;
  return static_cast<int>(static_cast<JpipContext *>(handle)->canvas_w);
}

EMSCRIPTEN_KEEPALIVE
int jpip_get_canvas_height(void *handle) {
  if (!handle) return 0;
  return static_cast<int>(static_cast<JpipContext *>(handle)->canvas_h);
}

EMSCRIPTEN_KEEPALIVE
int jpip_get_num_components(void *handle) {
  if (!handle) return 0;
  return static_cast<int>(static_cast<JpipContext *>(handle)->idx->num_components());
}

EMSCRIPTEN_KEEPALIVE
int jpip_get_total_precincts(void *handle) {
  if (!handle) return 0;
  return static_cast<int>(static_cast<JpipContext *>(handle)->idx->total_precincts());
}

EMSCRIPTEN_KEEPALIVE
void jpip_set_reduce(void *handle, int n) {
  if (!handle) return;
  static_cast<JpipContext *>(handle)->reduce_NL = static_cast<uint8_t>(n);
}

EMSCRIPTEN_KEEPALIVE
void jpip_begin_frame(void *handle) {
  if (!handle) return;
  static_cast<JpipContext *>(handle)->set = {};
}

EMSCRIPTEN_KEEPALIVE
int jpip_add_response(void *handle, const uint8_t *jpp_stream, size_t len) {
  if (!handle || !jpp_stream || len == 0) return -1;
  auto *ctx = static_cast<JpipContext *>(handle);
  open_htj2k::jpip::DataBinSet tmp;
  if (!open_htj2k::jpip::parse_jpp_stream(jpp_stream, len, &tmp)) return -2;
  ctx->set.merge_from(tmp);
  return 0;
}

EMSCRIPTEN_KEEPALIVE
int jpip_end_frame(void *handle, uint8_t *rgb_out, int out_w, int out_h) {
  if (!handle || !rgb_out || out_w <= 0 || out_h <= 0) return -1;
  auto *ctx = static_cast<JpipContext *>(handle);

  // Reassemble the sparse codestream from the accumulated DataBinSet.
  std::vector<uint8_t> sparse_cs;
  auto rc = open_htj2k::jpip::reassemble_codestream_client(ctx->set, *ctx->idx, sparse_cs);
  if (rc != open_htj2k::jpip::ReassembleStatus::Ok) return -2;

  // Decode.
  open_htj2k::openhtj2k_decoder dec;
#ifdef OPENHTJ2K_THREAD
  dec.init(sparse_cs.data(), sparse_cs.size(), ctx->reduce_NL, 0);
#else
  dec.init(sparse_cs.data(), sparse_cs.size(), ctx->reduce_NL, 1);
#endif
  dec.parse();

  std::vector<uint32_t> widths, heights;
  std::vector<uint8_t>  depths;
  std::vector<bool>     signeds;
  const uint32_t ow = static_cast<uint32_t>(out_w);
  const uint32_t oh = static_cast<uint32_t>(out_h);

  try {
    dec.invoke_line_based_stream(
        [&](uint32_t y, int32_t *const *rows, uint16_t nc) {
          if (nc < 3 || widths.empty() || heights.empty()) return;
          const uint32_t cw = widths[0];
          const uint32_t ch = heights[0];
          const int32_t shift = (depths.empty() ? 0 : static_cast<int32_t>(depths[0]) - 8);
          const uint32_t ty =
              static_cast<uint32_t>(static_cast<uint64_t>(y) * oh / (ch > 0 ? ch : 1));
          if (ty >= oh) return;
          uint8_t *dst = rgb_out + static_cast<size_t>(ty) * ow * 4;  // RGBA
          for (uint32_t xw = 0; xw < ow; ++xw) {
            const uint32_t xc =
                static_cast<uint32_t>(static_cast<uint64_t>(xw) * cw / (ow > 0 ? ow : 1));
            auto to_u8 = [shift](int32_t v) -> uint8_t {
              if (shift > 0) v >>= shift;
              if (v < 0) return 0;
              if (v > 255) return 255;
              return static_cast<uint8_t>(v);
            };
            dst[4 * xw + 0] = to_u8(rows[0][xc]);
            dst[4 * xw + 1] = to_u8(rows[1][xc]);
            dst[4 * xw + 2] = to_u8(rows[2][xc]);
            dst[4 * xw + 3] = 255;  // alpha
          }
        },
        widths, heights, depths, signeds);
  } catch (...) {
    return -3;
  }
  return 0;
}

EMSCRIPTEN_KEEPALIVE
void jpip_destroy_context(void *handle) {
  delete static_cast<JpipContext *>(handle);
}

}  // extern "C"
#endif  // __EMSCRIPTEN__
