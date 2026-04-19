#ifdef __EMSCRIPTEN__
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

#include <emscripten.h>

#include <string>

#include "decoder.hpp"
#include "precinct_index.hpp"
#include "jpp_parser.hpp"
#include "codestream_assembler.hpp"
#include "jpp_message.hpp"
#include "cache_model.hpp"

// ────────────────────────────────────────────────────────────────────────────
// JPIP WASM API — thin C-linkage entry points for the browser demo.
//
// Lifecycle:
//   1. JS fetches the main-header data-bin from the JPIP server.
//   2. jpip_create_context(bin, len) → opaque handle.  Enables single-tile
//      decoder reuse so the parsed tile tree is cached across frames.
//   3. Per frame:
//      a. jpip_begin_frame(handle)              — no-op (reserved; kept for
//                                                  ABI compatibility).  The
//                                                  DataBinSet accumulates
//                                                  across frames so cached
//                                                  precincts from prior
//                                                  viewports are reused.
//      b. jpip_add_response(handle, data, len)  — parse a JPP-stream
//                                                  response and merge.
//         (called 1–3 times: fovea / parafovea / periphery)
//      c. jpip_end_frame(handle, rgb, w, h)     — reassemble + decode +
//                                                  write RGB8.
//   4. jpip_reset_cache(handle)                 — explicit full cache reset
//                                                  (e.g. on disconnect).
//   5. jpip_destroy_context(handle).
// ────────────────────────────────────────────────────────────────────────────

struct JpipContext {
  std::unique_ptr<open_htj2k::jpip::CodestreamIndex> idx;
  open_htj2k::jpip::DataBinSet set;
  uint32_t canvas_w  = 0;
  uint32_t canvas_h  = 0;
  uint8_t  reduce_NL = 0;
  // Persistent decoder.  enable_single_tile_reuse() caches the parsed tile
  // tree (codeblock allocations, tagtrees, line-decode ring buffers) across
  // init() calls whose main-header bytes (SIZ/COD/QCD) are byte-identical.
  // Every sparse codestream reassembled from the same CodestreamIndex has
  // the same main header, so the cache hits on every frame after the first.
  open_htj2k::openhtj2k_decoder dec;
  // Reassembled codestream kept alive between frames so the decoder's
  // init() doesn't have to memcpy into its own buffer from a local each
  // time; also lets us skip reassembly when no new data-bins arrived.
  std::vector<uint8_t> sparse_cs;
  bool dec_initialized = false;
  // Tracks the reduce_NL value the persistent decoder was last initialized
  // with.  reduce_NL is NOT part of the main-header bytes that the reuse
  // cache fingerprints, so a reduce change (user zoom) would silently reuse
  // a tile tree sized for a different resolution and corrupt the output.
  // When this differs from ctx->reduce_NL we manually flush the cache.
  uint8_t last_reduce_NL = 0xFF;
  // Set by add_response when new data-bins land; cleared after a successful
  // decode.  Lets jpip_end_frame*() skip reassembly when nothing changed
  // and the viewer's lastFetchKey guard was bypassed by an upstream resize.
  bool dirty = false;
  // Grow-only staging buffer for JPIP server responses.  Exposed to JS via
  // jpip_get_response_buffer() so the viewer and foveation demo can skip
  // the per-frame _malloc / wasmWrite / _free triplet (3× per foveation
  // frame) and the repeated WASM heap churn that came with it.
  std::vector<uint8_t> response_buf;
  // Cache model the client has authoritatively received — headers, tile
  // headers, metadata-bin-0.  Precincts are intentionally not tracked:
  // the foveation demo drops precinct state per frame so the periphery
  // decays when the gaze moves.  Exposed to JS via jpip_get_cache_model()
  // so the outgoing query can carry `&model=` and the server skips
  // redundant header bytes.  Rebuilt lazily into `cache_model_str` when
  // the JS caller asks for it.
  open_htj2k::jpip::CacheModel client_cache;
  std::string cache_model_str;
};

// Flush the single-tile reuse cache when a parameter that isn't part of the
// main-header fingerprint (currently: reduce_NL) has changed.  Toggling the
// flag off and back on drops the cached tile tree so the next init+parse
// builds a fresh one sized for the new resolution.
static void jpip_maybe_invalidate_reuse(JpipContext *ctx) {
  if (ctx->dec_initialized && ctx->last_reduce_NL != ctx->reduce_NL) {
    ctx->dec.enable_single_tile_reuse(false);
    ctx->dec.enable_single_tile_reuse(true);
    ctx->dec_initialized = false;
  }
}

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
  ctx->dec.enable_single_tile_reuse(true);
  // Seed the cache with the main-header data-bin that built the index,
  // and mark every non-precinct bin present in the client CacheModel so
  // the very first follow-up request carries an accurate &model= header.
  ctx->set.merge_from(tmp);
  for (const auto &kv : tmp.keys()) {
    if (kv.first == open_htj2k::jpip::kMsgClassPrecinct
        || kv.first == open_htj2k::jpip::kMsgClassExtPrecinct) continue;
    if (tmp.is_complete(kv.first, kv.second))
      ctx->client_cache.mark(kv.first, kv.second);
  }
  ctx->cache_model_str.clear();
  ctx->dirty = true;
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

// Returns a C string the JS caller can hand straight to UTF8ToString().
// The string is the client's current cache-model request-field body
// (§C.9) — e.g. "Hm,Ht0,M0" — which JS appends as `&model=<string>`
// to every JPIP query.  The returned pointer is owned by the context
// and is invalidated by the next call to this function or any call
// that marks new bins (jpip_add_response, jpip_create_context).
EMSCRIPTEN_KEEPALIVE
const char *jpip_get_cache_model(void *handle) {
  if (!handle) return "";
  auto *ctx = static_cast<JpipContext *>(handle);
  if (ctx->cache_model_str.empty() && ctx->client_cache.size() > 0) {
    ctx->cache_model_str = ctx->client_cache.format();
  }
  return ctx->cache_model_str.c_str();
}

// Kept as a no-op: the viewer calls it before every jpip_add_response() but
// clearing the DataBinSet here would throw away everything JPIP has cached
// for earlier viewports, forcing the server to resend main-header and
// previously-delivered precincts on every pan.  Use jpip_reset_cache() if
// you explicitly want to drop accumulated state (e.g. on disconnect).
EMSCRIPTEN_KEEPALIVE
void jpip_begin_frame(void *handle) {
  (void)handle;
}

// Drop accumulated precincts but keep headers / tile-headers / metadata
// bins.  The foveation demo calls this every frame so the periphery
// decays when the gaze moves (commit a6749fd); preserving the header
// bins means the reassembler still has them and the &model= header we
// advertise stays consistent with what WASM actually holds.  The
// client_cache isn't touched for the same reason — the server will keep
// skipping the headers it already sent.
EMSCRIPTEN_KEEPALIVE
void jpip_reset_cache(void *handle) {
  if (!handle) return;
  auto *ctx = static_cast<JpipContext *>(handle);
  open_htj2k::jpip::DataBinSet kept;
  for (const auto &kv : ctx->set.keys()) {
    if (kv.first == open_htj2k::jpip::kMsgClassPrecinct
        || kv.first == open_htj2k::jpip::kMsgClassExtPrecinct) continue;
    const auto &bin = ctx->set.get(kv.first, kv.second);
    kept.append(kv.first, kv.second, 0, bin.data(), bin.size(),
                ctx->set.is_complete(kv.first, kv.second));
  }
  ctx->set = std::move(kept);
  ctx->sparse_cs.clear();
  ctx->dec_initialized = false;
  ctx->last_reduce_NL = 0xFF;
  // We dropped precincts, so the next end_frame() has to reassemble.
  ctx->dirty = true;
}

// Return a pointer to the context's grow-only response staging buffer,
// resized to at least `min_size` bytes.  JS is expected to write the
// next JPIP response body into [ptr, ptr + min_size) and then call
// jpip_add_response(handle, ptr, actual_len) — no _malloc / _free
// round-trip per frame.  The returned pointer is stable across calls
// until a growth forces reallocation; callers MUST call this function
// (and take its return value) before every write, never cache the
// pointer across calls.
EMSCRIPTEN_KEEPALIVE
uint8_t *jpip_get_response_buffer(void *handle, size_t min_size) {
  if (!handle) return nullptr;
  auto *ctx = static_cast<JpipContext *>(handle);
  if (ctx->response_buf.size() < min_size) ctx->response_buf.resize(min_size);
  return ctx->response_buf.data();
}

EMSCRIPTEN_KEEPALIVE
int jpip_add_response(void *handle, const uint8_t *jpp_stream, size_t len) {
  if (!handle || !jpp_stream || len == 0) return -1;
  auto *ctx = static_cast<JpipContext *>(handle);
  open_htj2k::jpip::DataBinSet tmp;
  if (!open_htj2k::jpip::parse_jpp_stream(jpp_stream, len, &tmp)) return -2;
  ctx->set.merge_from(tmp);
  // Track completed non-precinct bins (headers, tile-headers, metadata) in
  // the client cache model so the next &model= advertisement lets the
  // server skip them.  Precincts are intentionally omitted — the foveation
  // demo's per-frame reset drops them anyway.
  bool cache_changed = false;
  for (const auto &kv : tmp.keys()) {
    if (kv.first == open_htj2k::jpip::kMsgClassPrecinct
        || kv.first == open_htj2k::jpip::kMsgClassExtPrecinct) continue;
    if (!tmp.is_complete(kv.first, kv.second)) continue;
    if (!ctx->client_cache.has(kv.first, kv.second)) {
      ctx->client_cache.mark(kv.first, kv.second);
      cache_changed = true;
    }
  }
  if (cache_changed) ctx->cache_model_str.clear();
  ctx->dirty = true;
  return 0;
}

EMSCRIPTEN_KEEPALIVE
int jpip_end_frame(void *handle, uint8_t *rgb_out, int out_w, int out_h) {
  if (!handle || !rgb_out || out_w <= 0 || out_h <= 0) return -1;
  auto *ctx = static_cast<JpipContext *>(handle);

  // Reassemble the sparse codestream from the accumulated DataBinSet.  Only
  // rebuild when new data arrived since the last decode — otherwise the
  // existing ctx->sparse_cs is already correct and the persistent decoder's
  // cached tile tree is valid.
  if (ctx->dirty || ctx->sparse_cs.empty()) {
    ctx->sparse_cs.clear();
    auto rc = open_htj2k::jpip::reassemble_codestream_client(ctx->set, *ctx->idx, ctx->sparse_cs);
    if (rc != open_htj2k::jpip::ReassembleStatus::Ok) return -2;
  }

  // Decode.  First init after create_context uses build-default threading;
  // subsequent inits pass num_threads=1 to preserve the ThreadPool that was
  // spun up on the first call.
  jpip_maybe_invalidate_reuse(ctx);
  auto &dec = ctx->dec;
  if (!ctx->dec_initialized) {
#ifdef OPENHTJ2K_THREAD
    dec.init(ctx->sparse_cs.data(), ctx->sparse_cs.size(), ctx->reduce_NL, 0);
#else
    dec.init(ctx->sparse_cs.data(), ctx->sparse_cs.size(), ctx->reduce_NL, 1);
#endif
    ctx->dec_initialized = true;
  } else {
    dec.init(ctx->sparse_cs.data(), ctx->sparse_cs.size(), ctx->reduce_NL, 1);
  }
  ctx->last_reduce_NL = ctx->reduce_NL;
  dec.parse();

  std::vector<uint32_t> widths, heights;
  std::vector<uint8_t>  depths;
  std::vector<bool>     signeds;
  const uint32_t ow = static_cast<uint32_t>(out_w);
  const uint32_t oh = static_cast<uint32_t>(out_h);

  // Column index LUT — xc[xw] = xw · cw / ow.  Built on the first row
  // callback (cw only becomes known once the decoder populates `widths`).
  // Hoisting this out of the pixel loop eliminates an O(ow · oh) worth of
  // per-pixel 64-bit divides that WASM can't strength-reduce.
  std::vector<uint32_t> xc_lut;
  uint32_t last_ch = 0;

  try {
    dec.invoke_line_based_stream_reuse(
        [&](uint32_t y, int32_t *const *rows, uint16_t nc) {
          if (nc < 3 || widths.empty() || heights.empty()) return;
          const uint32_t cw = widths[0];
          const uint32_t ch = heights[0];
          if (xc_lut.size() != ow || last_ch != ch) {
            xc_lut.resize(ow);
            for (uint32_t xw = 0; xw < ow; ++xw) {
              xc_lut[xw] = static_cast<uint32_t>(
                  static_cast<uint64_t>(xw) * cw / (ow > 0 ? ow : 1));
            }
            last_ch = ch;
          }
          const int32_t shift = (depths.empty() ? 0 : static_cast<int32_t>(depths[0]) - 8);
          const int32_t shift_pos = shift > 0 ? shift : 0;  // no-op when bit-depth <= 8
          const uint32_t ty0 =
              static_cast<uint32_t>(static_cast<uint64_t>(y) * oh / (ch > 0 ? ch : 1));
          const uint32_t ty1 =
              static_cast<uint32_t>(static_cast<uint64_t>(y + 1) * oh / (ch > 0 ? ch : 1));
          if (ty0 >= oh) return;
          uint8_t *dst = rgb_out + static_cast<size_t>(ty0) * ow * 4;
          const int32_t *__restrict__ r0 = rows[0];
          const int32_t *__restrict__ r1 = rows[1];
          const int32_t *__restrict__ r2 = rows[2];
          const uint32_t *__restrict__ lut = xc_lut.data();
          for (uint32_t xw = 0; xw < ow; ++xw) {
            const uint32_t xc = lut[xw];
            int32_t v0 = r0[xc] >> shift_pos;
            int32_t v1 = r1[xc] >> shift_pos;
            int32_t v2 = r2[xc] >> shift_pos;
            if (v0 < 0) v0 = 0; else if (v0 > 255) v0 = 255;
            if (v1 < 0) v1 = 0; else if (v1 > 255) v1 = 255;
            if (v2 < 0) v2 = 0; else if (v2 > 255) v2 = 255;
            dst[4 * xw + 0] = static_cast<uint8_t>(v0);
            dst[4 * xw + 1] = static_cast<uint8_t>(v1);
            dst[4 * xw + 2] = static_cast<uint8_t>(v2);
            dst[4 * xw + 3] = 255;
          }
          for (uint32_t ty = ty0 + 1; ty < ty1 && ty < oh; ++ty) {
            std::memcpy(rgb_out + static_cast<size_t>(ty) * ow * 4, dst, ow * 4);
          }
        },
        widths, heights, depths, signeds);
  } catch (...) {
    return -3;
  }
  ctx->dirty = false;
  return 0;
}

// Viewport-region decode: decodes only the visible portion of the canvas
// and outputs it at 1:1 to the output buffer.  The region is specified
// in canvas coordinates (before reduce_NL scaling).
EMSCRIPTEN_KEEPALIVE
int jpip_end_frame_region(void *handle, uint8_t *rgb_out, int out_w, int out_h,
                          int region_x, int region_y, int region_w, int region_h) {
  if (!handle || !rgb_out || out_w <= 0 || out_h <= 0) return -1;
  if (region_w <= 0 || region_h <= 0) return -1;
  auto *ctx = static_cast<JpipContext *>(handle);

  // Only rebuild the sparse codestream when new data-bins arrived.  When the
  // user pans within cached precincts (dirty==false), ctx->sparse_cs is still
  // valid and re-parsing the main header via the decoder's reuse cache is a
  // no-op on the cache path.
  if (ctx->dirty || ctx->sparse_cs.empty()) {
    ctx->sparse_cs.clear();
    auto rc = open_htj2k::jpip::reassemble_codestream_client(ctx->set, *ctx->idx, ctx->sparse_cs);
    if (rc != open_htj2k::jpip::ReassembleStatus::Ok) return -2;
  }

  jpip_maybe_invalidate_reuse(ctx);
  auto &dec = ctx->dec;
  if (!ctx->dec_initialized) {
#ifdef OPENHTJ2K_THREAD
    dec.init(ctx->sparse_cs.data(), ctx->sparse_cs.size(), ctx->reduce_NL, 0);
#else
    dec.init(ctx->sparse_cs.data(), ctx->sparse_cs.size(), ctx->reduce_NL, 1);
#endif
    ctx->dec_initialized = true;
  } else {
    dec.init(ctx->sparse_cs.data(), ctx->sparse_cs.size(), ctx->reduce_NL, 1);
  }
  ctx->last_reduce_NL = ctx->reduce_NL;
  dec.parse();

  std::vector<uint32_t> widths, heights;
  std::vector<uint8_t>  depths;
  std::vector<bool>     signeds;
  const uint32_t ow = static_cast<uint32_t>(out_w);
  const uint32_t oh = static_cast<uint32_t>(out_h);
  const uint32_t red = ctx->reduce_NL;

  // Canvas extent at the active reduce level — used to clamp the region
  // bounds so the row callback's column LUT is guaranteed in-range and
  // can drop its per-pixel `xc >= cw` guard.
  const uint32_t cw_at_red = (ctx->canvas_w + ((1u << red) - 1)) >> red;
  const uint32_t ch_at_red = (ctx->canvas_h + ((1u << red) - 1)) >> red;

  const uint32_t ry0_u = static_cast<uint32_t>(region_y) >> red;
  uint32_t ry1 = (static_cast<uint32_t>(region_y + region_h) + ((1u << red) - 1)) >> red;
  const uint32_t rx0_u = static_cast<uint32_t>(region_x) >> red;
  uint32_t rx1 = (static_cast<uint32_t>(region_x + region_w) + ((1u << red) - 1)) >> red;
  if (ry1 > ch_at_red) ry1 = ch_at_red;
  if (rx1 > cw_at_red) rx1 = cw_at_red;
  const uint32_t ry0 = ry0_u;
  const uint32_t rx0 = rx0_u;
  if (rx1 <= rx0 || ry1 <= ry0) return -1;
  const uint32_t rw = rx1 - rx0;
  const uint32_t rh = ry1 - ry0;

  // The row callback below writes every output pixel when it fires for
  // every y in [ry0, ry1) — which it does on the success path.  The
  // memset is only a safety net for partial-region panning where not
  // every output row is written.  For the common full-canvas case
  // (viewer at fit-zoom, initial load) the entire buffer is overwritten,
  // so skip the O(ow·oh·4) wipe.
  const bool full_coverage =
      (region_x == 0 && region_y == 0
       && static_cast<uint32_t>(region_x + region_w) >= ctx->canvas_w
       && static_cast<uint32_t>(region_y + region_h) >= ctx->canvas_h);
  if (!full_coverage)
    std::memset(rgb_out, 0, static_cast<size_t>(ow) * oh * 4);

  // Precompute the column LUT upfront — xc depends only on (rx0, rw, ow),
  // all of which are known before the decoder starts emitting rows.
  std::vector<uint32_t> xc_lut(ow);
  for (uint32_t xw = 0; xw < ow; ++xw) {
    xc_lut[xw] = rx0 + static_cast<uint32_t>(static_cast<uint64_t>(xw) * rw / ow);
  }

  dec.set_row_limit(ry1);
  dec.set_col_range(rx0, rx1);
  try {
    dec.invoke_line_based_stream_reuse(
        [&](uint32_t y, int32_t *const *rows, uint16_t nc) {
          if (nc < 3 || widths.empty() || heights.empty()) return;
          if (y < ry0 || y >= ry1) return;
          const int32_t shift = (depths.empty() ? 0 : static_cast<int32_t>(depths[0]) - 8);
          const int32_t shift_pos = shift > 0 ? shift : 0;  // no-op when bit-depth <= 8
          const uint32_t ty = static_cast<uint32_t>(static_cast<uint64_t>(y - ry0) * oh / rh);
          if (ty >= oh) return;
          uint8_t *dst = rgb_out + static_cast<size_t>(ty) * ow * 4;
          const int32_t *__restrict__ r0 = rows[0];
          const int32_t *__restrict__ r1 = rows[1];
          const int32_t *__restrict__ r2 = rows[2];
          const uint32_t *__restrict__ lut = xc_lut.data();
          for (uint32_t xw = 0; xw < ow; ++xw) {
            const uint32_t xc = lut[xw];
            int32_t v0 = r0[xc] >> shift_pos;
            int32_t v1 = r1[xc] >> shift_pos;
            int32_t v2 = r2[xc] >> shift_pos;
            if (v0 < 0) v0 = 0; else if (v0 > 255) v0 = 255;
            if (v1 < 0) v1 = 0; else if (v1 > 255) v1 = 255;
            if (v2 < 0) v2 = 0; else if (v2 > 255) v2 = 255;
            dst[4 * xw + 0] = static_cast<uint8_t>(v0);
            dst[4 * xw + 1] = static_cast<uint8_t>(v1);
            dst[4 * xw + 2] = static_cast<uint8_t>(v2);
            dst[4 * xw + 3] = 255;
          }
          const uint32_t ty1_out = static_cast<uint32_t>(static_cast<uint64_t>(y - ry0 + 1) * oh / rh);
          for (uint32_t ty2 = ty + 1; ty2 < ty1_out && ty2 < oh; ++ty2)
            std::memcpy(rgb_out + static_cast<size_t>(ty2) * ow * 4, dst, ow * 4);
        },
        widths, heights, depths, signeds);
  } catch (...) {
    return -3;
  }
  ctx->dirty = false;
  return 0;
}

EMSCRIPTEN_KEEPALIVE
void jpip_destroy_context(void *handle) {
  delete static_cast<JpipContext *>(handle);
}

}  // extern "C"
#endif  // __EMSCRIPTEN__
