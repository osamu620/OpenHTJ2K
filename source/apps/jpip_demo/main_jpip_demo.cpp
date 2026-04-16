// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.
//
// JPIP mouse-driven foveation demo.
//
// Loads a JPEG 2000 codestream, builds a CodestreamIndex once, opens a
// window, and redecodes the image every frame with a JPIP precinct filter
// that picks precincts by concentric cones around the mouse cursor:
//
//   fovea     : fsiz = canvas             → all resolutions, tight RoI
//   parafovea : fsiz = canvas / 2         → drop the finest resolution, wider RoI
//   periphery : fsiz = canvas / 4         → drop the top two resolutions, whole image
//
// The unioned precinct set becomes the decoder's set_precinct_filter, the
// decoder streams rows via invoke_line_based_stream(), each row is
// nearest-neighbour downsampled into a fixed-size RGB buffer, and that
// buffer is uploaded to the rtp_recv renderer (Metal on macOS, OpenGL 3.3
// elsewhere).
//
// Decoupling the window/texture size from the canvas size lets the demo
// run on canvases that exceed the GPU texture limit (Metal: 16384 wide on
// Apple silicon) — including the 21600 × 10800 NASA Blue Marble.  Peak RSS
// is proportional to the ring-buffer depth rather than canvas W × H.
//
// Usage:
//   open_htj2k_jpip_demo <input.j2c>
//       [--fovea-radius N=256] [--parafovea-radius N=512]
//       [--window-size WxH=1920x1080]
//       [--decode-on-move-only] [--no-vsync]
//
// Exits on window close or ESC.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_set>
#include <vector>

#include <GLFW/glfw3.h>

#include "decoder.hpp"
#include "precinct_index.hpp"
#include "view_window.hpp"
#include "renderer.hpp"

using open_htj2k::rtp_recv::Renderer;
using open_htj2k::jpip::CodestreamIndex;
using open_htj2k::jpip::PrecinctKey;
using open_htj2k::jpip::ViewWindow;

namespace {

struct Options {
  std::string infile;
  uint32_t    fovea_radius     = 256;
  uint32_t    parafovea_radius = 512;
  // Window/texture dimensions, decoupled from canvas size.  The decoder still
  // operates at canvas resolution; the row-callback downsamples into a
  // window-sized RGB buffer before upload.  Default 1920×1080 fits the Metal
  // 16384 texture limit for arbitrarily large canvases (the original demo
  // used canvas size directly and aborted on the 21600-wide NASA Blue Marble
  // asset because Metal rejects MTLTextureDescriptor.width > 16384).
  uint32_t    window_w         = 1920;
  uint32_t    window_h         = 1080;
  bool        decode_on_move   = false;
  bool        vsync            = true;
};

bool parse_args(int argc, char **argv, Options &opt) {
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--fovea-radius" && i + 1 < argc)      opt.fovea_radius = static_cast<uint32_t>(std::stoul(argv[++i]));
    else if (a == "--parafovea-radius" && i + 1 < argc) opt.parafovea_radius = static_cast<uint32_t>(std::stoul(argv[++i]));
    else if (a == "--window-size" && i + 1 < argc) {
      // "WxH" or "W,H".
      const std::string s = argv[++i];
      const auto sep = s.find_first_of("x,");
      if (sep == std::string::npos) {
        std::fprintf(stderr, "ERROR: --window-size expects WxH or W,H, got '%s'\n", s.c_str());
        return false;
      }
      opt.window_w = static_cast<uint32_t>(std::stoul(s.substr(0, sep)));
      opt.window_h = static_cast<uint32_t>(std::stoul(s.substr(sep + 1)));
    }
    else if (a == "--decode-on-move-only")          opt.decode_on_move = true;
    else if (a == "--no-vsync")                     opt.vsync = false;
    else if (a.size() > 0 && a[0] != '-' && opt.infile.empty()) opt.infile = a;
    else {
      std::fprintf(stderr, "ERROR: unknown arg '%s'\n", a.c_str());
      return false;
    }
  }
  return !opt.infile.empty();
}

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

// Build a ViewWindow describing `radius` canvas-coordinate samples around
// (gx, gy) at the discard level corresponding to `fsiz_ratio` (1.0 = full
// canvas / all resolutions, 0.5 = half resolution, 0.25 = quarter, …).
// A fsiz_ratio of 0 means "whole image at that resolution" (periphery).
ViewWindow make_view_window(const CodestreamIndex &idx, uint32_t gx, uint32_t gy,
                            uint32_t radius, float fsiz_ratio, bool whole_image) {
  const auto &g = idx.geometry();
  ViewWindow vw;
  vw.fx = static_cast<uint32_t>(static_cast<float>(g.canvas_size.x) * fsiz_ratio);
  vw.fy = static_cast<uint32_t>(static_cast<float>(g.canvas_size.y) * fsiz_ratio);
  if (vw.fx == 0) vw.fx = 1;
  if (vw.fy == 0) vw.fy = 1;
  if (whole_image) {
    vw.ox = 0;
    vw.oy = 0;
    vw.sx = vw.fx;
    vw.sy = vw.fy;
  } else {
    // Scale (gx, gy, radius) from canvas to the fsiz grid.
    const uint32_t gx_f = static_cast<uint32_t>(static_cast<uint64_t>(gx) * vw.fx / g.canvas_size.x);
    const uint32_t gy_f = static_cast<uint32_t>(static_cast<uint64_t>(gy) * vw.fy / g.canvas_size.y);
    const uint32_t r_f  = static_cast<uint32_t>(static_cast<uint64_t>(radius) * vw.fx / g.canvas_size.x);
    vw.ox = (gx_f > r_f) ? (gx_f - r_f) : 0u;
    vw.oy = (gy_f > r_f) ? (gy_f - r_f) : 0u;
    vw.sx = 2u * r_f;
    vw.sy = 2u * r_f;
  }
  return vw;
}

// Build an I-indexed hash set from three concentric view-window calls.
std::unordered_set<uint64_t> foveated_i_set(const CodestreamIndex &idx,
                                            uint32_t gx, uint32_t gy,
                                            const Options &opt) {
  std::unordered_set<uint64_t> out;
  auto add = [&](const std::vector<PrecinctKey> &keys) {
    out.reserve(out.size() + keys.size());
    for (const auto &k : keys) out.insert(idx.I(k.t, k.c, k.r, k.p_rc));
  };

  // Fovea: full resolution, tight RoI centred on gaze.
  auto vw_f = make_view_window(idx, gx, gy, opt.fovea_radius,     1.00f, false);
  add(open_htj2k::jpip::resolve_view_window(idx, vw_f));

  // Parafovea: half resolution, wider RoI.
  auto vw_p = make_view_window(idx, gx, gy, opt.parafovea_radius, 0.50f, false);
  add(open_htj2k::jpip::resolve_view_window(idx, vw_p));

  // Periphery: quarter resolution, whole image (covers everything at coarse detail).
  auto vw_q = make_view_window(idx, gx, gy, 0, 0.25f, true);
  add(open_htj2k::jpip::resolve_view_window(idx, vw_q));

  return out;
}

}  // namespace

int main(int argc, char **argv) {
  Options opt;
  if (!parse_args(argc, argv, opt)) {
    std::fprintf(stderr,
                 "Usage: open_htj2k_jpip_demo <input.j2c> "
                 "[--fovea-radius N] [--parafovea-radius N] "
                 "[--decode-on-move-only] [--no-vsync]\n");
    return EXIT_FAILURE;
  }

  auto bytes = read_file(opt.infile.c_str());
  if (bytes.empty()) return EXIT_FAILURE;

  std::unique_ptr<CodestreamIndex> idx;
  try {
    idx = CodestreamIndex::build(bytes.data(), bytes.size());
  } catch (std::exception &e) {
    std::fprintf(stderr, "CodestreamIndex build failed: %s\n", e.what());
    return EXIT_FAILURE;
  }
  const uint32_t canvas_w = idx->geometry().canvas_size.x;
  const uint32_t canvas_h = idx->geometry().canvas_size.y;
  const uint64_t total_p  = idx->total_precincts();
  std::printf("loaded %s: canvas %u×%u, %u components, %llu precincts\n",
              opt.infile.c_str(), canvas_w, canvas_h, idx->num_components(),
              static_cast<unsigned long long>(total_p));

  // The decoder's codestream cursor advances during each invoke(), so every
  // frame constructs a fresh openhtj2k_decoder (see the per-frame init()+
  // parse() block below).  The codestream buffer `bytes` lives for the
  // program lifetime, so the re-init is zero-copy and dominated by marker
  // parsing — sub-millisecond for a single-tile stream.

  // Clamp the requested window to a known-safe upper bound so we never feed
  // the renderer something that violates its texture size limit (Metal: 16384
  // on Apple silicon; GL: implementation-defined but typically ≥ 16384 too).
  if (opt.window_w == 0) opt.window_w = canvas_w;
  if (opt.window_h == 0) opt.window_h = canvas_h;
  constexpr uint32_t kMaxTex = 16384;
  if (opt.window_w > kMaxTex) opt.window_w = kMaxTex;
  if (opt.window_h > kMaxTex) opt.window_h = kMaxTex;

  Renderer renderer;
  if (!renderer.init(static_cast<int>(opt.window_w), static_cast<int>(opt.window_h),
                     "OpenHTJ2K JPIP Foveation Demo", opt.vsync)) {
    std::fprintf(stderr, "FATAL: renderer.init failed\n");
    return EXIT_FAILURE;
  }
  GLFWwindow *window = renderer.get_window();
  std::printf("window %u×%u  (downsample factor %.2f×%.2f from canvas)\n",
              opt.window_w, opt.window_h,
              static_cast<double>(canvas_w) / opt.window_w,
              static_cast<double>(canvas_h) / opt.window_h);

  std::vector<uint8_t> rgb;
  uint64_t frames = 0;
  int32_t  last_gx = -1, last_gy = -1;

  using Clock = std::chrono::steady_clock;
  auto last_log = Clock::now();
  uint64_t frames_since_log = 0;
  uint64_t precincts_since_log = 0;

  while (!renderer.should_close()) {
    renderer.poll_events();

    double mx = 0.0, my = 0.0;
    glfwGetCursorPos(window, &mx, &my);
    int win_w = 1, win_h = 1;
    glfwGetWindowSize(window, &win_w, &win_h);
    if (win_w < 1) win_w = 1;
    if (win_h < 1) win_h = 1;
    double rx = std::max(0.0, std::min(1.0, mx / win_w));
    double ry = std::max(0.0, std::min(1.0, my / win_h));
    uint32_t gx = static_cast<uint32_t>(rx * (canvas_w - 1));
    uint32_t gy = static_cast<uint32_t>(ry * (canvas_h - 1));

    if (opt.decode_on_move && static_cast<int32_t>(gx) == last_gx
        && static_cast<int32_t>(gy) == last_gy && frames > 0) {
      continue;
    }
    last_gx = static_cast<int32_t>(gx);
    last_gy = static_cast<int32_t>(gy);

    auto keep = foveated_i_set(*idx, gx, gy, opt);
    precincts_since_log += keep.size();

    // Fresh decoder per frame — init() + parse() rewind the codestream cursor.
    open_htj2k::openhtj2k_decoder dec;
    dec.init(bytes.data(), bytes.size(), /*reduce_NL=*/0, /*num_threads=*/1);
    dec.parse();

    auto *idx_ptr = idx.get();
    dec.set_precinct_filter(
        [idx_ptr, keep_moved = std::move(keep)](
            uint16_t t, uint16_t c, uint8_t r, uint32_t p_rc) {
          return keep_moved.count(idx_ptr->I(t, c, r, p_rc)) > 0;
        });

    // Line-based stream decode + on-the-fly nearest-neighbour downsample
    // into the window-sized RGB buffer.  Peak RSS is now proportional to
    // (ring depth × canvas_w) instead of (canvas_w × canvas_h) — the 233 MP
    // NASA Blue Marble crop drops from ~10 GB to a few hundred MB.
    if (rgb.size() != static_cast<std::size_t>(opt.window_w) * opt.window_h * 3u) {
      rgb.assign(static_cast<std::size_t>(opt.window_w) * opt.window_h * 3u, 0);
    }
    std::vector<uint32_t> w, h;
    std::vector<uint8_t>  depth;
    std::vector<bool>     sgn;
    // Track which window rows have been written this frame so we don't
    // overwrite a row when several canvas rows map to it (first canvas row
    // in each window-row bucket wins — nearest-neighbour vertical downsample).
    static thread_local std::vector<uint8_t> row_written;
    row_written.assign(opt.window_h, 0);
    bool ok      = true;
    bool dims_ok = true;
    try {
      dec.invoke_line_based_stream(
          [&](uint32_t y, int32_t *const *rows, uint16_t nc) {
            if (nc < 3 || w.empty() || h.empty()) { dims_ok = false; return; }
            const uint32_t cw = w[0];
            const uint32_t ch = h[0];
            // Map this canvas row to a window row.  Floor-div is fine for
            // nearest-neighbour; any canvas row whose mapped target_y has
            // already been written is dropped.
            const uint32_t target_y =
                static_cast<uint32_t>(static_cast<uint64_t>(y) * opt.window_h / std::max(1u, ch));
            if (target_y >= opt.window_h || row_written[target_y]) return;
            row_written[target_y] = 1;
            uint8_t *dst = rgb.data() + static_cast<std::size_t>(target_y) * opt.window_w * 3u;
            for (uint32_t x_w = 0; x_w < opt.window_w; ++x_w) {
              const uint32_t x_c =
                  static_cast<uint32_t>(static_cast<uint64_t>(x_w) * cw / std::max(1u, opt.window_w));
              auto clamp_u8 = [](int32_t v) -> uint8_t {
                if (v < 0) return 0;
                if (v > 255) return 255;
                return static_cast<uint8_t>(v);
              };
              dst[3u * x_w + 0] = clamp_u8(rows[0][x_c]);
              dst[3u * x_w + 1] = clamp_u8(rows[1][x_c]);
              dst[3u * x_w + 2] = clamp_u8(rows[2][x_c]);
            }
          },
          w, h, depth, sgn);
    } catch (std::exception &e) {
      std::fprintf(stderr, "decode failed: %s\n", e.what());
      break;
    }
    if (!dims_ok) ok = false;

    if (ok) renderer.upload_and_draw(rgb.data(), static_cast<int>(opt.window_w),
                                     static_cast<int>(opt.window_h));

    ++frames;
    ++frames_since_log;
    const auto now = Clock::now();
    if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_log).count() > 1000) {
      const double secs = std::chrono::duration<double>(now - last_log).count();
      const double fps  = frames_since_log / std::max(secs, 1e-6);
      const uint64_t avg_precincts = frames_since_log ? (precincts_since_log / frames_since_log) : 0;
      const double coverage_pct =
          total_p ? (100.0 * static_cast<double>(avg_precincts) / static_cast<double>(total_p)) : 0.0;
      std::printf("gaze=(%u,%u) precincts=%llu (%.1f%% of %llu)  %.1f fps\n",
                  gx, gy, static_cast<unsigned long long>(avg_precincts), coverage_pct,
                  static_cast<unsigned long long>(total_p), fps);
      std::fflush(stdout);
      last_log            = now;
      frames_since_log    = 0;
      precincts_since_log = 0;
    }
  }

  renderer.shutdown();
  return EXIT_SUCCESS;
}
