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
// By default (Phase 2 JPP round-trip mode), each frame's foveated precinct
// set is serialised to a JPP-stream, parsed back into a DataBinSet,
// reassembled into a sparse J2C codestream, and decoded.  The --use-filter
// flag falls back to the Phase-1 direct set_precinct_filter path for A/B
// performance comparison.  Both paths produce visually identical output;
// the JPP path exercises every byte of the JPIP wire format.
//
// Decoupling the window/texture size from the canvas size lets the demo
// run on canvases that exceed the GPU texture limit (Metal: 16384 wide on
// Apple silicon) — including the 21600 × 10800 NASA Blue Marble.  Peak RSS
// is proportional to the ring-buffer depth rather than canvas W × H.
//
// Usage:
//   open_htj2k_jpip_demo <input.j2c>
//       [--fovea-radius N]          (canvas px; default = canvas_w / 15)
//       [--parafovea-radius N]      (canvas px; default = canvas_w / 8)
//       [--parafovea-ratio F=0.5]   (fsiz ratio; lower = coarser)
//       [--periphery-ratio F=0.125] (fsiz ratio; default drops 3 of 5 DWT levels)
//       [--window-size WxH=1920x1080]
//       [--reduce N=0]             (DWT reduce levels; trades resolution for speed)
//       [--server-h3 host:port]     (HTTP/3 QUIC server mode)
//       [--use-filter]              (Phase-1 direct filter, skip JPP round-trip)
//       [--decode-on-move-only] [--no-vsync]
//
// Exits on window close or ESC.

#include <algorithm>
#include <chrono>
#include <future>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_set>
#include <vector>

#include <GLFW/glfw3.h>

#include "codestream_assembler.hpp"
#include "codestream_walker.hpp"
#include "data_bin_emitter.hpp"
#include "decoder.hpp"
#include "cache_model.hpp"
#include "jpip_client.hpp"
#include "jpp_parser.hpp"
#include "packet_locator.hpp"
#include "precinct_index.hpp"
#include "view_window.hpp"
#ifdef OPENHTJ2K_ENABLE_QUIC
#include "h3_client.hpp"
#endif
#include "renderer.hpp"

using open_htj2k::rtp_recv::Renderer;
using open_htj2k::jpip::CodestreamIndex;
using open_htj2k::jpip::PrecinctKey;
using open_htj2k::jpip::ViewWindow;

namespace {

void merge_headers_only(open_htj2k::jpip::DataBinSet &dst,
                        const open_htj2k::jpip::DataBinSet &src) {
  for (const auto &kv : src.keys()) {
    if (kv.first == open_htj2k::jpip::kMsgClassPrecinct ||
        kv.first == open_htj2k::jpip::kMsgClassExtPrecinct)
      continue;
    const auto &data = src.get(kv.first, kv.second);
    dst.append(kv.first, kv.second, 0, data.data(), data.size(),
               src.is_complete(kv.first, kv.second));
  }
}

struct Options {
  std::string infile;
  // Foveation cone radii in canvas pixels.  0 = auto-scale from canvas
  // width (fovea = W/15, parafovea = W/8) so the demo "just works" at
  // any resolution without manually tuning radii.
  uint32_t    fovea_radius     = 0;
  uint32_t    parafovea_radius = 0;
  // fsiz ratios for the parafovea and periphery cones — control how many
  // DWT resolutions are kept.  0.5 → r*=1 (drop the finest), 0.25 → r*=2
  // (drop the top two), 0.125 → r*=3 (drop the top three).  Lower values
  // produce a more dramatic quality drop outside the fovea.
  float       parafovea_ratio  = 0.5f;
  float       periphery_ratio  = 0.125f;
  uint32_t    window_w         = 1920;
  uint32_t    window_h         = 1080;
  bool        decode_on_move   = false;
  bool        vsync            = true;
  uint8_t     reduce           = 0;
  bool        dual_res         = false;
  bool        use_filter       = false;
  // When non-empty, the demo fetches each frame's JPP-stream from the
  // given JPIP server instead of doing in-process round-trip.
  std::string server_host;
  uint16_t    server_port      = 8080;
  std::string server_h3_host;
  uint16_t    server_h3_port   = 8080;
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
    else if (a == "--parafovea-ratio" && i + 1 < argc) opt.parafovea_ratio = std::stof(argv[++i]);
    else if (a == "--periphery-ratio" && i + 1 < argc) opt.periphery_ratio = std::stof(argv[++i]);
    else if (a == "--reduce" && i + 1 < argc)        opt.reduce = static_cast<uint8_t>(std::stoul(argv[++i]));
    else if (a == "--dual-res")                      opt.dual_res = true;
    else if (a == "--decode-on-move-only")          opt.decode_on_move = true;
    else if (a == "--use-filter")                   opt.use_filter = true;
    else if (a == "--no-vsync")                     opt.vsync = false;
    else if (a == "--server" && i + 1 < argc) {
      const std::string hp = argv[++i];
      const auto colon = hp.rfind(':');
      if (colon == std::string::npos) {
        opt.server_host = hp;
      } else {
        opt.server_host = hp.substr(0, colon);
        opt.server_port = static_cast<uint16_t>(std::stoul(hp.substr(colon + 1)));
      }
    }
    else if (a == "--server-h3" && i + 1 < argc) {
      const std::string hp = argv[++i];
      const auto colon = hp.rfind(':');
      if (colon == std::string::npos) {
        opt.server_h3_host = hp;
      } else {
        opt.server_h3_host = hp.substr(0, colon);
        opt.server_h3_port = static_cast<uint16_t>(std::stoul(hp.substr(colon + 1)));
      }
    }
    else if (a.size() > 0 && a[0] != '-' && opt.infile.empty()) opt.infile = a;
    else {
      std::fprintf(stderr, "ERROR: unknown arg '%s'\n", a.c_str());
      return false;
    }
  }
  return !opt.infile.empty() || !opt.server_host.empty() || !opt.server_h3_host.empty();
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
  auto vw_f = make_view_window(idx, gx, gy, opt.fovea_radius, 1.00f, false);
  add(open_htj2k::jpip::resolve_view_window(idx, vw_f));

  // Parafovea: reduced resolution, wider RoI.
  auto vw_p = make_view_window(idx, gx, gy, opt.parafovea_radius,
                               opt.parafovea_ratio, false);
  add(open_htj2k::jpip::resolve_view_window(idx, vw_p));

  // Periphery: aggressively reduced resolution, whole image.
  auto vw_q = make_view_window(idx, gx, gy, 0, opt.periphery_ratio, true);
  add(open_htj2k::jpip::resolve_view_window(idx, vw_q));

  return out;
}

}  // namespace

int main(int argc, char **argv) {
  Options opt;
  if (!parse_args(argc, argv, opt)) {
    std::fprintf(stderr,
                 "Usage: open_htj2k_jpip_demo [<input.j2c>] [options]\n"
                 "\n"
                 "Options:\n"
                 "  --server host:port        Fetch from a JPIP server (no local file needed)\n"
                 "  --fovea-radius N          Fovea radius in canvas px (default: auto)\n"
                 "  --parafovea-radius N      Parafovea radius in canvas px (default: auto)\n"
                 "  --parafovea-ratio F       fsiz ratio for parafovea (default: 0.5)\n"
                 "  --periphery-ratio F       fsiz ratio for periphery (default: 0.125)\n"
                 "  --window-size WxH         Window/texture size (default: 1920x1080)\n"
                 "  --use-filter              Phase-1 direct filter (skip JPP round-trip)\n"
                 "  --decode-on-move-only     Skip redecode when cursor is stationary\n"
                 "  --no-vsync                Unlock frame rate\n"
                 "\n"
                 "When --server is given, no local codestream file is needed — the demo\n"
                 "fetches everything from the server.  Without --server, <input.j2c> is\n"
                 "required for the in-process JPP round-trip or --use-filter path.\n");
    return EXIT_FAILURE;
  }

  // In --server mode, the local codestream is optional.
  std::vector<uint8_t> bytes;
  if (!opt.infile.empty()) {
    bytes = read_file(opt.infile.c_str());
    if (bytes.empty()) return EXIT_FAILURE;
  } else if (opt.server_host.empty() && opt.server_h3_host.empty()) {
    std::fprintf(stderr, "ERROR: either <input.j2c> or --server/--server-h3 host:port is required\n");
    return EXIT_FAILURE;
  }

  std::unique_ptr<CodestreamIndex> idx;
  if (!bytes.empty()) {
    try {
      idx = CodestreamIndex::build(bytes.data(), bytes.size());
    } catch (std::exception &e) {
      std::fprintf(stderr, "CodestreamIndex build failed: %s\n", e.what());
      return EXIT_FAILURE;
    }
  }
#ifdef OPENHTJ2K_ENABLE_QUIC
  else if (!opt.server_h3_host.empty()) {
    // H3 server-only mode: fetch initial main-header over HTTP/3.
    open_htj2k::jpip::H3Client h3_init;
    if (!h3_init.connect(opt.server_h3_host, opt.server_h3_port, false)) {
      std::fprintf(stderr, "H3 initial connect: %s\n", h3_init.last_error().c_str());
      return EXIT_FAILURE;
    }
    auto init_body = h3_init.fetch("/jpip?fsiz=1,1&type=jpp-stream");
    if (init_body.empty()) {
      std::fprintf(stderr, "H3 initial fetch empty\n");
      return EXIT_FAILURE;
    }
    open_htj2k::jpip::DataBinSet init_set;
    open_htj2k::jpip::parse_jpp_stream(init_body.data(), init_body.size(), &init_set);
    const auto &mh_bin = init_set.get(open_htj2k::jpip::kMsgClassMainHeader, 0);
    if (mh_bin.empty()) {
      std::fprintf(stderr, "H3 server response missing main-header data-bin\n");
      return EXIT_FAILURE;
    }
    idx = CodestreamIndex::build_from_main_header_bin(mh_bin);
    if (!idx) {
      std::fprintf(stderr, "CodestreamIndex build from H3 main-header bin failed\n");
      return EXIT_FAILURE;
    }
    std::printf("built index from H3 server's main-header data-bin\n");
  }
#endif
  else {
    // HTTP/1.1 server-only mode: fetch an initial full-image request to get
    // the main-header data-bin, then build the index from it.
    open_htj2k::jpip::JpipClient client;
    open_htj2k::jpip::DataBinSet init_set;
    open_htj2k::jpip::ViewWindow init_vw;
    init_vw.fx = 1; init_vw.fy = 1;  // minimal fsiz to get headers only
    if (!client.fetch(opt.server_host, opt.server_port, init_vw, &init_set)) {
      std::fprintf(stderr, "initial server fetch: %s\n", client.last_error().c_str());
      return EXIT_FAILURE;
    }
    const auto &mh_bin = init_set.get(open_htj2k::jpip::kMsgClassMainHeader, 0);
    if (mh_bin.empty()) {
      std::fprintf(stderr, "server response missing main-header data-bin\n");
      return EXIT_FAILURE;
    }
    idx = CodestreamIndex::build_from_main_header_bin(mh_bin);
    if (!idx) {
      std::fprintf(stderr, "CodestreamIndex build from main-header bin failed\n");
      return EXIT_FAILURE;
    }
    std::printf("built index from server's main-header data-bin\n");
  }
  const uint32_t canvas_w = idx->geometry().canvas_size.x;
  const uint32_t canvas_h = idx->geometry().canvas_size.y;
  const uint64_t total_p  = idx->total_precincts();

  // Auto-scale foveation radii if not explicitly set (0 = auto).  The
  // defaults produce a proportionally-sized fovea regardless of canvas
  // resolution — about 1/15 of the canvas width for the fovea and 1/8
  // for the parafovea.
  if (opt.fovea_radius == 0)     opt.fovea_radius     = std::max(16u, canvas_w / 15u);
  if (opt.parafovea_radius == 0) opt.parafovea_radius = std::max(32u, canvas_w / 8u);
  std::printf("foveation: fovea=%u  parafovea=%u  ratios=%.3f/%.3f (canvas px)\n",
              opt.fovea_radius, opt.parafovea_radius,
              static_cast<double>(opt.parafovea_ratio),
              static_cast<double>(opt.periphery_ratio));
  std::printf("loaded %s: canvas %u×%u, %u components, %llu precincts\n",
              opt.infile.empty() ? "(server)" : opt.infile.c_str(), canvas_w, canvas_h, idx->num_components(),
              static_cast<unsigned long long>(total_p));

  // The decoder's codestream cursor advances during each invoke(), so every
  // frame constructs a fresh openhtj2k_decoder (see the per-frame init()+
  // parse() block below).  The codestream buffer `bytes` lives for the
  // program lifetime, so the re-init is zero-copy and dominated by marker
  // parsing — sub-millisecond for a single-tile stream.

  // JPP round-trip mode (default) needs a one-time PacketLocator build.
  // This drives the decoder once with a packet observer to learn per-
  // precinct byte ranges; on the 1920×1920 asset it costs ~25 ms.
  // Build the packet locator + layout only for in-process modes (not for
  // server mode — the client-side reassembler doesn't need them).
  open_htj2k::jpip::CodestreamLayout layout;
  std::unique_ptr<open_htj2k::jpip::PacketLocator> locator;
  if (!opt.server_host.empty()) {
    std::printf("network mode: server %s:%u (no local codestream needed)\n",
                opt.server_host.c_str(), opt.server_port);
  } else if (!opt.use_filter && !bytes.empty()) {
    open_htj2k::jpip::walk_codestream(bytes.data(), bytes.size(), &layout);
    locator = open_htj2k::jpip::PacketLocator::build(bytes.data(), bytes.size(), *idx, layout);
    if (!locator) {
      std::fprintf(stderr, "WARN: PacketLocator build failed; falling back to --use-filter\n");
      opt.use_filter = true;
    }
    if (!opt.use_filter) {
      const uint8_t po = idx->progression_order();
      if (po == 0 || po == 1) {
        std::fprintf(stderr,
                     "WARN: progression order %u (LRCP/RLCP) not supported by "
                     "JPP reassembler; falling back to --use-filter\n", po);
        opt.use_filter = true;
      }
    }
    if (!opt.use_filter) {
      std::printf("JPP round-trip mode enabled (locator: %zu packet ranges)\n",
                  locator->size());
    }
  }
  if (opt.use_filter) {
    std::printf("direct-filter mode enabled\n");
  }

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
  open_htj2k::jpip::CacheModel client_cache;
  open_htj2k::jpip::DataBinSet header_cache;  // persistent: main header + tile headers + metadata
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

    // Build the view-window for this frame's gaze position.
    // For --server mode we send the view-window directly to the server;
    // for in-process modes we use the foveated I-set.
    open_htj2k::jpip::ViewWindow frame_vw;
    {
      auto vw_f = make_view_window(*idx, gx, gy, opt.fovea_radius, 1.00f, false);
      frame_vw = vw_f;  // use the fovea's full-res fsiz for the server request
      // For the server, we send one request with the fovea's fsiz; the
      // server resolves view-window → precincts.  For in-process, we
      // still build the three-cone union ourselves.
    }

    auto keep = foveated_i_set(*idx, gx, gy, opt);
    precincts_since_log += keep.size();

    // Build the codestream to decode this frame — via network, in-process
    // JPP round-trip, or the legacy precinct-filter path.
    std::vector<uint8_t> frame_cs;
    if (!opt.server_host.empty()) {
      // ── Network path: 3 concurrent JPIP requests (pipelined) ──
      // Each request is a standard-compliant view-window query sent on
      // its own TCP connection.  Running them concurrently reduces the
      // total latency from sum(3 RTTs) to max(3 RTTs).
      auto vw_fov  = make_view_window(*idx, gx, gy, opt.fovea_radius, 1.00f, false);
      auto vw_para = make_view_window(*idx, gx, gy, opt.parafovea_radius, opt.parafovea_ratio, false);
      auto vw_peri = make_view_window(*idx, gx, gy, 0, opt.periphery_ratio, true);

      auto do_fetch = [&](const open_htj2k::jpip::ViewWindow &vw) {
        open_htj2k::jpip::JpipClient c;
        open_htj2k::jpip::DataBinSet s;
        c.fetch(opt.server_host, opt.server_port, vw, &s, &client_cache);
        return s;
      };

      auto f1 = std::async(std::launch::async, do_fetch, vw_fov);
      auto f2 = std::async(std::launch::async, do_fetch, vw_para);
      auto f3 = std::async(std::launch::async, do_fetch, vw_peri);

      open_htj2k::jpip::DataBinSet set = f1.get();
      set.merge_from(f2.get());
      set.merge_from(f3.get());
      // Cache headers persistently; update model for all received bins.
      merge_headers_only(header_cache, set);
      for (const auto &kv : set.keys()) {
        if (set.is_complete(kv.first, kv.second) && kv.first != open_htj2k::jpip::kMsgClassPrecinct)
          client_cache.mark(kv.first, kv.second);
      }
      // Reassemble from headers (cached) + this frame's precincts.
      // Don't use accumulated precincts — that would erase foveation.
      open_htj2k::jpip::DataBinSet frame_set;
      frame_set.merge_from(header_cache);
      frame_set.merge_from(set);
      const auto rc = open_htj2k::jpip::reassemble_codestream_client(frame_set, *idx, frame_cs);
      if (rc != open_htj2k::jpip::ReassembleStatus::Ok) {
        std::fprintf(stderr, "reassemble (client) failed status=%d\n", static_cast<int>(rc));
        break;
      }
    }
#ifdef OPENHTJ2K_ENABLE_QUIC
    else if (!opt.server_h3_host.empty()) {
      // ── HTTP/3 network path ──
      static open_htj2k::jpip::H3Client h3c;
      static bool h3_connected = false;
      if (!h3_connected) {
        if (!h3c.connect(opt.server_h3_host, opt.server_h3_port, false)) {
          std::fprintf(stderr, "H3 connect failed: %s\n", h3c.last_error().c_str());
          break;
        }
        h3_connected = true;
      }
      // Build 3 query paths and fetch concurrently on multiplexed QUIC streams.
      auto make_query = [&](const open_htj2k::jpip::ViewWindow &vw) {
        std::string q = "/jpip?" + open_htj2k::jpip::format_view_window_query(vw);
        if (client_cache.size() > 0) q += "&model=" + client_cache.format();
        return q;
      };
      std::vector<std::string> paths = {
        make_query(make_view_window(*idx, gx, gy, opt.fovea_radius, 1.00f, false)),
        make_query(make_view_window(*idx, gx, gy, opt.parafovea_radius, opt.parafovea_ratio, false)),
        make_query(make_view_window(*idx, gx, gy, 0, opt.periphery_ratio, true)),
      };
      auto bodies = h3c.fetch_multi(paths);

      open_htj2k::jpip::DataBinSet set;
      for (const auto &body : bodies) {
        if (!body.empty()) {
          open_htj2k::jpip::DataBinSet tmp;
          open_htj2k::jpip::parse_jpp_stream(body.data(), body.size(), &tmp);
          set.merge_from(tmp);
        }
      }
      merge_headers_only(header_cache, set);
      for (const auto &kv : set.keys()) {
        if (set.is_complete(kv.first, kv.second) && kv.first != open_htj2k::jpip::kMsgClassPrecinct)
          client_cache.mark(kv.first, kv.second);
      }

      open_htj2k::jpip::DataBinSet frame_set;
      frame_set.merge_from(header_cache);
      frame_set.merge_from(set);
      const auto rc = open_htj2k::jpip::reassemble_codestream_client(frame_set, *idx, frame_cs);
      if (rc != open_htj2k::jpip::ReassembleStatus::Ok) {
        std::fprintf(stderr, "reassemble (H3 client) failed status=%d\n", static_cast<int>(rc));
        break;
      }
    }
#endif
    else if (!opt.use_filter) {
      // ── In-process JPP round-trip: emit → parse → reassemble ──
      std::vector<uint8_t> stream;
      open_htj2k::jpip::MessageHeaderContext enc_ctx;
      open_htj2k::jpip::emit_main_header_databin(bytes.data(), bytes.size(), layout, enc_ctx, stream);
      for (uint32_t t = 0; t < idx->num_tiles(); ++t) {
        open_htj2k::jpip::emit_tile_header_databin(bytes.data(), bytes.size(),
                                                    static_cast<uint16_t>(t), layout, enc_ctx, stream);
      }
      open_htj2k::jpip::emit_metadata_bin_zero(enc_ctx, stream);
      for (uint32_t t = 0; t < idx->num_tiles(); ++t) {
        for (uint16_t c = 0; c < idx->num_components(); ++c) {
          const auto &info = idx->tile_component(static_cast<uint16_t>(t), c);
          for (uint8_t r = 0; r <= info.NL; ++r) {
            const uint32_t n = info.npw[r] * info.nph[r];
            for (uint32_t p = 0; p < n; ++p) {
              const uint64_t I = idx->I(static_cast<uint16_t>(t), c, r, p);
              if (keep.count(I)) {
                open_htj2k::jpip::emit_precinct_databin(
                    bytes.data(), bytes.size(),
                    static_cast<uint16_t>(t), c, r, p, *idx, *locator, enc_ctx, stream);
              }
            }
          }
        }
      }
      open_htj2k::jpip::DataBinSet set;
      open_htj2k::jpip::parse_jpp_stream(stream.data(), stream.size(), &set);
      const auto rc = open_htj2k::jpip::reassemble_codestream(
          bytes.data(), bytes.size(), set, *idx, layout, *locator, frame_cs);
      if (rc != open_htj2k::jpip::ReassembleStatus::Ok) {
        std::fprintf(stderr, "reassemble failed status=%d\n", static_cast<int>(rc));
        break;
      }
    }

    // Fresh decoder per frame — init() + parse() rewind the codestream cursor.
    open_htj2k::openhtj2k_decoder dec;
    const uint8_t *dec_buf = opt.use_filter ? bytes.data() : frame_cs.data();
    const std::size_t dec_len = opt.use_filter ? bytes.size() : frame_cs.size();
    dec.init(dec_buf, dec_len, opt.reduce, /*num_threads=*/1);
    dec.parse();

    if (opt.use_filter) {
      auto *idx_ptr = idx.get();
      dec.set_precinct_filter(
          [idx_ptr, keep_moved = std::move(keep)](
              uint16_t t, uint16_t c, uint8_t r, uint32_t p_rc) {
            return keep_moved.count(idx_ptr->I(t, c, r, p_rc)) > 0;
          });
    }

    std::vector<uint32_t> w, h;
    std::vector<uint8_t>  depth;
    std::vector<bool>     sgn;
    uint32_t tex_w = 0, tex_h = 0;
    bool ok      = true;
    bool dims_ok = true;

    auto row_to_rgb = [](uint32_t y, int32_t *const *rows, uint16_t nc,
                         int32_t shift, uint8_t *rgb_buf, uint32_t cw) {
      uint8_t *dst = rgb_buf + static_cast<std::size_t>(y) * cw * 3u;
      for (uint32_t x = 0; x < cw; ++x) {
        auto to_u8 = [shift](int32_t v) -> uint8_t {
          if (shift > 0) v >>= shift;
          if (v < 0) return 0;
          if (v > 255) return 255;
          return static_cast<uint8_t>(v);
        };
        dst[3u * x + 0] = to_u8(rows[0][x]);
        dst[3u * x + 1] = to_u8(rows[1][x]);
        dst[3u * x + 2] = to_u8(rows[2][x]);
      }
    };

    try {
      if (opt.dual_res) {
        // ── Phase 4B dual-resolution decode ──────────────────────────
        // Pass 1: LL at max reduce → tiny coarse background
        // Pass 2: Full-res with precinct filter → sharp fovea overlay
        const uint8_t max_red = dec.get_max_safe_reduce_NL();
        std::vector<uint32_t> ll_w, ll_h, fov_w, fov_h;
        std::vector<uint8_t> ll_rgb;
        uint32_t ll_tw = 0, ll_th = 0;

        dec.invoke_dual_resolution(
            // LL callback: store coarse image
            [&](uint32_t y, int32_t *const *rows, uint16_t nc) {
              if (nc < 3 || ll_w.empty()) { dims_ok = false; return; }
              const uint32_t cw = ll_w[0], ch = ll_h[0];
              if (ll_tw != cw || ll_th != ch) {
                ll_tw = cw; ll_th = ch;
                ll_rgb.assign(static_cast<std::size_t>(cw) * ch * 3u, 0);
              }
              if (y >= ch) return;
              const int32_t shift = (depth.empty() ? 0 : static_cast<int32_t>(depth[0]) - 8);
              row_to_rgb(y, rows, nc, shift, ll_rgb.data(), cw);
            },
            // Foveal callback: overwrite onto upscaled LL
            [&](uint32_t y, int32_t *const *rows, uint16_t nc) {
              if (nc < 3 || fov_w.empty()) { dims_ok = false; return; }
              const uint32_t cw = fov_w[0], ch = fov_h[0];
              if (tex_w != cw || tex_h != ch) {
                tex_w = cw; tex_h = ch;
                // Upscale LL into the full-canvas RGB buffer
                rgb.assign(static_cast<std::size_t>(cw) * ch * 3u, 0);
                if (ll_tw > 0 && ll_th > 0) {
                  for (uint32_t fy = 0; fy < ch; ++fy) {
                    const uint32_t ly = static_cast<uint32_t>(
                        static_cast<uint64_t>(fy) * ll_th / ch);
                    const uint8_t *src = ll_rgb.data() + static_cast<size_t>(ly) * ll_tw * 3u;
                    uint8_t *dst = rgb.data() + static_cast<size_t>(fy) * cw * 3u;
                    for (uint32_t fx = 0; fx < cw; ++fx) {
                      const uint32_t lx = static_cast<uint32_t>(
                          static_cast<uint64_t>(fx) * ll_tw / cw);
                      dst[3u * fx + 0] = src[3u * lx + 0];
                      dst[3u * fx + 1] = src[3u * lx + 1];
                      dst[3u * fx + 2] = src[3u * lx + 2];
                    }
                  }
                }
              }
              if (y >= ch) return;
              // Check if this row has data (non-zero from foveal precincts)
              uint32_t acc = 0;
              for (uint32_t x = 0; x < cw && acc == 0; ++x)
                acc |= static_cast<uint32_t>(rows[0][x] | rows[1][x] | rows[2][x]);
              if (acc == 0) return;  // absent-precinct row — keep LL background
              const int32_t shift = (depth.empty() ? 0 : static_cast<int32_t>(depth[0]) - 8);
              row_to_rgb(y, rows, nc, shift, rgb.data(), cw);
            },
            max_red, ll_w, ll_h, fov_w, fov_h, depth, sgn);
      } else {
        // ── Single-pass decode (original path) ──────────────────────
        dec.invoke_line_based_stream(
            [&](uint32_t y, int32_t *const *rows, uint16_t nc) {
              if (nc < 3 || w.empty() || h.empty()) { dims_ok = false; return; }
              const uint32_t cw = w[0], ch = h[0];
              if (tex_w != cw || tex_h != ch) {
                tex_w = cw; tex_h = ch;
                rgb.assign(static_cast<std::size_t>(cw) * ch * 3u, 0);
              }
              if (y >= ch) return;
              const int32_t shift = (depth.empty() ? 0 : static_cast<int32_t>(depth[0]) - 8);
              row_to_rgb(y, rows, nc, shift, rgb.data(), cw);
            },
            w, h, depth, sgn);
      }
    } catch (std::exception &e) {
      std::fprintf(stderr, "decode failed: %s\n", e.what());
      break;
    }
    if (!dims_ok) ok = false;

    if (ok && tex_w > 0 && tex_h > 0)
      renderer.upload_and_draw(rgb.data(), static_cast<int>(tex_w),
                               static_cast<int>(tex_h));

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
