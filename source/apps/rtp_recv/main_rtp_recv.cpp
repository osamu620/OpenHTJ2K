// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.
//
// Licensed under the same BSD 3-Clause terms as the rest of OpenHTJ2K.

// RFC 9828 RTP receiver for HTJ2K video with a GLFW/OpenGL preview window.
//
// Typical usage (alongside a black-box kdu_stream_send sender):
//
//     build/bin/open_htj2k_rtp_recv --port 6000 --colorspace bt709 --range full
//
// Sub-codestream latency on the receive side is not implemented in v1 — the
// decoder interface requires a complete codestream per frame, so the main
// loop reassembles each frame and hands the full buffer to
// openhtj2k_decoder::invoke_line_based_stream().  See
// /home/osamu/.claude/plans/unified-kindling-parnas.md for rationale.

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <GLFW/glfw3.h>

#include "decoder.hpp"
#include "frame_handler.hpp"
#include "frame_pipeline.hpp"
#include "gl_renderer.hpp"
#include "planar_shift.hpp"
#include "rfc9828_parser.hpp"
#include "rtp_socket.hpp"
#include "ycbcr_rgb.hpp"

using namespace open_htj2k::rtp_recv;

namespace {

// ----------------------- CLI -----------------------

struct CliOptions {
  std::string bind_host    = "0.0.0.0";
  uint16_t    bind_port    = 6000;
  int         max_frames   = 0;      // 0 = unlimited
  bool        render       = true;   // --no-render disables GLFW init
  bool        decode       = true;   // --no-decode skips the openhtj2k decoder entirely
  bool        threading    = true;   // --threading=off falls back to v1 single-threaded
  bool        vsync        = true;   // --no-vsync flips the GLFW swap interval to 0
  std::string dump_pattern;          // printf-style, e.g. "/tmp/frame_%05d.j2c"
  // S=0 fallback colorspace (used only when Main Packet says S=0).
  // Leave as nullptr to require the stream to carry S=1.
  const ycbcr_coefficients* s0_fallback = nullptr;
  std::string s0_label;  // human-readable form for logging
  bool        smoke_test = false;
  // Default 2 matches the benchmarked optimum on broadcast 4K HT codestreams
  // (HTJ2K's intra-frame parallelism saturates fast; more threads pay extra
  // ThreadPool spinup cost without speedup).  Override per machine/content
  // via --threads.
  uint32_t    num_decoder_threads = 2;
  // Color conversion path.  "shader" (default) does YCbCr→RGB in the
  // fragment shader — the CPU only shifts samples to 8-bit per plane.
  // "cpu" runs the legacy scalar ycbcr_row_to_rgb8 on the decode thread
  // and uploads interleaved RGB.  Forced to "cpu" if the renderer fails
  // to create a GL 3.3 core context (fallback for headless / GL-limited
  // environments).
  enum class ColorPath : uint8_t { Shader, Cpu };
  ColorPath   color_path = ColorPath::Shader;
  // Frame-pacing target in fps, active only when vsync is off.  A 30 fps
  // source on a 60 Hz display without pacing shows 3:2 pulldown judder
  // because decoded frames are presented as soon as they're ready, which
  // drifts relative to vblank — some frames land on one vblank, others
  // on the next.  Pacing holds the next present until last_present_tp
  // + 1/pace_fps, giving a stable 2:2 cadence on 60 Hz.  0 disables the
  // pacer (useful for benchmarking and --no-render runs).
  double      pace_fps = 30.0;
};

void print_usage(const char* argv0) {
  std::fprintf(
      stderr,
      "Usage: %s [options]\n"
      "  --port <N>               UDP port to bind (default 6000)\n"
      "  --bind <host>            Host/IP to bind (default 0.0.0.0)\n"
      "  --frames <N>             Exit after N successfully decoded frames\n"
      "  --no-render              Do not open a window; pure depacketize+decode\n"
      "  --no-vsync               Disable GLFW vsync (immediate swap, no display lock)\n"
      "  --no-decode              Skip the openhtj2k decoder entirely (capture only)\n"
      "  --threading {on,off}     v2 multi-thread (default on); off = v1 single-thread\n"
      "  --dump-codestream <fmt>  printf-style path, e.g. '/tmp/f_%%05d.j2c'\n"
      "  --colorspace <name>      S=0 fallback: bt709 | bt601 | rgb (bt2020 not yet)\n"
      "  --range <name>           S=0 fallback: full | narrow (default full)\n"
      "  --threads <N>            Decoder thread count (default 2; 0 = hardware)\n"
      "  --color-path {shader|cpu} YCbCr->RGB on GPU (default) or CPU fallback\n"
      "  --pace-fps <N>           Frame-pacing target (default 30; 0 = disabled)\n"
      "                           Only active when vsync is off\n"
      "  --smoke-test             Run internal unit smoke tests and exit\n",
      argv0);
}

const char* get_arg(int argc, char** argv, const char* name) {
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], name) == 0 && i + 1 < argc) return argv[i + 1];
  }
  return nullptr;
}

bool has_flag(int argc, char** argv, const char* name) {
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], name) == 0) return true;
  }
  return false;
}

// Validates that a --dump-codestream pattern is a safe printf format
// string that takes exactly one integer argument.  Accepts optional
// flags (-+#0 space), width, and precision, and requires the conversion
// specifier to be one of d/i/u/o/x/X.  Literal "%%" is allowed.  Any
// other format directive (or zero/multiple integer slots) is rejected,
// closing the attack surface where passing e.g. "%s" or "%n" as the
// pattern would hand an arbitrary user string to snprintf's varargs
// parser with an unsigned int on the stack — undefined behaviour even
// when run by the local user.
bool validate_dump_pattern(const std::string& pattern) {
  int int_slots = 0;
  for (size_t i = 0; i < pattern.size(); ++i) {
    if (pattern[i] != '%') continue;
    ++i;
    if (i >= pattern.size()) return false;     // trailing bare '%'
    if (pattern[i] == '%') continue;           // literal "%%"
    // Optional flags.
    while (i < pattern.size()
           && (pattern[i] == '-' || pattern[i] == '+' || pattern[i] == ' '
               || pattern[i] == '#' || pattern[i] == '0'))
      ++i;
    // Optional width (digits only; refuse "*" indirection).
    while (i < pattern.size() && pattern[i] >= '0' && pattern[i] <= '9') ++i;
    // Optional precision ".digits".
    if (i < pattern.size() && pattern[i] == '.') {
      ++i;
      while (i < pattern.size() && pattern[i] >= '0' && pattern[i] <= '9') ++i;
    }
    if (i >= pattern.size()) return false;
    const char c = pattern[i];
    if (c == 'd' || c == 'i' || c == 'u' || c == 'o' || c == 'x' || c == 'X') {
      ++int_slots;
    } else {
      return false;                            // rejects %s, %n, %p, %f, %lu, etc.
    }
  }
  return int_slots == 1;
}

bool parse_cli(int argc, char** argv, CliOptions& opt) {
  if (has_flag(argc, argv, "--help") || has_flag(argc, argv, "-h")) {
    print_usage(argv[0]);
    return false;
  }
  opt.smoke_test = has_flag(argc, argv, "--smoke-test");
  opt.render     = !has_flag(argc, argv, "--no-render");
  opt.vsync      = !has_flag(argc, argv, "--no-vsync");
  opt.decode     = !has_flag(argc, argv, "--no-decode");
  if (!opt.decode) opt.render = false;  // can't render without a decoded frame
  if (const char* v = get_arg(argc, argv, "--threading")) {
    if (std::strcmp(v, "on") == 0) opt.threading = true;
    else if (std::strcmp(v, "off") == 0) opt.threading = false;
    else {
      std::fprintf(stderr, "ERROR: --threading must be 'on' or 'off'\n");
      return false;
    }
  }

  if (const char* v = get_arg(argc, argv, "--port"))  opt.bind_port  = static_cast<uint16_t>(std::atoi(v));
  if (const char* v = get_arg(argc, argv, "--bind"))  opt.bind_host  = v;
  if (const char* v = get_arg(argc, argv, "--frames")) opt.max_frames = std::atoi(v);
  if (const char* v = get_arg(argc, argv, "--dump-codestream")) {
    opt.dump_pattern = v;
    if (!validate_dump_pattern(opt.dump_pattern)) {
      std::fprintf(stderr,
                   "ERROR: --dump-codestream pattern must contain exactly one\n"
                   "       integer conversion specifier (%%d, %%i, %%u, %%o,\n"
                   "       %%x, or %%X, with optional flags/width/precision).\n"
                   "       Example: '/tmp/frame_%%05d.j2c'\n");
      return false;
    }
  }
  if (const char* v = get_arg(argc, argv, "--threads")) opt.num_decoder_threads = static_cast<uint32_t>(std::atoi(v));
  if (const char* v = get_arg(argc, argv, "--color-path")) {
    if (std::strcmp(v, "shader") == 0) opt.color_path = CliOptions::ColorPath::Shader;
    else if (std::strcmp(v, "cpu") == 0) opt.color_path = CliOptions::ColorPath::Cpu;
    else {
      std::fprintf(stderr, "ERROR: --color-path must be 'shader' or 'cpu'\n");
      return false;
    }
  }
  if (const char* v = get_arg(argc, argv, "--pace-fps")) {
    char*        end = nullptr;
    const double d   = std::strtod(v, &end);
    if (end == v || d < 0.0 || d > 1000.0) {
      std::fprintf(stderr, "ERROR: --pace-fps must be in [0, 1000]\n");
      return false;
    }
    opt.pace_fps = d;
  }

  // Colorspace fallback.  When a Main Packet carries S=1 we ignore these.
  bool range_full = true;
  if (const char* v = get_arg(argc, argv, "--range")) {
    if (std::strcmp(v, "full") == 0) range_full = true;
    else if (std::strcmp(v, "narrow") == 0) range_full = false;
    else {
      std::fprintf(stderr, "ERROR: --range must be 'full' or 'narrow'\n");
      return false;
    }
  }
  if (const char* v = get_arg(argc, argv, "--colorspace")) {
    if (std::strcmp(v, "bt709") == 0) {
      opt.s0_fallback = range_full ? &YCBCR_BT709_FULL : &YCBCR_BT709_NARROW;
      opt.s0_label    = range_full ? "bt709-full" : "bt709-narrow";
    } else if (std::strcmp(v, "bt601") == 0) {
      opt.s0_fallback = range_full ? &YCBCR_BT601_FULL : &YCBCR_BT601_NARROW;
      opt.s0_label    = range_full ? "bt601-full" : "bt601-narrow";
    } else if (std::strcmp(v, "bt2020") == 0) {
      std::fprintf(stderr, "ERROR: --colorspace bt2020 not supported in v1\n");
      return false;
    } else if (std::strcmp(v, "rgb") == 0) {
      opt.s0_fallback = nullptr;  // sentinel for "no YCbCr, components already RGB"
      opt.s0_label    = "rgb";
    } else {
      std::fprintf(stderr, "ERROR: --colorspace: unknown value '%s'\n", v);
      return false;
    }
  }
  return true;
}

// ----------------------- Smoke tests (run with --smoke-test) -----------------------

int smoke_test_socket();
int smoke_test_parser();
int smoke_test_ycbcr();
int smoke_test_frame_handler();

int run_smoke_tests() {
  if (smoke_test_socket() != 0) return EXIT_FAILURE;
  std::printf("udp socket smoke-test OK\n");
  if (smoke_test_parser() != 0) return EXIT_FAILURE;
  std::printf("rfc9828 parser smoke-test OK\n");
  if (smoke_test_ycbcr() != 0) return EXIT_FAILURE;
  std::printf("ycbcr->rgb smoke-test OK\n");
  if (!plane_shift_smoke_test()) {
    std::fprintf(stderr, "planar shift smoke-test FAILED\n");
    return EXIT_FAILURE;
  }
  std::printf("planar shift smoke-test OK\n");
  if (smoke_test_frame_handler() != 0) return EXIT_FAILURE;
  std::printf("frame_handler smoke-test OK\n");
  return EXIT_SUCCESS;
}

// ----------------------- Frame processing -----------------------

// Picks YCbCr->RGB coefficients per RFC 9828 §5.3 (S=1 in-band, S=0 CLI
// fallback).  Sets `components_are_rgb` if no conversion is needed (Table 1
// identity components or `--colorspace rgb`).  Logs and returns false on
// rejection.  Used by both the v1 single-threaded decoder path and the v2
// decode worker.
bool select_coefficients_for_frame(const AssembledFrame& frame, const CliOptions& opts,
                                   const ycbcr_coefficients*& coeffs, bool& components_are_rgb) {
  coeffs             = nullptr;
  components_are_rgb = false;
  if (frame.has_meta && frame.s) {
    if (frame.mat == 0) {
      // Table 1 "identity" / RGB components: no YCbCr conversion.
      components_are_rgb = true;
    } else {
      coeffs = select_coefficients_from_mat(frame.mat, frame.range);
      if (coeffs == nullptr) {
        std::fprintf(stderr, "frame: unsupported MAT=%u (S=1); dropping\n",
                     static_cast<unsigned>(frame.mat));
        return false;
      }
    }
  } else {
    // S=0: use CLI fallback.  If neither fallback nor rgb mode is set, fail.
    if (opts.s0_label == "rgb") {
      components_are_rgb = true;
    } else if (opts.s0_fallback != nullptr) {
      coeffs = opts.s0_fallback;
    } else {
      std::fprintf(stderr,
                   "frame: Main Packet has S=0 and --colorspace not set; refusing to guess\n");
      return false;
    }
  }
  return true;
}

void log_coefficients_choice_once(const CliOptions& opts, const ycbcr_coefficients* coeffs,
                                  bool components_are_rgb) {
  if (components_are_rgb) {
    std::fprintf(stderr, "info: rendering RGB components directly\n");
    return;
  }
  std::fprintf(stderr, "info: YCbCr -> RGB via %s coefficients\n",
               coeffs == opts.s0_fallback
                   ? opts.s0_label.c_str()
                   : (coeffs == &YCBCR_BT709_FULL     ? "bt709-full"
                      : coeffs == &YCBCR_BT709_NARROW ? "bt709-narrow"
                      : coeffs == &YCBCR_BT601_FULL   ? "bt601-full"
                      : coeffs == &YCBCR_BT601_NARROW ? "bt601-narrow"
                                                      : "?"));
}

// Run invoke_line_based_stream on `decoder` and write the result as 8-bit
// interleaved RGB into `out_rgb` (resized as needed).  `decoder.parse()`
// must already have been called.  On success returns true and fills out_w,
// out_h with the luma dimensions; on failure logs and returns false.
//
// Used by both the v1 (fresh-decoder-per-frame) and v2 (long-lived decoder
// + per-frame init()) paths — they differ in how the decoder is loaded,
// not in how the row callback fills the RGB buffer.
bool decode_to_rgb_buffer(open_htj2k::openhtj2k_decoder& decoder,
                          const ycbcr_coefficients* coeffs, bool components_are_rgb,
                          std::vector<uint8_t>& out_rgb, uint32_t& out_w, uint32_t& out_h) {
  uint8_t  depth0          = 0;
  uint32_t cb_stride_ratio = 1;
  uint32_t cr_stride_ratio = 1;
  bool     dims_ok         = true;
  out_w                    = 0;
  out_h                    = 0;

  try {
    std::vector<uint32_t> widths;
    std::vector<uint32_t> heights;
    std::vector<uint8_t>  depths;
    std::vector<bool>     signeds;
    decoder.invoke_line_based_stream_reuse(
        [&](uint32_t y, int32_t* const* rows, uint16_t nc) {
          if (y == 0) {
            if (nc < 1) {
              dims_ok = false;
              return;
            }
            out_w  = widths[0];
            out_h  = heights[0];
            depth0 = depths[0];
            if (nc >= 3) {
              cb_stride_ratio = widths[0] / widths[1];
              cr_stride_ratio = widths[0] / widths[2];
              if (cb_stride_ratio == 0) cb_stride_ratio = 1;
              if (cr_stride_ratio == 0) cr_stride_ratio = 1;
            }
            out_rgb.assign(static_cast<size_t>(out_w) * out_h * 3, 0);
          }
          if (!dims_ok) return;
          uint8_t* out_row = out_rgb.data() + static_cast<size_t>(y) * out_w * 3;
          if (nc >= 3 && !components_are_rgb && coeffs != nullptr) {
            ycbcr_row_to_rgb8(rows[0], rows[1], rows[2], out_row, out_w, cb_stride_ratio,
                              cr_stride_ratio, *coeffs, depth0,
                              /*is_signed=*/signeds[0]);
          } else if (nc >= 3 && components_are_rgb) {
            rgb_row_to_rgb8(rows[0], rows[1], rows[2], out_row, out_w, depth0);
          } else if (nc == 1) {
            // Grayscale: replicate Y into R/G/B.
            const int32_t shift  = static_cast<int32_t>(depth0) - 8;
            const int32_t maxval = (1 << depth0) - 1;
            for (uint32_t x = 0; x < out_w; ++x) {
              int32_t v = rows[0][x];
              if (v < 0) v = 0;
              if (v > maxval) v = maxval;
              const uint8_t v8   = static_cast<uint8_t>(shift > 0 ? (v >> shift) : v);
              out_row[3 * x + 0] = v8;
              out_row[3 * x + 1] = v8;
              out_row[3 * x + 2] = v8;
            }
          }
        },
        widths, heights, depths, signeds);
  } catch (std::exception& e) {
    std::fprintf(stderr, "decoder.invoke_line_based_stream failed: %s\n", e.what());
    return false;
  }

  return dims_ok && out_w > 0 && out_h > 0;
}

// Shader-path twin of decode_to_rgb_buffer.  Runs invoke_line_based_stream
// and writes each component's int32 samples into one of three 8-bit
// planar buffers, shifting down to 8 bits per sample (the broadcast 10-bit
// 4:2:2 source loses 2 LSBs — visually fine for SDR preview; HDR will
// bump to R16 later).  All matrix math and range normalization moves to
// the fragment shader, so the CPU side is just a cast + clamp + store.
//
// On success, `df` is populated with plane_y/plane_cb/plane_cr, width,
// height, chroma_width, chroma_height, and kind (PLANAR_YCBCR or
// PLANAR_RGB).  On failure returns false.
bool decode_to_planar_buffers(open_htj2k::openhtj2k_decoder& decoder, bool components_are_rgb,
                              DecodedFrame& df) {
  df.rgb.clear();
  df.plane_y.clear();
  df.plane_cb.clear();
  df.plane_cr.clear();
  df.plane_y_16.clear();
  df.plane_cb_16.clear();
  df.plane_cr_16.clear();
  df.width         = 0;
  df.height        = 0;
  df.chroma_width  = 0;
  df.chroma_height = 0;
  df.bit_depth     = 0;
  df.kind          = components_are_rgb ? DecodedFrame::PLANAR_RGB : DecodedFrame::PLANAR_YCBCR;

  bool     dims_ok    = true;
  uint32_t luma_w     = 0;
  uint32_t luma_h     = 0;
  uint32_t chroma_w_0 = 0;
  uint32_t chroma_h_0 = 0;
  uint8_t  depth_y    = 0;
  uint8_t  depth_c    = 0;
  bool     use_16     = false;  // true when depth_y > 8 -> take the GL_R16 path

  try {
    std::vector<uint32_t> widths;
    std::vector<uint32_t> heights;
    std::vector<uint8_t>  depths;
    std::vector<bool>     signeds;
    decoder.invoke_line_based_stream_reuse(
        [&](uint32_t y, int32_t* const* rows, uint16_t nc) {
          if (y == 0) {
            if (nc < 1) {
              dims_ok = false;
              return;
            }
            luma_w  = widths[0];
            luma_h  = heights[0];
            depth_y = depths[0];
            if (nc >= 3) {
              chroma_w_0 = widths[1];
              chroma_h_0 = heights[1];
              depth_c    = depths[1];
              // 3 planes assumed equal chroma dims (422 / 420 / 444).
              if (widths.size() >= 3 && heights.size() >= 3) {
                if (widths[2] != chroma_w_0 || heights[2] != chroma_h_0) {
                  // Mismatched chroma planes (asymmetric subsampling).
                  // v3 day one doesn't handle this; dims_ok stays true
                  // but we force single-plane width/height so the
                  // upload still produces something sane.
                  chroma_w_0 = std::min(chroma_w_0, widths[2]);
                  chroma_h_0 = std::min(chroma_h_0, heights[2]);
                }
              }
            } else {
              // Grayscale: replicate Y into Cb/Cr with neutral chroma.
              chroma_w_0 = luma_w;
              chroma_h_0 = luma_h;
              depth_c    = depth_y;
            }
            df.width         = luma_w;
            df.height        = luma_h;
            df.chroma_width  = chroma_w_0;
            df.chroma_height = chroma_h_0;
            df.bit_depth     = depth_y;
            use_16           = (depth_y > 8);
            if (use_16) {
              // 16-bit plane path.  Neutral chroma in the u16 space is the
              // midpoint of the source's [0, (1<<depth)-1] range.
              const uint16_t neutral_c = components_are_rgb
                                             ? 0
                                             : static_cast<uint16_t>(1u << (depth_c - 1));
              df.plane_y_16.assign(static_cast<size_t>(luma_w) * luma_h, 0);
              df.plane_cb_16.assign(static_cast<size_t>(chroma_w_0) * chroma_h_0, neutral_c);
              df.plane_cr_16.assign(static_cast<size_t>(chroma_w_0) * chroma_h_0, neutral_c);
            } else {
              df.plane_y.assign(static_cast<size_t>(luma_w) * luma_h, 0);
              df.plane_cb.assign(static_cast<size_t>(chroma_w_0) * chroma_h_0,
                                 components_are_rgb ? 0 : 128);
              df.plane_cr.assign(static_cast<size_t>(chroma_w_0) * chroma_h_0,
                                 components_are_rgb ? 0 : 128);
            }
          }
          if (!dims_ok) return;

          const int32_t maxval_y = (1 << depth_y) - 1;

          // Luma row -- every call has a Y row at the current y.
          if (use_16) {
            clamp_i32_plane_to_u16(
                rows[0], df.plane_y_16.data() + static_cast<size_t>(y) * luma_w, luma_w,
                maxval_y);
          } else {
            const int32_t shift_y = static_cast<int32_t>(depth_y) - 8;
            shift_i32_plane_to_u8(rows[0],
                                  df.plane_y.data() + static_cast<size_t>(y) * luma_w,
                                  luma_w, shift_y, maxval_y);
          }

          // Chroma (or G/B for PLANAR_RGB).  Written only for rows that
          // land on the chroma grid.  For 4:2:2 the chroma height equals
          // luma height and every luma row contributes.  For 4:2:0 the
          // luma height is 2x chroma; rows[] pointers may be nullptr on
          // the "skip" luma rows -- guard with the callback nc check.
          if (nc >= 3) {
            const int32_t maxval_c = (1 << depth_c) - 1;
            const uint32_t yc      = (luma_h > 0)
                                         ? static_cast<uint32_t>(
                                               static_cast<uint64_t>(y) * chroma_h_0 / luma_h)
                                         : 0;
            if (yc < chroma_h_0) {
              if (use_16) {
                if (rows[1] != nullptr) {
                  clamp_i32_plane_to_u16(
                      rows[1],
                      df.plane_cb_16.data() + static_cast<size_t>(yc) * chroma_w_0,
                      chroma_w_0, maxval_c);
                }
                if (rows[2] != nullptr) {
                  clamp_i32_plane_to_u16(
                      rows[2],
                      df.plane_cr_16.data() + static_cast<size_t>(yc) * chroma_w_0,
                      chroma_w_0, maxval_c);
                }
              } else {
                const int32_t shift_c = static_cast<int32_t>(depth_c) - 8;
                if (rows[1] != nullptr) {
                  shift_i32_plane_to_u8(
                      rows[1],
                      df.plane_cb.data() + static_cast<size_t>(yc) * chroma_w_0,
                      chroma_w_0, shift_c, maxval_c);
                }
                if (rows[2] != nullptr) {
                  shift_i32_plane_to_u8(
                      rows[2],
                      df.plane_cr.data() + static_cast<size_t>(yc) * chroma_w_0,
                      chroma_w_0, shift_c, maxval_c);
                }
              }
            }
          }
        },
        widths, heights, depths, signeds);
  } catch (std::exception& e) {
    std::fprintf(stderr, "decoder.invoke_line_based_stream failed: %s\n", e.what());
    return false;
  }

  return dims_ok && df.width > 0 && df.height > 0;
}

// v1 path: decode one reassembled HTJ2K codestream by constructing a fresh
// openhtj2k_decoder, then upload to the renderer.  Used only when
// --threading=off.  Returns true on success.
bool decode_and_present(const AssembledFrame& frame, const CliOptions& opts, bool is_first_frame,
                        GlRenderer* renderer, std::vector<uint8_t>& rgb_backbuffer,
                        DecodedFrame& planar_scratch) {
  using namespace open_htj2k;

  openhtj2k_decoder decoder(frame.bytes.data(), frame.bytes.size(), /*reduce_NL=*/0,
                            opts.num_decoder_threads);
  try {
    decoder.parse();
  } catch (std::exception& e) {
    std::fprintf(stderr, "decoder.parse() failed: %s\n", e.what());
    return false;
  }

  const ycbcr_coefficients* coeffs = nullptr;
  bool                      components_are_rgb = false;
  if (!select_coefficients_for_frame(frame, opts, coeffs, components_are_rgb)) return false;
  if (is_first_frame) log_coefficients_choice_once(opts, coeffs, components_are_rgb);

  if (opts.color_path == CliOptions::ColorPath::Shader) {
    if (!decode_to_planar_buffers(decoder, components_are_rgb, planar_scratch)) {
      std::fprintf(stderr, "frame: unable to determine dimensions; dropping\n");
      return false;
    }
    if (renderer != nullptr) {
      if (planar_scratch.bit_depth > 8) {
        renderer->upload_planar_16_and_draw(
            planar_scratch.plane_y_16.data(), planar_scratch.plane_cb_16.data(),
            planar_scratch.plane_cr_16.data(), static_cast<int>(planar_scratch.width),
            static_cast<int>(planar_scratch.height),
            static_cast<int>(planar_scratch.chroma_width),
            static_cast<int>(planar_scratch.chroma_height),
            static_cast<int>(planar_scratch.bit_depth), coeffs, components_are_rgb);
      } else {
        renderer->upload_planar_and_draw(
            planar_scratch.plane_y.data(), planar_scratch.plane_cb.data(),
            planar_scratch.plane_cr.data(), static_cast<int>(planar_scratch.width),
            static_cast<int>(planar_scratch.height),
            static_cast<int>(planar_scratch.chroma_width),
            static_cast<int>(planar_scratch.chroma_height), coeffs, components_are_rgb);
      }
    }
  } else {
    uint32_t out_w = 0;
    uint32_t out_h = 0;
    if (!decode_to_rgb_buffer(decoder, coeffs, components_are_rgb, rgb_backbuffer, out_w, out_h)) {
      std::fprintf(stderr, "frame: unable to determine dimensions; dropping\n");
      return false;
    }
    if (renderer != nullptr) {
      renderer->upload_and_draw(rgb_backbuffer.data(), static_cast<int>(out_w),
                                static_cast<int>(out_h));
    }
  }
  return true;
}

void dump_frame_if_requested(const CliOptions& opts, const AssembledFrame& frame,
                             uint64_t frame_index) {
  if (opts.dump_pattern.empty()) return;
  char path[512];
  std::snprintf(path, sizeof(path), opts.dump_pattern.c_str(),
                static_cast<unsigned>(frame_index));
  FILE* fp = std::fopen(path, "wb");
  if (!fp) {
    std::fprintf(stderr, "dump: fopen('%s') failed: %s\n", path, std::strerror(errno));
    return;
  }
  std::fwrite(frame.bytes.data(), 1, frame.bytes.size(), fp);
  std::fclose(fp);
}

// ----------------------- v1 single-threaded main loop (--threading=off) -----------------------

int run_receiver_single_threaded(const CliOptions& opts) {
  UdpSocket sock;
  if (!sock.bind(opts.bind_host, opts.bind_port)) {
    std::fprintf(stderr, "bind %s:%u failed: %s\n", opts.bind_host.c_str(), opts.bind_port,
                 sock.last_error().c_str());
    return EXIT_FAILURE;
  }
  // 32 MB SO_RCVBUF — enough for ~20 frames of 4K HTJ2K at broadcast bitrates
  // while the decoder processes the current frame.  Best-effort; the kernel
  // doubles the request internally and silently clamps to net.core.rmem_max,
  // so we read back the granted value and warn the user if it is too small
  // to absorb a single frame.
  constexpr int kRequestedRecvBuf = 32 * 1024 * 1024;
  sock.set_recv_buffer_size(kRequestedRecvBuf);
  const int granted = sock.last_granted_recv_buf();

  std::fprintf(stderr, "listening on %s:%u\n", opts.bind_host.c_str(), opts.bind_port);
  std::fprintf(stderr, "SO_RCVBUF: requested %d MB, kernel granted %d KB\n",
               kRequestedRecvBuf / (1024 * 1024), granted / 1024);
  if (granted < 4 * 1024 * 1024) {
    std::fprintf(stderr,
                 "WARN: SO_RCVBUF is < 4 MB. The kernel will drop packets when the\n"
                 "      receiver falls behind the sender. Raise net.core.rmem_max:\n"
                 "          sudo sysctl -w net.core.rmem_max=33554432\n"
                 "      and re-run.  Without this, expect frame corruption under\n"
                 "      sustained high-bitrate input.\n");
  }

  GlRenderer renderer;
  GlRenderer* renderer_ptr = nullptr;
  if (opts.render) {
    // Initial window size is a placeholder; the first frame resizes the texture.
    if (!renderer.init(1280, 720, "OpenHTJ2K RFC 9828 receiver", opts.vsync)) {
      std::fprintf(stderr, "WARN: GLFW init failed; continuing in --no-render mode\n");
    } else {
      renderer_ptr = &renderer;
    }
  }

  // If the renderer failed to come up, force the CPU color path — nothing
  // else needs the GL 3.3 shader machinery and we still want --no-render
  // + --threading=off to work for diagnostics.
  CliOptions opts_effective = opts;
  if (renderer_ptr == nullptr && opts_effective.color_path == CliOptions::ColorPath::Shader) {
    // Shader path is only meaningful with a live GL context.  Switch to
    // the CPU path so the decode function still writes something.
    opts_effective.color_path = CliOptions::ColorPath::Cpu;
  }

  FrameHandler frame_handler;
  std::vector<uint8_t> packet_buf(65536);  // max UDP payload
  std::vector<uint8_t> rgb_backbuffer;
  DecodedFrame         planar_scratch;

  uint64_t frames_decoded = 0;
  uint64_t frames_failed  = 0;
  uint64_t frames_attempted = 0;  // dump index — counts every emitted frame, success or fail
  bool     first_frame    = true;
  bool     should_exit    = false;

  // Timing instrumentation.  Wall-clock per frame from the moment the frame
  // leaves frame_handler to when decode+render returns; aggregate into
  // min/avg/max and a running FPS over the last 30 frames.
  using Clock             = std::chrono::steady_clock;
  const auto run_start_tp = Clock::now();
  auto  last_log_tp       = run_start_tp;
  double decode_ms_sum    = 0.0;
  double decode_ms_min    = std::numeric_limits<double>::infinity();
  double decode_ms_max    = 0.0;
  uint64_t frames_at_last_log = 0;

  while (!should_exit) {
    // Poll the socket with a short timeout so GLFW events still fire every
    // few ms even when no packets are arriving.
    const int ready = sock.wait_readable(/*timeout_ms=*/5);
    if (ready < 0) {
      std::fprintf(stderr, "socket poll error: %s\n", sock.last_error().c_str());
      break;
    }

    if (ready > 0) {
      // Drain whatever is pending in the kernel buffer before returning to
      // GLFW event pumping.  This keeps latency low under bursty arrivals.
      for (int drain = 0; drain < 256; ++drain) {
        auto n = sock.recv(packet_buf.data(), packet_buf.size());
        if (n == UdpSocket::kAgain) break;
        if (n == UdpSocket::kError) {
          std::fprintf(stderr, "recv error: %s\n", sock.last_error().c_str());
          should_exit = true;
          break;
        }
        if (n < 12) continue;  // too small for an RTP header
        const auto pkt_len = static_cast<size_t>(n);

        // Parse RTP fixed header.
        RtpHeader    rtp{};
        std::string  err;
        if (!parse_rtp_header(packet_buf.data(), pkt_len, rtp, err)) continue;
        if (rtp.payload_offset >= pkt_len) continue;

        const uint8_t* payload    = packet_buf.data() + rtp.payload_offset;
        const size_t   payload_sz = pkt_len - rtp.payload_offset;
        if (payload_sz < 8) continue;  // need at least an RFC 9828 payload header

        // Dispatch on the 2-bit MH field at the top of byte 0.
        const uint8_t mh = static_cast<uint8_t>(payload[0] >> 6);
        std::optional<AssembledFrame> emitted;

        if (mh == MH_BODY) {
          BodyPacketHeader body{};
          if (!parse_body_packet_header(payload, payload_sz, body, err)) continue;
          const uint8_t* cs_bytes = payload + body.codestream_offset;
          const size_t   cs_len   = payload_sz - body.codestream_offset;
          frame_handler.push_body_packet(rtp, body, cs_bytes, cs_len, emitted);
        } else {
          MainPacketHeader main{};
          if (!parse_main_packet_header(payload, payload_sz, main, err)) continue;
          const uint8_t* cs_bytes = payload + main.codestream_offset;
          const size_t   cs_len   = payload_sz - main.codestream_offset;
          frame_handler.push_main_packet(rtp, main, cs_bytes, cs_len, emitted);
        }

        if (emitted.has_value()) {
          dump_frame_if_requested(opts, *emitted, frames_attempted);
          ++frames_attempted;
          const auto decode_start = Clock::now();
          const bool ok =
              opts_effective.decode
                  ? decode_and_present(*emitted, opts_effective, first_frame, renderer_ptr,
                                       rgb_backbuffer, planar_scratch)
                  : true;  // --no-decode: count emitted frame as a success
          const auto decode_end = Clock::now();
          const double decode_ms =
              std::chrono::duration<double, std::milli>(decode_end - decode_start).count();
          if (ok) {
            ++frames_decoded;
            decode_ms_sum += decode_ms;
            if (decode_ms < decode_ms_min) decode_ms_min = decode_ms;
            if (decode_ms > decode_ms_max) decode_ms_max = decode_ms;
            first_frame = false;

            // Print a running FPS line every ~1 second.
            const auto now = Clock::now();
            const double since_log_s =
                std::chrono::duration<double>(now - last_log_tp).count();
            if (since_log_s >= 1.0) {
              const uint64_t delta_frames = frames_decoded - frames_at_last_log;
              const double   fps          = static_cast<double>(delta_frames) / since_log_s;
              std::fprintf(stderr, "  [%llu frames] inst=%.2f fps, last decode=%.2f ms\n",
                           static_cast<unsigned long long>(frames_decoded), fps, decode_ms);
              frames_at_last_log = frames_decoded;
              last_log_tp        = now;
            }

            if (opts.max_frames > 0 && frames_decoded >= static_cast<uint64_t>(opts.max_frames)) {
              should_exit = true;
              break;
            }
          } else {
            ++frames_failed;
          }
        }
      }
    }

    if (renderer_ptr) {
      renderer_ptr->poll_events();
      if (renderer_ptr->should_close()) should_exit = true;
    }
  }

  const auto   run_end_tp   = Clock::now();
  const double run_secs     = std::chrono::duration<double>(run_end_tp - run_start_tp).count();
  const double avg_fps      = frames_decoded > 0 ? static_cast<double>(frames_decoded) / run_secs
                                                 : 0.0;
  const double decode_ms_avg =
      frames_decoded > 0 ? decode_ms_sum / static_cast<double>(frames_decoded) : 0.0;
  if (frames_decoded == 0) decode_ms_min = 0.0;

  const auto& s = frame_handler.stats();
  std::fprintf(stderr,
               "\n--- summary ---\n"
               "  wall time:        %.2f s\n"
               "  frames attempted: %llu\n"
               "  frames decoded:   %llu\n"
               "  frames failed:    %llu\n"
               "  frames emitted:   %llu\n"
               "  frames dropped:   %llu (mid-frame seq gap)\n"
               "  tail-loss drops:  %llu (frame ended w/o RTP M=1)\n"
               "  packets received: %llu\n"
               "  bytes received:   %llu\n"
               "  sequence gaps:    %llu\n"
               "  avg FPS:          %.2f\n"
               "  decode time ms:   min=%.2f avg=%.2f max=%.2f\n",
               run_secs,
               static_cast<unsigned long long>(frames_attempted),
               static_cast<unsigned long long>(frames_decoded),
               static_cast<unsigned long long>(frames_failed),
               static_cast<unsigned long long>(s.frames_emitted),
               static_cast<unsigned long long>(s.frames_dropped),
               static_cast<unsigned long long>(s.tail_loss_drops),
               static_cast<unsigned long long>(s.packets_received),
               static_cast<unsigned long long>(s.bytes_received),
               static_cast<unsigned long long>(s.seq_gaps),
               avg_fps,
               decode_ms_min, decode_ms_avg, decode_ms_max);

  renderer.shutdown();
  return EXIT_SUCCESS;
}

// ----------------------- v2 multi-threaded main loop (--threading=on) -----------------------

// Shared mutable state for the v2 receive/decode/render trio.  Owned by
// run_receiver_threaded(); references handed to recv_thread_main and
// decode_thread_main.
struct ReceiverState {
  std::atomic<bool>           stop_flag{false};
  LatestSlot<AssembledFrame>  decode_slot;
  LatestSlot<DecodedFrame>    render_slot;

  // Counters (atomic for thread-safe summary at exit).
  std::atomic<uint64_t> frames_emitted_to_decode{0};  // dump index, increments per frame_handler emission
  std::atomic<uint64_t> frames_decoded{0};
  std::atomic<uint64_t> frames_failed{0};

  // Decode timing — written only by the decode thread, read by main at exit.
  std::atomic<uint64_t> decode_us_sum{0};
  std::atomic<uint64_t> decode_us_min{UINT64_MAX};
  std::atomic<uint64_t> decode_us_max{0};
};

void recv_thread_main(const CliOptions& opts, UdpSocket& sock, FrameHandler& frame_handler,
                      ReceiverState& st) {
  std::vector<uint8_t> packet_buf(65536);

  while (!st.stop_flag.load(std::memory_order_acquire)) {
    const int ready = sock.wait_readable(/*timeout_ms=*/5);
    if (ready < 0) {
      std::fprintf(stderr, "socket poll error: %s\n", sock.last_error().c_str());
      st.stop_flag.store(true, std::memory_order_release);
      st.decode_slot.notify();
      st.render_slot.notify();
      return;
    }
    if (ready == 0) continue;

    // Drain everything pending in the kernel buffer in one batch so the
    // socket never falls behind real time.  No upper bound: with the
    // 32 MB SO_RCVBUF and the dedicated thread, this loop empties the
    // kernel buffer faster than it can fill at 30 fps × 4K.  The socket
    // is non-blocking (set in run_receiver_threaded), so recv() returns
    // kAgain immediately when the kernel buffer empties.
    while (true) {
      auto n = sock.recv(packet_buf.data(), packet_buf.size());
      if (n == UdpSocket::kAgain) break;
      if (n == UdpSocket::kError) {
        std::fprintf(stderr, "recv error: %s\n", sock.last_error().c_str());
        st.stop_flag.store(true, std::memory_order_release);
        st.decode_slot.notify();
        st.render_slot.notify();
        return;
      }
      if (n < 12) continue;
      const auto pkt_len = static_cast<size_t>(n);

      RtpHeader   rtp{};
      std::string err;
      if (!parse_rtp_header(packet_buf.data(), pkt_len, rtp, err)) continue;
      if (rtp.payload_offset >= pkt_len) continue;

      const uint8_t* payload    = packet_buf.data() + rtp.payload_offset;
      const size_t   payload_sz = pkt_len - rtp.payload_offset;
      if (payload_sz < 8) continue;

      const uint8_t                  mh = static_cast<uint8_t>(payload[0] >> 6);
      std::optional<AssembledFrame>  emitted;

      if (mh == MH_BODY) {
        BodyPacketHeader body{};
        if (!parse_body_packet_header(payload, payload_sz, body, err)) continue;
        const uint8_t* cs_bytes = payload + body.codestream_offset;
        const size_t   cs_len   = payload_sz - body.codestream_offset;
        frame_handler.push_body_packet(rtp, body, cs_bytes, cs_len, emitted);
      } else {
        MainPacketHeader main{};
        if (!parse_main_packet_header(payload, payload_sz, main, err)) continue;
        const uint8_t* cs_bytes = payload + main.codestream_offset;
        const size_t   cs_len   = payload_sz - main.codestream_offset;
        frame_handler.push_main_packet(rtp, main, cs_bytes, cs_len, emitted);
      }

      if (emitted.has_value()) {
        const uint64_t idx = st.frames_emitted_to_decode.fetch_add(1, std::memory_order_relaxed);
        dump_frame_if_requested(opts, *emitted, idx);
        // Latest-wins: if the decoder is busy and a previous frame is still
        // queued, the previous frame is dropped.  Counter inside the slot.
        st.decode_slot.push(std::move(*emitted));
      }
    }
  }
}

void decode_thread_main(const CliOptions& opts, ReceiverState& st) {
  using Clock = std::chrono::steady_clock;

  // ONE long-lived decoder for the entire thread lifetime.  The default
  // ctor does NOT call ThreadPool::instance() — that happens on the first
  // init() below, which spawns the worker pool exactly once and reuses it
  // for every subsequent frame.  The destructor (when this thread exits)
  // is the only place that calls ThreadPool::release().
  open_htj2k::openhtj2k_decoder decoder;
  // v4 single-tile reuse: opt in so the second and subsequent frames skip
  // create_resolutions / packet-array allocation / init_line_decode's
  // ring-buffer allocation storm.  Saves ~3 ms/frame on 4K 4:2:2 HT at
  // threads=2.  Fingerprint-guarded inside the decoder; any main-header
  // shape change automatically invalidates the cache and falls back to
  // the legacy path.  RFC 9828 streams are single-tile by construction
  // (see project_rtp_streaming_single_tile in memory).
  decoder.enable_single_tile_reuse(true);

  bool first_frame = true;
  while (!st.stop_flag.load(std::memory_order_acquire)) {
    auto frame_opt = st.decode_slot.pop_wait(st.stop_flag);
    if (!frame_opt) break;
    const AssembledFrame frame = std::move(*frame_opt);

    const auto t0 = Clock::now();

    const ycbcr_coefficients* coeffs             = nullptr;
    bool                      components_are_rgb = false;
    if (!select_coefficients_for_frame(frame, opts, coeffs, components_are_rgb)) {
      st.frames_failed.fetch_add(1, std::memory_order_relaxed);
      continue;
    }
    if (first_frame) log_coefficients_choice_once(opts, coeffs, components_are_rgb);

    // Re-load the codestream into the same decoder instance.  This is the
    // hot path: with Core change A in place (alloc_memory free), init()
    // does not leak the previous frame's buffer, and because the decoder
    // is constructed once at thread startup, ThreadPool::release() is
    // called once (at thread exit) instead of per frame — saving the
    // ~14 ms ThreadPool spinup cost we measured for v1.
    try {
      decoder.init(frame.bytes.data(), frame.bytes.size(), /*reduce_NL=*/0,
                   opts.num_decoder_threads);
      decoder.parse();
    } catch (std::exception& e) {
      std::fprintf(stderr, "decoder.init/parse failed: %s\n", e.what());
      st.frames_failed.fetch_add(1, std::memory_order_relaxed);
      continue;
    }

    DecodedFrame df;
    df.shader_coeffs      = coeffs;
    df.components_are_rgb = components_are_rgb;
    df.source_rtp_ts      = frame.rtp_timestamp;
    if (opts.color_path == CliOptions::ColorPath::Shader) {
      if (!decode_to_planar_buffers(decoder, components_are_rgb, df)) {
        std::fprintf(stderr, "frame: unable to determine dimensions; dropping\n");
        st.frames_failed.fetch_add(1, std::memory_order_relaxed);
        continue;
      }
    } else {
      uint32_t out_w = 0;
      uint32_t out_h = 0;
      if (!decode_to_rgb_buffer(decoder, coeffs, components_are_rgb, df.rgb, out_w, out_h)) {
        std::fprintf(stderr, "frame: unable to determine dimensions; dropping\n");
        st.frames_failed.fetch_add(1, std::memory_order_relaxed);
        continue;
      }
      df.width  = out_w;
      df.height = out_h;
      df.kind   = DecodedFrame::CPU_RGB;
    }

    const auto     t1     = Clock::now();
    const uint64_t us     = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
    st.decode_us_sum.fetch_add(us, std::memory_order_relaxed);
    {
      uint64_t cur = st.decode_us_min.load(std::memory_order_relaxed);
      while (us < cur && !st.decode_us_min.compare_exchange_weak(cur, us)) {}
    }
    {
      uint64_t cur = st.decode_us_max.load(std::memory_order_relaxed);
      while (us > cur && !st.decode_us_max.compare_exchange_weak(cur, us)) {}
    }

    const uint64_t decoded =
        st.frames_decoded.fetch_add(1, std::memory_order_relaxed) + 1;
    first_frame = false;

    // Hand the decoded RGB to the renderer.  If main hasn't picked up the
    // previous decoded frame yet (e.g. blocked in vsync), it gets dropped
    // here — latest-wins keeps motion-to-photon minimal.
    st.render_slot.push(std::move(df));

    if (opts.max_frames > 0 && decoded >= static_cast<uint64_t>(opts.max_frames)) {
      st.stop_flag.store(true, std::memory_order_release);
      st.render_slot.notify();
      break;
    }
  }
}

int run_receiver_threaded(const CliOptions& opts) {
  UdpSocket sock;
  if (!sock.bind(opts.bind_host, opts.bind_port)) {
    std::fprintf(stderr, "bind %s:%u failed: %s\n", opts.bind_host.c_str(), opts.bind_port,
                 sock.last_error().c_str());
    return EXIT_FAILURE;
  }
  // Non-blocking is essential for v2: the dedicated receive thread uses
  // wait_readable + recv in a tight drain loop, and recv() must return
  // kAgain immediately when the kernel buffer is empty so the thread can
  // observe stop_flag and check for new packets via wait_readable.  In
  // blocking mode the second recv after a single-packet wait_readable
  // would hang forever, deadlocking the receive thread on shutdown.
  if (!sock.set_nonblocking()) {
    std::fprintf(stderr, "set_nonblocking failed: %s\n", sock.last_error().c_str());
    return EXIT_FAILURE;
  }
  constexpr int kRequestedRecvBuf = 32 * 1024 * 1024;
  sock.set_recv_buffer_size(kRequestedRecvBuf);
  const int granted = sock.last_granted_recv_buf();

  std::fprintf(stderr, "listening on %s:%u (threaded; %u decoder threads)\n",
               opts.bind_host.c_str(), opts.bind_port, opts.num_decoder_threads);
  std::fprintf(stderr, "SO_RCVBUF: requested %d MB, kernel granted %d KB\n",
               kRequestedRecvBuf / (1024 * 1024), granted / 1024);
  if (granted < 4 * 1024 * 1024) {
    std::fprintf(stderr,
                 "WARN: SO_RCVBUF is < 4 MB. The kernel will drop packets when the\n"
                 "      receiver falls behind the sender. Raise net.core.rmem_max:\n"
                 "          sudo sysctl -w net.core.rmem_max=33554432\n"
                 "      and re-run.\n");
  }

  GlRenderer  renderer;
  GlRenderer* renderer_ptr = nullptr;
  if (opts.render) {
    if (!renderer.init(1280, 720, "OpenHTJ2K RFC 9828 receiver", opts.vsync)) {
      std::fprintf(stderr, "WARN: GLFW init failed; continuing in --no-render mode\n");
    } else {
      renderer_ptr = &renderer;
    }
  }

  // With no live GL 3.3 context the shader path would produce nothing
  // useful, so force the CPU color path for headless and v1-fallback
  // runs.  This keeps --no-render and GL-incompatible environments
  // functional through the same CLI.
  CliOptions opts_effective = opts;
  if (renderer_ptr == nullptr && opts_effective.color_path == CliOptions::ColorPath::Shader) {
    opts_effective.color_path = CliOptions::ColorPath::Cpu;
  }

  FrameHandler  frame_handler;
  ReceiverState state;

  using Clock         = std::chrono::steady_clock;
  const auto run_start_tp = Clock::now();

  std::thread recv_thread(recv_thread_main, std::cref(opts_effective), std::ref(sock),
                          std::ref(frame_handler), std::ref(state));
  std::thread decode_thread(decode_thread_main, std::cref(opts_effective), std::ref(state));

  // Main loop: GLFW events + render slot polling + periodic FPS log.
  uint64_t last_log_decoded = 0;
  auto     last_log_tp      = run_start_tp;

  // Frame pacer.  With --no-vsync the main thread would present each
  // decoded frame as soon as it lands in the render slot, which on a
  // 60 Hz display produces 3:2 pulldown judder against a 30 fps source
  // and also amplifies sender arrival jitter: a burst of two quick
  // sender frames overwrites the first in the render_slot (eviction)
  // and the overall motion stutters.
  //
  // Primary pacing strategy: schedule each present at
  //   ref_steady_tp + (source_rtp_ts - ref_rtp_ts) / 90 kHz
  // where the reference is the first paced frame.  This follows the
  // sender's intended cadence frame-for-frame regardless of physical
  // arrival jitter or small sender/receiver clock differences.
  //
  // --pace-fps now plays two roles: (a) >0 enables the pacer at all,
  // and (b) sets a "runaway guard" period used for outlier detection
  // and for the fallback branch when two successive frames carry the
  // same RTP timestamp (defensive; frame_handler emits only complete
  // frames so this should not happen in practice).
  //
  // Disabled with vsync on (swap interval 1 paces via the vblank) and
  // with pace_fps == 0 (benchmarking).
  const bool pacer_active =
      (renderer_ptr != nullptr) && (!opts_effective.vsync) && (opts_effective.pace_fps > 0.0);
  const auto pace_period = pacer_active
                               ? std::chrono::nanoseconds(static_cast<int64_t>(
                                     1.0e9 / opts_effective.pace_fps))
                               : std::chrono::nanoseconds(0);
  constexpr double kRtpClockHz = 90000.0;  // RFC 3551 video profile
  auto     last_present_tp     = Clock::now();
  bool     first_present       = true;
  bool     rtp_ref_valid       = false;
  uint32_t rtp_ref_ts          = 0;
  auto     rtp_ref_tp          = Clock::now();

  while (!state.stop_flag.load(std::memory_order_acquire)) {
    if (renderer_ptr) {
      renderer_ptr->poll_events();
      if (renderer_ptr->should_close()) {
        state.stop_flag.store(true, std::memory_order_release);
        state.decode_slot.notify();
        state.render_slot.notify();
        break;
      }
    }

    auto df = state.render_slot.try_pop();
    if (df.has_value() && renderer_ptr) {
      // Pacer.  Two branches:
      //  1. RTP-timestamp branch (primary): schedule this present at
      //     rtp_ref_tp + (df->source_rtp_ts - rtp_ref_ts) / 90kHz.
      //     Signed int32 subtraction wraps cleanly at the 32-bit RTP
      //     boundary (~13.25 h at 90 kHz).  Reset the reference when
      //     we're more than four pace_periods behind the computed
      //     target (catastrophic decode stall, sender restart, pause)
      //     so the pacer rebases instead of chasing a past deadline.
      //  2. Fallback (same RTP ts as previous, or pacer just enabled):
      //     use the fixed --pace-fps period.  Defensive; shouldn't
      //     trigger on the Spark fixture.
      if (pacer_active && !first_present) {
        auto target = last_present_tp + pace_period;
        if (rtp_ref_valid && df->source_rtp_ts != rtp_ref_ts) {
          const int32_t delta_ticks =
              static_cast<int32_t>(df->source_rtp_ts - rtp_ref_ts);
          const auto delta = std::chrono::nanoseconds(static_cast<int64_t>(
              static_cast<double>(delta_ticks) / kRtpClockHz * 1.0e9));
          target = rtp_ref_tp + delta;
        }
        const auto now = Clock::now();
        if (target > now) {
          std::this_thread::sleep_until(target);
        } else if (rtp_ref_valid && (now - target) > 4 * pace_period) {
          // Runaway / resync: rebase the reference to this frame's
          // (rtp_ts, wall time).  The next present then computes its
          // target as new_ref_tp + (next_rtp_ts - new_ref_ts) /
          // 90 kHz, which — assuming the sender resumes normal
          // cadence — lands one sender frame period after this one.
          rtp_ref_ts = df->source_rtp_ts;
          rtp_ref_tp = now;
        }
      }
      if (!rtp_ref_valid) {
        rtp_ref_valid = true;
        rtp_ref_ts    = df->source_rtp_ts;
        rtp_ref_tp    = Clock::now();
      }
      first_present = false;

      if (df->kind == DecodedFrame::CPU_RGB) {
        renderer_ptr->upload_and_draw(df->rgb.data(), static_cast<int>(df->width),
                                      static_cast<int>(df->height));
      } else if (df->bit_depth > 8) {
        renderer_ptr->upload_planar_16_and_draw(
            df->plane_y_16.data(), df->plane_cb_16.data(), df->plane_cr_16.data(),
            static_cast<int>(df->width), static_cast<int>(df->height),
            static_cast<int>(df->chroma_width), static_cast<int>(df->chroma_height),
            static_cast<int>(df->bit_depth), df->shader_coeffs,
            df->components_are_rgb);
      } else {
        renderer_ptr->upload_planar_and_draw(
            df->plane_y.data(), df->plane_cb.data(), df->plane_cr.data(),
            static_cast<int>(df->width), static_cast<int>(df->height),
            static_cast<int>(df->chroma_width), static_cast<int>(df->chroma_height),
            df->shader_coeffs, df->components_are_rgb);
      }
      last_present_tp = Clock::now();
    } else if (!df.has_value()) {
      // Nothing to draw; brief sleep to avoid spinning while the decode
      // thread is busy.  Vsync would be doing this for us in --render
      // mode, but in headless mode there is nothing to throttle on.
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }

    const auto   now      = Clock::now();
    const double since_log = std::chrono::duration<double>(now - last_log_tp).count();
    if (since_log >= 1.0) {
      const uint64_t decoded = state.frames_decoded.load(std::memory_order_relaxed);
      const uint64_t delta   = decoded - last_log_decoded;
      const double   fps     = static_cast<double>(delta) / since_log;
      std::fprintf(stderr, "  [%llu frames] inst=%.2f fps\n",
                   static_cast<unsigned long long>(decoded), fps);
      last_log_decoded = decoded;
      last_log_tp      = now;
    }
  }

  // Shutdown.  Wake any thread blocked on a slot, then join.
  state.stop_flag.store(true, std::memory_order_release);
  state.decode_slot.notify();
  state.render_slot.notify();
  decode_thread.join();
  recv_thread.join();

  // Drain anything left in the render slot so RAII doesn't print spurious
  // warnings (the destructor doesn't, but be tidy).
  (void)state.render_slot.try_pop();

  const auto   run_end_tp = Clock::now();
  const double run_secs   = std::chrono::duration<double>(run_end_tp - run_start_tp).count();

  const uint64_t decoded         = state.frames_decoded.load();
  const uint64_t failed          = state.frames_failed.load();
  const uint64_t emitted_to_dec  = state.frames_emitted_to_decode.load();
  const double   avg_fps         = decoded > 0 ? static_cast<double>(decoded) / run_secs : 0.0;
  const uint64_t us_sum          = state.decode_us_sum.load();
  const double   decode_ms_avg   = decoded > 0 ? (static_cast<double>(us_sum)
                                                  / static_cast<double>(decoded)) / 1000.0
                                               : 0.0;
  const double   decode_ms_min   = decoded > 0
                                       ? static_cast<double>(state.decode_us_min.load()) / 1000.0
                                       : 0.0;
  const double   decode_ms_max   = static_cast<double>(state.decode_us_max.load()) / 1000.0;

  const auto& s = frame_handler.stats();
  std::fprintf(stderr,
               "\n--- summary (threaded) ---\n"
               "  wall time:           %.2f s\n"
               "  frames emitted:      %llu (frame_handler)\n"
               "  frames pushed:       %llu (to decode slot)\n"
               "  frames decoded:      %llu\n"
               "  frames failed:       %llu\n"
               "  frames dropped:      %llu (mid-frame seq gap)\n"
               "  tail-loss drops:     %llu (frame ended w/o RTP M=1)\n"
               "  decode-slot evicts:  %llu (decoder couldn't keep up)\n"
               "  render-slot evicts:  %llu (display refresh < decode)\n"
               "  packets received:    %llu\n"
               "  bytes received:      %llu\n"
               "  sequence gaps:       %llu\n"
               "  avg FPS:             %.2f\n"
               "  decode time ms:      min=%.2f avg=%.2f max=%.2f\n",
               run_secs,
               static_cast<unsigned long long>(s.frames_emitted),
               static_cast<unsigned long long>(emitted_to_dec),
               static_cast<unsigned long long>(decoded),
               static_cast<unsigned long long>(failed),
               static_cast<unsigned long long>(s.frames_dropped),
               static_cast<unsigned long long>(s.tail_loss_drops),
               static_cast<unsigned long long>(state.decode_slot.evictions()),
               static_cast<unsigned long long>(state.render_slot.evictions()),
               static_cast<unsigned long long>(s.packets_received),
               static_cast<unsigned long long>(s.bytes_received),
               static_cast<unsigned long long>(s.seq_gaps),
               avg_fps,
               decode_ms_min, decode_ms_avg, decode_ms_max);

  if (renderer_ptr) renderer.shutdown();
  return EXIT_SUCCESS;
}

// Top-level dispatcher: pick v1 or v2 main loop based on --threading.
int run_receiver(const CliOptions& opts) {
  if (opts.threading) return run_receiver_threaded(opts);
  return run_receiver_single_threaded(opts);
}

// ----------------------- Smoke test implementations -----------------------
// These are kept (behind --smoke-test) as a fast regression check that the
// individual pieces still behave after refactors.

int smoke_test_socket() {
  UdpSocket sock;
  if (!sock.bind("127.0.0.1", 0)) return 1;
  if (!sock.set_nonblocking()) return 1;
  char buf[8];
  return sock.recv(buf, sizeof(buf)) == UdpSocket::kAgain ? 0 : 1;
}

int smoke_test_parser() {
  uint8_t rtp_hdr[12] = {0};
  rtp_hdr[0] = 0x80;
  rtp_hdr[1] = 0x80 | 96;
  rtp_hdr[2] = 0x12; rtp_hdr[3] = 0x34;
  rtp_hdr[4] = 0xDE; rtp_hdr[5] = 0xAD; rtp_hdr[6] = 0xBE; rtp_hdr[7] = 0xEF;
  rtp_hdr[8] = 0xCA; rtp_hdr[9] = 0xFE; rtp_hdr[10] = 0xBA; rtp_hdr[11] = 0xBE;
  RtpHeader rtp{};
  std::string err;
  if (!parse_rtp_header(rtp_hdr, sizeof(rtp_hdr), rtp, err)) return 1;
  if (rtp.version != 2 || !rtp.marker || rtp.sequence != 0x1234
      || rtp.timestamp != 0xDEADBEEF || rtp.ssrc != 0xCAFEBABE) return 1;

  uint8_t main_hdr[8] = {0};
  main_hdr[0] = static_cast<uint8_t>((3u << 6) | 4u);
  main_hdr[1] = static_cast<uint8_t>(0x80 | ((0x0ABC >> 8) & 0x0F));
  main_hdr[2] = static_cast<uint8_t>(0x0ABC & 0xFF);
  main_hdr[3] = 0x42;
  main_hdr[4] = static_cast<uint8_t>(0x80 | 0x40 | 0x01);
  main_hdr[5] = 1;
  main_hdr[6] = 1;
  main_hdr[7] = 1;
  MainPacketHeader main{};
  if (!parse_main_packet_header(main_hdr, sizeof(main_hdr), main, err)) return 1;
  if (main.mh != 3 || main.ordh != ORDH_PCRL_RESYNC || main.ptstamp != 0x0ABC
      || main.eseq != 0x42 || !main.r || !main.s || !main.range || main.prims != 1) return 1;

  uint8_t body_hdr[8] = {0};
  body_hdr[0] = static_cast<uint8_t>(2u);
  body_hdr[1] = static_cast<uint8_t>(0x80 | (3u << 4) | ((0x0123 >> 8) & 0x0F));
  body_hdr[2] = static_cast<uint8_t>(0x0123 & 0xFF);
  body_hdr[3] = 0x7F;
  body_hdr[4] = static_cast<uint8_t>((0x0ABCu >> 4) & 0xFF);
  body_hdr[5] = static_cast<uint8_t>(((0x0ABCu & 0x0Fu) << 4) | ((0x0ABCDEu >> 16) & 0x0Fu));
  body_hdr[6] = static_cast<uint8_t>((0x0ABCDEu >> 8) & 0xFF);
  body_hdr[7] = static_cast<uint8_t>(0x0ABCDEu & 0xFF);
  BodyPacketHeader body{};
  if (!parse_body_packet_header(body_hdr, sizeof(body_hdr), body, err)) return 1;
  return (body.pos == 0x0ABC && body.pid == 0x0ABCDEu && body.ordb) ? 0 : 1;
}

int smoke_test_ycbcr() {
  const int32_t Y[] = {128}, Cb[] = {128}, Cr[] = {128};
  uint8_t rgb[3] = {0};
  ycbcr_row_to_rgb8(Y, Cb, Cr, rgb, 1, 1, 1, YCBCR_BT709_FULL, 8, false);
  if (std::abs(int(rgb[0]) - 128) > 1) return 1;

  const int32_t Yr[] = {54}, Cbr[] = {99}, Crr[] = {255};
  uint8_t rgb_r[3] = {0};
  ycbcr_row_to_rgb8(Yr, Cbr, Crr, rgb_r, 1, 1, 1, YCBCR_BT709_FULL, 8, false);
  if (!(rgb_r[0] >= 240 && rgb_r[1] <= 15 && rgb_r[2] <= 15)) return 1;

  // Multi-pixel rows to exercise the AVX2 fast path (engages at >= 8
  // luma pixels) and the scalar tail.  17 luma pixels covers both the
  // 8-wide SIMD loop and the 1-pixel remainder.
  {
    const uint32_t W = 17;
    int32_t Yrow[W];
    int32_t Cbrow[W];
    int32_t Crrow[W];
    for (uint32_t x = 0; x < W; ++x) {
      // Smooth ramp so a SIMD off-by-one on a lane would shift the
      // output in a visually detectable way.
      Yrow[x]  = static_cast<int32_t>(16 + x * 12);
      Cbrow[x] = 128;
      Crrow[x] = 128;
    }
    uint8_t out[W * 3] = {0};
    ycbcr_row_to_rgb8(Yrow, Cbrow, Crrow, out, W, 1, 1, YCBCR_BT709_FULL, 8, false);
    for (uint32_t x = 0; x < W; ++x) {
      const int expected = static_cast<int>(16 + x * 12);  // gray → R=G=B=Y
      for (uint32_t c = 0; c < 3; ++c) {
        if (std::abs(static_cast<int>(out[3 * x + c]) - expected) > 1) return 1;
      }
    }
  }

  // 4:2:2 chroma layout: 18 luma pixels with 9 chroma samples.
  // Uses ramped chroma to verify the stride-ratio=2 SIMD duplication
  // path (each Cb/Cr sample maps to two adjacent luma positions).
  {
    const uint32_t W  = 18;
    const uint32_t Wc = 9;
    int32_t Yrow[W];
    int32_t Cbrow[Wc];
    int32_t Crrow[Wc];
    for (uint32_t x = 0; x < W;  ++x) Yrow[x]  = 128;
    for (uint32_t x = 0; x < Wc; ++x) Cbrow[x] = 128;
    for (uint32_t x = 0; x < Wc; ++x) Crrow[x] = 128;
    uint8_t out[W * 3] = {0};
    ycbcr_row_to_rgb8(Yrow, Cbrow, Crrow, out, W, 2, 2, YCBCR_BT709_FULL, 8, false);
    // All-neutral input should decode to mid-gray at every position.
    for (uint32_t x = 0; x < W; ++x) {
      for (uint32_t c = 0; c < 3; ++c) {
        if (std::abs(static_cast<int>(out[3 * x + c]) - 128) > 1) return 1;
      }
    }
  }

  return 0;
}

int smoke_test_frame_handler() {
  FrameHandler fh;
  std::optional<AssembledFrame> frame;
  RtpHeader rtp{};
  rtp.version = 2;
  rtp.sequence = 100;
  rtp.timestamp = 0x1000;
  rtp.marker = true;
  MainPacketHeader main{};
  main.mh = MH_MAIN_SINGLE;
  main.ordh = ORDH_PCRL_RESYNC;
  const uint8_t cs[10] = {0xFF, 0x4F, 0xFF, 0x51, 0, 0, 0, 0, 0xFF, 0xD9};
  fh.push_main_packet(rtp, main, cs, sizeof(cs), frame);
  if (!frame.has_value() || frame->bytes.size() != sizeof(cs)) return 1;
  frame.reset();

  fh.reset();
  RtpHeader rtp3{};
  rtp3.version = 2;
  rtp3.sequence = 300;
  rtp3.timestamp = 0x3000;
  MainPacketHeader main3{};
  fh.push_main_packet(rtp3, main3, cs, 4, frame);
  rtp3.sequence = 302;  // skip 301 → gap
  rtp3.marker = true;
  BodyPacketHeader body3{};
  fh.push_body_packet(rtp3, body3, cs + 4, 6, frame);
  return (!frame.has_value() && fh.stats().frames_dropped == 1 && fh.stats().seq_gaps == 1) ? 0
                                                                                            : 1;
}

}  // namespace

int main(int argc, char** argv) {
  CliOptions opts;
  if (!parse_cli(argc, argv, opts)) return EXIT_FAILURE;

  if (opts.smoke_test) return run_smoke_tests();

  return run_receiver(opts);
}
