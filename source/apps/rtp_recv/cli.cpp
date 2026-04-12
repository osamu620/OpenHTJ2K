// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.
//
// Licensed under the same BSD 3-Clause terms as the rest of OpenHTJ2K.

#include "cli.hpp"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "ycbcr_rgb.hpp"

namespace open_htj2k::rtp_recv {

namespace {

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

}  // namespace

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
      "  --colorspace <name>      S=0 fallback: bt709 | bt601 | bt2020 | rgb\n"
      "  --range <name>           S=0 fallback: full | narrow (default full)\n"
      "  --threads <N>            Decoder thread count (default 4; 0 = hardware)\n"
      "  --color-path {shader|cpu} YCbCr->RGB on GPU (default) or CPU fallback\n"
      "  --transfer {auto|gamma|pq|hlg}\n"
      "                           Inverse EOTF (default auto, reads H.273 TRANS)\n"
      "  --display-primaries {bt709|bt2020}\n"
      "                           Target primaries for gamut stage (default bt709)\n"
      "  --display-encoding {srgb|gamma22|linear}\n"
      "                           Framebuffer encoding (default srgb)\n"
      "  --pace-fps <N>           Frame-pacing target (default 30; 0 = disabled)\n"
      "                           Only active when vsync is off\n"
      "  --smoke-test             Run internal unit smoke tests and exit\n",
      argv0);
}

bool parse_cli(int argc, char* argv[], CliOptions& opt) {
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
  if (const char* v = get_arg(argc, argv, "--transfer")) {
    if (std::strcmp(v, "auto") == 0)       opt.transfer = CliOptions::TransferMode::Auto;
    else if (std::strcmp(v, "gamma") == 0) opt.transfer = CliOptions::TransferMode::Gamma;
    else if (std::strcmp(v, "pq") == 0)    opt.transfer = CliOptions::TransferMode::Pq;
    else if (std::strcmp(v, "hlg") == 0)   opt.transfer = CliOptions::TransferMode::Hlg;
    else {
      std::fprintf(stderr, "ERROR: --transfer must be auto|gamma|pq|hlg\n");
      return false;
    }
  }
  if (const char* v = get_arg(argc, argv, "--display-primaries")) {
    if (std::strcmp(v, "bt709") == 0)       opt.display_primaries = CliOptions::DisplayPrimaries::Bt709;
    else if (std::strcmp(v, "bt2020") == 0) opt.display_primaries = CliOptions::DisplayPrimaries::Bt2020;
    else {
      std::fprintf(stderr, "ERROR: --display-primaries must be bt709|bt2020\n");
      return false;
    }
  }
  if (const char* v = get_arg(argc, argv, "--display-encoding")) {
    if (std::strcmp(v, "srgb") == 0)          opt.display_encoding = CliOptions::DisplayEncoding::Srgb;
    else if (std::strcmp(v, "gamma22") == 0)  opt.display_encoding = CliOptions::DisplayEncoding::Gamma22;
    else if (std::strcmp(v, "linear") == 0)   opt.display_encoding = CliOptions::DisplayEncoding::Linear;
    else {
      std::fprintf(stderr, "ERROR: --display-encoding must be srgb|gamma22|linear\n");
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
      opt.s0_fallback = range_full ? &YCBCR_BT2020_FULL : &YCBCR_BT2020_NARROW;
      opt.s0_label    = range_full ? "bt2020-full" : "bt2020-narrow";
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

}  // namespace open_htj2k::rtp_recv
