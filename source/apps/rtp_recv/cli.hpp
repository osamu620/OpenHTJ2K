// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.
//
// Licensed under the same BSD 3-Clause terms as the rest of OpenHTJ2K.

#pragma once

#include <cstdint>
#include <string>

#include "color_pipeline.hpp"
#include "ycbcr_rgb.hpp"

namespace open_htj2k::rtp_recv {

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
  // Default 4 matches the benchmarked optimum on broadcast 4K HT codestreams
  // after component-parallel IDWT dispatch landed (earlier code peaked at 2
  // threads because IDWT was main-thread serial; component-parallel IDWT
  // pushes the peak to threads=4..6 on 4K 4:2:2 HT).  Override per
  // machine/content via --threads.
  uint32_t    num_decoder_threads = 4;
  // Color conversion path.  "shader" (default) does YCbCr->RGB in the
  // fragment shader — the CPU only shifts samples to 8-bit per plane.
  // "cpu" runs the legacy scalar ycbcr_row_to_rgb8 on the decode thread
  // and uploads interleaved RGB.  Forced to "cpu" if the renderer fails
  // to create a GL 3.3 core context (fallback for headless / GL-limited
  // environments).
  enum class ColorPath : uint8_t { Shader, Cpu };
  ColorPath   color_path = ColorPath::Shader;
  // HDR colour pipeline overrides.  `auto` for transfer reads the
  // RFC 9828 Main Packet TRANS field and maps per H.273 Table 3:
  // TRANS=1 -> gamma2.2, TRANS=16 -> PQ, TRANS=18 -> HLG.  Any other
  // value falls through to `transfer_fallback` (default gamma2.2).
  // `display_primaries` selects the gamut matrix; `bt709` is the normal
  // target and `bt2020` is an identity stub for a future HDR output
  // path.  `display_encoding` selects the final non-linear write: sRGB
  // is the normal target, `gamma22` is a cheaper debugging alternative,
  // and `linear` writes linear light directly (diagnostic).
  enum class TransferMode : uint8_t { Auto, Gamma, Pq, Hlg };
  enum class DisplayPrimaries : uint8_t { Bt709, Bt2020 };
  enum class DisplayEncoding : uint8_t { Srgb, Gamma22, Linear };
  TransferMode     transfer          = TransferMode::Auto;
  int              transfer_fallback = TRANSFER_GAMMA22;  // used when Auto + S=0 or unknown TRANS
  DisplayPrimaries display_primaries = DisplayPrimaries::Bt709;
  DisplayEncoding  display_encoding  = DisplayEncoding::Srgb;
  // Frame-pacing target in fps, active only when vsync is off.  A 30 fps
  // source on a 60 Hz display without pacing shows 3:2 pulldown judder
  // because decoded frames are presented as soon as they're ready, which
  // drifts relative to vblank — some frames land on one vblank, others
  // on the next.  Pacing holds the next present until last_present_tp
  // + 1/pace_fps, giving a stable 2:2 cadence on 60 Hz.  0 disables the
  // pacer (useful for benchmarking and --no-render runs).
  double      pace_fps = 30.0;
};

void print_usage(const char* argv0);
bool parse_cli(int argc, char* argv[], CliOptions& opts);

}  // namespace open_htj2k::rtp_recv
