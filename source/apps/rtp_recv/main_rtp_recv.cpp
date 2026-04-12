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

#include <cstdlib>

#include "cli.hpp"
#include "pipeline_multi_threaded.hpp"
#include "pipeline_single_threaded.hpp"
#include "smoke_tests.hpp"

using namespace open_htj2k::rtp_recv;

static int run_receiver(const CliOptions& opts) {
  if (opts.threading) return run_receiver_threaded(opts);
  return run_receiver_single_threaded(opts);
}

int main(int argc, char** argv) {
  CliOptions opts;
  if (!parse_cli(argc, argv, opts)) return EXIT_FAILURE;

  if (opts.smoke_test) return run_smoke_tests(opts);

  return run_receiver(opts);
}
