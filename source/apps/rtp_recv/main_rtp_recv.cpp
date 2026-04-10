// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.
//
// Licensed under the same BSD 3-Clause terms as the rest of OpenHTJ2K.

// RFC 9828 RTP receiver for HTJ2K video with GLFW/OpenGL rendering.
//
// This is the v1 scaffold: it only verifies that the build pipeline wires up
// correctly (GLFW discovery, OpenHTJ2K library link, executable output).  The
// real depacketizer, frame handler, decoder invocation, and renderer land in
// subsequent commits on this branch.

#include <cstdio>

#include <GLFW/glfw3.h>

#include "decoder.hpp"

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  std::printf("open_htj2k_rtp_recv: scaffold build, GLFW %d.%d.%d\n",
              GLFW_VERSION_MAJOR, GLFW_VERSION_MINOR, GLFW_VERSION_REVISION);
  return 0;
}
