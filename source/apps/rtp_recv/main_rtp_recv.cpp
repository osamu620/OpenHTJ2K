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
#include <cstdlib>

#include <GLFW/glfw3.h>

#include "decoder.hpp"
#include "rtp_socket.hpp"

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  std::printf("open_htj2k_rtp_recv: scaffold build, GLFW %d.%d.%d\n",
              GLFW_VERSION_MAJOR, GLFW_VERSION_MINOR, GLFW_VERSION_REVISION);

  // Smoke-test the UDP socket wrapper: bind to an ephemeral loopback port,
  // set non-blocking, confirm recv returns kAgain, then close.
  open_htj2k::rtp_recv::UdpSocket sock;
  if (!sock.bind("127.0.0.1", 0)) {
    std::fprintf(stderr, "bind failed: %s\n", sock.last_error().c_str());
    return EXIT_FAILURE;
  }
  if (!sock.set_nonblocking()) {
    std::fprintf(stderr, "set_nonblocking failed: %s\n", sock.last_error().c_str());
    return EXIT_FAILURE;
  }
  char buf[8];
  auto n = sock.recv(buf, sizeof(buf));
  if (n != open_htj2k::rtp_recv::UdpSocket::kAgain) {
    std::fprintf(stderr, "expected kAgain from empty non-blocking socket, got %zd\n", n);
    return EXIT_FAILURE;
  }
  std::printf("udp socket smoke-test OK\n");
  return 0;
}
