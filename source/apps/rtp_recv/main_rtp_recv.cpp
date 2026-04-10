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

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <GLFW/glfw3.h>

#include "decoder.hpp"
#include "rfc9828_parser.hpp"
#include "rtp_socket.hpp"
#include "ycbcr_rgb.hpp"

namespace {

int smoke_test_socket() {
  open_htj2k::rtp_recv::UdpSocket sock;
  if (!sock.bind("127.0.0.1", 0)) {
    std::fprintf(stderr, "bind failed: %s\n", sock.last_error().c_str());
    return 1;
  }
  if (!sock.set_nonblocking()) {
    std::fprintf(stderr, "set_nonblocking failed: %s\n", sock.last_error().c_str());
    return 1;
  }
  char buf[8];
  auto n = sock.recv(buf, sizeof(buf));
  if (n != open_htj2k::rtp_recv::UdpSocket::kAgain) {
    std::fprintf(stderr, "expected kAgain from empty non-blocking socket, got %zd\n", n);
    return 1;
  }
  return 0;
}

int smoke_test_parser() {
  using namespace open_htj2k::rtp_recv;

  // Synthesize a 12-byte RTP header: V=2, no pad/ext/CSRC, M=1, PT=96,
  // seq=0x1234, ts=0xDEADBEEF, SSRC=0xCAFEBABE.
  uint8_t rtp_hdr[12] = {0};
  rtp_hdr[0]  = 0x80;                    // V=2, P=0, X=0, CC=0
  rtp_hdr[1]  = 0x80 | 96;               // M=1, PT=96
  rtp_hdr[2]  = 0x12; rtp_hdr[3]  = 0x34;
  rtp_hdr[4]  = 0xDE; rtp_hdr[5]  = 0xAD; rtp_hdr[6] = 0xBE; rtp_hdr[7] = 0xEF;
  rtp_hdr[8]  = 0xCA; rtp_hdr[9]  = 0xFE; rtp_hdr[10] = 0xBA; rtp_hdr[11] = 0xBE;

  RtpHeader rtp{};
  std::string err;
  if (!parse_rtp_header(rtp_hdr, sizeof(rtp_hdr), rtp, err)) {
    std::fprintf(stderr, "rtp parse failed: %s\n", err.c_str());
    return 1;
  }
  if (rtp.version != 2 || !rtp.marker || rtp.payload_type != 96 || rtp.sequence != 0x1234
      || rtp.timestamp != 0xDEADBEEF || rtp.ssrc != 0xCAFEBABE || rtp.payload_offset != 12) {
    std::fprintf(stderr, "rtp parse wrong fields\n");
    return 1;
  }

  // Main packet: MH=3 (single), TP=0, ORDH=4 (PCRL+resync), P=1, XTRAC=0,
  // PTSTAMP=0x0ABC, ESEQ=0x42, R=1, S=1, C=0, RANGE=1, PRIMS=1, TRANS=1, MAT=1.
  uint8_t main_hdr[8] = {0};
  main_hdr[0] = static_cast<uint8_t>((3u << 6) | (0u << 3) | 4u);
  main_hdr[1] = static_cast<uint8_t>(0x80 | (0u << 4) | ((0x0ABC >> 8) & 0x0F));
  main_hdr[2] = static_cast<uint8_t>(0x0ABC & 0xFF);
  main_hdr[3] = 0x42;
  main_hdr[4] = static_cast<uint8_t>(0x80 | 0x40 | 0x00 | 0x01);  // R=1 S=1 C=0 RSVD=0 RANGE=1
  main_hdr[5] = 1;
  main_hdr[6] = 1;
  main_hdr[7] = 1;

  MainPacketHeader main{};
  if (!parse_main_packet_header(main_hdr, sizeof(main_hdr), main, err)) {
    std::fprintf(stderr, "main parse failed: %s\n", err.c_str());
    return 1;
  }
  if (main.mh != 3 || main.tp != 0 || main.ordh != ORDH_PCRL_RESYNC || !main.p
      || main.xtrac != 0 || main.ptstamp != 0x0ABC || main.eseq != 0x42 || !main.r || !main.s
      || main.c || !main.range || main.prims != 1 || main.trans != 1 || main.mat != 1
      || main.codestream_offset != 8) {
    std::fprintf(stderr, "main parse wrong fields\n");
    return 1;
  }

  // Body packet: MH=0, TP=0, RES=2, ORDB=1, QUAL=3, PTSTAMP=0x0123, ESEQ=0x7F,
  // POS=0x0ABC (12-bit), PID=0x0ABCDE (20-bit).
  uint8_t body_hdr[8] = {0};
  body_hdr[0] = static_cast<uint8_t>((0u << 6) | (0u << 3) | 2u);
  body_hdr[1] = static_cast<uint8_t>(0x80 | (3u << 4) | ((0x0123 >> 8) & 0x0F));
  body_hdr[2] = static_cast<uint8_t>(0x0123 & 0xFF);
  body_hdr[3] = 0x7F;
  body_hdr[4] = static_cast<uint8_t>((0x0ABCu >> 4) & 0xFF);                  // POS[11:4]
  body_hdr[5] = static_cast<uint8_t>(((0x0ABCu & 0x0Fu) << 4)                 // POS[3:0]
                                     | ((0x0ABCDEu >> 16) & 0x0Fu));          // PID[19:16]
  body_hdr[6] = static_cast<uint8_t>((0x0ABCDEu >> 8) & 0xFF);                // PID[15:8]
  body_hdr[7] = static_cast<uint8_t>(0x0ABCDEu & 0xFF);                       // PID[7:0]

  BodyPacketHeader body{};
  if (!parse_body_packet_header(body_hdr, sizeof(body_hdr), body, err)) {
    std::fprintf(stderr, "body parse failed: %s\n", err.c_str());
    return 1;
  }
  if (body.mh != 0 || body.tp != 0 || body.res != 2 || !body.ordb || body.qual != 3
      || body.ptstamp != 0x0123 || body.eseq != 0x7F || body.pos != 0x0ABC
      || body.pid != 0x0ABCDEu || body.codestream_offset != 8) {
    std::fprintf(stderr, "body parse wrong fields: pos=0x%04x pid=0x%06x\n",
                 static_cast<unsigned>(body.pos), static_cast<unsigned>(body.pid));
    return 1;
  }

  return 0;
}

int smoke_test_ycbcr() {
  using namespace open_htj2k::rtp_recv;

  // 8-bit BT.709 full-range: pure gray (Y=128, Cb=Cr=128) should map to ~(128,128,128).
  const int32_t Y[]  = {128, 128};
  const int32_t Cb[] = {128, 128};
  const int32_t Cr[] = {128, 128};
  uint8_t rgb[6]     = {0};
  ycbcr_row_to_rgb8(Y, Cb, Cr, rgb, /*width=*/2, /*cb_stride=*/1, /*cr_stride=*/1,
                    YCBCR_BT709_FULL, /*depth=*/8, /*is_signed=*/false);
  // Accept ±1 slack for float round-off.
  auto close_to = [](uint8_t v, int target) { return std::abs(int(v) - target) <= 1; };
  if (!close_to(rgb[0], 128) || !close_to(rgb[1], 128) || !close_to(rgb[2], 128)) {
    std::fprintf(stderr, "ycbcr gray: got (%u,%u,%u)\n", rgb[0], rgb[1], rgb[2]);
    return 1;
  }

  // Pure red in YCbCr BT.709 full-range: R=255,G=0,B=0 encodes as Y=54, Cb=99, Cr=255
  // (Cr clips at 255 because the ideal value is 255.5).  Reverse transform should return
  // nearly (255, 0, 0).
  const int32_t Yr[]  = {54};
  const int32_t Cbr[] = {99};
  const int32_t Crr[] = {255};
  uint8_t rgb_r[3]    = {0};
  ycbcr_row_to_rgb8(Yr, Cbr, Crr, rgb_r, 1, 1, 1, YCBCR_BT709_FULL, 8, false);
  if (rgb_r[0] < 240 || rgb_r[1] > 15 || rgb_r[2] > 15) {
    std::fprintf(stderr, "ycbcr red: got (%u,%u,%u)\n", rgb_r[0], rgb_r[1], rgb_r[2]);
    return 1;
  }

  // MAT selector round-trip.
  if (select_coefficients_from_mat(1, true) != &YCBCR_BT709_FULL) return 1;
  if (select_coefficients_from_mat(5, false) != &YCBCR_BT601_NARROW) return 1;
  if (select_coefficients_from_mat(9, true) != nullptr) return 1;  // BT.2020 not yet supported
  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  std::printf("open_htj2k_rtp_recv: scaffold build, GLFW %d.%d.%d\n",
              GLFW_VERSION_MAJOR, GLFW_VERSION_MINOR, GLFW_VERSION_REVISION);

  if (smoke_test_socket() != 0) return EXIT_FAILURE;
  std::printf("udp socket smoke-test OK\n");

  if (smoke_test_parser() != 0) return EXIT_FAILURE;
  std::printf("rfc9828 parser smoke-test OK\n");

  if (smoke_test_ycbcr() != 0) return EXIT_FAILURE;
  std::printf("ycbcr->rgb smoke-test OK\n");

  return 0;
}
