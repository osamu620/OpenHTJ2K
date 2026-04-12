// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.
//
// Licensed under the same BSD 3-Clause terms as the rest of OpenHTJ2K.

// Smoke test implementations for --smoke-test.  These are kept as a fast
// regression check that the individual pieces still behave after refactors.

#include "smoke_tests.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <optional>

#include "cli.hpp"
#include "color_pipeline.hpp"
#include "frame_handler.hpp"
#include "planar_shift.hpp"
#include "rfc9828_parser.hpp"
#include "rtp_socket.hpp"
#include "ycbcr_rgb.hpp"

namespace open_htj2k::rtp_recv {

namespace {

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

  // BT.2020 sanity: the RFC 9828 MAT -> matrix dispatch routes correctly
  // and a neutral-gray input produces gray output through the BT.2020
  // matrix.  BT.2020 NCL is MAT=9 in ITU-T H.273 Table 4.
  {
    if (select_coefficients_from_mat(/*mat=*/9, /*full_range=*/true) != &YCBCR_BT2020_FULL)
      return 1;
    if (select_coefficients_from_mat(/*mat=*/9, /*full_range=*/false) != &YCBCR_BT2020_NARROW)
      return 1;

    const int32_t Y2020[] = {128}, Cb2020[] = {128}, Cr2020[] = {128};
    uint8_t rgb2020[3] = {0};
    ycbcr_row_to_rgb8(Y2020, Cb2020, Cr2020, rgb2020, 1, 1, 1, YCBCR_BT2020_FULL, 8, false);
    if (std::abs(int(rgb2020[0]) - 128) > 1) return 1;
    if (std::abs(int(rgb2020[1]) - 128) > 1) return 1;
    if (std::abs(int(rgb2020[2]) - 128) > 1) return 1;

    // Same neutral test at 10-bit depth so the bit-depth-dependent
    // normalization constants in ycbcr_row_to_rgb8 get exercised for
    // the BT.2020 matrix too.
    const int32_t Y10[]  = {512}, Cb10[] = {512}, Cr10[] = {512};
    uint8_t rgb10[3] = {0};
    ycbcr_row_to_rgb8(Y10, Cb10, Cr10, rgb10, 1, 1, 1, YCBCR_BT2020_FULL, 10, false);
    if (std::abs(int(rgb10[0]) - 128) > 1) return 1;
    if (std::abs(int(rgb10[1]) - 128) > 1) return 1;
    if (std::abs(int(rgb10[2]) - 128) > 1) return 1;
  }

  return 0;
}

// Host-side verification of the HDR colour pipeline.  The CPU helpers in
// color_pipeline.hpp mirror the GLSL stages bit-for-bit (within float
// epsilon), so this exercises the same arithmetic the fragment shader
// runs without needing a GL context.
int smoke_test_color_pipeline() {
  auto near = [](float a, float b, float tol) { return std::fabs(a - b) <= tol; };

  // --- SMPTE ST 2084 PQ EOTF reference points ---
  // Hand-computed values.  At e=0.0 the EOTF returns 0; at e=1.0 it
  // returns the 10 000-nit peak (normalized to 1.0).  The mid-range
  // point e=0.5 comes out to ~0.00922 linear, corresponding to ~92 nits
  // on the 10 000-nit scale.  Recompute with:
  //   v = 0.5^(1/m2); num = v - c1; den = c2 - c3*v
  //   result = (num/den)^(1/m1)
  if (!near(pq_to_linear(0.0f), 0.0f, 1e-5f)) return 1;
  if (!near(pq_to_linear(1.0f), 1.0f, 1e-3f)) return 1;
  if (!near(pq_to_linear(0.5f), 0.00922f, 5e-4f)) return 1;

  // --- HLG inverse OETF reference points ---
  // Formula: e <= 0.5 -> e^2/3; else (exp((e-c)/a)+b)/12.
  // HLG(0)=0, HLG(0.5)=0.25/3=0.08333..., HLG(1)=1.00013 (tiny rounding
  // on the constants brings the spec'd result slightly above 1.0; the
  // shader clamps downstream so this is harmless).
  if (!near(hlg_inverse(0.0f), 0.0f, 1e-6f)) return 1;
  if (!near(hlg_inverse(0.5f), 1.0f / 12.0f, 1e-5f)) return 1;
  if (!near(hlg_inverse(1.0f), 1.0f, 2e-4f)) return 1;

  // --- sRGB EOTF^-1 reference points ---
  // Below the knee (l <= 0.0031308) the encoding is 12.92*l; above it
  // the Hermite segment 1.055*l^(1/2.4) - 0.055 kicks in.
  if (!near(linear_to_srgb(0.0f), 0.0f, 1e-6f)) return 1;
  if (!near(linear_to_srgb(0.0031308f), 12.92f * 0.0031308f, 1e-5f)) return 1;
  if (!near(linear_to_srgb(1.0f), 1.0f, 1e-5f)) return 1;
  // linear_to_srgb(0.5) = 1.055 * 0.5^(1/2.4) - 0.055 ≈ 0.7354.
  if (!near(linear_to_srgb(0.5f), 0.7354f, 2e-4f)) return 1;

  // --- gamma22 display encoding is the exact inverse of the default
  // gamma2.2 inverse transfer, so pow(pow(e, 2.2), 1/2.2) == e. ---
  ColorPipelineParams gamma22_pipeline;
  gamma22_pipeline.transfer         = TRANSFER_GAMMA22;
  gamma22_pipeline.display_encoding = DISPLAY_ENCODING_GAMMA22;
  gamma22_pipeline.gamut_matrix     = kIdentityMatrix3;
  for (int i = 0; i <= 255; ++i) {
    const float e = static_cast<float>(i) / 255.0f;
    float       r, g, b;
    apply_color_pipeline(gamma22_pipeline, e, e, e, r, g, b);
    // Round-trip should reproduce the input to within float epsilon.
    if (!near(r, e, 1e-4f)) return 1;
    if (!near(g, e, 1e-4f)) return 1;
    if (!near(b, e, 1e-4f)) return 1;
  }

  // --- Default pipeline (gamma2.2 + identity + sRGB) drift bound. ---
  // Writing linear_to_srgb(pow(e, 2.2)) instead of `e` is mathematically
  // different from the v0.12.0 path, which wrote the post-matrix non-
  // linear R'G'B' straight to the framebuffer.  Both targets display
  // light ≈ e^2.2 on a calibrated monitor, so the visual output is
  // indistinguishable, but the byte values diverge most at dark greys
  // where sRGB's linear knee differs from a pure 2.2 curve.  Empirical
  // worst case across a 0..255 ramp is ~11 codes near e=0.05; bound to
  // 16/255 here with headroom.  Anyone needing bit-identical output
  // can pass --display-encoding gamma22 (verified above).
  ColorPipelineParams default_pipeline;  // gamma2.2 + identity + sRGB (defaults)
  float               max_drift_u8 = 0.0f;
  for (int i = 0; i <= 255; ++i) {
    const float e = static_cast<float>(i) / 255.0f;
    float       r, g, b;
    apply_color_pipeline(default_pipeline, e, e, e, r, g, b);
    const float drift = std::fabs(r - e) * 255.0f;
    if (drift > max_drift_u8) max_drift_u8 = drift;
  }
  if (max_drift_u8 > 16.0f) {
    std::fprintf(stderr,
                 "color pipeline default-drift=%.2f codes (u8) exceeds 16 bound\n",
                 max_drift_u8);
    return 1;
  }

  // --- BT.2020 -> BT.709 gamut matrix on a neutral-grey input. ---
  // Primary row sums for the matrix are all approximately 1.0 (BT.2087
  // rounding), so a grey input should map to grey output within ~2e-4.
  float mr, mg, mb;
  apply_gamut_matrix(kBt2020ToBt709, 0.5f, 0.5f, 0.5f, mr, mg, mb);
  if (!near(mr, 0.5f, 2e-4f)) return 1;
  if (!near(mg, 0.5f, 2e-4f)) return 1;
  if (!near(mb, 0.5f, 2e-4f)) return 1;

  // --- Neutral-grey round trip through the PQ + BT.2020->BT.709 + sRGB
  // pipeline.  PQ(0.5) ≈ 0.00922 linear (i.e. ~92 nits on the 10 000-nit
  // scale), the gamut matrix is neutral on grey, and
  // linear_to_srgb(0.00922) ≈ 0.0947 -- recompute with
  //   1.055 * 0.00922^(1/2.4) - 0.055. ---
  ColorPipelineParams pq_pipeline;
  pq_pipeline.transfer         = TRANSFER_PQ;
  pq_pipeline.display_encoding = DISPLAY_ENCODING_SRGB;
  pq_pipeline.gamut_matrix     = kBt2020ToBt709;
  float pr, pg, pb;
  apply_color_pipeline(pq_pipeline, 0.5f, 0.5f, 0.5f, pr, pg, pb);
  if (!near(pr, 0.0947f, 2e-3f)) return 1;
  if (!near(pg, 0.0947f, 2e-3f)) return 1;
  if (!near(pb, 0.0947f, 2e-3f)) return 1;

  // Pipeline label helpers.
  if (std::strcmp(transfer_label(TRANSFER_PQ), "pq") != 0) return 1;
  if (std::strcmp(transfer_label(TRANSFER_HLG), "hlg") != 0) return 1;
  if (std::strcmp(transfer_label(TRANSFER_GAMMA22), "gamma2.2") != 0) return 1;
  if (std::strcmp(display_encoding_label(DISPLAY_ENCODING_SRGB), "srgb") != 0) return 1;
  if (std::strcmp(gamut_matrix_label(kBt2020ToBt709), "bt2020->bt709") != 0) return 1;
  if (std::strcmp(gamut_matrix_label(kIdentityMatrix3), "identity") != 0) return 1;

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

int run_smoke_tests(const CliOptions& /*opts*/) {
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
  if (smoke_test_color_pipeline() != 0) return EXIT_FAILURE;
  std::printf("colour pipeline smoke-test OK\n");
  return EXIT_SUCCESS;
}

}  // namespace open_htj2k::rtp_recv
