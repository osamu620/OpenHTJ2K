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

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include <GLFW/glfw3.h>

#include "decoder.hpp"
#include "frame_handler.hpp"
#include "frame_pipeline.hpp"
#include "gl_renderer.hpp"
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
  std::string dump_pattern;          // printf-style, e.g. "/tmp/frame_%05d.j2c"
  // S=0 fallback colorspace (used only when Main Packet says S=0).
  // Leave as nullptr to require the stream to carry S=1.
  const ycbcr_coefficients* s0_fallback = nullptr;
  std::string s0_label;  // human-readable form for logging
  bool        smoke_test = false;
  uint32_t    num_decoder_threads = 4;
};

void print_usage(const char* argv0) {
  std::fprintf(
      stderr,
      "Usage: %s [options]\n"
      "  --port <N>             UDP port to bind (default 6000)\n"
      "  --bind <host>          Host/IP to bind (default 0.0.0.0)\n"
      "  --frames <N>           Exit after N successfully decoded frames\n"
      "  --no-render            Do not open a window; pure depacketize+decode\n"
      "  --no-decode            Skip the openhtj2k decoder entirely (capture only)\n"
      "  --dump-codestream <fmt>  printf-style path, e.g. '/tmp/f_%%05d.j2c'\n"
      "  --colorspace <name>    S=0 fallback: bt709 | bt601 (v1) | bt2020 (v2, rejected)\n"
      "  --range <name>         S=0 fallback: full | narrow (default full)\n"
      "  --threads <N>          Decoder thread count (default 4)\n"
      "  --smoke-test           Run internal unit smoke tests and exit\n",
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

bool parse_cli(int argc, char** argv, CliOptions& opt) {
  if (has_flag(argc, argv, "--help") || has_flag(argc, argv, "-h")) {
    print_usage(argv[0]);
    return false;
  }
  opt.smoke_test = has_flag(argc, argv, "--smoke-test");
  opt.render     = !has_flag(argc, argv, "--no-render");
  opt.decode     = !has_flag(argc, argv, "--no-decode");
  if (!opt.decode) opt.render = false;  // can't render without a decoded frame

  if (const char* v = get_arg(argc, argv, "--port"))  opt.bind_port  = static_cast<uint16_t>(std::atoi(v));
  if (const char* v = get_arg(argc, argv, "--bind"))  opt.bind_host  = v;
  if (const char* v = get_arg(argc, argv, "--frames")) opt.max_frames = std::atoi(v);
  if (const char* v = get_arg(argc, argv, "--dump-codestream")) opt.dump_pattern = v;
  if (const char* v = get_arg(argc, argv, "--threads")) opt.num_decoder_threads = static_cast<uint32_t>(std::atoi(v));

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
  if (smoke_test_frame_handler() != 0) return EXIT_FAILURE;
  std::printf("frame_handler smoke-test OK\n");
  return EXIT_SUCCESS;
}

// ----------------------- Frame processing -----------------------

// Decode one reassembled HTJ2K codestream and, if render is enabled, upload
// it to the renderer.  Returns true on success.
bool decode_and_present(const AssembledFrame& frame, const CliOptions& opts, bool is_first_frame,
                        GlRenderer* renderer, std::vector<uint8_t>& rgb_backbuffer) {
  using namespace open_htj2k;

  // Build a fresh decoder from the reassembled bytes.  The ctor copies/holds
  // the buffer pointer, so keeping `frame` alive through invoke_line_based_stream
  // is sufficient.
  openhtj2k_decoder decoder(frame.bytes.data(), frame.bytes.size(), /*reduce_NL=*/0,
                            opts.num_decoder_threads);
  try {
    decoder.parse();
  } catch (std::exception& e) {
    std::fprintf(stderr, "decoder.parse() failed: %s\n", e.what());
    return false;
  }

  // Colorspace selection per RFC 9828 §5.3.
  const ycbcr_coefficients* coeffs = nullptr;
  bool components_are_rgb          = false;
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

  if (is_first_frame) {
    if (components_are_rgb) {
      std::fprintf(stderr, "info: rendering RGB components directly\n");
    } else {
      std::fprintf(stderr, "info: YCbCr -> RGB via %s coefficients\n",
                   coeffs == opts.s0_fallback
                       ? opts.s0_label.c_str()
                       : (coeffs == &YCBCR_BT709_FULL     ? "bt709-full"
                          : coeffs == &YCBCR_BT709_NARROW ? "bt709-narrow"
                          : coeffs == &YCBCR_BT601_FULL   ? "bt601-full"
                          : coeffs == &YCBCR_BT601_NARROW ? "bt601-narrow"
                                                          : "?"));
    }
  }

  uint32_t out_w  = 0;
  uint32_t out_h  = 0;
  uint8_t  depth0 = 0;
  uint16_t ncomp  = 0;
  // Chroma subsampling ratios; populated on the first row callback.
  uint32_t cb_stride_ratio = 1;
  uint32_t cr_stride_ratio = 1;

  bool dims_ok = true;

  try {
    std::vector<uint32_t> widths;
    std::vector<uint32_t> heights;
    std::vector<uint8_t>  depths;
    std::vector<bool>     signeds;
    decoder.invoke_line_based_stream(
        [&](uint32_t y, int32_t* const* rows, uint16_t nc) {
          if (y == 0) {
            if (nc < 1) { dims_ok = false; return; }
            ncomp  = nc;
            out_w  = widths[0];
            out_h  = heights[0];
            depth0 = depths[0];
            if (nc >= 3) {
              cb_stride_ratio = widths[0] / widths[1];
              cr_stride_ratio = widths[0] / widths[2];
              if (cb_stride_ratio == 0) cb_stride_ratio = 1;
              if (cr_stride_ratio == 0) cr_stride_ratio = 1;
            }
            rgb_backbuffer.assign(static_cast<size_t>(out_w) * out_h * 3, 0);
          }
          if (!dims_ok) return;
          uint8_t* out_row = rgb_backbuffer.data() + static_cast<size_t>(y) * out_w * 3;
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
              const uint8_t v8 = static_cast<uint8_t>(shift > 0 ? (v >> shift) : v);
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

  if (!dims_ok || out_w == 0 || out_h == 0) {
    std::fprintf(stderr, "frame: unable to determine dimensions; dropping\n");
    return false;
  }

  if (renderer != nullptr) {
    renderer->upload_and_draw(rgb_backbuffer.data(), static_cast<int>(out_w),
                              static_cast<int>(out_h));
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

// ----------------------- Main loop -----------------------

int run_receiver(const CliOptions& opts) {
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
    if (!renderer.init(1280, 720, "OpenHTJ2K RFC 9828 receiver")) {
      std::fprintf(stderr, "WARN: GLFW init failed; continuing in --no-render mode\n");
    } else {
      renderer_ptr = &renderer;
    }
  }

  FrameHandler frame_handler;
  std::vector<uint8_t> packet_buf(65536);  // max UDP payload
  std::vector<uint8_t> rgb_backbuffer;

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
              opts.decode
                  ? decode_and_present(*emitted, opts, first_frame, renderer_ptr, rgb_backbuffer)
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
  return (rgb_r[0] >= 240 && rgb_r[1] <= 15 && rgb_r[2] <= 15) ? 0 : 1;
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
