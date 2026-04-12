// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.
//
// Licensed under the same BSD 3-Clause terms as the rest of OpenHTJ2K.

#include "pipeline_single_threaded.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <optional>
#include <vector>

#include "cli.hpp"
#include "decode_helpers.hpp"
#include "frame_handler.hpp"
#include "frame_pipeline.hpp"
#include "gl_renderer.hpp"
#include "rfc9828_parser.hpp"
#include "rtp_socket.hpp"

namespace open_htj2k::rtp_recv {

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

}  // namespace open_htj2k::rtp_recv
