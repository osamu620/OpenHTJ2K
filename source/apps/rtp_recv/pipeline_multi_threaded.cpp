// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.
//
// Licensed under the same BSD 3-Clause terms as the rest of OpenHTJ2K.

#include "pipeline_multi_threaded.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

#include "cli.hpp"
#include "decode_helpers.hpp"
#include "decoder.hpp"
#include "frame_handler.hpp"
#include "frame_pipeline.hpp"
#include "gl_renderer.hpp"
#include "rfc9828_parser.hpp"
#include "rtp_socket.hpp"

namespace open_htj2k::rtp_recv {

namespace {

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
    const ColorPipelineParams pipeline = select_color_pipeline_for_frame(frame, opts);
    if (first_frame)
      log_coefficients_choice_once(opts, coeffs, components_are_rgb, pipeline);

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
    df.pipeline           = pipeline;
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

}  // namespace

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
            df->components_are_rgb, df->pipeline);
      } else {
        renderer_ptr->upload_planar_and_draw(
            df->plane_y.data(), df->plane_cb.data(), df->plane_cr.data(),
            static_cast<int>(df->width), static_cast<int>(df->height),
            static_cast<int>(df->chroma_width), static_cast<int>(df->chroma_height),
            df->shader_coeffs, df->components_are_rgb, df->pipeline);
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

}  // namespace open_htj2k::rtp_recv
