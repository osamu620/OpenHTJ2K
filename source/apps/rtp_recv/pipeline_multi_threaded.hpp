// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.
//
// Licensed under the same BSD 3-Clause terms as the rest of OpenHTJ2K.

#pragma once

#include <atomic>
#include <cstdint>

#include "cli.hpp"
#include "frame_handler.hpp"
#include "frame_pipeline.hpp"

namespace open_htj2k::rtp_recv {

// Shared mutable state for the v2 receive/decode/render trio.  Owned by
// run_receiver_threaded(); references handed to recv_thread_main and
// decode_thread_main.
struct ReceiverState {
  std::atomic<bool>           stop_flag{false};
  LatestSlot<AssembledFrame>  decode_slot;
  LatestSlot<DecodedFrame>    render_slot;

  // Counters (atomic for thread-safe summary at exit).
  std::atomic<uint64_t> frames_emitted_to_decode{0};  // dump index, increments per frame_handler emission
  std::atomic<uint64_t> frames_decoded{0};
  std::atomic<uint64_t> frames_failed{0};

  // Decode timing — written only by the decode thread, read by main at exit.
  std::atomic<uint64_t> decode_us_sum{0};
  std::atomic<uint64_t> decode_us_min{UINT64_MAX};
  std::atomic<uint64_t> decode_us_max{0};
};

// v2 multi-threaded main loop (--threading=on, default).
int run_receiver_threaded(const CliOptions& opts);

}  // namespace open_htj2k::rtp_recv
