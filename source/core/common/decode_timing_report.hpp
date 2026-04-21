// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause
//
// Public report struct produced by the decoder's per-stage timing
// instrumentation.  Populated only in builds compiled with
// -DOPENHTJ2K_DECODE_TIMING=ON; otherwise the callback is never invoked
// and all counters stay zero.

#pragma once

#include <cstdint>

namespace open_htj2k {

enum class DecodeStage : unsigned {
  Parse          = 0,
  BlockDecode    = 1,
  IDWT           = 2,
  ColorTransform = 3,
  Finalize       = 4,
  kCount         = 5,
};

inline const char *decode_stage_name(DecodeStage s) {
  switch (s) {
    case DecodeStage::Parse:          return "parse";
    case DecodeStage::BlockDecode:    return "block_decode";
    case DecodeStage::IDWT:           return "idwt";
    case DecodeStage::ColorTransform: return "color_transform";
    case DecodeStage::Finalize:       return "finalize";
    default:                          return "?";
  }
}

// Wall-clock nanoseconds summed across invocations of each stage over the
// lifetime of a single public decode call (parse() or invoke*()).  The
// report is emitted exactly once per call via the registered sink.
//
// pool_wait_ns / pool_work_ns are aggregated across all worker threads in
// the shared pool, not just the ones touched by this decode call — they
// are process-lifetime counters and only meaningful when diffed between
// the start and end of a given decode, or when the caller controls pool
// usage (e.g. a single decode at a time with no background work).
struct DecodeTimingReport {
  uint64_t stage_ns[static_cast<unsigned>(DecodeStage::kCount)]    = {0, 0, 0, 0, 0};
  uint64_t stage_count[static_cast<unsigned>(DecodeStage::kCount)] = {0, 0, 0, 0, 0};

  // Pool-wide worker-thread accounting.  Sums wall-clock ns each worker
  // spent blocked on the task-queue condvar vs executing tasks.  When
  // timing is disabled at compile time these stay zero.
  uint64_t pool_wait_ns = 0;
  uint64_t pool_work_ns = 0;
  uint32_t pool_workers = 0;
};

}  // namespace open_htj2k
