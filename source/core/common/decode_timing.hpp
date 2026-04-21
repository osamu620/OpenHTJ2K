// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause
//
// Internal timing infrastructure for decode-path instrumentation.
//
// When compiled with -DOPENHTJ2K_DECODE_TIMING=ON, an accumulator is
// attached to each decoder_impl for the duration of a public call.
// Scope macros sprinkled through the decode pipeline accumulate wall
// clock into that accumulator via a thread-local pointer.  At the end
// of the public call, the impl emits a DecodeTimingReport to the
// registered sink.
//
// When the macro is undefined, every scope expands to (void)0 and all
// instrumentation vanishes.  No atomic ops, no chrono calls, no
// per-stage branching — zero measurable overhead in normal Release
// builds.

#pragma once

#include "decode_timing_report.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>

namespace open_htj2k {
namespace internal {

class DecodeTimingAccumulator {
 public:
  DecodeTimingAccumulator() { reset(); }

  void add(DecodeStage s, uint64_t ns) {
    const auto i = static_cast<unsigned>(s);
    stage_ns_[i].fetch_add(ns, std::memory_order_relaxed);
    stage_count_[i].fetch_add(1, std::memory_order_relaxed);
  }

  void reset() {
    for (unsigned i = 0; i < static_cast<unsigned>(DecodeStage::kCount); ++i) {
      stage_ns_[i].store(0, std::memory_order_relaxed);
      stage_count_[i].store(0, std::memory_order_relaxed);
    }
  }

  DecodeTimingReport snapshot() const {
    DecodeTimingReport r;
    for (unsigned i = 0; i < static_cast<unsigned>(DecodeStage::kCount); ++i) {
      r.stage_ns[i]    = stage_ns_[i].load(std::memory_order_relaxed);
      r.stage_count[i] = stage_count_[i].load(std::memory_order_relaxed);
    }
    return r;
  }

 private:
  std::atomic<uint64_t> stage_ns_[static_cast<unsigned>(DecodeStage::kCount)];
  std::atomic<uint64_t> stage_count_[static_cast<unsigned>(DecodeStage::kCount)];
};

// Thread-local pointer set by the decoder_impl for the duration of a
// public call.  Phase scopes read this; workers don't — worker-thread
// accounting lives in the ThreadPool counters instead.
inline thread_local DecodeTimingAccumulator *g_decode_accumulator = nullptr;

// RAII scope: samples steady_clock on construction and destruction;
// adds the elapsed ns to the current thread-local accumulator if one
// is attached.  Safe to construct when no accumulator is attached —
// becomes a pair of steady_clock reads with no storage effect.
class DecodeTimingScope {
 public:
  DecodeTimingScope(DecodeStage s) noexcept : stage_(s) {
    if (g_decode_accumulator) {
      t0_ = std::chrono::steady_clock::now();
    }
  }
  ~DecodeTimingScope() {
    if (g_decode_accumulator) {
      const auto t1 = std::chrono::steady_clock::now();
      const auto ns =
          std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0_).count();
      g_decode_accumulator->add(stage_, static_cast<uint64_t>(ns));
    }
  }
  DecodeTimingScope(const DecodeTimingScope &)            = delete;
  DecodeTimingScope &operator=(const DecodeTimingScope &) = delete;

 private:
  DecodeStage stage_;
  std::chrono::steady_clock::time_point t0_;
};

// RAII attach/detach for the thread-local accumulator.  Use at the
// top of parse() / invoke*() in decoder_impl.
class DecodeTimingAttach {
 public:
  explicit DecodeTimingAttach(DecodeTimingAccumulator *a) noexcept : prev_(g_decode_accumulator) {
    g_decode_accumulator = a;
  }
  ~DecodeTimingAttach() { g_decode_accumulator = prev_; }
  DecodeTimingAttach(const DecodeTimingAttach &)            = delete;
  DecodeTimingAttach &operator=(const DecodeTimingAttach &) = delete;

 private:
  DecodeTimingAccumulator *prev_;
};

}  // namespace internal
}  // namespace open_htj2k

// ─── Scope macros ───────────────────────────────────────────────────────────
// OPENHTJ2K_TIME_SCOPE(Parse)         — expands to an RAII scope covering
//                                        the remainder of the enclosing block
// OPENHTJ2K_TIME_ATTACH(acc_ptr)      — attach an accumulator for the remainder
//                                        of the enclosing block (decoder_impl use)
//
// Concatenation of __LINE__ ensures multiple scopes within the same function
// get distinct variable names.

#ifdef OPENHTJ2K_DECODE_TIMING
  #define OPENHTJ2K_TIMING_CONCAT_(a, b) a##b
  #define OPENHTJ2K_TIMING_CONCAT(a, b)  OPENHTJ2K_TIMING_CONCAT_(a, b)
  #define OPENHTJ2K_TIME_SCOPE(stage)                                            \
    ::open_htj2k::internal::DecodeTimingScope OPENHTJ2K_TIMING_CONCAT(           \
        _ohtj2k_tscope_, __LINE__)(::open_htj2k::DecodeStage::stage)
  #define OPENHTJ2K_TIME_ATTACH(acc_ptr)                                         \
    ::open_htj2k::internal::DecodeTimingAttach OPENHTJ2K_TIMING_CONCAT(          \
        _ohtj2k_tattach_, __LINE__)(acc_ptr)
#else
  #define OPENHTJ2K_TIME_SCOPE(stage)    ((void)0)
  #define OPENHTJ2K_TIME_ATTACH(acc_ptr) ((void)0)
#endif
