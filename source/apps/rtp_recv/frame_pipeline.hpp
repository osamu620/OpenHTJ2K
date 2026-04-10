// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.
//
// Licensed under the same BSD 3-Clause terms as the rest of OpenHTJ2K.

#pragma once

// Inter-thread plumbing for the v2 multi-threaded RTP receiver:
//
//   - LatestSlot<T>: a 1-deep "latest frame wins" handoff between two
//     threads. The producer push() always succeeds; if the slot is already
//     occupied, the previous item is dropped (eviction count incremented)
//     and replaced with the new one. The consumer can pop_wait() blocking
//     on a condition variable, or try_pop() non-blocking. This is the
//     right shape for a real-time pipeline where stale frames are useless
//     — we want minimum latency, not maximum throughput.
//
//   - DecodedFrame: the POD that carries a decoded RGB image from the
//     decode worker to the GLFW main thread.
//
// Both pieces are header-only — they're small enough that pulling them into
// a translation unit isn't worth the build-system noise.

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

namespace open_htj2k::rtp_recv {

template <typename T>
class LatestSlot {
 public:
  LatestSlot() = default;

  LatestSlot(const LatestSlot&)            = delete;
  LatestSlot& operator=(const LatestSlot&) = delete;

  // Producer-side. Atomically replaces any existing item with `value`.
  // Returns true if a previous item was discarded (and increments
  // evictions()), false if the slot was empty. Wakes any waiting consumer.
  bool push(T value) {
    bool evicted = false;
    {
      std::lock_guard<std::mutex> lk(mu_);
      if (slot_.has_value()) {
        evicted = true;
        ++evictions_;
      }
      slot_.emplace(std::move(value));
    }
    cv_.notify_one();
    return evicted;
  }

  // Consumer-side. Blocks until an item is available or `stop` becomes
  // true. Returns std::nullopt if the wait was woken by stop, otherwise
  // moves the item out of the slot.
  //
  // The consumer is responsible for periodically observing `stop` outside
  // pop_wait too — pop_wait only checks it while the cv lock is held.
  std::optional<T> pop_wait(const std::atomic<bool>& stop) {
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [&] { return slot_.has_value() || stop.load(std::memory_order_acquire); });
    if (!slot_.has_value()) return std::nullopt;
    std::optional<T> out;
    out.swap(slot_);
    return out;
  }

  // Consumer-side, non-blocking. Returns and clears the slot if there is
  // one, otherwise std::nullopt. Used by the GLFW main thread which has
  // its own event-pump cadence and shouldn't block on the decode worker.
  std::optional<T> try_pop() {
    std::lock_guard<std::mutex> lk(mu_);
    if (!slot_.has_value()) return std::nullopt;
    std::optional<T> out;
    out.swap(slot_);
    return out;
  }

  // Wake any waiting consumer without writing a value. Used during
  // shutdown so a thread blocked in pop_wait() observes the stop flag.
  void notify() { cv_.notify_all(); }

  // Cumulative count of items overwritten by push() because the slot was
  // already occupied. Read by the receiver's exit summary.
  uint64_t evictions() const {
    std::lock_guard<std::mutex> lk(mu_);
    return evictions_;
  }

 private:
  mutable std::mutex      mu_;
  std::condition_variable cv_;
  std::optional<T>        slot_;
  uint64_t                evictions_ = 0;
};

// Carries one decoded frame from the decode worker to the renderer.
// The decoder converts to 8-bit RGB on its own thread; the main thread
// just uploads `rgb` to a GL_RGB8 texture and draws.
struct DecodedFrame {
  std::vector<uint8_t> rgb;       // width * height * 3 bytes
  uint32_t             width  = 0;
  uint32_t             height = 0;
};

}  // namespace open_htj2k::rtp_recv
