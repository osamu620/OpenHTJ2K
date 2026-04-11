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

#include "color_pipeline.hpp"

namespace open_htj2k::rtp_recv {

struct ycbcr_coefficients;  // from ycbcr_rgb.hpp

template <typename T>
class LatestSlot {
 public:
  LatestSlot() = default;

  LatestSlot(const LatestSlot&)            = delete;
  LatestSlot& operator=(const LatestSlot&) = delete;

  // Producer-side. Atomically replaces any existing item with `value`.
  // Returns true if a previous item was discarded (and increments
  // evictions()), false if the slot was empty. Wakes any waiting consumer.
  //
  // The evicted item (if any) is moved out of the slot under the lock
  // and destroyed *after* the lock is released.  For T = DecodedFrame
  // that destructor frees multi-megabyte plane buffers, and holding the
  // mutex across those free() calls would serialize the producer and
  // consumer for the duration of the deallocation.  Moving the destroy
  // outside the critical section keeps lock hold time proportional to
  // the pointer swap, not to T's destructor cost.
  bool push(T value) {
    bool             evicted = false;
    std::optional<T> stale;
    {
      std::lock_guard<std::mutex> lk(mu_);
      if (slot_.has_value()) {
        evicted = true;
        ++evictions_;
        stale.swap(slot_);
      }
      slot_.emplace(std::move(value));
    }
    // `stale` destructor runs here, outside the mutex.
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
//
// Two content shapes share the struct (selected by `kind`):
//
//   - CPU_RGB: the decode worker ran YCbCr→RGB on the CPU and filled
//     `rgb` with width*height*3 interleaved bytes.  Used by the
//     --color-path=cpu fallback and by GL-incompatible environments.
//   - PLANAR_YCBCR / PLANAR_RGB: the decode worker wrote per-plane
//     samples into either the 8-bit or 16-bit plane vectors,
//     selected by `bit_depth`.  Luma dims are in `width/height`;
//     chroma dims in `chroma_width/chroma_height` (which equal the
//     luma dims for 4:4:4 / PLANAR_RGB).  The GPU shader applies
//     the YCbCr→RGB matrix in the fragment stage.
//
//     - bit_depth == 8: `plane_y/cb/cr` hold uint8_t samples, the
//       `_16` vectors are empty.  Uploaded as GL_R8 by the renderer.
//     - bit_depth >  8: `plane_y_16/cb_16/cr_16` hold uint16_t samples
//       (clamp-only, NOT shifted to 8-bit), the u8 vectors are empty.
//       Uploaded as GL_R16 by the renderer.  The fragment shader
//       renormalizes via uNormScale before bias/scale.
//
// The u8 and u16 vector sets are mutually exclusive per frame, so
// std::move-ing a DecodedFrame only moves whichever set is populated.
struct DecodedFrame {
  enum Kind : uint8_t {
    CPU_RGB      = 0,
    PLANAR_YCBCR = 1,
    PLANAR_RGB   = 2,
  };

  std::vector<uint8_t>      rgb;          // CPU_RGB: width * height * 3 bytes
  std::vector<uint8_t>      plane_y;      // PLANAR_*, 8-bit source: width * height bytes
  std::vector<uint8_t>      plane_cb;     // PLANAR_*, 8-bit source
  std::vector<uint8_t>      plane_cr;     // PLANAR_*, 8-bit source
  std::vector<uint16_t>     plane_y_16;   // PLANAR_*, >8-bit source: width * height shorts
  std::vector<uint16_t>     plane_cb_16;  // PLANAR_*, >8-bit source
  std::vector<uint16_t>     plane_cr_16;  // PLANAR_*, >8-bit source
  uint32_t                  width         = 0;  // luma width
  uint32_t                  height        = 0;  // luma height
  uint32_t                  chroma_width  = 0;  // 0 in CPU_RGB
  uint32_t                  chroma_height = 0;  // 0 in CPU_RGB
  uint8_t                   bit_depth     = 8;  // luma bit depth; 0 in CPU_RGB
  Kind                      kind          = CPU_RGB;
  // For PLANAR_YCBCR: points at one of the static-const YCBCR_* constants
  // in ycbcr_rgb.hpp (BT601/709 × full/narrow).  Never owns.  Null in
  // PLANAR_RGB / CPU_RGB.  The pointer is safe to move across threads
  // because it aliases immortal constexpr data.
  const ycbcr_coefficients* shader_coeffs      = nullptr;
  bool                      components_are_rgb = false;
  // HDR colour pipeline (inverse transfer, gamut matrix, display encoding).
  // Selected by the decode thread per-frame from the Main Packet
  // TRANS/PRIMS fields and the CLI fallbacks.  Defaults to the v0.12.0-
  // equivalent gamma2.2 + identity + sRGB pipeline for SDR BT.709 sources.
  ColorPipelineParams       pipeline;
  // Sender's RTP timestamp (90 kHz per RFC 3551 video profile).  Copied
  // from the AssembledFrame by the decode thread so the main-thread
  // frame pacer can schedule presentation at the sender's intended
  // cadence instead of a fixed --pace-fps rate.  0 is not a reliable
  // "absent" sentinel — the first frame of a stream can legitimately
  // have rtp_timestamp=0 — so pacing code must rely on its own
  // "reference established" flag rather than on source_rtp_ts != 0.
  uint32_t                  source_rtp_ts      = 0;
};

}  // namespace open_htj2k::rtp_recv
