// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.
//
// Licensed under the same BSD 3-Clause terms as the rest of OpenHTJ2K.

#pragma once

// Per-RTP-stream frame reassembler.
//
// Shape ported from rtp_decoder_test@onthefly-trial's frame_handler.hpp,
// but with the uvgRTP dependency removed: instead of rtp_frame* the public
// API takes our own parsed RtpHeader + MainPacketHeader / BodyPacketHeader
// structs and a pointer/length for the codestream bytes in that packet.
//
// Responsibilities:
//   - Accumulate codestream bytes for the current frame in a std::vector
//     buffer.  Frame boundary = RTP marker bit or RTP timestamp change.
//   - Track the 24-bit extended sequence number (ESEQ << 16 | rtp_seq) per
//     RFC 9828 §5.2 and flag the current frame as lossy on any gap.  v1
//     drops the whole frame on loss; reorder tolerance is a v2 item.
//   - Handle main-header reuse: when R=1, cache the Main Packet codestream
//     bytes so a later frame that begins with a Body Packet can prepend the
//     cached main header.
//   - Collect Main Packet metadata (TP, ORDH, R/S/C, PRIMS/TRANS/MAT/RANGE)
//     so the caller can pick YCbCr coefficients per §5.3.
//
// NOT implemented in v1:
//   - Reorder buffer for UDP in-flight swaps.  Any gap drops the frame.
//   - Code-block caching (C=1).  Treated as an invalid flag for now.
//   - Multi-codestream targets (only one SSRC at a time).

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "rfc9828_parser.hpp"

namespace open_htj2k::rtp_recv {

struct AssembledFrame {
  std::vector<uint8_t> bytes;       // complete JPEG 2000 codestream ready for openhtj2k_decoder
  uint32_t rtp_timestamp   = 0;
  uint32_t eseq_first      = 0;     // 24-bit extended seq of first packet in frame
  uint32_t eseq_last       = 0;     // 24-bit extended seq of last packet in frame
  size_t   packet_count    = 0;

  // Latest Main Packet metadata observed while building this frame.
  bool    has_meta  = false;
  uint8_t tp        = 0;
  uint8_t ordh      = 0;
  bool    r         = false;
  bool    s         = false;
  bool    c         = false;
  bool    range     = false;
  uint8_t prims     = 0;
  uint8_t trans     = 0;
  uint8_t mat       = 0;
};

struct FrameHandlerStats {
  uint64_t frames_emitted    = 0;
  uint64_t frames_dropped    = 0;
  uint64_t packets_received  = 0;
  uint64_t bytes_received    = 0;  // codestream bytes only, not RTP/9828 headers
  uint64_t seq_gaps          = 0;
  uint64_t tail_loss_drops   = 0;  // frames dropped because they ended without M=1
};

class FrameHandler {
 public:
  FrameHandler();

  // Reserve capacity for a worst-case 4K HTJ2K frame.  Optional; resize-on-
  // push handles it regardless.
  void reserve_frame_capacity(size_t bytes);

  // Reset every piece of state — invoked when the caller detects an RTP SSRC
  // change, or when it wants to discard a partially-assembled frame.
  void reset();

  // Feed a Main Packet.  `codestream_bytes` must point at the bytes past the
  // 8-byte RFC 9828 Main Packet payload header (and any XTRAB extension,
  // currently always zero).  On return, out_frame is populated if this push
  // completed a frame.
  //
  // Returns true if the packet was accepted (with or without emitting a frame),
  // false on a fatal invariant violation (caller should treat as protocol error).
  bool push_main_packet(const RtpHeader& rtp, const MainPacketHeader& main,
                        const uint8_t* codestream_bytes, size_t codestream_len,
                        std::optional<AssembledFrame>& out_frame);

  // Feed a Body Packet.  Same semantics as push_main_packet.
  bool push_body_packet(const RtpHeader& rtp, const BodyPacketHeader& body,
                        const uint8_t* codestream_bytes, size_t codestream_len,
                        std::optional<AssembledFrame>& out_frame);

  const FrameHandlerStats& stats() const { return stats_; }

 private:
  static uint32_t make_ext_seq(uint8_t eseq, uint16_t rtp_seq) {
    return (static_cast<uint32_t>(eseq) << 16) | static_cast<uint32_t>(rtp_seq);
  }

  // Common per-packet bookkeeping: extended-sequence tracking and per-frame
  // counters.  Returns false on gap (caller flags the frame lossy).
  void track_sequence(uint32_t ext_seq);

  // Start a new frame with the given RTP timestamp, copying the cached main
  // header first if R=1 is in effect and the current packet is a Body.
  void start_frame_with_main_prepend();
  void start_frame_empty();

  // Finalize the current frame.  If frame_intact_ == true, emit via out_frame;
  // otherwise drop and increment frames_dropped.  Resets per-frame state.
  void finalize_frame(std::optional<AssembledFrame>& out_frame);

  // Codestream accumulator.
  std::vector<uint8_t> accum_;

  // Cached main header from the last R=1 Main Packet.
  std::vector<uint8_t> cached_main_;
  bool have_cached_main_ = false;

  // Per-frame state.
  bool     have_frame_        = false;
  uint32_t current_ts_        = 0;
  uint32_t frame_eseq_first_  = 0;
  uint32_t frame_eseq_last_   = 0;
  size_t   frame_packet_count_ = 0;
  bool     frame_intact_      = true;
  // True iff the most recent packet pushed into the current frame had the
  // RTP marker bit set.  finalize_frame() consults this when it was triggered
  // by a timestamp change rather than by the marker bit, and treats the frame
  // as lossy if the previous packet did not carry M=1 — that's how we catch
  // tail-end packet loss the sequence-gap detector misses.
  bool     last_pkt_was_marker_ = false;

  // Latest Main Packet metadata for this frame.
  bool     frame_has_meta_ = false;
  uint8_t  frame_tp_       = 0;
  uint8_t  frame_ordh_     = 0;
  bool     frame_r_        = false;
  bool     frame_s_        = false;
  bool     frame_c_        = false;
  bool     frame_range_    = false;
  uint8_t  frame_prims_    = 0;
  uint8_t  frame_trans_    = 0;
  uint8_t  frame_mat_      = 0;

  // Global session state.
  bool     first_packet_ever_ = true;
  uint32_t last_ext_seq_      = 0;

  FrameHandlerStats stats_;
};

}  // namespace open_htj2k::rtp_recv
