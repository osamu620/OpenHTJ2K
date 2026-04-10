// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.
//
// Licensed under the same BSD 3-Clause terms as the rest of OpenHTJ2K.

#include "frame_handler.hpp"

#include <cstring>
#include <utility>

namespace open_htj2k::rtp_recv {

namespace {
// Default reserve size: 4 MB matches the onthefly-trial buffer, which was
// empirically large enough for 4K HTJ2K frames at broadcast bitrates.
constexpr size_t kDefaultFrameCapacity = 4 * 1024 * 1024;
}  // namespace

FrameHandler::FrameHandler() { accum_.reserve(kDefaultFrameCapacity); }

void FrameHandler::reserve_frame_capacity(size_t bytes) { accum_.reserve(bytes); }

void FrameHandler::reset() {
  accum_.clear();
  cached_main_.clear();
  have_cached_main_     = false;
  have_frame_           = false;
  current_ts_           = 0;
  frame_eseq_first_     = 0;
  frame_eseq_last_      = 0;
  frame_packet_count_   = 0;
  frame_intact_         = true;
  last_pkt_was_marker_  = false;
  frame_has_meta_       = false;
  frame_tp_           = 0;
  frame_ordh_         = 0;
  frame_r_            = false;
  frame_s_            = false;
  frame_c_            = false;
  frame_range_        = false;
  frame_prims_        = 0;
  frame_trans_        = 0;
  frame_mat_          = 0;
  first_packet_ever_  = true;
  last_ext_seq_       = 0;
  stats_              = {};
}

void FrameHandler::track_sequence(uint32_t ext_seq) {
  if (first_packet_ever_) {
    first_packet_ever_ = false;
    last_ext_seq_      = ext_seq;
    return;
  }
  const uint32_t expected = (last_ext_seq_ + 1) & 0x00FFFFFFu;
  if (ext_seq != expected) {
    ++stats_.seq_gaps;
    frame_intact_ = false;
  }
  last_ext_seq_ = ext_seq;
}

void FrameHandler::start_frame_empty() {
  accum_.clear();
  have_frame_          = true;
  frame_packet_count_  = 0;
  frame_intact_        = true;
  last_pkt_was_marker_ = false;
  frame_has_meta_      = false;
}

void FrameHandler::start_frame_with_main_prepend() {
  accum_.clear();
  if (have_cached_main_) {
    accum_.insert(accum_.end(), cached_main_.begin(), cached_main_.end());
  } else {
    // No cached main header yet — this Body-only frame can't be decoded.
    // Flag lossy so finalize_frame drops it.
    frame_intact_ = true;  // reset first
  }
  have_frame_          = true;
  frame_packet_count_  = 0;
  frame_intact_        = have_cached_main_;
  last_pkt_was_marker_ = false;
  frame_has_meta_      = false;
}

void FrameHandler::finalize_frame(std::optional<AssembledFrame>& out_frame) {
  if (!have_frame_) return;

  // Tail-loss check: if we are finalizing because of an external trigger
  // (timestamp change on the next packet) and the most recent packet of the
  // current frame did not carry M=1, we know packet(s) at the end of the
  // frame were dropped — the codestream is missing its tail.
  if (frame_intact_ && !last_pkt_was_marker_) {
    frame_intact_ = false;
    ++stats_.tail_loss_drops;
  }

  if (frame_intact_ && !accum_.empty()) {
    AssembledFrame f;
    f.bytes         = std::move(accum_);
    f.rtp_timestamp = current_ts_;
    f.eseq_first    = frame_eseq_first_;
    f.eseq_last     = frame_eseq_last_;
    f.packet_count  = frame_packet_count_;
    f.has_meta      = frame_has_meta_;
    f.tp            = frame_tp_;
    f.ordh          = frame_ordh_;
    f.r             = frame_r_;
    f.s             = frame_s_;
    f.c             = frame_c_;
    f.range         = frame_range_;
    f.prims         = frame_prims_;
    f.trans         = frame_trans_;
    f.mat           = frame_mat_;
    out_frame       = std::move(f);
    ++stats_.frames_emitted;
    accum_.clear();  // std::move left us in a valid but empty state
    accum_.reserve(kDefaultFrameCapacity);
  } else {
    ++stats_.frames_dropped;
    accum_.clear();
  }

  have_frame_         = false;
  frame_packet_count_ = 0;
  frame_intact_       = true;
  frame_has_meta_     = false;
}

bool FrameHandler::push_main_packet(const RtpHeader& rtp, const MainPacketHeader& main,
                                    const uint8_t* codestream_bytes, size_t codestream_len,
                                    std::optional<AssembledFrame>& out_frame) {
  ++stats_.packets_received;
  stats_.bytes_received += codestream_len;

  const uint32_t ext_seq = make_ext_seq(main.eseq, rtp.sequence);

  // Detect timestamp change mid-assembly: emit whatever we had, start fresh.
  if (have_frame_ && rtp.timestamp != current_ts_) {
    finalize_frame(out_frame);  // may populate out_frame
  }

  if (!have_frame_) {
    current_ts_       = rtp.timestamp;
    frame_eseq_first_ = ext_seq;
    start_frame_empty();
  }

  track_sequence(ext_seq);
  frame_eseq_last_ = ext_seq;
  ++frame_packet_count_;

  // Append this packet's codestream bytes to the accumulator.
  if (codestream_bytes != nullptr && codestream_len > 0) {
    accum_.insert(accum_.end(), codestream_bytes, codestream_bytes + codestream_len);
  }

  // Track marker bit so finalize_frame() can detect tail-loss when triggered
  // by a later timestamp change.
  last_pkt_was_marker_ = rtp.marker;

  // Capture metadata from the most recent Main Packet in this frame.
  frame_has_meta_ = true;
  frame_tp_       = main.tp;
  frame_ordh_     = main.ordh;
  frame_r_        = main.r;
  frame_s_        = main.s;
  frame_c_        = main.c;
  frame_range_    = main.range;
  frame_prims_    = main.prims;
  frame_trans_    = main.trans;
  frame_mat_      = main.mat;

  // Cache main-header bytes if R=1.  The Main Packet payload contains the
  // JPEG 2000 main header (SOC..first SOD); for R=1 streams we stash it so
  // later frames that arrive without a Main Packet can prepend it.
  if (main.r && codestream_len > 0) {
    cached_main_.assign(codestream_bytes, codestream_bytes + codestream_len);
    have_cached_main_ = true;
  }

  // If the RTP marker bit is set, this is the last packet of a frame
  // (typically the one that carries EOC).
  if (rtp.marker) {
    finalize_frame(out_frame);
  }

  return true;
}

bool FrameHandler::push_body_packet(const RtpHeader& rtp, const BodyPacketHeader& body,
                                    const uint8_t* codestream_bytes, size_t codestream_len,
                                    std::optional<AssembledFrame>& out_frame) {
  ++stats_.packets_received;
  stats_.bytes_received += codestream_len;

  const uint32_t ext_seq = make_ext_seq(body.eseq, rtp.sequence);

  if (have_frame_ && rtp.timestamp != current_ts_) {
    finalize_frame(out_frame);
  }

  if (!have_frame_) {
    current_ts_       = rtp.timestamp;
    frame_eseq_first_ = ext_seq;
    // Body-first frame start: prepend cached main header (if any).  If no
    // cache is available, start_frame_with_main_prepend marks the frame
    // lossy so finalize drops it — we cannot synthesize a main header.
    start_frame_with_main_prepend();
  }

  track_sequence(ext_seq);
  frame_eseq_last_ = ext_seq;
  ++frame_packet_count_;

  if (codestream_bytes != nullptr && codestream_len > 0) {
    accum_.insert(accum_.end(), codestream_bytes, codestream_bytes + codestream_len);
  }

  last_pkt_was_marker_ = rtp.marker;

  if (rtp.marker) {
    finalize_frame(out_frame);
  }

  return true;
}

}  // namespace open_htj2k::rtp_recv
