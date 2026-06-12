// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.
//
// JPIP channel/session manager (ISO/IEC 15444-9 §B.2, §C.3).
//
// A JPIP session is created when the server grants a New Channel request
// (`cnew=`, §C.3.3) by returning a `JPIP-cnew: cid=…` response header
// (§D.2.3).  Granting a channel is a contract: the server must keep a
// per-session cache model and never re-send data-bins it has already
// delivered on that session (§B.2, Table D.2 reason codes 1/2).  A server
// unwilling to keep that state shall NOT return the header — the request
// then degrades to stateless service where the client carries the model
// in `&model=` itself.
//
// This manager maps channel-id → CacheModel with an LRU cap.  Channels and
// sessions are 1:1 here (one channel per session); §C.3.4 `cclose` with a
// list of cids or "*" therefore closes exactly the named channels.
//
// Access pattern: `snapshot()` copies the model out so the (possibly slow)
// response emission never holds the lock; `commit()` merges the bins that
// were actually delivered back in afterwards.  Bins whose send aborted are
// simply not committed, so the next request re-sends them.

#ifndef OPENHTJ2K_CHANNEL_MANAGER_HPP
#define OPENHTJ2K_CHANNEL_MANAGER_HPP

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "cache_model.hpp"

#if defined(_MSC_VER) && !defined(OHTJ2K_STATIC)
  #define OPENHTJ2K_JPIP_EXPORT __declspec(dllexport)
#else
  #define OPENHTJ2K_JPIP_EXPORT
#endif

namespace open_htj2k {
namespace jpip {

// One data-bin's delivery outcome within a response, as committed to the
// channel's cache model: the client now holds `end_bytes` payload bytes of
// the bin, and `complete` records whether the is_last message went out.
struct SentBin {
  uint8_t  class_id    = 0;
  uint64_t in_class_id = 0;
  uint64_t end_bytes   = 0;
  bool     complete    = false;
};

class OPENHTJ2K_JPIP_EXPORT ChannelManager {
 public:
  explicit ChannelManager(std::size_t max_channels = 64) : max_channels_(max_channels) {}

  // Pick a transport from the §C.3.3 `cnew` value (comma-separated list of
  // transport names, e.g. "http-tcp,http").  Table D.1: the granted
  // transport shall be one of the values the client supplied — exact token
  // match, never substring ("http-tcp" must not be answered with "http"
  // unless "http" itself is also in the list).  Returns the chosen token,
  // or an empty string when nothing in the list is supported (the caller
  // must then serve the request statelessly without a JPIP-cnew header).
  // An empty list counts as unsupported.
  static std::string negotiate_transport(const std::string &cnew_list);

  // Create a new channel and return its channel-id (IDTOKEN, §C.3.2).
  // Evicts the least-recently-used channel when the cap is reached.
  std::string open();

  // Copy the channel's cache model into *out.  Returns false (and leaves
  // *out untouched) when the cid is unknown — expired, evicted, or never
  // issued.  Touches the channel's LRU recency.
  bool snapshot(const std::string &cid, CacheModel *out);

  // Merge data-bins delivered on this channel into its cache model:
  // complete bins are marked held-in-full, partial deliveries record
  // their byte offset so the next request resumes there.  Unknown cids
  // are ignored (the channel may have been evicted while the response
  // was streaming).
  void commit(const std::string &cid, const std::vector<SentBin> &sent);

  // Fold a request's §C.9 `model=` statements into the channel's model
  // (additive statements mark, subtractive/partial unmark — see
  // CacheModel::apply).  Sessions still accept client model updates, e.g.
  // when the client discards cached bins.  Returns false if cid unknown.
  bool apply_model(const std::string &cid, const std::string &model_str);

  // §C.3.4 cclose: close one channel.  Returns true if it existed.
  bool close(const std::string &cid);

  std::size_t size() const;

 private:
  struct Channel {
    CacheModel model;
    uint64_t   last_used = 0;
  };

  mutable std::mutex mtx_;
  std::unordered_map<std::string, Channel> channels_;
  std::size_t max_channels_;
  uint64_t    next_cid_  = 1;
  uint64_t    use_clock_ = 0;
};

}  // namespace jpip
}  // namespace open_htj2k

#endif  // OPENHTJ2K_CHANNEL_MANAGER_HPP
