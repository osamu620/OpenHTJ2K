// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.
//
// JPP-stream parser: consume a byte buffer of JPP-stream messages
// (produced by B3's data-bin emitters or by a JPIP server) and aggregate
// them into a DataBinSet keyed by (class, in-class identifier).
//
// In-order assumption: the v1 parser requires each message's msg_offset
// to equal the already-accumulated byte count for the target data-bin —
// i.e. a strict contiguous stream with no gaps and no duplicates.  This
// matches what our B3 emitter produces and is sufficient for the local
// round-trip use case in PHASE2_PLAN.md.  Out-of-order support can be
// layered on top once we have a transport that reorders (Phase 3+).
//
// The parser ignores the CSn field for now — Phase 1/2 assets always
// declare a single codestream (cs_n = 0).  DataBinSet's key therefore
// omits cs_n; if multi-codestream support is needed later, the key
// widens to a (cs_n, class, in_class_id) tuple.
#pragma once
#include <cstddef>
#include <cstdint>
#include <map>
#include <utility>
#include <vector>

#if defined(_MSC_VER) && !defined(OHTJ2K_STATIC)
  #define OPENHTJ2K_JPIP_EXPORT __declspec(dllexport)
#else
  #define OPENHTJ2K_JPIP_EXPORT
#endif

namespace open_htj2k {
namespace jpip {

class OPENHTJ2K_JPIP_EXPORT DataBinSet {
 public:
  // True iff at least one message for this (class, id) has been received.
  bool contains(uint8_t class_id, uint64_t in_class_id) const;

  // True iff the data-bin's last message has been received (is_last bit).
  bool is_complete(uint8_t class_id, uint64_t in_class_id) const;

  // Accumulated bytes for this data-bin.  Returns a reference to an empty
  // vector when the bin is unknown (rather than throwing), matching the
  // spec's treatment of "not yet received".
  const std::vector<uint8_t> &get(uint8_t class_id, uint64_t in_class_id) const;

  // Number of distinct data-bins known to the set.
  std::size_t size() const { return bins_.size(); }

  // (class, in_class_id) keys in deterministic ascending order.
  std::vector<std::pair<uint8_t, uint64_t>> keys() const;

  // Feed one message's body into the set.  Returns true on success, false
  // if the message violates the in-order assumption (non-contiguous
  // msg_offset, or further bytes arriving on a bin that was already
  // marked complete).  Intended for internal use by parse_jpp_stream; the
  // public entry point is below.
  bool append(uint8_t class_id, uint64_t in_class_id, uint64_t msg_offset,
              const uint8_t *payload, std::size_t payload_len, bool is_last);

 private:
  struct Entry {
    std::vector<uint8_t> bytes;
    bool                 is_last = false;
  };
  using Key = std::pair<uint8_t, uint64_t>;
  std::map<Key, Entry> bins_;
};

// Parse a complete JPP-stream byte buffer, appending every message's body
// into `*out`'s DataBinSet.  Returns true iff the entire buffer was
// consumed cleanly and every message obeyed the in-order assumption.
OPENHTJ2K_JPIP_EXPORT bool parse_jpp_stream(const uint8_t *bytes, std::size_t len,
                                            DataBinSet *out);

}  // namespace jpip
}  // namespace open_htj2k
