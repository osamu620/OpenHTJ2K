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

#include "jpp_message.hpp"

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

  // Merge all bins from `other` into this set.  For bins that exist in
  // both, the larger of the two is kept (by byte count).  Used by the
  // client to union multiple view-window responses into one set before
  // reassembly.
  void merge_from(const DataBinSet &other);

  // (class, in_class_id) keys in deterministic ascending order.
  std::vector<std::pair<uint8_t, uint64_t>> keys() const;

  bool has_eor() const { return eor_received_; }
  uint8_t eor_reason() const { return eor_reason_; }
  void set_eor(uint8_t reason) { eor_received_ = true; eor_reason_ = reason; }

  bool append(uint8_t class_id, uint64_t in_class_id, uint64_t msg_offset,
              const uint8_t *payload, std::size_t payload_len, bool is_last);

  // Remove a bin from the set.  Used by the browser viewer's LRU
  // precinct cache to free bytes when the cache exceeds its budget —
  // the next server response for the same view-window will re-fill
  // the erased bin, so we must drop both the bytes and the is_last
  // flag (append rejects further bytes on a closed bin).  Returns
  // true if a bin was erased, false if the key was not present.
  bool erase(uint8_t class_id, uint64_t in_class_id);

 private:
  struct Entry {
    std::vector<uint8_t> bytes;
    bool                 is_last = false;
  };
  using Key = std::pair<uint8_t, uint64_t>;
  std::map<Key, Entry> bins_;
  bool    eor_received_ = false;
  uint8_t eor_reason_   = 0;
};

// Parse a complete JPP-stream byte buffer, appending every message's body
// into `*out`'s DataBinSet.  Returns true iff the entire buffer was
// consumed cleanly and every message obeyed the in-order assumption.
OPENHTJ2K_JPIP_EXPORT bool parse_jpp_stream(const uint8_t *bytes, std::size_t len,
                                            DataBinSet *out);

// Resumable JPP-stream parser for progressive delivery.  Unlike
// `parse_jpp_stream`, which is one-shot and assumes the caller has the
// full response buffered, StreamingJppParser can be called with
// arbitrary-sized chunks — including chunk boundaries that fall
// mid-message or mid-VBAS — and will buffer any incomplete tail until
// the next `feed()` supplies the rest.  The MessageHeaderContext used to
// honour dependent-form headers (§A.2.1 Table A.1: omitted Class/CSn)
// is carried across calls.
//
// This is the primitive the HTTP-chunked clients (browser viewer, C++
// foveation demo) use to start decoding while the response is still
// arriving on the wire.
class OPENHTJ2K_JPIP_EXPORT StreamingJppParser {
 public:
  // Feed `len` bytes into the parser.  Complete messages are appended to
  // `*out`.  Returns true on success (including the case where only a
  // partial message is buffered internally and the caller should feed
  // more), and false iff a definitive protocol error was detected (bad
  // VBAS, class-id out of range, out-of-order msg_offset, an already
  // closed bin receiving more bytes, or a header that exceeds the
  // worst-case header size without decoding).
  bool feed(const uint8_t *bytes, std::size_t len, DataBinSet *out);

  // Signal end-of-stream.  Returns true iff the parser has no partial
  // message buffered — i.e. the response ended cleanly at a message
  // boundary.  A false return indicates the peer closed the connection
  // mid-message.
  bool finish() const { return pending_.empty(); }

  // Reset all state (drop pending bytes, reset MessageHeaderContext,
  // clear EOR flag) so the parser can be reused for a fresh response.
  void reset();

  // Number of bytes currently buffered (waiting for the rest of an
  // incomplete message).  Useful for tests and for back-pressure
  // accounting; 0 means the parser is at a clean message boundary.
  std::size_t pending() const { return pending_.size(); }

 private:
  std::vector<uint8_t> pending_;
  MessageHeaderContext ctx_;
};

}  // namespace jpip
}  // namespace open_htj2k
