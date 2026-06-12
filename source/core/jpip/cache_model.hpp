// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.
//
// JPIP client cache model (§C.9) — tracks which data-bins the client
// has received so the server can skip redundant data.

#ifndef OPENHTJ2K_CACHE_MODEL_HPP
#define OPENHTJ2K_CACHE_MODEL_HPP

#include <cstdint>
#include <string>
#include <unordered_map>

#if defined(_MSC_VER) && !defined(OHTJ2K_STATIC)
  #define OPENHTJ2K_JPIP_EXPORT __declspec(dllexport)
#else
  #define OPENHTJ2K_JPIP_EXPORT
#endif

namespace open_htj2k {
namespace jpip {

class OPENHTJ2K_JPIP_EXPORT CacheModel {
 public:
  // Mark a bin as held in full.
  void mark(uint8_t class_id, uint64_t in_class_id);
  // Mark a bin as held up to `bytes` payload bytes (§C.9.2 ":bytes"
  // qualifier).  Holdings only grow: a smaller `bytes` than already
  // recorded is a no-op, and a bin already complete stays complete.
  void mark_partial(uint8_t class_id, uint64_t in_class_id, uint64_t bytes);
  // Remove a previously marked bin from the cache model.  Used by the
  // LRU precinct cache when a bin is evicted so the client stops
  // advertising it in the `&model=` field.
  void unmark(uint8_t class_id, uint64_t in_class_id);
  // True only when the bin is held in full — a partial holding does not
  // satisfy a skip decision.
  bool has(uint8_t class_id, uint64_t in_class_id) const;
  // Payload bytes held for the bin: 0 if absent, the partial byte count,
  // or the recorded count for complete bins (callers check has() first
  // for completeness).  Servers use this as the resume offset (BinWindow
  // skip) when continuing a byte-limited delivery.
  uint64_t received_bytes(uint8_t class_id, uint64_t in_class_id) const;
  void clear();
  size_t size() const { return bins_.size(); }

  // Format as a JPIP model request field value (§C.9).
  // Uses Hm (main header), Ht (tile header), M (metadata), Hp (precinct)
  // with range compression for consecutive IDs.
  std::string format() const;

  // Parse a model request field value into a CacheModel.
  static CacheModel parse(const std::string &model_str);

  // Apply a model request field value onto this model in statement order
  // (§C.9.2): additive statements mark bins, "-"-prefixed subtractive
  // statements unmark them.  Partial-bin statements (":bytes" qualifier)
  // unmark — we only emit complete bins, so a partial holding must be
  // re-sent in full rather than skipped.  Used by session servers to fold
  // a request's `model=` updates into the channel's persistent model.
  void apply(const std::string &model_str);

 private:
  static uint64_t key(uint8_t cls, uint64_t id) {
    return (static_cast<uint64_t>(cls) << 56) | id;
  }
  struct Entry {
    uint64_t bytes    = 0;
    bool     complete = false;
  };
  std::unordered_map<uint64_t, Entry> bins_;
};

}  // namespace jpip
}  // namespace open_htj2k

#endif  // OPENHTJ2K_CACHE_MODEL_HPP
