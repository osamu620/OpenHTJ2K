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

// Windows DLL linkage: export while building the library (CMake defines
// open_htj2k_EXPORTS for its sources), import in code that uses it, and
// nothing when linking the static library (OHTJ2K_STATIC).
#if defined(_MSC_VER) && !defined(OHTJ2K_STATIC)
  #if defined(open_htj2k_EXPORTS)
    #define OPENHTJ2K_JPIP_EXPORT __declspec(dllexport)
  #else
    #define OPENHTJ2K_JPIP_EXPORT __declspec(dllimport)
  #endif
#else
  #define OPENHTJ2K_JPIP_EXPORT
#endif

// MSVC C4251: the exported classes keep standard-library members.  The DLL and its
// users here are built by the same compiler and C++ runtime, which makes that safe.
#if defined(_MSC_VER)
  #pragma warning(push)
  #pragma warning(disable : 4251)
#endif

namespace open_htj2k {
namespace jpip {

class OPENHTJ2K_JPIP_EXPORT CacheModel {
 public:
  // What a model statement may name and how much a model may hold.  A model
  // costs memory per data-bin, and a statement is a few bytes that can name
  // 2^64 of them: a server gives apply() and parse() the limits of the image
  // it serves (CodestreamIndex::precinct_id_bound(), its number of tiles, the
  // metadata-bins it has), so that a client cannot make it remember bins the
  // image does not have.  The defaults bound the identifiers by nothing and
  // the model by `bins` alone.
  struct Limits {
    uint64_t precinct_ids    = UINT64_MAX;  // identifiers below this exist (exclusive bound)
    uint64_t tile_header_ids = UINT64_MAX;
    uint64_t metadata_ids    = UINT64_MAX;
    // Data-bins the model may hold: statements that would add one more are
    // ignored (the cache model is advisory, see apply()).
    uint64_t bins = static_cast<uint64_t>(1) << 22;
  };

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
  static CacheModel parse(const std::string &model_str, const Limits &limits);

  // Apply a model request field value onto this model in statement order
  // (§C.9.2): additive statements mark bins, "-"-prefixed subtractive
  // statements unmark them.  Partial-bin statements (":bytes" qualifier)
  // unmark — we only emit complete bins, so a partial holding must be
  // re-sent in full rather than skipped.  Used by session servers to fold
  // a request's `model=` updates into the channel's persistent model.
  void apply(const std::string &model_str);
  // The same within `limits`: a statement about a bin past the identifier
  // bounds is ignored, and so is one that would take the model past
  // limits.bins.  Ignoring a statement only makes the server send again
  // what the client already holds.
  void apply(const std::string &model_str, const Limits &limits);

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

#if defined(_MSC_VER)
  #pragma warning(pop)
#endif

#endif  // OPENHTJ2K_CACHE_MODEL_HPP
