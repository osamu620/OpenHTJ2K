// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.
//
// JPIP client cache model (§C.9) — tracks which data-bins the client
// has received so the server can skip redundant data.

#ifndef OPENHTJ2K_CACHE_MODEL_HPP
#define OPENHTJ2K_CACHE_MODEL_HPP

#include <cstdint>
#include <string>
#include <unordered_set>

#if defined(_MSC_VER) && !defined(OHTJ2K_STATIC)
  #define OPENHTJ2K_JPIP_EXPORT __declspec(dllexport)
#else
  #define OPENHTJ2K_JPIP_EXPORT
#endif

namespace open_htj2k {
namespace jpip {

class OPENHTJ2K_JPIP_EXPORT CacheModel {
 public:
  void mark(uint8_t class_id, uint64_t in_class_id);
  bool has(uint8_t class_id, uint64_t in_class_id) const;
  void clear();
  size_t size() const { return bins_.size(); }

  // Format as a JPIP model request field value (§C.9).
  // Uses Hm (main header), Ht (tile header), M (metadata), Hp (precinct)
  // with range compression for consecutive IDs.
  std::string format() const;

  // Parse a model request field value into a CacheModel.
  static CacheModel parse(const std::string &model_str);

 private:
  static uint64_t key(uint8_t cls, uint64_t id) {
    return (static_cast<uint64_t>(cls) << 56) | id;
  }
  std::unordered_set<uint64_t> bins_;
};

}  // namespace jpip
}  // namespace open_htj2k

#endif  // OPENHTJ2K_CACHE_MODEL_HPP
