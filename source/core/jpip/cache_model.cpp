// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.

#include "cache_model.hpp"
#include "jpp_message.hpp"

#include <algorithm>
#include <cstdlib>
#include <sstream>
#include <vector>

namespace open_htj2k {
namespace jpip {

void CacheModel::mark(uint8_t class_id, uint64_t in_class_id) {
  bins_.insert(key(class_id, in_class_id));
}

bool CacheModel::has(uint8_t class_id, uint64_t in_class_id) const {
  return bins_.count(key(class_id, in_class_id)) > 0;
}

void CacheModel::clear() { bins_.clear(); }

// Class descriptor prefix per §C.9.
static const char *class_prefix(uint8_t cls) {
  switch (cls) {
    case kMsgClassMainHeader: return "Hm";
    case kMsgClassTileHeader: return "Ht";
    case kMsgClassPrecinct:   return "Hp";
    case kMsgClassMetadata:   return "M";
    default:                  return nullptr;
  }
}

static uint8_t prefix_to_class(const std::string &s, size_t &pos) {
  if (pos + 2 <= s.size()) {
    if (s[pos] == 'H' && s[pos + 1] == 'm') { pos += 2; return kMsgClassMainHeader; }
    if (s[pos] == 'H' && s[pos + 1] == 't') { pos += 2; return kMsgClassTileHeader; }
    if (s[pos] == 'H' && s[pos + 1] == 'p') { pos += 2; return kMsgClassPrecinct; }
  }
  if (pos < s.size() && s[pos] == 'M') { pos += 1; return kMsgClassMetadata; }
  return 0xFF;
}

std::string CacheModel::format() const {
  // Group bins by class, sort IDs, compress into ranges.
  struct ClassGroup { uint8_t cls; std::vector<uint64_t> ids; };
  std::vector<ClassGroup> groups;
  for (uint64_t k : bins_) {
    uint8_t cls = static_cast<uint8_t>(k >> 56);
    uint64_t id = k & 0x00FFFFFFFFFFFFFF;
    auto it = std::find_if(groups.begin(), groups.end(),
                           [cls](const ClassGroup &g) { return g.cls == cls; });
    if (it == groups.end()) {
      groups.push_back({cls, {id}});
    } else {
      it->ids.push_back(id);
    }
  }

  std::string out;
  for (auto &g : groups) {
    const char *pfx = class_prefix(g.cls);
    if (!pfx) continue;
    std::sort(g.ids.begin(), g.ids.end());

    // Main header (class 6) has only id=0 — emit just "Hm"
    if (g.cls == kMsgClassMainHeader) {
      if (!out.empty()) out += ',';
      out += pfx;
      continue;
    }

    // Compress into ranges: 0-3,5,7-10
    size_t i = 0;
    while (i < g.ids.size()) {
      if (!out.empty()) out += ',';
      out += pfx;
      uint64_t start = g.ids[i];
      uint64_t end = start;
      while (i + 1 < g.ids.size() && g.ids[i + 1] == end + 1) { ++end; ++i; }
      out += std::to_string(start);
      if (end != start) { out += '-'; out += std::to_string(end); }
      ++i;
    }
  }
  return out;
}

CacheModel CacheModel::parse(const std::string &model_str) {
  CacheModel m;
  if (model_str.empty()) return m;

  // Split by comma
  std::istringstream ss(model_str);
  std::string token;
  while (std::getline(ss, token, ',')) {
    size_t pos = 0;
    uint8_t cls = prefix_to_class(token, pos);
    if (cls == 0xFF) continue;

    // Main header: just "Hm" with no ID
    if (cls == kMsgClassMainHeader) {
      m.mark(cls, 0);
      continue;
    }

    // Parse ID or range: "5" or "5-10"
    if (pos >= token.size()) continue;
    const char *numstart = token.c_str() + pos;
    char *numend = nullptr;
    uint64_t start = std::strtoull(numstart, &numend, 10);
    uint64_t end = start;
    if (numend && *numend == '-') {
      end = std::strtoull(numend + 1, nullptr, 10);
    }
    for (uint64_t id = start; id <= end; ++id) {
      m.mark(cls, id);
    }
  }
  return m;
}

}  // namespace jpip
}  // namespace open_htj2k
