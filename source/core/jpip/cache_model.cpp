// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.

#include "cache_model.hpp"
#include "jpp_message.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace open_htj2k {
namespace jpip {

// Upper bound on the number of data-bin operations a single apply() call may
// perform.  A malformed range such as "P0-18446744073709551615" would
// otherwise spin ~2^64 iterations, each inserting a map entry — unbounded
// memory and CPU.  The cache model is advisory: under-applying it only makes
// the server re-send data the client already holds (correct, just not
// optimal), so stopping early is always safe.  The cap sits far above the
// data-bin count of any realistic gigapixel session (~10^5–10^6) while
// keeping worst-case memory bounded (~40 bytes per map entry).
static constexpr uint64_t kMaxModelBinOps = 1u << 22;  // 4,194,304

void CacheModel::mark(uint8_t class_id, uint64_t in_class_id) {
  bins_[key(class_id, in_class_id)].complete = true;
}

void CacheModel::mark_partial(uint8_t class_id, uint64_t in_class_id, uint64_t bytes) {
  Entry &e = bins_[key(class_id, in_class_id)];
  if (!e.complete && bytes > e.bytes) e.bytes = bytes;
}

void CacheModel::unmark(uint8_t class_id, uint64_t in_class_id) {
  bins_.erase(key(class_id, in_class_id));
}

bool CacheModel::has(uint8_t class_id, uint64_t in_class_id) const {
  auto it = bins_.find(key(class_id, in_class_id));
  return it != bins_.end() && it->second.complete;
}

uint64_t CacheModel::received_bytes(uint8_t class_id, uint64_t in_class_id) const {
  auto it = bins_.find(key(class_id, in_class_id));
  return it == bins_.end() ? 0 : it->second.bytes;
}

void CacheModel::clear() { bins_.clear(); }

// Class descriptor prefix per §C.9.3.1.  Precinct data-bins use "P", NOT
// "Hp" — "Hp" was an earlier misreading of the spec that made our server
// silently discard every precinct entry clients put in their outgoing
// cache model, so the server re-sent precincts the client already had.
// On stateful clients this filled the session cache with duplicates.
static const char *class_prefix(uint8_t cls) {
  switch (cls) {
    case kMsgClassMainHeader: return "Hm";
    case kMsgClassTileHeader: return "H";
    case kMsgClassPrecinct:   return "P";
    case kMsgClassMetadata:   return "M";
    default:                  return nullptr;
  }
}

static uint8_t prefix_to_class(const char *&p, const char *end) {
  // Order matters — "Hm" and "Ht" must be tried before bare "H".  Accept
  // legacy "Hp" / "Ht" forms too so old clients (and our own pre-fix
  // serialiser output that may be cached anywhere) round-trip cleanly.
  if (p + 2 <= end) {
    if (p[0] == 'H' && p[1] == 'm') { p += 2; return kMsgClassMainHeader; }
    if (p[0] == 'H' && p[1] == 't') { p += 2; return kMsgClassTileHeader; }
    if (p[0] == 'H' && p[1] == 'p') { p += 2; return kMsgClassPrecinct; }
  }
  if (p < end && *p == 'P') { p += 1; return kMsgClassPrecinct; }
  if (p < end && *p == 'H') { p += 1; return kMsgClassTileHeader; }
  if (p < end && *p == 'M') { p += 1; return kMsgClassMetadata; }
  return 0xFF;
}

std::string CacheModel::format() const {
  // Group bins by class, sort IDs, compress complete bins into ranges.
  // Partial holdings are emitted individually with the §C.9.2 ":bytes"
  // qualifier and never participate in range compression.
  struct ClassGroup {
    uint8_t cls;
    std::vector<uint64_t> complete_ids;
    std::vector<std::pair<uint64_t, uint64_t>> partial;  // (id, bytes)
  };
  std::vector<ClassGroup> groups;
  for (const auto &kv : bins_) {
    uint8_t cls = static_cast<uint8_t>(kv.first >> 56);
    uint64_t id = kv.first & 0x00FFFFFFFFFFFFFF;
    auto it = std::find_if(groups.begin(), groups.end(),
                           [cls](const ClassGroup &g) { return g.cls == cls; });
    if (it == groups.end()) {
      groups.push_back({cls, {}, {}});
      it = groups.end() - 1;
    }
    if (kv.second.complete) {
      it->complete_ids.push_back(id);
    } else if (kv.second.bytes > 0) {
      it->partial.emplace_back(id, kv.second.bytes);
    }
  }

  std::string out;
  for (auto &g : groups) {
    const char *pfx = class_prefix(g.cls);
    if (!pfx) continue;
    std::sort(g.complete_ids.begin(), g.complete_ids.end());
    std::sort(g.partial.begin(), g.partial.end());

    // Main header (class 6) has only id=0 — emit just "Hm" when complete.
    if (g.cls == kMsgClassMainHeader) {
      if (!g.complete_ids.empty()) {
        if (!out.empty()) out += ',';
        out += pfx;
      }
      for (const auto &p : g.partial) {
        if (!out.empty()) out += ',';
        out += pfx;
        out += ':';
        out += std::to_string(p.second);
      }
      continue;
    }

    // Compress complete bins into ranges: 0-3,5,7-10
    size_t i = 0;
    while (i < g.complete_ids.size()) {
      if (!out.empty()) out += ',';
      out += pfx;
      uint64_t start = g.complete_ids[i];
      uint64_t end = start;
      while (i + 1 < g.complete_ids.size() && g.complete_ids[i + 1] == end + 1) { ++end; ++i; }
      out += std::to_string(start);
      if (end != start) { out += '-'; out += std::to_string(end); }
      ++i;
    }
    for (const auto &p : g.partial) {
      if (!out.empty()) out += ',';
      out += pfx;
      out += std::to_string(p.first);
      out += ':';
      out += std::to_string(p.second);
    }
  }
  return out;
}

CacheModel CacheModel::parse(const std::string &model_str) {
  CacheModel m;
  m.apply(model_str);
  return m;
}

void CacheModel::apply(const std::string &model_str) {
  CacheModel &m = *this;
  if (model_str.empty()) return;

  // Walk comma-separated tokens in place.  Viewers on gigapixel sessions
  // send model strings that grow with their cache, so this runs on every
  // stateless request with potentially long input — no per-token string
  // or stringstream allocations.  strtoull naturally stops at ',' / '\0',
  // so number parses never need a bounded copy; only character searches
  // (memchr) are clamped to the token.
  const char *p         = model_str.c_str();
  const char *const end = p + model_str.size();
  uint64_t budget       = kMaxModelBinOps;
  while (p < end) {
    const char *tok_end =
        static_cast<const char *>(std::memchr(p, ',', static_cast<size_t>(end - p)));
    if (tok_end == nullptr) tok_end = end;
    const char *q = p;
    p             = (tok_end < end) ? tok_end + 1 : end;  // advance for next iteration

    // §C.9.2: a "-" prefix makes the statement subtractive — the client
    // has discarded the bin and the server must forget it was sent.
    bool subtractive = false;
    if (q < tok_end && *q == '-') { subtractive = true; ++q; }
    uint8_t cls = prefix_to_class(q, tok_end);
    if (cls == 0xFF) continue;

    // Main header: just "Hm" with no ID.  A ":bytes" qualifier (§C.9.2,
    // partial holding) can still follow.
    if (cls == kMsgClassMainHeader) {
      const char *colon =
          static_cast<const char *>(std::memchr(q, ':', static_cast<size_t>(tok_end - q)));
      if (subtractive) {
        m.unmark(cls, 0);
      } else if (colon != nullptr) {
        m.mark_partial(cls, 0, std::strtoull(colon + 1, nullptr, 10));
      } else {
        m.mark(cls, 0);
      }
      continue;
    }

    // Parse ID or range: "5" or "5-10", optionally with a ":bytes"
    // qualifier (§C.9.2 explicit-bin-descriptor) recording a partial
    // holding.  Partial bins resume from their byte offset on the next
    // delivery (BinWindow skip).  Subtractive statements discard the
    // holding entirely — conservative for "-P5:1234" too, so the whole
    // bin is re-sent rather than risking a withheld tail.
    if (q >= tok_end) continue;
    char *numend   = nullptr;
    uint64_t start = std::strtoull(q, &numend, 10);
    if (numend == q) continue;
    uint64_t end_id = start;
    if (numend < tok_end && *numend == '-') {
      const char *hs = numend + 1;
      end_id = std::strtoull(hs, &numend, 10);
      if (numend == hs || end_id < start) continue;
    }
    uint64_t partial_bytes = 0;
    const bool partial     = (numend < tok_end && *numend == ':');
    if (partial) partial_bytes = std::strtoull(numend + 1, nullptr, 10);
    for (uint64_t id = start; id <= end_id; ++id) {
      if (budget == 0) return;  // bin-operation budget spent — ignore the rest
      --budget;
      if (subtractive) m.unmark(cls, id);
      else if (partial) m.mark_partial(cls, id, partial_bytes);
      else m.mark(cls, id);
    }
  }
}

}  // namespace jpip
}  // namespace open_htj2k
