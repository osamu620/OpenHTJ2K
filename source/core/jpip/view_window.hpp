// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.
//
// JPIP view-window → precinct-set resolver.  Implements ISO/IEC 15444-9
// §C.4 (fsiz/roff/rsiz) + §M.4.1 (server determination of relevant
// precincts).  Used by the Phase-1 demo and by any future JPIP server to
// translate a client's view-window request into the set of precincts that
// must be delivered for a correct foveated reconstruction.
#pragma once
#include <cstdint>
#include <vector>

#include "precinct_index.hpp"

namespace open_htj2k {
namespace jpip {

struct ViewWindow {
  // §C.4.2 Frame Size: target image resolution.
  uint32_t fx = 0;
  uint32_t fy = 0;
  // §C.4.3/§C.4.4 Offset and Region Size (on the fx/fy grid).
  uint32_t ox = 0;
  uint32_t oy = 0;
  uint32_t sx = 0;
  uint32_t sy = 0;
  // §C.4.8 Components (empty = all).
  std::vector<uint16_t> comps;
  // §C.4.2 round-direction.
  enum class Round : uint8_t { Down = 0, Up = 1, Closest = 2 };
  Round round = Round::Down;
};

// Resolve a view-window into the set of precincts a server must deliver.
// Returned keys are deterministic and sorted by (t, c, r, p_rc).  Duplicates
// never appear.  Codestreams with DFS/POC are NOT yet supported (Phase 1).
OPENHTJ2K_JPIP_EXPORT std::vector<PrecinctKey>
resolve_view_window(const CodestreamIndex &idx, const ViewWindow &vw);

// Pick the discard level r* per §C.4.2 Table C.1.  Exposed for testing.
OPENHTJ2K_JPIP_EXPORT uint8_t pick_discard_level(const CodestreamIndex &idx,
                                                 const ViewWindow &vw);

}  // namespace jpip
}  // namespace open_htj2k
