// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.

#include "view_window.hpp"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>

namespace open_htj2k {
namespace jpip {

namespace {

inline uint32_t ceil_div_u(uint32_t a, uint32_t b) { return (a + b - 1u) / b; }

// Canvas frame size at discard level r per §C.4.1 Eq. C-1:
//   fx'(r) = ceil(Xsiz / 2^r) - ceil(XOsiz / 2^r)
// Works for r far larger than any tile-component NL — at that point the
// result saturates at 1 (or 0 if the canvas is empty).
void fsiz_at(const CodestreamIndex &idx, uint8_t r, uint32_t &fx, uint32_t &fy) {
  const ImageGeometry &g = idx.geometry();
  const uint32_t div = 1u << r;
  fx = ceil_div_u(g.canvas_size.x, div) - ceil_div_u(g.canvas_origin.x, div);
  fy = ceil_div_u(g.canvas_size.y, div) - ceil_div_u(g.canvas_origin.y, div);
}

// §M.4.1 DWT-synthesis expansion margin on the subband reference grid for
// resolution r in a tile-component with `levels_above` IDWT steps remaining
// between this resolution and the reconstructed image.  Upper bound of the
// geometric series L_synth * (1 + 1/2 + 1/4 + ...) < 2·L_synth.
uint32_t dwt_margin_subband(bool irreversible, uint8_t levels_above) {
  if (levels_above == 0) return 0;
  const uint32_t L_synth = irreversible ? 4u : 2u;  // 9/7 vs 5/3 synthesis support
  return 2u * L_synth;
}

// Saturating-subtraction on uint32_t.
inline uint32_t sub_sat(uint32_t a, uint32_t b) { return (a > b) ? (a - b) : 0u; }

}  // namespace

uint8_t pick_discard_level(const CodestreamIndex &idx, const ViewWindow &vw) {
  // Cap search at max_NL across all (t, c) — r larger than that collapses
  // to the LL subband and no further reduction is possible.
  const uint8_t r_max = idx.max_NL();
  const uint32_t req_fx = vw.fx ? vw.fx : idx.geometry().canvas_size.x;
  const uint32_t req_fy = vw.fy ? vw.fy : idx.geometry().canvas_size.y;

  switch (vw.round) {
    case ViewWindow::Round::Down: {
      // Largest available resolution that fits inside (req_fx, req_fy).
      // Iterate r=0 → r_max, pick the smallest r where fx'(r) ≤ req.  If
      // none fits (can only happen when req is smaller than the LL), fall
      // back to r_max (smallest available).
      for (uint8_t r = 0; r <= r_max; ++r) {
        uint32_t fx = 0, fy = 0;
        fsiz_at(idx, r, fx, fy);
        if (fx <= req_fx && fy <= req_fy) return r;
      }
      return r_max;
    }
    case ViewWindow::Round::Up: {
      // Smallest available resolution that contains (req_fx, req_fy).  Walk
      // r = r_max → 0; pick the largest r (smallest fx') where fx' ≥ req.
      for (int r = r_max; r >= 0; --r) {
        uint32_t fx = 0, fy = 0;
        fsiz_at(idx, static_cast<uint8_t>(r), fx, fy);
        if (fx >= req_fx && fy >= req_fy) return static_cast<uint8_t>(r);
      }
      return 0;
    }
    case ViewWindow::Round::Closest:
    default: {
      // Minimise |fx'(r)*fy'(r) - req_fx*req_fy|; ties → larger area (smaller r).
      const uint64_t target = static_cast<uint64_t>(req_fx) * req_fy;
      uint64_t best_diff = UINT64_MAX;
      uint8_t  best_r    = 0;
      for (uint8_t r = 0; r <= r_max; ++r) {
        uint32_t fx = 0, fy = 0;
        fsiz_at(idx, r, fx, fy);
        const uint64_t area = static_cast<uint64_t>(fx) * fy;
        const uint64_t diff = (area > target) ? (area - target) : (target - area);
        if (diff < best_diff) {
          best_diff = diff;
          best_r    = r;
        }
      }
      return best_r;
    }
  }
}

std::vector<PrecinctKey> resolve_view_window(const CodestreamIndex &idx,
                                             const ViewWindow &vw) {
  std::vector<PrecinctKey> out;
  if (idx.num_components() == 0 || idx.num_tiles() == 0) return out;

  // 1. Discard level — common across all tile-components; capped per-(t,c) below.
  const uint8_t r_star = pick_discard_level(idx, vw);

  // 2. Map (ox, oy, sx, sy) to canvas reference grid.  If Region Size is
  //    omitted (sx == 0 && sy == 0), treat the request as whole-image at the
  //    chosen discard level.
  const ImageGeometry &g = idx.geometry();
  const uint32_t scale = 1u << r_star;

  uint32_t A_x = vw.ox * scale + g.canvas_origin.x;
  uint32_t A_y = vw.oy * scale + g.canvas_origin.y;
  uint32_t B_x = (vw.sx == 0 && vw.sy == 0) ? g.canvas_size.x
                                            : (vw.ox + vw.sx) * scale + g.canvas_origin.x;
  uint32_t B_y = (vw.sx == 0 && vw.sy == 0) ? g.canvas_size.y
                                            : (vw.oy + vw.sy) * scale + g.canvas_origin.y;
  // Clip to canvas.
  A_x = std::min(A_x, g.canvas_size.x);
  A_y = std::min(A_y, g.canvas_size.y);
  B_x = std::min(B_x, g.canvas_size.x);
  B_y = std::min(B_y, g.canvas_size.y);
  if (A_x >= B_x || A_y >= B_y) return out;  // empty view-window

  // 3. Component selection.
  std::vector<uint16_t> comps = vw.comps;
  if (comps.empty()) {
    comps.reserve(idx.num_components());
    for (uint16_t c = 0; c < idx.num_components(); ++c) comps.push_back(c);
  }

  // 4. Enumerate intersecting tiles.
  const uint32_t tcol0 = sub_sat(A_x, g.tile_origin.x) / g.tile_size.x;
  const uint32_t trow0 = sub_sat(A_y, g.tile_origin.y) / g.tile_size.y;
  const uint32_t tcol1 = std::min(idx.num_tiles_x(), ceil_div_u(B_x - g.tile_origin.x, g.tile_size.x));
  const uint32_t trow1 = std::min(idx.num_tiles_y(), ceil_div_u(B_y - g.tile_origin.y, g.tile_size.y));

  for (uint32_t trow = trow0; trow < trow1; ++trow) {
    for (uint32_t tcol = tcol0; tcol < tcol1; ++tcol) {
      const uint16_t t = static_cast<uint16_t>(trow * idx.num_tiles_x() + tcol);
      const uint32_t tile_x0 = std::max(g.tile_origin.x + tcol * g.tile_size.x, g.canvas_origin.x);
      const uint32_t tile_y0 = std::max(g.tile_origin.y + trow * g.tile_size.y, g.canvas_origin.y);
      const uint32_t tile_x1 = std::min(g.tile_origin.x + (tcol + 1u) * g.tile_size.x, g.canvas_size.x);
      const uint32_t tile_y1 = std::min(g.tile_origin.y + (trow + 1u) * g.tile_size.y, g.canvas_size.y);

      // Clip canvas region [A, B) to this tile.
      const uint32_t clipA_x = std::max(A_x, tile_x0);
      const uint32_t clipA_y = std::max(A_y, tile_y0);
      const uint32_t clipB_x = std::min(B_x, tile_x1);
      const uint32_t clipB_y = std::min(B_y, tile_y1);
      if (clipA_x >= clipB_x || clipA_y >= clipB_y) continue;

      for (uint16_t c : comps) {
        if (c >= idx.num_components()) continue;
        const Point2 sub = idx.subsampling(c);
        // Tile-component coordinates on subsampled grid.
        const uint32_t tc_A_x = ceil_div_u(clipA_x, sub.x);
        const uint32_t tc_A_y = ceil_div_u(clipA_y, sub.y);
        const uint32_t tc_B_x = ceil_div_u(clipB_x, sub.x);
        const uint32_t tc_B_y = ceil_div_u(clipB_y, sub.y);
        if (tc_A_x >= tc_B_x || tc_A_y >= tc_B_y) continue;

        const TileComponentInfo &tc = idx.tile_component(t, c);
        const uint8_t NL_tc = tc.NL;

        // §M.4.1: if r_star > NL_tc, only the LL survives; else keep r in
        // [0, NL_tc - r_star] inclusive (drop the top r_star resolutions).
        const uint8_t eff_top_r =
            (r_star >= NL_tc) ? 0u : static_cast<uint8_t>(NL_tc - r_star);

        for (uint8_t r = 0; r <= eff_top_r; ++r) {
          const uint32_t scale_r = 1u << (NL_tc - r);
          // Subband coordinates for the clipped tile-component region.
          uint32_t sb_A_x = tc_A_x / scale_r;
          uint32_t sb_A_y = tc_A_y / scale_r;
          uint32_t sb_B_x = ceil_div_u(tc_B_x, scale_r);
          uint32_t sb_B_y = ceil_div_u(tc_B_y, scale_r);

          // DWT expansion: overshoot on the subband grid.
          const uint32_t margin =
              dwt_margin_subband(idx.is_irreversible(), static_cast<uint8_t>(NL_tc - r));
          sb_A_x = sub_sat(sb_A_x, margin);
          sb_A_y = sub_sat(sb_A_y, margin);
          sb_B_x += margin;
          sb_B_y += margin;

          const uint32_t PPx = 1u << tc.log2PPx[r];
          const uint32_t PPy = 1u << tc.log2PPy[r];
          const uint32_t idxoff_x = tc.respos0_x[r] / PPx;
          const uint32_t idxoff_y = tc.respos0_y[r] / PPy;

          // Precinct grid extents on this resolution.
          int64_t px_lo = static_cast<int64_t>(sb_A_x / PPx) - idxoff_x;
          int64_t py_lo = static_cast<int64_t>(sb_A_y / PPy) - idxoff_y;
          int64_t px_hi = static_cast<int64_t>(ceil_div_u(sb_B_x, PPx)) - idxoff_x;
          int64_t py_hi = static_cast<int64_t>(ceil_div_u(sb_B_y, PPy)) - idxoff_y;
          px_lo = std::max<int64_t>(0, px_lo);
          py_lo = std::max<int64_t>(0, py_lo);
          px_hi = std::min<int64_t>(tc.npw[r], px_hi);
          py_hi = std::min<int64_t>(tc.nph[r], py_hi);

          for (int64_t py = py_lo; py < py_hi; ++py) {
            for (int64_t px = px_lo; px < px_hi; ++px) {
              PrecinctKey k;
              k.t    = t;
              k.c    = c;
              k.r    = r;
              k.p_rc = static_cast<uint32_t>(py * tc.npw[r] + px);
              out.push_back(k);
            }
          }
        }
      }
    }
  }
  return out;
}

}  // namespace jpip
}  // namespace open_htj2k
