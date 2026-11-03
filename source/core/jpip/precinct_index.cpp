// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.

#include "precinct_index.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

#include "codestream.hpp"
#include "j2kmarkers.hpp"
#include "open_htj2k_typedef.hpp"
#include "utils.hpp"

namespace open_htj2k {
namespace jpip {

namespace {

// ceil(a / b), b > 0, without the overflow of (a + b - 1) / b for a near 2^32.
inline uint32_t ceil_div_u(uint32_t a, uint32_t b) { return a / b + ((a % b != 0) ? 1u : 0u); }

// Most tile-components an index is built for (65535 tiles x 16 components fit).
const uint64_t kMaxTileComponents = uint64_t{1} << 20;

// Find the COC marker that overrides component `c`, or nullptr if none.
const COC_marker *find_coc(const j2k_main_header &mh, uint16_t c) {
  for (const auto &coc : mh.COC) {
    if (coc->get_component_index() == c) return coc.get();
  }
  return nullptr;
}

// Mirrors the per-resolution geometry computation in j2k_tile_component::
// create_resolutions (coding_units.cpp:2820+).  No DFS support yet — fine for
// the Phase-1 test assets, which use a fixed isotropic dyadic decomposition.
void build_tile_component(j2k_main_header &mh, uint16_t c,
                          uint32_t tile_x0, uint32_t tile_y0,
                          uint32_t tile_x1, uint32_t tile_y1,
                          TileComponentInfo &out) {
  // get_subsampling_factor / get_precinct_size are non-const in the existing
  // marker classes, so we accept a non-const main_header here.
  element_siz sub;
  mh.SIZ->get_subsampling_factor(sub, c);
  const uint32_t tc_x0 = ceil_div_u(tile_x0, sub.x);
  const uint32_t tc_y0 = ceil_div_u(tile_y0, sub.y);
  const uint32_t tc_x1 = ceil_div_u(tile_x1, sub.x);
  const uint32_t tc_y1 = ceil_div_u(tile_y1, sub.y);

  COC_marker *coc = const_cast<COC_marker *>(find_coc(mh, c));
  const uint8_t NL_tc =
      coc ? coc->get_dwt_levels() : mh.COD->get_dwt_levels();

  out.NL = NL_tc;
  out.npw.assign(NL_tc + 1u, 0);
  out.nph.assign(NL_tc + 1u, 0);
  out.s_offset.assign(NL_tc + 1u, 0);
  out.log2PPx.assign(NL_tc + 1u, 0);
  out.log2PPy.assign(NL_tc + 1u, 0);
  out.respos0_x.assign(NL_tc + 1u, 0);
  out.respos0_y.assign(NL_tc + 1u, 0);
  out.tc_pos0.x = tc_x0;
  out.tc_pos0.y = tc_y0;
  out.tc_pos1.x = tc_x1;
  out.tc_pos1.y = tc_y1;

  uint64_t cumulative = 0;
  for (uint8_t r = 0; r <= NL_tc; ++r) {
    // Resolution r lies NL_tc - r levels down, which reaches 32 with 32 DWT
    // levels: no 32-bit shift by it.
    const uint32_t depth     = static_cast<uint32_t>(NL_tc - r);
    const uint32_t respos0_x = ceil_div_pow2(tc_x0, depth);
    const uint32_t respos0_y = ceil_div_pow2(tc_y0, depth);
    const uint32_t respos1_x = ceil_div_pow2(tc_x1, depth);
    const uint32_t respos1_y = ceil_div_pow2(tc_y1, depth);

    element_siz log2PP;
    if (coc) {
      coc->get_precinct_size(log2PP, r);
    } else {
      mh.COD->get_precinct_size(log2PP, r);
    }

    const uint32_t npw = (respos1_x > respos0_x)
                             ? (ceil_div_pow2(respos1_x, log2PP.x) - floor_div_pow2(respos0_x, log2PP.x))
                             : 0u;
    const uint32_t nph = (respos1_y > respos0_y)
                             ? (ceil_div_pow2(respos1_y, log2PP.y) - floor_div_pow2(respos0_y, log2PP.y))
                             : 0u;

    out.npw[r]       = npw;
    out.nph[r]       = nph;
    out.s_offset[r]  = static_cast<uint32_t>(cumulative);
    out.log2PPx[r]   = static_cast<uint8_t>(log2PP.x);
    out.log2PPy[r]   = static_cast<uint8_t>(log2PP.y);
    out.respos0_x[r] = respos0_x;
    out.respos0_y[r] = respos0_y;
    cumulative += uint64_t{npw} * nph;
    // Precinct indices and sequence numbers are 32-bit (B.6, A.3.2.1).
    if (cumulative > UINT32_MAX) {
      throw std::runtime_error("CodestreamIndex::build: too many precincts in a tile-component");
    }
  }
  out.total = static_cast<uint32_t>(cumulative);
}

}  // namespace

std::unique_ptr<CodestreamIndex> CodestreamIndex::build(const uint8_t *codestream,
                                                       std::size_t len) {
  if (codestream == nullptr || len < 2) {
    throw std::runtime_error("CodestreamIndex::build: empty input");
  }
  if (len > UINT32_MAX) {
    throw std::runtime_error("CodestreamIndex::build: codestream > 4 GiB unsupported");
  }

  // Borrow the caller's buffer; main_header.read() only consumes through SOC..SOT.
  // The marker readers require 16 bytes of readable padding past the end (SIMD
  // over-read).  borrow_memory documents this requirement; for raw J2C parsing
  // through the main header it is safe in practice because SIZ/COD/QCD parsing
  // does not over-read the buffer end.  We copy into a fresh j2c_src_memory
  // when the caller-supplied buffer is too tight (no padding contract).
  j2c_src_memory in;
  in.alloc_memory(len);
  // alloc_memory zero-pads internally; copy the codestream bytes in.
  std::copy_n(codestream, len, in.get_buf_pos());

  j2k_main_header mh;
  // read() returns EXIT_SUCCESS or throws on malformed input.
  mh.read(in);

  if (!mh.SIZ || !mh.COD) {
    throw std::runtime_error("CodestreamIndex::build: missing SIZ or COD");
  }

  std::unique_ptr<CodestreamIndex> idx(new CodestreamIndex());
  idx->num_components_  = mh.SIZ->get_num_components();
  idx->progression_     = mh.COD->get_progression_order();
  idx->num_layers_      = mh.COD->get_number_of_layers();
  // COD Scod (§A.6.1 Table A.13): bit 1 = SOP in use, bit 2 = EPH in use.
  idx->use_SOP_         = mh.COD->is_use_SOP();
  idx->use_EPH_         = mh.COD->is_use_EPH();
  idx->is_irreversible_ = (mh.COD->get_transformation() == 0);
  mh.get_number_of_tiles(idx->num_tiles_x_, idx->num_tiles_y_);

  element_siz canvas, origin, tsize, torigin;
  mh.SIZ->get_image_size(canvas);
  mh.SIZ->get_image_origin(origin);
  mh.SIZ->get_tile_size(tsize);
  mh.SIZ->get_tile_origin(torigin);
  idx->geometry_.canvas_size   = {canvas.x, canvas.y};
  idx->geometry_.canvas_origin = {origin.x, origin.y};
  idx->geometry_.tile_size     = {tsize.x, tsize.y};
  idx->geometry_.tile_origin   = {torigin.x, torigin.y};

  idx->subsampling_.resize(idx->num_components_);
  for (uint16_t c = 0; c < idx->num_components_; ++c) {
    element_siz sub;
    mh.SIZ->get_subsampling_factor(sub, c);
    idx->subsampling_[c] = {sub.x, sub.y};
  }

  const uint64_t total_tc =
      static_cast<uint64_t>(idx->num_tiles()) * idx->num_components_;
  // The index holds every tile-component up front, about 450 bytes each: the
  // 65535 tiles and 16384 components a main header may declare would take
  // hundreds of GB.  Allow what 65535 tiles of 16 components need.
  if (total_tc > kMaxTileComponents) {
    throw std::runtime_error("CodestreamIndex::build: " + std::to_string(idx->num_tiles()) + " tiles x "
                             + std::to_string(idx->num_components_) + " components exceed the "
                             + std::to_string(kMaxTileComponents) + " tile-components the index supports");
  }
  idx->tcinfo_.resize(static_cast<std::size_t>(total_tc));

  for (uint32_t ty = 0; ty < idx->num_tiles_y_; ++ty) {
    for (uint32_t tx = 0; tx < idx->num_tiles_x_; ++tx) {
      const uint32_t t = ty * idx->num_tiles_x_ + tx;
      // 64 bits: the end of the last tile may lie past 2^32.
      const uint32_t tile_x0 =
          static_cast<uint32_t>(std::max<uint64_t>(torigin.x + uint64_t{tx} * tsize.x, origin.x));
      const uint32_t tile_y0 =
          static_cast<uint32_t>(std::max<uint64_t>(torigin.y + uint64_t{ty} * tsize.y, origin.y));
      const uint32_t tile_x1 =
          static_cast<uint32_t>(std::min<uint64_t>(torigin.x + (tx + uint64_t{1}) * tsize.x, canvas.x));
      const uint32_t tile_y1 =
          static_cast<uint32_t>(std::min<uint64_t>(torigin.y + (ty + uint64_t{1}) * tsize.y, canvas.y));
      for (uint16_t c = 0; c < idx->num_components_; ++c) {
        build_tile_component(mh, c, tile_x0, tile_y0, tile_x1, tile_y1,
                             idx->tcinfo_[static_cast<std::size_t>(t) *
                                              idx->num_components_ + c]);
      }
    }
  }

  return idx;
}

const TileComponentInfo &CodestreamIndex::tile_component(uint16_t t, uint16_t c) const {
  if (t >= num_tiles() || c >= num_components_) {
    throw std::out_of_range("CodestreamIndex::tile_component");
  }
  return tcinfo_[static_cast<std::size_t>(t) * num_components_ + c];
}

uint32_t CodestreamIndex::s(uint16_t t, uint16_t c, uint8_t r, uint32_t p_rc) const {
  const TileComponentInfo &info = tile_component(t, c);
  if (r > info.NL || p_rc >= info.npw[r] * info.nph[r]) {
    throw std::out_of_range("CodestreamIndex::s");
  }
  return info.s_offset[r] + p_rc;
}

uint64_t CodestreamIndex::I(uint16_t t, uint16_t c, uint8_t r, uint32_t p_rc) const {
  const uint64_t s_val = s(t, c, r, p_rc);
  return static_cast<uint64_t>(t)
         + (static_cast<uint64_t>(c) + s_val * num_components_) * num_tiles();
}

std::unique_ptr<CodestreamIndex> CodestreamIndex::build_from_main_header_bin(
    const std::vector<uint8_t> &bin) {
  if (bin.empty()) return nullptr;
  // The main-header data-bin is [SOC .. last-main-header-marker).  The
  // internal parser (j2k_main_header::read) expects to stop at an SOT
  // marker.  Appending a fake SOT (FF 90) + a valid Lsot (00 0A) +
  // 8 zero bytes for the SOT body makes the parser terminate cleanly.
  std::vector<uint8_t> buf;
  buf.reserve(bin.size() + 12);
  buf.insert(buf.end(), bin.begin(), bin.end());
  // Fake SOT marker segment: FF90 000A 0000 00000000 00 00
  const uint8_t fake_sot[] = {0xFF, 0x90, 0x00, 0x0A, 0x00, 0x00,
                              0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
  buf.insert(buf.end(), fake_sot, fake_sot + sizeof(fake_sot));
  return build(buf.data(), buf.size());
}

uint64_t CodestreamIndex::total_precincts() const {
  uint64_t sum = 0;
  for (const auto &info : tcinfo_) sum += info.total;
  return sum;
}

uint8_t CodestreamIndex::max_NL() const {
  uint8_t m = 0;
  for (const auto &info : tcinfo_) {
    if (info.NL > m) m = info.NL;
  }
  return m;
}

Point2 CodestreamIndex::subsampling(uint16_t c) const {
  if (c >= num_components_) {
    throw std::out_of_range("CodestreamIndex::subsampling");
  }
  return subsampling_[c];
}

}  // namespace jpip
}  // namespace open_htj2k
