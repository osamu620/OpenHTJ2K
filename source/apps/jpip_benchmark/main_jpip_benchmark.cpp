// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.
//
// JPIP foveation benchmark — measures bandwidth and decode workload
// for foveated vs full-image decode, no GUI or server needed.
//
// Usage:
//   open_htj2k_jpip_benchmark <input.j2c>
//       [--gaze-grid NxN=5]
//       [--reduce N=0]
//       [--csv output.csv]
//
// For each gaze position on an NxN grid over the canvas, measures:
//   - Foveated: 3-cone (fovea + parafovea + periphery)
//   - Full image: all precincts at full resolution
//
// Outputs per-position and summary statistics for:
//   - JPP-stream bytes (bandwidth)
//   - Precinct count (coverage)
//   - Decode time (ms)

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_set>
#include <vector>

#include "codestream_assembler.hpp"
#include "codestream_walker.hpp"
#include "data_bin_emitter.hpp"
#include "decoder.hpp"
#include "jpp_parser.hpp"
#include "packet_locator.hpp"
#include "precinct_index.hpp"
#include "view_window.hpp"

using namespace open_htj2k::jpip;
using Clock = std::chrono::steady_clock;

namespace {

std::vector<uint8_t> read_file(const char *path) {
  FILE *f = std::fopen(path, "rb");
  if (!f) { std::fprintf(stderr, "ERROR: cannot open %s\n", path); return {}; }
  std::fseek(f, 0, SEEK_END);
  auto sz = static_cast<std::size_t>(std::ftell(f));
  std::fseek(f, 0, SEEK_SET);
  std::vector<uint8_t> buf(sz);
  std::fread(buf.data(), 1, sz, f);
  std::fclose(f);
  return buf;
}

ViewWindow make_view_window(const CodestreamIndex &idx, uint32_t gx, uint32_t gy,
                            uint32_t radius, float fsiz_ratio, bool whole_image) {
  const auto &g = idx.geometry();
  ViewWindow vw;
  vw.fx = static_cast<uint32_t>(static_cast<float>(g.canvas_size.x) * fsiz_ratio);
  vw.fy = static_cast<uint32_t>(static_cast<float>(g.canvas_size.y) * fsiz_ratio);
  if (vw.fx == 0) vw.fx = 1;
  if (vw.fy == 0) vw.fy = 1;
  if (whole_image) {
    vw.ox = 0; vw.oy = 0; vw.sx = vw.fx; vw.sy = vw.fy;
  } else {
    const uint32_t gx_f = static_cast<uint32_t>(static_cast<uint64_t>(gx) * vw.fx / g.canvas_size.x);
    const uint32_t gy_f = static_cast<uint32_t>(static_cast<uint64_t>(gy) * vw.fy / g.canvas_size.y);
    const uint32_t r_f  = static_cast<uint32_t>(static_cast<uint64_t>(radius) * vw.fx / g.canvas_size.x);
    vw.ox = (gx_f > r_f) ? (gx_f - r_f) : 0u;
    vw.oy = (gy_f > r_f) ? (gy_f - r_f) : 0u;
    vw.sx = 2u * r_f;
    vw.sy = 2u * r_f;
  }
  return vw;
}

struct BenchResult {
  uint32_t gx, gy;
  // Foveated
  size_t   fov_precincts;
  size_t   fov_bytes;
  double   fov_decode_ms;
  // Per-cone bytes
  size_t   fovea_bytes;
  size_t   para_bytes;
  size_t   peri_bytes;
  // Full image
  size_t   full_precincts;
  size_t   full_bytes;
  double   full_decode_ms;
};

std::vector<uint8_t> build_jpp(const std::vector<uint8_t> &cs, const CodestreamIndex &idx,
                               const CodestreamLayout &layout, const PacketLocator &locator,
                               const std::unordered_set<uint64_t> &keep) {
  std::vector<uint8_t> stream;
  MessageHeaderContext ctx;
  emit_main_header_databin(cs.data(), cs.size(), layout, ctx, stream);
  for (uint32_t t = 0; t < idx.num_tiles(); ++t)
    emit_tile_header_databin(cs.data(), cs.size(), static_cast<uint16_t>(t), layout, ctx, stream);
  emit_metadata_bin_zero(ctx, stream);
  for (uint32_t t = 0; t < idx.num_tiles(); ++t) {
    for (uint16_t c = 0; c < idx.num_components(); ++c) {
      const auto &info = idx.tile_component(static_cast<uint16_t>(t), c);
      for (uint8_t r = 0; r <= info.NL; ++r) {
        const uint32_t n = info.npw[r] * info.nph[r];
        for (uint32_t p = 0; p < n; ++p) {
          if (keep.count(idx.I(static_cast<uint16_t>(t), c, r, p)))
            emit_precinct_databin(cs.data(), cs.size(), static_cast<uint16_t>(t), c, r, p,
                                  idx, locator, ctx, stream);
        }
      }
    }
  }
  return stream;
}

double decode_jpp(const std::vector<uint8_t> &jpp, const CodestreamIndex &idx,
                  uint8_t reduce_NL) {
  DataBinSet set;
  parse_jpp_stream(jpp.data(), jpp.size(), &set);
  std::vector<uint8_t> sparse_cs;
  auto rc = reassemble_codestream_client(set, idx, sparse_cs);
  if (rc != ReassembleStatus::Ok) return -1.0;

  open_htj2k::openhtj2k_decoder dec;
  dec.init(sparse_cs.data(), sparse_cs.size(), reduce_NL, 1);
  dec.parse();

  std::vector<uint32_t> w, h;
  std::vector<uint8_t> d;
  std::vector<bool> s;

  const auto t0 = Clock::now();
  try {
    dec.invoke_line_based_stream(
        [](uint32_t, int32_t *const *, uint16_t) {},
        w, h, d, s);
  } catch (...) { return -1.0; }
  const auto t1 = Clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr,
        "Usage: open_htj2k_jpip_benchmark <input.j2c>\n"
        "       [--gaze-grid N=5] [--reduce N=0] [--csv output.csv]\n");
    return EXIT_FAILURE;
  }

  std::string infile = argv[1];
  int grid_n = 5;
  uint8_t reduce_NL = 0;
  std::string csv_path;

  for (int i = 2; i < argc; ++i) {
    if (std::strcmp(argv[i], "--gaze-grid") == 0 && i + 1 < argc)
      grid_n = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--reduce") == 0 && i + 1 < argc)
      reduce_NL = static_cast<uint8_t>(std::atoi(argv[++i]));
    else if (std::strcmp(argv[i], "--csv") == 0 && i + 1 < argc)
      csv_path = argv[++i];
  }

  auto cs = read_file(infile.c_str());
  if (cs.empty()) return EXIT_FAILURE;

  auto idx = CodestreamIndex::build(cs.data(), cs.size());
  if (!idx) { std::fprintf(stderr, "CodestreamIndex build failed\n"); return EXIT_FAILURE; }

  CodestreamLayout layout;
  walk_codestream(cs.data(), cs.size(), &layout);

  auto locator = PacketLocator::build(cs.data(), cs.size(), *idx, layout);
  if (!locator) { std::fprintf(stderr, "PacketLocator build failed\n"); return EXIT_FAILURE; }

  const uint32_t cw = idx->geometry().canvas_size.x;
  const uint32_t ch = idx->geometry().canvas_size.y;
  const uint64_t total_p = idx->total_precincts();
  const uint32_t fovea_r = std::max(16u, cw / 15u);
  const uint32_t para_r  = std::max(32u, cw / 8u);
  const float    para_ratio = 0.5f;
  const float    peri_ratio = 0.125f;

  std::printf("Image: %s (%u x %u, %llu precincts, reduce=%u)\n",
              infile.c_str(), cw, ch, static_cast<unsigned long long>(total_p), reduce_NL);
  std::printf("Foveation: fovea_r=%u  para_r=%u  para_ratio=%.3f  peri_ratio=%.3f\n",
              fovea_r, para_r, static_cast<double>(para_ratio), static_cast<double>(peri_ratio));
  std::printf("Gaze grid: %dx%d = %d positions\n\n", grid_n, grid_n, grid_n * grid_n);

  // Build full-image I-set once
  ViewWindow full_vw;
  full_vw.fx = cw; full_vw.fy = ch;
  full_vw.ox = 0;  full_vw.oy = 0;
  full_vw.sx = cw; full_vw.sy = ch;
  auto full_keys = resolve_view_window(*idx, full_vw);
  std::unordered_set<uint64_t> full_set;
  for (const auto &k : full_keys) full_set.insert(idx->I(k.t, k.c, k.r, k.p_rc));
  auto full_jpp = build_jpp(cs, *idx, layout, *locator, full_set);
  double full_decode_ms = decode_jpp(full_jpp, *idx, reduce_NL);

  std::printf("Full image: %zu precincts, %zu bytes (%.1f KB), decode=%.1f ms\n\n",
              full_set.size(), full_jpp.size(), static_cast<double>(full_jpp.size()) / 1024.0, full_decode_ms);

  // Header
  std::printf("%-10s %-10s │ %8s %10s %8s │ %8s %8s %8s │ %6s %6s\n",
              "gaze_x", "gaze_y",
              "precincts", "bytes", "decode",
              "fovea_B", "para_B", "peri_B",
              "bw_%", "dec_%");
  std::printf("──────────────────────┼──────────────────────────────┼──────────────────────────┼──────────────\n");

  std::vector<BenchResult> results;

  for (int yi = 0; yi < grid_n; ++yi) {
    for (int xi = 0; xi < grid_n; ++xi) {
      uint32_t gx = (grid_n == 1) ? cw / 2 : static_cast<uint32_t>(static_cast<uint64_t>(xi) * (cw - 1) / static_cast<uint64_t>(grid_n - 1));
      uint32_t gy = (grid_n == 1) ? ch / 2 : static_cast<uint32_t>(static_cast<uint64_t>(yi) * (ch - 1) / static_cast<uint64_t>(grid_n - 1));

      // Foveated I-set (union of 3 cones)
      std::unordered_set<uint64_t> fov_set;
      auto add = [&](const std::vector<PrecinctKey> &keys) {
        for (const auto &k : keys) fov_set.insert(idx->I(k.t, k.c, k.r, k.p_rc));
      };
      add(resolve_view_window(*idx, make_view_window(*idx, gx, gy, fovea_r, 1.00f, false)));
      add(resolve_view_window(*idx, make_view_window(*idx, gx, gy, para_r, para_ratio, false)));
      add(resolve_view_window(*idx, make_view_window(*idx, gx, gy, 0, peri_ratio, true)));

      // Per-cone byte counts
      auto jpp_fovea = build_jpp(cs, *idx, layout, *locator, [&]{
        std::unordered_set<uint64_t> s;
        for (const auto &k : resolve_view_window(*idx, make_view_window(*idx, gx, gy, fovea_r, 1.00f, false)))
          s.insert(idx->I(k.t, k.c, k.r, k.p_rc));
        return s;
      }());
      auto jpp_para = build_jpp(cs, *idx, layout, *locator, [&]{
        std::unordered_set<uint64_t> s;
        for (const auto &k : resolve_view_window(*idx, make_view_window(*idx, gx, gy, para_r, para_ratio, false)))
          s.insert(idx->I(k.t, k.c, k.r, k.p_rc));
        return s;
      }());
      auto jpp_peri = build_jpp(cs, *idx, layout, *locator, [&]{
        std::unordered_set<uint64_t> s;
        for (const auto &k : resolve_view_window(*idx, make_view_window(*idx, gx, gy, 0, peri_ratio, true)))
          s.insert(idx->I(k.t, k.c, k.r, k.p_rc));
        return s;
      }());

      // Combined foveated JPP for decode timing
      auto fov_jpp = build_jpp(cs, *idx, layout, *locator, fov_set);
      double fov_decode_ms = decode_jpp(fov_jpp, *idx, reduce_NL);

      BenchResult r;
      r.gx = gx; r.gy = gy;
      r.fov_precincts = fov_set.size();
      r.fov_bytes     = fov_jpp.size();
      r.fov_decode_ms = fov_decode_ms;
      r.fovea_bytes   = jpp_fovea.size();
      r.para_bytes    = jpp_para.size();
      r.peri_bytes    = jpp_peri.size();
      r.full_precincts = full_set.size();
      r.full_bytes     = full_jpp.size();
      r.full_decode_ms = full_decode_ms;
      results.push_back(r);

      double bw_pct  = 100.0 * static_cast<double>(r.fov_bytes) / static_cast<double>(r.full_bytes);
      double dec_pct = 100.0 * r.fov_decode_ms / full_decode_ms;

      std::printf("%-10u %-10u │ %8zu %10zu %7.1fms │ %8zu %8zu %8zu │ %5.1f%% %5.1f%%\n",
                  gx, gy,
                  r.fov_precincts, r.fov_bytes, r.fov_decode_ms,
                  r.fovea_bytes, r.para_bytes, r.peri_bytes,
                  bw_pct, dec_pct);
    }
  }

  // Summary
  double avg_bw_pct = 0, avg_dec_pct = 0;
  size_t avg_precincts = 0;
  for (const auto &r : results) {
    avg_bw_pct  += 100.0 * static_cast<double>(r.fov_bytes) / static_cast<double>(r.full_bytes);
    avg_dec_pct += 100.0 * r.fov_decode_ms / full_decode_ms;
    avg_precincts += r.fov_precincts;
  }
  const size_t n = results.size();
  avg_bw_pct    /= static_cast<double>(n);
  avg_dec_pct   /= static_cast<double>(n);
  avg_precincts /= n;

  std::printf("──────────────────────┼──────────────────────────────┼──────────────────────────┼──────────────\n");
  std::printf("AVERAGE    %-10s │ %8zu %10s %8s │ %8s %8s %8s │ %5.1f%% %5.1f%%\n",
              "", avg_precincts, "", "", "", "", "", avg_bw_pct, avg_dec_pct);
  std::printf("\nBandwidth reduction: %.1f%%  |  Decode speedup: %.1fx\n",
              100.0 - avg_bw_pct, full_decode_ms / (avg_dec_pct * full_decode_ms / 100.0));

  // CSV output
  if (!csv_path.empty()) {
    FILE *f = std::fopen(csv_path.c_str(), "w");
    if (f) {
      std::fprintf(f, "gaze_x,gaze_y,fov_precincts,fov_bytes,fov_decode_ms,"
                      "fovea_bytes,para_bytes,peri_bytes,"
                      "full_precincts,full_bytes,full_decode_ms,"
                      "bw_pct,decode_pct\n");
      for (const auto &r : results) {
        std::fprintf(f, "%u,%u,%zu,%zu,%.2f,%zu,%zu,%zu,%zu,%zu,%.2f,%.1f,%.1f\n",
                    r.gx, r.gy, r.fov_precincts, r.fov_bytes, r.fov_decode_ms,
                    r.fovea_bytes, r.para_bytes, r.peri_bytes,
                    r.full_precincts, r.full_bytes, r.full_decode_ms,
                    100.0 * static_cast<double>(r.fov_bytes) / static_cast<double>(r.full_bytes),
                    100.0 * r.fov_decode_ms / full_decode_ms);
      }
      std::fclose(f);
      std::printf("\nCSV written to %s\n", csv_path.c_str());
    }
  }

  return EXIT_SUCCESS;
}
