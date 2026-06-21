// m2_leverb_bench: quantify M2 "Lever B" — precinct-indexed byte-range region
// decode — for the OpenSeadragon-integration evaluation.
//
// M2 Lever A (m2_precinct_bench) showed set_precinct_filter cuts the full-width
// strip block decode (L0 -47% native / -53% WASM), but leaves a "mode-B floor"
// (75-94% of the filtered time) = the 72 MB init copy + the whole-codestream
// packet-header walk (all 3180 precinct headers still parsed, because the
// filter drops bodies, not headers).  Lever B removes BOTH: an index of every
// precinct's byte range lets us fetch + decode ONLY the tile's precincts.
//
// This bench builds the index once (JPIP CodestreamIndex + CodestreamLayout +
// PacketLocator), then per tile:
//   1. resolve_view_window -> the precinct set overlapping the tile (+ §M.4.1
//      DWT margin, so the windowed IDWT is byte-exact).
//   2. Emit a JPP-stream of ONLY those precinct data-bins (+ main/tile headers),
//      parse it, and reassemble a SPARSE J2C codestream — precincts not in the
//      set become 1-byte empty-packet placeholders.  This is the wire-format
//      equivalent of a byte-range client that fetched only the tile's precincts;
//      the sparse codestream's size is the bytes such a client downloads.
//   3. Decode the sparse codestream (reuse path + col/row range), measure, and
//      byte-exact-verify the window vs a full decode.
//
// Reported per level/window (single thread, median ms):
//   A  = full codestream, col/row range            (M1 baseline)
//   C  = full + col/row + set_precinct_filter       (Lever A)
//   LB = sparse codestream + col/row range          (Lever B)
//   plus sparse_bytes (vs the 72 MB original) and the one-time index_ms.
//
// Usage: m2_leverb_bench <input.j2k|.jph> [-threads T] [-iter K] [-warmup W]
//                        [-win 256,512,1024] [-maxlevel L] [-noverify] [-csv]
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
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

static double now_ms() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

struct Args {
  std::string infile;
  uint32_t threads = 1;
  int iter         = 15;
  int warmup       = 3;
  std::vector<uint32_t> wins;
  int maxlevel = -1;
  bool verify  = true;
  bool csv     = false;
};

static std::vector<uint32_t> parse_csv_uints(const char *s) {
  std::vector<uint32_t> v;
  const char *p = s;
  while (*p) {
    char *end       = nullptr;
    unsigned long n = strtoul(p, &end, 10);
    if (end == p) break;
    if (n > 0) v.push_back(static_cast<uint32_t>(n));
    p = (*end == ',') ? end + 1 : end;
  }
  return v;
}

static bool parse_args(int argc, char *argv[], Args &a) {
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "-threads") && i + 1 < argc) {
      a.threads = static_cast<uint32_t>(strtoul(argv[++i], nullptr, 10));
    } else if (!strcmp(argv[i], "-iter") && i + 1 < argc) {
      a.iter = atoi(argv[++i]);
    } else if (!strcmp(argv[i], "-warmup") && i + 1 < argc) {
      a.warmup = atoi(argv[++i]);
    } else if (!strcmp(argv[i], "-win") && i + 1 < argc) {
      a.wins = parse_csv_uints(argv[++i]);
    } else if (!strcmp(argv[i], "-maxlevel") && i + 1 < argc) {
      a.maxlevel = atoi(argv[++i]);
    } else if (!strcmp(argv[i], "-noverify")) {
      a.verify = false;
    } else if (!strcmp(argv[i], "-csv")) {
      a.csv = true;
    } else {
      a.infile = argv[i];
    }
  }
  if (a.wins.empty()) a.wins = {256, 512, 1024};
  std::sort(a.wins.begin(), a.wins.end());
  return !a.infile.empty();
}

static std::vector<uint8_t> read_file(const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f) {
    printf("ERROR: cannot open %s\n", path);
    return {};
  }
  fseek(f, 0, SEEK_END);
  size_t sz = static_cast<size_t>(ftell(f));
  fseek(f, 0, SEEK_SET);
  std::vector<uint8_t> buf(sz);
  size_t rd = fread(buf.data(), 1, sz, f);
  fclose(f);
  if (rd != sz) {
    printf("ERROR: partial read of %s\n", path);
    return {};
  }
  return buf;
}

struct DecodeBufs {
  std::vector<std::vector<int32_t>> out;
  std::vector<uint32_t> widths, heights;
  std::vector<uint8_t> depths;
  std::vector<bool> is_signed;
};

struct Timing {
  double setup;
  double decode;
  double total() const { return setup + decode; }
};

using PFilter = std::function<bool(uint16_t, uint16_t, uint8_t, uint32_t)>;

// One reuse-path region decode of `data` under an optional precinct filter.
static Timing decode_region(open_htj2k::openhtj2k_decoder &dec, const uint8_t *data, size_t len,
                            uint8_t reduce_NL, uint32_t threads, uint32_t x0, uint32_t y0, uint32_t w,
                            uint32_t h, bool store, const PFilter &filter, DecodeBufs &b) {
  const double t0 = now_ms();
  dec.init(data, len, reduce_NL, threads);
  dec.parse();
  dec.set_col_range(x0, x0 + w);
  dec.set_row_range(y0, y0 + h);
  dec.set_precinct_filter(filter);  // empty => keep-all
  const double t1 = now_ms();
  auto cb         = [&](uint32_t y, int32_t *const *rows, uint16_t nc) {
    if (!store) return;
    if (y < y0 || y >= y0 + h) return;
    if (b.out.size() < nc) b.out.resize(nc);
    const uint32_t ly = y - y0;
    for (uint16_t c = 0; c < nc; ++c) {
      if (b.out[c].size() < static_cast<size_t>(w) * h) b.out[c].assign(static_cast<size_t>(w) * h, 0);
      const uint32_t cw = (c < b.widths.size()) ? b.widths[c] : 0;
      const uint32_t xs = std::min(x0, cw);
      const uint32_t xe = std::min(x0 + w, cw);
      if (xe > xs)
        std::memcpy(b.out[c].data() + static_cast<size_t>(ly) * w, rows[c] + xs,
                            (xe - xs) * sizeof(int32_t));
    }
  };
  dec.invoke_line_based_stream_reuse(cb, b.widths, b.heights, b.depths, b.is_signed);
  const double t2 = now_ms();
  return {t1 - t0, t2 - t1};
}

// Per-(component, resolution) kept-precinct rectangle, from the resolve_view_window
// key list (one contiguous rectangle per (c, r)).  Used for the Lever-A filter.
struct CRRect {
  uint32_t npw = 0;
  int px_lo = INT32_MAX, px_hi = 0, py_lo = INT32_MAX, py_hi = 0;
  bool empty() const { return px_hi <= px_lo || py_hi <= py_lo; }
};

static std::vector<PrecinctKey> window_keys(const CodestreamIndex &idx, uint32_t levelW, uint32_t levelH,
                                            uint32_t x0, uint32_t y0, uint32_t w, uint32_t h) {
  ViewWindow vw;
  vw.fx    = levelW;  // force pick_discard_level -> reduce_NL (subsampling 1:1)
  vw.fy    = levelH;
  vw.ox    = x0;
  vw.oy    = y0;
  vw.sx    = w;
  vw.sy    = h;
  vw.round = ViewWindow::Round::Down;
  return resolve_view_window(idx, vw);
}

static PFilter filter_from_keys(const CodestreamIndex &idx, const std::vector<PrecinctKey> &keys) {
  const uint16_t nc   = idx.num_components();
  const uint8_t maxNL = idx.max_NL();
  auto rects          = std::make_shared<std::vector<std::vector<CRRect>>>(
      nc, std::vector<CRRect>(static_cast<size_t>(maxNL) + 1));
  for (const auto &k : keys) {
    if (k.t != 0 || k.c >= nc || k.r > maxNL) continue;
    CRRect &R    = (*rects)[k.c][k.r];
    R.npw        = idx.tile_component(0, k.c).npw[k.r];
    const int px = R.npw ? static_cast<int>(k.p_rc % R.npw) : 0;
    const int py = R.npw ? static_cast<int>(k.p_rc / R.npw) : 0;
    R.px_lo      = std::min(R.px_lo, px);
    R.px_hi      = std::max(R.px_hi, px + 1);
    R.py_lo      = std::min(R.py_lo, py);
    R.py_hi      = std::max(R.py_hi, py + 1);
  }
  return [rects, nc, maxNL](uint16_t, uint16_t c, uint8_t r, uint32_t p) -> bool {
    if (c >= nc || r > maxNL) return false;
    const CRRect &R = (*rects)[c][r];
    if (R.empty() || R.npw == 0) return false;
    const int px = static_cast<int>(p % R.npw);
    const int py = static_cast<int>(p / R.npw);
    return px >= R.px_lo && px < R.px_hi && py >= R.py_lo && py < R.py_hi;
  };
}

// Assemble a SPARSE codestream containing only `keys`' precincts (others become
// empty-packet placeholders), by emitting just those data-bins and running the
// JPIP reassembler.  Returns false if the codestream is outside reassembler v1
// scope (LRCP/RLCP/SOP/EPH/multi-tile-part).
static bool build_sparse(const std::vector<uint8_t> &orig, const CodestreamIndex &idx,
                         const CodestreamLayout &layout, const PacketLocator &loc,
                         const std::vector<PrecinctKey> &keys, std::vector<uint8_t> &sparse) {
  std::vector<uint8_t> stream;
  MessageHeaderContext ctx;
  emit_main_header_databin(orig.data(), orig.size(), layout, ctx, stream);
  for (uint32_t t = 0; t < idx.num_tiles(); ++t)
    emit_tile_header_databin(orig.data(), orig.size(), static_cast<uint16_t>(t), layout, ctx, stream);
  emit_metadata_bin_zero(ctx, stream);
  for (const auto &k : keys)
    emit_precinct_databin(orig.data(), orig.size(), k.t, k.c, k.r, k.p_rc, idx, loc, ctx, stream);

  DataBinSet set;
  if (!parse_jpp_stream(stream.data(), stream.size(), &set)) return false;
  sparse.clear();
  const auto st = reassemble_codestream(orig.data(), orig.size(), set, idx, layout, loc, sparse);
  return st == ReassembleStatus::Ok;
}

static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

static void measure(open_htj2k::openhtj2k_decoder &dec, const uint8_t *data, size_t len, uint8_t li,
                    uint32_t threads, uint32_t x0, uint32_t y0, uint32_t w, uint32_t h,
                    const PFilter &filter, int warmup, int iter, double &med_tot, double &med_dec) {
  DecodeBufs b;
  for (int i = 0; i < warmup; ++i)
    decode_region(dec, data, len, li, threads, x0, y0, w, h, false, filter, b);
  std::vector<double> tot, dc;
  for (int i = 0; i < iter; ++i) {
    Timing t = decode_region(dec, data, len, li, threads, x0, y0, w, h, false, filter, b);
    tot.push_back(t.total());
    dc.push_back(t.decode);
  }
  med_tot = median(tot);
  med_dec = median(dc);
}

// Byte-exact check: a windowed decode of `cand` must equal the full keep-all
// decode of `orig` within the window.  `cand` is either the original (with a
// filter) or the sparse codestream (filter empty).
static bool verify_window(const uint8_t *orig, size_t olen, const uint8_t *cand, size_t clen, uint8_t li,
                          uint32_t threads, uint32_t x0, uint32_t y0, uint32_t w, uint32_t h,
                          const PFilter &filter, const char *label) {
  open_htj2k::openhtj2k_decoder dref, dc;
  dref.enable_single_tile_reuse(true);
  dc.enable_single_tile_reuse(true);
  DecodeBufs br, bc;
  PFilter none;
  decode_region(dref, orig, olen, li, threads, x0, y0, w, h, true, none, br);  // warm
  br.out.clear();
  decode_region(dref, orig, olen, li, threads, x0, y0, w, h, true, none, br);  // reference
  decode_region(dc, cand, clen, li, threads, x0, y0, w, h, true, filter, bc);  // warm
  bc.out.clear();
  decode_region(dc, cand, clen, li, threads, x0, y0, w, h, true, filter, bc);  // candidate
  for (size_t c = 0; c < br.out.size() && c < bc.out.size(); ++c) {
    const uint32_t cw = (c < br.widths.size()) ? br.widths[c] : 0;
    const uint32_t ch = (c < br.heights.size()) ? br.heights[c] : 0;
    for (uint32_t ly = 0; ly < h; ++ly)
      for (uint32_t lx = 0; lx < w; ++lx) {
        const uint32_t gx = x0 + lx, gy = y0 + ly;
        if (gx >= cw || gy >= ch) continue;
        const int32_t r = br.out[c][static_cast<size_t>(ly) * w + lx];
        const int32_t g = bc.out[c][static_cast<size_t>(ly) * w + lx];
        if (r != g) {
          printf("# VERIFY FAIL (%s) level %u comp %zu pixel (%u,%u): ref=%d got=%d\n", label, li, c, gx,
                 gy, r, g);
          return false;
        }
      }
  }
  return true;
}

int main(int argc, char *argv[]) {
  Args a;
  if (!parse_args(argc, argv, a)) {
    printf(
        "Usage: m2_leverb_bench <input.j2k|.jph> [-threads T] [-iter K] [-warmup W]\n"
        "                       [-win 256,512,1024] [-maxlevel L] [-noverify] [-csv]\n");
    return 1;
  }
  std::vector<uint8_t> file = read_file(a.infile.c_str());
  if (file.empty()) return 1;
  const uint8_t *data = file.data();
  const size_t len    = file.size();

  // ── One-time precinct index build (CodestreamIndex + Layout + PacketLocator) ──
  const double idx_t0 = now_ms();
  auto idx            = CodestreamIndex::build(data, len);
  if (!idx) {
    printf("ERROR: CodestreamIndex::build failed\n");
    return 1;
  }
  CodestreamLayout layout;
  if (!walk_codestream(data, len, &layout)) {
    printf("ERROR: walk_codestream failed\n");
    return 1;
  }
  auto loc = PacketLocator::build(data, len, *idx, layout);
  if (!loc) {
    printf("ERROR: PacketLocator::build failed (codestream outside v1 scope?)\n");
    return 1;
  }
  const double index_ms = now_ms() - idx_t0;

  uint8_t max_reduce = 0;
  uint32_t full_w = 0, full_h = 0;
  uint16_t nc = 0;
  try {
    open_htj2k::openhtj2k_decoder probe;
    probe.init(data, len, 0, a.threads);
    probe.parse();
    max_reduce = probe.get_max_safe_reduce_NL();
    full_w     = probe.get_component_width(0);
    full_h     = probe.get_component_height(0);
    nc         = probe.get_num_component();
  } catch (std::exception &e) {
    printf("ERROR: parse failed: %s\n", e.what());
    return 1;
  }
  const int maxlevel = (a.maxlevel >= 0) ? std::min<int>(a.maxlevel, max_reduce) : max_reduce;

  if (a.csv) {
    printf(
        "level,level_w,level_h,win_w,win_h,threads,kept,total,sparse_bytes,orig_bytes,"
        "A_tot,C_tot,LB_tot,A_dec,C_dec,LB_dec,asm_ms,verifyC,verifyLB\n");
  } else {
    printf("# m2_leverb_bench  file=%s  (%zu bytes)\n", a.infile.c_str(), len);
    printf(
        "# full=%ux%u comps=%u max_safe_reduce=%u total_precincts=%llu prog=%u  index_build=%.1f ms "
        "(one-time, amortized)\n",
        full_w, full_h, nc, max_reduce, (unsigned long long)idx->total_precincts(),
        idx->progression_order(), index_ms);
    printf(
        "# A=full+colrow(M1)  C=full+colrow+filter(LeverA)  LB=sparse+colrow(LeverB); ms=median total\n");
    printf(
        "# asm_ms = per-tile sparse build (JPIP wire round-trip; UPPER bound vs a real byte-range "
        "splice)\n");
    printf("%-5s  %-13s  %-9s  %-10s  %-10s  %-7s  %-7s  %-8s  %-8s\n", "level", "level_dims", "window",
           "kept/tot", "sparse_KB", "A(M1)", "C(LvA)", "LB(LvB)", "asm_ms");
  }

  int fails = 0;
  for (int li = 0; li <= maxlevel; ++li) {
    open_htj2k::openhtj2k_decoder decA, decC, decLB;
    decA.enable_single_tile_reuse(true);
    decC.enable_single_tile_reuse(true);
    decLB.enable_single_tile_reuse(true);
    DecodeBufs probe;
    uint32_t W = 0, H = 0;
    try {
      PFilter none;
      decode_region(decA, data, len, static_cast<uint8_t>(li), a.threads, 0, 0, 1, 1, false, none, probe);
      if (!probe.widths.empty()) {
        W = probe.widths[0];
        H = probe.heights[0];
      }
    } catch (std::exception &e) {
      printf("level %d: probe failed: %s\n", li, e.what());
      continue;
    }
    if (W == 0 || H == 0) continue;

    uint64_t seen = 0;
    for (uint32_t S : a.wins) {
      const uint32_t w = std::min(S, W), h = std::min(S, H);
      const uint64_t key = (static_cast<uint64_t>(w) << 32) | h;
      if (key == seen) continue;
      seen              = key;
      const uint32_t x0 = (W > w) ? (W - w) / 2 : 0;
      const uint32_t y0 = (H > h) ? (H - h) / 2 : 0;

      std::vector<PrecinctKey> keys = window_keys(*idx, W, H, x0, y0, w, h);
      PFilter fwin                  = filter_from_keys(*idx, keys);

      std::vector<uint8_t> sparse;
      if (!build_sparse(file, *idx, layout, *loc, keys, sparse)) {
        printf("level %d win %ux%u: build_sparse failed (outside reassembler v1 scope)\n", li, w, h);
        continue;
      }

      bool okC = true, okLB = true;
      if (a.verify) {
        okC = verify_window(data, len, data, len, static_cast<uint8_t>(li), a.threads, x0, y0, w, h, fwin,
                            "Lever A");
        PFilter none;
        okLB = verify_window(data, len, sparse.data(), sparse.size(), static_cast<uint8_t>(li), a.threads,
                             x0, y0, w, h, none, "Lever B");
        if (!okC) ++fails;
        if (!okLB) ++fails;
      }

      // Per-tile sparse-assembly cost (emit selected bins + parse + reassemble).
      // This is the JPIP wire-format round-trip — an UPPER BOUND on a real
      // byte-range client, which just splices fetched ranges into a skeleton.
      std::vector<double> asm_ms;
      for (int i = 0; i < std::max(3, a.iter / 3); ++i) {
        std::vector<uint8_t> tmp;
        const double s0 = now_ms();
        build_sparse(file, *idx, layout, *loc, keys, tmp);
        asm_ms.push_back(now_ms() - s0);
      }
      const double Asm = median(asm_ms);

      const uint8_t lvl = static_cast<uint8_t>(li);
      double At, Ad, Ct, Cd, Lt, Ld;
      PFilter none;
      measure(decA, data, len, lvl, a.threads, x0, y0, w, h, none, a.warmup, a.iter, At, Ad);
      measure(decC, data, len, lvl, a.threads, x0, y0, w, h, fwin, a.warmup, a.iter, Ct, Cd);
      measure(decLB, sparse.data(), sparse.size(), lvl, a.threads, x0, y0, w, h, none, a.warmup, a.iter, Lt,
              Ld);

      const double sparse_kb = sparse.size() / 1024.0;
      char winlbl[24], kt[24];
      snprintf(winlbl, sizeof(winlbl), "%ux%u", w, h);
      snprintf(kt, sizeof(kt), "%zu/%llu", keys.size(), (unsigned long long)idx->total_precincts());
      if (a.csv) {
        printf("%d,%u,%u,%u,%u,%u,%zu,%llu,%zu,%zu,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%s,%s\n", li, W, H, w,
               h, a.threads, keys.size(), (unsigned long long)idx->total_precincts(), sparse.size(), len,
               At, Ct, Lt, Ad, Cd, Ld, Asm, a.verify ? (okC ? "PASS" : "FAIL") : "-",
               a.verify ? (okLB ? "PASS" : "FAIL") : "-");
      } else {
        char skb[16];
        snprintf(skb, sizeof(skb), "%.1f", sparse_kb);
        printf("%-5d  %5ux%-7u  %-9s  %-10s  %-10s  %-7.2f  %-7.2f  %-8.2f  %-8.2f\n", li, W, H, winlbl, kt,
               skb, At, Ct, Lt, Asm);
      }
      fflush(stdout);
    }
  }
  if (a.verify) printf("# verify: %s (%d failures)\n", fails ? "FAIL" : "PASS", fails);
  return fails ? 2 : 0;
}
