// m2_precinct_bench: quantify how much of the M1 per-tile decode tax the JPIP
// precinct filter removes — the OpenSeadragon-integration M2 spike.
//
// M1 showed each tile pays a ~15 ms fixed tax = ~4 ms (72 MB init copy) +
// ~11 ms (whole-codestream packet-header replay), and that set_col_range prunes
// only the IDWT — a "256-wide" tile still ENTROPY-DECODES a full-WIDTH strip.
// set_precinct_filter (ISO/IEC 15444-9 §M.4.1) drops the *body* bytes of
// precincts outside the tile while the packet-header bitstream still advances,
// so it attacks exactly that full-width-strip block-decode waste — but NOT the
// header walk or the init copy (those need byte-range / a precinct index = M2's
// Lever B). This bench measures the split.
//
// For each pyramid level (reduce_NL) and centred square window, it decodes via
// the WASM-viewer reuse path (enable_single_tile_reuse + set_col_range +
// set_row_range + invoke_line_based_stream_reuse) under three precinct-filter
// modes:
//   A  keep-all   filter unset            = the M1 baseline.
//   B  drop-all   filter returns false    = headers walked, every body dropped
//                                           = init-copy + header-walk FLOOR
//                                           (the residual only Lever B can cut).
//   C  keep-window resolve_view_window set = realistic M2; byte-exact-verified
//                                           vs A within the window.
// Derived: A.dec - B.dec = block-decode the filter CAN remove (ceiling);
//          A.total - C.total = the actual M2 per-tile win; B.total = Lever-B residual.
//
// The window->precinct set reuses the JPIP geometry (CodestreamIndex +
// resolve_view_window), which already adds the §M.4.1 DWT-synthesis margin, so
// masked precincts never feed the window's IDWT (verified byte-exact).
//
// Usage: m2_precinct_bench <input.j2k|.jph> [-threads T] [-iter K] [-warmup W]
//                          [-win 256,512,1024] [-maxlevel L] [-noverify] [-csv]
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "decoder.hpp"
#include "precinct_index.hpp"
#include "view_window.hpp"
#ifdef __EMSCRIPTEN__
  #include <emscripten/emscripten.h>
#endif

using open_htj2k::jpip::CodestreamIndex;
using open_htj2k::jpip::PrecinctKey;
using open_htj2k::jpip::ViewWindow;

// Monotonic wall clock in ms.  Under Emscripten use emscripten_get_now()
// (= performance.now(), high-resolution) — std::chrono::steady_clock can be
// coarse / zero-resolution in a non-threaded Wasm build.  Keeps the timed code
// path identical native/Wasm.
static double now_ms() {
#ifdef __EMSCRIPTEN__
  return emscripten_get_now();
#else
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch())
      .count();
#endif
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

// One reuse-path region decode under a given precinct filter.
static Timing decode_region(open_htj2k::openhtj2k_decoder &dec, const uint8_t *data, size_t len,
                            uint8_t reduce_NL, uint32_t threads, uint32_t x0, uint32_t y0, uint32_t w,
                            uint32_t h, bool store, const PFilter &filter, DecodeBufs &b) {
  const double t0 = now_ms();
  dec.init(data, len, reduce_NL, threads);
  dec.parse();
  dec.set_col_range(x0, x0 + w);
  dec.set_row_range(y0, y0 + h);
  dec.set_precinct_filter(filter);  // empty function => keep-all (mode A)
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
  Timing tm;
  tm.setup  = t1 - t0;
  tm.decode = t2 - t1;
  return tm;
}

// Per-(component, resolution) kept-precinct rectangle, recovered from the flat
// resolve_view_window key list (which is one contiguous rectangle per (c, r)).
// Filtering by rectangle avoids a per-packet hash lookup polluting the timing.
struct CRRect {
  uint32_t npw = 0;
  int px_lo = INT32_MAX, px_hi = 0, py_lo = INT32_MAX, py_hi = 0;
  bool empty() const { return px_hi <= px_lo || py_hi <= py_lo; }
};

// Build the keep-window precinct filter for a centred window at the pyramid
// level whose dims are levelW x levelH.  Returns the filter; also reports
// kept/total precinct counts for the report.  The discard level is forced to
// match the decode's reduce_NL by passing the level dims as the frame size.
static PFilter build_window_filter(const CodestreamIndex &idx, uint32_t levelW, uint32_t levelH,
                                   uint32_t x0, uint32_t y0, uint32_t w, uint32_t h, uint64_t &kept_out,
                                   uint64_t &total_out) {
  ViewWindow vw;
  vw.fx                         = levelW;  // force pick_discard_level -> reduce_NL (subsampling 1:1)
  vw.fy                         = levelH;
  vw.ox                         = x0;
  vw.oy                         = y0;
  vw.sx                         = w;
  vw.sy                         = h;
  vw.round                      = ViewWindow::Round::Down;
  std::vector<PrecinctKey> keys = resolve_view_window(idx, vw);

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
  kept_out  = keys.size();
  total_out = idx.total_precincts();

  return [rects, nc, maxNL](uint16_t /*t*/, uint16_t c, uint8_t r, uint32_t p) -> bool {
    if (c >= nc || r > maxNL) return false;
    const CRRect &R = (*rects)[c][r];
    if (R.empty() || R.npw == 0) return false;
    const int px = static_cast<int>(p % R.npw);
    const int py = static_cast<int>(p / R.npw);
    return px >= R.px_lo && px < R.px_hi && py >= R.py_lo && py < R.py_hi;
  };
}

static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

// Decode `iter` times under a filter, return {median total, median decode, median setup}.
static void measure(open_htj2k::openhtj2k_decoder &dec, const uint8_t *data, size_t len, uint8_t li,
                    uint32_t threads, uint32_t x0, uint32_t y0, uint32_t w, uint32_t h,
                    const PFilter &filter, int warmup, int iter, double &med_tot, double &med_dec,
                    double &med_set) {
  DecodeBufs b;
  for (int i = 0; i < warmup; ++i)
    decode_region(dec, data, len, li, threads, x0, y0, w, h, false, filter, b);
  std::vector<double> tot, dc, st;
  for (int i = 0; i < iter; ++i) {
    Timing t = decode_region(dec, data, len, li, threads, x0, y0, w, h, false, filter, b);
    tot.push_back(t.total());
    dc.push_back(t.decode);
    st.push_back(t.setup);
  }
  med_tot = median(tot);
  med_dec = median(dc);
  med_set = median(st);
}

// Byte-exact check: keep-window decode must equal keep-all within the window.
static bool verify_window(const uint8_t *data, size_t len, uint8_t li, uint32_t threads, uint32_t x0,
                          uint32_t y0, uint32_t w, uint32_t h, const PFilter &filter) {
  open_htj2k::openhtj2k_decoder da, dc;
  da.enable_single_tile_reuse(true);
  dc.enable_single_tile_reuse(true);
  DecodeBufs ba, bc;
  PFilter none;
  decode_region(da, data, len, li, threads, x0, y0, w, h, true, none, ba);  // warm
  ba.out.clear();
  decode_region(da, data, len, li, threads, x0, y0, w, h, true, none, ba);    // ref (keep-all)
  decode_region(dc, data, len, li, threads, x0, y0, w, h, true, filter, bc);  // warm
  bc.out.clear();
  decode_region(dc, data, len, li, threads, x0, y0, w, h, true, filter, bc);  // keep-window
  for (size_t c = 0; c < ba.out.size() && c < bc.out.size(); ++c) {
    const uint32_t cw = (c < ba.widths.size()) ? ba.widths[c] : 0;
    const uint32_t ch = (c < ba.heights.size()) ? ba.heights[c] : 0;
    for (uint32_t ly = 0; ly < h; ++ly)
      for (uint32_t lx = 0; lx < w; ++lx) {
        const uint32_t gx = x0 + lx, gy = y0 + ly;
        if (gx >= cw || gy >= ch) continue;
        const int32_t r = ba.out[c][static_cast<size_t>(ly) * w + lx];
        const int32_t g = bc.out[c][static_cast<size_t>(ly) * w + lx];
        if (r != g) {
          printf("# VERIFY FAIL level %u comp %zu pixel (%u,%u): keepall=%d keepwin=%d\n", li, c, gx, gy, r,
                 g);
          return false;
        }
      }
  }
  return true;
}

#ifdef __EMSCRIPTEN__
// JS-callable entry point for the Wasm runtime (main() is not auto-run under
// -sINVOKE_RUN=0; the runner drives this via ccall, like region_decode_bench).
//
// Times the three precinct-filter modes (A keep-all, B drop-all, C keep-window)
// for one reduce level / window edge, after `warmup` untimed decodes, and
// optionally byte-exact-verifies C vs A within the window.  Writes 11 doubles:
//   [0]=A.total [1]=A.dec  [2]=B.total [3]=B.dec  [4]=C.total [5]=C.dec
//   [6]=kept_precincts [7]=total_precincts [8]=level W [9]=level H
//   [10]=verify (1=byte-exact pass, 0=fail; 1 when do_verify==0)
// Returns 0 on success, 1 on probe/decode error, 2 on exception.
extern "C" EMSCRIPTEN_KEEPALIVE int m2bench_modes(const uint8_t *data, int len, int reduce, int win,
                                                  int threads, int iter, int warmup, int do_verify,
                                                  double *out) {
  for (int i = 0; i < 11; ++i) out[i] = 0.0;
  try {
    const size_t L   = static_cast<size_t>(len);
    const uint8_t r  = static_cast<uint8_t>(reduce);
    const uint32_t t = static_cast<uint32_t>(threads);
    auto idx         = CodestreamIndex::build(data, L);

    open_htj2k::openhtj2k_decoder dec;
    dec.enable_single_tile_reuse(true);
    DecodeBufs probe;
    PFilter none;
    decode_region(dec, data, L, r, t, 0, 0, 1, 1, false, none, probe);  // probe dims + warm reuse cache
    if (probe.widths.empty() || probe.widths[0] == 0 || probe.heights[0] == 0) return 1;
    const uint32_t W = probe.widths[0], H = probe.heights[0];
    const uint32_t w = std::min<uint32_t>(win, W), h = std::min<uint32_t>(win, H);
    const uint32_t x0 = (W > w) ? (W - w) / 2 : 0, y0 = (H > h) ? (H - h) / 2 : 0;

    uint64_t kept = 0, total = 0;
    PFilter fwin  = build_window_filter(*idx, W, H, x0, y0, w, h, kept, total);
    PFilter fdrop = [](uint16_t, uint16_t, uint8_t, uint32_t) { return false; };

    int verify = 1;
    if (do_verify) verify = verify_window(data, L, r, t, x0, y0, w, h, fwin) ? 1 : 0;

    double At, Ad, As, Bt, Bd, Bs, Ct, Cd, Cs;
    measure(dec, data, L, r, t, x0, y0, w, h, none, warmup, iter, At, Ad, As);
    measure(dec, data, L, r, t, x0, y0, w, h, fdrop, warmup, iter, Bt, Bd, Bs);
    measure(dec, data, L, r, t, x0, y0, w, h, fwin, warmup, iter, Ct, Cd, Cs);

    out[0]  = At;
    out[1]  = Ad;
    out[2]  = Bt;
    out[3]  = Bd;
    out[4]  = Ct;
    out[5]  = Cd;
    out[6]  = static_cast<double>(kept);
    out[7]  = static_cast<double>(total);
    out[8]  = static_cast<double>(W);
    out[9]  = static_cast<double>(H);
    out[10] = static_cast<double>(verify);
    return 0;
  } catch (std::exception &e) {
    printf("m2bench_modes error: %s\n", e.what());
    return 2;
  }
}
#endif  // __EMSCRIPTEN__

int main(int argc, char *argv[]) {
  Args a;
  if (!parse_args(argc, argv, a)) {
    printf(
        "Usage: m2_precinct_bench <input.j2k|.jph> [-threads T] [-iter K] [-warmup W]\n"
        "                         [-win 256,512,1024] [-maxlevel L] [-noverify] [-csv]\n");
    return 1;
  }
  std::vector<uint8_t> file = read_file(a.infile.c_str());
  if (file.empty()) return 1;
  const uint8_t *data = file.data();
  const size_t len    = file.size();

  std::unique_ptr<CodestreamIndex> idx;
  try {
    idx = CodestreamIndex::build(data, len);
  } catch (std::exception &e) {
    printf("ERROR: CodestreamIndex::build failed: %s\n", e.what());
    return 1;
  }

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
        "level,level_w,level_h,win_w,win_h,threads,kept_prec,total_prec,"
        "A_tot,A_dec,B_tot,B_dec,C_tot,C_dec,win_ms,floorB_ms,verify\n");
  } else {
    printf("# m2_precinct_bench  file=%s  (%zu bytes)\n", a.infile.c_str(), len);
    printf("# full=%ux%u comps=%u max_safe_reduce=%u  total_precincts=%llu  prog=%u layers=%u\n", full_w,
           full_h, nc, max_reduce, (unsigned long long)idx->total_precincts(), idx->progression_order(),
           idx->num_layers());
    printf(
        "# A=keep-all(M1)  B=drop-all(floor)  C=keep-window(M2); reuse + col/row range; ms = median "
        "total\n");
    printf("%-5s  %-13s  %-9s  %-11s  %-7s  %-7s  %-7s  %-7s  %-8s  %-8s  %-7s\n", "level", "level_dims",
           "window", "kept/total", "A(M1)", "B(flr)", "C(M2)", "A.dec", "B.dec", "C.dec", "verify");
  }

  int fails = 0;
  for (int li = 0; li <= maxlevel; ++li) {
    open_htj2k::openhtj2k_decoder dec;
    dec.enable_single_tile_reuse(true);
    DecodeBufs probe;
    uint32_t W = 0, H = 0;
    try {
      PFilter none;
      decode_region(dec, data, len, static_cast<uint8_t>(li), a.threads, 0, 0, 1, 1, false, none, probe);
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

      uint64_t kept = 0, total = 0;
      PFilter fwin = build_window_filter(*idx, W, H, x0, y0, w, h, kept, total);
      PFilter fnone;
      PFilter fdrop = [](uint16_t, uint16_t, uint8_t, uint32_t) { return false; };

      bool ok = true;
      if (a.verify) {
        ok = verify_window(data, len, static_cast<uint8_t>(li), a.threads, x0, y0, w, h, fwin);
        if (!ok) ++fails;
      }

      double At, Ad, As, Bt, Bd, Bs, Ct, Cd, Cs;
      const uint8_t lvl = static_cast<uint8_t>(li);
      measure(dec, data, len, lvl, a.threads, x0, y0, w, h, fnone, a.warmup, a.iter, At, Ad, As);
      measure(dec, data, len, lvl, a.threads, x0, y0, w, h, fdrop, a.warmup, a.iter, Bt, Bd, Bs);
      measure(dec, data, len, lvl, a.threads, x0, y0, w, h, fwin, a.warmup, a.iter, Ct, Cd, Cs);

      char winlbl[24], kt[24];
      snprintf(winlbl, sizeof(winlbl), "%ux%u", w, h);
      snprintf(kt, sizeof(kt), "%llu/%llu", (unsigned long long)kept, (unsigned long long)total);
      if (a.csv) {
        printf("%d,%u,%u,%u,%u,%u,%llu,%llu,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%s\n", li, W, H, w, h,
               a.threads, (unsigned long long)kept, (unsigned long long)total, At, Ad, Bt, Bd, Ct, Cd,
               At - Ct, Bt, a.verify ? (ok ? "PASS" : "FAIL") : "-");
      } else {
        printf("%-5d  %5ux%-7u  %-9s  %-11s  %-7.2f  %-7.2f  %-7.2f  %-7.2f  %-8.2f  %-8.2f  %-7s\n", li, W,
               H, winlbl, kt, At, Bt, Ct, Ad, Bd, Cd, a.verify ? (ok ? "PASS" : "FAIL") : "-");
      }
      fflush(stdout);
    }
  }
  if (a.verify) printf("# verify: %s (%d failures)\n", fails ? "FAIL" : "PASS", fails);
  return fails ? 2 : 0;
}
