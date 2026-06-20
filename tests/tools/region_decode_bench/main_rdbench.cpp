// region_decode_bench: measure HTJ2K region-decode latency / throughput at
// OpenSeadragon tile cadence, via the single-tile reuse path.
//
// For each pyramid level (reduce_NL) and each square window size, it decodes a
// centred window using exactly the path the WASM viewer (and a future
// OpenSeadragon HTJ2KTileSource) uses:
//
//     dec.enable_single_tile_reuse(true);
//     dec.set_col_range(x0, x0 + w);
//     dec.set_row_range(y0, y0 + h);
//     dec.invoke_line_based_stream_reuse(cb, ...);
//
// and reports ms/region (median / min) and regions/s.  This is the M1
// deliverable for the OpenSeadragon-integration evaluation: the make-or-break
// number is region-decode latency, used to build an Iris-style comparison
// table (region decode vs DZI tile fetch).
//
// Why init() and not init_borrow(): the scalar HT cleanup decode mutates the
// codeblock bytes in place (modDcup writes Dcup[Lcup-1]=0xFF, Dcup[Lcup-2]|=0x0F
// at ht_block_decoding.cpp:1196).  With a *borrowed* buffer those writes corrupt
// the source, so the 2nd+ reuse decode recomputes Scup=4095 and aborts every
// block ("WARNING: cleanup pass suffix length 4095 is invalid").  init() copies
// the codestream per call, giving each decode pristine bytes — the same path
// col_range_compare validates byte-exact.  The copy is therefore a real,
// measured component of the per-tile cost; the timing is split into:
//   setup  = init (codestream copy) + parse (header walk) + set ranges
//   decode = invoke_line_based_stream_reuse (packet replay + block + IDWT)
// so we can see whether the headline is dominated by the whole-codestream copy
// /parse (→ M2 byte-range precinct selection is the lever) or by region decode.
//
// Threading is a process-global singleton (ThreadPool::instance is
// first-call-wins), so the thread count is fixed per process via -threads;
// sweep it with run_bench.sh.  One decoder is created per level (reduce_NL is
// bound at init) and reused across window sizes / iterations so the single-tile
// reuse cache stays warm.
//
// Usage:
//   region_decode_bench <input.j2k|.jph> [-threads T] [-iter K] [-warmup W]
//                       [-win 256,512,1024] [-maxlevel L] [-full] [-verify]
//                       [-fullcap MP] [-csv]
//
// -verify byte-exact-checks the col+row windowed reuse decode against an
// independent full non-reuse decode (where the reduced level fits under
// -fullcap MP); exits 2 on any mismatch.
//
// Defaults: -threads 1  -iter 25  -warmup 3  -win 256,512,1024
//           -maxlevel <get_max_safe_reduce_NL()>  -fullcap 16 (MP)
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "decoder.hpp"

struct Args {
  std::string infile;
  uint32_t threads = 1;
  int iter         = 25;
  int warmup       = 3;
  std::vector<uint32_t> wins;  // square window edge lengths
  int maxlevel      = -1;      // -1 => get_max_safe_reduce_NL()
  bool full         = false;
  bool verify       = false;
  double fullcap_mp = 16.0;
  bool csv          = false;
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
    } else if (!strcmp(argv[i], "-full")) {
      a.full = true;
    } else if (!strcmp(argv[i], "-verify")) {
      a.verify = true;
    } else if (!strcmp(argv[i], "-fullcap") && i + 1 < argc) {
      a.fullcap_mp = atof(argv[++i]);
    } else if (!strcmp(argv[i], "-csv")) {
      a.csv = true;
    } else {
      a.infile = argv[i];
    }
  }
  if (a.wins.empty()) a.wins = {256, 512, 1024};
  std::sort(a.wins.begin(), a.wins.end());  // monotonic so the clamp-dedupe is adjacent
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

// Reused scratch for one decode (output rows + decoder-populated metadata).
struct DecodeBufs {
  std::vector<std::vector<int32_t>> out;  // extracted window, one plane/component
  std::vector<uint32_t> widths, heights;  // full reduced-level component dims
  std::vector<uint8_t> depths;
  std::vector<bool> is_signed;
};

struct Timing {
  double setup;   // init (copy) + parse + set ranges
  double decode;  // invoke_line_based_stream_reuse
  double total() const { return setup + decode; }
};

// One reuse-path region decode.  When store is true the centred
// [x0,x0+w) x [y0,y0+h) window is copied out of the full-width rows (the work a
// tile source actually does); the full-level baseline passes store=false to
// avoid allocating a multi-GB canvas.
static Timing decode_region(open_htj2k::openhtj2k_decoder &dec, const uint8_t *data, size_t len,
                            uint8_t reduce_NL, uint32_t threads, bool restrict_region, uint32_t x0,
                            uint32_t y0, uint32_t w, uint32_t h, bool store, DecodeBufs &b) {
  auto t0 = std::chrono::steady_clock::now();
  dec.init(data, len, reduce_NL, threads);
  dec.parse();
  if (restrict_region) {
    dec.set_col_range(x0, x0 + w);
    dec.set_row_range(y0, y0 + h);
  }
  auto t1 = std::chrono::steady_clock::now();
  auto cb = [&](uint32_t y, int32_t *const *rows, uint16_t nc) {
    if (!store) return;
    if (y < y0 || y >= y0 + h) return;  // row_range should already clip; guard anyway
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
  auto t2 = std::chrono::steady_clock::now();
  Timing tm;
  tm.setup  = std::chrono::duration<double, std::milli>(t1 - t0).count();
  tm.decode = std::chrono::duration<double, std::milli>(t2 - t1).count();
  return tm;
}

// Byte-exact check of the timed path: decode the centred window through the
// reuse path (col_range + row_range together) and compare against a full-level
// reference produced by the independent, reuse-disabled non-reuse line-based
// path.  col_range_compare and row_range_compare each validate only one axis;
// this is the only check of the 2D combination the benchmark actually times.
// Returns true on exact match within the window.
static bool verify_window(const uint8_t *data, size_t len, uint8_t reduce_NL, uint32_t threads, uint32_t x0,
                          uint32_t y0, uint32_t w, uint32_t h) {
  std::vector<std::vector<int32_t>> ref;
  std::vector<uint32_t> rw, rh;
  std::vector<uint8_t> rd;
  std::vector<bool> rs;
  {
    open_htj2k::openhtj2k_decoder dec;  // reuse disabled (independent reference)
    dec.init(data, len, reduce_NL, threads);
    dec.parse();
    auto cb = [&](uint32_t y, int32_t *const *rows, uint16_t nc) {
      if (ref.empty()) {
        ref.resize(nc);
        for (uint16_t c = 0; c < nc; ++c) ref[c].assign(static_cast<size_t>(rw[c]) * rh[c], 0);
      }
      for (uint16_t c = 0; c < nc; ++c)
        if (y < rh[c])
          std::memcpy(ref[c].data() + static_cast<size_t>(y) * rw[c], rows[c], rw[c] * sizeof(int32_t));
    };
    dec.invoke_line_based_stream(cb, rw, rh, rd, rs);
  }
  if (ref.empty()) return false;

  DecodeBufs b;
  open_htj2k::openhtj2k_decoder dec;
  dec.enable_single_tile_reuse(true);
  decode_region(dec, data, len, reduce_NL, threads, true, x0, y0, w, h, true, b);  // warm
  b.out.clear();
  decode_region(dec, data, len, reduce_NL, threads, true, x0, y0, w, h, true, b);  // measured-equivalent
  for (size_t c = 0; c < ref.size() && c < b.out.size(); ++c) {
    const uint32_t cw = rw[c];
    for (uint32_t ly = 0; ly < h; ++ly) {
      for (uint32_t lx = 0; lx < w; ++lx) {
        const uint32_t gx = x0 + lx, gy = y0 + ly;
        if (gx >= cw || gy >= rh[c]) continue;
        const int32_t r = ref[c][static_cast<size_t>(gy) * cw + gx];
        const int32_t g = b.out[c][static_cast<size_t>(ly) * w + lx];
        if (r != g) {
          printf("# VERIFY FAIL level %u comp %zu pixel (%u,%u): ref=%d got=%d\n", reduce_NL, c, gx, gy, r,
                 g);
          return false;
        }
      }
    }
  }
  return true;
}

static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}
static double minimum(const std::vector<double> &v) { return *std::min_element(v.begin(), v.end()); }

// Print one measured row (total = setup + decode).
static void print_row(const Args &a, int li, uint32_t W, uint32_t H, const char *winlbl, uint32_t winw,
                      uint32_t winh, int iters, const std::vector<double> &tot,
                      const std::vector<double> &dec, const std::vector<double> &set) {
  const double med_tot = median(tot), min_tot = minimum(tot);
  const double med_dec = median(dec), med_set = median(set);
  const double rps    = 1000.0 / med_tot;
  const double mpixps = static_cast<double>(winw) * winh / med_tot / 1000.0;
  if (a.csv) {
    printf("%d,%u,%u,%s,%u,%u,%llu,%u,%d,%.4f,%.4f,%.4f,%.4f,%.1f,%.1f\n", li, W, H, winlbl, winw, winh,
           (unsigned long long)winw * winh, a.threads, iters, med_tot, med_dec, med_set, min_tot, rps,
           mpixps);
  } else {
    printf("%-5d  %5ux%-7u  %-11s  %-9llu  %-5d  %-9.4f  %-9.4f  %-9.4f  %-9.4f  %-10.1f  %-8.1f\n", li, W,
           H, winlbl, (unsigned long long)winw * winh, iters, med_tot, med_dec, med_set, min_tot, rps,
           mpixps);
  }
  fflush(stdout);
}

int main(int argc, char *argv[]) {
  Args a;
  if (!parse_args(argc, argv, a)) {
    printf(
        "Usage: region_decode_bench <input.j2k|.jph> [-threads T] [-iter K] [-warmup W]\n"
        "                           [-win 256,512,1024] [-maxlevel L] [-full] [-verify] [-fullcap MP] "
        "[-csv]\n");
    return 1;
  }

  std::vector<uint8_t> file = read_file(a.infile.c_str());
  if (file.empty()) return 1;
  const uint8_t *data = file.data();
  const size_t len    = file.size();

  // ── Banner: query stream geometry from a level-0 parse ──────────────────
  uint16_t nc     = 0;
  uint32_t full_w = 0, full_h = 0, cs = 0;
  uint8_t depth = 0, mct = 0, max_reduce = 0;
  try {
    open_htj2k::openhtj2k_decoder probe;
    probe.init(data, len, 0, a.threads);
    probe.parse();
    nc         = probe.get_num_component();
    full_w     = probe.get_component_width(0);
    full_h     = probe.get_component_height(0);
    depth      = probe.get_component_depth(0);
    mct        = probe.get_mct();
    cs         = probe.get_colorspace();
    max_reduce = probe.get_max_safe_reduce_NL();
  } catch (std::exception &e) {
    printf("ERROR: parse failed: %s\n", e.what());
    return 1;
  }
  const int maxlevel = (a.maxlevel >= 0) ? std::min<int>(a.maxlevel, max_reduce) : max_reduce;

  if (a.csv) {
    printf(
        "level,level_w,level_h,window,win_w,win_h,win_px,threads,iters,med_ms,dec_ms,set_ms,min_ms,"
        "regions_s,mpix_s\n");
  } else {
    printf("# region_decode_bench  file=%s  (%zu bytes)\n", a.infile.c_str(), len);
    printf("# full=%ux%u  comps=%u  depth=%u  mct=%u  colorspace=%u  max_safe_reduce=%u\n", full_w, full_h,
           nc, depth, mct, cs, max_reduce);
    printf(
        "# threads=%u  iter=%d  warmup=%d  levels=0..%d   reuse path; per-tile = setup(copy+parse) + "
        "decode(invoke)\n",
        a.threads, a.iter, a.warmup, maxlevel);
    printf("%-5s  %-13s  %-11s  %-9s  %-5s  %-9s  %-9s  %-9s  %-9s  %-10s  %-8s\n", "level", "level_dims",
           "window", "win_px", "iters", "med_ms", "dec_ms", "set_ms", "min_ms", "regions/s", "Mpix/s");
  }

  int verify_fail = 0;
  for (int li = 0; li <= maxlevel; ++li) {
    open_htj2k::openhtj2k_decoder dec;
    dec.enable_single_tile_reuse(true);
    DecodeBufs b;
    uint32_t W = 0, H = 0;
    try {
      // Reduced-level dims: get_component_width() returns FULL-res dims (it
      // ignores reduce_NL), so query them the way col_range_compare does — from
      // the decoder callback.  A 1x1 restricted decode is cheap (it still emits
      // full-width metadata) and doubles as the reuse-cache warm-up.
      decode_region(dec, data, len, static_cast<uint8_t>(li), a.threads, true, 0, 0, 1, 1, false, b);
      if (!b.widths.empty()) {
        W = b.widths[0];
        H = b.heights[0];
      }
    } catch (std::exception &e) {
      printf("level %d: probe failed: %s\n", li, e.what());
      continue;
    }
    if (W == 0 || H == 0) continue;

    // Optional full-level baseline (no region restriction, no store).
    if (a.full) {
      const double mp = static_cast<double>(W) * H / 1e6;
      if (mp <= a.fullcap_mp) {
        decode_region(dec, data, len, li, a.threads, false, 0, 0, 0, 0, false, b);  // warm
        const int it = std::max(1, a.iter / 4);  // full decode is dear; fewer iters
        std::vector<double> tot, dc, st;
        for (int i = 0; i < it; ++i) {
          Timing t = decode_region(dec, data, len, li, a.threads, false, 0, 0, 0, 0, false, b);
          tot.push_back(t.total());
          dc.push_back(t.decode);
          st.push_back(t.setup);
        }
        print_row(a, li, W, H, "full", W, H, it, tot, dc, st);
      } else if (!a.csv) {
        printf("%-5d  %5ux%-7u  %-11s  (%.0f MP > fullcap %.0f MP, skipped)\n", li, W, H, "full", mp,
               a.fullcap_mp);
      }
    }

    uint64_t seen_win = 0;  // skip window sizes that clamp to the same WxH at this level
    for (uint32_t S : a.wins) {
      const uint32_t w   = std::min(S, W);
      const uint32_t h   = std::min(S, H);
      const uint64_t key = (static_cast<uint64_t>(w) << 32) | h;
      if (key == seen_win) continue;  // e.g. 512 and 1024 both clamp to a small level
      seen_win          = key;
      const uint32_t x0 = (W > w) ? (W - w) / 2 : 0;
      const uint32_t y0 = (H > h) ? (H - h) / 2 : 0;

      // Optional byte-exact correctness check of the col+row windowed path
      // (only where the full reference canvas is affordable).
      if (a.verify && static_cast<double>(W) * H / 1e6 <= a.fullcap_mp) {
        const bool ok = verify_window(data, len, static_cast<uint8_t>(li), a.threads, x0, y0, w, h);
        if (!ok) ++verify_fail;
        printf("# verify  level %d  %ux%u  -> %s\n", li, w, h, ok ? "PASS (byte-exact vs full)" : "FAIL");
        fflush(stdout);
      }

      b.out.clear();  // fresh sizing for this window stride
      for (int i = 0; i < a.warmup; ++i)
        decode_region(dec, data, len, li, a.threads, true, x0, y0, w, h, true, b);
      std::vector<double> tot, dc, st;
      tot.reserve(a.iter);
      dc.reserve(a.iter);
      st.reserve(a.iter);
      for (int i = 0; i < a.iter; ++i) {
        Timing t = decode_region(dec, data, len, li, a.threads, true, x0, y0, w, h, true, b);
        tot.push_back(t.total());
        dc.push_back(t.decode);
        st.push_back(t.setup);
      }
      char winlbl[24];
      snprintf(winlbl, sizeof(winlbl), "%ux%u", w, h);
      print_row(a, li, W, H, winlbl, w, h, a.iter, tot, dc, st);
    }
  }
  if (a.verify) printf("# verify summary: %s (%d failures)\n", verify_fail ? "FAIL" : "PASS", verify_fail);
  return verify_fail ? 2 : 0;
}
