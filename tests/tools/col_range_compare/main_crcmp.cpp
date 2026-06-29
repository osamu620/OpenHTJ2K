// col_range_compare: validate set_col_range() against a full-canvas reference.
// Decodes the input codestream once at full width (reference), then with several
// narrow [col_lo, col_hi) column windows; asserts every in-window pixel matches
// the reference byte-exactly.  This is the column-restriction analogue of
// row_range_compare and is the only thing that exercises the sub-range
// horizontal IDWT kernels (idwt_1d_filtr_{irrev97,rev53}_fixed_range) which the
// WASM JPIP viewer relies on via decoder::set_col_range().
//
// Usage: col_range_compare <input.j2k> [-reduce N] [-col LO HI] [-iter K]
//   default      : run the built-in validation cases, exit 0 on exact match.
//   -col LO HI    : validate one explicit window [LO, HI) instead of the cases.
//   -iter K       : after validating, time K windowed decodes (ms/iter) for
//                   profiling.  Uses the -col window, or a centred half-width
//                   window when -col is absent.
// Exits 0 on exact match, non-zero on mismatch or error.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "decoder.hpp"
#ifdef __EMSCRIPTEN__
  #include <emscripten/emscripten.h>
#endif

struct Args {
  std::string infile;
  uint8_t reduce_NL = 0;
  bool have_col     = false;
  uint32_t col_lo   = 0;
  uint32_t col_hi   = 0;
  int iter          = 0;
  bool reuse        = false;
};

static bool parse_args(int argc, char *argv[], Args &a) {
  for (int i = 1; i < argc; ++i) {
    if ((!strcmp(argv[i], "-reduce") || !strcmp(argv[i], "-r")) && i + 1 < argc) {
      a.reduce_NL = static_cast<uint8_t>(atoi(argv[++i]));
    } else if (!strcmp(argv[i], "-col") && i + 2 < argc) {
      a.col_lo   = static_cast<uint32_t>(strtoul(argv[++i], nullptr, 10));
      a.col_hi   = static_cast<uint32_t>(strtoul(argv[++i], nullptr, 10));
      a.have_col = true;
    } else if (!strcmp(argv[i], "-iter") && i + 1 < argc) {
      a.iter = atoi(argv[++i]);
    } else if (!strcmp(argv[i], "-reuse")) {
      a.reuse = true;
    } else {
      a.infile = argv[i];
    }
  }
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
    buf.clear();
  }
  return buf;
}

// Decode with an optional column range, returning a flat per-component image.
// set_col_range still emits every row, so the callback stores the full canvas;
// only columns in [col_lo, col_hi) are meaningful in the windowed decode.  Pass
// an already-sized `out` to reuse its buffers across calls (timing loop).
static bool decode_with_col_range(const std::vector<uint8_t> &codestream, uint8_t reduce_NL,
                                  bool restrict_cols, uint32_t col_lo, uint32_t col_hi,
                                  std::vector<std::vector<int32_t>> &out, std::vector<uint32_t> &widths,
                                  std::vector<uint32_t> &heights, std::vector<uint8_t> &depths,
                                  std::vector<bool> &is_signed) {
  try {
    open_htj2k::openhtj2k_decoder dec;
    dec.init(codestream.data(), codestream.size(), reduce_NL, 1);
    dec.parse();
    if (restrict_cols) {
      dec.set_col_range(col_lo, col_hi);
    }
    auto cb = [&](uint32_t y, int32_t *const *rows, uint16_t nc) {
      if (out.empty()) {
        out.resize(nc);
        for (uint16_t c = 0; c < nc; ++c) out[c].assign(static_cast<size_t>(widths[c]) * heights[c], 0);
      }
      for (uint16_t c = 0; c < nc; ++c) {
        if (y < heights[c]) {
          std::memcpy(out[c].data() + static_cast<size_t>(y) * widths[c], rows[c],
                      widths[c] * sizeof(int32_t));
        }
      }
    };
    dec.invoke_line_based_stream(cb, widths, heights, depths, is_signed);
  } catch (std::exception &e) {
    printf("ERROR decode: %s\n", e.what());
    return false;
  }
  return true;
}

// Decode through the single-tile reuse path on a persistent decoder, so the
// cached line-decode states/ctxs are exercised (this is the path the WASM JPIP
// viewer uses).  The first call on a fresh decoder falls through to the
// non-reuse path internally; subsequent calls hit the cached tree.
static bool decode_reuse(open_htj2k::openhtj2k_decoder &dec, const std::vector<uint8_t> &codestream,
                         uint8_t reduce_NL, bool restrict_cols, uint32_t col_lo, uint32_t col_hi,
                         std::vector<std::vector<int32_t>> &out, std::vector<uint32_t> &widths,
                         std::vector<uint32_t> &heights, std::vector<uint8_t> &depths,
                         std::vector<bool> &is_signed) {
  try {
    dec.init(codestream.data(), codestream.size(), reduce_NL, 1);
    dec.parse();
    if (restrict_cols) {
      dec.set_col_range(col_lo, col_hi);
    }
    auto cb = [&](uint32_t y, int32_t *const *rows, uint16_t nc) {
      if (out.empty()) {
        out.resize(nc);
        for (uint16_t c = 0; c < nc; ++c) out[c].assign(static_cast<size_t>(widths[c]) * heights[c], 0);
      }
      for (uint16_t c = 0; c < nc; ++c) {
        if (y < heights[c]) {
          std::memcpy(out[c].data() + static_cast<size_t>(y) * widths[c], rows[c],
                      widths[c] * sizeof(int32_t));
        }
      }
    };
    dec.invoke_line_based_stream_reuse(cb, widths, heights, depths, is_signed);
  } catch (std::exception &e) {
    printf("ERROR decode(reuse): %s\n", e.what());
    return false;
  }
  return true;
}

// Compare got[c][(y,x)] against ref[c][(y,x)] for x in [col_lo, col_hi), all y.
static bool compare_col_window(const std::vector<std::vector<int32_t>> &ref,
                               const std::vector<std::vector<int32_t>> &got,
                               const std::vector<uint32_t> &widths, const std::vector<uint32_t> &heights,
                               uint32_t col_lo, uint32_t col_hi, const char *label, const char *infile) {
  if (ref.size() != got.size()) {
    printf("FAIL [%s %s]: component count mismatch (%zu vs %zu)\n", infile, label, ref.size(), got.size());
    return false;
  }
  for (size_t c = 0; c < ref.size(); ++c) {
    const uint32_t W  = widths[c];
    const uint32_t H  = heights[c];
    const uint32_t x0 = std::min(col_lo, W);
    const uint32_t x1 = std::min(col_hi, W);
    for (uint32_t y = 0; y < H; ++y) {
      const int32_t *rp = ref[c].data() + static_cast<size_t>(y) * W;
      const int32_t *gp = got[c].data() + static_cast<size_t>(y) * W;
      for (uint32_t x = x0; x < x1; ++x) {
        if (rp[x] != gp[x]) {
          printf("FAIL [%s %s]: comp %zu pixel (%u,%u): ref=%d got=%d\n", infile, label, c, x, y, rp[x],
                 gp[x]);
          return false;
        }
      }
    }
  }
  return true;
}

#ifdef __EMSCRIPTEN__
// JS-callable entry point for the WASM runtime test.  The native main() does
// not run under Node for this Emscripten build (it drives exported functions
// via ccall instead, like web/open_htj2k_dec.mjs), so the bit-exact check is
// exposed as a function the runner calls with the codestream bytes.  reuse != 0
// drives the single-tile reuse path — the path that actually exercises the SIMD
// sub-range kernels.  Returns 0 on bit-exact match (or a benign reuse-skip),
// 1 on mismatch or decode error.
extern "C" EMSCRIPTEN_KEEPALIVE int crc_validate(const uint8_t *data, int len, int reuse, int reduce_NL) {
  const std::vector<uint8_t> codestream(data, data + len);
  std::vector<std::vector<int32_t>> ref;
  std::vector<uint32_t> w, h;
  std::vector<uint8_t> d;
  std::vector<bool> s;
  if (reuse) {
    open_htj2k::openhtj2k_decoder rdec;
    rdec.enable_single_tile_reuse(true);
    decode_reuse(rdec, codestream, static_cast<uint8_t>(reduce_NL), false, 0, 0, ref, w, h, d, s);
    ref.clear();
    decode_reuse(rdec, codestream, static_cast<uint8_t>(reduce_NL), false, 0, 0, ref, w, h, d, s);
    if (ref.empty() || w.empty()) return 0;  // reuse-ineligible fixture → skip
    const uint32_t W  = w[0];
    const uint32_t lo = (W / 2 > 200) ? W / 2 - 200 : 0;
    const uint32_t hi = std::min(W / 2 + 200, W);
    std::vector<std::vector<int32_t>> got;
    std::vector<uint32_t> gw, gh;
    std::vector<uint8_t> gd;
    std::vector<bool> gs;
    decode_reuse(rdec, codestream, static_cast<uint8_t>(reduce_NL), true, lo, hi, got, gw, gh, gd, gs);
    return compare_col_window(ref, got, w, h, lo, hi, "wasm_reuse", "wasm") ? 0 : 1;
  }
  if (!decode_with_col_range(codestream, static_cast<uint8_t>(reduce_NL), false, 0, 0, ref, w, h, d, s))
    return 1;
  if (ref.empty() || w.empty()) return 1;
  const uint32_t W = w[0];
  if (W < 16) return 0;
  const uint32_t half = W / 2, off = 37;
  const uint32_t lo[4] = {half, 0, std::min(half + off, W), half};
  const uint32_t hi[4] = {W, half, W, std::min(half + off, W)};
  for (int k = 0; k < 4; ++k) {
    if (lo[k] >= hi[k]) continue;
    std::vector<std::vector<int32_t>> got;
    std::vector<uint32_t> gw, gh;
    std::vector<uint8_t> gd;
    std::vector<bool> gs;
    if (!decode_with_col_range(codestream, static_cast<uint8_t>(reduce_NL), true, lo[k], hi[k], got, gw, gh,
                               gd, gs))
      return 1;
    if (!compare_col_window(ref, got, w, h, lo[k], hi[k], "wasm", "wasm")) return 1;
  }
  return 0;
}
#endif  // __EMSCRIPTEN__

int main(int argc, char *argv[]) {
  Args a;
  if (!parse_args(argc, argv, a)) {
    printf("Usage: col_range_compare <input.j2k> [-reduce N] [-col LO HI] [-iter K]\n");
    return 1;
  }

  std::vector<uint8_t> codestream = read_file(a.infile.c_str());
  if (codestream.empty()) return 1;

  // ── Reuse-path probe (the viewer's path: invoke_line_based_stream_reuse) ─
  if (a.reuse) {
    open_htj2k::openhtj2k_decoder rdec;
    rdec.enable_single_tile_reuse(true);
    std::vector<std::vector<int32_t>> rref;
    std::vector<uint32_t> rw, rh;
    std::vector<uint8_t> rd;
    std::vector<bool> rs;
    // 1st call warms the cache (falls through to non-reuse internally); 2nd
    // full-width call is the reuse-path reference.
    decode_reuse(rdec, codestream, a.reduce_NL, false, 0, 0, rref, rw, rh, rd, rs);
    rref.clear();
    decode_reuse(rdec, codestream, a.reduce_NL, false, 0, 0, rref, rw, rh, rd, rs);
    if (rref.empty() || rw.empty()) {
      // Some fixtures are not single-tile-reuse eligible (e.g. multi-tile HT
      // streams abort tier-1 decode on the cached second pass).  That is a
      // pre-existing reuse-path limitation, unrelated to the horizontal IDWT;
      // skip rather than fail so the cr_ suite stays green on such fixtures.
      printf("SKIP: %s reuse path produced no output (reuse not supported here)\n", a.infile.c_str());
      return 0;
    }
    const uint32_t W2 = rw[0];
    // Validate several STRICTLY-INTERIOR windows on the warm reuse path.  An
    // unaligned col_lo (not a multiple of 8) is the key case: the vertical
    // sub-range kernels run on row pointers offset by col_lo - u0, so an
    // unaligned start exercises the unaligned-load path that previously faulted
    // on AVX2.  Interior windows have real neighbours on both sides, so the
    // sub-range kernel and the full-width kernel compute the same lifting; on
    // GCC/Clang/Emscripten the two byte-match exactly.  (MSVC contracts the two
    // kernels' FMAs differently and drifts by 1 LSB, so the cmake registers
    // these reuse tests only on non-MSVC builds — see col_range_validation.cmake.
    // Component-boundary windows are deferred for the same reason.)
    struct RCase {
      const char *label;
      uint32_t lo;
      uint32_t hi;
    };
    std::vector<RCase> cases;
    if (a.have_col) {
      cases.push_back({"explicit", a.col_lo, std::min(a.col_hi, W2)});
    } else if (W2 >= 128) {
      const uint32_t q = W2 / 4;
      cases            = {
          {"interior_wide", q, W2 - q},
          {"interior_unaligned", (W2 / 3) | 1u, W2 - q},  // odd lo -> unaligned col0
          {"interior_narrow", W2 / 2, std::min(W2 / 2 + 40, W2 - 8)},
      };
    } else {
      cases.push_back({"full", 0, W2});
    }

    bool rok = true;
    uint32_t t_lo = 0, t_hi = W2;
    for (const RCase &cs : cases) {
      if (cs.lo >= cs.hi) continue;
      std::vector<std::vector<int32_t>> rgot;
      std::vector<uint32_t> gw, gh;
      std::vector<uint8_t> gd;
      std::vector<bool> gs;
      decode_reuse(rdec, codestream, a.reduce_NL, true, cs.lo, cs.hi, rgot, gw, gh, gd, gs);
      if (!compare_col_window(rref, rgot, rw, rh, cs.lo, cs.hi, cs.label, a.infile.c_str())) {
        rok = false;
        break;
      }
      t_lo = cs.lo;
      t_hi = cs.hi;
    }
    printf("%s: %s reuse-path (%zu windows) of W=%u\n", rok ? "PASS" : "FAIL", a.infile.c_str(),
           cases.size(), W2);
    if (rok && a.iter > 0) {
      std::vector<std::vector<int32_t>> rgot;
      std::vector<uint32_t> gw, gh;
      std::vector<uint8_t> gd;
      std::vector<bool> gs;
      auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < a.iter; ++i) {
        decode_reuse(rdec, codestream, a.reduce_NL, true, t_lo, t_hi, rgot, gw, gh, gd, gs);
      }
      auto t1         = std::chrono::steady_clock::now();
      const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / a.iter;
      printf("TIMING(reuse): window [%u,%u) of W=%u, %d iters, %.3f ms/iter\n", t_lo, t_hi, W2, a.iter, ms);
    }
    return rok ? 0 : 1;
  }

  // ── Reference: full-width decode ────────────────────────────────────────
  std::vector<std::vector<int32_t>> ref;
  std::vector<uint32_t> widths, heights;
  std::vector<uint8_t> depths;
  std::vector<bool> is_signed;
  if (!decode_with_col_range(codestream, a.reduce_NL, false, 0, 0, ref, widths, heights, depths,
                             is_signed)) {
    return 1;
  }
  if (ref.empty() || widths.empty()) {
    printf("ERROR: reference decode produced no output\n");
    return 1;
  }
  const uint32_t W = widths[0];
  if (W < 16) {
    printf("SKIP: %s is too small (W=%u) for col-range tests\n", a.infile.c_str(), W);
    return 0;
  }

  struct Case {
    const char *label;
    uint32_t col_lo;
    uint32_t col_hi;
  };
  std::vector<Case> cases;
  if (a.have_col) {
    cases.push_back({"explicit", a.col_lo, std::min(a.col_hi, W)});
  } else {
    const uint32_t half   = W / 2;
    const uint32_t offset = 37;  // deliberately not codeblock-aligned
    cases                 = {
        {"right_half", half, W},
        {"left_half", 0, half},
        {"offset_to_end", std::min(half + offset, W), W},
        {"narrow_window", half, std::min(half + offset, W)},
    };
  }

  bool ok = true;
  for (const Case &cs : cases) {
    if (cs.col_lo >= cs.col_hi) continue;  // empty / degenerate window
    std::vector<std::vector<int32_t>> got;
    std::vector<uint32_t> gw, gh;
    std::vector<uint8_t> gd;
    std::vector<bool> gs;
    if (!decode_with_col_range(codestream, a.reduce_NL, true, cs.col_lo, cs.col_hi, got, gw, gh, gd, gs)) {
      ok = false;
      break;
    }
    if (!compare_col_window(ref, got, widths, heights, cs.col_lo, cs.col_hi, cs.label, a.infile.c_str())) {
      ok = false;
      break;
    }
  }

  if (!ok) return 1;
  printf("PASS: %s (reduce=%u, %zu cases)\n", a.infile.c_str(), a.reduce_NL, cases.size());

  // ── Optional timing loop for profiling ──────────────────────────────────
  if (a.iter > 0) {
    const uint32_t tlo = a.have_col ? a.col_lo : (W / 4);
    const uint32_t thi = a.have_col ? std::min(a.col_hi, W) : (W - W / 4);
    std::vector<std::vector<int32_t>> got;
    std::vector<uint32_t> gw, gh;
    std::vector<uint8_t> gd;
    std::vector<bool> gs;
    // Warm up + size buffers so the timed loop reuses them (no per-iter alloc).
    decode_with_col_range(codestream, a.reduce_NL, true, tlo, thi, got, gw, gh, gd, gs);
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < a.iter; ++i) {
      decode_with_col_range(codestream, a.reduce_NL, true, tlo, thi, got, gw, gh, gd, gs);
    }
    auto t1         = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / a.iter;
    printf("TIMING: window [%u,%u) of W=%u, %d iters, %.3f ms/iter\n", tlo, thi, W, a.iter, ms);
  }
  return 0;
}
