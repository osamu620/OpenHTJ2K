// row_range_compare: validate set_row_range() against a full-canvas reference.
// Decodes the input codestream twice: once with the full row range (reference),
// then with several narrow [row_lo, row_hi) windows.  Asserts every in-window
// pixel matches the reference byte-exactly.
// Usage: row_range_compare <input.j2k> [-reduce N]
// Exits 0 on exact match, non-zero on mismatch or error.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include "decoder.hpp"

static bool parse_args(int argc, char *argv[], std::string &infile, uint8_t &reduce_NL) {
  infile    = "";
  reduce_NL = 0;
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "-reduce") == 0 || strcmp(argv[i], "-r") == 0) {
      if (++i < argc) reduce_NL = static_cast<uint8_t>(atoi(argv[i]));
    } else {
      infile = argv[i];
    }
  }
  return !infile.empty();
}

static std::vector<uint8_t> read_file(const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f) { printf("ERROR: cannot open %s\n", path); return {}; }
  fseek(f, 0, SEEK_END);
  size_t sz = static_cast<size_t>(ftell(f));
  fseek(f, 0, SEEK_SET);
  std::vector<uint8_t> buf(sz);
  size_t rd = fread(buf.data(), 1, sz, f);
  fclose(f);
  if (rd != sz) { printf("ERROR: partial read of %s\n", path); buf.clear(); }
  return buf;
}

// Decode with an optional row range, returning a flat per-component image.
// When row_lo == 0 and row_hi == UINT32_MAX the decoder runs without any
// set_row_range call (equivalent to the default path).  Otherwise
// set_row_range(row_lo, row_hi) is applied and only rows inside the window
// are stored; rows outside remain zero in the output buffers.
static bool decode_with_range(const std::vector<uint8_t> &codestream, uint8_t reduce_NL,
                              uint32_t row_lo, uint32_t row_hi,
                              std::vector<std::vector<int32_t>> &out,
                              std::vector<uint32_t> &widths,
                              std::vector<uint32_t> &heights,
                              std::vector<uint8_t>  &depths,
                              std::vector<bool>     &is_signed) {
  try {
    open_htj2k::openhtj2k_decoder dec;
    dec.init(codestream.data(), codestream.size(), reduce_NL, 1);
    dec.parse();
    if (row_lo != 0 || row_hi != UINT32_MAX) {
      dec.set_row_range(row_lo, row_hi);
    }
    auto cb = [&](uint32_t y, int32_t *const *rows, uint16_t nc) {
      if (out.empty()) {
        out.resize(nc);
        for (uint16_t c = 0; c < nc; ++c)
          out[c].assign(static_cast<size_t>(widths[c]) * heights[c], 0);
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

// Compare out[c][(y,x)] against ref[c][(y,x)] for y in [row_lo, row_hi).
// Returns true if every pixel matches.
static bool compare_window(const std::vector<std::vector<int32_t>> &ref,
                           const std::vector<std::vector<int32_t>> &got,
                           const std::vector<uint32_t> &widths,
                           const std::vector<uint32_t> &heights,
                           uint32_t row_lo, uint32_t row_hi,
                           const char *label, const char *infile) {
  if (ref.size() != got.size()) {
    printf("FAIL [%s %s]: component count mismatch (%zu vs %zu)\n",
           infile, label, ref.size(), got.size());
    return false;
  }
  for (size_t c = 0; c < ref.size(); ++c) {
    const uint32_t W = widths[c];
    const uint32_t H = heights[c];
    const uint32_t y0 = std::min(row_lo, H);
    const uint32_t y1 = std::min(row_hi, H);
    for (uint32_t y = y0; y < y1; ++y) {
      const int32_t *rp = ref[c].data() + static_cast<size_t>(y) * W;
      const int32_t *gp = got[c].data() + static_cast<size_t>(y) * W;
      for (uint32_t x = 0; x < W; ++x) {
        if (rp[x] != gp[x]) {
          printf("FAIL [%s %s]: comp %zu pixel (%u,%u): ref=%d got=%d\n",
                 infile, label, c, x, y, rp[x], gp[x]);
          return false;
        }
      }
    }
  }
  return true;
}

int main(int argc, char *argv[]) {
  std::string infile;
  uint8_t reduce_NL = 0;
  if (!parse_args(argc, argv, infile, reduce_NL)) {
    printf("Usage: row_range_compare <input.j2k> [-reduce N]\n");
    return 1;
  }

  std::vector<uint8_t> codestream = read_file(infile.c_str());
  if (codestream.empty()) return 1;

  // ── Reference: full-canvas decode via invoke_line_based_stream ──────────
  std::vector<std::vector<int32_t>> ref;
  std::vector<uint32_t> widths, heights;
  std::vector<uint8_t>  depths;
  std::vector<bool>     is_signed;
  if (!decode_with_range(codestream, reduce_NL, 0, UINT32_MAX,
                         ref, widths, heights, depths, is_signed)) {
    return 1;
  }
  if (ref.empty() || heights.empty()) {
    printf("ERROR: reference decode produced no output\n");
    return 1;
  }
  const uint32_t H = heights[0];
  if (H < 16) {
    printf("SKIP: %s is too small (H=%u) for row-range tests\n", infile.c_str(), H);
    return 0;
  }

  struct Case {
    const char *label;
    uint32_t    row_lo;
    uint32_t    row_hi;
  };
  const uint32_t half   = H / 2;
  const uint32_t offset = 37;                          // deliberately not codeblock-aligned
  const uint32_t narrow_lo = std::min(half, H - 1);
  const uint32_t narrow_hi = std::min(half + offset, H);
  std::vector<Case> cases = {
      {"bottom_half",   half,                     H},
      {"top_half",      0,                        half},
      {"offset_to_end", std::min(half + offset, H), H},
      {"narrow_window", narrow_lo,                narrow_hi},
  };

  bool ok = true;
  for (const Case &cs : cases) {
    // Empty or degenerate windows: nothing to verify.
    if (cs.row_lo >= cs.row_hi) continue;
    std::vector<std::vector<int32_t>> got;
    std::vector<uint32_t> gw, gh;
    std::vector<uint8_t>  gd;
    std::vector<bool>     gs;
    if (!decode_with_range(codestream, reduce_NL, cs.row_lo, cs.row_hi,
                           got, gw, gh, gd, gs)) {
      ok = false;
      break;
    }
    if (!compare_window(ref, got, widths, heights, cs.row_lo, cs.row_hi,
                        cs.label, infile.c_str())) {
      ok = false;
      break;
    }
  }

  if (ok) {
    printf("PASS: %s (reduce=%u, %zu cases)\n", infile.c_str(), reduce_NL, cases.size());
    return 0;
  }
  return 1;
}
