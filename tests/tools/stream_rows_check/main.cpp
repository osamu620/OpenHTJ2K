// stream_rows_check: the rows of the streaming entry points against the buffered decode.
// Decodes the input with invoke_line_based() as the reference, then with
// invoke_line_based_stream(), invoke_line_based_stream_reuse() (two frames),
// invoke_line_based_direct() (with and without reuse, components of up to 16 bits,
// unsigned) and with set_row_range() / set_row_limit() / set_col_range() windows, and
// checks the row layout documented in decoder.hpp: the callback runs for output rows
// 0 .. get_stream_row_count() - 1 in order (those of the window), row_ptrs[c] holds row
// get_stream_component_row(c, y) of component c, equal to the reference row, the same
// row while it spans several output rows, zeros where no row of the component covers
// the output row, and every component row covering the window is delivered.
// Usage: stream_rows_check <input> [-reduce N] [-threads N]
// Prints PASS and exits 0, or prints what differs (or the decoder's error) and exits 1.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "decoder.hpp"

namespace {
std::vector<uint8_t> cs;
uint8_t reduce_NL = 0;
uint32_t threads  = 1;
int failures      = 0;

struct Image {
  std::vector<std::vector<int32_t>> plane;
  std::vector<uint32_t> w, h;
  std::vector<uint8_t> depth;
  std::vector<bool> is_signed;
};

void fail(const std::string &label, const char *what) {
  if (failures < 20) printf("FAIL [%s]: %s\n", label.c_str(), what);
  ++failures;
}

// Samples deeper than a float's 24-bit significand pass through the float pipeline, and the
// partial-column and reuse paths round differently from a full decode there (2-3 units in
// the last place seen, depending on compiler and optimisation): allow 16 units in the last
// place of a float at full scale.  A wrong row or column is off by far more.  Samples of
// up to 24 bits must match exactly.
int64_t sample_tolerance(uint8_t depth) { return (depth > 24) ? (int64_t{1} << (depth - 20)) : 0; }

// XRsiz of component c, from the SIZ marker segment
uint32_t siz_xr(size_t c) {
  for (size_t i = 0; i + 4 < cs.size(); ++i) {
    if (cs[i] == 0xFF && cs[i + 1] == 0x4F && cs[i + 2] == 0xFF && cs[i + 3] == 0x51) {
      const size_t p = i + 42 + 3 * c + 1;  // SOC, SIZ, Lsiz, Rsiz to Csiz; then Ssiz XRsiz YRsiz
      return (p < cs.size() && cs[p] != 0) ? cs[p] : 1;
    }
  }
  return 1;
}

bool buffered(Image &ref) {
  try {
    open_htj2k::openhtj2k_decoder dec(cs.data(), cs.size(), reduce_NL, threads);
    dec.parse();
    std::vector<int32_t *> buf;
    dec.invoke_line_based(buf, ref.w, ref.h, ref.depth, ref.is_signed);
    ref.plane.resize(buf.size());
    for (size_t c = 0; c < buf.size(); ++c) {
      ref.plane[c].assign(buf[c], buf[c] + static_cast<size_t>(ref.w[c]) * ref.h[c]);
      delete[] buf[c];
    }
  } catch (std::exception &e) {
    printf("ERROR: %s\n", e.what());
    return false;
  }
  return true;
}

// reuse: decode two frames through invoke_line_based_stream_reuse and check the second.
// [lo, hi): set_row_range (lo == 0: set_row_limit(hi)); [clo, chi): set_col_range.
void stream(const Image &ref, bool reuse, uint32_t lo, uint32_t hi, uint32_t clo, uint32_t chi,
            const std::string &label) {
  const size_t NC = ref.plane.size();
  try {
    open_htj2k::openhtj2k_decoder dec;
    if (reuse) dec.enable_single_tile_reuse(true);
    for (int frame = 0; frame < (reuse ? 2 : 1); ++frame) {
      dec.init(cs.data(), cs.size(), reduce_NL, threads);
      dec.parse();
      if (lo != 0)
        dec.set_row_range(lo, hi);
      else if (hi != UINT32_MAX)
        dec.set_row_limit(hi);
      if (clo != 0 || chi != UINT32_MAX) dec.set_col_range(clo, chi);
      const uint32_t N   = dec.get_stream_row_count();
      const uint32_t end = std::min(hi, N);
      std::vector<std::vector<char>> seen(NC);
      std::vector<std::vector<int32_t>> last(NC);
      std::vector<uint32_t> last_row(NC, UINT32_MAX), x0(NC), x1(NC);
      for (size_t c = 0; c < NC; ++c) {
        seen[c].assign(ref.h[c], 0);
        x0[c] = 0;
        x1[c] = ref.w[c];
        if (clo != 0 || chi != UINT32_MAX) {  // the columns the window keeps (decoder.hpp)
          const uint32_t xr = siz_xr(c);
          x0[c]             = std::min(clo / xr, ref.w[c]);
          x1[c]             = static_cast<uint32_t>(
              std::min<uint64_t>((static_cast<uint64_t>(chi) + xr - 1) / xr, ref.w[c]));
        }
      }
      uint32_t next_y = std::min(lo, N);
      bool bad        = false;
      std::vector<uint32_t> W, H;
      std::vector<uint8_t> D;
      std::vector<bool> S;
      auto cb = [&](uint32_t y, int32_t *const *rows, uint16_t nc) {
        if (bad) return;
        char msg[256];
        if (y != next_y || y >= end || nc != NC) {
          snprintf(msg, sizeof(msg), "callback row %u (nc %u), expected row %u of [%u, %u)", y, nc, next_y,
                   lo, end);
          fail(label, msg);
          bad = true;
          return;
        }
        next_y = y + 1;
        for (uint16_t c = 0; c < nc; ++c) {
          if (rows[c] == nullptr) {
            fail(label, "null row pointer");
            bad = true;
            return;
          }
          const uint32_t j = dec.get_stream_component_row(c, y);
          if (j == UINT32_MAX) {
            for (uint32_t x = x0[c]; x < x1[c]; ++x) {
              if (rows[c][x] != 0) {
                snprintf(msg, sizeof(msg), "component %u, output row %u covers no row but holds %d", c, y,
                         rows[c][x]);
                fail(label, msg);
                bad = true;
                return;
              }
            }
            continue;
          }
          if (j >= ref.h[c] || (last_row[c] != UINT32_MAX && j < last_row[c])) {
            snprintf(msg, sizeof(msg), "component %u, output row %u maps to row %u", c, y, j);
            fail(label, msg);
            bad = true;
            return;
          }
          if (j == last_row[c]) {  // a row spanning several output rows stays the same
            if (memcmp(last[c].data(), rows[c] + x0[c], (x1[c] - x0[c]) * sizeof(int32_t)) != 0) {
              snprintf(msg, sizeof(msg), "component %u row %u changed at output row %u", c, j, y);
              fail(label, msg);
              bad = true;
              return;
            }
            continue;
          }
          last_row[c] = j;
          last[c].assign(rows[c] + x0[c], rows[c] + x1[c]);
          seen[c][j]       = 1;
          const int32_t *r  = ref.plane[c].data() + static_cast<size_t>(j) * ref.w[c];
          const int64_t tol = sample_tolerance(ref.depth[c]);
          for (uint32_t x = x0[c]; x < x1[c]; ++x) {
            const int64_t d = static_cast<int64_t>(rows[c][x]) - r[x];
            if (d > tol || d < -tol) {
              snprintf(msg, sizeof(msg), "component %u sample (%u, %u) is %d, reference %d", c, x, j,
                       rows[c][x], r[x]);
              fail(label, msg);
              bad = true;
              return;
            }
          }
        }
      };
      if (reuse)
        dec.invoke_line_based_stream_reuse(cb, W, H, D, S);
      else
        dec.invoke_line_based_stream(cb, W, H, D, S);
      if (bad || frame + 1 < (reuse ? 2 : 1)) continue;
      if (lo < end && next_y != end) {
        char msg[128];
        snprintf(msg, sizeof(msg), "the callback stopped at output row %u of [%u, %u)", next_y, lo, end);
        fail(label, msg);
      }
      // every component row covering an output row of the window was delivered
      for (size_t c = 0; c < NC; ++c) {
        for (uint32_t y = lo; y < end; ++y) {
          const uint32_t j = dec.get_stream_component_row(static_cast<uint16_t>(c), y);
          if (j != UINT32_MAX && j < ref.h[c] && !seen[c][j]) {
            char msg[128];
            snprintf(msg, sizeof(msg), "row %u of component %zu was not delivered", j, c);
            fail(label, msg);
            break;
          }
        }
        if (lo == 0 && end == N && ref.w[c] > 0) {
          for (uint32_t j = 0; j < ref.h[c]; ++j) {
            if (!seen[c][j]) {
              char msg[128];
              snprintf(msg, sizeof(msg), "row %u of component %zu was not delivered", j, c);
              fail(label, msg);
              break;
            }
          }
        }
      }
    }
  } catch (std::exception &e) {
    fail(label, e.what());
  }
}

void direct(const Image &ref, bool reuse) {
  const size_t NC = ref.plane.size();
  for (size_t c = 0; c < NC; ++c) {
    if (ref.depth[c] > 16 || ref.is_signed[c]) return;  // 16-bit unsigned planes only
  }
  const std::string label = reuse ? "direct_reuse" : "direct";
  try {
    open_htj2k::openhtj2k_decoder dec;
    if (reuse) dec.enable_single_tile_reuse(true);
    dec.init(cs.data(), cs.size(), reduce_NL, threads);
    dec.parse();
    std::vector<std::vector<uint16_t>> planes(NC);
    std::vector<open_htj2k::PlanarOutputDesc> descs(NC);
    for (size_t c = 0; c < NC; ++c) {
      planes[c].assign(static_cast<size_t>(ref.w[c]) * ref.h[c] + 1, 0xBEEF);  // one guard sample
      open_htj2k::PlanarOutputDesc &d = descs[c];
      d.base                          = planes[c].data();
      d.stride                        = ref.w[c];
      d.width                         = ref.w[c];
      d.height                        = ref.h[c];
      d.yr                            = 1;  // not used by the decoder
      d.dc                            = 1 << (ref.depth[c] - 1);
      d.maxval                        = (1 << ref.depth[c]) - 1;
      d.minval                        = 0;
      d.depth_shift                   = 0;
      d.is_16bit                      = true;
    }
    std::vector<uint32_t> W, H;
    std::vector<uint8_t> D;
    std::vector<bool> S;
    dec.invoke_line_based_direct(descs.data(), static_cast<uint16_t>(NC), W, H, D, S);
    for (size_t c = 0; c < NC; ++c) {
      for (size_t i = 0; i < ref.plane[c].size(); ++i) {
        if (planes[c][i] != static_cast<uint16_t>(ref.plane[c][i])) {
          char msg[128];
          snprintf(msg, sizeof(msg), "component %zu sample %zu is %u, reference %d", c, i, planes[c][i],
                   ref.plane[c][i]);
          fail(label, msg);
          break;
        }
      }
      if (planes[c].back() != 0xBEEF) fail(label, "wrote past the plane");
    }
  } catch (std::exception &e) {
    fail(label, e.what());
  }
}
}  // namespace

int main(int argc, char *argv[]) {
  std::string infile;
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "-reduce") == 0 && i + 1 < argc) {
      reduce_NL = static_cast<uint8_t>(atoi(argv[++i]));
    } else if (strcmp(argv[i], "-threads") == 0 && i + 1 < argc) {
      threads = static_cast<uint32_t>(atoi(argv[++i]));
    } else {
      infile = argv[i];
    }
  }
  if (infile.empty()) {
    printf("Usage: stream_rows_check <input> [-reduce N] [-threads N]\n");
    return 1;
  }
  FILE *f = fopen(infile.c_str(), "rb");
  if (f == nullptr) {
    printf("ERROR: cannot open %s\n", infile.c_str());
    return 1;
  }
  fseek(f, 0, SEEK_END);
  cs.resize(static_cast<size_t>(ftell(f)));
  fseek(f, 0, SEEK_SET);
  const size_t got = fread(cs.data(), 1, cs.size(), f);
  fclose(f);
  if (got != cs.size()) {
    printf("ERROR: cannot read %s\n", infile.c_str());
    return 1;
  }

  Image ref;
  if (!buffered(ref)) return 1;
  uint32_t N = 0;
  try {
    open_htj2k::openhtj2k_decoder dec(cs.data(), cs.size(), reduce_NL, threads);
    dec.parse();
    N = dec.get_stream_row_count();
  } catch (std::exception &e) {
    printf("ERROR: %s\n", e.what());
    return 1;
  }

  stream(ref, false, 0, UINT32_MAX, 0, UINT32_MAX, "stream");
  stream(ref, true, 0, UINT32_MAX, 0, UINT32_MAX, "reuse");
  direct(ref, false);
  direct(ref, true);
  if (N > 2) {
    // row windows: halves, a third, one row, and the limit alone
    const uint32_t win[][2] = {{N / 2, N}, {1, N / 2}, {N / 3, (2 * N) / 3 + 1}, {N - 1, N}, {0, N / 2}};
    for (const auto &w : win) {
      if (w[0] >= w[1]) continue;
      const std::string tag = "rows[" + std::to_string(w[0]) + "," + std::to_string(w[1]) + ")";
      stream(ref, false, w[0], w[1], 0, UINT32_MAX, "stream " + tag);
      stream(ref, true, w[0], w[1], 0, UINT32_MAX, "reuse " + tag);
    }
  }
  const uint32_t W0 = ref.w.empty() ? 0 : ref.w[0];
  if (W0 > 4) {
    // column windows (the reuse path narrows the decode to them), alone and with rows
    const uint32_t win[][2] = {{W0 / 3, (2 * W0) / 3 + 1}, {W0 / 2, UINT32_MAX}};
    for (const auto &w : win) {
      const std::string tag = "cols[" + std::to_string(w[0]) + "," + std::to_string(w[1]) + ")";
      stream(ref, true, 0, UINT32_MAX, w[0], w[1], "reuse " + tag);
      if (N > 2) stream(ref, true, N / 3, N - 1, w[0], w[1], "reuse rows " + tag);
    }
  }
  if (failures != 0) return 1;
  printf("PASS: %s (%u output rows, %zu components, reduce %u, %u threads)\n", infile.c_str(), N,
         ref.plane.size(), reduce_NL, threads);
  return 0;
}
