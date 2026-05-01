// jpip_partial_decode_check: Phase-1 ctest harness for the decoder-level
// precinct filter injection added in commit 3.  Validates two invariants:
//
//   --identity   Decode with a filter that returns true for every precinct
//                must produce byte-identical output to an unfiltered decode.
//                This exercises the `skip_body` plumbing in the keep-everything
//                branch: the body-attach path runs, but routed through the
//                filter-aware call site.
//
//   --empty      Decode with a filter that returns false for every precinct
//                must produce a uniform field — every sample within each
//                component equals the DC level (no codeblock contributes any
//                non-zero passes).  For lossy 9/7 + MCT on 8-bit unsigned the
//                expected value is 128.  The test only asserts uniformity
//                within each component; any small integer rounding noise from
//                the inverse MCT is tolerated via a PAE cap.
//
// Usage: jpip_partial_decode_check <input.j2k> [--identity] [--empty N]
//   N = maximum absolute difference between any two samples of the same
//       component (default 0 for lossless, pass 1 or 2 for lossy 9/7).
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "decoder.hpp"

static std::vector<uint8_t> read_file(const char *path) {
  FILE *f = std::fopen(path, "rb");
  if (!f) {
    std::fprintf(stderr, "ERROR: cannot open %s\n", path);
    return {};
  }
  std::fseek(f, 0, SEEK_END);
  std::size_t sz = static_cast<std::size_t>(std::ftell(f));
  std::fseek(f, 0, SEEK_SET);
  std::vector<uint8_t> buf(sz);
  std::size_t rd = std::fread(buf.data(), 1, sz, f);
  std::fclose(f);
  if (rd != sz) {
    std::fprintf(stderr, "ERROR: partial read of %s\n", path);
    buf.clear();
  }
  return buf;
}

static int decode(const std::vector<uint8_t> &codestream,
                  std::function<bool(uint16_t, uint16_t, uint8_t, uint32_t)> filter,
                  std::vector<std::vector<int32_t>> &planes,
                  std::vector<uint32_t> &w, std::vector<uint32_t> &h) {
  open_htj2k::openhtj2k_decoder dec;
  dec.init(codestream.data(), codestream.size(), 0, 1);
  dec.parse();
  if (filter) dec.set_precinct_filter(std::move(filter));
  std::vector<int32_t *>  buf;
  std::vector<uint8_t>    depth;
  std::vector<bool>       is_signed;
  try {
    dec.invoke(buf, w, h, depth, is_signed);
  } catch (std::exception &e) {
    std::fprintf(stderr, "ERROR invoke: %s\n", e.what());
    return -1;
  }
  // invoke() writes into caller-owned planes via its int32_t* vector.
  // buf[c] points into internal storage; copy it out so we can compare
  // across calls (the next invoke() may overwrite it).
  planes.resize(buf.size());
  for (std::size_t c = 0; c < buf.size(); ++c) {
    planes[c].assign(buf[c], buf[c] + static_cast<std::ptrdiff_t>(w[c]) * h[c]);
  }
  return 0;
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::fprintf(stderr,
                 "Usage: jpip_partial_decode_check <input.j2k> "
                 "(--identity | --empty <PAE>)\n");
    return 1;
  }
  std::string infile   = argv[1];
  bool        identity = false;
  bool        empty    = false;
  int         empty_pae = 0;
  for (int i = 2; i < argc; ++i) {
    if (std::strcmp(argv[i], "--identity") == 0) {
      identity = true;
    } else if (std::strcmp(argv[i], "--empty") == 0 && i + 1 < argc) {
      empty    = true;
      empty_pae = std::atoi(argv[++i]);
    } else {
      std::fprintf(stderr, "ERROR: unknown arg %s\n", argv[i]);
      return 1;
    }
  }

  auto bytes = read_file(infile.c_str());
  if (bytes.empty()) return 1;

  // Reference decode — no filter.
  std::vector<std::vector<int32_t>> ref;
  std::vector<uint32_t> w, h;
  if (decode(bytes, {}, ref, w, h) != 0) return 1;

  if (identity) {
    // Filter returns true for every precinct.
    std::vector<std::vector<int32_t>> got;
    std::vector<uint32_t> gw, gh;
    auto keep_all = [](uint16_t, uint16_t, uint8_t, uint32_t) { return true; };
    if (decode(bytes, keep_all, got, gw, gh) != 0) return 1;
    if (got.size() != ref.size() || gw != w || gh != h) {
      std::fprintf(stderr, "FAIL --identity: component/shape mismatch\n");
      return 1;
    }
    for (std::size_t c = 0; c < ref.size(); ++c) {
      if (got[c] != ref[c]) {
        // Find first diff for useful output.
        for (std::size_t i = 0; i < ref[c].size(); ++i) {
          if (got[c][i] != ref[c][i]) {
            std::fprintf(stderr,
                         "FAIL --identity: c=%zu i=%zu ref=%d got=%d\n",
                         c, i, ref[c][i], got[c][i]);
            return 1;
          }
        }
      }
    }
    std::printf("OK --identity (%zu components, %u×%u)\n", ref.size(), w[0], h[0]);
  }

  if (empty) {
    // Filter returns false for every precinct → uniform DC-only output.
    std::vector<std::vector<int32_t>> got;
    std::vector<uint32_t> gw, gh;
    auto drop_all = [](uint16_t, uint16_t, uint8_t, uint32_t) { return false; };
    if (decode(bytes, drop_all, got, gw, gh) != 0) return 1;
    for (std::size_t c = 0; c < got.size(); ++c) {
      if (got[c].empty()) continue;
      int32_t mn = got[c][0], mx = got[c][0];
      std::size_t mn_i = 0, mx_i = 0;
      for (std::size_t i = 1; i < got[c].size(); ++i) {
        if (got[c][i] < mn) { mn = got[c][i]; mn_i = i; }
        if (got[c][i] > mx) { mx = got[c][i]; mx_i = i; }
      }
      const int32_t range = mx - mn;
      if (range > empty_pae) {
        std::fprintf(stderr,
                     "FAIL --empty c=%zu: range %d (%d..%d) exceeds PAE cap %d "
                     "(min at i=%zu, max at i=%zu)\n",
                     c, range, mn, mx, empty_pae, mn_i, mx_i);
        return 1;
      }
    }
    std::printf("OK --empty (%zu components, PAE ≤ %d)\n", got.size(), empty_pae);
  }

  return 0;
}
