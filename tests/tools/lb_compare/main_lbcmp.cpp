// lb_compare: validate invoke_line_based() / invoke_line_based_stream() against
// invoke() for a given j2k file.
// Usage: lb_compare <input.j2k> [-reduce N] [--predecoded] [--stream]
// Exits 0 on exact match, non-zero on mismatch or error.
// --predecoded: use invoke_line_based_predecoded().
// --stream:     use invoke_line_based_stream() instead of invoke_line_based().
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include "decoder.hpp"

static bool parse_args(int argc, char *argv[], std::string &infile, uint8_t &reduce_NL,
                       bool &predecoded, bool &stream_mode) {
  infile      = "";
  reduce_NL   = 0;
  predecoded  = false;
  stream_mode = false;
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "-reduce") == 0 || strcmp(argv[i], "-r") == 0) {
      if (++i < argc) reduce_NL = static_cast<uint8_t>(atoi(argv[i]));
    } else if (strcmp(argv[i], "--predecoded") == 0) {
      predecoded = true;
    } else if (strcmp(argv[i], "--stream") == 0) {
      stream_mode = true;
    } else {
      infile = argv[i];
    }
  }
  return !infile.empty();
}

// Read entire file into a buffer.
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

int main(int argc, char *argv[]) {
  std::string infile;
  uint8_t     reduce_NL   = 0;
  bool        predecoded  = false;
  bool        stream_mode = false;
  if (!parse_args(argc, argv, infile, reduce_NL, predecoded, stream_mode)) {
    printf("Usage: lb_compare <input.j2k> [-reduce N] [--predecoded] [--stream]\n");
    return 1;
  }

  std::vector<uint8_t> codestream = read_file(infile.c_str());
  if (codestream.empty()) return 1;

  // ── Reference decode via invoke() ────────────────────────────────────────
  std::vector<int32_t *> ref_buf;
  std::vector<uint32_t>  ref_w, ref_h;
  std::vector<uint8_t>   ref_depth;
  std::vector<bool>      ref_signed;
  try {
    open_htj2k::openhtj2k_decoder dec_ref;
    dec_ref.init(codestream.data(), codestream.size(), reduce_NL, 1);
    dec_ref.parse();
    dec_ref.invoke(ref_buf, ref_w, ref_h, ref_depth, ref_signed);
  } catch (std::exception &e) {
    printf("ERROR invoke(): %s\n", e.what());
    return 1;
  }

  // ── Line-based decode ─────────────────────────────────────────────────────
  std::vector<int32_t *> lb_buf;
  std::vector<uint32_t>  lb_w, lb_h;
  std::vector<uint8_t>   lb_depth;
  std::vector<bool>      lb_signed;
  const char *lb_mode = stream_mode      ? "invoke_line_based_stream()"
                        : predecoded     ? "invoke_line_based_predecoded()"
                                         : "invoke_line_based()";
  try {
    open_htj2k::openhtj2k_decoder dec_lb;
    dec_lb.init(codestream.data(), codestream.size(), reduce_NL, 1);
    dec_lb.parse();
    if (stream_mode) {
      // Collect stream callback rows into flat per-component buffers.
      std::vector<std::vector<int32_t>> flat;
      auto stream_cb = [&](uint32_t y, int32_t *const *rows, uint16_t nc) {
        if (flat.empty()) {
          flat.resize(nc);
          for (uint16_t c = 0; c < nc; ++c)
            flat[c].assign(static_cast<size_t>(lb_w[c]) * lb_h[c], 0);
        }
        for (uint16_t c = 0; c < nc; ++c) {
          if (y < lb_h[c])
            std::memcpy(flat[c].data() + static_cast<size_t>(y) * lb_w[c], rows[c],
                        lb_w[c] * sizeof(int32_t));
        }
      };
      dec_lb.invoke_line_based_stream(stream_cb, lb_w, lb_h, lb_depth, lb_signed);
      lb_buf.resize(flat.size());
      for (size_t c = 0; c < flat.size(); ++c) {
        lb_buf[c] = new int32_t[flat[c].size()];
        std::memcpy(lb_buf[c], flat[c].data(), flat[c].size() * sizeof(int32_t));
      }
    } else if (predecoded) {
      dec_lb.invoke_line_based_predecoded(lb_buf, lb_w, lb_h, lb_depth, lb_signed);
    } else {
      dec_lb.invoke_line_based(lb_buf, lb_w, lb_h, lb_depth, lb_signed);
    }
  } catch (std::exception &e) {
    printf("ERROR %s: %s\n", lb_mode, e.what());
    return 1;
  }

  // ── Compare ───────────────────────────────────────────────────────────────
  const uint16_t NC = static_cast<uint16_t>(ref_buf.size());
  if (lb_buf.size() != NC) {
    printf("FAIL: component count mismatch (%zu vs %zu)\n", ref_buf.size(), lb_buf.size());
    return 1;
  }

  bool ok = true;
  for (uint16_t c = 0; c < NC && ok; ++c) {
    if (ref_w[c] != lb_w[c] || ref_h[c] != lb_h[c]) {
      printf("FAIL: comp %u dimension mismatch (%ux%u vs %ux%u)\n",
             c, ref_w[c], ref_h[c], lb_w[c], lb_h[c]);
      ok = false;
      break;
    }
    const size_t nsamp = static_cast<size_t>(ref_w[c]) * ref_h[c];
    for (size_t i = 0; i < nsamp; ++i) {
      if (ref_buf[c][i] != lb_buf[c][i]) {
        const uint32_t x = static_cast<uint32_t>(i % ref_w[c]);
        const uint32_t y = static_cast<uint32_t>(i / ref_w[c]);
        printf("FAIL [%s]: comp %u pixel (%u,%u): invoke=%d  %s=%d\n",
               infile.c_str(), c, x, y, ref_buf[c][i], lb_mode, lb_buf[c][i]);
        ok = false;
        break;
      }
    }
  }

  // Clean up
  for (auto *p : ref_buf) delete[] p;
  for (auto *p : lb_buf)  delete[] p;

  if (ok) {
    printf("PASS: %s (reduce=%u)\n", infile.c_str(), reduce_NL);
    return 0;
  }
  return 1;
}
