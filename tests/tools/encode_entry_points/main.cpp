// encode_entry_points: the two encoder entry points must accept the same
// images and agree with each other.
//
//   invoke_line_based()         buffered: the caller hands in whole planes
//   invoke_line_based_stream()  streaming: rows are pulled through a callback
//
// The command-line encoder only uses the streaming entry point, so the
// buffered one is otherwise exercised only by library users.  For every case
// in a matrix of component count, MCT on/off, reversible 5/3 / irreversible
// 9/7, DWT levels, bit depth / signedness and (odd) image size, this tool
// encodes through both entry points, requires byte-identical codestreams, and
// decodes the result: reversible cases must reproduce the input exactly,
// irreversible ones within a small tolerance.
//
// Usage: encode_entry_points [-v]
// Exits 0 when every case passes.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>
#include "decoder.hpp"
#include "encoder.hpp"

namespace {

constexpr uint8_t NO_QFACTOR = 0xFF;

struct Case {
  uint32_t width, height;
  uint16_t nc;
  uint8_t mct;
  uint8_t transformation;  // 1 = reversible 5/3, 0 = irreversible 9/7
  uint8_t levels;
  uint8_t depth;
  bool is_signed;
  std::string name() const {
    char s[128];
    snprintf(s, sizeof(s), "%ux%u nc=%u mct=%u %s levels=%u %u-bit %s", width, height, nc, mct,
             transformation ? "rev53" : "irv97", levels, depth, is_signed ? "signed" : "unsigned");
    return s;
  }
};

std::vector<std::vector<int32_t>> make_planes(const Case &k) {
  std::vector<std::vector<int32_t>> planes(k.nc);
  const int32_t lo   = k.is_signed ? -(1 << (k.depth - 1)) : 0;
  const int32_t span = 1 << k.depth;
  uint32_t s         = 2463534242u;
  for (uint16_t c = 0; c < k.nc; ++c) {
    planes[c].resize(static_cast<size_t>(k.width) * k.height);
    for (uint32_t y = 0; y < k.height; ++y) {
      for (uint32_t x = 0; x < k.width; ++x) {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        // smooth ramp plus a little noise, kept inside the sample range
        const int32_t ramp =
            static_cast<int32_t>(((x + 2 * y + 37 * c) * 5u) % static_cast<uint32_t>(span));
        const int32_t noise = static_cast<int32_t>(s % 9u) - 4;
        int32_t v           = ramp + noise;
        if (v < 0) v = 0;
        if (v >= span) v = span - 1;
        planes[c][static_cast<size_t>(y) * k.width + x] = v + lo;
      }
    }
  }
  return planes;
}

std::vector<uint8_t> encode(const Case &k, const std::vector<std::vector<int32_t>> &src, bool streaming) {
  open_htj2k::siz_params siz;
  siz.Rsiz   = 0;
  siz.Xsiz   = k.width;
  siz.Ysiz   = k.height;
  siz.XOsiz  = 0;
  siz.YOsiz  = 0;
  siz.XTsiz  = k.width;
  siz.YTsiz  = k.height;
  siz.XTOsiz = 0;
  siz.YTOsiz = 0;
  siz.Csiz   = k.nc;
  for (uint16_t c = 0; c < k.nc; ++c) {
    siz.Ssiz.push_back(static_cast<uint8_t>((k.depth - 1) | (k.is_signed ? 0x80 : 0)));
    siz.XRsiz.push_back(1);
    siz.YRsiz.push_back(1);
  }
  open_htj2k::cod_params cod;
  cod.blkwidth          = 4;  // 64x64 code-blocks (log2 - 2)
  cod.blkheight         = 4;
  cod.is_max_precincts  = true;
  cod.use_SOP           = false;
  cod.use_EPH           = false;
  cod.progression_order = 0;
  cod.number_of_layers  = 1;
  cod.use_color_trafo   = k.mct;
  cod.dwt_levels        = k.levels;
  cod.codeblock_style   = 0x040;  // HT
  cod.transformation    = k.transformation;
  open_htj2k::qcd_params qcd;
  qcd.is_derived          = false;
  qcd.number_of_guardbits = 1;
  qcd.base_step           = 1.0 / static_cast<double>(1 << k.depth);

  // Private copy: the comparison must not depend on whether the encoder
  // leaves its input untouched.
  std::vector<std::vector<int32_t>> planes = src;
  std::vector<int32_t *> input;
  if (!streaming) {
    for (auto &p : planes) input.push_back(p.data());
  }
  std::vector<uint8_t> out;
  open_htj2k::openhtj2k_encoder enc("", input, siz, cod, qcd, NO_QFACTOR, false, 0, 1);
  enc.set_output_buffer(out);
  if (streaming) {
    enc.invoke_line_based_stream([&planes, &k](uint32_t y, int32_t **rows, uint16_t nc) {
      for (uint16_t c = 0; c < nc; ++c)
        memcpy(rows[c], planes[c].data() + static_cast<size_t>(y) * k.width, sizeof(int32_t) * k.width);
    });
  } else {
    enc.invoke_line_based();
  }
  return out;
}

// Largest absolute sample error after decoding, or -1 on a shape mismatch.
long max_error(const Case &k, const std::vector<uint8_t> &j2c,
               const std::vector<std::vector<int32_t>> &ref) {
  open_htj2k::openhtj2k_decoder dec(j2c.data(), j2c.size(), 0, 1);
  std::vector<int32_t *> buf;
  std::vector<uint32_t> w, h;
  std::vector<uint8_t> depth;
  std::vector<bool> is_signed;
  dec.parse();
  dec.invoke_line_based(buf, w, h, depth, is_signed);
  long worst = 0;
  if (buf.size() != k.nc) worst = -1;
  for (size_t c = 0; worst >= 0 && c < buf.size(); ++c) {
    if (w[c] != k.width || h[c] != k.height) {
      worst = -1;
      break;
    }
    for (size_t i = 0; i < ref[c].size(); ++i) {
      const long d = std::labs(static_cast<long>(buf[c][i]) - static_cast<long>(ref[c][i]));
      if (d > worst) worst = d;
    }
  }
  for (auto p : buf) delete[] p;
  return worst;
}

}  // namespace

int main(int argc, char *argv[]) {
  const bool verbose        = argc > 1 && strcmp(argv[1], "-v") == 0;
  const uint32_t sizes[][2] = {{64, 126}, {130, 130}, {517, 383}};
  struct Comp {
    uint16_t nc;
    uint8_t mct;
  };
  const Comp comps[]   = {{1, 0}, {3, 0}, {3, 1}, {4, 0}, {4, 1}};
  const uint8_t lvls[] = {0, 1, 5};
  struct Depth {
    uint8_t bits;
    bool is_signed;
  };
  const Depth depths[] = {{8, false}, {12, true}};

  int total = 0, failed = 0;
  for (auto &sz : sizes)
    for (auto &cp : comps)
      for (uint8_t xf = 0; xf < 2; ++xf)
        for (uint8_t lv : lvls)
          for (auto &dp : depths) {
            const Case k{sz[0], sz[1], cp.nc, cp.mct, xf, lv, dp.bits, dp.is_signed};
            ++total;
            std::string err;
            try {
              const auto planes   = make_planes(k);
              const auto buffered = encode(k, planes, false);
              const auto stream   = encode(k, planes, true);
              const long e        = max_error(k, buffered, planes);
              // 9/7 (with or without the ICT) at a step of 1/2^depth stays within a
              // few code values; the byte comparison above is the main check.
              const long tol = k.transformation ? 0 : 8;
              if (buffered != stream) {
                err = "buffered and streaming codestreams differ";
              } else if (e < 0) {
                err = "decoded shape does not match the input";
              } else if (e > tol) {
                err = "decoded samples off by " + std::to_string(e) + " (tolerance " + std::to_string(tol)
                      + ")";
              }
            } catch (std::exception &ex) {
              err = std::string("exception: ") + ex.what();
            }
            if (!err.empty()) {
              ++failed;
              printf("FAIL %s: %s\n", k.name().c_str(), err.c_str());
            } else if (verbose) {
              printf("ok   %s\n", k.name().c_str());
            }
          }
  printf("encode_entry_points: %d cases, %d failed\n", total, failed);
  return failed == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
