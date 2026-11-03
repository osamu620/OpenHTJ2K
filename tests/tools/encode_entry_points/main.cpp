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
// irreversible ones within a small tolerance.  Some cases also place the image
// at an offset on the reference grid and split it into tiles, which moves the
// subbands off the code-block grid.  The irreversible cases run again with a
// target codestream size (rate control), with the HT block coder and with the
// Part 1 one: each entry point's codestream must fit the target and decode.
// (They need not agree: the rate control estimates its threshold from the
// code-blocks in the order they are coded, which differs between the entry
// points.)  The offset and tile cases also run the Part 1 one in three quality
// layers (a quarter, a half and all of the target), and the three-component ones
// in three layers without a target, with both block coders.  Last, the file
// formats of set_file_format(): JP2 files of 1 to 4 components from both entry
// points must hold the raw Part 1 codestream after the expected boxes and decode
// to the input; JP2 must refuse HT code-blocks and Part 2 capabilities; JPH set
// that way must equal the constructor's JPH.  First of all, a small image with
// 20 and 30 to 32 DWT levels: the quantization step sizes once took memory
// doubling with every level, so the 20-level encode must stay small (where the
// peak memory can be read) before the deeper ones run.
//
// Usage: encode_entry_points [-v]
// Exits 0 when every case passes.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>
#include "decoder.hpp"
#include "encoder.hpp"
#if defined(__unix__) || defined(__APPLE__)
  #include <sys/resource.h>
#endif

namespace {

constexpr uint8_t NO_QFACTOR = 0xFF;

// Peak resident memory of this process so far in MiB, or -1 where it cannot be read.
double peak_rss_mib() {
#if defined(__unix__) || defined(__APPLE__)
  struct rusage ru;
  if (getrusage(RUSAGE_SELF, &ru) != 0) return -1.0;
  #if defined(__APPLE__)
  return static_cast<double>(ru.ru_maxrss) / (1024.0 * 1024.0);  // bytes
  #else
  return static_cast<double>(ru.ru_maxrss) / 1024.0;  // KiB
  #endif
#else
  return -1.0;
#endif
}

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

// Image offset on the reference grid and tile size (0: one tile).
struct Layout {
  uint32_t ox, oy, tw, th;
};

// part1: the Part 1 block coder instead of HT; layers: quality layers of a target size,
// at target / 2^(layers - 1 - l) bytes for layer l.
std::vector<uint8_t> encode(const Case &k, const std::vector<std::vector<int32_t>> &src, bool streaming,
                            const Layout &lay, uint64_t target, bool part1, uint16_t layers) {
  open_htj2k::siz_params siz;
  siz.Rsiz   = 0;
  siz.Xsiz   = k.width + lay.ox;
  siz.Ysiz   = k.height + lay.oy;
  siz.XOsiz  = lay.ox;
  siz.YOsiz  = lay.oy;
  siz.XTsiz  = lay.tw ? lay.tw : k.width + lay.ox;
  siz.YTsiz  = lay.th ? lay.th : k.height + lay.oy;
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
  cod.number_of_layers  = layers;
  cod.use_color_trafo   = k.mct;
  cod.dwt_levels        = k.levels;
  cod.codeblock_style   = part1 ? 0x000 : 0x040;  // Part 1 default mode or HT
  cod.transformation    = k.transformation;
  open_htj2k::qcd_params qcd;
  qcd.is_derived          = false;
  qcd.number_of_guardbits = 1;
  qcd.base_step           = 1.0 / static_cast<double>(1 << (k.depth + (target ? 3 : 0)));

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
  std::vector<uint64_t> targets;
  for (uint16_t l = 0; target && l < layers; ++l) targets.push_back(target >> (layers - 1 - l));
  enc.set_target_sizes(targets);
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

// --- File formats (set_file_format) -------------------------------------------

// A reversible encode of a 67x45 image whose components have the given Ssiz values
// (bit depth - 1, sign in bit 7).  style: 0x40 for the HT block coder, 0 for Part 1.
struct FileCase {
  const char *name;
  std::vector<uint8_t> ssiz;
  uint8_t style;
  uint8_t progression;  // 5: PRCL, an ISO/IEC 15444-2 order
  uint16_t rsiz;
  uint8_t color_space;  // 0: RGB, 1: YCC
};
constexpr uint32_t FILE_W = 67, FILE_H = 45;

std::vector<std::vector<int32_t>> file_planes(const FileCase &f) {
  std::vector<std::vector<int32_t>> planes(f.ssiz.size());
  for (size_t c = 0; c < planes.size(); ++c) {
    const uint32_t span = 1u << ((f.ssiz[c] & 0x7F) + 1);
    const int32_t lo    = (f.ssiz[c] & 0x80) ? -static_cast<int32_t>(span / 2) : 0;
    planes[c].resize(static_cast<size_t>(FILE_W) * FILE_H);
    for (uint32_t i = 0; i < FILE_W * FILE_H; ++i)
      planes[c][i] = lo + static_cast<int32_t>((i * 37u + static_cast<uint32_t>(c) * 101u) % span);
  }
  return planes;
}

// is_jph: the constructor's argument; format < 0 leaves set_file_format() uncalled.
std::vector<uint8_t> encode_file(const FileCase &f, const std::vector<std::vector<int32_t>> &src,
                                 bool streaming, bool is_jph, int format) {
  open_htj2k::siz_params siz;
  siz.Rsiz   = f.rsiz;
  siz.Xsiz   = FILE_W;
  siz.Ysiz   = FILE_H;
  siz.XOsiz  = 0;
  siz.YOsiz  = 0;
  siz.XTsiz  = FILE_W;
  siz.YTsiz  = FILE_H;
  siz.XTOsiz = 0;
  siz.YTOsiz = 0;
  siz.Csiz   = static_cast<uint16_t>(f.ssiz.size());
  siz.Ssiz   = f.ssiz;
  siz.XRsiz.assign(f.ssiz.size(), 1);
  siz.YRsiz.assign(f.ssiz.size(), 1);
  open_htj2k::cod_params cod;
  cod.blkwidth          = 4;
  cod.blkheight         = 4;
  cod.is_max_precincts  = true;
  cod.use_SOP           = false;
  cod.use_EPH           = false;
  cod.progression_order = f.progression;
  cod.number_of_layers  = 1;
  cod.use_color_trafo   = 0;
  cod.dwt_levels        = 3;
  cod.codeblock_style   = f.style;
  cod.transformation    = 1;
  open_htj2k::qcd_params qcd;
  qcd.is_derived          = false;
  qcd.number_of_guardbits = 1;
  qcd.base_step           = 1.0 / 256;

  std::vector<std::vector<int32_t>> planes = src;
  std::vector<int32_t *> input;
  if (!streaming) {
    for (auto &p : planes) input.push_back(p.data());
  }
  std::vector<uint8_t> out;
  open_htj2k::openhtj2k_encoder enc("", input, siz, cod, qcd, NO_QFACTOR, is_jph, f.color_space, 1);
  enc.set_output_buffer(out);
  if (format >= 0) enc.set_file_format(static_cast<open_htj2k::file_format>(format));
  if (streaming) {
    enc.invoke_line_based_stream([&planes](uint32_t y, int32_t **rows, uint16_t nc) {
      for (uint16_t c = 0; c < nc; ++c)
        memcpy(rows[c], planes[c].data() + static_cast<size_t>(y) * FILE_W, sizeof(int32_t) * FILE_W);
    });
  } else {
    enc.invoke_line_based();
  }
  return out;
}

uint32_t be32(const uint8_t *p) {
  return (uint32_t{p[0]} << 24) | (uint32_t{p[1]} << 16) | (uint32_t{p[2]} << 8) | p[3];
}

struct Box {
  std::string type;
  std::vector<uint8_t> payload;
};

// The boxes of [p, end); false if a length is out of range.
bool split_boxes(const uint8_t *p, const uint8_t *end, std::vector<Box> &boxes) {
  while (p < end) {
    if (end - p < 8) return false;
    const uint32_t len = be32(p);
    if (len < 8 || len > static_cast<size_t>(end - p)) return false;
    boxes.push_back(
        Box{std::string(reinterpret_cast<const char *>(p) + 4, 4), std::vector<uint8_t>(p + 8, p + len)});
    p += len;
  }
  return true;
}

// Checks a JP2 (brand "jp2 ") or JPH (brand "jph ") file against the case and the raw
// codestream; returns an error message, empty if none.
std::string check_file(const FileCase &f, const std::vector<uint8_t> &file, const std::vector<uint8_t> &raw,
                       const char *brand) {
  std::vector<Box> top, hdr;
  if (!split_boxes(file.data(), file.data() + file.size(), top)) return "malformed box lengths";
  if (top.size() != 4 || top[0].type != "jP  " || top[1].type != "ftyp" || top[2].type != "jp2h"
      || top[3].type != "jp2c")
    return "boxes are not signature, file type, JP2 header, contiguous codestream";
  const std::vector<uint8_t> sig = {0x0D, 0x0A, 0x87, 0x0A};
  if (top[0].payload != sig) return "wrong signature";
  // Brand, MinV 0, and the brand as the one compatible brand.
  std::vector<uint8_t> ftyp(brand, brand + 4);
  ftyp.insert(ftyp.end(), 4, 0);
  ftyp.insert(ftyp.end(), brand, brand + 4);
  if (top[1].payload != ftyp) return "wrong file type box";
  if (top[3].payload != raw) return "contiguous codestream box does not hold the raw codestream";
  if (!split_boxes(top[2].payload.data(), top[2].payload.data() + top[2].payload.size(), hdr))
    return "malformed JP2 header box";
  const size_t nc = f.ssiz.size();
  bool same       = true;
  for (size_t c = 1; c < nc; ++c) same = same && f.ssiz[c] == f.ssiz[0];
  const uint8_t bpc   = same ? f.ssiz[0] : 0xFF;
  const bool has_cdef = (nc == 2 || nc == 4);
  const size_t want   = 2u + (same ? 0u : 1u) + (has_cdef ? 1u : 0u);
  if (hdr.size() != want) return "JP2 header box holds " + std::to_string(hdr.size()) + " boxes";
  size_t i              = 0;
  const auto &ihdr      = hdr[i++].payload;
  const uint8_t ihdr_[] = {0, 0, 0, FILE_H, 0, 0, 0, FILE_W, 0, static_cast<uint8_t>(nc), bpc, 7, 0, 0};
  if (hdr[0].type != "ihdr" || ihdr != std::vector<uint8_t>(ihdr_, ihdr_ + sizeof(ihdr_)))
    return "wrong image header box";
  if (!same && (hdr[i].type != "bpcc" || hdr[i++].payload != f.ssiz)) return "wrong bits per component box";
  const uint8_t colr_[] = {
      1, 0, 0, 0, 0, 0, static_cast<uint8_t>((nc <= 2) ? 17 : f.color_space ? 18 : 16)};
  if (hdr[i].type != "colr" || hdr[i++].payload != std::vector<uint8_t>(colr_, colr_ + sizeof(colr_)))
    return "wrong colour specification box";
  if (has_cdef) {
    // N, then per component Cn = c, Typ and Asoc: the last component is opacity (Typ 1)
    // of the whole image (Asoc 0), the others colours (Typ 0) c + 1.  16 bits each.
    std::vector<uint8_t> cdef = {0, static_cast<uint8_t>(nc)};
    for (size_t c = 0; c < nc; ++c) {
      const bool alpha      = (c == nc - 1);
      const uint8_t entry[] = {0, static_cast<uint8_t>(c),
                               0, static_cast<uint8_t>(alpha ? 1 : 0),
                               0, static_cast<uint8_t>(alpha ? 0 : c + 1)};
      cdef.insert(cdef.end(), entry, entry + sizeof(entry));
    }
    if (hdr[i].type != "cdef" || hdr[i].payload != cdef) return "wrong channel definition box";
  }
  return "";
}

// Decodes a file and compares it with the input; returns an error message, empty if none.
std::string check_decode(const FileCase &f, const std::vector<uint8_t> &file,
                         const std::vector<std::vector<int32_t>> &ref) {
  open_htj2k::openhtj2k_decoder dec(file.data(), file.size(), 0, 1);
  std::vector<int32_t *> buf;
  std::vector<uint32_t> w, h;
  std::vector<uint8_t> depth;
  std::vector<bool> is_signed;
  dec.parse();
  const uint32_t enum_cs = dec.get_colorspace();
  dec.invoke_line_based(buf, w, h, depth, is_signed);
  std::string err;
  const uint32_t want_cs = (f.ssiz.size() <= 2) ? 17 : f.color_space ? 18 : 16;
  if (enum_cs != want_cs) err = "decoder reads EnumCS " + std::to_string(enum_cs);
  if (err.empty() && buf.size() != f.ssiz.size()) err = "decoded component count differs";
  for (size_t c = 0; err.empty() && c < buf.size(); ++c) {
    if (w[c] != FILE_W || h[c] != FILE_H || depth[c] != (f.ssiz[c] & 0x7F) + 1
        || is_signed[c] != ((f.ssiz[c] & 0x80) != 0))
      err = "decoded component " + std::to_string(c) + " has another shape or depth";
    else if (memcmp(buf[c], ref[c].data(), sizeof(int32_t) * ref[c].size()) != 0)
      err = "decoded component " + std::to_string(c) + " differs from the input";
  }
  for (auto p : buf) delete[] p;
  return err;
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
  // 14 and 16 bits exceed FRACBITS (13): the irreversible input is scaled in
  // float instead of shifted, and the two entry points must still agree.
  const Depth depths[] = {{8, false}, {12, true}, {14, true}, {16, false}};

  int total = 0, failed = 0;
  // target: codestream bytes for rate control, 0 for none; part1: the Part 1 block coder;
  // layers: quality layers (with a target).
  auto run = [&](const Case &k, const Layout &lay, uint64_t target, bool part1, uint16_t layers = 1) {
    ++total;
    std::string err;
    try {
      const auto planes   = make_planes(k);
      const auto buffered = encode(k, planes, false, lay, target, part1, layers);
      const auto stream   = encode(k, planes, true, lay, target, part1, layers);
      const long e        = max_error(k, buffered, planes);
      // 9/7 (with or without the ICT) at a step of 1/2^depth stays within a
      // few code values; the byte comparison above is the main check.  A target
      // size makes the error depend on it: only the decode is checked.
      const long tol = target ? 0x7FFFFFFF : k.transformation ? 0 : 8;
      const long es  = target ? max_error(k, stream, planes) : 0;
      if (!target && buffered != stream) {
        err = "buffered and streaming codestreams differ";
      } else if (target && std::max(buffered.size(), stream.size()) > target) {
        err = "codestream of " + std::to_string(std::max(buffered.size(), stream.size()))
              + " bytes exceeds the target " + std::to_string(target);
      } else if (e < 0 || es < 0) {
        err = "decoded shape does not match the input";
      } else if (e > tol) {
        err = "decoded samples off by " + std::to_string(e) + " (tolerance " + std::to_string(tol) + ")";
      }
    } catch (std::exception &ex) {
      err = std::string("exception: ") + ex.what();
    }
    if (!err.empty()) {
      ++failed;
      printf("FAIL %s offset %u,%u tiles %ux%u%s%s, %u layer(s): %s\n", k.name().c_str(), lay.ox, lay.oy,
             lay.tw, lay.th, part1 ? " Part 1" : "", target ? " (target size)" : "", layers, err.c_str());
    } else if (verbose) {
      printf("ok   %s offset %u,%u tiles %ux%u%s%s, %u layer(s)\n", k.name().c_str(), lay.ox, lay.oy,
             lay.tw, lay.th, part1 ? " Part 1" : "", target ? " (target size)" : "", layers);
    }
  };

  // Deep decompositions, run first while the peak memory is still low.  The step
  // sizes once took memory doubling with every DWT level (about 0.5 GB at 20
  // levels, all of a 91 GB machine before 32), so the 30- to 32-level cases run
  // only when the 20-level encodes stayed small, or where the peak memory cannot
  // be read.
  // The 5-level encodes first take the memory any encode does (per-thread scratch).
  {
    for (uint8_t xf = 0; xf < 2; ++xf)
      run(Case{61, 47, 1, 0, xf, 5, 8, false}, Layout{0, 0, 0, 0}, 0, false);
    const double before = peak_rss_mib();
    for (uint8_t xf = 0; xf < 2; ++xf)
      run(Case{61, 47, 1, 0, xf, 20, 8, false}, Layout{0, 0, 0, 0}, 0, false);
    const double grown = peak_rss_mib() - before;
    if (verbose && before >= 0) printf("     20 DWT levels: peak memory grew by %.1f MiB\n", grown);
    if (before >= 0 && grown > 64.0) {
      ++total;
      ++failed;
      printf(
          "FAIL 20 DWT levels: peak memory grew by %.0f MiB (at most 64 expected); "
          "30 to 32 levels skipped\n",
          grown);
    } else {
      // Up to the 32 levels the standard allows: 31 and 32 index the per-resolution
      // arrays at their ends, and 32 levels divide the lowest resolution by 2^32.
      const uint8_t deep_levels[] = {30, 31, 32};
      for (uint8_t lv : deep_levels)
        for (auto &cp : {Comp{1, 0}, Comp{3, 1}})
          for (uint8_t xf = 0; xf < 2; ++xf)
            run(Case{61, 47, cp.nc, cp.mct, xf, lv, 8, false}, Layout{0, 0, 0, 0}, 0, false);
    }
  }

  for (auto &sz : sizes)
    for (auto &cp : comps)
      for (uint8_t xf = 0; xf < 2; ++xf)
        for (uint8_t lv : lvls)
          for (auto &dp : depths) {
            const Case k{sz[0], sz[1], cp.nc, cp.mct, xf, lv, dp.bits, dp.is_signed};
            run(k, Layout{0, 0, 0, 0}, 0, false);
            // One bit per sample, above what the headers take.
            const uint64_t target = std::max<uint64_t>(600, uint64_t{sz[0]} * sz[1] * cp.nc / 8);
            if (xf == 0) {
              run(k, Layout{0, 0, 0, 0}, target, false);
              run(k, Layout{0, 0, 0, 0}, target, true);
            }
          }

  // Images smaller than the DWT levels need: the coarsest levels then have
  // empty subbands (no code-blocks at all), which the encoder must handle.
  const uint32_t tiny_sizes[][2] = {{1, 1}, {2, 2}, {3, 5}, {1, 9}, {9, 1}, {17, 33}};
  const uint8_t tiny_lvls[]      = {1, 2, 5, 7};
  const Comp tiny_comps[]        = {{1, 0}, {3, 1}};
  for (auto &sz : tiny_sizes)
    for (auto &cp : tiny_comps)
      for (uint8_t xf = 0; xf < 2; ++xf)
        for (uint8_t lv : tiny_lvls) {
          const Case k{sz[0], sz[1], cp.nc, cp.mct, xf, lv, 8, false};
          run(k, Layout{0, 0, 0, 0}, 0, false);
          if (xf == 0) {
            run(k, Layout{0, 0, 0, 0}, 2000, false);
            run(k, Layout{0, 0, 0, 0}, 2000, true);
          }
        }

  // Offsets and tiles that start subbands on odd rows or between code-block rows.
  const Layout layouts[] = {{3, 0, 0, 0},   {0, 1, 0, 0},     {0, 2, 0, 0},     {5, 3, 100, 77},
                            {0, 0, 96, 96}, {33, 17, 64, 64}, {0, 0, 300, 200}, {0, 0, 768, 136}};
  const Comp lay_comps[] = {{1, 0}, {3, 1}};
  for (auto &lay : layouts)
    for (auto &cp : lay_comps)
      for (uint8_t xf = 0; xf < 2; ++xf) {
        const Case k{517, 383, cp.nc, cp.mct, xf, 5, 8, false};
        run(k, lay, 0, false);
        if (xf == 0) {
          run(k, lay, uint64_t{k.width} * k.height * k.nc / 8, false);
          run(k, lay, uint64_t{k.width} * k.height * k.nc / 8, true);
          run(k, lay, uint64_t{k.width} * k.height * k.nc / 8, true, 3);
        }
        // Three quality layers without a target: all the data goes in the first one.
        if (cp.nc == 3) {
          run(k, lay, 0, false, 3);
          run(k, lay, 0, true, 3);
        }
      }

  // File formats.  JP2: the raw Part 1 codestream after the boxes, from both entry points,
  // decoding to the input; components that differ in depth or signedness need a Bits Per
  // Component box.  JPH set with set_file_format() is the constructor's JPH.
  auto file_case = [&](const std::string &what, const std::string &err) {
    ++total;
    if (!err.empty()) {
      ++failed;
      printf("FAIL file format: %s: %s\n", what.c_str(), err.c_str());
    } else if (verbose) {
      printf("ok   file format: %s\n", what.c_str());
    }
  };
  const int CODESTREAM       = static_cast<int>(open_htj2k::file_format::codestream);
  const int JPH              = static_cast<int>(open_htj2k::file_format::jph);
  const int JP2              = static_cast<int>(open_htj2k::file_format::jp2);
  const FileCase jp2_cases[] = {
      {"gray", {7}, 0, 0, 0, 0},
      {"gray + alpha", {7, 7}, 0, 0, 0, 0},
      {"RGB", {7, 7, 7}, 0, 0, 0, 0},
      {"YCbCr", {7, 7, 7}, 0, 0, 0, 1},
      {"RGB + alpha", {7, 7, 7, 7}, 0, 0, 0, 0},
      {"16-bit signed RGB", {0x8F, 0x8F, 0x8F}, 0, 0, 0, 0},
      {"12-bit gray + 8-bit alpha", {11, 7}, 0, 0, 0, 0},
      {"unsigned and signed 8-bit", {7, 0x87, 7}, 0, 0, 0, 0},
  };
  for (const auto &f : jp2_cases) {
    std::string err;
    try {
      const auto planes   = file_planes(f);
      const auto raw      = encode_file(f, planes, false, false, -1);
      const auto buffered = encode_file(f, planes, false, false, JP2);
      const auto stream   = encode_file(f, planes, true, false, JP2);
      if (buffered != stream)
        err = "buffered and streaming files differ";
      else
        err = check_file(f, buffered, raw, "jp2 ");
      if (err.empty()) err = check_decode(f, buffered, planes);
    } catch (std::exception &ex) {
      err = std::string("exception: ") + ex.what();
    }
    file_case(std::string("JP2 ") + f.name, err);
  }
  {
    const FileCase f = {"HT RGB + alpha", {7, 7, 7, 7}, 0x40, 0, 0, 0};
    std::string err;
    try {
      const auto planes = file_planes(f);
      const auto raw    = encode_file(f, planes, false, false, -1);
      const auto jph    = encode_file(f, planes, false, true, -1);
      if (encode_file(f, planes, false, false, JPH) != jph
          || encode_file(f, planes, true, false, JPH) != jph)
        err = "set_file_format(jph) differs from the constructor's JPH";
      else if (encode_file(f, planes, false, true, CODESTREAM) != raw)
        err = "set_file_format(codestream) does not override the constructor's JPH";
      else
        err = check_file(f, jph, raw, "jph ");
      if (err.empty()) err = check_decode(f, jph, planes);
    } catch (std::exception &ex) {
      err = std::string("exception: ") + ex.what();
    }
    file_case(std::string("JPH ") + f.name, err);
  }
  // A JP2 file refuses a codestream that is not Part 1, from both entry points.
  const FileCase refused[] = {
      {"HT block coder", {7, 7, 7}, 0x40, 0, 0, 0},
      {"PRCL progression order", {7, 7, 7}, 0, 5, 0, 0},
      {"Rsiz with the Part 2 bit", {7, 7, 7}, 0, 0, 0x8000, 0},
      {"Rsiz with the CAP bit", {7, 7, 7}, 0, 0, 0x4000, 0},
  };
  for (const auto &f : refused) {
    const auto planes = file_planes(f);
    for (int streaming = 0; streaming < 2; ++streaming) {
      std::string err;
      try {
        encode_file(f, planes, streaming != 0, false, JP2);
        err = "not refused";
      } catch (std::exception &) {
      }
      file_case(std::string("JP2 refuses ") + f.name + (streaming ? " (streaming)" : " (buffered)"), err);
    }
  }

  printf("encode_entry_points: %d cases, %d failed\n", total, failed);
  return failed == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
