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
// peak memory can be read) before the deeper ones run.  Last, parameter edge cases:
// subsampled components with odd sizes, origins and tiles; 1-bit samples at 6 levels; more
// guard bits than a low bit depth needs; 300 quality layers; tile size 0 with an image
// origin; 23-bit reversible samples; Qfactor 0; 30 magnitude bit-planes; derived
// quantization at 20 and 29 levels; rate control at 24 levels within 64 MiB; Rsiz with PRCL;
// RPCL and PCRL refused for subsampling factors that are not powers of two (CPRL takes any);
// and out-of-range parameters, which both entry points must refuse with an exception.
//
// Usage: encode_entry_points [-v]
// Exits 0 when every case passes.
#include <algorithm>
#include <cmath>
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

// --- Parameter edge cases -------------------------------------------------------

// A whole encode: geometry, coding and quantization parameters.  Defaults: a 67x45 8-bit
// grey image, reversible, HT, three DWT levels.
struct Spec {
  uint32_t w = 67, h = 45, ox = 0, oy = 0, tw = 0, th = 0, tox = 0, toy = 0;
  std::vector<uint8_t> ssiz = {7};  // per component: bit depth - 1, sign in bit 7
  std::vector<uint8_t> xr = {1}, yr = {1};
  uint8_t levels = 3, xf = 1, mct = 0, style = 0x40, order = 0, guard = 1, bw = 4, bh = 4;
  uint16_t layers = 1;
  bool derived = false, maxprec = true;
  std::vector<uint8_t> ppx, ppy;
  double step     = 1.0 / 256;
  uint8_t qfactor = NO_QFACTOR;
  std::vector<uint64_t> targets;
  bool checkerboard = false;  // samples at the extremes of the range, instead of a ramp
  uint16_t nc() const { return static_cast<uint16_t>(ssiz.size()); }
  // Component size: ceil((origin + size) / factor) - ceil(origin / factor).
  uint32_t cw(uint16_t c) const { return (ox + w + xr[c] - 1) / xr[c] - (ox + xr[c] - 1) / xr[c]; }
  uint32_t ch(uint16_t c) const { return (oy + h + yr[c] - 1) / yr[c] - (oy + yr[c] - 1) / yr[c]; }
};

std::vector<std::vector<int32_t>> spec_planes(const Spec &k) {
  std::vector<std::vector<int32_t>> planes(k.nc());
  uint32_t s = 2463534242u;
  for (uint16_t c = 0; c < k.nc(); ++c) {
    if (c >= k.xr.size() || c >= k.yr.size() || k.xr[c] == 0 || k.yr[c] == 0) continue;
    const int64_t span = int64_t{1} << ((k.ssiz[c] & 0x7F) + 1);
    const int64_t lo   = (k.ssiz[c] & 0x80) ? -span / 2 : 0;
    planes[c].resize(static_cast<size_t>(k.cw(c)) * k.ch(c));
    for (uint32_t y = 0; y < k.ch(c); ++y)
      for (uint32_t x = 0; x < k.cw(c); ++x) {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        const int64_t v                                 = k.checkerboard ? (((x + y) & 1) ? span - 1 : 0)
                                                                         : static_cast<int64_t>((uint64_t{x + 2 * y + 37 * c} * 5u + s % 9u)
                                                                                                % static_cast<uint64_t>(span));
        planes[c][static_cast<size_t>(y) * k.cw(c) + x] = static_cast<int32_t>(v + lo);
      }
  }
  return planes;
}

// Throws whatever the encoder throws.
std::vector<uint8_t> spec_encode(const Spec &k, const std::vector<std::vector<int32_t>> &src,
                                 bool streaming) {
  open_htj2k::siz_params siz;
  siz.Rsiz   = 0;
  siz.Xsiz   = k.w + k.ox;
  siz.Ysiz   = k.h + k.oy;
  siz.XOsiz  = k.ox;
  siz.YOsiz  = k.oy;
  siz.XTsiz  = k.tw;
  siz.YTsiz  = k.th;
  siz.XTOsiz = k.tox;
  siz.YTOsiz = k.toy;
  siz.Csiz   = k.nc();
  siz.Ssiz   = k.ssiz;
  siz.XRsiz  = k.xr;
  siz.YRsiz  = k.yr;
  open_htj2k::cod_params cod;
  cod.blkwidth          = k.bw;
  cod.blkheight         = k.bh;
  cod.is_max_precincts  = k.maxprec;
  cod.use_SOP           = false;
  cod.use_EPH           = false;
  cod.progression_order = k.order;
  cod.number_of_layers  = k.layers;
  cod.use_color_trafo   = k.mct;
  cod.dwt_levels        = k.levels;
  cod.codeblock_style   = k.style;
  cod.transformation    = k.xf;
  cod.PPx               = k.ppx;
  cod.PPy               = k.ppy;
  open_htj2k::qcd_params qcd;
  qcd.is_derived                           = k.derived;
  qcd.number_of_guardbits                  = k.guard;
  qcd.base_step                            = k.step;
  std::vector<std::vector<int32_t>> planes = src;
  std::vector<int32_t *> input;
  if (!streaming)
    for (auto &p : planes) input.push_back(p.data());
  std::vector<uint8_t> out;
  open_htj2k::openhtj2k_encoder enc("", input, siz, cod, qcd, k.qfactor, false, 0, 1);
  enc.set_output_buffer(out);
  enc.set_target_sizes(k.targets);
  if (streaming) {
    // Component c takes its row y / (YRsiz[c] / YRsiz[0]), as the command line's readers give it.
    enc.invoke_line_based_stream([&](uint32_t y, int32_t **rows, uint16_t nc) {
      for (uint16_t c = 0; c < nc; ++c) {
        if (k.cw(c) == 0 || k.ch(c) == 0) continue;
        const uint32_t cy = std::min<uint32_t>(y / (k.yr[c] / k.yr[0]), k.ch(c) - 1);
        memcpy(rows[c], planes[c].data() + static_cast<size_t>(cy) * k.cw(c), sizeof(int32_t) * k.cw(c));
      }
    });
  } else {
    enc.invoke_line_based();
  }
  return out;
}

// Largest sample error of the decoded codestream, or -1 if its shape differs.
long spec_error(const Spec &k, const std::vector<uint8_t> &j2c,
                const std::vector<std::vector<int32_t>> &ref) {
  open_htj2k::openhtj2k_decoder dec(j2c.data(), j2c.size(), 0, 1);
  std::vector<int32_t *> buf;
  std::vector<uint32_t> w, h;
  std::vector<uint8_t> depth;
  std::vector<bool> sg;
  dec.parse();
  dec.invoke_line_based(buf, w, h, depth, sg);
  long worst = (buf.size() == k.nc()) ? 0 : -1;
  for (size_t c = 0; worst >= 0 && c < buf.size(); ++c) {
    const uint16_t cc = static_cast<uint16_t>(c);
    if (w[c] != k.cw(cc) || h[c] != k.ch(cc)) {
      worst = -1;
      break;
    }
    for (size_t i = 0; i < ref[c].size(); ++i)
      worst = std::max(worst, std::labs(static_cast<long>(buf[c][i]) - static_cast<long>(ref[c][i])));
  }
  for (auto p : buf) delete[] p;
  return worst;
}

// Rsiz of a codestream (the SIZ marker segment follows SOC).
uint16_t rsiz_of(const std::vector<uint8_t> &j2c) {
  return (j2c.size() < 8) ? 0 : static_cast<uint16_t>((j2c[6] << 8) | j2c[7]);
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

  // Parameter edge cases.  Each must encode through both entry points to the same bytes and
  // decode within `tol` (0: lossless), or, if `refused`, be refused by both with an exception.
  auto edge = [&](const char *what, const Spec &k, long tol, bool refused, bool same_bytes) {
    ++total;
    std::string err;
    const auto planes = spec_planes(k);
    for (int streaming = 0; streaming < 2 && err.empty(); ++streaming) {
      try {
        const auto j2c = spec_encode(k, planes, streaming != 0);
        if (refused) {
          err = streaming ? "the streaming entry point does not refuse it"
                          : "the buffered entry point does not refuse it";
        } else {
          const long e = spec_error(k, j2c, planes);
          if (e < 0 || e > tol)
            err = std::string(streaming ? "streaming" : "buffered") + ": decoded samples off by "
                  + std::to_string(e);
          if (err.empty() && streaming && same_bytes && j2c != spec_encode(k, planes, false))
            err = "buffered and streaming codestreams differ";
        }
      } catch (std::exception &ex) {
        if (!refused) err = std::string("exception: ") + ex.what();
      }
    }
    if (!err.empty()) {
      ++failed;
      printf("FAIL edge case: %s: %s\n", what, err.c_str());
    } else if (verbose) {
      printf("ok   edge case: %s\n", what);
    }
  };
  auto check = [&](const char *what, bool ok) {
    ++total;
    if (!ok) {
      ++failed;
      printf("FAIL edge case: %s\n", what);
    } else if (verbose) {
      printf("ok   edge case: %s\n", what);
    }
  };
  {
    // Subsampled components with odd sizes, an image origin and tiles: a component's plane
    // starts at ceil(origin / factor) and is ceil((origin + size) / factor) less that wide; the
    // streaming entry point takes a component's row where component 0's row is on its grid.
    struct Sub {
      uint32_t w, h, ox, oy, tw, th, tox, toy;
    };
    const Sub subs[]             = {{64, 64, 0, 0, 0, 0, 0, 0},
                                    {65, 47, 0, 0, 0, 0, 0, 0},
                                    {65, 47, 3, 1, 0, 0, 3, 1},
                                    {130, 75, 5, 3, 40, 17, 0, 0},
                                    {97, 61, 1, 2, 33, 7, 1, 2}};
    const uint8_t factors[][2]   = {{2, 2}, {2, 1}, {1, 2}, {3, 4}, {4, 3}};
    const uint8_t block_styles[] = {0x40, 0x00};
    for (const auto &g : subs)
      for (const auto &f : factors)
        for (uint8_t style : block_styles) {
          Spec k;
          k.w = g.w, k.h = g.h, k.ox = g.ox, k.oy = g.oy;
          k.tw = g.tw, k.th = g.th, k.tox = g.tox, k.toy = g.toy;
          k.ssiz  = {7, 7, 7};
          k.xr    = {1, f[0], f[1]};
          k.yr    = {1, f[1], f[0]};
          k.style = style;
          char what[160];
          snprintf(what, sizeof(what), "%ux%u at %u,%u, tiles %ux%u, subsampling %ux%u and %ux%u%s", g.w,
                   g.h, g.ox, g.oy, g.tw, g.th, f[0], f[1], f[1], f[0], style ? "" : ", Part 1");
          edge(what, k, 0, false, true);
        }
    // Component 0 subsampled vertically by 2 and component 1 by 3: the buffered entry point
    // only; the streaming one refuses it.
    Spec k;
    k.ssiz            = {7, 7};
    k.xr              = {2, 1};
    k.yr              = {2, 3};
    const auto planes = spec_planes(k);
    bool refused      = false;
    try {
      spec_encode(k, planes, true);
    } catch (std::exception &) {
      refused = true;
    }
    check("YRsiz 2 and 3: the streaming entry point refuses them", refused);
    long e = -1;
    try {
      e = spec_error(k, spec_encode(k, planes, false), planes);
    } catch (std::exception &) {
    }
    check("YRsiz 2 and 3: the buffered entry point reproduces the input", e == 0);
    // RPCL and PCRL need subsampling factors that are powers of two (ISO/IEC 15444-1,
    // B.12.1.3 and B.12.1.4; refused otherwise, below); CPRL takes any.
    struct Prog {
      uint8_t order;
      uint8_t xr, yr;
    };
    // Component 0 subsampled horizontally more than the others: their rows are wider than
    // component 0's, and the streaming entry point once kept every row in component 0's
    // width (a heap overflow).  The 3-column tiles leave component 0 without samples in some.
    struct Wide {
      uint8_t x0, y0, x1, y1;
      uint32_t tw;
    };
    const Wide wides[] = {
        {4, 1, 1, 1, 0}, {3, 2, 1, 2, 0}, {5, 1, 2, 3, 0}, {4, 2, 1, 4, 0}, {5, 1, 1, 1, 3}};
    for (const auto &wd : wides) {
      Spec kw;
      kw.w = 83, kw.h = 41, kw.ox = 3, kw.oy = 5, kw.tox = 3, kw.toy = 5, kw.tw = wd.tw,
      kw.th   = wd.tw ? 41 : 0;
      kw.ssiz = {7, 7};
      kw.xr   = {wd.x0, wd.x1};
      kw.yr   = {wd.y0, wd.y1};
      char what[96];
      snprintf(what, sizeof(what), "component 0 subsampled %ux%u, component 1 %ux%u%s", wd.x0, wd.y0, wd.x1,
               wd.y1, wd.tw ? ", 3-column tiles" : "");
      edge(what, kw, 0, false, true);
    }
    const Prog progs[] = {{4, 3, 5}, {4, 5, 3}, {2, 2, 4}, {3, 4, 2}, {2, 1, 1}};
    for (const auto &pg : progs) {
      Spec kp;
      kp.w = 97, kp.h = 61, kp.ox = 3, kp.oy = 5, kp.tox = 3, kp.toy = 5;
      kp.ssiz  = {7, 7, 7};
      kp.xr    = {1, pg.xr, 1};
      kp.yr    = {1, pg.yr, 1};
      kp.order = pg.order;
      char what[96];
      snprintf(what, sizeof(what), "%s with subsampling %ux%u",
               pg.order == 4   ? "CPRL"
               : pg.order == 2 ? "RPCL"
                               : "PCRL",
               pg.xr, pg.yr);
      edge(what, kp, 0, false, true);
    }
  }
  {
    // Lossless corners.
    Spec k;
    k.ssiz = {0};  // 1-bit samples: their coarsest subbands needed one more bit-plane at 6 levels
    k.w = 149, k.h = 776, k.ox = 64, k.oy = 270, k.tox = 64, k.toy = 270, k.levels = 6;
    edge("1-bit samples, 6 levels", k, 0, false, true);
    k.style = 0x00;
    edge("1-bit samples, 6 levels, Part 1", k, 0, false, true);
    k       = Spec();
    k.ssiz  = {1};
    k.guard = 4;  // more guard bits than 2-bit samples leave room for: exponent 0, not 255
    edge("2-bit samples, 4 guard bits", k, 0, false, true);
    k        = Spec();
    k.layers = 300;  // the inclusion tag tree counted layers in 8 bits
    edge("300 quality layers", k, 0, false, true);
    k    = Spec();
    k.ox = 40, k.oy = 30;  // tile size 0: one tile, from the tile origin (0, 0) to the far edge
    edge("tile size 0 with an image origin", k, 0, false, true);
    k        = Spec();
    k.ssiz   = {22, 22, 22};  // 23 bits: the most the reversible path codes losslessly
    k.xr     = {1, 1, 1};
    k.yr     = {1, 1, 1};
    k.levels = 5;
    edge("23-bit samples", k, 0, false, true);
    k.mct = 1;  // 24 bits after the reversible colour transform
    edge("23-bit samples with the colour transform (refused)", k, 0, true, true);
    k       = Spec();
    k.guard = 0;  // the reversible step sizes do not depend on the guard bits
    edge("reversible, 0 guard bits", k, 0, false, true);
    k         = Spec();
    k.maxprec = false, k.ppx = {5, 0}, k.ppy = {5, 0}, k.levels = 1;  // exponent 0 at resolution 0
    edge("precinct exponent 0 at resolution 0", k, 0, false, true);
    // Lossy corners.
    k         = Spec();
    k.xf      = 0;
    k.qfactor = 0;  // taken as 1: 50 / 0 made every step infinite
    k.ssiz    = {7, 7, 7};
    k.xr      = {1, 1, 1};
    k.yr      = {1, 1, 1};
    k.mct     = 1;
    edge("Qfactor 0", k, 255, false, true);
    {
      Spec k1           = k;
      k1.qfactor        = 1;
      const auto planes = spec_planes(k);
      check("Qfactor 0 codes as Qfactor 1",
            spec_encode(k, planes, false) == spec_encode(k1, planes, false));
    }
    k              = Spec();
    k.xf           = 0;
    k.ssiz         = {1, 1, 1};
    k.xr           = {1, 1, 1};
    k.yr           = {1, 1, 1};
    k.mct          = 1;
    k.checkerboard = true;
    k.step         = 1e-9;  // 30 magnitude bit-planes in every sample: the longest HT cleanup segments
    k.levels       = 1;
    k.w = 64, k.h = 64;
    edge("30 magnitude bit-planes, alternating extremes", k, 3, false, true);
    k      = Spec();
    k.xf   = 0;
    k.step = 1e300;  // the largest step there is
    edge("base step 1e300", k, 255, false, true);
    // The deepest samples the decoder holds: 30 bits unsigned, 31 signed (refused above).
    k      = Spec();
    k.xf   = 0;
    k.step = 1e-7;
    k.ssiz = {29};
    edge("30-bit unsigned samples, irreversible", k, 1L << 22, false, true);
    k.ssiz = {0x9E};
    edge("31-bit signed samples, irreversible", k, 1L << 22, false, true);
    k         = Spec();
    k.xf      = 0;
    k.derived = true;
    k.levels  = 20;  // the derived exponents epsilon_0 - NL + n_b stay >= 0: epsilon_0 >= 19
    edge("derived quantization, 20 levels", k, 255, false, true);
    k.levels = 29;
    k.guard  = 7;  // epsilon_0 can be at most 31 - 7 = 24, less than 28
    edge("derived quantization, 29 levels, 7 guard bits (refused)", k, 0, true, true);
    k                   = Spec();
    k.xf                = 0;
    k.levels            = 24;
    k.targets           = {400};  // the rate control's basis functions doubled in length per level
    const double before = peak_rss_mib();
    edge("target size, 24 levels", k, 255, false, false);
    const double grown = peak_rss_mib() - before;
    check("target size, 24 levels: peak memory grew by at most 64 MiB", before < 0 || grown <= 64.0);
    // PRCL comes with a CAP marker for its ISO/IEC 15444-2 capability: Rsiz bits 15 and 14.
    k                 = Spec();
    k.order           = 5;
    const auto planes = spec_planes(k);
    const auto j2c    = spec_encode(k, planes, false);
    check("PRCL: Rsiz bits 15 and 14",
          (rsiz_of(j2c) & 0xC000) == 0xC000 && spec_error(k, j2c, planes) == 0);
  }
  {
    // Out-of-range parameters: refused with an exception by both entry points.
    struct Bad {
      const char *what;
      void (*apply)(Spec &);
    };
    const Bad bad[] = {
        {"no components", [](Spec &k) { k.ssiz.clear(), k.xr.clear(), k.yr.clear(); }},
        {"32-bit samples", [](Spec &k) { k.ssiz = {31}, k.xf = 0; }},
        {"31-bit unsigned samples", [](Spec &k) { k.ssiz = {30}, k.xf = 0, k.step = 1e-7; }},
        {"32-bit signed samples", [](Spec &k) { k.ssiz = {0x9F}, k.xf = 0, k.step = 1e-7; }},
        {"24-bit reversible samples", [](Spec &k) { k.ssiz = {23}; }},
        {"subsampling factor 0", [](Spec &k) { k.ssiz = {7, 7}, k.xr = {1, 0}, k.yr = {1, 1}; }},
        {"empty image", [](Spec &k) { k.w = 0; }},
        {"image extent beyond 2^31 - 1", [](Spec &k) { k.ox = k.tox = 0x7FFFFFF0u; }},
        {"33 DWT levels", [](Spec &k) { k.levels = 33; }},
        {"code-block exponent 9", [](Spec &k) { k.bw = 9, k.bh = 0; }},
        {"code-blocks of 8192 samples", [](Spec &k) { k.bw = 5, k.bh = 6; }},
        {"progression order 6", [](Spec &k) { k.order = 6; }},
        {"0 quality layers", [](Spec &k) { k.layers = 0; }},
        {"use_color_trafo 2",
         [](Spec &k) { k.ssiz = {7, 7, 7}, k.xr = {1, 1, 1}, k.yr = {1, 1, 1}, k.mct = 2; }},
        {"transformation 2", [](Spec &k) { k.xf = 2; }},
        {"8 guard bits", [](Spec &k) { k.guard = 8; }},
        {"irreversible, 0 guard bits", [](Spec &k) { k.xf = 0, k.guard = 0; }},
        {"base step 0", [](Spec &k) { k.xf = 0, k.step = 0; }},
        {"negative base step", [](Spec &k) { k.xf = 0, k.step = -1; }},
        {"infinite base step", [](Spec &k) { k.xf = 0, k.step = HUGE_VAL; }},
        {"base step not a number", [](Spec &k) { k.xf = 0, k.step = std::nan(""); }},
        {"precinct exponent 16", [](Spec &k) { k.maxprec = false, k.ppx = {16}, k.ppy = {16}; }},
        {"precinct exponent 0 above resolution 0",
         [](Spec &k) { k.maxprec = false, k.ppx = {0}, k.ppy = {0}; }},
        {"no precinct sizes", [](Spec &k) { k.maxprec = false; }},
        {"RPCL with XRsiz 3", [](Spec &k) { k.order = 2, k.ssiz = {7, 7}, k.xr = {1, 3}, k.yr = {1, 1}; }},
        {"RPCL with YRsiz 5", [](Spec &k) { k.order = 2, k.ssiz = {7, 7}, k.xr = {1, 1}, k.yr = {1, 5}; }},
        {"PCRL with XRsiz 3 on component 0", [](Spec &k) { k.order = 3, k.xr = {3}; }},
        {"PCRL with YRsiz 6", [](Spec &k) { k.order = 3, k.ssiz = {7, 7}, k.xr = {1, 2}, k.yr = {1, 6}; }},
    };
    for (const auto &b : bad) {
      Spec k;
      b.apply(k);
      std::string what = std::string(b.what) + " (refused)";
      edge(what.c_str(), k, 0, true, true);
    }
  }

  printf("encode_entry_points: %d cases, %d failed\n", total, failed);
  return failed == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
