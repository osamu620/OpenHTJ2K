// reuse_switch_check: the single-tile reuse path when the coding parameters change between
// frames.  enable_single_tile_reuse(true) keeps the tile's structure (subbands, step sizes,
// packet order) from one frame to the next; a frame whose headers describe another
// structure must be decoded as a fresh decoder decodes it, not with the cached one.
//
// From one single-tile codestream F the tool makes variants V that decode to other samples
// but keep the coded data valid, by editing marker segments in memory:
//   qcd       the step sizes of the main-header QCD (Sqcd style 2) changed, all but the
//             first (that of the LL band); if a component uses the QCD
//   qcc       those of the first main-header QCC changed likewise (if F has a QCC)
//   tp-qcd    the changed QCD added to the tile-part header (if F has no main-header
//             QCC, which a tile-part QCD would override with other exponents)
//   tp-qcc    the changed first main-header QCC, or a QCC for component 0 made from the
//             changed QCD, added to the tile-part header
//   reduce    F itself, decoded at one more resolution reduction
// (The other marker segments that shape the tile, RGN among them, are hashed the same
// way; an RGN shift does not change the samples of data coded without a region.)
// and decodes F, V, F on one decoder with reuse enabled, through
// invoke_line_based_stream_reuse() and (components of up to 16 bits, unsigned)
// invoke_line_based_direct().  Each frame must equal the same codestream decoded by a
// fresh decoder, and each V must decode differently from F (else the variant tests
// nothing).
// Usage: reuse_switch_check <input> [-threads N]
// Prints PASS and exits 0, or prints what differs (or the decoder's error) and exits 1.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>
#include "decoder.hpp"

namespace {
using Bytes = std::vector<uint8_t>;

uint32_t threads = 1;
int failures     = 0;

void fail(const std::string &what) {
  printf("FAIL %s\n", what.c_str());
  ++failures;
}

uint16_t be16(const Bytes &b, size_t i) { return static_cast<uint16_t>((b[i] << 8) | b[i + 1]); }
void put_be16(Bytes &b, size_t i, uint16_t v) {
  b[i]     = static_cast<uint8_t>(v >> 8);
  b[i + 1] = static_cast<uint8_t>(v);
}

// A marker segment of the main header: its code, and the offset of its first byte (the
// marker code) and its size including the code.
struct Segment {
  uint16_t code;
  size_t pos, size;
};

std::vector<Segment> main_header_segments(const Bytes &cs, size_t &sot) {
  std::vector<Segment> segs;
  size_t i = 2;  // after SOC
  while (i + 4 <= cs.size() && be16(cs, i) != 0xFF90) {
    const size_t size = 2u + be16(cs, i + 2);
    segs.push_back({be16(cs, i), i, size});
    i += size;
  }
  sot = i;
  return segs;
}

const Segment *find(const std::vector<Segment> &segs, uint16_t code) {
  for (const Segment &s : segs)
    if (s.code == code) return &s;
  return nullptr;
}

// Changes the mantissas of the step sizes of a QCD/QCC segment but the first (Sq style 2:
// two bytes per subband, the mantissa in the low 11 bits).  Returns false for another
// style or a single step size.
bool change_steps(Bytes &seg, size_t sq_offset) {
  if ((seg[sq_offset] & 0x1F) != 2 || seg.size() < sq_offset + 5) return false;
  for (size_t i = sq_offset + 1 + 2 + 1; i < seg.size(); i += 2) seg[i] ^= 0x55;
  return true;
}

// Inserts a marker segment at the start of the first tile-part header, growing Psot.
Bytes insert_in_tile_part(const Bytes &cs, size_t sot, const Bytes &seg) {
  Bytes out(cs.begin(), cs.begin() + static_cast<std::ptrdiff_t>(sot + 12));
  out.insert(out.end(), seg.begin(), seg.end());
  out.insert(out.end(), cs.begin() + static_cast<std::ptrdiff_t>(sot + 12), cs.end());
  const uint32_t psot = (static_cast<uint32_t>(be16(cs, sot + 6)) << 16) | be16(cs, sot + 8);
  if (psot != 0) {
    const uint32_t grown = psot + static_cast<uint32_t>(seg.size());
    put_be16(out, sot + 6, static_cast<uint16_t>(grown >> 16));
    put_be16(out, sot + 8, static_cast<uint16_t>(grown));
  }
  return out;
}

struct Image {
  std::vector<uint32_t> w, h;
  std::vector<uint8_t> depth;
  std::vector<bool> is_signed;
  std::vector<std::vector<int32_t>> plane;
  bool operator==(const Image &o) const { return w == o.w && h == o.h && plane == o.plane; }
};

// Each component row once, as invoke_line_based_stream() documents (decoder.hpp).
std::function<void(uint32_t, int32_t *const *, uint16_t)> collector(open_htj2k::openhtj2k_decoder &dec,
                                                                    Image &img,
                                                                    std::vector<uint32_t> &last) {
  return [&dec, &img, &last](uint32_t y, int32_t *const *rows, uint16_t nc) {
    if (img.plane.empty()) {
      img.plane.resize(img.w.size());
      for (size_t c = 0; c < img.w.size(); ++c)
        img.plane[c].assign(static_cast<size_t>(img.w[c]) * img.h[c], 0);
      last.assign(img.w.size(), UINT32_MAX);
    }
    const uint16_t n = static_cast<uint16_t>(std::min<size_t>(nc, img.w.size()));
    for (uint16_t c = 0; c < n; ++c) {
      const uint32_t cy = dec.get_stream_component_row(c, y);
      if (cy == UINT32_MAX || cy == last[c] || cy >= img.h[c]) continue;
      last[c] = cy;
      std::memcpy(img.plane[c].data() + static_cast<size_t>(cy) * img.w[c], rows[c],
                  img.w[c] * sizeof(int32_t));
    }
  };
}

// One frame on `dec`: through the streaming reuse entry point, or the direct one.
Image decode(open_htj2k::openhtj2k_decoder &dec, const Bytes &cs, uint8_t reduce, bool direct) {
  Image img;
  dec.init(cs.data(), cs.size(), reduce, threads);
  dec.parse();
  if (!direct) {
    std::vector<uint32_t> last;
    // The callback reads img.w/h, which the call fills in before the first row.
    dec.invoke_line_based_stream_reuse(collector(dec, img, last), img.w, img.h, img.depth, img.is_signed);
    return img;
  }
  // The plane sizes come from a parse of the same codestream on a scratch decoder.
  {
    open_htj2k::openhtj2k_decoder probe;
    probe.init(cs.data(), cs.size(), reduce, 1);
    probe.parse();
    std::vector<uint32_t> last;
    Image tmp;
    probe.invoke_line_based_stream(collector(probe, tmp, last), tmp.w, tmp.h, tmp.depth, tmp.is_signed);
    img.w = tmp.w, img.h = tmp.h, img.depth = tmp.depth, img.is_signed = tmp.is_signed;
  }
  const size_t nc = img.w.size();
  std::vector<std::vector<uint16_t>> planes(nc);
  std::vector<open_htj2k::PlanarOutputDesc> descs(nc);
  for (size_t c = 0; c < nc; ++c) {
    planes[c].assign(static_cast<size_t>(img.w[c]) * img.h[c], 0);
    open_htj2k::PlanarOutputDesc &d = descs[c];
    d.base                          = planes[c].data();
    d.stride                        = img.w[c];
    d.width                         = img.w[c];
    d.height                        = img.h[c];
    d.yr                            = 1;
    d.dc                            = 1 << (img.depth[c] - 1);
    d.maxval                        = (1 << img.depth[c]) - 1;
    d.minval                        = 0;
    d.depth_shift                   = 0;
    d.is_16bit                      = true;
  }
  std::vector<uint32_t> W, H;
  std::vector<uint8_t> D;
  std::vector<bool> S;
  dec.invoke_line_based_direct(descs.data(), static_cast<uint16_t>(nc), W, H, D, S);
  img.plane.resize(nc);
  for (size_t c = 0; c < nc; ++c) img.plane[c].assign(planes[c].begin(), planes[c].end());
  return img;
}

Image fresh(const Bytes &cs, uint8_t reduce, bool direct) {
  open_htj2k::openhtj2k_decoder dec;
  return decode(dec, cs, reduce, direct);
}

void check_variant(const std::string &name, const Bytes &base, const Bytes &variant, uint8_t vreduce,
                   bool direct) {
  // base is decoded at reduction 0, the variant at vreduce.
  const std::string label = name + (direct ? " (direct)" : " (stream)");
  try {
    const Image ref_base = fresh(base, 0, direct);
    const Image ref_var  = fresh(variant, vreduce, direct);
    if (ref_base == ref_var) {
      fail(label + ": the variant decodes like the input, so it tests nothing");
      return;
    }
    open_htj2k::openhtj2k_decoder dec;
    dec.enable_single_tile_reuse(true);
    if (!(decode(dec, base, 0, direct) == ref_base)) fail(label + ": frame 1 (input) differs");
    if (!(decode(dec, variant, vreduce, direct) == ref_var))
      fail(label + ": frame 2 (variant) differs from a fresh decode");
    if (!(decode(dec, base, 0, direct) == ref_base))
      fail(label + ": frame 3 (input again) differs from a fresh decode");
  } catch (std::exception &e) {
    fail(label + ": " + e.what());
  }
}
}  // namespace

int main(int argc, char *argv[]) {
  std::string infile;
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "-threads") == 0 && i + 1 < argc) {
      threads = static_cast<uint32_t>(atoi(argv[++i]));
    } else {
      infile = argv[i];
    }
  }
  if (infile.empty()) {
    printf("usage: reuse_switch_check <input> [-threads N]\n");
    return 1;
  }
  FILE *fp = fopen(infile.c_str(), "rb");
  if (fp == nullptr) {
    printf("cannot open %s\n", infile.c_str());
    return 1;
  }
  Bytes cs;
  uint8_t chunk[65536];
  size_t n;
  while ((n = fread(chunk, 1, sizeof(chunk), fp)) > 0) cs.insert(cs.end(), chunk, chunk + n);
  fclose(fp);

  size_t sot                      = 0;
  const std::vector<Segment> segs = main_header_segments(cs, sot);
  const Segment *siz = find(segs, 0xFF51), *qcd = find(segs, 0xFF5C);
  if (cs.size() < 4 || be16(cs, 0) != 0xFF4F || siz == nullptr || qcd == nullptr || sot + 12 > cs.size()) {
    printf("%s: not a codestream this tool understands\n", infile.c_str());
    return 1;
  }
  if (find(segs, 0xFF55) != nullptr) {  // the inserted tile-part markers would not match TLM
    printf("%s: has a TLM marker; pick an input without one\n", infile.c_str());
    return 1;
  }
  const uint16_t csiz    = be16(cs, siz->pos + 38);
  const size_t comp_size = (csiz < 257) ? 1 : 2;  // Cqcc / Crgn bytes

  // qcd: the main-header QCD with its last step size changed.
  Bytes qcd_seg(cs.begin() + static_cast<std::ptrdiff_t>(qcd->pos),
                cs.begin() + static_cast<std::ptrdiff_t>(qcd->pos + qcd->size));
  if (!change_steps(qcd_seg, 4)) {
    printf("%s: its QCD does not list a step size per subband (Sqcd style 2)\n", infile.c_str());
    return 1;
  }
  Bytes v_qcd = cs;
  std::copy(qcd_seg.begin(), qcd_seg.end(), v_qcd.begin() + static_cast<std::ptrdiff_t>(qcd->pos));

  bool direct_ok = true;  // invoke_line_based_direct writes unsigned planes of up to 16 bits
  for (uint16_t c = 0; c < csiz; ++c) {
    const uint8_t ssiz = cs[siz->pos + 40 + 3 * c];
    if ((ssiz & 0x80) != 0 || (ssiz & 0x7F) + 1 > 16) direct_ok = false;
  }

  struct Variant {
    std::string name;
    Bytes cs;
    uint8_t reduce;
  };
  std::vector<Variant> variants;
  size_t num_qcc = 0;
  for (const Segment &seg : segs) num_qcc += (seg.code == 0xFF5D);
  if (num_qcc < csiz) variants.push_back({"qcd", v_qcd, 0});
  // The tile-part QCC: the first main-header QCC with its step sizes changed, or one for
  // component 0 made from the changed QCD (Cqcc after Lqcc).
  Bytes tp_qcc;
  if (const Segment *qcc = find(segs, 0xFF5D)) {
    Bytes seg(cs.begin() + static_cast<std::ptrdiff_t>(qcc->pos),
              cs.begin() + static_cast<std::ptrdiff_t>(qcc->pos + qcc->size));
    if (change_steps(seg, 4 + comp_size)) {
      Bytes v = cs;
      std::copy(seg.begin(), seg.end(), v.begin() + static_cast<std::ptrdiff_t>(qcc->pos));
      variants.push_back({"qcc", v, 0});
      tp_qcc = seg;
    }
  } else {
    tp_qcc.assign(qcd_seg.begin(), qcd_seg.begin() + 4);
    tp_qcc[1] = 0x5D;
    tp_qcc.insert(tp_qcc.end(), comp_size, 0);
    tp_qcc.insert(tp_qcc.end(), qcd_seg.begin() + 4, qcd_seg.end());
    put_be16(tp_qcc, 2, static_cast<uint16_t>(tp_qcc.size() - 2));
    variants.push_back({"tp-qcd", insert_in_tile_part(cs, sot, qcd_seg), 0});
  }
  if (!tp_qcc.empty()) variants.push_back({"tp-qcc", insert_in_tile_part(cs, sot, tp_qcc), 0});
  variants.push_back({"reduce", cs, 1});

  for (const Variant &v : variants) {
    check_variant(v.name, cs, v.cs, v.reduce, false);
    if (direct_ok) check_variant(v.name, cs, v.cs, v.reduce, true);
  }
  if (failures != 0) {
    printf("%d check(s) failed\n", failures);
    return 1;
  }
  printf("PASS (%zu variants%s)\n", variants.size(), direct_ok ? ", stream and direct" : ", stream");
  return 0;
}
