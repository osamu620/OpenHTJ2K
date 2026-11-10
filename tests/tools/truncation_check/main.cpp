// truncation_check: decoding a codestream that is cut short (a truncated file or download).
//
// For each length L from the end of the main header to the full length, in steps of S,
// and for the last three lengths, the first L bytes must decode without an error through
// each entry point:
//   line     invoke_line_based()
//   stream   invoke_line_based_stream()
//   reuse    invoke_line_based_stream_reuse(), reuse on: one decoder for every length
//   direct   invoke_line_based_direct(), reuse on, likewise (when every component is
//            unsigned with up to 16 bits)
// and every entry point must give the same samples.  is_truncated() must be true for each
// L that ends before the EOC marker does and false for the whole input (bytes after EOC
// are allowed), and a codestream that lacks only its EOC marker, or the last byte of it,
// must decode to the samples of the whole one.
// The decoders that kept their structure must decode the full codestream again afterwards
// as a fresh decoder does.  A cut before the marker code of the first SOT marker segment,
// where parse() cannot tell that the main header is whole, must be refused by parse() with
// an exception (no tile could be decoded anyway).
// Usage: truncation_check <codestream> [-step S] [-threads N]
// Prints PASS and exits 0, or prints what failed and exits 1.
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
  if (failures < 20) printf("FAIL %s\n", what.c_str());
  ++failures;
}

uint16_t be16(const Bytes &b, size_t i) { return static_cast<uint16_t>((b[i] << 8) | b[i + 1]); }

struct Image {
  std::vector<uint32_t> w, h;
  std::vector<uint8_t> depth;
  std::vector<bool> is_signed;
  std::vector<std::vector<int32_t>> plane;
  bool truncated = false;
  bool same_samples(const Image &o) const { return w == o.w && h == o.h && plane == o.plane; }
};

void allocate(Image &img) {
  img.plane.resize(img.w.size());
  for (size_t c = 0; c < img.w.size(); ++c)
    img.plane[c].assign(static_cast<size_t>(img.w[c]) * img.h[c], 0);
}

Image decode_line(const Bytes &cs) {
  open_htj2k::openhtj2k_decoder dec;
  dec.init(cs.data(), cs.size(), 0, threads);
  dec.parse();
  Image img;
  std::vector<int32_t *> buf;
  dec.invoke_line_based(buf, img.w, img.h, img.depth, img.is_signed);
  img.plane.resize(buf.size());
  for (size_t c = 0; c < buf.size(); ++c) {
    img.plane[c].assign(buf[c], buf[c] + static_cast<size_t>(img.w[c]) * img.h[c]);
    delete[] buf[c];
  }
  img.truncated = dec.is_truncated();
  return img;
}

// Each component row once, as invoke_line_based_stream() documents (decoder.hpp).
Image decode_stream(open_htj2k::openhtj2k_decoder &dec, const Bytes &cs, bool reuse) {
  dec.init(cs.data(), cs.size(), 0, threads);
  dec.parse();
  Image img;
  std::vector<uint32_t> last;
  auto cb = [&](uint32_t y, int32_t *const *rows, uint16_t nc) {
    if (img.plane.empty()) {
      allocate(img);
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
  if (reuse)
    dec.invoke_line_based_stream_reuse(cb, img.w, img.h, img.depth, img.is_signed);
  else
    dec.invoke_line_based_stream(cb, img.w, img.h, img.depth, img.is_signed);
  if (img.plane.empty()) allocate(img);
  img.truncated = dec.is_truncated();
  return img;
}

// Unsigned 16-bit planes shaped like `shape` (the samples are the same, within the range).
Image decode_direct(open_htj2k::openhtj2k_decoder &dec, const Bytes &cs, const Image &shape) {
  dec.init(cs.data(), cs.size(), 0, threads);
  dec.parse();
  const size_t nc = shape.w.size();
  std::vector<std::vector<uint16_t>> planes(nc);
  std::vector<open_htj2k::PlanarOutputDesc> descs(nc);
  for (size_t c = 0; c < nc; ++c) {
    planes[c].assign(static_cast<size_t>(shape.w[c]) * shape.h[c], 0);
    open_htj2k::PlanarOutputDesc &d = descs[c];
    d.base                          = planes[c].data();
    d.stride                        = shape.w[c];
    d.width                         = shape.w[c];
    d.height                        = shape.h[c];
    d.yr                            = 1;
    d.dc                            = 1 << (shape.depth[c] - 1);
    d.maxval                        = (1 << shape.depth[c]) - 1;
    d.minval                        = 0;
    d.depth_shift                   = 0;
    d.is_16bit                      = true;
  }
  Image img;
  dec.invoke_line_based_direct(descs.data(), static_cast<uint16_t>(nc), img.w, img.h, img.depth,
                               img.is_signed);
  img.plane.resize(nc);
  for (size_t c = 0; c < nc; ++c) img.plane[c].assign(planes[c].begin(), planes[c].end());
  img.truncated = dec.is_truncated();
  return img;
}

struct Decoders {
  open_htj2k::openhtj2k_decoder reuse, direct;
  bool use_direct = false;
  Decoders() {
    reuse.enable_single_tile_reuse(true);
    direct.enable_single_tile_reuse(true);
  }
};

// Decodes the first L bytes through every entry point; checks that they agree and that
// is_truncated() is `truncated`.  Returns the samples, or an empty image on a failure.
Image check_length(Decoders &decs, const Bytes &cs, size_t L, bool truncated) {
  const Bytes cut(cs.begin(), cs.begin() + static_cast<std::ptrdiff_t>(L));
  const std::string at = "length " + std::to_string(L) + ": ";
  Image line;
  try {
    line = decode_line(cut);
    open_htj2k::openhtj2k_decoder fresh;
    const Image stream       = decode_stream(fresh, cut, false);
    const Image reuse        = decode_stream(decs.reuse, cut, true);
    const Image *const all[] = {&line, &stream, &reuse};
    for (const Image *img : all) {
      if (img->truncated != truncated) {
        fail(at + "is_truncated() is " + (img->truncated ? "true" : "false"));
        break;
      }
    }
    if (!stream.same_samples(line))
      fail(at + "invoke_line_based_stream() differs from invoke_line_based()");
    if (!reuse.same_samples(line)) fail(at + "invoke_line_based_stream_reuse() differs");
    if (decs.use_direct) {
      const Image direct = decode_direct(decs.direct, cut, line);
      if (direct.truncated != truncated)
        fail(at + "is_truncated() after invoke_line_based_direct() is wrong");
      if (!direct.same_samples(line)) fail(at + "invoke_line_based_direct() differs");
    }
  } catch (std::exception &e) {
    fail(at + "decode failed: " + e.what());
    return Image();
  }
  return line;
}
}  // namespace

int main(int argc, char *argv[]) {
  std::string infile;
  size_t step = 1;
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "-threads") == 0 && i + 1 < argc) {
      threads = static_cast<uint32_t>(atoi(argv[++i]));
    } else if (strcmp(argv[i], "-step") == 0 && i + 1 < argc) {
      step = static_cast<size_t>(std::max(1, atoi(argv[++i])));
    } else {
      infile = argv[i];
    }
  }
  if (infile.empty()) {
    printf("usage: truncation_check <codestream> [-step S] [-threads N]\n");
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

  // The main header ends at the first SOT marker; the codestream, at its EOC marker (the
  // last 0xFFD9: coded data holds no marker codes above 0xFF8F), which bytes may follow.
  size_t sot = 2, siz = 0;
  while (sot + 4 <= cs.size() && be16(cs, sot) != 0xFF90) {
    const uint16_t marker = be16(cs, sot);
    if (marker == 0xFF51) siz = sot;
    sot += (marker >= 0xFF30 && marker <= 0xFF3F) ? 2u : 2u + be16(cs, sot + 2);  // A.1.4
  }
  const size_t N = cs.size();
  size_t eoc_end = 0;  // one past the EOC marker
  for (size_t i = N; i >= 2 && eoc_end == 0; --i)
    if (cs[i - 2] == 0xFF && cs[i - 1] == 0xD9) eoc_end = i;
  if (N < 4 || be16(cs, 0) != 0xFF4F || siz == 0 || sot + 12 > N || eoc_end <= sot) {
    printf("%s: not a codestream with an EOC marker\n", infile.c_str());
    return 1;
  }
  Decoders decs;
  decs.use_direct = true;  // invoke_line_based_direct writes unsigned planes of up to 16 bits
  for (uint16_t c = 0, csiz = be16(cs, siz + 38); c < csiz; ++c) {
    const uint8_t ssiz = cs[siz + 40 + 3 * c];
    if ((ssiz & 0x80) != 0 || (ssiz & 0x7F) + 1 > 16) decs.use_direct = false;
  }

  const Image full = check_length(decs, cs, N, false);
  if (full.plane.empty()) return 1;

  // Cuts inside the main header, or before the marker code that ends it.
  size_t refused = 0;
  for (size_t L = 1; L < sot + 2; L += step) {
    const Bytes cut(cs.begin(), cs.begin() + static_cast<std::ptrdiff_t>(L));
    try {
      open_htj2k::openhtj2k_decoder dec;
      dec.init(cut.data(), cut.size(), 0, threads);
      dec.parse();
      fail("length " + std::to_string(L) + ": parse() accepted a main header cut short");
    } catch (std::exception &) {
      ++refused;
    }
  }

  // Cuts after it.
  std::vector<size_t> lengths;
  for (size_t L = sot + 2; L + 3 < eoc_end; L += step) lengths.push_back(L);
  for (size_t L = eoc_end - 3; L < eoc_end; ++L) lengths.push_back(L);
  for (size_t L : lengths) {
    const Image img = check_length(decs, cs, L, true);
    if (L + 2 >= eoc_end && !img.plane.empty() && !img.same_samples(full))
      fail("length " + std::to_string(L)
           + ": a codestream without (all of) its EOC marker decodes differently");
  }

  // The decoders that decoded the cuts with their structure kept decode it whole again.
  const Image again = check_length(decs, cs, N, false);
  if (!again.plane.empty() && !again.same_samples(full))
    fail("the full codestream decodes differently after its cuts");

  if (failures != 0) {
    printf("%d check(s) failed\n", failures);
    return 1;
  }
  printf("PASS (%zu cuts decoded, %zu main-header cuts refused; line, stream, reuse%s)\n", lengths.size(),
         refused, decs.use_direct ? ", direct" : "");
  return 0;
}
