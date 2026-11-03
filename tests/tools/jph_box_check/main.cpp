// jph_box_check: the box lengths of JPH and JP2 files (ISO/IEC 15444-1 I.4).
//
// A box gives its length, header included, in the 32-bit LBox.  A longer box has
// LBox = 1 and the length in the 64-bit XLBox after the box type; LBox = 0 means
// that the box extends to the end of the file.
//
// Writer: the encoder writes the Contiguous Codestream box header with
// jp2c_box_header().  A codestream of 4 GiB cannot be written in a test, so the
// header is checked byte by byte for code lengths below, at and above the 32-bit
// limit; small JPH and JP2 files encoded through the library must hold the plain
// LBox header and report their own length.
//
// Reader: the same files with the Contiguous Codestream box header rewritten to
// LBox = 1 + XLBox and to LBox = 0, and with every box using XLBox, must decode
// to the input.  XLBox values smaller than the 16-byte header or past the end of
// the file, and a header cut short, must be rejected with an exception.  Each
// buffer is allocated at its exact size, so that a sanitizer build sees any read
// past it.
//
// Usage: jph_box_check
// Exits 0 when every case passes.
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <vector>
#include "decoder.hpp"
#include "encoder.hpp"
#include "jph.hpp"

namespace {

int failures = 0;

void check(bool ok, const std::string &what) {
  if (!ok) {
    ++failures;
    printf("FAIL: %s\n", what.c_str());
  }
}

std::string hex(const uint8_t *p, size_t n) {
  std::string s;
  char b[3];
  for (size_t i = 0; i < n; ++i) {
    snprintf(b, sizeof(b), "%02x", p[i]);
    s += b;
  }
  return s;
}

void put_be32(uint8_t *p, uint32_t v) {
  for (int i = 0; i < 4; ++i) p[i] = static_cast<uint8_t>(v >> (24 - 8 * i));
}
void put_be64(uint8_t *p, uint64_t v) {
  for (int i = 0; i < 8; ++i) p[i] = static_cast<uint8_t>(v >> (56 - 8 * i));
}
uint32_t get_be32(const uint8_t *p) {
  return (static_cast<uint32_t>(p[0]) << 24) | (static_cast<uint32_t>(p[1]) << 16)
         | (static_cast<uint32_t>(p[2]) << 8) | p[3];
}

// --- Writer -------------------------------------------------------------------

void check_header(uint64_t code_len, const std::string &want) {
  uint8_t hdr[16];
  memset(hdr, 0xAA, sizeof(hdr));
  const size_t n        = open_htj2k::jp2c_box_header(code_len, hdr);
  const std::string got = hex(hdr, n);
  char name[64];
  snprintf(name, sizeof(name), "jp2c header for %llu code bytes",
           static_cast<unsigned long long>(code_len));
  check(got == want, std::string(name) + ": " + got + ", expected " + want);
}

void check_writer() {
  // LBox = code length + 8 while it fits in 32 bits.
  check_header(0, "000000086a703263");
  check_header(1000, "000003f06a703263");
  check_header(0xFFFFFFF6u, "fffffffe6a703263");
  check_header(0xFFFFFFF7u, "ffffffff6a703263");  // the largest LBox
  // Then LBox = 1 and XLBox = code length + 16.
  check_header(0xFFFFFFF8u, "000000016a7032630000000100000008");
  check_header(0xFFFFFFFFu, "000000016a703263000000010000000f");
  check_header(0x100000000ull, "000000016a7032630000000100000010");
  check_header(0x140000000ull + 7, "000000016a7032630000000140000017");  // 5 GiB + 7
  check_header(0x123456789ABCull, "000000016a7032630000123456789acc");
}

// --- Files from the encoder -------------------------------------------------------

constexpr uint32_t W = 37, H = 23;
// Two components of different bit depth and sign: 12-bit unsigned, 8-bit signed.
const uint8_t SSIZ[2] = {11, 0x80 | 7};

std::vector<std::vector<int32_t>> make_planes() {
  std::vector<std::vector<int32_t>> planes(2);
  for (size_t c = 0; c < 2; ++c) {
    const uint32_t span = 1u << ((SSIZ[c] & 0x7F) + 1);
    const int32_t lo    = (SSIZ[c] & 0x80) ? -static_cast<int32_t>(span / 2) : 0;
    planes[c].resize(static_cast<size_t>(W) * H);
    for (uint32_t i = 0; i < W * H; ++i)
      planes[c][i] = lo + static_cast<int32_t>((i * 37u + static_cast<uint32_t>(c) * 101u) % span);
  }
  return planes;
}

// A lossless file: JPH with the HT block coder, or JP2 with the Part 1 one.
std::vector<uint8_t> encode(const std::vector<std::vector<int32_t>> &src, bool jp2, size_t &reported) {
  open_htj2k::siz_params siz;
  siz.Rsiz   = 0;
  siz.Xsiz   = W;
  siz.Ysiz   = H;
  siz.XOsiz  = 0;
  siz.YOsiz  = 0;
  siz.XTsiz  = W;
  siz.YTsiz  = H;
  siz.XTOsiz = 0;
  siz.YTOsiz = 0;
  siz.Csiz   = 2;
  siz.Ssiz.assign(SSIZ, SSIZ + 2);
  siz.XRsiz.assign(2, 1);
  siz.YRsiz.assign(2, 1);
  open_htj2k::cod_params cod;
  cod.blkwidth          = 4;
  cod.blkheight         = 4;
  cod.is_max_precincts  = true;
  cod.use_SOP           = false;
  cod.use_EPH           = false;
  cod.progression_order = 0;
  cod.number_of_layers  = 1;
  cod.use_color_trafo   = 0;
  cod.dwt_levels        = 3;
  cod.codeblock_style   = jp2 ? 0x00 : 0x40;
  cod.transformation    = 1;
  open_htj2k::qcd_params qcd;
  qcd.is_derived          = false;
  qcd.number_of_guardbits = 1;
  qcd.base_step           = 1.0 / 256;

  std::vector<std::vector<int32_t>> planes = src;
  std::vector<int32_t *> input;
  for (auto &p : planes) input.push_back(p.data());
  std::vector<uint8_t> out;
  open_htj2k::openhtj2k_encoder enc("", input, siz, cod, qcd, 0xFF, false, 0, 1);
  enc.set_output_buffer(out);
  enc.set_file_format(jp2 ? open_htj2k::file_format::jp2 : open_htj2k::file_format::jph);
  reported = enc.invoke_line_based();
  return out;
}

// Offsets of the top-level boxes of a file with plain LBox values; empty if malformed.
std::vector<size_t> box_offsets(const std::vector<uint8_t> &f) {
  std::vector<size_t> offs;
  size_t p = 0;
  while (p < f.size()) {
    if (f.size() - p < 8) return std::vector<size_t>();
    const uint32_t len = get_be32(&f[p]);
    if (len < 8 || len > f.size() - p) return std::vector<size_t>();
    offs.push_back(p);
    p += len;
  }
  return offs;
}

// Decodes buf (copied to an allocation of exactly its size); returns an error message,
// empty when it decodes to src with the colour space of two components (greyscale).
std::string decode(const std::vector<uint8_t> &buf, const std::vector<std::vector<int32_t>> &src) {
  std::unique_ptr<uint8_t[]> exact(new uint8_t[buf.size()]);
  memcpy(exact.get(), buf.data(), buf.size());
  open_htj2k::openhtj2k_decoder dec(exact.get(), buf.size(), 0, 1);
  exact.reset();  // the decoder holds its own copy of the codestream
  std::vector<int32_t *> out;
  std::vector<uint32_t> w, h;
  std::vector<uint8_t> depth;
  std::vector<bool> is_signed;
  dec.parse();
  const uint32_t cs = dec.get_colorspace();
  dec.invoke_line_based(out, w, h, depth, is_signed);
  std::string err;
  if (cs != open_htj2k::ENUMCS_GRAYSCALE) err = "colour space " + std::to_string(cs);
  if (err.empty() && out.size() != src.size()) err = "component count " + std::to_string(out.size());
  for (size_t c = 0; err.empty() && c < out.size(); ++c) {
    if (w[c] != W || h[c] != H || depth[c] != (SSIZ[c] & 0x7F) + 1
        || is_signed[c] != ((SSIZ[c] & 0x80) != 0))
      err = "component " + std::to_string(c) + " has another shape or depth";
    else if (memcmp(out[c], src[c].data(), sizeof(int32_t) * src[c].size()) != 0)
      err = "component " + std::to_string(c) + " differs from the input";
  }
  for (auto p : out) delete[] p;
  return err;
}

void expect_decodes(const std::vector<uint8_t> &buf, const std::vector<std::vector<int32_t>> &src,
                    const std::string &what) {
  std::string err;
  try {
    err = decode(buf, src);
  } catch (std::exception &e) {
    err = std::string("exception: ") + e.what();
  }
  check(err.empty(), what + ": " + err);
}

void expect_rejected(const std::vector<uint8_t> &buf, const std::vector<std::vector<int32_t>> &src,
                     const std::string &what) {
  bool thrown = false;
  try {
    decode(buf, src);
  } catch (std::exception &) {
    thrown = true;
  }
  check(thrown, what + ": not rejected");
}

// f with the 8-byte box header at off replaced by hdr.
std::vector<uint8_t> with_header(const std::vector<uint8_t> &f, size_t off,
                                 const std::vector<uint8_t> &hdr) {
  std::vector<uint8_t> g(f.begin(), f.begin() + static_cast<std::ptrdiff_t>(off));
  g.insert(g.end(), hdr.begin(), hdr.end());
  g.insert(g.end(), f.begin() + static_cast<std::ptrdiff_t>(off + 8), f.end());
  return g;
}

// The 16-byte header of a box of type tbox (from the 8-byte header at p) with the given XLBox.
std::vector<uint8_t> xl_header(const uint8_t *p, uint64_t xlbox) {
  std::vector<uint8_t> hdr(16);
  put_be32(&hdr[0], 1);
  memcpy(&hdr[4], p + 4, 4);
  put_be64(&hdr[8], xlbox);
  return hdr;
}

void check_file(bool jp2) {
  const std::string name       = jp2 ? "JP2" : "JPH";
  const auto planes            = make_planes();
  size_t reported              = 0;
  const std::vector<uint8_t> f = encode(planes, jp2, reported);
  check(reported == f.size(), name + ": the encoder reports " + std::to_string(reported)
                                  + " bytes for a file of " + std::to_string(f.size()));
  const std::vector<size_t> offs = box_offsets(f);
  check(offs.size() == 4, name + ": the file does not hold four boxes with plain lengths");
  if (offs.size() != 4) return;
  const size_t jp2h = offs[2], jp2c = offs[3];
  check(get_be32(&f[jp2c + 4]) == 0x6A703263u,
        name + ": the last box is not the Contiguous Codestream box");
  const uint64_t code_len = f.size() - jp2c - 8;
  check(get_be32(&f[jp2c]) == code_len + 8, name + ": the jp2c LBox is not the codestream length + 8");
  check(f[jp2c + 8] == 0xFF && f[jp2c + 9] == 0x4F, name + ": the codestream does not start with SOC");
  expect_decodes(f, planes, name + " as written");

  // jp2c with LBox = 1 and XLBox.
  expect_decodes(with_header(f, jp2c, xl_header(&f[jp2c], code_len + 16)), planes,
                 name + " jp2c with XLBox");
  // jp2c with LBox = 0, to the end of the file.
  {
    std::vector<uint8_t> g = f;
    put_be32(&g[jp2c], 0);
    expect_decodes(g, planes, name + " jp2c with LBox 0");
  }
  // Every box with XLBox, the boxes in the JP2 header box included, but the signature
  // box, whose LBox of 12 identifies the file.
  {
    std::vector<uint8_t> g(f.begin(), f.begin() + static_cast<std::ptrdiff_t>(offs[1]));
    for (size_t i = 1; i < 4; ++i) {
      const size_t off = offs[i];
      const size_t len = (i < 3) ? offs[i + 1] - off : f.size() - off;
      std::vector<uint8_t> body(f.begin() + static_cast<std::ptrdiff_t>(off + 8),
                                f.begin() + static_cast<std::ptrdiff_t>(off + len));
      if (i == 2) {
        // The boxes in the JP2 header box each grow by 8 bytes.
        body.clear();
        for (size_t q = off + 8; q < off + len;) {
          const size_t sub               = get_be32(&f[q]);
          const std::vector<uint8_t> hdr = xl_header(&f[q], sub + 8);
          body.insert(body.end(), hdr.begin(), hdr.end());
          body.insert(body.end(), f.begin() + static_cast<std::ptrdiff_t>(q + 8),
                      f.begin() + static_cast<std::ptrdiff_t>(q + sub));
          q += sub;
        }
      }
      const std::vector<uint8_t> hdr = xl_header(&f[off], body.size() + 16);
      g.insert(g.end(), hdr.begin(), hdr.end());
      g.insert(g.end(), body.begin(), body.end());
    }
    // The decoder reads the colour space from the colour specification box in there.
    expect_decodes(g, planes, name + " every box but the signature box with XLBox");
  }

  // Malformed jp2c lengths.
  const uint64_t bad_xl[] = {0,
                             1,
                             8,
                             15,
                             code_len + 17,
                             code_len + 1000,
                             0x100000000ull + code_len + 16,
                             0x7FFFFFFFFFFFFFFFull,
                             0xFFFFFFFFFFFFFFF0ull,
                             0xFFFFFFFFFFFFFFFFull};
  for (uint64_t x : bad_xl)
    expect_rejected(with_header(f, jp2c, xl_header(&f[jp2c], x)), planes,
                    name + " jp2c with XLBox " + std::to_string(static_cast<unsigned long long>(x)));
  for (uint32_t l = 2; l < 8; ++l) {
    std::vector<uint8_t> g = f;
    put_be32(&g[jp2c], l);
    expect_rejected(g, planes, name + " jp2c with LBox " + std::to_string(l));
  }
  // LBox = 1 with the file ending inside XLBox (or right after the box type).
  for (size_t keep = 8; keep < 16; ++keep) {
    std::vector<uint8_t> g = with_header(f, jp2c, xl_header(&f[jp2c], code_len + 16));
    g.resize(jp2c + keep);
    expect_rejected(g, planes, name + " jp2c header cut after " + std::to_string(keep) + " bytes");
  }
  // The JP2 header box with an XLBox past the end of the file: no codestream box is found.
  expect_rejected(with_header(f, jp2h, xl_header(&f[jp2h], f.size() + 9)), planes,
                  name + " jp2h with XLBox past the end of the file");
}

}  // namespace

int main() {
  check_writer();
  try {
    check_file(false);
    check_file(true);
  } catch (std::exception &e) {
    check(false, std::string("exception: ") + e.what());
  }
  if (failures) {
    printf("jph_box_check: %d case(s) failed\n", failures);
    return 1;
  }
  printf("jph_box_check: all cases passed\n");
  return 0;
}
