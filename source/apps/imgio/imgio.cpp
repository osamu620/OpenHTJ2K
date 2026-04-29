// Copyright (c) 2019 - 2026, Osamu Watanabe
// All rights reserved. (See imgio.hpp for the full license header.)

#include "imgio.hpp"

#include <cctype>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#if defined(OPENHTJ2K_TIFF_SUPPORT)
  #include <tiffio.h>
#endif

namespace imgio {

// ---------------------------------------------------------------------------
// PnmStreamReader — reads a PNM (P5/P6) file one row at a time
// ---------------------------------------------------------------------------
class PnmStreamReader : public StreamReader {
  FILE *fp_           = nullptr;
  uint32_t width_     = 0;
  uint32_t height_    = 0;
  uint16_t nc_        = 0;
  uint8_t bitdepth_   = 0;
  uint32_t bps_       = 0;  // bytes per sample
  bool isPPM_         = false;
  long data_start_    = 0;  // file offset of first pixel byte
  std::vector<uint8_t> raw_;

  // Skip whitespace and PNM comments
  static void skip_ws(FILE *fp) {
    int c;
    while ((c = fgetc(fp)) != EOF) {
      if (c == '#') {
        while ((c = fgetc(fp)) != EOF && c != '\n') {}
      } else if (c != ' ' && c != '\t' && c != '\n' && c != '\r') {
        ungetc(c, fp);
        break;
      }
    }
  }

  static uint32_t read_uint(FILE *fp) {
    skip_ws(fp);
    uint32_t v = 0;
    int c;
    while ((c = fgetc(fp)) != EOF && c >= '0' && c <= '9') v = v * 10 + (uint32_t)(c - '0');
    if (c != EOF) ungetc(c, fp);
    return v;
  }

 public:
  ~PnmStreamReader() override {
    if (fp_) {
      fclose(fp_);
      fp_ = nullptr;
    }
  }

  int open(const std::string &filename) {
    fp_ = fopen(filename.c_str(), "rb");
    if (!fp_) return -1;

    char magic[3] = {};
    if (fread(magic, 1, 2, fp_) != 2) return -1;
    if (magic[0] != 'P') return -1;
    if (magic[1] == '5') {
      isPPM_ = false;
      nc_    = 1;
    } else if (magic[1] == '6') {
      isPPM_ = true;
      nc_    = 3;
    } else {
      return -1;
    }

    width_  = read_uint(fp_);
    height_ = read_uint(fp_);
    uint32_t maxval = read_uint(fp_);
    // Per the netpbm PPM/PGM binary spec (P5/P6), the maxval field is followed
    // by EXACTLY ONE whitespace character; the very next byte starts the
    // raster.  read_uint() already ungetc'd the trailing whitespace, so we
    // consume that single byte here.  A previous version called skip_ws()
    // which kept eating bytes greedily — fine until the first pixel value
    // happened to be 0x09/0x0A/0x0D/0x20 (e.g. NASA Blue Marble crops where
    // R≈10), at which point the first 1–3 pixel bytes were misread as
    // "header whitespace" and every fread shifted, eventually overrunning
    // EOF on the last row and aborting with "fread: failed to read row data".
    {
      const int sep = fgetc(fp_);
      if (sep != ' ' && sep != '\t' && sep != '\n' && sep != '\r') return -1;
    }

    if (maxval == 0 || maxval > 65535) return -1;
    // Compute the exact bit depth from maxval (e.g. 4095 → 12, 255 → 8).
    uint8_t bd = 1;
    while ((1U << bd) <= maxval) bd++;
    bitdepth_ = bd;
    bps_      = (bitdepth_ > 8) ? 2 : 1;

    raw_.resize(static_cast<size_t>(width_) * nc_ * bps_);
    // Record the file position of the first pixel byte for random-access reads.
    data_start_ = ftell(fp_);
    return 0;
  }

  uint32_t get_width() const override { return width_; }
  uint32_t get_height() const override { return height_; }
  uint16_t get_num_components() const override { return nc_; }
  uint8_t  get_bitdepth(uint16_t /*c*/ = 0) const override { return bitdepth_; }
  // PNM is always unsigned; Ssiz = bitdepth - 1 with no sign bit.
  uint8_t  get_Ssiz(uint16_t /*c*/) const override { return static_cast<uint8_t>(bitdepth_ - 1); }

  // Fill rows[0..nc-1][0..width-1] with int32 pixels for the given absolute row y.
  // Seeks to the correct file position so multi-tile images are handled correctly.
  void get_row(uint32_t y, int32_t **rows, uint16_t /*nc*/) override {
    const long row_bytes = static_cast<long>(raw_.size());
    fseek(fp_, data_start_ + y * row_bytes, SEEK_SET);
    if (fread(raw_.data(), 1, raw_.size(), fp_) != raw_.size()) {
      throw std::runtime_error("fread: failed to read row data");
    }
    if (bps_ == 1) {
      if (isPPM_) {
        for (uint32_t x = 0; x < width_; ++x) {
          rows[0][x] = static_cast<int32_t>(raw_[x * 3 + 0]);
          rows[1][x] = static_cast<int32_t>(raw_[x * 3 + 1]);
          rows[2][x] = static_cast<int32_t>(raw_[x * 3 + 2]);
        }
      } else {
        for (uint32_t x = 0; x < width_; ++x) rows[0][x] = static_cast<int32_t>(raw_[x]);
      }
    } else {
      if (isPPM_) {
        for (uint32_t x = 0; x < width_; ++x) {
          rows[0][x] = static_cast<int32_t>((raw_[(x * 3 + 0) * 2] << 8) | raw_[(x * 3 + 0) * 2 + 1]);
          rows[1][x] = static_cast<int32_t>((raw_[(x * 3 + 1) * 2] << 8) | raw_[(x * 3 + 1) * 2 + 1]);
          rows[2][x] = static_cast<int32_t>((raw_[(x * 3 + 2) * 2] << 8) | raw_[(x * 3 + 2) * 2 + 1]);
        }
      } else {
        for (uint32_t x = 0; x < width_; ++x)
          rows[0][x] = static_cast<int32_t>((raw_[x * 2] << 8) | raw_[x * 2 + 1]);
      }
    }
  }
};

// ---------------------------------------------------------------------------
// PgxStreamReader — reads one PGX file per component, one row at a time
// ---------------------------------------------------------------------------
class PgxStreamReader : public StreamReader {
  struct Component {
    FILE    *fp          = nullptr;
    uint32_t width       = 0;
    uint32_t height      = 0;
    uint8_t  bitdepth    = 0;
    bool     isSigned    = false;
    bool     isBigendian = false;
    long     data_start  = 0;
    std::vector<uint8_t> raw;
  };
  std::vector<Component> comps_;
  std::vector<uint8_t>   xr_, yr_;

  static int parse_pgx_header(const std::string &fname, Component &c) {
    c.fp = fopen(fname.c_str(), "rb");
    if (!c.fp) return -1;
    // Expect "PG"
    if (fgetc(c.fp) != 'P' || fgetc(c.fp) != 'G') return -1;
    // Skip to byte-order indicator: 'M' (ML = big-endian) or 'L' (LM = little-endian)
    int d;
    do { d = fgetc(c.fp); } while (d != 'M' && d != 'L' && d != EOF);
    if (d == EOF) return -1;
    c.isBigendian = (d == 'M');
    fgetc(c.fp);  // consume the second letter of 'ML' or 'LM'
    // Skip to sign character or first digit of bit-depth
    do { d = fgetc(c.fp); } while (d != '+' && d != '-' && !isdigit(d) && d != EOF);
    if (d == EOF) return -1;
    if (d == '+' || d == '-') {
      c.isSigned = (d == '-');
      do { d = fgetc(c.fp); } while (!isdigit(d) && d != EOF);
      if (d == EOF) return -1;
    }
    // Read bit-depth
    uint32_t bd = 0;
    do { bd = bd * 10 + static_cast<uint32_t>(d - '0'); d = fgetc(c.fp); } while (isdigit(d));
    c.bitdepth = static_cast<uint8_t>(bd);
    // Skip whitespace, read width
    while (d == ' ' || d == '\n' || d == '\r') d = fgetc(c.fp);
    uint32_t w = 0;
    while (isdigit(d)) { w = w * 10 + static_cast<uint32_t>(d - '0'); d = fgetc(c.fp); }
    c.width = w;
    // Skip whitespace, read height
    while (d == ' ' || d == '\n' || d == '\r') d = fgetc(c.fp);
    uint32_t h = 0;
    while (isdigit(d)) { h = h * 10 + static_cast<uint32_t>(d - '0'); d = fgetc(c.fp); }
    c.height = h;
    // Per the PGX convention exactly ONE whitespace separates the header from
    // the binary raster — do NOT read further or we'll swallow data bytes that
    // happen to be 0x09/0x0a/0x0d/0x20 (e.g. files whose first sample's high
    // byte is 0x0d/CR shift every row by one and the last fread overruns EOF).
    // Same class of bug the PnmStreamReader was fixed for.
    if (d != ' ' && d != '\t' && d != '\n' && d != '\r') return -1;
    c.data_start = ftell(c.fp);
    c.raw.resize(static_cast<size_t>(c.width) * ((c.bitdepth > 8) ? 2u : 1u));
    return 0;
  }

 public:
  ~PgxStreamReader() override {
    for (auto &c : comps_) if (c.fp) { fclose(c.fp); c.fp = nullptr; }
  }

  int open(const std::vector<std::string> &fnames) {
    for (const auto &fname : fnames) {
      Component c;
      if (parse_pgx_header(fname, c) != 0) {
        printf("ERROR: Failed to parse PGX header: %s\n", fname.c_str());
        return -1;
      }
      comps_.push_back(std::move(c));
    }
    xr_.resize(comps_.size(), 1);
    yr_.resize(comps_.size(), 1);
    for (size_t i = 1; i < comps_.size(); ++i) {
      if (comps_[0].width % comps_[i].width != 0 || comps_[0].height % comps_[i].height != 0) {
        printf("ERROR: PGX component %zu dimensions (%ux%u) are not integer submultiples of"
               " component 0 (%ux%u).\n",
               i, comps_[i].width, comps_[i].height, comps_[0].width, comps_[0].height);
        return -1;
      }
      xr_[i] = static_cast<uint8_t>(comps_[0].width  / comps_[i].width);
      yr_[i] = static_cast<uint8_t>(comps_[0].height / comps_[i].height);
    }
    return 0;
  }

  uint32_t get_width() const override { return comps_.empty() ? 0 : comps_[0].width; }
  uint32_t get_height() const override { return comps_.empty() ? 0 : comps_[0].height; }
  uint16_t get_num_components() const override { return static_cast<uint16_t>(comps_.size()); }
  uint8_t  get_bitdepth(uint16_t c = 0) const override {
    return (c < comps_.size()) ? comps_[c].bitdepth : 0;
  }
  uint8_t  get_Ssiz(uint16_t c) const override {
    if (c >= comps_.size()) return 0;
    uint8_t bd = comps_[c].bitdepth;
    return static_cast<uint8_t>(comps_[c].isSigned ? ((bd - 1) | 0x80u) : (bd - 1));
  }
  uint8_t get_XRsiz(uint16_t c) const override { return (c < xr_.size()) ? xr_[c] : 1; }
  uint8_t get_YRsiz(uint16_t c) const override { return (c < yr_.size()) ? yr_[c] : 1; }

  void get_row(uint32_t y, int32_t **rows, uint16_t nc) override {
    for (uint16_t ci = 0; ci < nc && ci < static_cast<uint16_t>(comps_.size()); ++ci) {
      auto    &c         = comps_[ci];
      uint32_t bps       = (c.bitdepth > 8) ? 2u : 1u;
      long     row_bytes = static_cast<long>(c.width * bps);
      uint32_t file_y    = (ci < yr_.size()) ? (y / yr_[ci]) : y;
      fseek(c.fp, c.data_start + static_cast<long>(file_y) * row_bytes, SEEK_SET);
      if (fread(c.raw.data(), 1, c.raw.size(), c.fp) != c.raw.size())
        throw std::runtime_error("fread: failed to read PGX row data");
      if (bps == 1) {
        for (uint32_t x = 0; x < c.width; ++x)
          rows[ci][x] = c.isSigned ? static_cast<int32_t>(static_cast<int8_t>(c.raw[x]))
                                   : static_cast<int32_t>(c.raw[x]);
      } else {
        for (uint32_t x = 0; x < c.width; ++x) {
          uint8_t  b0    = c.raw[x * 2];
          uint8_t  b1    = c.raw[x * 2 + 1];
          uint16_t raw16 = c.isBigendian ? static_cast<uint16_t>((b0 << 8) | b1)
                                         : static_cast<uint16_t>((b1 << 8) | b0);
          rows[ci][x]    = c.isSigned ? static_cast<int32_t>(static_cast<int16_t>(raw16))
                                      : static_cast<int32_t>(raw16);
        }
      }
    }
  }
};

#if defined(OPENHTJ2K_TIFF_SUPPORT)
// ---------------------------------------------------------------------------
// TiffStreamReader — reads a basic TIFF (8/16-bit, RGB or grayscale,
// uncompressed or strip-compressed) one row at a time via libtiff's scanline
// API. Tiled TIFFs are rejected; for those use the buffered (-batch) path.
// Both PLANARCONFIG_CONTIG and PLANARCONFIG_SEPARATE are supported.
// ---------------------------------------------------------------------------
class TiffStreamReader : public StreamReader {
  TIFF    *tif_              = nullptr;
  uint32_t width_            = 0;
  uint32_t height_           = 0;
  uint16_t nc_               = 0;
  uint16_t bits_per_sample_  = 0;
  uint16_t planar_           = 0;
  uint32_t bytes_per_sample_ = 0;
  std::vector<uint8_t>              raw_;           // contig: one full scanline
  std::vector<std::vector<uint8_t>> raw_separate_;  // separate: one scanline per component

 public:
  ~TiffStreamReader() override {
    if (tif_) {
      TIFFClose(tif_);
      tif_ = nullptr;
    }
  }

  int open(const std::string &filename) {
    tif_ = TIFFOpen(filename.c_str(), "r");
    if (!tif_) return -1;

    if (TIFFIsTiled(tif_)) {
      printf("ERROR: tiled TIFF is not supported by the streaming reader; use -batch.\n");
      return -1;
    }

    uint16_t samples_per_pixel = 0;
    uint16_t photometric       = 0;
    TIFFGetField(tif_, TIFFTAG_IMAGEWIDTH,      &width_);
    TIFFGetField(tif_, TIFFTAG_IMAGELENGTH,     &height_);
    TIFFGetField(tif_, TIFFTAG_BITSPERSAMPLE,   &bits_per_sample_);
    TIFFGetField(tif_, TIFFTAG_SAMPLESPERPIXEL, &samples_per_pixel);
    TIFFGetField(tif_, TIFFTAG_PHOTOMETRIC,     &photometric);
    TIFFGetField(tif_, TIFFTAG_PLANARCONFIG,    &planar_);

    if (samples_per_pixel < 1) samples_per_pixel = 1;
    if (planar_ == 0)          planar_ = PLANARCONFIG_CONTIG;

    if (bits_per_sample_ != 8 && bits_per_sample_ != 16) {
      printf("ERROR: TIFF bit depth %u is not supported (8 or 16 only).\n", bits_per_sample_);
      return -1;
    }
    if (photometric != PHOTOMETRIC_RGB && photometric != PHOTOMETRIC_MINISBLACK) {
      printf("ERROR: TIFF photometric %u is not supported (RGB or grayscale only).\n", photometric);
      return -1;
    }

    nc_               = samples_per_pixel;
    bytes_per_sample_ = (bits_per_sample_ + 7u) / 8u;

    const tmsize_t scanline = TIFFScanlineSize(tif_);
    if (scanline <= 0) {
      printf("ERROR: TIFFScanlineSize returned %lld.\n", static_cast<long long>(scanline));
      return -1;
    }
    if (planar_ == PLANARCONFIG_SEPARATE) {
      raw_separate_.assign(nc_, std::vector<uint8_t>(static_cast<size_t>(scanline)));
    } else {
      raw_.assign(static_cast<size_t>(scanline), 0);
    }
    return 0;
  }

  uint32_t get_width()         const override { return width_; }
  uint32_t get_height()        const override { return height_; }
  uint16_t get_num_components() const override { return nc_; }
  uint8_t  get_bitdepth(uint16_t /*c*/ = 0) const override {
    return static_cast<uint8_t>(bits_per_sample_);
  }
  // TIFF samples are unsigned in this reader (matches the buffered TIFF path).
  uint8_t  get_Ssiz(uint16_t /*c*/) const override {
    return static_cast<uint8_t>(bits_per_sample_ - 1);
  }

  void get_row(uint32_t y, int32_t **rows, uint16_t /*nc*/) override {
    if (planar_ == PLANARCONFIG_SEPARATE) {
      for (uint16_t c = 0; c < nc_; ++c) {
        if (TIFFReadScanline(tif_, raw_separate_[c].data(), y, c) < 0) {
          throw std::runtime_error("TIFFReadScanline failed (PLANARCONFIG_SEPARATE)");
        }
      }
      for (uint16_t c = 0; c < nc_; ++c) {
        const uint8_t *src = raw_separate_[c].data();
        int32_t       *dst = rows[c];
        if (bytes_per_sample_ == 1) {
          for (uint32_t x = 0; x < width_; ++x) dst[x] = static_cast<int32_t>(src[x]);
        } else {
          // libtiff returns multi-byte samples in host byte order.
          for (uint32_t x = 0; x < width_; ++x) {
            uint16_t v;
            std::memcpy(&v, src + x * 2u, 2);
            dst[x] = static_cast<int32_t>(v);
          }
        }
      }
    } else {
      // PLANARCONFIG_CONTIG
      if (TIFFReadScanline(tif_, raw_.data(), y, 0) < 0) {
        throw std::runtime_error("TIFFReadScanline failed (PLANARCONFIG_CONTIG)");
      }
      const uint32_t stride = static_cast<uint32_t>(nc_) * bytes_per_sample_;
      if (bytes_per_sample_ == 1) {
        for (uint16_t c = 0; c < nc_; ++c) {
          int32_t       *dst = rows[c];
          const uint8_t *src = raw_.data() + c;
          for (uint32_t x = 0; x < width_; ++x) dst[x] = static_cast<int32_t>(src[x * stride]);
        }
      } else {
        for (uint16_t c = 0; c < nc_; ++c) {
          int32_t       *dst = rows[c];
          const uint8_t *src = raw_.data() + c * 2u;
          for (uint32_t x = 0; x < width_; ++x) {
            uint16_t v;
            std::memcpy(&v, src + x * stride, 2);
            dst[x] = static_cast<int32_t>(v);
          }
        }
      }
    }
  }
};
#endif  // OPENHTJ2K_TIFF_SUPPORT

// ---------------------------------------------------------------------------
// Factory: dispatch by first file's extension.
// ---------------------------------------------------------------------------
std::unique_ptr<StreamReader> open_stream_reader(const std::vector<std::string> &fnames) {
  if (fnames.empty()) {
    printf("ERROR: open_stream_reader called with no filenames.\n");
    return nullptr;
  }
  std::string ext;
  auto pos = fnames[0].find_last_of('.');
  if (pos != std::string::npos) {
    ext = fnames[0].substr(pos);
    for (auto &ch : ext) ch = static_cast<char>(tolower(ch));
  }

  if (ext == ".pgx") {
    auto r = std::unique_ptr<PgxStreamReader>(new PgxStreamReader());
    if (r->open(fnames) != 0) {
      printf("ERROR: Failed to open PGX input file(s) for streaming.\n");
      return nullptr;
    }
    return r;
  }
  if (ext == ".tif" || ext == ".tiff") {
#if defined(OPENHTJ2K_TIFF_SUPPORT)
    auto r = std::unique_ptr<TiffStreamReader>(new TiffStreamReader());
    if (r->open(fnames[0]) != 0) {
      printf("ERROR: Failed to open TIFF input file for streaming: %s\n", fnames[0].c_str());
      return nullptr;
    }
    return r;
#else
    printf("ERROR: TIFF support was not built (libtiff not found at configure time): %s\n",
           fnames[0].c_str());
    return nullptr;
#endif
  }
  // Default: PNM (P5/P6).
  auto r = std::unique_ptr<PnmStreamReader>(new PnmStreamReader());
  if (r->open(fnames[0]) != 0) {
    printf("ERROR: Failed to open input file for streaming: %s\n", fnames[0].c_str());
    return nullptr;
  }
  return r;
}

}  // namespace imgio
