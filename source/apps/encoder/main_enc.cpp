// Copyright (c) 2019 - 2022, Osamu Watanabe
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
//    modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
//    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
//    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
//    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
//    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// open_htj2k_enc: An encoder implementation of ITU-T Rec. 814 | ISO/IEC 15444-15
// (a.k.a HTJ2K)
//
// This software is currently compliant to limited part of the standard.
// Supported markers: SIZ, CAP, COD, QCD, QCC, COM. Other features are undone and future work.
// (c) 2021 Osamu Watanabe, Takushoku University
// (c) 2022 Osamu Watanabe, Takushoku University

#include <chrono>
#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || __cplusplus >= 201703L)
  #include <filesystem>
#else
  #include <sys/stat.h>
#endif
#include <cstdio>
#include <functional>
#include <vector>
#include <exception>
#include "encoder.hpp"
#include "enc_utils.hpp"

// ---------------------------------------------------------------------------
// PnmStreamReader — reads a PNM (P5/P6) file one row at a time
// ---------------------------------------------------------------------------
class PnmStreamReader {
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
  ~PnmStreamReader() {
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
    // Skip exactly one whitespace byte after maxval
    fgetc(fp_);

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

  uint32_t get_width() const { return width_; }
  uint32_t get_height() const { return height_; }
  uint16_t get_num_components() const { return nc_; }
  uint8_t get_bitdepth() const { return bitdepth_; }

  // Fill rows[0..nc-1][0..width-1] with int32 pixels for the given absolute row y.
  // Seeks to the correct file position so multi-tile images are handled correctly.
  void get_row(uint32_t y, int32_t **rows, uint16_t /*nc*/) {
    const long row_bytes = static_cast<long>(raw_.size());
    fseek(fp_, data_start_ + y * row_bytes, SEEK_SET);
    fread(raw_.data(), 1, raw_.size(), fp_);
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

int main(int argc, char *argv[]) {
  j2k_argset args(argc, argv);  // parsed command line
  std::vector<std::string> fnames = args.ifnames;
  for (const auto &fname : fnames) {
#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || __cplusplus >= 201703L)
    try {
      if (!std::filesystem::exists(fname)) {
        throw std::exception();
      }
    }
#else
    try {
      struct stat st;
      if (stat(fname.c_str(), &st)) {
        throw std::exception();
      }
    }
#endif
    catch (std::exception &exc) {
      printf("ERROR: File %s is not found.\n", fname.c_str());
      return EXIT_FAILURE;
    }
  }
  bool isJPH               = false;
  std::string out_filename = args.ofname;
  bool toFile              = true;
  if (out_filename.empty()) {
    toFile = false;
  } else {
    std::string::size_type pos = out_filename.find_last_of('.');
    std::string fext           = out_filename.substr(pos, 4);
    if (fext == ".jph" || fext == ".JPH") {
      isJPH = true;
    } else if (fext.compare(".j2c") && fext.compare(".j2k") && fext.compare(".jphc") && fext.compare(".J2C")
               && fext.compare(".J2K") && fext.compare(".JPHC")) {
      printf("ERROR: invalid extension for output file\n");
      exit(EXIT_FAILURE);
    }
  }

  size_t total_size      = 0;
  int32_t num_iterations = args.num_iteration;
  uint32_t stat_width = 0, stat_height = 0;
  // memory buffer for output codestream/file
  std::vector<uint8_t> outbuf;
  auto start = std::chrono::high_resolution_clock::now();

  if (args.line_based) {
    // ── Streaming path: read rows on demand; never allocate the full image ──
    PnmStreamReader reader;
    if (reader.open(fnames[0]) != 0) {
      printf("ERROR: Failed to open input file for streaming: %s\n", fnames[0].c_str());
      return EXIT_FAILURE;
    }
    element_siz_local image_origin_s = args.get_origin();
    element_siz_local image_size_s(reader.get_width(), reader.get_height());
    stat_width  = reader.get_width();
    stat_height = reader.get_height();
    uint16_t nc_s = reader.get_num_components();
    uint8_t bd_s  = reader.get_bitdepth();

    element_siz_local tile_size_s   = args.get_tile_size();
    element_siz_local tile_origin_s = args.get_tile_origin();
    if (image_origin_s.x != 0 && tile_origin_s.x == 0) tile_origin_s.x = image_origin_s.x;
    if (image_origin_s.y != 0 && tile_origin_s.y == 0) tile_origin_s.y = image_origin_s.y;

    open_htj2k::siz_params siz_s;
    siz_s.Rsiz   = 0;
    siz_s.Xsiz   = image_size_s.x + image_origin_s.x;
    siz_s.Ysiz   = image_size_s.y + image_origin_s.y;
    siz_s.XOsiz  = image_origin_s.x;
    siz_s.YOsiz  = image_origin_s.y;
    siz_s.XTsiz  = tile_size_s.x;
    siz_s.YTsiz  = tile_size_s.y;
    siz_s.XTOsiz = tile_origin_s.x;
    siz_s.YTOsiz = tile_origin_s.y;
    siz_s.Csiz   = nc_s;
    for (uint16_t c = 0; c < nc_s; ++c) {
      siz_s.Ssiz.push_back(static_cast<uint8_t>(bd_s - 1));  // unsigned: Ssiz = bitdepth - 1
      siz_s.XRsiz.push_back(1);
      siz_s.YRsiz.push_back(1);
    }

    open_htj2k::cod_params cod_s;
    element_siz_local cblk_size_s       = args.get_cblk_size();
    cod_s.blkwidth                      = static_cast<uint16_t>(cblk_size_s.x);
    cod_s.blkheight                     = static_cast<uint16_t>(cblk_size_s.y);
    cod_s.is_max_precincts              = args.is_max_precincts();
    cod_s.use_SOP                       = args.is_use_sop();
    cod_s.use_EPH                       = args.is_use_eph();
    cod_s.progression_order             = args.get_progression();
    cod_s.number_of_layers              = 1;
    cod_s.use_color_trafo               = args.get_ycc();
    cod_s.dwt_levels                    = args.get_dwt_levels();
    cod_s.codeblock_style               = 0x040;
    cod_s.transformation                = args.get_transformation();
    std::vector<element_siz_local> PP_s = args.get_prct_size();
    for (auto &i : PP_s) {
      cod_s.PPx.push_back(static_cast<unsigned char>(i.x));
      cod_s.PPy.push_back(static_cast<unsigned char>(i.y));
    }

    open_htj2k::qcd_params qcd_s{};
    qcd_s.is_derived          = args.is_derived();
    qcd_s.number_of_guardbits = args.get_num_guard();
    qcd_s.base_step           = args.get_basestep_size();
    if (qcd_s.base_step == 0.0) {
      qcd_s.base_step = 1.0f / static_cast<float>(1 << bd_s);
    }

    std::vector<int32_t *> empty_input;
    uint8_t color_space_s = args.jph_color_space;
    for (int iter = 0; iter < num_iterations; ++iter) {
      open_htj2k::openhtj2k_encoder encoder(out_filename.c_str(), empty_input, siz_s, cod_s, qcd_s,
                                            args.get_qfactor(), isJPH, color_space_s, args.num_threads);
      if (!toFile) encoder.set_output_buffer(outbuf);
      try {
        total_size = encoder.invoke_line_based_stream(
            [&reader](uint32_t y, int32_t **rows, uint16_t nc) { reader.get_row(y, rows, nc); });
      } catch (std::exception &exc) {
        return EXIT_FAILURE;
      }
    }
  } else {
    // ── Buffered path: load the full image then encode ──
    auto fstart = std::chrono::high_resolution_clock::now();
    open_htj2k::image img(fnames);
    auto fduration = std::chrono::high_resolution_clock::now() - fstart;
    auto fcount    = std::chrono::duration_cast<std::chrono::microseconds>(fduration).count();
    double ftime   = static_cast<double>(fcount) / 1000.0;
    printf("elapsed time for reading inputs %-15.3lf[ms]\n", ftime);
    auto fbytes = img.get_width() * img.get_height() * img.get_num_components() * 2;
    printf("%f [MB/s]\n", (double)fbytes / ftime / 1000);

    stat_width  = img.get_width();
    stat_height = img.get_height();

    element_siz_local image_origin = args.get_origin();
    element_siz_local image_size(img.get_width(), img.get_height());
    uint16_t num_components = img.get_num_components();
    std::vector<int32_t *> input_buf;
    for (uint16_t c = 0; c < num_components; ++c) input_buf.push_back(img.get_buf(c));

    element_siz_local tile_size   = args.get_tile_size();
    element_siz_local tile_origin = args.get_tile_origin();
    if (image_origin.x != 0 && tile_origin.x == 0) tile_origin.x = image_origin.x;
    if (image_origin.y != 0 && tile_origin.y == 0) tile_origin.y = image_origin.y;

    open_htj2k::siz_params siz;
    siz.Rsiz   = 0;
    siz.Xsiz   = image_size.x + image_origin.x;
    siz.Ysiz   = image_size.y + image_origin.y;
    siz.XOsiz  = image_origin.x;
    siz.YOsiz  = image_origin.y;
    siz.XTsiz  = tile_size.x;
    siz.YTsiz  = tile_size.y;
    siz.XTOsiz = tile_origin.x;
    siz.YTOsiz = tile_origin.y;
    siz.Csiz   = num_components;
    for (uint16_t c = 0; c < siz.Csiz; ++c) {
      siz.Ssiz.push_back(img.get_Ssiz_value(c));
      auto compw = img.get_component_width(c);
      auto comph = img.get_component_height(c);
      siz.XRsiz.push_back(static_cast<unsigned char>(((siz.Xsiz - siz.XOsiz) + compw - 1) / compw));
      siz.YRsiz.push_back(static_cast<unsigned char>(((siz.Ysiz - siz.YOsiz) + comph - 1) / comph));
    }

    open_htj2k::cod_params cod;
    element_siz_local cblk_size       = args.get_cblk_size();
    cod.blkwidth                      = static_cast<uint16_t>(cblk_size.x);
    cod.blkheight                     = static_cast<uint16_t>(cblk_size.y);
    cod.is_max_precincts              = args.is_max_precincts();
    cod.use_SOP                       = args.is_use_sop();
    cod.use_EPH                       = args.is_use_eph();
    cod.progression_order             = args.get_progression();
    cod.number_of_layers              = 1;
    cod.use_color_trafo               = args.get_ycc();
    cod.dwt_levels                    = args.get_dwt_levels();
    cod.codeblock_style               = 0x040;
    cod.transformation                = args.get_transformation();
    std::vector<element_siz_local> PP = args.get_prct_size();
    for (auto &i : PP) {
      cod.PPx.push_back(static_cast<unsigned char>(i.x));
      cod.PPy.push_back(static_cast<unsigned char>(i.y));
    }

    open_htj2k::qcd_params qcd{};
    qcd.is_derived          = args.is_derived();
    qcd.number_of_guardbits = args.get_num_guard();
    qcd.base_step           = args.get_basestep_size();
    if (qcd.base_step == 0.0) {
      qcd.base_step = 1.0f / static_cast<float>(1 << img.get_max_bpp());
    }
    uint8_t color_space = args.jph_color_space;

    for (int i = 0; i < num_iterations; ++i) {
      open_htj2k::openhtj2k_encoder encoder(out_filename.c_str(), input_buf, siz, cod, qcd,
                                            args.get_qfactor(), isJPH, color_space, args.num_threads);
      if (!toFile) encoder.set_output_buffer(outbuf);
      try {
        total_size = encoder.invoke();
      } catch (std::exception &exc) {
        return EXIT_FAILURE;
      }
    }
  }

  auto duration = std::chrono::high_resolution_clock::now() - start;
  auto count    = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
  double time   = static_cast<double>(count) / 1000.0 / static_cast<double>(num_iterations);
  double bpp    = (double)total_size * 8 / (stat_width * stat_height);

  // show stats
  printf("Codestream bytes  = %zu = %f [bits/pixel]\n", total_size, bpp);
  printf("elapsed time %-15.3lf[ms]\n", time);
  return EXIT_SUCCESS;
}
