// Copyright (c) 2019 - 2026, Osamu Watanabe
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

#pragma once
#include <cstdint>
#include <cmath>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#ifndef OPENHTJ2K_NODISCARD
  #if ((defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || __cplusplus >= 201703L)
    #define OPENHTJ2K_NODISCARD [[nodiscard]]
    #define OPENHTJ2K_MAYBE_UNUSED [[maybe_unused]]
  #else
    #define OPENHTJ2K_NODISCARD
    #define OPENHTJ2K_MAYBE_UNUSED
  #endif
#endif

#if defined(_MSC_VER) && !defined(OHTJ2K_STATIC)
  #define OPENHTJ2K_EXPORT __declspec(dllexport)
#else
  #define OPENHTJ2K_EXPORT
#endif

#if defined(OPENHTJ2K_TIFF_SUPPORT)
  #include <tiffio.h>
#endif
namespace open_htj2k {
class image {
 private:
  uint32_t width;
  uint32_t height;
  uint16_t num_components;
  std::vector<uint32_t> component_width;
  std::vector<uint32_t> component_height;
  std::unique_ptr<std::unique_ptr<int32_t[]>[]> buf;
  std::vector<uint8_t> bits_per_pixel;
  std::vector<bool> is_signed;

 public:
OPENHTJ2K_EXPORT  explicit image(const std::vector<std::string> &filenames);
OPENHTJ2K_EXPORT  int read_pnmpgx(const std::string &filename, uint16_t nc);
  #if defined(OPENHTJ2K_TIFF_SUPPORT)
OPENHTJ2K_EXPORT  int read_tiff(const std::string &filename);
  #endif

  OPENHTJ2K_NODISCARD uint32_t get_width() const { return this->width; }
  OPENHTJ2K_NODISCARD uint32_t get_height() const { return this->height; }
  OPENHTJ2K_NODISCARD uint32_t get_component_width(uint16_t c) const {
    if (c > num_components) {
      printf("ERROR: component index %d is larger than maximum value %d.\n", c, num_components);
      throw std::exception();
    }
    return this->component_width[c];
  }
  OPENHTJ2K_NODISCARD uint32_t get_component_height(uint16_t c) const {
    if (c > num_components) {
      printf("ERROR: component index %d is larger than maximum value %d.\n", c, num_components);
      throw std::exception();
    }
    return this->component_height[c];
  }
  OPENHTJ2K_NODISCARD uint16_t get_num_components() const { return this->num_components; }
  uint8_t get_Ssiz_value(uint16_t c) {
    uint8_t val = static_cast<uint8_t>(this->bits_per_pixel[c] - 1);
    if (this->is_signed[c]) {
      val = val | 0x80;
    }
    return val;
  }
  uint8_t get_max_bpp() {
    uint8_t max = 0;
    for (auto &v : bits_per_pixel) {
      max = (max < v) ? v : max;
    }
    return max;
  }
  int32_t *get_buf(uint16_t c) { return this->buf[c].get(); }
};

struct siz_params {
  uint16_t Rsiz;
  uint32_t Xsiz;
  uint32_t Ysiz;
  uint32_t XOsiz;
  uint32_t YOsiz;
  uint32_t XTsiz;
  uint32_t YTsiz;
  uint32_t XTOsiz;
  uint32_t YTOsiz;
  uint16_t Csiz;
  std::vector<uint8_t> Ssiz;
  std::vector<uint8_t> XRsiz;
  std::vector<uint8_t> YRsiz;
  // uint8_t bpp;
};

struct cod_params {
  uint16_t blkwidth;
  uint16_t blkheight;
  bool is_max_precincts;
  bool use_SOP;
  bool use_EPH;
  uint8_t progression_order;
  uint16_t number_of_layers;
  uint8_t use_color_trafo;
  uint8_t dwt_levels;
  uint8_t codeblock_style;
  uint8_t transformation;
  std::vector<uint8_t> PPx, PPy;
};

struct qcd_params {
  uint8_t number_of_guardbits;
  bool is_derived;
  double base_step;
};

class openhtj2k_encoder {
 private:
  std::unique_ptr<class openhtj2k_encoder_impl> impl;

 public:
OPENHTJ2K_EXPORT openhtj2k_encoder(const char *, const std::vector<int32_t *> &input_buf, siz_params &siz, cod_params &cod,
                    qcd_params &qcd, uint8_t qfactor, bool isJPH, uint8_t color_space,
                    uint32_t num_threads);
OPENHTJ2K_EXPORT void set_output_buffer(std::vector<uint8_t> &output_buf);
OPENHTJ2K_EXPORT size_t invoke();
OPENHTJ2K_EXPORT size_t invoke_line_based();
OPENHTJ2K_EXPORT size_t invoke_line_based_stream(std::function<void(uint32_t, int32_t **, uint16_t)> src_fn);
OPENHTJ2K_EXPORT ~openhtj2k_encoder();
};
}  // namespace open_htj2k
