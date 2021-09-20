// Copyright (c) 2019 - 2021, Osamu Watanabe
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

#include <cstring>
#include <fstream>
#include <j2kmarkers.hpp>
#include "encoder.hpp"
#include "coding_units.hpp"
#include "jph.hpp"
#include "ThreadPool.hpp"

#define NO_QFACTOR 0xFF

#define sRGB 0
#define sYCC 1

namespace open_htj2k {
// openhtj2k_encoder_impl shall not be public
class openhtj2k_encoder_impl {
 private:
  std::string outfile;
  const std::vector<int32_t *> *buf;
  siz_params *siz;
  cod_params *cod;
  qcd_params *qcd;
  uint8_t qfactor;
  bool isJPH;
  uint8_t color_space;

 public:
  openhtj2k_encoder_impl(const char *, const std::vector<int32_t *> &, siz_params &, cod_params &,
                         qcd_params &, uint8_t, bool, uint8_t);
  ~openhtj2k_encoder_impl();
  size_t invoke();
};

openhtj2k_encoder_impl::openhtj2k_encoder_impl(const char *filename,
                                               const std::vector<int32_t *> &input_buf, siz_params &s,
                                               cod_params &c, qcd_params &q, uint8_t qf, bool flag,
                                               uint8_t cs)
    : buf(&input_buf), siz(&s), cod(&c), qcd(&q), qfactor(qf), isJPH(flag), color_space(cs) {
  this->outfile = filename;
}

openhtj2k_encoder_impl::~openhtj2k_encoder_impl() = default;

size_t openhtj2k_encoder_impl::invoke() {
  std::vector<uint8_t> Ssiz;
  std::vector<uint8_t> XRsiz, YRsiz;

  if ((siz->XOsiz > siz->Xsiz) || (siz->YOsiz > siz->Ysiz)) {
    printf("ERROR: image origin exceeds the size of input image.\n");
    exit(EXIT_FAILURE);
  }
  if ((siz->XTOsiz > siz->XOsiz) || (siz->YTOsiz > siz->YOsiz)) {
    printf("ERROR: tile origin shall be no greater than the image origin.\n");
    exit(EXIT_FAILURE);
  }
  if (siz->XTsiz * siz->YTsiz == 0) {
    siz->XTsiz = siz->Xsiz;
    siz->YTsiz = siz->Ysiz;
  }
  if (((siz->XTOsiz + siz->XTsiz) <= siz->XOsiz) || ((siz->YTOsiz + siz->YTsiz) <= siz->YOsiz)) {
    printf("ERROR: tile size plus tile origin shall be greater than the image origin.\n");
    exit(EXIT_FAILURE);
  }

  for (auto c = 0; c < siz->Csiz; ++c) {
    Ssiz.push_back(siz->Ssiz[c]);
    XRsiz.push_back(siz->XRsiz[c]);
    YRsiz.push_back(siz->YRsiz[c]);
  }

  // check component size
  if (siz->Csiz == 3 && cod->use_color_trafo == 1 && (XRsiz[0] != XRsiz[1] || XRsiz[1] != XRsiz[2])
      && (YRsiz[0] != YRsiz[1] || YRsiz[1] != YRsiz[2])) {
    cod->use_color_trafo = 0;
    printf("WARNING: Cycc is set to 'no' because size of each component is not identical.\n");
  }

  // check number of components
  if (siz->Csiz != 3 && cod->use_color_trafo == 1) {
    cod->use_color_trafo = 0;
    printf("WARNING: Cycc is set to 'no' because the number of components is not equal to 3.\n");
  }
  // force RGB->YCbCr when Qfactor feature is enabled
  if (qfactor != NO_QFACTOR) {
    if (siz->Csiz == 3) {
      if (cod->use_color_trafo == 0) {
        printf("WARNING: Color conversion is OFF while Qfactor feature is enabled.\n");
        printf("         It is OK if the inputs are in YCbCr color space.\n");
      }
    } else if (siz->Csiz != 1) {
      printf("WARNING: Qfactor is designed for only gray-scale or RGB or YCbCr input.\n");
    }
  }

  // create required marker segments
  SIZ_marker main_SIZ(siz->Rsiz, siz->Xsiz, siz->Ysiz, siz->XOsiz, siz->YOsiz, siz->XTsiz, siz->YTsiz,
                      siz->XTOsiz, siz->YTOsiz, siz->Csiz, Ssiz, XRsiz, YRsiz, true);
  COD_marker main_COD(cod->is_max_precincts, cod->use_SOP, cod->use_EPH, cod->progression_order,
                      cod->number_of_layers, cod->use_color_trafo, cod->dwt_levels, cod->blkwidth,
                      cod->blkheight, cod->codeblock_style, cod->transformation, cod->PPx, cod->PPy);
  QCD_marker main_QCD(qcd->number_of_guardbits, cod->dwt_levels, cod->transformation, qcd->is_derived,
                      Ssiz[0], cod->use_color_trafo, qcd->base_step, qfactor);
  // parameters for CAP marker
  uint16_t bits14_15 = 0;                     // 0: HTONLY, 2: HTDECLARED, 3: MIXED
  uint16_t bit13     = 0;                     // 0: SINGLEHT, 1: MULTIHT
  uint16_t bit12     = 0;                     // 0: RGNFREE, 1: RGN
  uint16_t bit11     = 0;                     // 0: HOMOGENEOUS, 1: HETEROGENEOUS
  uint16_t bit5      = !cod->transformation;  // 0: HTREV, 1: HTIRV
  uint16_t bits0_4;
  uint8_t MAGB = main_QCD.get_MAGB();
  if (MAGB < 27) {
    bits0_4 = (MAGB > 8) ? MAGB - 8 : 0;
  } else if (MAGB <= 71) {
    bits0_4 = (MAGB - 27) / 4 + 19;
  } else {
    bits0_4 = 31;
  }
  uint16_t Ccap15 =
      (bits14_15 << 14) + (bit13 << 13) + (bit12 << 12) + (bit11 << 11) + (bit5 << 5) + bits0_4;
  CAP_marker main_CAP;
  main_CAP.set_Ccap(Ccap15, 15);

  // create main header
  j2k_main_header main_header(&main_SIZ, &main_COD, &main_QCD, &main_CAP, qfactor);
  COM_marker main_COM("OpenHTJ2K version 0", true);
  main_header.add_COM_marker(main_COM);

  j2c_dst_memory j2c_dst, jph_dst;
  j2c_dst.put_word(_SOC);
  main_header.flush(j2c_dst);

  element_siz numTiles;
  main_header.get_number_of_tiles(numTiles.x, numTiles.y);

  auto tileSet = std::make_unique<j2k_tile[]>(numTiles.x * numTiles.y);
  for (uint32_t i = 0; i < numTiles.x * numTiles.y; ++i) {
    tileSet[i].enc_init(i, main_header, *buf);
  }
  for (uint32_t i = 0; i < numTiles.x * numTiles.y; ++i) {
    tileSet[i].perform_dc_offset(main_header);
    tileSet[i].rgb_to_ycbcr(main_header);
    tileSet[i].encode(main_header);
    tileSet[i].construct_packets(main_header);
  }
  for (uint32_t i = 0; i < numTiles.x * numTiles.y; ++i) {
    tileSet[i].write_packets(j2c_dst);
  }
  j2c_dst.put_word(_EOC);
  uint32_t codestream_size = j2c_dst.get_length();

  // prepare jph box-based format, if necessary
  if (isJPH) {
    bool isSRGB = (color_space == sRGB) ? true : false;
    jph_boxes jph_info(main_header, 1, isSRGB, codestream_size);
    size_t file_format_size = jph_info.write(jph_dst);
    codestream_size += file_format_size - codestream_size;
  }
  std::ofstream dst;
  dst.open(this->outfile, std::ios::out | std::ios::binary);
  if (isJPH) {
    jph_dst.flush(dst);
  }
  j2c_dst.flush(dst);
  dst.close();
  return codestream_size;
}

// public interface
openhtj2k_encoder::openhtj2k_encoder(const char *fname, const std::vector<int32_t *> &input_buf,
                                     siz_params &siz, cod_params &cod, qcd_params &qcd, uint8_t qfactor,
                                     bool isJPH, uint8_t color_space, uint32_t num_threads) {
  if (qfactor != NO_QFACTOR) {
    if (qfactor > 100) {
      printf("Value of Qfactor shall be in the range [0, 100]\n");
      exit(EXIT_FAILURE);
    }
  }
  ThreadPool::instance(num_threads);
  this->impl = std::make_unique<openhtj2k_encoder_impl>(fname, input_buf, siz, cod, qcd, qfactor, isJPH,
                                                        color_space);
}
size_t openhtj2k_encoder::invoke() { return this->impl->invoke(); }
openhtj2k_encoder::~openhtj2k_encoder() = default;
}  // namespace open_htj2k
