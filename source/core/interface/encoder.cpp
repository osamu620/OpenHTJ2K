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
#include <j2kmarkers.hpp>
#include "encoder.hpp"
#include "coding_units.hpp"

namespace open_htj2k {
// openhtj2k_encoder_impl shall not be public
class openhtj2k_encoder_impl {
 private:
  std::string outfile;
  const std::vector<int32_t *> *buf;
  siz_params *siz;
  cod_params *cod;
  qcd_params *qcd;

 public:
  openhtj2k_encoder_impl(const char *, const std::vector<int32_t *> &, siz_params &, cod_params &,
                         qcd_params &);
  ~openhtj2k_encoder_impl();
  size_t invoke();
};

openhtj2k_encoder_impl::openhtj2k_encoder_impl(const char *filename,
                                               const std::vector<int32_t *> &input_buf, siz_params &s,
                                               cod_params &c, qcd_params &q)
    : buf(&input_buf), siz(&s), cod(&c), qcd(&q) {
  this->outfile = filename;
}

openhtj2k_encoder_impl::~openhtj2k_encoder_impl() = default;

size_t openhtj2k_encoder_impl::invoke() {
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
  std::vector<uint8_t> bit_depth;
  std::vector<uint8_t> XRsiz, YRsiz;
  for (auto c = 0; c < siz->Csiz; ++c) {
    bit_depth.push_back(siz->bpp);
    XRsiz.push_back(1);
    YRsiz.push_back(1);
  }
  if (siz->Csiz != 3 && cod->use_color_trafo == 1) {
    cod->use_color_trafo = 0;
    printf("WARNING: Cycc is set to 'no' because the number of components is not equal to 3.\n");
  }

  // create required marker segments
  SIZ_marker main_SIZ(siz->Rsiz, siz->Xsiz, siz->Ysiz, siz->XOsiz, siz->YOsiz, siz->XTsiz, siz->YTsiz,
                      siz->XTOsiz, siz->YTOsiz, siz->Csiz, bit_depth, XRsiz, YRsiz, false, true);
  COD_marker main_COD(cod->is_max_precincts, cod->use_SOP, cod->use_EPH, cod->progression_order,
                      cod->number_of_layers, cod->use_color_trafo, cod->dwt_levels, cod->blkwidth,
                      cod->blkheight, cod->codeblock_style, cod->transformation, cod->PPx, cod->PPy);
  QCD_marker main_QCD(qcd->number_of_guardbits, cod->dwt_levels, cod->transformation, qcd->is_derived,
                      bit_depth[0], cod->use_color_trafo, qcd->base_step);
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
  j2k_main_header main_header(&main_SIZ, &main_COD, &main_QCD, &main_CAP);
  COM_marker main_COM("OpenHTJ2K version 0", true);
  main_header.add_COM_marker(main_COM);

  j2c_dst_memory j2c_dst;
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
  uint32_t codestream_size = j2c_dst.flush(this->outfile);
  return codestream_size;
}

// public interface
openhtj2k_encoder::openhtj2k_encoder(const char *fname, const std::vector<int32_t *> &input_buf,
                                     siz_params &siz, cod_params &cod, qcd_params &qcd) {
  this->impl = std::make_unique<openhtj2k_encoder_impl>(fname, input_buf, siz, cod, qcd);
}
size_t openhtj2k_encoder::invoke() { return this->impl->invoke(); }
openhtj2k_encoder::~openhtj2k_encoder() = default;
}  // namespace open_htj2k
