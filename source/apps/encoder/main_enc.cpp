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

// open_htj2k_enc: An encoder implementation of ITU-T Rec. 814 | ISO/IEC 15444-15
// (a.k.a HTJ2K)
//
// This software is currently compliant to limited part of the standard.
// Supported markers: SIZ, CAP, COD, QCD, COM. Other features are undone and future work.
// (c) 2021 Osamu Watanabe, Takushoku University

#include <chrono>
#include <vector>
#include "encoder.hpp"
#include "image.hpp"
#include "enc_utils.hpp"

int main(int argc, char *argv[]) {
  j2k_argset args(argc, argv);  // parsed command line

  image img(args.get_infile());  // input image
  element_siz_local image_origin = args.get_origin();
  element_siz_local image_size(img.get_width(), img.get_height());

  uint32_t num_components = img.get_num_components();
  std::vector<int32_t *> input_buf;
  for (auto c = 0; c < num_components; ++c) {
    input_buf.push_back(img.get_buf(c));
  }
  uint8_t img_depth;
  img_depth                     = img.get_bpp(0);  // suppose all components have the same bit-depth
  std::string out_filename      = args.get_outfile();
  element_siz_local tile_size   = args.get_tile_size();
  element_siz_local tile_origin = args.get_tile_origin();
  open_htj2k::siz_params siz;  // information of input image
  siz.Rsiz   = 0;
  siz.Xsiz   = image_size.x;
  siz.Ysiz   = image_size.y;
  siz.XOsiz  = image_origin.x;
  siz.YOsiz  = image_origin.y;
  siz.XTsiz  = tile_size.x;
  siz.YTsiz  = tile_size.y;
  siz.XTOsiz = tile_origin.x;
  siz.YTOsiz = tile_origin.y;
  siz.Csiz   = num_components;
  siz.bpp    = img_depth;

  open_htj2k::cod_params cod;  // parameters related to COD marker
  element_siz_local cblk_size       = args.get_cblk_size();
  cod.blkwidth                      = cblk_size.x;
  cod.blkheight                     = cblk_size.y;
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
    cod.PPx.push_back(i.x);
    cod.PPy.push_back(i.y);
  }

  open_htj2k::qcd_params qcd;  // parameters related to QCD marker
  qcd.is_derived          = args.is_derived();
  qcd.number_of_guardbits = args.get_num_guard();
  qcd.base_step           = args.get_basestep_size();
  if (qcd.base_step == 0.0) {
    qcd.base_step = 1.0f / static_cast<float>(1 << siz.bpp);
  }
  size_t total_size;
  int32_t num_iterations = args.get_num_iteration();
  auto start             = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < num_iterations; ++i) {
    // create encoder
    open_htj2k::openhtj2k_encoder encoder(args.get_outfile().c_str(), input_buf, siz, cod, qcd,
                                          args.get_qfactor());

    // invoke encoding
    total_size = encoder.invoke();
  }
  auto duration = std::chrono::high_resolution_clock::now() - start;
  auto count    = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
  double time   = count / 1000.0 / static_cast<double>(num_iterations);
  double bpp    = (double)total_size * 8 / (img.get_width() * img.get_height());

  // show stats
  printf("Codestream bytes  = %d = %f [bits/pixel]\n", total_size, bpp);
  printf("elapsed time %-15.3lf[ms]\n", time);
  return EXIT_SUCCESS;
}