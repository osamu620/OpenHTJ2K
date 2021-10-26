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

#include <cstdio>
#include "decoder.hpp"
#include <sys/stat.h>
#include "coding_units.hpp"
#include "ThreadPool.hpp"
#ifdef _OPENMP
  #include <omp.h>
#endif
namespace open_htj2k {
off_t get_file_size(const std::string &filename) {
  struct stat st;
  if (stat(filename.c_str(), &st) != 0) {
    throw new std::exception;
  }
  return st.st_size;
}

class openhtj2k_decoder_impl {
 private:
  j2c_src_memory in;
  const uint8_t reduce_NL;

 public:
  openhtj2k_decoder_impl(const char *, uint8_t reduce_NL, uint32_t num_threads);
  ~openhtj2k_decoder_impl();
  void invoke(std::vector<int32_t *> &, std::vector<uint32_t> &, std::vector<uint32_t> &,
              std::vector<uint8_t> &, std::vector<bool> &);
};

openhtj2k_decoder_impl::openhtj2k_decoder_impl(const char *filename, const uint8_t r, uint32_t num_threads)
    : reduce_NL(r) {
  off_t file_size = get_file_size(filename);  // supports 32-bits file size
  struct stat st;
  if (stat(filename, &st) != 0) {
    printf("ERROR: input file %s is not found.\n", filename);
    throw new std::exception;
  }
  ThreadPool::instance(num_threads);
  // open codestream and store it in memory
  FILE *fp = fopen(filename, "rb");
  in.alloc_memory(static_cast<uint32_t>(file_size));
  uint8_t *p = in.get_buf_pos();
  fread(p, sizeof(uint8_t), static_cast<uint32_t>(file_size), fp);
  fclose(fp);
}

void openhtj2k_decoder_impl::invoke(std::vector<int32_t *> &buf, std::vector<uint32_t> &width,
                                    std::vector<uint32_t> &height, std::vector<uint8_t> &depth,
                                    std::vector<bool> &is_signed) {
  // read main header
  j2k_main_header main_header;
  main_header.read(in);
  in.rewind_2bytes();
  element_siz numTiles;
  main_header.get_number_of_tiles(numTiles.x, numTiles.y);
  printf("Tile num x = %d, y = %d\n", numTiles.x, numTiles.y);

  // create output buffer
  uint16_t num_components = main_header.SIZ->get_num_components();
  std::vector<uint32_t> x0(num_components), x1(num_components), y0(num_components), y1(num_components);
  element_siz siz, Osiz, Tsiz, TOsiz, Rsiz;
  main_header.SIZ->get_image_size(siz);
  main_header.SIZ->get_image_origin(Osiz);
  main_header.SIZ->get_tile_size(Tsiz);
  main_header.SIZ->get_tile_origin(TOsiz);
  for (uint16_t c = 0; c < num_components; c++) {
    main_header.SIZ->get_subsampling_factor(Rsiz, c);
    x0[c] = ceil_int(Osiz.x, Rsiz.x);
    x1[c] = ceil_int(siz.x, Rsiz.x);
    y0[c] = ceil_int(Osiz.y, Rsiz.y);
    y1[c] = ceil_int(siz.y, Rsiz.y);
    width.push_back(ceil_int(x1[c] - x0[c], (1 << reduce_NL)));
    height.push_back(ceil_int(y1[c] - y0[c], (1 << reduce_NL)));
    buf.emplace_back(new int32_t[width[c] * height[c]]);
    depth.push_back(main_header.SIZ->get_bitdepth(c) - 0);
    is_signed.push_back(main_header.SIZ->is_signed(c));
  }

  auto tileSet = std::make_unique<j2k_tile[]>(numTiles.x * numTiles.y);
  for (uint32_t i = 0; i < numTiles.x * numTiles.y; ++i) {
    tileSet[i].dec_init(i, main_header, reduce_NL);
  }

  uint16_t word;
  SOT_marker tmpSOT;
  uint16_t tile_index;
  // read all tile parts
  while ((word = in.get_word()) != _EOC) {
    if (word != _SOT) {
      printf("ERROR: SOT marker segment expected but %04X is found\n", word);
      throw new std::exception;
    }
    tmpSOT     = SOT_marker(in);
    tile_index = tmpSOT.get_tile_index();
    tileSet[tile_index].add_tile_part(tmpSOT, in, main_header);
  }

  // read codestream and decode
  //#pragma omp parallel for
  for (uint32_t i = 0; i < numTiles.x * numTiles.y; i++) {
    tileSet[i].create_tile_buf(main_header);
    tileSet[i].decode(main_header);
    tileSet[i].ycbcr_to_rgb(main_header);
    tileSet[i].finalize(main_header);
  }

  // prepare output buffer
#pragma omp parallel for
  for (uint16_t c = 0; c < num_components; c++) {
    for (uint32_t i = 0; i < numTiles.x * numTiles.y; i++) {
      j2k_tile_component *tile_component = tileSet[i].get_tile_component(c);
      element_siz component_pos          = tile_component->get_pos0();
      element_siz component_size;
      tile_component->get_size(component_size);
      uint32_t component_width  = component_size.x;
      uint32_t component_height = component_size.y;
      uint32_t x_offset         = component_pos.x - ceil_int(x0[c], (1 << reduce_NL));
      uint32_t y_offset         = component_pos.y - ceil_int(y0[c], (1 << reduce_NL));
      for (uint32_t y = 0; y < component_height; y++) {
        int32_t *sp = tile_component->get_sample_address(0, y);
        memcpy(&buf[c][(0 + x_offset) + (y + y_offset) * width[c]], sp, sizeof(int32_t) * component_width);
      }
    }
  }
}

openhtj2k_decoder_impl::~openhtj2k_decoder_impl() = default;

// public interface
openhtj2k_decoder::openhtj2k_decoder(const char *fname, const uint8_t reduce_NL, uint32_t num_threads) {
  this->impl = std::make_unique<openhtj2k_decoder_impl>(fname, reduce_NL, num_threads);
}
void openhtj2k_decoder::invoke(std::vector<int32_t *> &buf, std::vector<uint32_t> &width,
                               std::vector<uint32_t> &height, std::vector<uint8_t> &depth,
                               std::vector<bool> &is_signed) {
  this->impl->invoke(buf, width, height, depth, is_signed);
}
openhtj2k_decoder::~openhtj2k_decoder() = default;
}  // namespace open_htj2k
