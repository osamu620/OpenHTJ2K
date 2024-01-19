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
#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || __cplusplus >= 201703L)
  #include <filesystem>
#else
  #include <sys/stat.h>
#endif
#include "coding_units.hpp"
#include "ThreadPool.hpp"
#ifdef _OPENMP
  #include <omp.h>
#endif
namespace open_htj2k {
class openhtj2k_decoder_impl {
 private:
  j2c_src_memory in;
  uint8_t reduce_NL;
  bool is_codestream_set;
  bool is_parsed;
  j2k_main_header main_header;

 public:
  openhtj2k_decoder_impl();
  openhtj2k_decoder_impl(const char *, uint8_t reduce_NL, uint32_t num_threads);
  openhtj2k_decoder_impl(const uint8_t *, size_t, uint8_t reduce_NL, uint32_t num_threads);
  ~openhtj2k_decoder_impl();
  void init(const uint8_t *, size_t, uint8_t reduce_NL, uint32_t num_threads);
  void parse();
  uint16_t get_num_component();
  uint32_t get_component_width(uint16_t);
  uint32_t get_component_height(uint16_t);
  uint8_t get_component_depth(uint16_t);
  bool get_component_signedness(uint16_t);
  uint8_t get_minimum_DWT_levels();

  void invoke(std::vector<int32_t *> &, std::vector<uint32_t> &, std::vector<uint32_t> &,
              std::vector<uint8_t> &, std::vector<bool> &);

  void destroy();
};

openhtj2k_decoder_impl::openhtj2k_decoder_impl() {
  reduce_NL         = 0;
  is_codestream_set = false;
  is_parsed         = false;
}

openhtj2k_decoder_impl::openhtj2k_decoder_impl(const char *filename, const uint8_t r, uint32_t num_threads)
    : reduce_NL(r), is_codestream_set(false), is_parsed(false) {
  uintmax_t file_size;
#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || __cplusplus >= 201703L)
  try {
    file_size = std::filesystem::file_size(filename);
  } catch (std::filesystem::filesystem_error &err) {
    printf("ERROR: input file %s is not found.\n", filename);
    exit(EXIT_FAILURE);
  }
#else
  struct stat st;
  if (stat(filename, &st) != 0) {
    printf("ERROR: input file %s is not found.\n", filename);
    exit(EXIT_FAILURE);
  }
  file_size = static_cast<uintmax_t>(st.st_size);
#endif
#ifdef OPENHTJ2K_THREAD
  ThreadPool::instance(num_threads);
#endif
  // open codestream and store it in memory
  FILE *fp = fopen(filename, "rb");
  in.alloc_memory(static_cast<uint32_t>(file_size));
  uint8_t *p        = in.get_buf_pos();
  size_t bytes_read = fread(p, sizeof(uint8_t), static_cast<size_t>(file_size), fp);
  if (bytes_read < file_size) {
    printf("ERROR: %s seems to have not enough data.\n", filename);
    throw std::exception();
  }
  fclose(fp);
  is_codestream_set = true;
}

openhtj2k_decoder_impl::openhtj2k_decoder_impl(const uint8_t *buf, const size_t length, const uint8_t r,
                                               uint32_t num_threads)
    : reduce_NL(r), is_codestream_set(false), is_parsed(false) {
  if (buf == nullptr) {
  }
#ifdef OPENHTJ2K_THREAD
  ThreadPool::instance(num_threads);
#endif
  // open codestream and store it in memory
  in.alloc_memory(static_cast<uint32_t>(length));
  uint8_t *p = in.get_buf_pos();
  memcpy(p, buf, length);
  is_codestream_set = true;
}

void openhtj2k_decoder_impl::init(const uint8_t *buf, const size_t length, const uint8_t r,
                                  uint32_t num_threads) {
  reduce_NL = r;
  if (buf == nullptr) {
  }
#ifdef OPENHTJ2K_THREAD
  ThreadPool::instance(num_threads);
#endif
  // open codestream and store it in memory
  in.alloc_memory(static_cast<uint32_t>(length));
  uint8_t *p = in.get_buf_pos();
  memcpy(p, buf, length);
  is_codestream_set = true;
}

void openhtj2k_decoder_impl::parse() {
  if (is_codestream_set == false) {
    printf(
        "ERROR: openhtj2k_decoder_impl::openhtj2k_decoder_impl() shall be called before calling "
        "openhtj2k_decoder_impl::parse().\n");
    throw std::exception();
  }
  // Read main header
  main_header.read(in);
  in.rewind_2bytes();
  is_parsed = true;
}

uint16_t openhtj2k_decoder_impl::get_num_component() { return main_header.SIZ->get_num_components(); }
uint32_t openhtj2k_decoder_impl::get_component_width(uint16_t c) {
  // Currently does not return component specific width
  element_siz size, origin;
  main_header.SIZ->get_image_size(size);
  main_header.SIZ->get_image_origin(origin);

  return size.x - origin.x;
}
uint32_t openhtj2k_decoder_impl::get_component_height(uint16_t c) {
  // Currently does not return component specific height
  element_siz size, origin;
  main_header.SIZ->get_image_size(size);
  main_header.SIZ->get_image_origin(origin);

  return size.y - origin.y;
}
uint8_t openhtj2k_decoder_impl::get_component_depth(uint16_t c) { return main_header.SIZ->get_bitdepth(c); }
bool openhtj2k_decoder_impl::get_component_signedness(uint16_t c) { return main_header.SIZ->is_signed(c); }

uint8_t openhtj2k_decoder_impl::get_minimum_DWT_levels() {
  uint8_t NL = main_header.COD->get_dwt_levels();
  if (main_header.COC.empty() == false) {
    size_t i = 0;
    for (uint16_t c = 0; c < this->get_num_component(); ++c) {
      if (main_header.COC[i]->get_component_index() == c) {
        if (NL > main_header.COC[i]->get_dwt_levels()) {
          NL = main_header.COC[i]->get_dwt_levels();
        }
        ++i;
      }
    }
  }
  return NL;
}

void openhtj2k_decoder_impl::destroy() {}
void openhtj2k_decoder_impl::invoke(std::vector<int32_t *> &buf, std::vector<uint32_t> &width,
                                    std::vector<uint32_t> &height, std::vector<uint8_t> &depth,
                                    std::vector<bool> &is_signed) {
  if (is_parsed == false) {
    printf(
        "ERROR: openhtj2k_decoder_impl::parse() shall be called before calling "
        "openhtj2k_decoder_impl::invoke().\n");
    throw std::exception();
  }
  if (reduce_NL > this->get_minimum_DWT_levels()) {
    throw std::runtime_error(
        "Attempting to access a non-existent resolution level within some\n"
        "tile-component.  Problem almost certainly caused by trying to discard more\n"
        "resolution levels than the number of DWT levels used to compress a\n"
        "tile-component.");
  }
  element_siz numTiles;
  main_header.get_number_of_tiles(numTiles.x, numTiles.y);
  // printf("Tile num x = %d, y = %d\n", numTiles.x, numTiles.y);

  // Create output buffer
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
    width.push_back(ceil_int(x1[c] - x0[c], (1U << reduce_NL)));
    height.push_back(ceil_int(y1[c] - y0[c], (1U << reduce_NL)));
    buf.emplace_back(new int32_t[width[c] * height[c]]);
    depth.push_back(main_header.SIZ->get_bitdepth(c) - 0);
    is_signed.push_back(main_header.SIZ->is_signed(c));
  }
  if (numTiles.x * numTiles.y > 65535) {
    printf("ERROR: The number of tiles exceeds its allowable maximum (65535).\n");
    throw std::exception();
  }

  //  auto tileSet = MAKE_UNIQUE<j2k_tile[]>(static_cast<size_t>(numTiles.x) * numTiles.y);
  std::vector<j2k_tile> tileSet;
  tileSet.resize(static_cast<size_t>(numTiles.x) * numTiles.y);
  for (uint16_t i = 0; i < static_cast<uint16_t>(numTiles.x * numTiles.y); ++i) {
    tileSet[i].dec_init(i, main_header, reduce_NL);
  }

  uint16_t word;
  SOT_marker tmpSOT;
  uint16_t tile_index;
  // Read all tile parts
  while ((word = in.get_word()) != _EOC) {
    if (word != _SOT) {
      printf("ERROR: SOT marker segment expected but %04X is found\n", word);
      throw std::exception();
    }
    tmpSOT     = SOT_marker(in);
    tile_index = tmpSOT.get_tile_index();
    tileSet[tile_index].add_tile_part(tmpSOT, in, main_header);
  }

  // Read codestream and decode it
  for (uint32_t i = 0; i < numTiles.x * numTiles.y; i++) {
    try {
      tileSet[i].create_tile_buf(main_header);
    } catch (std::exception &exc) {
      printf("ERROR: %s\n", exc.what());
      tileSet[i].destroy();
      throw std::runtime_error("Abort Decoding!");
    };

    tileSet[i].decode();
    tileSet[i].ycbcr_to_rgb();
    tileSet[i].finalize(main_header, reduce_NL, buf);  // Copy reconstructed image to output buffer
  }
}

openhtj2k_decoder_impl::~openhtj2k_decoder_impl() {
#ifdef OPENHTJ2K_THREAD
  ThreadPool::release();
#endif
}

// public interface
openhtj2k_decoder::openhtj2k_decoder() { this->impl = MAKE_UNIQUE<openhtj2k_decoder_impl>(); }
openhtj2k_decoder::openhtj2k_decoder(const char *fname, const uint8_t reduce_NL, uint32_t num_threads) {
  this->impl = MAKE_UNIQUE<openhtj2k_decoder_impl>(fname, reduce_NL, num_threads);
}
// on memory decoding
openhtj2k_decoder::openhtj2k_decoder(const uint8_t *buf, size_t length, const uint8_t reduce_NL,
                                     uint32_t num_threads) {
  this->impl = MAKE_UNIQUE<openhtj2k_decoder_impl>(buf, length, reduce_NL, num_threads);
}
void openhtj2k_decoder::init(const uint8_t *buf, size_t length, const uint8_t reduce_NL,
                             uint32_t num_threads) {
  this->impl->init(buf, length, reduce_NL, num_threads);
}
void openhtj2k_decoder::parse() { this->impl->parse(); }

uint16_t openhtj2k_decoder::get_num_component() { return this->impl->get_num_component(); }
uint32_t openhtj2k_decoder::get_component_width(uint16_t c) { return this->impl->get_component_width(c); }
uint32_t openhtj2k_decoder::get_component_height(uint16_t c) { return this->impl->get_component_height(c); }
uint8_t openhtj2k_decoder::get_component_depth(uint16_t c) { return this->impl->get_component_depth(c); }
bool openhtj2k_decoder::get_component_signedness(uint16_t c) {
  return this->impl->get_component_signedness(c);
}
uint8_t openhtj2k_decoder::get_minumum_DWT_levels() { return this->impl->get_minimum_DWT_levels(); }

void openhtj2k_decoder::invoke(std::vector<int32_t *> &buf, std::vector<uint32_t> &width,
                               std::vector<uint32_t> &height, std::vector<uint8_t> &depth,
                               std::vector<bool> &is_signed) {
  this->impl->invoke(buf, width, height, depth, is_signed);
}

void openhtj2k_decoder::destroy() { this->impl->destroy(); }

openhtj2k_decoder::~openhtj2k_decoder() = default;
}  // namespace open_htj2k
