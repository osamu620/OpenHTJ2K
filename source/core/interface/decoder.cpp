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

#include <algorithm>
#include <cstring>
#include <cstdio>
#include <functional>
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
  [[nodiscard]] uint16_t get_num_component() const;
  [[nodiscard]] uint32_t get_component_width(uint16_t) const;
  [[nodiscard]] uint32_t get_component_height(uint16_t) const;
  [[nodiscard]] uint8_t get_component_depth(uint16_t) const;
  [[nodiscard]] bool get_component_signedness(uint16_t) const;
  uint8_t get_minimum_DWT_levels();

  void invoke(std::vector<int32_t *> &, std::vector<uint32_t> &, std::vector<uint32_t> &,
              std::vector<uint8_t> &, std::vector<bool> &);
  void invoke_line_based(std::vector<int32_t *> &, std::vector<uint32_t> &, std::vector<uint32_t> &,
                         std::vector<uint8_t> &, std::vector<bool> &);
  void invoke_line_based_stream(std::function<void(uint32_t, int32_t *const *, uint16_t)> cb,
                                std::vector<uint32_t> &, std::vector<uint32_t> &,
                                std::vector<uint8_t> &, std::vector<bool> &);
  void invoke_line_based_predecoded(std::vector<int32_t *> &, std::vector<uint32_t> &,
                                    std::vector<uint32_t> &, std::vector<uint8_t> &,
                                    std::vector<bool> &);

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

uint16_t openhtj2k_decoder_impl::get_num_component() const { return main_header.SIZ->get_num_components(); }
uint32_t openhtj2k_decoder_impl::get_component_width(uint16_t c) const {
  element_siz size, origin, subsampling_factor;
  main_header.SIZ->get_image_size(size);
  main_header.SIZ->get_image_origin(origin);
  main_header.SIZ->get_subsampling_factor(subsampling_factor, c);

  return ceil_int(size.x - origin.x, subsampling_factor.x);
}
uint32_t openhtj2k_decoder_impl::get_component_height(uint16_t c) const {
  element_siz size, origin, subsampling_factor;
  main_header.SIZ->get_image_size(size);
  main_header.SIZ->get_image_origin(origin);
  main_header.SIZ->get_subsampling_factor(subsampling_factor, c);

  return ceil_int(size.y - origin.y, subsampling_factor.y);
}
uint8_t openhtj2k_decoder_impl::get_component_depth(uint16_t c) const {
  return main_header.SIZ->get_bitdepth(c);
}
bool openhtj2k_decoder_impl::get_component_signedness(uint16_t c) const {
  return main_header.SIZ->is_signed(c);
}

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
    tileSet[i].destroy();  // Release tile-internal buffers immediately (output is in buf)
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

// ─────────────────────────────────────────────────────────────────────────────
void openhtj2k_decoder_impl::invoke_line_based(std::vector<int32_t *> &buf,
                                               std::vector<uint32_t> &width,
                                               std::vector<uint32_t> &height,
                                               std::vector<uint8_t> &depth,
                                               std::vector<bool> &is_signed) {
  if (!is_parsed) {
    printf(
        "ERROR: openhtj2k_decoder_impl::parse() shall be called before calling "
        "openhtj2k_decoder_impl::invoke_line_based().\n");
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

  // Allocate output buffers (identical to invoke()).
  uint16_t num_components = main_header.SIZ->get_num_components();
  element_siz siz, Osiz, Tsiz, TOsiz, Rsiz;
  main_header.SIZ->get_image_size(siz);
  main_header.SIZ->get_image_origin(Osiz);
  main_header.SIZ->get_tile_size(Tsiz);
  main_header.SIZ->get_tile_origin(TOsiz);
  for (uint16_t c = 0; c < num_components; c++) {
    main_header.SIZ->get_subsampling_factor(Rsiz, c);
    const uint32_t x0 = ceil_int(Osiz.x, Rsiz.x);
    const uint32_t x1 = ceil_int(siz.x, Rsiz.x);
    const uint32_t y0 = ceil_int(Osiz.y, Rsiz.y);
    const uint32_t y1 = ceil_int(siz.y, Rsiz.y);
    width.push_back(ceil_int(x1 - x0, (1U << reduce_NL)));
    height.push_back(ceil_int(y1 - y0, (1U << reduce_NL)));
    buf.emplace_back(new int32_t[width[c] * height[c]]);
    depth.push_back(main_header.SIZ->get_bitdepth(c));
    is_signed.push_back(main_header.SIZ->is_signed(c));
  }
  if (numTiles.x * numTiles.y > 65535) {
    printf("ERROR: The number of tiles exceeds its allowable maximum (65535).\n");
    throw std::exception();
  }

  std::vector<j2k_tile> tileSet;
  tileSet.resize(static_cast<size_t>(numTiles.x) * numTiles.y);
  for (uint16_t i = 0; i < static_cast<uint16_t>(numTiles.x * numTiles.y); ++i) {
    tileSet[i].dec_init(i, main_header, reduce_NL);
  }

  uint16_t word;
  SOT_marker tmpSOT;
  uint16_t tile_index;
  while ((word = in.get_word()) != _EOC) {
    if (word != _SOT) {
      printf("ERROR: SOT marker segment expected but %04X is found\n", word);
      throw std::exception();
    }
    tmpSOT     = SOT_marker(in);
    tile_index = tmpSOT.get_tile_index();
    tileSet[tile_index].add_tile_part(tmpSOT, in, main_header);
  }

  for (uint32_t i = 0; i < numTiles.x * numTiles.y; i++) {
    try {
      tileSet[i].line_based_decode = true;
      tileSet[i].create_tile_buf(main_header);
    } catch (std::exception &exc) {
      printf("ERROR: %s\n", exc.what());
      tileSet[i].destroy();
      throw std::runtime_error("Abort Decoding!");
    }
    // decode_line_based() replaces decode() + ycbcr_to_rgb() + finalize().
    tileSet[i].decode_line_based(main_header, reduce_NL, buf);
    tileSet[i].destroy();  // Release tile-internal buffers immediately (output is in buf)
  }
}

void openhtj2k_decoder::invoke_line_based(std::vector<int32_t *> &buf, std::vector<uint32_t> &width,
                                          std::vector<uint32_t> &height, std::vector<uint8_t> &depth,
                                          std::vector<bool> &is_signed) {
  this->impl->invoke_line_based(buf, width, height, depth, is_signed);
}

void openhtj2k_decoder_impl::invoke_line_based_stream(
    std::function<void(uint32_t, int32_t *const *, uint16_t)> cb, std::vector<uint32_t> &width,
    std::vector<uint32_t> &height, std::vector<uint8_t> &depth, std::vector<bool> &is_signed) {
  if (!is_parsed) {
    printf(
        "ERROR: openhtj2k_decoder_impl::parse() shall be called before calling "
        "openhtj2k_decoder_impl::invoke_line_based_stream().\n");
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

  uint16_t num_components = main_header.SIZ->get_num_components();
  element_siz siz, Osiz, Tsiz, TOsiz, Rsiz;
  main_header.SIZ->get_image_size(siz);
  main_header.SIZ->get_image_origin(Osiz);
  main_header.SIZ->get_tile_size(Tsiz);
  main_header.SIZ->get_tile_origin(TOsiz);
  for (uint16_t c = 0; c < num_components; c++) {
    main_header.SIZ->get_subsampling_factor(Rsiz, c);
    const uint32_t x0 = ceil_int(Osiz.x, Rsiz.x);
    const uint32_t x1 = ceil_int(siz.x, Rsiz.x);
    const uint32_t y0 = ceil_int(Osiz.y, Rsiz.y);
    const uint32_t y1 = ceil_int(siz.y, Rsiz.y);
    width.push_back(ceil_int(x1 - x0, (1U << reduce_NL)));
    height.push_back(ceil_int(y1 - y0, (1U << reduce_NL)));
    depth.push_back(main_header.SIZ->get_bitdepth(c));
    is_signed.push_back(main_header.SIZ->is_signed(c));
  }
  if (numTiles.x * numTiles.y > 65535) {
    printf("ERROR: The number of tiles exceeds its allowable maximum (65535).\n");
    throw std::exception();
  }

  std::vector<j2k_tile> tileSet;
  tileSet.resize(static_cast<size_t>(numTiles.x) * numTiles.y);
  for (uint16_t i = 0; i < static_cast<uint16_t>(numTiles.x * numTiles.y); ++i) {
    tileSet[i].dec_init(i, main_header, reduce_NL);
  }

  uint16_t word;
  SOT_marker tmpSOT;
  uint16_t tile_index;
  while ((word = in.get_word()) != _EOC) {
    if (word != _SOT) {
      printf("ERROR: SOT marker segment expected but %04X is found\n", word);
      throw std::exception();
    }
    tmpSOT     = SOT_marker(in);
    tile_index = tmpSOT.get_tile_index();
    tileSet[tile_index].add_tile_part(tmpSOT, in, main_header);
  }

  const uint32_t rscale = 1U << reduce_NL;

  // Compute tile-component geometry (pixel offset + size at active resolution).
  auto get_tile_geom = [&](uint32_t tx, uint32_t ty, uint16_t c, uint32_t &x_off,
                           uint32_t &y_off, uint32_t &t_w, uint32_t &t_h) {
    element_siz Rsiz_c;
    main_header.SIZ->get_subsampling_factor(Rsiz_c, c);
    const uint32_t x0_c  = ceil_int(Osiz.x, Rsiz_c.x);
    const uint32_t y0_c  = ceil_int(Osiz.y, Rsiz_c.y);
    const uint32_t gtx0  = std::max(TOsiz.x + tx * Tsiz.x, Osiz.x);
    const uint32_t gtx1  = std::min(TOsiz.x + (tx + 1) * Tsiz.x, siz.x);
    const uint32_t gty0  = std::max(TOsiz.y + ty * Tsiz.y, Osiz.y);
    const uint32_t gty1  = std::min(TOsiz.y + (ty + 1) * Tsiz.y, siz.y);
    const uint32_t tcx0  = ceil_int(gtx0, Rsiz_c.x);
    const uint32_t tcx1  = ceil_int(gtx1, Rsiz_c.x);
    const uint32_t tcy0  = ceil_int(gty0, Rsiz_c.y);
    const uint32_t tcy1  = ceil_int(gty1, Rsiz_c.y);
    x_off = ceil_int(tcx0, rscale) - ceil_int(x0_c, rscale);
    y_off = ceil_int(tcy0, rscale) - ceil_int(y0_c, rscale);
    t_w   = ceil_int(tcx1, rscale) - ceil_int(tcx0, rscale);
    t_h   = ceil_int(tcy1, rscale) - ceil_int(tcy0, rscale);
  };

  uint32_t global_y = 0;

  if (numTiles.x == 1) {
    // Fast path: single tile column — deliver rows directly to user callback, no accumulator.
    for (uint32_t ty = 0; ty < numTiles.y; ++ty) {
      uint32_t x_off, y_off, t_w, band_h0;
      get_tile_geom(0, ty, 0, x_off, y_off, t_w, band_h0);
      const uint32_t tile_idx = ty;
      try {
        tileSet[tile_idx].line_based_decode = true;
        tileSet[tile_idx].create_tile_buf(main_header);
      } catch (std::exception &exc) {
        printf("ERROR: %s\n", exc.what());
        tileSet[tile_idx].destroy();
        throw std::runtime_error("Abort Decoding!");
      }
      tileSet[tile_idx].decode_line_based_stream(main_header, reduce_NL,
          [&](uint32_t y_local, int32_t *const *rows, uint16_t nc) {
            cb(global_y + y_local, rows, nc);
          });
      tileSet[tile_idx].destroy();
      global_y += band_h0;
    }
    return;
  }

  // General path: multiple tile columns — scatter into per-band-row accumulators.
  for (uint32_t ty = 0; ty < numTiles.y; ++ty) {
    // Compute band height per component (same for all tx in this tile row).
    std::vector<uint32_t> band_h(num_components);
    {
      uint32_t x_off, y_off, t_w;
      for (uint16_t c = 0; c < num_components; ++c)
        get_tile_geom(0, ty, c, x_off, y_off, t_w, band_h[c]);
    }

    // Allocate full-width accumulator buffers for this tile-row band.
    std::vector<std::vector<int32_t>> accum(num_components);
    for (uint16_t c = 0; c < num_components; ++c)
      accum[c].assign(static_cast<size_t>(band_h[c]) * width[c], 0);

    for (uint32_t tx = 0; tx < numTiles.x; ++tx) {
      const uint32_t tile_idx = ty * numTiles.x + tx;

      // Compute per-component x-offsets and widths for this tile.
      std::vector<uint32_t> tile_x_off(num_components), tile_w(num_components);
      {
        uint32_t x_off, y_off, t_w, t_h;
        for (uint16_t c = 0; c < num_components; ++c) {
          get_tile_geom(tx, ty, c, x_off, y_off, t_w, t_h);
          tile_x_off[c] = x_off;
          tile_w[c]     = t_w;
        }
      }

      // Internal scatter callback: copy tile-local rows into accumulator.
      auto scatter = [&](uint32_t y_local, int32_t *const *rows, uint16_t nc) {
        for (uint16_t c = 0; c < nc; ++c) {
          if (tile_w[c] == 0 || y_local >= band_h[c]) continue;
          int32_t *dst =
              accum[c].data() + static_cast<ptrdiff_t>(y_local) * width[c] + tile_x_off[c];
          std::memcpy(dst, rows[c], tile_w[c] * sizeof(int32_t));
        }
      };

      try {
        tileSet[tile_idx].line_based_decode = true;
        tileSet[tile_idx].create_tile_buf(main_header);
      } catch (std::exception &exc) {
        printf("ERROR: %s\n", exc.what());
        tileSet[tile_idx].destroy();
        throw std::runtime_error("Abort Decoding!");
      }
      tileSet[tile_idx].decode_line_based_stream(main_header, reduce_NL, scatter);
      tileSet[tile_idx].destroy();  // Release tile-internal buffers immediately
    }

    // Deliver complete rows for this tile-row band to the user callback.
    std::vector<int32_t *> row_ptrs(num_components);
    for (uint32_t y = 0; y < band_h[0]; ++y) {
      for (uint16_t c = 0; c < num_components; ++c) {
        // For subsampled components, clamp to last valid row.
        const uint32_t cy = (band_h[c] > 0) ? std::min(y, band_h[c] - 1) : 0;
        row_ptrs[c] = accum[c].data() + static_cast<ptrdiff_t>(cy) * width[c];
      }
      cb(global_y + y, row_ptrs.data(), num_components);
    }
    global_y += band_h[0];
  }
}

void openhtj2k_decoder::invoke_line_based_stream(
    std::function<void(uint32_t y, int32_t *const *, uint16_t nc)> cb, std::vector<uint32_t> &width,
    std::vector<uint32_t> &height, std::vector<uint8_t> &depth, std::vector<bool> &is_signed) {
  this->impl->invoke_line_based_stream(std::move(cb), width, height, depth, is_signed);
}

void openhtj2k_decoder_impl::invoke_line_based_predecoded(std::vector<int32_t *> &buf,
                                                          std::vector<uint32_t> &width,
                                                          std::vector<uint32_t> &height,
                                                          std::vector<uint8_t> &depth,
                                                          std::vector<bool> &is_signed) {
  if (!is_parsed) {
    printf(
        "ERROR: openhtj2k_decoder_impl::parse() shall be called before calling "
        "openhtj2k_decoder_impl::invoke_line_based_predecoded().\n");
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

  uint16_t num_components = main_header.SIZ->get_num_components();
  element_siz siz, Osiz, Tsiz, TOsiz, Rsiz;
  main_header.SIZ->get_image_size(siz);
  main_header.SIZ->get_image_origin(Osiz);
  main_header.SIZ->get_tile_size(Tsiz);
  main_header.SIZ->get_tile_origin(TOsiz);
  for (uint16_t c = 0; c < num_components; c++) {
    main_header.SIZ->get_subsampling_factor(Rsiz, c);
    const uint32_t x0 = ceil_int(Osiz.x, Rsiz.x);
    const uint32_t x1 = ceil_int(siz.x, Rsiz.x);
    const uint32_t y0 = ceil_int(Osiz.y, Rsiz.y);
    const uint32_t y1 = ceil_int(siz.y, Rsiz.y);
    width.push_back(ceil_int(x1 - x0, (1U << reduce_NL)));
    height.push_back(ceil_int(y1 - y0, (1U << reduce_NL)));
    buf.emplace_back(new int32_t[width[c] * height[c]]);
    depth.push_back(main_header.SIZ->get_bitdepth(c));
    is_signed.push_back(main_header.SIZ->is_signed(c));
  }
  if (numTiles.x * numTiles.y > 65535) {
    printf("ERROR: The number of tiles exceeds its allowable maximum (65535).\n");
    throw std::exception();
  }

  std::vector<j2k_tile> tileSet;
  tileSet.resize(static_cast<size_t>(numTiles.x) * numTiles.y);
  for (uint16_t i = 0; i < static_cast<uint16_t>(numTiles.x * numTiles.y); ++i) {
    tileSet[i].dec_init(i, main_header, reduce_NL);
  }

  uint16_t word;
  SOT_marker tmpSOT;
  uint16_t tile_index;
  while ((word = in.get_word()) != _EOC) {
    if (word != _SOT) {
      printf("ERROR: SOT marker segment expected but %04X is found\n", word);
      throw std::exception();
    }
    tmpSOT     = SOT_marker(in);
    tile_index = tmpSOT.get_tile_index();
    tileSet[tile_index].add_tile_part(tmpSOT, in, main_header);
  }

  for (uint32_t i = 0; i < numTiles.x * numTiles.y; i++) {
    try {
      tileSet[i].create_tile_buf(main_header);
    } catch (std::exception &exc) {
      printf("ERROR: %s\n", exc.what());
      tileSet[i].destroy();
      throw std::runtime_error("Abort Decoding!");
    }
    tileSet[i].decode_line_based_predecoded(main_header, reduce_NL, buf);
  }
}

void openhtj2k_decoder::invoke_line_based_predecoded(std::vector<int32_t *> &buf,
                                                     std::vector<uint32_t> &width,
                                                     std::vector<uint32_t> &height,
                                                     std::vector<uint8_t> &depth,
                                                     std::vector<bool> &is_signed) {
  this->impl->invoke_line_based_predecoded(buf, width, height, depth, is_signed);
}

void openhtj2k_decoder::destroy() { this->impl->destroy(); }

openhtj2k_decoder::~openhtj2k_decoder() = default;
}  // namespace open_htj2k
