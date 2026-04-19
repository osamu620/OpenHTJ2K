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

#include <algorithm>
#include <cstring>
#include <cstdio>
#include <functional>
#include "decoder.hpp"
#include "jph.hpp"
// GCC 7.x claims C++17 (__cplusplus >= 201703L) but ships <filesystem>
// only as <experimental/filesystem>, so the language-version check alone
// is not sufficient.  Use __has_include when available (GCC 5+, Clang,
// MSVC 2017+) and fall back to the language-version test otherwise.
#if defined(__has_include)
  #if __has_include(<filesystem>) && ((defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || __cplusplus >= 201703L)
    #define OPENHTJ2K_HAS_FILESYSTEM 1
  #endif
#elif ((defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || __cplusplus >= 201703L)
  #define OPENHTJ2K_HAS_FILESYSTEM 1
#endif

#ifdef OPENHTJ2K_HAS_FILESYSTEM
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
  uint32_t row_limit_ = UINT32_MAX;
  uint32_t col_lo_    = 0;
  uint32_t col_hi_    = UINT32_MAX;
  bool is_codestream_set;
  bool is_parsed;
  j2k_main_header main_header;
  uint32_t enum_cs;             // EnumCS from JPH colr box; 0 for raw codestreams
  std::vector<uint8_t> jph_file_buf;  // owns full file data when input is a JPH file

  // ── Single-tile cache (RFC 9828 streaming opt-in) ─────────────────────
  // When single_tile_reuse_enabled_ is set, invoke_line_based_stream_reuse()
  // keeps the decoded tile alive across calls so the next frame can reuse
  // every aligned allocation hanging off tcomp[] → resolution[] → precinct[]
  // → j2k_codeblock (compressed buffers, tagtrees, ring buffers, line_dec
  // sub-objects).  Only valid when the main-header marker bytes are byte-
  // identical across frames; cached_header_fingerprint_ is the FNV-1a hash
  // of SIZ/COD/COC/QCD/QCC/RGN used as the cache-validity check.
  bool single_tile_reuse_enabled_      = false;
  std::vector<j2k_tile> cached_tileSet_;
  uint64_t cached_header_fingerprint_  = 0;

  // JPIP partial-decode filter: invoked per-packet during every subsequent
  // invoke*() call, once per (t, c, r, p_rc).  An empty std::function means
  // "no filter" — the tile-level precinct_filter_ stays unset and all
  // precincts decode normally.
  std::function<bool(uint16_t, uint16_t, uint8_t, uint32_t)> precinct_filter_;

  // JPIP packet observer: called with per-packet byte ranges relative to
  // each tile's tile_buf.  Empty std::function means "do not observe".
  std::function<void(uint16_t, uint16_t, uint8_t, uint32_t, uint16_t, uint64_t, uint64_t)>
      packet_observer_;

 public:
  openhtj2k_decoder_impl();
  openhtj2k_decoder_impl(const char *, uint8_t reduce_NL, uint32_t num_threads);
  openhtj2k_decoder_impl(const uint8_t *, size_t, uint8_t reduce_NL, uint32_t num_threads);
  ~openhtj2k_decoder_impl();
  void init(const uint8_t *, size_t, uint8_t reduce_NL, uint32_t num_threads);
  void init_borrow(uint8_t *, size_t, uint8_t reduce_NL, uint32_t num_threads);
  void parse();
  OPENHTJ2K_NODISCARD uint16_t get_num_component() const;
  OPENHTJ2K_NODISCARD uint32_t get_component_width(uint16_t) const;
  OPENHTJ2K_NODISCARD uint32_t get_component_height(uint16_t) const;
  OPENHTJ2K_NODISCARD uint8_t get_component_depth(uint16_t) const;
  OPENHTJ2K_NODISCARD bool get_component_signedness(uint16_t) const;
  OPENHTJ2K_NODISCARD uint32_t get_colorspace() const { return enum_cs; }
  uint8_t get_minimum_DWT_levels();
  uint8_t get_max_safe_reduce_NL();

  void invoke(std::vector<int32_t *> &, std::vector<uint32_t> &, std::vector<uint32_t> &,
              std::vector<uint8_t> &, std::vector<bool> &);
  void invoke_line_based(std::vector<int32_t *> &, std::vector<uint32_t> &, std::vector<uint32_t> &,
                         std::vector<uint8_t> &, std::vector<bool> &);
  void invoke_line_based_stream(std::function<void(uint32_t, int32_t *const *, uint16_t)> cb,
                                std::vector<uint32_t> &, std::vector<uint32_t> &,
                                std::vector<uint8_t> &, std::vector<bool> &);
  void invoke_line_based_stream_reuse(std::function<void(uint32_t, int32_t *const *, uint16_t)> cb,
                                       std::vector<uint32_t> &, std::vector<uint32_t> &,
                                       std::vector<uint8_t> &, std::vector<bool> &);
  void invoke_line_based_direct(PlanarOutputDesc *descs, uint16_t nc,
                                std::vector<uint32_t> &, std::vector<uint32_t> &,
                                std::vector<uint8_t> &, std::vector<bool> &);
  void invoke_line_based_predecoded(std::vector<int32_t *> &, std::vector<uint32_t> &,
                                    std::vector<uint32_t> &, std::vector<uint8_t> &,
                                    std::vector<bool> &);
  void enable_single_tile_reuse(bool on);
  void set_row_limit(uint32_t limit) { row_limit_ = limit; }
  void set_col_range(uint32_t col_lo, uint32_t col_hi) {
    col_lo_ = col_lo;
    col_hi_ = col_hi;
  }
  void set_precinct_filter(std::function<bool(uint16_t, uint16_t, uint8_t, uint32_t)> f);
  void set_packet_observer(
      std::function<void(uint16_t, uint16_t, uint8_t, uint32_t, uint16_t, uint64_t, uint64_t)> f);

  // Binds precinct_filter_'s (t, c, r, p_rc) signature into a (c, r, p_rc)
  // closure for the given tile and installs it on the tile (or clears it
  // when precinct_filter_ is empty).  Called just before each
  // create_tile_buf() so the packet-parsing loop sees the current filter.
  void install_precinct_filter(j2k_tile &tile, uint16_t tile_idx);

  // Analogous to install_precinct_filter but for the packet observer.
  void install_packet_observer(j2k_tile &tile, uint16_t tile_idx);

  void destroy();
};

openhtj2k_decoder_impl::openhtj2k_decoder_impl() {
  reduce_NL         = 0;
  is_codestream_set = false;
  is_parsed         = false;
  enum_cs           = 0;
}

openhtj2k_decoder_impl::openhtj2k_decoder_impl(const char *filename, const uint8_t r, uint32_t num_threads)
    : reduce_NL(r), is_codestream_set(false), is_parsed(false), enum_cs(0) {
  uintmax_t file_size;
#ifdef OPENHTJ2K_HAS_FILESYSTEM
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
  // Read the entire file into a temporary buffer for JPH detection.
  FILE *fp = fopen(filename, "rb");
  jph_file_buf.resize(static_cast<size_t>(file_size));
  size_t bytes_read = fread(jph_file_buf.data(), sizeof(uint8_t), jph_file_buf.size(), fp);
  fclose(fp);
  if (bytes_read < static_cast<size_t>(file_size)) {
    printf("ERROR: %s seems to have not enough data.\n", filename);
    throw std::exception();
  }
  // Detect JPH/JP2 signature and extract embedded codestream if found.
  jph_info info;
  if (jph_parse_buffer(jph_file_buf.data(), jph_file_buf.size(), info)) {
    enum_cs = info.enum_cs;
    in.alloc_memory(static_cast<uint32_t>(info.cs_size));
    memcpy(in.get_buf_pos(), info.cs_data, info.cs_size);
    jph_file_buf.clear();  // no longer needed; codestream is copied into `in`
    jph_file_buf.shrink_to_fit();
  } else {
    // Raw codestream: transfer directly.
    in.alloc_memory(static_cast<uint32_t>(file_size));
    memcpy(in.get_buf_pos(), jph_file_buf.data(), static_cast<size_t>(file_size));
    jph_file_buf.clear();
    jph_file_buf.shrink_to_fit();
  }
  is_codestream_set = true;
}

openhtj2k_decoder_impl::openhtj2k_decoder_impl(const uint8_t *buf, const size_t length, const uint8_t r,
                                               uint32_t num_threads)
    : reduce_NL(r), is_codestream_set(false), is_parsed(false), enum_cs(0) {
  if (buf == nullptr) {
  }
#ifdef OPENHTJ2K_THREAD
  ThreadPool::instance(num_threads);
#endif
  // Detect JPH/JP2 and extract the embedded codestream if present.
  jph_info info;
  if (jph_parse_buffer(buf, length, info)) {
    enum_cs = info.enum_cs;
    in.alloc_memory(static_cast<uint32_t>(info.cs_size));
    memcpy(in.get_buf_pos(), info.cs_data, info.cs_size);
  } else {
    in.alloc_memory(static_cast<uint32_t>(length));
    memcpy(in.get_buf_pos(), buf, length);
  }
  is_codestream_set = true;
}

void openhtj2k_decoder_impl::init(const uint8_t *buf, const size_t length, const uint8_t r,
                                  uint32_t num_threads) {
  reduce_NL = r;
  enum_cs   = 0;
  if (buf == nullptr) {
  }
#ifdef OPENHTJ2K_THREAD
  ThreadPool::instance(num_threads);
#endif
  jph_info info;
  if (jph_parse_buffer(buf, length, info)) {
    enum_cs = info.enum_cs;
    in.alloc_memory(static_cast<uint32_t>(info.cs_size));
    memcpy(in.get_buf_pos(), info.cs_data, info.cs_size);
  } else {
    in.alloc_memory(static_cast<uint32_t>(length));
    memcpy(in.get_buf_pos(), buf, length);
  }
  is_codestream_set = true;
}

void openhtj2k_decoder_impl::init_borrow(uint8_t *buf, const size_t length, const uint8_t r,
                                          uint32_t num_threads) {
  reduce_NL = r;
  enum_cs   = 0;
  if (buf == nullptr) {
  }
#ifdef OPENHTJ2K_THREAD
  ThreadPool::instance(num_threads);
#endif
  // Zero-copy: lend the caller's buffer to the decoder.  The caller must
  // keep the data alive through parse() + invoke*().  16 bytes of readable
  // padding past buf+length are required for SIMD over-reads.
  // JPH/JP2 wrapping is not expected on the RTP path; fall back to copy
  // if detected.
  jph_info info;
  if (jph_parse_buffer(buf, length, info)) {
    // JPH container detected — must copy the embedded codestream.
    enum_cs = info.enum_cs;
    in.alloc_memory(static_cast<uint32_t>(info.cs_size));
    memcpy(in.get_buf_pos(), info.cs_data, info.cs_size);
  } else {
    in.borrow_memory(buf, static_cast<uint32_t>(length));
  }
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
        // When DFS is active, SPcoc[0] encodes the DFS index, not level count.
        // Skip this COC for minimum level calculation; use COD's NL instead.
        if (!main_header.COC[i]->is_dfs_defined()) {
          if (NL > main_header.COC[i]->get_dwt_levels()) {
            NL = main_header.COC[i]->get_dwt_levels();
          }
        }
        ++i;
      }
    }
  }
  return NL;
}

uint8_t openhtj2k_decoder_impl::get_max_safe_reduce_NL() {
  // Start with the non-DFS upper bound.
  uint8_t max_r = get_minimum_DWT_levels();
  // For each DFS-active component, further limit to consecutive BIDIR levels from
  // the finest: reducing through a HONLY or VONLY level produces an image that
  // only has half the resolution in one spatial dimension, which is meaningless.
  if (!main_header.COC.empty()) {
    size_t i = 0;
    for (uint16_t c = 0; c < this->get_num_component(); ++c) {
      if (i < main_header.COC.size() && main_header.COC[i]->get_component_index() == c) {
        if (main_header.COC[i]->is_dfs_defined()) {
          const DFS_marker *dfs = main_header.get_dfs_marker(main_header.COC[i]->get_dfs_index());
          if (dfs != nullptr) {
            uint8_t dfs_safe = dfs->get_max_safe_reduce();
            if (max_r > dfs_safe) max_r = dfs_safe;
          }
        }
        ++i;
      }
    }
  }
  return max_r;
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
  if (reduce_NL > this->get_max_safe_reduce_NL()) {
    throw std::runtime_error(
        "Attempting to access a non-existent resolution level: -reduce exceeds the\n"
        "maximum safe value for this codestream.  For DFS streams this is the count\n"
        "of consecutive bidirectional DWT levels from the finest; for other streams\n"
        "it is the minimum DWT level count across all tile-components.");
  }
  element_siz numTiles;
  main_header.get_number_of_tiles(numTiles.x, numTiles.y);
  // printf("Tile num x = %d, y = %d\n", numTiles.x, numTiles.y);
  // Validate tile count BEFORE allocating output buffers — otherwise the raw `new int32_t[]`
  // pointers pushed into `buf` would leak on throw (caller owns `buf` but only after return).
  if (numTiles.x * numTiles.y > 65535) {
    printf("ERROR: The number of tiles exceeds its allowable maximum (65535).\n");
    throw std::exception();
  }

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

  //  auto tileSet = MAKE_UNIQUE<j2k_tile[]>(static_cast<size_t>(numTiles.x) * numTiles.y);
  std::vector<j2k_tile> tileSet;
  tileSet.resize(static_cast<size_t>(numTiles.x) * numTiles.y);
  for (uint16_t i = 0; i < static_cast<uint16_t>(numTiles.x * numTiles.y); ++i) {
    tileSet[i].dec_init(i, main_header, reduce_NL);
  }

  uint16_t word;
  SOT_marker tmpSOT;
  uint16_t tile_index;
  // Read all tile parts; treat EOF as EOC for truncated codestreams.
  while (in.get_remaining() >= 2 && (word = in.get_word()) != _EOC) {
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
      install_precinct_filter(tileSet[i], static_cast<uint16_t>(i));
      install_packet_observer(tileSet[i], static_cast<uint16_t>(i));
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
void openhtj2k_decoder::init_borrow(uint8_t *buf, size_t length, const uint8_t reduce_NL,
                                    uint32_t num_threads) {
  this->impl->init_borrow(buf, length, reduce_NL, num_threads);
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
uint8_t openhtj2k_decoder::get_max_safe_reduce_NL() { return this->impl->get_max_safe_reduce_NL(); }
uint32_t openhtj2k_decoder::get_colorspace() { return this->impl->get_colorspace(); }

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
  if (reduce_NL > this->get_max_safe_reduce_NL()) {
    throw std::runtime_error(
        "Attempting to access a non-existent resolution level: -reduce exceeds the\n"
        "maximum safe value for this codestream.  For DFS streams this is the count\n"
        "of consecutive bidirectional DWT levels from the finest; for other streams\n"
        "it is the minimum DWT level count across all tile-components.");
  }

  element_siz numTiles;
  main_header.get_number_of_tiles(numTiles.x, numTiles.y);
  // Validate tile count BEFORE allocating output buffers — see invoke() for rationale.
  if (numTiles.x * numTiles.y > 65535) {
    printf("ERROR: The number of tiles exceeds its allowable maximum (65535).\n");
    throw std::exception();
  }

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

  std::vector<j2k_tile> tileSet;
  tileSet.resize(static_cast<size_t>(numTiles.x) * numTiles.y);
  for (uint16_t i = 0; i < static_cast<uint16_t>(numTiles.x * numTiles.y); ++i) {
    tileSet[i].dec_init(i, main_header, reduce_NL);
  }

  uint16_t word;
  SOT_marker tmpSOT;
  uint16_t tile_index;
  // Read all tile parts; treat EOF as EOC for truncated codestreams.
  while (in.get_remaining() >= 2 && (word = in.get_word()) != _EOC) {
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
      install_precinct_filter(tileSet[i], static_cast<uint16_t>(i));
      install_packet_observer(tileSet[i], static_cast<uint16_t>(i));
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
  if (reduce_NL > this->get_max_safe_reduce_NL()) {
    throw std::runtime_error(
        "Attempting to access a non-existent resolution level: -reduce exceeds the\n"
        "maximum safe value for this codestream.  For DFS streams this is the count\n"
        "of consecutive bidirectional DWT levels from the finest; for other streams\n"
        "it is the minimum DWT level count across all tile-components.");
  }

  element_siz numTiles;
  main_header.get_number_of_tiles(numTiles.x, numTiles.y);
  // Validate tile count BEFORE populating output metadata — see invoke() for rationale.
  if (numTiles.x * numTiles.y > 65535) {
    printf("ERROR: The number of tiles exceeds its allowable maximum (65535).\n");
    throw std::exception();
  }

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

  std::vector<j2k_tile> tileSet;
  tileSet.resize(static_cast<size_t>(numTiles.x) * numTiles.y);
  for (uint16_t i = 0; i < static_cast<uint16_t>(numTiles.x * numTiles.y); ++i) {
    tileSet[i].dec_init(i, main_header, reduce_NL);
  }

  uint16_t word;
  SOT_marker tmpSOT;
  uint16_t tile_index;
  // Read all tile parts; treat EOF as EOC for truncated codestreams.
  while (in.get_remaining() >= 2 && (word = in.get_word()) != _EOC) {
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
        install_precinct_filter(tileSet[tile_idx], static_cast<uint16_t>(tile_idx));
        install_packet_observer(tileSet[tile_idx], static_cast<uint16_t>(tile_idx));
        tileSet[tile_idx].create_tile_buf(main_header);
      } catch (std::exception &exc) {
        printf("ERROR: %s\n", exc.what());
        tileSet[tile_idx].destroy();
        throw std::runtime_error("Abort Decoding!");
      }
      tileSet[tile_idx].decode_line_based_stream(main_header, reduce_NL,
          [&](uint32_t y_local, int32_t *const *rows, uint16_t nc) {
            cb(global_y + y_local, rows, nc);
          }, row_limit_, col_lo_, col_hi_);
      tileSet[tile_idx].destroy();
      global_y += band_h0;
    }
    return;
  }

  // General path: multiple tile columns — scatter into per-band-row accumulators.
  // Hoist all per-iteration scratch vectors out of the loops so the heap allocations
  // happen once instead of (numTiles.y * (1 + numTiles.x + 1)) times.
  std::vector<uint32_t>             band_h(num_components);
  std::vector<uint32_t>             tile_x_off(num_components);
  std::vector<uint32_t>             tile_w(num_components);
  std::vector<int32_t *>            row_ptrs(num_components);
  std::vector<std::vector<int32_t>> accum(num_components);

  for (uint32_t ty = 0; ty < numTiles.y; ++ty) {
    // Compute band height per component (same for all tx in this tile row).
    {
      uint32_t x_off, y_off, t_w;
      for (uint16_t c = 0; c < num_components; ++c)
        get_tile_geom(0, ty, c, x_off, y_off, t_w, band_h[c]);
    }

    // (Re)allocate full-width accumulator buffers for this tile-row band.
    // assign() reuses existing capacity when the new size fits.
    for (uint16_t c = 0; c < num_components; ++c)
      accum[c].assign(static_cast<size_t>(band_h[c]) * width[c], 0);

    for (uint32_t tx = 0; tx < numTiles.x; ++tx) {
      const uint32_t tile_idx = ty * numTiles.x + tx;

      // Compute per-component x-offsets and widths for this tile.
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
              accum[c].data() + static_cast<ptrdiff_t>(y_local) * static_cast<ptrdiff_t>(width[c]) + tile_x_off[c];
          std::memcpy(dst, rows[c], tile_w[c] * sizeof(int32_t));
        }
      };

      try {
        tileSet[tile_idx].line_based_decode = true;
        install_precinct_filter(tileSet[tile_idx], static_cast<uint16_t>(tile_idx));
        install_packet_observer(tileSet[tile_idx], static_cast<uint16_t>(tile_idx));
        tileSet[tile_idx].create_tile_buf(main_header);
      } catch (std::exception &exc) {
        printf("ERROR: %s\n", exc.what());
        tileSet[tile_idx].destroy();
        throw std::runtime_error("Abort Decoding!");
      }
      tileSet[tile_idx].decode_line_based_stream(main_header, reduce_NL, scatter, row_limit_,
                                                  col_lo_, col_hi_);
      tileSet[tile_idx].destroy();  // Release tile-internal buffers immediately
    }

    // Deliver complete rows for this tile-row band to the user callback.
    for (uint32_t y = 0; y < band_h[0]; ++y) {
      for (uint16_t c = 0; c < num_components; ++c) {
        // For subsampled components, clamp to last valid row.
        const uint32_t cy = (band_h[c] > 0) ? std::min(y, band_h[c] - 1) : 0;
        row_ptrs[c] = accum[c].data() + static_cast<ptrdiff_t>(cy) * static_cast<ptrdiff_t>(width[c]);
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

void openhtj2k_decoder::invoke_line_based_stream_reuse(
    std::function<void(uint32_t y, int32_t *const *, uint16_t nc)> cb, std::vector<uint32_t> &width,
    std::vector<uint32_t> &height, std::vector<uint8_t> &depth, std::vector<bool> &is_signed) {
  this->impl->invoke_line_based_stream_reuse(std::move(cb), width, height, depth, is_signed);
}

void openhtj2k_decoder::invoke_line_based_direct(
    PlanarOutputDesc *descs, uint16_t nc, std::vector<uint32_t> &width,
    std::vector<uint32_t> &height, std::vector<uint8_t> &depth, std::vector<bool> &is_signed) {
  this->impl->invoke_line_based_direct(descs, nc, width, height, depth, is_signed);
}

void openhtj2k_decoder::enable_single_tile_reuse(bool on) {
  this->impl->enable_single_tile_reuse(on);
}

void openhtj2k_decoder_impl::enable_single_tile_reuse(bool on) {
  single_tile_reuse_enabled_ = on;
  if (!on) {
    // Drop the cache so the next call starts from a clean slate.
    cached_tileSet_.clear();
    cached_header_fingerprint_ = 0;
  }
}

void openhtj2k_decoder::set_row_limit(uint32_t limit) {
  this->impl->set_row_limit(limit);
}

void openhtj2k_decoder::set_col_range(uint32_t col_lo, uint32_t col_hi) {
  this->impl->set_col_range(col_lo, col_hi);
}

void openhtj2k_decoder::set_precinct_filter(
    std::function<bool(uint16_t, uint16_t, uint8_t, uint32_t)> f) {
  this->impl->set_precinct_filter(std::move(f));
}

void openhtj2k_decoder_impl::set_precinct_filter(
    std::function<bool(uint16_t, uint16_t, uint8_t, uint32_t)> f) {
  precinct_filter_ = std::move(f);
}

void openhtj2k_decoder_impl::install_precinct_filter(j2k_tile &tile, uint16_t tile_idx) {
  if (precinct_filter_) {
    auto pf = precinct_filter_;  // capture by value — filter may be replaced between invokes
    tile.set_precinct_filter([pf, tile_idx](uint16_t c, uint8_t r, uint32_t p_rc) {
      return pf(tile_idx, c, r, p_rc);
    });
  } else {
    tile.set_precinct_filter({});
  }
}

void openhtj2k_decoder::set_packet_observer(
    std::function<void(uint16_t, uint16_t, uint8_t, uint32_t, uint16_t, uint64_t, uint64_t)> f) {
  this->impl->set_packet_observer(std::move(f));
}

void openhtj2k_decoder_impl::set_packet_observer(
    std::function<void(uint16_t, uint16_t, uint8_t, uint32_t, uint16_t, uint64_t, uint64_t)> f) {
  packet_observer_ = std::move(f);
}

void openhtj2k_decoder_impl::install_packet_observer(j2k_tile &tile, uint16_t tile_idx) {
  if (packet_observer_) {
    auto obs = packet_observer_;
    tile.set_packet_observer([obs, tile_idx](uint16_t c, uint8_t r, uint32_t p_rc,
                                             uint16_t layer, uint64_t offset,
                                             uint64_t length) {
      obs(tile_idx, c, r, p_rc, layer, offset, length);
    });
  } else {
    tile.set_packet_observer({});
  }
}

// Single-tile reuse entry point.  Mirrors the single-tile fast path of
// invoke_line_based_stream (numTiles.x == 1 branch) but keeps the j2k_tile
// parked on cached_tileSet_ so the next call skips create_resolutions,
// packet[] allocation, and the per-codeblock aligned_mem_alloc storm in
// subband_row_buf::init.  See reference_rtp_recv_v3_profile_2026_04_11.md
// in the user's memory for the measurements that motivated this.
//
// Multi-tile streams, multi-tile-column streams, and grayscale-only streams
// with numTiles.x != 1 are NOT supported — they fall through to the legacy
// invoke_line_based_stream which allocates per-call.  rtp_recv only emits
// single-tile codestreams for RFC 9828 (see project memory).
void openhtj2k_decoder_impl::invoke_line_based_stream_reuse(
    std::function<void(uint32_t, int32_t *const *, uint16_t)> cb, std::vector<uint32_t> &width,
    std::vector<uint32_t> &height, std::vector<uint8_t> &depth, std::vector<bool> &is_signed) {
  if (!is_parsed) {
    printf(
        "ERROR: openhtj2k_decoder_impl::parse() shall be called before calling "
        "openhtj2k_decoder_impl::invoke_line_based_stream_reuse().\n");
    throw std::exception();
  }
  if (!single_tile_reuse_enabled_) {
    invoke_line_based_stream(std::move(cb), width, height, depth, is_signed);
    return;
  }
  if (reduce_NL > this->get_max_safe_reduce_NL()) {
    throw std::runtime_error(
        "Attempting to access a non-existent resolution level: -reduce exceeds the\n"
        "maximum safe value for this codestream.  For DFS streams this is the count\n"
        "of consecutive bidirectional DWT levels from the finest; for other streams\n"
        "it is the minimum DWT level count across all tile-components.");
  }

  element_siz numTiles;
  main_header.get_number_of_tiles(numTiles.x, numTiles.y);
  // Only single-tile codestreams are eligible for the cache path.  Anything
  // else — including the multi-tile-row single-column case used by some
  // JPEG 2000 encoders — falls through to the legacy path.  The cache is
  // dropped to prevent us from mixing shapes across calls.
  if (numTiles.x != 1 || numTiles.y != 1) {
    cached_tileSet_.clear();
    cached_header_fingerprint_ = 0;
    invoke_line_based_stream(std::move(cb), width, height, depth, is_signed);
    return;
  }

  // Compute a fingerprint of the marker segments that the tile tree is
  // structurally sensitive to.  SIZ/COD/COC/QCD/QCC/RGN cover every field
  // read by create_resolutions / create_subbands / create_precincts; if
  // any of their encoded bytes change, we must drop the cache.  The
  // fingerprint is a cheap FNV-1a over the marker's get_length() /
  // parameter-derived values obtained from the parsed main_header —
  // computing over the raw bytes would require keeping a copy of the
  // codestream prefix, which is overkill for a safety check.
  auto fnv1a_u64 = [](uint64_t h, const void *data, size_t len) {
    const uint8_t *p = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < len; ++i) {
      h ^= p[i];
      h *= 1099511628211ull;
    }
    return h;
  };
  uint64_t fp = 14695981039346656037ull;  // FNV-1a offset basis
  auto mix_u64 = [&](uint64_t v) {
    fp = fnv1a_u64(fp, &v, sizeof(v));
  };
  {
    element_siz s_siz, s_osiz, s_tsiz, s_tosiz;
    main_header.SIZ->get_image_size(s_siz);
    main_header.SIZ->get_image_origin(s_osiz);
    main_header.SIZ->get_tile_size(s_tsiz);
    main_header.SIZ->get_tile_origin(s_tosiz);
    const uint16_t ncomp = main_header.SIZ->get_num_components();
    mix_u64((static_cast<uint64_t>(s_siz.x)  << 32) | s_siz.y);
    mix_u64((static_cast<uint64_t>(s_osiz.x) << 32) | s_osiz.y);
    mix_u64((static_cast<uint64_t>(s_tsiz.x) << 32) | s_tsiz.y);
    mix_u64((static_cast<uint64_t>(s_tosiz.x)<< 32) | s_tosiz.y);
    mix_u64(ncomp);
    for (uint16_t c = 0; c < ncomp; ++c) {
      element_siz sub;
      const uint8_t bd = main_header.SIZ->get_bitdepth(c);
      const bool    sn = main_header.SIZ->is_signed(c);
      main_header.SIZ->get_subsampling_factor(sub, c);
      mix_u64((static_cast<uint64_t>(bd) << 56)
              | (static_cast<uint64_t>(sn ? 1 : 0) << 48)
              | (static_cast<uint64_t>(sub.x) << 16)
              | static_cast<uint64_t>(sub.y));
    }
    if (main_header.COD != nullptr) {
      const uint8_t  dl  = main_header.COD->get_dwt_levels();
      const uint8_t  po  = main_header.COD->get_progression_order();
      const uint16_t nl  = main_header.COD->get_number_of_layers();
      const uint8_t  mct = main_header.COD->use_color_trafo();
      const uint8_t  cm  = main_header.COD->get_Cmodes();
      const uint8_t  tx  = main_header.COD->get_transformation();
      element_siz    cbsz; main_header.COD->get_codeblock_size(cbsz);
      mix_u64((static_cast<uint64_t>(dl) << 56) | (static_cast<uint64_t>(po) << 48)
              | (static_cast<uint64_t>(nl) << 32)
              | (static_cast<uint64_t>(mct) << 24) | (static_cast<uint64_t>(cm) << 16)
              | static_cast<uint64_t>(tx));
      mix_u64((static_cast<uint64_t>(cbsz.x) << 32) | cbsz.y);
      for (uint8_t r = 0; r <= dl; ++r) {
        element_siz pp; main_header.COD->get_precinct_size(pp, r);
        mix_u64((static_cast<uint64_t>(pp.x) << 32) | pp.y);
      }
    }
    if (main_header.QCD != nullptr) {
      const uint8_t qs = main_header.QCD->get_quantization_style();
      const uint8_t gb = main_header.QCD->get_number_of_guardbits();
      const uint8_t n0 = main_header.QCD->get_num_entries();
      mix_u64((static_cast<uint64_t>(qs) << 16) | (static_cast<uint64_t>(gb) << 8)
              | static_cast<uint64_t>(n0));
      if (n0 > 0) {
        const uint8_t  e0 = main_header.QCD->get_exponents(0);
        const uint16_t m0 = main_header.QCD->get_mantissas(0);
        mix_u64((static_cast<uint64_t>(e0) << 16) | m0);
      }
    }
    // COC / QCC / RGN lists: hash the count only.  rtp_recv streams do not
    // use per-component overrides or region-of-interest, so any non-zero
    // count is an unexpected-shape signal that invalidates the cache.
    mix_u64(static_cast<uint64_t>(main_header.COC.size()));
    mix_u64(static_cast<uint64_t>(main_header.QCC.size()));
    mix_u64(static_cast<uint64_t>(main_header.RGN.size()));
  }

  // Populate the output metadata vectors the same way the legacy path does.
  uint16_t num_components = main_header.SIZ->get_num_components();
  element_siz siz, Osiz, Tsiz, TOsiz, Rsiz;
  main_header.SIZ->get_image_size(siz);
  main_header.SIZ->get_image_origin(Osiz);
  main_header.SIZ->get_tile_size(Tsiz);
  main_header.SIZ->get_tile_origin(TOsiz);
  for (uint16_t c = 0; c < num_components; ++c) {
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

  // Cache hit if: we have a cached tile whose tree is built AND whose
  // fingerprint matches the current frame's main-header.  Otherwise drop
  // the cache and treat this frame as a first frame.
  const bool cache_hit = !cached_tileSet_.empty()
                         && cached_tileSet_[0].is_structure_built()
                         && cached_header_fingerprint_ == fp;
  if (!cache_hit) {
    cached_tileSet_.clear();
    cached_tileSet_.resize(1);
    cached_tileSet_[0].dec_init(0, main_header, reduce_NL);
  } else {
    cached_tileSet_[0].prepare_for_next_frame();
    // dec_init is a scalar-only update that's safe to call repeatedly;
    // calling it ensures tile.index / num_components / CCap15 / reduce_NL
    // track any ABI-compatible main-header tweaks (same fingerprint).
    cached_tileSet_[0].dec_init(0, main_header, reduce_NL);
  }

  // Read tile-parts from the current bitstream into the cached (or fresh) tile.
  {
    uint16_t word;
    SOT_marker tmpSOT;
    while (in.get_remaining() >= 2 && (word = in.get_word()) != _EOC) {
      if (word != _SOT) {
        printf("ERROR: SOT marker segment expected but %04X is found\n", word);
        throw std::exception();
      }
      tmpSOT = SOT_marker(in);
      const uint16_t tile_index = tmpSOT.get_tile_index();
      if (tile_index != 0) {
        throw std::runtime_error(
            "openhtj2k_decoder_impl::invoke_line_based_stream_reuse expected a "
            "single-tile codestream (tile_index != 0 found)");
      }
      cached_tileSet_[0].add_tile_part(tmpSOT, in, main_header);
    }
  }

  // Flip persistence on tile_components BEFORE decode_line_based_stream so
  // finalize_line_decode inside the decode call skips teardown.  On the
  // first frame this is the point at which the flag starts holding true;
  // on subsequent frames prepare_for_next_frame already set it.
  cached_tileSet_[0].set_line_decode_persistent_all(true);

  try {
    cached_tileSet_[0].line_based_decode = true;
    install_precinct_filter(cached_tileSet_[0], 0);
    install_packet_observer(cached_tileSet_[0], 0);
    cached_tileSet_[0].create_tile_buf(main_header);
  } catch (std::exception &exc) {
    printf("ERROR: %s\n", exc.what());
    // On a create_tile_buf failure the cache is likely in a bad state —
    // drop it and rethrow.
    cached_tileSet_.clear();
    cached_header_fingerprint_ = 0;
    throw std::runtime_error("Abort Decoding!");
  }

  cached_tileSet_[0].decode_line_based_stream(main_header, reduce_NL, cb, row_limit_,
                                               col_lo_, col_hi_);

  cached_header_fingerprint_ = fp;
}

// ── Direct-to-planar decode ───────────────────────────────────────────────
// Reuses the exact same fingerprint / cache / tile-part machinery as
// invoke_line_based_stream_reuse, but calls decode_line_based_stream_planar
// instead of decode_line_based_stream.  Falls back to the callback path
// when the codestream is multi-tile or when reuse is disabled.
void openhtj2k_decoder_impl::invoke_line_based_direct(
    PlanarOutputDesc *descs, uint16_t nc, std::vector<uint32_t> &width,
    std::vector<uint32_t> &height, std::vector<uint8_t> &depth, std::vector<bool> &is_signed) {
  if (!is_parsed) {
    printf(
        "ERROR: openhtj2k_decoder_impl::parse() shall be called before calling "
        "openhtj2k_decoder_impl::invoke_line_based_direct().\n");
    throw std::exception();
  }
  if (!single_tile_reuse_enabled_) {
    // No single-tile cache — fall back to the callback path with a
    // synthesized callback that writes to the planar descriptors.
    invoke_line_based_stream(
        [&](uint32_t y, int32_t *const *rows, uint16_t nc_cb) {
          for (uint16_t c = 0; c < nc_cb && c < nc; ++c) {
            const PlanarOutputDesc &d = descs[c];
            if (y % d.yr != 0) continue;
            const uint32_t cy = y / d.yr;
            if (cy >= d.height) continue;
            if (d.is_16bit) {
              auto *dst = static_cast<uint16_t *>(d.base) + static_cast<size_t>(cy) * d.stride;
              for (uint32_t x = 0; x < d.width; ++x) {
                int32_t v = rows[c][x];
                if (v < d.minval) v = d.minval;
                if (v > d.maxval) v = d.maxval;
                dst[x] = static_cast<uint16_t>(v);
              }
            } else {
              auto *dst = static_cast<uint8_t *>(d.base) + static_cast<size_t>(cy) * d.stride;
              for (uint32_t x = 0; x < d.width; ++x) {
                int32_t v = rows[c][x];
                if (v < d.minval) v = d.minval;
                if (v > d.maxval) v = d.maxval;
                dst[x] = static_cast<uint8_t>(d.depth_shift > 0 ? (v >> d.depth_shift) : v);
              }
            }
          }
        },
        width, height, depth, is_signed);
    return;
  }
  if (reduce_NL > this->get_max_safe_reduce_NL()) {
    throw std::runtime_error(
        "Attempting to access a non-existent resolution level: -reduce exceeds the\n"
        "maximum safe value for this codestream.  For DFS streams this is the count\n"
        "of consecutive bidirectional DWT levels from the finest; for other streams\n"
        "it is the minimum DWT level count across all tile-components.");
  }

  element_siz numTiles;
  main_header.get_number_of_tiles(numTiles.x, numTiles.y);
  if (numTiles.x != 1 || numTiles.y != 1) {
    cached_tileSet_.clear();
    cached_header_fingerprint_ = 0;
    // Multi-tile: fall back via recursive call with reuse disabled.
    bool was_enabled = single_tile_reuse_enabled_;
    single_tile_reuse_enabled_ = false;
    invoke_line_based_direct(descs, nc, width, height, depth, is_signed);
    single_tile_reuse_enabled_ = was_enabled;
    return;
  }

  // Fingerprint — identical to invoke_line_based_stream_reuse.
  auto fnv1a_u64 = [](uint64_t h, const void *data, size_t len) {
    const uint8_t *p = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < len; ++i) {
      h ^= p[i];
      h *= 1099511628211ull;
    }
    return h;
  };
  uint64_t fp = 14695981039346656037ull;
  auto mix_u64 = [&](uint64_t v) {
    fp = fnv1a_u64(fp, &v, sizeof(v));
  };
  {
    element_siz s_siz, s_osiz, s_tsiz, s_tosiz;
    main_header.SIZ->get_image_size(s_siz);
    main_header.SIZ->get_image_origin(s_osiz);
    main_header.SIZ->get_tile_size(s_tsiz);
    main_header.SIZ->get_tile_origin(s_tosiz);
    const uint16_t ncomp = main_header.SIZ->get_num_components();
    mix_u64((static_cast<uint64_t>(s_siz.x)  << 32) | s_siz.y);
    mix_u64((static_cast<uint64_t>(s_osiz.x) << 32) | s_osiz.y);
    mix_u64((static_cast<uint64_t>(s_tsiz.x) << 32) | s_tsiz.y);
    mix_u64((static_cast<uint64_t>(s_tosiz.x)<< 32) | s_tosiz.y);
    mix_u64(ncomp);
    for (uint16_t c = 0; c < ncomp; ++c) {
      element_siz sub;
      const uint8_t bd = main_header.SIZ->get_bitdepth(c);
      const bool    sn = main_header.SIZ->is_signed(c);
      main_header.SIZ->get_subsampling_factor(sub, c);
      mix_u64((static_cast<uint64_t>(bd) << 56)
              | (static_cast<uint64_t>(sn ? 1 : 0) << 48)
              | (static_cast<uint64_t>(sub.x) << 16)
              | static_cast<uint64_t>(sub.y));
    }
    if (main_header.COD != nullptr) {
      const uint8_t  dl  = main_header.COD->get_dwt_levels();
      const uint8_t  po  = main_header.COD->get_progression_order();
      const uint16_t nl  = main_header.COD->get_number_of_layers();
      const uint8_t  mct = main_header.COD->use_color_trafo();
      const uint8_t  cm  = main_header.COD->get_Cmodes();
      const uint8_t  tx  = main_header.COD->get_transformation();
      element_siz    cbsz; main_header.COD->get_codeblock_size(cbsz);
      mix_u64((static_cast<uint64_t>(dl) << 56) | (static_cast<uint64_t>(po) << 48)
              | (static_cast<uint64_t>(nl) << 32)
              | (static_cast<uint64_t>(mct) << 24) | (static_cast<uint64_t>(cm) << 16)
              | static_cast<uint64_t>(tx));
      mix_u64((static_cast<uint64_t>(cbsz.x) << 32) | cbsz.y);
      for (uint8_t r = 0; r <= dl; ++r) {
        element_siz pp; main_header.COD->get_precinct_size(pp, r);
        mix_u64((static_cast<uint64_t>(pp.x) << 32) | pp.y);
      }
    }
    if (main_header.QCD != nullptr) {
      const uint8_t qs = main_header.QCD->get_quantization_style();
      const uint8_t gb = main_header.QCD->get_number_of_guardbits();
      const uint8_t n0 = main_header.QCD->get_num_entries();
      mix_u64((static_cast<uint64_t>(qs) << 16) | (static_cast<uint64_t>(gb) << 8)
              | static_cast<uint64_t>(n0));
      if (n0 > 0) {
        const uint8_t  e0 = main_header.QCD->get_exponents(0);
        const uint16_t m0 = main_header.QCD->get_mantissas(0);
        mix_u64((static_cast<uint64_t>(e0) << 16) | m0);
      }
    }
    mix_u64(static_cast<uint64_t>(main_header.COC.size()));
    mix_u64(static_cast<uint64_t>(main_header.QCC.size()));
    mix_u64(static_cast<uint64_t>(main_header.RGN.size()));
  }

  // Populate output metadata.
  uint16_t num_components = main_header.SIZ->get_num_components();
  element_siz siz, Osiz, Tsiz, TOsiz, Rsiz;
  main_header.SIZ->get_image_size(siz);
  main_header.SIZ->get_image_origin(Osiz);
  main_header.SIZ->get_tile_size(Tsiz);
  main_header.SIZ->get_tile_origin(TOsiz);
  for (uint16_t c = 0; c < num_components; ++c) {
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

  const bool cache_hit = !cached_tileSet_.empty()
                         && cached_tileSet_[0].is_structure_built()
                         && cached_header_fingerprint_ == fp;
  if (!cache_hit) {
    cached_tileSet_.clear();
    cached_tileSet_.resize(1);
    cached_tileSet_[0].dec_init(0, main_header, reduce_NL);
  } else {
    cached_tileSet_[0].prepare_for_next_frame();
    cached_tileSet_[0].dec_init(0, main_header, reduce_NL);
  }

  {
    uint16_t word;
    SOT_marker tmpSOT;
    while (in.get_remaining() >= 2 && (word = in.get_word()) != _EOC) {
      if (word != _SOT) {
        printf("ERROR: SOT marker segment expected but %04X is found\n", word);
        throw std::exception();
      }
      tmpSOT = SOT_marker(in);
      const uint16_t tile_index = tmpSOT.get_tile_index();
      if (tile_index != 0) {
        throw std::runtime_error(
            "openhtj2k_decoder_impl::invoke_line_based_direct expected a "
            "single-tile codestream (tile_index != 0 found)");
      }
      cached_tileSet_[0].add_tile_part(tmpSOT, in, main_header);
    }
  }

  cached_tileSet_[0].set_line_decode_persistent_all(true);

  try {
    cached_tileSet_[0].line_based_decode = true;
    install_precinct_filter(cached_tileSet_[0], 0);
    install_packet_observer(cached_tileSet_[0], 0);
    cached_tileSet_[0].create_tile_buf(main_header);
  } catch (std::exception &exc) {
    printf("ERROR: %s\n", exc.what());
    cached_tileSet_.clear();
    cached_header_fingerprint_ = 0;
    throw std::runtime_error("Abort Decoding!");
  }

  cached_tileSet_[0].decode_line_based_stream_planar(main_header, reduce_NL, descs, nc);

  cached_header_fingerprint_ = fp;
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
  if (reduce_NL > this->get_max_safe_reduce_NL()) {
    throw std::runtime_error(
        "Attempting to access a non-existent resolution level: -reduce exceeds the\n"
        "maximum safe value for this codestream.  For DFS streams this is the count\n"
        "of consecutive bidirectional DWT levels from the finest; for other streams\n"
        "it is the minimum DWT level count across all tile-components.");
  }

  element_siz numTiles;
  main_header.get_number_of_tiles(numTiles.x, numTiles.y);
  // Validate tile count BEFORE allocating output buffers — see invoke() for rationale.
  if (numTiles.x * numTiles.y > 65535) {
    printf("ERROR: The number of tiles exceeds its allowable maximum (65535).\n");
    throw std::exception();
  }

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
      install_precinct_filter(tileSet[i], static_cast<uint16_t>(i));
      install_packet_observer(tileSet[i], static_cast<uint16_t>(i));
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
