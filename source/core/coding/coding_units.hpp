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

#include "j2kmarkers.hpp"

#include <atomic>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>

/********************************************************************************
 * cblk_data_pool — monotonic bump allocator for HTJ2K codeblock bitstreams
 *
 * Single-threaded bump allocator for HTJ2K encoder compressed bitstreams.
 * Each encoder thread owns its own instance via a per-thread slot (g_tl_pool_slot),
 * so no synchronisation is needed — bump() and trim() are plain arithmetic.
 * Replaces ~11,520 individual malloc() calls per 4K tile with a handful of 2 MB slabs.
 *******************************************************************************/
class cblk_data_pool {
  static constexpr size_t SLAB_BYTES = 2u << 20;  // 2 MB per slab
  struct Slab {
    uint8_t *ptr;
    size_t used;
    size_t cap;
  };
  std::vector<Slab> slabs;

 public:
  cblk_data_pool() { slabs.reserve(8); }
  cblk_data_pool(const cblk_data_pool &)          = delete;
  cblk_data_pool &operator=(const cblk_data_pool &) = delete;

  ~cblk_data_pool() {
    for (auto &s : slabs) free(s.ptr);
  }

  // Allocate n bytes (rounded to 8-byte alignment).  NOT thread-safe.
  uint8_t *bump(size_t n) {
    n = (n + 7u) & ~7u;
    if (slabs.empty() || slabs.back().used + n > slabs.back().cap) {
      const size_t cap = (n > SLAB_BYTES) ? n : SLAB_BYTES;
      slabs.push_back({static_cast<uint8_t *>(malloc(cap)), 0u, cap});
    }
    uint8_t *p = slabs.back().ptr + slabs.back().used;
    slabs.back().used += n;
    return p;
  }

  // Release the last n bytes of the most-recent bump() (must still be in the same slab).
  void trim(size_t n) noexcept {
    if (!slabs.empty() && slabs.back().used >= n) slabs.back().used -= n;
  }

  // Reset all slabs to empty, keeping the allocated pages so they are already
  // faulted-in for the next tile encode (avoids repeated mmap/munmap overhead).
  void reset() noexcept {
    for (auto &s : slabs) s.used = 0;
    // Collapse to one slab to bound long-term memory footprint.
    while (slabs.size() > 1) {
      free(slabs.back().ptr);
      slabs.pop_back();
    }
  }
};

/********************************************************************************
 * j2k_region
 *******************************************************************************/
class j2k_region {
 public:
  // top-left coordinate (inclusive) of a region in the reference grid
  element_siz pos0;
  // bottom-right coordinate (exclusive) of a region in the reference grid
  element_siz pos1;
  // width for line buffer
  uint32_t stride;

  // return top-left coordinate (inclusive)
  OPENHTJ2K_NODISCARD element_siz get_pos0() const { return pos0; }
  // return bottom-right coordinate (exclusive)
  OPENHTJ2K_NODISCARD element_siz get_pos1() const { return pos1; }
  // get size of a region
  void get_size(element_siz &out) const {
    out.x = pos1.x - pos0.x;
    out.y = pos1.y - pos0.y;
  }
  // set top-left coordinate (inclusive)
  void set_pos0(element_siz in) { pos0 = in; }
  // set bottom-right coordinate (exclusive)
  void set_pos1(element_siz in) { pos1 = in; }
  j2k_region() = default;
  j2k_region(element_siz p0, element_siz p1) : pos0(p0), pos1(p1), stride(round_up(pos1.x - pos0.x, 32U)) {}
};

/********************************************************************************
 * j2k_codeblock
 *******************************************************************************/
class j2k_codeblock : public j2k_region {
 public:
  const element_siz size;

 private:
  uint8_t *compressed_data;
  uint8_t *current_address;
  const uint8_t band;
  const uint8_t M_b;
  OPENHTJ2K_MAYBE_UNUSED const uint32_t index;
  // When true, compressed_data was bump-allocated by a cblk_data_pool owned by
  // j2k_tile. The destructor must NOT free it; the pool manages the lifetime.
  bool compressed_is_pooled = false;

 public:
  int32_t *sample_buf;
  size_t blksampl_stride;
  uint8_t *block_states;
  size_t blkstate_stride;
  sprec_t *i_samples;
  const uint32_t band_stride;
  OPENHTJ2K_MAYBE_UNUSED const uint8_t R_b;
  const uint8_t transformation;
  const float stepsize;

  const uint16_t num_layers;

  uint32_t length;
  uint16_t Cmodes;
  uint8_t num_passes;
  uint8_t num_ZBP;
  uint8_t fast_skip_passes;
  uint8_t Lblock;
  // length of a coding pass in bytes (fixed array, max 128 coding passes)
  uint32_t pass_length[128];
  uint8_t pass_length_count;
  // non-owning: pooled by the owning j2k_precinct_subband
  uint8_t *layer_start;
  uint8_t *layer_passes;
  bool already_included;
  bool refsegment;

  j2k_codeblock(const uint32_t &idx, uint8_t orientation, uint8_t M_b, uint8_t R_b, uint8_t transformation,
                float stepsize, uint32_t band_stride, sprec_t *ibuf, uint32_t offset,
                const uint16_t &numlayers, const uint8_t &codeblock_style, const element_siz &p0,
                const element_siz &p1, const element_siz &s);
  ~j2k_codeblock() {
    if (compressed_data != nullptr && !compressed_is_pooled) {
      free(compressed_data);
    }
  }
  //  void modify_state(const std::function<void(uint8_t &, uint8_t)> &callback, uint8_t val, int16_t j1,
  //                    int16_t j2) {
  //    callback(
  //        block_states[static_cast<uint32_t>(j1 + 1) * (blkstate_stride) + static_cast<uint32_t>(j2 + 1)],
  //        val);
  //  }
  //  uint8_t get_state(const std::function<uint8_t(uint8_t &)> &callback, int16_t j1, int16_t j2) const {
  //    return (uint8_t)callback(
  //        block_states[static_cast<uint32_t>(j1 + 1) * (blkstate_stride) + static_cast<uint32_t>(j2 +
  //        1)]);
  //  }
  OPENHTJ2K_NODISCARD uint8_t get_orientation() const { return band; }

  //  OPENHTJ2K_NODISCARD uint8_t get_context_label_sig(const uint32_t &j1, const uint32_t &j2) const;
  //  OPENHTJ2K_NODISCARD uint8_t get_signLUT_index(const uint32_t &j1, const uint32_t &j2) const;
  OPENHTJ2K_NODISCARD uint8_t get_Mb() const;
  uint8_t *get_compressed_data();
  void set_compressed_data(uint8_t *buf, uint16_t size, uint16_t Lref = 0);
  void create_compressed_buffer(buf_chain *tile_buf, int32_t buf_limit, const uint16_t &layer);
  // Single-tile reuse: clear every field touched by parse_packet_header /
  // create_compressed_buffer so the codeblock returns to its post-ctor
  // "ready to parse" state, without deallocating structural resources.
  // Frees compressed_data if it is a non-pooled owned buffer (matching
  // the destructor's ownership convention).
  void reset_for_next_frame();
  //  void update_sample(const uint8_t &symbol, const uint8_t &p, const int16_t &j1, const int16_t &j2)
  //  const; void update_sign(const int8_t &val, const uint32_t &j1, const uint32_t &j2) const;
  //  OPENHTJ2K_NODISCARD uint8_t get_sign(const uint32_t &j1, const uint32_t &j2) const;
  void quantize(uint32_t &or_val);
  uint8_t calc_mbr(uint32_t i, uint32_t j, uint8_t causal_cond) const;
  void dequantize(uint8_t ROIshift) const;
};

/********************************************************************************
 * j2k_subband
 *******************************************************************************/
class j2k_subband : public j2k_region {
 public:
  uint8_t orientation;
  uint8_t transformation;
  uint8_t R_b;
  OPENHTJ2K_MAYBE_UNUSED uint8_t epsilon_b;
  OPENHTJ2K_MAYBE_UNUSED uint16_t mantissa_b;
  uint8_t M_b;
  float delta;
  OPENHTJ2K_MAYBE_UNUSED float nominal_range;
  sprec_t *i_samples;

  // j2k_subband();
  j2k_subband(element_siz p0, element_siz p1, uint8_t orientation, uint8_t transformation, uint8_t R_b,
              uint8_t epsilon_b, uint16_t mantissa_b, uint8_t M_b, float delta, float nominal_range,
              sprec_t *ibuf, bool no_alloc = false);
  ~j2k_subband();
  void destroy() {
    if (orientation != BAND_LL) {
      if (i_samples != nullptr) aligned_mem_free(i_samples - DWT_LEFT_SLACK);
      i_samples = nullptr;
    }
  }
};

/********************************************************************************
 * j2k_precinct_subband
 *******************************************************************************/
class j2k_precinct_subband : public j2k_region {
 private:
  OPENHTJ2K_MAYBE_UNUSED const uint8_t orientation;
  // Embedded directly (was previously raw `tagtree *` with `new`/`delete`).
  // Default-constructed when num_codeblocks==0 (stays empty); otherwise move-assigned in ctor.
  tagtree inclusion_info;
  tagtree ZBP_info;
  // Flat array of codeblock objects (placement-new'd into a single allocation)
  j2k_codeblock *codeblocks;
  // Single slab backing layer_start[] and layer_passes[] for all codeblocks
  std::unique_ptr<uint8_t[]> cb_layer_pool;

 public:
  uint32_t num_codeblock_x;
  uint32_t num_codeblock_y;
  j2k_precinct_subband(uint8_t orientation, uint8_t M_b, uint8_t R_b, uint8_t transformation,
                       float stepsize, sprec_t *ibuf, const element_siz &bp0, const element_siz &p0,
                       const element_siz &p1, const uint32_t stride, const uint16_t &num_layers,
                       const element_siz &codeblock_size, const uint8_t &Cmodes);
  ~j2k_precinct_subband() {
    const uint32_t N = num_codeblock_x * num_codeblock_y;
    for (uint32_t i = 0; i < N; ++i) {
      codeblocks[i].~j2k_codeblock();
    }
    operator delete[](codeblocks);
    // tagtree members and cb_layer_pool unique_ptr destruct automatically.
  }
  //  void destroy_codeblocks() {
  //    for (uint32_t i = 0; i < num_codeblock_x * num_codeblock_y; ++i) {
  //      delete codeblocks[i];
  //    }
  //    delete[] codeblocks;
  //  }
  tagtree_node *get_inclusion_node(uint32_t i);
  tagtree_node *get_ZBP_node(uint32_t i);
  j2k_codeblock *access_codeblock(uint32_t i);
  void parse_packet_header(buf_chain *packet_header, uint16_t layer_idx, uint16_t Ccap15);
  void generate_packet_header(packet_header_writer &header, uint16_t layer_idx);
  // Single-tile reuse: reset both tagtrees and every codeblock's per-frame
  // state so parse_packet_header can be called again.  Keeps all structural
  // allocations alive (codeblocks[], cb_layer_pool, tagtree_node arrays).
  void reset_for_next_frame();
};

/********************************************************************************
 * j2k_precinct
 *******************************************************************************/
class j2k_precinct : public j2k_region {
 private:
  // index of this precinct
  OPENHTJ2K_MAYBE_UNUSED const uint32_t index;
  // index of resolution level to which this precinct belongs
  const uint8_t resolution;
  // number of subbands in this precinct (DFS-dependent)
  uint8_t num_bands;
  // length which includes packet header and body, used only for encoder
  uint32_t length;
  // Flat array of precinct-subbands (placement-new'd into a single allocation).
  // Replaces the old `unique_ptr<unique_ptr<T>[]>` double-indirection — one heap
  // allocation per precinct instead of (1 + num_bands).
  j2k_precinct_subband *pband;

 public:
  // buffer for generated packet header: only for encoding
  std::unique_ptr<uint8_t[]> packet_header;
  // length of packet header
  uint32_t packet_header_length;

 public:
  j2k_precinct(const uint8_t &r, const uint32_t &idx, const element_siz &p0, const element_siz &p1,
               const j2k_subband *subband, const uint16_t &num_layers,
               const element_siz &codeblock_size, const uint8_t &Cmodes, uint8_t nb = 0,
               dwt_type dfs_dir = DWT_BIDIR);
  ~j2k_precinct() {
    for (uint8_t i = 0; i < num_bands; ++i) {
      pband[i].~j2k_precinct_subband();
    }
    operator delete[](pband);
  }
  // Disable copy/move — owns raw pointer with manual destruction.
  j2k_precinct(const j2k_precinct &)            = delete;
  j2k_precinct &operator=(const j2k_precinct &) = delete;
  j2k_precinct(j2k_precinct &&)                 = delete;
  j2k_precinct &operator=(j2k_precinct &&)      = delete;

  j2k_precinct_subband *access_pband(uint8_t b);
  void set_length(uint32_t len) { length = len; }
  OPENHTJ2K_NODISCARD uint32_t get_length() const { return length; }
  OPENHTJ2K_NODISCARD uint8_t get_num_bands() const { return num_bands; }
};

/********************************************************************************
 * j2c_packet
 *******************************************************************************/
class j2c_packet {
 public:
  OPENHTJ2K_MAYBE_UNUSED uint16_t layer;
  OPENHTJ2K_MAYBE_UNUSED uint8_t resolution;
  OPENHTJ2K_MAYBE_UNUSED uint16_t component;
  OPENHTJ2K_MAYBE_UNUSED uint32_t precinct;
  OPENHTJ2K_MAYBE_UNUSED buf_chain *header;
  OPENHTJ2K_MAYBE_UNUSED buf_chain *body;
  // only for encoder
  std::unique_ptr<uint8_t[]> buf;
  uint32_t length;

  j2c_packet()
      : layer(0), resolution(0), component(0), precinct(0), header(nullptr), body(nullptr), length(0){};
  // constructor for decoding
  j2c_packet(const uint16_t l, const uint8_t r, const uint16_t c, const uint32_t p,
             buf_chain *const h = nullptr, buf_chain *const bo = nullptr)
      : layer(l), resolution(r), component(c), precinct(p), header(h), body(bo), length(0) {}
  // constructor for encoding
  j2c_packet(uint16_t l, uint8_t r, uint16_t c, uint32_t p, j2k_precinct *cp, uint8_t num_bands);
};

/********************************************************************************
 * j2k_resolution
 *******************************************************************************/
class j2k_resolution : public j2k_region {
 private:
  // resolution level
  const uint8_t index;
  // Flat array of precincts (placement-new'd, single allocation per resolution).
  j2k_precinct *precincts;
  uint32_t num_precincts;  // == npw * nph for non-empty resolutions, else 0
  // Flat array of subbands (placement-new'd, single allocation per resolution).
  j2k_subband *subbands;
  // nominal ranges of subbands
  float child_ranges[4]{};

 public:
  // number of subbands (DFS-dependent; 1 for LL and HORZ/VERT levels, 3 for BIDIR)
  uint8_t num_bands;
  // DWT type for this resolution (DWT_BIDIR for standard, DWT_HORZ/VERT when DFS active)
  dwt_type transform_direction;
  // number of precincts wide
  const uint32_t npw;
  // number of precincts height
  const uint32_t nph;
  // a resolution is empty if it has no precincts
  const bool is_empty;
  // post-shift value for inverse DWT
  uint8_t normalizing_upshift;
  // pre-shift value for forward DWT
  uint8_t normalizing_downshift;
  sprec_t *i_samples;
  j2k_resolution(const uint8_t &r, const element_siz &p0, const element_siz &p1, const uint32_t &npw,
                 const uint32_t &nph, bool no_alloc = false, uint8_t nb = 0,
                 dwt_type dir = DWT_BIDIR);
  ~j2k_resolution();
  OPENHTJ2K_MAYBE_UNUSED uint8_t get_index() const { return index; }
  void create_subbands(element_siz &p0, element_siz &p1, uint8_t NL, uint8_t transformation,
                       std::vector<uint8_t> &exponents, std::vector<uint16_t> &mantissas,
                       uint8_t num_guard_bits, uint8_t qstyle, uint8_t bitdepth,
                       bool line_based = false, const DFS_marker *dfs = nullptr);
  void create_precincts(element_siz PP, uint16_t num_layers, element_siz codeblock_size, uint8_t Cmodes);

  j2k_precinct *access_precinct(uint32_t p);
  j2k_subband *access_subband(uint8_t b);
  void set_nominal_ranges(const float *ranges) {
    child_ranges[0] = ranges[0];
    child_ranges[1] = ranges[1];
    child_ranges[2] = ranges[2];
    child_ranges[3] = ranges[3];
  }
  void destroy() {
    if (i_samples != nullptr) aligned_mem_free(i_samples - DWT_LEFT_SLACK);
    i_samples = nullptr;
    if (subbands != nullptr) {
      for (uint8_t b = 0; b < num_bands; ++b) {
        subbands[b].destroy();
      }
    }
  }
  // Disable copy/move — owns raw arrays with manual destruction.
  j2k_resolution(const j2k_resolution &)            = delete;
  j2k_resolution &operator=(const j2k_resolution &) = delete;
  j2k_resolution(j2k_resolution &&)                 = delete;
  j2k_resolution &operator=(j2k_resolution &&)      = delete;
};

/********************************************************************************
 * j2k_tile_part
 *******************************************************************************/
class j2k_tile_part {
 private:
  // tile index to which this tile-part belongs
  uint16_t tile_index;
  // tile-part index
  uint8_t tile_part_index;
  // pointer to tile-part buffer
  uint8_t *body;
  // length of tile-part
  uint32_t length;

 public:
  // pointer to tile-part header
  std::unique_ptr<j2k_tilepart_header> header;
  explicit j2k_tile_part(uint16_t num_components);
  void set_SOT(SOT_marker &tmpSOT);
  int read(j2c_src_memory &);
  OPENHTJ2K_MAYBE_UNUSED OPENHTJ2K_NODISCARD uint16_t get_tile_index() const;
  OPENHTJ2K_MAYBE_UNUSED OPENHTJ2K_NODISCARD uint8_t get_tile_part_index() const;
  OPENHTJ2K_NODISCARD uint32_t get_length() const;
  uint8_t *get_buf();
  void set_tile_index(uint16_t t);
  void set_tile_part_index(uint8_t tp);
};

/********************************************************************************
 * j2k_tile_base
 *******************************************************************************/
class j2k_tile_base : public j2k_region {
 public:
  // number of DWT decomposition levels
  uint8_t NL;
  // resolution reduction
  uint8_t reduce_NL;
  // code-block width and height
  element_siz codeblock_size;
  // codeblock style (Table A.19)
  uint8_t Cmodes;
  // DWT type (Table A.20), 0:9x7, 1:5x3
  uint8_t transformation;
  // precinct width and height as exponents of the power of 2
  std::vector<element_siz> precinct_size;
  // quantization style (Table A.28)
  uint8_t quantization_style;
  // exponents of step sizes
  std::vector<uint8_t> exponents;
  // mantissas of step sizes
  std::vector<uint16_t> mantissas;
  // number of guard bits
  uint8_t num_guard_bits;
  j2k_tile_base() : reduce_NL(0) {}
};

/********************************************************************************
 * j2k_tile_component
 *******************************************************************************/
class j2k_tile_component : public j2k_tile_base {
 private:
  // component index
  uint16_t index;
  // pointer to sample buffer (integer)
  int32_t *samples;
  // shift value for ROI
  uint8_t ROIshift;
  // DFS index for this component (0 = DFS not active)
  uint8_t dfs_index = 0;
  // Flat array of j2k_resolution objects (placement-new'd, single allocation per tile component).
  j2k_resolution *resolution;
  uint8_t num_resolutions;  // == NL + 1 when allocated, else 0
  // opaque line-decode state (allocated by init_line_decode, freed by finalize_line_decode)
  std::unique_ptr<struct j2k_tcomp_line_dec> line_dec;
  // opaque line-encode state (allocated by init_line_encode, freed by finalize_line_encode)
  std::unique_ptr<struct j2k_tcomp_line_enc> line_enc;
  // Single-tile reuse: when true, init_line_decode() short-circuits to a
  // cursor-only reset if line_dec is already allocated, and
  // finalize_line_decode() leaves line_dec alive.  The destructor clears the
  // flag before calling finalize_line_decode() so full teardown still runs
  // when the j2k_tile_component itself is destroyed.
  bool line_dec_persistent_ = false;
  // set members related to COC marker
  void setCOCparams(COC_marker *COC);
  // set members related to QCC marker
  void setQCCparams(QCC_marker *QCC);
  // set ROIshift from RGN marker
  void setRGNparams(RGN_marker *RGN);

 public:
  // component bit-depth
  uint8_t bitdepth;
  // LB encode: raw input pointer + dc-offset parameters (avoids allocating tcomp->samples)
  const int32_t *lb_src_ptr = nullptr;
  uint32_t lb_src_stride    = 0;
  int32_t lb_dc_offset      = 0;
  int32_t lb_dc_shiftup     = 0;
  bool lb_enc_mode          = false;  // true in any LB encode path (streaming or buffered)
  // default constructor
  j2k_tile_component();
  // destructor
  ~j2k_tile_component();
  // Disable copy/move — owns raw pointer arrays (resolution[]) with manual destruction.
  j2k_tile_component(const j2k_tile_component &)            = delete;
  j2k_tile_component &operator=(const j2k_tile_component &) = delete;
  j2k_tile_component(j2k_tile_component &&)                 = delete;
  j2k_tile_component &operator=(j2k_tile_component &&)      = delete;
  // initialization of coordinates and parameters defined in tile-part markers
  void init(j2k_main_header *hdr, j2k_tilepart_header *tphdr, j2k_tile_base *tile, uint16_t c,
            std::vector<int32_t *> img = {}, bool lb_enc = false);
  int32_t *get_sample_address(uint32_t x, uint32_t y);
  uint8_t get_dwt_levels();
  uint8_t get_transformation();
  OPENHTJ2K_MAYBE_UNUSED OPENHTJ2K_NODISCARD uint8_t get_Cmodes() const;
  OPENHTJ2K_MAYBE_UNUSED OPENHTJ2K_NODISCARD uint8_t get_bitdepth() const;
  element_siz get_precinct_size(uint8_t r);
  OPENHTJ2K_MAYBE_UNUSED element_siz get_codeblock_size();
  OPENHTJ2K_MAYBE_UNUSED OPENHTJ2K_NODISCARD uint8_t get_ROIshift() const;
  j2k_resolution *access_resolution(uint8_t r);
  void create_resolutions(uint16_t numlayers, bool line_based = false, bool enc_lb = false);
  // DFS/ATK pointers resolved during init; nullptr = not active for this component.
  const DFS_marker *dfs_info = nullptr;
  const ATK_marker *atk_info = nullptr;

  void perform_dc_offset(uint8_t transformation, bool is_signed);

  // ── Line-based decode API ─────────────────────────────────────────────────
  // init_line_decode(): must be called after all packets are parsed.
  // pull_line():        returns the next decoded row (float) into out[0..width-1].
  //                     Returns false when all rows are exhausted.
  // pull_line_ref():    zero-copy variant — returns a pointer into the internal ring
  //                     buffer (valid until the next pull call).  Returns nullptr
  //                     when exhausted.  The caller may modify the row in-place.
  // finalize_line_decode(): frees state allocated by init_line_decode().
  void init_line_decode(bool ring_mode = false);
  bool pull_line(sprec_t *out);
  sprec_t *pull_line_ref();
  // Pull `count` consecutive rows via pull_line_ref() + memcpy into a
  // per-component aligned strip scratch buffer held by line_dec.  Returns a
  // pointer to the first row (stride = stride_floats), or nullptr if
  // line_dec is not initialised.  Grows the scratch buffer on demand.
  // Intended for the strip-granular decode_line_based_stream driver — the
  // copy is necessary because idwt_2d_state's ring is too shallow to keep a
  // whole outer strip pinned.
  sprec_t *pull_strip_into_buf(uint32_t count, uint32_t stride_floats);
  void finalize_line_decode();
  // Mark all subband row bufs in line_dec as bypass (for pre-decoded diagnostic).
  void mark_line_dec_predecoded();
  // Single-tile reuse: reset line-decode cursor state (next_row, strip_y0,
  // prefetch_y0, etc. on every subband_row_buf) without freeing ring buffers
  // or idwt states.  A subsequent init_line_decode() must detect that
  // line_dec is already allocated and short-circuit the allocation path.
  void reset_line_decode_cursors();
  // Single-tile reuse opt-in.  When true, init_line_decode reuses an existing
  // line_dec (cursor reset only) and finalize_line_decode is a no-op.  When
  // false, both functions behave as before.  The destructor forces false
  // before calling finalize_line_decode to guarantee cleanup.
  void set_line_decode_persistent(bool on) { line_dec_persistent_ = on; }

  // ── Line-based encode API ─────────────────────────────────────────────────
  // init_line_encode():     allocates FDWT state chain; call after enc_init().
  // push_line_enc(in):      feeds one float input row into the FDWT chain.
  // finalize_line_encode(): flushes states, fills subband buffers, frees state.
  void init_line_encode();
  void push_line_enc(const sprec_t *in);
  void finalize_line_encode();
  struct j2k_tcomp_line_enc *get_line_enc() { return line_enc.get(); }

  void destroy() {
    if (resolution != nullptr) {
      for (uint8_t r = 0; r < this->NL; ++r) {
        resolution[r].destroy();
      }
    }
  }
};

/********************************************************************************
 * j2k_tile
 *******************************************************************************/
class j2k_tile : public j2k_tile_base {
 private:
  // vector array of tile-parts
  std::vector<std::unique_ptr<j2k_tile_part>> tile_part;
  // index of this tile
  uint16_t index;
  // number of components
  uint16_t num_components;
  // SOP is used or not (Table A.13)
  bool use_SOP;
  // EPH is used or not (Table A.13)
  bool use_EPH;
  // progression order (Table A.16)
  uint8_t progression_order;
  // number of layers (Table A.14)
  uint16_t numlayers;
  // multiple component transform (Table A.17)
  uint8_t MCT;

  // length of tile (in bytes)
  uint32_t length;
  // pointer to tile buffer
  std::unique_ptr<buf_chain> tile_buf;
  // pointer to packet header
  buf_chain *packet_header;
  // buffer for PPM marker segments
  buf_chain sbst_packet_header;
  // number of tile-parts
  uint8_t num_tile_part;
  // position of current tile-part
  int current_tile_part_pos;
  // unique pointer to tile-components
  std::unique_ptr<j2k_tile_component[]> tcomp;
  // pointer to packet header in PPT marker segments
  std::unique_ptr<buf_chain> ppt_header;
  // number_of_packets (for encoder only)
  uint32_t num_packets;
  // unique pointer to packets
  std::unique_ptr<j2c_packet[]> packet;
  // value of Ccap15 parameter in CAP marker segment
  uint16_t Ccap15;
  // progression order information for both COD and POC
  POC_marker porder_info;
  // Set to true after create_tile_buf has allocated the resolution/precinct/
  // codeblock tree.  When openhtj2k_decoder_impl is driving this tile through
  // the single-tile reuse path, prepare_for_next_frame() clears per-frame
  // mutable state but leaves structure_built_ set, and create_tile_buf
  // short-circuits its allocation steps on the next call.
  bool structure_built_ = false;
 public:
  // Bump-allocator pool for HTJ2K encode compressed bitstreams (one pool per thread).
  struct EncodePoolCtx {
    std::vector<std::unique_ptr<cblk_data_pool>> pools;
    uint32_t gen = 0;
    std::atomic<int> slot_cnt{0};
    // Grow-only scratch buffers for codeblock sample/state data during HT block encoding.
    // Allocated once per tile (sized to the largest resolution's codeblock count) and
    // reused across all resolution levels and precincts — eliminating per-resolution
    // malloc/free cycles that cause expensive mmap/munmap page-fault pressure on Linux.
    int32_t *gbuf      = nullptr;
    uint8_t *sgbuf     = nullptr;
    size_t   gbuf_cap  = 0;  // capacity in int32_t elements
    size_t   sgbuf_cap = 0;  // capacity in uint8_t elements

    ~EncodePoolCtx() {
      std::free(gbuf);
      std::free(sgbuf);
    }

    // Ensure gbuf/sgbuf have at least the requested capacity (never shrink).
    void reserve_scratch(size_t need_g, size_t need_sg) {
      if (need_g > gbuf_cap) {
        std::free(gbuf);
        gbuf     = static_cast<int32_t *>(std::malloc(need_g * sizeof(int32_t)));
        gbuf_cap = need_g;
      }
      if (need_sg > sgbuf_cap) {
        std::free(sgbuf);
        sgbuf     = static_cast<uint8_t *>(std::malloc(need_sg));
        sgbuf_cap = need_sg;
      }
    }
  };
 private:
  std::unique_ptr<EncodePoolCtx> encode_pool_ctx;
  // return SOP is used or not
  OPENHTJ2K_NODISCARD bool is_use_SOP() const { return this->use_SOP; }
  // return EPH is used or not
  OPENHTJ2K_MAYBE_UNUSED OPENHTJ2K_NODISCARD bool is_use_EPH() const { return this->use_EPH; }
  // set members related to COD marker
  void setCODparams(COD_marker *COD);
  // set members related to QCD marker
  void setQCDparams(QCD_marker *QCD);
  // read packets
  void read_packet(j2k_precinct *current_precint, uint16_t layer, uint8_t num_band);
  // function to retrieve greatest common divisor of precinct size among resolution levels
  void find_gcd_of_precinct_size(element_siz &out);

 public:
  // When true, j2k_subband constructors skip large non-LL buffer allocation.
  // Set this before calling create_tile_buf() to enable ring-mode decoder RSS savings.
  bool line_based_decode = false;
  j2k_tile();
  void destroy() {
    for (uint16_t c = 0; c < this->num_components; ++c) {
      tcomp[c].destroy();
    }
  }
  // Decoding
  // Initialization with tile-index
  void dec_init(uint16_t idx, j2k_main_header &main_header, uint8_t reduce_levels);
  // read and add a tile_part into a tile
  void add_tile_part(SOT_marker &tmpSOT, j2c_src_memory &in, j2k_main_header &main_header);
  // create buffer to store compressed data for decoding
  void create_tile_buf(j2k_main_header &main_header);
  // Single-tile reuse: clear every piece of mutable per-frame state while
  // keeping the tile/component/resolution/precinct/codeblock allocations
  // (and their sub-allocations — ring buffers, compressed buffers, tagtrees,
  // line_dec) alive.  After this returns, the caller may re-populate the
  // tile via the normal add_tile_part() loop and then re-call
  // create_tile_buf() — which will short-circuit its structural allocations
  // because structure_built_ stays true — and then decode_line_based_stream().
  void prepare_for_next_frame();
  // Read-only accessor used by openhtj2k_decoder_impl to decide whether a
  // cached tile is in "tree is allocated" state (= reuse path) or not (=
  // still a fresh j2k_tile that has never seen a codestream).
  OPENHTJ2K_NODISCARD bool is_structure_built() const { return structure_built_; }
  // Flip persistence on all tile_components of this tile.  Called by the
  // reuse entry point on the first frame, just before decode_line_based_stream,
  // so finalize_line_decode() skips teardown and keeps line_dec alive for
  // the next call.  Calling it is idempotent.
  void set_line_decode_persistent_all(bool on);
  // decoding (does block decoding and IDWT) function for a tile
  void decode();
  // inverse color transform
  void ycbcr_to_rgb();
  // inverse DC offset and clipping
  void finalize(j2k_main_header &main_header, uint8_t reduce_NL, std::vector<int32_t *> &dst);
  // Line-based decode: parses packets first (via create_tile_buf), then lazily
  // pulls float rows component-by-component, applies per-row YCbCr→RGB and
  // float→int32 conversion.  Does NOT call decode() / ycbcr_to_rgb() / finalize().
  void decode_line_based(j2k_main_header &main_header, uint8_t reduce_NL,
                         std::vector<int32_t *> &dst);
  // Streaming variant: same as decode_line_based() but outputs one row at a time via
  // a callback instead of writing to a pre-allocated full-image buffer.
  // The callback receives (y, row_ptrs[NC], NC) where row_ptrs[c] points to one
  // decoded int32_t row for component c.  Allocates only per-row scratch buffers.
  void decode_line_based_stream(
      j2k_main_header &main_header, uint8_t reduce_NL,
      const std::function<void(uint32_t y, int32_t *const *, uint16_t nc)> &cb);
  // Diagnostic variant: decodes all codeblocks first (no IDWT), then uses
  // the pre-decoded sb->i_samples to bypass decode_strip() in row_ptr().
  // Used by lb_compare to isolate decode_strip bugs from IDWT state machine bugs.
  void decode_line_based_predecoded(j2k_main_header &main_header, uint8_t reduce_NL,
                                    std::vector<int32_t *> &dst);

  // Encoding
  // Initialization with tile-index
  void enc_init(uint16_t idx, j2k_main_header &main_header, std::vector<int32_t *> img,
                bool line_based = false, bool streaming = false);
  // DC offsetting
  int perform_dc_offset(j2k_main_header &main_header);
  // forward color transform
  void rgb_to_ycbcr();
  // encoding (does block encoding and FDWT) function for a tile
  uint8_t *encode();
  // line-based encoding: uses stateful FDWT instead of batch FDWT
  uint8_t *encode_line_based();
  // streaming line-based encoding: pulls rows via callback instead of pre-allocated buffer
  uint8_t *encode_line_based_stream(
      std::function<void(uint32_t y, int32_t **rows, uint16_t nc)> src_fn,
      const std::vector<uint32_t> &img_comp_widths);
  // create packets in encoding
  void construct_packets(j2k_main_header &main_header);
  // write packets into destination
  void write_packets(j2c_dst_memory &outbuf);

  // getters
  OPENHTJ2K_MAYBE_UNUSED OPENHTJ2K_NODISCARD uint16_t get_numlayers() const { return this->numlayers; }
  j2k_tile_component *get_tile_component(uint16_t c);

  OPENHTJ2K_MAYBE_UNUSED OPENHTJ2K_MAYBE_UNUSED uint8_t get_byte_from_tile_buf();
  OPENHTJ2K_MAYBE_UNUSED uint8_t get_bit_from_tile_buf();
  OPENHTJ2K_NODISCARD uint32_t get_length() const;
  OPENHTJ2K_MAYBE_UNUSED uint32_t get_buf_length();
};

int32_t htj2k_encode(j2k_codeblock *block, uint8_t ROIshift) noexcept;
