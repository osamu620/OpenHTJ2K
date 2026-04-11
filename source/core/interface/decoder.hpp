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
#include <functional>
#include <memory>
#include <vector>
#if defined(_MSC_VER) && !defined(OHTJ2K_STATIC)
  #define OPENHTJ2K_EXPORT __declspec(dllexport)
#else
  #define OPENHTJ2K_EXPORT
#endif
namespace open_htj2k {
// EnumCS values from the JPH/JP2 colour specification box (ISO/IEC 15444-1 Annex I).
// get_colorspace() returns one of these; 0 means the input was a raw codestream.
static constexpr uint32_t ENUMCS_SRGB      = 16u;
static constexpr uint32_t ENUMCS_GRAYSCALE = 17u;
static constexpr uint32_t ENUMCS_YCBCR     = 18u;

class openhtj2k_decoder {
 private:
  std::unique_ptr<class openhtj2k_decoder_impl> impl;

 public:
  OPENHTJ2K_EXPORT openhtj2k_decoder();
  OPENHTJ2K_EXPORT openhtj2k_decoder(const char *, uint8_t reduce_NL, uint32_t num_threads);
  OPENHTJ2K_EXPORT openhtj2k_decoder(const uint8_t *, size_t, uint8_t reduce_NL, uint32_t num_threads);
  OPENHTJ2K_EXPORT void init(const uint8_t *, size_t, uint8_t reduce_NL, uint32_t num_threads);
  OPENHTJ2K_EXPORT void parse();
  OPENHTJ2K_EXPORT uint16_t get_num_component();
  OPENHTJ2K_EXPORT uint32_t get_component_width(uint16_t);
  OPENHTJ2K_EXPORT uint32_t get_component_height(uint16_t);
  OPENHTJ2K_EXPORT uint8_t get_component_depth(uint16_t);
  OPENHTJ2K_EXPORT bool get_component_signedness(uint16_t);
  OPENHTJ2K_EXPORT uint8_t get_minumum_DWT_levels();  // note: typo preserved for ABI compat
  OPENHTJ2K_EXPORT uint8_t get_max_safe_reduce_NL();
  // Returns the EnumCS value from the JPH/JP2 colour specification box, or 0 for raw codestreams.
  // Compare against open_htj2k::ENUMCS_SRGB / ENUMCS_GRAYSCALE / ENUMCS_YCBCR.
  OPENHTJ2K_EXPORT uint32_t get_colorspace();
  OPENHTJ2K_EXPORT void invoke(std::vector<int32_t *> &, std::vector<uint32_t> &, std::vector<uint32_t> &,
                               std::vector<uint8_t> &, std::vector<bool> &);
  // Line-based decode: same signature as invoke() but uses stateful row-pull
  // instead of full-tile IDWT.  Peak memory proportional to DWT ring depth
  // rather than image size.
  OPENHTJ2K_EXPORT void invoke_line_based(std::vector<int32_t *> &, std::vector<uint32_t> &,
                                          std::vector<uint32_t> &, std::vector<uint8_t> &,
                                          std::vector<bool> &);
  // Streaming variant of invoke_line_based(): outputs one row at a time via a callback
  // instead of writing to a pre-allocated full-image buffer.  Avoids allocating W×H
  // output buffers entirely — only per-row scratch is needed.  width/height/depth/is_signed
  // are populated before the first callback invocation so the callback can use them.
  // The callback receives (y, row_ptrs[NC], NC) and must copy the data if needed.
  OPENHTJ2K_EXPORT void invoke_line_based_stream(
      std::function<void(uint32_t y, int32_t *const *, uint16_t nc)> cb,
      std::vector<uint32_t> &width, std::vector<uint32_t> &height, std::vector<uint8_t> &depth,
      std::vector<bool> &is_signed);
  // Opt-in single-tile streaming variant: when enable_single_tile_reuse(true)
  // has been set, consecutive calls reuse the decoded tile tree (codeblock
  // allocations, precinct tagtrees, line-decode ring buffers) so per-frame
  // init cost is proportional to the new bitstream's packet headers instead
  // of to a full tile rebuild.  Only valid for codestreams whose main-header
  // bytes (SIZ/COD/COC/QCD/QCC/RGN) are byte-identical across calls — a
  // fingerprint check invalidates the cache automatically when that changes.
  // Falls through to invoke_line_based_stream() for the first call after a
  // cache invalidation or when reuse is disabled.
  OPENHTJ2K_EXPORT void invoke_line_based_stream_reuse(
      std::function<void(uint32_t y, int32_t *const *, uint16_t nc)> cb,
      std::vector<uint32_t> &width, std::vector<uint32_t> &height, std::vector<uint8_t> &depth,
      std::vector<bool> &is_signed);
  // Enable the single-tile reuse optimization (default off).  Call once
  // after constructing the decoder and before the first init()/parse()
  // sequence for the stream you want to keep cached.  Passing false drops
  // any cached state and returns the decoder to the legacy per-frame path.
  OPENHTJ2K_EXPORT void enable_single_tile_reuse(bool on);
  // Diagnostic: pre-decodes codeblocks via the tile-at-a-time path, then runs
  // the line-based IDWT using those pre-decoded values.  If this matches invoke()
  // but invoke_line_based() does not, the bug is in decode_strip(); otherwise
  // the bug is in idwt_2d_state.
  OPENHTJ2K_EXPORT void invoke_line_based_predecoded(std::vector<int32_t *> &, std::vector<uint32_t> &,
                                                     std::vector<uint32_t> &, std::vector<uint8_t> &,
                                                     std::vector<bool> &);
  OPENHTJ2K_EXPORT void destroy();
  OPENHTJ2K_EXPORT ~openhtj2k_decoder();
};
}  // namespace open_htj2k
