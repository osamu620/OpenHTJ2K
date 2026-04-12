// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.
//
// Licensed under the same BSD 3-Clause terms as the rest of OpenHTJ2K.

#pragma once

#include <cstdint>
#include <vector>

#include "cli.hpp"
#include "color_pipeline.hpp"
#include "frame_handler.hpp"
#include "frame_pipeline.hpp"
#include "gl_renderer.hpp"
#include "ycbcr_rgb.hpp"

// Forward declaration from the core library.
namespace open_htj2k {
class openhtj2k_decoder;
}

namespace open_htj2k::rtp_recv {

// Picks YCbCr->RGB coefficients per RFC 9828 §5.3 (S=1 in-band, S=0 CLI
// fallback).  Sets `components_are_rgb` if no conversion is needed (Table 1
// identity components or `--colorspace rgb`).  Logs and returns false on
// rejection.  Used by both the v1 single-threaded decoder path and the v2
// decode worker.
bool select_coefficients_for_frame(const AssembledFrame& frame, const CliOptions& opts,
                                   const ycbcr_coefficients*& coeffs,
                                   bool& components_are_rgb);

void log_coefficients_choice_once(const CliOptions& opts, const ycbcr_coefficients* coeffs,
                                  bool components_are_rgb,
                                  const ColorPipelineParams& pipeline);

// Select the HDR colour pipeline params for this frame.
ColorPipelineParams select_color_pipeline_for_frame(const AssembledFrame& frame,
                                                    const CliOptions&     opts);

// Run invoke_line_based_stream on `decoder` and write the result as 8-bit
// interleaved RGB into `out_rgb` (resized as needed).  `decoder.parse()`
// must already have been called.  On success returns true and fills out_w,
// out_h with the luma dimensions; on failure logs and returns false.
bool decode_to_rgb_buffer(open_htj2k::openhtj2k_decoder& decoder,
                          const ycbcr_coefficients* coeffs, bool components_are_rgb,
                          std::vector<uint8_t>& out_rgb, uint32_t& out_w, uint32_t& out_h);

// Shader-path twin of decode_to_rgb_buffer.  Runs invoke_line_based_stream
// and writes each component's int32 samples into one of three 8-bit or
// 16-bit planar buffers.  On success, `df` is populated with plane data,
// dimensions, and kind.  On failure returns false.
bool decode_to_planar_buffers(open_htj2k::openhtj2k_decoder& decoder, bool components_are_rgb,
                              DecodedFrame& df);

void dump_frame_if_requested(const CliOptions& opts, const AssembledFrame& frame,
                             uint64_t frame_index);

// v1 path: decode one reassembled HTJ2K codestream by constructing a fresh
// openhtj2k_decoder, then upload to the renderer.  Used only when
// --threading=off.  Returns true on success.
bool decode_and_present(const AssembledFrame& frame, const CliOptions& opts, bool is_first_frame,
                        GlRenderer* renderer, std::vector<uint8_t>& rgb_backbuffer,
                        DecodedFrame& planar_scratch);

}  // namespace open_htj2k::rtp_recv
