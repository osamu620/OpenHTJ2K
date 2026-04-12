// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.
//
// Licensed under the same BSD 3-Clause terms as the rest of OpenHTJ2K.

#include "decode_helpers.hpp"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "cli.hpp"
#include "color_pipeline.hpp"
#include "decoder.hpp"
#include "frame_handler.hpp"
#include "frame_pipeline.hpp"
#include "gl_renderer.hpp"
#include "planar_shift.hpp"
#include "ycbcr_rgb.hpp"

namespace open_htj2k::rtp_recv {

bool select_coefficients_for_frame(const AssembledFrame& frame, const CliOptions& opts,
                                   const ycbcr_coefficients*& coeffs,
                                   bool& components_are_rgb) {
  coeffs             = nullptr;
  components_are_rgb = false;
  if (frame.has_meta && frame.s) {
    if (frame.mat == 0) {
      // Table 1 "identity" / RGB components: no YCbCr conversion.
      components_are_rgb = true;
    } else {
      coeffs = select_coefficients_from_mat(frame.mat, frame.range);
      if (coeffs == nullptr) {
        std::fprintf(stderr, "frame: unsupported MAT=%u (S=1); dropping\n",
                     static_cast<unsigned>(frame.mat));
        return false;
      }
    }
  } else {
    // S=0: use CLI fallback.  If neither fallback nor rgb mode is set, fail.
    if (opts.s0_label == "rgb") {
      components_are_rgb = true;
    } else if (opts.s0_fallback != nullptr) {
      coeffs = opts.s0_fallback;
    } else {
      std::fprintf(stderr,
                   "frame: Main Packet has S=0 and --colorspace not set; refusing to guess\n");
      return false;
    }
  }
  return true;
}

void log_coefficients_choice_once(const CliOptions& opts, const ycbcr_coefficients* coeffs,
                                  bool components_are_rgb,
                                  const ColorPipelineParams& pipeline) {
  if (components_are_rgb) {
    std::fprintf(stderr, "info: rendering RGB components directly\n");
  } else {
    std::fprintf(stderr, "info: YCbCr -> RGB via %s coefficients\n",
                 coeffs == opts.s0_fallback
                     ? opts.s0_label.c_str()
                     : (coeffs == &YCBCR_BT709_FULL     ? "bt709-full"
                        : coeffs == &YCBCR_BT709_NARROW ? "bt709-narrow"
                        : coeffs == &YCBCR_BT601_FULL   ? "bt601-full"
                        : coeffs == &YCBCR_BT601_NARROW ? "bt601-narrow"
                        : coeffs == &YCBCR_BT2020_FULL  ? "bt2020-full"
                        : coeffs == &YCBCR_BT2020_NARROW? "bt2020-narrow"
                                                        : "?"));
  }
  std::fprintf(stderr,
               "info: colour pipeline transfer=%s gamut=%s display_encoding=%s\n",
               transfer_label(pipeline.transfer),
               gamut_matrix_label(pipeline.gamut_matrix),
               display_encoding_label(pipeline.display_encoding));
}

ColorPipelineParams select_color_pipeline_for_frame(const AssembledFrame& frame,
                                                    const CliOptions&     opts) {
  ColorPipelineParams p;

  // Transfer.  Auto reads the Main Packet TRANS field per H.273 Table 3.
  if (opts.transfer == CliOptions::TransferMode::Gamma) {
    p.transfer = TRANSFER_GAMMA22;
  } else if (opts.transfer == CliOptions::TransferMode::Pq) {
    p.transfer = TRANSFER_PQ;
  } else if (opts.transfer == CliOptions::TransferMode::Hlg) {
    p.transfer = TRANSFER_HLG;
  } else {
    // Auto: use TRANS when S=1, else the CLI fallback.
    p.transfer = opts.transfer_fallback;
    if (frame.has_meta && frame.s) {
      switch (frame.trans) {
        case 1:  // BT.709 gamma
        case 6:  // BT.601
        case 14: // BT.2020 10-bit non-constant-luminance
        case 15: // BT.2020 12-bit non-constant-luminance
          p.transfer = TRANSFER_GAMMA22;
          break;
        case 16:  // SMPTE ST 2084 PQ
          p.transfer = TRANSFER_PQ;
          break;
        case 18:  // ARIB STD-B67 HLG
          p.transfer = TRANSFER_HLG;
          break;
        default:
          // Unknown TRANS: leave on the CLI fallback (default gamma2.2).
          break;
      }
    }
  }

  // Gamut matrix.  The source primaries come from PRIMS (S=1) or from
  // the CLI colourspace fallback; BT.2020 sources targeting a BT.709
  // display need the BT.2020 -> BT.709 matrix, anything else is identity.
  bool source_is_bt2020 = false;
  if (frame.has_meta && frame.s) {
    // H.273 Table 2: PRIMS=9 is BT.2020.
    if (frame.prims == 9) source_is_bt2020 = true;
  } else if (opts.s0_fallback == &YCBCR_BT2020_FULL
             || opts.s0_fallback == &YCBCR_BT2020_NARROW) {
    source_is_bt2020 = true;
  }
  const bool display_is_bt2020 =
      (opts.display_primaries == CliOptions::DisplayPrimaries::Bt2020);
  if (source_is_bt2020 && !display_is_bt2020) {
    p.gamut_matrix = kBt2020ToBt709;
  } else {
    p.gamut_matrix = kIdentityMatrix3;
  }

  // Display encoding: pure CLI pick.
  switch (opts.display_encoding) {
    case CliOptions::DisplayEncoding::Gamma22: p.display_encoding = DISPLAY_ENCODING_GAMMA22; break;
    case CliOptions::DisplayEncoding::Linear:  p.display_encoding = DISPLAY_ENCODING_LINEAR;  break;
    case CliOptions::DisplayEncoding::Srgb:
    default:                                   p.display_encoding = DISPLAY_ENCODING_SRGB;    break;
  }

  return p;
}

bool decode_to_rgb_buffer(open_htj2k::openhtj2k_decoder& decoder,
                          const ycbcr_coefficients* coeffs, bool components_are_rgb,
                          std::vector<uint8_t>& out_rgb, uint32_t& out_w, uint32_t& out_h) {
  uint8_t  depth0          = 0;
  uint32_t cb_stride_ratio = 1;
  uint32_t cr_stride_ratio = 1;
  bool     dims_ok         = true;
  out_w                    = 0;
  out_h                    = 0;

  try {
    std::vector<uint32_t> widths;
    std::vector<uint32_t> heights;
    std::vector<uint8_t>  depths;
    std::vector<bool>     signeds;
    decoder.invoke_line_based_stream_reuse(
        [&](uint32_t y, int32_t* const* rows, uint16_t nc) {
          if (y == 0) {
            if (nc < 1) {
              dims_ok = false;
              return;
            }
            out_w  = widths[0];
            out_h  = heights[0];
            depth0 = depths[0];
            if (nc >= 3) {
              cb_stride_ratio = widths[0] / widths[1];
              cr_stride_ratio = widths[0] / widths[2];
              if (cb_stride_ratio == 0) cb_stride_ratio = 1;
              if (cr_stride_ratio == 0) cr_stride_ratio = 1;
            }
            out_rgb.assign(static_cast<size_t>(out_w) * out_h * 3, 0);
          }
          if (!dims_ok) return;
          uint8_t* out_row = out_rgb.data() + static_cast<size_t>(y) * out_w * 3;
          if (nc >= 3 && !components_are_rgb && coeffs != nullptr) {
            ycbcr_row_to_rgb8(rows[0], rows[1], rows[2], out_row, out_w, cb_stride_ratio,
                              cr_stride_ratio, *coeffs, depth0,
                              /*is_signed=*/signeds[0]);
          } else if (nc >= 3 && components_are_rgb) {
            rgb_row_to_rgb8(rows[0], rows[1], rows[2], out_row, out_w, depth0);
          } else if (nc == 1) {
            // Grayscale: replicate Y into R/G/B.
            const int32_t shift  = static_cast<int32_t>(depth0) - 8;
            const int32_t maxval = (1 << depth0) - 1;
            for (uint32_t x = 0; x < out_w; ++x) {
              int32_t v = rows[0][x];
              if (v < 0) v = 0;
              if (v > maxval) v = maxval;
              const uint8_t v8   = static_cast<uint8_t>(shift > 0 ? (v >> shift) : v);
              out_row[3 * x + 0] = v8;
              out_row[3 * x + 1] = v8;
              out_row[3 * x + 2] = v8;
            }
          }
        },
        widths, heights, depths, signeds);
  } catch (std::exception& e) {
    std::fprintf(stderr, "decoder.invoke_line_based_stream failed: %s\n", e.what());
    return false;
  }

  return dims_ok && out_w > 0 && out_h > 0;
}

bool decode_to_planar_buffers(open_htj2k::openhtj2k_decoder& decoder, bool components_are_rgb,
                              DecodedFrame& df) {
  df.rgb.clear();
  df.plane_y.clear();
  df.plane_cb.clear();
  df.plane_cr.clear();
  df.plane_y_16.clear();
  df.plane_cb_16.clear();
  df.plane_cr_16.clear();
  df.width         = 0;
  df.height        = 0;
  df.chroma_width  = 0;
  df.chroma_height = 0;
  df.bit_depth     = 0;
  df.kind          = components_are_rgb ? DecodedFrame::PLANAR_RGB : DecodedFrame::PLANAR_YCBCR;

  bool     dims_ok    = true;
  uint32_t luma_w     = 0;
  uint32_t luma_h     = 0;
  uint32_t chroma_w_0 = 0;
  uint32_t chroma_h_0 = 0;
  uint8_t  depth_y    = 0;
  uint8_t  depth_c    = 0;
  bool     use_16     = false;  // true when depth_y > 8 -> take the GL_R16 path

  try {
    std::vector<uint32_t> widths;
    std::vector<uint32_t> heights;
    std::vector<uint8_t>  depths;
    std::vector<bool>     signeds;
    decoder.invoke_line_based_stream_reuse(
        [&](uint32_t y, int32_t* const* rows, uint16_t nc) {
          if (y == 0) {
            if (nc < 1) {
              dims_ok = false;
              return;
            }
            luma_w  = widths[0];
            luma_h  = heights[0];
            depth_y = depths[0];
            if (nc >= 3) {
              chroma_w_0 = widths[1];
              chroma_h_0 = heights[1];
              depth_c    = depths[1];
              // 3 planes assumed equal chroma dims (422 / 420 / 444).
              if (widths.size() >= 3 && heights.size() >= 3) {
                if (widths[2] != chroma_w_0 || heights[2] != chroma_h_0) {
                  // Mismatched chroma planes (asymmetric subsampling).
                  // v3 day one doesn't handle this; dims_ok stays true
                  // but we force single-plane width/height so the
                  // upload still produces something sane.
                  chroma_w_0 = std::min(chroma_w_0, widths[2]);
                  chroma_h_0 = std::min(chroma_h_0, heights[2]);
                }
              }
            } else {
              // Grayscale: replicate Y into Cb/Cr with neutral chroma.
              chroma_w_0 = luma_w;
              chroma_h_0 = luma_h;
              depth_c    = depth_y;
            }
            df.width         = luma_w;
            df.height        = luma_h;
            df.chroma_width  = chroma_w_0;
            df.chroma_height = chroma_h_0;
            df.bit_depth     = depth_y;
            use_16           = (depth_y > 8);
            if (use_16) {
              // 16-bit plane path.  Neutral chroma in the u16 space is the
              // midpoint of the source's [0, (1<<depth)-1] range.
              const uint16_t neutral_c = components_are_rgb
                                             ? 0
                                             : static_cast<uint16_t>(1u << (depth_c - 1));
              df.plane_y_16.assign(static_cast<size_t>(luma_w) * luma_h, 0);
              df.plane_cb_16.assign(static_cast<size_t>(chroma_w_0) * chroma_h_0, neutral_c);
              df.plane_cr_16.assign(static_cast<size_t>(chroma_w_0) * chroma_h_0, neutral_c);
            } else {
              df.plane_y.assign(static_cast<size_t>(luma_w) * luma_h, 0);
              df.plane_cb.assign(static_cast<size_t>(chroma_w_0) * chroma_h_0,
                                 components_are_rgb ? 0 : 128);
              df.plane_cr.assign(static_cast<size_t>(chroma_w_0) * chroma_h_0,
                                 components_are_rgb ? 0 : 128);
            }
          }
          if (!dims_ok) return;

          const int32_t maxval_y = (1 << depth_y) - 1;

          // Luma row -- every call has a Y row at the current y.
          if (use_16) {
            clamp_i32_plane_to_u16(
                rows[0], df.plane_y_16.data() + static_cast<size_t>(y) * luma_w, luma_w,
                maxval_y);
          } else {
            const int32_t shift_y = static_cast<int32_t>(depth_y) - 8;
            shift_i32_plane_to_u8(rows[0],
                                  df.plane_y.data() + static_cast<size_t>(y) * luma_w,
                                  luma_w, shift_y, maxval_y);
          }

          // Chroma (or G/B for PLANAR_RGB).  Written only for rows that
          // land on the chroma grid.  For 4:2:2 the chroma height equals
          // luma height and every luma row contributes.  For 4:2:0 the
          // luma height is 2x chroma; rows[] pointers may be nullptr on
          // the "skip" luma rows -- guard with the callback nc check.
          if (nc >= 3) {
            const int32_t maxval_c = (1 << depth_c) - 1;
            const uint32_t yc      = (luma_h > 0)
                                         ? static_cast<uint32_t>(
                                               static_cast<uint64_t>(y) * chroma_h_0 / luma_h)
                                         : 0;
            if (yc < chroma_h_0) {
              if (use_16) {
                if (rows[1] != nullptr) {
                  clamp_i32_plane_to_u16(
                      rows[1],
                      df.plane_cb_16.data() + static_cast<size_t>(yc) * chroma_w_0,
                      chroma_w_0, maxval_c);
                }
                if (rows[2] != nullptr) {
                  clamp_i32_plane_to_u16(
                      rows[2],
                      df.plane_cr_16.data() + static_cast<size_t>(yc) * chroma_w_0,
                      chroma_w_0, maxval_c);
                }
              } else {
                const int32_t shift_c = static_cast<int32_t>(depth_c) - 8;
                if (rows[1] != nullptr) {
                  shift_i32_plane_to_u8(
                      rows[1],
                      df.plane_cb.data() + static_cast<size_t>(yc) * chroma_w_0,
                      chroma_w_0, shift_c, maxval_c);
                }
                if (rows[2] != nullptr) {
                  shift_i32_plane_to_u8(
                      rows[2],
                      df.plane_cr.data() + static_cast<size_t>(yc) * chroma_w_0,
                      chroma_w_0, shift_c, maxval_c);
                }
              }
            }
          }
        },
        widths, heights, depths, signeds);
  } catch (std::exception& e) {
    std::fprintf(stderr, "decoder.invoke_line_based_stream failed: %s\n", e.what());
    return false;
  }

  return dims_ok && df.width > 0 && df.height > 0;
}

bool decode_and_present(const AssembledFrame& frame, const CliOptions& opts, bool is_first_frame,
                        GlRenderer* renderer, std::vector<uint8_t>& rgb_backbuffer,
                        DecodedFrame& planar_scratch) {
  using namespace open_htj2k;

  openhtj2k_decoder decoder(frame.bytes.data(), frame.bytes.size(), /*reduce_NL=*/0,
                            opts.num_decoder_threads);
  try {
    decoder.parse();
  } catch (std::exception& e) {
    std::fprintf(stderr, "decoder.parse() failed: %s\n", e.what());
    return false;
  }

  const ycbcr_coefficients* coeffs = nullptr;
  bool                      components_are_rgb = false;
  if (!select_coefficients_for_frame(frame, opts, coeffs, components_are_rgb)) return false;
  const ColorPipelineParams pipeline = select_color_pipeline_for_frame(frame, opts);
  if (is_first_frame)
    log_coefficients_choice_once(opts, coeffs, components_are_rgb, pipeline);

  if (opts.color_path == CliOptions::ColorPath::Shader) {
    if (!decode_to_planar_buffers(decoder, components_are_rgb, planar_scratch)) {
      std::fprintf(stderr, "frame: unable to determine dimensions; dropping\n");
      return false;
    }
    if (renderer != nullptr) {
      if (planar_scratch.bit_depth > 8) {
        renderer->upload_planar_16_and_draw(
            planar_scratch.plane_y_16.data(), planar_scratch.plane_cb_16.data(),
            planar_scratch.plane_cr_16.data(), static_cast<int>(planar_scratch.width),
            static_cast<int>(planar_scratch.height),
            static_cast<int>(planar_scratch.chroma_width),
            static_cast<int>(planar_scratch.chroma_height),
            static_cast<int>(planar_scratch.bit_depth), coeffs, components_are_rgb,
            pipeline);
      } else {
        renderer->upload_planar_and_draw(
            planar_scratch.plane_y.data(), planar_scratch.plane_cb.data(),
            planar_scratch.plane_cr.data(), static_cast<int>(planar_scratch.width),
            static_cast<int>(planar_scratch.height),
            static_cast<int>(planar_scratch.chroma_width),
            static_cast<int>(planar_scratch.chroma_height), coeffs, components_are_rgb,
            pipeline);
      }
    }
  } else {
    uint32_t out_w = 0;
    uint32_t out_h = 0;
    if (!decode_to_rgb_buffer(decoder, coeffs, components_are_rgb, rgb_backbuffer, out_w, out_h)) {
      std::fprintf(stderr, "frame: unable to determine dimensions; dropping\n");
      return false;
    }
    if (renderer != nullptr) {
      renderer->upload_and_draw(rgb_backbuffer.data(), static_cast<int>(out_w),
                                static_cast<int>(out_h));
    }
  }
  return true;
}

void dump_frame_if_requested(const CliOptions& opts, const AssembledFrame& frame,
                             uint64_t frame_index) {
  if (opts.dump_pattern.empty()) return;
  char path[512];
  std::snprintf(path, sizeof(path), opts.dump_pattern.c_str(),
                static_cast<unsigned>(frame_index));
  FILE* fp = std::fopen(path, "wb");
  if (!fp) {
    std::fprintf(stderr, "dump: fopen('%s') failed: %s\n", path, std::strerror(errno));
    return;
  }
  std::fwrite(frame.bytes.data(), 1, frame.bytes.size(), fp);
  std::fclose(fp);
}

}  // namespace open_htj2k::rtp_recv
