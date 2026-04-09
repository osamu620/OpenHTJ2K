#ifdef __EMSCRIPTEN__
  #include <cstdint>
  #include <cstdio>
  #include <cstring>
  #include <string>
  #include <vector>
  #include <exception>

  #include <emscripten.h>
  #if defined(OPENHTJ2K_ENABLE_WASM_SIMD)
    #include <wasm_simd128.h>
  #endif

  #include "decoder.hpp"

open_htj2k::openhtj2k_decoder* cpp_create_decoder(uint8_t* data, size_t size, uint8_t reduce_NL) {
  return new open_htj2k::openhtj2k_decoder(data, size, reduce_NL, 1);
}

void cpp_parse_j2c_data(open_htj2k::openhtj2k_decoder* dec) { dec->parse(); }
void cpp_invoke_decoder(open_htj2k::openhtj2k_decoder* dec, int32_t* out) {
  const uint16_t C = dec->get_num_component();
  std::vector<int32_t*> buf;
  std::vector<uint32_t> img_width;
  std::vector<uint32_t> img_height;
  std::vector<uint8_t> img_depth;
  std::vector<bool> img_signed;
  dec->invoke(buf, img_width, img_height, img_depth, img_signed);

  const uint32_t W = img_width[0];
  const uint32_t H = img_height[0];

  if (C == 1) {
    // Grayscale: direct copy — no interleaving needed
    std::memcpy(out, buf[0], (size_t)W * H * sizeof(int32_t));
  } else {
    // Planar → interleaved row by row.
    // Inner loop reads buf[c] sequentially (cache-friendly) and writes
    // with stride C; LLVM auto-vectorizes to v128 scatter ops.
    for (uint32_t y = 0; y < H; ++y) {
      int32_t* __restrict__ dst = out + (size_t)y * W * C;
      for (uint16_t c = 0; c < C; ++c) {
        const int32_t* __restrict__ src = buf[c] + (size_t)y * W;
        for (uint32_t x = 0; x < W; ++x) {
          dst[x * C + c] = src[x];
        }
      }
    }
  }
}

void cpp_release_j2c_data(open_htj2k::openhtj2k_decoder* dec) { delete dec; }

uint16_t cpp_get_num_components(open_htj2k::openhtj2k_decoder* dec) { return dec->get_num_component(); };

uint32_t cpp_get_width(open_htj2k::openhtj2k_decoder* dec, uint16_t c) {
  return dec->get_component_width(c);
}

uint32_t cpp_get_height(open_htj2k::openhtj2k_decoder* dec, uint16_t c) {
  return dec->get_component_height(c);
}

uint8_t cpp_get_depth(open_htj2k::openhtj2k_decoder* dec, uint16_t c) {
  return dec->get_component_depth(c);
}

bool cpp_get_signed(open_htj2k::openhtj2k_decoder* dec, uint16_t c) {
  return dec->get_component_signedness(c);
}

uint8_t cpp_get_minimum_DWT_levels(open_htj2k::openhtj2k_decoder* dec) {
  return dec->get_minumum_DWT_levels();
}

//////////////////////////////////////////////////////////////////////////////////////////////////
extern "C" {
EMSCRIPTEN_KEEPALIVE
open_htj2k::openhtj2k_decoder* create_decoder(uint8_t* data, size_t size, uint8_t reduce_NL) {
  return cpp_create_decoder(data, size, reduce_NL);
}

EMSCRIPTEN_KEEPALIVE
void parse_j2c_data(open_htj2k::openhtj2k_decoder* dec) { cpp_parse_j2c_data(dec); }

EMSCRIPTEN_KEEPALIVE
void invoke_decoder(open_htj2k::openhtj2k_decoder* dec, int32_t* out) { cpp_invoke_decoder(dec, out); }

// Decode and write per-component planar buffers at native resolution.
// dst_ptrs[c] receives W[c] × H[c] int32 values for component c.
// Handles different component sizes (subsampled images) correctly.
EMSCRIPTEN_KEEPALIVE
void invoke_decoder_planar(open_htj2k::openhtj2k_decoder* dec, int32_t** dst_ptrs) {
  std::vector<int32_t*> buf;
  std::vector<uint32_t> img_width, img_height;
  std::vector<uint8_t> img_depth;
  std::vector<bool> img_signed;
  dec->invoke(buf, img_width, img_height, img_depth, img_signed);
  uint16_t nc = static_cast<uint16_t>(buf.size());
  for (uint16_t c = 0; c < nc; ++c) {
    std::memcpy(dst_ptrs[c], buf[c], (size_t)img_width[c] * img_height[c] * sizeof(int32_t));
  }
}

EMSCRIPTEN_KEEPALIVE
void release_j2c_data(open_htj2k::openhtj2k_decoder* dec) { cpp_release_j2c_data(dec); }

EMSCRIPTEN_KEEPALIVE
uint16_t get_num_components(open_htj2k::openhtj2k_decoder* dec) { return cpp_get_num_components(dec); }

EMSCRIPTEN_KEEPALIVE
uint32_t get_width(open_htj2k::openhtj2k_decoder* dec, uint16_t c) { return cpp_get_width(dec, c); }

EMSCRIPTEN_KEEPALIVE
uint32_t get_height(open_htj2k::openhtj2k_decoder* dec, uint16_t c) { return cpp_get_height(dec, c); }

EMSCRIPTEN_KEEPALIVE
uint32_t get_depth(open_htj2k::openhtj2k_decoder* dec, uint16_t c) { return cpp_get_depth(dec, c); }

EMSCRIPTEN_KEEPALIVE
uint32_t get_signed(open_htj2k::openhtj2k_decoder* dec, uint16_t c) { return cpp_get_signed(dec, c); }

EMSCRIPTEN_KEEPALIVE
uint32_t get_minimum_DWT_levels(open_htj2k::openhtj2k_decoder* dec) {
  return cpp_get_minimum_DWT_levels(dec);
}

EMSCRIPTEN_KEEPALIVE
uint32_t get_max_safe_reduce_NL(open_htj2k::openhtj2k_decoder* dec) {
  return dec->get_max_safe_reduce_NL();
}

EMSCRIPTEN_KEEPALIVE
uint32_t get_colorspace(open_htj2k::openhtj2k_decoder* dec) {
  return dec->get_colorspace();
}

// invoke_decoder_to_rgba: decode and convert directly to 8-bit RGBA in one pass.
// rgba_dst must be pre-allocated as W × H × 4 bytes.
// Handles all bit depths (1–16) and signed/unsigned samples; correct down/up-shift
// and rounding (half-pel bias) are applied in C++ rather than in JS.
// WASM-SIMD paths process 4 pixels per iteration for both grayscale and color.
EMSCRIPTEN_KEEPALIVE
void invoke_decoder_to_rgba(open_htj2k::openhtj2k_decoder* dec, uint8_t* rgba_dst) {
  std::vector<uint32_t> width, height;
  std::vector<uint8_t>  depth;
  std::vector<bool>     is_signed;

  dec->invoke_line_based_stream(
    [&](uint32_t y, int32_t* const* rows, uint16_t nc) {
      const uint32_t W        = width[0];
      // For subsampled formats (e.g. 4:2:2), chroma components have smaller width.
      // Detect this and map output pixel x to chroma sample x>>1 in the scalar paths.
      const uint32_t chromaW  = (nc >= 2) ? width[1] : W;
      const bool     subsamp  = (chromaW < W);
      const uint8_t  d        = depth[0];
      const int32_t  down_sh  = (d >= 8) ? (d - 8) : 0;
      const int32_t  up_sh    = (d < 8)  ? (8 - d) : 0;
      const int32_t  half     = (d > 8)  ? (1 << (down_sh - 1)) : 0;
      const int32_t  offset   = is_signed[0] ? (1 << (d - 1)) : 0;
      const int32_t  bias     = half + offset;

      uint8_t* __restrict__ row = rgba_dst + (size_t)y * W * 4;

#if defined(OPENHTJ2K_ENABLE_WASM_SIMD)
      const v128_t vbias  = wasm_i32x4_splat(bias);
      const v128_t vzero  = wasm_i32x4_const_splat(0);
      const v128_t vmax   = wasm_i32x4_const_splat(255);
      const v128_t valpha = wasm_i8x16_const_splat(-1);  // 0xFF bytes
      uint32_t x = 0;
      if (nc == 1) {
        for (; x + 4 <= W; x += 4) {
          v128_t v = wasm_v128_load(rows[0] + x);
          v = wasm_i32x4_add(v, vbias);
          v = wasm_i32x4_shr(v, down_sh);
          v = wasm_i32x4_shl(v, up_sh);
          v = wasm_i32x4_min(wasm_i32x4_max(v, vzero), vmax);
          // narrow int32×4 → uint8×4 (values sit in byte positions 0-3)
          v128_t n8 = wasm_u8x16_narrow_i16x8(wasm_i16x8_narrow_i32x4(v, v),
                                              wasm_i16x8_narrow_i32x4(v, v));
          // replicate each gray byte to RGB and insert alpha=0xFF
          // [g0,g0,g0,FF, g1,g1,g1,FF, g2,g2,g2,FF, g3,g3,g3,FF]
          v128_t rgba = wasm_i8x16_shuffle(n8, valpha,
              0,0,0,16, 1,1,1,16, 2,2,2,16, 3,3,3,16);
          wasm_v128_store(row + x * 4, rgba);
        }
      } else if (!subsamp) {
        // Full-resolution color SIMD path (not used for subsampled chroma — falls to scalar tail).
        for (; x + 4 <= W; x += 4) {
          // Process R, G, B channels — each loads 4 int32 values and narrows
          v128_t rv = wasm_v128_load(rows[0] + x);
          rv = wasm_i32x4_min(wasm_i32x4_max(wasm_i32x4_shl(wasm_i32x4_shr(
               wasm_i32x4_add(rv, vbias), down_sh), up_sh), vzero), vmax);
          v128_t gv = wasm_v128_load(rows[1] + x);
          gv = wasm_i32x4_min(wasm_i32x4_max(wasm_i32x4_shl(wasm_i32x4_shr(
               wasm_i32x4_add(gv, vbias), down_sh), up_sh), vzero), vmax);
          v128_t bv = wasm_v128_load(rows[2] + x);
          bv = wasm_i32x4_min(wasm_i32x4_max(wasm_i32x4_shl(wasm_i32x4_shr(
               wasm_i32x4_add(bv, vbias), down_sh), up_sh), vzero), vmax);
          // narrow each channel: int32×4 → uint8 bytes at positions 0-3
          v128_t rn = wasm_u8x16_narrow_i16x8(wasm_i16x8_narrow_i32x4(rv, rv),
                                              wasm_i16x8_narrow_i32x4(rv, rv));
          v128_t gn = wasm_u8x16_narrow_i16x8(wasm_i16x8_narrow_i32x4(gv, gv),
                                              wasm_i16x8_narrow_i32x4(gv, gv));
          v128_t bn = wasm_u8x16_narrow_i16x8(wasm_i16x8_narrow_i32x4(bv, bv),
                                              wasm_i16x8_narrow_i32x4(bv, bv));
          // interleave: RG pairs then BA pairs, then combine
          // rn[0..3]=r0..r3, gn[0..3]=g0..g3, bn[0..3]=b0..b3
          v128_t rg = wasm_i8x16_shuffle(rn, gn,
              0,16,1,17,2,18,3,19, 0,16,1,17,2,18,3,19); // [r0,g0,r1,g1,r2,g2,r3,g3,...]
          v128_t ba = wasm_i8x16_shuffle(bn, valpha,
              0,16,1,17,2,18,3,19, 0,16,1,17,2,18,3,19); // [b0,FF,b1,FF,b2,FF,b3,FF,...]
          v128_t rgba = wasm_i8x16_shuffle(rg, ba,
              0,1,16,17, 2,3,18,19, 4,5,20,21, 6,7,22,23); // [r0,g0,b0,FF,...]
          wasm_v128_store(row + x * 4, rgba);
        }
      }
      // scalar tail (and full scalar for subsamp, nc>3, or up_sh paths)
      if (nc == 1) {
        for (; x < W; ++x) {
          int32_t v = ((rows[0][x] + bias) >> down_sh) << up_sh;
          if (v < 0) v = 0; else if (v > 255) v = 255;
          uint8_t u = static_cast<uint8_t>(v);
          row[x*4+0] = row[x*4+1] = row[x*4+2] = u; row[x*4+3] = 255;
        }
      } else {
        for (; x < W; ++x) {
          for (uint16_t c = 0; c < 3 && c < nc; ++c) {
            uint32_t sx = (c == 0 || !subsamp) ? x : (x >> 1);  // nearest-neighbour chroma upsample
            int32_t v = ((rows[c][sx] + bias) >> down_sh) << up_sh;
            if (v < 0) v = 0; else if (v > 255) v = 255;
            row[x*4+c] = static_cast<uint8_t>(v);
          }
          row[x*4+3] = 255;
        }
      }
#else
      if (nc == 1) {
        for (uint32_t x = 0; x < W; ++x) {
          int32_t v = ((rows[0][x] + bias) >> down_sh) << up_sh;
          if (v < 0) v = 0; else if (v > 255) v = 255;
          uint8_t u = static_cast<uint8_t>(v);
          row[x*4+0] = row[x*4+1] = row[x*4+2] = u; row[x*4+3] = 255;
        }
      } else {
        for (uint32_t x = 0; x < W; ++x) {
          for (uint16_t c = 0; c < 3 && c < nc; ++c) {
            uint32_t sx = (c == 0 || !subsamp) ? x : (x >> 1);  // nearest-neighbour chroma upsample
            int32_t v = ((rows[c][sx] + bias) >> down_sh) << up_sh;
            if (v < 0) v = 0; else if (v > 255) v = 255;
            row[x*4+c] = static_cast<uint8_t>(v);
          }
          row[x*4+3] = 255;
        }
      }
#endif
    },
    width, height, depth, is_signed);
}

// invoke_decoder_stream: decode using invoke_line_based_stream() so that the
// internal planar tile buffers and the full W×H×C int32 output buffer are
// never simultaneously live.  Rows are interleaved and packed directly into
// dst (already sized W × H × nc × bytes_per_sample) as they are produced.
// Peak WASM heap is reduced by ~(W × H × nc × 4) bytes vs invoke_decoder().
EMSCRIPTEN_KEEPALIVE
void invoke_decoder_stream(open_htj2k::openhtj2k_decoder* dec, uint8_t* dst,
                           int32_t maxval, int32_t bytes_per_sample,
                           int32_t apply_dc_offset = 1) {
  std::vector<uint32_t> width, height;
  std::vector<uint8_t>  depth;
  std::vector<bool>     is_signed;

  // Precompute per-component DC level shift offsets for signed components.
  // JPEG 2000 stores signed data with a DC offset of 2^(depth-1); the inverse
  // transform subtracts it, leaving negative values.  For PGM/PPM output we
  // add it back so values fall in [0, maxval].  For PGX, skip (raw signed).
  int32_t dc_offset[16] = {};
  // defer filling until nc is known (inside the lambda on first call)
  bool dc_offset_ready = false;

  dec->invoke_line_based_stream(
    [&](uint32_t y, int32_t* const* rows, uint16_t nc) {
      if (!dc_offset_ready) {
        if (apply_dc_offset) {
          for (uint16_t c = 0; c < nc; ++c)
            dc_offset[c] = is_signed[c] ? (1 << (depth[c] - 1)) : 0;
        }
        dc_offset_ready = true;
      }
      const uint32_t W       = width[0];
      // Detect chroma subsampling (e.g. 4:2:2): chroma rows are narrower than luma.
      const uint32_t chromaW = (nc >= 2) ? width[1] : W;
      const bool     subsamp = (chromaW < W);
      uint8_t* row_dst = dst + (size_t)y * W * nc * bytes_per_sample;
      if (nc == 1) {
        // Grayscale — single component, already "interleaved"
        const int32_t off0 = dc_offset[0];
#if defined(OPENHTJ2K_ENABLE_WASM_SIMD)
        const v128_t vmx   = wasm_i32x4_splat(maxval);
        const v128_t vzero = wasm_i32x4_const_splat(0);
        const v128_t voff  = wasm_i32x4_splat(off0);
        uint32_t x = 0;
        if (bytes_per_sample == 1) {
          for (; x + 16 <= W; x += 16) {
            v128_t a = wasm_i32x4_min(wasm_i32x4_max(wasm_i32x4_add(wasm_v128_load(rows[0] + x),      voff), vzero), vmx);
            v128_t b = wasm_i32x4_min(wasm_i32x4_max(wasm_i32x4_add(wasm_v128_load(rows[0] + x +  4), voff), vzero), vmx);
            v128_t c = wasm_i32x4_min(wasm_i32x4_max(wasm_i32x4_add(wasm_v128_load(rows[0] + x +  8), voff), vzero), vmx);
            v128_t d = wasm_i32x4_min(wasm_i32x4_max(wasm_i32x4_add(wasm_v128_load(rows[0] + x + 12), voff), vzero), vmx);
            wasm_v128_store(row_dst + x,
                            wasm_u8x16_narrow_i16x8(wasm_i16x8_narrow_i32x4(a, b),
                                                    wasm_i16x8_narrow_i32x4(c, d)));
          }
          for (; x < W; ++x) {
            int32_t v = rows[0][x] + off0; if (v < 0) v = 0; else if (v > maxval) v = maxval;
            row_dst[x] = static_cast<uint8_t>(v);
          }
        } else {
          uint8_t* d = row_dst;
          for (; x + 8 <= W; x += 8) {
            v128_t a = wasm_i32x4_min(wasm_i32x4_max(wasm_i32x4_add(wasm_v128_load(rows[0] + x),     voff), vzero), vmx);
            v128_t b = wasm_i32x4_min(wasm_i32x4_max(wasm_i32x4_add(wasm_v128_load(rows[0] + x + 4), voff), vzero), vmx);
            v128_t u16le = wasm_u16x8_narrow_i32x4(a, b);
            v128_t u16be = wasm_i8x16_shuffle(u16le, u16le,
                                              1,0, 3,2, 5,4, 7,6, 9,8, 11,10, 13,12, 15,14);
            wasm_v128_store(d, u16be); d += 16;
          }
          for (; x < W; ++x) {
            int32_t v = rows[0][x] + off0; if (v < 0) v = 0; else if (v > maxval) v = maxval;
            *d++ = static_cast<uint8_t>(v >> 8); *d++ = static_cast<uint8_t>(v & 0xff);
          }
        }
#else
        if (bytes_per_sample == 1) {
          for (uint32_t x = 0; x < W; ++x) {
            int32_t v = rows[0][x] + off0; if (v < 0) v = 0; else if (v > maxval) v = maxval;
            row_dst[x] = static_cast<uint8_t>(v);
          }
        } else {
          uint8_t* d = row_dst;
          for (uint32_t x = 0; x < W; ++x) {
            int32_t v = rows[0][x] + off0; if (v < 0) v = 0; else if (v > maxval) v = maxval;
            *d++ = static_cast<uint8_t>(v >> 8); *d++ = static_cast<uint8_t>(v & 0xff);
          }
        }
#endif
      } else {
        // Multi-component: interleave components and pack in one pass.
        // For subsampled chroma (e.g. 4:2:2), upsample with nearest-neighbour: rows[c][x>>1].
        if (bytes_per_sample == 1) {
#if defined(OPENHTJ2K_ENABLE_WASM_SIMD)
          // Fast SIMD path for the common case: 3 non-subsampled 8-bit channels.
          // Processes 16 pixels at a time, producing 48 bytes (3 × v128_t stores).
          // Uses a 3-way byte-interleave via 6 wasm_i8x16_shuffle operations.
          if (nc == 3 && !subsamp) {
            const v128_t vmx   = wasm_i32x4_splat(maxval);
            const v128_t vzero = wasm_i32x4_const_splat(0);
            const v128_t voff0 = wasm_i32x4_splat(dc_offset[0]);
            const v128_t voff1 = wasm_i32x4_splat(dc_offset[1]);
            const v128_t voff2 = wasm_i32x4_splat(dc_offset[2]);
            uint32_t x = 0;
            for (; x + 16 <= W; x += 16) {
              // Clamp 16 samples per channel and narrow to bytes.
              auto clamp_narrow16 = [&](const int32_t* src, v128_t voff) -> v128_t {
                v128_t a = wasm_i32x4_min(wasm_i32x4_max(wasm_i32x4_add(wasm_v128_load(src),      voff), vzero), vmx);
                v128_t b = wasm_i32x4_min(wasm_i32x4_max(wasm_i32x4_add(wasm_v128_load(src + 4),  voff), vzero), vmx);
                v128_t c = wasm_i32x4_min(wasm_i32x4_max(wasm_i32x4_add(wasm_v128_load(src + 8),  voff), vzero), vmx);
                v128_t d = wasm_i32x4_min(wasm_i32x4_max(wasm_i32x4_add(wasm_v128_load(src + 12), voff), vzero), vmx);
                return wasm_u8x16_narrow_i16x8(wasm_i16x8_narrow_i32x4(a, b),
                                               wasm_i16x8_narrow_i32x4(c, d));
              };
              v128_t R = clamp_narrow16(rows[0] + x, voff0);
              v128_t G = clamp_narrow16(rows[1] + x, voff1);
              v128_t B = clamp_narrow16(rows[2] + x, voff2);
              // 3-channel byte interleave: R,G,B[0..15] → RGB[0..47] (3 stores).
              // Step 1: interleave R and G into pairs.
              v128_t rg0 = wasm_i8x16_shuffle(R, G, 0,16,1,17,2,18,3,19,4,20,5,21,6,22,7,23);
              v128_t rg1 = wasm_i8x16_shuffle(R, G, 8,24,9,25,10,26,11,27,12,28,13,29,14,30,15,31);
              // Step 2: insert B bytes at every third position across three output vectors.
              // out0 = [R0,G0,B0, R1,G1,B1, R2,G2,B2, R3,G3,B3, R4,G4,B4, R5]  (bytes 0-15)
              v128_t out0 = wasm_i8x16_shuffle(rg0, B, 0,1,16, 2,3,17, 4,5,18, 6,7,19, 8,9,20, 10);
              // mid: bridge rg0[11..15] + rg1[0..10] for out1 construction.
              v128_t mid  = wasm_i8x16_shuffle(rg0, rg1, 11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26);
              // out1 = [G5,B5, R6,G6,B6, R7,G7,B7, R8,G8,B8, R9,G9,B9, R10,G10]  (bytes 16-31)
              v128_t out1 = wasm_i8x16_shuffle(mid, B, 0,21, 1,2,22, 3,4,23, 5,6,24, 7,8,25, 9,10);
              // out2 = [B10, R11,G11,B11, R12,G12,B12, R13,G13,B13, R14,G14,B14, R15,G15,B15]  (bytes 32-47)
              v128_t out2 = wasm_i8x16_shuffle(B, rg1, 10, 22,23,11, 24,25,12, 26,27,13, 28,29,14, 30,31,15);
              wasm_v128_store(row_dst,      out0); row_dst += 16;
              wasm_v128_store(row_dst,      out1); row_dst += 16;
              wasm_v128_store(row_dst,      out2); row_dst += 16;
            }
            for (; x < W; ++x) {
              int32_t rv = rows[0][x] + dc_offset[0]; if (rv < 0) rv = 0; else if (rv > maxval) rv = maxval;
              int32_t gv = rows[1][x] + dc_offset[1]; if (gv < 0) gv = 0; else if (gv > maxval) gv = maxval;
              int32_t bv = rows[2][x] + dc_offset[2]; if (bv < 0) bv = 0; else if (bv > maxval) bv = maxval;
              *row_dst++ = static_cast<uint8_t>(rv);
              *row_dst++ = static_cast<uint8_t>(gv);
              *row_dst++ = static_cast<uint8_t>(bv);
            }
          } else {
#endif
          for (uint32_t x = 0; x < W; ++x) {
            for (uint16_t c = 0; c < nc; ++c) {
              uint32_t sx = (c == 0 || !subsamp) ? x : (x >> 1);
              int32_t v = rows[c][sx] + dc_offset[c]; if (v < 0) v = 0; else if (v > maxval) v = maxval;
              *row_dst++ = static_cast<uint8_t>(v);
            }
          }
#if defined(OPENHTJ2K_ENABLE_WASM_SIMD)
          }
#endif
        } else {
          for (uint32_t x = 0; x < W; ++x) {
            for (uint16_t c = 0; c < nc; ++c) {
              uint32_t sx = (c == 0 || !subsamp) ? x : (x >> 1);
              int32_t v = rows[c][sx] + dc_offset[c]; if (v < 0) v = 0; else if (v > maxval) v = maxval;
              *row_dst++ = static_cast<uint8_t>(v >> 8);
              *row_dst++ = static_cast<uint8_t>(v & 0xff);
            }
          }
        }
      }
    },
    width, height, depth, is_signed);
}

// pack_samples: convert interleaved int32 pixels → packed uint8 or uint16 big-endian.
// src[0..count-1] are clamped to [0, maxval] then written into dst.
// bytes_per_sample: 1 → P5/P6 8-bit, 2 → P5/P6 16-bit big-endian.
// WASM-SIMD paths process 16 (8-bit) or 8 (16-bit) samples per iteration.
EMSCRIPTEN_KEEPALIVE
void pack_samples(const int32_t* __restrict__ src, uint8_t* __restrict__ dst,
                  uint32_t count, int32_t maxval, int32_t bytes_per_sample) {
#if defined(OPENHTJ2K_ENABLE_WASM_SIMD)
  const v128_t vmx   = wasm_i32x4_splat(maxval);
  const v128_t vzero = wasm_i32x4_const_splat(0);
  uint32_t i = 0;
  if (bytes_per_sample == 1) {
    for (; i + 16 <= count; i += 16) {
      v128_t a = wasm_i32x4_min(wasm_i32x4_max(wasm_v128_load(src + i),      vzero), vmx);
      v128_t b = wasm_i32x4_min(wasm_i32x4_max(wasm_v128_load(src + i +  4), vzero), vmx);
      v128_t c = wasm_i32x4_min(wasm_i32x4_max(wasm_v128_load(src + i +  8), vzero), vmx);
      v128_t d = wasm_i32x4_min(wasm_i32x4_max(wasm_v128_load(src + i + 12), vzero), vmx);
      wasm_v128_store(dst + i, wasm_u8x16_narrow_i16x8(wasm_i16x8_narrow_i32x4(a, b),
                                                        wasm_i16x8_narrow_i32x4(c, d)));
    }
    for (; i < count; i++) {
      int32_t v = src[i]; if (v < 0) v = 0; else if (v > maxval) v = maxval;
      dst[i] = static_cast<uint8_t>(v);
    }
  } else {
    uint8_t* d = dst;
    for (; i + 8 <= count; i += 8) {
      v128_t a = wasm_i32x4_min(wasm_i32x4_max(wasm_v128_load(src + i),     vzero), vmx);
      v128_t b = wasm_i32x4_min(wasm_i32x4_max(wasm_v128_load(src + i + 4), vzero), vmx);
      // narrow int32 → uint16 LE, then byte-swap each pair for big-endian output
      v128_t u16le = wasm_u16x8_narrow_i32x4(a, b);
      v128_t u16be = wasm_i8x16_shuffle(u16le, u16le, 1,0, 3,2, 5,4, 7,6, 9,8, 11,10, 13,12, 15,14);
      wasm_v128_store(d, u16be);
      d += 16;
    }
    for (; i < count; i++) {
      int32_t v = src[i]; if (v < 0) v = 0; else if (v > maxval) v = maxval;
      *d++ = static_cast<uint8_t>(v >> 8); *d++ = static_cast<uint8_t>(v & 0xff);
    }
  }
#else
  if (bytes_per_sample == 1) {
    for (uint32_t i = 0; i < count; i++) {
      int32_t v = src[i]; if (v < 0) v = 0; else if (v > maxval) v = maxval;
      dst[i] = static_cast<uint8_t>(v);
    }
  } else {
    uint8_t* d = dst;
    for (uint32_t i = 0; i < count; i++) {
      int32_t v = src[i]; if (v < 0) v = 0; else if (v > maxval) v = maxval;
      *d++ = static_cast<uint8_t>(v >> 8); *d++ = static_cast<uint8_t>(v & 0xff);
    }
  }
#endif
}

// ── In-place YCbCr→RGB conversion ──────────────────────────────────────────
// Input RGBA: R=Y, G=Cb, B=Cr, A=255 (as produced by invoke_decoder_to_rgba
// when MCT=0 / colour transform disabled).  Converts in-place to sRGB.
// n_pixels = width * height.

// Shared helper: clamp an int32 to [0, 255] and return as uint8.
static inline uint8_t clamp_u8(int32_t v) {
  return static_cast<uint8_t>(v < 0 ? 0 : v > 255 ? 255 : v);
}

// BT.601 full-range (JFIF / JPEG convention, typical for photos):
//   R = Y + 1.402*(Cr-128)
//   G = Y - 0.344136*(Cb-128) - 0.714136*(Cr-128)
//   B = Y + 1.772*(Cb-128)
EMSCRIPTEN_KEEPALIVE
void apply_ycbcr_bt601_to_rgba(uint8_t* rgba, uint32_t n_pixels) {
#if defined(OPENHTJ2K_ENABLE_WASM_SIMD)
  // Fixed-point coefficients scaled by 16384 (>>14), bias +8192 for rounding.
  const v128_t v22970 = wasm_i32x4_const_splat(22970);
  const v128_t v5638  = wasm_i32x4_const_splat(5638);
  const v128_t v11698 = wasm_i32x4_const_splat(11698);
  const v128_t v29032 = wasm_i32x4_const_splat(29032);
  const v128_t vrnd   = wasm_i32x4_const_splat(8192);
  const v128_t v128   = wasm_i32x4_const_splat(128);
  const v128_t v255   = wasm_i32x4_const_splat(255);
  const v128_t vzero  = wasm_i32x4_const_splat(0);
  const v128_t vmask  = wasm_i32x4_const_splat(0xFF);
  uint32_t i = 0;
  for (; i + 4 <= n_pixels; i += 4) {
    // Load 4 RGBA pixels (16 bytes) and extract channels as i32x4.
    v128_t pix = wasm_v128_load(rgba + i * 4);
    v128_t vY  = wasm_v128_and(pix, vmask);                                    // R (= Y)
    v128_t vCb = wasm_i32x4_sub(wasm_v128_and(wasm_u32x4_shr(pix, 8),  vmask), v128);  // G - 128
    v128_t vCr = wasm_i32x4_sub(wasm_v128_and(wasm_u32x4_shr(pix, 16), vmask), v128);  // B - 128
    v128_t vA  = wasm_u32x4_shr(pix, 24);                                      // alpha unchanged
    v128_t newR = wasm_i32x4_add(vY, wasm_i32x4_shr(
        wasm_i32x4_add(wasm_i32x4_mul(v22970, vCr), vrnd), 14));
    v128_t newG = wasm_i32x4_sub(vY, wasm_i32x4_shr(
        wasm_i32x4_add(wasm_i32x4_add(wasm_i32x4_mul(v5638,  vCb),
                                      wasm_i32x4_mul(v11698, vCr)), vrnd), 14));
    v128_t newB = wasm_i32x4_add(vY, wasm_i32x4_shr(
        wasm_i32x4_add(wasm_i32x4_mul(v29032, vCb), vrnd), 14));
    newR = wasm_i32x4_min(wasm_i32x4_max(newR, vzero), v255);
    newG = wasm_i32x4_min(wasm_i32x4_max(newG, vzero), v255);
    newB = wasm_i32x4_min(wasm_i32x4_max(newB, vzero), v255);
    // Narrow i32x4 → i16x8 → u8x16, then transpose planar→interleaved.
    // rg = [R0,R1,R2,R3,G0,G1,G2,G3], ba = [B0,B1,B2,B3,A0,A1,A2,A3]
    v128_t packed = wasm_u8x16_narrow_i16x8(wasm_i16x8_narrow_i32x4(newR, newG),
                                            wasm_i16x8_narrow_i32x4(newB, vA));
    // packed = [R0..R3, G0..G3, B0..B3, A0..A3]; shuffle to RGBA pixels.
    wasm_v128_store(rgba + i * 4,
        wasm_i8x16_shuffle(packed, packed, 0,4,8,12, 1,5,9,13, 2,6,10,14, 3,7,11,15));
  }
  for (; i < n_pixels; ++i) {
    uint8_t* p  = rgba + i * 4;
    int32_t Y   = p[0];
    int32_t Cb  = static_cast<int32_t>(p[1]) - 128;
    int32_t Cr  = static_cast<int32_t>(p[2]) - 128;
    p[0] = clamp_u8(Y + ((22970 * Cr + 8192) >> 14));
    p[1] = clamp_u8(Y - ((5638 * Cb + 11698 * Cr + 8192) >> 14));
    p[2] = clamp_u8(Y + ((29032 * Cb + 8192) >> 14));
  }
#else
  for (uint32_t i = 0; i < n_pixels; ++i) {
    uint8_t* p  = rgba + i * 4;
    int32_t Y   = p[0];
    int32_t Cb  = static_cast<int32_t>(p[1]) - 128;
    int32_t Cr  = static_cast<int32_t>(p[2]) - 128;
    // Fixed-point coefficients scaled by 16384 (>>14), bias +8192 for rounding.
    p[0] = clamp_u8(Y + ((22970 * Cr + 8192) >> 14));
    p[1] = clamp_u8(Y - ((5638 * Cb + 11698 * Cr + 8192) >> 14));
    p[2] = clamp_u8(Y + ((29032 * Cb + 8192) >> 14));
    // p[3] (alpha) unchanged
  }
#endif
}

// BT.709 full-range (HDTV / H.264 / H.265 convention):
//   R = Y + 1.5748*(Cr-128)
//   G = Y - 0.187324*(Cb-128) - 0.468124*(Cr-128)
//   B = Y + 1.8556*(Cb-128)
EMSCRIPTEN_KEEPALIVE
void apply_ycbcr_bt709_to_rgba(uint8_t* rgba, uint32_t n_pixels) {
#if defined(OPENHTJ2K_ENABLE_WASM_SIMD)
  const v128_t v25801 = wasm_i32x4_const_splat(25801);
  const v128_t v3069  = wasm_i32x4_const_splat(3069);
  const v128_t v7672  = wasm_i32x4_const_splat(7672);
  const v128_t v30397 = wasm_i32x4_const_splat(30397);
  const v128_t vrnd   = wasm_i32x4_const_splat(8192);
  const v128_t v128   = wasm_i32x4_const_splat(128);
  const v128_t v255   = wasm_i32x4_const_splat(255);
  const v128_t vzero  = wasm_i32x4_const_splat(0);
  const v128_t vmask  = wasm_i32x4_const_splat(0xFF);
  uint32_t i = 0;
  for (; i + 4 <= n_pixels; i += 4) {
    v128_t pix = wasm_v128_load(rgba + i * 4);
    v128_t vY  = wasm_v128_and(pix, vmask);
    v128_t vCb = wasm_i32x4_sub(wasm_v128_and(wasm_u32x4_shr(pix, 8),  vmask), v128);
    v128_t vCr = wasm_i32x4_sub(wasm_v128_and(wasm_u32x4_shr(pix, 16), vmask), v128);
    v128_t vA  = wasm_u32x4_shr(pix, 24);
    v128_t newR = wasm_i32x4_add(vY, wasm_i32x4_shr(
        wasm_i32x4_add(wasm_i32x4_mul(v25801, vCr), vrnd), 14));
    v128_t newG = wasm_i32x4_sub(vY, wasm_i32x4_shr(
        wasm_i32x4_add(wasm_i32x4_add(wasm_i32x4_mul(v3069, vCb),
                                      wasm_i32x4_mul(v7672, vCr)), vrnd), 14));
    v128_t newB = wasm_i32x4_add(vY, wasm_i32x4_shr(
        wasm_i32x4_add(wasm_i32x4_mul(v30397, vCb), vrnd), 14));
    newR = wasm_i32x4_min(wasm_i32x4_max(newR, vzero), v255);
    newG = wasm_i32x4_min(wasm_i32x4_max(newG, vzero), v255);
    newB = wasm_i32x4_min(wasm_i32x4_max(newB, vzero), v255);
    v128_t packed = wasm_u8x16_narrow_i16x8(wasm_i16x8_narrow_i32x4(newR, newG),
                                            wasm_i16x8_narrow_i32x4(newB, vA));
    wasm_v128_store(rgba + i * 4,
        wasm_i8x16_shuffle(packed, packed, 0,4,8,12, 1,5,9,13, 2,6,10,14, 3,7,11,15));
  }
  for (; i < n_pixels; ++i) {
    uint8_t* p  = rgba + i * 4;
    int32_t Y   = p[0];
    int32_t Cb  = static_cast<int32_t>(p[1]) - 128;
    int32_t Cr  = static_cast<int32_t>(p[2]) - 128;
    p[0] = clamp_u8(Y + ((25801 * Cr + 8192) >> 14));
    p[1] = clamp_u8(Y - ((3069 * Cb + 7672 * Cr + 8192) >> 14));
    p[2] = clamp_u8(Y + ((30397 * Cb + 8192) >> 14));
  }
#else
  for (uint32_t i = 0; i < n_pixels; ++i) {
    uint8_t* p  = rgba + i * 4;
    int32_t Y   = p[0];
    int32_t Cb  = static_cast<int32_t>(p[1]) - 128;
    int32_t Cr  = static_cast<int32_t>(p[2]) - 128;
    p[0] = clamp_u8(Y + ((25801 * Cr + 8192) >> 14));
    p[1] = clamp_u8(Y - ((3069 * Cb + 7672 * Cr + 8192) >> 14));
    p[2] = clamp_u8(Y + ((30397 * Cb + 8192) >> 14));
  }
#endif
}
}

#endif