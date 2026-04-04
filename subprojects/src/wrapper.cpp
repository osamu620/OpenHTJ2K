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
                           int32_t maxval, int32_t bytes_per_sample) {
  std::vector<uint32_t> width, height;
  std::vector<uint8_t>  depth;
  std::vector<bool>     is_signed;

  dec->invoke_line_based_stream(
    [&](uint32_t y, int32_t* const* rows, uint16_t nc) {
      const uint32_t W       = width[0];
      // Detect chroma subsampling (e.g. 4:2:2): chroma rows are narrower than luma.
      const uint32_t chromaW = (nc >= 2) ? width[1] : W;
      const bool     subsamp = (chromaW < W);
      uint8_t* row_dst = dst + (size_t)y * W * nc * bytes_per_sample;
      if (nc == 1) {
        // Grayscale — single component, already "interleaved"
#if defined(OPENHTJ2K_ENABLE_WASM_SIMD)
        const v128_t vmx   = wasm_i32x4_splat(maxval);
        const v128_t vzero = wasm_i32x4_const_splat(0);
        uint32_t x = 0;
        if (bytes_per_sample == 1) {
          for (; x + 16 <= W; x += 16) {
            v128_t a = wasm_i32x4_min(wasm_i32x4_max(wasm_v128_load(rows[0] + x),      vzero), vmx);
            v128_t b = wasm_i32x4_min(wasm_i32x4_max(wasm_v128_load(rows[0] + x +  4), vzero), vmx);
            v128_t c = wasm_i32x4_min(wasm_i32x4_max(wasm_v128_load(rows[0] + x +  8), vzero), vmx);
            v128_t d = wasm_i32x4_min(wasm_i32x4_max(wasm_v128_load(rows[0] + x + 12), vzero), vmx);
            wasm_v128_store(row_dst + x,
                            wasm_u8x16_narrow_i16x8(wasm_i16x8_narrow_i32x4(a, b),
                                                    wasm_i16x8_narrow_i32x4(c, d)));
          }
          for (; x < W; ++x) {
            int32_t v = rows[0][x]; if (v < 0) v = 0; else if (v > maxval) v = maxval;
            row_dst[x] = static_cast<uint8_t>(v);
          }
        } else {
          uint8_t* d = row_dst;
          for (; x + 8 <= W; x += 8) {
            v128_t a = wasm_i32x4_min(wasm_i32x4_max(wasm_v128_load(rows[0] + x),     vzero), vmx);
            v128_t b = wasm_i32x4_min(wasm_i32x4_max(wasm_v128_load(rows[0] + x + 4), vzero), vmx);
            v128_t u16le = wasm_u16x8_narrow_i32x4(a, b);
            v128_t u16be = wasm_i8x16_shuffle(u16le, u16le,
                                              1,0, 3,2, 5,4, 7,6, 9,8, 11,10, 13,12, 15,14);
            wasm_v128_store(d, u16be); d += 16;
          }
          for (; x < W; ++x) {
            int32_t v = rows[0][x]; if (v < 0) v = 0; else if (v > maxval) v = maxval;
            *d++ = static_cast<uint8_t>(v >> 8); *d++ = static_cast<uint8_t>(v & 0xff);
          }
        }
#else
        if (bytes_per_sample == 1) {
          for (uint32_t x = 0; x < W; ++x) {
            int32_t v = rows[0][x]; if (v < 0) v = 0; else if (v > maxval) v = maxval;
            row_dst[x] = static_cast<uint8_t>(v);
          }
        } else {
          uint8_t* d = row_dst;
          for (uint32_t x = 0; x < W; ++x) {
            int32_t v = rows[0][x]; if (v < 0) v = 0; else if (v > maxval) v = maxval;
            *d++ = static_cast<uint8_t>(v >> 8); *d++ = static_cast<uint8_t>(v & 0xff);
          }
        }
#endif
      } else {
        // Multi-component: interleave components and pack in one pass.
        // For subsampled chroma (e.g. 4:2:2), upsample with nearest-neighbour: rows[c][x>>1].
        if (bytes_per_sample == 1) {
          for (uint32_t x = 0; x < W; ++x) {
            for (uint16_t c = 0; c < nc; ++c) {
              uint32_t sx = (c == 0 || !subsamp) ? x : (x >> 1);
              int32_t v = rows[c][sx]; if (v < 0) v = 0; else if (v > maxval) v = maxval;
              *row_dst++ = static_cast<uint8_t>(v);
            }
          }
        } else {
          for (uint32_t x = 0; x < W; ++x) {
            for (uint16_t c = 0; c < nc; ++c) {
              uint32_t sx = (c == 0 || !subsamp) ? x : (x >> 1);
              int32_t v = rows[c][sx]; if (v < 0) v = 0; else if (v > maxval) v = maxval;
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
}

// BT.709 full-range (HDTV / H.264 / H.265 convention):
//   R = Y + 1.5748*(Cr-128)
//   G = Y - 0.187324*(Cb-128) - 0.468124*(Cr-128)
//   B = Y + 1.8556*(Cb-128)
EMSCRIPTEN_KEEPALIVE
void apply_ycbcr_bt709_to_rgba(uint8_t* rgba, uint32_t n_pixels) {
  for (uint32_t i = 0; i < n_pixels; ++i) {
    uint8_t* p  = rgba + i * 4;
    int32_t Y   = p[0];
    int32_t Cb  = static_cast<int32_t>(p[1]) - 128;
    int32_t Cr  = static_cast<int32_t>(p[2]) - 128;
    p[0] = clamp_u8(Y + ((25801 * Cr + 8192) >> 14));
    p[1] = clamp_u8(Y - ((3069 * Cb + 7672 * Cr + 8192) >> 14));
    p[2] = clamp_u8(Y + ((30397 * Cb + 8192) >> 14));
  }
}
}

#endif