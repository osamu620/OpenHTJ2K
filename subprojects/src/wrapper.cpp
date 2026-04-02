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
}

#endif