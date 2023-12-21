#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include <exception>
#ifdef __EMSCRIPTEN__
  #include <emscripten.h>
#endif

#ifndef __EMSCRIPTEN__
  #define EMSCRIPTEN_KEEPALIVE [[maybe_unused]]
#endif

#include "decoder.hpp"

open_htj2k::openhtj2k_decoder* cpp_create_decoder(uint8_t* data, size_t size, uint8_t reduce_NL) {
  return new open_htj2k::openhtj2k_decoder(data, size, reduce_NL, 1);
}

void cpp_parse_j2c_data(open_htj2k::openhtj2k_decoder* dec) { dec->parse(); }
void cpp_invoke_decoder(open_htj2k::openhtj2k_decoder* dec, int32_t* out) {
  const uint16_t num_components = dec->get_num_component();
  std::vector<int32_t*> buf;
  std::vector<uint32_t> img_width;
  std::vector<uint32_t> img_height;
  std::vector<uint8_t> img_depth;
  std::vector<bool> img_signed;
  dec->invoke(buf, img_width, img_height, img_depth, img_signed);
  for (uint32_t y = 0; y < img_height[0]; ++y) {
    for (uint32_t x = 0; x < img_width[0]; ++x) {
      for (uint16_t c = 0; c < num_components; ++c) {
        out[y * img_width[0] * num_components + x * num_components + c] = buf[c][y * img_width[0] + x];
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
void show(open_htj2k::openhtj2k_decoder* dec) { dec->show(); }
}