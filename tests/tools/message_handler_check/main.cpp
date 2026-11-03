// Copyright (c) 2026, Osamu Watanabe
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

// Checks how the library reports problems:
//   * it writes nothing to stdout or stderr by itself ("silent" mode, run by CTest with a
//     FAIL_REGULAR_EXPRESSION that matches any output);
//   * its warnings reach a handler installed with open_htj2k::set_message_handler() ("handler" mode);
//   * its errors arrive as exceptions whose what() describes the problem (both modes).
//
// Usage: message_handler_check silent|handler <malformed codestream>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <iterator>
#include <mutex>
#include <string>
#include <vector>

#include "decoder.hpp"
#include "encoder.hpp"

namespace {
const uint32_t kWidth  = 37;
const uint32_t kHeight = 23;

std::mutex messages_mutex;
int warning_count = 0;
std::string last_warning;

void count_message(open_htj2k::message_level level, const char *text, void *) {
  std::lock_guard<std::mutex> lock(messages_mutex);
  if (level == open_htj2k::message_level::warning) {
    ++warning_count;
    last_warning = text;
  }
}

// Encodes a small one-component image with the colour transform requested, which the encoder
// turns off with a warning. Returns false if the encode fails.
bool encode_with_warning() {
  open_htj2k::siz_params siz;
  siz.Rsiz   = 0;
  siz.Xsiz   = kWidth;
  siz.Ysiz   = kHeight;
  siz.XOsiz  = 0;
  siz.YOsiz  = 0;
  siz.XTsiz  = kWidth;
  siz.YTsiz  = kHeight;
  siz.XTOsiz = 0;
  siz.YTOsiz = 0;
  siz.Csiz   = 1;
  siz.Ssiz.push_back(7);
  siz.XRsiz.push_back(1);
  siz.YRsiz.push_back(1);
  open_htj2k::cod_params cod;
  cod.blkwidth          = 4;
  cod.blkheight         = 4;
  cod.is_max_precincts  = true;
  cod.use_SOP           = false;
  cod.use_EPH           = false;
  cod.progression_order = 0;
  cod.number_of_layers  = 1;
  cod.use_color_trafo   = 1;  // not possible with one component: the encoder warns
  cod.dwt_levels        = 3;
  cod.codeblock_style   = 0x040;
  cod.transformation    = 1;
  open_htj2k::qcd_params qcd{};
  qcd.is_derived          = false;
  qcd.number_of_guardbits = 1;
  qcd.base_step           = 1.0 / 256.0;
  std::vector<int32_t *> no_input;
  std::vector<uint8_t> out;
  try {
    open_htj2k::openhtj2k_encoder enc("", no_input, siz, cod, qcd, 0xFF, false, 0, 1);
    enc.set_output_buffer(out);
    enc.invoke_line_based_stream([](uint32_t y, int32_t **rows, uint16_t) {
      for (uint32_t x = 0; x < kWidth; ++x) rows[0][x] = static_cast<int32_t>((x * 7 + y * 3) & 0xFF);
    });
  } catch (std::exception &) {
    return false;
  }
  return !out.empty();
}

// Decodes a malformed codestream from memory. Returns the exception's what(), or an empty string
// if no exception was thrown.
std::string decode_malformed(const std::vector<uint8_t> &data) {
  try {
    open_htj2k::openhtj2k_decoder dec;
    dec.init(data.data(), data.size(), 0, 1);
    dec.parse();
    std::vector<uint32_t> w, h;
    std::vector<uint8_t> depth;
    std::vector<bool> is_signed;
    dec.invoke_line_based_stream([](uint32_t, int32_t *const *, uint16_t) {}, w, h, depth, is_signed);
  } catch (std::exception &exc) {
    const std::string text = exc.what();
    return text.empty() ? std::string("(empty what())") : text;
  }
  return std::string();
}
}  // namespace

int main(int argc, char *argv[]) {
  if (argc != 3 || (std::strcmp(argv[1], "silent") != 0 && std::strcmp(argv[1], "handler") != 0)) {
    std::printf("usage: %s silent|handler <malformed codestream>\n", argv[0]);
    return 2;
  }
  const bool with_handler = std::strcmp(argv[1], "handler") == 0;
  std::ifstream file(argv[2], std::ios::binary);
  const std::vector<uint8_t> data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  if (data.empty()) {
    std::printf("cannot read %s\n", argv[2]);
    return 2;
  }
  if (with_handler) {
    open_htj2k::set_message_handler(count_message, nullptr);
  }
  const bool encoded       = encode_with_warning();
  const std::string error  = decode_malformed(data);
  const bool generic_error = error == "std::exception" || error == "Unknown exception";
  if (!encoded || error.empty() || generic_error) {
    std::printf("FAIL: encoded=%d error=\"%s\"\n", encoded ? 1 : 0, error.c_str());
    return 1;
  }
  if (with_handler) {
    open_htj2k::set_message_handler(nullptr, nullptr);
    std::printf("warnings=%d last=\"%s\" error=\"%s\"\n", warning_count, last_warning.c_str(),
                error.c_str());
    return warning_count > 0 ? 0 : 1;
  }
  return 0;  // silent mode: no output at all
}
