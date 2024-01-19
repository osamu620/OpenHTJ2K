// Copyright (c) 2019 - 2021, Osamu Watanabe
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
#include <memory>
#include <vector>
#if defined(_MSC_VER) && !defined(OHTJ2K_STATIC)
  #define OPENHTJ2K_EXPORT __declspec(dllexport)
#else
  #define OPENHTJ2K_EXPORT
#endif
namespace open_htj2k {
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
  OPENHTJ2K_EXPORT uint8_t get_minumum_DWT_levels();
  OPENHTJ2K_EXPORT void invoke(std::vector<int32_t *> &, std::vector<uint32_t> &, std::vector<uint32_t> &,
                               std::vector<uint8_t> &, std::vector<bool> &);
  OPENHTJ2K_EXPORT void destroy();
  OPENHTJ2K_EXPORT ~openhtj2k_decoder();
};
}  // namespace open_htj2k
