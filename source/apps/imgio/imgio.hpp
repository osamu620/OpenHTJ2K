// Copyright (c) 2019 - 2026, Osamu Watanabe
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
// THE POSSIBILITY OF SUCH DAMAGE.

#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace imgio {

// Row-at-a-time reader interface used by the encoder's line-based path.
struct StreamReader {
  virtual ~StreamReader()                                            = default;
  virtual uint32_t get_width() const                                 = 0;
  virtual uint32_t get_height() const                                = 0;
  virtual uint16_t get_num_components() const                        = 0;
  virtual uint8_t  get_bitdepth(uint16_t c = 0) const                = 0;
  // SIZ Ssiz byte for component c (bit-depth minus 1; bit 7 set if signed).
  virtual uint8_t  get_Ssiz(uint16_t c) const                        = 0;
  // SIZ XRsiz/YRsiz for component c (1 = no subsampling).
  virtual uint8_t  get_XRsiz(uint16_t /*c*/) const { return 1; }
  virtual uint8_t  get_YRsiz(uint16_t /*c*/) const { return 1; }
  virtual void     get_row(uint32_t y, int32_t **rows, uint16_t nc)  = 0;
};

// Opens a stream reader chosen by the first file's extension:
//   .pgx          → PGX (one file per component; uses all of fnames)
//   .tif / .tiff  → TIFF (uses fnames[0]; requires OPENHTJ2K_TIFF_SUPPORT)
//   otherwise     → PNM (P5/P6, uses fnames[0])
// On failure prints a diagnostic to stdout and returns nullptr.
std::unique_ptr<StreamReader> open_stream_reader(const std::vector<std::string> &fnames);

}  // namespace imgio
