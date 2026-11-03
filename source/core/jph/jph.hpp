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
#include <cstdio>
#include <cstring>
#include <vector>
#include "j2kmarkers.hpp"
#include "codestream.hpp"
#include "open_htj2k_typedef.hpp"

namespace open_htj2k {

// ---------------------------------------------------------------------------
// JPH / JP2 box reader (decoder side)
// ---------------------------------------------------------------------------
// EnumCS values defined in ISO/IEC 15444-1 Annex I.
static constexpr uint32_t OPENHTJ2K_ENUMCS_SRGB      = 16u;
static constexpr uint32_t OPENHTJ2K_ENUMCS_GRAYSCALE = 17u;
static constexpr uint32_t OPENHTJ2K_ENUMCS_YCBCR     = 18u;

// Information extracted from a JPH/JP2 file.
struct jph_info {
  const uint8_t *cs_data = nullptr;  // pointer into caller-owned buffer at J2K codestream start
  size_t         cs_size = 0;        // codestream length in bytes
  uint32_t       enum_cs = 0;        // EnumCS from colr box (0 = not found / raw codestream)
};

// Parse a JPH/JP2 buffer.  Returns true if a valid signature and codestream box
// were found; fills `out`.  Does NOT copy data — `buf` must outlive `out`.
// Returns false without the signature (a raw codestream); a buffer with the
// signature but without a valid codestream box is an error (throws).
bool jph_parse_buffer(const uint8_t *buf, size_t len, jph_info &out);

// Convenience: returns true when the first 12 bytes match the JP2/JPH signature.
bool jph_is_signature(const uint8_t *buf, size_t len);
// ---------------------------------------------------------------------------

class box_base {
 public:
  uint32_t LBox;
  uint32_t TBox;
  uint64_t XLBox;

 public:
  box_base(uint32_t l, uint32_t t) : LBox(l), TBox(t), XLBox(0) {}
  virtual size_t write(j2c_dst_memory &dst) = 0;
  void base_write(j2c_dst_memory &dst) const {
    dst.put_dword(LBox);
    dst.put_dword(TBox);
  }
};

class signature_box : public box_base {
 private:
  uint32_t signature;

 public:
  signature_box();
  size_t write(j2c_dst_memory &dst) override;
};

class file_type_box : public box_base {
 private:
  uint32_t BR;
  const uint32_t MinV;
  std::vector<uint32_t> CLi;

 public:
  file_type_box(uint8_t type);
  size_t write(j2c_dst_memory &dst) override;
};

class image_header_box : public box_base {
 private:
  uint32_t HEIGHT;
  uint32_t WIDTH;
  uint16_t NC;
  uint8_t BPC;
  const uint8_t C;
  uint8_t UnkC;
  uint8_t IPR;

 public:
  image_header_box(j2k_main_header &hdr);
  bool needBPCC();
  size_t write(j2c_dst_memory &dst) override;
};

class bits_per_component_box : public box_base {
 private:
  std::vector<uint8_t> BPC;

 public:
  bits_per_component_box(j2k_main_header &hdr);
  size_t write(j2c_dst_memory &dst) override;
};

class colour_specification_box : public box_base {
 private:
  uint8_t METH;
  uint8_t PREC;
  uint8_t APPROX;
  uint32_t EnumCS;
  OPENHTJ2K_MAYBE_UNUSED uint8_t PROFILE;
  OPENHTJ2K_MAYBE_UNUSED uint16_t COLPRIMS;
  OPENHTJ2K_MAYBE_UNUSED uint16_t TRANSFC;
  OPENHTJ2K_MAYBE_UNUSED uint16_t MATCOEFFS;
  OPENHTJ2K_MAYBE_UNUSED bool VIDFRNG;
  OPENHTJ2K_MAYBE_UNUSED uint8_t VIDFRNG_RSVD;

 public:
  colour_specification_box(j2k_main_header &hdr, bool isSRGB);
  size_t write(j2c_dst_memory &dst) override;
};

// Channel Definition box (ISO/IEC 15444-1 I.5.3.6).  Written for 2 and 4
// components, which are gray + alpha and colour + alpha: the last component is
// declared as opacity for the whole image, the others as the colours of the
// colour specification box in order.
class channel_definition_box : public box_base {
 private:
  std::vector<uint16_t> Cn;
  std::vector<uint16_t> Typ;
  std::vector<uint16_t> Asoc;

 public:
  explicit channel_definition_box(j2k_main_header &hdr);
  bool is_needed() const { return !Cn.empty(); }
  size_t write(j2c_dst_memory &dst) override;
};

class header_box : public box_base {
 private:
  image_header_box ihdr;
  bits_per_component_box bpcc;
  colour_specification_box colr;
  channel_definition_box cdef;

 public:
  header_box(j2k_main_header &hdr, bool isSRGB);
  size_t write(j2c_dst_memory &dst) override;
};

// The header of a Contiguous Codestream box that holds code_len bytes (ISO/IEC 15444-1 I.4):
// LBox, the length of the box including its header, and TBox 'jp2c'.  When that length does
// not fit in the 32 bits of LBox, LBox is 1 and the 64-bit XLBox after TBox gives it, which
// makes the header 16 bytes.  Writes the header to hdr and returns its length, 8 or 16.
inline size_t jp2c_box_header(uint64_t code_len, uint8_t (&hdr)[16]) {
  const bool extended = code_len > 0xFFFFFFFFu - 8u;
  const uint64_t lbox = extended ? 1u : code_len + 8u;
  const uint32_t tbox = 0x6A703263u;  // 'jp2c'
  for (int i = 0; i < 4; ++i) {
    hdr[i]     = static_cast<uint8_t>(lbox >> (24 - 8 * i));
    hdr[4 + i] = static_cast<uint8_t>(tbox >> (24 - 8 * i));
  }
  if (!extended) return 8;
  const uint64_t xlbox = code_len + 16u;
  for (int i = 0; i < 8; ++i) hdr[8 + i] = static_cast<uint8_t>(xlbox >> (56 - 8 * i));
  return 16;
}

class contiguous_codestream_box : public box_base {
 private:
  uint64_t code_len;

 public:
  // LBox and XLBox are given by jp2c_box_header() when the header is written.
  explicit contiguous_codestream_box(size_t len);
  // Writes the box header; returns the length of the box, codestream included.
  size_t write(j2c_dst_memory &dst) override;
};

// The boxes of a JPH or JP2 file up to the Contiguous Codestream box header; the
// codestream (code_len bytes) follows them.  type is the brand of the file type
// box: 0 for JP2 (ISO/IEC 15444-1 Annex I), 1 for JPH (ISO/IEC 15444-15).
class jph_boxes {
 private:
  signature_box sig;
  file_type_box ftyp;
  header_box jp2h;
  contiguous_codestream_box jp2c;

 public:
  jph_boxes(j2k_main_header &hdr, uint8_t type, bool isSRGB, size_t code_len);
  // Writes the boxes; returns the length of the file, codestream included.
  size_t write(j2c_dst_memory &dst);
};
}  // namespace open_htj2k