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

#include "jph.hpp"
#include "message_internal.hpp"

namespace open_htj2k {

// ---------------------------------------------------------------------------
// JPH / JP2 box reader implementation
// ---------------------------------------------------------------------------
static inline uint32_t jph_read_be32(const uint8_t *p) {
  return (static_cast<uint32_t>(p[0]) << 24) | (static_cast<uint32_t>(p[1]) << 16)
         | (static_cast<uint32_t>(p[2]) << 8) | static_cast<uint32_t>(p[3]);
}
static inline uint64_t jph_read_be64(const uint8_t *p) {
  return (static_cast<uint64_t>(jph_read_be32(p)) << 32)
         | static_cast<uint64_t>(jph_read_be32(p + 4));
}

static constexpr uint32_t BOX_JP2H = 0x6A703268u;  // "jp2h"
static constexpr uint32_t BOX_COLR = 0x636F6C72u;  // "colr"
static constexpr uint32_t BOX_JP2C = 0x6A703263u;  // "jp2c"

// JP2/JPH superbox nesting is shallow in practice; cap recursion so a crafted
// chain of zero-length jp2h superboxes cannot exhaust the stack.
static constexpr int JPH_MAX_BOX_DEPTH = 16;
static void jph_scan_boxes(const uint8_t *begin, const uint8_t *end, jph_info &out, int depth = 0) {
  if (depth > JPH_MAX_BOX_DEPTH) return;
  const uint8_t *p = begin;
  // Lengths are compared with the remaining byte count in the integer domain: a pointer
  // past `end` must not be formed, and `p + xlen` overflows for a large crafted xlen.
  while (end - p >= 8) {
    const uint64_t remaining = static_cast<uint64_t>(end - p);
    uint32_t lbox            = jph_read_be32(p);
    uint32_t tbox            = jph_read_be32(p + 4);
    const uint8_t *payload, *box_end;
    if (lbox == 1) {
      // XLBox: the length of the box, its 16-byte header included (ISO/IEC 15444-1 I.4).
      if (remaining < 16) return;
      uint64_t xlen = jph_read_be64(p + 8);
      if (xlen < 16 || xlen > remaining) return;
      payload = p + 16;
      box_end = p + static_cast<size_t>(xlen);
    } else if (lbox == 0) {
      // The box extends to the end of the file (or of the enclosing superbox).
      payload = p + 8;
      box_end = end;
    } else {
      // Clamp to file end: handles jp2c boxes whose declared lbox exceeds
      // the actual file size (e.g. some streaming writers pre-declare a
      // larger lbox; the codestream ends at EOF via its own EOC marker).
      if (lbox < 8) return;
      payload  = p + 8;
      box_end  = (lbox <= remaining) ? (p + lbox) : end;
    }
    if (tbox == BOX_JP2H) {
      jph_scan_boxes(payload, box_end, out, depth + 1);
    } else if (tbox == BOX_COLR) {
      // METH(1) PREC(1) APPROX(1) EnumCS(4) — only for Enumerated Colourspace (METH==1)
      if (box_end - payload >= 7 && payload[0] == 1) out.enum_cs = jph_read_be32(payload + 3);
    } else if (tbox == BOX_JP2C) {
      out.cs_data = payload;
      out.cs_size = static_cast<size_t>(box_end - payload);
    }
    p = box_end;
  }
}

bool jph_is_signature(const uint8_t *buf, size_t len) {
  // Signature box: LBox=12, TBox="jP  " (0x6A502020), payload=0x0D0A870A
  return len >= 12 && jph_read_be32(buf) == 12 && jph_read_be32(buf + 4) == 0x6A502020u
         && jph_read_be32(buf + 8) == 0x0D0A870Au;
}

bool jph_parse_buffer(const uint8_t *buf, size_t len, jph_info &out) {
  if (!jph_is_signature(buf, len)) return false;
  jph_scan_boxes(buf, buf + len, out);
  if (out.cs_data == nullptr || out.cs_size == 0) {
    // Not a raw codestream either: it starts with the signature box.
    open_htj2k::detail::fail("JPH/JP2 file without a valid Contiguous Codestream box.");
  }
  return true;
}
// ---------------------------------------------------------------------------

signature_box::signature_box() : box_base(12, 0x6A502020), signature(0x0D0A870A) {}

size_t signature_box::write(j2c_dst_memory &dst) {
  base_write(dst);
  dst.put_dword(signature);
  return LBox;
}

file_type_box::file_type_box(uint8_t type) : box_base(16, 0x66747970), MinV(0) {
  if (type == 0) {
    // jp2
    BR = 0x6A703220;
    CLi.push_back(0x6A703220);
  } else if (type == 1) {
    // jph
    BR = 0x6A706820;
    CLi.push_back(0x6A706820);
  } else {
    open_htj2k::detail::fail("unsupported type for file_type_box");
  }
  for (size_t i = 0; i < CLi.size(); ++i) {
    LBox += 4;
  }
}

size_t file_type_box::write(j2c_dst_memory &dst) {
  base_write(dst);
  dst.put_dword(BR);
  dst.put_dword(MinV);
  for (auto &c : CLi) {
    dst.put_dword(c);
  }
  return LBox;
}

image_header_box::image_header_box(j2k_main_header &hdr) : box_base(22, 0x69686472), C(7), UnkC(0), IPR(0) {
  element_siz siz, Osiz;
  hdr.SIZ->get_image_size(siz);
  hdr.SIZ->get_image_origin(Osiz);
  HEIGHT = siz.y - Osiz.y;
  WIDTH  = siz.x - Osiz.x;
  NC     = hdr.SIZ->get_num_components();
  // Bit depth minus 1, with the sign in bit 7; 255 when the components differ in either,
  // which a Bits Per Component box then gives per component.
  BPC = static_cast<uint8_t>((hdr.SIZ->get_bitdepth(0) - 1) | (hdr.SIZ->is_signed(0) ? 0x80 : 0));
  for (uint16_t c = 1; c < NC; ++c) {
    if (BPC != static_cast<uint8_t>((hdr.SIZ->get_bitdepth(c) - 1) | (hdr.SIZ->is_signed(c) ? 0x80 : 0))) {
      BPC = 0xFF;
      break;
    }
  }
}
bool image_header_box::needBPCC() {
  bool val = false;
  if (BPC == 0xFF) {
    val = true;
  }
  return val;
}
size_t image_header_box::write(j2c_dst_memory &dst) {
  base_write(dst);
  dst.put_dword(HEIGHT);
  dst.put_dword(WIDTH);
  dst.put_word(NC);
  dst.put_byte(BPC);
  dst.put_byte(C);
  dst.put_byte(UnkC);
  dst.put_byte(IPR);
  return LBox;
}

bits_per_component_box::bits_per_component_box(j2k_main_header &hdr) : box_base(8, 0x62706363) {
  for (uint16_t c = 0; c < hdr.SIZ->get_num_components(); ++c) {
    uint8_t val = static_cast<uint8_t>(hdr.SIZ->get_bitdepth(c) - 1);
    val         = val | static_cast<uint8_t>((hdr.SIZ->is_signed(c)) ? 0x80 : 0);
    BPC.push_back(val);
    LBox++;
  }
}
size_t bits_per_component_box::write(j2c_dst_memory &dst) {
  base_write(dst);
  for (auto &b : BPC) {
    dst.put_byte(b);
  }
  return LBox;
}

colour_specification_box::colour_specification_box(j2k_main_header &hdr, bool isSRGB)
    : box_base(15, 0x636F6C72), METH(1), PREC(0), APPROX(0) {
  // 2 and 4 components carry an alpha channel after the colour channels; see
  // channel_definition_box.
  const uint16_t nc = hdr.SIZ->get_num_components();
  if (nc == 3 || nc == 4) {
    if (isSRGB) {
      EnumCS = 16;
    } else {
      EnumCS = 18;
    }
  } else if (nc == 1 || nc == 2) {
    EnumCS = 17;
  } else {
    open_htj2k::detail::fail(
        "the JPH and JP2 file writer takes 1 to 4 components (greyscale or colour, each optionally "
        "followed by an alpha channel), but the image has %u: write a bare codestream instead.",
        static_cast<unsigned>(nc));
  }
}
size_t colour_specification_box::write(j2c_dst_memory &dst) {
  base_write(dst);
  dst.put_byte(METH);
  dst.put_byte(PREC);
  dst.put_byte(APPROX);
  dst.put_dword(EnumCS);
  return LBox;
}

channel_definition_box::channel_definition_box(j2k_main_header &hdr) : box_base(10, 0x63646566) {
  const uint16_t nc = hdr.SIZ->get_num_components();
  if (nc != 2 && nc != 4) return;
  for (uint16_t c = 0; c < nc; ++c) {
    const bool alpha = (c == nc - 1);
    Cn.push_back(c);
    Typ.push_back(alpha ? 1 : 0);                              // 1: opacity, 0: colour
    Asoc.push_back(alpha ? 0 : static_cast<uint16_t>(c + 1));  // 0: the whole image
    LBox += 6;
  }
}
size_t channel_definition_box::write(j2c_dst_memory &dst) {
  base_write(dst);
  dst.put_word(static_cast<uint16_t>(Cn.size()));
  for (size_t i = 0; i < Cn.size(); ++i) {
    dst.put_word(Cn[i]);
    dst.put_word(Typ[i]);
    dst.put_word(Asoc[i]);
  }
  return LBox;
}

header_box::header_box(j2k_main_header &hdr, bool isSRGB)
    : box_base(8, 0x6A703268), ihdr(hdr), bpcc(hdr), colr(hdr, isSRGB), cdef(hdr) {
  LBox += ihdr.LBox + colr.LBox;
  if (ihdr.needBPCC()) {
    LBox += bpcc.LBox;
  }
  if (cdef.is_needed()) {
    LBox += cdef.LBox;
  }
}
size_t header_box::write(j2c_dst_memory &dst) {
  base_write(dst);
  ihdr.write(dst);
  if (ihdr.needBPCC()) {
    bpcc.write(dst);
  }
  colr.write(dst);
  if (cdef.is_needed()) {
    cdef.write(dst);
  }
  return LBox;
}

contiguous_codestream_box::contiguous_codestream_box(size_t len)
    : box_base(0, 0x6A703263), code_len(static_cast<uint64_t>(len)) {}

size_t contiguous_codestream_box::write(j2c_dst_memory &dst) {
  uint8_t hdr[16];
  const size_t hdr_len = jp2c_box_header(code_len, hdr);
  dst.put_N_bytes(hdr, static_cast<uint32_t>(hdr_len));
  return static_cast<size_t>(hdr_len + code_len);
}

jph_boxes::jph_boxes(j2k_main_header &hdr, uint8_t type, bool isSRGB, size_t code_len)
    : sig(), ftyp(type), jp2h(hdr, isSRGB), jp2c(code_len) {}

size_t jph_boxes::write(j2c_dst_memory &dst) {
  size_t len = 0;
  len += sig.write(dst);
  len += ftyp.write(dst);
  len += jp2h.write(dst);
  len += jp2c.write(dst);
  return len;
}
}  // namespace open_htj2k
