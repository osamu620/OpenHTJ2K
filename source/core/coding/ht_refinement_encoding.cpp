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

// HT SigProp and HT MagRef encoding (ISO/IEC 15444-15: clauses 7.4 and 7.5 define
// the passes, F.4 the bit packing and the termination of the HT refinement segment).
//
// A code-block coded here carries one HT set of three passes: the HT cleanup pass at
// bit-plane p (p >= 1), then the HT SigProp and HT MagRef passes of plane p - 1.  The
// SigProp pass only reaches samples next to a significant one, so plane p - 1 is not
// coded completely: the result is close to, but not, a full-precision code-block.
//
// The cleanup pass is produced by the regular (SIMD) cleanup coder, unchanged: the
// refinement passes run first and leave the samples with their magnitudes shifted
// down by p, in place.  The refinement passes are plain C++, bit-parallel over 64-bit
// words that each hold 16 columns of a stripe of four rows in scan order (bit 4c + r
// is column c, row r).  SigProp's spread of new significance along a row, a serial
// chain from column to column, becomes one carry-propagating addition per word.

#include <algorithm>
#include <cstdlib>
#include <cstring>

#include "coding_units.hpp"
#include "ht_sigprop.hpp"
#include "utils.hpp"

// With BMI2, PEXT extracts a coding group's bits in one instruction.  Zen 1 and Zen 2
// run PEXT as a slow microcoded loop, so builds tuned for them use the table instead.
#if (defined(__BMI2__) || (defined(_MSC_VER) && defined(__AVX2__))) && !defined(__znver1__) \
    && !defined(__znver2__)
  #include <immintrin.h>
  #define OPENHTJ2K_REFINE_PEXT 1
#elif defined(OPENHTJ2K_ENABLE_ARM_NEON) && defined(__aarch64__)
  // AArch64 NEON extracts the bits of a whole word at once (per-lane shifts and
  // 128-bit lane operations that 32-bit ARM lacks; it keeps the table).
  #include <arm_neon.h>
  #define OPENHTJ2K_REFINE_NEON 1
#endif

namespace {

// Magnitude of a sample in the quantiser's HT format: bit 31 marks a non-zero
// sample, whose magnitude m is stored as ((m - 1) << 1) | sign.
inline uint32_t sample_magnitude(uint32_t u) { return ((u & 0x7FFFFFFFu) >> 1) + (u >> 31); }

#if !defined(OPENHTJ2K_REFINE_PEXT) && !defined(OPENHTJ2K_REFINE_NEON)
// The bits of v at the set positions of the 4-bit mask m, packed from bit 0 (a 4-bit
// parallel bit extract), and the number of set bits of m.
struct nibble_tables {
  uint8_t extract[256];  // index (m << 4) | v
  uint8_t count[16];
  nibble_tables() {
    for (uint32_t m = 0; m < 16; ++m) {
      count[m] = static_cast<uint8_t>((m & 1) + ((m >> 1) & 1) + ((m >> 2) & 1) + (m >> 3));
      for (uint32_t v = 0; v < 16; ++v) {
        uint32_t r = 0, k = 0;
        for (uint32_t i = 0; i < 4; ++i)
          if ((m >> i) & 1) r |= ((v >> i) & 1) << k++;
        extract[(m << 4) | v] = static_cast<uint8_t>(r);
      }
    }
  }
};
const nibble_tables nt;
#endif

#ifdef OPENHTJ2K_REFINE_NEON
// The 16 nibbles of w, one per byte lane, in order.
inline uint8x16_t nibble_lanes(uint64_t w) {
  const uint8x8_t x  = vcreate_u8(w);
  const uint8x8_t lo = vand_u8(x, vdup_n_u8(0x0F));
  const uint8x8_t hi = vshr_n_u8(x, 4);
  return vcombine_u8(vzip1_u8(lo, hi), vzip2_u8(lo, hi));
}

// Per byte lane, the bits of the 4-bit value v at the set positions of the 4-bit mask
// m packed from bit 0, as two 64-bit lanes: each the concatenation of eight byte
// lanes' bits in lane order (at most 32 bits), with their number in n.
inline uint64x2_t extract_lanes(uint8x16_t v, uint8x16_t m, uint64x2_t &n) {
  // Within each pair of bits, the upper bit moves down where the lower mask bit is 0;
  // then the upper pair moves down to follow the lower pair's bits.
  const uint8x16_t hole = vbicq_u8(vdupq_n_u8(0x5), m);
  uint8x16_t x          = vandq_u8(v, m);
  x                     = vorrq_u8(vbicq_u8(x, vshlq_n_u8(hole, 1)), vandq_u8(vshrq_n_u8(x, 1), hole));
  const uint8x16_t c_lo = vcntq_u8(vandq_u8(m, vdupq_n_u8(0x3)));
  const uint8x16_t e =
      vorrq_u8(vandq_u8(x, vdupq_n_u8(0x3)), vshlq_u8(vshrq_n_u8(x, 2), vreinterpretq_s8_u8(c_lo)));
  // Join neighbouring lanes: the upper one's bits follow the lower one's.
  const uint8x16_t c8 = vcntq_u8(m);
  const uint16x8_t e8 = vreinterpretq_u16_u8(e);
  const uint16x8_t k8 = vreinterpretq_u16_u8(c8);
  const uint16x8_t e16 =
      vorrq_u16(vandq_u16(e8, vdupq_n_u16(0xFF)),
                vshlq_u16(vshrq_n_u16(e8, 8), vreinterpretq_s16_u16(vandq_u16(k8, vdupq_n_u16(0xFF)))));
  const uint16x8_t c16  = vpaddlq_u8(c8);
  const uint32x4_t e16w = vreinterpretq_u32_u16(e16);
  const uint32x4_t k16  = vreinterpretq_u32_u16(c16);
  const uint32x4_t e32  = vorrq_u32(
      vandq_u32(e16w, vdupq_n_u32(0xFFFF)),
      vshlq_u32(vshrq_n_u32(e16w, 16), vreinterpretq_s32_u32(vandq_u32(k16, vdupq_n_u32(0xFFFF)))));
  const uint32x4_t c32  = vpaddlq_u16(c16);
  const uint64x2_t e32w = vreinterpretq_u64_u32(e32);
  const uint64x2_t k32  = vreinterpretq_u64_u32(c32);
  n                     = vpaddlq_u32(c32);
  return vorrq_u64(
      vandq_u64(e32w, vdupq_n_u64(0xFFFFFFFFull)),
      vshlq_u64(vshrq_n_u64(e32w, 32), vreinterpretq_s64_u64(vandq_u64(k32, vdupq_n_u64(0xFFFFFFFFull)))));
}
#endif

#ifndef OPENHTJ2K_REFINE_NEON
// The bits of v at the set positions of m, packed from bit 0 in order; n receives
// their number.
inline uint32_t extract_bits(uint32_t v, uint32_t m, uint32_t &n) {
  #ifdef OPENHTJ2K_REFINE_PEXT
  n = static_cast<uint32_t>(popcount32(m));
  return _pext_u32(v, m);
  #else
  if (m == 0) {  // common: no newly significant sample, or none significant
    n = 0;
    return 0;
  }
  if (m == 0xFFFFFFFFu) {  // common in dense code-blocks
    n = 32;
    return v;
  }
  uint32_t r = 0, k = 0;
  for (uint32_t i = 0; i < 32; i += 4) {
    const uint32_t mi = (m >> i) & 0xF;
    r |= static_cast<uint32_t>(nt.extract[(mi << 4) | ((v >> i) & 0xF)]) << k;
    k += nt.count[mi];
  }
  n = k;
  return r;
  #endif
}
#endif

// Stores the bytes of x at p, least (le) or most (be) significant first.
inline void store_le32(uint8_t *p, uint32_t x) {
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  x = (x >> 24) | ((x >> 8) & 0xFF00u) | ((x << 8) & 0xFF0000u) | (x << 24);
#endif
  std::memcpy(p, &x, 4);
}
inline void store_be32(uint8_t *p, uint32_t x) {
#if !defined(__BYTE_ORDER__) || __BYTE_ORDER__ != __ORDER_BIG_ENDIAN__
  x = (x >> 24) | ((x >> 8) & 0xFF00u) | ((x << 8) & 0xFF0000u) | (x << 24);
#endif
  std::memcpy(p, &x, 4);
}

// Both packers collect bits LSB first in a 64-bit accumulator.  While at least 32 bits
// are pending they write bytes, four at a time when none of them involves a stuffed
// bit.  finish() writes the whole bytes left and keeps the partial byte in tmp and
// bits for termination, as the bit-serial packers of F.4 would.

// F.4 initSPPacker / emitSPBit: SigProp bits go LSB first; the byte after an 0xFF
// holds only 7 bits.
struct sp_packer {
  uint8_t *buf;
  uint32_t pos  = 0;
  uint32_t bits = 0;  // pending bits in acc; after finish(), in tmp
  uint32_t max  = 8;  // bits in the next byte
  uint32_t tmp  = 0;
  uint64_t acc  = 0;
  explicit sp_packer(uint8_t *b) : buf(b) {}
  // Appends the n (<= 32) low bits of v, first bit in bit 0.  The next four bytes are
  // stored every time and kept once 32 bits are pending.  The only branch is for an
  // 0xFF byte among them or a 7-bit byte next, both rare; a byte not yet complete has
  // zeros above its bits, so it never looks like 0xFF.  buf needs 3 bytes of room past
  // the segment.
  void emit(uint32_t v, uint32_t n) {
    acc |= static_cast<uint64_t>(v) << bits;
    bits += n;
    const uint32_t x = static_cast<uint32_t>(acc);
    if ((((~x - 0x01010101u) & x & 0x80808080u) | (max ^ 8)) != 0) {
      while (bits >= 32) put_byte();
      return;
    }
    const uint32_t full = bits & 32;  // bits < 64
    store_le32(buf + pos, x);
    pos += full >> 3;
    acc >>= full;
    bits -= full;
  }
  void put_byte() {
    const uint32_t byte = static_cast<uint32_t>(acc) & ((1u << max) - 1);
    buf[pos++]          = static_cast<uint8_t>(byte);
    acc >>= max;
    bits -= max;
    max = (byte == 0xFF) ? 7 : 8;
  }
  void finish() {
    while (bits >= max) put_byte();
    tmp = static_cast<uint32_t>(acc);
  }
};

// F.4 initMRPacker / emitMRBit: MagRef bits go LSB first; the bytes are placed in
// reverse order at the end of the refinement segment, so they are written here from
// the end of buf (MR_BUF_SIZE bytes) down.  A byte whose 7 LSBs are all 1 after a byte
// above 0x8F gets a stuffed 0 in its MSB, so it holds 7 bits.
constexpr uint32_t MR_BUF_SIZE = HT_MAX_LREF;
struct mr_packer {
  uint8_t *buf;
  uint32_t pos  = 0;  // bytes written, at buf[MR_BUF_SIZE - pos] up
  uint32_t bits = 0;  // pending bits in acc; after finish(), in tmp
  uint32_t tmp  = 0;
  uint32_t last = 0xFF;
  uint64_t acc  = 0;
  explicit mr_packer(uint8_t *b) : buf(b) {}
  // Appends the n (<= 32) low bits of v, first bit in bit 0, storing the next four
  // bytes every time as the SigProp packer does.  The rare branch is for a byte whose
  // 7 LSBs are all 1 among them (it may get a stuffed bit).  buf needs 4 bytes of room
  // below the segment.
  void emit(uint32_t v, uint32_t n) {
    acc |= static_cast<uint64_t>(v) << bits;
    bits += n;
    const uint32_t x = static_cast<uint32_t>(acc);
    if ((((x & 0x7F7F7F7Fu) + 0x01010101u) & 0x80808080u) != 0) {
      while (bits >= 32) put_byte();
      return;
    }
    const uint32_t full = bits & 32;  // bits < 64
    store_be32(buf + MR_BUF_SIZE - pos - 4, x);
    last = full ? x >> 24 : last;
    pos += full >> 3;
    acc >>= full;
    bits -= full;
  }
  bool stuffed() const { return last > 0x8F && (acc & 0x7F) == 0x7F; }
  void put_byte() {
    uint32_t byte = static_cast<uint32_t>(acc) & 0xFF;
    uint32_t used = 8;
    if (stuffed()) {
      byte = 0x7F;
      used = 7;
    }
    buf[MR_BUF_SIZE - ++pos] = static_cast<uint8_t>(byte);
    last                     = byte;
    acc >>= used;
    bits -= used;
  }
  // Writes bytes while 8 bits are pending, or 7 for a byte that gets a stuffed bit.
  void finish() {
    while (bits >= 8 || (bits == 7 && stuffed())) put_byte();
    tmp = static_cast<uint32_t>(acc);
  }
};

// F.4 termSPandMRPackers: the refinement segment is the SigProp bytes followed by the
// MagRef bytes in reverse order.  The two final partial bytes share one byte when
// their bits agree.  sp.buf is the output; returns the segment length Lref.  info, when
// given, receives the segment that termination gives with no MagRef bits (the set cut
// after SigProp): the same bytes up to the partial one, which then stands alone.
uint32_t terminate_refinement(sp_packer &sp, mr_packer &mr, ht_refinement_info *info) {
  sp.finish();
  mr.finish();
  uint32_t sp_mask = 0xFFu >> (8 - sp.bits);  // 0 when no SigProp bit is pending
  sp_mask |= (1u << sp.max) & 0x80u;          // a pending byte after 0xFF keeps its MSB 0
  if (info != nullptr) {
    info->Lsp     = sp.pos + (sp_mask != 0);
    info->sp_tail = static_cast<uint8_t>((sp_mask != 0) ? sp.tmp : (sp.pos ? sp.buf[sp.pos - 1] : 0));
  }
  const uint32_t mr_mask = 0xFFu >> (8 - mr.bits);
  if ((sp_mask | mr_mask) != 0) {
    const uint32_t fuse = sp.tmp | mr.tmp;
    if ((((fuse ^ sp.tmp) & sp_mask) | ((fuse ^ mr.tmp) & mr_mask)) == 0) {
      sp.buf[sp.pos] = static_cast<uint8_t>(fuse);  // below 0x80: no false marker
    } else {
      sp.buf[sp.pos]                 = static_cast<uint8_t>(sp.tmp);
      mr.buf[MR_BUF_SIZE - ++mr.pos] = static_cast<uint8_t>(mr.tmp);
    }
    ++sp.pos;
  }
  std::memcpy(sp.buf + sp.pos, mr.buf + MR_BUF_SIZE - mr.pos, mr.pos);
  return sp.pos + mr.pos;
}

using ht_sigprop::MAX_WORDS;

struct stripe_words {
  uint64_t sig[MAX_WORDS];   // significant after the cleanup pass: magnitude at least 2^p
  uint64_t bit[MAX_WORDS];   // bit p - 1 of the magnitude
  uint64_t sign[MAX_WORDS];  // sign
};

// 16 bytes that each hold a 4-bit value, as one word (byte c at bits 4c to 4c + 3).
inline uint64_t pack_nibbles(const uint8_t *b) {
  uint64_t lo = 0, hi = 0;
  for (uint32_t i = 0; i < 8; ++i) {
    lo |= static_cast<uint64_t>(b[i]) << (8 * i);
    hi |= static_cast<uint64_t>(b[i + 8]) << (8 * i);
  }
  lo = (lo | (lo >> 4)) & 0x00FF00FF00FF00FFull;
  hi = (hi | (hi >> 4)) & 0x00FF00FF00FF00FFull;
  lo = (lo | (lo >> 8)) & 0x0000FFFF0000FFFFull;
  hi = (hi | (hi >> 8)) & 0x0000FFFF0000FFFFull;
  lo = (lo | (lo >> 16)) & 0x00000000FFFFFFFFull;
  hi = (hi | (hi >> 16)) & 0x00000000FFFFFFFFull;
  return lo | (hi << 32);
}

// Codes the SigProp and MagRef passes of plane q = p - 1 after a cleanup pass at
// plane p into out (at least HT_MAX_LREF bytes) and returns Lref.  The samples are
// left with their magnitudes shifted down by p, ready for the cleanup pass.  info, when
// given, receives the segment cut after SigProp and, if info->newly is set, the samples
// SigProp makes significant (stripe s, word k at newly[s * ((W + 15) / 16) + k]).
uint32_t encode_refinement_passes(j2k_codeblock *block, uint32_t p, uint8_t *out,
                                  ht_refinement_info *info) {
  const uint32_t W         = block->size.x;
  const uint32_t H         = block->size.y;
  const size_t stride      = block->blksampl_stride;
  uint32_t *samples        = reinterpret_cast<uint32_t *>(block->sample_buf);
  const uint32_t q         = p - 1;
  const bool causal        = (block->Cmodes & CAUSAL) != 0;
  const uint32_t ns        = (H + 3) / 4;                                      // stripes
  const uint32_t nk        = (W + 15) / 16;                                    // words per stripe
  const uint64_t last_cols = (W & 15) ? (1ull << (4 * (W & 15))) - 1 : ~0ull;  // in word nk - 1

  // A 4096-sample block needs at most ~1.1 KB of SigProp and ~0.6 KB of MagRef bytes.
  uint8_t mr_buf[MR_BUF_SIZE];
  sp_packer sp(out);
  mr_packer mr(mr_buf);

  uint8_t sg[1024], bt[1024], sn[1024];  // a stripe per column (bit r = row r), packed below
  stripe_words words[2];                 // the stripe being coded and the next one
  uint64_t above[MAX_WORDS] = {};        // at row 3: the stripe above's last row after SigProp
  uint64_t newly[MAX_WORDS];             // became significant in this SigProp pass
  uint64_t coded[MAX_WORDS];             // coded by this SigProp pass
  // Each step reads stripe t, then codes the stripe before it, whose SigProp
  // neighbourhood reaches the first row of stripe t.
  for (uint32_t t = 0; t <= ns; ++t) {
    stripe_words &next = words[t & 1];
    if (t < ns) {
      // All four rows: the quantiser fills rows up to a multiple of 8, past H with zero.
      const uint32_t *r0 = samples + 4 * t * stride;
      const uint32_t *r1 = r0 + stride;
      const uint32_t *r2 = r1 + stride;
      const uint32_t *r3 = r2 + stride;
      for (uint32_t j = 0; j < W; ++j) {
        const uint32_t u0 = r0[j], u1 = r1[j], u2 = r2[j], u3 = r3[j];
        const uint32_t m0 = sample_magnitude(u0), m1 = sample_magnitude(u1);
        const uint32_t m2 = sample_magnitude(u2), m3 = sample_magnitude(u3);
        sg[j] = static_cast<uint8_t>(((m0 >> p) != 0) | (((m1 >> p) != 0) << 1) | (((m2 >> p) != 0) << 2)
                                     | (((m3 >> p) != 0) << 3));
        bt[j] = static_cast<uint8_t>(((m0 >> q) & 1) | (((m1 >> q) & 1) << 1) | (((m2 >> q) & 1) << 2)
                                     | (((m3 >> q) & 1) << 3));
        sn[j] = static_cast<uint8_t>((u0 & 1) | ((u1 & 1) << 1) | ((u2 & 1) << 2) | ((u3 & 1) << 3));
      }
      const size_t len = 16 * static_cast<size_t>(nk);
      std::memset(sg + W, 0, len - W);
      std::memset(bt + W, 0, len - W);
      std::memset(sn + W, 0, len - W);
      for (uint32_t k = 0; k < nk; ++k) {
        next.sig[k]  = pack_nibbles(sg + 16 * k);
        next.bit[k]  = pack_nibbles(bt + 16 * k);
        next.sign[k] = pack_nibbles(sn + 16 * k);
      }
    } else {
      std::memset(next.sig, 0, nk * sizeof(uint64_t));
    }
    if (t == 0) continue;
    const uint32_t s        = t - 1;
    const stripe_words &cur = words[s & 1];
    const uint32_t rows     = std::min(4u, H - 4 * s);
    ht_sigprop::solve_stripe<>(cur.sig, cur.bit, next.sig, above, nk, rows, last_cols, causal, newly,
                               coded);

    if (info != nullptr && info->newly != nullptr)
      std::memcpy(info->newly + s * nk, newly, nk * sizeof(uint64_t));
    for (uint32_t k = 0; k < nk; ++k) {
      const uint64_t n    = newly[k];
      const uint64_t cand = coded[k];
      const uint64_t b    = cur.bit[k];
      const uint64_t sgn  = cur.sign[k];
      const uint64_t g    = cur.sig[k];
      // Per group, the magnitude bits and then the sign bits: one extract from a value
      // and mask holding the magnitudes in the low half and the signs in the high half.
      // Groups past the last column have nothing to code and emit no bits.  MagRef
      // (7.5) codes every sample significant after the cleanup pass, in the same order,
      // into its own packer.
#ifdef OPENHTJ2K_REFINE_NEON
      const uint32x4_t bv = vreinterpretq_u32_u8(nibble_lanes(b));
      const uint32x4_t sv = vreinterpretq_u32_u8(nibble_lanes(sgn));
      const uint32x4_t cm = vreinterpretq_u32_u8(nibble_lanes(cand));
      const uint32x4_t nm = vreinterpretq_u32_u8(nibble_lanes(n));
      uint64x2_t n01, n23, nmr;
      const uint64x2_t g01 = extract_lanes(vreinterpretq_u8_u32(vzip1q_u32(bv, sv)),
                                           vreinterpretq_u8_u32(vzip1q_u32(cm, nm)), n01);
      const uint64x2_t g23 = extract_lanes(vreinterpretq_u8_u32(vzip2q_u32(bv, sv)),
                                           vreinterpretq_u8_u32(vzip2q_u32(cm, nm)), n23);
      const uint64x2_t ref = extract_lanes(vreinterpretq_u8_u32(bv), nibble_lanes(g), nmr);
      sp.emit(static_cast<uint32_t>(vgetq_lane_u64(g01, 0)), static_cast<uint32_t>(vgetq_lane_u64(n01, 0)));
      sp.emit(static_cast<uint32_t>(vgetq_lane_u64(g01, 1)), static_cast<uint32_t>(vgetq_lane_u64(n01, 1)));
      sp.emit(static_cast<uint32_t>(vgetq_lane_u64(g23, 0)), static_cast<uint32_t>(vgetq_lane_u64(n23, 0)));
      sp.emit(static_cast<uint32_t>(vgetq_lane_u64(g23, 1)), static_cast<uint32_t>(vgetq_lane_u64(n23, 1)));
      mr.emit(static_cast<uint32_t>(vgetq_lane_u64(ref, 0)), static_cast<uint32_t>(vgetq_lane_u64(nmr, 0)));
      mr.emit(static_cast<uint32_t>(vgetq_lane_u64(ref, 1)), static_cast<uint32_t>(vgetq_lane_u64(nmr, 1)));
#else
      auto group = [&](uint32_t sh) {
        const uint32_t v =
            (static_cast<uint32_t>(b >> sh) & 0xFFFF) | (static_cast<uint32_t>(sgn >> sh) << 16);
        const uint32_t m =
            (static_cast<uint32_t>(cand >> sh) & 0xFFFF) | (static_cast<uint32_t>(n >> sh) << 16);
        uint32_t nbits;
        const uint32_t bits = extract_bits(v, m, nbits);
        sp.emit(bits, nbits);
      };
      group(0);
      group(16);
      group(32);
      group(48);
      uint32_t nref;
      uint32_t ref = extract_bits(static_cast<uint32_t>(b), static_cast<uint32_t>(g), nref);
      mr.emit(ref, nref);
      ref = extract_bits(static_cast<uint32_t>(b >> 32), static_cast<uint32_t>(g >> 32), nref);
      mr.emit(ref, nref);
#endif
    }
  }

  // The cleanup pass codes the magnitudes shifted down by p, in place; zero stays
  // zero, and so does the padding the quantiser cleared.
  const size_t area = H * stride;
  for (size_t k = 0; k < area; ++k) {
    const uint32_t u  = samples[k];
    const uint32_t ms = sample_magnitude(u) >> p;
    samples[k]        = (0x80000000u | ((ms - 1) << 1) | (u & 1)) & (0u - static_cast<uint32_t>(ms != 0));
  }
  return terminate_refinement(sp, mr, info);
}

}  // namespace

int32_t htj2k_encode_refined(j2k_codeblock *block, uint8_t ROIshift, ht_refinement_info *info) noexcept {
  uint32_t or_val = 0;
  if (!block->pre_quantized)
    block->quantize(or_val);
  else
    or_val = block->pre_or_val;
  block->pre_quantized = true;
  block->pre_or_val    = or_val;

  const uint32_t H     = block->size.y;
  const size_t stride  = block->blksampl_stride;
  const uint32_t *src  = reinterpret_cast<const uint32_t *>(block->sample_buf);
  const size_t visible = H * stride;  // the columns past W are zero padding
  uint32_t max_mag     = 0;
  if (or_val) {
    // A non-zero sample holds ((m - 1) << 1) | sign below bit 31 and a zero one holds 0,
    // so the largest of those gives the largest magnitude.  They are below 2^31, so a
    // signed maximum does it (SSE2 has no unsigned one; the loop vectorises with it).
    int32_t top = 0;
    for (size_t k = 0; k < visible; ++k) top = std::max(top, static_cast<int32_t>(src[k] & 0x7FFFFFFFu));
    max_mag = (static_cast<uint32_t>(top) >> 1) + 1;
  }
  // Keep at least one significant sample in the cleanup pass.
  const uint32_t msb = max_mag ? 32 - count_leading_zeros(max_mag) : 0;
  const uint32_t p   = (msb == 0) ? 0 : std::min<uint32_t>(block->ht_refine_planes, msb - 1);
  if (p == 0) {  // cleanup pass only
    if (info != nullptr) info->Lsp = 0;
    return htj2k_encode(block, ROIshift);
  }

  // The refinement passes read the samples and leave their magnitudes shifted down by
  // p for the cleanup pass, in place: nothing reads them after this block is coded.
  alignas(16) uint8_t ref[HT_MAX_LREF];
  const uint32_t Lref = encode_refinement_passes(block, p, ref, info);
  block->refsegment   = true;  // the cleanup coder reserves HT_MAX_LREF bytes after its segment
  htj2k_cleanup_encode(block, ROIshift);

  const uint32_t Lcup = block->length;
  // The MagRef pass codes every significant sample, so the segment is never empty;
  // an empty one would make decoders skip both refinement passes (B.3, Z_blk = 1).
  block->append_refinement_segment(ref, Lref, HT_MAX_LREF);
  block->ensure_pass_capacity(3);
  block->pass_length[1]    = 0;
  block->pass_length[2]    = Lref;  // SigProp and MagRef share one segment
  block->pass_length_count = 3;
  block->num_passes        = 3;
  block->layer_passes[0]   = 3;
  block->length            = Lcup + Lref;
  block->num_ZBP           = static_cast<uint8_t>(block->get_Mb() - 1 - p);
  return EXIT_SUCCESS;
}
