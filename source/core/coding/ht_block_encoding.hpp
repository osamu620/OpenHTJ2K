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

// Portable HT block encoder state (no SIMD).  The bit writers follow the
// AVX2 version in ht_block_encoding_avx2.hpp and produce the same bytes.

#if !defined(OPENHTJ2K_ENABLE_ARM_NEON) && (!defined(__AVX2__) || !defined(OPENHTJ2K_TRY_AVX2))
  #pragma once
  #include <algorithm>
  #include <cstdint>
  #include <cstring>
  #include "open_htj2k_typedef.hpp"
  #include "utils.hpp"

  #ifndef FORCE_INLINE
    #if defined(__GNUC__) || defined(__clang__)
      #define FORCE_INLINE inline __attribute__((always_inline))
    #elif defined(_MSC_VER)
      #define FORCE_INLINE __forceinline
    #else
      #define FORCE_INLINE inline
    #endif
  #endif

  #define MAX_Lcup 16384
  #define MAX_Scup 4079

/********************************************************************************
 * state_MS_enc: state class for MagSgn encoding
 *******************************************************************************/
class state_MS_enc {
 private:
  uint64_t Creg;       // up to 64 pending bits, LSB first
  uint32_t ctreg;      // number of pending bits in Creg
  int32_t pos;         // current position in the buffer
  uint8_t last;        // last byte written
  uint8_t *const buf;  // buffer for MagSgn

  // Write the low 32 bits of cr as four bytes.  A byte that follows 0xFF
  // carries only 7 bits (the stuffed MSB is 0), so fewer than 32 bits may be
  // consumed.
  static FORCE_INLINE void flush_dword(uint64_t &cr, uint32_t &ct, uint8_t &la, int32_t &po,
                                       uint8_t *const b) {
    const uint32_t val = static_cast<uint32_t>(cr);
    // Fast path: no stuffing when none of {last, val[7:0], val[15:8], val[23:16]} is 0xFF.
    if (la < 0xFF && (val & 0xFF) < 0xFF && ((val >> 8) & 0xFF) < 0xFF && ((val >> 16) & 0xFF) < 0xFF) {
      la = static_cast<uint8_t>(val >> 24);
      cr >>= 32;
      ct -= 32;
      store_u32_unaligned(b + po, val);
      po += 4;
      return;
    }
    uint32_t bl = 0, t = 0, stuff = (la == 0xFF), tmp;
    tmp = val & ((1U << (8U - stuff)) - 1U);
    t |= tmp;
    bl += 8 - stuff;
    stuff = (tmp == 0xFF);
    tmp   = (val >> bl) & ((1U << (8U - stuff)) - 1U);
    t |= tmp << 8;
    bl += 8 - stuff;
    stuff = (tmp == 0xFF);
    tmp   = (val >> bl) & ((1U << (8U - stuff)) - 1U);
    t |= tmp << 16;
    bl += 8 - stuff;
    stuff = (tmp == 0xFF);
    tmp   = (val >> bl) & ((1U << (8U - stuff)) - 1U);
    t |= tmp << 24;
    bl += 8 - stuff;
    la = static_cast<uint8_t>(tmp);
    cr >>= bl;
    ct -= bl;
    store_u32_unaligned(b + po, t);
    po += 4;
  }

 public:
  explicit state_MS_enc(uint8_t *p) : Creg(0), ctreg(0), pos(0), last(0), buf(p) {}

  // Emit count codewords: v[i] holds m[i] bits (m[i] <= 32).
  #ifdef _MSC_VER
  __declspec(noinline)
  #else
  __attribute__((noinline))
  #endif
  void
  emitFlat(const uint32_t *v, const uint32_t *m, int32_t count) {
    uint64_t cr = Creg;
    uint32_t ct = ctreg;
    uint8_t la  = last;
    int32_t po  = pos;
    int32_t i   = 0;
    for (; i + 3 < count; i += 4) {
      const uint32_t m_sum = m[i] + m[i + 1] + m[i + 2] + m[i + 3];
      if (ct + m_sum < 64) {
        uint32_t s0 = ct;
        cr |= static_cast<uint64_t>(v[i]) << s0;
        s0 += m[i];
        cr |= static_cast<uint64_t>(v[i + 1]) << s0;
        s0 += m[i + 1];
        cr |= static_cast<uint64_t>(v[i + 2]) << s0;
        s0 += m[i + 2];
        cr |= static_cast<uint64_t>(v[i + 3]) << s0;
        ct += m_sum;
        if (ct >= 32) flush_dword(cr, ct, la, po, buf);
      } else {
        for (int32_t k = 0; k < 4; ++k) {
          cr |= static_cast<uint64_t>(v[i + k]) << ct;
          ct += m[i + k];
          if (ct >= 32) flush_dword(cr, ct, la, po, buf);
        }
      }
    }
    for (; i < count; ++i) {
      cr |= static_cast<uint64_t>(v[i]) << ct;
      ct += m[i];
      while (ct >= 32) flush_dword(cr, ct, la, po, buf);
    }
    Creg  = cr;
    ctreg = ct;
    last  = la;
    pos   = po;
  }

  int32_t termMS() {
    while (true) {
      if (last == 0xFF) {
        if (ctreg < 7) break;
        last = static_cast<uint8_t>(Creg & 0x7F);
        Creg >>= 7;
        ctreg -= 7;
      } else {
        if (ctreg < 8) break;
        last = static_cast<uint8_t>(Creg & 0xFF);
        Creg >>= 8;
        ctreg -= 8;
      }
      buf[pos++] = last;
    }
    bool last_was_FF = (last == 0xFF);
    uint8_t fill_mask, cwd;
    if (ctreg > 0) {
      fill_mask = static_cast<uint8_t>(0xFF << ctreg);
      if (last_was_FF) {
        fill_mask &= 0x7F;
      }
      cwd = static_cast<uint8_t>(Creg |= fill_mask);
      if (cwd != 0xFF) {
        buf[pos++] = cwd;
      }
    } else if (last_was_FF) {
      pos--;
      buf[pos] = 0x00;  // may be not necessary
    }
    return pos;  // return current position as Pcup
  }
};

class state_MEL_enc;  // forward declaration for friend function "termMELandVLC()"
/********************************************************************************
 * state_VLC_enc: state class for VLC encoding (written backwards)
 *******************************************************************************/
class state_VLC_enc {
 private:
  uint64_t Creg;
  uint32_t ctreg;
  uint8_t *const buf;
  uint8_t tmp;
  uint8_t last;
  uint8_t bits;
  int32_t pos;

  friend int32_t termMELandVLC(state_VLC_enc &, state_MEL_enc &);

  FORCE_INLINE void emit_dword() {
    const uint32_t val = static_cast<uint32_t>(Creg);
    // Fast path: every byte of {last, val[7:0], val[15:8], val[23:16]} is
    // 0x00-0x0F or 0x80-0x8F, so none exceeds 0x8F and no bit is stuffed.
    if (((last | (val & 0xFF) | ((val >> 8) & 0xFF) | ((val >> 16) & 0xFF)) & 0x70) == 0) {
      last = static_cast<uint8_t>(val >> 24);
      Creg >>= 32;
      ctreg -= 32;
      store_u32_unaligned(buf + pos - 3, openhtj2k_bswap32(val));
      pos -= 4;
      return;
    }
    uint32_t bits_local = 0;
    uint32_t temp, t = 0;
    uint32_t stuff;
    uint32_t last_byte = last;
    for (int32_t k = 3; k >= 0; --k) {
      temp      = (val >> bits_local) & 0xFF;
      stuff     = (last_byte > 0x8F) && ((temp & 0x7F) == 0x7F);
      last_byte = temp & ((1U << (8 - stuff)) - 1);
      t |= last_byte << (8 * k);
      bits_local += 8 - stuff;
    }
    last = static_cast<uint8_t>(last_byte);
    Creg >>= bits_local;
    ctreg -= bits_local;
    store_u32_unaligned(buf + pos - 3, t);
    pos -= 4;
  }

  FORCE_INLINE void termVLC() {
    while (true) {
      uint8_t t = Creg & 0x7F;
      if ((last > 0x8F) && (t == 0x7F)) {
        if (ctreg < 7) break;
        last = static_cast<uint8_t>(Creg & 0x7F);
        Creg >>= 7;
        ctreg -= 7;
      } else {
        if (ctreg < 8) break;
        last = static_cast<uint8_t>(Creg & 0xFF);
        Creg >>= 8;
        ctreg -= 8;
      }
      buf[pos--] = last;
    }
    bits = static_cast<uint8_t>(ctreg & 0xFF);
    tmp  = Creg & 0xFF;
  }

 public:
  explicit state_VLC_enc(uint8_t *p)
      : Creg(0xF), ctreg(4), buf(p), tmp(0xF), last(0xFF), bits(4), pos(MAX_Scup - 2) {
    buf[pos + 1] = 0xFF;
  }

  // Append len bits of cwd (len + pending bits must stay below 64).
  FORCE_INLINE void emitVLCBits(uint64_t cwd, uint32_t len) {
    Creg |= cwd << ctreg;
    ctreg += len;
    while (ctreg >= 32) {
      emit_dword();
    }
  }
};

/********************************************************************************
 * state_MEL_enc: state class for MEL encoding
 *******************************************************************************/
class state_MEL_enc {
 private:
  int8_t MEL_k;
  uint8_t MEL_run;
  const uint8_t MEL_E[13];
  uint8_t MEL_t;
  int32_t pos;
  uint8_t rem;
  uint8_t tmp;
  uint8_t *const buf;
  FORCE_INLINE void emitMELbit(uint8_t bit) {
    tmp = static_cast<uint8_t>((tmp << 1) + bit);
    rem--;
    if (rem == 0) {
      buf[pos] = tmp;
      pos++;
      rem = (tmp == 0xFF) ? 7 : 8;
      tmp = 0;
    }
  }

  friend int32_t termMELandVLC(state_VLC_enc &, state_MEL_enc &);

 public:
  explicit state_MEL_enc(uint8_t *p)
      : MEL_k(0),
        MEL_run(0),
        MEL_E{0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 5},
        MEL_t(static_cast<uint8_t>(1 << MEL_E[MEL_k])),
        pos(0),
        rem(8),
        tmp(0),
        buf(p) {}
  FORCE_INLINE void encodeMEL(uint8_t smel) {
    uint8_t eval;
    if (smel == 0) {
      MEL_run++;
      if (MEL_run >= MEL_t) {
        emitMELbit(1);
        MEL_run = 0;
        MEL_k   = (int8_t)std::min(12, MEL_k + 1);
        eval    = MEL_E[MEL_k];
        MEL_t   = static_cast<uint8_t>(1 << eval);
      }
    } else {
      emitMELbit(0);
      eval = MEL_E[MEL_k];
      while (eval > 0) {
        eval--;
        // (MEL_run >> eval) & 1 = msb
        emitMELbit((MEL_run >> eval) & 1);
      }
      MEL_run = 0;
      MEL_k   = (int8_t)std::max(0, MEL_k - 1);
      eval    = MEL_E[MEL_k];
      MEL_t   = static_cast<uint8_t>(1 << eval);
    }
  }
  void termMEL() {
    if (MEL_run > 0) {
      emitMELbit(1);
    }
  }
};

#endif
