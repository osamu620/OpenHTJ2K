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
#include <exception>

#if __GNUC__ || __has_attribute(always_inline)
  #define FORCE_INLINE inline __attribute__((always_inline))
  #define openhtj2k_arm_clzll(x) __builtin_clzll((x))
#elif defined(_MSC_VER)
  #define FORCE_INLINE __forceinline
  #define openhtj2k_arm_clzll(x) _CountLeadingZeros64((x))
#else
  #define FORCE_INLINE inline
  #define openhtj2k_arm_clzll(x) __builtin_clzll((x))
#endif

// LUT for UVLC decoding in initial line-pair
//   index (8bits) : [bit   7] u_off_1 (1bit)
//                   [bit   6] u_off_0 (1bit)
//                   [bit 5-0] LSB bits from VLC codeword
//   the index is incremented by 64 when both u_off_0 and u_off_1 are 0
//
//   output        : [bit 0-2] length of prefix (l_p) for quads 0 and 1
//                 : [bit 3-6] length of suffix (l_s) for quads 0 and 1
//                 : [bit 7-9] ength of suffix (l_s) for quads 0
constexpr uint16_t uvlc_dec_0[256 + 64] = {
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x16ab,
    0x0401, 0x0802, 0x0401, 0x0c8b, 0x0401, 0x0802, 0x0401, 0x16ab, 0x0401, 0x0802, 0x0401, 0x0c8b, 0x0401,
    0x0802, 0x0401, 0x16ab, 0x0401, 0x0802, 0x0401, 0x0c8b, 0x0401, 0x0802, 0x0401, 0x16ab, 0x0401, 0x0802,
    0x0401, 0x0c8b, 0x0401, 0x0802, 0x0401, 0x16ab, 0x0401, 0x0802, 0x0401, 0x0c8b, 0x0401, 0x0802, 0x0401,
    0x16ab, 0x0401, 0x0802, 0x0401, 0x0c8b, 0x0401, 0x0802, 0x0401, 0x16ab, 0x0401, 0x0802, 0x0401, 0x0c8b,
    0x0401, 0x0802, 0x0401, 0x16ab, 0x0401, 0x0802, 0x0401, 0x0c8b, 0x0401, 0x0802, 0x0401, 0xa02b, 0x2001,
    0x4002, 0x2001, 0x600b, 0x2001, 0x4002, 0x2001, 0xa02b, 0x2001, 0x4002, 0x2001, 0x600b, 0x2001, 0x4002,
    0x2001, 0xa02b, 0x2001, 0x4002, 0x2001, 0x600b, 0x2001, 0x4002, 0x2001, 0xa02b, 0x2001, 0x4002, 0x2001,
    0x600b, 0x2001, 0x4002, 0x2001, 0xa02b, 0x2001, 0x4002, 0x2001, 0x600b, 0x2001, 0x4002, 0x2001, 0xa02b,
    0x2001, 0x4002, 0x2001, 0x600b, 0x2001, 0x4002, 0x2001, 0xa02b, 0x2001, 0x4002, 0x2001, 0x600b, 0x2001,
    0x4002, 0x2001, 0xa02b, 0x2001, 0x4002, 0x2001, 0x600b, 0x2001, 0x4002, 0x2001, 0x36ac, 0xa42c, 0xa82d,
    0x2402, 0x2c8c, 0x4403, 0x2803, 0x2402, 0x56ac, 0x640c, 0x4804, 0x2402, 0x4c8c, 0x4403, 0x2803, 0x2402,
    0x36ac, 0xa42c, 0x680d, 0x2402, 0x2c8c, 0x4403, 0x2803, 0x2402, 0x56ac, 0x640c, 0x4804, 0x2402, 0x4c8c,
    0x4403, 0x2803, 0x2402, 0x36ac, 0xa42c, 0xa82d, 0x2402, 0x2c8c, 0x4403, 0x2803, 0x2402, 0x56ac, 0x640c,
    0x4804, 0x2402, 0x4c8c, 0x4403, 0x2803, 0x2402, 0x36ac, 0xa42c, 0x680d, 0x2402, 0x2c8c, 0x4403, 0x2803,
    0x2402, 0x56ac, 0x640c, 0x4804, 0x2402, 0x4c8c, 0x4403, 0x2803, 0x2402, 0xfed6, 0xec2c, 0xf02d, 0x6c02,
    0xf4b6, 0x8c03, 0x7003, 0x6c02, 0x7eac, 0xac0c, 0x9004, 0x6c02, 0x748c, 0x8c03, 0x7003, 0x6c02, 0x9ead,
    0xec2c, 0xb00d, 0x6c02, 0x948d, 0x8c03, 0x7003, 0x6c02, 0x7eac, 0xac0c, 0x9004, 0x6c02, 0x748c, 0x8c03,
    0x7003, 0x6c02, 0xbeb6, 0xec2c, 0xf02d, 0x6c02, 0xb496, 0x8c03, 0x7003, 0x6c02, 0x7eac, 0xac0c, 0x9004,
    0x6c02, 0x748c, 0x8c03, 0x7003, 0x6c02, 0x9ead, 0xec2c, 0xb00d, 0x6c02, 0x948d, 0x8c03, 0x7003, 0x6c02,
    0x7eac, 0xac0c, 0x9004, 0x6c02, 0x748c, 0x8c03, 0x7003, 0x6c02};

// LUT for UVLC decoding in non-initial line-pair
//   index (8bits) : [bit   7] u_off_1 (1bit)
//                   [bit   6] u_off_0 (1bit)
//                   [bit 5-0] LSB bits from VLC codeword
//
//   output        : [bit 0-2] length of prefix (l_p) for quads 0 and 1
//                 : [bit 3-6] length of suffix (l_s) for quads 0 and 1
//                 : [bit 7-9] ength of suffix (l_s) for quads 0
constexpr uint16_t uvlc_dec_1[256] = {
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x16ab,
    0x0401, 0x0802, 0x0401, 0x0c8b, 0x0401, 0x0802, 0x0401, 0x16ab, 0x0401, 0x0802, 0x0401, 0x0c8b, 0x0401,
    0x0802, 0x0401, 0x16ab, 0x0401, 0x0802, 0x0401, 0x0c8b, 0x0401, 0x0802, 0x0401, 0x16ab, 0x0401, 0x0802,
    0x0401, 0x0c8b, 0x0401, 0x0802, 0x0401, 0x16ab, 0x0401, 0x0802, 0x0401, 0x0c8b, 0x0401, 0x0802, 0x0401,
    0x16ab, 0x0401, 0x0802, 0x0401, 0x0c8b, 0x0401, 0x0802, 0x0401, 0x16ab, 0x0401, 0x0802, 0x0401, 0x0c8b,
    0x0401, 0x0802, 0x0401, 0x16ab, 0x0401, 0x0802, 0x0401, 0x0c8b, 0x0401, 0x0802, 0x0401, 0xa02b, 0x2001,
    0x4002, 0x2001, 0x600b, 0x2001, 0x4002, 0x2001, 0xa02b, 0x2001, 0x4002, 0x2001, 0x600b, 0x2001, 0x4002,
    0x2001, 0xa02b, 0x2001, 0x4002, 0x2001, 0x600b, 0x2001, 0x4002, 0x2001, 0xa02b, 0x2001, 0x4002, 0x2001,
    0x600b, 0x2001, 0x4002, 0x2001, 0xa02b, 0x2001, 0x4002, 0x2001, 0x600b, 0x2001, 0x4002, 0x2001, 0xa02b,
    0x2001, 0x4002, 0x2001, 0x600b, 0x2001, 0x4002, 0x2001, 0xa02b, 0x2001, 0x4002, 0x2001, 0x600b, 0x2001,
    0x4002, 0x2001, 0xa02b, 0x2001, 0x4002, 0x2001, 0x600b, 0x2001, 0x4002, 0x2001, 0xb6d6, 0xa42c, 0xa82d,
    0x2402, 0xacb6, 0x4403, 0x2803, 0x2402, 0x36ac, 0x640c, 0x4804, 0x2402, 0x2c8c, 0x4403, 0x2803, 0x2402,
    0x56ad, 0xa42c, 0x680d, 0x2402, 0x4c8d, 0x4403, 0x2803, 0x2402, 0x36ac, 0x640c, 0x4804, 0x2402, 0x2c8c,
    0x4403, 0x2803, 0x2402, 0x76b6, 0xa42c, 0xa82d, 0x2402, 0x6c96, 0x4403, 0x2803, 0x2402, 0x36ac, 0x640c,
    0x4804, 0x2402, 0x2c8c, 0x4403, 0x2803, 0x2402, 0x56ad, 0xa42c, 0x680d, 0x2402, 0x4c8d, 0x4403, 0x2803,
    0x2402, 0x36ac, 0x640c, 0x4804, 0x2402, 0x2c8c, 0x4403, 0x2803, 0x2402};

/********************************************************************************
 * MEL_dec:
 *******************************************************************************/
// this class implementation is borrowed from OpenJPH
class MEL_dec {
 private:
  const uint8_t *buf;
  uint64_t tmp;
  int bits;
  int32_t length;
  bool unstuff;
  int MEL_k;

  int32_t num_runs;
  uint64_t runs;

 public:
  MEL_dec(const uint8_t *Dcup, int32_t Lcup, int32_t Scup)
      : buf(Dcup + Lcup - Scup),
        tmp(0),
        bits(0),
        length(Scup - 1),  // length is the length of MEL+VLC-1
        unstuff(false),
        MEL_k(0),
        num_runs(0),
        runs(0) {
    int num = 4 - static_cast<int>(reinterpret_cast<intptr_t>(buf) & 0x3);
    for (int32_t i = 0; i < num; ++i) {
      uint64_t d = (length > 0) ? *buf : 0xFF;  // if buffer is exhausted, set data to 0xFF
      if (length == 1) {
        d |= 0xF;  // if this is MEL+VLC+1, set LSBs to 0xF (see the spec)
      }
      buf += length-- > 0;  // increment if the end is not reached
      int d_bits = 8 - unstuff;
      tmp        = (tmp << d_bits) | d;
      bits += d_bits;
      unstuff = ((d & 0xFF) == 0xFF);
    }
    tmp <<= (64 - bits);
  }

  inline void read() {
    if (bits > 32) {  // there are enough bits in tmp, return without any reading
      return;
    }

    uint32_t val = 0xFFFFFFFF;  // feed in 0xFF if buffer is exhausted
    if (length > 4) {           // if there is data in the MEL segment
      val = *reinterpret_cast<uint32_t *>(const_cast<uint8_t *>(buf));  // read 32 bits from MEL data
      buf += 4;                                                         // advance pointer
      length -= 4;                                                      // reduce counter
    } else if (length > 0) {
      // 4 or less
      int i = 0;
      while (length > 1) {
        uint32_t v = *buf++;                // read one byte at a time
        uint32_t m = ~(0xFFU << i);         // mask of location
        val        = (val & m) | (v << i);  // put one byte in its correct location
        --length;
        i += 8;
      }
      // length equal to 1
      uint32_t v = *buf++;  // the one before the last is different
      v |= 0xF;             // MEL and VLC segments may be overlapped
      uint32_t m = ~(0xFFU << i);
      val        = (val & m) | (v << i);
      --length;
    } else {
      // error
    }

    // next we unstuff them before adding them to the buffer
    int bits_local =
        32 - unstuff;  // number of bits in val, subtract 1 if the previously read byte requires unstuffing

    // data is unstuffed and accumulated in t
    // bits_local has the number of bits in t
    uint32_t t        = val & 0xFF;
    bool unstuff_flag = ((val & 0xFF) == 0xFF);
    bits_local -= unstuff_flag;
    t = t << (8 - unstuff_flag);

    t |= (val >> 8) & 0xFF;
    unstuff_flag = (((val >> 8) & 0xFF) == 0xFF);
    bits_local -= unstuff_flag;
    t = t << (8 - unstuff_flag);

    t |= (val >> 16) & 0xFF;
    unstuff_flag = (((val >> 16) & 0xFF) == 0xFF);
    bits_local -= unstuff_flag;
    t = t << (8 - unstuff_flag);

    t |= (val >> 24) & 0xFF;
    unstuff = (((val >> 24) & 0xFF) == 0xFF);

    // move to tmp, and push the result all the way up, so we read from the MSB
    tmp |= (static_cast<uint64_t>(t)) << (64 - bits_local - bits);
    bits += bits_local;
  }

  inline void decode() {
    constexpr int32_t MEL_E[13] = {0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 5};
    if (bits < 6) {  // if there are less than 6 bits in tmp then read from the MEL bitstream 6 bits that is
      // the largest decodable MEL codeword.
      read();
    }
    // repeat so long that there is enough decodable bits in tmp, and the runs store is not full
    // (num_runs < 8)
    while (bits >= 6 && num_runs < 8) {
      int32_t eval = MEL_E[MEL_k];
      int32_t run  = 0;
      // The next bit to decode (stored in MSB)
      if (tmp & (1ULL << 63)) {
        // "1" is found
        run = 1 << eval;
        run--;                                        // consecutive runs of 0 events - 1
        MEL_k = ((MEL_k + 1) < 12) ? MEL_k + 1 : 12;  // increment, max is 12
        tmp <<= 1;                                    // consume one bit from tmp
        bits--;
        run <<= 1;  // a stretch of zeros not terminating in one
      } else {
        // "0" is found
        run   = static_cast<int32_t>(tmp >> (63 - eval)) & ((1 << eval) - 1);
        MEL_k = ((MEL_k - 1) > 0) ? MEL_k - 1 : 0;  // decrement, min is 0
        tmp <<= eval + 1;                           // consume eval + 1 bits (max is 6)
        bits -= eval + 1;
        run = (run << 1) + 1;  // a stretch of zeros terminating with one
      }
      eval = num_runs * 7;                             // 7 bits per run
      runs &= ~(static_cast<uint64_t>(0x3F) << eval);  // 6 bits are sufficient
      runs |= (static_cast<uint64_t>(run)) << eval;    // store the value in runs
      num_runs++;                                      // increment count
    }
  }

  inline int32_t get_run() {
    if (num_runs == 0) {  // if no runs, decode more bit from MEL segment
      decode();
    }
    int32_t t = static_cast<int32_t>(runs & 0x7F);  // retrieve one run
    runs >>= 7;                                     // remove the retrieved run
    num_runs--;
    return t;  // return run
  }
};

/********************************************************************************
 * rev_buf:
 *******************************************************************************/
// this class implementation is borrowed from OpenJPH
class rev_buf {
 private:
  uint8_t *buf;
  uint64_t Creg;
  uint32_t bits;
  int32_t length;
  uint32_t unstuff;

 public:
  rev_buf(uint8_t *Dcup, int32_t Lcup, int32_t Scup)
      : buf(Dcup + Lcup - 2), Creg(0), bits(0), length(Scup - 2), unstuff(0) {
    uint32_t d = *buf--;  // read a byte (only use it's half byte)
    Creg       = d >> 4;
    bits       = 4 - ((Creg & 0x07) == 0x07);
    unstuff    = (d | 0x0F) > 0x8f;

    int32_t p = static_cast<int32_t>(reinterpret_cast<intptr_t>(buf) & 0x03);
    //    p &= 0x03;
    int32_t num  = 1 + p;
    int32_t tnum = (num < length) ? num : length;
    for (auto i = 0; i < tnum; ++i) {
      uint64_t d64;
      d64             = *buf--;
      uint32_t d_bits = 8 - static_cast<uint32_t>(unstuff & ((d64 & 0x7F) == 0x7F));
      Creg |= d64 << bits;
      bits += d_bits;
      unstuff = d64 > 0x8F;
    }
    length -= tnum;
    read();
  }

  inline void read() {
    // process 4 bytes at a time
    if (bits > 32) {
      // if there are already more than 32 bits, do nothing to prevent overflow of Creg
      return;
    }
    uint32_t val = 0;
    if (length > 3) {  // Common case; we have at least 4 bytes available
      val = *reinterpret_cast<uint32_t *>(buf - 3);
      buf -= 4;
      length -= 4;
    } else if (length > 0) {  // we have less than 4 bytes
      int i = 24;
      while (length > 0) {
        uint32_t v = *buf--;
        val |= (v << i);
        --length;
        i -= 8;
      }
    } else {
      // error
    }

    // accumulate in tmp, number of bits in tmp are stored in bits
    uint32_t tmp = val >> 24;  // start with the MSB byte
    uint32_t bits_local;

    // test unstuff (previous byte is >0x8F), and this byte is 0x7F
    // Use bitwise & instead of && to avoid short-circuit branches.
    bits_local        = 8 - static_cast<uint32_t>(unstuff & (((val >> 24) & 0x7F) == 0x7F));
    bool unstuff_flag = (val >> 24) > 0x8F;  // this is for the next byte

    tmp |= ((val >> 16) & 0xFF) << bits_local;  // process the next byte
    bits_local += 8 - static_cast<uint32_t>(unstuff_flag & (((val >> 16) & 0x7F) == 0x7F));
    unstuff_flag = ((val >> 16) & 0xFF) > 0x8F;

    tmp |= ((val >> 8) & 0xFF) << bits_local;
    bits_local += 8 - static_cast<uint32_t>(unstuff_flag & (((val >> 8) & 0x7F) == 0x7F));
    unstuff_flag = ((val >> 8) & 0xFF) > 0x8F;

    tmp |= (val & 0xFF) << bits_local;
    bits_local += 8 - static_cast<uint32_t>(unstuff_flag & ((val & 0x7F) == 0x7F));
    unstuff_flag = (val & 0xFF) > 0x8F;

    // now move the read and unstuffed bits into this->Creg
    Creg |= static_cast<uint64_t>(tmp) << bits;
    bits += bits_local;
    unstuff = unstuff_flag;  // this for the next read
  }

  FORCE_INLINE uint32_t fetch() {
    if (bits < 32) {
      read();
      if (bits < 32) {
        read();
      }
    }
    return static_cast<uint32_t>(Creg);
  }

  inline uint32_t advance(uint32_t num_bits) {
    if (num_bits > bits) {
      printf("ERROR: VLC require %d bits but there are %d bits left\n", num_bits, bits);
      throw std::exception();
    }
    Creg >>= num_bits;
    bits -= num_bits;
    return static_cast<uint32_t>(Creg);
  }
};

/********************************************************************************
 * fwd_buf:
 *******************************************************************************/
#if defined(OPENHTJ2K_ENABLE_ARM_NEON)
  #include <arm_neon.h>
// NEON does not provide a version of this function, here is an article about
// some ways to repro the results.
// http://stackoverflow.com/questions/11870910/sse-mm-movemask-epi8-equivalent-method-for-arm-neon
// Creates a 16-bit mask from the most significant bits of the 16 signed or
// unsigned 8-bit integers in a and zero extends the upper bits.
// https://msdn.microsoft.com/en-us/library/vstudio/s090c8fk(v=vs.100).aspx
FORCE_INLINE int aarch64_movemask_epi8(int32x4_t _a) {
  uint8x16_t input     = vreinterpretq_u8_s32(_a);
  uint16x8_t high_bits = vreinterpretq_u16_u8(vshrq_n_u8(input, 7));
  uint32x4_t paired16  = vreinterpretq_u32_u16(vsraq_n_u16(high_bits, high_bits, 7));
  uint64x2_t paired32  = vreinterpretq_u64_u32(vsraq_n_u32(paired16, paired16, 14));
  uint8x16_t paired64  = vreinterpretq_u8_u64(vsraq_n_u64(paired32, paired32, 28));
  return vgetq_lane_u8(paired64, 0) | ((int)vgetq_lane_u8(paired64, 8) << 8);
  //  static const int8_t __attribute__((aligned(16))) xr[8] = {-7, -6, -5, -4, -3, -2, -1, 0};
  //  uint8x8_t mask_and                                     = vdup_n_u8(0x80);
  //  int8x8_t mask_shift                                    = vld1_s8(xr);
  //
  //  uint8x8_t lo = vget_low_u8(input);
  //  uint8x8_t hi = vget_high_u8(input);
  //
  //  lo = vand_u8(lo, mask_and);
  //  lo = vshl_u8(lo, mask_shift);
  //
  //  hi = vand_u8(hi, mask_and);
  //  hi = vshl_u8(hi, mask_shift);
  //
  //  lo = vpadd_u8(lo, lo);
  //  lo = vpadd_u8(lo, lo);
  //  lo = vpadd_u8(lo, lo);
  //
  //  hi = vpadd_u8(hi, hi);
  //  hi = vpadd_u8(hi, hi);
  //  hi = vpadd_u8(hi, hi);
  //
  //  return ((hi[0] << 8) | (lo[0] & 0xFF));
}

  #define aarch64_srli_epi64(a, imm)                                               \
    ({                                                                             \
      int32x4_t ret;                                                               \
      if ((imm) <= 0) {                                                            \
        ret = a;                                                                   \
      } else if ((imm) > 63) {                                                     \
        ret = vdupq_n_s32(0);                                                      \
      } else {                                                                     \
        ret = vreinterpretq_s32_u64(vshrq_n_u64(vreinterpretq_u64_s32(a), (imm))); \
      }                                                                            \
      ret;                                                                         \
    })

  // Shifts the 128 - bit value in a right by imm bytes while shifting in
  // zeros.imm must be an immediate.
  // https://msdn.microsoft.com/en-us/library/305w28yz(v=vs.100).aspx
  // FORCE_INLINE aarch64_srli_si128(__m128i a, __constrange(0,255) int imm)
  #define aarch64_srli_si128(a, imm)                                                         \
    ({                                                                                       \
      int32x4_t ret;                                                                         \
      if ((imm) <= 0) {                                                                      \
        ret = a;                                                                             \
      } else if ((imm) > 15) {                                                               \
        ret = vdupq_n_s32(0);                                                                \
      } else {                                                                               \
        ret = vreinterpretq_s32_s8(vextq_s8(vreinterpretq_s8_s32(a), vdupq_n_s8(0), (imm))); \
      }                                                                                      \
      ret;                                                                                   \
    })

  #define aarch64_slli_epi64(a, imm)                                               \
    ({                                                                             \
      int32x4_t ret;                                                               \
      if ((imm) <= 0) {                                                            \
        ret = a;                                                                   \
      } else if ((imm) > 64) {                                                     \
        ret = vdupq_n_s32(0);                                                      \
      } else {                                                                     \
        ret = vreinterpretq_s32_s64(vshlq_n_s64(vreinterpretq_s64_s32(a), (imm))); \
      }                                                                            \
      ret;                                                                         \
    })

FORCE_INLINE int32x4_t aarch64_sll_epi64(int32x4_t a, int32x4_t b) {
  return vreinterpretq_s32_s64(vshlq_s64(vreinterpretq_s64_s32(a), vreinterpretq_s64_s32(b)));
}

FORCE_INLINE int32x4_t aarch64_srl_epi64(int32x4_t a, uint8_t b) {
  // following 5 lines are problematic with clang!!!
  //  uint64_t tmp[2];
  //  vst1q_u64(tmp, a);
  //  auto vtmp = vld1q_u64(tmp);
  //  vtmp >>= b;
  //  return vtmp;
  return vreinterpretq_s32_s64(vshlq_u64(vreinterpretq_s64_s32(a), vreinterpretq_s64_s32(vdupq_n_s32(-b))));
}

  // Shifts the 128-bit value in a left by imm bytes while shifting in zeros. imm
  // must be an immediate.
  // https://msdn.microsoft.com/en-us/library/34d3k2kt(v=vs.100).aspx
  // FORCE_INLINE __m128i aarch64_slli_si128(__m128i a, __constrange(0,255) int imm)
  #define aarch64_slli_si128(a, imm)                                                              \
    ({                                                                                            \
      int32x4_t ret;                                                                              \
      if ((imm) <= 0) {                                                                           \
        ret = a;                                                                                  \
      } else if ((imm) > 15) {                                                                    \
        ret = vdupq_n_s32(0);                                                                     \
      } else {                                                                                    \
        ret = vreinterpretq_s32_s8(vextq_s8(vdupq_n_s8(0), vreinterpretq_s8_s32(a), 16 - (imm))); \
      }                                                                                           \
      ret;                                                                                        \
    })

  // Extracts the selected signed or unsigned 16-bit integer from a and zero
  // extends.  https://msdn.microsoft.com/en-us/library/6dceta0c(v=vs.100).aspx
  // FORCE_INLINE int aarch64_extract_epi16(__m128i a, __constrange(0,8) int imm)
  #define aarch64_extract_epi16(a, imm)                               \
    ({                                                                \
      (vgetq_lane_s16(vreinterpretq_s16_s32(a), (imm)) & 0x0000ffff); \
    })  // modified from 0x0000ffffUL to suppress compiler warnings

//************************************************************************/
/** @brief State structure for reading and unstuffing of forward-growing
 *         bitstreams; these are: MagSgn and SPP bitstreams
 */
// this class implementation is borrowed from OpenJPH and modified for ARM NEON
template <int X>
class fwd_buf {
 private:
  const uint8_t *data;  //!< pointer to bitstream
  uint8_t tmp[48];      //!< temporary buffer of read data + 16 extra
  uint32_t bits;        //!< number of bits stored in tmp
  uint32_t unstuff;     //!< 1 if a bit needs to be unstuffed from next byte
  int size;             //!< size of data
 public:
  fwd_buf(const uint8_t *data, int size) : data(data), bits(0), unstuff(0), size(size) {
    //    this->data = data;
    vst1q_u8(this->tmp, vdupq_n_u8(0));
    vst1q_u8(this->tmp + 16, vdupq_n_u8(0));
    vst1q_u8(this->tmp + 32, vdupq_n_u8(0));

    //    this->bits    = 0;
    //    this->unstuff = 0;
    //    this->size    = size;

    read();  // read 128 bits more
  }

  //************************************************************************/
  /** @brief Read and unstuffs 16 bytes from forward-growing bitstream
   *
   *  A template is used to accommodate a different requirement for
   *  MagSgn and SPP bitstreams; in particular, when MagSgn bitstream is
   *  consumed, 0xFF's are fed, while when SPP is exhausted 0's are fed in.
   *  X controls this value.
   *
   *  Unstuffing prevent sequences that are more than 0xFF7F from appearing
   *  in the conpressed sequence.  So whenever a value of 0xFF is coded, the
   *  MSB of the next byte is set 0 and must be ignored during decoding.
   *
   *  Reading can go beyond the end of buffer by up to 16 bytes.
   *
   *
   */

  inline void read() {
    assert(this->bits <= 128);

    uint8x16_t offset, val, validity, all_xff;
    val       = vld1q_u8(this->data);
    int bytes = this->size >= 16 ? 16 : this->size;
    validity  = vdupq_n_s8((char)bytes);
    this->data += bytes;
    this->size -= bytes;
    uint32_t bits_local = 128;
    // offset = _mm_set_epi64x(0x0F0E0D0C0B0A0908, 0x0706050403020100);
    int64_t local_tmp[2] = {0x0706050403020100, 0x0F0E0D0C0B0A0908};
    offset               = vreinterpretq_u8_s64(vld1q_s64(local_tmp));
    validity             = vcgtq_s8(validity, offset);
    all_xff              = vdupq_n_s8(-1);
    if (X == 0xFF)  // the compiler should remove this if statement
    {
      auto t = veorq_u8(validity, all_xff);  // complement
      val    = vorrq_u8(t, val);             // fill with 0xFF
    } else if (X == 0)
      val = vandq_u8(validity, val);  // fill with zeros
    else
      assert(0);

    uint16x8_t ff_bytes = vceqq_u8(val, all_xff);

    // movemask: SEE BELOW
    // https://community.arm.com/arm-community-blogs/b/infrastructure-solutions-blog/posts/porting-x86-vector-bitmask-optimizations-to-arm-neon
    uint64_t flags_arm =
        vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(ff_bytes), 4)), 0);  // movemask
    uint32_t next_unstuff_arm = flags_arm >> 63;
    flags_arm <<= 4;  // unstuff following byte
    flags_arm |= this->unstuff << 3;
    flags_arm |= this->unstuff << 2;
    flags_arm |= this->unstuff << 1;
    flags_arm |= this->unstuff;

    while (flags_arm) {  // bit unstuffing occurs on average once every 256 bytes
      // therefore it is not an issue if it is a bit slow
      // here we process 16 bytes
      --bits_local;  // consuming one stuffing bit

      int32_t loc_arm = static_cast<int32_t>(15 - (openhtj2k_arm_clzll(flags_arm) >> 2));
      flags_arm ^= (uint64_t)0xF << (loc_arm << 2);
      uint8x16_t m, t, c;
      t = vdupq_n_s8(static_cast<int8_t>(loc_arm));
      m = vcgtq_s8(offset, t);

      t = vandq_u8(m, val);  // keep bits_local at locations larger than loc
      // c = aarch64_srli_epi64(t, 1);  // 1 bits_local left
      c = vreinterpretq_s32_u64(vshrq_n_u64(vreinterpretq_u64_s32(t), 1));  // 1 bits_local left
      // t = aarch64_srli_si128(t, 8);   // 8 bytes left
      t = vreinterpretq_s32_s8(vextq_s8(vreinterpretq_s8_s32(t), vdupq_n_s8(0), 8));  // 8 bytes left
      // t = aarch64_slli_epi64(t, 63);                                          // keep the MSB only
      t = vreinterpretq_s32_s64(vshlq_n_s64(vreinterpretq_s64_s32(t), 63));  // keep the MSB only
      t = vorrq_u8(t, c);                                                    // combine the above 3 steps

      val = vorrq_u8(t, vbicq_s8(val, m));
    }

    // combine with earlier data
    assert(this->bits >= 0 && this->bits <= 128);
    uint32_t cur_bytes = this->bits >> 3;
    uint32_t cur_bits  = this->bits & 7;
    uint8x16_t b1, b2;
    b1 = aarch64_sll_epi64(val, vdupq_n_s64(cur_bits));

    //    b2 = aarch64_slli_si128(val, 8);  // 8 bytes right
    b2 = vreinterpretq_s32_s8(vextq_s8(vdupq_n_s8(0), vreinterpretq_s8_s32(val), 16 - 8));  // 8 bytes left
    b2 = aarch64_srl_epi64(b2, static_cast<uint8_t>(64 - cur_bits));
    b1 = vorrq_u8(b1, b2);
    b2 = vld1q_u8(tmp + cur_bytes);
    b2 = vorrq_u8(b1, b2);
    vst1q_u8(tmp + cur_bytes, b2);

    uint32_t consumed_bits = bits_local < 128 - cur_bits ? bits_local : 128 - cur_bits;
    cur_bytes              = (this->bits + (uint32_t)consumed_bits + 7) >> 3;  // round up
    // int upper = aarch64_extract_epi16(val, 7);
    int upper = vgetq_lane_s16(vreinterpretq_s16_s32(val), 7) & 0x0000ffff;

    upper >>= consumed_bits - 128 + 16;
    this->tmp[cur_bytes] = (uint8_t)upper;  // copy byte

    this->bits += (uint32_t)bits_local;
    this->unstuff = next_unstuff_arm;  // next unstuff
    assert(this->unstuff == 0 || this->unstuff == 1);
  }

  //************************************************************************/
  /** @brief Consume num_bits bits from the bitstream of fwd_buf
   *
   *  @param [in]  num_bits is the number of bit to consume
   */
  inline void advance(uint32_t num_bits) {
    if (!num_bits) return;
    assert(num_bits > 0 && num_bits <= this->bits && num_bits < 128);
    this->bits -= num_bits;

    auto *p = (this->tmp + ((num_bits >> 3) & 0x18));
    num_bits &= 63;

    uint16x8_t v0, v1, c0, c1, t;
    v0 = vld1q_u8(p);
    v1 = vld1q_u8(p + 16);

    // shift right by num_bits
    c0 = aarch64_srl_epi64(v0, static_cast<uint8_t>(num_bits));
    //    t  = aarch64_srli_si128(v0, 8);
    t  = vreinterpretq_s32_s8(vextq_s8(vreinterpretq_s8_s32(v0), vdupq_n_s8(0), 8));
    t  = aarch64_sll_epi64(t, vdupq_n_s64(64 - num_bits));
    c0 = vorrq_u8(c0, t);
    //    t  = aarch64_slli_si128(v1, 8);
    t  = vreinterpretq_s32_s8(vextq_s8(vdupq_n_s8(0), vreinterpretq_s8_s32(v1), 16 - 8));
    t  = aarch64_sll_epi64(t, vdupq_n_s64(64 - num_bits));
    c0 = vorrq_u8(c0, t);

    vst1q_u8(this->tmp, c0);

    c1 = aarch64_srl_epi64(v1, static_cast<uint8_t>(num_bits));
    //    t  = aarch64_srli_si128(v1, 8);
    t  = vreinterpretq_s32_s8(vextq_s8(vreinterpretq_s8_s32(v1), vdupq_n_s8(0), 8));
    t  = aarch64_sll_epi64(t, vdupq_n_s64(64 - num_bits));
    c1 = vorrq_u8(c1, t);

    vst1q_u8(this->tmp + 16, c1);
  }

  //************************************************************************/
  /** @brief Fetches 32 bits from the fwd_buf bitstream
   *
   *  @param [in]  m is a reference to a vector of m_n bits
   */
  FORCE_INLINE int32x4_t fetch(const int32x4_t &m) {
    if (this->bits <= 128) {
      read();
      if (this->bits <= 128)  // need to test
        read();
    }
    auto t = vld1q_u8(this->tmp);
  #if defined(_MSC_VER)
    int32x4_t msvec, c, v;
    msvec = vsetq_lane_s32(vgetq_lane_s32(t, 0) & 0xFFFFFFFF, msvec, 0);
    c     = aarch64_srl_epi64(t, static_cast<uint8_t>(vgetq_lane_s32(m, 0)));
    v     = vreinterpretq_s32_s8(vextq_s8(vreinterpretq_s8_s32(t), vdupq_n_s8(0), 8));
    v     = aarch64_sll_epi64(v, vdupq_n_s64(64 - vgetq_lane_s32(m, 0)));
    t     = vorrq_u8(c, v);
    msvec = vsetq_lane_s32(vgetq_lane_s32(t, 0) & 0xFFFFFFFF, msvec, 1);
    c     = aarch64_srl_epi64(t, static_cast<uint8_t>(vgetq_lane_s32(m, 1)));
    v     = vreinterpretq_s32_s8(vextq_s8(vreinterpretq_s8_s32(t), vdupq_n_s8(0), 8));
    v     = aarch64_sll_epi64(v, vdupq_n_s64(64 - vgetq_lane_s32(m, 1)));
    t     = vorrq_u8(c, v);
    msvec = vsetq_lane_s32(vgetq_lane_s32(t, 0) & 0xFFFFFFFF, msvec, 2);
    c     = aarch64_srl_epi64(t, static_cast<uint8_t>(vgetq_lane_s32(m, 2)));
    v     = vreinterpretq_s32_s8(vextq_s8(vreinterpretq_s8_s32(t), vdupq_n_s8(0), 8));
    v     = aarch64_sll_epi64(v, vdupq_n_s64(64 - vgetq_lane_s32(m, 2)));
    t     = vorrq_u8(c, v);
    msvec = vsetq_lane_s32(vgetq_lane_s32(t, 0) & 0xFFFFFFFF, msvec, 3);
    advance(vaddvq_u32(m));
    return msvec;
  #else
    //    uint32_t vtmp[4];
    //    vtmp[0] = v128i & 0xFFFFFFFFU;
    //    v128i >>= m[0];
    //    vtmp[1] = v128i & 0xFFFFFFFFU;
    //    v128i >>= m[1];
    //    vtmp[2] = v128i & 0xFFFFFFFFU;
    //    v128i >>= m[2];
    //    vtmp[3] = v128i & 0xFFFFFFFFU;
    //    return vld1q_u32(vtmp);

    __uint128_t v128i = (__uint128_t)t;
    int32x4_t vtmp;
    vtmp[0] = static_cast<int32_t>(v128i & 0xFFFFFFFFU);
    v128i >>= m[0];
    vtmp[1] = static_cast<int32_t>(v128i & 0xFFFFFFFFU);
    v128i >>= m[1];
    vtmp[2] = static_cast<int32_t>(v128i & 0xFFFFFFFFU);
    v128i >>= m[2];
    vtmp[3] = static_cast<int32_t>(v128i & 0xFFFFFFFFU);
    advance(vaddvq_u32(m));
    return vtmp;
  #endif
  }

  //************************************************************************/
  /** @brief Fetches raw 128 bits from the fwd_buf bitstream without
   *         per-lane extraction.  Caller handles bit-level extraction
   *         (e.g. via vqtbl1q_u8).
   */
  FORCE_INLINE uint8x16_t fetch_raw() {
    if (this->bits <= 128) {
      read();
      if (this->bits <= 128)
        read();
    }
    return vld1q_u8(this->tmp);
  }

  //************************************************************************/
  /** @brief Decode 2 quads (8 samples) using 16-bit arithmetic.
   *
   *  Active when pLSB > 16 (mmsbp2 < 16, output fits in int16_t).
   *  Uses vqtbl1q_u8 for batch bit extraction instead of sequential
   *  __uint128_t shifts, providing better ILP on AArch64.
   *
   *  @param [in]     tv0       raw VLC table entry for quad 0
   *  @param [in]     tv1       raw VLC table entry for quad 1
   *  @param [in]     U0        kappa-adjusted U value for quad 0
   *  @param [in]     U1        kappa-adjusted U value for quad 1
   *  @param [in]     pLSB_adj  pLSB - 16
   *  @param [in,out] v_n       4 x int16_t per-column row-1 magnitudes (ORed in)
   *  @return int16x8_t [r0c0,r1c0,r0c1,r1c1,r0c2,r1c2,r0c3,r1c3]
   *          sign at bit 15; expand to int32 via vshll_n_s16(x, 16).
   */
  FORCE_INLINE int16x8_t decode_two_quads_16bit(uint16_t tv0, uint16_t tv1,
                                                  uint16_t U0, uint16_t U1,
                                                  uint8_t pLSB_adj, int16x4_t &v_n) {
    const int16x8_t vone16  = vdupq_n_s16(1);
    const int16x8_t vtwo16  = vdupq_n_s16(2);
    const int16x8_t vzero16 = vdupq_n_s16(0);
    int16x8_t row           = vzero16;

    // Broadcast inf words: tv0 to lanes 0-3, tv1 to lanes 4-7.
    int16x8_t w0 = vcombine_s16(vdup_n_s16(static_cast<int16_t>(tv0)),
                                 vdup_n_s16(static_cast<int16_t>(tv1)));

    // Extract per-sample significance/EMB flags.
    // flag_mask[i] tests bit i of rho (bit 4+i), emb_1 (bit 8+i), emb_k (bit 12+i).
    alignas(16) static const int16_t flag_mask_arr[8] = {
        (int16_t)0x1110, 0x2220, 0x4440, (int16_t)0x8880,
        (int16_t)0x1110, 0x2220, 0x4440, (int16_t)0x8880};
    int16x8_t flags    = vandq_s16(w0, vld1q_s16(flag_mask_arr));
    uint16x8_t insig   = vceqq_s16(flags, vzero16);

    // Early exit if all 8 samples are insignificant.
    if (vmaxvq_u16(vreinterpretq_u16_s16(flags)) == 0) {
      return row;
    }

    // Broadcast U values: U0 to lanes 0-3, U1 to lanes 4-7.
    int16x8_t U_vec = vcombine_s16(vdup_n_s16(static_cast<int16_t>(U0)),
                                    vdup_n_s16(static_cast<int16_t>(U1)));

    // Normalize flags: multiply by {8,4,2,1} to align all samples to
    // e_k @ bit 15, e_1 @ bit 11, rho @ bit 7.
    alignas(16) static const int16_t mul_arr[8] = {8, 4, 2, 1, 8, 4, 2, 1};
    flags = vmulq_s16(flags, vld1q_s16(mul_arr));

    // Compute m_n = U - e_k.  Zero inactive lanes before prefix sum.
    uint16x8_t emb_k = vshrq_n_u16(vreinterpretq_u16_s16(flags), 15);
    int16x8_t m_n    = vsubq_s16(U_vec, vreinterpretq_s16_u16(emb_k));
    m_n              = vbicq_s16(m_n, vreinterpretq_s16_u16(insig));

    // Inclusive prefix sum of m_n (8 x 16-bit).
    int16x8_t inc_sum = m_n;
    inc_sum = vaddq_s16(inc_sum, vextq_s16(vzero16, inc_sum, 7));
    inc_sum = vaddq_s16(inc_sum, vextq_s16(vzero16, inc_sum, 6));
    inc_sum = vaddq_s16(inc_sum, vextq_s16(vzero16, inc_sum, 4));
    int total_mn = vgetq_lane_s16(inc_sum, 7);

    // Fetch raw MagSgn bits (gated on total_mn > 0).
    uint8x16_t ms_raw = vdupq_n_u8(0);
    if (total_mn > 0) {
      ms_raw = this->fetch_raw();
      this->advance(static_cast<uint32_t>(total_mn));
    }

    // Exclusive prefix sum = inclusive shifted right by 1 element.
    int16x8_t ex_sum = vextq_s16(vzero16, inc_sum, 7);

    // Byte-level extraction via vqtbl1q_u8 (NEON equivalent of vpshufb).
    uint16x8_t byte_idx = vshrq_n_u16(vreinterpretq_u16_s16(ex_sum), 3);
    uint16x8_t bit_idx  = vandq_u16(vreinterpretq_u16_s16(ex_sum), vdupq_n_u16(7));

    // Duplicate low byte of each 16-bit byte_idx to both bytes.
    alignas(16) static const uint8_t dup_lo[16] = {
        0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14};
    uint8x16_t bidx = vqtbl1q_u8(vreinterpretq_u8_u16(byte_idx), vld1q_u8(dup_lo));
    // Add [0,1] per 16-bit element to fetch consecutive byte pairs [n, n+1].
    alignas(16) static const uint8_t add_01[16] = {
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
    bidx          = vaddq_u8(bidx, vld1q_u8(add_01));
    uint8x16_t d0 = vqtbl1q_u8(ms_raw, bidx);
    bidx          = vaddq_u8(bidx, vdupq_n_u8(1));
    uint8x16_t d1 = vqtbl1q_u8(ms_raw, bidx);

    // Bit-level alignment: multiply-shift to extract bits within each byte pair.
    // For 16-bit elements, high byte of bit_idx is 0 so vqtbl1q reads table[0]=0xFF;
    // after +0x0101 the high byte wraps to 0x00 giving the correct multiplier.
    alignas(16) static const uint8_t bit_tab[16] = {
        0xFF, 127, 63, 31, 15, 7, 3, 1, 0xFF, 127, 63, 31, 15, 7, 3, 1};
    uint8x16_t bit_shift   = vqtbl1q_u8(vld1q_u8(bit_tab), vreinterpretq_u8_u16(bit_idx));
    uint16x8_t bit_shift16 = vaddq_u16(vreinterpretq_u16_u8(bit_shift), vdupq_n_u16(0x0101));

    uint16x8_t d0_16 = vmulq_u16(vreinterpretq_u16_u8(d0), bit_shift16);
    d0_16            = vshrq_n_u16(d0_16, 8);
    uint16x8_t d1_16 = vmulq_u16(vreinterpretq_u16_u8(d1), bit_shift16);
    d1_16            = vandq_u16(d1_16, vdupq_n_u16(0xFF00));
    uint16x8_t ms_vec = vorrq_u16(d0_16, d1_16);

    // Compute 2^m_n = (2 - e_k) << (U_q - 1).  NEON per-lane variable shift.
    int16x8_t w0_val  = vsubq_s16(vtwo16, vreinterpretq_s16_u16(emb_k));
    int16x8_t Uq_m1   = vsubq_s16(U_vec, vone16);
    uint16x8_t shift_v = vshlq_u16(vreinterpretq_u16_s16(w0_val), Uq_m1);

    // Mask ms_vec to m_n magnitude bits.
    ms_vec = vandq_u16(ms_vec, vsubq_u16(shift_v, vreinterpretq_u16_s16(vone16)));

    // Place e_1 at bit position m_n: OR shift where e_1 flag is set.
    uint16x8_t emb1_absent = vceqq_u16(
        vandq_u16(vreinterpretq_u16_s16(flags), vdupq_n_u16(0x800)), vdupq_n_u16(0));
    ms_vec = vorrq_u16(ms_vec, vbicq_u16(shift_v, emb1_absent));

    // Build final sample: sign at bit 15, magnitude at pLSB_adj - 1.
    uint16x8_t tvn    = vbicq_u16(ms_vec, insig);                         // save for v_n
    uint16x8_t w_sign = vshlq_n_u16(ms_vec, 15);                          // sign bit -> bit 15
    ms_vec            = vorrq_u16(ms_vec, vreinterpretq_u16_s16(vone16));  // bin center
    ms_vec            = vaddq_u16(ms_vec, vreinterpretq_u16_s16(vtwo16));  // + 2
    ms_vec = vshlq_u16(ms_vec, vdupq_n_s16(pLSB_adj - 1));               // runtime shift
    ms_vec = vorrq_u16(ms_vec, w_sign);                                    // sign
    row    = vreinterpretq_s16_u16(vbicq_u16(ms_vec, insig));

    // Update v_n: extract row-1 magnitudes (odd elements) per column.
    int16x4_t tvn_lo = vget_low_s16(vreinterpretq_s16_u16(tvn));
    int16x4_t tvn_hi = vget_high_s16(vreinterpretq_s16_u16(tvn));
    v_n              = vorr_s16(v_n, vuzp2_s16(tvn_lo, tvn_hi));

    return row;
  }
};
#elif defined(OPENHTJ2K_ENABLE_WASM_SIMD)
  #include <wasm_simd128.h>

template <int X>
class fwd_buf {
 private:
  const uint8_t *data;
  uint8_t tmp[48];
  uint32_t bits;
  uint32_t unstuff;
  int size;

  static inline v128_t srli_si128_8(v128_t a) {
    return wasm_i8x16_shuffle(a, wasm_i32x4_const_splat(0),
                              8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23);
  }
  static inline v128_t slli_si128_8(v128_t a) {
    return wasm_i8x16_shuffle(wasm_i32x4_const_splat(0), a,
                              0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23);
  }

 public:
  fwd_buf(const uint8_t *data, int size) : data(data), bits(0), unstuff(0), size(size) {
    wasm_v128_store(this->tmp,      wasm_i32x4_const_splat(0));
    wasm_v128_store(this->tmp + 16, wasm_i32x4_const_splat(0));
    wasm_v128_store(this->tmp + 32, wasm_i32x4_const_splat(0));
    read();
  }

  inline void read() {
    assert(this->bits <= 128);
    v128_t offset, val, validity, all_xff;
    val       = wasm_v128_load(this->data);
    int bytes = this->size >= 16 ? 16 : this->size;
    validity  = wasm_i8x16_splat((char)bytes);
    this->data += bytes;
    this->size -= bytes;
    int bits = 128;
    offset   = wasm_i8x16_const(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    validity = wasm_i8x16_gt(validity, offset);
    all_xff  = wasm_i8x16_const_splat(-1);
    if (X == 0xFF) {
      v128_t t = wasm_v128_xor(validity, all_xff);
      val      = wasm_v128_or(t, val);
    } else if (X == 0)
      val = wasm_v128_and(validity, val);
    else
      assert(0);

    v128_t ff_bytes;
    ff_bytes       = wasm_i8x16_eq(val, all_xff);
    ff_bytes       = wasm_v128_and(ff_bytes, validity);
    uint32_t flags = (uint32_t)wasm_i8x16_bitmask(ff_bytes);
    flags <<= 1;
    uint32_t next_unstuff = flags >> 16;
    flags |= this->unstuff;
    flags &= 0xFFFF;

    while (flags) {
      --bits;
      uint32_t loc = 31 - __builtin_clz(flags);
      flags ^= 1U << loc;
      v128_t m, t, c;
      t = wasm_i8x16_splat((char)loc);
      m = wasm_i8x16_gt(offset, t);
      t = wasm_v128_and(m, val);
      c = wasm_u64x2_shr(t, 1);
      t = srli_si128_8(t);
      t = wasm_i64x2_shl(t, 63);
      t = wasm_v128_or(t, c);
      val = wasm_v128_or(t, wasm_v128_andnot(val, m));
    }

    assert(this->bits >= 0 && this->bits <= 128);
    uint32_t cur_bytes = this->bits >> 3;
    int cur_bits       = this->bits & 7;
    v128_t b1, b2;
    b1 = wasm_i64x2_shl(val, cur_bits);
    if (cur_bits > 0) {
      b2 = slli_si128_8(val);
      b2 = wasm_u64x2_shr(b2, 64 - cur_bits);
      b1 = wasm_v128_or(b1, b2);
    }
    b2 = wasm_v128_load(this->tmp + cur_bytes);
    b2 = wasm_v128_or(b1, b2);
    wasm_v128_store(this->tmp + cur_bytes, b2);

    int consumed_bits = bits < 128 - cur_bits ? bits : 128 - cur_bits;
    cur_bytes         = (this->bits + (uint32_t)consumed_bits + 7) >> 3;
    int upper         = wasm_i16x8_extract_lane(val, 7) & 0xFFFF;
    upper >>= consumed_bits - 128 + 16;
    this->tmp[cur_bytes] = (uint8_t)upper;

    this->bits += (uint32_t)bits;
    this->unstuff = next_unstuff;
    assert(this->unstuff == 0 || this->unstuff == 1);
  }

  FORCE_INLINE void advance(uint32_t num_bits) {
    if (!(num_bits <= this->bits && num_bits < 128)) {
      printf("Value of numbits = %d is out of range.\n", num_bits);
      throw std::exception();
    }
    this->bits -= num_bits;
    v128_t *p = (v128_t *)(this->tmp + ((num_bits >> 3) & 0x18));
    num_bits &= 63;

    v128_t v0, v1, c0, c1, t;
    v0 = wasm_v128_load(p);
    v1 = wasm_v128_load(p + 1);

    c0 = wasm_u64x2_shr(v0, num_bits);
    if (num_bits > 0) {
      t  = srli_si128_8(v0);
      t  = wasm_i64x2_shl(t, 64 - num_bits);
      c0 = wasm_v128_or(c0, t);
      t  = slli_si128_8(v1);
      t  = wasm_i64x2_shl(t, 64 - num_bits);
      c0 = wasm_v128_or(c0, t);
    }
    wasm_v128_store(this->tmp, c0);

    c1 = wasm_u64x2_shr(v1, num_bits);
    if (num_bits > 0) {
      t  = srli_si128_8(v1);
      t  = wasm_i64x2_shl(t, 64 - num_bits);
      c1 = wasm_v128_or(c1, t);
    }
    wasm_v128_store(this->tmp + 16, c1);
  }

  FORCE_INLINE v128_t fetch(const v128_t &m) {
    if (this->bits <= 128) {
      read();
      if (this->bits <= 128)
        read();
    }
    v128_t t = wasm_v128_load(this->tmp);
    uint64_t lo = (uint64_t)wasm_i64x2_extract_lane(t, 0);
    uint64_t hi = (uint64_t)wasm_i64x2_extract_lane(t, 1);
    __uint128_t v128i = ((__uint128_t)hi << 64) | lo;
    v128_t vtmp;
    int32_t m0 = wasm_i32x4_extract_lane(m, 0);
    int32_t m1 = wasm_i32x4_extract_lane(m, 1);
    int32_t m2 = wasm_i32x4_extract_lane(m, 2);
    int32_t m3 = wasm_i32x4_extract_lane(m, 3);
    int32_t r0 = (int32_t)(v128i & 0xFFFFFFFFU); v128i >>= m0;
    int32_t r1 = (int32_t)(v128i & 0xFFFFFFFFU); v128i >>= m1;
    int32_t r2 = (int32_t)(v128i & 0xFFFFFFFFU); v128i >>= m2;
    int32_t r3 = (int32_t)(v128i & 0xFFFFFFFFU);
    vtmp = wasm_i32x4_make(r0, r1, r2, r3);
    advance((uint32_t)(m0 + m1 + m2 + m3));
    return vtmp;
  }

  /** @brief Fetch 16 raw bytes from the current bit position for batch bit extraction. */
  FORCE_INLINE v128_t fetch_raw() {
    if (this->bits <= 128) {
      read();
      if (this->bits <= 128)
        read();
    }
    return wasm_v128_load(this->tmp);
  }

  /** @brief Decode 2 quads (8 samples) using 16-bit arithmetic.
   *  Active when pLSB > 16 (coefficients fit in int16_t).
   *  Uses wasm_i8x16_swizzle for batch bit extraction.
   *  @param tv0,tv1     VLC table entries for quads 0 and 1
   *  @param U0,U1       kappa-adjusted U values (must fit in uint16_t)
   *  @param pLSB_adj    pLSB - 16
   *  @param v_n         4×int16 per-column row-1 magnitudes (ORed in, low 4 lanes only)
   *  @return v128_t     [r0c0,r1c0,r0c1,r1c1,r0c2,r1c2,r0c3,r1c3] as int16
   *                     sign at bit 15; expand to int32 via shl 16 + extend
   */
  FORCE_INLINE v128_t decode_two_quads_16bit_wasm(uint16_t tv0, uint16_t tv1,
                                                    uint16_t U0, uint16_t U1,
                                                    uint8_t pLSB_adj, v128_t &v_n) {
    const v128_t vone16  = wasm_i16x8_const_splat(1);
    const v128_t vtwo16  = wasm_i16x8_const_splat(2);
    const v128_t vzero16 = wasm_i16x8_const_splat(0);
    v128_t row           = vzero16;

    // Broadcast tv0 to lanes 0-3, tv1 to lanes 4-7.
    v128_t w0 = wasm_i16x8_shuffle(wasm_i16x8_splat((int16_t)tv0),
                                    wasm_i16x8_splat((int16_t)tv1), 0, 1, 2, 3, 8, 9, 10, 11);

    // Extract per-sample significance/EMB flags.
    alignas(16) static const int16_t flag_mask_arr[8] = {
        (int16_t)0x1110, 0x2220, 0x4440, (int16_t)0x8880,
        (int16_t)0x1110, 0x2220, 0x4440, (int16_t)0x8880};
    v128_t flags = wasm_v128_and(w0, wasm_v128_load(flag_mask_arr));
    v128_t insig = wasm_i16x8_eq(flags, vzero16);  // all-1 where insignificant

    // Early exit if all 8 samples are insignificant.
    if (!wasm_v128_any_true(flags)) return row;

    // Broadcast U values: U0 to lanes 0-3, U1 to lanes 4-7.
    v128_t U_vec = wasm_i16x8_shuffle(wasm_i16x8_splat((int16_t)U0),
                                       wasm_i16x8_splat((int16_t)U1), 0, 1, 2, 3, 8, 9, 10, 11);

    // Normalize flags: emb_k → bit 15, emb_1 → bit 11, rho → bit 7.
    alignas(16) static const int16_t mul_arr[8] = {8, 4, 2, 1, 8, 4, 2, 1};
    flags = wasm_i16x8_mul(flags, wasm_v128_load(mul_arr));

    // m_n = U - e_k; zero inactive lanes.
    v128_t emb_k = wasm_u16x8_shr(flags, 15);          // 0 or 1
    v128_t m_n   = wasm_i16x8_sub(U_vec, emb_k);
    m_n          = wasm_v128_andnot(m_n, insig);        // zero where insignificant

    // Inclusive prefix sum of m_n (8 × int16).
    v128_t inc_sum = m_n;
    inc_sum = wasm_i16x8_add(inc_sum, wasm_i16x8_shuffle(vzero16, inc_sum, 7, 8, 9, 10, 11, 12, 13, 14));
    inc_sum = wasm_i16x8_add(inc_sum, wasm_i16x8_shuffle(vzero16, inc_sum, 6, 7, 8, 9, 10, 11, 12, 13));
    inc_sum = wasm_i16x8_add(inc_sum, wasm_i16x8_shuffle(vzero16, inc_sum, 4, 5, 6, 7, 8, 9, 10, 11));
    int total_mn = wasm_i16x8_extract_lane(inc_sum, 7);

    // Fetch raw MagSgn bits.
    v128_t ms_raw = vzero16;
    if (total_mn > 0) {
      ms_raw = this->fetch_raw();
      this->advance((uint32_t)total_mn);
    }

    // Exclusive prefix sum (inclusive shifted right by 1 element).
    v128_t ex_sum = wasm_i16x8_shuffle(vzero16, inc_sum, 7, 8, 9, 10, 11, 12, 13, 14);

    // Byte/bit index for each sample's bit offset within ms_raw.
    v128_t byte_idx = wasm_u16x8_shr(ex_sum, 3);
    v128_t bit_idx  = wasm_v128_and(ex_sum, wasm_i16x8_const_splat(7));

    // Duplicate low byte of each 16-bit byte_idx to both bytes of the pair.
    alignas(16) static const uint8_t dup_lo[16] = {
        0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14};
    v128_t bidx = wasm_i8x16_swizzle(byte_idx, wasm_v128_load(dup_lo));
    alignas(16) static const uint8_t add_01[16] = {
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
    bidx       = wasm_i8x16_add(bidx, wasm_v128_load(add_01));
    v128_t d0  = wasm_i8x16_swizzle(ms_raw, bidx);
    bidx       = wasm_i8x16_add(bidx, wasm_i8x16_const_splat(1));
    v128_t d1  = wasm_i8x16_swizzle(ms_raw, bidx);

    // Bit-level alignment: multiply-shift to align bits to bit 0.
    alignas(16) static const uint8_t bit_tab[16] = {
        0xFF, 127, 63, 31, 15, 7, 3, 1, 0xFF, 127, 63, 31, 15, 7, 3, 1};
    v128_t bit_shift   = wasm_i8x16_swizzle(wasm_v128_load(bit_tab), bit_idx);
    v128_t bit_shift16 = wasm_i16x8_add(bit_shift, wasm_i16x8_const_splat(0x0101));

    v128_t d0_16  = wasm_u16x8_shr(wasm_i16x8_mul(d0, bit_shift16), 8);
    v128_t d1_16  = wasm_v128_and(wasm_i16x8_mul(d1, bit_shift16),
                                   wasm_i16x8_const_splat((int16_t)0xFF00));
    v128_t ms_vec = wasm_v128_or(d0_16, d1_16);

    // shift_v = (2 - e_k) << (U - 1): U0 for lanes 0-3, U1 for lanes 4-7.
    v128_t w0_val   = wasm_i16x8_sub(vtwo16, emb_k);
    v128_t shift_lo = wasm_i16x8_shl(w0_val, U0 - 1);
    v128_t shift_hi = wasm_i16x8_shl(w0_val, U1 - 1);
    v128_t shift_v  = wasm_i16x8_shuffle(shift_lo, shift_hi, 0, 1, 2, 3, 12, 13, 14, 15);

    // Mask ms_vec to m_n magnitude bits.
    ms_vec = wasm_v128_and(ms_vec, wasm_i16x8_sub(shift_v, vone16));

    // Place e_1 at bit position m_n.
    v128_t emb1_absent = wasm_i16x8_eq(
        wasm_v128_and(flags, wasm_i16x8_const_splat((int16_t)0x800)), vzero16);
    ms_vec = wasm_v128_or(ms_vec, wasm_v128_andnot(shift_v, emb1_absent));

    // Build final sample: sign at bit 15, magnitude at pLSB_adj-1.
    v128_t tvn    = wasm_v128_andnot(ms_vec, insig);        // save pre-shift for v_n
    v128_t w_sign = wasm_i16x8_shl(ms_vec, 15);             // sign bit → bit 15
    ms_vec        = wasm_v128_or(ms_vec, vone16);            // bin center
    ms_vec        = wasm_i16x8_add(ms_vec, vtwo16);          // +2
    ms_vec        = wasm_i16x8_shl(ms_vec, pLSB_adj - 1);   // runtime shift
    ms_vec        = wasm_v128_or(ms_vec, w_sign);            // apply sign
    row           = wasm_v128_andnot(ms_vec, insig);         // zero inactive

    // Update v_n: odd lanes (1,3,5,7) → low 4 lanes of v_n.
    v128_t tvn_odd = wasm_i16x8_shuffle(tvn, vzero16, 1, 3, 5, 7, 8, 8, 8, 8);
    v_n = wasm_v128_or(v_n, tvn_odd);

    return row;
  }
};
#elif defined(OPENHTJ2K_TRY_AVX2) && defined(__AVX2__)
// https://stackoverflow.com/questions/6996764/fastest-way-to-do-horizontal-sse-vector-sum-or-other-reduction
FORCE_INLINE int32_t hsum_epi32_sse2(__m128i x) {
  __m128i hi64 =
      _mm_unpackhi_epi64(x, x);  // 3-operand non-destructive AVX lets us save a byte without needing a mov
  __m128i sum64 = _mm_add_epi32(hi64, x);
  __m128i hi32  = _mm_shufflelo_epi16(sum64, _MM_SHUFFLE(1, 0, 3, 2));  // Swap the low two elements
  __m128i sum32 = _mm_add_epi32(sum64, hi32);
  return _mm_cvtsi128_si32(sum32);  // SSE2 movd
  // return _mm_extract_epi32(hl, 0);     // SSE4, even though it compiles to movd instead of a literal
  // pextrd r32,xmm,0
}
  #if defined(_MSC_VER)
FORCE_INLINE __m128i mm_bitshift_right(__m128i x, unsigned count) {
  __m128i hi = _mm_srli_si128(x, 8);  // shifted by 8 byte right, take hi 64 bit
  if (count >= 64) return _mm_srli_epi64(hi, count - 64);
  hi = _mm_slli_epi64(hi, 64 - count);

  x = _mm_srli_epi64(x, count);
  return _mm_or_si128(x, hi);
}
  #endif
// this class implementation is borrowed from OpenJPH and modified
//************************************************************************/
/** @brief State structure for reading and unstuffing of forward-growing
 *         bitstreams; these are: MagSgn and SPP bitstreams
 */
template <int X>
class fwd_buf {
 private:
  const uint8_t *data;  //!< pointer to bitstream
  uint8_t tmp[48];      //!< temporary buffer of read data + 16 extra
  uint32_t bits;        //!< number of bits stored in tmp
  uint32_t unstuff;     //!< 1 if a bit needs to be unstuffed from next byte
  int size;             //!< size of data
 public:
  //************************************************************************/
  /** @brief Initialize frwd_struct struct and reads some bytes
   *
   *  @tparam      X is the value fed in when the bitstream is exhausted.
   *               See frwd_read regarding the template
   *  @param [in]  data is a pointer to the start of data
   *  @param [in]  size is the number of byte in the bitstream
   */
  fwd_buf(const uint8_t *data, int size) : data(data), bits(0), unstuff(0), size(size) {
    _mm_storeu_si128((__m128i *)this->tmp, _mm_setzero_si128());
    _mm_storeu_si128((__m128i *)this->tmp + 1, _mm_setzero_si128());
    _mm_storeu_si128((__m128i *)this->tmp + 2, _mm_setzero_si128());

    read();  // read 128 bits more
  }

  //************************************************************************/
  /** @brief Read and unstuffs 16 bytes from forward-growing bitstream
   *
   *  A template is used to accommodate a different requirement for
   *  MagSgn and SPP bitstreams; in particular, when MagSgn bitstream is
   *  consumed, 0xFF's are fed, while when SPP is exhausted 0's are fed in.
   *  X controls this value.
   *
   *  Unstuffing prevent sequences that are more than 0xFF7F from appearing
   *  in the conpressed sequence.  So whenever a value of 0xFF is coded, the
   *  MSB of the next byte is set 0 and must be ignored during decoding.
   *
   *  Reading can go beyond the end of buffer by up to 16 bytes.
   *
   *  @tparam       X is the value fed in when the bitstream is exhausted
   *  @param  [in]  msp is a pointer to frwd_struct structure
   *
   */

  FORCE_INLINE void read() {
    assert(this->bits <= 128);

    __m128i offset, val, validity, all_xff;
    val       = _mm_loadu_si128((__m128i *)this->data);
    int bytes = this->size >= 16 ? 16 : this->size;
    validity  = _mm_set1_epi8((char)bytes);
    this->data += bytes;
    this->size -= bytes;
    int bits = 128;
    offset   = _mm_set_epi64x(0x0F0E0D0C0B0A0908, 0x0706050403020100);
    validity = _mm_cmpgt_epi8(validity, offset);
    all_xff  = _mm_set1_epi8(-1);
    if (X == 0xFF)  // the compiler should remove this if statement
    {
      __m128i t = _mm_xor_si128(validity, all_xff);  // complement
      val       = _mm_or_si128(t, val);              // fill with 0xFF
    } else if (X == 0)
      val = _mm_and_si128(validity, val);  // fill with zeros
    else
      assert(0);

    __m128i ff_bytes;
    ff_bytes       = _mm_cmpeq_epi8(val, all_xff);
    ff_bytes       = _mm_and_si128(ff_bytes, validity);
    uint32_t flags = (uint32_t)_mm_movemask_epi8(ff_bytes);
    flags <<= 1;  // unstuff following byte
    uint32_t next_unstuff = flags >> 16;
    flags |= this->unstuff;
    flags &= 0xFFFF;
    while (flags) {  // bit unstuffing occurs on average once every 256 bytes
      // therefore it is not an issue if it is a bit slow
      // here we process 16 bytes
      --bits;  // consuming one stuffing bit

      uint32_t loc = 31 - count_leading_zeros(flags);
      flags ^= 1U << loc;

      __m128i m, t, c;
      t = _mm_set1_epi8((char)loc);
      m = _mm_cmpgt_epi8(offset, t);

      t = _mm_and_si128(m, val);  // keep bits at locations larger than loc
      c = _mm_srli_epi64(t, 1);   // 1 bits left
      t = _mm_srli_si128(t, 8);   // 8 bytes left
      t = _mm_slli_epi64(t, 63);  // keep the MSB only
      t = _mm_or_si128(t, c);     // combine the above 3 steps

      val = _mm_or_si128(t, _mm_andnot_si128(m, val));
    }

    // combine with earlier data
    assert(this->bits >= 0 && this->bits <= 128);
    uint32_t cur_bytes = this->bits >> 3;
    int cur_bits       = this->bits & 7;
    __m128i b1, b2;
    b1 = _mm_sll_epi64(val, _mm_set1_epi64x(cur_bits));
    b2 = _mm_slli_si128(val, 8);  // 8 bytes right
    b2 = _mm_srl_epi64(b2, _mm_set1_epi64x(64 - cur_bits));
    b1 = _mm_or_si128(b1, b2);
    b2 = _mm_loadu_si128((__m128i *)(this->tmp + cur_bytes));
    b2 = _mm_or_si128(b1, b2);
    _mm_storeu_si128((__m128i *)(this->tmp + cur_bytes), b2);

    int consumed_bits = bits < 128 - cur_bits ? bits : 128 - cur_bits;
    cur_bytes         = (this->bits + (uint32_t)consumed_bits + 7) >> 3;  // round up
    int upper         = _mm_extract_epi16(val, 7);
    upper >>= consumed_bits - 128 + 16;
    this->tmp[cur_bytes] = (uint8_t)upper;  // copy byte

    this->bits += (uint32_t)bits;
    this->unstuff = next_unstuff;  // next unstuff
    assert(this->unstuff == 0 || this->unstuff == 1);
  }

  //************************************************************************/
  /** @brief Consume num_bits bits from the bitstream of frwd_struct
   *
   *  @param [in]  num_bits is the number of bit to consume
   */
  FORCE_INLINE void advance(uint32_t num_bits) {
    // if (!num_bits) return;
    if (!(num_bits <= this->bits && num_bits < 128)) {
      printf("Value of numbits = %d is out of range.\n", num_bits);
      throw std::exception();
    }
    this->bits -= num_bits;

    __m128i *p = (__m128i *)(this->tmp + ((num_bits >> 3) & 0x18));
    num_bits &= 63;

    __m128i v0, v1, c0, c1, t;
    v0 = _mm_loadu_si128(p);
    v1 = _mm_loadu_si128(p + 1);

    // shift right by num_bits
    c0 = _mm_srli_epi64(v0, static_cast<int32_t>(num_bits));
    t  = _mm_srli_si128(v0, 8);
    t  = _mm_slli_epi64(t, 64 - static_cast<int32_t>(num_bits));
    c0 = _mm_or_si128(c0, t);
    t  = _mm_slli_si128(v1, 8);
    t  = _mm_slli_epi64(t, 64 - static_cast<int32_t>(num_bits));
    c0 = _mm_or_si128(c0, t);

    _mm_storeu_si128((__m128i *)this->tmp, c0);

    c1 = _mm_srli_epi64(v1, static_cast<int32_t>(num_bits));
    t  = _mm_srli_si128(v1, 8);
    t  = _mm_slli_epi64(t, 64 - static_cast<int32_t>(num_bits));
    c1 = _mm_or_si128(c1, t);

    _mm_storeu_si128((__m128i *)this->tmp + 1, c1);
  }

  //************************************************************************/
  /** @brief Fetches 32 bits from the frwd_struct bitstream
   *
   *  @tparam      X is the value fed in when the bitstream is exhausted.
   *               See frwd_read regarding the template
   */
  FORCE_INLINE __m128i fetch() {
    if (this->bits <= 128) {
      read();
      if (this->bits <= 128)  // need to test
        read();
    }
    const __m128i t = _mm_loadu_si128((__m128i *)this->tmp);

    return t;
    //   //_mm_extract_epi32(m, 0)
    //   const uint32_t m0 = static_cast<uint32_t>(_mm_cvtsi128_si32(m));
    //   //_mm_extract_epi32(m, 1)
    //   const uint32_t m1 = m0 + static_cast<uint32_t>(_mm_cvtsi128_si32(_mm_srli_si128(m, 4)));
    //   //_mm_extract_epi32(m, 2)
    //   const uint32_t m2 = m1 + static_cast<uint32_t>(_mm_cvtsi128_si32(_mm_srli_si128(m, 8)));
    //   //_mm_extract_epi32(m, 3)
    //   const uint32_t m3 = m2 + static_cast<uint32_t>(_mm_cvtsi128_si32(_mm_srli_si128(m, 12)));

    //   uint32_t vtmp[4];
    // #if defined(_MSC_VER)
    //   vtmp[0] = _mm_extract_epi32(t, 0);
    //   vtmp[1] = _mm_extract_epi32(mm_bitshift_right(t, m0), 0);
    //   vtmp[2] = _mm_extract_epi32(mm_bitshift_right(t, m1), 0);
    //   vtmp[3] = _mm_extract_epi32(mm_bitshift_right(t, m2), 0);
    // #else
    //   const __uint128_t v128i = (__uint128_t)t;

    //   vtmp[0] = v128i & 0xFFFFFFFFU;
    //   vtmp[1] = (v128i >> m0) & 0xFFFFFFFFU;
    //   vtmp[2] = (v128i >> m1) & 0xFFFFFFFFU;
    //   vtmp[3] = (v128i >> m2) & 0xFFFFFFFFU;
    // #endif
    //   advance(m3);
    //   return *(__m128i *)vtmp;
  }

  template <int N>
  FORCE_INLINE __m128i decode_one_quad(__m128i qinf, __m128i U_q, uint8_t pLSB, __m128i &v_n) {
    const __m128i vone = _mm_set1_epi32(1);
    __m128i mu_n       = _mm_setzero_si128();
    __m128i w0         = _mm_shuffle_epi32(qinf, _MM_SHUFFLE(N, N, N, N));
    __m128i flags      = _mm_and_si128(w0, _mm_set_epi32(0x8880, 0x4440, 0x2220, 0x1110));
    __m128i insig      = _mm_cmpeq_epi32(flags, _mm_setzero_si128());
    if (_mm_movemask_epi8(insig) != 0xFFFF)  // are all insignificant?
    {
      flags          = _mm_mullo_epi16(flags, _mm_set_epi16(1, 1, 2, 2, 4, 4, 8, 8));
      w0             = _mm_srli_epi32(flags, 15);  // emb_k
      U_q            = _mm_shuffle_epi32(U_q, _MM_SHUFFLE(N, N, N, N));
      __m128i m_n    = _mm_sub_epi32(U_q, w0);
      m_n            = _mm_andnot_si128(insig, m_n);
      w0             = _mm_and_si128(_mm_srli_epi32(flags, 11), vone);  // emb_1
      __m128i mask   = _mm_sub_epi32(_mm_sllv_epi32(vone, m_n), vone);
      __m128i ms_vec = this->fetch();

      /* */
      // find cumulative sums to find at which bit in ms_vec the sample starts
      __m128i inc_sum = m_n;  // inclusive scan
      inc_sum         = _mm_add_epi32(inc_sum, _mm_bslli_si128(inc_sum, 4));
      inc_sum         = _mm_add_epi32(inc_sum, _mm_bslli_si128(inc_sum, 8));
      int total_mn    = _mm_extract_epi16(inc_sum, 6);
      __m128i ex_sum  = _mm_bslli_si128(inc_sum, 4);  // exclusive scan

      // find the starting byte and starting bit
      __m128i byte_idx = _mm_srli_epi32(ex_sum, 3);
      __m128i bit_idx  = _mm_and_si128(ex_sum, _mm_set1_epi32(7));
      byte_idx = _mm_shuffle_epi8(byte_idx, _mm_set_epi32(0x0C0C0C0C, 0x08080808, 0x04040404, 0x00000000));
      byte_idx = _mm_add_epi32(byte_idx, _mm_set1_epi32(0x03020100));
      __m128i d0 = _mm_shuffle_epi8(ms_vec, byte_idx);
      byte_idx   = _mm_add_epi32(byte_idx, _mm_set1_epi32(0x01010101));
      __m128i d1 = _mm_shuffle_epi8(ms_vec, byte_idx);

      // shift samples values to correct location
      bit_idx           = _mm_or_si128(bit_idx, _mm_slli_epi32(bit_idx, 16));
      __m128i bit_shift = _mm_shuffle_epi8(
          _mm_set_epi8(1, 3, 7, 15, 31, 63, 127, -1, 1, 3, 7, 15, 31, 63, 127, -1), bit_idx);
      bit_shift = _mm_add_epi16(bit_shift, _mm_set1_epi16(0x0101));
      d0        = _mm_mullo_epi16(d0, bit_shift);
      d0        = _mm_srli_epi16(d0, 8);  // we should have 8 bits in the LSB
      d1        = _mm_mullo_epi16(d1, bit_shift);
      d1        = _mm_and_si128(d1, _mm_set1_epi32((int32_t)0xFF00FF00));  // 8 in MSB
      ms_vec    = _mm_or_si128(d0, d1);
      /* */

      ms_vec = _mm_and_si128(ms_vec, mask);
      ms_vec = _mm_or_si128(ms_vec, _mm_sllv_epi32(w0, m_n));  // v = 2(mu-1) + sign (0 or 1)
      mu_n   = _mm_add_epi32(ms_vec, _mm_set1_epi32(2));       // 2(mu-1) + sign + 2 = 2mu + sign
      // Add center bin (would be used for lossy and truncated lossless codestreams)
      mu_n = _mm_or_si128(mu_n, vone);  // This cancels the effect of a sign bit in LSB
      mu_n = _mm_slli_epi32(mu_n, pLSB - 1);
      mu_n = _mm_or_si128(mu_n, _mm_slli_epi32(ms_vec, 31));
      mu_n = _mm_andnot_si128(insig, mu_n);

      w0 = ms_vec;
      if (N == 0) {
        w0 = _mm_shuffle_epi8(w0, _mm_set_epi32(-1, -1, 0x0F0E0D0C, 0x07060504));
      } else if (N == 1) {
        w0 = _mm_shuffle_epi8(w0, _mm_set_epi32(0x0F0E0D0C, 0x07060504, -1, -1));
      }
      v_n = _mm_or_si128(v_n, w0);

      if (total_mn) {
        this->advance(static_cast<uint32_t>(total_mn));
      }
    }
    return mu_n;
  }

  // ── AVX2 256-bit two-quad decoder ─────────────────────────────────────────
  // Processes quads 0 and 1 simultaneously:
  //   qinf256 lower 128 = qinf[0] broadcast (quad 0 context)
  //   qinf256 upper 128 = qinf[1] broadcast (quad 1 context)
  //   U_q256            = _mm256_srli_epi32(qinf256, 16)  (or kappa-adjusted)
  // Returns __m256i: lower 128 = mu for quad 0, upper 128 = mu for quad 1.
  // v_n is updated with the magnitude-bit positions of both quads.
  // Fetches and advances the MagSgn bit-stream sequentially (one call per quad).
#if defined(__AVX2__)
  FORCE_INLINE __m256i decode_two_quads(__m256i qinf256, __m256i U_q256, uint8_t pLSB,
                                        __m128i &v_n) {
    const __m256i vone256 = _mm256_set1_epi32(1);
    __m256i row256        = _mm256_setzero_si256();

    // Significance flags for all 8 samples (both quads) at once.
    __m256i flags = _mm256_and_si256(
        qinf256, _mm256_set_epi32(0x8880, 0x4440, 0x2220, 0x1110,
                                   0x8880, 0x4440, 0x2220, 0x1110));
    __m256i insig = _mm256_cmpeq_epi32(flags, _mm256_setzero_si256());

    if ((uint32_t)_mm256_movemask_epi8(insig) != 0xFFFFFFFFu) {
      flags = _mm256_mullo_epi16(
          flags, _mm256_set_epi16(1, 1, 2, 2, 4, 4, 8, 8, 1, 1, 2, 2, 4, 4, 8, 8));

      __m256i emb_k = _mm256_srli_epi32(flags, 15);
      __m256i m_n   = _mm256_andnot_si256(insig, _mm256_sub_epi32(U_q256, emb_k));
      __m256i emb_1 = _mm256_and_si256(_mm256_srli_epi32(flags, 11), vone256);

      // Prefix sum within each 128-bit lane (quad-local — _mm256_bslli_epi128
      // shifts each 128-bit lane independently, which is exactly what we need).
      __m256i mask    = _mm256_sub_epi32(_mm256_sllv_epi32(vone256, m_n), vone256);
      __m256i inc_sum = m_n;
      inc_sum         = _mm256_add_epi32(inc_sum, _mm256_bslli_epi128(inc_sum, 4));
      inc_sum         = _mm256_add_epi32(inc_sum, _mm256_bslli_epi128(inc_sum, 8));
      int total_mn0   = _mm256_extract_epi16(inc_sum, 6);   // quad 0 total bits
      int total_mn1   = _mm256_extract_epi16(inc_sum, 14);  // quad 1 total bits

      // Fetch MagSgn data sequentially; advance quad 0 before fetching quad 1.
      __m128i ms_vec0 = this->fetch();
      if (total_mn0) this->advance(static_cast<uint32_t>(total_mn0));
      __m128i ms_vec1 = this->fetch();
      if (total_mn1) this->advance(static_cast<uint32_t>(total_mn1));

      // Combine into 256-bit: lower lane = quad 0 data, upper lane = quad 1.
      __m256i ms_vec = _mm256_set_m128i(ms_vec1, ms_vec0);

      __m256i ex_sum = _mm256_bslli_epi128(inc_sum, 4);  // exclusive scan

      // Extract each sample's magnitude bits from its respective fetch window.
      // vpshufb (_mm256_shuffle_epi8) operates within each 128-bit lane, so
      // quad 0 reads from ms_vec0 and quad 1 from ms_vec1 independently.
      __m256i byte_idx = _mm256_srli_epi32(ex_sum, 3);
      __m256i bit_idx  = _mm256_and_si256(ex_sum, _mm256_set1_epi32(7));
      byte_idx         = _mm256_shuffle_epi8(
          byte_idx, _mm256_set_epi32(0x0C0C0C0C, 0x08080808, 0x04040404, 0x00000000,
                                      0x0C0C0C0C, 0x08080808, 0x04040404, 0x00000000));
      byte_idx      = _mm256_add_epi32(byte_idx, _mm256_set1_epi32(0x03020100));
      __m256i d0    = _mm256_shuffle_epi8(ms_vec, byte_idx);
      byte_idx      = _mm256_add_epi32(byte_idx, _mm256_set1_epi32(0x01010101));
      __m256i d1    = _mm256_shuffle_epi8(ms_vec, byte_idx);

      bit_idx           = _mm256_or_si256(bit_idx, _mm256_slli_epi32(bit_idx, 16));
      __m128i tab128    = _mm_set_epi8(1, 3, 7, 15, 31, 63, 127, -1, 1, 3, 7, 15, 31, 63, 127, -1);
      __m256i bit_shift = _mm256_shuffle_epi8(_mm256_set_m128i(tab128, tab128), bit_idx);
      bit_shift         = _mm256_add_epi16(bit_shift, _mm256_set1_epi16(0x0101));
      d0                = _mm256_mullo_epi16(d0, bit_shift);
      d0                = _mm256_srli_epi16(d0, 8);
      d1                = _mm256_mullo_epi16(d1, bit_shift);
      d1                = _mm256_and_si256(d1, _mm256_set1_epi32((int32_t)0xFF00FF00));
      ms_vec            = _mm256_or_si256(d0, d1);

      ms_vec = _mm256_and_si256(ms_vec, mask);
      ms_vec = _mm256_or_si256(ms_vec, _mm256_sllv_epi32(emb_1, m_n));

      __m256i mu = _mm256_add_epi32(ms_vec, _mm256_set1_epi32(2));
      mu         = _mm256_or_si256(mu, vone256);
      mu         = _mm256_slli_epi32(mu, pLSB - 1);
      mu         = _mm256_or_si256(mu, _mm256_slli_epi32(ms_vec, 31));
      row256     = _mm256_andnot_si256(insig, mu);

      // Update v_n: for quad 0 (N=0) place ms_vec lanes 1,3 into v_n[0..7];
      //             for quad 1 (N=1) place ms_vec lanes 1,3 into v_n[8..15].
      // Uses a single 256-bit vpshufb with per-lane shuffles.
      __m256i tvn = _mm256_andnot_si256(insig, ms_vec);
      tvn         = _mm256_shuffle_epi8(
          tvn, _mm256_set_epi8(
                   // Upper lane (quad 1, N=1): lanes 1,3 → offsets 8..15
                   0x0F, 0x0E, 0x0D, 0x0C, 0x07, 0x06, 0x05, 0x04,
                   -1, -1, -1, -1, -1, -1, -1, -1,
                   // Lower lane (quad 0, N=0): lanes 1,3 → offsets 0..7
                   -1, -1, -1, -1, -1, -1, -1, -1,
                   0x0F, 0x0E, 0x0D, 0x0C, 0x07, 0x06, 0x05, 0x04));
      v_n = _mm_or_si128(v_n, _mm_or_si128(_mm256_castsi256_si128(tvn),
                                            _mm256_extracti128_si256(tvn, 1)));
    }
    return row256;
  }

  // ── AVX2 256-bit four-quad decoder (16-bit arithmetic) ────────────────────
  // Processes quads 0-3 simultaneously using 16-bit lanes.
  // Requires pLSB_adj = pLSB - 16 so the output fits in int16_t.
  // Active when pLSB > 16 (i.e. mmsbp2 = 32 - pLSB < 16).
  //
  // inf_u_q: 8 × uint16_t = [inf0, u0, inf1, u1, inf2, u2, inf3, u3]
  // U_q:     4 × uint32_t  = [U0, U1, U2, U3] (already kappa-adjusted)
  // Returns __m256i of 16 × int16_t: interleaved row0/row1 per quad column.
  // Sign bit is in bit 15 of each int16_t.
  // Caller must expand to int32_t via zero-extend then slli_epi32(x, 16).
  FORCE_INLINE __m256i decode_four_quads(__m128i inf_u_q, __m128i U_q, uint8_t pLSB_adj,
                                          __m128i &v_n) {
    const __m256i vone16 = _mm256_set1_epi16(1);
    const __m256i vtwo16 = _mm256_set1_epi16(2);
    __m256i row256       = _mm256_setzero_si256();

    // Broadcast each quad's inf word to all 4 sample slots in its 64-bit group.
    // inf_u_q = [inf0(16b), u0(16b), inf1(16b), u1(16b), inf2(16b), u2(16b), inf3(16b), u3(16b)]
    // Step 1: duplicate each inf to both 16-bit positions of its 32-bit slot.
    __m128i ddd = _mm_shuffle_epi8(
        inf_u_q, _mm_set_epi32(0x0D0C0D0C, 0x09080908, 0x05040504, 0x01000100));
    // ddd (8 × 16-bit): [inf0,inf0, inf1,inf1, inf2,inf2, inf3,inf3]
    // Step 2: broadcast each 32-bit pair to two 32-bit slots in the 256-bit register.
    __m256i w0 = _mm256_permutevar8x32_epi32(
        _mm256_castsi128_si256(ddd), _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3));
    // w0 (16 × 16-bit): [inf0 ×4, inf1 ×4, inf2 ×4, inf3 ×4]

    // Extract significance/EMB flags per sample.
    __m256i flags = _mm256_and_si256(
        w0, _mm256_set_epi16((int16_t)0x8880, 0x4440, 0x2220, 0x1110,
                              (int16_t)0x8880, 0x4440, 0x2220, 0x1110,
                              (int16_t)0x8880, 0x4440, 0x2220, 0x1110,
                              (int16_t)0x8880, 0x4440, 0x2220, 0x1110));
    __m256i insig = _mm256_cmpeq_epi16(flags, _mm256_setzero_si256());

    if ((uint32_t)_mm256_movemask_epi8(insig) != 0xFFFFFFFFu) {
      // Broadcast each U_q value (uint32_t → uint16_t pair) to 4 samples per quad.
      ddd          = _mm_or_si128(_mm_bslli_si128(U_q, 2), U_q);
      __m256i U_q_avx = _mm256_permutevar8x32_epi32(
          _mm256_castsi128_si256(ddd), _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3));
      // U_q_avx (16 × 16-bit): [U0 ×4, U1 ×4, U2 ×4, U3 ×4]

      flags = _mm256_mullo_epi16(
          flags, _mm256_set_epi16(1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8));
      // After mullo: e_k at bit 15, e_1 at bit 11, rho at bit 7.

      __m256i emb_k = _mm256_srli_epi16(flags, 15);
      __m256i m_n   = _mm256_sub_epi16(U_q_avx, emb_k);
      m_n           = _mm256_andnot_si256(insig, m_n);

      // Inclusive prefix sum of m_n within each 128-bit lane (= one quad-pair).
      __m256i inc_sum = m_n;
      inc_sum = _mm256_add_epi16(inc_sum, _mm256_bslli_epi128(inc_sum, 2));
      inc_sum = _mm256_add_epi16(inc_sum, _mm256_bslli_epi128(inc_sum, 4));
      inc_sum = _mm256_add_epi16(inc_sum, _mm256_bslli_epi128(inc_sum, 8));
      int total_mn0 = _mm256_extract_epi16(inc_sum, 7);   // quads 0+1 total bits
      int total_mn1 = _mm256_extract_epi16(inc_sum, 15);  // quads 2+3 total bits

      __m128i ms_vec0 = _mm_setzero_si128();
      __m128i ms_vec1 = _mm_setzero_si128();
      if (total_mn0 > 0) {
        ms_vec0 = this->fetch();
        this->advance(static_cast<uint32_t>(total_mn0));
      }
      if (total_mn1 > 0) {
        ms_vec1 = this->fetch();
        this->advance(static_cast<uint32_t>(total_mn1));
      }
      __m256i ms_vec = _mm256_inserti128_si256(_mm256_castsi128_si256(ms_vec0), ms_vec1, 1);

      __m256i ex_sum = _mm256_bslli_epi128(inc_sum, 2);  // exclusive scan

      // Extract each sample's magnitude bits from its 128-bit fetch window.
      __m256i byte_idx = _mm256_srli_epi16(ex_sum, 3);
      __m256i bit_idx  = _mm256_and_si256(ex_sum, _mm256_set1_epi16(7));
      byte_idx         = _mm256_shuffle_epi8(
          byte_idx, _mm256_set_epi16(0x0E0E, 0x0C0C, 0x0A0A, 0x0808, 0x0606, 0x0404,
                                      0x0202, 0x0000, 0x0E0E, 0x0C0C, 0x0A0A, 0x0808,
                                      0x0606, 0x0404, 0x0202, 0x0000));
      byte_idx      = _mm256_add_epi16(byte_idx, _mm256_set1_epi16(0x0100));
      __m256i d0    = _mm256_shuffle_epi8(ms_vec, byte_idx);
      byte_idx      = _mm256_add_epi16(byte_idx, _mm256_set1_epi16(0x0101));
      __m256i d1    = _mm256_shuffle_epi8(ms_vec, byte_idx);

      // For 16-bit elements, bit_idx high byte must stay 0 so vpshufb maps it to
      // table[0]=0xFF; after +0x0101 the high byte wraps to 0x00, giving the correct
      // single-byte multiplier for mullo_epi16.  (The 32-bit decode_two_quads path
      // duplicates across 16-bit halves of a 32-bit element, which is different.)
      __m256i bit_shift = _mm256_shuffle_epi8(
          _mm256_set_epi8(1, 3, 7, 15, 31, 63, 127, -1, 1, 3, 7, 15, 31, 63, 127, -1,
                          1, 3, 7, 15, 31, 63, 127, -1, 1, 3, 7, 15, 31, 63, 127, -1),
          bit_idx);
      bit_shift = _mm256_add_epi16(bit_shift, _mm256_set1_epi16(0x0101));
      d0        = _mm256_mullo_epi16(d0, bit_shift);
      d0        = _mm256_srli_epi16(d0, 8);
      d1        = _mm256_mullo_epi16(d1, bit_shift);
      d1        = _mm256_and_si256(d1, _mm256_set1_epi16((int16_t)0xFF00));
      ms_vec    = _mm256_or_si256(d0, d1);

      // Compute mask = 2^m_n - 1 and place e_1 at bit position m_n.
      // AVX2 has no _mm256_sllv_epi16; split into four _mm_sll_epi16 calls,
      // one per quad (each quad has a uniform shift = U_q - 1).
      // w0 = (2 - e_k): 1 when e_k=1, 2 when e_k=0.
      w0              = _mm256_sub_epi16(vtwo16, emb_k);
      __m256i Uq_m1   = _mm256_sub_epi16(U_q_avx, vone16);
      // Shift count for even quads (0 and 2) within each 128-bit lane.
      __m256i Uq_evn  = _mm256_and_si256(Uq_m1, _mm256_set_epi32(0, 0, 0, 0x1F, 0, 0, 0, 0x1F));
      // Shift count for odd quads (1 and 3) — move to element 0 of each lane.
      __m256i Uq_odd  = _mm256_bsrli_epi128(Uq_m1, 14);
      __m256i t_evn   = _mm256_and_si256(w0, _mm256_set_epi64x(0, -1, 0, -1));
      __m256i t_odd   = _mm256_and_si256(w0, _mm256_set_epi64x(-1, 0, -1, 0));
      {  // no _mm256_sllv_epi16 in AVX2 — use four _mm_sll_epi16 calls instead
        __m128i lo, hi;
        lo    = _mm_sll_epi16(_mm256_castsi256_si128(t_evn), _mm256_castsi256_si128(Uq_evn));
        hi    = _mm_sll_epi16(_mm256_extracti128_si256(t_evn, 1),
                               _mm256_extracti128_si256(Uq_evn, 1));
        t_evn = _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
        lo    = _mm_sll_epi16(_mm256_castsi256_si128(t_odd), _mm256_castsi256_si128(Uq_odd));
        hi    = _mm_sll_epi16(_mm256_extracti128_si256(t_odd, 1),
                               _mm256_extracti128_si256(Uq_odd, 1));
        t_odd = _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
      }
      __m256i shift = _mm256_or_si256(t_evn, t_odd);  // = (2 - e_k) << (U_q - 1) = 2^m_n

      ms_vec = _mm256_and_si256(ms_vec, _mm256_sub_epi16(shift, vone16));  // m_n magnitude bits

      // Place e_1 at bit position m_n.
      __m256i emb1_mask = _mm256_cmpeq_epi16(
          _mm256_and_si256(flags, _mm256_set1_epi16(0x800)), _mm256_setzero_si256());
      ms_vec = _mm256_or_si256(ms_vec, _mm256_andnot_si256(emb1_mask, shift));

      // Build decoded sample: sign at bit 15, magnitude shifted by (pLSB_adj - 1).
      __m256i tvn    = ms_vec;                                 // save for v_n update (before |=1)
      __m256i w_sign = _mm256_slli_epi16(ms_vec, 15);        // sign bit → bit 15
      ms_vec         = _mm256_or_si256(ms_vec, vone16);       // bin center
      ms_vec         = _mm256_add_epi16(ms_vec, vtwo16);      // + 2
      ms_vec         = _mm256_slli_epi16(ms_vec, pLSB_adj - 1);
      ms_vec         = _mm256_or_si256(ms_vec, w_sign);       // sign
      row256         = _mm256_andnot_si256(insig, ms_vec);

      // Update v_n (8 × int16_t): pairwise OR of row-0 and row-1 magnitudes per column.
      // Gather odd-indexed 16-bit elements (row 1 only) into lower 64 bits of each 128-bit lane.
      tvn = _mm256_andnot_si256(insig, tvn);
      __m256i vn256 = _mm256_shuffle_epi8(
          tvn, _mm256_set_epi8(-1,-1,-1,-1,-1,-1,-1,-1,
                                0x0F,0x0E,0x0B,0x0A,0x07,0x06,0x03,0x02,
                                -1,-1,-1,-1,-1,-1,-1,-1,
                                0x0F,0x0E,0x0B,0x0A,0x07,0x06,0x03,0x02));
      // Combine: lower 64 bits of each lane into the lower 128 bits.
      vn256 = _mm256_permute4x64_epi64(vn256, _MM_SHUFFLE(2, 0, 2, 0));
      v_n   = _mm_or_si128(v_n, _mm256_castsi256_si128(vn256));
    }
    return row256;
  }
#endif  // defined(__AVX2__)
};
#else
template <int X>
class fwd_buf {
 private:
  const uint8_t *buf;
  uint64_t Creg;
  uint32_t bits;
  uint32_t unstuff;
  int32_t length;
  uint32_t pos;

 public:
  fwd_buf(const uint8_t *Dcup, int32_t Pcup)
      : buf(Dcup), Creg(0), bits(0), unstuff(0), length(Pcup), pos(0) {
    // for alignment
    auto p = reinterpret_cast<intptr_t>(buf);
    p &= 0x03;
    auto num = 4 - p;
    for (auto i = 0; i < num; ++i) {
      uint64_t d;
      if (length-- > 0) {
        d = *buf++;
        pos++;
      } else {
        d = (uint64_t)X;
      }
      Creg |= (d << bits);
      bits += 8 - unstuff;
      unstuff = ((d & 0xFF) == 0xFF);  // bit-unstuffing for next byte
    }
    read();
  }

  inline void read() {
    if (bits > 32) {
      printf("ERROR: in MagSgn reading\n");
      throw std::exception();
    }

    uint32_t val = 0;
    if (length > 3) {
      val = *(uint32_t *)(buf);
      buf += 4;
      pos += 4;
      length -= 4;
    } else if (length > 0) {
      int i = 0;
      val   = (X != 0) ? 0xFFFFFFFFU : 0;
      while (length > 0) {
        uint32_t v = *buf++;
        pos++;
        uint32_t m = ~(0xFFU << i);
        val        = (val & m) | (v << i);  // put one byte in its correct location
        --length;
        i += 8;
      }
    } else {
      val = (X != 0) ? 0xFFFFFFFFU : 0;
    }

    // we accumulate in t and keep a count of the number of bits_local in bits_local
    uint32_t bits_local   = 8 - unstuff;
    uint32_t t            = val & 0xFF;
    uint32_t unstuff_flag = ((val & 0xFF) == 0xFF);  // Do we need unstuffing next?

    t |= ((val >> 8) & 0xFF) << bits_local;
    bits_local += 8 - unstuff_flag;
    unstuff_flag = (((val >> 8) & 0xFF) == 0xFF);

    t |= ((val >> 16) & 0xFF) << bits_local;
    bits_local += 8 - unstuff_flag;
    unstuff_flag = (((val >> 16) & 0xFF) == 0xFF);

    t |= ((val >> 24) & 0xFF) << bits_local;
    bits_local += 8 - unstuff_flag;
    unstuff = (((val >> 24) & 0xFF) == 0xFF);  // for next byte

    Creg |= ((uint64_t)t) << bits;  // move data to this->tmp
    bits += bits_local;
  }

  inline void advance(uint32_t n) {
    if (n > bits) {
      printf("ERROR: illegal attempt to advance %d bits but there are %d bits left in MagSgn advance\n", n,
             bits);
  #if defined(__clang__)
      // the following code might be problem with GCC, TODO: to be investigated
      throw std::exception();
  #endif
    }
    Creg >>= n;  // consume n bits
    bits -= n;
  }

  inline uint32_t fetch() {
    if (bits < 32) {
      read();
      if (bits < 32)  // need to test
        read();
    }
    return (uint32_t)Creg;
  }
};
#endif

/********************************************************************************
 * state_MS: state class for MagSgn decoding
 *******************************************************************************/
// class state_MS_dec {
//  private:
//   uint32_t pos;
//   uint8_t bits;
//   uint8_t tmp;
//   uint8_t last;
//   const uint8_t *buf;
//   const uint32_t length;
//   uint64_t Creg;
//   uint8_t ctreg;
//
//  public:
//   state_MS_dec(const uint8_t *Dcup, uint32_t Pcup)
//       : pos(0), bits(0), tmp(0), last(0), buf(Dcup), length(Pcup), Creg(0), ctreg(0) {
//     while (ctreg < 32) {
//       loadByte();
//     }
//   }
//   void loadByte();
//   void close(int32_t num_bits);
//   uint8_t importMagSgnBit();
//   int32_t decodeMagSgnValue(int32_t m_n, int32_t i_n);
// };
//
///********************************************************************************
// * state_MEL_unPacker and state_MEL: state classes for MEL decoding
// *******************************************************************************/
// class state_MEL_unPacker {
// private:
//  int32_t pos;
//  int8_t bits;
//  uint8_t tmp;
//  const uint8_t *buf;
//  uint32_t length;
//
// public:
//  state_MEL_unPacker(const uint8_t *Dcup, uint32_t Lcup, int32_t Pcup)
//      : pos(Pcup), bits(0), tmp(0), buf(Dcup), length(Lcup) {}
//  uint8_t importMELbit();
//};
//
// class state_MEL_decoder {
// private:
//  uint8_t MEL_k;
//  uint8_t MEL_run;
//  uint8_t MEL_one;
//  const uint8_t MEL_E[13];
//  state_MEL_unPacker *MEL_unPacker;
//
// public:
//  explicit state_MEL_decoder(state_MEL_unPacker &unpacker)
//      : MEL_k(0),
//        MEL_run(0),
//        MEL_one(0),
//        MEL_E{0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 5},
//        MEL_unPacker(&unpacker) {}
//  uint8_t decodeMELSym();
//};
//
// #define ADVANCED
// #ifdef ADVANCED
//  #define getbitfunc getVLCbit()
// #else
//  #define getbitfunc importVLCBit()
// #endif
///********************************************************************************
// * state_VLC: state class for VLC decoding
// *******************************************************************************/
// class state_VLC_dec {
// private:
//  int32_t pos;
//  uint8_t last;
// #ifndef ADVANCED
//  uint8_t tmp;
//  uint32_t rev_length;
// #else
//  int32_t ctreg;
//  uint64_t Creg;
// #endif
//  uint8_t bits;
//  uint8_t *buf;
//
// public:
//  state_VLC_dec(uint8_t *Dcup, uint32_t Lcup, int32_t Pcup)
// #ifndef ADVANCED
//      : pos((Lcup > 2) ? Lcup - 3 : 0),
//        last(*(Dcup + Lcup - 2)),
//        tmp(last >> 4),
//        rev_length(Pcup),
//        bits(((tmp & 0x07) < 7) ? 4 : 3),
//        buf(Dcup) {
//  }
//  uint8_t importVLCBit();
// #else
//      : pos(static_cast<int32_t>(Lcup) - 2 - Pcup), ctreg(0), Creg(0), bits(0), buf(Dcup + Pcup) {
//    load_bytes();
//    ctreg -= 4;
//    Creg >>= 4;
//    while (ctreg < 32) {
//      load_bytes();
//    }
//  }
//  void load_bytes();
//  uint8_t getVLCbit();
//  void close32(int32_t num_bits);
// #endif
//  void decodeCxtVLC(const uint16_t &context, uint8_t (&u_off)[2], uint8_t (&rho)[2], uint8_t (&emb_k)[2],
//                    uint8_t (&emb_1)[2], const uint8_t &first_or_second, const uint16_t
//                    *dec_CxtVLC_table);
//  uint8_t decodeUPrefix();
//  uint8_t decodeUSuffix(const uint32_t &u_pfx);
//  uint8_t decodeUExtension(const uint32_t &u_sfx);
//};
/********************************************************************************
 * SP_dec: state class for HT SigProp decoding
 *******************************************************************************/
class SP_dec {
 private:
  const uint32_t Lref;
  uint8_t bits;
  uint8_t tmp;
  uint8_t last;
  uint32_t pos;
  const uint8_t *Dref;

 public:
  SP_dec(const uint8_t *HT_magref_segment, uint32_t magref_length)
      : Lref(magref_length),
        bits(0),
        tmp(0),
        last(0),
        pos(0),
        Dref((Lref == 0) ? nullptr : HT_magref_segment) {}
  uint8_t importSigPropBit();
};

/********************************************************************************
 * MR_dec: state class for HT MagRef decoding
 *******************************************************************************/
class MR_dec {
 private:
  const uint32_t Lref;
  uint8_t bits;
  uint8_t last;
  uint8_t tmp;
  int32_t pos;
  const uint8_t *Dref;

 public:
  MR_dec(const uint8_t *HT_magref_segment, uint32_t magref_length)
      : Lref(magref_length),
        bits(0),
        last(0xFF),
        tmp(0),
        pos((Lref == 0) ? -1 : static_cast<int32_t>(magref_length - 1)),
        Dref((Lref == 0) ? nullptr : HT_magref_segment) {}
  uint8_t importMagRefBit();
};

///********************************************************************************
// * functions for state_MS: state class for MagSgn decoding
// *******************************************************************************/
// void state_MS_dec::loadByte() {
//  tmp  = 0xFF;
//  bits = (last == 0xFF) ? 7 : 8;
//  if (pos < length) {
//    tmp = buf[pos];
//    pos++;
//    last = tmp;
//  }
//
//  Creg |= static_cast<uint64_t>(tmp) << ctreg;
//  ctreg = static_cast<uint8_t>(ctreg + bits);
//}
// void state_MS_dec::close(int32_t num_bits) {
//  Creg >>= num_bits;
//  ctreg = static_cast<uint8_t>(ctreg - static_cast<uint8_t>(num_bits));
//  while (ctreg < 32) {
//    loadByte();
//  }
//}
//
//OPENHTJ2K_MAYBE_UNUSED uint8_t state_MS_dec::importMagSgnBit() {
//  uint8_t val;
//  if (bits == 0) {
//    bits = (last == 0xFF) ? 7 : 8;
//    if (pos < length) {
//      tmp = *(buf + pos);  // modDcup(MS->pos, Lcup);
//      if ((static_cast<uint16_t>(tmp) & static_cast<uint16_t>(1 << bits)) != 0) {
//        printf("ERROR: importMagSgnBit error\n");
//        throw std::exception();
//      }
//    } else if (pos == length) {
//      tmp = 0xFF;
//    } else {
//      printf("ERROR: importMagSgnBit error\n");
//      throw std::exception();
//    }
//    last = tmp;
//    pos++;
//  }
//  val = tmp & 1;
//  tmp = static_cast<uint8_t>(tmp >> 1);
//  --bits;
//  return val;
//}
//
//OPENHTJ2K_MAYBE_UNUSED int32_t state_MS_dec::decodeMagSgnValue(int32_t m_n, int32_t i_n) {
//  int32_t val = 0;
//  // uint8_t bit;
//  if (m_n > 0) {
//    val = static_cast<int32_t>(bitmask32[m_n] & (int32_t)Creg);
//    //      for (int i = 0; i < m_n; i++) {
//    //        bit = MS->importMagSgnBit();
//    //        val += (bit << i);
//    //      }
//    val += (i_n << m_n);
//    close(m_n);
//  } else {
//    val = 0;
//  }
//  return val;
//}
//
///********************************************************************************
// * functions for state_MEL_unPacker and state_MEL: state classes for MEL decoding
// *******************************************************************************/
// uint8_t state_MEL_unPacker::importMELbit() {
//  if (bits == 0) {
//    bits = (tmp == 0xFF) ? 7 : 8;
//    if (pos < length) {
//      tmp = *(buf + pos);  //+ modDcup(MEL_unPacker->pos, Lcup);
//      //        MEL_unPacker->tmp = modDcup()
//      pos++;
//    } else {
//      tmp = 0xFF;
//    }
//  }
//  bits--;
//  return (tmp >> bits) & 1;
//}
//
// uint8_t state_MEL_decoder::decodeMELSym() {
//  uint8_t eval;
//  uint8_t bit;
//  if (MEL_run == 0 && MEL_one == 0) {
//    eval = this->MEL_E[MEL_k];
//    bit  = MEL_unPacker->importMELbit();
//    if (bit == 1) {
//      MEL_run = static_cast<uint8_t>(1 << eval);
//      MEL_k   = static_cast<uint8_t>((12 < MEL_k + 1) ? 12 : MEL_k + 1);
//    } else {
//      MEL_run = 0;
//      while (eval > 0) {
//        bit     = MEL_unPacker->importMELbit();
//        MEL_run = static_cast<uint8_t>((MEL_run << 1) + bit);
//        eval--;
//      }
//      MEL_k   = static_cast<uint8_t>((0 > MEL_k - 1) ? 0 : MEL_k - 1);
//      MEL_one = 1;
//    }
//  }
//  if (MEL_run > 0) {
//    MEL_run--;
//    return 0;
//  } else {
//    MEL_one = 0;
//    return 1;
//  }
//}
//
///********************************************************************************
// * functions for state_VLC: state class for VLC decoding
// *******************************************************************************/
// #ifndef ADVANCED
// uint8_t state_VLC::importVLCBit() {
//  uint8_t val;
//  if (bits == 0) {
//    if (pos >= rev_length) {
//      tmp = *(buf + pos);  // modDcup(VLC->pos, Lcup);
//    } else {
//      printf("ERROR: import VLCBits error\n");
//      throw std::exception();
//    }
//    bits = 8;
//    if (last > 0x8F && (tmp & 0x7F) == 0x7F) {
//      bits = 7;  // bit-un-stuffing
//    }
//    last = tmp;
//    // To prevent overflow of pos
//    if (pos > 0) {
//      pos--;
//    }
//  }
//  val = tmp & 1;
//  tmp >>= 1;
//  bits--;
//  return val;
//}
// #else
// void state_VLC_dec::load_bytes() {
//  uint64_t load_val = 0;
//  int32_t new_bits  = 32;
//  last              = buf[pos + 1];
//  if (pos >= 3) {  // Common case; we have at least 4 bytes available
//    load_val = buf[pos - 3];
//    load_val = (load_val << 8) | buf[pos - 2];
//    load_val = (load_val << 8) | buf[pos - 1];
//    load_val = (load_val << 8) | buf[pos];
//    load_val = (load_val << 8) | last;  // For stuffing bit detection
//    pos -= 4;
//  } else {
//    if (pos >= 2) {
//      load_val = buf[pos - 2];
//    }
//    if (pos >= 1) {
//      load_val = (load_val << 8) | buf[pos - 1];
//    }
//    if (pos >= 0) {
//      load_val = (load_val << 8) | buf[pos];
//    }
//    pos      = 0;
//    load_val = (load_val << 8) | last;  // For stuffing bit detection
//  }
//  // Now remove any stuffing bits, shifting things down as we go
//  if ((load_val & 0x7FFF000000) > 0x7F8F000000) {
//    load_val &= 0x7FFFFFFFFF;
//    new_bits--;
//  }
//  if ((load_val & 0x007FFF0000) > 0x007F8F0000) {
//    load_val = (load_val & 0x007FFFFFFF) + ((load_val & 0xFF00000000) >> 1);
//    new_bits--;
//  }
//  if ((load_val & 0x00007FFF00) > 0x00007F8F00) {
//    load_val = (load_val & 0x00007FFFFF) + ((load_val & 0xFFFF000000) >> 1);
//    new_bits--;
//  }
//  if ((load_val & 0x0000007FFF) > 0x0000007F8F) {
//    load_val = (load_val & 0x0000007FFF) + ((load_val & 0xFFFFFF0000) >> 1);
//    new_bits--;
//  }
//  load_val >>= 8;  // Shifts away the extra byte we imported
//  Creg |= (load_val << ctreg);
//  ctreg += new_bits;
//}
//
// uint8_t state_VLC_dec::getVLCbit() {
//  // "bits" is not actually bits, but a bit
//  bits = (uint8_t)(Creg & 0x01);
//  close32(1);
//  return bits;
//}
//
// void state_VLC_dec::close32(int32_t num_bits) {
//  Creg >>= num_bits;
//  ctreg -= num_bits;
//  while (ctreg < 32) {
//    load_bytes();
//  }
//}
// #endif
//
//OPENHTJ2K_MAYBE_UNUSED void state_VLC_dec::decodeCxtVLC(const uint16_t &context, uint8_t (&u_off)[2],
//                                                  uint8_t (&rho)[2], uint8_t (&emb_k)[2],
//                                                  uint8_t (&emb_1)[2], const uint8_t &first_or_second,
//                                                  const uint16_t *dec_CxtVLC_table) {
// #ifndef ADVANCED
//  uint8_t b_low = tmp;
//  uint8_t b_upp = *(buf + pos);  // modDcup(VLC->pos, Lcup);
//  uint16_t word = (b_upp << bits) + b_low;
//  uint8_t cwd   = word & 0x7F;
// #else
//  uint8_t cwd = Creg & 0x7f;
// #endif
//  uint16_t idx           = static_cast<uint16_t>(cwd + (context << 7));
//  uint16_t value         = dec_CxtVLC_table[idx];
//  u_off[first_or_second] = value & 1;
//  // value >>= 1;
//  // uint8_t len = value & 0x07;
//  // value >>= 3;
//  // rho[first_or_second] = value & 0x0F;
//  // value >>= 4;
//  // emb_k[first_or_second] = value & 0x0F;
//  // value >>= 4;
//  // emb_1[first_or_second] = value & 0x0F;
//  uint8_t len            = static_cast<uint8_t>((value & 0x000F) >> 1);
//  rho[first_or_second]   = static_cast<uint8_t>((value & 0x00F0) >> 4);
//  emb_k[first_or_second] = static_cast<uint8_t>((value & 0x0F00) >> 8);
//  emb_1[first_or_second] = static_cast<uint8_t>((value & 0xF000) >> 12);
//
// #ifndef ADVANCED
//  for (int i = 0; i < len; i++) {
//    importVLCBit();
//  }
// #else
//  close32(len);
// #endif
//}
//
//OPENHTJ2K_MAYBE_UNUSED uint8_t state_VLC_dec::decodeUPrefix() {
//  if (getbitfunc == 1) {
//    return 1;
//  }
//  if (getbitfunc == 1) {
//    return 2;
//  }
//  if (getbitfunc == 1) {
//    return 3;
//  } else {
//    return 5;
//  }
//}
//
//OPENHTJ2K_MAYBE_UNUSED uint8_t state_VLC_dec::decodeUSuffix(const uint32_t &u_pfx) {
//  uint8_t bit, val;
//  if (u_pfx < 3) {
//    return 0;
//  }
//  val = getbitfunc;
//  if (u_pfx == 3) {
//    return val;
//  }
//  for (int i = 1; i < 5; i++) {
//    bit = getbitfunc;
//    val = static_cast<uint8_t>(val + (bit << i));
//  }
//  return val;
//}
//OPENHTJ2K_MAYBE_UNUSED uint8_t state_VLC_dec::decodeUExtension(const uint32_t &u_sfx) {
//  uint8_t bit, val;
//  if (u_sfx < 28) {
//    return 0;
//  }
//  val = getbitfunc;
//  for (int i = 1; i < 4; i++) {
//    bit = getbitfunc;
//    val = static_cast<uint8_t>(val + (bit << i));
//  }
//  return val;
//}
/********************************************************************************
 * functions for SP_dec: state class for HT SigProp decoding
 *******************************************************************************/
FORCE_INLINE uint8_t SP_dec::importSigPropBit() {
  if (bits == 0) {
    bits = (last == 0xFF) ? 7 : 8;
    if (pos < Lref) {
      tmp = *(Dref + pos);
      pos++;
      if ((tmp & (1 << bits)) != 0) {
        printf("ERROR: importSigPropBit error\n");
        throw std::exception();
      }
    } else {
      tmp = 0;
    }
    last = tmp;
  }
  uint8_t val = tmp & 1;
  tmp = static_cast<uint8_t>(tmp >> 1);
  bits--;
  return val;
}

/********************************************************************************
 * MR_dec: state class for HT MagRef decoding
 *******************************************************************************/
FORCE_INLINE uint8_t MR_dec::importMagRefBit() {
  if (bits == 0) {
    if (pos >= 0) {
      tmp = *(Dref + pos);
      pos--;
    } else {
      tmp = 0;
    }
    bits = 8;
    if (last > 0x8F && (tmp & 0x7F) == 0x7F) {
      bits = 7;
    }
    last = tmp;
  }
  uint8_t val = tmp & 1;
  tmp = static_cast<uint8_t>(tmp >> 1);
  bits--;
  return val;
}

//OPENHTJ2K_MAYBE_UNUSED auto decodeSigEMB = [](state_MEL_decoder &MEL_decoder, rev_buf &VLC_dec,
//                                        const uint16_t &context, uint8_t (&u_off)[2], uint8_t (&rho)[2],
//                                        uint8_t (&emb_k)[2], uint8_t (&emb_1)[2],
//                                        const uint8_t &first_or_second, const uint16_t *dec_CxtVLC_table)
//                                        {
//  uint8_t sym;
//  if (context == 0) {
//    sym = MEL_decoder.decodeMELSym();
//    if (sym == 0) {
//      rho[first_or_second] = u_off[first_or_second] = emb_k[first_or_second] = emb_1[first_or_second] = 0;
//      return;
//    }
//  }
//  uint32_t vlcval        = VLC_dec.fetch();
//  uint16_t value         = dec_CxtVLC_table[(vlcval & 0x7F) + (context << 7)];
//  u_off[first_or_second] = value & 1;
//  uint32_t len           = static_cast<uint8_t>((value & 0x000F) >> 1);
//  rho[first_or_second]   = static_cast<uint8_t>((value & 0x00F0) >> 4);
//  emb_k[first_or_second] = static_cast<uint8_t>((value & 0x0F00) >> 8);
//  emb_1[first_or_second] = static_cast<uint8_t>((value & 0xF000) >> 12);
//  VLC_dec.advance(len);
//  //  VLC_dec.decodeCxtVLC(context, u_off, rho, emb_k, emb_1, first_or_second, dec_CxtVLC_table);
//};
