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
  int bits;
  uint64_t tmp;
  const uint8_t *buf;
  int32_t length;
  bool unstuff;
  int MEL_k;

  int32_t num_runs;
  uint64_t runs;

 public:
  MEL_dec(const uint8_t *Dcup, int32_t Lcup, int32_t Scup)
      : bits(0),
        tmp(0),
        buf(Dcup + Lcup - Scup),
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
  uint32_t bits;
  uint64_t Creg;
  uint32_t unstuff;
  uint8_t *buf;
  int32_t length;

 public:
  rev_buf(uint8_t *Dcup, int32_t Lcup, int32_t Scup)
      : bits(0), Creg(0), unstuff(0), buf(Dcup + Lcup - 2), length(Scup - 2) {
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
      uint32_t d_bits = 8 - ((unstuff && ((d64 & 0x7F) == 0x7F)) ? 1 : 0);
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
    bits_local        = 8 - ((unstuff && (((val >> 24) & 0x7F) == 0x7F)) ? 1 : 0);
    bool unstuff_flag = (val >> 24) > 0x8F;  // this is for the next byte

    tmp |= ((val >> 16) & 0xFF) << bits_local;  // process the next byte
    bits_local += 8 - ((unstuff_flag && (((val >> 16) & 0x7F) == 0x7F)) ? 1 : 0);
    unstuff_flag = ((val >> 16) & 0xFF) > 0x8F;

    tmp |= ((val >> 8) & 0xFF) << bits_local;
    bits_local += 8 - ((unstuff_flag && (((val >> 8) & 0x7F) == 0x7F)) ? 1 : 0);
    unstuff_flag = ((val >> 8) & 0xFF) > 0x8F;

    tmp |= (val & 0xFF) << bits_local;
    bits_local += 8 - ((unstuff_flag && ((val & 0x7F) == 0x7F)) ? 1 : 0);
    unstuff_flag = (val & 0xFF) > 0x8F;

    // now move the read and unstuffed bits into this->Creg
    Creg |= static_cast<uint64_t>(tmp) << bits;
    bits += bits_local;
    unstuff = unstuff_flag;  // this for the next read
  }

  inline uint32_t fetch() {
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
// this class implementation is borrowed from OpenJPH
template <int X>
class fwd_buf {
 private:
  uint32_t pos;
  uint32_t bits;
  uint64_t Creg;
  uint32_t unstuff;
  const uint8_t *buf;
  int32_t length;

 public:
  fwd_buf(const uint8_t *Dcup, int32_t Pcup)
      : pos(0), bits(0), Creg(0), unstuff(0), buf(Dcup), length(Pcup) {
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

    Creg |= ((uint64_t)t) << bits;  // move data to msp->tmp
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
//#define ADVANCED
//#ifdef ADVANCED
//  #define getbitfunc getVLCbit()
//#else
//  #define getbitfunc importVLCBit()
//#endif
///********************************************************************************
// * state_VLC: state class for VLC decoding
// *******************************************************************************/
// class state_VLC_dec {
// private:
//  int32_t pos;
//  uint8_t last;
//#ifndef ADVANCED
//  uint8_t tmp;
//  uint32_t rev_length;
//#else
//  int32_t ctreg;
//  uint64_t Creg;
//#endif
//  uint8_t bits;
//  uint8_t *buf;
//
// public:
//  state_VLC_dec(uint8_t *Dcup, uint32_t Lcup, int32_t Pcup)
//#ifndef ADVANCED
//      : pos((Lcup > 2) ? Lcup - 3 : 0),
//        last(*(Dcup + Lcup - 2)),
//        tmp(last >> 4),
//        rev_length(Pcup),
//        bits(((tmp & 0x07) < 7) ? 4 : 3),
//        buf(Dcup) {
//  }
//  uint8_t importVLCBit();
//#else
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
//#endif
//  void decodeCxtVLC(const uint16_t &context, uint8_t (&u_off)[2], uint8_t (&rho)[2], uint8_t (&emb_k)[2],
//                    uint8_t (&emb_1)[2], const uint8_t &first_or_second, const uint16_t
//                    *dec_CxtVLC_table);
//  uint8_t decodeUPrefix();
//  uint8_t decodeUSuffix(const uint32_t &u_pfx);
//  uint8_t decodeUExtension(const uint32_t &u_sfx);
//};
//
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
//[[maybe_unused]] uint8_t state_MS_dec::importMagSgnBit() {
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
//[[maybe_unused]] int32_t state_MS_dec::decodeMagSgnValue(int32_t m_n, int32_t i_n) {
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
//#ifndef ADVANCED
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
//#else
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
//#endif
//
//[[maybe_unused]] void state_VLC_dec::decodeCxtVLC(const uint16_t &context, uint8_t (&u_off)[2],
//                                                  uint8_t (&rho)[2], uint8_t (&emb_k)[2],
//                                                  uint8_t (&emb_1)[2], const uint8_t &first_or_second,
//                                                  const uint16_t *dec_CxtVLC_table) {
//#ifndef ADVANCED
//  uint8_t b_low = tmp;
//  uint8_t b_upp = *(buf + pos);  // modDcup(VLC->pos, Lcup);
//  uint16_t word = (b_upp << bits) + b_low;
//  uint8_t cwd   = word & 0x7F;
//#else
//  uint8_t cwd = Creg & 0x7f;
//#endif
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
//#ifndef ADVANCED
//  for (int i = 0; i < len; i++) {
//    importVLCBit();
//  }
//#else
//  close32(len);
//#endif
//}
//
//[[maybe_unused]] uint8_t state_VLC_dec::decodeUPrefix() {
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
//[[maybe_unused]] uint8_t state_VLC_dec::decodeUSuffix(const uint32_t &u_pfx) {
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
//[[maybe_unused]] uint8_t state_VLC_dec::decodeUExtension(const uint32_t &u_sfx) {
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
//
//[[maybe_unused]] auto decodeSigEMB = [](state_MEL_decoder &MEL_decoder, rev_buf &VLC_dec,
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
