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
#include <algorithm>  // for max{a,b,c,d}

const int32_t bitmask32[32] = {
    0x00000000, 0x00000001, 0x00000003, 0x00000007, 0x0000000F, 0x0000001F, 0x0000003F, 0x0000007F,
    0x000000FF, 0x000001FF, 0x000003FF, 0x000007FF, 0x00000FFF, 0x00001FFF, 0x00003FFF, 0x00007FFF,
    0x0000FFFF, 0x0001FFFF, 0x0003FFFF, 0x0007FFFF, 0x000FFFFF, 0x001FFFFF, 0x003FFFFF, 0x007FFFFF,
    0x00FFFFFF, 0x01FFFFFF, 0x03FFFFFF, 0x07FFFFFF, 0x0FFFFFFF, 0x1FFFFFFF, 0x3FFFFFFF, 0x7FFFFFFF};
/********************************************************************************
 * rev_buf:
 *******************************************************************************/
class rev_buf {
 private:
  int32_t pos;
  uint32_t bits;
  uint64_t Creg;
  uint32_t unstuff;
  uint8_t *buf;
  int32_t length;

 public:
  rev_buf(uint8_t *Dcup, uint32_t Lcup, int32_t Scup)
      : pos(Scup - 2), bits(0), Creg(0), unstuff(0), buf(Dcup + Lcup - 2), length(Scup - 2) {
    uint32_t d = *buf--;  // read a byte (only use it's half byte)
    Creg       = d >> 4;
    bits       = 4 - ((Creg & 0x07) == 0x07);
    unstuff    = (d | 0x0F) > 0x8f;

    auto p = reinterpret_cast<intptr_t>(buf);
    p &= 0x03;
    auto num  = 1 + p;
    auto tnum = (num < length) ? num : length;
    for (auto i = 0; i < tnum; ++i) {
      uint64_t d;
      d               = *buf--;
      uint32_t d_bits = 8 - ((unstuff && ((d & 0x7F) == 0x7F)) ? 1 : 0);
      Creg |= d << bits;
      bits += d_bits;
      unstuff = d > 0x8F;
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
      printf("ERROR:VLC require %d bits but %d bits left", num_bits, bits);
      throw std::exception();
    }
    Creg >>= num_bits;
    bits -= num_bits;
    return static_cast<uint32_t>(Creg);
  }

  inline void decodeCxtVLC(const uint16_t &context, uint8_t (&u_off)[2], uint8_t (&rho)[2],
                           uint8_t (&emb_k)[2], uint8_t (&emb_1)[2], const uint8_t &first_or_second,
                           const uint16_t *dec_CxtVLC_table) {
    fetch();
    uint8_t cwd            = Creg & 0x7f;
    uint16_t idx           = static_cast<uint16_t>(cwd + (context << 7));
    uint16_t value         = dec_CxtVLC_table[idx];
    u_off[first_or_second] = value & 1;
    uint8_t len            = static_cast<uint8_t>((value & 0x000F) >> 1);
    rho[first_or_second]   = static_cast<uint8_t>((value & 0x00F0) >> 4);
    emb_k[first_or_second] = static_cast<uint8_t>((value & 0x0F00) >> 8);
    emb_1[first_or_second] = static_cast<uint8_t>((value & 0xF000) >> 12);
    advance(len);
  }

  inline uint8_t importVLCBit() {
    uint32_t cwd = fetch();
    advance(1);
    return (cwd & 1);
  }

  inline uint8_t decodeUPrefix() {
    uint8_t bit = importVLCBit();
    if (bit == 1) return 1;
    bit = importVLCBit();
    if (bit == 1) return 2;
    bit = importVLCBit();
    return (bit == 1) ? 3 : 5;
  }

  inline uint8_t decodeUSuffix(const uint32_t &u_pfx) {
    uint8_t val;
    if (u_pfx < 3) return 0;
    val = importVLCBit();
    if (u_pfx == 3) return val;
    //    for (int i = 1; i < 5; ++i) {
    //      uint8_t bit = importVLCBit();
    //      val += bit << i;
    //    }
    uint32_t cwd = fetch();
    advance(4);
    val += (cwd & 0x0F) << 1;
    return val;
  }

  inline uint8_t decodeUExtension(const uint32_t &u_sfx) {
    uint8_t val;
    if (u_sfx < 28) return 0;
    val = importVLCBit();
    //    for (int i = 1; i < 4; ++i) {
    //      uint8_t bit = importVLCBit();
    //      val += bit << i;
    //    }
    uint32_t cwd = fetch();
    advance(3);
    val += (cwd & 0x07) << 1;
    return val;
    return val;
  }
};

/********************************************************************************
 * fwd_buf:
 *******************************************************************************/
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
      printf("ERROR: ");
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
    uint32_t bits_local = 8 - unstuff;
    uint32_t t          = val & 0xFF;
    bool unstuff_flag   = ((val & 0xFF) == 0xFF);  // Do we need unstuffing next?

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
      printf("ERROR:");
      throw std::exception();
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
class state_MS_dec {
 private:
  uint32_t pos;
  uint8_t bits;
  uint8_t tmp;
  uint8_t last;
  const uint8_t *buf;
  const uint32_t length;
  uint64_t Creg;
  uint8_t ctreg;

 public:
  state_MS_dec(const uint8_t *Dcup, uint32_t Pcup)
      : pos(0), bits(0), tmp(0), last(0), buf(Dcup), length(Pcup), Creg(0), ctreg(0) {
    while (ctreg < 32) {
      loadByte();
    }
  }
  void loadByte();
  void close(int32_t num_bits);
  uint8_t importMagSgnBit();
  int32_t decodeMagSgnValue(int32_t m_n, int32_t i_n);
};

/********************************************************************************
 * state_MEL_unPacker and state_MEL: state classes for MEL decoding
 *******************************************************************************/
class state_MEL_unPacker {
 private:
  int32_t pos;
  int8_t bits;
  uint8_t tmp;
  const uint8_t *buf;
  uint32_t length;

 public:
  state_MEL_unPacker(const uint8_t *Dcup, uint32_t Lcup, int32_t Pcup)
      : pos(Pcup), bits(0), tmp(0), buf(Dcup), length(Lcup) {}
  uint8_t importMELbit();
};

class state_MEL_decoder {
 private:
  uint8_t MEL_k;
  uint8_t MEL_run;
  uint8_t MEL_one;
  const uint8_t MEL_E[13];
  state_MEL_unPacker *MEL_unPacker;

 public:
  explicit state_MEL_decoder(state_MEL_unPacker &unpacker)
      : MEL_k(0),
        MEL_run(0),
        MEL_one(0),
        MEL_E{0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 5},
        MEL_unPacker(&unpacker) {}
  uint8_t decodeMELSym();
};

#define ADVANCED
#ifdef ADVANCED
  #define getbitfunc getVLCbit()
#else
  #define getbitfunc importVLCBit()
#endif
/********************************************************************************
 * state_VLC: state class for VLC decoding
 *******************************************************************************/
class state_VLC_dec {
 private:
  int32_t pos;
  uint8_t last;
#ifndef ADVANCED
  uint8_t tmp;
  uint32_t rev_length;
#else
  int32_t ctreg;
  uint64_t Creg;
#endif
  uint8_t bits;
  uint8_t *buf;

 public:
  state_VLC_dec(uint8_t *Dcup, uint32_t Lcup, int32_t Pcup)
#ifndef ADVANCED
      : pos((Lcup > 2) ? Lcup - 3 : 0),
        last(*(Dcup + Lcup - 2)),
        tmp(last >> 4),
        rev_length(Pcup),
        bits(((tmp & 0x07) < 7) ? 4 : 3),
        buf(Dcup) {
  }
  uint8_t importVLCBit();
#else
      : pos(static_cast<int32_t>(Lcup) - 2 - Pcup), ctreg(0), Creg(0), bits(0), buf(Dcup + Pcup) {
    load_bytes();
    ctreg -= 4;
    Creg >>= 4;
    while (ctreg < 32) {
      load_bytes();
    }
  }
  void load_bytes();
  uint8_t getVLCbit();
  void close32(int32_t num_bits);
#endif
  void decodeCxtVLC(const uint16_t &context, uint8_t (&u_off)[2], uint8_t (&rho)[2], uint8_t (&emb_k)[2],
                    uint8_t (&emb_1)[2], const uint8_t &first_or_second, const uint16_t *dec_CxtVLC_table);
  uint8_t decodeUPrefix();
  uint8_t decodeUSuffix(const uint32_t &u_pfx);
  uint8_t decodeUExtension(const uint32_t &u_sfx);
};
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
