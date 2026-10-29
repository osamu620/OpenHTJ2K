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
#include "open_htj2k_typedef.hpp"
#include "utils.hpp"

// #define MQNAIVE
// #define CDP

class mq_decoder {
 public:
  int32_t A;  // was uint16_t
  int32_t t;  // was uint8_t
  // Lower-bound interval
  int32_t C;  // was uint32_t
#if defined(CDP)
  int32_t D;  // only for CDP implementation
#endif
  // Temporary byte register
  int32_t T;  // was uint8_t
  // position in byte-stream
  uint32_t L;
  // start position in byte-stream
  OPENHTJ2K_MAYBE_UNUSED uint32_t L_start;
  // position of current codeword segment boundary
  uint32_t Lmax;
  // dynamic table for context
  uint16_t dynamic_table[2][19];
  // Byte-stream buffer
  uint8_t const *byte_buffer;
  explicit mq_decoder(const uint8_t *buf);
  void fill_LSBs();
  void init(uint32_t buf_pos, uint32_t segment_length, bool is_bypass);
  void init_states_for_all_contexts();
  void renormalize_once();
  void renormalize();
  uint8_t decode(uint8_t label);
  uint8_t get_raw_symbol();
  void finish();
};

// The per-symbol routines are defined here so that the coding-pass loops in
// block_decoding.cpp can inline them and keep the coder state in registers.
#if defined(MQNAIVE)
  #define MQMINOFF 0
#else
  #define MQMINOFF 8
#endif
static constexpr int32_t Areg_min = 1 << (15 + MQMINOFF);

// static table for state transition (32bits)
// Xs (1bit), Sigma_mps (6bit in 7bit),Sigma_lps (6bit in 8bit), Qe (16bit)
static constexpr uint32_t static_table[47] = {
    0x81015601, 0x02063401, 0x03091801, 0x040c0ac1, 0x051d0521, 0x26210221, 0x87065601, 0x080e5401,
    0x090e4801, 0x0a0e3801, 0x0b113001, 0x0c122401, 0x0d141c01, 0x1d151601, 0x8f0e5601, 0x100e5401,
    0x110f5101, 0x12104801, 0x13113801, 0x14123401, 0x15133001, 0x16132801, 0x17142401, 0x18152201,
    0x19161c01, 0x1a171801, 0x1b181601, 0x1c191401, 0x1d1a1201, 0x1e1b1101, 0x1f1c0ac1, 0x201d09c1,
    0x211e08a1, 0x221f0521, 0x23200441, 0x242102a1, 0x25220221, 0x26230141, 0x27240111, 0x28250085,
    0x29260049, 0x2a270025, 0x2b280015, 0x2c290009, 0x2d2a0005, 0x2d2b0001, 0x2e2e5601};

inline void mq_decoder::renormalize() {
  int32_t n = static_cast<int32_t>(count_leading_zeros(static_cast<uint32_t>(A))) - 8;
  if (t == 0) {
    fill_LSBs();
  }
  if (n <= t) {
    A <<= n;
    C <<= n;
    t -= n;
  } else {
    A <<= t;
    C <<= t;
    n -= t;
    t = 0;
    do {
      fill_LSBs();
      int32_t shift = (n <= t) ? n : t;
      A <<= shift;
      C <<= shift;
      t -= shift;
      n -= shift;
    } while (n > 0);
  }
}

inline void mq_decoder::fill_LSBs() {
  t = 8;
  if (L == Lmax || (T == 0xFF && byte_buffer[L] > 0x8F)) {
    // Codeword exhausted; fill C with 1's from now on
    C += 0xFF;
  } else {
    if (T == 0xFF) {
      t = 7;
    }
    T = byte_buffer[L];
    L++;
    C += T << (8 - t);
  }
}

#if !defined(CDP) && !defined(MQNAIVE)
/**
 * @brief MQ Decode Procedure (from JPEG 2000 book P.646 - 647)
 * @param[in] label context label
 * @return uint8_t output-symbol
 */
inline uint8_t mq_decoder::decode(uint8_t label) {
  int32_t x;
  uint16_t Sigma_k         = dynamic_table[0][label];
  uint32_t val             = static_table[Sigma_k];
  const uint16_t Sigma_mps = (val >> 24) & 0x3F;                // static_table[0][Sigma_k];
  const uint16_t Sigma_lps = (val >> 16) & 0x3F;                // static_table[1][Sigma_k];
  const uint16_t Xs        = static_cast<uint16_t>(val >> 31);  // static_table[2][Sigma_k];
  // = p_bar (from the static table) << 8
  const int32_t p_shifted = (int32_t)((val & 0xFFFF) << 8);
  uint16_t sk             = dynamic_table[1][label];

  x = sk;  // set to MPS for now, since this is most likely
  A = A - p_shifted;
  if (C >= p_shifted) {  // identical to C_active >= p_bar (upper sub-interval selected),
    C -= p_shifted;      // identical to C_active -= p_bar, C_active is equal to (C & 0xFFFF00)
    if (A < Areg_min) {  // need renormalization and perhaps conditional exchange
      int32_t is_lps = (A < p_shifted) ? 1 : 0;
      x              = sk ^ is_lps;
      sk             = sk ^ (Xs & static_cast<uint16_t>(-is_lps));
      Sigma_k        = is_lps ? Sigma_lps : Sigma_mps;
      renormalize();
    }
  } else {  // lower sub-interval selected; renormalization is inevitable
    int32_t is_lps = (A >= p_shifted) ? 1 : 0;
    x              = sk ^ is_lps;
    sk             = sk ^ (Xs & static_cast<uint16_t>(-is_lps));
    Sigma_k        = is_lps ? Sigma_lps : Sigma_mps;
    A              = p_shifted;
    renormalize();
  }
  dynamic_table[0][label] = Sigma_k;
  dynamic_table[1][label] = sk;
  return static_cast<uint8_t>(x);
}
#endif

inline uint8_t mq_decoder::get_raw_symbol() {
  if (t == 0) {
    t = 8;
    if (L == Lmax) {
      T = 0xFF;
    } else {
      if (T == 0xFF) {
        t = 7;
      }
      T = byte_buffer[L];
      L++;
    }
  }
  t--;
  return ((T >> t) & 1);
}
