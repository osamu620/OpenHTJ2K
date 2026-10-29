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

#include "mq_decoder.hpp"

#include <cstdio>
#include <stdexcept>
#include "utils.hpp"

mq_decoder::mq_decoder(const uint8_t *const buf)
    : A(0), t(0), C(0), T(0), L(0), L_start(0), Lmax(0), dynamic_table{{}}, byte_buffer(buf) {}

void mq_decoder::init(uint32_t buf_pos, uint32_t segment_length, bool is_bypass) {
  L_start = buf_pos;
  Lmax    = buf_pos + segment_length;
  L       = buf_pos;  // this means L points the beginning of a codeword segment (L=0)
  T       = 0;
  if (is_bypass) {
    t = 0;
  } else {
    A = Areg_min;
    C = 0;
    fill_LSBs();
    C <<= t;
    fill_LSBs();
    C <<= 7;
    t -= 7;
#if defined(CDP)
    // only for CDP implementation
    D = A - Areg_min;
    D = (C < D) ? C : D;
    A -= D;
    C -= D;
#endif
  }
}

void mq_decoder::init_states_for_all_contexts() {
  for (uint8_t i = 0; i < 19; i++) {
    dynamic_table[0][i] = 0;
    dynamic_table[1][i] = 0;
  }
  dynamic_table[0][0]  = 4;
  dynamic_table[0][17] = 3;
  dynamic_table[0][18] = 46;
}

void mq_decoder::renormalize_once() {
  if (t == 0) {
    fill_LSBs();
  }
  A <<= 1;
  C <<= 1;
  t--;
}

#if defined(MQNAIVE)
// Naive implementation
uint8_t mq_decoder::decode(uint8_t label) {
  uint8_t x;
  uint16_t tmp;
  constexpr uint8_t min_C_active   = 8;
  constexpr uint32_t C_active_mask = 0xFFFF00;
  uint16_t *sk                     = &dynamic_table[1][label];
  uint16_t *Sigma_k                = &dynamic_table[0][label];
  uint32_t val                     = static_table[*Sigma_k];
  const uint16_t Sigma_mps         = (val >> 24) & 0x3F;  // static_table[0][Sigma_k];
  const uint16_t Sigma_lps         = (val >> 16) & 0x3F;  // static_table[1][Sigma_k];
  const uint16_t Xs                = val >> 31;           // static_table[2][Sigma_k];
  uint16_t p                       = static_cast<uint16_t>(val & 0xFFFF);
  uint16_t s                       = dynamic_table[1][label];

  if (s > 1) {
    printf("ERROR: mq_dec error in function decode()\n");
    throw std::exception();
  }

  A = static_cast<uint16_t>(A - p);
  if (A < p) {
    // Conditional exchange of MPS and LPS
    s = static_cast<uint16_t>(1 - s);
  }

  // Compare active region of C
  if (((C & C_active_mask) >> 8) < p) {
    x = static_cast<uint8_t>(1 - s);
    A = p;
  } else {
    x   = static_cast<uint8_t>(s);
    tmp = static_cast<uint16_t>(((C & C_active_mask) >> min_C_active) - static_cast<uint32_t>(p));
    C &= ~C_active_mask;
    C += static_cast<uint32_t>((tmp << min_C_active)) & C_active_mask;
  }
  if (A < 0x8000) {
    if (x == *sk) {
      // The x was a real MPS
      *Sigma_k = Sigma_mps;
    } else {
      // The x was a real LPS
      *sk      = *sk ^ Xs;
      *Sigma_k = Sigma_lps;
    }
  }

  while (A < 0x8000) {
    renormalize_once();
  }
  return x;
}
#endif

#if defined(CDP)
// Common Decoding Path (CDP) implementation, require a new state value D
uint8_t mq_decoder::decode(uint8_t label) {
  int32_t x;
  uint16_t Sigma_k         = dynamic_table[0][label];
  uint32_t val             = static_table[Sigma_k];
  const uint16_t Sigma_mps = (val >> 24) & 0x3F;  // static_table[0][Sigma_k];
  const uint16_t Sigma_lps = (val >> 16) & 0x3F;  // static_table[1][Sigma_k];
  const uint16_t Xs        = val >> 31;           // static_table[2][Sigma_k];
  // = p_bar (from the static table) << 8
  const int32_t p_shifted = (int32_t)((val & 0xFFFF) << 8);
  uint16_t sk             = dynamic_table[1][label];

  x = sk;  // set to MPS for now, since this is most likely
  D -= p_shifted;
  if (D < 0) {
    A += D;
    C += D;
    if (C >= 0) {
      if (A < p_shifted) {  // conditional exchange, LPS decoded
        x       = 1 - sk;
        sk      = sk ^ Xs;
        Sigma_k = Sigma_lps;
      } else {  // MPS decoded
        Sigma_k = Sigma_mps;
      }
      renormalize();
    } else {
      C += p_shifted;
      if (A < p_shifted) {  // conditional exchange, MPS decoded
        Sigma_k = Sigma_mps;
      } else {  // LPS decoded
        x       = 1 - sk;
        sk      = sk ^ Xs;
        Sigma_k = Sigma_lps;
      }
      A = p_shifted;
      renormalize();
    }
    D = A - Areg_min;
    D = (C < D) ? C : D;
    A -= D;
    C -= D;
  }
  dynamic_table[0][label] = Sigma_k;
  dynamic_table[1][label] = sk;
  return static_cast<uint8_t>(x);
}
#endif

void mq_decoder::finish() {
  // TODO: ERTERM
}
