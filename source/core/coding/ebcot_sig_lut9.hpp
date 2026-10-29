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
#include "EBCOTtables.hpp"

namespace ebcot {

// 9-bit-indexed zero-coding context LUT for the packed stripe-column word (book
// §17.1.2 figure 17.2), built once from the 8-bit sig_LUT by remapping the
// neighbourhood-bit layout:
//   8-bit sig_LUT:  TL=0 TR=1 BL=2 BR=3 T=4 B=5 L=6 R=7
//   9-bit window:   TL=0 T=1 TR=2 L=3 self=4 R=5 BL=6 B=7 BR=8
// The self bit (bit 4) is a don't-care since the context is only queried for an
// insignificant sample; all 512 entries are populated so the index never needs
// masking.  Shared by the Part 1 block decoder and encoder.
// clang-format off
struct SigLUT9 {
  uint8_t data[4][512];
  SigLUT9() {
    for (uint32_t idx9 = 0; idx9 < 512; ++idx9) {
      const uint8_t idx8 = static_cast<uint8_t>(
          ((idx9 >> 0) & 1u)                 // TL bit 0 → bit 0
          | (((idx9 >> 2) & 1u) << 1)        // TR bit 2 → bit 1
          | (((idx9 >> 6) & 1u) << 2)        // BL bit 6 → bit 2
          | (((idx9 >> 8) & 1u) << 3)        // BR bit 8 → bit 3
          | (((idx9 >> 1) & 1u) << 4)        // T  bit 1 → bit 4
          | (((idx9 >> 7) & 1u) << 5)        // B  bit 7 → bit 5
          | (((idx9 >> 3) & 1u) << 6)        // L  bit 3 → bit 6
          | (((idx9 >> 5) & 1u) << 7));      // R  bit 5 → bit 7
      for (int o = 0; o < 4; ++o) data[o][idx9] = sig_LUT[o][idx8];
    }
  }
};
// clang-format on

}  // namespace ebcot

// One table per translation unit that includes this header (C++11 has no inline variables).
static const ebcot::SigLUT9 sig_LUT9_table;
