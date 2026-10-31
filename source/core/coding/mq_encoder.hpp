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
#include "mq_decoder.hpp"  // static_table: Qe, NMPS, NLPS and SWITCH packed per state

// MQ arithmetic encoder of ISO/IEC 15444-1 Annex C (the CODEMPS / CODELPS /
// RENORME / BYTEOUT / FLUSH procedures of C.2).  The coding passes call
// encode() once per decision; flush() or erterm() terminates the codeword
// segment and returns its length in bytes.
//
// The byte "before the start" of the standard's BYTEOUT (B at BP = BPST) is kept
// in a register, never in the buffer: a carry into it is simply discarded, and
// the output buffer is never written before its first byte.
class mq_encoder {
 public:
  explicit mq_encoder(uint8_t *out) { init(out); }

  // INITENC (C.2.8): start a new codeword segment at out.  The context states are
  // kept, so a segment that follows a termination continues with them.
  void init(uint8_t *out) {
    A   = 0x8000;
    C   = 0;
    CT  = 12;
    B   = 0;
    BP  = -1;
    buf = out;
  }

  // Table D.7 initial states: all contexts at state 0 except the all-insignificant
  // zero-coding context (4), run-length (3) and uniform (46).
  void init_states_for_all_contexts() {
    for (int i = 0; i < 19; ++i) {
      state[i] = 0;
      mps[i]   = 0;
    }
    state[0]  = 4;
    state[17] = 3;
    state[18] = 46;
  }

  // Code decision d (0 or 1) in context label.
  inline void encode(uint32_t label, uint32_t d) {
    const uint32_t val = static_table[state[label]];
    const uint32_t qe  = val & 0xFFFFu;
    A -= qe;
    if (d == mps[label]) {
      // CODEMPS
      if ((A & 0x8000u) == 0) {
        if (A < qe) {
          A = qe;
        } else {
          C += qe;
        }
        state[label] = static_cast<uint8_t>((val >> 24) & 0x3Fu);  // NMPS
        renormalize();
      } else {
        C += qe;
      }
    } else {
      // CODELPS
      if (A < qe) {
        C += qe;
      } else {
        A = qe;
      }
      if (val >> 31) mps[label] ^= 1u;                           // SWITCH
      state[label] = static_cast<uint8_t>((val >> 16) & 0x3Fu);  // NLPS
      renormalize();
    }
  }

  // Bytes of the segment that later coding cannot change: those before the byte still
  // held in B (a carry can still reach B, never the bytes before it).
  uint32_t settled_bytes() const { return BP > 0 ? static_cast<uint32_t>(BP) : 0u; }

  // Writes the upper end of the current interval, C + A, as segment bytes from
  // settled_bytes() on (the bytes the coder has yet to write) and returns where they
  // end.  Returns 0 when the upper end carries out of the segment's first byte (it is
  // then 1, above every codeword).  Used on a copy of the coder.
  //
  // A codeword segment cut after L bytes is decoded as if 1 bits followed (D.4.1), so a
  // cut decodes every symbol coded so far iff the cut bytes followed by 1s stay below
  // that upper end.  With the codeword's own bytes, the shortest such cut ends one byte
  // after the first byte where the codeword is below the upper end's bytes (a 0xFF
  // followed by a byte above 0x7F encodes a carry into the 0xFF; the cut can then be a
  // byte longer than needed, never too short).
  uint32_t upper_end() {
    C += A;
    C <<= CT;
    if (BP < 0 && C >= 0x8000000u) return 0;  // a carry into the byte before the start
    byteout();
    while (C != 0) {
      C <<= CT;
      byteout();
    }
    emit();
    return static_cast<uint32_t>(BP);
  }

  // FLUSH (C.2.9).  The optional trailing 0xFF (and the 0xFFAC marker) are not
  // written: a decoder reading past the end of the segment sees 0xFF bytes.
  inline uint32_t flush() {
    const uint32_t tempc = C + A;  // SETBITS
    C |= 0xFFFFu;
    if (C >= tempc) C -= 0x8000u;
    C <<= CT;
    byteout();
    C <<= CT;
    byteout();
    if (B != 0xFFu) emit();  // keep the last byte unless it is a 0xFF
    return static_cast<uint32_t>(BP);
  }

  // Predictable termination (D.4.2), required when ERTERM is signalled: push out
  // only the k = (11 - CT) + 1 bits of C the decoder needs, then the byte buffer
  // unless it holds 0xFF.  The bytes are fixed by the coded symbols alone.
  inline uint32_t erterm() {
    int32_t k = 12 - CT;
    while (k > 0) {
      C <<= CT;
      CT = 0;
      byteout();
      k -= CT;
    }
    if (B != 0xFFu) byteout();
    return static_cast<uint32_t>(BP);
  }

 private:
  uint32_t A;  // interval
  uint32_t C;  // code register: bits 0..15 active, 16..18 spacer, 19..26 byte out, 27 carry
  int32_t CT;  // shifts until the next byte out
  uint32_t B;  // the byte at BP, not yet stored
  int32_t BP;  // index of B in buf; -1 for the byte before the start
  uint8_t *buf;
  uint8_t state[19];
  uint8_t mps[19];

  // BP = BP + 1: store the byte at BP (if it is inside the buffer) and advance.
  inline void emit() {
    if (BP >= 0) buf[BP] = static_cast<uint8_t>(B);
    ++BP;
  }

  inline void byteout() {
    if (B == 0xFFu) {
      stuff();
    } else if (C < 0x8000000u) {
      nostuff();
    } else {
      ++B;  // carry into the previous byte
      if (B == 0xFFu) {
        C &= 0x7FFFFFFu;
        stuff();
      } else {
        nostuff();
      }
    }
  }
  inline void stuff() {  // after a 0xFF only 7 bits go out, so the next byte is < 0x90
    emit();
    B = C >> 20;
    C &= 0xFFFFFu;
    CT = 7;
  }
  inline void nostuff() {
    emit();
    B = C >> 19;
    C &= 0x7FFFFu;
    CT = 8;
  }

  inline void renormalize() {
    do {
      if (CT == 0) byteout();
      A <<= 1;
      C <<= 1;
      --CT;
    } while ((A & 0x8000u) == 0);
  }
};

// Raw bit packer for the passes the selective arithmetic coding bypass (D.6)
// takes out of the MQ coder: bits go into bytes MSB first, and the byte after an
// 0xFF carries only 7 bits under a stuffed 0 MSB.
class raw_encoder {
 public:
  void init(uint8_t *out) {
    buf = out;
    pos = 0;
    c   = 0;
    ct  = 8;
  }

  inline void put(uint32_t bit) {
    c = (c << 1) | bit;
    if (--ct == 0) {
      buf[pos++] = static_cast<uint8_t>(c);
      ct         = (c == 0xFFu) ? 7 : 8;
      c          = 0;
    }
  }

  // Bytes written so far; later bits never change them.
  uint32_t settled_bytes() const { return pos; }
  // Bytes that hold every bit put so far: the written ones and a partial byte.
  uint32_t needed_bytes() const {
    const bool after_ff = pos > 0 && buf[pos - 1] == 0xFFu;
    return pos + ((ct < (after_ff ? 7 : 8)) ? 1u : 0u);
  }

  // Pads the last byte to the byte boundary and returns the segment length; the
  // segment never ends in 0xFF.  With ERTERM the padding is 0101... (D.6).
  // Otherwise it is all 1s and a final 0xFF is dropped: the decoder extends the
  // segment with 0xFF bytes (D.4.1), so it reads the same bits.
  inline uint32_t terminate(bool erterm) {
    const bool after_ff = pos > 0 && buf[pos - 1] == 0xFFu;
    if (ct < (after_ff ? 7 : 8)) {  // a partial byte is pending
      c = (c << ct) | (erterm ? (0x55u >> (8 - ct)) : ((1u << ct) - 1));
      if (erterm || c != 0xFFu) buf[pos++] = static_cast<uint8_t>(c);
    } else if (after_ff) {
      if (erterm)
        buf[pos++] = 0x2A;  // the stuffed 0, then 0101010
      else
        --pos;
    }
    return pos;
  }

 private:
  uint8_t *buf = nullptr;
  uint32_t pos = 0;
  uint32_t c   = 0;  // bits of the byte being assembled
  int32_t ct   = 8;  // bits still free in that byte
};
