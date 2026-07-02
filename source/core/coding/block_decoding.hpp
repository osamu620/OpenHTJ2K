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
#include "coding_units.hpp"

void j2k_decode(j2k_codeblock *block, uint8_t ROIshift);
bool htj2k_decode(j2k_codeblock *block, uint8_t ROIshift);

// Decode n HT codeblocks, interleaving the serial step-1 (MEL/VLC/UVLC)
// dependency chains of same-sized neighbours when the active ISA provides a
// batched kernel (htj2k_dec_batch_lanes > 1).  Caller guarantees every block
// is an HT block with num_passes > 0 and non-null compressed data.
// results[i] receives what htj2k_decode(blocks[i], ROIshift) would have
// returned; the decoded output is byte-identical to per-block decoding.
// Returns true iff every block decoded successfully.
bool htj2k_decode_batch(j2k_codeblock *const *blocks, uint32_t n, uint8_t ROIshift, bool *results);

// Number of step-1 lanes the active ISA's batched kernel interleaves;
// 1 means htj2k_decode_batch is a plain per-block loop (link-time constant —
// SIMD dispatch is compile-time in this project).
extern const uint32_t htj2k_dec_batch_lanes;
