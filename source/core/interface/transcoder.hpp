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

// EXPERIMENTAL (prototype): transcoding of a JPEG 2000 Part 1 (EBCOT/MQ) codestream into an
// HTJ2K (Part 15) codestream, in the coefficient domain.
//
// The Part 1 code-blocks are decoded to their quantisation indices and the indices are coded
// again with the HT block coder.  There is no wavelet transform, no re-quantisation and no
// colour transform, so the image a decoder reconstructs from the result is the one it
// reconstructs from the input (also for code-blocks that a lossy Part 1 encoder cut part-way
// through a bit-plane, unless transcode_partial_plane::keep or drop is chosen).  The marker segments that
// describe the image (SIZ, QCD, QCC, COM, tiling, precincts, progression order, SOP/EPH) are kept; the
// code-block style becomes HT, the CAP marker segment is added with a CPF marker segment that holds the
// input's profile (Rsiz plus one, for a transcode back), and the packets are written again.  An HT
// block is one HT set (a cleanup segment, with an optional refinement segment), not refined layer by layer
// as a Part 1 block is.  By default the layers of the input are merged into one; transcode_layers::keep
// keeps them with placeholder passes (see there).  The input's PLT marker segments become PLT marker
// segments for the output's packets; its PLM, PPM, POC and TLM marker segments are not carried over.
//
// Memory: the input and the output are held whole.  The tiles are transcoded one after the other,
// and only the tile in hand has its structures, so memory is about the input, the output and one
// tile (the whole image if it has one tile, which then takes about three to four times the input).
// With transcode_options::input_padded the input is read where it is, else it is copied first.  A
// codestream of 4 GB or more is not supported (the reader keeps 32-bit positions).
//
// The input is a raw codestream or a JP2/JPH file (found by its signature box).  With
// transcode_options::jph_output the result is a JPH file: the boxes of a JP2 input (image header,
// colour, resolution, XML, ...) are kept, the file type box gets the brand 'jph '; a raw codestream
// gets boxes made the way the encoder makes them.
//
// RGN (region of interest, Maxshift): the RGN marker segments are kept and the shifted indices are
// coded as they are.  Reversible streams reconstruct exactly.  For a lossy stream the Part 1 decoder
// takes every bit-plane of a region-of-interest block as decoded and its mid-point at half a step,
// so those blocks are coded down to the last bit with that half step added to the coefficients of
// the region; the HT decoder's own mid-point then leaves the result within about 2^-roi of a step per
// coefficient of the image the Part 1 decoder gives (a few levels of 4095 at most in the tests).
//
// Limits (the function throws, with a message, when the input has one): the input must use the
// Part 1 block coder (not HT), have no DFS or ATK marker segments (Part 2), and be less than 4 GB.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include "decoder.hpp"  // OPENHTJ2K_EXPORT

namespace open_htj2k {
// A Part 1 code-block is coded down to some pass.  When that is a cleanup pass, every sample
// is known down to that bit-plane, and the HT block coder codes the block at that plane: the
// HT decoder's bin centre then equals the Part 1 mid-point, so the reconstruction is the same
// and the bit-planes below the cut cost nothing.  When the last pass is a SigProp or MagRef
// pass, the block ends part-way through a bit-plane, which one HT cleanup pass cannot hold.
enum class transcode_partial_plane {
  keep,   // code the indices to the last bit, partial plane included (reconstruction off by half a step)
  drop,   // code the block at its last complete cleanup plane and drop the partial plane
  refine  // (default) code the block at its last complete cleanup plane and the partial plane with HT
          // SigProp (and MagRef) passes, the way the Part 1 passes were cut: the reconstruction is exact
};

// The quality layers of the output.
//
// keep follows the WG1 guideline on placeholder passes (WG1 N101237, "Guideline on Placeholder Passes and
// Multiple HT Sets in HTJ2K codestreams", section 6).  The output has the layers of the input, and each
// packet adds to each code-block the number of coding passes it added in the input: the passes before the
// block's last Part 1 cleanup pass become placeholder passes (ISO/IEC 15444-15, B.3), which carry no
// bytes, and the HT cleanup, SigProp and MagRef passes go in the packets of the Part 1 passes they come
// from.  The layer boundaries of the input are therefore kept in the packet headers, and the output can be
// transcoded back into a Part 1 codestream with the same layers.  In the first packet that includes a
// block, a multiple of three placeholder passes is replaced by as many more missing MSBs, leaving at most
// two, as the guideline recommends; when that would leave the packet with no pass of the block, the
// number of missing MSBs of the input is kept instead.  The output stays SINGLEHT (one HT set per block).
// It decodes to the same image as the merged output; a decode of fewer layers is not the gradual
// refinement of the Part 1 file, since a block is decoded only from the layer that holds its cleanup pass
// on.  keep requires transcode_partial_plane::refine, and refuses a lossy stream with a region of
// interest (whose blocks are coded below their last pass).  A block whose passes decode to zero has
// nothing to code and is left out, as in merge.
enum class transcode_layers {
  merge,  // (default) one quality layer
  keep    // the layers of the input, with placeholder passes
};

// What the code-blocks that had coding passes were found to end with.
struct transcode_stats {
  uint64_t blocks  = 0;  // code-blocks with coding passes
  uint64_t full    = 0;  // decoded to the last bit-plane, ending on a cleanup pass
  uint64_t coarser = 0;  // ending on a cleanup pass above the last bit-plane
  uint64_t partial = 0;  // ending part-way through a bit-plane
  uint64_t refined = 0;  // of those, the blocks coded with HT refinement passes (refine mode)
};

// The colour space a JPH file written for a raw codestream declares (its colour specification box).
enum class transcode_colour_space {
  automatic,  // sRGB for three or more components (default), greyscale for one or two
  rgb,        // sRGB (three or four components: colour, then an optional alpha channel)
  ycc         // sYCC, for components that are Y, Cb, Cr
};

struct transcode_options {
  // True if the 16 bytes after the input buffer (in + in_len) can be read.  The transcoder then reads the
  // input where it is; else it takes a copy of it first, which costs memory the size of the input.
  bool input_padded = false;
  transcode_colour_space jph_colour_space =
      transcode_colour_space::automatic;  // JPH output of a raw codestream
  transcode_partial_plane partial_plane = transcode_partial_plane::refine;
  transcode_layers layers               = transcode_layers::merge;
  transcode_stats *stats                = nullptr;  // optional: receives the block statistics
  bool jph_output      = false;  // write a JPH file (ISO/IEC 15444-15) instead of the codestream alone
  unsigned num_threads = 0;      // code-block worker threads; 0 = all hardware threads, 1 = none
};

// Transcodes the Part 1 codestream [in, in + in_len) into the HTJ2K codestream `out`.
// Returns the size of `out`.  Throws std::exception on input it cannot transcode.
OPENHTJ2K_EXPORT size_t transcode_part1_to_htj2k(const uint8_t *in, size_t in_len,
                                                 std::vector<uint8_t> &out);
OPENHTJ2K_EXPORT size_t transcode_part1_to_htj2k(const uint8_t *in, size_t in_len,
                                                 std::vector<uint8_t> &out,
                                                 const transcode_options &options);

// EXPERIMENTAL (prototype): the other direction, an HTJ2K codestream into a Part 1 codestream.
//
// Every code-block is decoded to its quantisation indices (an HT block with its cleanup, SigProp and
// MagRef passes; a Part 1 block of a mixed codestream with its own passes) and coded again with the
// Part 1 block coder, down to the same coding pass, so a Part 1 decoder reconstructs the image an
// HTJ2K decoder reconstructs from the input.  The number of missing MSBs of a block is that of its
// indices (an HT block may signal fewer zero bit-planes or more), and its coding passes are numbered
// from there.
//
// The quality layers of the input are kept by default (WG1 N101237, section 6): a packet adds to each
// code-block the passes it added in the input.  Placeholder passes become the Part 1 passes they stand
// for, and the passes above the block's first packet's (those an HT encoder turned into missing MSBs)
// go into that packet.  A cut inside a codeword segment keeps the shortest prefix of the coded data
// that decodes the passes before it, so every layer decodes to what the input's layers decode to in a
// Part 1 decoder, and an HTJ2K codestream made by transcode_part1_to_htj2k() with
// transcode_layers::keep goes back to the layers of its Part 1 source.
//
// The marker segments are kept as in transcode_part1_to_htj2k(); the code-block style becomes that
// of the Part 1 coder, the CAP marker segment is removed, and bit 14 of Rsiz is cleared.  The output
// is a raw codestream; the input is a raw codestream or a JPH/JP2 file.
//
// Limits (the function throws, with a message): the input must be an HTJ2K codestream (a code-block
// style with the HT bit, or a CAP marker segment for Part 15) with at most one HT set per code-block
// (not MULTIHT), no Part 2 marker segments and no capabilities besides Part 15 in the CAP marker
// segment, and be less than 4 GB.  A Part 1 code-block cannot hold more than 65535 bytes here.
struct transcode_to_part1_options {
  // As in transcode_options.
  bool input_padded = false;
  // keep (default): the layers of the input; merge: one quality layer.
  transcode_layers layers = transcode_layers::keep;
  // The Part 1 code-block style switches of the output (bits 0-5 of SPcod: BYPASS 0x01, RESET 0x02,
  // RESTART 0x04, CAUSAL 0x08, ERTERM 0x10, SEGMARK 0x20).  -1 (default): those bits of the input's
  // code-block style, which an HT code-block uses only for CAUSAL and a Part 1 one of a mixed
  // codestream for all of them.
  int32_t code_block_style = -1;
  unsigned num_threads     = 0;  // code-block worker threads; 0 = all hardware threads, 1 = none
};

OPENHTJ2K_EXPORT size_t transcode_htj2k_to_part1(const uint8_t *in, size_t in_len,
                                                 std::vector<uint8_t> &out);
OPENHTJ2K_EXPORT size_t transcode_htj2k_to_part1(const uint8_t *in, size_t in_len,
                                                 std::vector<uint8_t> &out,
                                                 const transcode_to_part1_options &options);

// True if [in, in + in_len), a raw codestream or a JP2/JPH file, is an HTJ2K codestream: its main
// header has a CAP marker segment for Part 15 or a code-block style (COD or COC) with the HT bit.
// Throws std::exception if its main header cannot be read.
OPENHTJ2K_EXPORT bool transcode_input_is_htj2k(const uint8_t *in, size_t in_len);
}  // namespace open_htj2k
