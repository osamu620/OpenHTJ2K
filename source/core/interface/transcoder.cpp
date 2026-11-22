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

#include <algorithm>
#include <atomic>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>
#include <cstring>
#include <memory>
#include <vector>
#include "transcoder.hpp"
#include "jph.hpp"
#include "coding_units.hpp"
#include "block_decoding.hpp"
#include "message_internal.hpp"
#ifdef OPENHTJ2K_THREAD
  #include "ThreadPool.hpp"
#endif

namespace open_htj2k {
namespace {

// A tile whose tile-parts hold fewer bytes than this is transcoded on the calling thread only.  Measured
// on 2 cores with tiles of 64 x 64 to 256 x 256 at 0.5 to 4 bits per sample: the pool lost up to 5 times
// below about 8 KB per tile and won by 10 to 40% above it.
constexpr uint64_t kInlineTileBytes = 8192;

// Bits 0-4 of Ccap15 for a magnitude bound B (ISO/IEC 15444-15, A.3.7); see encoder.cpp.
uint16_t magb_field(int32_t bound) {
  if (bound <= 8) return 0;
  if (bound <= 27) return static_cast<uint16_t>(bound - 8);
  if (bound <= 71) return static_cast<uint16_t>(19 + (bound - 27 + 3) / 4);
  return 31;
}

// The code-block style of the output: HT, with the input's vertically causal context switch
// (CAUSAL), which the HT SigProp pass follows as the Part 1 one does.  The other Part 1 switches
// do not apply to HT code-blocks.
uint8_t ht_style(uint8_t part1_style) { return static_cast<uint8_t>(HT | (part1_style & CAUSAL)); }

// Per-thread scratch of the block transcoder.
struct scratch_t {
  std::vector<int32_t> dec;   // decoded Part 1 samples: sign in bit 31, magnitude below
  std::vector<uint32_t> ctx;  // Part 1 context words
  std::vector<int32_t> enc;   // the same samples in the form the HT encoder takes
};

// Decodes the Part 1 code-block `bin` to its quantisation indices and codes them with the HT
// block coder into `bout`, the same code-block of the output tile.
void transcode_block(j2k_codeblock *bin, j2k_codeblock *bout, uint8_t roi, scratch_t &sc,
                     const transcode_options &opt, transcode_stats &stats) {
  // A region-of-interest shift (RGN) makes the indices `roi` bit-planes deeper; the block coder
  // of the output counts them, and the RGN marker segment is kept for the decoder to undo them.
  bout->add_ROIshift(roi);
  const uint32_t W      = bin->size.x;
  const uint32_t H      = bin->size.y;
  const uint32_t stride = round_up(W, 8U);
  const uint32_t QH     = round_up(H, 8U);

  uint32_t or_val = 0;
  uint32_t p_c    = 0;      // bit-plane (above the index LSB) the HT cleanup pass is coded at
  bool refine     = false;  // the partial plane goes into HT SigProp (and MagRef) passes
  bool sp_only    = false;  // ... and the Part 1 block ended after its SigProp pass
  if (bin->num_passes > 0 && bin->get_compressed_data() != nullptr) {
    sc.dec.assign(static_cast<size_t>(stride) * QH, 0);
    sc.ctx.assign(static_cast<size_t>(QH / 4 + 2) * (stride + 2), 0);
    bin->sample_buf            = sc.dec.data();
    bin->blksampl_stride       = stride;
    bin->block_contexts        = sc.ctx.data();
    bin->block_contexts_stride = stride + 2;
    j2k_decode_indices(bin, roi);

    // The decoder holds a magnitude in Q(31 - M_b): its index is the sample shifted down by
    // pLSB = 31 - M_b (j2k_dequant does the same for the reversible transform).  A marker below
    // that bit is the half-LSB mid-point and is dropped; one at or above it belongs to a sample
    // a truncated stream decoded to a coarser bit-plane, and stays in the index.
    // The HT encoder takes 0x80000000 | ((m - 1) << 1) | sign for an index m > 0, in a buffer
    // padded to multiples of 8 in both directions.
    const int32_t Mb   = static_cast<int32_t>(bin->get_Mb()) + static_cast<int32_t>(roi);
    const int32_t pLSB = 31 - Mb;
    if (pLSB < 0) open_htj2k::detail::fail("transcoder: a sub-band has more than 31 magnitude bit-planes.");

    // The passes are the cleanup pass of the first bit-plane, then SigProp, MagRef and cleanup
    // for each further one.  After n passes the cleanup passes of (n + 2) / 3 bit-planes are
    // done, so every sample is known down to the bit-plane `last_plane` (an index bit-plane,
    // 0 being the LSB); a block that ends on its SigProp or MagRef pass has some samples known
    // one plane further.
    const uint32_t n      = bin->num_passes;
    const int32_t planes  = static_cast<int32_t>((n + 2) / 3);
    const int32_t last_pl = Mb - static_cast<int32_t>(bin->num_ZBP) - planes;
    if (last_pl < 0) open_htj2k::detail::fail("transcoder: a code-block has more passes than bit-planes.");
    const bool partial = ((n - 1) % 3) != 0;
    ++stats.blocks;
    if (partial) {
      ++stats.partial;
    } else if (last_pl == 0) {
      ++stats.full;
    } else {
      ++stats.coarser;
    }
    // Ending on a cleanup pass, or dropping the partial plane, or carrying it in HT refinement
    // passes: the HT cleanup pass is coded at that plane.  Otherwise (keep) every decoded bit is
    // coded, down to plane 0.
    if (!partial || opt.partial_plane != transcode_partial_plane::keep)
      p_c = static_cast<uint32_t>(last_pl);
    // The HT set of cleanup at plane p, SigProp and MagRef of plane p - 1 is the Part 1 sequence
    // cleanup, SigProp, MagRef of the plane below.  A block that ended after SigProp has the
    // set cut there (the refinement segment up to the SigProp bits).
    refine  = partial && opt.partial_plane == transcode_partial_plane::refine && p_c >= 1;
    sp_only = refine && (n % 3 == 2);

    // A lossy block with a region-of-interest shift is reconstructed by the Part 1 decoder with
    // every bit-plane counted as decoded and the mid-point at half a step of the index m, whether
    // the block was cut or not (a coefficient of the region is m * 2^roi in the stream).  The HT
    // decoder puts the mid-point at half of the last coded bit of the shifted value instead, which
    // is 2^-roi of a step for the coefficients of the region.  So the indices are coded down to
    // the last bit, and a coefficient of the region gets the bit below its lowest one (roi - 1)
    // set, which is that half step (the HT mid-point adds 2^-(roi+1) of a step on top of it).
    const bool roi_lossy = roi > 0 && bin->transformation != 1;
    if (roi_lossy && opt.layers == transcode_layers::keep) {
      // Coded below their last pass, these blocks have no place for their passes in the layers.
      open_htj2k::detail::fail(
          "transcoder: -layers keep does not support a lossy stream with a region of interest (RGN).");
    }
    if (roi_lossy) {
      p_c     = 0;
      refine  = false;
      sp_only = false;
    }

    // Fills the encoder's samples from the decoded buffer: the magnitude is shifted down by
    // `sh` bits (the bits below, the markers among them, are gone) and up again by `up`.
    // Returns the largest magnitude.
    auto fill = [&](int32_t sh, uint32_t up) -> uint32_t {
      uint32_t mx = 0;
      or_val      = 0;
      sc.enc.assign(static_cast<size_t>(stride) * QH, 0);
      for (uint32_t y = 0; y < H; ++y) {
        const uint32_t *sp =
            reinterpret_cast<const uint32_t *>(sc.dec.data()) + static_cast<size_t>(y) * stride;
        int32_t *dp = sc.enc.data() + static_cast<size_t>(y) * stride;
        for (uint32_t x = 0; x < W; ++x) {
          uint32_t mu = ((sp[x] & 0x7FFFFFFFu) >> sh) << up;
          if (roi_lossy && mu >= (1u << roi)) mu |= 1u << (roi - 1);
          or_val |= mu;
          mx    = std::max(mx, mu);
          dp[x] = mu ? static_cast<int32_t>(0x80000000u | ((mu - 1) << 1) | (sp[x] >> 31)) : 0;
        }
      }
      return mx;
    };

    if (refine) {
      // Index bit-plane q = p_c - 1 of the decoder's buffer is bit pLSB + q.  The samples keep
      // their bits down to q, which the SigProp and MagRef passes code; the Part 1 mid-point
      // markers below q are cleared.
      const uint32_t q  = p_c - 1;
      const uint32_t mx = fill(pLSB + static_cast<int32_t>(q), q);
      // The HT encoder keeps the cleanup plane below the block's top bit-plane (and within M_b);
      // a block that does not allow p_c cannot be coded as a set, and drops the plane instead.
      const uint32_t msb = mx ? 32u - count_leading_zeros(mx) : 0u;
      if (p_c + 1 > std::min(msb, static_cast<uint32_t>(Mb))) {
        refine  = false;
        sp_only = false;
      }
    }
    if (refine) {
      ++stats.refined;
    } else {
      // Index bit-plane p_c of the decoder's buffer is bit pLSB + p_c; the markers below it (the
      // mid-point of the last plane) are shifted out.
      fill(pLSB + static_cast<int32_t>(p_c), 0);
    }
    bin->sample_buf     = nullptr;
    bin->block_contexts = nullptr;
  }

  bout->sample_buf       = or_val ? sc.enc.data() : nullptr;
  bout->blksampl_stride  = stride;
  bout->pre_quantized    = true;
  bout->pre_or_val       = or_val ? 1 : 0;
  bout->ht_refine_planes = refine ? static_cast<uint8_t>(p_c) : 0;
  bout->refsegment       = false;
  if (refine) {
    // Cleanup at plane p_c, then SigProp and MagRef of plane p_c - 1 in one refinement segment;
    // the encoder signals num_ZBP = M_b - 1 - p_c itself.
    ht_refinement_info rinfo;
    htj2k_encode_refined(bout, 0, &rinfo);
    bout->sample_buf       = nullptr;
    bout->ht_refine_planes = 0;
    const bool keep_layers = opt.layers == transcode_layers::keep;
    if (sp_only && bout->num_passes == 3) {
      // The Part 1 block ended after its SigProp pass: keep the cleanup segment and the
      // refinement segment as far as the SigProp bits, whose last byte stands alone (the cut
      // rate control makes).  Without SigProp bits it is the cleanup pass alone, unless the
      // layers are kept: the SigProp pass then stays, with no bytes, so that its packet still
      // adds it (an empty refinement segment is not decoded, ISO/IEC 15444-15, B.3).
      const uint32_t Lref     = bout->pass_length[2];
      const uint32_t Lcup     = bout->length - Lref;
      const uint8_t passes    = (rinfo.Lsp != 0 || keep_layers) ? 2 : 1;
      bout->layer_start[0]    = 0;
      bout->layer_passes[0]   = passes;
      bout->num_passes        = passes;
      bout->pass_length[0]    = Lcup;
      bout->pass_length[1]    = (passes == 2) ? rinfo.Lsp : 0;
      bout->pass_length[2]    = 0;
      bout->pass_length_count = passes;
      bout->length            = Lcup + ((passes == 2) ? rinfo.Lsp : 0);
      if (rinfo.Lsp != 0) bout->get_compressed_data()[Lcup + rinfo.Lsp - 1] = rinfo.sp_tail;
    } else if (keep_layers && bout->num_passes == 3) {
      // The SigProp and MagRef passes may be in different layers: the SigProp pass gets the
      // bytes of the segment up to its last bit (which a decode without the MagRef pass reads),
      // the MagRef pass the rest.  Together they are the one refinement segment.
      const uint32_t Lref  = bout->pass_length[2];
      bout->pass_length[1] = rinfo.Lsp;
      bout->pass_length[2] = Lref - rinfo.Lsp;
    }
    return;
  }
  encode_codeblock(bout, 0);
  bout->sample_buf = nullptr;
  // The cleanup pass codes the magnitudes of the shifted indices, whatever plane they stand for;
  // the plane only enters through the number of missing MSBs the packet header signals,
  // M_b - 1 - p (ISO/IEC 15444-15, 7.6; the same as the encoder's refinement path sets).
  if (p_c > 0 && bout->num_passes > 0) bout->num_ZBP = static_cast<uint8_t>(bout->get_Mb() - 1 - p_c);
}

// The coding passes each packet of the input added to a code-block, in the order of its layers:
// (layer, passes) for the layers that added some.
struct layer_log {
  uint8_t total = 0;  // the passes recorded so far
  std::vector<std::pair<uint16_t, uint8_t>> runs;
};

// transcode_layers::keep (WG1 N101237, 6): gives the HT code-block `bout`, coded from the Part 1
// block `bin`, the passes of each layer that the input had, with the passes before the last Part 1
// cleanup pass as placeholder passes.
void place_layers(const j2k_codeblock *bin, j2k_codeblock *bout, const layer_log *log) {
  const uint16_t Q = bout->num_layers;
  for (uint16_t l = 0; l < Q; ++l) bout->layer_passes[l] = 0;
  // A block whose passes decode to zero has nothing to code: it is left out, as in merge.
  const uint32_t n_ht = bout->num_passes;
  if (n_ht == 0) return;
  // The HT cleanup pass stands for the last Part 1 cleanup pass, z0; the SigProp and MagRef passes
  // for the passes after it.
  const uint32_t Z  = bin->num_passes;
  const uint32_t z0 = 3 * ((Z - 1) / 3);
  if (Z - z0 != n_ht || log == nullptr)
    open_htj2k::detail::fail("transcoder: the HT passes of a code-block do not match its Part 1 passes.");
  // The packet header gave each layer's passes; those of a block that a cut packet left without
  // data were dropped at the end, and so are dropped here from the last layers.
  uint32_t left = Z;
  for (const auto &run : log->runs) {
    if (run.first >= Q) open_htj2k::detail::fail("transcoder: a packet of a layer the tile does not have.");
    const uint32_t w              = std::min<uint32_t>(run.second, left);
    bout->layer_passes[run.first] = static_cast<uint8_t>(bout->layer_passes[run.first] + w);
    left -= w;
  }
  if (left != 0) open_htj2k::detail::fail("transcoder: the passes of a code-block were not all recorded.");

  // S_blk = P_b + P0, the bit-planes above the cleanup pass, is what the HT encoder signalled.
  const uint32_t P0 = z0 / 3;
  if (bout->num_ZBP != bin->num_ZBP + P0)
    open_htj2k::detail::fail("transcoder: the cleanup pass of a code-block is not at its last Part 1 one.");
  // The first packet with passes of the block: 3k of its placeholder passes become k more missing
  // MSBs, leaving at most two (WG1 N101237, 6.1).  Should that leave the packet without a pass of
  // the block, P_b stays that of the input (which the guideline allows as well).
  uint16_t f = 0;
  while (bout->layer_passes[f] == 0) ++f;
  uint32_t k = std::min<uint32_t>(bout->layer_passes[f], z0) / 3;
  if (bout->layer_passes[f] == 3 * k) k = 0;
  bout->layer_passes[f] = static_cast<uint8_t>(bout->layer_passes[f] - 3 * k);
  bout->num_ZBP         = static_cast<uint8_t>(bin->num_ZBP + k);
  const uint32_t phld   = 3 * (P0 - k);

  // The placeholder passes go in front of the HT passes, with no bytes.
  const uint32_t total = phld + n_ht;
  if (total > 128) open_htj2k::detail::fail("transcoder: a code-block has more than 128 coding passes.");
  bout->ensure_pass_capacity(static_cast<uint8_t>(total));
  for (uint32_t j = n_ht; j-- > 0;) bout->pass_length[phld + j] = bout->pass_length[j];
  for (uint32_t j = 0; j < phld; ++j) bout->pass_length[j] = 0;
  bout->pass_length_count = static_cast<uint8_t>(total);
  bout->num_passes        = static_cast<uint8_t>(total);
  bout->ht_phld_passes    = static_cast<uint8_t>(phld);
}

/********************************************************************************
 * HTJ2K -> Part 1
 *******************************************************************************/

// The Part 1 code-block style of the output: the switches given, or (style < 0) those of the input's
// style (bits 0-5; an HT code-block uses CAUSAL only, a Part 1 one of a mixed codestream all of them).
uint8_t part1_style(uint8_t input_style, int32_t style) {
  return static_cast<uint8_t>((style >= 0) ? (style & 0x3F) : (input_style & 0x3F));
}

// Per-thread scratch of the HT -> Part 1 block transcoder.
struct p1_scratch_t {
  std::vector<uint8_t> data;    // a copy of the input block's bytes (the HT decoder writes into them)
  std::vector<int32_t> dec;     // decoded samples: sign in bit 31, magnitude below
  std::vector<uint8_t> states;  // HT decoder states
  std::vector<uint32_t> ctx;    // Part 1 decoder contexts (a Part 1 block of a mixed codestream)
  std::vector<int32_t> enc;     // the indices in the form the encoder takes
  std::vector<uint32_t> cut;    // the passes of the output block up to each layer
  ht_rd_info rd;                // the encoder's record of every pass (lengths, prefixes)
};

// A code-block of the output with no passes; it keeps the largest number of missing MSBs, so that its
// tag-tree parents do not drop.
void leave_out(j2k_codeblock *b) {
  for (uint16_t l = 0; l < b->num_layers; ++l) {
    b->layer_passes[l] = 0;
    b->layer_start[l]  = 0;
  }
  b->num_passes        = 0;
  b->pass_length_count = 0;
  b->length            = 0;
  b->num_ZBP           = static_cast<uint8_t>(b->get_Mb() - 1);
}

// Decodes the code-block `bin` of an HTJ2K codestream (an HT block, or a Part 1 block of a mixed one) to
// its quantisation indices, and codes them with the Part 1 block coder into `bout`, the same code-block of
// the output tile, down to the coding pass the input ends with.  `log` holds the passes each packet of the
// input added (null for one output layer).
void part1_block(j2k_codeblock *bin, j2k_codeblock *bout, uint8_t roi, const layer_log *log,
                 p1_scratch_t &sc) {
  // As in transcode_block(): the indices are `roi` bit-planes deeper, and the RGN marker segment stays.
  bout->add_ROIshift(roi);
  const int32_t Mb = static_cast<int32_t>(bin->get_Mb()) + static_cast<int32_t>(roi);
  if (Mb != static_cast<int32_t>(bout->get_Mb()) || Mb > 31)
    open_htj2k::detail::fail("transcoder: a sub-band has more than 31 magnitude bit-planes.");
  const bool is_ht = (bin->Cmodes & HT) != 0;
  // An HT block with placeholder passes only (no bytes) decodes to zero.
  if (bin->num_passes == 0 || bin->get_compressed_data() == nullptr || (is_ht && bin->length == 0)) {
    leave_out(bout);
    return;
  }

  const uint32_t W      = bin->size.x;
  const uint32_t H      = bin->size.y;
  const uint32_t stride = round_up(W, 8U);
  const uint32_t QH     = round_up(H, 8U);
  sc.dec.assign(static_cast<size_t>(stride) * (QH + 2), 0);
  bin->sample_buf      = sc.dec.data();
  bin->blksampl_stride = stride;

  // Z_in: the passes of the input block, numbered as Part 1 numbers them from the cleanup pass of the
  // bit-plane below its missing MSBs (an HT block's placeholder passes included), that the decoded
  // samples hold.
  uint32_t Z_in = bin->num_passes;
  if (is_ht) {
    // The placeholder passes, counted as the decoder counts them (ISO/IEC 15444-15, B.3): the
    // passes before the first one with bytes, in sets of three.
    uint32_t P0 = 0;
    if (bin->num_passes > 3) {
      while (P0 < bin->pass_length_count && bin->pass_length[P0] == 0) ++P0;
      P0 /= 3;
    }
    if (bin->num_passes < 3 * P0 + 1)
      open_htj2k::detail::fail("transcoder: an HT code-block has placeholder passes only, but bytes.");
    if (bin->num_passes - 3 * P0 > 3)
      open_htj2k::detail::fail("transcoder: a code-block with more than one HT set is not supported.");
    const int32_t p = Mb - 1 - static_cast<int32_t>(bin->num_ZBP + P0);  // the cleanup pass's bit-plane
    if (p < 0) open_htj2k::detail::fail("transcoder: an HT code-block has more passes than bit-planes.");
    // The HT decoder changes the bytes it reads and may read a few past them: it reads a copy.
    sc.data.assign(static_cast<size_t>(bin->length) + 64, 0);
    std::memcpy(sc.data.data(), bin->get_compressed_data(), bin->length);
    bool pooled         = false;
    uint8_t *const orig = bin->detach_compressed_data(pooled);
    bin->attach_compressed_data(sc.data.data(), true);  // not freed with the code-block
    sc.states.assign(static_cast<size_t>(stride + 2) * (QH + 2), 0);
    bin->block_states    = sc.states.data();
    bin->blkstate_stride = stride + 2;
    const bool ok        = htj2k_decode_indices(bin);
    bool unused          = false;
    bin->detach_compressed_data(unused);
    bin->attach_compressed_data(orig, pooled);
    bin->block_states = nullptr;
    if (!ok) open_htj2k::detail::fail("transcoder: an HT code-block could not be decoded.");
    // SigProp and MagRef passes below the LSB (of a cleanup pass at bit-plane 0) hold nothing.
    if (p == 0) Z_in = 3 * P0 + 1;
  } else {
    sc.ctx.assign(static_cast<size_t>(QH / 4 + 2) * (stride + 2), 0);
    bin->block_contexts        = sc.ctx.data();
    bin->block_contexts_stride = stride + 2;
    j2k_decode_indices(bin, roi);
    bin->block_contexts = nullptr;
  }
  bin->sample_buf = nullptr;

  // Pass z (0 being the cleanup pass of the input's top bit-plane, top_in) is in bit-plane
  // top_in - (z + 2) / 3; the decoded samples are exact down to the last one, in bit-plane q.  A sample
  // the last pass did not reach has its decoder's mid-point in bit-plane q, which the passes after it
  // (left out) would code.
  const int32_t top_in = Mb - 1 - static_cast<int32_t>(bin->num_ZBP);
  const int32_t q      = top_in - static_cast<int32_t>((Z_in - 1 + 2) / 3);
  if (q < 0) open_htj2k::detail::fail("transcoder: a code-block has more passes than bit-planes.");
  // The decoders hold a magnitude in Q(31 - M_b): index bit-plane j is bit 31 - M_b + j.  The encoder
  // takes 0x80000000 | ((m - 1) << 1) | sign for an index m > 0.
  const int32_t sh = 31 - Mb + q;
  uint32_t or_mag  = 0;
  sc.enc.assign(static_cast<size_t>(stride) * QH, 0);
  for (uint32_t y = 0; y < H; ++y) {
    const uint32_t *sp =
        reinterpret_cast<const uint32_t *>(sc.dec.data()) + static_cast<size_t>(y) * stride;
    int32_t *dp = sc.enc.data() + static_cast<size_t>(y) * stride;
    for (uint32_t x = 0; x < W; ++x) {
      const uint32_t m = ((sp[x] & 0x7FFFFFFFu) >> sh) << q;
      or_mag |= m;
      dp[x] = m ? static_cast<int32_t>(0x80000000u | ((m - 1) << 1) | (sp[x] >> 31)) : 0;
    }
  }
  if (or_mag == 0) {  // nothing to code
    leave_out(bout);
    return;
  }

  // The output block's passes start at the cleanup pass of its own top bit-plane: d bit-planes below
  // the input's (d < 0: above it, when an HT encoder signalled more missing MSBs than there are).  Its
  // pass z is the input's pass z + 3d; the input's passes in the bit-planes above are dropped, and the
  // ones it does not have go into the block's first packet.
  const int32_t msb_len = 32 - static_cast<int32_t>(count_leading_zeros(or_mag));
  const int32_t d       = (Mb - msb_len) - static_cast<int32_t>(bin->num_ZBP);
  const int32_t Z_out   = static_cast<int32_t>(Z_in) - 3 * d;
  const int32_t Z_coded = 3 * (msb_len - q) - 2;  // the encoder codes down to the cleanup pass of q
  if (Z_out < 1 || Z_out > Z_coded)
    open_htj2k::detail::fail("transcoder: the passes of a code-block do not match its bit-planes.");

  // The passes of the output block up to each layer.
  const uint16_t Q = bout->num_layers;
  sc.cut.assign(Q, static_cast<uint32_t>(Z_out));
  if (Q > 1) {
    if (log == nullptr)
      open_htj2k::detail::fail("transcoder: the passes of a code-block were not all recorded.");
    std::fill(sc.cut.begin(), sc.cut.end(), 0u);
    for (const auto &run : log->runs) {
      if (run.first >= Q)
        open_htj2k::detail::fail("transcoder: a packet of a layer the tile does not have.");
      sc.cut[run.first] += run.second;
    }
    // The passes a cut packet left without data were dropped by the decoder, at the end.  The layers
    // before the block's first packet stay without passes.
    uint32_t in_cum = 0;
    for (uint16_t l = 0; l < Q; ++l) {
      in_cum += sc.cut[l];
      const int32_t c = static_cast<int32_t>(std::min<uint32_t>(in_cum, Z_in)) - 3 * d;
      sc.cut[l] =
          (in_cum == 0) ? 0u : static_cast<uint32_t>(std::max<int32_t>(0, std::min<int32_t>(c, Z_out)));
    }
    if (sc.cut[Q - 1] != static_cast<uint32_t>(Z_out))
      open_htj2k::detail::fail("transcoder: the passes of a code-block were not all recorded.");
  }
  // A cut inside a codeword segment keeps the shortest prefix of the coded data that decodes it (and
  // so does the last cut when the encoder's termination there does not fit); the encoder records them.
  bool inner = Z_out < Z_coded;
  for (uint16_t l = 0; l < Q; ++l)
    inner = inner || (sc.cut[l] > 0 && sc.cut[l] < static_cast<uint32_t>(Z_out));

  ht_rd_info &rd        = sc.rd;
  rd.p_coded            = static_cast<uint8_t>(q);
  rd.p1_prefix          = inner;
  bout->rd              = &rd;
  bout->sample_buf      = sc.enc.data();
  bout->blksampl_stride = stride;
  bout->pre_quantized   = true;
  bout->pre_or_val      = 1;
  const int32_t coded   = ebcot_encode(bout, 0);
  bout->rd              = nullptr;
  bout->sample_buf      = nullptr;
  if (coded < 0 || rd.p1_passes != static_cast<uint32_t>(Z_coded))
    open_htj2k::detail::fail(
        "transcoder: a code-block cannot be coded with the Part 1 block coder (65535 bytes at most).");

  // pass_length[] holds the bytes from one boundary to the next, a boundary being the end of a layer's
  // passes or of a codeword segment (as the encoder's rate control sets them).
  const p1_rd_pass *pt = rd.p1.get();
  const uint32_t full  = bout->length;
  uint32_t done        = 0;
  uint16_t li          = 0;
  for (uint32_t z = 0; z < static_cast<uint32_t>(Z_out); ++z) {
    while (li < Q && sc.cut[li] < z + 1) ++li;
    uint32_t at;
    if (pt[z].ends_segment) {
      at = pt[z].keep;
    } else if (z + 1 == static_cast<uint32_t>(Z_out)) {
      // The last cut: the encoder's termination there, unless it would change the bytes of a cut
      // below or run past the coded data; then a prefix of the coded data.
      if (pt[z].keep >= done && pt[z].keep + pt[z].tail_len <= full) {
        if (pt[z].tail_len)
          std::memcpy(bout->get_compressed_data() + pt[z].keep, pt[z].tail, pt[z].tail_len);
        at = pt[z].keep + pt[z].tail_len;
      } else {
        at = std::max(pt[z].pre, done);
      }
    } else if (li < Q && sc.cut[li] == z + 1) {
      at = std::max(pt[z].pre, done);  // a longer prefix decodes the passes as well
    } else {
      bout->pass_length[z] = 0;
      continue;
    }
    if (at < done || at > full)
      open_htj2k::detail::fail("transcoder: a code-block's cuts are out of order.");
    bout->pass_length[z] = at - done;
    done                 = at;
  }
  uint32_t prev = 0;
  for (uint16_t l = 0; l < Q; ++l) {
    bout->layer_start[l]  = static_cast<uint8_t>(prev);
    bout->layer_passes[l] = static_cast<uint8_t>(sc.cut[l] - prev);
    prev                  = sc.cut[l];
  }
  bout->num_passes        = static_cast<uint8_t>(Z_out);
  bout->pass_length_count = static_cast<uint8_t>(Z_out);
  bout->length            = done;
  rd.p1.reset();
}

// Calls fn(block_in, block_out, ROI shift of the component) for every code-block, the two tiles being of
// the same structure.
template <typename Fn>
void for_each_block_pair(j2k_tile &tin, j2k_tile &tout, Fn fn) {
  for (uint16_t c = 0; c < tin.get_num_components(); ++c) {
    j2k_tile_component *tci = tin.get_tile_component(c);
    j2k_tile_component *tco = tout.get_tile_component(c);
    const uint8_t NL        = tci->get_dwt_levels();
    if (tco->get_dwt_levels() != NL)
      open_htj2k::detail::fail("transcoder: tile structures differ (levels).");
    for (uint8_t r = 0; r <= NL; ++r) {
      j2k_resolution *cri = tci->access_resolution(r);
      j2k_resolution *cro = tco->access_resolution(r);
      if (cri->is_empty != cro->is_empty || cri->npw != cro->npw || cri->nph != cro->nph
          || cri->num_bands != cro->num_bands) {
        open_htj2k::detail::fail("transcoder: tile structures differ (resolution %u).", unsigned(r));
      }
      if (cri->is_empty) continue;
      for (uint32_t pi = 0; pi < cri->npw * cri->nph; ++pi) {
        j2k_precinct *cpi = cri->access_precinct(pi);
        j2k_precinct *cpo = cro->access_precinct(pi);
        for (uint8_t b = 0; b < cri->num_bands; ++b) {
          j2k_precinct_subband *pbi = cpi->access_pband(b);
          j2k_precinct_subband *pbo = cpo->access_pband(b);
          const uint32_t n          = pbi->num_codeblock_x * pbi->num_codeblock_y;
          if (n != pbo->num_codeblock_x * pbo->num_codeblock_y) {
            open_htj2k::detail::fail("transcoder: tile structures differ (code-blocks).");
          }
          // No packet of this precinct-band was in the input, so its code-blocks were never made:
          // they stay empty in the output as well.
          if (n == 0 || pbi->access_codeblock(0) == nullptr) continue;
          for (uint32_t i = 0; i < n; ++i)
            fn(pbi->access_codeblock(i), pbo->access_codeblock(i), tci->get_ROIshift());
        }
      }
    }
  }
}

// The magnitude bound B of the transcoded HT code-blocks (ISO/IEC 15444-15, 8.7.3); the same
// computation as the encoder's coded_MAGB(), with S_blk = num_ZBP + P0 for the placeholder passes.
int32_t coded_magb(j2k_tile *tiles, uint32_t num_tiles, const std::vector<uint8_t> &present) {
  int32_t B = 0;
  for (uint32_t t = 0; t < num_tiles; ++t) {
    if (!present[t]) continue;
    for (uint16_t c = 0; c < tiles[t].get_num_components(); ++c) {
      j2k_tile_component *tc = tiles[t].get_tile_component(c);
      const int32_t NL       = tc->get_dwt_levels();
      for (uint8_t r = 0; r <= NL; ++r) {
        j2k_resolution *cr = tc->access_resolution(r);
        if (cr->is_empty) continue;
        const int32_t nb = (r == 0) ? NL : NL - r + 1;
        for (uint32_t pi = 0; pi < cr->npw * cr->nph; ++pi) {
          j2k_precinct *cp = cr->access_precinct(pi);
          for (uint8_t b = 0; b < cr->num_bands; ++b) {
            j2k_precinct_subband *cpb = cp->access_pband(b);
            const uint32_t n          = cpb->num_codeblock_x * cpb->num_codeblock_y;
            for (uint32_t i = 0; i < n; ++i) {
              const j2k_codeblock *block = cpb->access_codeblock(i);
              if ((block->Cmodes & HT) == 0 || block->num_passes == 0) continue;
              const int32_t S_blk = block->num_ZBP + block->ht_phld_passes / 3;
              B = std::max<int32_t>(B, S_blk + 1 - ((block->transformation == 1) ? 0 : nb - 1));
            }
          }
        }
      }
    }
  }
  return B;
}

// PLT marker segments (ISO/IEC 15444-1, A.7.3) holding the packet lengths: each Iplt is a base 128
// number, most significant group first, with the top bit set on every byte but the last.  A
// segment has at most 65532 bytes of lengths and holds whole numbers.
std::vector<uint8_t> plt_segments(const std::vector<uint32_t> &lengths) {
  std::vector<uint8_t> out;
  size_t i     = 0;
  uint8_t zplt = 0;
  while (i < lengths.size()) {
    std::vector<uint8_t> body;
    while (i < lengths.size()) {
      uint8_t tmp[5];
      int n      = 0;
      uint32_t v = lengths[i];
      tmp[n++]   = static_cast<uint8_t>(v & 0x7F);
      for (v >>= 7; v != 0; v >>= 7) tmp[n++] = static_cast<uint8_t>(0x80 | (v & 0x7F));
      if (body.size() + static_cast<size_t>(n) > 65532U) break;
      while (n > 0) body.push_back(tmp[--n]);
      ++i;
    }
    const size_t L = body.size() + 3;
    out.push_back(0xFF);
    out.push_back(0x58);
    out.push_back(static_cast<uint8_t>(L >> 8));
    out.push_back(static_cast<uint8_t>(L & 0xFF));
    out.push_back(zplt++);
    out.insert(out.end(), body.begin(), body.end());
  }
  return out;
}

void check_supported(const j2k_main_header &h) {
  if (!h.DFS.empty() || !h.ATK.empty())
    open_htj2k::detail::fail("transcoder: Part 2 (DFS/ATK) marker segments are not supported yet.");
  if (h.COD == nullptr || h.QCD == nullptr || h.SIZ == nullptr)
    open_htj2k::detail::fail("transcoder: the main header lacks SIZ, COD or QCD.");
  if (h.COD->get_Cmodes() & HT)
    open_htj2k::detail::fail("transcoder: the input already uses the HT block coder.");
}

// Reads the main header of a codestream held in `in` (positioned on SOC).
// With borrow the reader works on `data` itself (the caller guarantees the 16 bytes after it are
// readable); else it takes a copy.
void read_main_header(j2c_src_memory &src, const uint8_t *data, size_t len, j2k_main_header &hdr,
                      bool borrow = false) {
  if (borrow) {
    src.borrow_memory(const_cast<uint8_t *>(data), len);
  } else {
    src.alloc_memory(len);
    std::memcpy(src.get_buf_pos(), data, len);
  }
  hdr.read(src);
  src.rewind_2bytes();
}

// The length of the main header of the codestream in data: up to and including the first SOT
// marker code.  The whole length if no SOT is found (the header parser then reports the problem).
size_t main_header_extent(const uint8_t *data, size_t len) {
  size_t p = 2;  // after SOC
  while (p + 2 <= len && data[p] == 0xFF) {
    if (data[p + 1] == 0x90) return p + 2;  // SOT
    if (data[p + 1] >= 0x30 && data[p + 1] <= 0x3F) {
      p += 2;
      continue;
    }  // no length field
    if (p + 4 > len) break;
    p += 2 + ((static_cast<size_t>(data[p + 2]) << 8) | data[p + 3]);
  }
  return len;
}

uint32_t be32(const uint8_t *p) {
  return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) | uint32_t(p[3]);
}

// The boxes of a JP2 file that the JPH file keeps: those before the codestream box (the
// signature box, the file type box and the codestream box excepted) go to `pre`, those after it to
// `post`.  Only the top level is walked; a box is copied as it is, header included.
void collect_file_boxes(const uint8_t *buf, size_t len, std::vector<uint8_t> &pre,
                        std::vector<uint8_t> &post) {
  const uint8_t *p = buf, *end = buf + len;
  bool seen_jp2c = false;
  while (static_cast<size_t>(end - p) >= 8) {
    const size_t remaining = static_cast<size_t>(end - p);
    const uint32_t lbox    = be32(p);
    const uint32_t tbox    = be32(p + 4);
    size_t blen;
    if (lbox == 1) {
      if (remaining < 16) break;
      uint64_t x = 0;
      for (int i = 0; i < 8; ++i) x = (x << 8) | p[8 + i];
      if (x < 16 || x > remaining) break;
      blen = static_cast<size_t>(x);
    } else if (lbox == 0) {
      blen = remaining;
    } else {
      if (lbox < 8) break;
      blen = (lbox <= remaining) ? lbox : remaining;
    }
    if (tbox == 0x6A703263u) {  // jp2c
      seen_jp2c = true;
    } else if (tbox != 0x6A502020u && tbox != 0x66747970u) {  // not jP, ftyp
      std::vector<uint8_t> &dst = seen_jp2c ? post : pre;
      dst.insert(dst.end(), p, p + blen);
    }
    p += blen;
  }
}

// True if the main header describes an HTJ2K codestream: a CAP marker segment for Part 15 or a code-block
// style with the HT bit.
bool header_is_htj2k(j2k_main_header &h) {
  if (h.CAP != nullptr && (h.CAP->get_Pcap() & 0x00020000u)) return true;
  if (h.COD != nullptr && (h.COD->get_Cmodes() & HT)) return true;
  for (auto &c : h.COC)
    if (c->get_Cmodes() & HT) return true;
  return false;
}

// Locates the codestream of a raw codestream or a JP2/JPH file.
void find_codestream(const uint8_t *&in, size_t &in_len, jph_info &finfo, bool &is_file) {
  is_file = (in != nullptr) && jph_parse_buffer(in, in_len, finfo);
  if (is_file) {
    in     = finfo.cs_data;
    in_len = finfo.cs_size;
  }
}

// The direction and the options of a transcode: `fwd` for Part 1 -> HTJ2K, `rev` for HTJ2K -> Part 1.
size_t transcode_impl(const uint8_t *in, size_t in_len, std::vector<uint8_t> &out,
                      const transcode_options *fwd, const transcode_to_part1_options *rev);

}  // namespace

size_t transcode_part1_to_htj2k(const uint8_t *in, size_t in_len, std::vector<uint8_t> &out) {
  return transcode_part1_to_htj2k(in, in_len, out, transcode_options());
}

size_t transcode_part1_to_htj2k(const uint8_t *in, size_t in_len, std::vector<uint8_t> &out,
                                const transcode_options &options) {
  return transcode_impl(in, in_len, out, &options, nullptr);
}

size_t transcode_htj2k_to_part1(const uint8_t *in, size_t in_len, std::vector<uint8_t> &out) {
  return transcode_htj2k_to_part1(in, in_len, out, transcode_to_part1_options());
}

size_t transcode_htj2k_to_part1(const uint8_t *in, size_t in_len, std::vector<uint8_t> &out,
                                const transcode_to_part1_options &options) {
  if (options.code_block_style < -1 || options.code_block_style > 0x3F)
    open_htj2k::detail::fail("transcoder: a Part 1 code-block style takes bits 0-5 only.");
  return transcode_impl(in, in_len, out, nullptr, &options);
}

bool transcode_input_is_htj2k(const uint8_t *in, size_t in_len) {
  jph_info finfo;
  bool is_file = false;
  find_codestream(in, in_len, finfo, is_file);
  if (in == nullptr || in_len < 4 || in[0] != 0xFF || in[1] != 0x4F) {
    open_htj2k::detail::fail(
        "transcoder: the input is neither a JP2/JPH file nor a raw JPEG 2000 codestream.");
  }
  j2c_src_memory src;
  j2k_main_header h;
  read_main_header(src, in, main_header_extent(in, in_len), h);
  return header_is_htj2k(h);
}

namespace {
size_t transcode_impl(const uint8_t *in, size_t in_len, std::vector<uint8_t> &out,
                      const transcode_options *fwd, const transcode_to_part1_options *rev) {
  const bool to_part1             = rev != nullptr;
  const transcode_options options = to_part1 ? transcode_options() : *fwd;
  const bool input_padded         = to_part1 ? rev->input_padded : options.input_padded;
  const unsigned num_threads      = to_part1 ? rev->num_threads : options.num_threads;
  const int32_t p1_style          = to_part1 ? rev->code_block_style : -1;
  transcode_stats stats;
  const bool keep_layers = (to_part1 ? rev->layers : options.layers) == transcode_layers::keep;
  // The layers are kept with the passes of the input, which refine codes and the others do not.
  if (!to_part1 && keep_layers && options.partial_plane != transcode_partial_plane::refine)
    open_htj2k::detail::fail("transcoder: -layers keep works with -partial refine only.");
  // A JP2 file: transcode its codestream, keep its other boxes for a JPH output.
  const uint8_t *file_in = in;
  const size_t file_len  = in_len;
  jph_info finfo;
  const bool is_file = (in != nullptr) && jph_parse_buffer(in, in_len, finfo);
  if (options.jph_colour_space != transcode_colour_space::automatic) {
    if (!options.jph_output) open_htj2k::detail::fail("transcoder: a colour space is for JPH output only.");
    if (is_file)
      open_htj2k::detail::fail(
          "transcoder: the input file has its own colour specification, which is kept; a colour space is "
          "for a raw "
          "codestream.");
  }
  if (is_file) {
    in     = finfo.cs_data;
    in_len = finfo.cs_size;
  }
  // The codestream reader keeps positions in 32 bits.
  if (in_len > 0xFFFFFFFFull - 64)
    open_htj2k::detail::fail("transcoder: a codestream of 4 GB or more is not supported.");
  if (in == nullptr || in_len < 4 || in[0] != 0xFF || in[1] != 0x4F) {
    open_htj2k::detail::fail(
        "transcoder: the input is neither a JP2/JPH file nor a raw JPEG 2000 codestream.");
  }

  // Input: main header and tiles with their packets parsed.  Nothing is decoded here.
  j2c_src_memory src;
  j2k_main_header hin;
  read_main_header(src, in, in_len, hin, input_padded);
  if (to_part1) {
    if (!hin.DFS.empty() || !hin.ATK.empty())
      open_htj2k::detail::fail("transcoder: Part 2 (DFS/ATK) marker segments are not supported yet.");
    if (hin.COD == nullptr || hin.QCD == nullptr || hin.SIZ == nullptr)
      open_htj2k::detail::fail("transcoder: the main header lacks SIZ, COD or QCD.");
    if (!header_is_htj2k(hin))
      open_htj2k::detail::fail("transcoder: the input does not use the HT block coder.");
    if (hin.CAP != nullptr && hin.CAP->get_Pcap() != 0x00020000u)
      open_htj2k::detail::fail("transcoder: the CAP marker segment signals capabilities besides Part 15.");
  } else {
    check_supported(hin);
  }

  element_siz numTiles;
  hin.get_number_of_tiles(numTiles.x, numTiles.y);
  const uint32_t num_tiles = numTiles.x * numTiles.y;
  if (num_tiles == 0 || num_tiles > 65535) open_htj2k::detail::fail("transcoder: bad number of tiles.");

  // Pass 1: where the tile-parts of each tile are.  Only their headers are read, so a tile costs
  // nothing until its turn comes.  The tiles are then transcoded one after the other, and a tile's
  // input and output structures are freed before the next one is built.
  struct tile_part_loc {
    uint32_t pos;    // read position of the SOT marker segment (after the marker code)
    uint32_t index;  // the tile-part's place in the codestream (for PPM)
    uint32_t bytes;  // its length, from the SOT marker segment to its last byte
  };
  std::vector<std::vector<tile_part_loc>> tile_parts(num_tiles);
  {
    uint16_t word;
    uint32_t count = 0;
    while (src.get_remaining() >= 2 && (word = src.get_word()) != _EOC) {
      if (word != _SOT)
        open_htj2k::detail::fail("transcoder: SOT marker segment expected but %04X is found", word);
      const uint32_t at = src.get_pos();
      SOT_marker sot(src);
      const uint16_t ti = sot.get_tile_index();
      if (ti >= num_tiles)
        open_htj2k::detail::fail("transcoder: SOT tile index %u is out of range.", unsigned(ti));
      j2k_tile_part part(hin.SIZ->get_num_components());
      part.set_SOT(sot);
      part.read(src);
      tile_parts[ti].push_back({at, count++, src.get_pos() - at});
    }
  }
  // A tile with no tile-part in the input has no data to transcode.  It is left out of the output,
  // as the input leaves it out, so a header that declares many tiles costs nothing for them.
  std::vector<uint8_t> present(num_tiles, 0);
  uint32_t num_present = 0;
  for (uint32_t i = 0; i < num_tiles; ++i) {
    if (tile_parts[i].empty()) continue;
    present[i] = 1;
    ++num_present;
  }
  if (num_present == 0) open_htj2k::detail::fail("transcoder: the codestream has no tile-part.");

  // Output: the same header with the HT code-block style, one quality layer (those of the input
  // with keep_layers) and a CAP marker segment.  It is parsed from the input again, to be changed
  // without touching the input's.
  j2c_src_memory src2;
  j2k_main_header hout;
  // Only the header is read again, not the tile-parts after it.
  read_main_header(src2, in, main_header_extent(in, in_len), hout);
  // CAP bit 5 (ISO/IEC 15444-15, A.3.6): 0 (HTREV) only when every component is reversible; a
  // codestream with an irreversible component, a mixed one included, is in the HTIRV set (8.7.2).
  // The transformation of a component is that of its COC if it has one, else that of the COD; a
  // tile-part header's COC and COD take precedence over the main header's for that tile.
  const uint16_t ncomp = hout.SIZ->get_num_components();
  std::vector<uint8_t> main_rev(ncomp, hout.COD->get_transformation() == 1 ? 1 : 0);
  for (const auto &c : hout.COC)
    if (c->get_component_index() < ncomp)
      main_rev[c->get_component_index()] = (c->get_transformation() == 1);
  bool reversible = true;
  for (uint8_t r : main_rev) reversible = reversible && (r != 0);
  const auto NL0 = hout.COD->get_dwt_levels();
  // The code-block style of the output: HT (Part 1 -> HTJ2K) or the Part 1 switches (HTJ2K -> Part 1).
  auto out_style = [&](uint8_t s) { return to_part1 ? part1_style(s, p1_style) : ht_style(s); };
  hout.COD->set_Cmodes(out_style(hout.COD->get_Cmodes()));
  for (auto &c : hout.COC) c->set_Cmodes(out_style(c->get_Cmodes()));
  if (!keep_layers) hout.COD->set_number_of_layers(1);
  if (to_part1) {
    // Rsiz without bit 14 (no CAP marker segment).  A CPF marker segment (ISO/IEC 15444-15) gives
    // the profile of the corresponding Part 1 codestream, its Rsiz plus one: Rsiz is restored from it
    // (the HTJ2K conformance codestreams carry the Rsiz of their Part 1 sources so), and dropped.
    uint16_t rsiz = static_cast<uint16_t>(hout.SIZ->get_Rsiz() & ~0x4000);
    if (hin.CPF != nullptr && hin.CPF->get_Pcpf().size() == 1 && hin.CPF->get_Pcpf()[0] >= 1
        && ((hin.CPF->get_Pcpf()[0] - 1) & 0x4000) == 0)
      rsiz = static_cast<uint16_t>(hin.CPF->get_Pcpf()[0] - 1);
    hout.SIZ->set_Rsiz(rsiz);
    hout.CPF.reset();
  } else {
    // The profile of the input goes into a CPF marker segment, for a transcode back to Part 1.
    const uint16_t rsiz_in = hout.SIZ->get_Rsiz();
    hout.CPF.reset(new CPF_marker());
    hout.CPF->set_Pcpf(static_cast<uint16_t>(rsiz_in + 1));
    hout.SIZ->set_cap_capability();
  }
  // The input packets are read in the order its POC marker segments give.  They are written again
  // in the order of the COD, a progression order every packet set can be put in, so the output
  // needs no progression order changes.
  hout.POC.reset();
  // Likewise the packet headers are read from the PPM marker segments, if any, and written in the
  // packets.
  hout.PPM.clear();
  hout.ppm_header.reset();
  hout.ppm_buf.reset();
  hout.TLM.clear();
  hout.PLM.clear();
  int32_t magb_main = hout.QCD->get_MAGB(NL0);
  for (const auto &q : hout.QCC) magb_main = std::max<int32_t>(magb_main, q->get_MAGB(NL0));
  // A Part 1 codestream has no CAP marker segment (only Part 15 capabilities are allowed in the input).
  if (to_part1)
    hout.CAP.reset();
  else
    hout.CAP = std::unique_ptr<CAP_marker>(new CAP_marker());
  // CAP bit 12 (RGN): the output keeps an RGN marker segment, in the main header or (found with the
  // tiles) a tile-part header.  Bit 11 (HETEROGENEOUS): a tile-part header of the output has coding
  // parameters (COD, COC, QCD, QCC or RGN; it has no POC or PPT).
  bool has_rgn       = !hout.RGN.empty();
  bool heterogeneous = false;

  // The header's length does not depend on the values that are known at the end (the magnitude
  // bound, the reversibility, the RGN and HETEROGENEOUS flags and the tile-part lengths), so the
  // tile-parts are written after a header that is filled in then.
  std::vector<uint16_t> tlm_indices;
  if (num_tiles > 1) {
    for (uint32_t i = 0; i < num_tiles; ++i)
      if (present[i]) tlm_indices.push_back(static_cast<uint16_t>(i));
  }
  auto make_header = [&](const std::vector<uint32_t> &tlm_lengths, int32_t magb) {
    hout.TLM.clear();
    if (num_tiles > 1)
      hout.TLM.push_back(
          std::unique_ptr<TLM_marker>(new TLM_marker(static_cast<uint8_t>(0), tlm_indices, tlm_lengths)));
    const auto bit5  = static_cast<uint16_t>(reversible ? 0 : 1);     // HTREV / HTIRV
    const auto bit11 = static_cast<uint16_t>(heterogeneous ? 1 : 0);  // HOMOGENEOUS / HETEROGENEOUS
    const auto bit12 = static_cast<uint16_t>(has_rgn ? 1 : 0);        // RGNFREE / RGN
    if (!to_part1)
      hout.CAP->set_Ccap(
          static_cast<uint16_t>((bit12 << 12) | (bit11 << 11) | (bit5 << 5) | magb_field(magb)), 15);
    j2c_dst_memory hdr_buf;
    hdr_buf.put_word(_SOC);
    hout.flush(hdr_buf);
    std::vector<uint8_t> bytes;
    hdr_buf.flush(&bytes);
    return bytes;
  };
  const size_t hdr_len = make_header(std::vector<uint32_t>(tlm_indices.size(), 0), magb_main).size();

  // JPH output: the boxes in front of the codestream and after it.  Those in front hold the
  // codestream's length, which is known at the end; their own length is the same unless the
  // codestream is longer than 4 GB, and they are made again then.
  std::vector<uint8_t> file_mid, file_post;
  if (options.jph_output && is_file) collect_file_boxes(file_in, file_len, file_mid, file_post);
  auto make_pre = [&](size_t cs_len) {
    std::vector<uint8_t> pre;
    if (!options.jph_output) return pre;
    if (is_file) {
      j2c_dst_memory sig_buf;
      signature_box sig;
      file_type_box ftyp(1);  // brand 'jph '
      sig.write(sig_buf);
      ftyp.write(sig_buf);
      sig_buf.flush(&pre);
      pre.insert(pre.end(), file_mid.begin(), file_mid.end());
      uint8_t jp2c[16];
      const size_t n = jp2c_box_header(cs_len, jp2c);
      pre.insert(pre.end(), jp2c, jp2c + n);
    } else {
      // A raw codestream: boxes made as the encoder makes them; sRGB for three or more
      // components, grayscale otherwise.
      const uint16_t nc = hout.SIZ->get_num_components();
      if (options.jph_colour_space != transcode_colour_space::automatic && nc != 3 && nc != 4)
        open_htj2k::detail::fail(
            "transcoder: sRGB and sYCC are for three or four components, the image has %u.",
            static_cast<unsigned>(nc));
      const bool srgb = (options.jph_colour_space == transcode_colour_space::automatic)
                            ? (nc >= 3)
                            : (options.jph_colour_space == transcode_colour_space::rgb);
      j2c_dst_memory box_buf;
      jph_boxes boxes(hout, 1, srgb, cs_len);
      boxes.write(box_buf);
      box_buf.flush(&pre);
    }
    return pre;
  };
  const size_t pre_len = make_pre(hdr_len + 2).size();

  // The output buffer; the boxes and the header are filled in at the end.
  out.clear();
  out.reserve(pre_len + in_len + in_len / 8 + hdr_len + (size_t(1) << 16));
  out.resize(pre_len + hdr_len);

  std::vector<uint32_t> tile_lengths;
  int32_t magb_max = 0;
  const std::vector<uint8_t> one_present(1, 1);
  std::vector<int32_t *> empty_buf;
#ifdef OPENHTJ2K_THREAD
  // The pool is made when the first tile that is not small needs it: an image of small tiles
  // transcodes on this thread and does not start or stop the pool's threads.
  std::unique_ptr<ThreadPoolLease> pool_lease;
  ThreadPool *pool = nullptr;
#endif
  for (uint32_t i = 0; i < num_tiles; ++i) {
    if (!present[i]) continue;
#ifdef OPENHTJ2K_THREAD
    // A tile with little compressed data is transcoded on this thread alone, in the setup of the
    // input and output tiles as in the block loop: the pool's dispatches cost more than its work
    // (4K in 64 x 64 tiles took 5 times as long with 2 threads as with 1).  The decoder does the
    // same for small tiles (see j2k_tile::decode_inline()).
    uint64_t tile_bytes = 0;
    for (const tile_part_loc &loc : tile_parts[i]) tile_bytes += loc.bytes;
    const bool inline_tile = tile_bytes < kInlineTileBytes;
    ThreadPool::SingleThreadScope inline_scope(inline_tile);
#endif
    // The input tile, with its packets parsed.  Nothing is decoded here.
    j2k_tile tin;
    tin.dec_init(static_cast<uint16_t>(i), hin, 0);
    for (const tile_part_loc &loc : tile_parts[i]) {
      src.set_pos(loc.pos);
      SOT_marker sot(src);
      hin.num_tile_parts_read = loc.index;
      tin.add_tile_part(sot, src, hin);
    }
    tin.line_based_decode = true;
    // keep_layers: after each packet, the passes it added to each code-block of its precinct.  The
    // packets of a precinct come in the order of their layers.
    std::unordered_map<const j2k_codeblock *, layer_log> logs;
    if (keep_layers) {
      tin.set_packet_observer(
          [&tin, &logs](uint16_t c, uint8_t r, uint32_t p, uint16_t layer, uint64_t, uint64_t) {
            j2k_resolution *cr = tin.get_tile_component(c)->access_resolution(r);
            j2k_precinct *cp   = cr->access_precinct(p);
            for (uint8_t b = 0; b < cr->num_bands; ++b) {
              j2k_precinct_subband *pb = cp->access_pband(b);
              if (!pb->is_built()) continue;
              // Only the code-blocks this packet gave passes to (a walk over all of the
              // precinct's for each of its packets takes layers x code-blocks steps).
              const uint32_t *included = nullptr;
              const size_t n           = tin.packet_blocks(b, included);
              for (size_t j = 0; j < n; ++j) {
                const j2k_codeblock *blk = pb->access_codeblock(included[j]);
                if (blk->num_passes == 0) continue;
                layer_log &g = logs[blk];
                if (blk->num_passes > g.total) {
                  g.runs.emplace_back(layer, static_cast<uint8_t>(blk->num_passes - g.total));
                  g.total = blk->num_passes;
                }
              }
            }
          });
    }
    tin.create_tile_buf(hin);
    // The output tile is an encoder tile of the same structure.  It builds every code-block (the
    // input tile builds those its packets have data for), each with its passes per layer, so its
    // state too is sized by header values: it counts against the codestream's memory budget with
    // the input tile's, before it is built.
    tin.charge_output_tile(keep_layers ? tin.get_numlayers() : static_cast<uint16_t>(1));

    // A COD, COC, QCD or QCC in the input's tile-part header is kept for the tile, with the HT
    // code-block style and one layer, and written in the tile-part header of the output.
    j2k_tilepart_header ov(hout.SIZ->get_num_components());
    bool has_override = false;
    // Comments of the input's tile-part header are kept in the output's.
    std::vector<COM_marker> tile_com;
    bool want_plt = false;
    if (const j2k_tilepart_header *th = tin.first_tile_part_header()) {
      want_plt = !th->PLT.empty();
      for (const auto &c : th->COM) tile_com.push_back(*c);
      if (th->COD != nullptr) {
        ov.COD = std::unique_ptr<COD_marker>(new COD_marker(*th->COD));
        ov.COD->set_Cmodes(out_style(ov.COD->get_Cmodes()));
        if (!keep_layers) ov.COD->set_number_of_layers(1);
        has_override = true;
      }
      for (const auto &c : th->COC) {
        ov.COC.push_back(std::unique_ptr<COC_marker>(new COC_marker(*c)));
        ov.COC.back()->set_Cmodes(out_style(ov.COC.back()->get_Cmodes()));
        has_override = true;
      }
      if (th->QCD != nullptr) {
        ov.QCD       = std::unique_ptr<QCD_marker>(new QCD_marker(*th->QCD));
        has_override = true;
      }
      for (const auto &q : th->QCC) {
        ov.QCC.push_back(std::unique_ptr<QCC_marker>(new QCC_marker(*q)));
        has_override = true;
      }
      for (const auto &g : th->RGN) {
        ov.RGN.push_back(std::unique_ptr<RGN_marker>(new RGN_marker(*g)));
        has_override = true;
      }
    }
    if (!ov.RGN.empty()) has_rgn = true;
    if (has_override) heterogeneous = true;
    if (has_override) {
      std::vector<uint8_t> rev(main_rev);
      if (ov.COD != nullptr) rev.assign(ncomp, ov.COD->get_transformation() == 1 ? 1 : 0);
      for (const auto &c : ov.COC)
        if (c->get_component_index() < ncomp)
          rev[c->get_component_index()] = (c->get_transformation() == 1);
      for (uint8_t r : rev) reversible = reversible && (r != 0);
    }
    j2k_tile tout;
    tout.enc_init(static_cast<uint16_t>(i), hout, empty_buf, true, true, has_override ? &ov : nullptr,
                  /*structure_only=*/true);
    std::vector<uint8_t> tile_hdr_bytes;
    if (has_override || !tile_com.empty()) {
      j2c_dst_memory tb;
      if (ov.COD != nullptr) ov.COD->write(tb);
      for (auto &c : ov.COC) c->write(tb);
      if (ov.QCD != nullptr) ov.QCD->write(tb);
      for (auto &q : ov.QCC) q->write(tb);
      for (auto &g : ov.RGN) g->write(tb);
      for (auto &c : tile_com) c.write(tb);
      tb.flush(&tile_hdr_bytes);
      tout.set_tile_part_header_bytes(tile_hdr_bytes);
    }

    // Code-blocks.  Each block is independent, so they run on the thread pool in fixed chunks; every
    // chunk has its own statistics, summed in chunk order, so the output does not depend on the
    // number of threads.
    {
      struct block_job {
        j2k_codeblock *in, *out;
        uint8_t roi;
        const layer_log *log;  // keep_layers: the passes of each layer
      };
      std::vector<block_job> pairs;
      for_each_block_pair(tin, tout, [&](j2k_codeblock *bi, j2k_codeblock *bo, uint8_t roi) {
        const auto it = logs.find(bi);
        pairs.push_back({bi, bo, roi, it == logs.end() ? nullptr : &it->second});
      });
      constexpr size_t CHUNK = 8;
      const size_t nchunks   = (pairs.size() + CHUNK - 1) / CHUNK;
      std::vector<transcode_stats> cstats(nchunks);
      std::atomic<bool> failed{false};
      std::exception_ptr error;
      std::mutex error_mutex;
      auto run_chunk = [&](size_t k) {
        if (failed.load(std::memory_order_relaxed)) return;
        thread_local scratch_t sc;
        thread_local p1_scratch_t sc1;
        try {
          const size_t end = std::min(pairs.size(), (k + 1) * CHUNK);
          for (size_t j = k * CHUNK; j < end; ++j) {
            if (to_part1) {
              part1_block(pairs[j].in, pairs[j].out, pairs[j].roi, pairs[j].log, sc1);
              continue;
            }
            transcode_block(pairs[j].in, pairs[j].out, pairs[j].roi, sc, options, cstats[k]);
            if (keep_layers) place_layers(pairs[j].in, pairs[j].out, pairs[j].log);
          }
        } catch (...) {
          std::lock_guard<std::mutex> lk(error_mutex);
          if (!error) error = std::current_exception();
          failed.store(true);
        }
      };
      bool done = false;
#ifdef OPENHTJ2K_THREAD
      if (!inline_tile && pool == nullptr && num_threads != 1 && nchunks > 1) {
        pool_lease.reset(new ThreadPoolLease());
        pool = ThreadPool::instance(num_threads);
      }
      if (!inline_tile && pool != nullptr && pool->num_threads() > 1 && nchunks > 1) {
        std::atomic<size_t> next{0};
        std::atomic<int> running{0};
        auto drain = [&]() {
          for (size_t k = next.fetch_add(1); k < nchunks; k = next.fetch_add(1)) run_chunk(k);
        };
        const int workers = static_cast<int>(std::min<size_t>(pool->num_threads(), nchunks));
        running.store(workers, std::memory_order_relaxed);
        for (int w = 0; w < workers; ++w) {
          pool->push([&]() {
            drain();
            running.fetch_sub(1, std::memory_order_release);
          });
        }
        drain();
        while (running.load(std::memory_order_acquire) > 0) {
          if (!pool->try_run_one()) std::this_thread::yield();
        }
        done = true;
      }
#endif
      if (!done) {
        for (size_t k = 0; k < nchunks; ++k) run_chunk(k);
      }
      if (error) std::rethrow_exception(error);
      for (const transcode_stats &cs : cstats) {
        stats.blocks += cs.blocks;
        stats.full += cs.full;
        stats.coarser += cs.coarser;
        stats.partial += cs.partial;
        stats.refined += cs.refined;
      }
      tout.build_packet_headers();
      if (want_plt) {
        std::vector<uint8_t> bytes     = tile_hdr_bytes;
        const std::vector<uint8_t> plt = plt_segments(tout.get_packet_lengths(hout));
        bytes.insert(bytes.end(), plt.begin(), plt.end());
        tout.set_tile_part_header_bytes(std::move(bytes));
      }
    }

    // The tile-part goes to the end of the output; its input and output structures are freed.
    if (!to_part1) magb_max = std::max(magb_max, coded_magb(&tout, 1, one_present));
    const size_t tile_size = tout.get_tile_part_size();
    const size_t at        = out.size();
    out.resize(at + tile_size);
    const size_t written = tout.write_packets_direct(hout, out.data() + at);
    if (written != tile_size) open_htj2k::detail::fail("transcoder: a tile-part is shorter than its size.");
    tile_lengths.push_back(static_cast<uint32_t>(tile_size));
    // Give the input tile's charges back to the codestream's memory budget, as the decoder does
    // after each tile; otherwise the budget fills up tile by tile on an image of many tiles.
    // (The output tile's state was charged through the input tile, and goes back with it.)
    tin.destroy();
  }

  // EOC, the boxes after the codestream, then the header and the boxes in front, now complete.
  out.push_back(0xFF);
  out.push_back(0xD9);
  const size_t codestream_len = out.size() - pre_len;
  out.insert(out.end(), file_post.begin(), file_post.end());
  {
    const std::vector<uint8_t> hdr = make_header(tile_lengths, magb_max);
    if (hdr.size() != hdr_len) open_htj2k::detail::fail("transcoder: the main header changed its length.");
    std::memcpy(out.data() + pre_len, hdr.data(), hdr.size());
    const std::vector<uint8_t> pre = make_pre(codestream_len);
    if (pre.size() == pre_len) {
      if (!pre.empty()) std::memcpy(out.data(), pre.data(), pre.size());
    } else {
      out.erase(out.begin(), out.begin() + static_cast<std::ptrdiff_t>(pre_len));
      out.insert(out.begin(), pre.begin(), pre.end());
    }
  }
  const size_t total = out.size();
  if (options.stats != nullptr) *options.stats = stats;
  return total;
}
}  // namespace

}  // namespace open_htj2k
