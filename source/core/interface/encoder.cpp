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
#include <cstring>
#include <fstream>
#include <j2kmarkers.hpp>
#include "encoder.hpp"
#include "coding_units.hpp"
#include "ht_rate_control.hpp"
#include "jph.hpp"
#include "ThreadPool.hpp"
#if defined(OPENHTJ2K_TIFF_SUPPORT)
  #include <tiffio.h>
  #include "message_internal.hpp"
#endif

#define NO_QFACTOR 0xFF

#define sRGB 0
#define sYCC 1

namespace open_htj2k {

// Per-thread reusable output buffers shared by both encoder entry points
// (`invoke_internal` and `invoke_line_based_stream`).  A single
// thread_local instance keeps the ~22 MB output capacity mapped across
// consecutive encodes on the same thread, eliminating the page-fault
// storm that glibc's mmap-free of large allocations would otherwise
// cause.  Function-local thread_locals were avoided so a thread that
// calls both entry points doesn't retain two separate sets.
namespace {
struct EncoderThreadBuffers {
  j2c_dst_memory tmp;      // per-tile measurement for the TLM marker
  j2c_dst_memory j2c_dst;  // final codestream
  j2c_dst_memory jph_dst;  // optional JPH wrapper
};
inline EncoderThreadBuffers &enc_thread_bufs() {
  static thread_local EncoderThreadBuffers tbuf;
  return tbuf;
}

// The QCD marker is derived from component 0's bit depth: its exponents follow that dynamic
// range.  A component of another depth needs its own quantization parameters, or a deeper one loses
// its top bit-planes and a lossless encode is not lossless.  Returns a QCC marker for each such
// component; none with Qfactor, for which the main header writes a QCC marker for every component.
std::vector<std::unique_ptr<QCC_marker>> qcc_for_other_depths(const std::vector<uint8_t> &Ssiz,
                                                              const cod_params &cod, const qcd_params &qcd,
                                                              uint8_t qfactor,
                                                              const visual_weighting_params &vw) {
  std::vector<std::unique_ptr<QCC_marker>> qcc;
  if (qfactor != NO_QFACTOR) return qcc;
  const uint16_t Csiz = static_cast<uint16_t>(Ssiz.size());
  for (uint16_t c = 1; c < Csiz; ++c) {
    const uint8_t depth = static_cast<uint8_t>((Ssiz[c] & 0x7F) + 1U);
    if (depth == static_cast<uint8_t>((Ssiz[0] & 0x7F) + 1U)) continue;
    QCD_marker component_QCD(qcd.number_of_guardbits, cod.dwt_levels, cod.transformation, qcd.is_derived,
                             depth, cod.use_color_trafo, qcd.base_step, NO_QFACTOR, vw);
    qcc.push_back(MAKE_UNIQUE<QCC_marker>(Csiz, c, component_QCD));
  }
  return qcc;
}

// Bits 0-4 of Ccap15 for a magnitude bound: the smallest P whose parameter B (ISO/IEC 15444-15,
// A.3.7: 8 for P = 0, P + 8 for P < 20, 4 (P - 19) + 27 for P < 31, 74 for P = 31) is at least
// `bound`.  A B smaller than the coded magnitudes need makes the codestream non-conforming.
uint16_t magb_field(int32_t bound) {
  if (bound <= 8) return 0;
  if (bound <= 27) return static_cast<uint16_t>(bound - 8);
  if (bound <= 71) return static_cast<uint16_t>(19 + (bound - 27 + 3) / 4);
  return 31;
}

// The magnitude bound the QCD and QCC markers of the main header allow (QCD_marker::get_MAGB()):
// what the CAP marker holds until the code-blocks are coded.
int32_t quantization_MAGB(const j2k_main_header &main_header, uint8_t NL) {
  int32_t B = main_header.QCD->get_MAGB(NL);
  for (const auto &q : main_header.QCC) B = std::max<int32_t>(B, q->get_MAGB(NL));
  return B;
}

// The magnitude bound of the HT code-blocks as coded (ISO/IEC 15444-15, 8.7.3), which covers the
// HT cleanup pass only.  A cleanup pass at bit-plane p of a sub-band with M_b magnitude
// bit-planes codes S_blk + 1 = M_b - p of them (7.6), S_blk being the zero bit-planes the encoder
// signals (num_ZBP; it codes no placeholder passes, B.3): not the plane its HT SigProp and
// MagRef passes refine, nor those a rate control dropped.  A code-block without coding passes
// contributes nothing.  For an irreversible transform the bound of a sub-band at decomposition
// level n_b is 2^(B + n_b - 1), so B takes n_b - 1 fewer planes there.
int32_t coded_MAGB(j2k_tile *tiles, uint32_t num_tiles) {
  int32_t B = 0;
  for (uint32_t t = 0; t < num_tiles; ++t) {
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
              B = std::max<int32_t>(B, block->num_ZBP + 1 - ((block->transformation == 1) ? 0 : nb - 1));
            }
          }
        }
      }
    }
  }
  return B;
}

// Why the last image::read_pnmpgx() or image::read_tiff() call on this thread failed.
thread_local std::string image_read_error;

// Resolution r of a component with NL decomposition levels takes the precinct size entry
// min(NL - r, entries - 1) of cod_params::PPx/PPy: the first entry is for the highest
// resolution, and the last one repeats for the lower resolutions (COD_marker).
size_t precinct_entry(uint8_t NL, uint8_t r, size_t entries) {
  return std::min(static_cast<size_t>(NL - r), entries - 1);
}

// The parameters the markers and the tile structures take as given.  Values outside the
// ranges of ISO/IEC 15444-1 Annex A, or beyond what this encoder implements, are refused
// here: they used to give invalid codestreams, wrong samples or memory errors.
void check_parameters(const siz_params &siz, const cod_params &cod, const qcd_params &qcd,
                      uint8_t qfactor) {
  if (siz.Csiz == 0 || siz.Csiz > 16384) {
    open_htj2k::detail::fail("the number of components (Csiz) shall be 1 to 16384, not %u.",
                             static_cast<unsigned>(siz.Csiz));
  }
  if (siz.Ssiz.size() < siz.Csiz || siz.XRsiz.size() < siz.Csiz || siz.YRsiz.size() < siz.Csiz) {
    open_htj2k::detail::fail("Ssiz, XRsiz and YRsiz shall hold a value for each of the %u components.",
                             static_cast<unsigned>(siz.Csiz));
  }
  if (siz.Xsiz <= siz.XOsiz || siz.Ysiz <= siz.YOsiz) {
    open_htj2k::detail::fail(
        "the image is empty: Xsiz and Ysiz (%u, %u) shall exceed XOsiz and YOsiz (%u, %u).", siz.Xsiz,
        siz.Ysiz, siz.XOsiz, siz.YOsiz);
  }
  // Coordinates are 32-bit signed integers in places (the decoder refuses larger extents too).
  if (siz.Xsiz > 0x7FFFFFFFu || siz.Ysiz > 0x7FFFFFFFu) {
    open_htj2k::detail::fail(
        "the image extent (Xsiz %u, Ysiz %u) exceeds 2147483647, the largest supported.", siz.Xsiz,
        siz.Ysiz);
  }
  for (uint16_t c = 0; c < siz.Csiz; ++c) {
    // The samples are 32-bit integers, and the decoder's DC offset and clamp to the sample range
    // hold at most 30 unsigned or 31 signed bits (it refuses deeper components).
    const unsigned depth = (siz.Ssiz[c] & 0x7FU) + 1U;
    const bool is_signed = (siz.Ssiz[c] & 0x80U) != 0;
    const unsigned limit = is_signed ? 31U : 30U;
    if (depth > limit) {
      open_htj2k::detail::fail("component %u has %u-bit %s samples; at most %u bits are supported.",
                               static_cast<unsigned>(c), depth, is_signed ? "signed" : "unsigned", limit);
    }
    if (siz.XRsiz[c] == 0 || siz.YRsiz[c] == 0) {
      open_htj2k::detail::fail("the subsampling factors XRsiz and YRsiz of component %u shall be 1 to 255.",
                               static_cast<unsigned>(c));
    }
  }
  // ISO/IEC 15444-1, B.12.1.3 and B.12.1.4: "To use this progression, XRsiz and YRsiz values
  // must be [shall be] powers of two for each component" for RPCL and PCRL (not for CPRL).
  // The encoder writes no POC marker, so the COD progression order is the only one.
  if (cod.progression_order == 2 || cod.progression_order == 3) {
    for (uint16_t c = 0; c < siz.Csiz; ++c) {
      const unsigned xr = siz.XRsiz[c], yr = siz.YRsiz[c];
      if ((xr & (xr - 1)) != 0 || (yr & (yr - 1)) != 0) {
        open_htj2k::detail::fail(
            "progression order %s needs subsampling factors that are powers of two (ISO/IEC 15444-1, "
            "B.12.1.%u), but component %u has XRsiz %u and YRsiz %u; CPRL takes any.",
            cod.progression_order == 2 ? "RPCL" : "PCRL", cod.progression_order == 2 ? 3U : 4U,
            static_cast<unsigned>(c), xr, yr);
      }
    }
  }
  if (cod.dwt_levels > 32) {
    open_htj2k::detail::fail("the number of DWT levels shall be 0 to 32, not %u.",
                             static_cast<unsigned>(cod.dwt_levels));
  }
  // Code-block width and height exponents (log2 - 2) of 0 to 8, an area of at most 4096.
  if (cod.blkwidth > 8 || cod.blkheight > 8 || cod.blkwidth + cod.blkheight > 8) {
    open_htj2k::detail::fail(
        "code-blocks shall be 4 to 1024 samples wide and high and hold at most 4096 samples "
        "(blkwidth %u, blkheight %u: log2 of the size less 2).",
        static_cast<unsigned>(cod.blkwidth), static_cast<unsigned>(cod.blkheight));
  }
  if (cod.progression_order > 5) {
    open_htj2k::detail::fail("the progression order shall be 0 to 5, not %u.",
                             static_cast<unsigned>(cod.progression_order));
  }
  if (cod.number_of_layers == 0) {
    open_htj2k::detail::fail("the number of quality layers shall be 1 to 65535.");
  }
  if (cod.use_color_trafo > 1) {
    open_htj2k::detail::fail("use_color_trafo shall be 0 or 1, not %u.",
                             static_cast<unsigned>(cod.use_color_trafo));
  }
  if (cod.transformation > 1) {
    open_htj2k::detail::fail("transformation shall be 0 (irreversible 9/7) or 1 (reversible 5/3), not %u.",
                             static_cast<unsigned>(cod.transformation));
  }
  if (!cod.is_max_precincts) {
    if (cod.PPx.empty() || cod.PPx.size() != cod.PPy.size()) {
      open_htj2k::detail::fail("user-defined precincts need as many PPx as PPy values, at least one.");
    }
    // Exponents of 0 to 15 (four bits each in the COD marker), and of at least 1 above the
    // lowest resolution (ISO/IEC 15444-1, A.6.1).
    for (uint8_t r = 0; r <= cod.dwt_levels; ++r) {
      const size_t i    = precinct_entry(cod.dwt_levels, r, cod.PPx.size());
      const unsigned lo = (r == 0) ? 0 : 1;
      if (cod.PPx[i] < lo || cod.PPy[i] < lo || cod.PPx[i] > 15 || cod.PPy[i] > 15) {
        open_htj2k::detail::fail(
            "the precinct size exponents of resolution %u are (%u, %u); they shall be %u to 15.",
            static_cast<unsigned>(r), static_cast<unsigned>(cod.PPx[i]), static_cast<unsigned>(cod.PPy[i]),
            lo);
      }
    }
  }
  if (qcd.number_of_guardbits > 7) {
    open_htj2k::detail::fail("the number of guard bits shall be 0 to 7, not %u.",
                             static_cast<unsigned>(qcd.number_of_guardbits));
  }
  // A subband has G + epsilon_b - 1 magnitude bit-planes (ISO/IEC 15444-1, E-2), and an
  // irreversible step of 2^-epsilon_b holds coefficients below 2^(G-1) of the nominal range.
  // Without guard bits, coefficients at the edge of that range (the most negative sample, at
  // the least) need one bit-plane more than the codestream gives them, which decoders reject
  // or decode differently.  The reversible step sizes do not depend on G.
  if (cod.transformation == 0 && qcd.number_of_guardbits == 0) {
    open_htj2k::detail::fail(
        "the irreversible transform needs at least 1 guard bit; with none, quantized coefficients can "
        "need more magnitude bit-planes than the codestream signals.");
  }
  // The base step size sets the irreversible quantization unless Qfactor does.
  if (cod.transformation == 0 && qfactor == NO_QFACTOR
      && !(qcd.base_step > 0.0 && std::isfinite(qcd.base_step))) {
    open_htj2k::detail::fail("the base step size shall be a positive finite number, not %g.",
                             qcd.base_step);
  }
}

// The reversible path is lossless as long as the samples and the transform keep within
// single-precision floating point: up to 23 bits per sample, one bit less for the
// components the reversible colour transform widens.  Checked after the colour transform
// has been decided.
void check_reversible_precision(const siz_params &siz, const cod_params &cod) {
  if (cod.transformation != 1 || cod.dwt_levels == 0) return;
  for (uint16_t c = 0; c < siz.Csiz; ++c) {
    const unsigned depth = (siz.Ssiz[c] & 0x7FU) + 1U;
    const bool widened   = cod.use_color_trafo == 1 && c < 3;
    if (depth + (widened ? 1U : 0U) > 23) {
      open_htj2k::detail::fail(
          "the reversible transform is lossless for at most 23 bits per sample (22 with the colour "
          "transform); component %u has %u.",
          static_cast<unsigned>(c), depth);
    }
  }
}

// Origin and tile checks shared by the entry points.  Tile size 0 (XTsiz or YTsiz) means one
// tile, reaching from the tile origin to the far edge of the image.
void check_tiling(siz_params &siz) {
  if ((siz.XTOsiz > siz.XOsiz) || (siz.YTOsiz > siz.YOsiz)) {
    open_htj2k::detail::fail("tile origin shall be no greater than the image origin.");
  }
  if (siz.XTsiz == 0 || siz.YTsiz == 0) {
    siz.XTsiz = siz.Xsiz - siz.XTOsiz;
    siz.YTsiz = siz.Ysiz - siz.YTOsiz;
  }
  if ((uint64_t{siz.XTOsiz} + siz.XTsiz <= siz.XOsiz) || (uint64_t{siz.YTOsiz} + siz.YTsiz <= siz.YOsiz)) {
    open_htj2k::detail::fail("tile size plus tile origin shall be greater than the image origin.");
  }
  const uint64_t tiles_x = (uint64_t{siz.Xsiz} - siz.XTOsiz + siz.XTsiz - 1) / siz.XTsiz;
  const uint64_t tiles_y = (uint64_t{siz.Ysiz} - siz.YTOsiz + siz.YTsiz - 1) / siz.YTsiz;
  if (tiles_x * tiles_y > 65535) {
    open_htj2k::detail::fail("The number of tiles exceeds its allowable maximum (65535).");
  }
}

// SPcod code-block style.  Part 1 code-blocks take any of the six mode switches
// of ISO/IEC 15444-1 Table A.19; HT code-blocks only the vertically causal one
// (ISO/IEC 15444-15 Table A.3).  Mixed HT/Part 1 tile-components (bit 7) are not
// supported.
void check_codeblock_style(uint8_t style) {
  const bool is_ht       = (style & HT) != 0;
  const unsigned allowed = is_ht ? (HT | CAUSAL) : (BYPASS | RESET | RESTART | CAUSAL | ERTERM | SEGMARK);
  if (style & ~allowed) {
    if (is_ht)
      open_htj2k::detail::fail(
          "HT code-blocks accept only the CAUSAL mode switch (code-block style 0x%02X).", style);
    else
      open_htj2k::detail::fail("unsupported code-block style 0x%02X.", style);
  }
}
}  // namespace

image::image(const std::vector<std::string> &filenames) : width(0), height(0), buf(nullptr) {
  size_t num_files = filenames.size();
  if (num_files > 16384) {
    open_htj2k::detail::fail("over 16384 components are not supported in the spec.");
  }
  num_components = static_cast<uint16_t>(num_files);  // num_components may change after parsing PPM header
  uint16_t c     = 0;
  char ext_name[256];
  char *infile_ext_name = ext_name;
  for (const auto &fname : filenames) {
    bool is_pnm_pgx = false;
    bool is_tiff    = false;
    infile_ext_name = strrchr(const_cast<char *>(fname.c_str()), '.');
    if (strcmp(infile_ext_name, ".pgm") == 0 || strcmp(infile_ext_name, ".PGM") == 0
        || strcmp(infile_ext_name, ".ppm") == 0 || strcmp(infile_ext_name, ".PPM") == 0
        || strcmp(infile_ext_name, ".pgx") == 0 || strcmp(infile_ext_name, ".PGX") == 0) {
      is_pnm_pgx = true;
    }
#if defined(OPENHTJ2K_TIFF_SUPPORT)
    if (strcmp(infile_ext_name, ".tif") == 0 || strcmp(infile_ext_name, ".TIF") == 0
        || strcmp(infile_ext_name, ".tiff") == 0 || strcmp(infile_ext_name, ".TIFF") == 0) {
      is_tiff = true;
    }
#endif

    if (!is_pnm_pgx && !is_tiff) {
      throw std::runtime_error("unsupported input file type (expected .pgm/.ppm/.pgx/.tif)");
    }
    image_read_error.clear();
    if (is_pnm_pgx) {
      if (read_pnmpgx(fname, c)) {
        throw std::runtime_error(image_read_error.empty() ? "cannot read " + fname : image_read_error);
      }
      c++;
    } else if (is_tiff) {
#if defined(OPENHTJ2K_TIFF_SUPPORT)
      if (read_tiff(fname)) {
        throw std::runtime_error(image_read_error.empty() ? "cannot read " + fname : image_read_error);
      }
      c++;
#endif
    }
  }
}

int image::read_pnmpgx(const std::string &filename, const uint16_t nc) {
  constexpr int READ_WIDTH  = 0;
  constexpr int READ_HEIGHT = 1;
  constexpr int READ_MAXVAL = 2;
  constexpr int DONE        = 3;
  constexpr char SP         = ' ';
  constexpr char LF         = '\n';
  constexpr char CR         = 0x0d;
  bool isASCII              = false;
  bool isBigendian          = false;
  bool isSigned             = false;

  FILE *fp = fopen(filename.c_str(), "rb");
  if (fp == nullptr) {
    image_read_error = open_htj2k::detail::format("File %s is not found.", filename.c_str());
    return EXIT_FAILURE;
  }
  int status = READ_WIDTH;
  int d;
  uint32_t val = 0;
  d            = fgetc(fp);
  if (d != 'P') {
    image_read_error = open_htj2k::detail::format("%s is not a PNM/PGX file.", filename.c_str());
    fclose(fp);
    return EXIT_FAILURE;
  }

  bool isPPM = false;
  bool isPGX = false;
  uint32_t compw, comph;
  uint8_t bitDepth;

  d = fgetc(fp);
  switch (d) {
    // PGM
    case '2':
      isASCII = true;
      /* FALLTHRU */
    case '5':
      isBigendian = true;
      break;
      // PPM
    case '3':
      isASCII = true;
      /* FALLTHRU */
    case '6':
      isPPM = true;
      // number of components shall be three here
      num_components = 3;
      isBigendian    = true;
      break;
      // PGX
    case 'G':
      isPGX = true;
      // read endian
      do {
        d = fgetc(fp);
      } while (d != 'M' && d != 'L');
      switch (d) {
        case 'M':
          isBigendian = true;
          d           = fgetc(fp);
          if (d != 'L') {
            open_htj2k::detail::warn("input PGX file %s is broken.", filename.c_str());
          }
          break;
        case 'L':
          d = fgetc(fp);
          if (d != 'M') {
            open_htj2k::detail::warn("input PGX file %s is broken.", filename.c_str());
          }
          break;
        default:
          image_read_error = open_htj2k::detail::format("input file does not conform to PGX format.");
          return EXIT_FAILURE;
      }
      // check signed or not
      do {
        d = fgetc(fp);
      } while (d != '+' && d != '-' && isdigit(d) == false);
      if (d == '+' || d == '-') {
        if (d == '-') {
          isSigned = true;
        }
        do {
          d = fgetc(fp);
        } while (isdigit(d) == false);
      }
      do {
        val *= 10;
        val += static_cast<unsigned int>(d - '0');
        d = fgetc(fp);
      } while (d != SP && d != LF && d != CR);
      bitDepth = static_cast<uint8_t>(val);
      val      = 0;
      break;
      // PBM (not supported)
    case '1':
    case '4':
      image_read_error = open_htj2k::detail::format("PBM file is not supported.");
      fclose(fp);
      return EXIT_FAILURE;
      break;
      // error
    default:
      image_read_error = open_htj2k::detail::format("%s is not a PNM/PGX file.", filename.c_str());
      fclose(fp);
      return EXIT_FAILURE;
      break;
  }
  while (status != DONE) {
    d = fgetc(fp);
    // eat white/LF/CR and comments
    while (d == SP || d == LF || d == CR) {
      d = fgetc(fp);
      if (d == '#') {
        // PNM comments run to end-of-line and have no length limit per the spec;
        // consume byte-by-byte instead of through a fixed-size fgets buffer.
        do {
          d = fgetc(fp);
        } while (d != EOF && d != LF && d != CR);
        if (d == EOF) {
          throw std::runtime_error("PNM/PGX header comment read error");
        }
        d = fgetc(fp);
      }
    }
    // read numerical value
    while (d != SP && d != LF && d != CR) {
      val *= 10;
      val += static_cast<unsigned int>(d - '0');
      d = fgetc(fp);
    }
    // update status
    switch (status) {
      case READ_WIDTH:
        compw       = val;
        this->width = (this->width < compw) ? compw : this->width;
        val         = 0;
        status      = READ_HEIGHT;
        break;
      case READ_HEIGHT:
        comph        = val;
        this->height = (this->height < comph) ? comph : this->height;
        val          = 0;
        if (isPGX) {
          status = DONE;
        } else {
          status = READ_MAXVAL;
        }
        break;
      case READ_MAXVAL:
        bitDepth = static_cast<uint8_t>(log2(static_cast<float>(val)) + 1.0f);
        val      = 0;
        status   = DONE;
        break;
      default:
        break;
    }
  }
  uint16_t num_iterations = 1;
  if (isPPM) {
    num_iterations = 3;
  }
  // setting bit-depth for components
  for (int i = 0; i < num_iterations; ++i) {
    this->component_width.push_back(compw);
    this->component_height.push_back(comph);
    this->bits_per_pixel.push_back(bitDepth);
    this->is_signed.push_back(isSigned);
  }

  // easting trailing spaces/LF/CR or comments
  d = fgetc(fp);
  while (d == SP || d == LF || d == CR) {
    d = fgetc(fp);
    if (d == '#') {
      do {
        d = fgetc(fp);
      } while (d != EOF && d != LF && d != CR);
      if (d == EOF) {
        throw std::runtime_error("PNM/PGX header comment read error");
      }
      d = fgetc(fp);
    }
  }
  fseek(fp, -1, SEEK_CUR);

  const uint32_t byte_per_sample      = (bitDepth + 8U - 1U) / 8U;
  const uint32_t component_gap        = num_iterations * byte_per_sample;
  const uint32_t line_width           = component_gap * compw;
  std::unique_ptr<uint8_t[]> line_buf = MAKE_UNIQUE<uint8_t[]>(line_width);

  // allocate memory once
  if (this->buf == nullptr) {
    this->buf = MAKE_UNIQUE<std::unique_ptr<int32_t[]>[]>(this->num_components);
  }
  if (isPPM) {
    for (size_t i = 0; i < this->num_components; ++i) {
      this->buf[i] = MAKE_UNIQUE<int32_t[]>(static_cast<size_t>(compw) * comph);
    }
  } else {
    this->buf[nc] = MAKE_UNIQUE<int32_t[]>(static_cast<size_t>(compw) * comph);
  }

  if (!isASCII) {
    for (size_t i = 0; i < comph; ++i) {
      if (fread(line_buf.get(), sizeof(uint8_t), line_width, fp) < line_width) {
        image_read_error = open_htj2k::detail::format("not enough samples in the given pnm file.");
        fclose(fp);
        return EXIT_FAILURE;
      }

      for (size_t c = 0; c < num_iterations; ++c) {
        uint8_t *src;
        int32_t *dst;
        src = &line_buf[c * byte_per_sample];
        if (isPPM) {
          dst = &this->buf[c][i * compw];
        } else {
          dst = &this->buf[nc][i * compw];
        }
        switch (byte_per_sample) {
          case 1:
            for (size_t j = 0; j < compw; ++j) {
              *dst = (isSigned) ? static_cast<int8_t>(*src) : *src;
              dst++;
              src += component_gap;
            }
            break;
          case 2:
            for (size_t j = 0; j < compw; ++j) {
              if (isSigned) {
                if (isBigendian) {
                  *dst = static_cast<int_least16_t>((static_cast<uint_least16_t>(src[0]) << 8)
                                                    | static_cast<uint_least16_t>(src[1]));
                } else {
                  *dst = static_cast<int_least16_t>(static_cast<uint_least16_t>(src[0])
                                                    | (static_cast<uint_least16_t>(src[1]) << 8));
                }
              } else {
                if (isBigendian) {
                  *dst = (src[0] << 8) | src[1];
                } else {
                  *dst = src[0] | (src[1] << 8);
                }
              }
              dst++;
              src += component_gap;
            }
            break;
          default:
            image_read_error = open_htj2k::detail::format("bit-depth over 16 is not supported.");
            fclose(fp);
            return EXIT_FAILURE;
            break;
        }
      }
    }
  } else {
    for (size_t i = 0; i < static_cast<size_t>(compw) * comph; ++i) {
      for (size_t c = 0; c < num_iterations; ++c) {
        val = 0;
        d   = fgetc(fp);
        while (d != SP && d != CR && d != LF && d != EOF) {
          val *= 10;
          val += static_cast<unsigned int>(d - '0');
          d = fgetc(fp);
        }
        this->buf[c][i] = static_cast<int>(val);
      }
    }
  }
  fclose(fp);
  return EXIT_SUCCESS;
}

#if defined(OPENHTJ2K_TIFF_SUPPORT)
int image::read_tiff(const std::string &filename) {
  TIFF *tiff_handle;
  size_t bytes_per_line;
  uint16_t planar_configuration;
  if ((tiff_handle = TIFFOpen(filename.c_str(), "r")) == nullptr) {
    image_read_error = open_htj2k::detail::format("File %s is not found.", filename.c_str());
    return EXIT_FAILURE;
  }

  uint32_t tiff_width  = 0;
  uint32_t tiff_height = 0;
  TIFFGetField(tiff_handle, TIFFTAG_IMAGEWIDTH, &tiff_width);
  TIFFGetField(tiff_handle, TIFFTAG_IMAGELENGTH, &tiff_height);

  uint16_t tiff_bits_per_sample   = 0;
  uint16_t tiff_samples_per_pixel = 0;
  TIFFGetField(tiff_handle, TIFFTAG_BITSPERSAMPLE, &tiff_bits_per_sample);
  TIFFGetField(tiff_handle, TIFFTAG_SAMPLESPERPIXEL, &tiff_samples_per_pixel);
  // some TIFs have tiff_samples_per_pixel=0 when it is a single channel
  // image - set to 1
  tiff_samples_per_pixel = (tiff_samples_per_pixel < 1) ? 1 : tiff_samples_per_pixel;

  uint16_t tiff_planar_configuration = 0;
  uint16_t tiff_photometric          = 0;
  TIFFGetField(tiff_handle, TIFFTAG_PLANARCONFIG, &tiff_planar_configuration);
  TIFFGetField(tiff_handle, TIFFTAG_PHOTOMETRIC, &tiff_photometric);

  planar_configuration = tiff_planar_configuration;

  uint16_t tiff_compression    = 0;
  uint32_t tiff_rows_per_strip = 0;
  TIFFGetField(tiff_handle, TIFFTAG_COMPRESSION, &tiff_compression);
  TIFFGetField(tiff_handle, TIFFTAG_ROWSPERSTRIP, &tiff_rows_per_strip);

  if (tiff_planar_configuration == PLANARCONFIG_SEPARATE) {
    bytes_per_line = tiff_samples_per_pixel * TIFFScanlineSize64(tiff_handle);
  } else {
    bytes_per_line = TIFFScanlineSize64(tiff_handle);
  }
  // Error on known incompatilbe input formats
  if (tiff_bits_per_sample != 8 && tiff_bits_per_sample != 16) {
    image_read_error = open_htj2k::detail::format(
        "TIFF input is limited to files with TIFFTAG_BITSPERSAMPLE=8 or 16; %s has "
        "TIFFTAG_BITSPERSAMPLE=%d.",
        filename.c_str(), tiff_bits_per_sample);
    TIFFClose(tiff_handle);
    return EXIT_FAILURE;
  }
  if (TIFFIsTiled(tiff_handle)) {
    image_read_error = open_htj2k::detail::format(
        "TIFF input is limited to files without tiles; %s is tiled.", filename.c_str());
    TIFFClose(tiff_handle);
    return EXIT_FAILURE;
  }
  if (PHOTOMETRIC_RGB != tiff_photometric && PHOTOMETRIC_MINISBLACK != tiff_photometric) {
    image_read_error = open_htj2k::detail::format(
        "TIFF input is limited to TIFFTAG_PHOTOMETRIC=PHOTOMETRIC_MINISBLACK (%d) or PHOTOMETRIC_RGB (%d); "
        "%s "
        "has TIFFTAG_PHOTOMETRIC=%d.",
        PHOTOMETRIC_MINISBLACK, PHOTOMETRIC_RGB, filename.c_str(), tiff_photometric);
    TIFFClose(tiff_handle);
    return EXIT_FAILURE;
  }
  this->num_components    = tiff_samples_per_pixel;
  uint16_t num_iterations = this->num_components;
  // setting bit-depth for components
  this->width  = tiff_width;
  this->height = tiff_height;
  for (int i = 0; i < num_iterations; ++i) {
    this->component_width.push_back(width);
    this->component_height.push_back(height);
    this->bits_per_pixel.push_back(static_cast<uint8_t>(tiff_bits_per_sample));
    this->is_signed.push_back(false);
  }

  const uint32_t byte_per_sample = (tiff_bits_per_sample + 8U - 1U) / 8U;
  const uint32_t component_gap   = num_iterations * byte_per_sample;
  // const uint32_t line_width = component_gap * tiff_width;
  std::unique_ptr<uint8_t[]> line_buf = MAKE_UNIQUE<uint8_t[]>(bytes_per_line);

  // allocate memory once
  if (this->buf == nullptr) {
    this->buf = MAKE_UNIQUE<std::unique_ptr<int32_t[]>[]>(this->num_components);
  }
  for (size_t i = 0; i < this->num_components; ++i) {
    this->buf[i] =
        MAKE_UNIQUE<int32_t[]>(static_cast<size_t>(this->component_width[i]) * this->component_height[i]);
  }

  if (PLANARCONFIG_SEPARATE == planar_configuration) {
    // PLANAR read
  } else if (PLANARCONFIG_CONTIG == planar_configuration) {
    // NORMAL read
    for (uint32_t i = 0; i < tiff_height; ++i) {
      if (TIFFReadScanline(tiff_handle, line_buf.get(), i) < 0) {
        image_read_error = open_htj2k::detail::format("not enough samples in %s.", filename.c_str());
        TIFFClose(tiff_handle);
        return EXIT_FAILURE;
      }

      for (size_t c = 0; c < num_iterations; ++c) {
        uint8_t *src;
        int32_t *dst;
        src = &line_buf[c * byte_per_sample];
        dst = &this->buf[c][static_cast<size_t>(i) * tiff_width];

        switch (byte_per_sample) {
          case 1:
            for (size_t j = 0; j < tiff_width; ++j) {
              *dst = *src;
              dst++;
              src += component_gap;
            }
            break;
          case 2:
            for (size_t j = 0; j < tiff_width; ++j) {
              // suppose little-endian
              *dst = src[0] | (src[1] << 8);
              // *dst >>= 16 - tiff_bits_per_sample; // for future use
              dst++;
              src += component_gap;
            }
            break;
          default:
            image_read_error = open_htj2k::detail::format("bit-depth over 16 is not supported.");
            TIFFClose(tiff_handle);
            return EXIT_FAILURE;
            break;
        }
      }
    }
  }
  return EXIT_SUCCESS;
}
#else
// Built without libtiff: the declaration is unconditional so that the public
// header does not depend on the build configuration.
int image::read_tiff(const std::string &) { return EXIT_FAILURE; }
#endif

// An instance created with num_threads == 1 runs on the calling thread only, even when
// another instance has created the shared pool (see ThreadPool::SingleThreadScope).
#ifdef OPENHTJ2K_THREAD
  #define OPENHTJ2K_SINGLE_THREAD_SCOPE(single) ThreadPool::SingleThreadScope single_thread_scope(single)
#else
  #define OPENHTJ2K_SINGLE_THREAD_SCOPE(single) (void)(single)
#endif

// openhtj2k_encoder_impl shall not be public
class openhtj2k_encoder_impl {
 private:
#ifdef OPENHTJ2K_THREAD
  // First member: keeps the shared pool alive until everything below is gone.
  ThreadPoolLease pool_lease_;
#endif
  std::string outfile;
  const std::vector<int32_t *> *buf;
  std::vector<uint8_t> *outbuf;
  siz_params *siz;
  cod_params *cod;
  qcd_params *qcd;
  uint8_t qfactor;
  file_format format;  // codestream, or the codestream in a JPH or JP2 file
  bool single_thread_ = false;  // num_threads == 1: never use the shared pool
  uint8_t color_space;
  // EXPERIMENTAL: analytic visual (CSF) weighting. Default-constructed selects
  // the legacy table, so output is unchanged unless set_visual_weighting() is called.
  visual_weighting_params vw{};
  // EXPERIMENTAL: HT cleanup pass this many bit-planes above the LSB, followed by
  // HT SigProp and MagRef passes (0 = cleanup pass only).  See set_ht_refinement().
  uint8_t ht_refine = 0;
  // EXPERIMENTAL: codestream size the code-blocks are truncated to (0 = off), and the
  // size of each quality layer with the layers below it (the last is target_size).
  uint64_t target_size = 0;
  std::vector<uint64_t> layer_targets;

  size_t invoke_internal();
  // The brand of the file type box (file_type_box): 0 for JP2, 1 for JPH.
  uint8_t brand() const { return (format == file_format::jp2) ? 0 : 1; }
  void check_file_format() const;
  void check_ht_refinement() const;
  void check_rate_control() const;
  // Codestream bytes outside the tile-parts (SOC, main header with the TLM marker that
  // multi-tile codestreams get, EOC); main_header is complete except for the TLM marker.
  static uint64_t fixed_bytes(j2k_main_header &main_header, uint32_t num_tiles);
  // The threshold estimate the code-blocks are coded for, with every tile's code-blocks.
  std::unique_ptr<ht_rc_state> start_rate_control(j2k_tile *tiles, uint32_t num_tiles,
                                                  j2k_main_header &main_header) const;
  // Truncates the code-blocks of every tile to the target size and builds their packet
  // headers.
  void apply_rate_control(j2k_tile *tiles, uint32_t num_tiles, j2k_main_header &main_header) const;

 public:
  openhtj2k_encoder_impl(const char *, const std::vector<int32_t *> &, siz_params &, cod_params &,
                         qcd_params &, uint8_t, bool, uint8_t);
  void set_output_buffer(std::vector<uint8_t> &);
  void set_file_format(file_format f) { format = f; }
  void set_single_thread(bool single) { single_thread_ = single; }
  bool single_thread() const { return single_thread_; }
  void set_visual_weighting(uint8_t model, double ref_ppd, double zoom);
  void set_component_types(uint8_t c0, uint8_t c1, uint8_t c2);
  void set_chroma_csf_reuse_luma(bool reuse_luma);
  void set_ht_refinement(uint8_t planes) { ht_refine = planes; }
  void set_target_size(uint64_t bytes) {
    target_size = bytes;
    layer_targets.assign(bytes ? 1 : 0, bytes);
  }
  void set_target_sizes(const std::vector<uint64_t> &bytes) {
    layer_targets = bytes;
    target_size   = bytes.empty() ? 0 : bytes.back();
  }
  ~openhtj2k_encoder_impl();
  size_t invoke_line_based();
  size_t invoke_line_based_stream(std::function<void(uint32_t, int32_t **, uint16_t)> src_fn);
};

openhtj2k_encoder_impl::openhtj2k_encoder_impl(const char *filename,
                                               const std::vector<int32_t *> &input_buf, siz_params &s,
                                               cod_params &c, qcd_params &q, uint8_t qf, bool flag,
                                               uint8_t cs)
    : buf(&input_buf),
      siz(&s),
      cod(&c),
      qcd(&q),
      qfactor(qf),
      format(flag ? file_format::jph : file_format::codestream),
      color_space(cs) {
  this->outfile = filename;
  this->outbuf  = nullptr;
}

void openhtj2k_encoder_impl::set_output_buffer(std::vector<uint8_t> &output_buf) {
  this->outbuf = &output_buf;
}

// A JP2 file (ISO/IEC 15444-1 Annex I) carries a Part 1 codestream: the HT block coder
// needs a JPH file, and Part 2 extensions a JPX file.
void openhtj2k_encoder_impl::check_file_format() const {
  if (format != file_format::codestream && format != file_format::jph && format != file_format::jp2) {
    open_htj2k::detail::fail("unknown file format %d.", static_cast<int>(format));
  }
  if (format != file_format::jp2) return;
  const char *why = nullptr;
  if (cod->codeblock_style & HT)
    why = "the Part 1 block coder (Cmodes without HT); HT code-blocks go in a JPH file";
  else if (cod->progression_order == 5)
    why = "a Part 1 progression order (PRCL is an ISO/IEC 15444-2 extension)";
  else if (siz->Rsiz & 0xC000)
    why = "Rsiz without the ISO/IEC 15444-2 (bit 15) and CAP marker (bit 14) capabilities";
  if (why != nullptr) {
    open_htj2k::detail::fail("the JP2 file format requires %s.", why);
  }
}

void openhtj2k_encoder_impl::check_ht_refinement() const {
  if (ht_refine && (cod->codeblock_style & HT) == 0) {
    open_htj2k::detail::fail("HT SigProp/MagRef refinement requires the HT block coder.");
  }
}

void openhtj2k_encoder_impl::check_rate_control() const {
  if (target_size == 0) return;
  const char *why = nullptr;
  bool rising     = true;
  for (size_t l = 0; l < layer_targets.size(); ++l)
    if (layer_targets[l] == 0 || (l > 0 && layer_targets[l] < layer_targets[l - 1])) rising = false;
  if (cod->transformation != 0)
    why = "the irreversible 9/7 wavelet (Creversible=no)";
  else if (layer_targets.size() != cod->number_of_layers)
    why = "one target per quality layer";
  else if (!rising)
    why = "positive targets that do not decrease from layer to layer";
  else if (layer_targets.size() > 1 && (cod->codeblock_style & HT))
    why = "the Part 1 block coder (Cmodes without HT) for more than one quality layer";
  else if (qfactor != NO_QFACTOR)
    why = "a base step size instead of Qfactor";
  else if (ht_refine != 0)
    why = "Cht_refine=0";
  if (why != nullptr) {
    open_htj2k::detail::fail("a target size requires %s.", why);
  }
}

uint64_t openhtj2k_encoder_impl::fixed_bytes(j2k_main_header &main_header, uint32_t num_tiles) {
  // The TLM marker takes 6 bytes per tile, and 6 for each marker segment of at most 10921
  // tiles (TLM_marker::write()).
  j2c_dst_memory hdr;
  hdr.put_word(_SOC);
  main_header.flush(hdr);
  uint64_t fixed = hdr.get_length() + 2;
  if (num_tiles > 1) fixed += 6 * ((num_tiles + 10920) / 10921) + 6 * static_cast<uint64_t>(num_tiles);
  return fixed;
}

std::unique_ptr<ht_rc_state> openhtj2k_encoder_impl::start_rate_control(
    j2k_tile *tiles, uint32_t num_tiles, j2k_main_header &main_header) const {
  // The code-blocks' share: the target less the fixed bytes, the tile-part markers and
  // (a guess, for the estimate only) 1.5% for the packet headers.
  const uint64_t fixed = fixed_bytes(main_header, num_tiles) + 14 * static_cast<uint64_t>(num_tiles);
  const double room    = (target_size > fixed) ? static_cast<double>(target_size - fixed) : 0.0;
  // Part 1 code-blocks: a larger budget for the estimate, whose models are HT's.
  const double scale = (cod->codeblock_style & HT) ? 1.0 : P1_ESTIMATE_BUDGET;
  auto rc            = MAKE_UNIQUE<ht_rc_state>(static_cast<uint64_t>(room * 0.985 * scale));
  for (uint32_t i = 0; i < num_tiles; ++i) {
    rc->add_tile(tiles[i]);
    tiles[i].enc_rc = rc.get();
  }
  return rc;
}

void openhtj2k_encoder_impl::apply_rate_control(j2k_tile *tiles, uint32_t num_tiles,
                                                j2k_main_header &main_header) const {
  const uint64_t fixed = fixed_bytes(main_header, num_tiles);
  const uint64_t total = (cod->codeblock_style & HT)
                             ? ht_rate_control(tiles, num_tiles, fixed, target_size)
                             : p1_rate_control(tiles, num_tiles, fixed, layer_targets);
  if (total > target_size)
    open_htj2k::detail::warn("the codestream takes %llu bytes, more than the target %llu.",
                             static_cast<unsigned long long>(total),
                             static_cast<unsigned long long>(target_size));
  // The lower quality layers (the packet headers are built).
  for (size_t l = 0; l + 1 < layer_targets.size(); ++l) {
    uint64_t bytes = fixed;
    for (uint32_t t = 0; t < num_tiles; ++t) bytes += tiles[t].get_layers_size(static_cast<uint16_t>(l));
    if (bytes > layer_targets[l])
      open_htj2k::detail::warn(
          "quality layer %zu and the layers below it take %llu bytes, more than the target "
          "%llu.",
          l, static_cast<unsigned long long>(bytes), static_cast<unsigned long long>(layer_targets[l]));
  }
}

void openhtj2k_encoder_impl::set_visual_weighting(uint8_t model, double ref_ppd, double zoom) {
  // model: 0 = legacy table (default, bit-identical), 1 = Mannos-Sakrison, 2 = Daly.
  switch (model) {
    case 1:
      vw.model = csf_model::mannos_sakrison;
      break;
    case 2:
      vw.model = csf_model::daly;
      break;
    default:
      vw.model = csf_model::legacy_table;
      break;
  }
  if (ref_ppd > 0.0) vw.ref_ppd = ref_ppd;
  if (zoom > 0.0) vw.zoom = zoom;

  // EXPERIMENTAL status / guard (qfactor is already set by the constructor).
  if (vw.model != csf_model::legacy_table) {
    if (qfactor == NO_QFACTOR) {
      open_htj2k::detail::warn("analytic visual weighting (Qcsf) has no effect without Qfactor.");
    } else {
      open_htj2k::detail::info(
          "EXPERIMENTAL: analytic visual weighting (model=%d, ref_ppd=%.1f, zoom=%.2f).",
          static_cast<int>(vw.model), vw.ref_ppd, vw.zoom);
    }
  }
}

void openhtj2k_encoder_impl::set_component_types(uint8_t c0, uint8_t c1, uint8_t c2) {
  const uint8_t t[3] = {c0, c1, c2};
  for (int i = 0; i < 3; ++i) {
    // 0 = generic, 1 = Y, 2 = Cb, 3 = Cr; anything else falls back to generic.
    vw.ctype_hint[i] = (t[i] <= 3) ? static_cast<component_type>(t[i]) : component_type::generic;
  }
}

void openhtj2k_encoder_impl::set_chroma_csf_reuse_luma(bool reuse_luma) {
  vw.chroma_reuse_luma_csf = reuse_luma;
}

openhtj2k_encoder_impl::~openhtj2k_encoder_impl() = default;

size_t openhtj2k_encoder_impl::invoke_internal() {
  std::vector<uint8_t> Ssiz;
  std::vector<uint8_t> XRsiz, YRsiz;

  check_parameters(*siz, *cod, *qcd, qfactor);
  check_tiling(*siz);

  for (size_t c = 0; c < siz->Csiz; ++c) {
    Ssiz.push_back(siz->Ssiz[c]);
    XRsiz.push_back(siz->XRsiz[c]);
    YRsiz.push_back(siz->YRsiz[c]);
  }

  // check component size
  // The MCT (on components 0-2) requires identically-sized components: a mismatch on EITHER axis
  // (e.g. 4:2:2, which differs only horizontally) must force it off, or the
  // color transform runs across differently-sized buffers and corrupts memory.
  if (siz->Csiz >= 3 && cod->use_color_trafo == 1
      && (XRsiz[0] != XRsiz[1] || XRsiz[1] != XRsiz[2] || YRsiz[0] != YRsiz[1] || YRsiz[1] != YRsiz[2])) {
    cod->use_color_trafo = 0;
    open_htj2k::detail::warn("Cycc is set to 'no' because size of each component is not identical.");
  }
  // The colour transform mixes the samples of the three components, so they also need the same bit
  // depth and signedness; otherwise no component decodes losslessly.
  if (siz->Csiz >= 3 && cod->use_color_trafo == 1 && (Ssiz[0] != Ssiz[1] || Ssiz[1] != Ssiz[2])) {
    cod->use_color_trafo = 0;
    open_htj2k::detail::warn(
        "Cycc is set to 'no' because the components differ in bit depth or signedness.");
  }

  // check number of components
  if (siz->Csiz < 3 && cod->use_color_trafo == 1) {
    cod->use_color_trafo = 0;
    open_htj2k::detail::warn("Cycc is set to 'no' because there are fewer than 3 components.");
  }
  check_reversible_precision(*siz, *cod);
  // Qfactor assumes a YCbCr (luma/chroma) decomposition; warn (do not force) if
  // the color transform is off or the component count is unsupported.
  if (qfactor != NO_QFACTOR) {
    if (siz->Csiz == 3 || siz->Csiz == 4) {
      if (cod->use_color_trafo == 0) {
        open_htj2k::detail::warn(
            "Color conversion is OFF while Qfactor feature is enabled. It is OK if the inputs are in YCbCr "
            "color space.");
      }
    } else if (siz->Csiz != 1 && siz->Csiz != 2) {
      open_htj2k::detail::warn(
          "Qfactor is designed for gray-scale or RGB or YCbCr input, with or without an alpha channel.");
    }
  }

  // create required marker segments
  // Part 15 (HT) block coding is signalled by Rsiz bit 14 and a CAP marker, Part 1
  // (EBCOT) block coding by neither.
  const bool is_ht = (cod->codeblock_style & HT) != 0;
  check_codeblock_style(cod->codeblock_style);
  // PRCL is an ISO/IEC 15444-2 progression order, signalled in CAP (Ccap2): Rsiz bit 15 marks
  // the Part 2 codestream that the CAP marker's Part 2 capabilities belong to.
  const auto Rsiz = static_cast<uint16_t>(siz->Rsiz | ((cod->progression_order == 5) ? 0x8000 : 0));
  SIZ_marker main_SIZ(Rsiz, siz->Xsiz, siz->Ysiz, siz->XOsiz, siz->YOsiz, siz->XTsiz, siz->YTsiz,
                      siz->XTOsiz, siz->YTOsiz, siz->Csiz, Ssiz, XRsiz, YRsiz,
                      is_ht || cod->progression_order == 5);  // Rsiz bit 14: a CAP marker follows
  COD_marker main_COD(cod->is_max_precincts, cod->use_SOP, cod->use_EPH, cod->progression_order,
                      cod->number_of_layers, cod->use_color_trafo, cod->dwt_levels,
                      static_cast<uint8_t>(cod->blkwidth), static_cast<uint8_t>(cod->blkheight),
                      cod->codeblock_style, cod->transformation, cod->PPx, cod->PPy);
  QCD_marker main_QCD(qcd->number_of_guardbits, cod->dwt_levels, cod->transformation, qcd->is_derived,
                      static_cast<uint8_t>((Ssiz[0] & 0x7F) + 1U), cod->use_color_trafo, qcd->base_step,
                      qfactor, vw);
  // parameters for CAP marker
  uint16_t bits14_15 = 0;                     // 0: HTONLY, 2: HTDECLARED, 3: MIXED
  uint16_t bit13     = 0;                     // 0: SINGLEHT, 1: MULTIHT
  uint16_t bit12     = 0;                     // 0: RGNFREE, 1: RGN
  uint16_t bit11     = 0;                     // 0: HOMOGENEOUS, 1: HETEROGENEOUS
  uint16_t bit5      = !cod->transformation;  // 0: HTREV, 1: HTIRV

  std::vector<std::unique_ptr<QCC_marker>> main_QCC = qcc_for_other_depths(Ssiz, *cod, *qcd, qfactor, vw);
  // Bits 0-4, the magnitude bound, are set once the main header has its QCC markers, and again
  // once the code-blocks are coded.
  const auto Ccap15 = static_cast<uint16_t>((bits14_15 << 14) + (bit13 << 13) + (bit12 << 12)
                                            + (bit11 << 11) + (bit5 << 5));
  CAP_marker main_CAP;
  bool need_CAP = is_ht;
  if (is_ht) main_CAP.set_Ccap(Ccap15, 15);
  if (cod->progression_order == 5) {
    need_CAP = true;
    // ISO/IEC 15444-2, Table A.49: the position-resolution level-component-layer progression
    // order (PRCL) is a Part-2 extension; signal it via bit-14 of Ccap2.
    main_CAP.set_Ccap(static_cast<uint16_t>(1U << 14), 2);
  }

  // create main header
  j2k_main_header main_header(&main_SIZ, &main_COD, &main_QCD, need_CAP ? &main_CAP : nullptr, qfactor, vw);
  for (auto &q : main_QCC) main_header.QCC.push_back(std::move(q));
  if (is_ht)
    main_header.CAP->set_Ccap(
        static_cast<uint16_t>(Ccap15 | magb_field(quantization_MAGB(main_header, cod->dwt_levels))), 15);
  COM_marker main_COM("OpenHTJ2K version 0", true);
  main_header.add_COM_marker(main_COM);

  element_siz numTiles;
  main_header.get_number_of_tiles(numTiles.x, numTiles.y);
  if (numTiles.x * numTiles.y > 65535) {
    open_htj2k::detail::fail("The number of tiles exceeds its allowable maximum (65535).");
  }

  check_file_format();
  check_ht_refinement();
  check_rate_control();
  auto tileSet = MAKE_UNIQUE<j2k_tile[]>(static_cast<size_t>(numTiles.x) * numTiles.y);
  for (uint16_t i = 0; i < static_cast<uint16_t>(numTiles.x * numTiles.y); ++i) {
    tileSet[i].enc_ht_refine    = ht_refine;
    tileSet[i].enc_rate_control = (target_size != 0);
    tileSet[i].enc_init(i, main_header, *buf, true);
  }
  std::unique_ptr<ht_rc_state> rc;
  if (target_size != 0) rc = start_rate_control(tileSet.get(), numTiles.x * numTiles.y, main_header);
  for (uint32_t i = 0; i < numTiles.x * numTiles.y; ++i) {
    tileSet[i].perform_dc_offset(main_header);
    tileSet[i].rgb_to_ycbcr();
    tileSet[i].encode_line_based();
  }

  const uint32_t num_tiles = numTiles.x * numTiles.y;
  if (target_size != 0) apply_rate_control(tileSet.get(), num_tiles, main_header);
  if (is_ht)
    main_header.CAP->set_Ccap(
        static_cast<uint16_t>(Ccap15 | magb_field(coded_MAGB(tileSet.get(), num_tiles))), 15);

  // Multi-tile: generate TLM marker from computed tile-part lengths.
  if (num_tiles > 1) {
    std::vector<uint16_t> tile_indices(num_tiles);
    std::vector<uint32_t> tile_lengths(num_tiles);
    for (uint32_t i = 0; i < num_tiles; ++i) {
      tile_indices[i] = static_cast<uint16_t>(i);
      tile_lengths[i] = tileSet[i].get_tile_part_size();
    }
    main_header.TLM.push_back(MAKE_UNIQUE<TLM_marker>(static_cast<uint8_t>(0), tile_indices, tile_lengths));
  }

  // Flush SOC + main header into a small buffer to measure its size and to
  // have content ready for both the buffer and file output paths.
  j2c_dst_memory hdr_buf;
  hdr_buf.put_word(_SOC);
  main_header.flush(hdr_buf);

  // Compute total codestream size upfront (needed for JPH boxes and return).
  size_t codestream_size = hdr_buf.get_length();  // SOC + main header
  for (uint32_t i = 0; i < num_tiles; ++i) codestream_size += tileSet[i].get_tile_part_size();
  codestream_size += 2;  // EOC

  if (outbuf != nullptr) {
    // Buffer output: write directly into the caller's vector.
    // Extract header bytes via flush to a small temporary vector.
    std::vector<uint8_t> hdr_bytes;
    hdr_buf.flush(&hdr_bytes);

    if (format != file_format::codestream) {
      bool isSRGB = (color_space == static_cast<uint8_t>(sRGB));
      jph_boxes jph_info(main_header, brand(), isSRGB, codestream_size);
      j2c_dst_memory jph_tmp;
      size_t file_format_size = jph_info.write(jph_tmp);
      std::vector<uint8_t> jph_bytes;
      jph_tmp.flush(&jph_bytes);

      outbuf->resize(jph_bytes.size() + codestream_size);
      uint8_t *p = outbuf->data();
      std::memcpy(p, jph_bytes.data(), jph_bytes.size());
      p += jph_bytes.size();
      std::memcpy(p, hdr_bytes.data(), hdr_bytes.size());
      p += hdr_bytes.size();
      for (uint32_t i = 0; i < num_tiles; ++i) p += tileSet[i].write_packets_direct(main_header, p);
      p[0]            = 0xFF;
      p[1]            = 0xD9;
      codestream_size = file_format_size;
    } else {
      outbuf->resize(codestream_size);
      uint8_t *p = outbuf->data();
      std::memcpy(p, hdr_bytes.data(), hdr_bytes.size());
      p += hdr_bytes.size();
      for (uint32_t i = 0; i < num_tiles; ++i) p += tileSet[i].write_packets_direct(main_header, p);
      p[0] = 0xFF;
      p[1] = 0xD9;
    }
  } else {
    // File output: write directly to ofstream, bypassing j2c_dst entirely.
    std::ofstream dst(this->outfile, std::ios::out | std::ios::binary);
    if (format != file_format::codestream) {
      bool isSRGB = (color_space == static_cast<uint8_t>(sRGB));
      jph_boxes jph_info(main_header, brand(), isSRGB, codestream_size);
      auto &jph_dst = enc_thread_bufs().jph_dst;
      jph_dst.clear();
      size_t file_format_size = jph_info.write(jph_dst);
      jph_dst.flush(dst);
      codestream_size = file_format_size;
    }
    hdr_buf.flush(dst);
    for (uint32_t i = 0; i < num_tiles; ++i) tileSet[i].write_packets_direct(main_header, dst);
    uint8_t eoc[2] = {0xFF, 0xD9};
    dst.write(reinterpret_cast<const char *>(eoc), 2);
    dst.close();
  }
  return codestream_size;
}

size_t openhtj2k_encoder_impl::invoke_line_based() { return invoke_internal(); }

size_t openhtj2k_encoder_impl::invoke_line_based_stream(
    std::function<void(uint32_t, int32_t **, uint16_t)> src_fn) {
  std::vector<uint8_t> Ssiz;
  std::vector<uint8_t> XRsiz, YRsiz;

  check_parameters(*siz, *cod, *qcd, qfactor);
  check_tiling(*siz);

  for (size_t c = 0; c < siz->Csiz; ++c) {
    Ssiz.push_back(siz->Ssiz[c]);
    XRsiz.push_back(siz->XRsiz[c]);
    YRsiz.push_back(siz->YRsiz[c]);
  }

  // The MCT (on components 0-2) requires identically-sized components: a mismatch on EITHER axis
  // (e.g. 4:2:2, which differs only horizontally) must force it off, or the
  // color transform runs across differently-sized buffers and corrupts memory.
  if (siz->Csiz >= 3 && cod->use_color_trafo == 1
      && (XRsiz[0] != XRsiz[1] || XRsiz[1] != XRsiz[2] || YRsiz[0] != YRsiz[1] || YRsiz[1] != YRsiz[2])) {
    cod->use_color_trafo = 0;
    open_htj2k::detail::warn("Cycc is set to 'no' because size of each component is not identical.");
  }
  // The colour transform mixes the samples of the three components, so they also need the same bit
  // depth and signedness; otherwise no component decodes losslessly.
  if (siz->Csiz >= 3 && cod->use_color_trafo == 1 && (Ssiz[0] != Ssiz[1] || Ssiz[1] != Ssiz[2])) {
    cod->use_color_trafo = 0;
    open_htj2k::detail::warn(
        "Cycc is set to 'no' because the components differ in bit depth or signedness.");
  }
  if (siz->Csiz < 3 && cod->use_color_trafo == 1) {
    cod->use_color_trafo = 0;
    open_htj2k::detail::warn("Cycc is set to 'no' because there are fewer than 3 components.");
  }
  check_reversible_precision(*siz, *cod);
  // Rows are pulled for component 0; a component takes one when component 0's row is on its
  // sampling grid, which needs its vertical subsampling factor to be a multiple of component 0's.
  for (uint16_t c = 1; c < siz->Csiz; ++c) {
    if (siz->YRsiz[c] % siz->YRsiz[0] != 0) {
      open_htj2k::detail::fail(
          "invoke_line_based_stream() needs the vertical subsampling factor of every component to be a "
          "multiple of component 0's (component %u: YRsiz %u, component 0: YRsiz %u); "
          "invoke_line_based() takes any.",
          static_cast<unsigned>(c), static_cast<unsigned>(siz->YRsiz[c]),
          static_cast<unsigned>(siz->YRsiz[0]));
    }
  }
  if (qfactor != NO_QFACTOR) {
    if (siz->Csiz == 3 || siz->Csiz == 4) {
      if (cod->use_color_trafo == 0) {
        open_htj2k::detail::warn(
            "Color conversion is OFF while Qfactor feature is enabled. It is OK if the inputs are in YCbCr "
            "color space.");
      }
    } else if (siz->Csiz != 1 && siz->Csiz != 2) {
      open_htj2k::detail::warn(
          "Qfactor is designed for gray-scale or RGB or YCbCr input, with or without an alpha channel.");
    }
  }

  // Part 15 (HT) block coding is signalled by Rsiz bit 14 and a CAP marker, Part 1
  // (EBCOT) block coding by neither.
  const bool is_ht = (cod->codeblock_style & HT) != 0;
  check_codeblock_style(cod->codeblock_style);
  // PRCL is an ISO/IEC 15444-2 progression order, signalled in CAP (Ccap2): Rsiz bit 15 marks
  // the Part 2 codestream that the CAP marker's Part 2 capabilities belong to.
  const auto Rsiz = static_cast<uint16_t>(siz->Rsiz | ((cod->progression_order == 5) ? 0x8000 : 0));
  SIZ_marker main_SIZ(Rsiz, siz->Xsiz, siz->Ysiz, siz->XOsiz, siz->YOsiz, siz->XTsiz, siz->YTsiz,
                      siz->XTOsiz, siz->YTOsiz, siz->Csiz, Ssiz, XRsiz, YRsiz,
                      is_ht || cod->progression_order == 5);  // Rsiz bit 14: a CAP marker follows
  COD_marker main_COD(cod->is_max_precincts, cod->use_SOP, cod->use_EPH, cod->progression_order,
                      cod->number_of_layers, cod->use_color_trafo, cod->dwt_levels,
                      static_cast<uint8_t>(cod->blkwidth), static_cast<uint8_t>(cod->blkheight),
                      cod->codeblock_style, cod->transformation, cod->PPx, cod->PPy);
  QCD_marker main_QCD(qcd->number_of_guardbits, cod->dwt_levels, cod->transformation, qcd->is_derived,
                      static_cast<uint8_t>((Ssiz[0] & 0x7F) + 1U), cod->use_color_trafo, qcd->base_step,
                      qfactor, vw);
  uint16_t bits14_15 = 0;
  uint16_t bit13     = 0;
  uint16_t bit12     = 0;
  uint16_t bit11     = 0;
  uint16_t bit5      = !cod->transformation;

  std::vector<std::unique_ptr<QCC_marker>> main_QCC = qcc_for_other_depths(Ssiz, *cod, *qcd, qfactor, vw);
  // Bits 0-4, the magnitude bound, are set once the main header has its QCC markers, and again
  // once the code-blocks are coded.
  const auto Ccap15 = static_cast<uint16_t>((bits14_15 << 14) + (bit13 << 13) + (bit12 << 12)
                                            + (bit11 << 11) + (bit5 << 5));
  CAP_marker main_CAP;
  bool need_CAP = is_ht;
  if (is_ht) main_CAP.set_Ccap(Ccap15, 15);
  if (cod->progression_order == 5) {
    need_CAP = true;
    // ISO/IEC 15444-2, Table A.49: the position-resolution level-component-layer progression
    // order (PRCL) is a Part-2 extension; signal it via bit-14 of Ccap2.
    main_CAP.set_Ccap(static_cast<uint16_t>(1U << 14), 2);
  }

  j2k_main_header main_header(&main_SIZ, &main_COD, &main_QCD, need_CAP ? &main_CAP : nullptr, qfactor, vw);
  for (auto &q : main_QCC) main_header.QCC.push_back(std::move(q));
  if (is_ht)
    main_header.CAP->set_Ccap(
        static_cast<uint16_t>(Ccap15 | magb_field(quantization_MAGB(main_header, cod->dwt_levels))), 15);
  COM_marker main_COM("OpenHTJ2K version 0", true);
  main_header.add_COM_marker(main_COM);

  element_siz numTiles;
  main_header.get_number_of_tiles(numTiles.x, numTiles.y);
  if (numTiles.x * numTiles.y > 65535) {
    open_htj2k::detail::fail("The number of tiles exceeds its allowable maximum (65535).");
  }

  // Empty input buffer for streaming (img is not used)
  std::vector<int32_t *> empty_buf;
  const uint32_t num_tiles_lbs = numTiles.x * numTiles.y;
  check_file_format();
  check_ht_refinement();
  check_rate_control();
  auto tileSet = MAKE_UNIQUE<j2k_tile[]>(static_cast<size_t>(num_tiles_lbs));
  for (uint16_t i = 0; i < static_cast<uint16_t>(num_tiles_lbs); ++i) {
    tileSet[i].enc_ht_refine    = ht_refine;
    tileSet[i].enc_rate_control = (target_size != 0);
    tileSet[i].enc_init(i, main_header, empty_buf, true, true);
  }
  std::unique_ptr<ht_rc_state> rc;
  if (target_size != 0) rc = start_rate_control(tileSet.get(), num_tiles_lbs, main_header);
  for (uint32_t i = 0; i < num_tiles_lbs; ++i) {
    std::vector<uint32_t> img_comp_widths(static_cast<size_t>(siz->Csiz));
    for (size_t c = 0; c < siz->Csiz; ++c) {
      const uint32_t xr = static_cast<uint32_t>(siz->XRsiz[c]);
      img_comp_widths[c] =
          (static_cast<uint32_t>(siz->Xsiz) - static_cast<uint32_t>(siz->XOsiz) + xr - 1) / xr;
    }
    tileSet[i].perform_dc_offset(main_header);
    tileSet[i].encode_line_based_stream(src_fn, img_comp_widths);
  }
  if (target_size != 0) apply_rate_control(tileSet.get(), num_tiles_lbs, main_header);
  if (is_ht)
    main_header.CAP->set_Ccap(
        static_cast<uint16_t>(Ccap15 | magb_field(coded_MAGB(tileSet.get(), num_tiles_lbs))), 15);

  // Multi-tile: generate TLM marker from computed tile-part lengths.
  if (num_tiles_lbs > 1) {
    std::vector<uint16_t> tile_indices(num_tiles_lbs);
    std::vector<uint32_t> tile_lengths(num_tiles_lbs);
    for (uint32_t i = 0; i < num_tiles_lbs; ++i) {
      tile_indices[i] = static_cast<uint16_t>(i);
      tile_lengths[i] = tileSet[i].get_tile_part_size();
    }
    main_header.TLM.push_back(MAKE_UNIQUE<TLM_marker>(static_cast<uint8_t>(0), tile_indices, tile_lengths));
  }

  // Flush SOC + main header into a small buffer to measure its size and to
  // have content ready for the file output path.
  j2c_dst_memory hdr_buf;
  hdr_buf.put_word(_SOC);
  main_header.flush(hdr_buf);

  // Compute total codestream size upfront (needed for JPH boxes and return).
  size_t codestream_size = hdr_buf.get_length();
  for (uint32_t i = 0; i < num_tiles_lbs; ++i) codestream_size += tileSet[i].get_tile_part_size();
  codestream_size += 2;  // EOC

  if (outbuf != nullptr) {
    // Buffer output: write directly into the caller's vector.
    std::vector<uint8_t> hdr_bytes;
    hdr_buf.flush(&hdr_bytes);

    if (format != file_format::codestream) {
      bool isSRGB = (color_space == static_cast<uint8_t>(sRGB));
      jph_boxes jph_info(main_header, brand(), isSRGB, codestream_size);
      j2c_dst_memory jph_tmp;
      size_t file_format_size = jph_info.write(jph_tmp);
      std::vector<uint8_t> jph_bytes;
      jph_tmp.flush(&jph_bytes);

      outbuf->resize(jph_bytes.size() + codestream_size);
      uint8_t *p = outbuf->data();
      std::memcpy(p, jph_bytes.data(), jph_bytes.size());
      p += jph_bytes.size();
      std::memcpy(p, hdr_bytes.data(), hdr_bytes.size());
      p += hdr_bytes.size();
      for (uint32_t i = 0; i < num_tiles_lbs; ++i) p += tileSet[i].write_packets_direct(main_header, p);
      p[0]            = 0xFF;
      p[1]            = 0xD9;
      codestream_size = file_format_size;
    } else {
      outbuf->resize(codestream_size);
      uint8_t *p = outbuf->data();
      std::memcpy(p, hdr_bytes.data(), hdr_bytes.size());
      p += hdr_bytes.size();
      for (uint32_t i = 0; i < num_tiles_lbs; ++i) p += tileSet[i].write_packets_direct(main_header, p);
      p[0] = 0xFF;
      p[1] = 0xD9;
    }
  } else {
    // File output: write directly to ofstream, bypassing j2c_dst entirely.
    std::ofstream dst(this->outfile, std::ios::out | std::ios::binary);
    if (format != file_format::codestream) {
      bool isSRGB = (color_space == static_cast<uint8_t>(sRGB));
      jph_boxes jph_info(main_header, brand(), isSRGB, codestream_size);
      auto &jph_dst = enc_thread_bufs().jph_dst;
      jph_dst.clear();
      size_t file_format_size = jph_info.write(jph_dst);
      jph_dst.flush(dst);
      codestream_size = file_format_size;
    }
    hdr_buf.flush(dst);
    for (uint32_t i = 0; i < num_tiles_lbs; ++i) tileSet[i].write_packets_direct(main_header, dst);
    uint8_t eoc[2] = {0xFF, 0xD9};
    dst.write(reinterpret_cast<const char *>(eoc), 2);
    dst.close();
  }
  return codestream_size;
}

openhtj2k_encoder::openhtj2k_encoder(const char *fname, const std::vector<int32_t *> &input_buf,
                                     siz_params &siz, cod_params &cod, qcd_params &qcd, uint8_t qfactor,
                                     bool isJPH, uint8_t color_space, uint32_t num_threads) {
  if (qfactor != NO_QFACTOR) {
    if (qfactor > open_htj2k::qfactor_max) {
      open_htj2k::detail::fail("Value of Qfactor shall be in the range [0, %u].",
                               static_cast<unsigned>(open_htj2k::qfactor_max));
    }
  }
  // The impl takes a pool reference on construction, so create it before
  // asking for the pool.
  this->impl =
      MAKE_UNIQUE<openhtj2k_encoder_impl>(fname, input_buf, siz, cod, qcd, qfactor, isJPH, color_space);
  this->impl->set_single_thread(num_threads == 1);
#ifdef OPENHTJ2K_THREAD
  ThreadPool::instance(num_threads);
#endif
}

void openhtj2k_encoder::set_output_buffer(std::vector<uint8_t> &output_buf) {
  this->impl->set_output_buffer(output_buf);
}

void openhtj2k_encoder::set_file_format(file_format format) { this->impl->set_file_format(format); }

void openhtj2k_encoder::set_ht_refinement(uint8_t planes) { this->impl->set_ht_refinement(planes); }

void openhtj2k_encoder::set_target_size(uint64_t bytes) { this->impl->set_target_size(bytes); }

void openhtj2k_encoder::set_target_sizes(const std::vector<uint64_t> &bytes) {
  this->impl->set_target_sizes(bytes);
}

void openhtj2k_encoder::set_visual_weighting(uint8_t model, double ref_ppd, double zoom) {
  this->impl->set_visual_weighting(model, ref_ppd, zoom);
}

void openhtj2k_encoder::set_component_types(uint8_t c0, uint8_t c1, uint8_t c2) {
  this->impl->set_component_types(c0, c1, c2);
}

void openhtj2k_encoder::set_chroma_csf_reuse_luma(bool reuse_luma) {
  this->impl->set_chroma_csf_reuse_luma(reuse_luma);
}

size_t openhtj2k_encoder::invoke_line_based() {
  OPENHTJ2K_SINGLE_THREAD_SCOPE(this->impl->single_thread());
  return this->impl->invoke_line_based();
}

size_t openhtj2k_encoder::invoke_line_based_stream(
    std::function<void(uint32_t, int32_t **, uint16_t)> src_fn) {
  OPENHTJ2K_SINGLE_THREAD_SCOPE(this->impl->single_thread());
  return this->impl->invoke_line_based_stream(src_fn);
}

// The pool reference is dropped by the impl's ThreadPoolLease.
openhtj2k_encoder::~openhtj2k_encoder() = default;
}  // namespace open_htj2k
