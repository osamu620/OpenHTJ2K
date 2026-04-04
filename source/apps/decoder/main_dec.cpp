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

// open_htj2k_dec: A decoder implementation for JPEG 2000 Part 1 and 15
// (ITU-T Rec. 814 | ISO/IEC 15444-15 and ITU-T Rec. 814 | ISO/IEC 15444-15)
//
// (c) 2019 - 2021 Osamu Watanabe, Takushoku University, Vrije Universiteit Brussels

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#ifdef _OPENMP
  #include <omp.h>
#endif

#include "decoder.hpp"
#include "dec_utils.hpp"

// ---------------------------------------------------------------------------
// Minimal JPH / JP2 box parser
// ---------------------------------------------------------------------------
// Box type constants (big-endian 4CC as uint32_t)
static constexpr uint32_t BOX_JP   = 0x6A502020u;  // "jP  " — signature
static constexpr uint32_t BOX_JP2H = 0x6A703268u;  // "jp2h" — header superbox
static constexpr uint32_t BOX_COLR = 0x636F6C72u;  // "colr" — colour spec
static constexpr uint32_t BOX_JP2C = 0x6A703263u;  // "jp2c" — codestream

// EnumCS values in the colr box
static constexpr uint32_t ENUMCS_SRGB      = 16u;
static constexpr uint32_t ENUMCS_GRAYSCALE = 17u;
static constexpr uint32_t ENUMCS_YCBCR     = 18u;

static inline uint32_t read_be32(const uint8_t *p) {
  return (static_cast<uint32_t>(p[0]) << 24) | (static_cast<uint32_t>(p[1]) << 16)
         | (static_cast<uint32_t>(p[2]) << 8) | static_cast<uint32_t>(p[3]);
}
static inline uint64_t read_be64(const uint8_t *p) {
  return (static_cast<uint64_t>(read_be32(p)) << 32) | static_cast<uint64_t>(read_be32(p + 4));
}

struct JphInfo {
  const uint8_t *cs_data = nullptr;  // pointer into file buffer at start of J2K codestream
  size_t         cs_size = 0;        // length of codestream in bytes
  uint32_t       enum_cs = 0;        // EnumCS from colr box (0 = not found)
};

// Iterates boxes in [begin, end) and fills `out`.  Recurses into jp2h.
static bool scan_boxes(const uint8_t *begin, const uint8_t *end, JphInfo &out) {
  const uint8_t *p = begin;
  while (p + 8 <= end) {
    uint32_t lbox = read_be32(p);
    uint32_t tbox = read_be32(p + 4);
    const uint8_t *payload;
    const uint8_t *box_end;
    if (lbox == 1) {
      if (p + 16 > end) return false;
      uint64_t xlen = read_be64(p + 8);
      if (xlen < 16 || p + xlen > end) return false;
      payload = p + 16;
      box_end = p + xlen;
    } else if (lbox == 0) {
      payload = p + 8;
      box_end = end;
    } else {
      if (lbox < 8 || p + lbox > end) return false;
      payload = p + 8;
      box_end = p + lbox;
    }
    if (tbox == BOX_JP2H) {
      scan_boxes(payload, box_end, out);
    } else if (tbox == BOX_COLR) {
      // METH(1) PREC(1) APPROX(1) EnumCS(4) — only for METH==1
      if (payload + 7 <= box_end && payload[0] == 1)
        out.enum_cs = read_be32(payload + 3);
    } else if (tbox == BOX_JP2C) {
      out.cs_data = payload;
      out.cs_size = static_cast<size_t>(box_end - payload);
    }
    p = box_end;
  }
  return true;
}

// Reads the file at `path` into `buf`, then extracts JPH metadata.
// Returns true on success; `buf` keeps the data alive.
static bool parse_jph_file(const char *path, std::vector<uint8_t> &buf, JphInfo &out) {
  FILE *fp = fopen(path, "rb");
  if (!fp) { printf("ERROR: Cannot open %s\n", path); return false; }
  fseek(fp, 0, SEEK_END);
  long flen = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  if (flen <= 0) { fclose(fp); printf("ERROR: Empty file %s\n", path); return false; }
  buf.resize(static_cast<size_t>(flen));
  if (fread(buf.data(), 1, buf.size(), fp) != buf.size()) {
    fclose(fp); printf("ERROR: Failed to read %s\n", path); return false;
  }
  fclose(fp);
  // Validate signature box (LBox=0x0000000C, TBox="jP  ", payload=0x0D0A870A)
  if (buf.size() < 12 || read_be32(buf.data()) != 12 || read_be32(buf.data() + 4) != BOX_JP
      || read_be32(buf.data() + 8) != 0x0D0A870Au) {
    printf("ERROR: Not a valid JPH/JP2 file (bad signature).\n"); return false;
  }
  scan_boxes(buf.data(), buf.data() + buf.size(), out);
  if (!out.cs_data || out.cs_size == 0) {
    printf("ERROR: No codestream box (jp2c) found in %s\n", path); return false;
  }
  return true;
}
// ---------------------------------------------------------------------------

void print_help(char *cmd) {
  printf("JPEG 2000 Part 1 and Part 15 decoder\n");
  printf("USAGE: %s [options]\n\n", cmd);
  printf("OPTIONS:\n");
  printf("-i: Input file. .j2k, .j2c, .jhc, and .jph are supported.\n");
  printf("-o: Output file. Supported formats are PPM, PGM, PGX and RAW.\n");
  printf("-reduce n: Number of DWT resolution reduction.\n");
  printf("-iter n: Repeat decoding n times (for benchmarking). Output is written once.\n");
  printf("-num_threads n: Number of threads (0 = auto).\n");
  printf("-batch: Use batch (full-image buffer) decode path instead of the default streaming path.\n");
  printf("-ycbcr bt601|bt709: [EXPERIMENTAL] Convert YCbCr to RGB (PPM output only).\n");
}

int main(int argc, char *argv[]) {
  // parse input args
  char *infile_name, *infile_ext_name;
  char *outfile_name, *outfile_ext_name;
  if (command_option_exists(argc, argv, "-h") || argc < 2) {
    print_help(argv[0]);
    exit(EXIT_SUCCESS);
  }
  if (nullptr == (infile_name = get_command_option(argc, argv, "-i"))) {
    printf("ERROR: Input file is missing. Use -i to specify input file.\n");
    exit(EXIT_FAILURE);
  }
  infile_ext_name = strrchr(infile_name, '.');
  const bool is_jph = (strcmp(infile_ext_name, ".jph") == 0 || strcmp(infile_ext_name, ".JPH") == 0);
  if (!is_jph && strcmp(infile_ext_name, ".j2k") != 0 && strcmp(infile_ext_name, ".j2c") != 0
      && strcmp(infile_ext_name, ".jhc") != 0) {
    printf("ERROR: Supported extensions are .j2k, .j2c, .jhc, .jph\n");
    exit(EXIT_FAILURE);
  }
  if (nullptr == (outfile_name = get_command_option(argc, argv, "-o"))) {
    printf(
        "ERROR: Output files are missing. Use -o to specify output file "
        "names.\n");
    exit(EXIT_FAILURE);
  }
  outfile_ext_name = strrchr(outfile_name, '.');
  if (strcmp(outfile_ext_name, ".pgm") != 0 && strcmp(outfile_ext_name, ".ppm") != 0
      && strcmp(outfile_ext_name, ".raw") != 0 && strcmp(outfile_ext_name, ".pgx") != 0) {
    printf("ERROR: Unsupported output file type.\n");
    exit(EXIT_FAILURE);
  }
  char *tmp_param, *endptr;
  long tmp_val;
  uint8_t reduce_NL;
  if (nullptr == (tmp_param = get_command_option(argc, argv, "-reduce"))) {
    reduce_NL = 0;
  } else {
    tmp_val = strtol(tmp_param, &endptr, 10);
    if (tmp_val >= 0 && tmp_val <= 32 && tmp_param != endptr) {
      reduce_NL = static_cast<uint8_t>(tmp_val);
    } else {
      printf("ERROR: -reduce takes non-negative integer in the range from 0 to 32.\n");
      exit(EXIT_FAILURE);
    }
  }
  int32_t num_iterations;
  if (nullptr == (tmp_param = get_command_option(argc, argv, "-iter"))) {
    num_iterations = 1;
  } else {
    tmp_val = strtol(tmp_param, &endptr, 10);
    if (tmp_param == endptr) {
      printf("ERROR: -iter takes positive integer.\n");
      exit(EXIT_FAILURE);
    }
    if (tmp_val < 1 || tmp_val > INT32_MAX) {
      printf("ERROR: -iter takes positive integer ( < INT32_MAX).\n");
      exit(EXIT_FAILURE);
    }
    num_iterations = static_cast<int32_t>(tmp_val);
  }

  uint32_t num_threads;
  if (nullptr == (tmp_param = get_command_option(argc, argv, "-num_threads"))) {
    num_threads = 0;
  } else {
    tmp_val = strtol(tmp_param, &endptr, 10);
    if (tmp_param == endptr) {
      printf("ERROR: -num_threads takes non-negative integer.\n");
      exit(EXIT_FAILURE);
    }
    if (tmp_val < 0 || tmp_val > UINT32_MAX) {
      printf("ERROR: -num_threads takes non-negative integer ( < UINT32_MAX).\n");
      exit(EXIT_FAILURE);
    }
    //    num_iterations = static_cast<int32_t>(tmp_val);
    num_threads = static_cast<uint32_t>(tmp_val);  // strtoul(tmp_param, nullptr, 10);
  }
  // Reject any unrecognised flags.
  {
    static const char *const known[] = {
        "-h", "-i", "-o", "-reduce", "-iter", "-num_threads", "-batch", "-ycbcr", nullptr};
    for (int i = 1; i < argc; ++i) {
      if (argv[i][0] != '-') continue;
      bool recognised = false;
      for (int k = 0; known[k]; ++k) {
        if (strcmp(argv[i], known[k]) == 0) { recognised = true; break; }
      }
      if (!recognised) {
        printf("ERROR: unknown option %s\n", argv[i]);
        exit(EXIT_FAILURE);
      }
    }
  }

  // For JPH inputs: parse boxes once, extract codestream and colorspace.
  std::vector<uint8_t> jph_buf;
  JphInfo              jph_info;
  if (is_jph) {
    if (!parse_jph_file(infile_name, jph_buf, jph_info)) exit(EXIT_FAILURE);
    if (jph_info.enum_cs == ENUMCS_SRGB)
      printf("INFO: JPH colorspace: sRGB\n");
    else if (jph_info.enum_cs == ENUMCS_GRAYSCALE)
      printf("INFO: JPH colorspace: Grayscale\n");
    else if (jph_info.enum_cs == ENUMCS_YCBCR)
      printf("INFO: JPH colorspace: YCbCr\n");
    else if (jph_info.enum_cs != 0)
      printf("INFO: JPH colorspace: unknown EnumCS %u\n", jph_info.enum_cs);
  }

  // Parse experimental -ycbcr flag (PPM output only).
  bool               do_ycbcr  = false;
  ycbcr_coefficients ycbcr_coeff{};
  // Auto-enable for JPH files with YCbCr colorspace (BT.601 by default).
  if (is_jph && jph_info.enum_cs == ENUMCS_YCBCR && strcmp(outfile_ext_name, ".ppm") == 0) {
    do_ycbcr    = true;
    ycbcr_coeff = YCBCR_BT601;
  }
  if (nullptr != (tmp_param = get_command_option(argc, argv, "-ycbcr"))) {
    if (strcmp(tmp_param, "bt601") == 0) {
      do_ycbcr    = true;
      ycbcr_coeff = YCBCR_BT601;
    } else if (strcmp(tmp_param, "bt709") == 0) {
      do_ycbcr    = true;
      ycbcr_coeff = YCBCR_BT709;
    } else {
      printf("ERROR: -ycbcr takes 'bt601' or 'bt709'.\n");
      exit(EXIT_FAILURE);
    }
    if (strcmp(outfile_ext_name, ".ppm") != 0) {
      printf("WARNING: -ycbcr has no effect for non-PPM output.\n");
    }
  }
  if (do_ycbcr && strcmp(outfile_ext_name, ".ppm") == 0) {
    const bool is601 = (ycbcr_coeff.cr_to_r == YCBCR_BT601.cr_to_r);
    if (is_jph && jph_info.enum_cs == ENUMCS_YCBCR && get_command_option(argc, argv, "-ycbcr") == nullptr)
      printf("INFO: YCbCr→RGB conversion auto-enabled (BT.601). Use -ycbcr bt709 to override.\n");
    else
      printf("INFO: YCbCr→RGB conversion enabled (%s).\n", is601 ? "BT.601" : "BT.709");
  }

  const bool use_batch = command_option_exists(argc, argv, "-batch");

  std::vector<uint32_t> img_width;
  std::vector<uint32_t> img_height;
  std::vector<uint8_t> img_depth;
  std::vector<bool> img_signed;

  auto start = std::chrono::high_resolution_clock::now();

  // Batch path: decode entire image into full-image buffers, then write.
  if (use_batch) {
    std::vector<int32_t *> buf;
    for (int32_t i = 0; i < num_iterations; ++i) {
      open_htj2k::openhtj2k_decoder decoder =
          is_jph ? open_htj2k::openhtj2k_decoder(jph_info.cs_data, jph_info.cs_size, reduce_NL,
                                                  num_threads)
                 : open_htj2k::openhtj2k_decoder(infile_name, reduce_NL, num_threads);
      for (auto &p : buf) delete[] p;
      buf.clear();
      img_width.clear();
      img_height.clear();
      img_depth.clear();
      img_signed.clear();
      try {
        decoder.parse();
        decoder.invoke(buf, img_width, img_height, img_depth, img_signed);
      } catch (std::exception &exc) {
        printf("ERROR: %s\n", exc.what());
        return EXIT_FAILURE;
      }
    }
    auto duration       = std::chrono::high_resolution_clock::now() - start;
    auto num_components = static_cast<uint16_t>(img_depth.size());
    if (strcmp(outfile_ext_name, ".ppm") == 0) {
      write_ppm(outfile_name, outfile_ext_name, buf, img_width, img_height, img_depth, img_signed,
                do_ycbcr ? &ycbcr_coeff : nullptr);
    } else {
      write_components(outfile_name, outfile_ext_name, buf, img_width, img_height, img_depth,
                       img_signed);
    }
    uint32_t total_samples = 0;
    for (uint16_t c = 0; c < num_components; ++c) {
      total_samples += img_width[c] * img_height[c];
      delete[] buf[c];
    }
    auto count = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
    printf("elapsed time %-15.3lf[ms]\n",
           static_cast<double>(count) / 1000.0 / static_cast<double>(num_iterations));
    printf("throughput %lf [Msamples/s]\n",
           total_samples * static_cast<double>(num_iterations) / static_cast<double>(count));
    printf("throughput %lf [usec/sample]\n",
           static_cast<double>(count) / static_cast<double>(num_iterations) / total_samples);
    return EXIT_SUCCESS;
  }

  const bool want_ppm = (strcmp(outfile_ext_name, ".ppm") == 0);
  const bool want_pgm = (strcmp(outfile_ext_name, ".pgm") == 0);
  const bool want_pgx = (strcmp(outfile_ext_name, ".pgx") == 0);

  // Output file handles (opened lazily on the last iteration's first row).
  std::vector<FILE *> fps;
  std::vector<uint8_t> row_buf;
  uint8_t bpp            = 0;
  int32_t pnm_offset     = 0;
  uint32_t total_samples = 0;
  // State for experimental YCbCr→RGB streaming conversion.
  int32_t ycbcr_cb_center = 0, ycbcr_cr_center = 0, ycbcr_maxval = 0;

  for (int32_t i = 0; i < num_iterations; ++i) {
    const bool is_last = (i == num_iterations - 1);

    open_htj2k::openhtj2k_decoder decoder =
        is_jph ? open_htj2k::openhtj2k_decoder(jph_info.cs_data, jph_info.cs_size, reduce_NL,
                                                num_threads)
               : open_htj2k::openhtj2k_decoder(infile_name, reduce_NL, num_threads);
    try {
      decoder.parse();
    } catch (std::exception &exc) {
      printf("ERROR: %s\n", exc.what());
      return EXIT_FAILURE;
    }

    try {
      decoder.invoke_line_based_stream(
          [&](uint32_t y, int32_t *const *rows, uint16_t nc) {
            if (!is_last) return;  // warm-up iterations: decode but discard output

            if (y == 0) {
              total_samples = 0;
              bpp           = static_cast<uint8_t>(ceil_int(static_cast<int32_t>(img_depth[0]), 8));
              pnm_offset    = (want_pgm && img_signed[0]) ? (1 << (img_depth[0] - 1)) : 0;
              if (do_ycbcr && nc >= 3) {
                ycbcr_maxval    = (1 << img_depth[0]) - 1;
                ycbcr_cb_center = img_signed[1] ? 0 : (1 << (img_depth[1] - 1));
                ycbcr_cr_center = img_signed[2] ? 0 : (1 << (img_depth[2] - 1));
              }
              if (want_ppm && nc == 3) {
                // Single PPM file: chroma is nearest-neighbour upsampled during row write.
                char fname[256], base[256];
                memcpy(base, outfile_name, static_cast<size_t>(outfile_ext_name - outfile_name));
                base[outfile_ext_name - outfile_name] = '\0';
                snprintf(fname, sizeof(fname), "%s%s", base, outfile_ext_name);
                FILE *fp = fopen(fname, "wb");
                fprintf(fp, "P6 %d %d %d\n", img_width[0], img_height[0],
                        (1 << img_depth[0]) - 1);
                fps.push_back(fp);
                fps.push_back(nullptr);
                fps.push_back(nullptr);
                row_buf.resize(static_cast<size_t>(img_width[0]) * 3 * bpp);
              } else {
                // One file per component (PGM, PGX, or RAW)
                fps.resize(nc, nullptr);
                for (uint16_t c = 0; c < nc; ++c) {
                  char fname[256], base[256];
                  memcpy(base, outfile_name,
                         static_cast<size_t>(outfile_ext_name - outfile_name));
                  base[outfile_ext_name - outfile_name] = '\0';
                  snprintf(fname, sizeof(fname), "%s_%02d%s", base, c, outfile_ext_name);
                  fps[c] = fopen(fname, "wb");
                  if (want_pgm)
                    fprintf(fps[c], "P5 %d %d %d\n", img_width[c], img_height[c],
                            (1 << img_depth[c]) - 1);
                  if (want_pgx) {
                    char sign = img_signed[c] ? '-' : '+';
                    fprintf(fps[c], "PG LM %c %d %d %d\n", sign, img_depth[c], img_width[c],
                            img_height[c]);
                  }
                }
                uint32_t max_w = 0;
                for (uint16_t c = 0; c < nc; ++c)
                  max_w = std::max(max_w, img_width[c]);
                row_buf.resize(static_cast<size_t>(max_w) * bpp);
              }
              for (uint16_t c = 0; c < nc; ++c)
                total_samples += img_width[c] * img_height[c];
            }

            const uint16_t nc_all = static_cast<uint16_t>(fps.size());
            if (want_ppm && nc_all >= 3 && fps[0] != nullptr && fps[1] == nullptr) {
              // Interleaved PPM row
              uint8_t *out = row_buf.data();
              if (do_ycbcr) {
                // YCbCr→RGB conversion (experimental); chroma upsampled as needed.
                if (bpp == 1) {
                  for (uint32_t n = 0; n < img_width[0]; ++n) {
                    const uint32_t n1 = n * img_width[1] / img_width[0];
                    const uint32_t n2 = n * img_width[2] / img_width[0];
                    const int32_t Y   = rows[0][n];
                    const int32_t Cb  = rows[1][n1] - ycbcr_cb_center;
                    const int32_t Cr  = rows[2][n2] - ycbcr_cr_center;
                    int32_t r = Y + ((ycbcr_coeff.cr_to_r * Cr + 8192) >> 14);
                    int32_t g = Y - ((ycbcr_coeff.cb_to_g * Cb + ycbcr_coeff.cr_to_g * Cr + 8192) >> 14);
                    int32_t b = Y + ((ycbcr_coeff.cb_to_b * Cb + 8192) >> 14);
                    *out++ = static_cast<uint8_t>(r < 0 ? 0 : r > ycbcr_maxval ? ycbcr_maxval : r);
                    *out++ = static_cast<uint8_t>(g < 0 ? 0 : g > ycbcr_maxval ? ycbcr_maxval : g);
                    *out++ = static_cast<uint8_t>(b < 0 ? 0 : b > ycbcr_maxval ? ycbcr_maxval : b);
                  }
                } else {
                  for (uint32_t n = 0; n < img_width[0]; ++n) {
                    const uint32_t n1 = n * img_width[1] / img_width[0];
                    const uint32_t n2 = n * img_width[2] / img_width[0];
                    const int32_t Y   = rows[0][n];
                    const int32_t Cb  = rows[1][n1] - ycbcr_cb_center;
                    const int32_t Cr  = rows[2][n2] - ycbcr_cr_center;
                    int32_t r = Y + ((ycbcr_coeff.cr_to_r * Cr + 8192) >> 14);
                    int32_t g = Y - ((ycbcr_coeff.cb_to_g * Cb + ycbcr_coeff.cr_to_g * Cr + 8192) >> 14);
                    int32_t b = Y + ((ycbcr_coeff.cb_to_b * Cb + 8192) >> 14);
                    r         = r < 0 ? 0 : r > ycbcr_maxval ? ycbcr_maxval : r;
                    g         = g < 0 ? 0 : g > ycbcr_maxval ? ycbcr_maxval : g;
                    b         = b < 0 ? 0 : b > ycbcr_maxval ? ycbcr_maxval : b;
                    *out++    = static_cast<uint8_t>(r >> 8); *out++ = static_cast<uint8_t>(r);
                    *out++    = static_cast<uint8_t>(g >> 8); *out++ = static_cast<uint8_t>(g);
                    *out++    = static_cast<uint8_t>(b >> 8); *out++ = static_cast<uint8_t>(b);
                  }
                }
              } else if (bpp == 1) {
                for (uint32_t n = 0; n < img_width[0]; ++n) {
                  const uint32_t n1 = n * img_width[1] / img_width[0];
                  const uint32_t n2 = n * img_width[2] / img_width[0];
                  *out++ = static_cast<uint8_t>(rows[0][n] + pnm_offset);
                  *out++ = static_cast<uint8_t>(rows[1][n1] + pnm_offset);
                  *out++ = static_cast<uint8_t>(rows[2][n2] + pnm_offset);
                }
              } else {
                for (uint32_t n = 0; n < img_width[0]; ++n) {
                  const uint32_t n1 = n * img_width[1] / img_width[0];
                  const uint32_t n2 = n * img_width[2] / img_width[0];
                  int32_t r = rows[0][n] + pnm_offset;
                  int32_t g = rows[1][n1] + pnm_offset;
                  int32_t b = rows[2][n2] + pnm_offset;
                  *out++    = static_cast<uint8_t>(r >> 8);
                  *out++    = static_cast<uint8_t>(r);
                  *out++    = static_cast<uint8_t>(g >> 8);
                  *out++    = static_cast<uint8_t>(g);
                  *out++    = static_cast<uint8_t>(b >> 8);
                  *out++    = static_cast<uint8_t>(b);
                }
              }
              fwrite(row_buf.data(), 1, row_buf.size(), fps[0]);
            } else {
              // Per-component files.
              // For vertically subsampled components (yr_c > 1, e.g. 4:2:0 chroma):
              // decode_line_based_stream holds the component row stable across yr_c luma
              // rows and only advances the ring buffer every yr_c luma rows.  So we write
              // once per yr_c luma rows (at y%yr_c==0) and use y/yr_c as the component-
              // space row boundary.  For non-subsampled components (yr_c==1) the
              // behaviour is identical to the original.
              // Use ceiling division to recover YRsiz: ceil(H0/Hc), which is exact for
              // any valid JPEG 2000 subsampling factor (including odd-height images where
              // integer division would give the wrong answer).
              for (uint16_t c = 0; c < nc_all; ++c) {
                if (fps[c] == nullptr) continue;
                const uint32_t h0 = img_height[0], hc = img_height[c];
                const uint32_t yr_c = (hc > 0 && hc < h0) ? (h0 + hc - 1) / hc : 1u;
                if (y % yr_c != 0) continue;         // duplicate luma row: skip
                if (y / yr_c >= hc) continue;         // past end of component
                uint8_t *out = row_buf.data();
                if (bpp == 1) {
                  for (uint32_t n = 0; n < img_width[c]; ++n)
                    *out++ = static_cast<uint8_t>(rows[c][n] + pnm_offset);
                } else if (want_pgx) {
                  // PGX: little-endian (LM = LSB first), no offset
                  for (uint32_t n = 0; n < img_width[c]; ++n) {
                    int32_t v = rows[c][n];
                    *out++    = static_cast<uint8_t>(v);
                    *out++    = static_cast<uint8_t>(v >> 8);
                  }
                } else {
                  // PGM / other: big-endian with offset
                  for (uint32_t n = 0; n < img_width[c]; ++n) {
                    int32_t v = rows[c][n] + pnm_offset;
                    *out++    = static_cast<uint8_t>(v >> 8);
                    *out++    = static_cast<uint8_t>(v);
                  }
                }
                fwrite(row_buf.data(), 1, static_cast<size_t>(img_width[c]) * bpp, fps[c]);
              }
            }
          },
          img_width, img_height, img_depth, img_signed);
    } catch (std::exception &exc) {
      printf("ERROR: %s\n", exc.what());
    }
  }

  for (FILE *fp : fps)
    if (fp) fclose(fp);

  auto duration = std::chrono::high_resolution_clock::now() - start;
  auto count    = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
  printf("elapsed time %-15.3lf[ms]\n",
         static_cast<double>(count) / 1000.0 / static_cast<double>(num_iterations));
  printf("throughput %lf [Msamples/s]\n",
         total_samples * static_cast<double>(num_iterations) / static_cast<double>(count));
  printf("throughput %lf [usec/sample]\n",
         static_cast<double>(count) / static_cast<double>(num_iterations) / total_samples);
  return EXIT_SUCCESS;
}
