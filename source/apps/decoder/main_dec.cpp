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

void print_help(char *cmd) {
  printf("JPEG 2000 Part 1 and Part 15 decoder\n");
  printf("USAGE: %s [options]\n\n", cmd);
  printf("OPTIONS:\n");
  printf("-i: Input file. .j2k, .j2c, and .jhc are supported.\n");
  printf("    .jp2 and .jph (box based file-format) are not supported.\n");
  printf("-o: Output file. Supported formats are PPM, PGM, PGX and RAW.\n");
  printf("-reduce n: Number of DWT resolution reduction.\n");
  printf("-batch: Use batch (full-image) decode path instead of the default line-based path.\n");
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
  if (strcmp(infile_ext_name, ".j2k") != 0 && strcmp(infile_ext_name, ".j2c") != 0
      && strcmp(infile_ext_name, ".jhc") != 0) {
    printf("ERROR: Supported extensions are .j2k, .j2c, .jhc\n");
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
  const bool use_line_based = !command_option_exists(argc, argv, "-batch");

  std::vector<int32_t *> buf;
  std::vector<uint32_t> img_width;
  std::vector<uint32_t> img_height;
  std::vector<uint8_t> img_depth;
  std::vector<bool> img_signed;
  auto start = std::chrono::high_resolution_clock::now();

  // Streaming line-based decode: no full-image output buffers; writes rows to file as decoded.
  if (use_line_based && num_iterations == 1) {
    open_htj2k::openhtj2k_decoder decoder(infile_name, reduce_NL, num_threads);
    try {
      decoder.parse();
    } catch (std::exception &exc) {
      printf("ERROR: %s\n", exc.what());
      return EXIT_FAILURE;
    }

    // Output file state (opened lazily on first row).
    const bool want_ppm = (strcmp(outfile_ext_name, ".ppm") == 0);
    const bool want_pgm = (strcmp(outfile_ext_name, ".pgm") == 0);
    const bool want_pgx = (strcmp(outfile_ext_name, ".pgx") == 0);
    std::vector<FILE *> fps;
    std::vector<uint8_t> row_buf;  // per-row byte buffer (reused)
    uint8_t bpp            = 0;
    int32_t pnm_offset     = 0;
    uint32_t total_samples = 0;

    try {
      decoder.invoke_line_based_stream(
          [&](uint32_t y, int32_t *const *rows, uint16_t nc) {
            if (y == 0) {
              bpp = static_cast<uint8_t>(ceil_int(static_cast<int32_t>(img_depth[0]), 8));
              // PGM shifts signed values to unsigned; PGX and RAW store values as-is
              pnm_offset = (want_pgm && img_signed[0]) ? (1 << (img_depth[0] - 1)) : 0;
              if (want_ppm && nc == 3 && img_width[0] == img_width[1] &&
                  img_width[0] == img_width[2]) {
                // Single PPM file: interleaved RGB
                char fname[256], base[256];
                memcpy(base, outfile_name,
                       static_cast<size_t>(outfile_ext_name - outfile_name));
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

            const uint16_t nc_all = static_cast<uint16_t>(fps.size() > 0 ? fps.size() : 0);
            if (want_ppm && nc_all >= 3 && fps[0] != nullptr && fps[1] == nullptr) {
              // Interleaved PPM row
              uint8_t *out = row_buf.data();
              if (bpp == 1) {
                for (uint32_t n = 0; n < img_width[0]; ++n) {
                  *out++ = static_cast<uint8_t>(rows[0][n] + pnm_offset);
                  *out++ = static_cast<uint8_t>(rows[1][n] + pnm_offset);
                  *out++ = static_cast<uint8_t>(rows[2][n] + pnm_offset);
                }
              } else {
                for (uint32_t n = 0; n < img_width[0]; ++n) {
                  int32_t r = rows[0][n] + pnm_offset;
                  int32_t g = rows[1][n] + pnm_offset;
                  int32_t b = rows[2][n] + pnm_offset;
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
              // Per-component files
              for (uint16_t c = 0; c < static_cast<uint16_t>(fps.size()); ++c) {
                if (fps[c] == nullptr || y >= img_height[c]) continue;
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
    for (FILE *fp : fps)
      if (fp) fclose(fp);

    auto duration2  = std::chrono::high_resolution_clock::now() - start;
    auto count2     = std::chrono::duration_cast<std::chrono::microseconds>(duration2).count();
    double time2    = static_cast<double>(count2) / 1000.0;
    printf("elapsed time %-15.3lf[ms]\n", time2);
    printf("throughput %lf [Msamples/s]\n", total_samples / (double)count2);
    printf("throughput %lf [usec/sample]\n", (double)count2 / total_samples);
    return EXIT_SUCCESS;
  }

  for (int i = 0; i < num_iterations; ++i) {
    // create decoder
    open_htj2k::openhtj2k_decoder decoder(infile_name, reduce_NL, num_threads);
    for (auto &j : buf) {
      delete[] j;
    }
    buf.clear();
    img_width.clear();
    img_height.clear();
    img_depth.clear();
    img_signed.clear();
    // invoke decoding
    try {
      decoder.parse();
      if (use_line_based) {
        decoder.invoke_line_based(buf, img_width, img_height, img_depth, img_signed);
      } else {
        decoder.invoke(buf, img_width, img_height, img_depth, img_signed);
      }
    } catch (std::exception &exc) {
      printf("ERROR: %s\n", exc.what());
      return EXIT_FAILURE;
    }
  }
  auto duration = std::chrono::high_resolution_clock::now() - start;

  // write decoded components
  bool compositable   = false;
  auto num_components = static_cast<uint16_t>(img_depth.size());
  if (num_components == 3 && strcmp(outfile_ext_name, ".ppm") == 0) {
    compositable = true;
    for (uint16_t c = 0; c < num_components - 1; c++) {
      if (img_width[c] != img_width[c + 1U] || img_height[c] != img_height[c + 1U]) {
        compositable = false;
        break;
      }
    }
  }
  if (strcmp(outfile_ext_name, ".ppm") == 0) {
    // PPM
    if (!compositable) {
      printf("ERROR: the number of components of the input is not three.");
      exit(EXIT_FAILURE);
    }
    write_ppm(outfile_name, outfile_ext_name, buf, img_width, img_height, img_depth, img_signed);

  } else {
    // PGM or RAW
    write_components(outfile_name, outfile_ext_name, buf, img_width, img_height, img_depth, img_signed);
  }

  uint32_t total_samples = 0;
  for (uint16_t c = 0; c < num_components; ++c) {
    total_samples += img_width[c] * img_height[c];
    delete[] buf[c];
  }

  // show stats
  auto count  = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
  double time = static_cast<double>(count) / 1000.0 / static_cast<double>(num_iterations);
  printf("elapsed time %-15.3lf[ms]\n", time);
  printf("throughput %lf [Msamples/s]\n",
         total_samples * static_cast<double>(num_iterations) / (double)count);
  printf("throughput %lf [usec/sample]\n",
         (double)count / static_cast<double>(num_iterations) / total_samples);
  return EXIT_SUCCESS;
}
