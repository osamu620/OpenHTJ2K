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

#pragma once
#include <cstdint>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

namespace open_htj2k {
class image {
 private:
  uint32_t width;
  uint32_t height;
  uint16_t num_components;
  std::vector<uint32_t> component_width;
  std::vector<uint32_t> component_height;
  std::unique_ptr<std::unique_ptr<int32_t[]>[]> buf;
  std::vector<uint8_t> bits_per_pixel;
  std::vector<bool> is_signed;

 public:
  explicit image(const std::vector<std::string> &filenames) : width(0), height(0), buf(nullptr) {
    size_t num_files = filenames.size();
    if (num_files > 16384) {
      printf("ERROR: over 16384 components are not supported in the spec.\n");
      throw std::exception();
    }
    num_components = num_files;  // num_components may change after parsing PPM header
    uint16_t c     = 0;
    for (const auto &fname : filenames) {
      read_pnmpgx(fname, c);
      c++;
    }
  }

  int read_pnmpgx(const std::string &filename, const uint16_t nc) {
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
      printf("ERROR: File %s is not found.\n", filename.c_str());
      return EXIT_FAILURE;
    }
    int status = READ_WIDTH;
    int d;
    uint32_t val = 0;
    char comment[256];
    d = fgetc(fp);
    if (d != 'P') {
      printf("ERROR: %s is not a PNM/PGX file.\n", filename.c_str());
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
      case '5':
        isBigendian = true;
        break;
      // PPM
      case '3':
        isASCII = true;
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
              printf("ERROR: input PGX file %s is broken.\n", filename.c_str());
            }
            break;
          case 'L':
            d = fgetc(fp);
            if (d != 'M') {
              printf("ERROR: input PGX file %s is broken.\n", filename.c_str());
            }
            break;
          default:
            printf("ERROR: input file does not conform to PGX format.\n");
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
          val += d - '0';
          d = fgetc(fp);
        } while (d != SP && d != LF && d != CR);
        bitDepth = val;
        val      = 0;
        break;
      // PBM (not supported)
      case '1':
      case '4':
        printf("ERROR: PBM file is not supported.\n");
        fclose(fp);
        return EXIT_FAILURE;
        break;
      // error
      default:
        printf("ERROR: %s is not a PNM/PGX file.\n", filename.c_str());
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
          static_cast<void>(fgets(comment, sizeof(comment), fp));
          d = fgetc(fp);
        }
      }
      // read numerical value
      while (d != SP && d != LF && d != CR) {
        val *= 10;
        val += d - '0';
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
        static_cast<void>(fgets(comment, sizeof(comment), fp));
        d = fgetc(fp);
      }
    }
    fseek(fp, -1, SEEK_CUR);

    const uint32_t byte_per_sample      = (bitDepth + 8 - 1) / 8;
    const uint32_t component_gap        = num_iterations * byte_per_sample;
    const uint32_t line_width           = component_gap * compw;
    std::unique_ptr<uint8_t[]> line_buf = std::make_unique<uint8_t[]>(line_width);

    // allocate memory once
    if (this->buf == nullptr) {
      this->buf = std::make_unique<std::unique_ptr<int32_t[]>[]>(this->num_components);
    }
    if (isPPM) {
      for (size_t i = 0; i < this->num_components; ++i) {
        this->buf[i] = std::make_unique<int32_t[]>(compw * comph);
      }
    } else {
      this->buf[nc] = std::make_unique<int32_t[]>(compw * comph);
    }

    if (!isASCII) {
      for (size_t i = 0; i < comph; ++i) {
        if (fread(line_buf.get(), sizeof(uint8_t), line_width, fp) < line_width) {
          printf("ERROR: not enough samples in the given pnm file.\n");
          fclose(fp);
          return EXIT_FAILURE;
        }
#pragma omp parallel for
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
              printf("ERROR: bit-depth over 16 is not supported.\n");
              fclose(fp);
              return EXIT_FAILURE;
              break;
          }
        }
      }
    } else {
      for (size_t i = 0; i < compw * comph; ++i) {
        for (size_t c = 0; c < num_iterations; ++c) {
          val = 0;
          d   = fgetc(fp);
          while (d != SP && d != CR && d != LF && d != EOF) {
            val *= 10;
            val += d - '0';
            d = fgetc(fp);
          }
          this->buf[c][i] = val;
        }
      }
    }
    fclose(fp);
    return EXIT_SUCCESS;
  }

  uint32_t get_width() const { return this->width; }
  uint32_t get_height() const { return this->height; }
  uint32_t get_component_width(uint16_t c) const {
    if (c > num_components) {
      printf("ERROR: component index %d is larger than maximum value %d.\n", c, num_components);
      throw std::exception();
    }
    return this->component_width[c];
  }
  uint32_t get_component_height(uint16_t c) const {
    if (c > num_components) {
      printf("ERROR: component index %d is larger than maximum value %d.\n", c, num_components);
      throw std::exception();
    }
    return this->component_height[c];
  }
  uint16_t get_num_components() const { return this->num_components; }
  uint8_t get_Ssiz_value(uint16_t c) {
    return (this->is_signed[c]) ? (this->bits_per_pixel[c] - 1) | 0x80 : this->bits_per_pixel[c] - 1;
  }
  uint8_t get_max_bpp() {
    uint8_t max = 0;
    for (auto &v : bits_per_pixel) {
      max = (max < v) ? v : max;
    }
    return max;
  }
  int32_t *get_buf(uint16_t c) { return this->buf[c].get(); }
};

struct siz_params {
  uint16_t Rsiz;
  uint32_t Xsiz;
  uint32_t Ysiz;
  uint32_t XOsiz;
  uint32_t YOsiz;
  uint32_t XTsiz;
  uint32_t YTsiz;
  uint32_t XTOsiz;
  uint32_t YTOsiz;
  uint16_t Csiz;
  std::vector<uint8_t> Ssiz;
  std::vector<uint8_t> XRsiz;
  std::vector<uint8_t> YRsiz;
  // uint8_t bpp;
};

struct cod_params {
  uint16_t blkwidth;
  uint16_t blkheight;
  bool is_max_precincts;
  bool use_SOP;
  bool use_EPH;
  uint8_t progression_order;
  uint16_t number_of_layers;
  uint8_t use_color_trafo;
  uint8_t dwt_levels;
  uint8_t codeblock_style;
  uint8_t transformation;
  std::vector<uint8_t> PPx, PPy;
};

struct qcd_params {
  uint8_t number_of_guardbits;
  bool is_derived;
  double base_step;
};

class openhtj2k_encoder {
 private:
  std::unique_ptr<class openhtj2k_encoder_impl> impl;

 public:
#if defined(_MSC_VER)
  __declspec(dllexport)
      openhtj2k_encoder(const char *, const std::vector<int32_t *> &input_buf, siz_params &siz,
                        cod_params &cod, qcd_params &qcd, uint8_t qfactor, bool isJPH, uint8_t color_space,
                        uint32_t num_threads);
  __declspec(dllexport) size_t invoke();
  __declspec(dllexport) ~openhtj2k_encoder();
#else
  openhtj2k_encoder(const char *, const std::vector<int32_t *> &input_buf, siz_params &siz, cod_params &cod,
                    qcd_params &qcd, uint8_t qfactor, bool isJPH, uint8_t color_space,
                    uint32_t num_threads);
  size_t invoke();
  ~openhtj2k_encoder();
#endif
};
}  // namespace open_htj2k
