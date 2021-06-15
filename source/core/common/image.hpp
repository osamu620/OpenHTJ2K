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

#include <cstdio>
#include <cassert>
#include <string>
#include <cstdint>
#include <vector>
#include <memory>

#include "utils.hpp"

#ifdef DEPRECATED_VIEWER
  #include <opencv2/core.hpp>       // for viewer
  #include <opencv2/highgui.hpp>    // for viewer
  #include <opencv2/imgcodecs.hpp>  // for viewer
  #include <opencv2/opencv.hpp>
#endif

class image {
 private:
  uint32_t width;
  uint32_t height;
  uint16_t num_components;
  std::vector<uint32_t> component_width;
  std::vector<uint32_t> component_height;
  std::vector<bool> is_signed;
  std::unique_ptr<std::unique_ptr<int32_t[]>[]> buf;
  std::vector<uint8_t> bits_per_pixel;

 public:
  image() : width(0), height(0), num_components(0), buf(nullptr) {
    // avoid compier warnings
    (void)width;
    (void)height;
  }
  explicit image(uint16_t c) {
    this->num_components = c;
    for (uint16_t i = 0; i < c; i++) {
      this->component_width.push_back(0);
      this->component_height.push_back(0);
      this->is_signed.push_back(false);
      this->bits_per_pixel.push_back(0);
    }
    this->buf = std::make_unique<std::unique_ptr<int32_t[]>[]>(this->num_components);
  }

  explicit image(const std::string &filename) {
    FILE *fp;
    int d, val = 0;
    int status = 0;
    char com[256];
    fp = fopen(filename.c_str(), "rb");
    if (fp == nullptr) {
      printf("ERROR: input file %s is not found\n", filename.c_str());
      exit(EXIT_FAILURE);
    }
    d = fgetc(fp);
    if (d != 'P') {
      printf("ERROR: input file is not compiant with PNM format.\n");
      exit(EXIT_FAILURE);
    }
    d = fgetc(fp);
    switch (d) {
      case '5':
        this->num_components = 1;
        break;
      case '6':
        this->num_components = 3;
        break;
      default:
        printf("ERROR: input file is not compiant with PNM format.\n");
        exit(EXIT_FAILURE);
        break;
    }

    while (status < 3) {
      d = fgetc(fp);
      while (d == ' ' || d == '\n') {
        d = fgetc(fp);
        if (d == '#') {
          fgets(com, sizeof(com), fp);
          d = fgetc(fp);
        }
      }

      while (d != ' ' && d != '\n') {
        val *= 10;
        val += d - '0';
        d = fgetc(fp);
      }

      switch (status) {
        case 0:
          this->width = val;
          for (int i = 0; i < this->num_components; ++i) {
            component_width.push_back(this->width);
          }
          val = 0;
          status++;
          break;
        case 1:
          this->height = val;
          for (int i = 0; i < this->num_components; ++i) {
            component_height.push_back(this->height);
          }
          val = 0;
          status++;
          break;
        case 2:
          for (int i = 0; i < this->num_components; ++i) {
            this->bits_per_pixel.push_back(static_cast<uint8_t>(int_log2(val) + 1));
          }
          val = 0;
          status++;
          break;
        default:
          break;
      }
    }
    this->buf = std::make_unique<std::unique_ptr<int32_t[]>[]>(this->num_components);
    for (int i = 0; i < this->num_components; ++i) {
      is_signed.push_back(false);
      this->buf[i] = std::make_unique<int32_t[]>(this->component_width[i] * this->component_height[i]);
    }

    const uint32_t byte_per_sample   = ceil_int(this->bits_per_pixel[0], 8);
    const uint32_t component_gap     = this->num_components * byte_per_sample;
    const uint32_t line_width        = component_gap * this->width;
    std::unique_ptr<uint8_t[]> imbuf = std::make_unique<uint8_t[]>(line_width);
    for (int i = 0; i < this->height; ++i) {
      fread(imbuf.get(), sizeof(uint8_t), line_width, fp);
#pragma omp parallel for
      for (int c = 0; c < this->num_components; ++c) {
        uint8_t *src = &imbuf[c * byte_per_sample];
        int32_t *dst = &this->buf[c][i * this->width];
        if (this->bits_per_pixel[c] > 16) {
          printf("ERROR: over  16 bit/pixel is not supported.\n");
          exit(EXIT_FAILURE);
        } else if (this->bits_per_pixel[c] > 8) {
          for (int j = 0; j < this->width; ++j) {
            *dst = (*src) << 8;  // suppose big endian
            *dst += *(src + 1);
            dst++;
            src += component_gap;
          }
        } else {
          for (int j = 0; j < this->width; ++j) {
            *dst = *src;
            dst++;
            src += component_gap;
          }
        }
      }
    }
    fclose(fp);
  }

  uint32_t get_width() const { return this->width; }

  uint32_t get_height() const { return this->height; }

  uint16_t get_num_components() const { return this->num_components; }

  uint8_t get_bpp(uint16_t c) { return this->bits_per_pixel[c]; }

  int32_t *get_buf(uint16_t c) { return this->buf[c].get(); }

  void show_params() {
    printf("width = %d, height = %d, num_components = %d\n", this->width, this->height,
           this->num_components);
    for (int c = 0; c < this->num_components; ++c) {
      printf("width[%d] = %d, height[%d] = %d, bpp[%d] = %d\n", c, this->component_width[c], c,
             this->component_height[c], c, this->bits_per_pixel[c]);
      // for (int i = 0; i < this->height; ++i) {
      //   for (int j = 0; j < this->width; ++j) {
      //     printf("%3d ", this->buf[c][i * this->width + j]);
      //   }
      //   printf("\n");
      // }
    }
  }
#ifdef DEPRECATED_VIEWER
  int show_decoded_image() {
    uint32_t cw       = component_width[0];
    uint32_t ch       = component_height[0];
    bool compositable = (this->num_components == 3);
    for (uint16_t c = 0; c < this->num_components; c++) {
      if (cw != component_width[c] || ch != component_height[c]) {
        compositable = false;
        break;
      }
    }
    std::vector<cv::Mat> orig_img;
    for (uint16_t c = 0; c < this->num_components; c++) {
      orig_img.emplace_back(this->component_height[c], this->component_width[c], CV_32SC1,
                            this->buf[c].get());
    }
    cv::Mat img;
    char window_name[256];
    sprintf(window_name, "test");
    std::string str = window_name;
    cv::namedWindow(str);
    int component   = 0;
    int cvtype_mono = (this->bits_per_pixel[0] > 8) ? CV_16UC1 : CV_8UC1;
    int cvtype_colr = (this->bits_per_pixel[0] > 8) ? CV_16UC3 : CV_8UC3;
    double cv_alpha = pow(2.0, (double)(ceil_int(this->bits_per_pixel[0], 8) * 8 - (bits_per_pixel[0])));
    double cv_beta  = (this->is_signed[0]) ? pow(2.0, (double)this->bits_per_pixel[0] - 1) : 0;
    cv_beta *= cv_alpha;
    orig_img[component].convertTo(img, cvtype_mono, cv_alpha, cv_beta);
    cv::imshow(str, img);
    int keycode;
    while (true) {
      keycode = cv::waitKey(0);

      switch (keycode) {
        case 43:  // '+', next component
          component++;
          if (component > this->num_components - 1) {
            component = 0;
          }
          orig_img[component].convertTo(img, cvtype_mono, cv_alpha, cv_beta);
          // sprintf(window_name, "component %d", component);
          cv::imshow(window_name, img);
          break;
        case 45:  // '-', previous component
          component--;
          if (component < 0) {
            component = this->num_components - 1;
          }
          orig_img[component].convertTo(img, cvtype_mono, cv_alpha, cv_beta);
          // sprintf(window_name, "component %d", component);
          cv::imshow(window_name, img);
          break;
        case 99:  // 'c', composit
          if (compositable) {
            cv::Mat cimg(this->component_height[0], this->component_width[0], cvtype_colr);
            cv::Mat tmp;

            cv::merge(orig_img, tmp);
            tmp.reshape(3);
            tmp.convertTo(cimg, cvtype_colr, cv_alpha, cv_beta);
            cv::Mat cimg2;
            cv::cvtColor(cimg, cimg2, cv::COLOR_BGR2RGB);
            // sprintf(window_name, "composite");
            cv::imshow(window_name, cimg2);
            // cv::imwrite("tmpcv.ppm", cimg2);
          }
        default:

          break;
      }
      printf("c = %d, keycode = %d\n", component, keycode);
      if (keycode == 113) {
        break;
      }
    }
    return 0;  // SUCCESS
  }
#endif
  uint32_t get_total_samples() {
    uint32_t val = 0;
    for (uint16_t c = 0; c < this->num_components; c++) {
      val += this->component_width[c] * this->component_height[c];
    }
    return val;
  }
  uint32_t get_total_bytes() {
    uint32_t val = 0;
    for (uint16_t c = 0; c < this->num_components; c++) {
      val += this->component_width[c] * this->component_height[c] * ceil_int(this->bits_per_pixel[c], 8);
    }
    return val;
  }
  void set_component_size(uint32_t w, uint32_t h, bool s, uint8_t bpp, uint16_t c) {
    assert(this->num_components != 0);
    assert(this->component_width.size() > c);
    this->component_width[c]  = w;
    this->component_height[c] = h;
    this->is_signed[c]        = s;
    this->bits_per_pixel[c]   = bpp;
    this->buf[c]              = std::make_unique<int32_t[]>(w * h);
  }
  void set_component_pixels(int32_t val, uint16_t c, const uint32_t x, const uint32_t y) {
    buf[c][x + y * component_width[c]] = val;
  }
};