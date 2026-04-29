// Copyright (c) 2019 - 2026, Osamu Watanabe
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
// THE POSSIBILITY OF SUCH DAMAGE.

#include <cmath>
#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "imgio.hpp"

int main(int argc, char *argv[]) {
  if (argc != 3 && argc != 5) {
    printf("\nusage: imgcmp file1 file2 [PAE MSE]\n");
    printf("  (accepts pnm, pgx, or tiff files)\n");
    printf("  - PAE and MSE are threshold for conformance testing.\n\n");
    return EXIT_FAILURE;
  }

  auto img0 = imgio::open_stream_reader({std::string(argv[1])});
  auto img1 = imgio::open_stream_reader({std::string(argv[2])});
  if (!img0 || !img1) return EXIT_FAILURE;

  const uint32_t w  = img0->get_width();
  const uint32_t h  = img0->get_height();
  const uint16_t nc = img0->get_num_components();
  const uint8_t  bd = img0->get_bitdepth(0);

  if (w != img1->get_width() || h != img1->get_height()) {
    printf("width and height shall be the same\n");
    return EXIT_FAILURE;
  }
  if (nc != img1->get_num_components()) {
    printf("number of components shall be the same\n");
    return EXIT_FAILURE;
  }
  if (bd != img1->get_bitdepth(0)) {
    printf("bit-depth shall be the same\n");
    return EXIT_FAILURE;
  }

  // Per-component planar row buffers used by the StreamReader interface.
  std::vector<std::vector<int32_t>> rows0(nc, std::vector<int32_t>(w));
  std::vector<std::vector<int32_t>> rows1(nc, std::vector<int32_t>(w));
  std::vector<int32_t *> row0_ptrs(nc), row1_ptrs(nc);
  for (uint16_t c = 0; c < nc; ++c) {
    row0_ptrs[c] = rows0[c].data();
    row1_ptrs[c] = rows1[c].data();
  }

  uint64_t PAE = 0;
  uint64_t sum = 0;  // sum of squared differences
  for (uint32_t y = 0; y < h; ++y) {
    img0->get_row(y, row0_ptrs.data(), nc);
    img1->get_row(y, row1_ptrs.data(), nc);
    for (uint16_t c = 0; c < nc; ++c) {
      const int32_t *a = rows0[c].data();
      const int32_t *b = rows1[c].data();
      for (uint32_t x = 0; x < w; ++x) {
        const int64_t  d   = static_cast<int64_t>(a[x]) - static_cast<int64_t>(b[x]);
        const uint64_t mag = (d < 0) ? static_cast<uint64_t>(-d) : static_cast<uint64_t>(d);
        if (mag > PAE) PAE = mag;
        sum += static_cast<uint64_t>(d * d);
      }
    }
  }

  const uint64_t length = static_cast<uint64_t>(w) * h * nc;
  const double   mse    = static_cast<double>(sum) / static_cast<double>(length);
  const double   maxval = static_cast<double>((1u << bd) - 1u);
  double         psnr   = 10.0 * log10((maxval * maxval) / mse);
  if (mse < DBL_EPSILON) psnr = INFINITY;

  printf("%4llu, %12.6f, %12.6f\n",
         static_cast<unsigned long long>(PAE), mse, psnr);

  if (argc == 5) {
    const uint64_t thPAE = static_cast<uint64_t>(std::stoi(argv[3]));
    const double   thMSE = std::stof(argv[4]);
    if (PAE > thPAE || mse > thMSE) {
      printf("conformance test failure.\n");
      return EXIT_FAILURE;
    }
  }
  return EXIT_SUCCESS;
}
