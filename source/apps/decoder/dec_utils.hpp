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
#include <cstring>
#include <memory>

#define ceil_int(a, b) ((a) + ((b)-1)) / (b)

bool command_option_exists(int argc, char *argv[], const char *option) {
  bool result = false;
  for (int i = 0; i < argc; i++) {
    if (strcmp(argv[i], option) == 0) {
      result = true;
    }
  }
  return result;
}

char *get_command_option(int argc, char *argv[], const char *option) {
  for (int i = 0; i < argc; i++) {
    if (strcmp(argv[i], option) == 0) {
      return argv[i + 1];
    }
  }
  return nullptr;
}

inline uint16_t swap16(uint16_t value) {
  uint16_t ret;
  ret = value << 8;
  ret |= value >> 8;
  return ret;
}

inline uint32_t swap32(uint32_t value) {
  uint32_t ret;
  ret = value << 24;
  ret |= (value & 0x0000FF00) << 8;
  ret |= (value & 0x00FF0000) >> 8;
  ret |= value >> 24;
  return ret;
}

void write_ppm(char *outfile_name, char *outfile_ext_name, std::vector<int32_t *> buf,
               std::vector<uint32_t> &width, std::vector<uint32_t> &height, std::vector<uint8_t> &depth,
               std::vector<bool> &is_signed) {
  // ppm does not allow negative value
  int32_t PNM_OFFSET      = (is_signed[0]) ? 1 << (depth[0] - 1) : 0;
  int32_t MAXVAL          = 0;
  uint8_t bytes_per_pixel = ceil_int(depth[0], 8);
  MAXVAL                  = (1 << depth[0]) - 1;
  char fname[256], tmpname[256];
  memcpy(tmpname, outfile_name, outfile_ext_name - outfile_name);
  tmpname[outfile_ext_name - outfile_name] = '\0';
  sprintf(fname, "%s%s", tmpname, outfile_ext_name);
  FILE *ofp = fopen(fname, "wb");
  fprintf(ofp, "P6 %d %d %d\n", width[0], height[0], MAXVAL);
  const uint32_t num_pixels = width[0] * height[0];
#if ((defined(_MSVC_LANG) && _MSVC_LANG < 201402L) || __cplusplus < 201402L)
  std::unique_ptr<uint8_t[]> ppm_out(new uint8_t[num_pixels * bytes_per_pixel * 3]);
#else
  auto ppm_out = std::make_unique<uint8_t[]>(num_pixels * bytes_per_pixel * 3);
#endif
  setvbuf(ofp, (char *)ppm_out.get(), _IOFBF, num_pixels);
  int32_t val0, val1, val2;

  for (uint32_t i = 0; i < num_pixels; ++i) {
    val0 = (buf[0][i] + PNM_OFFSET);
    val1 = (buf[1][i] + PNM_OFFSET);
    val2 = (buf[2][i] + PNM_OFFSET);
    switch (bytes_per_pixel) {
      case 1:
        ppm_out[3 * i]     = (uint8_t)val0;
        ppm_out[3 * i + 1] = (uint8_t)val1;
        ppm_out[3 * i + 2] = (uint8_t)val2;
        break;
      case 2:
        ppm_out[6 * i]     = (uint8_t)(val0 >> 8);
        ppm_out[6 * i + 1] = (uint8_t)val0;
        ppm_out[6 * i + 2] = (uint8_t)(val1 >> 8);
        ppm_out[6 * i + 3] = (uint8_t)val1;
        ppm_out[6 * i + 4] = (uint8_t)(val2 >> 8);
        ppm_out[6 * i + 5] = (uint8_t)val2;
      default:
        break;
    }
  }
  fwrite(ppm_out.get(), sizeof(uint8_t), num_pixels * bytes_per_pixel * 3, ofp);
  fclose(ofp);
}

template <class T>
void convert_component_buffer_class(T *outbuf, uint16_t c, bool is_PGM, std::vector<int32_t *> &buf,
                                    std::vector<uint32_t> &width, std::vector<uint32_t> &height,
                                    std::vector<uint8_t> &depth, std::vector<bool> &is_signed) {
  const uint32_t num_pixels = width[c] * height[c];
  // pgm does not allow negative value
  int32_t PNM_OFFSET            = 0;
  const uint8_t bytes_per_pixel = ceil_int(depth[c], 8);

  if (is_PGM) {
    PNM_OFFSET = (is_signed[c]) ? 1 << (depth[c] - 1) : 0;
  }

  int32_t val;
  if (is_PGM) {
    uint32_t uval, ret;
    for (uint32_t i = 0; i < num_pixels; i++) {
      val = (buf[c][i] + PNM_OFFSET);
      // Little to Big endian
      uval = (uint32_t)val;
      ret  = uval << 24;
      ret |= (uval & 0x0000FF00) << 8;
      ret |= (uval & 0x00FF0000) >> 8;
      ret |= uval >> 24;
      outbuf[i] = (T)(ret >> ((4 - bytes_per_pixel) * 8));
    }
  } else {
    for (uint32_t i = 0; i < num_pixels; i++) {
      outbuf[i] = (T)buf[c][i];
    }
  }
}

void write_components(char *outfile_name, char *outfile_ext_name, std::vector<int32_t *> &buf,
                      std::vector<uint32_t> &width, std::vector<uint32_t> &height,
                      std::vector<uint8_t> &depth, std::vector<bool> &is_signed) {
  bool is_PGM = (strcmp(outfile_ext_name, ".pgm") == 0);
  bool is_PGX = (strcmp(outfile_ext_name, ".pgx") == 0);
  for (uint16_t c = 0; c < depth.size(); c++) {
    char fname[256], tmpname[256];
    memcpy(tmpname, outfile_name, outfile_ext_name - outfile_name);
    tmpname[outfile_ext_name - outfile_name] = '\0';
    sprintf(fname, "%s_%02d%s", tmpname, c, outfile_ext_name);

    FILE *ofp           = fopen(fname, "wb");
    uint32_t num_pixels = width[c] * height[c];
    if (is_PGM) {
      fprintf(ofp, "P5 %d %d %d\n", width[c], height[c], (1 << depth[c]) - 1);
    }
    if (is_PGX) {
      char sign = is_signed[c] ? '-' : '+';
      fprintf(ofp, "PG LM %c %d %d %d\n", sign, depth[c], width[c], height[c]);
    }
    if (ceil_int(depth[c], 8) == 1) {
#if ((defined(_MSVC_LANG) && _MSVC_LANG < 201402L) || __cplusplus < 201402L)
      std::unique_ptr<uint8_t[]> outbuf(new uint8_t[num_pixels]);
#else
      std::unique_ptr<uint8_t[]> outbuf  = std::make_unique<uint8_t[]>(num_pixels);
#endif
      convert_component_buffer_class<uint8_t>(outbuf.get(), c, is_PGM, buf, width, height, depth,
                                              is_signed);
      fwrite(outbuf.get(), sizeof(uint8_t), num_pixels, ofp);
    } else if (ceil_int(depth[c], 8) == 2) {
#if ((defined(_MSVC_LANG) && _MSVC_LANG < 201402L) || __cplusplus < 201402L)
      std::unique_ptr<uint16_t[]> outbuf(new uint16_t[num_pixels]);
#else
      std::unique_ptr<uint16_t[]> outbuf = std::make_unique<uint16_t[]>(num_pixels);
#endif
      convert_component_buffer_class<uint16_t>(outbuf.get(), c, is_PGM, buf, width, height, depth,
                                               is_signed);
      fwrite(outbuf.get(), sizeof(uint16_t), num_pixels, ofp);
    } else if (ceil_int(depth[c], 8) == 4) {
#if ((defined(_MSVC_LANG) && _MSVC_LANG < 201402L) || __cplusplus < 201402L)
      std::unique_ptr<uint32_t[]> outbuf(new uint32_t[num_pixels]);
#else
      std::unique_ptr<uint32_t[]> outbuf = std::make_unique<uint32_t[]>(num_pixels);
#endif
      convert_component_buffer_class<uint32_t>(outbuf.get(), c, is_PGM, buf, width, height, depth,
                                               is_signed);
      fwrite(outbuf.get(), sizeof(uint32_t), num_pixels, ofp);
    } else {
      // error
    }
    fclose(ofp);
  }
}
