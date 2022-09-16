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
#if defined(OPENHTJ2K_ENABLE_ARM_NEON)
  #include <arm_neon.h>
#endif
#define ceil_int(a, b) (((a) + ((b)-1)) / (b))

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

void write_ppm(char *outfile_name, char *outfile_ext_name, std::vector<int32_t *> buf,
               std::vector<uint32_t> &width, std::vector<uint32_t> &height, std::vector<uint8_t> &depth,
               std::vector<bool> &is_signed) {
  // ppm does not allow negative value
  int32_t PNM_OFFSET      = (is_signed[0]) ? 1 << (depth[0] - 1) : 0;
  uint8_t bytes_per_pixel = static_cast<uint8_t>(ceil_int(static_cast<int32_t>(depth[0]), 8));
  int32_t MAXVAL          = (1 << depth[0]) - 1;
  char fname[256], tmpname[256];
  memcpy(tmpname, outfile_name, static_cast<size_t>(outfile_ext_name - outfile_name));
  tmpname[outfile_ext_name - outfile_name] = '\0';
  sprintf(fname, "%s%s", tmpname, outfile_ext_name);
  FILE *ofp = fopen(fname, "wb");
  fprintf(ofp, "P6 %d %d %d\n", width[0], height[0], MAXVAL);
  const uint32_t num_pixels = width[0] * height[0];

  // ppm_out buffer is allocated by malloc because it does not need value-initialization
  uint8_t *ppm_out = static_cast<uint8_t *>(malloc(num_pixels * bytes_per_pixel * 3));
  setvbuf(ofp, (char *)ppm_out, _IOFBF, num_pixels);
  int32_t val0, val1, val2;

#if defined(OPENHTJ2K_ENABLE_ARM_NEON)
  uint32_t len = num_pixels;
  int32_t *R, *G, *B;
  uint8_t *out = ppm_out;
  R            = buf[0];
  G            = buf[1];
  B            = buf[2];
  int32x4_t R0, R1, G0, G1, B0, B1;
  if (bytes_per_pixel == 1) {
    uint8x8_t voffset = vdup_n_u8(static_cast<uint8_t>(PNM_OFFSET));
    for (; len >= 8; len -= 8) {
      uint8x8x3_t vout;
      R0          = vld1q_s32(R);
      R1          = vld1q_s32(R + 4);
      G0          = vld1q_s32(G);
      G1          = vld1q_s32(G + 4);
      B0          = vld1q_s32(B);
      B1          = vld1q_s32(B + 4);
      vout.val[0] = vadd_u8(vmovn_u16(vcombine_u16(vmovn_s32(R0), vmovn_s32(R1))), voffset);
      vout.val[1] = vadd_u8(vmovn_u16(vcombine_u16(vmovn_s32(G0), vmovn_s32(G1))), voffset);
      vout.val[2] = vadd_u8(vmovn_u16(vcombine_u16(vmovn_s32(B0), vmovn_s32(B1))), voffset);

      vst3_u8(out, vout);
      R += 8;
      G += 8;
      B += 8;
      out += 24;
    }
    for (; len > 0; --len) {
      *out++ = static_cast<uint8_t>(*R++ + PNM_OFFSET);
      *out++ = static_cast<uint8_t>(*G++ + PNM_OFFSET);
      *out++ = static_cast<uint8_t>(*B++ + PNM_OFFSET);
    }
  } else {
    if (bytes_per_pixel != 2) {
      printf("ERROR: write PPM with Over 16bpp is not supported\n");
      throw std::exception();
    }
    uint16x8_t voffset = vdupq_n_u16(static_cast<uint16_t>(PNM_OFFSET));
    for (; len >= 8; len -= 8) {
      uint16x8x3_t vout;
      R0          = vld1q_s32(R);
      R1          = vld1q_s32(R + 4);
      G0          = vld1q_s32(G);
      G1          = vld1q_s32(G + 4);
      B0          = vld1q_s32(B);
      B1          = vld1q_s32(B + 4);
      vout.val[0] = vcombine_u16(vmovn_s32(R0), vmovn_s32(R1));
      vout.val[1] = vcombine_u16(vmovn_s32(G0), vmovn_s32(G1));
      vout.val[2] = vcombine_u16(vmovn_s32(B0), vmovn_s32(B1));

      vout.val[0] = vrev16q_u8(vaddq_u16(vout.val[0], voffset));
      vout.val[1] = vrev16q_u8(vaddq_u16(vout.val[1], voffset));
      vout.val[2] = vrev16q_u8(vaddq_u16(vout.val[2], voffset));
      vst3q_u16((uint16_t *)out, vout);
      R += 8;
      G += 8;
      B += 8;
      out += 48;
    }
    for (; len > 0; --len) {
      val0   = *R + PNM_OFFSET;
      val1   = *G + PNM_OFFSET;
      val2   = *B + PNM_OFFSET;
      *out++ = static_cast<uint8_t>(val0 >> 8);
      *out++ = static_cast<uint8_t>(val0);
      *out++ = static_cast<uint8_t>(val1 >> 8);
      *out++ = static_cast<uint8_t>(val1);
      *out++ = static_cast<uint8_t>(val2 >> 8);
      *out++ = static_cast<uint8_t>(val2);
      R++;
      G++;
      B++;
    }
  }
#else
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
#endif
  fwrite(ppm_out, sizeof(uint8_t), num_pixels * bytes_per_pixel * 3, ofp);
  fclose(ofp);
  free(ppm_out);
}

template <class T>
void convert_component_buffer_class(T *outbuf, uint16_t c, bool is_PGM, std::vector<int32_t *> &buf,
                                    std::vector<uint32_t> &width, std::vector<uint32_t> &height,
                                    std::vector<uint8_t> &depth, std::vector<bool> &is_signed) {
  const uint32_t num_pixels = width[c] * height[c];
  // pgm does not allow negative value
  int32_t PNM_OFFSET            = 0;
  const uint8_t bytes_per_pixel = static_cast<uint8_t>(ceil_int(static_cast<int32_t>(depth[c]), 8));

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
  for (uint16_t c = 0; c < static_cast<uint16_t>(depth.size()); c++) {
    char fname[256], tmpname[256];
    memcpy(tmpname, outfile_name, static_cast<size_t>(outfile_ext_name - outfile_name));
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
