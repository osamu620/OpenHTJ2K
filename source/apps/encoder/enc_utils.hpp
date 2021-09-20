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
#include <algorithm>
#include <iterator>
#include <string>

#define NO_QFACTOR 0xFF

void print_help(char *cmd) {
  printf("JPEG 2000 Part 15 encoder\n");
  printf("USAGE: %s -i inputimage(PNM format) -o output-codestream [options...]\n\n", cmd);
  printf("-i: Input file\n  PGM and PPM are supported.\n");
  printf("-o: Output codestream\n  `.jhc` or `.j2c` are recommended as the extension.\n\n");
  printf("OPTIONS:\n");
  printf(
      "Stiles=Size:\n  Size of tile. `Size` should be in the format "
      "{height, width}"
      ". Default is equal to the image size.\n");
  printf("Sorigin=Size:\n  Offset from the origin of the reference grid to the image area. Default is \n");
  printf("Stile_origin=Size\n  Offset from the origin of the reference grid to the first tile.\n");
  printf(
      "Clevels=Int:\n  Number of DWT decomposition.\n  Valid range for number of DWT levels is from 0 to "
      "32 (Default is 5.)\n");
  printf("Creversible=Bool:\n  yes for lossless mode, no for lossy mode.\n");
  printf("Cblk=Size:\n  Code-block size.\n");
  printf("Cprecincts=Size:\n  Precinct size. Shall be power of two.\n");
  printf("Cycc=Bool:\n  yes to use RGB->YCbCr color space conversion.\n");
  printf("Corder:\n  Progression order. Valid entry is one of LRCP, RLCP, RPCL, PCRL, CPRL.\n");
  printf("Cuse_sop=Bool:\n  yes to use SOP (Start Of Packet) marker segment.\n");
  printf("Cuse_eph=Bool:\n  yes to use EPH (End of Packet Header) marker.\n");
  printf("Qstep=Float:\n  Base step size for quantization.\n  0.0 < base step size <= 2.0.\n");
  printf("Qguard=Int:\n  Number of guard bits. Valid range is from 0 to 8 (Default is 1.)\n");
  printf("Qfactor=Int:\n  Quality factor. Valid range is from 0 to 100 (100 is for the best quality)\n");
  printf("  Note: If this option is present, Qstep is ignored and Cycc is set to `yes`.\n");
}

class element_siz_local {
 public:
  uint32_t x;
  uint32_t y;
  element_siz_local() : x(0), y(0) {}
  element_siz_local(uint32_t x0, uint32_t y0) {
    x = x0;
    y = y0;
  }
};

size_t popcount_local(uintmax_t num) {
  size_t precision = 0;
  while (num != 0) {
    if (1 == (num & 1)) {
      precision++;
    }
    num >>= 1;
  }
  return precision;
}

int32_t log2i32(int32_t x) {
  if (x <= 0) {
    printf("ERROR: cannot compute log2 of negative value.\n");
    exit(EXIT_FAILURE);
  }
  int32_t y = 0;
  while (x > 1) {
    y++;
    x >>= 1;
  }
  return y;
}

class j2k_argset {
 private:
  std::vector<std::string> args;
  element_siz_local origin;
  element_siz_local tile_origin;
  uint8_t transformation;
  uint8_t use_ycc;
  uint8_t dwt_levels;
  element_siz_local cblksize;
  bool max_precincts;
  std::vector<element_siz_local> prctsize;
  element_siz_local tilesize;
  uint8_t Porder;
  bool use_sop;
  bool use_eph;
  double base_step_size;
  uint8_t num_guard;
  bool qderived;
  uint8_t qfactor;

 public:
  j2k_argset(int argc, char *argv[])
      : origin(0, 0),
        tile_origin(0, 0),
        transformation(0),
        use_ycc(1),
        dwt_levels(5),
        cblksize(4, 4),
        max_precincts(true),
        tilesize(0, 0),
        Porder(0),
        use_sop(false),
        use_eph(false),
        base_step_size(0.0),
        num_guard(1),
        qderived(false),
        qfactor(NO_QFACTOR) {
    args.reserve(argc);
    // skip command itself
    for (int i = 1; i < argc; ++i) {
      args.emplace_back(argv[i]);
    }
    get_help(argc, argv);

    for (auto &arg : args) {
      char &c = arg.front();
      int pos0, pos1;
      std::string param, val;
      std::string subparam;
      element_siz_local tmpsiz;
      switch (c) {
        case 'S':
          pos0  = arg.find_first_of('=');
          param = arg.substr(1, pos0 - 1);
          if (param == "tiles") {
            pos0 = arg.find_first_of('=');
            if (args[pos0] != "{") {
            }
            pos0++;
            pos1       = arg.find_first_of('}');
            subparam   = arg.substr(pos0 + 1, pos1 - pos0 - 1);
            pos0       = subparam.find_first_of(',');
            tilesize.y = std::stoi(subparam.substr(0, pos0));
            tilesize.x = std::stoi(subparam.substr(pos0 + 1, 4));
            break;
          }
          if (param == "origin") {
            pos0 = arg.find_first_of('=');
            if (pos0 == std::string::npos) {
              printf("ERROR: Sorigin needs a coordinate for the origin {y,x}\n");
              exit(EXIT_FAILURE);
            }
            pos0 = arg.find_first_of('{');
            if (pos0 == std::string::npos) {
              printf("ERROR: Sorigin needs a coordinate for the origin {y,x}\n");
              exit(EXIT_FAILURE);
            }
            pos1 = arg.find_first_of('}');
            if (pos1 == std::string::npos) {
              printf("ERROR: Sorigin needs a coordinate for the origin {y,x}\n");
              exit(EXIT_FAILURE);
            }
            subparam = arg.substr(pos0 + 1, pos1 - pos0 - 1);
            pos0     = subparam.find_first_of(',');
            origin.y = std::stoi(subparam.substr(0, pos0));
            origin.x = std::stoi(subparam.substr(pos0 + 1, subparam.length()));
            break;
          }
          if (param == "tile_origin") {
            pos0 = arg.find_first_of('=');
            if (pos0 == std::string::npos) {
              printf("ERROR: Stile_origin needs a coordinate for the origin {y,x}\n");
              exit(EXIT_FAILURE);
            }
            pos0 = arg.find_first_of('{');
            if (pos0 == std::string::npos) {
              printf("ERROR: Stile_origin needs a coordinate for the origin {y,x}\n");
              exit(EXIT_FAILURE);
            }
            pos1 = arg.find_first_of('}');
            if (pos1 == std::string::npos) {
              printf("ERROR: Stile_origin needs a coordinate for the origin {y,x}\n");
              exit(EXIT_FAILURE);
            }
            subparam      = arg.substr(pos0 + 1, pos1 - pos0 - 1);
            pos0          = subparam.find_first_of(',');
            tile_origin.y = std::stoi(subparam.substr(0, pos0));
            tile_origin.x = std::stoi(subparam.substr(pos0 + 1, subparam.length()));
            break;
          }
          printf("ERROR: unknown parameter S%s\n", param.c_str());
          exit(EXIT_FAILURE);
          break;
        case 'C':
          pos0  = arg.find_first_of('=');
          param = arg.substr(1, pos0 - 1);
          if (param == "reversible") {
            pos0 = arg.find_first_of('=');
            if (pos0 == std::string::npos) {
              printf("ERROR: Creversible needs =yes or =no\n");
              exit(EXIT_FAILURE);
            }
            val = arg.substr(pos0 + 1, 3);
            if (val == "yes") {
              transformation = 1;
            }
            break;
          }
          if (param == "ycc") {
            pos0 = arg.find_first_of('=');
            if (pos0 == std::string::npos) {
              printf("ERROR: Cycc needs =yes or =no\n");
              exit(EXIT_FAILURE);
            }
            val = arg.substr(pos0 + 1, 3);
            if (val == "yes") {
              use_ycc = 1;
            } else {
              use_ycc = 0;
            }
            break;
          }
          if (param == "levels") {
            pos0 = arg.find_first_of('=');
            if (pos0 == std::string::npos) {
              printf("ERROR: Clevels needs =dwt_levels (0 - 32)\n");
              exit(EXIT_FAILURE);
            }
            val     = arg.substr(pos0 + 1, 3);
            int tmp = std::stoi(val);
            if (tmp < 0 || tmp > 32) {
              printf("ERROR: number of DWT levels shall be in the range of [0, 32]\n");
              exit(EXIT_FAILURE);
            }
            dwt_levels = static_cast<uint8_t>(tmp);
            break;
          }
          if (param == "blk") {
            pos0 = arg.find_first_of('=');
            if (pos0 == std::string::npos) {
              printf("ERROR: Cblk needs a size of codeblock {height,width}\n");
              exit(EXIT_FAILURE);
            }
            pos0 = arg.find_first_of('{');
            if (pos0 == std::string::npos) {
              printf("ERROR: Cblk needs a size of codeblock {height,width}\n");
              exit(EXIT_FAILURE);
            }
            pos1 = arg.find_first_of('}');
            if (pos1 == std::string::npos) {
              printf("ERROR: Cblk needs a size of codeblock {height,width}\n");
              exit(EXIT_FAILURE);
            }
            subparam = arg.substr(pos0 + 1, pos1 - pos0 - 1);
            pos0     = subparam.find_first_of(',');
            tmpsiz.y = std::stoi(subparam.substr(0, pos0));
            tmpsiz.x = std::stoi(subparam.substr(pos0 + 1, 4));
            if ((popcount_local(tmpsiz.y) > 1) || (popcount_local(tmpsiz.x) > 1)) {
              printf("ERROR: code block size must be power of two.\n");
              exit(EXIT_FAILURE);
            }
            if (tmpsiz.x < 4 || tmpsiz.y < 4) {
              printf("ERROR: code block size must be greater than four\n");
              exit(EXIT_FAILURE);
            }
            if (tmpsiz.x * tmpsiz.y > 4096) {
              printf("ERROR: code block area must be less than or equal to 4096.\n");
              exit(EXIT_FAILURE);
            }
            cblksize.x = log2i32(tmpsiz.x) - 2;
            cblksize.y = log2i32(tmpsiz.y) - 2;
            break;
          }
          if (param == "precincts") {
            max_precincts = false;
            pos0          = arg.find_first_of('=');
            if (pos0 == std::string::npos) {
              printf("ERROR: Cprecincts needs at least one precinct size {height,width}\n");
              exit(EXIT_FAILURE);
            }
            pos0 = arg.find_first_of('{');
            if (pos0 == std::string::npos) {
              printf("ERROR: Cprecincts needs at least one precinct size {height,width}\n");
              exit(EXIT_FAILURE);
            }
            while (pos0 != std::string::npos) {
              pos1 = arg.find(std::string("}"), pos0);
              if (pos1 == std::string::npos) {
                printf("ERROR: Cprecincts needs at least one precinct size {height,width}\n");
                exit(EXIT_FAILURE);
              }
              subparam = arg.substr(pos0 + 1, pos1 - pos0 - 1);
              pos0     = subparam.find_first_of(',');
              tmpsiz.y = std::stoi(subparam.substr(0, pos0));
              tmpsiz.x = std::stoi(subparam.substr(pos0 + 1, 5));
              if ((popcount_local(tmpsiz.y) > 1) || (popcount_local(tmpsiz.x) > 1)) {
                printf("ERROR: precinct size must be power of two.\n");
                exit(EXIT_FAILURE);
              }
              prctsize.emplace_back(log2i32(tmpsiz.x), log2i32(tmpsiz.y));
              pos0 = arg.find(std::string("{"), pos1);
            }
            break;
          }
          if (param == "order") {
            pos0 = arg.find_first_of('=');
            if (pos0 == std::string::npos) {
              printf("ERROR: Corder needs progression order =(LRCP, RLCP, RPCL, PCRL, CPRL)\n");
              exit(EXIT_FAILURE);
            }
            val = arg.substr(pos0 + 1, 4);
            if (val == "LRCP") {
              Porder = 0;
            } else if (val == "RLCP") {
              Porder = 1;
            } else if (val == "RPCL") {
              Porder = 2;
            } else if (val == "PCRL") {
              Porder = 3;
            } else if (val == "CPRL") {
              Porder = 4;
            } else {
              printf("ERROR: unknown progression order %s\n", val.c_str());
              exit(EXIT_FAILURE);
            }
            break;
          }
          if (param == "use_sop") {
            pos0 = arg.find_first_of('=');
            if (pos0 == std::string::npos) {
              printf("ERROR: Cuse_sop needs =yes or =no\n");
              exit(EXIT_FAILURE);
            }
            val = arg.substr(pos0 + 1, 3);
            if (val == "yes") {
              use_sop = true;
            }
            break;
          }
          if (param == "use_eph") {
            pos0 = arg.find_first_of('=');
            if (pos0 == std::string::npos) {
              printf("ERROR: Cuse_eph needs =yes or =no\n");
              exit(EXIT_FAILURE);
            }
            val = arg.substr(pos0 + 1, 3);
            if (val == "yes") {
              use_eph = true;
            }
            break;
          }
          printf("ERROR: unknown parameter C%s\n", param.c_str());
          exit(EXIT_FAILURE);
          break;
        case 'Q':
          pos0  = arg.find_first_of('=');
          param = arg.substr(1, pos0 - 1);
          if (param == "step") {
            pos0 = arg.find_first_of('=');
            if (pos0 == std::string::npos) {
              printf("ERROR: Qstep needs base step size in float (0.0 < step <= 2.0)\n");
              exit(EXIT_FAILURE);
            }
            val            = arg.substr(pos0 + 1, 50);
            base_step_size = std::stod(val);
            if (base_step_size <= 0.0 || base_step_size > 2.0) {
              printf("ERROR: base step size shall be in the range of (0.0, 2.0]\n");
              exit(EXIT_FAILURE);
            }
            break;
          }
          if (param == "guard") {
            pos0 = arg.find_first_of('=');
            if (pos0 == std::string::npos) {
              printf("ERROR: Qguard needs number of guard bits (0-7)\n");
              exit(EXIT_FAILURE);
            }
            val     = arg.substr(pos0 + 1, 2);
            int tmp = std::stoi(val);
            if (tmp < 0 || tmp > 7) {
              printf("ERROR: number of guard bits shall be in the range of [0, 7]\n");
              exit(EXIT_FAILURE);
            }
            num_guard = static_cast<uint8_t>(tmp);
            break;
          }
          if (param == "derived") {
            pos0 = arg.find_first_of('=');
            if (pos0 == std::string::npos) {
              printf("ERROR: Qderived needs =yes or =no\n");
              exit(EXIT_FAILURE);
            }
            val = arg.substr(pos0 + 1, 3);
            if (val == "yes") {
              qderived = true;
            }
            break;
          }
          if (param == "factor") {
            pos0 = arg.find_first_of('=');
            if (pos0 == std::string::npos) {
              printf("ERROR: Qfactor needs value of quality (0-100)\n");
              exit(EXIT_FAILURE);
            }
            val     = arg.substr(pos0 + 1, 3);
            int tmp = std::stoi(val);
            if (tmp < 0 || tmp > 100) {
              printf("ERROR: value of Qfactor shall be in the range of [0, 100]\n");
              exit(EXIT_FAILURE);
            }
            qfactor = static_cast<uint8_t>(tmp);
            break;
          }
          printf("ERROR: unknown parameter Q%s\n", param.c_str());
          exit(EXIT_FAILURE);
          break;
        default:
          break;
      }
    }
  }

  std::vector<std::string> get_infile() {
    auto p = std::find(args.begin(), args.end(), "-i");
    if (p == args.end()) {
      printf("ERROR: input file (\"-i\") is missing!\n");
      exit(EXIT_FAILURE);
    }
    auto idx = std::distance(args.begin(), p);
    if (idx + 1 > args.size() - 1) {
      printf("ERROR: file name for input is missing!\n");
      exit(EXIT_FAILURE);
    }
    const std::string buf = args[idx + 1];
    const std::string comma(",");
    std::string::size_type pos = 0;
    std::string::size_type newpos;
    std::vector<std::string> fnames;
    std::string::size_type aa = buf.length();
    while (true) {
      newpos = buf.find(comma, pos + comma.length());
      fnames.push_back(buf.substr(pos, newpos - pos));
      pos = newpos;
      if (pos != std::string::npos) {
        pos += 1;
      } else {
        break;
      }
    }
    return fnames;
    // return args[idx + 1].c_str();
  }

  std::string get_outfile() {
    auto p = std::find(args.begin(), args.end(), "-o");
    if (p == args.end()) {
      printf("ERROR: output file (\"-o\") is missing!\n");
      exit(EXIT_FAILURE);
    }
    auto idx = std::distance(args.begin(), p);
    if (idx + 1 > args.size() - 1) {
      printf("ERROR: file name for output is missing!\n");
      exit(EXIT_FAILURE);
    }
    return args[idx + 1];
  }

  int32_t get_num_iteration() {
    int32_t num_iteration = 1;
    auto p                = std::find(args.begin(), args.end(), "-iter");
    if (p == args.end()) {
      return num_iteration;
    }
    auto idx = std::distance(args.begin(), p);
    if (idx + 1 > args.size() - 1) {
      printf("ERROR: -iter requires number of iteration\n");
      exit(EXIT_FAILURE);
    }
    return std::stoi(args[idx + 1]);
  }

  uint32_t get_num_threads() {
    // zero implies all threads
    uint32_t num_threads = 0;
    auto p              = std::find(args.begin(), args.end(), "-num_threads");
    if (p == args.end()) {
      return num_threads;
    }
    auto idx = std::distance(args.begin(), p);
    if (idx + 1 > args.size() - 1) {
      printf("ERROR: -iter requires number of iteration\n");
      exit(EXIT_FAILURE);
    }
    return (uint32_t)std::stoul(args[idx + 1]);
  }

  uint8_t get_jph_color_space() {
    uint8_t val = 0;
    auto p      = std::find(args.begin(), args.end(), "-jph_color_space");
    if (p == args.end()) {
      return val;
    }
    auto idx = std::distance(args.begin(), p);
    if (idx + 1 > args.size() - 1) {
      printf("ERROR: -jph_color_space requires name of color-space\n");
      exit(EXIT_FAILURE);
    }
    if (args[idx + 1].compare("YCC") && args[idx + 1].compare("RGB")) {
      printf("ERROR: invalid name for color-space\n");
      exit(EXIT_FAILURE);
    } else if (args[idx + 1].compare("YCC") == 0) {
      val = 1;
    }
    return val;
  }
  void get_help(int argc, char *argv[]) {
    auto p = std::find(args.begin(), args.end(), "-h");
    if (p == args.end() && argc > 1) {
      return;
    }
    print_help(argv[0]);
    exit(EXIT_SUCCESS);
  }

  element_siz_local get_origin() const { return origin; }
  element_siz_local get_tile_origin() const { return tile_origin; }
  uint8_t get_transformation() const { return transformation; }
  uint8_t get_ycc() const { return use_ycc; }
  uint8_t get_dwt_levels() const { return dwt_levels; }
  element_siz_local get_cblk_size() { return cblksize; }
  bool is_max_precincts() const { return max_precincts; }
  std::vector<element_siz_local> get_prct_size() { return prctsize; }
  element_siz_local get_tile_size() { return tilesize; }
  uint8_t get_progression() const { return Porder; }
  bool is_use_sop() const { return use_sop; }
  bool is_use_eph() const { return use_eph; }
  double get_basestep_size() const { return base_step_size; }
  uint8_t get_num_guard() const { return num_guard; }
  bool is_derived() const { return qderived; }
  uint8_t get_qfactor() const { return qfactor; }
};
