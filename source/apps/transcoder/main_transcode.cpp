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

// open_htj2k_transcode (EXPERIMENTAL): Part 1 (EBCOT/MQ) codestream -> HTJ2K codestream.

#include <chrono>
#include <cstdio>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>
#include "../print_library_messages.hpp"
#include "transcoder.hpp"

static void usage(const char *prog) {
  std::printf(
      "Usage: %s -i <input.j2c> -o <output.j2c>\n"
      "  Transcodes a JPEG 2000 Part 1 (EBCOT/MQ) codestream into an HTJ2K (Part 15)\n"
      "  codestream with the same quantisation indices (EXPERIMENTAL).\n"
      "  -i  input: raw codestream (.j2c / .j2k) or JP2 file (.jp2), found by its signature\n"
      "  -o  output: raw codestream (.j2c / .jhc) or JPH file (.jph). For a JP2 input the .jph file\n"
      "      keeps the boxes of the input (colour, resolution, XML, ...); for a raw codestream they\n"
      "      are made as open_htj2k_enc makes them (sRGB for 3 or more components, else grayscale)\n"
      "  -jph_color_space RGB|YCC  the colour space a .jph file declares for a raw codestream with\n"
      "      three or four components: RGB = sRGB (default), YCC = sYCC, for components that are Y,\n"
      "      Cb, Cr (as in open_htj2k_enc). Not for a JP2 input, whose own colour box is kept.\n"
      "  -copy_input  the library copies the input before it reads it (memory the size of the input\n"
      "      more; the output is the same). By default it reads the input where it is.\n"
      "  -partial refine|drop|keep  a code-block that the Part 1 encoder cut part-way through a\n"
      "      bit-plane: refine = code the block at its last complete cleanup plane plus HT\n"
      "      SigProp/MagRef passes for the partial one (default; exact), drop = code it at that\n"
      "      plane and drop the partial plane (smallest, lossy), keep = code every decoded bit\n"
      "      (reconstruction off by half a step, and a much larger output). Blocks cut after a\n"
      "      cleanup pass are exact in all modes.\n"
      "  -num_threads N  worker threads for the code-block loop (0 = all hardware threads, default;\n"
      "      1 = single-threaded). The output is identical for every N.\n"
      "  -layers merge|keep  merge = one quality layer (default): an HT code-block is a single HT\n"
      "      set, which other decoders (OpenJPEG, for one) can read, and it cannot be refined layer\n"
      "      by layer as a Part 1 block can. keep = the layer count of the input, with a block's HT\n"
      "      set in the first layer the input had passes of it in: a decode of fewer layers shows\n"
      "      early blocks at full precision and later blocks missing (EXPERIMENTAL; a larger output\n"
      "      for the extra packet headers).\n"
      "  Memory: the input and the output are held whole, and the tiles are transcoded one at a\n"
      "  time, so a tiled image needs about the input plus the output plus one tile; an image of\n"
      "  one tile needs three to four times the input. A codestream of 4 GB or more is refused.\n"
      "  PLM marker segments are not written; PLT ones are.\n"
      "  -h  this help\n",
      prog);
}

int main(int argc, char *argv[]) {
  open_htj2k_apps::install_library_message_printer();
  std::string in_name, out_name;
  open_htj2k::transcode_options opt;
  open_htj2k::transcode_stats stats;
  opt.stats       = &stats;
  bool copy_input = false;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
      usage(argv[0]);
      return EXIT_SUCCESS;
    } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
      in_name = argv[++i];
    } else if (std::strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
      out_name = argv[++i];
    } else if (std::strcmp(argv[i], "-jph_color_space") == 0 && i + 1 < argc) {
      const char *v = argv[++i];
      if (std::strcmp(v, "RGB") == 0) {
        opt.jph_colour_space = open_htj2k::transcode_colour_space::rgb;
      } else if (std::strcmp(v, "YCC") == 0) {
        opt.jph_colour_space = open_htj2k::transcode_colour_space::ycc;
      } else {
        std::printf("ERROR: -jph_color_space takes RGB or YCC, not %s\n", v);
        return EXIT_FAILURE;
      }
    } else if (std::strcmp(argv[i], "-copy_input") == 0) {
      copy_input = true;
    } else if (std::strcmp(argv[i], "-layers") == 0 && i + 1 < argc) {
      const char *v = argv[++i];
      if (std::strcmp(v, "keep") == 0) {
        opt.layers = open_htj2k::transcode_layers::keep;
      } else if (std::strcmp(v, "merge") == 0) {
        opt.layers = open_htj2k::transcode_layers::merge;
      } else {
        std::printf("ERROR: -layers takes merge or keep, not %s\n", v);
        return EXIT_FAILURE;
      }
    } else if (std::strcmp(argv[i], "-partial") == 0 && i + 1 < argc) {
      const char *v = argv[++i];
      if (std::strcmp(v, "keep") == 0) {
        opt.partial_plane = open_htj2k::transcode_partial_plane::keep;
      } else if (std::strcmp(v, "drop") == 0) {
        opt.partial_plane = open_htj2k::transcode_partial_plane::drop;
      } else if (std::strcmp(v, "refine") == 0) {
        opt.partial_plane = open_htj2k::transcode_partial_plane::refine;
      } else {
        std::printf("ERROR: -partial takes keep, drop or refine, not %s\n", v);
        return EXIT_FAILURE;
      }
    } else if (std::strcmp(argv[i], "-num_threads") == 0 && i + 1 < argc) {
      opt.num_threads = static_cast<unsigned>(std::strtoul(argv[++i], nullptr, 10));
    } else {
      std::printf("ERROR: unknown or incomplete option %s\n", argv[i]);
      usage(argv[0]);
      return EXIT_FAILURE;
    }
  }
  if (in_name.empty() || out_name.empty()) {
    usage(argv[0]);
    return EXIT_FAILURE;
  }

  {
    const size_t dot = out_name.find_last_of('.');
    std::string ext  = dot == std::string::npos ? "" : out_name.substr(dot);
    for (char &ch : ext) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    if (ext == ".jph") {
      opt.jph_output = true;
    } else if (ext == ".jp2") {
      std::printf("ERROR: an HTJ2K codestream cannot be in a JP2 file; write .jph (or .j2c / .jhc)\n");
      return EXIT_FAILURE;
    }
  }

  std::ifstream ifs(in_name, std::ios::in | std::ios::binary);
  if (!ifs) {
    std::printf("ERROR: cannot open %s\n", in_name.c_str());
    return EXIT_FAILURE;
  }
  // The file is read in one piece, into a buffer of its size (reading it through iterators would grow
  // the buffer by doubling, and hold twice its size meanwhile).
  std::vector<uint8_t> in;
  try {
    ifs.seekg(0, std::ios::end);
    const std::streamoff file_size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    if (file_size < 0) throw std::runtime_error("cannot get the size");
    in.resize(static_cast<size_t>(file_size) + 16);  // 16 readable bytes after the file, for the transcoder
    ifs.read(reinterpret_cast<char *>(in.data()), file_size);
    if (!ifs) throw std::runtime_error("cannot read the whole file");
  } catch (std::exception &exc) {
    std::printf("ERROR: cannot read %s: %s\n", in_name.c_str(), exc.what());
    return EXIT_FAILURE;
  }

  std::vector<uint8_t> out;
  const auto t0 = std::chrono::high_resolution_clock::now();
  try {
    opt.input_padded = !copy_input;
    open_htj2k::transcode_part1_to_htj2k(in.data(), in.size() - 16, out, opt);
  } catch (std::exception &exc) {
    std::printf("ERROR: transcoding failed: %s\n", exc.what());
    return EXIT_FAILURE;
  }
  const auto t1 = std::chrono::high_resolution_clock::now();

  std::ofstream ofs(out_name, std::ios::out | std::ios::binary);
  if (!ofs) {
    std::printf("ERROR: cannot create %s\n", out_name.c_str());
    return EXIT_FAILURE;
  }
  ofs.write(reinterpret_cast<const char *>(out.data()), static_cast<std::streamsize>(out.size()));
  ofs.close();

  const size_t in_size = in.size() - 16;
  std::printf("Input (Part 1)   = %zu bytes\nOutput (HTJ2K)   = %zu bytes (%.1f%% of input)\n", in_size,
              out.size(), 100.0 * static_cast<double>(out.size()) / static_cast<double>(in_size));
  std::printf(
      "Code-blocks with passes = %llu: %llu full, %llu cut after a cleanup pass above the LSB, %llu cut "
      "part-way through a bit-plane (%llu coded with HT refinement passes)\n",
      static_cast<unsigned long long>(stats.blocks), static_cast<unsigned long long>(stats.full),
      static_cast<unsigned long long>(stats.coarser), static_cast<unsigned long long>(stats.partial),
      static_cast<unsigned long long>(stats.refined));
  std::printf("elapsed time %.3f [ms]\n", std::chrono::duration<double, std::milli>(t1 - t0).count());
  return EXIT_SUCCESS;
}
