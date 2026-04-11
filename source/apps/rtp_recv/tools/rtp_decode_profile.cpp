// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.
//
// Licensed under the same BSD 3-Clause terms as the rest of OpenHTJ2K.
//
// Offline decode profiler for the rtp_recv hot path.
//
// Loads a set of .j2c codestreams and drives the decoder exactly the way
// rtp_recv::decode_thread_main does: one long-lived openhtj2k_decoder,
// per-frame init() + parse() + invoke_line_based_stream() with a callback
// that writes 8-bit planar Y/Cb/Cr.  Per-stage timing is printed so the
// per-frame budget can be decomposed end-to-end, decoupled from the UDP
// socket, frame handler, and renderer.
//
// Built alongside open_htj2k_rtp_recv when -DOPENHTJ2K_RTP=ON.  The binary
// is named open_htj2k_rtp_decode_profile and lands in build/bin/.
//
// Typical flow — dump 250 codestreams from a .rtp fixture, then run the
// profiler on the dumped directory.  See source/apps/rtp_recv/tools/README.md
// for the exact commands (two-step capture with open_htj2k_rtp_recv in
// --no-decode mode plus rtp_file_replay.py, then the profiler).
//
// CLI: <codestream_dir> [max_frames=200] [loops=3] [threads=2]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "decoder.hpp"
#include "../planar_shift.hpp"

namespace fs = std::filesystem;
using Clock  = std::chrono::steady_clock;
using ns     = std::chrono::nanoseconds;

struct Frame {
  std::vector<uint8_t> bytes;
  std::string          name;
};

static double ms(ns d) {
  return std::chrono::duration<double, std::milli>(d).count();
}

static std::vector<Frame> load_dir(const std::string& dir, size_t max_frames) {
  std::vector<fs::path> paths;
  for (auto& e : fs::directory_iterator(dir)) {
    if (!e.is_regular_file()) continue;
    auto p = e.path();
    if (p.extension() != ".j2c") continue;
    paths.push_back(p);
  }
  std::sort(paths.begin(), paths.end());
  if (max_frames && paths.size() > max_frames) paths.resize(max_frames);

  std::vector<Frame> out;
  out.reserve(paths.size());
  for (auto& p : paths) {
    Frame f;
    f.name = p.filename().string();
    std::ifstream ifs(p, std::ios::binary | std::ios::ate);
    auto sz = ifs.tellg();
    ifs.seekg(0);
    f.bytes.resize(static_cast<size_t>(sz));
    ifs.read(reinterpret_cast<char*>(f.bytes.data()), sz);
    out.push_back(std::move(f));
  }
  return out;
}

struct Stats {
  double sum    = 0;
  double sq_sum = 0;
  double min    = 1e30;
  double max    = 0;
  size_t n      = 0;

  void add(double x) {
    sum += x;
    sq_sum += x * x;
    if (x < min) min = x;
    if (x > max) max = x;
    ++n;
  }
  double mean() const { return n ? sum / static_cast<double>(n) : 0.0; }
  double stddev() const {
    if (n < 2) return 0.0;
    double m = mean();
    return std::sqrt(std::max(0.0, sq_sum / static_cast<double>(n) - m * m));
  }
};

static void print_row(const char* label, const Stats& s) {
  std::printf("%-18s n=%-4zu  min=%7.3f  avg=%7.3f  max=%7.3f  stddev=%6.3f  sum=%8.1f ms\n",
              label, s.n, s.min, s.mean(), s.max, s.stddev(), s.sum);
}

int main(int argc, char** argv) {
  std::string dir        = argc > 1 ? argv[1] : "/tmp/spark_cs";
  size_t      max_frames = argc > 2 ? std::stoul(argv[2]) : 200;
  size_t      loops      = argc > 3 ? std::stoul(argv[3]) : 3;
  uint32_t    nthreads   = argc > 4 ? static_cast<uint32_t>(std::stoul(argv[4])) : 2;
  // Optional 5th arg: pass "reuse" to exercise the single-tile cache path;
  // anything else (or no arg) uses the legacy invoke_line_based_stream so
  // we can A/B the two on the same codestream set.
  const bool  use_reuse = (argc > 5 && std::string(argv[5]) == "reuse");
  // Optional 6th arg: path prefix for dumping frame 0's decoded planes
  // (suffix _y.bin / _cb.bin / _cr.bin is appended).  Used to verify
  // byte-equality between the legacy and reuse paths: run twice with
  // different prefixes and `cmp` the outputs.  Empty string = no dump.
  std::string dump_prefix = argc > 6 ? argv[6] : "";

  std::printf("loading up to %zu codestreams from %s ...\n", max_frames, dir.c_str());
  auto frames = load_dir(dir, max_frames);
  if (frames.empty()) {
    std::fprintf(stderr, "no .j2c files found in %s\n", dir.c_str());
    return 1;
  }
  size_t total_bytes = 0;
  for (auto& f : frames) total_bytes += f.bytes.size();
  std::printf("loaded %zu frames, %.1f MiB, avg %.1f KiB per frame\n", frames.size(),
              static_cast<double>(total_bytes) / (1024.0 * 1024.0),
              static_cast<double>(total_bytes) / 1024.0 / static_cast<double>(frames.size()));
  std::printf("threads=%u  loops=%zu  path=%s\n\n", nthreads, loops,
              use_reuse ? "invoke_line_based_stream_reuse" : "invoke_line_based_stream");

  open_htj2k::openhtj2k_decoder decoder;
  if (use_reuse) decoder.enable_single_tile_reuse(true);

  Stats s_init, s_parse, s_stream, s_cbshift, s_total;
  size_t iters                  = 0;

  // Persistent planar scratch buffers (reused across frames, like rtp_recv).
  std::vector<uint8_t> plane_y, plane_cb, plane_cr;

  // Warm-up: one full pass not counted, to prime caches and ThreadPool.
  // Also primes the single-tile reuse cache on the first frame so the
  // timed loop below sees the cached fast path from iteration 1.
  for (size_t i = 0; i < frames.size(); ++i) {
    try {
      decoder.init(frames[i].bytes.data(), frames[i].bytes.size(), 0, nthreads);
      decoder.parse();
      std::vector<uint32_t> widths, heights;
      std::vector<uint8_t>  depths;
      std::vector<bool>     signeds;
      if (use_reuse) {
        decoder.invoke_line_based_stream_reuse(
            [&](uint32_t, int32_t* const*, uint16_t) {},
            widths, heights, depths, signeds);
      } else {
        decoder.invoke_line_based_stream(
            [&](uint32_t, int32_t* const*, uint16_t) {},
            widths, heights, depths, signeds);
      }
    } catch (std::exception& e) {
      std::fprintf(stderr, "warm-up frame %zu failed: %s\n", i, e.what());
      return 2;
    }
  }
  std::printf("warm-up done.\n\n");

  for (size_t L = 0; L < loops; ++L) {
    for (size_t i = 0; i < frames.size(); ++i) {
      const auto& f = frames[i];

      const auto t0 = Clock::now();
      try {
        decoder.init(f.bytes.data(), f.bytes.size(), 0, nthreads);
      } catch (std::exception& e) {
        std::fprintf(stderr, "init[%zu] failed: %s\n", i, e.what());
        return 3;
      }
      const auto t1 = Clock::now();
      try {
        decoder.parse();
      } catch (std::exception& e) {
        std::fprintf(stderr, "parse[%zu] failed: %s\n", i, e.what());
        return 4;
      }
      const auto t2 = Clock::now();

      // Replicate the decode_to_planar_buffers callback shape: scalar
      // int32->u8 shift into Y/Cb/Cr planes.  Time spent inside the
      // callback is `cb_time_ns`; `stream_time_ns` is the full
      // invoke_line_based_stream call (HT decode + IDWT + callback).
      int64_t  cb_time_ns  = 0;
      uint32_t luma_w = 0, luma_h = 0;
      uint32_t chroma_w = 0, chroma_h = 0;
      uint8_t  depth_y = 0, depth_c = 0;
      bool     first_row = true;

      try {
        std::vector<uint32_t> widths, heights;
        std::vector<uint8_t>  depths;
        std::vector<bool>     signeds;
        auto stream_cb = [&](uint32_t y, int32_t* const* rows, uint16_t nc) {
              const auto cb0 = Clock::now();
              if (first_row) {
                first_row = false;
                if (nc < 1) return;
                luma_w  = widths[0];
                luma_h  = heights[0];
                depth_y = depths[0];
                if (nc >= 3) {
                  chroma_w = widths[1];
                  chroma_h = heights[1];
                  depth_c  = depths[1];
                }
                plane_y.assign(static_cast<size_t>(luma_w) * luma_h, 0);
                plane_cb.assign(static_cast<size_t>(chroma_w) * chroma_h, 128);
                plane_cr.assign(static_cast<size_t>(chroma_w) * chroma_h, 128);
              }

              const int32_t shift_y  = static_cast<int32_t>(depth_y) - 8;
              const int32_t maxval_y = (1 << depth_y) - 1;
              open_htj2k::rtp_recv::shift_i32_plane_to_u8(
                  rows[0], plane_y.data() + static_cast<size_t>(y) * luma_w, luma_w,
                  shift_y, maxval_y);
              if (nc >= 3 && chroma_h > 0) {
                const int32_t shift_c  = static_cast<int32_t>(depth_c) - 8;
                const int32_t maxval_c = (1 << depth_c) - 1;
                const uint32_t yc      = (luma_h > 0)
                                             ? static_cast<uint32_t>(
                                                   static_cast<uint64_t>(y) * chroma_h / luma_h)
                                             : 0;
                if (yc < chroma_h) {
                  if (rows[1]) {
                    open_htj2k::rtp_recv::shift_i32_plane_to_u8(
                        rows[1], plane_cb.data() + static_cast<size_t>(yc) * chroma_w,
                        chroma_w, shift_c, maxval_c);
                  }
                  if (rows[2]) {
                    open_htj2k::rtp_recv::shift_i32_plane_to_u8(
                        rows[2], plane_cr.data() + static_cast<size_t>(yc) * chroma_w,
                        chroma_w, shift_c, maxval_c);
                  }
                }
              }

              const auto cb1 = Clock::now();
              cb_time_ns += std::chrono::duration_cast<ns>(cb1 - cb0).count();
            };
        if (use_reuse) {
          decoder.invoke_line_based_stream_reuse(stream_cb, widths, heights, depths, signeds);
        } else {
          decoder.invoke_line_based_stream(stream_cb, widths, heights, depths, signeds);
        }
      } catch (std::exception& e) {
        std::fprintf(stderr, "stream[%zu] failed: %s\n", i, e.what());
        return 5;
      }
      const auto t3 = Clock::now();

      const double init_ms   = ms(t1 - t0);
      const double parse_ms  = ms(t2 - t1);
      const double stream_ms = ms(t3 - t2);
      const double cb_ms     = static_cast<double>(cb_time_ns) / 1e6;
      const double total_ms  = ms(t3 - t0);

      s_init.add(init_ms);
      s_parse.add(parse_ms);
      s_stream.add(stream_ms);
      s_cbshift.add(cb_ms);
      s_total.add(total_ms);
      ++iters;
    }
    std::printf("loop %zu/%zu done\n", L + 1, loops);
  }

  std::printf("\n---- per-frame timing across %zu iterations ----\n", iters);
  print_row("init()",                s_init);
  print_row("parse()",               s_parse);
  print_row("stream_total",          s_stream);
  print_row("  cb(shift_to_u8)",     s_cbshift);
  print_row("TOTAL",                 s_total);
  const double total_s = s_total.sum / 1000.0;
  std::printf("\nceiling fps (1 / avg total): %.2f\n", 1000.0 / std::max(s_total.mean(), 1e-9));
  std::printf("sustained fps (%zu frames / total wall): %.2f\n", iters,
              static_cast<double>(iters) / total_s);

  if (!dump_prefix.empty()) {
    // At this point plane_y / plane_cb / plane_cr hold the LAST-decoded
    // frame's planar bytes (loops × frames.size() iterations, last frame
    // of the last loop).  Write them out so an external `cmp` can check
    // byte-equality between paths.
    auto write_file = [](const std::string& path, const std::vector<uint8_t>& data) {
      std::ofstream ofs(path, std::ios::binary);
      ofs.write(reinterpret_cast<const char*>(data.data()),
                static_cast<std::streamsize>(data.size()));
    };
    write_file(dump_prefix + "_y.bin",  plane_y);
    write_file(dump_prefix + "_cb.bin", plane_cb);
    write_file(dump_prefix + "_cr.bin", plane_cr);
    std::printf("dumped last frame's planes to %s_{y,cb,cr}.bin\n", dump_prefix.c_str());
  }
  return 0;
}
