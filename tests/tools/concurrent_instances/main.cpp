// concurrent_instances: several application threads, each creating its own
// decoder / encoder instances, all running at the same time.
//
// This is how an image library embeds the codec (e.g. imread/imwrite called
// from many threads).  Every instance shares the one process-wide worker pool,
// so the test checks that (a) no instance tears the pool down while another is
// still using it and (b) concurrent encodes do not share per-thread scratch.
//
// Each thread repeats, for -iter rounds:
//   * decode the input codestream and compare a checksum of all samples
//     against a single-instance reference decode;
//   * encode a synthetic 3-component image through both encoder entry points
//     (buffered, multi-tile; streaming, single-tile) and compare the output
//     bytes against single-instance reference encodes.
//
// Usage: concurrent_instances <input.j2k> [-app T] [-iter K] [-threads N]
//                             [-size WxH] [-mixed]
//   -app      application threads running at once (default 8)
//   -iter     rounds per application thread (default 4)
//   -threads  num_threads passed to every instance (default 4)
//   -size     synthetic encode image size (default 1024x768)
//   -mixed    odd-numbered application threads pass num_threads = 1, so
//             single-threaded and pooled instances run side by side
// Exits 0 when every result matches its reference, non-zero otherwise.
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <thread>
#include <vector>
#include "decoder.hpp"
#include "encoder.hpp"

namespace {

constexpr uint8_t NO_QFACTOR = 0xFF;

std::vector<uint8_t> read_file(const char *path) {
  std::vector<uint8_t> data;
  FILE *f = fopen(path, "rb");
  if (!f) return data;
  fseek(f, 0, SEEK_END);
  const long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (n > 0) {
    data.resize(static_cast<size_t>(n));
    if (fread(data.data(), 1, data.size(), f) != data.size()) data.clear();
  }
  fclose(f);
  return data;
}

uint64_t fnv1a(uint64_t h, const void *p, size_t n) {
  const auto *b = static_cast<const uint8_t *>(p);
  for (size_t i = 0; i < n; ++i) {
    h ^= b[i];
    h *= 0x100000001b3ULL;
  }
  return h;
}

uint64_t decode_checksum(const std::vector<uint8_t> &cs, uint32_t num_threads) {
  open_htj2k::openhtj2k_decoder dec(cs.data(), cs.size(), 0, num_threads);
  std::vector<int32_t *> buf;
  std::vector<uint32_t> w, h;
  std::vector<uint8_t> depth;
  std::vector<bool> is_signed;
  dec.parse();
  dec.invoke_line_based(buf, w, h, depth, is_signed);
  uint64_t sum = 0xcbf29ce484222325ULL;
  for (size_t c = 0; c < buf.size(); ++c) {
    sum = fnv1a(sum, &w[c], sizeof(w[c]));
    sum = fnv1a(sum, &h[c], sizeof(h[c]));
    sum = fnv1a(sum, buf[c], sizeof(int32_t) * w[c] * h[c]);
    delete[] buf[c];
  }
  return sum;
}

// Deterministic noisy gradient: incompressible enough to give every
// codeblock real work.
struct SyntheticImage {
  uint32_t width, height;
  std::vector<std::vector<int32_t>> planes;
  SyntheticImage(uint32_t w, uint32_t h) : width(w), height(h), planes(3) {
    uint32_t s = 12345u;
    for (uint16_t c = 0; c < 3; ++c) {
      planes[c].resize(static_cast<size_t>(w) * h);
      for (uint32_t y = 0; y < h; ++y) {
        for (uint32_t x = 0; x < w; ++x) {
          s                                         = s * 1664525u + 1013904223u;
          const int32_t grad                        = static_cast<int32_t>((x + y + 40 * c) & 0xFF);
          planes[c][static_cast<size_t>(y) * w + x] = (grad + static_cast<int32_t>(s >> 27)) & 0xFF;
        }
      }
    }
  }
};

std::vector<uint8_t> encode(const SyntheticImage &img, bool streaming, uint32_t num_threads) {
  open_htj2k::siz_params siz;
  siz.Rsiz   = 0;
  siz.Xsiz   = img.width;
  siz.Ysiz   = img.height;
  siz.XOsiz  = 0;
  siz.YOsiz  = 0;
  siz.XTsiz  = streaming ? img.width : (img.width + 1) / 2;  // buffered: 2x2 tiles
  siz.YTsiz  = streaming ? img.height : (img.height + 1) / 2;
  siz.XTOsiz = 0;
  siz.YTOsiz = 0;
  siz.Csiz   = 3;
  for (uint16_t c = 0; c < 3; ++c) {
    siz.Ssiz.push_back(7);  // 8-bit unsigned
    siz.XRsiz.push_back(1);
    siz.YRsiz.push_back(1);
  }
  open_htj2k::cod_params cod;
  cod.blkwidth          = 4;  // log2(64) - 2
  cod.blkheight         = 4;
  cod.is_max_precincts  = true;
  cod.use_SOP           = false;
  cod.use_EPH           = false;
  cod.progression_order = 0;
  cod.number_of_layers  = 1;
  cod.use_color_trafo   = 1;
  cod.dwt_levels        = 5;
  cod.codeblock_style   = 0x040;  // HT
  cod.transformation    = 1;      // reversible 5/3
  open_htj2k::qcd_params qcd{};
  qcd.is_derived          = false;
  qcd.number_of_guardbits = 1;
  qcd.base_step           = 1.0 / 256.0;

  // Private copy per encode, so the comparison does not depend on whether the
  // encoder leaves its input untouched.
  std::vector<std::vector<int32_t>> planes;
  std::vector<int32_t *> input;
  if (!streaming) {
    planes = img.planes;
    for (auto &p : planes) input.push_back(p.data());
  }
  std::vector<uint8_t> out;
  open_htj2k::openhtj2k_encoder enc("", input, siz, cod, qcd, NO_QFACTOR, false, 0, num_threads);
  enc.set_output_buffer(out);
  if (streaming) {
    enc.invoke_line_based_stream([&img](uint32_t y, int32_t **rows, uint16_t nc) {
      for (uint16_t c = 0; c < nc; ++c)
        memcpy(rows[c], img.planes[c].data() + static_cast<size_t>(y) * img.width,
               sizeof(int32_t) * img.width);
    });
  } else {
    enc.invoke_line_based();
  }
  return out;
}

}  // namespace

int main(int argc, char *argv[]) {
  const char *infile = nullptr;
  int app_threads = 8, iterations = 4;
  uint32_t num_threads = 4, enc_w = 1024, enc_h = 768;
  bool mixed = false;
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "-app") == 0 && i + 1 < argc) {
      app_threads = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-iter") == 0 && i + 1 < argc) {
      iterations = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-threads") == 0 && i + 1 < argc) {
      num_threads = static_cast<uint32_t>(atoi(argv[++i]));
    } else if (strcmp(argv[i], "-size") == 0 && i + 1 < argc) {
      if (sscanf(argv[++i], "%ux%u", &enc_w, &enc_h) != 2) enc_w = enc_h = 0;
    } else if (strcmp(argv[i], "-mixed") == 0) {
      mixed = true;
    } else {
      infile = argv[i];
    }
  }
  if (infile == nullptr || app_threads < 1 || iterations < 1 || enc_w == 0 || enc_h == 0) {
    printf("usage: %s <input.j2k> [-app T] [-iter K] [-threads N] [-size WxH] [-mixed]\n", argv[0]);
    return EXIT_FAILURE;
  }
  const std::vector<uint8_t> cs = read_file(infile);
  if (cs.empty()) {
    printf("ERROR: cannot read %s\n", infile);
    return EXIT_FAILURE;
  }

  // References from one instance at a time.
  const SyntheticImage img(enc_w, enc_h);
  uint64_t ref_dec = 0;
  std::vector<uint8_t> ref_buffered, ref_stream;
  try {
    ref_dec      = decode_checksum(cs, num_threads);
    ref_buffered = encode(img, false, num_threads);
    ref_stream   = encode(img, true, num_threads);
  } catch (std::exception &e) {
    printf("ERROR: reference run failed: %s\n", e.what());
    return EXIT_FAILURE;
  }

  std::atomic<int> ok{0}, errors{0};
  std::atomic<int> bad_dec{0}, bad_buffered{0}, bad_stream{0};
  std::vector<std::thread> workers;
  for (int t = 0; t < app_threads; ++t) {
    const uint32_t nt = (mixed && (t & 1)) ? 1 : num_threads;
    workers.emplace_back([&, nt] {
      for (int i = 0; i < iterations; ++i) {
        try {
          (decode_checksum(cs, nt) == ref_dec ? ok : bad_dec)++;
          (encode(img, false, nt) == ref_buffered ? ok : bad_buffered)++;
          (encode(img, true, nt) == ref_stream ? ok : bad_stream)++;
        } catch (std::exception &e) {
          printf("ERROR: %s\n", e.what());
          errors++;
        }
      }
    });
  }
  for (auto &w : workers) w.join();

  const int bad = bad_dec.load() + bad_buffered.load() + bad_stream.load();
  printf(
      "concurrent_instances: %d app threads x %d rounds, num_threads=%u%s: %d ok, %d mismatch "
      "(decode %d, buffered encode %d, streaming encode %d), %d error\n",
      app_threads, iterations, num_threads, mixed ? " (mixed with 1)" : "", ok.load(), bad, bad_dec.load(),
      bad_buffered.load(), bad_stream.load(), errors.load());
  return (bad == 0 && errors.load() == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
