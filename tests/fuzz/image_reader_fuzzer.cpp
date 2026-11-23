// Fuzz target: open_htj2k::image (encoder.hpp), the reader of the encoder's input images, on
// untrusted PGM, PPM and PGX files.
//
// The reader takes file names, so the files are written to a temporary directory first:
// $TMPDIR, or /tmp (%TEMP%, or the working directory, on Windows).
//
// Input = the file(s), then 1 option byte (fuzz_common.hpp says why it comes last):
//   o0 bits0-1  number of files: 1, 2, 3, 1
//   o0 bits2-3  their extension: .pgm, .ppm, .pgx, .pgm
// With one file the input is the file.  With more, every file but the last one comes with
// its length in front, as 2 bytes, big-endian; the last one takes what is left.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>
#if defined(_WIN32)
  #include <process.h>
#else
  #include <unistd.h>
#endif
#include "encoder.hpp"
#include "fuzz_common.hpp"

namespace {
std::string temp_base() {
  const char *dir = std::getenv("TMPDIR");
#if defined(_WIN32)
  if (dir == nullptr) dir = std::getenv("TEMP");
  if (dir == nullptr) dir = ".";
  const long pid = _getpid();
#else
  if (dir == nullptr) dir = "/tmp";
  const long pid = getpid();
#endif
  return std::string(dir) + "/openhtj2k_fuzz_" + std::to_string(pid) + "_";
}
}  // namespace

extern "C" int LLVMFuzzerInitialize(int *, char ***) {
  open_htj2k::set_message_handler(fuzz::quiet, nullptr);
  return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  if (size < 2) return 0;
  static const std::string base  = temp_base();
  static const char *const ext[] = {".pgm", ".ppm", ".pgx", ".pgm"};
  const size_t len               = size - 1;
  const uint8_t o0               = data[len];
  const int count                = (o0 & 3) % 3 + 1;

  std::vector<std::string> names;
  size_t at = 0;
  for (int k = 0; k < count; ++k) {
    size_t n = len - at;
    if (k != count - 1) {
      if (at + 2 > len) break;
      n = (static_cast<size_t>(data[at]) << 8) | data[at + 1];
      at += 2;
      n = std::min(n, len - at);
    }
    const std::string name = base + std::to_string(k) + ext[(o0 >> 2) & 3];
    FILE *f                = std::fopen(name.c_str(), "wb");
    if (f == nullptr) break;
    const bool written = std::fwrite(data + at, 1, n, f) == n;
    std::fclose(f);
    names.push_back(name);
    if (!written) break;
    at += n;
  }
  if (static_cast<int>(names.size()) == count) {
    try {
      open_htj2k::image img(names);
      uint64_t acc      = 0;
      const uint16_t nc = img.get_num_components();
      for (uint16_t c = 0; c < nc; ++c) {
        const uint64_t n = static_cast<uint64_t>(img.get_component_width(c)) * img.get_component_height(c);
        const int32_t *p = img.get_buf(c);
        if (p == nullptr || n == 0) continue;
        acc += static_cast<uint32_t>(p[0]) + static_cast<uint32_t>(p[n - 1]) + img.get_Ssiz_value(c);
      }
      fuzz::use(acc);
    } catch (std::exception &) {
    }
  }
  for (const std::string &name : names) std::remove(name.c_str());
  return 0;
}
