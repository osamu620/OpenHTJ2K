// borrow_check: the decoder only reads a borrowed codestream (init_borrow()).
//
// Decodes one borrowed buffer four times: twice one after the other, then twice at
// once from two threads, each time with a decoder of its own.  Every decode must give
// the samples of the first, and the buffer must hold the same bytes afterwards.  A
// decoder that wrote into the codestream (the HT decoder once replaced the last two
// bytes of every cleanup segment) changed the caller's data, failed on the second
// decode of the same buffer, and could not decode a buffer two threads share.
//
// Usage: borrow_check <input.j2k|.j2c> [-threads N]
// Exits 0 when every check holds.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iterator>
#include <thread>
#include <vector>
#include "decoder.hpp"

namespace {

// Decodes buf[0, len) borrowed, with init_borrow(); returns false on an error.
bool decode(uint8_t *buf, size_t len, uint32_t threads, std::vector<int32_t> &out) {
  out.clear();
  try {
    open_htj2k::openhtj2k_decoder dec;
    dec.init_borrow(buf, len, 0, threads);
    dec.parse();
    std::vector<uint32_t> w, h;
    std::vector<uint8_t> d;
    std::vector<bool> s;
    dec.invoke_line_based_stream(
        [&](uint32_t, int32_t *const *rows, uint16_t nc) {
          for (uint16_t c = 0; c < nc; ++c) out.insert(out.end(), rows[c], rows[c] + w[c]);
        },
        w, h, d, s);
  } catch (std::exception &) {
    return false;
  }
  return true;
}

}  // namespace

int main(int argc, char *argv[]) {
  const char *path = nullptr;
  uint32_t threads = 1;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "-threads") && i + 1 < argc) {
      threads = static_cast<uint32_t>(std::atoi(argv[++i]));
    } else {
      path = argv[i];
    }
  }
  if (path == nullptr) {
    std::printf("usage: borrow_check <input> [-threads N]\n");
    return 2;
  }
  std::ifstream f(path, std::ios::binary);
  std::vector<uint8_t> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  if (buf.empty()) {
    std::printf("cannot read %s\n", path);
    return 2;
  }
  const size_t len = buf.size();
  buf.resize(len + 16, 0);  // init_borrow() wants 16 readable bytes past the end
  const std::vector<uint8_t> original(buf);

  int failures = 0;
  auto check_buffer = [&](const char *when) {
    if (std::memcmp(buf.data(), original.data(), len) != 0) {
      size_t n = 0;
      for (size_t i = 0; i < len; ++i) n += (buf[i] != original[i]);
      std::printf("FAIL: %s, %zu bytes of the borrowed codestream changed\n", when, n);
      ++failures;
    }
  };

  std::vector<int32_t> first, again;
  if (!decode(buf.data(), len, threads, first)) {
    std::printf("FAIL: the first decode failed\n");
    return 1;
  }
  check_buffer("after the first decode");
  if (!decode(buf.data(), len, threads, again) || again != first) {
    std::printf("FAIL: a second decode of the same buffer failed or gave other samples\n");
    ++failures;
  }
  check_buffer("after the second decode");

  std::vector<int32_t> a, b;
  bool ok_a = false, ok_b = false;
  std::thread ta([&] { ok_a = decode(buf.data(), len, threads, a); });
  std::thread tb([&] { ok_b = decode(buf.data(), len, threads, b); });
  ta.join();
  tb.join();
  if (!ok_a || !ok_b || a != first || b != first) {
    std::printf("FAIL: two decoders sharing the buffer at once failed or gave other samples\n");
    ++failures;
  }
  check_buffer("after the concurrent decodes");

  if (failures == 0) std::printf("PASS: %s (%zu samples)\n", path, first.size());
  return failures == 0 ? 0 : 1;
}
