// main() for a fuzz target built without a fuzzing engine: runs the target once on every
// file named on the command line.  The test suite replays tests/fuzz/corpus with it on every
// platform and compiler (a target linked with libFuzzer takes the same command line).
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <vector>

extern "C" int LLVMFuzzerInitialize(int *argc, char ***argv);
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size);

int main(int argc, char **argv) {
  LLVMFuzzerInitialize(&argc, &argv);
  int inputs = 0;
  for (int i = 1; i < argc; ++i) {
    std::ifstream file(argv[i], std::ios::binary);
    if (!file) {
      std::fprintf(stderr, "cannot open %s\n", argv[i]);
      return 1;
    }
    const std::vector<char> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    // A copy of the exact size, as a fuzzing engine gives: a read past the input is then a
    // read past the allocation, which AddressSanitizer reports.
    std::unique_ptr<uint8_t[]> input(new uint8_t[bytes.size() + (bytes.empty() ? 1 : 0)]);
    if (!bytes.empty()) std::memcpy(input.get(), bytes.data(), bytes.size());
    LLVMFuzzerTestOneInput(input.get(), bytes.size());
    ++inputs;
  }
  std::printf("%d inputs\n", inputs);
  return 0;
}
