# Fuzzing

The fuzz targets live in [`tests/fuzz`](../../tests/fuzz). Each one is a
`<name>_fuzzer.cpp` with an `LLVMFuzzerTestOneInput()` that feeds one input to
a part of the library that reads untrusted bytes.

| Target | What it runs | Input |
|---|---|---|
| `decode_fuzzer` | The decoder API (`decoder.hpp`): the four `invoke_line_based*()` entry points, resolution reduction, 1 and 4 threads, a borrowed buffer, row and column ranges, the precinct filter, the packet observer, and a second and third frame on the same decoder. | A codestream or a JPH/JP2 file, then 8 option bytes. |
| `transcode_fuzzer` | The transcoder (`transcoder.hpp`) in both directions with every option, then a decode of the output and a transcode back. | A codestream, then 4 option bytes. |
| `jpip_client_fuzzer` | What a JPIP client reads from a server: the JPP-stream parser (whole, and in pieces), HTTP framing, the reassembly of a codestream from data-bins, and its decode. | The bytes from the server, then 1 option byte. |
| `jpip_request_fuzzer` | What a JPIP server reads from a client: the request line, the query fields, the cache model, the channel requests. | The text of a request. |
| `jpip_server_fuzzer` | A JPIP server on a codestream to serve: index, packet locator, a view-window, the data-bins sent for it, and the client's side of that response. | A codestream, then 8 option bytes (the view-window). |
| `image_reader_fuzzer` | `open_htj2k::image`, the reader of PGM, PPM and PGX input images. | One to three files, then 1 option byte. |

The head of each source file gives the meaning of its option bytes. They come
last so that the file under test starts at byte 0: a codestream followed by
zero bytes is a valid seed, run with the default options.

The targets decode images of up to 2^22 samples, 1024 tiles and 64
components. Larger ones are well formed and only slow under the sanitizers,
so the targets return early for them.

## The corpus in the test suite

`tests/fuzz/corpus/<name>_fuzzer/` holds a few inputs per target: small
codestreams run through each option, and every input that once made a target
report something. The test suite builds the targets without a fuzzing engine
and runs each over its corpus:

```bash
ctest --test-dir build -R fuzz_replay
```

These six tests run with every compiler and on every platform the suite runs
on, so a finding made on x86-64 is checked against the NEON and MSVC builds
too. Without a sanitizer they catch crashes and hangs; in a sanitizer build
they catch what the sanitizer does.

To keep a new finding, copy the reported file to the corpus directory of its
target under a name that says what it is.

## Fuzzing on your own machine

You need clang with libFuzzer. `tests/fuzz/build.sh` builds the static library
and the targets, their dictionaries and their seed corpora, and
`tests/fuzz/run.sh` runs them all at once:

```bash
export CXX=clang++
export CXXFLAGS="-fsanitize=fuzzer-no-link,address,undefined -fno-sanitize-recover=undefined -fno-omit-frame-pointer -g"
export LIB_FUZZING_ENGINE=-fsanitize=fuzzer
export WORK=/tmp/openhtj2k-fuzz OUT=/tmp/openhtj2k-fuzz/out
tests/fuzz/build.sh
tests/fuzz/run.sh "$OUT" 600        # every target for 10 minutes
```

`run.sh` prints one line per target and exits with 1 if any target reported
something. The reported inputs are in `$OUT/run/<target>/artifacts`, and the
target's output in `$OUT/run/<target>/log.txt`. It uses the limits of
OSS-Fuzz: 25 s for one input and 2560 MB of memory, with leak detection on.
Six sanitizer processes can take several gigabytes together; on Linux,
`systemd-run --user --scope -p MemoryMax=16G tests/fuzz/run.sh ...` bounds
them.

To run one target for longer, or on several cores:

```bash
mkdir -p corpus && unzip -q "$OUT/decode_fuzzer_seed_corpus.zip" -d seeds
"$OUT/decode_fuzzer" -dict="$OUT/decode_fuzzer.dict" -timeout=25 -rss_limit_mb=2560 \
  -fork=8 -ignore_crashes=1 -ignore_timeouts=1 -ignore_ooms=1 corpus seeds
```

To look at a report, run the target on the file: `"$OUT/decode_fuzzer"
<file>`, or `build/bin/decode_fuzzer <file>` from an ordinary build (add
`-DCMAKE_CXX_FLAGS=-fsanitize=address,undefined` to see what the fuzzer saw).
Time a reported timeout or out-of-memory input with a release build before
calling it a finding: the sanitizers make a decode 10 to 30 times slower.

### Which code is fuzzed

SIMD code is chosen at compile time, so one build exercises one set of
kernels. `OPENHTJ2K_FUZZ_CPU` picks it for `build.sh` on x86-64:

- `generic` (the default): the SSE2 code. The targets run on any x86-64
  machine.
- `x86-64-v3`: the AVX2 code. The machines that run the targets need AVX2.

A complete run covers both. The AVX-512 and NEON kernels are covered by
building on such a machine with `-DOPENHTJ2K_TARGET_CPU=native` (x86-64) or
on AArch64.

### Without the script

`-DOPENHTJ2K_BUILD_FUZZERS=ON` links the targets with a fuzzing engine
instead of the replay `main()`. It needs the static library:

```bash
cmake -B build_fuzz -DCMAKE_CXX_COMPILER=clang++ -DBUILD_SHARED_LIBS=OFF \
  -DOPENHTJ2K_BUILD_FUZZERS=ON -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined"
cmake --build build_fuzz -j
```

When the environment sets `LIB_FUZZING_ENGINE` (OSS-Fuzz and ClusterFuzzLite
do), the targets link that and the caller's `CXXFLAGS` carry the
instrumentation. Otherwise the build adds `-fsanitize=fuzzer-no-link` to the
library and links libFuzzer.

## In CI

`.github/workflows/fuzz.yml` builds the targets with AddressSanitizer and
UndefinedBehaviorSanitizer and runs each for five minutes, with `build.sh`
and `run.sh` as above, on every pull request that changes `source/`,
`tests/fuzz/`, a `CMakeLists.txt` or the workflow itself. Inputs it reports
are uploaded as an artifact of the run. A few minutes of fuzzing can also
meet a fault that the pull request did not bring, so read a failure before
blaming the change; the job is not meant to be a required check. It can be
started by hand as well (`workflow_dispatch`, with the seconds per target).

## OSS-Fuzz and ClusterFuzzLite

`build.sh` leaves in `$OUT` what both expect: the targets,
`<target>.dict` and `<target>_seed_corpus.zip`. A project build script for
either one is these lines:

```bash
cd $SRC/OpenHTJ2K
tests/fuzz/build.sh
```

The targets are checked under AddressSanitizer and
UndefinedBehaviorSanitizer. MemorySanitizer has not been run on them yet.
