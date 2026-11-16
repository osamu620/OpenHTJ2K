[![CMake](https://github.com/osamu620/OpenHTJ2K/actions/workflows/cmake.yml/badge.svg?branch=main)](https://github.com/osamu620/OpenHTJ2K/actions/workflows/cmake.yml)
[![CodeQL](https://github.com/osamu620/OpenHTJ2K/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/osamu620/OpenHTJ2K/actions/workflows/codeql-analysis.yml)
[![Packaging status](https://repology.org/badge/tiny-repos/openhtj2k.svg)](https://repology.org/project/openhtj2k/versions)

# OpenHTJ2K

OpenHTJ2K is a C++ library with command-line tools for encoding and decoding
**High-Throughput JPEG 2000** (HTJ2K: Part 15, ITU-T Rec. T.814 | ISO/IEC
15444-15) and **JPEG 2000 Part 1** (ITU-T Rec. T.800 | ISO/IEC 15444-1) images.
It is fast on x86-64, Arm and WebAssembly, and free under the BSD 3-Clause
license.

**Links:** [🌐 Website](https://osamu620.github.io/OpenHTJ2K/) · [▶ Live demo](https://htj2k-demo.pages.dev/) · [📖 Docs](docs/README.md)

## Features

- **HTJ2K and Part 1**, both encode and decode. A Part 1 codestream from the
  encoder opens in any JPEG 2000 decoder. Part 2 is partly supported (DFS,
  ATK). Tested against the conformance codestreams of ISO/IEC 15444-4.
- **Fast:** SIMD code for AVX2, AVX-512, SSE2, NEON and WebAssembly, and a
  built-in thread pool.
- **Quality:** `Qfactor=0..100` sets the quality, much as in JPEG
  ([Qfactor](docs/qfactor.md)).
- **Size, for HTJ2K (experimental):** `-rate` makes an HTJ2K codestream of a
  given size, usually within a few bytes of it.
  - Post-compression rate-distortion optimisation (PCRD-opt) cuts each HT
    code-block after its cleanup, SigProp or MagRef pass.
  - It also works with the Part 1 block coder, which can add quality layers.
  - See [rate control](docs/cli_encoder.md#rate-control).
- **File formats:** codestreams (`.j2c`, `.j2k`, `.jhc`), JPH files (`.jph`,
  HTJ2K) and JP2 files (`.jp2`, Part 1). The tools read PGM, PPM, PGX and TIFF,
  and write PGM, PPM, PGX and RAW.
- **In the browser:** a WebAssembly build decodes inside the web page or
  under Node.js ([WebAssembly](docs/wasm.md)). Try the
  [live demo](https://htj2k-demo.pages.dev/).
- **Streaming (experimental):** a live RTP receiver (RFC 9828) that keeps up
  with 4K at 60 fps on a recent x86-64 CPU, and a JPIP server with foveated
  and gigapixel viewers.

Lossless coding of a 4096×2160 8-bit RGB frame on one thread
([how this was measured](docs/benchmarks.md)):

| Lossless, one thread | OpenJPEG 2.5.4 | OpenHTJ2K Part 1 | OpenHTJ2K HT |
|---|---:|---:|---:|
| Encode | 800 ms | 555 ms (1.4×) | 68 ms (11.8×) |
| Decode | 586 ms | 550 ms (1.1×) | 64 ms (9.2×) |

OpenHTJ2K's decode time includes writing the output file and OpenJPEG's does
not, so the comparison is conservative for OpenHTJ2K.

## Build and install

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
cmake --install build      # optional; --prefix <dir> to choose where
```

You need CMake 3.13 or later (3.15 for `cmake --install`; before that, use
`cmake --build build --target install`) and a C++11 compiler. TIFF input also needs
libtiff, which CMake finds by itself. The tools are in `build/bin/`. For other
platforms and build options, see [docs/building.md](docs/building.md); for
the WebAssembly build, [docs/wasm.md](docs/wasm.md).

## Quick start

```bash
open_htj2k_enc -i input.ppm -o out.jph Qfactor=90            # lossy HTJ2K, quality 90
open_htj2k_enc -i input.ppm -o out.j2c Creversible=yes       # lossless
open_htj2k_enc -i input.ppm -o out.jp2 Cmodes=0 Qfactor=90   # Part 1, for any JPEG 2000 decoder
open_htj2k_enc -i input.ppm -o out.j2c -rate 1               # at most 1 bit per pixel

open_htj2k_dec -i out.jph -o decoded.ppm
open_htj2k_dec -i out.jph -o half.ppm -reduce 1              # half the width and height
```

Each tool lists its options with `-h`. To run them from the build tree, use
`./build/bin/open_htj2k_enc`. Full references: [encoder](docs/cli_encoder.md),
[decoder](docs/cli_decoder.md).

## Using the library

```cmake
find_package(openhtj2k CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE openhtj2k::openhtj2k)
```

[docs/library.md](docs/library.md) gives short decode and encode examples and
covers threads, errors and warnings.

## Documentation

| Topic | Document |
|---|---|
| Building and installing | [docs/building.md](docs/building.md) |
| Encoder and decoder options | [docs/cli_encoder.md](docs/cli_encoder.md), [docs/cli_decoder.md](docs/cli_decoder.md) |
| What `Qfactor` does | [docs/qfactor.md](docs/qfactor.md) |
| The C++ API | [docs/library.md](docs/library.md) |
| WebAssembly: decoding in the browser and Node.js | [docs/wasm.md](docs/wasm.md) |
| Speed against OpenJPEG | [docs/benchmarks.md](docs/benchmarks.md) |
| RTP receiver, browser viewer, JPIP (experimental) | [docs/cli_rtp_recv.md](docs/cli_rtp_recv.md), [docs/wt_viewer.md](docs/wt_viewer.md), [docs/jpip.md](docs/jpip.md) |
| Release history | [CHANGELOG](CHANGELOG) |

To work on OpenHTJ2K itself, start with [CONTRIBUTING.md](CONTRIBUTING.md).

## License

BSD 3-Clause. See [LICENSE](LICENSE).
