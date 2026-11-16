# Contributing to OpenHTJ2K

This page is for people who change OpenHTJ2K itself. To use it, see the
[README](README.md).

## Build and test

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build -j               # the whole suite
ctest --test-dir build -R enc_lossless  # only the tests whose names match
```

- A Debug build (`-DCMAKE_BUILD_TYPE=Debug`) names the tools with a `_dbg`
  suffix, for example `open_htj2k_enc_dbg`.
- The tests are defined in `tests/*.cmake`. The conformance codestreams and
  reference images are in `conformance_data/`. The suite covers:
  - Part 1, Part 15 and Part 2 (DFS, ATK) decoding conformance;
  - encoder round trips;
  - line-based decoding;
  - JPIP protocol checks;
  - security regressions.
- Test helpers are in `tests/tools/`: `imgcmp` (pixel comparison),
  `row_range_compare`, the `jpip_*_check` programs, and others.
- [docs/building.md](docs/building.md) covers the build for users: options,
  portable x86-64 builds, installing, the WebAssembly build and the optional
  RTP and JPIP tools.

## Build options for development

- **C++ standard:** CMake uses the newest of C++17, C++14 and C++11 that the
  compiler supports. Pass `-DCMAKE_CXX_STANDARD=11` (or `14`, `17`) to choose
  one. All three build correctly and pass the test suite, and CI builds and
  tests C++11 on Linux (x86-64) and macOS (ARM64) to keep it that way.

  | Standard | Behaviour |
  |---|---|
  | C++17 | `[[nodiscard]]` and `[[maybe_unused]]` attributes are active; `std::filesystem` used for path handling |
  | C++14 | Attributes expand to nothing (no diagnostics lost at runtime); `stat()` fallback for path handling; `if constexpr` on template parameters becomes a plain `if` (the optimizer removes the dead branch) |
  | C++11 | Same as C++14; additionally uses a built-in `make_unique` shim and `std::result_of` instead of `std::invoke_result_t` |

- **SIMD code paths:**

  | Flag | Default | Effect |
  |---|---|---|
  | `-DENABLE_AVX2=OFF` | `ON` | Build x86-64 code without any AVX2 (or wider) instructions; same as `-DOPENHTJ2K_TARGET_CPU=generic`. |
  | `-DENABLE_SSE2=OFF` | `ON` | In x86-64 builds without AVX2, use the scalar code paths instead of the SSE2 ones (as other architectures do). |
  | `-DENABLE_ARM_NEON=OFF` | auto | Force-disable NEON dispatch on AArch64. |

- **Profiling:**
  - Native: build with `-DCMAKE_BUILD_TYPE=RelWithDebInfo`.
  - WebAssembly: `-DOPENHTJ2K_WASM_PROFILE=ON` keeps the function names. See
    [docs/wasm_bench.md](docs/wasm_bench.md).
- **Public headers:** the installed package passes on only what the public
  headers need. The macros that select the library's own code paths are
  private to its build. `ctest -R install_consumer` installs the build into a
  scratch prefix and checks three things, with warnings as errors under
  `-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion
  -Wold-style-cast` and similar flags:
  - it builds and runs a small program against it through CMake;
  - it does the same through pkg-config, when pkg-config is available;
  - it compiles every installed header on its own.
- **WebAssembly demo site:** `.github/workflows/deploy-wasm-demo.yml` rebuilds
  every variant and publishes https://htj2k-demo.pages.dev/ when a push to
  `main` changes its inputs. `rtp_demo.html` takes URL parameters for A/B
  tests:
  - `?variant=mt_simd|simd|mt|scalar` and `?renderer=auto|webgl|2d` force a
    build or a renderer;
  - `?verbose=1` prints diagnostics to the console every second.

## Source layout

```
source/core/
  interface/     public API: openhtj2k_encoder, openhtj2k_decoder, messages
  codestream/    J2K/JPH marker reading and writing
  coding/        HT and MQ block coding, coding units
  transform/     DWT and colour transforms
  common/        shared types, thread pool, utilities
  jph/           JPH/JP2 box parsing
  jpip/          Part 9 server and client
source/apps/     the command-line tools (encoder, decoder, estimate_qfactor,
                 rtp_recv, jpip_*) and imgio, the shared image readers
tests/           test definitions; tests/tools/ holds the test programs
web/             browser demos and the WebAssembly decoder worker
tools/           benchmark scripts and the wt_bridge relay
```

- **Encoding:** read the image → colour transform → forward DWT → block
  coding → codestream.
- **Decoding:** parse the markers → block decoding → inverse DWT → colour
  transform → output.

## Code style

- Format with `clang-format -i` on the files you change. The style in
  `.clang-format` is Google style with a 2-space indent and 108 columns, and
  it leaves the include order alone.
- Functions are `snake_case`, classes `CamelCase`.
- The library targets C++11. CMake picks C++17, C++14 or C++11, whichever
  the compiler supports, and the test suite passes in all three. Do not use
  C++14 or later features in `source/core` unless they are guarded.

## SIMD code

- Each core algorithm has one file per instruction set: `foo.cpp`
  (scalar), `foo_avx2.cpp`, `foo_avx512.cpp`, `foo_neon.cpp` and
  `foo_wasm.cpp`. The HT block encoder and the forward DWT also have
  `foo_sse2.cpp`, which must give the same output as the scalar file bit for
  bit.
- When you change an algorithm, check every variant. A fix made only in the
  scalar file is a common source of divergence.
- The instruction set is chosen at compile time (`OPENHTJ2K_ENABLE_AVX2`,
  `OPENHTJ2K_ENABLE_ARM_NEON`, `OPENHTJ2K_ENABLE_WASM_SIMD`,
  `OPENHTJ2K_ENABLE_SSE2`); there is no runtime detection.
- `OPENHTJ2K_ENABLE_ARM_NEON` is for AArch64 only, but code reached under
  `__ARM_NEON` also compiles for 32-bit ARMv7, for example in vcpkg's
  Android ARM build. There, use intrinsics that ARMv7 has, or split on
  `__aarch64__ || _M_ARM64`.
- Row buffers are not 32-byte aligned, so use unaligned loads and stores.

## Things that break easily

- **Sample type:** `sprec_t` is `float` everywhere (DWT, colour, sample
  buffers). The integer lifting paths use `FRACBITS = 13`.
- **Forward DWT order:** the forward DWT is vertical, then horizontal.
  Horizontal first breaks lossless 5/3 because of integer rounding.
- **Ring buffers:** index with `r % RING_DEPTH`, not
  `(r - origin) % DEPTH`.
- **Band order:** `BAND_LL=0`, `BAND_HL=1`, `BAND_LH=2`, `BAND_HH=3`.
- **Performance changes:** they need measurements before and after. Compare
  against `main` with two separate build trees, because each tool loads the
  shared library of its own tree.

## Pull requests

- Make one branch per change and open the pull request against `main`. The
  history is kept linear (no merge commits), so rebase your branch on `main`
  rather than merging `main` into it.
- Title commits and pull requests as `type(scope): summary`, for example
  `fix(decoder): …`, `feat(encoder): …`, `perf(decoder): …` or `docs: …`.
- Say what changed, why, and how you tested it. The description should make
  sense to someone who has not read the discussion that led to it.
- Say whether the change can alter encoded or decoded output, and how you
  checked, for example by comparing codestreams with those of `main`.
- Note user-visible changes in [CHANGELOG](CHANGELOG), under a
  `# [Unreleased]` heading at the top, using the section headings already
  there (`## Added`, `## Changed`, `## Fixed`, …).

## Background documents

[docs/README.md](docs/README.md#working-on-openhtj2k) lists the documents for
contributors, among them the benchmark method, the WebAssembly benchmark
harness, the SIMD IDWT kernels and the JPIP internals.
