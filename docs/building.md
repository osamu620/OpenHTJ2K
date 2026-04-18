# Building OpenHTJ2K

This document covers the full build matrix: the native C++ library and
CLI tools, the WebAssembly variant, and the experimental RFC 9828 RTP
receiver.

## Requirements

CMake 3.13 or later and a compiler supporting **C++11 or later**.

CMake automatically selects the highest standard supported by the
compiler (C++17 → C++14 → C++11). All three modes have been verified to
produce a correct build and pass the full conformance test suite.

| Standard | Behaviour |
|---|---|
| C++17 (recommended) | `[[nodiscard]]` and `[[maybe_unused]]` attributes are active; `std::filesystem` used for path handling |
| C++14 | Attributes expand to nothing (no diagnostics lost at runtime); `stat()` fallback for path handling |
| C++11 | Same as C++14; additionally uses a built-in `make_unique` shim and `std::result_of` instead of `std::invoke_result_t` |

## Native build

`./` is the root of the cloned repository and `${BUILD_DIR}` is a build
directory (for example `./build` or `../build`).

```bash
cmake -G "Unix Makefiles" -B ${BUILD_DIR} -DCMAKE_BUILD_TYPE=Release -DOPENHTJ2K_THREAD=ON
cmake --build ${BUILD_DIR} --config Release -j
```

Executables are placed in `${BUILD_DIR}/bin`. The shared library is
`libopen_htj2k.so` / `libopen_htj2k.dylib` / `open_htj2k.dll` depending
on platform.

### Common CMake flags

| Flag | Default | Meaning |
|---|---|---|
| `-DCMAKE_BUILD_TYPE=<Release\|Debug\|RelWithDebInfo>` | (none) | Optimization and debug info level. `RelWithDebInfo` is the recommended mode for profiling. |
| `-DOPENHTJ2K_THREAD=ON` | `OFF` | Enable the built-in thread pool for multi-threaded encode and decode. Strongly recommended. |
| `-DOPENHTJ2K_RTP=ON` | `OFF` | Build the experimental RFC 9828 RTP receiver (see below). Adds a GLFW + OpenGL dependency. Also builds the JPIP foveation demo. |
| `-DOPENHTJ2K_QUIC=ON` | `OFF` | Enable HTTP/3 over QUIC for the JPIP server and demo. Requires MsQuic + nghttp3 (`brew install libmsquic libnghttp3` on macOS). |
| `-DENABLE_AVX2=OFF` | auto | Force-disable AVX2 dispatch. Auto-detected via `-march=native` on x86-64. |
| `-DENABLE_ARM_NEON=OFF` | auto | Force-disable NEON dispatch on AArch64. |
| `-DBUILD_SHARED_LIBS=OFF` | `ON` | Build a static library instead of a shared one. |

### Generator notes

- `-G "Xcode"` generates an Xcode project on macOS.
- `-G "Visual Studio 17 2022"` generates a Visual Studio 2022 solution.
  See the [CMake Visual Studio generators docs](https://cmake.org/cmake/help/latest/manual/cmake-generators.7.html#visual-studio-generators)
  for other versions.
- `-G "Ninja"` is the fastest generator on Linux/macOS if Ninja is
  installed.

## Building for WebAssembly (WASM)

Requires [Emscripten](https://emscripten.org/) (tested with 3.x / 5.x).

Four variants are produced under `subprojects/build/html/`:

- `libopen_htj2k.js`           — scalar, single-threaded
- `libopen_htj2k_simd.js`      — WASM-SIMD 128-bit, single-threaded
  (recommended for most browsers)
- `libopen_htj2k_mt.js`        — scalar + pthreads (multi-threaded)
- `libopen_htj2k_mt_simd.js`   — WASM-SIMD + pthreads (fastest where
  available)

```bash
cd subprojects
mkdir -p build && cd build
emcmake cmake ..
cmake --build . -j
```

The multi-threaded variants use Emscripten's pthreads, which require
the page to be **cross-origin isolated** (responses carry both
`Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: require-corp` — or `credentialless`).
When served from a static host that can set headers (e.g. Cloudflare
Pages via a `_headers` file), this works on first load.  For local
development or `file://` access, the bundled `coi-serviceworker.js`
installs the headers via a service worker.

A live demo is available at **https://htj2k-demo.pages.dev/**.  It
hosts two pages:

- `index.html` — still-image decoder (preset dropdown + file upload,
  progressive-resolution grid).
- `rtp_demo.html` — RFC 9828 `.rtp` file replay (WebGL2 GPU renderer,
  YCbCr→RGB in fragment shader, planar Y/Cb/Cr textures, Display-P3
  on supported hardware).  URL parameters `?variant=mt_simd|simd|mt|
  scalar` and `?renderer=auto|webgl|2d` force a specific build /
  renderer for A/B testing.  `?verbose=1` enables per-second console
  diagnostics.

A GitHub Actions workflow (`.github/workflows/deploy-wasm-demo.yml`)
rebuilds all four variants and publishes the demo site on every
push to `main` or `feat/wasm-rtp-demo`.

### Node.js CLI decoder (`open_htj2k_dec.mjs`)

`open_htj2k_dec.mjs` is a Node.js ES module that wraps the WASM build
so you can decode J2C / J2K / JPH files from the terminal. It requires
the WASM build above but not a platform-native C++ toolchain on the
target machine.

**Requirements:** Node.js ≥ 18 and the WASM build.

**Usage:**
```bash
cd subprojects
node open_htj2k_dec.mjs -i <input.j2c|.j2k|.jph> -o <output.ppm|.pgm> \
     [-r <reduce_NL>] [-num_threads <N>] [-ycbcr bt601|bt709]
```

| Option | Description |
|--------|-------------|
| `-i`, `--input`     | Input codestream (`.j2c`, `.j2k`, `.jph`) |
| `-o`, `--output`    | Output image (`.ppm` for RGB, `.pgm` for grayscale, `.pgx` for per-component raw) |
| `-r`, `--reduce`    | Resolution reduction: skip `n` DWT levels (`0` = full resolution) |
| `-num_threads`, `-t` | Number of decode threads (`0` = auto-detect, `1` = single-threaded).  Selects the multi-threaded WASM build when `> 1` or `= 0`. |
| `-ycbcr`            | YCbCr→RGB conversion for PPM output: `bt601` or `bt709` (auto-detected from JPH `EnumCS` otherwise). |

**Examples:**
```bash
node open_htj2k_dec.mjs -i image.j2c -o image.ppm
node open_htj2k_dec.mjs -i image.j2c -o image_half.ppm -r 1   # half resolution
```

The script auto-selects the SIMD build (`libopen_htj2k_simd.js`) when
available, falling back to the scalar build. Decoding uses the
streaming `invoke_decoder_stream` path, keeping peak WASM heap well
below the full-image `int32` buffer approach (~52 MB peak for a 4K RGB
image versus ~486 MB with the batch path).

## Building the experimental RFC 9828 RTP receiver

Adds `open_htj2k_rtp_recv`, a live HTJ2K RTP receiver per RFC 9828 that
decodes incoming frames and displays them via GLFW/OpenGL. Off by
default so the rest of the project builds without a window system.

**Prerequisites:**

- GLFW 3.x development headers:
  - Debian/Ubuntu: `libglfw3-dev`
  - Fedora: `glfw-devel`
  - macOS (Homebrew): `brew install glfw`
- OpenGL 3.3 core profile at runtime

```bash
cmake -G "Unix Makefiles" -B ${BUILD_DIR} -DCMAKE_BUILD_TYPE=Release \
      -DOPENHTJ2K_THREAD=ON -DOPENHTJ2K_RTP=ON
cmake --build ${BUILD_DIR} --config Release -j
```

Produces `${BUILD_DIR}/bin/open_htj2k_rtp_recv` and, when
`-DOPENHTJ2K_RTP=ON`, the offline decode profiler
`${BUILD_DIR}/bin/open_htj2k_rtp_decode_profile` for reproducing the
performance measurements documented in
[cli_rtp_recv.md](cli_rtp_recv.md).

## Running the test suite

```bash
ctest --test-dir ${BUILD_DIR}
```

Runs the full conformance suite (~582 tests) including HTJ2K and Part 1
decode conformance, encoder round-trip checks, and line-based
streaming validation. `-j` is supported.
