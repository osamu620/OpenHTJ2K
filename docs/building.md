# Building OpenHTJ2K

This document covers the full build matrix: the native C++ library and
CLI tools, the WebAssembly variant, and the experimental RFC 9828 RTP
receiver.

## Requirements

**Mandatory:**

- CMake 3.13 or later
- A compiler supporting **C++11 or later** (GCC 4.8+, Clang 3.3+, MSVC 2015+, Apple Clang 6+)

CMake automatically selects the highest standard supported by the
compiler (C++17 → C++14 → C++11). All three modes have been verified to
produce a correct build and pass the full conformance test suite.

| Standard | Behaviour |
|---|---|
| C++17 (recommended) | `[[nodiscard]]` and `[[maybe_unused]]` attributes are active; `std::filesystem` used for path handling |
| C++14 | Attributes expand to nothing (no diagnostics lost at runtime); `stat()` fallback for path handling |
| C++11 | Same as C++14; additionally uses a built-in `make_unique` shim and `std::result_of` instead of `std::invoke_result_t` |

**Optional (auto-detected):**

| Dependency | Enables | Install |
|---|---|---|
| **libtiff** | TIFF input in `open_htj2k_enc` (8/16-bit, RGB or grayscale, both planar configurations; tiled TIFFs require `-batch`) | Debian/Ubuntu: `libtiff-dev`; Fedora: `libtiff-devel`; macOS (Homebrew): `brew install libtiff`; vcpkg: `vcpkg install tiff` |

When libtiff is not detected at configure time the build proceeds without
TIFF support; the encoder will reject `.tif` / `.tiff` inputs at runtime.
CMake prints `libtiff found` (or omits the line) so you can verify which
mode you got.

**Opt-in (explicit CMake flags):** see the table under *Common CMake
flags* below for `-DOPENHTJ2K_RTP` and `-DOPENHTJ2K_QUIC`, and the
dedicated sections later in this document for each one's additional
native-dependency requirements. The built-in thread pool is enabled
automatically whenever `find_package(Threads)` succeeds; there is no
explicit threading flag.

## Native build

`./` is the root of the cloned repository and `${BUILD_DIR}` is a build
directory (for example `./build` or `../build`).

```bash
cmake -G "Unix Makefiles" -B ${BUILD_DIR} -DCMAKE_BUILD_TYPE=Release
cmake --build ${BUILD_DIR} --config Release -j
```

Executables are placed in `${BUILD_DIR}/bin`. The shared library is
`libopen_htj2k.so` / `libopen_htj2k.dylib` / `open_htj2k.dll` depending
on platform.

### Common CMake flags

| Flag | Default | Meaning |
|---|---|---|
| `-DCMAKE_BUILD_TYPE=<Release\|Debug\|RelWithDebInfo>` | (none) | Optimization and debug info level. `RelWithDebInfo` is the recommended mode for profiling. |
| `-DOPENHTJ2K_RTP=ON` | `OFF` | Build the experimental RFC 9828 RTP receiver (see below). Adds a GLFW dependency on every platform; uses native Metal on macOS and OpenGL 3.3 elsewhere. Also gates `open_htj2k_jpip_demo`. |
| `-DOPENHTJ2K_QUIC=ON` | `OFF` | Enable HTTP/3 over QUIC for the JPIP server and demo. Requires MsQuic + nghttp3; see [JPIP HTTP/3 prerequisites](#jpip-http3-prerequisites-opt-in) below. |
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

Four variants are produced under `web/build/html/`:

- `libopen_htj2k.js`           — scalar, single-threaded
- `libopen_htj2k_simd.js`      — WASM-SIMD 128-bit, single-threaded
  (recommended for most browsers)
- `libopen_htj2k_mt.js`        — scalar + pthreads (multi-threaded)
- `libopen_htj2k_mt_simd.js`   — WASM-SIMD + pthreads (fastest where
  available)

```bash
cd web
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
cd web
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
decodes incoming frames and displays them in a windowed viewer. Off by
default so the rest of the project builds without a window system.

**Prerequisites:**

- GLFW 3.x development headers (every platform):
  - Debian/Ubuntu: `libglfw3-dev`
  - Fedora: `glfw-devel`
  - Arch: `glfw`
  - macOS (Homebrew): `brew install glfw`
  - Windows (vcpkg): `vcpkg install glfw3`
- Renderer at runtime:
  - **Linux / Windows**: OpenGL 3.3 core profile. Install the OpenGL
    development headers (Debian/Ubuntu: `libgl1-mesa-dev`; Fedora:
    `mesa-libGL-devel`). Use `--color-path cpu` on hosts without a
    GL 3.3 context (the receiver auto-falls back).
  - **macOS**: native Metal renderer via the Metal / QuartzCore / Cocoa
    frameworks (already present in any Xcode Command Line Tools
    install). No OpenGL dependency on Apple silicon — the
    `OPENHTJ2K_USE_METAL` define is set automatically by CMake.

The same `-DOPENHTJ2K_RTP=ON` flag also builds the
`open_htj2k_jpip_demo` foveation viewer (it shares the GLFW renderer
with `open_htj2k_rtp_recv`). If you only need the JPIP server +
benchmark + browser viewer, the flag is unnecessary.

```bash
cmake -G "Unix Makefiles" -B ${BUILD_DIR} -DCMAKE_BUILD_TYPE=Release \
      -DOPENHTJ2K_RTP=ON
cmake --build ${BUILD_DIR} --config Release -j
```

Produces `${BUILD_DIR}/bin/open_htj2k_rtp_recv` and the offline decode
profiler `${BUILD_DIR}/bin/open_htj2k_rtp_decode_profile` for reproducing
the performance measurements documented in
[cli_rtp_recv.md](cli_rtp_recv.md).

## JPIP HTTP/3 prerequisites (opt-in)

`-DOPENHTJ2K_QUIC=ON` enables HTTP/3 over QUIC for both
`open_htj2k_jpip_server` and `open_htj2k_jpip_demo`. The HTTP/1.1
transport is always available; this flag only adds the H3 path.

**Required libraries:**

- **MsQuic** — Microsoft's QUIC implementation, used for the transport.
- **nghttp3** — HTTP/3 framing on top of MsQuic.

| Platform | Install |
|---|---|
| macOS (Homebrew) | `brew install libmsquic libnghttp3` |
| Debian/Ubuntu | No official package yet. Build MsQuic from <https://github.com/microsoft/msquic> and nghttp3 from <https://github.com/ngtcp2/nghttp3>; install to `/usr/local`. |
| Fedora / RHEL | Same as Debian — build from source. |
| Windows (vcpkg) | `vcpkg install ms-quic nghttp3` |

CMake calls `find_library(msquic)` and `find_path(msquic.h)` (likewise
for nghttp3); when either is missing it prints a `WARNING` and silently
omits the H3 transport. Verify the configure log contains
`OPENHTJ2K_QUIC: MsQuic found` and the matching nghttp3 line before
running the H3 demos.

The HTTP/1.1 server has no extra dependencies beyond the core library.

## WebTransport browser viewer (experimental)

A separate stack — Go relay (`tools/wt_bridge/`) plus a single-page
WebTransport viewer (`web/wt_viewer/`) — lets a Chromium browser display
RFC 9828 streams without any native binary on the viewing host. Build
prerequisites (Go ≥ 1.22, Node.js ≥ 18, Emscripten, OpenSSL, Python 3),
the LAN launcher, and the URL parameter reference live in
[**wt_viewer.md**](wt_viewer.md). The viewer reuses the
multi-threaded SIMD WASM artefact built by the WebAssembly section
above, so build that first.

## Running the test suite

```bash
ctest --test-dir ${BUILD_DIR}
```

Runs the full conformance suite (~582 tests) including HTJ2K and Part 1
decode conformance, encoder round-trip checks, and line-based
streaming validation. `-j` is supported.
