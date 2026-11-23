# Building OpenHTJ2K

Most people need only this:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
cmake --install build --prefix /usr/local   # optional
```

The tools are then in `build/bin/`. The rest of this page covers:
- the [requirements](#requirements);
- [build options](#common-cmake-flags);
- [binaries for other machines](#portable-x86-64-builds);
- [using the installed library](#installing-and-using-the-library);
- the optional parts: [WebAssembly](wasm.md),
  the [RTP receiver](#building-the-experimental-rfc-9828-rtp-receiver),
  [JPIP over HTTP/3](#jpip-http3-prerequisites-opt-in) and the
  [WebTransport viewer](#webtransport-browser-viewer-experimental).

Options for working on OpenHTJ2K itself, such as the C++ standard and
turning SIMD code paths off, are in
[CONTRIBUTING.md](../CONTRIBUTING.md#build-options-for-development).

## Requirements

**Mandatory:**

- CMake 3.13 or later. Two commands in these documents need a newer one: `cmake --install`
  needs 3.15 (otherwise use `cmake --build <dir> --target install`), and `ctest --test-dir`
  needs 3.20 (otherwise run `ctest` inside the build directory).
- A compiler supporting **C++11 or later** (GCC 4.8+, Clang 3.3+, MSVC 2015+, Apple Clang 6+).
  CMake uses the newest of C++17, C++14 and C++11 that the compiler supports.
- A **little-endian** target (x86, x86-64, ARM, AArch64, RISC-V, WebAssembly, ...). The block
  coders assume little-endian byte order. Configuring for a big-endian target (s390x, big-endian
  PowerPC or MIPS, ...) stops with an error rather than building a library that would write and
  read wrong codestreams.

**Optional (auto-detected):**

| Dependency | Enables | Install |
|---|---|---|
| **libtiff** | TIFF input in `open_htj2k_enc` (8/16-bit, RGB or grayscale, both planar configurations; tiled TIFFs are not supported) | Debian/Ubuntu: `libtiff-dev`; Fedora: `libtiff-devel`; macOS (Homebrew): `brew install libtiff`; vcpkg: `vcpkg install tiff` |

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
`libopenhtj2k.so` / `libopenhtj2k.dylib` / `openhtj2k.dll` depending
on platform.

### Common CMake flags

| Flag | Default | Meaning |
|---|---|---|
| `-DCMAKE_BUILD_TYPE=<Release\|Debug\|RelWithDebInfo>` | (none) | Optimization and debug info level. Use `Release` unless you are debugging. |
| `-DOPENHTJ2K_TARGET_CPU=<native\|x86-64-v3\|x86-64-v4\|generic>` | `native` | x86-64 code generation target; see [Portable x86-64 builds](#portable-x86-64-builds). |
| `-DBUILD_SHARED_LIBS=OFF` | `ON` | Build a static library instead of a shared one. |
| `-DOPENHTJ2K_RTP=ON` | `OFF` | Build the experimental RFC 9828 RTP receiver (see below). Adds a GLFW dependency on every platform; uses native Metal on macOS and OpenGL 3.3 elsewhere. Also gates `open_htj2k_jpip_demo`. |
| `-DOPENHTJ2K_QUIC=ON` | `OFF` | Enable HTTP/3 over QUIC for the JPIP server and demo. Requires MsQuic + nghttp3; see [JPIP HTTP/3 prerequisites](#jpip-http3-prerequisites-opt-in) below. |
| `-DOPENHTJ2K_BUILD_APPS=OFF` | `ON` | Build only the library, without the command-line tools (and therefore without the test suite). |
| `-DOPENHTJ2K_BUILD_TESTS=OFF` | `ON` | Skip the test tools and the CTest suite. |
| `-DOPENHTJ2K_BUILD_FUZZERS=ON` | `OFF` | Link the fuzz targets of `tests/fuzz` with a fuzzing engine (clang, static library); see [dev/fuzzing.md](dev/fuzzing.md). |

### Portable x86-64 builds

SIMD code paths are chosen at compile time, so the x86-64 target decides
which CPUs the binaries run on. The default, `native`, compiles with
`-march=native`: the fastest code for the build machine, but binaries can
stop with an illegal-instruction error on other CPUs. Use one of the other
targets for packages, containers, CI artifacts or anything copied to
another machine:

| `OPENHTJ2K_TARGET_CPU` | Runs on | GCC/Clang flags | MSVC |
|---|---|---|---|
| `native` (default) | the build machine (and identical CPUs) | `-march=native -mtune=native` | `/arch:AVX2` |
| `x86-64-v3` | any CPU with AVX2 (Intel Haswell, AMD Zen and newer) | `-march=x86-64-v3` | `/arch:AVX2` |
| `x86-64-v4` | any CPU with AVX-512 (F/BW/CD/DQ/VL) | `-march=x86-64-v4` | `/arch:AVX512` |
| `generic` | any x86-64 CPU (no AVX2; SSE2 code paths for the HT block encoder, HT MagSgn decoding, the DWT and the decoder's color transform, scalar ones elsewhere) | `-march=x86-64 -mtune=generic` | none |

Compilers without `-march=x86-64-v3`/`-v4` (GCC < 11, Clang < 12) get the
equivalent `-mavx2 -mfma -mbmi ...` feature flags instead.

Cost of portability, measured on a Ryzen 9 9950X (AVX-512 capable;
3840×2160 8-bit RGB image, `Qfactor=90`, command-line tools including file
I/O, best of 5, relative to `native`):

| Target | Decode, 1 / 8 threads | Encode, 1 / 8 threads |
|---|---|---|
| `x86-64-v4` | 1.05× / 0.97× | 1.01× / 1.01× |
| `x86-64-v3` | 1.14× / 1.02× | 1.20× / 1.07× |
| `generic` | 1.43× / 1.44× | 1.51× / 1.34× |

### Generator notes

- `-G "Xcode"` generates an Xcode project on macOS.
- `-G "Visual Studio 17 2022"` generates a Visual Studio 2022 solution.
  See the [CMake Visual Studio generators docs](https://cmake.org/cmake/help/latest/manual/cmake-generators.7.html#visual-studio-generators)
  for other versions.
- `-G "Ninja"` is the fastest generator on Linux/macOS if Ninja is
  installed.

### Installing and using the library

```bash
cmake --install ${BUILD_DIR} --prefix /usr/local
```

| Installed | Location |
|---|---|
| `open_htj2k_dec`, `open_htj2k_enc` | `<prefix>/bin` |
| `libopenhtj2k` (`.so` / `.dylib` / `.a`; on Windows the `.dll` goes to `bin` and the import library to `lib`) | `<prefix>/lib` (or `lib64`, following `GNUInstallDirs`) |
| Public headers `decoder.hpp`, `encoder.hpp`, `openhtj2k_message.hpp`, `planar_output_desc.hpp` | `<prefix>/include/open_htj2k/interface` |
| `open_htj2k.pc` | `<prefix>/lib/pkgconfig` |
| CMake package (`openhtj2k-config.cmake` and friends) | `<prefix>/lib/cmake/openhtj2k` |

To use the installed library from a CMake project:

```cmake
find_package(openhtj2k CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE openhtj2k::openhtj2k)
```

(add the install prefix to `CMAKE_PREFIX_PATH` if it is not a standard
location). A project that includes OpenHTJ2K with `add_subdirectory()` can
link the same `openhtj2k::openhtj2k` name. Without CMake, use pkg-config:

```bash
c++ my_app.cpp $(pkg-config --cflags --libs open_htj2k) -o my_app
# static library: pkg-config --static --cflags --libs open_htj2k
```

Either way, include the API as
`#include <open_htj2k/interface/decoder.hpp>` /
`#include <open_htj2k/interface/encoder.hpp>`, which cannot collide with
another library's headers of the same name; the short form
`#include "decoder.hpp"` / `#include "encoder.hpp"` also works. (A project
that uses OpenHTJ2K through `add_subdirectory()` has only the short form.)
The paths in `open_htj2k.pc` are relative to its `prefix=` line, so a
relocated install only needs that line changed.

The headers compile on their own as C++11 or later, without warnings under
strict flags (`-Wall -Wextra -Wpedantic -Wconversion` and the like), and the
library's internal macros (SIMD, threading, TIFF and QUIC support) do not
leak into your code. A static library built with TIFF support needs libtiff
when linking: the CMake package finds it with `find_dependency(TIFF)`, and
`pkg-config --static` lists it.

[library.md](library.md) shows how to decode and encode with the API.

On Windows, code using the DLL sees the API declared `__declspec(dllimport)`;
when linking the static library, define `OHTJ2K_STATIC` (the CMake target
and the `.pc` file of a static build do this for you). MSVC builds use the
DLL C runtime (`/MD`, `/MDd`) unless `CMAKE_MSVC_RUNTIME_LIBRARY` says
otherwise.

## Building for WebAssembly (WASM)

With [Emscripten](https://emscripten.org/):

```bash
emcmake cmake -S web -B web/build
cmake --build web/build -j
```

The modules this builds, how to use them in a web page or under Node.js, and the
Node.js command-line decoder are described in [wasm.md](wasm.md).

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
ctest --test-dir ${BUILD_DIR} -j
```

Runs the whole suite: decoding conformance for Part 1, Part 15 and Part 2,
encoder round trips, line-based decoding and more. Its layout and the test
helpers are described in [CONTRIBUTING.md](../CONTRIBUTING.md#build-and-test).
