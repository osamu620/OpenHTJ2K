[![CMake](https://github.com/osamu620/OpenHTJ2K/actions/workflows/cmake.yml/badge.svg?branch=main)](https://github.com/osamu620/OpenHTJ2K/actions/workflows/cmake.yml)
[![CodeQL](https://github.com/osamu620/OpenHTJ2K/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/osamu620/OpenHTJ2K/actions/workflows/codeql-analysis.yml)
[![Packaging status](https://repology.org/badge/tiny-repos/openhtj2k.svg)](https://repology.org/project/openhtj2k/versions)
# OpenHTJ2K
OpenHTJ2K is an open source implementation of ITU-T Rec.814 | ISO/IEC 15444-15 (a.k.a. JPEG 2000 Part 15, High-Throughput JPEG 2000; HTJ2K)

# What OpenHTJ2K provides
OpenHTJ2K provides a shared library and sample applications with the following features:

**Decoding**
- Decodes ITU-T Rec.800 | ISO/IEC 15444-1 (JPEG 2000 Part 1) and ITU-T Rec.814 | ISO/IEC 15444-15 (HTJ2K) codestreams
- Fully compliant with conformance testing defined in ITU-T Rec.803 | ISO 15444-4
- Three decode APIs:
  - `invoke()` — batch (full-image) path
  - `invoke_line_based()` — streaming row-by-row output via callback, using per-subband ring buffers
  - `invoke_line_based_stream()` — streaming path for multi-tile images, assembling full-width rows
- The line-based path is the default; the batch path is available with the `-batch` flag

**Encoding**
- Encodes into HTJ2K-compliant codestreams (.jhc/.j2c) and JPH files (.jph)
- Optional markers (COC, POC, etc.) and HT SigProp/MagRef passes are not implemented
- Up to **16 bit** per component sample supported
- Quality control for lossy compression via the `Qfactor` parameter
- Two encode APIs:
  - `invoke()` — batch (full-image) path
  - `invoke_line_based_stream()` — streaming push-row path driven by a source callback

**Performance**
- DWT internal precision is float32 throughout (FDWT and IDWT)
- SIMD acceleration: AVX2 (x86-64), NEON (AArch64), and WASM SIMD 128-bit for Color Transform, DWT, and HT block coding
- Multi-threaded encode and decode via a built-in thread pool

# Requirements
cmake (version 3.14 or later) and a C++17 compliant compiler.

# Building
`./` is the root of the cloned repository and `${BUILD_DIR}` is a build directory (e.g. `../build` or `./build`).

- Specify `-DCMAKE_BUILD_TYPE=Debug` or `-DCMAKE_BUILD_TYPE=RelWithDebInfo` to include debug information.
- Specify `-G "Xcode"` to generate an Xcode project.
- Specify `-G "Visual Studio 17 2022"` for Visual Studio 2022. See
  https://cmake.org/cmake/help/latest/manual/cmake-generators.7.html#visual-studio-generators for other versions.
- Specify `-DOPENHTJ2K_THREAD=ON` to enable multi-threaded encode/decode (recommended).

```
cd ./
cmake -G "Unix Makefiles" -B ${BUILD_DIR} -DCMAKE_BUILD_TYPE=Release -DOPENHTJ2K_THREAD=ON
cd  ${BUILD_DIR}
make
```

Executables are placed in `${BUILD_DIR}/bin`.

## Building for WebAssembly (WASM)
Requires [Emscripten](https://emscripten.org/) (tested with 3.x / 5.x).

Two variants are produced under `subprojects/build/html/`:
- `libopen_htj2k.js` — scalar build
- `libopen_htj2k_simd.js` — WASM SIMD 128-bit build (recommended for modern browsers)

```bash
cd subprojects
mkdir -p build && cd build
emcmake cmake ..
cmake --build . -j
```

A live demo is available at **https://htj2k-demo.pages.dev/**

# Usage
## Encoder
Only Part 15 compliant encoding is supported. Both .j2c (codestream) and .jph (file format) are available.
```bash
./open_htj2k_enc -i input-image(s) -o output [options...]
```
The encoder accepts comma-separated multiple input files. For example, to encode YCbCr components:
```
./open_htj2k_enc -i inputY.pgx,inputCb.pgx,inputCr.pgx -o output
```

### Options
- `Stiles=Size`
  - Tile size in `{height,width}` format. Default is equal to the image size.
- `Sorigin=Size`
  - Offset from the reference grid origin to the image area. Default is `{0,0}`.
- `Stile_origin=Size`
  - Offset from the reference grid origin to the first tile. Default is `{0,0}`.
- `Clevels=Int`
  - Number of DWT decomposition levels. Valid range: 0–32. Default is **5**.
- `Creversible=yes or no`
  - `yes` for lossless mode, `no` for lossy mode. Default is **no**.
- `Cblk=Size`
  - Code-block size. Default is **{64,64}**.
- `Cprecincts=Size`
  - Precinct size. Must be a power of two.
- `Cycc=yes or no`
  - `yes` to apply RGB→YCbCr color space conversion. Default is **yes**.
- `Corder`
  - Progression order: `LRCP`, `RLCP`, `RPCL`, `PCRL`, `CPRL`. Default is **LRCP**.
- `Cuse_sop=yes or no`
  - `yes` to insert SOP (Start Of Packet) marker segments. Default is **no**.
- `Cuse_eph=yes or no`
  - `yes` to insert EPH (End of Packet Header) markers. Default is **no**.
- `Qstep=Float`
  - Base step size for quantization. Valid range: `0.0 < Qstep <= 2.0`.
- `Qguard=Int`
  - Number of guard bits. Valid range: 0–8. Default is **1**.
- `Qfactor=Int`
  - Quality factor for lossy compression. Valid range: 0–100 (100 = best quality).
  - When specified, `Qstep` is ignored and `Cycc` is set to `yes`.
- `-jph_color_space`
  - Declare the color space of input components: `RGB` or `YCC`.
  - Use `YCC` if the inputs are already in YCbCr.
- `-num_threads Int`
  - Number of threads. `0` (default) uses all available hardware threads.
- `-batch`
  - Use the batch (full-image) encode path. Loads the entire image into memory before encoding.
  - The default path is line-based (streaming).

## Decoder
Both Part 1 and Part 15 compliant decoding are supported.
```bash
./open_htj2k_dec -i codestream -o output [options...]
```

### Options
- `-reduce n`
  - Decode at a reduced resolution by skipping `n` DWT levels.
- `-batch`
  - Use the batch (full-image) decode path. The default path is line-based (streaming).

## Supported file formats
### Encoder input / Decoder output
| Format | Encoder input | Decoder output |
|--------|:---:|:---:|
| PGM / PPM | ✓ | ✓ |
| PGX | ✓ | ✓ |
| TIFF (libtiff required, 8/16 bpp) | ✓ | |
| RAW | | ✓ |

### Codestream formats
| Extension | Description |
|-----------|-------------|
| `.jhc`, `.j2c`, `.j2k` | HTJ2K / JPEG 2000 Part 1 codestream |
| `.jph` | HTJ2K file format (JPH); specifying this as encoder output triggers JPH creation |
