[![CMake](https://github.com/osamu620/OpenHTJ2K/actions/workflows/cmake.yml/badge.svg?branch=main)](https://github.com/osamu620/OpenHTJ2K/actions/workflows/cmake.yml)
[![CodeQL](https://github.com/osamu620/OpenHTJ2K/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/osamu620/OpenHTJ2K/actions/workflows/codeql-analysis.yml)
[![Packaging status](https://repology.org/badge/tiny-repos/openhtj2k.svg)](https://repology.org/project/openhtj2k/versions)
# OpenHTJ2K
OpenHTJ2K is an open source implementation of ITU-T Rec.800 | ISO/IEC 15444-1 (JPEG 2000 Part 1) and ITU-T Rec.814 | ISO/IEC 15444-15 (High-Throughput JPEG 2000; HTJ2K).

# What OpenHTJ2K provides
OpenHTJ2K provides a shared library and sample applications with the following features:

**Decoding**
- Decodes ITU-T Rec.800 | ISO/IEC 15444-1 (JPEG 2000 Part 1) and ITU-T Rec.814 | ISO/IEC 15444-15 (HTJ2K) codestreams
- Partial support for JPEG 2000 Part 2: Downsampling Factor Structures (DFS) and Arbitrary Transform Kernels (ATK) — irreversible 9/7-based and reversible 5/3-based ATK kernels
- Fully compliant with conformance testing defined in ITU-T Rec.803 | ISO 15444-4
- Three decode APIs:
  - `invoke()` — batch (full-image) path; writes decoded samples into a pre-allocated W×H buffer
  - `invoke_line_based()` — streaming IDWT via ring buffers; writes row-by-row into a pre-allocated full-image buffer (lower peak memory than `invoke()`)
  - `invoke_line_based_stream()` — same streaming IDWT but delivers rows via a callback; avoids allocating the W×H output buffer entirely
- The line-based path is the default; the batch path is available with the `-batch` flag

**Encoding**
- Encodes into HTJ2K-compliant codestreams (.jhc/.j2c) and JPH files (.jph)
- Optional markers (COC, POC, etc.) and HT SigProp/MagRef passes are not implemented
- Up to **16 bit** per component sample supported
- Quality control for lossy compression via the `Qfactor` parameter
- Encoder input supports PGM, PPM, PGX, and TIFF (with libtiff); PGX streaming works without `-batch`
- Two encode APIs:
  - `invoke()` — batch (full-image) path
  - `invoke_line_based_stream()` — streaming push-row path driven by a source callback

**Performance**
- DWT internal precision is float32 throughout (FDWT and IDWT)
- SIMD acceleration: AVX2 (x86-64), NEON (AArch64), and WASM SIMD 128-bit for Color Transform, DWT, and HT block coding
- Multi-threaded encode and decode via a built-in thread pool

# Requirements
CMake 3.13 or later and a compiler supporting **C++11 or later**.

CMake automatically selects the highest standard supported by the compiler (C++17 → C++14 → C++11).
All three modes have been verified to produce a correct build and pass the full conformance test suite.

| Standard | Behaviour |
|---|---|
| C++17 (recommended) | `[[nodiscard]]` and `[[maybe_unused]]` attributes are active; `std::filesystem` used for path handling |
| C++14 | Attributes expand to nothing (no diagnostics lost at runtime); `stat()` fallback for path handling |
| C++11 | Same as C++14; additionally uses a built-in `make_unique` shim and `std::result_of` instead of `std::invoke_result_t` |

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
cmake --build ${BUILD_DIR} --config Release -j
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

### Node.js CLI decoder (`open_htj2k_dec.mjs`)

`open_htj2k_dec.mjs` is a Node.js ES module that wraps the WASM build so you
can decode J2C / J2K / JPH files from the terminal. It requires the WASM build
(see above) but not a platform-native C++ toolchain on the target machine.

**Requirements:** Node.js ≥ 18 and the WASM build (see above).

**Usage:**
```bash
cd subprojects
node open_htj2k_dec.mjs -i <input.j2c|.j2k|.jph> -o <output.ppm|.pgm> [-r <reduce_NL>]
```

| Option | Description |
|--------|-------------|
| `-i`, `--input`  | Input codestream (`.j2c`, `.j2k`, `.jph`) |
| `-o`, `--output` | Output image (`.ppm` for RGB, `.pgm` for grayscale) |
| `-r`, `--reduce` | Resolution reduction: skip `n` DWT levels (0 = full resolution) |

**Example:**
```bash
node open_htj2k_dec.mjs -i image.j2c -o image.ppm
node open_htj2k_dec.mjs -i image.j2c -o image_half.ppm -r 1   # half resolution
```

The script auto-selects the SIMD build (`libopen_htj2k_simd.js`) when
available, falling back to the scalar build. Decoding uses the streaming
`invoke_decoder_stream` path, keeping peak WASM heap well below the
full-image `int32` buffer approach (~52 MB peak for a 4K RGB image vs ~486 MB
with the batch path).

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
  - When the codestream uses DFS markers (Part 2), the value is clamped to the
    number of consecutive bidirectional DWT levels, avoiding nonsensical
    HONLY/VONLY outputs.
- `-num_threads n`
  - Number of threads. `0` (default) uses all available hardware threads.
- `-iter n`
  - Repeat decoding `n` times (benchmarking). Output is written only once.
- `-batch`
  - Use the batch (full-image) decode path. The default path is line-based (streaming).
- `-ycbcr bt601|bt709` *(experimental)*
  - Convert YCbCr to RGB during PPM output using full-range ITU-R BT.601 or
    BT.709 coefficients. Handles 4:2:2 nearest-neighbour chroma upsampling.
    Has no effect when writing PGX, PGM, or RAW outputs.

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
