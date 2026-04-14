[![CMake](https://github.com/osamu620/OpenHTJ2K/actions/workflows/cmake.yml/badge.svg?branch=main)](https://github.com/osamu620/OpenHTJ2K/actions/workflows/cmake.yml)
[![CodeQL](https://github.com/osamu620/OpenHTJ2K/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/osamu620/OpenHTJ2K/actions/workflows/codeql-analysis.yml)
[![Packaging status](https://repology.org/badge/tiny-repos/openhtj2k.svg)](https://repology.org/project/openhtj2k/versions)

# OpenHTJ2K

OpenHTJ2K is an open-source C++ implementation of **JPEG 2000 Part 1**
(ITU-T Rec.800 | ISO/IEC 15444-1) and **High-Throughput JPEG 2000**
(Part 15; ITU-T Rec.814 | ISO/IEC 15444-15), with SIMD acceleration
across x86-64, AArch64, and WebAssembly, a built-in multi-threaded
pipeline, and a live RFC 9828 RTP receiver that sustains **4K @ 60 fps
on modern x86-64**.

## Highlights

**Standards compliance**
- Full HTJ2K encode + decode and Part 1 decode; partial Part 2
  (Downsampling Factor Structures, Arbitrary Transform Kernels).
- Fully conformance-tested against ITU-T Rec.803 | ISO 15444-4; 582
  tests in CI.
- JPH (`.jph`) file format, including colour specification box parsing
  for automatic YCbCr colorspace detection.

**Performance**
- SIMD: **AVX2**, **AVX-512** (x86-64), **NEON** (AArch64), and
  **WASM SIMD** 128-bit — for color transform, DWT, and HT block coding.
- Built-in thread pool for both encode and decode.
- Three decode APIs so callers can pick their memory/latency tradeoff:
  `invoke()` (batch), `invoke_line_based()` (streaming IDWT into a
  caller buffer), and `invoke_line_based_stream()` (row-callback;
  no intermediate W×H buffer).

**Deliverables**
- Shared library (`libopen_htj2k`) with C++ encoder/decoder APIs.
- CLI tools: `open_htj2k_enc`, `open_htj2k_dec`, and the experimental
  `open_htj2k_rtp_recv`.
- WebAssembly build (scalar / SIMD / pthreads / SIMD+pthreads) +
  Node.js CLI decoder + in-browser RTP replay demo (WebGL2 GPU
  rendering, Display-P3, planar Y/Cb/Cr textures with hardware
  chroma upsampling) — try both at **https://htj2k-demo.pages.dev/**.

**Live streaming (experimental)**
- `open_htj2k_rtp_recv` implements RFC 9828 (JPEG 2000 RTP with
  sub-codestream latency). Three-thread pipeline (receive / decode /
  render) with GPU shader rendering, HDR colour pipeline (PQ / HLG +
  BT.2020 gamut mapping), and an RTP-timestamp frame pacer.
  **Sustains 4K @ 60 fps** on 4K 4:2:2 1.7-bpp broadcast HT —
  `--threads 4` on modern x86-64 (AVX2) and `--threads 2` on Apple
  Silicon (M3 Max NEON, v0.13.2+).
  Opt-in via `-DOPENHTJ2K_RTP=ON`.

## Quick build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DOPENHTJ2K_THREAD=ON
cmake --build build -j
```

Executables land in `build/bin/`. For the full CMake flag reference,
the WebAssembly build, and the experimental RTP receiver build (GLFW
+ OpenGL), see [**docs/building.md**](docs/building.md).

## CLI quick start

Every CLI prints its full option reference via `-h`. The snippets
below show the most common invocations; fuller references and more
examples live under [`docs/`](docs/).

### Encoder — `open_htj2k_enc`

Part 15 HTJ2K encoder. Inputs: PGM, PPM, PGX, TIFF (libtiff).
Outputs: `.j2c` / `.jhc` (raw codestream) or `.jph` (JPH file format).

```bash
# Lossless encode
./build/bin/open_htj2k_enc -i input.ppm -o out.j2c Creversible=yes

# Lossy encode at quality 90
./build/bin/open_htj2k_enc -i input.ppm -o out.jph Qfactor=90

# Encode separate YCbCr component files
./build/bin/open_htj2k_enc -i Y.pgx,Cb.pgx,Cr.pgx -o out.jph -jph_color_space YCC
```

Key flags: `Creversible={yes|no}`, `Qfactor=0..100`, `Clevels=0..32`,
`Cblk={H,W}`, `Corder={LRCP|RLCP|RPCL|PCRL|CPRL}`,
`-num_threads N`. Full reference:
[**docs/cli_encoder.md**](docs/cli_encoder.md).

### Decoder — `open_htj2k_dec`

Part 1 and Part 15 decoder. Inputs: `.j2c` / `.j2k` / `.jph`.
Outputs: PPM / PGM / PGX / RAW.

```bash
# Decode to PPM (RGB output)
./build/bin/open_htj2k_dec -i input.j2c -o out.ppm

# Decode at half resolution
./build/bin/open_htj2k_dec -i input.j2c -o out_half.ppm -reduce 1

# Force BT.709 during YCbCr -> RGB matrix
./build/bin/open_htj2k_dec -i input.j2c -o out.ppm -ycbcr bt709
```

Key flags: `-reduce n`, `-num_threads n`, `-ycbcr {bt601|bt709}`,
`-batch`. Full reference:
[**docs/cli_decoder.md**](docs/cli_decoder.md).

### RFC 9828 RTP receiver — `open_htj2k_rtp_recv` (experimental)

Live HTJ2K over UDP per RFC 9828. Renders via GLFW / OpenGL 3.3
core. Opt-in at build time with `-DOPENHTJ2K_RTP=ON`.

```bash
# Default: bind 0.0.0.0:6000, render with GL 3.3 shader path
./build/bin/open_htj2k_rtp_recv

# Use --no-vsync + RTP-timestamp pacer (recommended on NVIDIA Wayland)
./build/bin/open_htj2k_rtp_recv --no-vsync

# Headless (no window), exit after 1000 frames
./build/bin/open_htj2k_rtp_recv --no-render --frames 1000
```

Key flags: `--port N`, `--bind host`, `--no-vsync`, `--frames N`,
`--threads 4` (x86-64 default) or `--threads 2` (Apple Silicon),
`--colorspace {bt709|bt601|bt2020}`.
Full reference, kernel `rmem_max` tuning, hardware requirements for
4K @ 60 fps sustained, and known issues:
[**docs/cli_rtp_recv.md**](docs/cli_rtp_recv.md).

## Supported file formats

### Library (codestream / file format)

| Extension | Encode | Decode | Description |
|-----------|:---:|:---:|-------------|
| `.jhc`, `.j2c`, `.j2k` | ✓ | ✓ | HTJ2K / JPEG 2000 Part 1 codestream |
| `.jph` | ✓ | ✓ | HTJ2K file format (JPH); colour specification box auto-detects YCbCr on decode |

### CLI I/O

| Format | `open_htj2k_enc` input | `open_htj2k_dec` output | Notes |
|--------|:---:|:---:|-------|
| PGM (`P5`) | ✓ | ✓ | Single grayscale component. Bit depth is auto-detected from `maxval` (8-bit or 16-bit). Decoder writes one file per component with a `_NN` suffix. |
| PPM (`P6`) | ✓ | ✓ | Packed RGB (3 equal-sized components). On decode, a subsampled YCbCr codestream is upsampled to luma resolution (nearest-neighbour for 4:2:2 / 4:2:0) before writing interleaved RGB — enable the YCbCr→RGB matrix with `-ycbcr bt601\|bt709`. |
| PGX | ✓ | ✓ | One component per file. Multi-component input is a comma-separated file list (e.g. `Y.pgx,Cb.pgx,Cr.pgx`) and the encoder computes `XRsiz` / `YRsiz` from each file's dimensions, so 4:2:2 / 4:2:0 Y/Cb/Cr inputs encode with the right subsampling. Both batch and streaming encode paths support this. Decoder writes one file per component with a `_NN` suffix. |
| TIFF | ✓ (batch only) | | Requires libtiff at build time. 8 or 16 bits **per sample**. TIFF input is supported on the batch (`-batch`) encode path only; the streaming encode path does not handle TIFF. Not a decoder output. |
| RAW | | ✓ | Decoder-only output. Packed samples, no header, one file per component with a `_NN` suffix. |

## Documentation

In-depth guides live under [`docs/`](docs/README.md):

- [docs/building.md](docs/building.md) — full CMake flag reference, native build, WASM + Node.js CLI, RTP receiver prerequisites
- [docs/cli_encoder.md](docs/cli_encoder.md) — `open_htj2k_enc` reference
- [docs/cli_decoder.md](docs/cli_decoder.md) — `open_htj2k_dec` reference
- [docs/cli_rtp_recv.md](docs/cli_rtp_recv.md) — `open_htj2k_rtp_recv` reference + operational guide

See also [CHANGELOG](CHANGELOG) for release history.

## Requirements

CMake 3.13+, a C++11-or-later compiler. CMake auto-selects C++17 →
C++14 → C++11 depending on compiler support; all three modes pass the
full conformance test suite. Per-standard feature differences are
covered in [docs/building.md](docs/building.md#requirements).

## License

BSD 3-Clause. See [LICENSE](LICENSE).
