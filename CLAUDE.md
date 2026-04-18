# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenHTJ2K is a C++ library implementing JPEG 2000 Part 1 (ISO/IEC 15444-1) decoding and Part 15 / HTJ2K (ISO/IEC 15444-15) encoding/decoding. Design priorities: **speed > memory efficiency > cache-friendliness**. OpenJPH is the primary benchmark target for single-threaded performance.

## Build

```bash
# Configure + build (from repo root)
cmake -G "Unix Makefiles" -B build -DCMAKE_BUILD_TYPE=Release -DOPENHTJ2K_THREAD=ON
cmake --build build --config Release -j

# Debug build
cmake -B build_dbg -DCMAKE_BUILD_TYPE=Debug -DOPENHTJ2K_THREAD=ON
cmake --build build_dbg -j
```

Executables in `build/bin/`. Release library: `open_htj2k_R`, Debug: `open_htj2k_d`. Binaries: `open_htj2k_enc`, `open_htj2k_dec`, `imgcmp`, `lb_compare`, `open_htj2k_jpip_server`, `open_htj2k_jpip_demo`, `open_htj2k_jpip_benchmark`.

SIMD and TIFF are auto-detected. On x86-64, `-march=native -mtune=native` is already set — **never add explicit `-mavx2` flags** in new SIMD files.

### WASM build
```bash
cd subprojects && mkdir -p build && cd build
emcmake cmake .. && cmake --build . -j
```

### QUIC build (HTTP/3 JPIP transport)
```bash
# macOS: brew install libmsquic libnghttp3
cmake -B build_quic -DCMAKE_BUILD_TYPE=Release -DOPENHTJ2K_QUIC=ON -DOPENHTJ2K_RTP=ON
cmake --build build_quic -j
```

## Testing

```bash
cd build

# Run all tests
ctest --build-config Release -VV

# Single test by name (regex)
ctest -R enc_lossless -VV

# Test categories
ctest -R "^dec_p0_ht_"  -VV   # HT Profile 0 decoder conformance
ctest -R "^dec_p1_"     -VV   # Part 1 decoder conformance
ctest -R "^dec_p2_dfs_" -VV   # Part 2 DFS/ATK
ctest -R "^batch_"      -VV   # Batch validation (invoke() path)
ctest -R "^lbs_"        -VV   # Line-based streaming validation
ctest -R "^enc_"        -VV   # Encoder round-trip
```

Conformance data in `conformance_data/`. Tests defined in `tests/*.cmake`.

Utilities: `imgcmp <decoded.pgx> <reference.pgx> <max_PAE> <min_PSNR_dB>` (PAE=0 for lossless). `lb_compare <codestream> [--stream]` compares line-based vs batch decode.

## Architecture

```
source/core/
  interface/   ← Public API: encoder.hpp, decoder.hpp (Pimpl, namespace open_htj2k)
  codestream/  ← Marker parsing & bitstream I/O
  coding/      ← HT block coding (encode+decode), MQ decoder, subband buffering
  transform/   ← DWT (FDWT/IDWT) and color transforms (ICT/RCT)
  jph/         ← JPH box-based file format parsing
  common/      ← ThreadPool, typedefs, utils
source/apps/   ← CLI tools: encoder, decoder, imgcmp, lb_compare
subprojects/   ← Emscripten/WASM build
```

### Decode pipeline
`parse()` → per-tile: HT block decode → dequantize → IDWT → inverse color transform. Three output modes: `invoke()` (batch), `invoke_line_based()` (ring-buffer streaming), `invoke_line_based_stream()` (callback, no W×H allocation).

### Encode pipeline
Optional RGB→YCbCr → FDWT → quantize → HT block encode. Two modes: `invoke()` (batch), `invoke_line_based_stream()` (push-row callback).

### Stateful DWT
- `fdwt_2d_state` — push-row (encoder); `idwt_2d_state` — pull-row (decoder)
- Ring buffers with `RING_DEPTH = 8`, slot = `row % RING_DEPTH`
- FDWT order is **vertical then horizontal** (V→H). Changing to H→V breaks lossless 5/3 due to integer rounding.

### SIMD dispatch (compile-time)
Each hot kernel has three variants: `*_generic()`, `*_avx2()`, `*_neon()` (+ `*_wasm()` for Emscripten). Selection via file-static function-pointer arrays under `#if` guards:

- x86-64: CMake sets `OPENHTJ2K_TRY_AVX2`; code checks both `OPENHTJ2K_TRY_AVX2` **and** `__AVX2__`
- AArch64: CMake sets `OPENHTJ2K_ENABLE_ARM_NEON`
- WASM: CMake sets `OPENHTJ2K_ENABLE_WASM_SIMD` with `-msimd128`

AVX2/NEON kernels use **unaligned** loads/stores. Row buffers are not required to be aligned.

SIMD files are guarded at the top so they compile to nothing on wrong platforms. Coverage across AVX2/NEON/WASM: FDWT, IDWT, color transform, and the HT block **CUP** (cleanup) pass. HT block **SPP/MRP** decode is present in all three files but is scalar SWAR (32-bit-word packing of a 4×4 quad + `ctz`-driven bit loop), not lane-parallel SIMD — the passes are bit-serial on the VLC stream, so intrinsic vectorization doesn't straightforwardly apply. AVX-512 exists for IDWT only (`idwt_avx512.cpp`).

### JPIP architecture (ISO/IEC 15444-9)
- `source/core/jpip/` — 16 files: precinct index, view-window resolver, JPP-stream wire format (VBAS, message headers, data-bins), codestream walker/assembler, packet locator, cache model, H3 server/client wrappers.
- `source/apps/jpip_server/` — stateless HTTP/1.1 + HTTP/3 server. Loads one J2C, builds index once, serves view-window requests.
- `source/apps/jpip_demo/` — mouse-driven foveation demo. 3-cone RoI (fovea/para/periphery). Modes: in-process JPP round-trip, HTTP/1.1 `--server`, HTTP/3 `--server-h3`.
- `source/apps/jpip_benchmark/` — bandwidth + decode workload comparison (foveated vs full-image) across an NxN gaze grid.
- **Phase 4 IDWT zero-skip**: `idwt_2d_state` tracks per-row zero flags in ring buffer; `adv_step()` skips lifting when both neighbor rows are zero. Cascades through DWT levels.
- **Phase 5 H3 transport**: MsQuic (QUIC) + nghttp3 (HTTP/3 framing). `H3Server`/`H3Client` wrappers in `h3_server.cpp`/`h3_client.cpp`. Build with `-DOPENHTJ2K_QUIC=ON`.
- **Data-bin classes**: 0=precinct, 2=tile-header, 6=main-header, 7=EOR, 8=metadata.
- **Cache model** (`cache_model.{hpp,cpp}`): client tracks received data-bins, sends `model=Hm,Ht0,M0` so server skips known headers.
- **TLM marker**: encoder emits tile-part lengths; decoder exposes via `TLM_marker` accessors.

## Key Types and Constants

Defined in `source/core/common/open_htj2k_typedef.hpp`:
- `sprec_t` = `float` — DWT internal precision. **Do not change to integer.**
- `SIMD_LEN_I32` = 8 (AVX2) / 4 (NEON)
- `SIMD_PADDING` = 32 — extra allocation margin for SIMD over-reads
- `DWT_VERT_STRIP` = 64 — column-strip width for vertical DWT (L1/L2 cache fit)

## Coding Style

`.clang-format`: Google base, 2-space indent, 108-char line limit, `SortIncludes: Never` (do not reorder includes), `IndentPPDirectives: BeforeHash`. All headers use `.hpp`.

## Memory Allocation

- `cblk_data_pool` — per-thread monotonic bump allocator for HT block bitstreams (replaces per-codeblock malloc)
- Vertical DWT scratch: single flat buffer allocated once, reused across strips

## Threading

Built-in `ThreadPool` (no OpenMP/TBB dependency). Enable with `-DOPENHTJ2K_THREAD=ON`. Runtime: `-num_threads 0` for hardware concurrency.

## Encoder Limitations

- HT SigProp/MagRef passes not implemented (Cleanup only)
- COC/POC markers not implemented
- Up to 16 bpp
