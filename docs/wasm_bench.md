# `wasm_bench.mjs` — WASM decoder benchmark harness

`subprojects/wasm_bench.mjs` is an iteration-loop benchmark driver for
the WebAssembly build of the OpenHTJ2K decoder. It pays the WASM
startup cost once, runs _N_ decodes of the same file in a loop, and
reports min/median/p95/mean wall-clock together with Msamples/s and
fps. It is the right tool for measuring steady-state decode throughput
and for byte-exact regression testing during WASM-side perf work.

For one-shot decode to a file, use `subprojects/open_htj2k_dec.mjs`
(documented under the WebAssembly section of
[`building.md`](building.md)). The two scripts share the same WASM
loader but serve different purposes: `open_htj2k_dec.mjs` writes a
single PPM/PGM/PGX output, while `wasm_bench.mjs` times repeated
decodes and optionally dumps planar buffers for comparison.

## Prerequisites

Build at least one WASM variant first. The bench driver looks in
`${SUBPROJECTS}/../build_wasm_prof/html/` by default; override with
`--build-dir` to point elsewhere.

```bash
emcmake cmake -S subprojects -B build_wasm_prof -DCMAKE_BUILD_TYPE=Release
cmake --build build_wasm_prof \
      --target libopen_htj2k_simd libopen_htj2k_mt_simd \
      -j$(nproc)
```

Add `-DOPENHTJ2K_WASM_PROFILE=ON` at configure time if you also intend
to collect CPU profiles; see [profiling](#profiling) below. The
`--profiling-funcs` linker flag preserves Wasm function names and adds
~15% to `.wasm` size with zero measured runtime cost.

## Synopsis

```bash
node subprojects/wasm_bench.mjs -i <codestream> [options...]
```

## Options

| Option | Default | Description |
|---|---|---|
| `-i`, `--input <file>` | — (required) | Input codestream (`.j2c`, `.j2k`, or `.jph`). |
| `--variant scalar\|simd\|mt\|mt_simd` | `simd` | WASM build to load. `mt*` variants require `--threads > 0`. |
| `--threads N` | `1` | Number of decode threads. Ignored by `scalar` / `simd`. `0` = auto (uses `navigator.hardwareConcurrency`). |
| `--iters N` | `20` | Number of measured iterations. |
| `--warmup N` | `3` | Number of unmeasured iterations before measurement. |
| `--mode stream\|planar_u8` | `stream` | Which decoder entry point to call. `stream` uses `invoke_decoder_stream` (PPM/PGM path); `planar_u8` uses `invoke_decoder_planar_u8` (WASM RTP demo path). |
| `--reduce N` | `0` | Resolution reduction level (0 = full resolution). |
| `--build-dir <path>` | `../build_wasm_prof/html` | Override WASM binary directory. |
| `--dump-planes <prefix>` | off | `planar_u8` only: on the final iteration write each component plane as `<prefix>_{Y,Cb,Cr}.pgm`. Used for byte-exact diff checks. |

The script prints a JSON object to stdout on completion:

```json
{
  "variant": "simd", "threads": 1,
  "input": "...", "dims": {"W": 3840, "H": 2160, "C": 3, "depth": 12},
  "iters": 20,
  "total_ms":  { "min": ..., "p50": ..., "p95": ..., "max": ..., "mean": ... },
  "parse_ms":  { ... },
  "decode_ms": { ... },
  "throughput_msamples_per_s_mean": ...,
  "fps_mean": ...
}
```

## Examples

### Baseline throughput check

```bash
node subprojects/wasm_bench.mjs \
     -i build-f32/bin/u05Q90.j2c \
     --variant simd --iters 20 --warmup 3
```

### Thread-scaling sweep on the multi-threaded variant

```bash
for t in 1 2 4 8; do
  echo "=== $t threads ==="
  node subprojects/wasm_bench.mjs \
       -i build-f32/bin/u05Q90.j2c \
       --variant mt_simd --threads $t --iters 15 --warmup 3
done
```

### WASM RTP-demo-shaped path (planar u8)

`invoke_decoder_planar_u8` writes per-component u8 buffers at native
(per-component) resolution — the shape the browser RTP demo
(`subprojects/rtp_demo.html`) uses to upload three R8 textures per
frame for GPU-side YCbCr→RGB.

```bash
node subprojects/wasm_bench.mjs \
     -i conformance_data/ATK_DFS_IRV.j2c \
     --variant mt_simd --threads 2 --iters 15 --warmup 3 \
     --mode planar_u8
```

### Byte-exact regression check

`--mode planar_u8` has no conformance-test coverage, so when modifying
that path record an explicit plane-level checksum:

```bash
# Capture baseline BEFORE your code change.
node subprojects/wasm_bench.mjs \
     -i some_file.j2c --variant simd --iters 1 \
     --mode planar_u8 --dump-planes /tmp/before/myfile

# ...edit wrapper.cpp, rebuild WASM...

# Capture AFTER and diff.
node subprojects/wasm_bench.mjs \
     -i some_file.j2c --variant simd --iters 1 \
     --mode planar_u8 --dump-planes /tmp/after/myfile

cmp /tmp/before/myfile_Y.pgm  /tmp/after/myfile_Y.pgm
cmp /tmp/before/myfile_Cb.pgm /tmp/after/myfile_Cb.pgm
cmp /tmp/before/myfile_Cr.pgm /tmp/after/myfile_Cr.pgm
```

For good coverage of the wrapper paths, diff at least:

- 8-bit 4:4:4 (e.g. `conformance_data/p0_04.j2k`)
- 12-bit 4:4:4 (any 12 bpc YCbCr file)
- 12-bit 4:2:2 (e.g. `conformance_data/ATK_DFS_IRV.j2c`)
- Sub-8-bit grayscale (e.g. `conformance_data/ds0_ht_03_b14.j2k`)

## Profiling

### Single-threaded variants: `node --cpu-prof`

```bash
# Produces bench.cpuprofile in the current directory.
# Load it via Chrome DevTools → Performance → "Load profile".
node --cpu-prof --cpu-prof-name=bench.cpuprofile --cpu-prof-interval=100 \
     subprojects/wasm_bench.mjs \
     -i build-f32/bin/u05Q90.j2c --variant simd --iters 30 --warmup 5
```

### Multi-threaded variants: `perf record`

`node --cpu-prof` **silently produces no output file** for the `mt` /
`mt_simd` variants — Emscripten's pthread teardown bypasses Node's
CPU-profile finalizer. Use Linux `perf` instead, with Node's
`--perf-basic-prof-only-functions` so Wasm JIT code is mapped to
readable names:

```bash
perf record -F 499 -g -o bench.perf.data --call-graph dwarf -- \
     node --perf-basic-prof-only-functions \
          subprojects/wasm_bench.mjs \
          -i build-f32/bin/u05Q90.j2c \
          --variant mt_simd --threads 2 --iters 20 --warmup 3

perf report -i bench.perf.data --stdio --no-children -g none | head -30
```

On most Linux hosts this needs `kernel.perf_event_paranoid` at 2 or
lower (`sudo sysctl kernel.perf_event_paranoid=2`).
