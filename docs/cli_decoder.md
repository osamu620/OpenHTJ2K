# `open_htj2k_dec` — decoder CLI reference

Both Part 1 (JPEG 2000) and Part 15 (HTJ2K) compliant decoding are
supported. Accepts `.j2c` / `.j2k` / `.jph` inputs and writes
PGM / PPM / PGX / RAW outputs.

Full runtime help: `open_htj2k_dec -h`.

## Synopsis

```bash
open_htj2k_dec -i <codestream> -o <output> [options...]
```

`-i` takes a single codestream file; `-o` takes the output path. The
output extension decides the writer:

- `.pgm` — grayscale single-plane output
- `.ppm` — RGB three-plane output (YCbCr sources get matrixed to RGB
  automatically; see `-ycbcr` below)
- `.pgx` — per-component PGX files; the library writes one file per
  component with a `_N` suffix
- `.raw` — packed raw samples, no header

## Options

- `-reduce n`
  - Decode at a reduced resolution by skipping `n` DWT levels.
  - When the codestream uses DFS markers (Part 2), the value is
    clamped to the number of consecutive bidirectional DWT levels,
    avoiding nonsensical HONLY / VONLY outputs.
- `-num_threads n`
  - Number of threads. `0` (default) uses all available hardware threads.
- `-iter n`
  - Repeat decoding `n` times (benchmarking). Output is written only once.
- `-batch`
  - Use the batch (full-image) decode path. The default path is
    line-based (streaming).
- `-ycbcr bt601|bt709` *(experimental)*
  - Convert YCbCr to RGB during PPM output using full-range
    ITU-R BT.601 or BT.709 coefficients. Handles 4:2:0 and 4:2:2
    nearest-neighbour chroma upsampling. Has no effect when writing
    PGX, PGM, or RAW outputs.
  - When decoding a `.jph` file whose colour specification box declares
    YCbCr (`EnumCS = 18`), BT.601 conversion is applied automatically;
    use `-ycbcr bt709` to override.

## Examples

```bash
# Decode to PPM (RGB output)
open_htj2k_dec -i input.j2c -o out.ppm

# Decode at half resolution
open_htj2k_dec -i input.j2c -o out_half.ppm -reduce 1

# Force BT.709 during YCbCr -> RGB matrix
open_htj2k_dec -i input.j2c -o out.ppm -ycbcr bt709

# Decode to per-component PGX files (writes out_0.pgx, out_1.pgx, out_2.pgx)
open_htj2k_dec -i input.j2c -o out.pgx

# Benchmark: decode 10 times, single-threaded
open_htj2k_dec -i input.j2c -o out.ppm -num_threads 1 -iter 10
```
