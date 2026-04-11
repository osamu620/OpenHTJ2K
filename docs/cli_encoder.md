# `open_htj2k_enc` — encoder CLI reference

Part 15 (HTJ2K) compliant encoder. Produces either a raw codestream
(`.j2c`, `.jhc`) or an HTJ2K JPH file (`.jph`). Accepts PGM, PPM,
PGX, and TIFF (with libtiff) inputs. PGX streaming supports
subsampled component sets (4:2:2, 4:2:0) without `-batch`.

Full runtime help: `open_htj2k_enc -h`.

## Synopsis

```bash
open_htj2k_enc -i <input-image(s)> -o <output> [options...]
```

Multiple input files can be passed as a comma-separated list. For
example, to encode separate YCbCr component files:

```bash
open_htj2k_enc -i inputY.pgx,inputCb.pgx,inputCr.pgx -o output
```

`-o` takes the base name; the extension decides the container:
`.j2c` / `.jhc` for raw HTJ2K codestream, `.jph` for the JPH file
format.

## Options

### Tile and image structure

- `Stiles=Size`
  - Tile size in `{height,width}` format. Default is equal to the image size.
- `Sorigin=Size`
  - Offset from the reference grid origin to the image area. Default is `{0,0}`.
- `Stile_origin=Size`
  - Offset from the reference grid origin to the first tile. Default is `{0,0}`.

### DWT and codeblock structure

- `Clevels=Int`
  - Number of DWT decomposition levels. Valid range: 0–32. Default is **5**.
- `Creversible=yes|no`
  - `yes` for lossless mode, `no` for lossy mode. Default is **no**.
- `Cblk=Size`
  - Code-block size. Default is **{64,64}**.
- `Cprecincts=Size`
  - Precinct size. Must be a power of two.
- `Cycc=yes|no`
  - `yes` to apply RGB→YCbCr color space conversion. Default is **yes**.
- `Corder=<LRCP|RLCP|RPCL|PCRL|CPRL>`
  - Progression order. Default is **LRCP**.
- `Cuse_sop=yes|no`
  - `yes` to insert SOP (Start Of Packet) marker segments. Default is **no**.
- `Cuse_eph=yes|no`
  - `yes` to insert EPH (End of Packet Header) markers. Default is **no**.

### Quantization and quality

- `Qstep=Float`
  - Base step size for quantization. Valid range: `0.0 < Qstep <= 2.0`.
- `Qguard=Int`
  - Number of guard bits. Valid range: 0–8. Default is **1**.
- `Qfactor=Int`
  - Quality factor for lossy compression. Valid range: 0–100
    (100 = best quality).
  - When specified, `Qstep` is ignored and `Cycc` is set to `yes`.

### JPH and component layout

- `-jph_color_space RGB|YCC`
  - Declare the color space of the input components. Use `YCC` if the
    inputs are already in YCbCr.

### Runtime

- `-num_threads Int`
  - Number of threads. `0` (default) uses all available hardware threads.
- `-batch`
  - Use the batch (full-image) encode path. Loads the entire image into
    memory before encoding. The default path is line-based (streaming).

## Examples

```bash
# Lossless encode, default DWT / codeblock settings
open_htj2k_enc -i input.ppm -o out.j2c Creversible=yes

# Lossy encode at quality 90, write to JPH file format
open_htj2k_enc -i input.ppm -o out.jph Qfactor=90

# Encode YCbCr components into a JPH file with explicit color space
open_htj2k_enc -i Y.pgx,Cb.pgx,Cr.pgx -o out.jph -jph_color_space YCC

# Lossless encode with 6 DWT levels and RPCL progression
open_htj2k_enc -i input.ppm -o out.j2c Creversible=yes Clevels=6 Corder=RPCL
```
