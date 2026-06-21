# OpenSeadragon × HTJ2K tile source (M3)

An [OpenSeadragon](https://openseadragon.github.io/) `TileSource` that serves a
deep-zoom image straight out of a **single HTJ2K codestream**, decoding each tile
on demand in a Web Worker with the OpenHTJ2K WASM decoder. No pre-tiling, no DZI
pyramid on disk — one `.j2c`/`.jph` file, region + resolution selected per tile.

This is **Lever A** productised for the browser: the whole codestream is fetched
once and kept resident in the worker's WASM heap, and every OSD tile request
becomes a windowed *reuse* decode (`set_col_range`/`set_row_range`) at the
matching reduce level. Byte-range delivery (**Lever B** — fetch only a tile's
precincts) is a later drop-in that swaps the whole-file fetch for ranged fetches.

Targets **OpenSeadragon ≥ 6.0** (the `downloadTileStart` / `context.finish`
data-type graph). Output is 8-bit SDR, handed to OSD as a `context2d` tile.

## Files

| File | Role |
|---|---|
| `htj2k_geometry.mjs`   | Pure OSD `(level,x,y)` → reduce + clipped window math. No deps. |
| `htj2k_decode_core.mjs`| Shared decode core: WASM fn bindings, codestream load, per-reduce region-decoder pool. Used by the worker **and** the test. |
| `htj2k_tile_worker.mjs`| Web Worker: owns the WASM module + codestream, decodes tiles, posts RGBA back. |
| `htj2k_tilesource.mjs` | `HTJ2KTileSource extends OpenSeadragon.TileSource`. Main-thread glue + stats. |
| `htj2k_osd_demo.html`  | Demo viewer (OSD from CDN) with a decode-latency HUD. |
| `test_tile_grid.mjs`   | Headless correctness proof (Node): every tile byte-exact vs a full-level decode. |
| `probe_fixture.mjs`    | Dev util: print a codestream's dims / components / max reduce. |

## Prerequisites

Build the single-thread SIMD WASM module (the artifacts the worker loads):

```bash
cmake --build web/build --target libopen_htj2k_simd
# -> web/build/html/libopen_htj2k_simd.{js,wasm}
```

## Headless correctness test

No browser needed — proves the geometry + the exact decode core the worker runs:

```bash
node web/osd/test_tile_grid.mjs ~/Downloads/heic0602a.j2c
# PASS: N tiles across L levels byte-exact vs full-level decode
```

## Run the demo

The single-thread build needs **no** COOP/COEP headers (no SharedArrayBuffer), so
any static server works — but everything must be same-origin (module worker +
dynamic WASM import). Serve the `web/` directory:

```bash
web/osd/serve.sh                 # symlinks the Carina fixture + serves web/ on :8000
# then open  http://localhost:8000/osd/htj2k_osd_demo.html
```

Query params: `?src=<codestream-url>` (default `./carina.j2c`),
`?wasmBase=<dir>` (default `../build/html/`), `?tile=<px>` (default `256`).

## How OSD maps to HTJ2K

```
maxLevel = ceil(log2(max(W,H)))          // full-res level (OSD's own formula)
reduce   = maxLevel - level              // HTJ2K resolution reduction
minLevel = maxLevel - maxSafeReduce      // coarsest level the codestream can make
tile(x,y): x0=x*ts, y0=y*ts, w=min(ts, levelW-x0), h=min(ts, levelH-y0)  // clipped edges
```

`minLevel`/`maxLevel` are computed once from the codestream and passed to the OSD
`TileSource` constructor, so OSD and the decoder share one pyramid definition.

## Status & next steps

- ✅ Region→RGBA primitive (`web/src/wrapper.cpp`), byte-exact reduce 0–5.
- ✅ Geometry + worker + `TileSource` + demo; tile grid byte-exact (this test).
- ▶ **Lever B**: derive a precinct→byte-range index and fetch only a tile's
  precincts instead of the whole file.
- ▶ **Scale-out**: a pool of N workers (round-robin tiles) — decode is
  embarrassingly parallel across tiles. One worker = one 72 MB heap copy.
- ▶ HDR / >8-bit via a custom WebGL2 drawer (OSD's stock drawers are 8-bit).
