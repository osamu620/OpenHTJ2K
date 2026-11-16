# WebAssembly: decoding in the browser and Node.js

The decoder is also built as WebAssembly modules that run in a web page or under
Node.js, with no plugin and no server-side decoding. The demos at
**https://htj2k-demo.pages.dev/** use them.

## Build

You need [Emscripten](https://emscripten.org/) (tested with 3.x and 5.x):

```bash
emcmake cmake -S web -B web/build
cmake --build web/build -j
```

The modules land in `web/build/html/`:

| Module | Contents | Use it for |
|---|---|---|
| `libopen_htj2k_simd.js` | WASM-SIMD (128-bit), one thread | most pages |
| `libopen_htj2k_mt_simd.js` | WASM-SIMD and threads | the fastest decoding, on a [cross-origin isolated](#threads-need-cross-origin-isolation) page |
| `libopen_htj2k.js`, `libopen_htj2k_mt.js` | as above without SIMD | browsers without WASM-SIMD |
| `libopen_htj2k_jpip.js`, `libopen_htj2k_jpip_mt_simd.js` | JPIP client and decoder | the JPIP demos ([jpip.md](jpip.md)) |

Each `.js` file is an ES module. Serve it with the `.wasm` file next to it, and with the
`.worker.js` file that older Emscripten versions write for the threaded modules.

## Decoding in a web page

```js
import createModule from './libopen_htj2k_simd.js';

const M = await createModule();
const bytes = new Uint8Array(await (await fetch('image.j2c')).arrayBuffer());
const src = M._malloc(bytes.length + 16);  // 16 bytes of padding for SIMD reads
M.HEAPU8.set(bytes, src);
const dec = M._create_decoder(src, bytes.length, 0);  // 0: full resolution
M._parse_j2c_data(dec);
const w = M._get_width(dec, 0), h = M._get_height(dec, 0);
const rgba = M._malloc(w * h * 4);
M._invoke_decoder_to_rgba(dec, rgba);
const pixels = M.HEAPU8.slice(rgba, rgba + w * h * 4);
M._release_j2c_data(dec);
M._free(rgba);
M._free(src);

const canvas = document.querySelector('canvas');
canvas.width = w;
canvas.height = h;
canvas.getContext('2d').putImageData(new ImageData(new Uint8ClampedArray(pixels.buffer), w, h), 0, 0);
```

- **Input:** the bytes can be a codestream (`.j2c`, `.j2k`, `.jhc`) or a JPH or JP2 file.
- **Lower resolution:** a third argument `n` to `create_decoder` drops `n` resolution levels.
  `get_width()` and `get_height()` still report the full size; the output is
  `ceil(full / 2^n)`.
- **YCbCr input without the colour transform:** for example, sub-sampled YCbCr, or a JPH
  file whose `get_colorspace(dec)` is 18. The RGBA buffer then holds Y, Cb and Cr. Convert
  it in place with `M._apply_ycbcr_bt601_to_rgba(rgba, w * h)`, or with the `bt709` variant.
- **Video:** to decode one frame after another, create the decoder once. For every later
  frame, call `M._reset_decoder_with_bytes(dec, src, length, 0)` and then
  `M._parse_j2c_data(dec)`. The RTP demo does this; see `web/shared/decoder_worker.mjs`.

The example's calls were checked under Node.js with a lossless codestream: the decoded
image equals the original pixels exactly. The other entry points are in
`web/src/wrapper.cpp`. For example, `invoke_decoder_planar_u8` writes the components to
separate planes for a GPU shader.

## Threads need cross-origin isolation

The multi-threaded modules use Emscripten's pthreads. Pthreads need a page that is
cross-origin isolated: its responses must carry `Cross-Origin-Opener-Policy: same-origin`
and `Cross-Origin-Embedder-Policy: require-corp` (or `credentialless`).

- **Hosts that can set headers:** on a static host that sets headers (for example
  Cloudflare Pages, through a `_headers` file), this works on the first load.
- **Local development:** for local development or `file://`, the bundled
  `web/coi-serviceworker.js` adds the headers through a service worker.

Create a threaded decoder with `M._create_decoder_mt(src, length, 0, threads)`.

- **Thread limit:** each module starts `min(navigator.hardwareConcurrency, 16)` worker
  threads when it loads. Ask for no more threads than that: decoder threads beyond it may
  never start, and the decode then hangs.
- **Reuse:** a decoder re-initialised with `reset_decoder_with_bytes()` keeps its thread
  count.

## Node.js command-line decoder

`web/open_htj2k_dec.mjs` decodes a file from the terminal with the WebAssembly build. It
needs Node.js 18 or later and the build above, and no native toolchain.

```bash
cd web
node open_htj2k_dec.mjs -i image.j2c -o image.ppm
node open_htj2k_dec.mjs -i image.j2c -o image_half.ppm -r 1   # half resolution
```

| Option | Description |
|--------|-------------|
| `-i`, `--input`     | Input codestream (`.j2c`, `.j2k`, `.jph`) |
| `-o`, `--output`    | Output image (`.ppm` for RGB, `.pgm` for grayscale, `.pgx` for per-component raw) |
| `-r`, `--reduce`    | Resolution reduction: skip `n` DWT levels (`0` = full resolution) |
| `-num_threads`, `-t` | Number of decode threads (`0` = auto-detect, `1` = single-threaded).  Selects the multi-threaded WASM build when `> 1` or `= 0`. |
| `-ycbcr`            | YCbCr→RGB conversion for PPM output: `bt601` or `bt709` (auto-detected from JPH `EnumCS` otherwise). |

The script uses the SIMD build (`libopen_htj2k_simd.js`) when it is there, otherwise the
scalar one. It decodes row by row, so a 4K RGB image needs about 52 MB of WASM memory.

## Demos

The site at **https://htj2k-demo.pages.dev/** is built this way.

| Page | What it shows |
|---|---|
| `index.html` | A still-image decoder: preset images, or a file you upload, shown at each resolution level. |
| `rtp_demo.html` | Replay of an RFC 9828 `.rtp` file. It renders with WebGL2, converting YCbCr to RGB on the GPU from planar Y, Cb and Cr textures, and uses Display-P3 on supported hardware. |
| `jpip_demo.html`, `jpip_viewer.html` | The JPIP demos; see [jpip.md](jpip.md). |

The live viewer for RFC 9828 streams over WebTransport is described in
[wt_viewer.md](wt_viewer.md). To benchmark the WebAssembly decoder, see
[dev/wasm_bench.md](dev/wasm_bench.md).
