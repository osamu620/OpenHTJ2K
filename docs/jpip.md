# JPIP — foveated streaming and gigapixel browsing

OpenHTJ2K's JPIP pipeline implements **ISO/IEC 15444-9** (JPIP, 3rd
edition) for on-demand delivery of JPEG 2000 codestreams at the precinct
level. Clients ask for a *view-window* (resolution frame size + region
offset + region size); the server returns only the precincts that
contribute to that region, in the JPP-stream wire format. Network
transports: HTTP/1.1 and HTTP/3 over QUIC.

Three demos ship with the library:

| Demo | Binary / Page | Purpose |
|---|---|---|
| Mouse-driven foveation | `open_htj2k_jpip_demo` (native GLFW), [`jpip_demo.html`](https://htj2k-demo.pages.dev/jpip_demo.html) (browser) | Three concentric cones around the cursor (fovea / parafovea / periphery) at full / reduced / coarse resolutions |
| Gigapixel pan + zoom viewer | [`jpip_viewer.html`](https://htj2k-demo.pages.dev/jpip_viewer.html) (browser) | Viewport-region decode for images larger than GPU texture limits |
| Foveation vs full-image benchmark | `open_htj2k_jpip_benchmark` | Bandwidth reduction + decode speedup across an NxN gaze grid |

All three share the same [`open_htj2k_jpip_server`](#server-open_htj2k_jpip_server)
and the core `source/core/jpip/` library.

## Quick start

Build the server, native demo, and benchmark:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DOPENHTJ2K_THREAD=ON
cmake --build build -j
```

Serve a codestream and drive it from the native demo:

```bash
# Terminal 1 — stateless HTTP/1.1 server
./build/bin/open_htj2k_jpip_server input.j2c --port 8080

# Terminal 2 — mouse-driven foveation demo, fetches from the server
./build/bin/open_htj2k_jpip_demo --server localhost:8080
```

Or drive the in-process JPP round-trip directly (no server needed, still
exercises every byte of the JPP-stream wire format):

```bash
./build/bin/open_htj2k_jpip_demo input.j2c
```

Browser demos are deployed at `https://htj2k-demo.pages.dev/`; point
them at your own server with the `?server=` URL parameter or the
**Server** field in the top bar.

## Server — `open_htj2k_jpip_server`

Stateless HTTP/1.1 (optionally HTTP/3) server. Loads one `.j2c`
codestream, builds the precinct index + packet locator once, then
serves view-window requests.

```
open_htj2k_jpip_server <input.j2c>
    [--port N=8080]
    [--h3 --cert server.cert --key server.key]   # HTTP/3 (requires -DOPENHTJ2K_QUIC=ON)
    [--chunked]                                  # opt in to Transfer-Encoding: chunked
    [--no-chunked]                               # accepted for back-compat (now the default)
```

Query grammar (§C.4):

```
GET /jpip?fsiz=<fx>,<fy>&roff=<ox>,<oy>&rsiz=<sx>,<sy>&type=jpp-stream
          [&len=<N>][&model=...][&cnew=http]
```

- `fsiz` — target resolution frame size; the server picks the
  smallest discard level whose decoded size is ≥ fsiz.
- `roff`, `rsiz` — region within the resolution frame (default:
  full frame).
- `type=jpp-stream` — JPP-stream response (the only wire format this
  server speaks).
- `len=<N>` — §C.6.1 Maximum Response Length, in bytes.  The server
  emits whole messages up to `N` bytes of JPP-stream payload (EOR
  does not count) and terminates with EOR reason=4 (`ByteLimit`)
  when it stops early, or reason=2 (`WindowDone`) when the entire
  view-window fit.
- `model` — §C.9 cache-model advertisement; see
  [cache-model field](#cache-model-field--c9).
- `cnew=http` — §C.3.3 session/channel establishment.  The server
  replies with a `JPIP-cnew:
  cid=C<n>,path=jpip,transport=http` response header.  This server
  is stateless — the `cid` is opaque and the cache-model field
  carries all session state — but interactive clients require the
  header before they will commit received data-bins to their
  persistent cache.

HTTP **POST** is also accepted (§C.1): the query string moves into
the request body when GET's URL-length limit would otherwise
truncate large cache-model advertisements.  The server caps the
body at 16 MB.

Responses are complete JPP-streams.  Per §A.3.6.1, metadata-bin 0
is emitted first — even empty, as a session-binding signal —
followed by main-header (§A.3.6), one tile-header data-bin per
tile whose index appears in the view-window result (§A.3.3),
every precinct data-bin selected by the view-window resolver, and
an EOR.  Large precinct data-bins are chunked into 987-byte
messages (`is_last=0` on all but the final chunk).  Precincts are
delivered in ascending in-class-id order so clients that iterate
their cache by bin-id see no gaps.  CORS preflight (`OPTIONS`) is
handled for cross-origin browser access.

### Progressive (chunked) delivery

The server supports `Transfer-Encoding: chunked` progressive delivery:
each completed JPP-stream message is flushed to the socket as soon as
the server produces it, so chunked-aware clients can start decoding
while later precincts are still being built.  On a 24 MB full-canvas
response, loopback time-to-first-byte drops from ~7.4 ms
(Content-Length path) to ~0.44 ms (~17×); chunked and Content-Length
emit byte-identical bodies, so the mode swap has no effect on
correctness.

**Default is Content-Length.**  The server ships with
`Transfer-Encoding: chunked` *off* because some interactive reference
JPIP clients do not implement chunked transfer parsing and report
"connection closed unexpectedly" when the Content-Length header they
expect is absent.  Our browser demos + C++ `JpipClient` accept either
format; the client-side chunked refactor that enabled the TTFB win
stays useful wherever the server is started with `--chunked`.  Opt in
with:

```
open_htj2k_jpip_server <input.j2c> --chunked
```

The §C.6.1 `len=` rollback still works in chunked mode — each message
is built into a reusable scratch buffer and only committed to a wire
chunk once it fits under the cap.  HTTP/3 continues to use the
buffered data-reader path; progressive DATA-frame delivery over H3
would require an nghttp3 data-reader refactor and is tracked as a
follow-up.

## Native demo — `open_htj2k_jpip_demo`

Mouse-driven foveation. Three concentric cones follow the cursor:

| Cone | fsiz ratio | Default radius | What it delivers |
|---|---|---|---|
| Fovea | 1.0 (full resolution) | `canvas_w / 15` | Sharp detail around the cursor |
| Parafovea | 0.5 (half res; configurable) | `canvas_w / 8` | Mid-resolution context |
| Periphery | 0.125 (1/8 res; configurable) | whole image | Low-res anchor |

```
open_htj2k_jpip_demo [<input.j2c>]
    [--fovea-radius N]            # canvas px; default = canvas_w / 15
    [--parafovea-radius N]        # canvas px; default = canvas_w / 8
    [--parafovea-ratio F=0.5]     # fsiz ratio
    [--periphery-ratio F=0.125]   # fsiz ratio
    [--window-size WxH=1920x1080] # decouple window from canvas
    [--reduce N=0]                # discard DWT levels
    [--server host:port]          # HTTP/1.1 server mode
    [--server-h3 host:port]       # HTTP/3 server mode (requires -DOPENHTJ2K_QUIC=ON)
    [--use-filter]                # direct precinct filter (skip JPP round-trip)
    [--decode-on-move-only] [--no-vsync]
```

Three modes are always available:

- **In-process JPP round-trip** (default, needs `<input.j2c>`): per
  frame, emit JPP messages → parse them back → reassemble a sparse
  codestream → decode. Exercises the full wire format locally; useful
  for benchmarking without networking.
- **HTTP/1.1 client** (`--server host:port`): three concurrent
  `JpipClient::fetch_streaming` calls per frame (one per cone),
  pipelined on their own TCP connections.  The client feeds each
  HTTP chunk into the `StreamingJppParser` as it arrives, so the
  `DataBinSet` fills up while later chunks are still in flight
  rather than after a drain-to-EOF buffer pass.
- **HTTP/3 client** (`--server-h3 host:port`): three requests
  multiplexed on one QUIC connection via `H3Client::fetch_multi`.

`--use-filter` bypasses the JPP round-trip and installs the precinct
filter directly on the decoder — an A/B comparison path that still
works against local files. The window size can exceed the
canvas (decouples the GPU texture scaler from the decoded resolution)
or be smaller for cheap previews; Metal's 16384-pixel texture cap is
honoured so 21600 × 10800 NASA Blue Marble-sized canvases run cleanly
on Apple silicon.

## Benchmark — `open_htj2k_jpip_benchmark`

Measures bandwidth and decode time for foveated vs full-image
delivery across a gaze grid. No GUI or server needed.

```
open_htj2k_jpip_benchmark <input.j2c>
    [--gaze-grid NxN=5]
    [--reduce N=0]
    [--csv output.csv]
```

Example output on the 1920 × 1920 `land_shallow_topo_1920_fov.j2c`
reference asset:

```
gaze_x  gaze_y  │ precincts   bytes   decode │ bw_%  dec_%
────────────────┼────────────────────────────┼─────────────
0       0       │      171    89305   15.5ms │ 25.6  67.8
...
AVERAGE         │      175                   │ 21.0  69.9

Bandwidth reduction: 79.0%  |  Decode speedup: 1.4x
```

## Browser demos

### Foveation — `jpip_demo.html`

Mouse-over a canvas; three concurrent `fetch()` calls per gaze move
return fovea + parafovea + periphery JPP-streams, merged into a
WASM-side `DataBinSet`, reassembled into a sparse codestream, decoded,
and painted via WebGL2 (with bilinear filtering). Uses the
multi-threaded WASM variant (pthreads + SIMD) when the page is
cross-origin-isolated, single-threaded SIMD otherwise.

Each per-cone response is drained via `response.body.getReader()` and
fed into the WASM `jpip_feed_stream*` API as the browser yields bytes,
so when the server is launched with `--chunked` the progressive
delivery round-trips all the way into the decoder without buffering
the full body on the JS side first.  Against the default
Content-Length server the same code path still works — the browser
just yields larger chunks.

Per-frame cache semantics: the JS side calls `jpip_reset_cache` each
frame so the previous gaze's high-res precincts decay — without it,
the periphery would freeze under the last foveal hit. Headers
(main-header, tile-headers, metadata-bin 0) are preserved across the
reset, so `&model=Hm,Ht*,M0` is advertised on every request and the
server stops re-sending tens of KB of headers per frame.

URL parameters:

| Param | Meaning |
|---|---|
| `server=host[:port]` | JPIP server (matches the top-bar field) |
| `reduce=N` | discard N DWT levels |
| `para_ratio=F` | parafovea fsiz ratio |
| `peri_ratio=F` | periphery fsiz ratio |
| `variant={st,mt}` | force single- or multi-threaded WASM |

### Gigapixel viewer — `jpip_viewer.html`

Pan + zoom for images larger than the GPU texture limit (Metal:
16384 pixels on Apple silicon). Only the precincts that fall inside
the current viewport are fetched and decoded; the decoder's
viewport-region API (`jpip_end_frame_region`) runs the IDWT only on
the rows within the visible rectangle and a column-limited horizontal
1D IDWT skips samples outside the visible columns.

Mouse drag = pan. `Ctrl`-wheel = zoom. Trackpad: two-finger scroll =
pan, pinch = zoom. Keyboard: arrows = pan, `+/-` = zoom,
`Home` = reset. Reduce level auto-selects from zoom:
`reduce = ceil(log2(1/zoom)) − 1` clamped to `[0, 5]`.

URL parameters:

| Param | Meaning |
|---|---|
| `server=host[:port]` | JPIP server |
| `debounce=N` | trailing-debounce window in ms (default 60; `0` disables) |
| `debug` | per-frame timing console dump |
| `variant={st,mt}` | force single- or multi-threaded WASM |
| `maxSize=WxH` | cap the WebGL render target to `WxH` (default `1920x1080`). Bounds per-frame precinct fetch + decode work on 4K / ultrawide displays. Pass `maxSize=window` to disable the cap and render at full window resolution. |
| `fit={stretch,contain}` | `stretch` (default) scales the render target up to fill the window with GPU-side `GL_LINEAR`; `contain` shows the canvas at native pixel scale centered in the window — like the foveation demo — with the `maxSize` cap defining the centered rectangle. |

Pan events are debounced + coalesced: during an in-flight fetch, new
events flip a "pending" slot rather than queue a second request, so
the final viewport the user aimed at is always what lands on screen.
Fetch responses are streamed into the WASM `jpip_feed_stream*` API
per chunk.  Against a `--chunked` server this matches the wire-level
chunked delivery; against the default Content-Length server the JS
side still avoids the full-buffer `arrayBuffer()` round-trip, it just
sees fewer/larger chunks from the browser.

## Core architecture

```
source/core/jpip/
  precinct_index.{hpp,cpp}       — (t, c, r, p_rc) → JPIP sequence number s, identifier I
  view_window.{hpp,cpp}          — §C.4 view-window → precinct set
  vbas.{hpp,cpp}                 — VBAS codec (§A.2.1)
  jpp_message.{hpp,cpp}          — JPP message headers (§A.2, Tables A.1, A.2)
  data_bin_emitter.{hpp,cpp}     — header + precinct data-bin emission
  packet_locator.{hpp,cpp}       — per-precinct byte ranges in the codestream
  codestream_walker.{hpp,cpp}    — one-time codestream layout pass
  jpp_parser.{hpp,cpp}           — wire stream → DataBinSet (one-shot
                                    `parse_jpp_stream` + resumable
                                    `StreamingJppParser` for chunked
                                    clients; they share a single
                                    try_decode_one_message helper)
  codestream_assembler.{hpp,cpp} — DataBinSet → sparse codestream
  cache_model.{hpp,cpp}          — §C.9 client cache model
  jpip_request.{hpp,cpp}         — query parser
  jpip_response.{hpp,cpp}        — HTTP framer; chunked + Content-Length
                                    writers + parsers
  jpip_client.{hpp,cpp}          — HTTP/1.1 client with incremental
                                    chunked-transfer state machine
                                    (fetch_streaming + progress callback)
  tcp_socket.{hpp,cpp}           — cross-platform TCP wrapper
  h3_server.{hpp,cpp}            — MsQuic + nghttp3 HTTP/3 server
  h3_client.{hpp,cpp}            — MsQuic + nghttp3 HTTP/3 client
```

### Data-bin classes

- 0 — precinct (JPP-stream)
- 1 — extended precinct (JPP-stream, has Aux)
- 2 — tile header
- 6 — main header
- 8 — metadata

The End-of-Response (EOR) message is deliberately not a class. Per
§D.3 it sits outside Annex A and uses a special 0x00 identifier byte
rather than a Bin-ID VBAS; see the dedicated EOR handling below.

### JPIP sequence number / identifier

Per §A.3.2.1:

```
s = Σ_{r' < r} npw[r'] · nph[r']  +  p_rc
I = t + (c + s · num_components) · num_tiles
```

`I` is the in-class identifier used in every precinct data-bin
message header. `CodestreamIndex::I(t, c, r, p_rc)` computes it;
the client reassembler decomposes `I` back to `(t, c, r, p_rc)` by
inverting the formula and linear-scanning `s_offset[]`.

### Zero-skip IDWT

Absent precincts appear in the sparse codestream as one-byte empty
packet headers (`0x00`). Rows that entirely derive from absent
precincts carry zero subband data, and `idwt_2d_state::adv_step()`
short-circuits the lifting pass when both its neighbour rows are
zero. The cost drops to ~10 % of a populated row (the counter
increments and zero checks stay).

### Viewport-region decode

`jpip_end_frame_region(handle, rgb, w, h, x, y, rw, rh)`:

1. **Precinct filter** — the server already sends only the precincts
   that overlap the region; absent rows naturally zero-skip.
2. **Row limit** — `set_row_limit(ry1)` tells the decoder to stop
   iterating after the last row of the region (`ry1 = ceil((y+rh) /
   2^reduce)`). Rows before `ry0` still iterate but zero-skip.
3. **Column range** — `set_col_range(rx0, rx1)` makes the horizontal
   1D IDWT process only the columns inside the region. Samples
   outside the range aren't stored or lifted.

Result: a centred viewport on a 42K × 10K image decodes in ~400 ms
in WASM at `reduce=0` instead of ~3000 ms for the full canvas.

### Cache-model field (§C.9)

A client-side `CacheModel` tracks which non-precinct data-bins have
been received. Its `format()` method emits the §C.9 model field body
used as `&model=` — for example `Hm,Ht0,Ht1-5,M0`. Range compression
(`Ht1-5` instead of `Ht1,Ht2,Ht3,Ht4,Ht5`) keeps the query short on
multi-tile images.

| Class | Prefix | Example |
|---|---|---|
| Main header (6) | `Hm` | `Hm` (id always 0) |
| Tile header (2) | `Ht` | `Ht0`, `Ht1-5` |
| Precinct (0) | `Hp` | *not sent by the demos* |
| Metadata (8) | `M` | `M0` |

The browser demos track only headers — precincts are intentionally
excluded so the foveation demo's periphery decays when the gaze
moves.

### WASM C API

Defined in `subprojects/src/jpip_wrapper.cpp`, exported via
`-sEXPORTED_FUNCTIONS` in `subprojects/CMakeLists.txt`:

| Function | Purpose |
|---|---|
| `jpip_create_context(jpp, len)` | parse initial JPP-stream, build `CodestreamIndex` |
| `jpip_get_canvas_{width,height}` | canvas dimensions |
| `jpip_get_num_components`, `jpip_get_total_precincts` | asset metadata |
| `jpip_set_reduce(n)` | discard N DWT levels |
| `jpip_get_response_buffer(ctx, min_size)` | grow-only staging pointer; avoid per-frame `_malloc` |
| `jpip_add_response(ctx, buf, len)` | one-shot: parse a complete JPP-stream buffer and merge into the `DataBinSet` |
| `jpip_feed_stream_begin(ctx)` | progressive path: reset the per-context streaming parser before the first chunk of a new response (v0.17.0) |
| `jpip_feed_stream(ctx, buf, len)` | feed the next HTTP chunk of a chunked response; incomplete messages are buffered internally until the next call supplies the rest |
| `jpip_feed_stream_end(ctx)` | finalize the response; returns 0 if the stream ended at a clean JPP message boundary, nonzero if a partial message was still pending |
| `jpip_get_cache_model(ctx)` | C-string for `&model=` advertisement |
| `jpip_reset_cache(ctx)` | soft reset — drops precincts, keeps headers |
| `jpip_end_frame(ctx, rgb, w, h)` | full-canvas decode |
| `jpip_end_frame_region(ctx, rgb, w, h, x, y, rw, rh)` | viewport-region decode |
| `jpip_destroy_context(ctx)` | release |

The browser demos use the `jpip_feed_stream*` trio (not
`jpip_add_response`) so they can feed HTTP chunks into the decoder
as `response.body.getReader()` yields them; `jpip_add_response` is
preserved for one-shot callers that have the full body in hand.

## Deployment

For production-scale hosting (Docker + Cloudflare Tunnel + TLS
certificates for HTTP/3), see [`deploy/README.md`](../deploy/README.md).

## Known limitations

- **Progression order**: the core reassembler patches the COD marker
  to LRCP. When using `--server` mode against natively-LRCP or
  natively-RLCP codestreams, the demo falls back to `--use-filter`.
- **Layers**: v1 supports single-layer streams. Multi-layer bins are
  re-emitted per layer as-is; per-layer byte offsets within a bin
  are not parsed.
- **SOP / EPH markers**: refused by the reassembler (v1 scope).
- **TLS for HTTP/3**: the server requires `--cert` + `--key`; no
  automatic certificate management.

## References

- ISO/IEC 15444-9 (3rd edition) — JPIP.
- RFC 9114 — HTTP/3.
- [Deployment guide](../deploy/README.md) — Docker + Cloudflare Tunnel for the server.
