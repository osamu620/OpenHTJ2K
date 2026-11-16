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

All three share the same [`open_htj2k_jpip_server`](#server--open_htj2k_jpip_server)
and the core `source/core/jpip/` library.

## Build prerequisites

The three native binaries have different build-flag requirements:

| Binary | Build flag | Native deps |
|---|---|---|
| `open_htj2k_jpip_server` | none (always built) | None beyond the core library |
| `open_htj2k_jpip_benchmark` | none (always built) | None beyond the core library |
| `open_htj2k_jpip_demo` | **`-DOPENHTJ2K_RTP=ON`** (shares the GLFW renderer with `open_htj2k_rtp_recv`) | GLFW 3.x; OpenGL 3.3 on Linux/Windows or Metal on macOS — see [building.md](building.md#building-the-experimental-rfc-9828-rtp-receiver) |

For HTTP/3 over QUIC on either the server or the native demo, add
`-DOPENHTJ2K_QUIC=ON`. Required libraries (MsQuic + nghttp3) and
per-platform install commands are documented in
[building.md → JPIP HTTP/3 prerequisites](building.md#jpip-http3-prerequisites-opt-in).
The HTTP/1.1 transport is always available.

## Quick start

Build the server, benchmark, and native demo (the demo needs the
extra `-DOPENHTJ2K_RTP=ON` flag because it shares its renderer with
`open_htj2k_rtp_recv`):

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DOPENHTJ2K_RTP=ON
cmake --build build -j
```

If you only need the server and benchmark (e.g. for CI or headless
deployment), drop `-DOPENHTJ2K_RTP=ON` and the cmake step has no
window-system or GPU dependencies.

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

Browser demos are deployed at `https://htj2k-demo.pages.dev/` and run
in any current Chromium- or Firefox-based browser; point them at your
own server with the `?server=` URL parameter or the **Server** field
in the top bar. Building the browser demos locally requires
Emscripten — see [building.md → Building for WebAssembly](building.md#building-for-webassembly-wasm).

## Server — `open_htj2k_jpip_server`

HTTP/1.1 (optionally HTTP/3) server. Loads one `.j2c` codestream,
builds the precinct index + packet locator once, then serves
view-window requests — statelessly by default (the client's `model`
field carries its cache state), or within §B.2 sessions when the
client establishes a channel with `cnew=http` (the server then keeps
a per-channel cache model; see `cnew`/`cid` below).  The HTTP/3 path
is stateless only.

```
open_htj2k_jpip_server <input.j2c>
    [--port N=8080]
    [--h3 --cert server.cert --key server.key]   # HTTP/3 (requires -DOPENHTJ2K_QUIC=ON)
    [--chunked]                                  # opt in to Transfer-Encoding: chunked
    [--no-chunked]                               # accepted for back-compat (now the default)
```

> **Warning** — `--chunked` is for the browser/WASM streaming demos
> only.  Some interactive reference JPIP clients (native desktop
> viewers) cannot parse chunked responses and fail with "connection
> closed unexpectedly"; serve those clients with the default
> `Content-Length` mode.  See
> [Progressive (chunked) delivery](#progressive-chunked-delivery).

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
  fills the response up to `N` bytes (EOR does not count) and
  terminates with EOR reason=4 (`ByteLimit`) when it stops early,
  or reason=2 (`WindowDone`) when the entire view-window fit.
  Delivery is resumable at byte granularity: a data-bin larger than
  the remaining budget contributes a prefix now and continues from
  that byte offset on the next request (tracked per-session, or via
  the `model` field's `:bytes` qualifier in stateless mode), so
  byte-limited clients always make forward progress.  `len=0` is
  legal and yields an EOR-only response.
- `model` — §C.9 cache-model advertisement; see
  [cache-model field](dev/jpip_internals.md#cache-model-field-c9).
- `cnew=<transports>` — §C.3.3 session/channel establishment.  When
  the comma-separated list contains `http` (the only transport this
  server grants), the reply carries `JPIP-cnew:
  cid=<token>,path=jpip,transport=http` and the server creates a
  session: a per-channel cache model remembers every data-bin (and
  partial-bin byte offset) delivered on that `cid`, so follow-up
  requests — which session clients send *without* a `model` field —
  never receive the same bytes twice (§B.2).  Once a window
  completes, repeating the request returns an empty body with EOR
  reason=2, as Table D.2 requires.  If no offered transport is
  supported (e.g. `cnew=http-tcp` alone), no `JPIP-cnew` header is
  returned and the request is served statelessly per §C.3.3 —
  never a transport the client did not offer (Table D.1).
- `cid=<token>` — request within an existing channel.  Unknown or
  expired cids get HTTP 404 so the client re-establishes rather
  than looping on stateless duplicates.  Channels are evicted LRU
  beyond 64 concurrent sessions.
- `cclose=*` (or a cid list) — §C.3.4 channel close, honoured after
  the response completes.
- `qid=<N>` — §C.3.5 request ID; echoed back verbatim in a
  `JPIP-qid` response header (§D.2.4).  Session responses also
  carry `Cache-Control: no-cache`.

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
expect is absent — a session that otherwise negotiates perfectly
(`JPIP-cnew` granted, image data flowing) fails at the HTTP layer with
no JPIP-level error to point at the cause.  Use the default mode for
native desktop viewers; reserve `--chunked` for the browser demos and
`JpipClient`, which accept either format.  The client-side chunked
refactor that enabled the TTFB win stays useful wherever the server
is started with `--chunked`.  Opt in with:

```
open_htj2k_jpip_server <input.j2c> --chunked
```

The §C.6.1 `len=` cap works identically in chunked mode — data-bins
are emitted through budget-aware byte windows, so each wire chunk is
already guaranteed to fit under the cap when it is flushed, and a bin
interrupted by the cap resumes from its recorded byte offset on the
next request.  HTTP/3 continues to use the
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
| `precinctCacheMB=N` | LRU precinct cache budget in megabytes (v0.18.0); default `64`, pass `0` to disable. Received precincts are tracked in `&model=` so the server skips redelivery on subsequent pans (70–95% byte reduction typical). |
| `prefetchMargin=N` | Adjacent-viewport halo prefetch (v0.18.1) — fetches precincts up to `N` canvas pixels outside the visible viewport so short follow-up pans hit in the LRU cache. Default `128`, pass `0` to disable. Cancelled on the next user interaction so continuous panning never pays bandwidth for it. |
| `prefetchDelayMs=N` | Idle delay in ms before the halo prefetch fires. Default `150`. |
| `midPaintMs=N` | Mid-decode paint (v0.18.2) — on slow fetches, render the partial DataBinSet after `N` ms so the user sees progress before the full response arrives. Default `200`; `0` disables. Fast fetches (LAN / cached) never trigger it. |

Pan events are debounced + coalesced: during an in-flight fetch, new
events flip a "pending" slot rather than queue a second request, so
the final viewport the user aimed at is always what lands on screen.
Fetch responses are streamed into the WASM `jpip_feed_stream*` API
per chunk.  Against a `--chunked` server this matches the wire-level
chunked delivery; against the default Content-Length server the JS
side still avoids the full-buffer `arrayBuffer()` round-trip, it just
sees fewer/larger chunks from the browser.

## How it works

The core library's structure, the data-bin classes, how JPIP sequence numbers
and identifiers are derived, the zero-skip IDWT, viewport-region decoding,
the cache-model field (§C.9) and the WASM C API are described in
[dev/jpip_internals.md](dev/jpip_internals.md).

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
