# WebTransport viewer (experimental)

A browser-native viewer for live HTJ2K streams. Receives an RFC 9828
RTP/UDP feed via a small relay (`wt_bridge`), forwards it as a
WebTransport stream to a Chromium-based browser, and decodes through
the existing WebAssembly decoder.

Any RFC 9828 compliant sender works on the producer side — Kakadu's
`kdu_stream_send`, a Raspberry Pi running an HTJ2K-capable rpicam-apps
fork, the in-repo `udp_replay.mjs` replayer for captured `.rtp` files,
or any custom packetizer that follows the RFC. The bridge does not
parse the payload.

This is the LAN companion to [`open_htj2k_rtp_recv`](cli_rtp_recv.md):
same wire format on the producer side, same WASM decoder on the receive
side, but no native binary required on the viewing host.

The whole pipeline is experimental — wire format and CLI defaults may
change.

```
┌────────────┐ RFC 9828 RTP/UDP ┌────────────┐ WebTransport ┌─────────┐
│  producer  │ ───────────────▶ │ wt_bridge  │ ───────────▶ │ browser │
│ (any 9828) │                  │   (Go)     │  (uni-stream)│ (WASM)  │
└────────────┘                  └────────────┘              └─────────┘
                                                                  ▲
                                                          HTTPS (static)
                                                                  │
                                                         ┌────────┴───────┐
                                                         │ web/perf/serve │
                                                         │   (viewer +    │
                                                         │    WASM)       │
                                                         └────────────────┘
```

## Prerequisites

| Component | Required for | Minimum version | Notes |
|---|---|---|---|
| **Go** | Building `wt_bridge` | **1.22** | Pinned by `tools/wt_bridge/go.mod`. `go build` will refuse older toolchains. |
| **Node.js** | Static server (`web/perf/serve.mjs`), fixture replayer (`udp_replay.mjs`) | **18** | Modern ESM + WebCrypto APIs are used. |
| **Emscripten** | WASM artefacts under `web/build_wt/` | 3.x or 5.x | See [building.md → Building for WebAssembly](building.md#building-for-webassembly-wasm). |
| **OpenSSL** | HTTPS dev cert for the static server | any 1.1+ | Required by `gen_static_cert.sh`. Skip with `HTTP_NO_TLS=1` (then WebTransport works only from `http://localhost` on the bridge host). |
| **Python 3** | URL-encoding inside `run_lan.sh` | 3.6+ | Any system `python3` works. |
| **Chromium-based browser** | Viewer | latest stable | Firefox WebTransport is partial / behind a flag; Safari has no implementation. See [Browser support](#caveats-and-constraints). |
| `iproute2` (`ip` command) | LAN-IP autodetect in `run_lan.sh` | — | Linux only. On macOS / BSD, set `LAN_IP=<addr>` before invoking the launcher. |

## Components

- [`tools/wt_bridge/`](../tools/wt_bridge/) — Go relay. Binds a UDP
  socket for the producer, accepts WebTransport sessions from viewers,
  forwards every UDP datagram as one length-prefixed message on a
  server-initiated unidirectional stream.
- [`web/wt_viewer/index.html`](../web/wt_viewer/index.html) — single-file
  browser viewer. Opens the WebTransport session, parses the framing,
  feeds packets into the WASM `rtp_session_*` API exported by
  [`web/src/wrapper.cpp`](../web/src/wrapper.cpp),
  decodes via `mt_simd` build with worker threads, renders to Canvas2D
  or WebGL2.
- [`web/perf/serve.mjs`](../web/perf/serve.mjs) — minimal HTTP/HTTPS
  server with the COOP/COEP headers required for `SharedArrayBuffer`
  (multi-threaded WASM). HTTPS is the default in the LAN launcher and
  removes WebTransport's secure-context restriction for cross-LAN
  viewers. Also serves the WASM artefacts under `/wasm/` and any
  `.rtp` fixture under `/fixtures/`.

## Quick start (LAN)

The launcher does the boring parts (build the bridge if needed, bind
to all interfaces, generate the dev certificate, print the connection
details):

```bash
./tools/wt_bridge/scripts/run_lan.sh
```

It prints something like:

```
 Bridge UDP listener:  0.0.0.0:6000        (point your RFC 9828 sender here)
 Bridge QUIC listener: 0.0.0.0:4433
 Static server:        https://0.0.0.0:8765/wt_viewer/
 Cert SHA-256 (WebTransport):
   ab:cd:ef:…
 Static-server cert: /tmp/wt_static_cert/cert.pem  (self-signed; click through once)

 ── Producer side ──────────────────────────────────────────────────────
 Point any RFC 9828 sender at 192.168.0.14:6000.  One example
 (rpicam-apps HTJ2K fork running on a Pi):

   rpicam-vid \
       --rtp-host 192.168.0.14 \
       --rtp-port 6000 \
       --rtp-prims 1 --rtp-trans 13 --rtp-mat 5 --rtp-range 0 \
       --width 1920 --height 1080 --framerate 30 --inline \
       --output -

 ── Browser (any LAN device) ───────────────────────────────────────────
   https://192.168.0.14:8765/wt_viewer/?autorun=1&url=…&certHash=ab:cd:…

   First load: Chrome shows "Your connection is not private" because
   the static server's cert is self-signed.  Click "Advanced → Proceed".
```

Point any RFC 9828 sender at `<bridge-host>:6000`, then open the
printed URL in Chromium on any host on the LAN. With no producer
running yet, use the in-repo replayer for a quick smoke:

```bash
node tools/wt_bridge/scripts/udp_replay.mjs \
    tools/wt_bridge/fixtures/1080p2997_30frames.rtp \
    --port 6000 --fps 30 --loop
```

The bridge log tails to stdout — you'll see `session accepted` when the
browser connects, then `session N forwarded=1000/2000/…` as packets
flow.

## Building the bridge

The bridge is a self-contained Go module. Build once:

```bash
cd tools/wt_bridge
go build -o wt_bridge .
```

Binary lands at `tools/wt_bridge/wt_bridge` (gitignored). No `cgo`, no
architecture-specific code. Cross-compile to ARM64 with
`GOOS=linux GOARCH=arm64 go build -o wt_bridge_arm64 .`.

The WASM artefacts are produced by the existing Emscripten build under
[`web/`](../web/) — the launcher expects
`web/build_wt/html/libopen_htj2k_mt_simd.{js,wasm}`. Build with:

```bash
cd web && rm -rf build_wt && mkdir build_wt && cd build_wt
emcmake cmake ..
cmake --build . -j -t libopen_htj2k_simd libopen_htj2k_mt_simd
```

## `wt_bridge` options

`wt_bridge -h` prints the full reference; the highlights:

- `--listen-udp <host:port>` — UDP bind for the producer. Default
  `0.0.0.0:6000`.
- `--listen-quic <host:port>` — QUIC bind for incoming WebTransport
  sessions. Default `0.0.0.0:4433`.
- `--max-clients <N>` — Concurrent viewer cap. Default `8`.
- `--queue-depth <N>` — Per-session packet queue. Drop-oldest on
  overrun. Default `8192` (sized for ~1 s at 30 fps × 200 packets/frame).
- `--dev` — Generate an ephemeral ECDSA-P256 certificate (13-day
  validity, `digitalSignature` + `serverAuth`) at startup and print its
  SHA-256 hash to stderr.
- `--cert <path> --key <path>` — Use a real PEM certificate chain
  instead of `--dev`. With a CA-issued cert, browsers validate
  normally and viewers don't need a `certHash` query parameter.
  Operational glue (auto-rotation, ACME integration) is a planned
  follow-up — for now, refresh manually before expiry.

The dev hash is what the browser pins via the WebTransport
`serverCertificateHashes` API — the cert otherwise wouldn't validate
because it's self-signed and the SAN list (`localhost`, `127.0.0.1`,
`::1`) isn't a real CA-issued name. Per W3C WebTransport, hash-pinning
bypasses the SAN/chain check entirely.

The bridge sends each UDP datagram as `[u16BE len][packet bytes]` on a
single server-initiated unidirectional stream. The viewer parses this
framing. Streams (not datagrams) because Chromium negotiates a
WebTransport `max_datagram_frame_size` of about 1170 B which is below
typical RFC 9828 packet sizes; on LAN the head-of-line cost vs
datagrams is negligible.

## Viewer URL parameters

The browser viewer is configured entirely via query string. The header
controls (URL, certHash, Connect/Stop) are also wired and editable.

- `url=<wt-url>` — WebTransport endpoint, e.g.
  `https://localhost:4433/`. Pre-fills the URL field.
- `certHash=<colon-separated-hex>` — SHA-256 of the bridge's dev cert.
  The launcher prints the exact value to paste.
- `autorun=1` — Click `Connect` automatically once the page loads.
- `renderer={webgl2|canvas2d}` — Force a renderer. Default tries WebGL2
  first; falls back to Canvas2D if WebGL2 init fails (e.g. browsers
  without GPU acceleration).
- `threads=<N>` — Number of WASM decoder workers. Default `4`. Use
  fewer (`2`) on resource-constrained hosts; more rarely helps because
  the HT block coder saturates around two threads.
- `debug=1` — Show a translucent overlay on the canvas with FPS,
  decode `p50/p95`, queue/drop counters, sequence-gap count, and RTP-
  vs-wall drift.
- `source_fps=<N>` — Declared source frame rate. Used only to flag the
  "decode-bound" status when rolling decode `p95` exceeds the source
  frame interval. Default `30`.
- `pace={live|rtp}` — Frame-display strategy. Default `live`: render the
  most-recent decoded frame at every rAF tick (lowest latency, frames
  silently overwritten if the producer outpaces vsync). `rtp` queues
  frames in a small ring and times each render against its
  RTP-timestamp-relative wall time after a brief pre-roll anchor —
  smoother under network jitter at the cost of ~1 frame-period extra
  latency. The stats line gains a `pace lag <ms> drops <N>` suffix in
  `rtp` mode (`pace lag` is the most-recent `rafTimestamp − target`;
  `drops` counts frames discarded on ring overflow).
- `preroll=<N>` — In `pace=rtp` mode, render the first `N` decoded
  frames immediately before taking the RTP-clock anchor. Defaults to
  `1`; raise it (e.g. `?pace=rtp&preroll=3`) to absorb a longer initial
  network burst before locking in the wall-clock anchor.
- `report=<ms>` — Period in milliseconds to POST a JSON snapshot of the
  current stats to `/report`. Used by the headless smoke and benchmark
  scripts; leave unset for normal viewing.
- `reconnect=0` — Disable auto-reconnect. Default behaviour retries on
  session loss with capped exponential back-off; pass `0` for a
  one-shot connection (useful when debugging initial-connection
  failures that would otherwise loop).

## Architecture notes

**WebTransport, not WebRTC.** RTCPeerConnection's media pipeline is
codec-aware; HTJ2K isn't one of the codecs it understands. WebTransport
is codec-opaque, so RFC 9828 packets pass through unchanged.

**No TypeScript port of the RFC 9828 parser.**
[`web/src/wrapper.cpp`](../web/src/wrapper.cpp) already
exports the full `rtp_session_*` API (`rtp_session_create`,
`rtp_push_packet`, `rtp_peek_frame_size`, `rtp_pop_frame`, plus the
H.273 metadata accessors and decoder-reuse helpers) via
`EMSCRIPTEN_KEEPALIVE`. The viewer just `cwrap`s them.

**Render decoupled from decode.** Decoding uploads textures and shader
uniforms but does not call `gl.drawArrays`. A separate
`requestAnimationFrame` loop calls `draw()` once per vsync. This locks
display cadence to monitor refresh and absorbs the per-frame jitter
that decode-then-draw would otherwise produce as visible flicker. The
cost is one frame of display latency (≤16.7 ms at 60 Hz).

**Drop-on-overrun.** Two layers absorb the case where decode is slower
than the source frame rate:

1. The C++ `rtp_session` ready queue caps at 2 frames; a third
   completed frame evicts the oldest.
2. Before each decode the JS layer also calls `rtp_drop_ready` while
   the queue holds more than one frame, so the decoder always works on
   the latest codestream available.

The overlay's `decode-bound` flag fires when rolling decode `p95`
exceeds the source frame interval; that's the signal to lower
resolution or frame rate at the producer.

## Caveats and constraints

**Secure-context requirement.** Browsers expose the WebTransport API
only on secure contexts. `http://localhost` qualifies; `http://<LAN-IP>`
does not. The launcher (`run_lan.sh`) handles this by serving the page
over HTTPS using a short-lived self-signed certificate
(`tools/wt_bridge/scripts/gen_static_cert.sh`). Browsers will show
"Your connection is not private" on first load — click
"Advanced → Proceed". The decision is remembered per-cert for ~13 days,
so subsequent loads are silent. The WebTransport session itself does not
trigger this prompt because its cert is hash-pinned.

The static server's `--cert/--key` flags can also be used standalone:

```bash
./tools/wt_bridge/scripts/gen_static_cert.sh /tmp/wt_static_cert <LAN_IP>
node web/perf/serve.mjs 8765 --bind \
    --cert /tmp/wt_static_cert/cert.pem \
    --key  /tmp/wt_static_cert/key.pem
```

Set `HTTP_NO_TLS=1` on the launcher to fall back to plain HTTP (only
useful when openssl isn't available); WebTransport then works only from
`http://localhost:<port>` on the bridge host itself, and cross-LAN
viewers need the
`--unsafely-treat-insecure-origin-as-secure="http://<LAN-IP>:<port>"`
Chrome flag.

**Browser support.** Chromium-based browsers only at the time of
writing. Firefox WebTransport is partial / behind a flag; Safari has
no implementation. For wider reach, a future fallback path (HLS or
WebSocket relay) will be needed; not in the current scope.

**Auto-reconnect.** When a WebTransport session ends due to a producer
reboot, network blip, or transient bridge restart, the viewer retries
transparently with capped exponential back-off (1, 2, 4, 8, 16, 30 s).
A session that lasted longer than ~10 s resets the back-off counter,
so a steady stream that occasionally hiccups recovers immediately. The
Stop button cancels any pending retry. Set `?reconnect=0` on the URL
to disable; useful when debugging an initial-connection failure that
would otherwise loop.

**4K is best-effort.** WASM decode of 4K@30 currently averages ~17 fps
in `mt_simd` with 4 threads on x86_64. The viewer drops cleanly down
to that ceiling; the producer should target FHD@30 for smooth playback.

**Kernel UDP buffer.** quic-go on Linux warns if it cannot raise
`SO_RCVBUF` past the kernel ceiling. Bump
`net.core.rmem_max` to ≥8 MiB on the bridge host to silence the warning
and keep packet loss low under burst:

```bash
sudo sysctl -w net.core.rmem_max=8388608
```

This is the same recommendation as in
[`docs/cli_rtp_recv.md`](cli_rtp_recv.md).

## Troubleshooting

**Browser shows nothing, no errors.** Most common cause: WebTransport
isn't available because the page isn't a secure context. Use HTTPS
(the launcher does so by default) or the Chrome flag workaround above.
The page checks for `WebTransport in window` and throws a visible
error if absent — open DevTools → Console.

**Chrome shows "Your connection is not private" and refuses to
proceed.** The static server's self-signed cert: click
"Advanced → Proceed to … (unsafe)". If the option doesn't appear, the
HSTS cache may have the host pinned — try a different port or run in
incognito.

**Bridge log: `failed to sufficiently increase receive buffer size`.**
The kernel clamped `SO_RCVBUF`. Raise `net.core.rmem_max`; see above.

**Bridge log: `session ended; forwarded=N dropped=M` with M ≫ 0.** The
viewer can't keep up with the network rate. Check the overlay's
`decode-bound` flag. Drop the producer's frame rate or resolution.

## Reproduction scripts

- [`tools/wt_bridge/scripts/run_lan.sh`](../tools/wt_bridge/scripts/run_lan.sh)
  — interactive LAN launcher.
- [`tools/wt_bridge/scripts/e2e_smoke.sh`](../tools/wt_bridge/scripts/e2e_smoke.sh)
  — headless Chromium end-to-end smoke against an `.rtp` fixture.
- [`tools/wt_bridge/scripts/udp_replay.mjs`](../tools/wt_bridge/scripts/udp_replay.mjs)
  — paced fixture replayer; useful when no producer is reachable.

The smoke script captures decode/loss telemetry via the viewer's
`?report=N` POST channel, so it can verify end-to-end correctness in
CI without screen-scraping.

## See also

- [cli_rtp_recv.md](cli_rtp_recv.md) — the native equivalent of this
  viewer, for desktop / native-tooling use cases.
- [building.md](building.md) — Emscripten/WASM build instructions.
- [RFC 9828](https://datatracker.ietf.org/doc/rfc9828/) — wire format
  contract.
