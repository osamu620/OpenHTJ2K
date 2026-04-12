# `open_htj2k_rtp_recv` — RFC 9828 RTP receiver (experimental)

Receives a live HTJ2K RTP stream per
[RFC 9828](https://datatracker.ietf.org/doc/rfc9828/), reassembles
frames, decodes through the line-based streaming decoder, and
displays the result in a letterboxed GLFW window (or dumps
codestreams to disk). The CLI and defaults are still experimental
and may change.

This binary is opt-in at build time; see
[building.md](building.md#building-the-experimental-rfc-9828-rtp-receiver).

Full runtime help: `open_htj2k_rtp_recv -h` (or `--help`).

## Synopsis

```bash
# Start the receiver (default bind 0.0.0.0:6000)
open_htj2k_rtp_recv --colorspace bt709 --range full
```

Run the sender from another host. The receiver prints running FPS
once per second; close with `Esc`, `Q`, or the window close button.
The exit summary reports frame/byte counts, decode timing
(min/avg/max), and per-slot eviction counters.

The receiver consumes any RFC 9828 compliant sender. A Python
loopback helper at
`source/apps/rtp_recv/tools/rtp_loopback_send.py` wraps a single
codestream as one Main Packet and sends it over UDP loopback for
quick local testing without a live sender.

## Options

### Binding and lifetime

- `--port <N>` — UDP port to bind. Default `6000`.
- `--bind <host>` — Bind address. Default `0.0.0.0`.
- `--frames <N>` — Exit after `N` successfully decoded frames.
  `0` = unlimited.

### Pipeline selection

- `--no-render` — Headless; depacketize + decode only, no GLFW window.
- `--no-vsync` — Immediate swap instead of display-locked swap.
  Combine with `--pace-fps` for smooth motion.
- `--no-decode` — Capture-only; skip the HTJ2K decoder entirely.
- `--threading {on,off}` — Multi-threaded pipeline. Default `on`;
  `off` falls back to a single-threaded loop.
- `--color-path {shader|cpu}` — YCbCr→RGB via a GL 3.3 fragment
  shader (default) or the AVX2 CPU path. Auto-forced to `cpu` if a
  GL 3.3 core context cannot be created.

### HDR colour pipeline (shader path only)

After the YCbCr→RGB matrix, the fragment shader runs an inverse
transfer (EOTF), a linear-light gamut matrix, hard clipping, and a
display-encoding stage. The three switches below pick each stage.

- `--transfer {auto|gamma|pq|hlg}` — Inverse EOTF applied to the
  post-matrix non-linear R′G′B′. Default `auto` reads the Main Packet
  `TRANS` field per ITU-T H.273 Table 3:
  - `TRANS = 1, 6, 14, 15` → `gamma` (BT.709 / BT.601 / BT.2020 NCL)
  - `TRANS = 16` → `pq` (SMPTE ST 2084)
  - `TRANS = 18` → `hlg` (ARIB STD-B67)
  - Any other value falls through to the CLI fallback (default `gamma`).
- `--display-primaries {bt709|bt2020}` — Target primaries for the
  linear-light gamut matrix. Default `bt709`. `bt2020` is an identity
  stub for a future HDR output path. The matrix is identity unless the
  source primaries are BT.2020 (H.273 `PRIMS = 9` under S=1 or the CLI
  `--colorspace bt2020` fallback) and the display primaries are not.
- `--display-encoding {srgb|gamma22|linear}` — Final non-linear
  encoding written to the framebuffer. Default `srgb` (IEC 61966-2-1
  piecewise). `gamma22` is a cheaper inverse of the default `gamma`
  transfer and, combined with `--transfer gamma`, is a bit-identical
  round-trip for SDR BT.709 sources. `linear` writes linear light
  directly and is diagnostic only.

The default pipeline for an SDR BT.709 source is
`--transfer auto` → `gamma` → `--display-primaries bt709` (identity)
→ `--display-encoding srgb`. This differs from the v0.12.0 shader,
which wrote the non-linear R′G′B′ directly to the framebuffer: both
targets the same display light through the monitor's own gamma, so
the two paths are visually indistinguishable. Byte values diverge
slightly (max ~9/255 in the deep-grey region); pass
`--display-encoding gamma22` for a bit-identical round-trip.

**Tone mapping** is currently a hard `clamp(rgb, 0, 1)` in linear
light, which is correct for any source below the display peak
(including all HLG content treated as display-referred) but clips
highlights on above-peak PQ content. The ITU-R BT.2390 EETF soft-knee
curve is a planned follow-up.

### Pacing and throughput

- `--pace-fps <N>` — Frame-pacing target, default `30`, `0`
  disables. Active only with `--no-vsync`. Uses RTP timestamp deltas
  when available.
- `--threads <N>` — Decoder thread count. Default `4`, optimal on
  4K with component-parallel IDWT dispatch (v0.13.0+).

### Color fallback (when the Main Packet declares S=0)

- `--colorspace {bt709|bt601|bt2020|rgb}` — Fallback colorspace.
  `bt2020` selects the BT.2020 NCL matrix (ITU-T H.273 MatrixCoefficients = 9)
  and also feeds the BT.2020 → BT.709 gamut matrix when the display
  primaries stay at the default `bt709`. The inverse transfer and
  display encoding come from the `--transfer` / `--display-encoding`
  switches above; pair with `--transfer pq` (or `hlg`) for a BT.2020
  HDR source served under S=0.
- `--range {full|narrow}` — Fallback range. Default `full`.

### Diagnostics

- `--dump-codestream <fmt>` — printf-style path, e.g.
  `/tmp/frame_%05d.j2c`. Writes each reassembled frame's codestream
  to disk for offline analysis.
- `--smoke-test` — Run built-in smoke tests and exit.

## Examples

```bash
# Default shader path, window + vsync
open_htj2k_rtp_recv

# --no-vsync with RTP-timestamp pacing (smoother on NVIDIA + Mutter)
open_htj2k_rtp_recv --no-vsync

# Headless capture + decode, exit after 1000 frames
open_htj2k_rtp_recv --no-render --frames 1000

# Capture and dump reassembled codestreams to /tmp
open_htj2k_rtp_recv --no-decode --frames 200 --dump-codestream /tmp/f_%05d.j2c

# Bit-identical with v0.12.0 on SDR BT.709 sources (gamma inverse + gamma22 encode)
open_htj2k_rtp_recv --transfer gamma --display-encoding gamma22

# Force a BT.2020 PQ source under S=0 (receiver has no TRANS/PRIMS to read)
open_htj2k_rtp_recv --colorspace bt2020 --transfer pq
```

## Kernel receive buffer

Live 4K HTJ2K easily exceeds Linux's default UDP receive buffer
(~200 KB). The receiver asks for a 32 MB buffer and warns if the
kernel grants less than 4 MB. Raise `net.core.rmem_max` before
running:

```bash
sudo sysctl -w net.core.rmem_max=33554432
```

To persist across reboots:

```bash
echo 'net.core.rmem_max = 33554432' | sudo tee /etc/sysctl.d/99-openhtj2k-rtp.conf
```

## Hardware requirements (4K @ 60 fps sustained)

Measured against a 4K 4:2:2 1.7-bpp broadcast HT fixture at
`--threads 4` on an AMD Ryzen 9 9950X running Linux. Reproduce with
the offline profiler at
`source/apps/rtp_recv/tools/rtp_decode_profile.cpp` (built as
`open_htj2k_rtp_decode_profile` when `-DOPENHTJ2K_RTP=ON`). Higher-
bitrate streams, 4:4:4 chroma, or deeper bit depths will not hit the
same numbers.

- **CPU**: recent high-end x86-64 with AVX2. HTJ2K decode is bounded
  by per-thread throughput — `--threads 4` (the default, v0.13.0+)
  is optimal on 4K with the component-parallel IDWT strip dispatch.
  The dev-box profiler peaks at ~95 fps (10.7 ms/frame) on the above
  fixture; the live `open_htj2k_rtp_recv --no-vsync` pipeline locks
  to the source cadence at 60 fps with zero decode-slot evictions,
  leaving comfortable headroom inside the 16.67 ms frame budget.
  Mid-range or older parts are unlikely to sustain 60 fps at 4K;
  non-AVX2 CPUs additionally fall back to the scalar YCbCr path and
  will not reach real-time.
- **GPU** (default `--color-path shader`): any integrated or discrete
  GPU with OpenGL 3.3 core. A modern IGP is ample; the YCbCr fragment
  shader is trivial.
- **Headless / no-GPU**: `--color-path cpu` uses the AVX2 color path
  on the same CPU class. Note that `--color-path cpu` does the full
  YCbCr→RGB matrix on the CPU (a different and heavier hot path than
  the shader path), so its ceiling is lower than the shader path's.
  It is auto-selected when GL 3.3 context creation fails, so the same
  binary runs on headless servers and in containers without X/Wayland.
- **Network**: LAN bandwidth for ~100 MB/s (broadcast 4K 4:2:2 HT at
  ~1.7 bpp, 60 fps); raise `net.core.rmem_max` as above.
- **Display**: with a 60 fps source and a 60 Hz display, the
  RTP-timestamp pacer naturally lands one present per vblank under
  `--no-vsync`. For 30 fps sources on a 60 Hz display, set
  `--pace-fps 30`. Other source rates need `--pace-fps` set
  accordingly (or `0` to rely purely on the RTP-timestamp pacer).

## Known issues

- **NVIDIA + Mutter + Wayland + vsync**: `glfwSwapInterval(1)` in
  fullscreen exhibits explicit-sync jitter (motion judder, title-bar
  wobble) on NVIDIA proprietary drivers. Workaround: `--no-vsync`
  and rely on the RTP-timestamp pacer. Tracks NVIDIA / Mutter
  explicit-sync upstream.
