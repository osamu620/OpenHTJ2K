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

### Pacing and throughput

- `--pace-fps <N>` — Frame-pacing target, default `30`, `0`
  disables. Active only with `--no-vsync`. Uses RTP timestamp deltas
  when available.
- `--threads <N>` — Decoder thread count. Default `2`, matches HT
  intra-frame parallelism saturation on 4K.

### Color fallback (when the Main Packet declares S=0)

- `--colorspace {bt709|bt601|bt2020|rgb}` — Fallback colorspace.
  `bt2020` selects the BT.2020 NCL matrix (ITU-T H.273 MatrixCoefficients = 9).
  Note that this slice only switches the YCbCr→RGB matrix; a PQ / HLG
  transfer function and gamut-mapping from BT.2020 primaries to the
  display primaries are not yet implemented, so a BT.2020 HDR source
  still renders with the display's native gamma curve.
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
`--threads 2` on an AMD Ryzen 9 9950X running Linux. Reproduce with
the offline profiler at
`source/apps/rtp_recv/tools/rtp_decode_profile.cpp` (built as
`open_htj2k_rtp_decode_profile` when `-DOPENHTJ2K_RTP=ON`). Higher-
bitrate streams, 4:4:4 chroma, or deeper bit depths will not hit the
same numbers.

- **CPU**: recent high-end x86-64 with AVX2. HTJ2K decode is bounded
  by per-thread throughput — `--threads 2` (the default) saturates HT
  intra-frame parallelism on 4K, so single-thread speed matters more
  than core count. The dev-box profiler peaks at ~80 fps on the above
  fixture; the live `open_htj2k_rtp_recv --no-vsync` pipeline locks
  to the source cadence at 60 fps with zero decode-slot evictions and
  ~13 ms average decode time, leaving ~3.5 ms of p99 headroom inside
  the 16.67 ms frame budget. Mid-range or older parts are unlikely
  to sustain 60 fps at 4K; non-AVX2 CPUs additionally fall back to
  the scalar YCbCr path and will not reach real-time.
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
