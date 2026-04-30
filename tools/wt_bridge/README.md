# wt_bridge — UDP→WebTransport relay

Receives RFC 9828 RTP packets on a UDP socket and fans them out to every
connected WebTransport client. No HTJ2K parsing — the browser viewer's
WASM decoder owns that.

Each connected viewer gets one server-initiated unidirectional WebTransport
stream. Packets are written length-prefixed: `[u16BE len][packet bytes]…`.
Streams (not datagrams) because Chromium's negotiated WebTransport datagram
size caps at ~1170 B, below typical RFC 9828 packet sizes (~1400 B). On LAN
the head-of-line cost vs datagrams is negligible.

## Quick start (dev mode)

```sh
go build -o wt_bridge .
./wt_bridge --listen-udp 0.0.0.0:6000 --listen-quic 0.0.0.0:4433 --dev
```

`--dev` generates an ephemeral self-signed ECDSA P-256 certificate (13-day
validity, key usage = digitalSignature, EKU = serverAuth) and prints its
SHA-256 hash to stderr:

```
[wt_bridge] dev cert SHA-256:
[wt_bridge]   ab:cd:…
[wt_bridge] viewer URL hint: ?certHash=ab:cd:…
```

Paste that hash into the viewer's `?certHash=…` URL parameter — Chromium's
`serverCertificateHashes` API trusts it for that one connection without a
public CA. Cert constraints come from the WebTransport spec: ECDSA-P256,
validity strictly less than two weeks.

Producer side: point rpicam-apps at the bridge's UDP port:

```sh
rpicam-vid --rtp-host <bridge-ip> --rtp-port 6000 …
```

For dev without rpicam-apps, replay a captured `.rtp` fixture:

```sh
node scripts/udp_replay.mjs <fixture.rtp> --port 6000 --fps 30 --loop
```

## CLI

```
wt_bridge [flags]

  --listen-udp   <addr>   UDP bind for incoming RTP                  [0.0.0.0:6000]
  --listen-quic  <addr>   QUIC bind for outgoing WebTransport        [0.0.0.0:4433]
  --max-clients  <N>      Max concurrent WT sessions                 [8]
  --queue-depth  <N>      Per-session packet queue depth             [8192]
                          Drop-oldest on overrun.
  --cert <path>           PEM cert chain (production; not yet wired)
  --key  <path>           PEM private key (production; not yet wired)
  --dev                   Ephemeral self-signed cert + hash printout
```

## Wire contract

- UDP in: arbitrary RFC 9828 packets, one packet per UDP datagram. The bridge
  does not validate the payload.
- WebTransport out, per session: one server-initiated unidirectional stream.
  Repeated framing of `[len:u16BE][bytes×len]`. Reading client must parse this
  framing.

## Build

```sh
go build -o wt_bridge .                                       # native (linux/amd64)
GOOS=linux GOARCH=arm64 go build -o wt_bridge_arm64 .         # cross to Pi etc.
```

No `cgo`, no architecture-specific code. Single static binary; ~10 MB stripped.

## Operational notes

- Bump `net.core.rmem_max` to ~8 MiB on the bridge host so `SO_RCVBUF`
  isn't kernel-clamped (quic-go logs this if it can't apply the requested
  size). The relay is intended to share a host with the producer or sit on
  a small LAN node — host-level tuning is fine.
- The bridge listens on QUIC + HTTP/3; CORS is wide open in dev mode.
  Tighten `CheckOrigin` and lock down the cert path for any non-LAN
  deployment (Phase C).
