#!/bin/bash
# Bring up the WebTransport viewer stack for LAN testing against any
# RFC 9828 sender (rpicam-apps fork, kdu_stream_send, the in-repo
# udp_replay.mjs replayer, …).  Prints the LAN IP, cert hash, and
# ready-to-paste browser URL on stdout, then runs in the foreground
# until Ctrl-C.
#
# Layout:
#   This host  → wt_bridge (UDP 6000 ← producer)  +  static server (HTTPS 8765)
#   Producer   → <RFC 9828 sender> --rtp-host <this-host-ip> --rtp-port 6000 …
#   Browser    → https://<this-host-ip>:8765/wt_viewer/?url=…&certHash=…
#
# HTTP_NO_TLS=1 falls back to plain HTTP (useful for very-local testing
# where openssl isn't available); WebTransport then only works from
# http://localhost on the bridge host itself.
set -e
cd "$(dirname "$0")/../../.."     # repo root

# Pick the first non-loopback IPv4 the host owns.  Override with $LAN_IP.
LAN_IP=${LAN_IP:-$(ip -4 addr show | awk '/inet / && $2 !~ /127\./ {print $2}' | head -1 | cut -d/ -f1)}
[ -n "$LAN_IP" ] || { echo "could not detect a non-loopback IPv4 — set \$LAN_IP" >&2; exit 1; }

UDP_PORT=${UDP_PORT:-6000}
QUIC_PORT=${QUIC_PORT:-4433}
HTTP_PORT=${HTTP_PORT:-8765}
CERT_DIR=${CERT_DIR:-/tmp/wt_static_cert}

# Make sure the WT bridge is built.
[ -x ./tools/wt_bridge/wt_bridge ] || (
  command -v go >/dev/null || export PATH=$HOME/go-toolchain/bin:$PATH
  cd tools/wt_bridge && go build -o wt_bridge .
)

# Pre-cleanup so re-running doesn't EADDRINUSE.  Use -x with the binary's
# basename so the script doesn't kill itself (its argv[0] also contains
# "wt_bridge" via the directory path).
pkill -f "node web/perf/serve" 2>/dev/null || true
pkill -x "wt_bridge"           2>/dev/null || true
sleep 0.5

trap 'echo; echo "[run_lan] shutting down"; pkill -P $$ 2>/dev/null; exit 0' INT TERM

# Generate or refresh the static-server cert (HTTPS).  Skipped when
# HTTP_NO_TLS=1; in that case the static server runs HTTP-only and only a
# browser at http://localhost:HTTP_PORT can use WebTransport (cross-LAN
# browsers will see `WebTransport is undefined` due to the secure-context
# requirement).
if [ "${HTTP_NO_TLS:-}" = "1" ] || ! command -v openssl >/dev/null; then
  SCHEME="http"
  SERVER_TLS_ARGS=()
else
  ./tools/wt_bridge/scripts/gen_static_cert.sh "$CERT_DIR" "$LAN_IP"
  SCHEME="https"
  SERVER_TLS_ARGS=(--cert "$CERT_DIR/cert.pem" --key "$CERT_DIR/key.pem")
fi

# Static server bound to all interfaces (--bind), serves /wt_viewer/, /wasm/,
# /perf/.  HTTPS when a cert was generated above.
node web/perf/serve.mjs "$HTTP_PORT" --bind "${SERVER_TLS_ARGS[@]}" \
  > /tmp/wtb_lan_serve.log 2>&1 &
SERVER_PID=$!
sleep 0.3

# Bridge bound to all interfaces.
./tools/wt_bridge/wt_bridge \
  --listen-udp  "0.0.0.0:${UDP_PORT}" \
  --listen-quic "0.0.0.0:${QUIC_PORT}" \
  --dev > /tmp/wtb_lan_bridge.log 2>&1 &
BRIDGE_PID=$!
sleep 1.0

HASH=$(grep "viewer URL hint" /tmp/wtb_lan_bridge.log | sed -E 's/.*\?certHash=//' | tr -d '\r\n')
[ -n "$HASH" ] || { echo "bridge failed to start; see /tmp/wtb_lan_bridge.log" >&2; cat /tmp/wtb_lan_bridge.log >&2; exit 2; }

# URL-encode the WebTransport URL for the browser query string.
QURL_ENC=$(python3 -c "import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1],safe=''))" "https://${LAN_IP}:${QUIC_PORT}/")

if [ "$SCHEME" = "https" ]; then
  PAGE_URL_LAN="https://${LAN_IP}:${HTTP_PORT}/wt_viewer/?autorun=1&url=${QURL_ENC}&certHash=${HASH}"
  cat <<EOF

========================================================================
 Stack is up.

 Bridge UDP listener:  0.0.0.0:${UDP_PORT}        (point your RFC 9828 sender here)
 Bridge QUIC listener: 0.0.0.0:${QUIC_PORT}
 Static server:        ${SCHEME}://0.0.0.0:${HTTP_PORT}/wt_viewer/
 Cert SHA-256 (WebTransport):
   ${HASH}
 Static-server cert: $CERT_DIR/cert.pem  (self-signed; click through once)

 ── Producer side ──────────────────────────────────────────────────────
 Point any RFC 9828 sender at ${LAN_IP}:${UDP_PORT}.  One example
 (rpicam-apps HTJ2K fork running on a Pi):

   rpicam-vid \\
       --rtp-host ${LAN_IP} \\
       --rtp-port ${UDP_PORT} \\
       --rtp-prims 1 --rtp-trans 13 --rtp-mat 5 --rtp-range 0 \\
       --width 1920 --height 1080 --framerate 30 --inline \\
       --output -

 ── Browser (any LAN device) ───────────────────────────────────────────
   ${PAGE_URL_LAN}

   First load: Chrome shows "Your connection is not private" because
   the static server's cert is self-signed.  Click "Advanced → Proceed
   to ${LAN_IP} (unsafe)".  The browser remembers the decision per-cert
   for ~13 days; subsequent loads are silent.  WebTransport itself
   does NOT trigger this prompt — its cert is hash-pinned.

 Logs:
   tail -f /tmp/wtb_lan_bridge.log
   tail -f /tmp/wtb_lan_serve.log

 Ctrl-C to stop.
========================================================================

EOF
else
  PAGE_URL_LOCAL="http://localhost:${HTTP_PORT}/wt_viewer/?autorun=1&url=${QURL_ENC}&certHash=${HASH}"
  PAGE_URL_LAN="http://${LAN_IP}:${HTTP_PORT}/wt_viewer/?autorun=1&url=${QURL_ENC}&certHash=${HASH}"
  cat <<EOF

========================================================================
 Stack is up — HTTP mode (HTTP_NO_TLS=1 or openssl unavailable).

 Bridge UDP listener:  0.0.0.0:${UDP_PORT}        (point your RFC 9828 sender here)
 Bridge QUIC listener: 0.0.0.0:${QUIC_PORT}
 Static server:        http://0.0.0.0:${HTTP_PORT}/wt_viewer/
 Cert SHA-256:
   ${HASH}

 ── Producer side ──────────────────────────────────────────────────────
 Point any RFC 9828 sender at ${LAN_IP}:${UDP_PORT}.  One example
 (rpicam-apps HTJ2K fork running on a Pi):

   rpicam-vid --rtp-host ${LAN_IP} --rtp-port ${UDP_PORT} \\
       --rtp-prims 1 --rtp-trans 13 --rtp-mat 5 --rtp-range 0 \\
       --width 1920 --height 1080 --framerate 30 --inline --output -

 ── Browser, on THIS host (recommended) ────────────────────────────────
   ${PAGE_URL_LOCAL}

 ── Browser, on another LAN device (needs a Chrome flag) ───────────────
   google-chrome \\
       --user-data-dir=/tmp/chrome-wt-lan \\
       --unsafely-treat-insecure-origin-as-secure="http://${LAN_IP}:${HTTP_PORT}" \\
       "${PAGE_URL_LAN}"

 Logs:
   tail -f /tmp/wtb_lan_bridge.log
   tail -f /tmp/wtb_lan_serve.log

 Ctrl-C to stop.
========================================================================

EOF
fi

# Tail the bridge log so the user sees session accept / per-thousand counters live.
tail -f /tmp/wtb_lan_bridge.log
