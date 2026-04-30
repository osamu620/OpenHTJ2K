#!/bin/bash
# Bring up the WebTransport viewer stack for LAN testing against a real
# RFC 9828 producer (rpicam-apps).  Prints the LAN IP, cert hash, and
# ready-to-paste browser URL on stdout, then runs in the foreground until
# Ctrl-C.
#
# Layout:
#   This host  → wt_bridge (UDP 6000  ← Pi)  +  static server (HTTP 8765)
#   Pi         → rpicam-vid --rtp-host <this-host-ip> --rtp-port 6000 …
#   Browser    → http://<this-host-ip>:8765/viewer/?url=…&certHash=…
set -e
cd "$(dirname "$0")/../../.."     # repo root

# Pick the first non-loopback IPv4 the host owns.  Override with $LAN_IP.
LAN_IP=${LAN_IP:-$(ip -4 addr show | awk '/inet / && $2 !~ /127\./ {print $2}' | head -1 | cut -d/ -f1)}
[ -n "$LAN_IP" ] || { echo "could not detect a non-loopback IPv4 — set \$LAN_IP" >&2; exit 1; }

UDP_PORT=${UDP_PORT:-6000}
QUIC_PORT=${QUIC_PORT:-4433}
HTTP_PORT=${HTTP_PORT:-8765}

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

# Static server bound to all interfaces (-bind), serves /viewer/, /wasm/, /perf/.
node web/perf/serve.mjs "$HTTP_PORT" --bind > /tmp/wtb_lan_serve.log 2>&1 &
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

# URL-encode the QUIC URL for the browser query string.
QURL_ENC=$(python3 -c "import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1],safe=''))" "https://${LAN_IP}:${QUIC_PORT}/")

QURL_LOCAL_ENC=$(python3 -c "import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1],safe=''))" "https://${LAN_IP}:${QUIC_PORT}/")

cat <<EOF

========================================================================
 Stack is up.

 Bridge UDP listener:  0.0.0.0:${UDP_PORT}        (point Pi producer here)
 Bridge QUIC listener: 0.0.0.0:${QUIC_PORT}
 Static server:        http://0.0.0.0:${HTTP_PORT}/viewer/
 Cert SHA-256:
   ${HASH}

 ── Pi side ────────────────────────────────────────────────────────────
   rpicam-vid \\
       --rtp-host ${LAN_IP} \\
       --rtp-port ${UDP_PORT} \\
       --rtp-prims 1 --rtp-trans 13 --rtp-mat 5 --rtp-range 0 \\
       --width 1920 --height 1080 --framerate 30 --inline \\
       --output -

 ── Browser, on THIS host (recommended) ────────────────────────────────
   Chrome here:

     http://localhost:${HTTP_PORT}/viewer/?autorun=1&url=${QURL_LOCAL_ENC}&certHash=${HASH}

 ── Browser, on another LAN device (needs a flag) ──────────────────────
   WebTransport requires a secure context.  http://localhost is one;
   http://${LAN_IP} is not.  Two workarounds for testing:

   1. Launch Chrome with the LAN origin whitelisted (paste in another shell):
        google-chrome \\
            --user-data-dir=/tmp/chrome-wt-lan \\
            --unsafely-treat-insecure-origin-as-secure="http://${LAN_IP}:${HTTP_PORT}" \\
            "http://${LAN_IP}:${HTTP_PORT}/viewer/?autorun=1&url=${QURL_ENC}&certHash=${HASH}"

   2. Or run Chrome with chrome://flags/#unsafely-treat-insecure-origin-as-secure
      and add http://${LAN_IP}:${HTTP_PORT} to the list.

 Logs:
   tail -f /tmp/wtb_lan_bridge.log
   tail -f /tmp/wtb_lan_serve.log

 Ctrl-C to stop.
========================================================================

EOF

# Tail the bridge log so the user sees session accept / per-thousand counters live.
tail -f /tmp/wtb_lan_bridge.log
