#!/bin/bash
# Split-host variant of run_lan.sh: run wt_bridge and the static server on
# separate machines so the UDP listener can sit on the same L2 subnet as the
# hardware encoder (no router hop for RTP packets).
#
# Usage:
#   # Machine A — same subnet as the RFC 9828 sender:
#   ./tools/wt_bridge/scripts/run_split.sh bridge
#
#   # Machine B — anywhere reachable from browsers:
#   BRIDGE_IP=<machine-A-ip> CERT_HASH=<hash> \
#       ./tools/wt_bridge/scripts/run_split.sh static
#
# Environment variables (all optional where noted):
#   LAN_IP      Override auto-detected local IPv4
#   UDP_PORT    Bridge UDP listen port           (default 6000)
#   QUIC_PORT   Bridge QUIC listen port          (default 4433)
#   HTTP_PORT   Static server listen port        (default 8765)
#   CERT_DIR    Static-server cert directory      (default /tmp/wt_static_cert)
#   BRIDGE_IP   IP of the bridge host            (required for "static" role)
#   CERT_HASH   WebTransport cert SHA-256 hash   (required for "static" role)
#   HTTP_NO_TLS Set to 1 to skip HTTPS on the static server
set -e

ROLE=${1:-}
if [ "$ROLE" != "bridge" ] && [ "$ROLE" != "static" ]; then
  cat >&2 <<'USAGE'
Usage:
  run_split.sh bridge               Run wt_bridge only (UDP + QUIC)
  run_split.sh static               Run static HTTPS server only

  See header comments for environment variables.
USAGE
  exit 1
fi

cd "$(dirname "$0")/../../.."   # repo root

detect_lan_ip() {
  local ip
  # macOS
  ip=$(ifconfig 2>/dev/null \
       | awk '/inet / && $2 !~ /127\./ {print $2; exit}')
  # Linux fallback
  [ -n "$ip" ] || ip=$(ip -4 addr show 2>/dev/null \
       | awk '/inet / && $2 !~ /127\./ {print $2}' \
       | head -1 | cut -d/ -f1)
  echo "$ip"
}

LAN_IP=${LAN_IP:-$(detect_lan_ip)}
[ -n "$LAN_IP" ] || { echo "could not detect a non-loopback IPv4 — set \$LAN_IP" >&2; exit 1; }

UDP_PORT=${UDP_PORT:-6000}
QUIC_PORT=${QUIC_PORT:-4433}
HTTP_PORT=${HTTP_PORT:-8765}
CERT_DIR=${CERT_DIR:-/tmp/wt_static_cert}

trap 'echo; echo "[run_split] shutting down"; pkill -P $$ 2>/dev/null; exit 0' INT TERM

# ── bridge role ──────────────────────────────────────────────────────────
if [ "$ROLE" = "bridge" ]; then
  [ -x ./tools/wt_bridge/wt_bridge ] || (
    command -v go >/dev/null || export PATH=$HOME/go-toolchain/bin:$PATH
    cd tools/wt_bridge && go build -o wt_bridge .
  )

  pkill -x "wt_bridge" 2>/dev/null || true
  sleep 0.3

  ./tools/wt_bridge/wt_bridge \
    --listen-udp  "0.0.0.0:${UDP_PORT}" \
    --listen-quic "0.0.0.0:${QUIC_PORT}" \
    --dev > /tmp/wtb_split_bridge.log 2>&1 &
  BRIDGE_PID=$!
  sleep 1.0

  HASH=$(grep "viewer URL hint" /tmp/wtb_split_bridge.log \
         | sed -E 's/.*\?certHash=//' | tr -d '\r\n')
  if [ -z "$HASH" ]; then
    echo "bridge failed to start; see /tmp/wtb_split_bridge.log" >&2
    cat /tmp/wtb_split_bridge.log >&2
    exit 2
  fi

  cat <<EOF

========================================================================
 wt_bridge is up.

 UDP  listener: 0.0.0.0:${UDP_PORT}   (point your RFC 9828 sender here)
 QUIC listener: 0.0.0.0:${QUIC_PORT}  (browsers connect here via WebTransport)
 This host IP:  ${LAN_IP}

 WebTransport cert SHA-256:
   ${HASH}

 ── Producer side ──────────────────────────────────────────────────────
 Point any RFC 9828 sender at ${LAN_IP}:${UDP_PORT}.

 ── Static-server side ─────────────────────────────────────────────────
 On the machine running the static server, start with:

   BRIDGE_IP=${LAN_IP} CERT_HASH=${HASH} \\
       ./tools/wt_bridge/scripts/run_split.sh static

 Log: tail -f /tmp/wtb_split_bridge.log
 Ctrl-C to stop.
========================================================================

EOF
  tail -f /tmp/wtb_split_bridge.log
fi

# ── static role ──────────────────────────────────────────────────────────
if [ "$ROLE" = "static" ]; then
  if [ -z "$BRIDGE_IP" ] || [ -z "$CERT_HASH" ]; then
    echo "BRIDGE_IP and CERT_HASH are required for the static role." >&2
    echo "Run 'run_split.sh bridge' on the bridge host first to obtain them." >&2
    exit 1
  fi

  pkill -f "node web/perf/serve" 2>/dev/null || true
  sleep 0.3

  if [ "${HTTP_NO_TLS:-}" = "1" ] || ! command -v openssl >/dev/null; then
    SCHEME="http"
    SERVER_TLS_ARGS=()
  else
    ./tools/wt_bridge/scripts/gen_static_cert.sh "$CERT_DIR" "$LAN_IP"
    SCHEME="https"
    SERVER_TLS_ARGS=(--cert "$CERT_DIR/cert.pem" --key "$CERT_DIR/key.pem")
  fi

  node web/perf/serve.mjs "$HTTP_PORT" --bind "${SERVER_TLS_ARGS[@]}" \
    > /tmp/wtb_split_serve.log 2>&1 &
  SERVER_PID=$!
  sleep 0.5

  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "static server failed to start; see /tmp/wtb_split_serve.log" >&2
    cat /tmp/wtb_split_serve.log >&2
    exit 2
  fi

  QURL_ENC=$(python3 -c \
    "import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1],safe=''))" \
    "https://${BRIDGE_IP}:${QUIC_PORT}/")

  if [ "$SCHEME" = "https" ]; then
    PAGE_URL="${SCHEME}://${LAN_IP}:${HTTP_PORT}/wt_viewer/?autorun=1&url=${QURL_ENC}&certHash=${CERT_HASH}"
    cat <<EOF

========================================================================
 Static server is up.

 Listening: ${SCHEME}://0.0.0.0:${HTTP_PORT}
 This host: ${LAN_IP}
 Bridge at: ${BRIDGE_IP}:${QUIC_PORT}

 ── Browser (any device that can reach both hosts) ─────────────────────
   ${PAGE_URL}

   First load: click through the self-signed cert warning for
   ${LAN_IP}:${HTTP_PORT}.  WebTransport to the bridge is hash-pinned
   and does not trigger a separate prompt.

 Log: tail -f /tmp/wtb_split_serve.log
 Ctrl-C to stop.
========================================================================

EOF
  else
    PAGE_URL_LOCAL="http://localhost:${HTTP_PORT}/wt_viewer/?autorun=1&url=${QURL_ENC}&certHash=${CERT_HASH}"
    PAGE_URL_LAN="http://${LAN_IP}:${HTTP_PORT}/wt_viewer/?autorun=1&url=${QURL_ENC}&certHash=${CERT_HASH}"
    cat <<EOF

========================================================================
 Static server is up — HTTP mode (no TLS).

 Listening: http://0.0.0.0:${HTTP_PORT}
 This host: ${LAN_IP}
 Bridge at: ${BRIDGE_IP}:${QUIC_PORT}

 ── Browser, on THIS host ──────────────────────────────────────────────
   ${PAGE_URL_LOCAL}

 ── Browser, on another device (needs Chrome flag) ─────────────────────
   google-chrome \\
       --user-data-dir=/tmp/chrome-wt-split \\
       --unsafely-treat-insecure-origin-as-secure="http://${LAN_IP}:${HTTP_PORT}" \\
       "${PAGE_URL_LAN}"

 Log: tail -f /tmp/wtb_split_serve.log
 Ctrl-C to stop.
========================================================================

EOF
  fi

  tail -f /tmp/wtb_split_serve.log
fi
