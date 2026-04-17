#!/bin/sh
set -e

J2C="${1:?Usage: docker run <image> <file.j2c> [--tunnel-token TOKEN]}"
shift

PORT="${JPIP_PORT:-8080}"
TUNNEL_TOKEN=""

while [ $# -gt 0 ]; do
  case "$1" in
    --tunnel-token) TUNNEL_TOKEN="$2"; shift 2 ;;
    --port)         PORT="$2"; shift 2 ;;
    *)              shift ;;
  esac
done

echo "Starting JPIP server on port ${PORT}..."
open_htj2k_jpip_server "$J2C" --port "$PORT" &
SERVER_PID=$!

if [ -n "$TUNNEL_TOKEN" ]; then
  echo "Starting named Cloudflare Tunnel..."
  cloudflared tunnel run --token "$TUNNEL_TOKEN" &
elif [ "${JPIP_TUNNEL:-0}" = "1" ]; then
  echo "Starting quick Cloudflare Tunnel..."
  cloudflared tunnel --url "http://localhost:${PORT}" &
fi

wait $SERVER_PID
