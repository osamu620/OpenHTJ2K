#!/bin/sh
# Expose a local JPIP server via Cloudflare Tunnel.
#
# Cloudflare terminates TLS and speaks HTTP/3 to browsers automatically.
# The local server runs plain HTTP/1.1 — no certs needed.
#
# Prerequisites:
#   brew install cloudflared    # macOS
#   # or: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/
#
# Usage:
#   # Terminal 1: start the JPIP server
#   open_htj2k_jpip_server input.j2c --port 8080
#
#   # Terminal 2: expose it via tunnel
#   ./scripts/jpip_tunnel.sh [port=8080]
#
#   # Open the WASM demo and paste the tunnel URL:
#   https://htj2k-demo.pages.dev/jpip_demo.html
#
# The tunnel prints a URL like:
#   https://random-words.trycloudflare.com
# Paste that into the JPIP demo's "Server" field.

set -e
PORT="${1:-8080}"

if ! command -v cloudflared >/dev/null 2>&1; then
  echo "ERROR: cloudflared not found. Install with: brew install cloudflared"
  exit 1
fi

echo "Exposing http://localhost:${PORT} via Cloudflare Tunnel..."
echo "Paste the https://...trycloudflare.com URL into the JPIP demo."
echo ""
cloudflared tunnel --url "http://localhost:${PORT}"
