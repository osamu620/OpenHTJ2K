#!/usr/bin/env bash
# Serve the web/ tree for the OSD × HTJ2K demo.
#
#   web/osd/serve.sh [port] [fixture]
#
# Symlinks the gigapixel fixture to web/osd/carina.j2c (the demo's default src),
# then serves the web/ directory so the demo, the ES module worker, the WASM
# artifacts and the fixture are all same-origin. The single-thread SIMD build
# needs no COOP/COEP headers.
set -euo pipefail

PORT="${1:-8000}"
FIXTURE="${2:-$HOME/Downloads/heic0602a.j2c}"
WEB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -f "$WEB_DIR/build/html/libopen_htj2k_simd.wasm" ]]; then
  echo "WASM module missing. Build it first:" >&2
  echo "  cmake --build web/build --target libopen_htj2k_simd" >&2
  exit 1
fi

if [[ -f "$FIXTURE" ]]; then
  ln -sf "$FIXTURE" "$WEB_DIR/osd/carina.j2c"
  echo "fixture: $FIXTURE -> web/osd/carina.j2c"
else
  echo "warning: fixture not found at $FIXTURE — pass ?src=<url> in the demo" >&2
fi

echo "serving $WEB_DIR on http://localhost:$PORT"
echo "  demo: http://localhost:$PORT/osd/htj2k_osd_demo.html"
cd "$WEB_DIR"
exec python3 -m http.server "$PORT"
