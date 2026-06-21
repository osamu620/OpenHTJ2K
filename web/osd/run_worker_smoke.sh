#!/usr/bin/env bash
# Headless-Chrome smoke for the OSD HTJ2K decode worker: spawns the real module
# Worker, dynamic-imports the WASM module inside it, decodes sample tiles and
# validates them. Captures the page's REPORT line and exits non-zero on failure.
#
#   web/osd/run_worker_smoke.sh [fixture]
set -euo pipefail
cd "$(dirname "$0")/../.."

FIXTURE="${1:-$HOME/Downloads/heic0602a.j2c}"

if [[ ! -f web/build/html/libopen_htj2k_simd.wasm ]]; then
  echo "WASM module missing: cmake --build web/build --target libopen_htj2k_simd" >&2
  exit 1
fi
[[ -f "$FIXTURE" ]] || { echo "fixture not found: $FIXTURE" >&2; exit 1; }
ln -sf "$FIXTURE" web/osd/carina.j2c

LOG=$(mktemp /tmp/osd_smoke_XXXXXX.log)
PROFILE=$(mktemp -d /tmp/osd_smoke_chrome_XXXXXX)
cleanup() {
  [[ -n "${CHROME_PID:-}" ]] && kill "$CHROME_PID" 2>/dev/null || true
  [[ -n "${SERVER_PID:-}" ]] && kill "$SERVER_PID" 2>/dev/null || true
  find "$PROFILE" -mindepth 0 -delete 2>/dev/null || true
  rm -f "$LOG" 2>/dev/null || true
}
trap cleanup EXIT

PORT=
for p in 8765 8766 8767 8768 8769; do
  if ! ss -tln 2>/dev/null | awk '{print $4}' | grep -q ":$p$"; then PORT=$p; break; fi
done
[[ -n "$PORT" ]] || { echo "no free port" >&2; exit 1; }

node web/perf/serve.mjs "$PORT" > "$LOG" 2>&1 &
SERVER_PID=$!
sleep 0.6

URL="http://127.0.0.1:$PORT/osd/worker_smoke.html"
google-chrome --headless=new --no-sandbox --disable-gpu --disable-dev-shm-usage \
  --user-data-dir="$PROFILE" --no-first-run --no-default-browser-check \
  "$URL" > /dev/null 2>&1 &
CHROME_PID=$!

for _ in $(seq 1 60); do
  grep -q "^REPORT " "$LOG" && break
  sleep 1
done

LINE=$(grep "^REPORT " "$LOG" | head -1 | sed 's/^REPORT //')
[[ -n "$LINE" ]] || { echo "no REPORT captured (timeout). server log:" >&2; tail -20 "$LOG" >&2; exit 2; }
echo "$LINE"
echo "$LINE" | grep -q '"pass":true' || { echo "SMOKE FAIL" >&2; exit 3; }
echo "SMOKE PASS"
