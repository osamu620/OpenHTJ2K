#!/bin/bash
# Run a single Chrome-headless benchmark, capture the REPORT line, exit.
# Args: fixture variant frames [threads]
set -e
cd "$(dirname "$0")/../.."

FIX=${1:-1080p5994-1200frames.rtp}
VAR=${2:-mt_simd}
N=${3:-200}
T=${4:-0}

LOG=$(mktemp /tmp/serveXXXXXX.log)
trap 'pkill -P $$ 2>/dev/null; rm -f "$LOG"' EXIT

# Find a free port near 8765.
for p in 8765 8766 8767 8768; do
  if ! ss -tln 2>/dev/null | awk '{print $4}' | grep -q ":$p$"; then PORT=$p; break; fi
done
[ -n "$PORT" ] || { echo "no free port" >&2; exit 1; }

# Start server.
node web/perf/serve.mjs "$PORT" > "$LOG" 2>&1 &
SERVER_PID=$!
sleep 0.5

# Run headless Chrome.
URL="http://127.0.0.1:$PORT/perf/?autorun=1&fixture=$FIX&variant=$VAR&frames=$N&threads=$T"
google-chrome --headless=new --no-sandbox --disable-gpu \
  --enable-features=SharedArrayBuffer \
  --user-data-dir="/tmp/chrome-bench-$$" \
  --no-first-run --no-default-browser-check \
  "$URL" > /dev/null 2>&1 &
CHROME_PID=$!

# Wait for REPORT in the server log (max 90s).
for i in $(seq 1 90); do
  if grep -q "^REPORT " "$LOG"; then break; fi
  sleep 1
done
kill $CHROME_PID 2>/dev/null || true
kill $SERVER_PID 2>/dev/null || true
rm -rf "/tmp/chrome-bench-$$"

# Emit just the JSON.
grep "^REPORT " "$LOG" | head -1 | sed 's/^REPORT //'
