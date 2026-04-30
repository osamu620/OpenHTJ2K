#!/bin/bash
# End-to-end smoke test: server + bridge + udp replayer + headless Chrome.
# Verifies the WT viewer decodes at least one frame.
#
# Usage:
#   scripts/e2e_smoke.sh                         # uses tools/wt_bridge/fixtures/1080p2997_30frames.rtp
#   scripts/e2e_smoke.sh /path/to/other.rtp      # custom fixture
#   FIXTURE=… EXIT_CODE=strict scripts/e2e_smoke.sh   # fail on no-decode (CI use)
#
# Env knobs:
#   FIXTURE     — path to .rtp file (overrides positional arg)
#   FPS         — replay rate (default 30)
#   WAIT_SECS   — seconds to wait for a decoded frame (default 15)
#   EXIT_CODE   — `strict` returns non-zero on no-decode; default is 0
set +e

REPO=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$REPO"

FIXTURE=${FIXTURE:-${1:-tools/wt_bridge/fixtures/1080p2997_30frames.rtp}}
FPS=${FPS:-30}
WAIT_SECS=${WAIT_SECS:-15}
EXIT_CODE=${EXIT_CODE:-lenient}

[ -f "$FIXTURE" ] || { echo "fixture not found: $FIXTURE"; exit 2; }

# Cleanup any prior runs.  Use -x so this script's own path (which contains
# "wt_bridge") doesn't get matched by -f.
pkill -f "node web/perf/serve"     2>/dev/null
pkill -x "wt_bridge"               2>/dev/null
pkill -f "node tools/wt_bridge/scripts/udp_replay" 2>/dev/null
sleep 1

: > /tmp/serve.log
: > /tmp/wtb.log
: > /tmp/replay.log
: > /tmp/chrome.log

# 1. Static server (with /report)
node web/perf/serve.mjs 8765 > /tmp/serve.log 2>&1 &
SERVER_PID=$!
sleep 0.5

# 2. Bridge
./tools/wt_bridge/wt_bridge \
  --listen-udp 127.0.0.1:6000 --listen-quic 127.0.0.1:4433 --dev \
  > /tmp/wtb.log 2>&1 &
BRIDGE_PID=$!
sleep 1.5

HASH=$(grep "viewer URL hint" /tmp/wtb.log | sed -E 's/.*\?certHash=//' | tr -d '\r\n')
echo "HASH: $HASH"
[ -n "$HASH" ] || { echo "no cert hash captured"; cat /tmp/wtb.log; exit 2; }

# 3. UDP replayer (looped, paced — runs until killed)
node tools/wt_bridge/scripts/udp_replay.mjs "$FIXTURE" \
  --port 6000 --fps "$FPS" --loop \
  > /tmp/replay.log 2>&1 &
REPLAY_PID=$!

# 4. Headless Chrome — connect to viewer, run for 8 seconds, post stats every 1 s.
URL="http://127.0.0.1:8765/viewer/?autorun=1&url=https%3A%2F%2F127.0.0.1%3A4433%2F&certHash=${HASH}&report=1000"
google-chrome --headless=new --no-sandbox --disable-gpu \
  --enable-features=SharedArrayBuffer \
  --user-data-dir=/tmp/chrome-e2e \
  --no-first-run --no-default-browser-check \
  --enable-quic --origin-to-force-quic-on=127.0.0.1:4433 \
  "$URL" > /tmp/chrome.log 2>&1 &
CHROME_PID=$!

# 5. Watch for any successful REPORT for up to $WAIT_SECS seconds.
DECODED_OK=0
for i in $(seq 1 "$WAIT_SECS"); do
  if grep -q '"decodedFrames":[1-9]' /tmp/serve.log; then DECODED_OK=1; break; fi
  sleep 1
done

echo ""
echo "=== bridge log (tail) ==="
tail -10 /tmp/wtb.log
echo "=== replay log (tail) ==="
tail -5  /tmp/replay.log
echo "=== server log REPORTs ==="
grep '^REPORT' /tmp/serve.log | tail -5
echo "=== server log non-REPORT ==="
grep -v '^REPORT' /tmp/serve.log | tail -10
echo "=== chrome log (tail) ==="
tail -20 /tmp/chrome.log

kill $CHROME_PID $REPLAY_PID $BRIDGE_PID $SERVER_PID 2>/dev/null
sleep 1
find /tmp/chrome-e2e -mindepth 1 -delete 2>/dev/null
rmdir /tmp/chrome-e2e 2>/dev/null
wait 2>/dev/null

if [ "$EXIT_CODE" = "strict" ] && [ "$DECODED_OK" = "0" ]; then
  echo "FAIL: no decoded frame within ${WAIT_SECS}s"
  exit 1
fi
true
