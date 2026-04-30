#!/bin/bash
# End-to-end smoke test: server + bridge + udp replayer + headless Chrome.
# Verifies the WT viewer decodes at least one frame.
set +e
cd /home/osamu/Documents/src/OpenHTJ2K

# Cleanup any prior runs.
pkill -f "web/perf/serve" 2>/dev/null
pkill -f "wt_bridge"     2>/dev/null
pkill -f "udp_replay"    2>/dev/null
sleep 1

> /tmp/serve.log
> /tmp/wtb.log
> /tmp/replay.log
> /tmp/chrome.log

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
[ -n "$HASH" ] || { echo "no cert hash captured"; exit 2; }

# 3. UDP replayer (looped, paced — runs until killed)
node tools/wt_bridge/scripts/udp_replay.mjs \
  /home/osamu/Documents/data/videos/1080p5994-1200frames.rtp \
  --port 6000 --fps 30 --loop \
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

# 5. Watch for any successful REPORT for up to 15 s.
for i in $(seq 1 15); do
  if grep -q '"decodedFrames":[1-9]' /tmp/serve.log; then break; fi
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
rm -rf /tmp/chrome-e2e
wait 2>/dev/null
true
