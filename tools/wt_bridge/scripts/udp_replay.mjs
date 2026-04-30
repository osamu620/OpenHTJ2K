#!/usr/bin/env node
// Replay a .rtp fixture file (Spark format: [FFFF][len:u16BE][packet]…) onto
// a UDP socket at a controlled pace.  Lets us drive wt_bridge from a fixture
// without needing the rpicam-apps producer running.
//
// Usage:
//   node udp_replay.mjs <fixture.rtp> [--host 127.0.0.1] [--port 6000]
//                       [--fps 30] [--loop] [--max-packets N]

import { openSync, readSync, closeSync, statSync } from 'fs';
import dgram from 'dgram';

const args = process.argv.slice(2);
function flag(name, def) {
  const i = args.indexOf('--' + name);
  if (i < 0) return def;
  return args[i + 1];
}
function bool(name) { return args.includes('--' + name); }

const FIXTURE = args.find(a => !a.startsWith('--'));
if (!FIXTURE) {
  console.error('usage: node udp_replay.mjs <fixture.rtp> [--host H] [--port P] [--fps N] [--loop]');
  process.exit(2);
}
const HOST  = flag('host', '127.0.0.1');
const PORT  = parseInt(flag('port', '6000'), 10);
const FPS   = parseFloat(flag('fps', '30'));
const LOOP  = bool('loop');
const MAXP  = parseInt(flag('max-packets', '0'), 10);

const sock = dgram.createSocket('udp4');
const fd   = openSync(FIXTURE, 'r');
const stat = statSync(FIXTURE);
console.error(`replay: ${FIXTURE}  (${(stat.size / 1e6).toFixed(1)} MB)  → udp://${HOST}:${PORT}  fps=${FPS}  loop=${LOOP}`);

// Stream-read in chunks; the .rtp files can be 2+ GB, so don't slurp.
const CHUNK = 1 << 20;
let buf = Buffer.allocUnsafe(8 * CHUNK);
let head = 0, tail = 0;
let pos  = 0;

function fillTo(n) {
  while (tail - head < n) {
    if (head > buf.length / 2) { buf.copyWithin(0, head, tail); tail -= head; head = 0; }
    if (tail + CHUNK > buf.length) {
      const grown = Buffer.allocUnsafe(buf.length * 2);
      buf.copy(grown, 0, head, tail); tail -= head; head = 0; buf = grown;
    }
    const got = readSync(fd, buf, tail, Math.min(CHUNK, buf.length - tail), pos);
    if (got === 0) return false;
    pos  += got;
    tail += got;
  }
  return true;
}

function rewind() { pos = 0; head = 0; tail = 0; }

let sent = 0;
let firstFrameStart = 0;     // hrtime in ns of first packet of current frame
let prevTimestamp = null;
const RTP_CLOCK_HZ = 90000;  // RFC 9828 timestamp clock

async function sleepNs(nsec) {
  if (nsec <= 0) return;
  await new Promise(r => setTimeout(r, Math.max(0, nsec / 1e6)));
}

(async () => {
  while (true) {
    if (!fillTo(4)) {
      if (LOOP) { rewind(); continue; }
      break;
    }
    if (buf[head] !== 0xFF || buf[head + 1] !== 0xFF) {
      console.error(`bad marker at file offset ${pos - (tail - head)}`); break;
    }
    const len = (buf[head + 2] << 8) | buf[head + 3];
    head += 4;
    if (!fillTo(len)) {
      if (LOOP) { rewind(); continue; }
      console.error('truncated fixture'); break;
    }
    const pkt = buf.subarray(head, head + len);
    head += len;

    // Pace by RTP timestamps.  Each packet's RTP-ts is at byte offset 4..7.
    // First packet sets the wall-clock anchor; later packets sleep until
    // (now - anchor) catches up to (rtp_ts - first_rtp_ts) / 90 kHz.
    const ts = ((pkt[4] << 24) | (pkt[5] << 16) | (pkt[6] << 8) | pkt[7]) >>> 0;
    if (prevTimestamp === null) {
      prevTimestamp = ts;
      firstFrameStart = process.hrtime.bigint();
    }
    let diff = (ts - prevTimestamp) >>> 0;
    if (diff > 0x80000000) diff -= 0x100000000;     // signed wrap
    // RFC 9828 packets in the same frame share a timestamp — only sleep
    // when ts advances.  At FPS override (--fps), pace by frame count
    // when the file's wall-clock would skip.
    if (diff !== 0) {
      // emulate FPS: 1/FPS seconds per frame regardless of fixture rate
      const wantNs = BigInt(Math.round(1e9 / FPS));
      await sleepNs(Number(wantNs));
      prevTimestamp = ts;
    }

    sock.send(pkt, PORT, HOST);
    sent++;
    if (MAXP && sent >= MAXP) break;
    if ((sent % 1000) === 0) process.stderr.write('.');
  }
  console.error(`\nsent ${sent} packets`);
  closeSync(fd);
  sock.close();
})().catch(e => { console.error(e); process.exit(1); });
