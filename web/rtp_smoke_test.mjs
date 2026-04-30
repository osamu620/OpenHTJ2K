#!/usr/bin/env node
// Smoke test for the RTP wrapper API — loads the WASM module, creates a session,
// feeds a synthetic minimal packet, verifies the stats-query path works.
// Does NOT exercise the decoder end-to-end (no real .rtp fixture in the repo yet).
//
// Usage:
//   node web/rtp_smoke_test.mjs [path/to/sample.rtp]
// If a .rtp file is given, also verifies packet parsing + frame reassembly.

import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { createRequire } from 'module';

const __dir = dirname(fileURLToPath(import.meta.url));

// Load the scalar build (no pthreads to keep Node setup simple).
function loadFactory(jsPath) {
  let src = readFileSync(jsPath, 'utf-8');
  src = src.replace(/import\.meta\.url/g, JSON.stringify('file://' + jsPath));
  src = src.replace(/;var isPthread=[\s\S]*$/, '');
  src = src.replace(/export\s+default\s+Module\s*;?\s*$/, '');
  const require = createRequire(jsPath);
  return new Function('require', '__filename', '__dirname', src + '\nreturn Module;')(
      require, jsPath, dirname(jsPath));
}

// Pick variant: set RTP_VARIANT=mt_simd to test the pthreads build.
const VARIANT  = process.env.RTP_VARIANT || 'simd';
const jsPath   = join(__dir, `build/html/libopen_htj2k_${VARIANT}.js`);
const wasmPath = join(__dir, `build/html/libopen_htj2k_${VARIANT}.wasm`);
const factory  = loadFactory(jsPath);
const M = await factory({ wasmBinary: readFileSync(wasmPath) });

// ── cwrap ──
const rtp_create     = M.cwrap('rtp_session_create',  'number', []);
const rtp_destroy    = M.cwrap('rtp_session_destroy', 'void',   ['number']);
const rtp_reset      = M.cwrap('rtp_session_reset',   'void',   ['number']);
const rtp_push       = M.cwrap('rtp_push_packet',     'number', ['number','number','number']);
const rtp_ready      = M.cwrap('rtp_ready_count',     'number', ['number']);
const rtp_peek       = M.cwrap('rtp_peek_frame_size', 'number', ['number']);
const rtp_pop        = M.cwrap('rtp_pop_frame',       'number', ['number','number','number']);
const rtp_frames     = M.cwrap('rtp_frames_emitted',  'number', ['number']);
const rtp_drops      = M.cwrap('rtp_frames_dropped',  'number', ['number']);
const rtp_gaps       = M.cwrap('rtp_seq_gaps',        'number', ['number']);
const rtp_last_error = M.cwrap('rtp_last_error',      'string', ['number']);

function assert(cond, msg) {
  if (!cond) { console.error('FAIL:', msg); process.exit(1); }
  console.log('  ok:', msg);
}

// ── Test 1: session lifecycle ──
console.log('[1] session lifecycle');
const s = rtp_create();
assert(s !== 0, 'rtp_session_create returns non-zero handle');
assert(rtp_ready(s) === 0, 'new session has empty ready queue');
assert(rtp_frames(s) === 0, 'new session has 0 frames emitted');

// ── Test 2: push garbage → expect parse error ──
console.log('[2] rejects malformed packet');
const garbage = new Uint8Array([0x00, 0x01, 0x02]);  // too short for RTP header
const pkt_ptr = M._malloc(16);
M.HEAPU8.set(garbage, pkt_ptr);
const r1 = rtp_push(s, pkt_ptr, garbage.length);
assert(r1 === -1, 'rtp_push on garbage returns -1');
const err1 = rtp_last_error(s);
assert(err1.length > 0, `error message set (got "${err1}")`);

// ── Test 3: push minimally-valid RTP + 9828 Main packet with 1 byte of payload ──
// RTP fixed header: V=2, no CSRCs, no ext, marker=1, PT=96, seq=1, ts=1000, ssrc=0xDEADBEEF
//   V=2,P=0,X=0,CC=0 => 0x80
//   M=1,PT=96        => 0x80 | 0x60 = 0xE0
//   seq=0x0001, ts=0x000003E8, ssrc=0xDEADBEEF
// RFC 9828 Main header (8 bytes): MH=3 (single main), TP=0, ORDH=0, P=0, XTRAC=0, PTSTAMP=0,
//   ESEQ=0, R=0, S=0, C=0, RANGE=0, PRIMS=0, TRANS=0, MAT=0
//   Byte 0 = (MH<<6) | (TP<<3) | ORDH    = (3<<6) = 0xC0
//   Byte 1 = (P<<7) | (XTRAC<<4) | PTSTAMP[11:8]    = 0x00
//   Bytes 2-3 = PTSTAMP[7:0] | ...                  = 0x00 0x00
//   Byte 4 = ESEQ = 0
//   Byte 5 = (R<<7) | (S<<6) | (C<<5) | (RANGE<<4)  = 0x00
//   Bytes 6-7 = PRIMS=0, TRANS=0 ... (8-byte fixed field)
console.log('[3] synthetic Main packet reassembles a 1-byte codestream');

// Build RTP header (12 bytes) + Main 9828 header (8 bytes) + payload (1 byte "codestream")
const rtp_hdr = new Uint8Array(12);
rtp_hdr[0] = 0x80;             // V=2
rtp_hdr[1] = 0xE0;             // M=1, PT=96
rtp_hdr[2] = 0x00; rtp_hdr[3] = 0x01;  // seq
rtp_hdr[4] = 0x00; rtp_hdr[5] = 0x00; rtp_hdr[6] = 0x03; rtp_hdr[7] = 0xE8; // ts
rtp_hdr[8] = 0xDE; rtp_hdr[9] = 0xAD; rtp_hdr[10] = 0xBE; rtp_hdr[11] = 0xEF; // ssrc

const nineK_main = new Uint8Array(8);
nineK_main[0] = 0xC0;   // MH=3

const codestream = new Uint8Array([0xAA]);  // synthetic 1-byte payload

const full = new Uint8Array(rtp_hdr.length + nineK_main.length + codestream.length);
full.set(rtp_hdr, 0);
full.set(nineK_main, rtp_hdr.length);
full.set(codestream, rtp_hdr.length + nineK_main.length);

rtp_reset(s);
M.HEAPU8.set(full, pkt_ptr);
const r2 = rtp_push(s, pkt_ptr, full.length);
if (r2 < 0) console.error('rtp_push error:', rtp_last_error(s));
assert(r2 === 1, 'rtp_push returns 1 (frame complete) on marker-set Main packet');
assert(rtp_ready(s) === 1, 'ready_count == 1');

const size = rtp_peek(s);
assert(size === codestream.length, `peek_frame_size == 1 (got ${size})`);

const out_ptr = M._malloc(16);
const got = rtp_pop(s, out_ptr, 16);
assert(got === 1, `rtp_pop returns 1 (got ${got})`);
assert(M.HEAPU8[out_ptr] === 0xAA, 'reassembled codestream byte matches input');
assert(rtp_ready(s) === 0, 'ready_count == 0 after pop');
assert(rtp_frames(s) === 1, 'frames_emitted == 1');

M._free(out_ptr);
M._free(pkt_ptr);
rtp_destroy(s);

// ── Test 4: optional — parse a real .rtp file if provided ──
if (process.argv[2]) {
  const rtp_path = process.argv[2];
  console.log(`[4] parsing real .rtp file: ${rtp_path}`);
  const data = readFileSync(rtp_path);
  let offset = 0;
  const s2 = rtp_create();
  const buf2 = M._malloc(2048);
  let pkts = 0, frames = 0, errs = 0;
  while (offset + 4 <= data.length) {
    if (data[offset] !== 0xFF || data[offset+1] !== 0xFF) {
      console.error(`bad marker at offset ${offset}`); process.exit(1);
    }
    const len = (data[offset+2] << 8) | data[offset+3];
    offset += 4;
    if (offset + len > data.length) { console.error('truncated'); break; }
    const pkt = data.subarray(offset, offset + len);
    if (len > 2048) { console.warn('packet too big, skip'); offset += len; continue; }
    M.HEAPU8.set(pkt, buf2);
    const rr = rtp_push(s2, buf2, len);
    pkts++;
    if (rr < 0) { errs++; continue; }
    if (rr === 1) {
      frames++;
      const sz = rtp_peek(s2);
      const fb = M._malloc(sz);
      rtp_pop(s2, fb, sz);
      M._free(fb);
    }
    offset += len;
  }
  console.log(`  packets=${pkts}  frames=${frames}  parse_errors=${errs}`);
  console.log(`  reported stats: frames_emitted=${rtp_frames(s2)} dropped=${rtp_drops(s2)} seq_gaps=${rtp_gaps(s2)}`);
  assert(frames > 0, 'at least one frame reassembled from .rtp file');
  M._free(buf2);
  rtp_destroy(s2);
}

console.log('\nALL TESTS PASSED');
