#!/usr/bin/env node
// Phase A0 — WASM decode latency benchmark.
//
// Feeds a .rtp fixture through the WASM rtp_session_* API, pops complete
// JPEG 2000 codestreams, decodes each via create_decoder + parse + invoke
// _decoder_to_rgba, and records per-frame decode latency.  Reports mean,
// p50, p95, p99, max in milliseconds and an estimate of sustained fps
// ceiling (1000 / mean).
//
// Usage:
//   node node_decode_bench.mjs <fixture.rtp> [variant=simd|mt_simd] [maxFrames=200]
//
// The variant selects which WASM artifact to load (single-threaded SIMD
// vs pthreads + SIMD).  The pthreads variant requires --experimental-vm-modules
// and `node --experimental-wasm-threads`; in this harness we stay single
// -threaded by default for portability across Node versions.

import { readFileSync, openSync, readSync, closeSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { createRequire } from 'module';

const __dir = dirname(fileURLToPath(import.meta.url));
const REPO  = join(__dir, '..', '..');

const FIXTURE    = process.argv[2];
const VARIANT    = process.argv[3] || 'simd';
const MAX_FRAMES = parseInt(process.argv[4] || '200', 10);

if (!FIXTURE) {
  console.error('usage: node node_decode_bench.mjs <fixture.rtp> [variant] [maxFrames]');
  process.exit(2);
}

const buildDir = join(REPO, 'subprojects', 'build_wt', 'html');
const jsPath   = join(buildDir, `libopen_htj2k_${VARIANT}.js`);
const wasmPath = join(buildDir, `libopen_htj2k_${VARIANT}.wasm`);

// Hack the emitted ES module so it can run under createRequire(): strip
// `import.meta.url`, the pthread-detection probe, and the default export.
function loadFactory(p) {
  let src = readFileSync(p, 'utf-8');
  src = src.replace(/import\.meta\.url/g, JSON.stringify('file://' + p));
  src = src.replace(/;var isPthread=[\s\S]*$/, '');
  src = src.replace(/export\s+default\s+Module\s*;?\s*$/, '');
  const require = createRequire(p);
  return new Function('require', '__filename', '__dirname', src + '\nreturn Module;')(
      require, p, dirname(p));
}

const factory = loadFactory(jsPath);
const M = await factory({ wasmBinary: readFileSync(wasmPath) });

// ── cwrap the API surface we use ──
const F = {
  rtp_create:        M.cwrap('rtp_session_create',     'number', []),
  rtp_destroy:       M.cwrap('rtp_session_destroy',    'void',   ['number']),
  rtp_push:          M.cwrap('rtp_push_packet',        'number', ['number','number','number']),
  rtp_peek:          M.cwrap('rtp_peek_frame_size',    'number', ['number']),
  rtp_pop:           M.cwrap('rtp_pop_frame',          'number', ['number','number','number']),
  rtp_last_error:    M.cwrap('rtp_last_error',         'string', ['number']),

  create_decoder:    M.cwrap('create_decoder',         'number', ['number','number','number']),
  reset_decoder:     M.cwrap('reset_decoder_with_bytes','void',  ['number','number','number','number']),
  parse_j2c:         M.cwrap('parse_j2c_data',         'void',   ['number']),
  invoke_to_rgba:    M.cwrap('invoke_decoder_to_rgba', 'void',   ['number','number']),
  apply_bt709:       M.cwrap('apply_ycbcr_bt709_to_rgba','void', ['number','number']),
  apply_bt601:       M.cwrap('apply_ycbcr_bt601_to_rgba','void', ['number','number']),
  release_decoder:   M.cwrap('release_j2c_data',       'void',   ['number']),
  get_width:         M.cwrap('get_width',              'number', ['number','number']),
  get_height:        M.cwrap('get_height',             'number', ['number','number']),
  get_num_components:M.cwrap('get_num_components',     'number', ['number']),
  get_colorspace:    M.cwrap('get_colorspace',         'number', ['number']),
  rtp_pop_matrix:    M.cwrap('rtp_pop_frame_matrix',   'number', ['number']),
};

// ── allocate scratch buffers ──
const PACKET_BUF_SIZE = 4096;
const FRAME_BUF_SIZE  = 16 * 1024 * 1024;   // up to ~16 MiB codestream
const RGBA_BUF_SIZE   = 3840 * 2160 * 4;    // sized for 4K RGBA

const packetPtr = M._malloc(PACKET_BUF_SIZE);
const framePtr  = M._malloc(FRAME_BUF_SIZE);
const rgbaPtr   = M._malloc(RGBA_BUF_SIZE);

const session  = F.rtp_create();

// ── feed the .rtp fixture and time decode of each frame ──
console.error(`[bench] variant=${VARIANT}  fixture=${FIXTURE}  maxFrames=${MAX_FRAMES}`);

// Stream-read the fixture in 8 MiB chunks via a RollingBuffer.  readFileSync()
// hits a 2 GiB ceiling on Node's Buffer; the .rtp fixtures are larger.
class RollingBuffer {
  constructor(cap = 16 << 20) { this._b = Buffer.allocUnsafe(cap); this._h = 0; this._t = 0; }
  get length() { return this._t - this._h; }
  bytes(off, len) { return this._b.subarray(this._h + off, this._h + off + len); }
  consume(n) {
    this._h += n;
    if (this._h > (this._b.length >> 1)) {
      this._b.copyWithin(0, this._h, this._t);
      this._t -= this._h; this._h = 0;
    }
  }
  appendFrom(fd) {
    const free = this._b.length - this._t;
    if (free < 65536) {
      const grown = Buffer.allocUnsafe(this._b.length * 2);
      this._b.copy(grown, 0, this._h, this._t);
      this._t -= this._h; this._h = 0; this._b = grown;
    }
    const n = readSync(fd, this._b, this._t, this._b.length - this._t, null);
    this._t += n;
    return n;
  }
}

const fd = openSync(FIXTURE, 'r');
const buf = new RollingBuffer();
const decodeMs = [];
let decoder = 0;
let firstW = 0, firstH = 0, firstC = 0;
let frames = 0;
let pkts = 0;
let pktErrs = 0;
let eof = false;

const t_start = process.hrtime.bigint();

while (frames < MAX_FRAMES) {
  // Ensure at least one full packet (header+payload) is buffered.
  while (!eof && buf.length < 4) {
    if (buf.appendFrom(fd) === 0) { eof = true; break; }
  }
  if (buf.length < 4) break;
  const hdr = buf.bytes(0, 4);
  if (hdr[0] !== 0xFF || hdr[1] !== 0xFF) {
    console.error(`bad fixture marker`); break;
  }
  const len = (hdr[2] << 8) | hdr[3];
  while (!eof && buf.length < 4 + len) {
    if (buf.appendFrom(fd) === 0) { eof = true; break; }
  }
  if (buf.length < 4 + len) break;
  if (len > PACKET_BUF_SIZE) { console.error(`packet too big (${len})`); break; }

  M.HEAPU8.set(buf.bytes(4, len), packetPtr);
  const r = F.rtp_push(session, packetPtr, len);
  buf.consume(4 + len);
  pkts++;
  if (r < 0) { pktErrs++; continue; }
  if (r !== 1) continue;   // not a complete frame yet

  // Frame complete — pop bytes and time the decode.
  const frameSize = F.rtp_peek(session);
  if (frameSize > FRAME_BUF_SIZE) { console.error(`frame too big (${frameSize})`); break; }
  F.rtp_pop(session, framePtr, FRAME_BUF_SIZE);

  const t0 = process.hrtime.bigint();
  if (decoder === 0) {
    decoder = F.create_decoder(framePtr, frameSize, 0);
  } else {
    F.reset_decoder(decoder, framePtr, frameSize, 0);
  }
  F.parse_j2c(decoder);
  if (firstW === 0) {
    firstW = F.get_width(decoder, 0);
    firstH = F.get_height(decoder, 0);
    firstC = F.get_num_components(decoder);
    console.error(`[bench] first frame: ${firstW}x${firstH} ${firstC}c`);
  }
  F.invoke_to_rgba(decoder, rgbaPtr);
  // Apply YCbCr→RGB if MCT is off (typical for sYCC streams).
  // get_colorspace returns 0=YCC after MCT, 1=YCbCr no MCT, 2=other.
  // Skipping the matrix in the timing keeps "decode" comparable; the
  // matrix is a cheap O(W*H) pass.
  const t1 = process.hrtime.bigint();
  decodeMs.push(Number(t1 - t0) / 1e6);
  frames++;
}

const t_end = process.hrtime.bigint();
const wall_s = Number(t_end - t_start) / 1e9;

if (decoder !== 0) F.release_decoder(decoder);
F.rtp_destroy(session);
M._free(packetPtr); M._free(framePtr); M._free(rgbaPtr);
closeSync(fd);

if (decodeMs.length === 0) {
  console.error('no frames decoded');
  process.exit(1);
}

// ── stats ──
decodeMs.sort((a, b) => a - b);
const mean = decodeMs.reduce((s, v) => s + v, 0) / decodeMs.length;
const pct  = q => decodeMs[Math.min(decodeMs.length - 1, Math.floor(q * decodeMs.length))];
const p50  = pct(0.50);
const p95  = pct(0.95);
const p99  = pct(0.99);
const max  = decodeMs[decodeMs.length - 1];
const min  = decodeMs[0];

const result = {
  fixture: FIXTURE.split('/').pop(),
  variant: VARIANT,
  resolution: `${firstW}x${firstH}`,
  components: firstC,
  frames: frames,
  packets: pkts,
  packet_errors: pktErrs,
  wall_s: +wall_s.toFixed(2),
  decode_ms: {
    min: +min.toFixed(2),
    mean: +mean.toFixed(2),
    p50: +p50.toFixed(2),
    p95: +p95.toFixed(2),
    p99: +p99.toFixed(2),
    max: +max.toFixed(2),
  },
  fps_ceiling_mean: +(1000 / mean).toFixed(1),
  fps_ceiling_p95:  +(1000 / p95).toFixed(1),
};

console.log(JSON.stringify(result, null, 2));
