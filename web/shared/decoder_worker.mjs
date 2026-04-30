// Decoder Web Worker.
//
// Owns the WASM module instance, the rtp_session_* lifecycle, and the
// HTJ2K decoder.  Receives RFC 9828 RTP packet bytes from the main thread,
// reassembles frames via rtp_session, decodes via invoke_decoder_planar_u8,
// and posts the planar Y/Cb/Cr back as transferable ArrayBuffers.
//
// Intentionally single-purpose: no UI, no networking source, no rendering.
// Both wt_viewer (live WebTransport stream) and rtp_demo (file replay)
// instantiate this worker via decoder_client.mjs.
//
// Message protocol — see decoder_client.mjs for the main-thread side.

/* global self */

let M = null;             // Emscripten Module
let F = null;             // cwrap'd functions
let session = 0;          // rtp_session handle
let decoder = 0;          // openhtj2k_decoder handle

let packetPtr = 0;        // staging buffer for incoming packets
let framePtr  = 0;        // staging buffer for completed codestreams
let yPtr = 0, cbPtr = 0, crPtr = 0;
let rgbaPtr = 0;          // only used when invoke_to_rgba path requested

const PACKET_BUF = 4096;
const FRAME_BUF  = 16 << 20;
const PLANE_BUF  = 3840 * 2160;
const RGBA_BUF   = 3840 * 2160 * 4;

// Stats throttling: post a snapshot at most every STATS_PERIOD_MS.
const STATS_PERIOD_MS = 500;
let lastStatsAt = 0;

let threadCount = 4;
let isMtBuild   = true;   // false for 'simd' / 'scalar' — disables create_decoder_mt path
let reduceNL    = 0;      // resolution-reduce; 0 = full, 1 = half, 2 = quarter
// 'planar' = post Y/Cb/Cr buffers (cheap; renderer applies matrix in shader)
// 'rgba'   = post a single RGBA8 buffer with WASM-side matrix already applied
//            (used by the Canvas2D fallback; ~2× the bytes but no main-thread work)
let outputMode = 'planar';

const VARIANT_FILE = {
  mt_simd: 'libopen_htj2k_mt_simd.js',
  simd:    'libopen_htj2k_simd.js',
  mt:      'libopen_htj2k_mt.js',
  scalar:  'libopen_htj2k.js',
};

async function init({ wasmBase = '/wasm/', threadCount: tc = 4, output = 'planar',
                      variant = 'mt_simd', reduceNL: rn = 0 }) {
  threadCount = tc;
  outputMode = output;
  reduceNL   = rn | 0;
  const file = VARIANT_FILE[variant] || VARIANT_FILE.mt_simd;
  isMtBuild  = (variant === 'mt_simd' || variant === 'mt');
  // mt_simd works in a nested worker as long as Emscripten's pthread
  // bootstrap doesn't fall back to a relative `import('./libopen_htj2k_*.js')`
  // — that import resolves against the outer worker's URL in a nested
  // context (not the inner worker.js's URL like it does from the main
  // thread), and 404s.  Setting `mainScriptUrlOrBlob` to the absolute
  // module URL makes Emscripten send it explicitly to the pthread
  // workers as `urlOrBlob`, bypassing the relative import.  Verified
  // via web/perf/mt_worker_diag.html — without mainScriptUrlOrBlob the
  // inner workers fire "Uncaught [object Event]" storms; with it, all
  // pthreads bootstrap cleanly.  Harmless on single-threaded builds.
  const factoryURL = new URL(file, wasmBase);
  const factory = (await import(factoryURL.href)).default;
  M = await factory({
    locateFile:         (path) => new URL(path, factoryURL.href).href,
    mainScriptUrlOrBlob: factoryURL.href,
  });

  F = {
    rtp_create:        M.cwrap('rtp_session_create',     'number', []),
    rtp_destroy:       M.cwrap('rtp_session_destroy',    'void',   ['number']),
    rtp_reset:         M.cwrap('rtp_session_reset',      'void',   ['number']),
    rtp_push:          M.cwrap('rtp_push_packet',        'number', ['number','number','number']),
    rtp_peek:          M.cwrap('rtp_peek_frame_size',    'number', ['number']),
    rtp_pop:           M.cwrap('rtp_pop_frame',          'number', ['number','number','number']),
    rtp_drop_ready:    M.cwrap('rtp_drop_ready',         'number', ['number']),
    rtp_ready_count:   M.cwrap('rtp_ready_count',        'number', ['number']),
    rtp_pop_ts:        M.cwrap('rtp_pop_frame_timestamp','number', ['number']),
    rtp_pop_matrix:    M.cwrap('rtp_pop_frame_matrix',   'number', ['number']),
    rtp_pop_range:     M.cwrap('rtp_pop_frame_range',    'number', ['number']),
    rtp_pop_primaries: M.cwrap('rtp_pop_frame_primaries','number', ['number']),
    rtp_pop_transfer:  M.cwrap('rtp_pop_frame_transfer', 'number', ['number']),
    rtp_frames:        M.cwrap('rtp_frames_emitted',     'number', ['number']),
    rtp_drops:         M.cwrap('rtp_frames_dropped',     'number', ['number']),
    rtp_gaps:          M.cwrap('rtp_seq_gaps',           'number', ['number']),
    rtp_last_error:    M.cwrap('rtp_last_error',         'string', ['number']),
    create_decoder:    M.cwrap('create_decoder',         'number', ['number','number','number']),
    create_decoder_mt: isMtBuild
      ? M.cwrap('create_decoder_mt',                     'number', ['number','number','number','number'])
      : null,
    reset_decoder:     M.cwrap('reset_decoder_with_bytes','void',  ['number','number','number','number']),
    parse_j2c:         M.cwrap('parse_j2c_data',         'void',   ['number']),
    invoke_planar_u8:  M.cwrap('invoke_decoder_planar_u8','void',  ['number','number','number','number']),
    invoke_to_rgba:    M.cwrap('invoke_decoder_to_rgba', 'void',   ['number','number']),
    apply_bt601:       M.cwrap('apply_ycbcr_bt601_to_rgba','void', ['number','number']),
    apply_bt709:       M.cwrap('apply_ycbcr_bt709_to_rgba','void', ['number','number']),
    release_decoder:   M.cwrap('release_j2c_data',       'void',   ['number']),
    get_width:         M.cwrap('get_width',              'number', ['number','number']),
    get_height:        M.cwrap('get_height',             'number', ['number','number']),
    get_num_components:M.cwrap('get_num_components',     'number', ['number']),
    get_depth:         M.cwrap('get_depth',              'number', ['number','number']),
    get_signed:        M.cwrap('get_signed',             'number', ['number','number']),
    get_colorspace:    M.cwrap('get_colorspace',         'number', ['number']),
  };

  packetPtr = M._malloc(PACKET_BUF);
  framePtr  = M._malloc(FRAME_BUF);
  yPtr      = M._malloc(PLANE_BUF);
  cbPtr     = M._malloc(PLANE_BUF);
  crPtr     = M._malloc(PLANE_BUF);
  rgbaPtr   = M._malloc(RGBA_BUF);
  session   = F.rtp_create();

  self.postMessage({ type: 'ready', variant });
}

function reset() {
  if (!F) return;
  if (decoder) { F.release_decoder(decoder); decoder = 0; }
  if (session) F.rtp_reset(session);
  lastStatsAt = 0;
}

// reduce_NL is set at decoder creation time and can't be changed on a live
// instance — main thread calls this when its 'auto' resolution heuristic
// flips between half and full after seeing the first frame.  We just drop
// the current decoder; the next drainReady() rebuilds it with the new value.
function setReduceNL(n) {
  reduceNL = n | 0;
  if (decoder) { F.release_decoder(decoder); decoder = 0; }
}

function makeDecoder(framePtr, fsz) {
  return isMtBuild
    ? F.create_decoder_mt(framePtr, fsz, reduceNL, threadCount)
    : F.create_decoder(framePtr, fsz, reduceNL);
}

function pushPacket(bytes) {
  if (!F || !session) return;
  const len = bytes.length;
  if (len > PACKET_BUF) return;       // oversized — drop silently (caller logs)
  M.HEAPU8.set(bytes, packetPtr);
  const r = F.rtp_push(session, packetPtr, len);
  if (r === 1) drainReady();
  maybePostStats();
}

// Process N packets from a single ArrayBuffer in one message-handler turn.
// `offsets[i]` is the exclusive end of packet i within `bytesBuf` (so
// packet i spans [offsets[i-1] || 0, offsets[i])).  Reduces postMessage
// cost ~Nx vs per-packet messages, which matters on high-bitrate streams
// (>>10k packets/s where the IPC overhead saturates the main thread).
//
// drainReady() runs after every frame-completing push within the batch
// (same as the per-packet path) — NOT just at end-of-batch.  The C++ ready
// queue caps at 2 (wrapper.cpp:932) and silently pops the front when full;
// if a batch contained two markers, deferring drainReady to the end would
// silently drop the earlier frame.
function pushPacketBatch(bytesBuf, offsets) {
  if (!F || !session) return;
  const u8 = new Uint8Array(bytesBuf);
  let prev = 0;
  for (let i = 0; i < offsets.length; i++) {
    const end = offsets[i];
    const len = end - prev;
    if (len > 0 && len <= PACKET_BUF) {
      M.HEAPU8.set(u8.subarray(prev, end), packetPtr);
      const r = F.rtp_push(session, packetPtr, len);
      if (r === 1) drainReady();
    }
    prev = end;
  }
  maybePostStats();
}

// Drop-old policy: when more than one frame is ready, skip everything but
// the latest before decoding.  Mirrors the wt_viewer's behaviour pre-worker
// (drop-on-overrun is essential for live streams under load).
function drainReady() {
  while (F.rtp_peek(session) > 0 && F.rtp_ready_count(session) > 1) {
    F.rtp_drop_ready(session);
  }
  while (true) {
    const fsz = F.rtp_peek(session);
    if (!fsz) return;
    if (fsz > FRAME_BUF) {
      self.postMessage({ type: 'error', msg: `frame too large: ${fsz}`, fatal: false });
      F.rtp_drop_ready(session);
      continue;
    }
    F.rtp_pop(session, framePtr, FRAME_BUF);
    const matrix    = F.rtp_pop_matrix(session);
    const range     = F.rtp_pop_range(session);
    const primaries = F.rtp_pop_primaries(session);
    const transfer  = F.rtp_pop_transfer(session);
    const rtpTs     = F.rtp_pop_ts(session);

    const t0 = performance.now();
    if (!decoder) decoder = makeDecoder(framePtr, fsz);
    else          F.reset_decoder(decoder, framePtr, fsz, reduceNL);
    F.parse_j2c(decoder);
    // get_* return FULL-resolution dims regardless of reduce_NL.  invoke_*
    // writes pixels at REDUCED dims = ceil(full / 2^reduce_NL).
    const shift = reduceNL | 0;
    const fullW = F.get_width(decoder, 0)  | 0;
    const fullH = F.get_height(decoder, 0) | 0;
    const w  = (fullW + (1 << shift) - 1) >> shift;
    const h  = (fullH + (1 << shift) - 1) >> shift;
    const nc = F.get_num_components(decoder) | 0;
    const fullCW = nc >= 2 ? (F.get_width(decoder, 1)  | 0) : fullW;
    const fullCH = nc >= 2 ? (F.get_height(decoder, 1) | 0) : fullH;
    const cw = (fullCW + (1 << shift) - 1) >> shift;
    const ch = (fullCH + (1 << shift) - 1) >> shift;
    const depth = F.get_depth(decoder, 0) | 0;
    // Per-component diagnostics (full-res dims, signedness, depth).  Capped
    // at 4 components — RGB/RGBA/CMYK/YCbCrA cover everything we expect.
    const compW = [], compH = [], compS = [], compD = [];
    const ncMeta = Math.min(nc, 4);
    for (let c = 0; c < ncMeta; c++) {
      compW.push(F.get_width(decoder, c)  | 0);
      compH.push(F.get_height(decoder, c) | 0);
      compS.push(F.get_signed(decoder, c) | 0);
      compD.push(F.get_depth(decoder, c)  | 0);
    }

    const colorspace = F.get_colorspace(decoder);
    const isRGB    = (nc >= 3 && colorspace === 16);   // ENUMCS_SRGB
    const isYCbCr  = (nc >= 3 && !isRGB);

    if (outputMode === 'rgba') {
      // RGBA path — used by the Canvas2D fallback.  invoke_decoder_to_rgba
      // does the planar→packed conversion (with chroma nearest-neighbour
      // upsampling); apply_ycbcr_bt601/709 then writes the matrix in place.
      F.invoke_to_rgba(decoder, rgbaPtr);
      if (isYCbCr) {
        if (matrix === 1)                                       F.apply_bt709(rgbaPtr, w * h);
        else if (matrix === 5 || matrix === 6 || matrix === 255) F.apply_bt601(rgbaPtr, w * h);
        // 0 (identity), 2 (unspec), or sRGB → leave as-is
      }
      const decodeMs = performance.now() - t0;
      const rgbaBuf = M.HEAPU8.slice(rgbaPtr, rgbaPtr + w * h * 4).buffer;
      self.postMessage(
        { type: 'frame', mode: 'rgba', w, h, fullW, fullH, nc, colorspace, depth,
          compW, compH, compS, compD,
          matrix, range, primaries, transfer, rtpTs, decodeMs,
          rgba: rgbaBuf },
        [rgbaBuf]
      );
    } else {
      // Planar path — used by the WebGL2 renderer.  Three R8 textures get
      // uploaded on main; the fragment shader does the matrix.
      F.invoke_planar_u8(decoder, yPtr, cbPtr, crPtr);
      const decodeMs = performance.now() - t0;
      // .slice() copies; the result owns its buffer and is transferable.
      // Subarray would share storage with the WASM heap and can't be transferred.
      const yBuf  = M.HEAPU8.slice(yPtr,  yPtr  + w  * h ).buffer;
      const cbBuf = M.HEAPU8.slice(cbPtr, cbPtr + cw * ch).buffer;
      const crBuf = M.HEAPU8.slice(crPtr, crPtr + cw * ch).buffer;
      self.postMessage(
        { type: 'frame', mode: 'planar', w, h, cw, ch, fullW, fullH, nc, colorspace, depth,
          compW, compH, compS, compD,
          isRGB, isYCbCr,
          matrix, range, primaries, transfer, rtpTs, decodeMs,
          y: yBuf, cb: cbBuf, cr: crBuf },
        [yBuf, cbBuf, crBuf]
      );
    }
  }
}

function maybePostStats() {
  const now = performance.now();
  if (now - lastStatsAt < STATS_PERIOD_MS) return;
  lastStatsAt = now;
  self.postMessage({
    type: 'stats',
    framesEmitted: F.rtp_frames(session),
    framesDropped: F.rtp_drops(session),
    seqGaps:       F.rtp_gaps(session),
    readyCount:    F.rtp_ready_count(session),
    lastError:     F.rtp_last_error(session) || '',
  });
}

self.addEventListener('message', async ({ data }) => {
  try {
    switch (data.type) {
      case 'init':
        await init(data);
        break;
      case 'packet':
        pushPacket(new Uint8Array(data.bytes));
        break;
      case 'packet_batch':
        pushPacketBatch(data.bytes, data.offsets);
        break;
      case 'reset':
        reset();
        break;
      case 'setReduceNL':
        setReduceNL(data.value);
        break;
      case 'drain':
        // postMessage delivery is FIFO, and pushPacket()'s decode runs
        // synchronously on this worker's thread.  By the time we handle
        // 'drain', every prior 'packet' message has been processed and any
        // resulting 'frame' message has already been posted to the main
        // thread.  The reply lets the main thread wait for the tail of
        // playback before tearing down (rtp_demo: avoids "Playback finished"
        // appearing while frames are still in flight).
        self.postMessage({ type: 'drained' });
        break;
      case 'close':
        if (decoder) { F.release_decoder(decoder); decoder = 0; }
        if (session) { F.rtp_destroy(session); session = 0; }
        self.close();
        break;
      default:
        self.postMessage({ type: 'error', msg: `unknown message type: ${data.type}`, fatal: false });
    }
  } catch (e) {
    self.postMessage({ type: 'error', msg: e?.stack || String(e), fatal: true });
  }
});
