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
// 'planar' = post Y/Cb/Cr buffers (cheap; renderer applies matrix in shader)
// 'rgba'   = post a single RGBA8 buffer with WASM-side matrix already applied
//            (used by the Canvas2D fallback; ~2× the bytes but no main-thread work)
let outputMode = 'planar';

async function init({ wasmBase = '/wasm/', threadCount: tc = 4, output = 'planar' }) {
  threadCount = tc;
  outputMode = output;
  // We deliberately load the *single-threaded* SIMD build here, not mt_simd.
  // Emscripten's pthreads-enabled build spawns its own pool of inner Web
  // Workers at module-init time, and that inner-worker bootstrap is fragile
  // when invoked from a nested worker context (relative-URL resolution +
  // dynamic-import-from-classic-worker semantics).  The single-threaded
  // SIMD build avoids the pthread machinery entirely and runs cleanly in
  // any worker.  Per Phase A0 this hits ~28 fps for FHD@30 in Chromium —
  // borderline but enough for the production target.  4K@30 is not viable
  // on this path; revisit when nested-pthread support stabilises.
  const factoryURL = new URL(`${wasmBase}libopen_htj2k_simd.js`, self.location.href);
  const factory = (await import(factoryURL.href)).default;
  M = await factory({
    locateFile: (path) => new URL(path, factoryURL.href).href,
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
    // The non-mt build only exports `create_decoder` (3 args, no thread count).
    create_decoder:    M.cwrap('create_decoder',         'number', ['number','number','number']),
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
    get_colorspace:    M.cwrap('get_colorspace',         'number', ['number']),
  };

  packetPtr = M._malloc(PACKET_BUF);
  framePtr  = M._malloc(FRAME_BUF);
  yPtr      = M._malloc(PLANE_BUF);
  cbPtr     = M._malloc(PLANE_BUF);
  crPtr     = M._malloc(PLANE_BUF);
  rgbaPtr   = M._malloc(RGBA_BUF);
  session   = F.rtp_create();

  self.postMessage({ type: 'ready' });
}

function reset() {
  if (!F) return;
  if (decoder) { F.release_decoder(decoder); decoder = 0; }
  if (session) F.rtp_reset(session);
  lastStatsAt = 0;
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
    if (!decoder) decoder = F.create_decoder(framePtr, fsz, 0);
    else          F.reset_decoder(decoder, framePtr, fsz, 0);
    F.parse_j2c(decoder);
    const w  = F.get_width(decoder, 0);
    const h  = F.get_height(decoder, 0);
    const nc = F.get_num_components(decoder);
    const cw = nc >= 2 ? F.get_width(decoder, 1)  : w;
    const ch = nc >= 2 ? F.get_height(decoder, 1) : h;

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
        { type: 'frame', mode: 'rgba', w, h, nc, colorspace,
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
        { type: 'frame', mode: 'planar', w, h, cw, ch, nc, colorspace,
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
      case 'reset':
        reset();
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
