// Node-side integration test for the WASM streaming feed API (issue #297
// follow-up).  Loads libopen_htj2k_jpip.js, creates a context from a
// JPIP main-header bin, then drives the same response bytes through two
// code paths:
//
//   A. One-shot: _jpip_add_response(buf, full_len) — the existing API.
//   B. Streaming: _jpip_feed_stream_begin/_feed/_end — new in issue #297.
//
// The two paths must produce identical visible state (canvas dimensions,
// bin/precinct totals, cache-model string).  The streaming path is then
// re-run with several chunk sizes (1, 7, 100, 1024, full) to confirm
// chunk-boundary handling tolerates every split.
//
// Expects the JPIP server to have been started separately on $JPIP_PORT
// (default 8094).  Invoke:
//   node tools/jpip_wasm_feed_test.mjs

import Module from '../subprojects/build/html/libopen_htj2k_jpip.js';

const PORT = parseInt(process.env.JPIP_PORT || '8094', 10);
const URL_BASE = `http://127.0.0.1:${PORT}/jpip`;

function assert(cond, msg) {
  if (!cond) { console.error(`FAIL: ${msg}`); process.exitCode = 1; throw new Error(msg); }
}

async function fetchBytes(q) {
  const r = await fetch(`${URL_BASE}?${q}`);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  const ab = await r.arrayBuffer();
  return new Uint8Array(ab);
}

function wasmWrite(M, arr, ptr) {
  if (M.writeArrayToMemory) M.writeArrayToMemory(arr, ptr);
  else M.HEAPU8.set(arr, ptr);
}

// Feed one response into the context via the streaming API, breaking
// the buffer into `chunkSize`-byte pieces.
function feedStreaming(M, ctx, bytes, chunkSize) {
  assert(M._jpip_feed_stream_begin(ctx) === 0, 'feed_stream_begin');
  for (let off = 0; off < bytes.length; off += chunkSize) {
    const end = Math.min(off + chunkSize, bytes.length);
    const slice = bytes.subarray(off, end);
    const ptr = M._jpip_get_response_buffer(ctx, slice.length);
    wasmWrite(M, slice, ptr);
    const rc = M._jpip_feed_stream(ctx, ptr, slice.length);
    assert(rc === 0, `feed_stream (off=${off}, chunk=${chunkSize}) rc=${rc}`);
  }
  const endRc = M._jpip_feed_stream_end(ctx);
  assert(endRc === 0, `feed_stream_end clean (chunk=${chunkSize}) rc=${endRc}`);
}

function snapshot(M, ctx) {
  const modelPtr = M._jpip_get_cache_model(ctx);
  return {
    w:     M._jpip_get_canvas_width(ctx),
    h:     M._jpip_get_canvas_height(ctx),
    ncomp: M._jpip_get_num_components(ctx),
    prec:  M._jpip_get_total_precincts(ctx),
    model: modelPtr ? M.UTF8ToString(modelPtr) : '',
  };
}

async function main() {
  const M = await Module();

  // A main-header-only request (roff=0, rsiz=1,1) gives the bins needed
  // to build the codestream index; then a full-image request exercises
  // precinct bins.  The test uses the default land_shallow_topo_1920_fov
  // asset that the caller started the server with.
  const probe = await fetchBytes('fsiz=1,1&roff=0,0&rsiz=1,1&type=jpp-stream');
  const probePtr = M._jpip_get_response_buffer(0, probe.length);
  // probePtr is null because handle is null; create_context doesn't need
  // it — we can pass a plain pointer by mallocing directly.
  const probeMem = M._malloc(probe.length);
  wasmWrite(M, probe, probeMem);
  const ctxA = M._jpip_create_context(probeMem, probe.length);
  assert(ctxA !== 0, 'create_context (A)');
  M._free(probeMem);

  // Now issue a full-image request (without `&model=` — full re-send so
  // the add_response vs feed_stream paths see identical input).
  const full = await fetchBytes('fsiz=1920,1920&type=jpp-stream');
  console.log(`full response: ${full.length} bytes`);

  // Path A: one-shot.
  {
    const ptr = M._jpip_get_response_buffer(ctxA, full.length);
    wasmWrite(M, full, ptr);
    const rc = M._jpip_add_response(ctxA, ptr, full.length);
    assert(rc === 0, 'add_response');
  }
  const snapA = snapshot(M, ctxA);
  console.log('add_response   →', snapA);

  // Path B: streaming, several chunk sizes.  A fresh context per run so
  // nothing carries over between attempts.
  for (const chunkSize of [1, 7, 100, 1024, full.length]) {
    const pbuf = await fetchBytes('fsiz=1,1&roff=0,0&rsiz=1,1&type=jpp-stream');
    const memP = M._malloc(pbuf.length);
    wasmWrite(M, pbuf, memP);
    const ctxBnew = M._jpip_create_context(memP, pbuf.length);
    assert(ctxBnew !== 0, `create_context (B chunk=${chunkSize})`);
    M._free(memP);
    feedStreaming(M, ctxBnew, full, chunkSize);
    const snapB = snapshot(M, ctxBnew);
    console.log(`feed_stream(${chunkSize}) →`, snapB);

    assert(snapB.w    === snapA.w,     `chunk=${chunkSize}: canvas_w mismatch`);
    assert(snapB.h    === snapA.h,     `chunk=${chunkSize}: canvas_h mismatch`);
    assert(snapB.prec === snapA.prec,  `chunk=${chunkSize}: total_precincts mismatch`);
    assert(snapB.model === snapA.model, `chunk=${chunkSize}: cache model mismatch`);
    M._jpip_destroy_context(ctxBnew);
  }

  M._jpip_destroy_context(ctxA);

  if (process.exitCode) {
    console.error('FAIL');
  } else {
    console.log('OK jpip_wasm_feed_test: add_response and feed_stream are equivalent');
  }
}

main().catch(e => { console.error(e); process.exit(1); });
