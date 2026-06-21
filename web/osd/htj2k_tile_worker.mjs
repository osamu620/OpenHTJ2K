// HTJ2K tile-decode Web Worker for the OpenSeadragon HTJ2KTileSource.
//
// Owns the WASM module, one in-heap copy of the codestream, and a pool of region
// decoders (one per reduce level).  Decodes each requested (level, x, y) tile to
// a tight w*h RGBA buffer via the shared decode core and posts it back as a
// transferable ArrayBuffer.  All UI / OSD wiring lives on the main thread in
// htj2k_tilesource.mjs; this worker is pure decode.
//
// Deployment model (see the OSD integration handoff): single-thread decode per
// tile, scale out across N workers.  This file is one such unit, so it loads the
// single-thread SIMD build (libopen_htj2k_simd.js) by default.
//
// Message protocol (main -> worker):
//   { type:'init', wasmBase, wasmFile?, codestream:ArrayBuffer, tileSize? }
//   { type:'decodeTile', id, level, x, y }
//   { type:'abort', id }
//   { type:'close' }
// (worker -> main):
//   { type:'ready', fullW, fullH, nc, maxReduce, maxLevel, minLevel, tileSize }
//   { type:'tile', id, w, h, rgba:ArrayBuffer }   // transferable
//   { type:'tile', id, empty:true }               // out-of-range tile
//   { type:'tile', id, error:string }
//   { type:'error', msg, fatal }

/* global self */

import { bindFns, loadCodestream, makeDecodeCore } from './htj2k_decode_core.mjs';

let M = null;
let F = null;
let core = null;
const aborted = new Set(); // ids cancelled before their decode started

async function init({ wasmBase, wasmFile = 'libopen_htj2k_simd.js', codestream, tileSize = 256 }) {
  const factoryURL = new URL(wasmFile, wasmBase);
  // Same incantation as web/shared/decoder_worker.mjs: setting locateFile +
  // mainScriptUrlOrBlob lets the Emscripten module bootstrap correctly even when
  // imported from inside a Worker (harmless on the single-thread build).
  const factory = (await import(factoryURL.href)).default;
  M = await factory({
    locateFile: (path) => new URL(path, factoryURL.href).href,
    mainScriptUrlOrBlob: factoryURL.href,
  });
  F = bindFns(M);

  const bytes = new Uint8Array(codestream);
  const { dataPtr, dataLen, pyramid, nc } = loadCodestream(M, F, bytes, tileSize);
  core = makeDecodeCore({ M, F, dataPtr, dataLen, pyramid });

  self.postMessage({
    type: 'ready',
    fullW: pyramid.fullW,
    fullH: pyramid.fullH,
    nc,
    maxReduce: pyramid.maxReduce,
    maxLevel: pyramid.maxLevel,
    minLevel: pyramid.minLevel,
    tileSize: pyramid.tileSize,
  });
}

function decodeTile({ id, level, x, y }) {
  if (aborted.delete(id)) return; // cancelled while queued
  let tile;
  const t0 = performance.now();
  try {
    tile = core.decodeTileToHeap(level, x, y);
  } catch (e) {
    self.postMessage({ type: 'tile', id, error: e?.message || String(e) });
    return;
  }
  if (!tile) {
    self.postMessage({ type: 'tile', id, empty: true });
    return;
  }
  const decodeMs = performance.now() - t0;
  // .slice() returns an owned ArrayBuffer (not a view onto the heap), so it is
  // safe to transfer.  Re-read HEAPU8 here: ALLOW_MEMORY_GROWTH may have
  // reallocated the heap during the decode, detaching any earlier view.
  const { ptr, w, h, reduce } = tile;
  const rgba = M.HEAPU8.slice(ptr, ptr + w * h * 4).buffer;
  self.postMessage({ type: 'tile', id, w, h, reduce, decodeMs, rgba }, [rgba]);
}

self.addEventListener('message', async ({ data }) => {
  try {
    switch (data.type) {
      case 'init':
        await init(data);
        break;
      case 'decodeTile':
        decodeTile(data);
        break;
      case 'abort':
        aborted.add(data.id);
        break;
      case 'close':
        if (core) core.dispose();
        self.close();
        break;
      default:
        self.postMessage({ type: 'error', msg: `unknown message type: ${data.type}`, fatal: false });
    }
  } catch (e) {
    self.postMessage({ type: 'error', msg: e?.stack || String(e), fatal: true });
  }
});
