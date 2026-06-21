// Shared HTJ2K region-decode core: the substantive decode logic used by BOTH
// the OSD Web Worker (browser) and the Node correctness test, so the two can
// never drift.  Knows nothing about `self`/postMessage/OpenSeadragon — it takes
// an already-instantiated Emscripten module and turns OSD tile coordinates into
// tight RGBA tiles in the WASM heap.
//
// Backed by the region->RGBA primitive in web/src/wrapper.cpp
// (create_region_decoder / decode_region_to_rgba / destroy_region_decoder),
// which is the productised Lever A: whole codestream resident in the heap, each
// tile a windowed reuse decode (set_col_range/set_row_range) at its reduce level.

import { computePyramid, tileRegion } from './htj2k_geometry.mjs';

// cwrap the exports this core needs.  One place, so the Worker and the test bind
// identical signatures.
export function bindFns(M) {
  return {
    create_decoder:         M.cwrap('create_decoder', 'number', ['number', 'number', 'number']),
    parse:                  M.cwrap('parse_j2c_data', 'void', ['number']),
    get_width:              M.cwrap('get_width', 'number', ['number', 'number']),
    get_height:             M.cwrap('get_height', 'number', ['number', 'number']),
    get_num_components:     M.cwrap('get_num_components', 'number', ['number']),
    get_max_safe_reduce:    M.cwrap('get_max_safe_reduce_NL', 'number', ['number']),
    release:                M.cwrap('release_j2c_data', 'void', ['number']),
    invoke_to_rgba:         M.cwrap('invoke_decoder_to_rgba', 'void', ['number', 'number']),
    create_region_decoder:  M.cwrap('create_region_decoder', 'number', []),
    decode_region_to_rgba:  M.cwrap('decode_region_to_rgba', 'number',
      ['number', 'number', 'number', 'number', 'number', 'number', 'number', 'number', 'number']),
    destroy_region_decoder: M.cwrap('destroy_region_decoder', 'void', ['number']),
    malloc: (n) => M._malloc(n),
    free:   (p) => M._free(p),
  };
}

// Copy the codestream into the WASM heap once and probe its geometry from the
// main header.  Returns { dataPtr, dataLen, pyramid, nc }.  The bytes stay
// resident; every tile re-inits the decoder from this pointer (reuse fires
// because the main-header fingerprint is unchanged).
export function loadCodestream(M, F, bytes, tileSize = 256) {
  const dataLen = bytes.length;
  const dataPtr = F.malloc(dataLen);
  M.HEAPU8.set(bytes, dataPtr);

  const probe = F.create_decoder(dataPtr, dataLen, 0);
  F.parse(probe);
  const fullW = F.get_width(probe, 0);
  const fullH = F.get_height(probe, 0);
  const nc = F.get_num_components(probe);
  const maxReduce = F.get_max_safe_reduce(probe);
  F.release(probe);

  const pyramid = computePyramid({ fullW, fullH, tileSize, maxReduce });
  return { dataPtr, dataLen, pyramid, nc };
}

// A decode core bound to one resident codestream.  Maintains a pool of region
// decoders keyed by reduce level (the single-tile-reuse fingerprint is the main
// header, which does NOT encode reduce_NL, so each reduce level needs its own
// decoder to keep its tile tree warm) and a reusable RGBA scratch buffer.
export function makeDecodeCore({ M, F, dataPtr, dataLen, pyramid }) {
  const decoders = new Map(); // reduce -> region decoder handle
  let dstPtr = 0;
  let dstCap = 0;

  function decoderFor(reduce) {
    let d = decoders.get(reduce);
    if (!d) {
      d = F.create_region_decoder();
      decoders.set(reduce, d);
    }
    return d;
  }

  function ensureDst(n) {
    if (dstCap >= n) return;
    if (dstPtr) F.free(dstPtr);
    dstPtr = F.malloc(n);
    dstCap = n;
  }

  return {
    pyramid,

    // Decode an OSD tile into the WASM heap as a tight w*h*4 RGBA tile.
    // Returns { ptr, w, h, reduce, x0, y0 } or null for an out-of-range tile.
    // The returned ptr is into the shared scratch buffer: copy it out before
    // the next decodeTileToHeap() call.
    decodeTileToHeap(level, x, y) {
      const r = tileRegion(level, x, y, pyramid);
      if (!r) return null;
      const need = r.w * r.h * 4;
      ensureDst(need);
      const rc = F.decode_region_to_rgba(
        decoderFor(r.reduce), dataPtr, dataLen, r.reduce, r.x0, r.y0, r.w, r.h, dstPtr);
      if (rc !== 0) throw new Error(`decode_region_to_rgba rc=${rc} (level=${level} x=${x} y=${y})`);
      return { ptr: dstPtr, w: r.w, h: r.h, reduce: r.reduce, x0: r.x0, y0: r.y0 };
    },

    dispose() {
      for (const d of decoders.values()) F.destroy_region_decoder(d);
      decoders.clear();
      if (dstPtr) {
        F.free(dstPtr);
        dstPtr = 0;
        dstCap = 0;
      }
    },
  };
}
