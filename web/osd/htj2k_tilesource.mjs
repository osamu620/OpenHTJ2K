// OpenSeadragon TileSource backed by an HTJ2K codestream, decoded in a Web
// Worker via the OpenHTJ2K WASM region-decode primitive.
//
// This is Lever A productised for the browser: the whole codestream is fetched
// once and kept resident in the worker's WASM heap; each OSD tile request is a
// windowed reuse decode at the matching reduce level.  Output is 8-bit SDR,
// handed to OSD as a 2D canvas context ("context2d" cache type).  Byte-range
// (Lever B) is a later optimisation that swaps the whole-file fetch for ranged
// fetches of just the tile's precincts.
//
// Requires OpenSeadragon >= 6.0 (downloadTileStart / context.finish data-type
// graph) and an HTJ2K WASM build (web/build/html/libopen_htj2k_simd.{js,wasm}).
//
// Usage (ES module):
//   import { HTJ2KTileSource } from './htj2k_tilesource.mjs';
//   const ts = await HTJ2KTileSource.fromUrl('carina.j2c', { wasmBase: '../build/html/' });
//   const viewer = OpenSeadragon({ id: 'osd', tileSources: ts });

/* global OpenSeadragon, document */

export class HTJ2KTileSource extends OpenSeadragon.TileSource {
  // info  : the worker's 'ready' payload (fullW, fullH, maxLevel, minLevel, ...)
  // worker: the decode Web Worker, already initialised
  constructor({ info, worker }) {
    super({
      width: info.fullW,
      height: info.fullH,
      tileSize: info.tileSize,
      tileOverlap: 0,
      minLevel: info.minLevel,
      maxLevel: info.maxLevel,
    });
    this._info = info;
    this._worker = worker;
    this._pending = new Map(); // id -> ImageJob context
    this._nextId = 1;
    this._stats = { tiles: 0, totalMs: 0, lastMs: 0, maxMs: 0 };
    this._onMessage = (e) => this._handleWorkerMessage(e.data);
    worker.addEventListener('message', this._onMessage);
  }

  // Rolling decode stats (host can poll this for an Iris-style latency HUD).
  getStats() {
    const s = this._stats;
    return { ...s, avgMs: s.tiles ? s.totalMs / s.tiles : 0 };
  }

  // Fetch the codestream, spin up + initialise the decode worker, and resolve to
  // a ready-to-open TileSource once the worker reports the image geometry.
  static async fromUrl(url, {
    wasmBase = './',
    wasmFile = 'libopen_htj2k_simd.js',
    tileSize = 256,
    workerUrl,
  } = {}) {
    const resp = await fetch(url);
    if (!resp.ok) {
      throw new Error(`HTJ2KTileSource: fetch ${url} -> HTTP ${resp.status}`);
    }
    const codestream = await resp.arrayBuffer();
    const wurl = workerUrl || new URL('./htj2k_tile_worker.mjs', import.meta.url);
    const worker = new Worker(wurl, { type: 'module' });

    const info = await new Promise((resolve, reject) => {
      const onMsg = (e) => {
        const d = e.data;
        if (d.type === 'ready') {
          worker.removeEventListener('message', onMsg);
          resolve(d);
        } else if (d.type === 'error' && d.fatal) {
          worker.removeEventListener('message', onMsg);
          reject(new Error(d.msg));
        }
      };
      worker.addEventListener('message', onMsg);
      worker.addEventListener('error', (ev) =>
        reject(new Error(`HTJ2KTileSource worker error: ${ev.message || ev}`)), { once: true });
      worker.postMessage(
        { type: 'init', wasmBase: new URL(wasmBase, location.href).href, wasmFile, codestream, tileSize },
        [codestream]);
    });
    return new HTJ2KTileSource({ info, worker });
  }

  // OSD calls these to identify a tile.  We decode in downloadTileStart rather
  // than fetch a URL, but getTileUrl must still return a unique, stable string
  // (it becomes context.src and part of the tile's identity).
  getTileUrl(level, x, y) {
    return `htj2k://${this._uniqueIdentifier}/${level}/${x}_${y}`;
  }

  getTileHashKey(level, x, y) {
    return `htj2k_${this._uniqueIdentifier}_${level}_${x}_${y}`;
  }

  downloadTileStart(context) {
    const id = this._nextId++;
    const t = context.tile;
    context.userData = context.userData || {};
    context.userData.htj2kId = id;
    this._pending.set(id, context);
    this._worker.postMessage({ type: 'decodeTile', id, level: t.level, x: t.x, y: t.y });
  }

  downloadTileAbort(context) {
    const id = context.userData && context.userData.htj2kId;
    if (id == null) return;
    this._pending.delete(id);
    this._worker.postMessage({ type: 'abort', id });
  }

  _handleWorkerMessage(d) {
    if (d.type !== 'tile') return;
    const context = this._pending.get(d.id);
    if (!context) return; // aborted / already settled
    this._pending.delete(d.id);

    if (d.error) return this._fail(context, d.error);
    if (d.empty) return this._fail(context, 'HTJ2KTileSource: out-of-range tile');

    if (typeof d.decodeMs === 'number') {
      const s = this._stats;
      s.tiles++;
      s.totalMs += d.decodeMs;
      s.lastMs = d.decodeMs;
      if (d.decodeMs > s.maxMs) s.maxMs = d.decodeMs;
    }

    try {
      // OSD's "context2d" cache type expects a CanvasRenderingContext2D (its
      // type map keys off '[object CanvasRenderingContext2D]'), so use a real
      // <canvas> 2D context here — not an OffscreenCanvas context, whose tag
      // differs and would not be recognised.
      const canvas = document.createElement('canvas');
      canvas.width = d.w;
      canvas.height = d.h;
      const ctx = canvas.getContext('2d');
      ctx.putImageData(new ImageData(new Uint8ClampedArray(d.rgba), d.w, d.h), 0, 0);
      context.finish(ctx, context.src, 'context2d');
    } catch (err) {
      this._fail(context, err?.message || String(err));
    }
  }

  _fail(context, msg) {
    if (typeof context.fail === 'function') context.fail(msg, context.src);
    else context.finish(null, context.src, msg);
  }

  // Not part of the OSD TileSource contract, but lets a host tear down the
  // worker when a source is discarded.
  destroy() {
    if (this._worker) {
      this._worker.removeEventListener('message', this._onMessage);
      this._worker.postMessage({ type: 'close' });
      this._worker.terminate();
      this._worker = null;
    }
    this._pending.clear();
  }
}

if (typeof OpenSeadragon !== 'undefined') {
  OpenSeadragon.HTJ2KTileSource = HTJ2KTileSource;
}
