// Main-thread client for the shared decoder Web Worker.
//
// Wraps web/shared/decoder_worker.mjs in a small class with callback hooks.
// Both wt_viewer and rtp_demo use this; the worker is an implementation
// detail.
//
// Usage:
//   const dec = new DecoderClient({
//     wasmBase: '/wasm/',         // where libopen_htj2k_*.{js,wasm} live
//     variant: 'mt_simd',         // mt_simd | simd | mt | scalar (default mt_simd)
//     threadCount: 4,             // WASM decoder workers (MT builds only)
//     reduceNL: 0,                // resolution reduce (0=full, 1=half, 2=quarter)
//     output: 'planar',           // planar | rgba
//     onFrame: ({ y, cb, cr, w, h, cw, ch, matrix, range, ... }) => {…},
//     onStats: ({ framesEmitted, framesDropped, seqGaps, … }) => {…},
//     onError: ({ msg, fatal }) => {…},
//   });
//   await dec.ready;            // resolves when the worker has loaded the WASM
//   dec.variant;                 // string of the variant that actually loaded
//   dec.pushPacket(uint8array); // send one RFC 9828 RTP packet
//   dec.setReduceNL(n);          // change DWT resolution-reduce (rebuilds decoder)
//   dec.reset();                 // discard in-flight state (e.g. on stream switch)
//   await dec.drain();           // wait for queued packets/frames to finish
//   dec.close();                 // tear the worker down

export class DecoderClient {
  constructor({ wasmBase = '/wasm/', variant = 'mt_simd', threadCount = 4,
                reduceNL = 0, output = 'planar',
                onFrame = () => {}, onStats = () => {}, onError = () => {} } = {}) {
    this.onFrame = onFrame;
    this.onStats = onStats;
    this.onError = onError;
    this.variant = variant;
    // Resolve wasmBase to absolute against the page so callers can pass a
    // page-relative '/wasm/', a directory-relative './', or a full URL — and
    // the worker (whose `self.location.href` is /shared/decoder_worker.mjs)
    // doesn't end up resolving './' against itself.
    const absBase = new URL(wasmBase, location.href).href;

    // The worker URL is resolved relative to this module so pages at
    // different paths (web/wt_viewer/index.html, web/rtp_demo.html) all
    // load the same file.  A cache-bust query string forces a fresh fetch
    // — Workers are sometimes cached more aggressively than the static
    // server's `Cache-Control: no-store` header would suggest, especially
    // across launcher restarts on the same browser session.
    const workerURL = new URL('./decoder_worker.mjs', import.meta.url);
    workerURL.searchParams.set('v', String(Date.now()));
    this.worker = new Worker(workerURL, { type: 'module' });

    this.ready = new Promise((resolve, reject) => {
      const handler = (ev) => {
        if (ev.data?.type === 'ready') {
          if (ev.data.variant) this.variant = ev.data.variant;
          this.worker.removeEventListener('message', handler);
          resolve(this.variant);
        } else if (ev.data?.type === 'error' && ev.data.fatal) {
          this.worker.removeEventListener('message', handler);
          reject(new Error(ev.data.msg));
        }
      };
      this.worker.addEventListener('message', handler);
      // Surface load-time failures (404, syntax error, uncaught throw before
      // the worker can postMessage) — otherwise the ready promise hangs
      // forever and the page sits on "awaiting connection…".
      this.worker.addEventListener('error', (ev) => {
        this.worker.removeEventListener('message', handler);
        reject(new Error(`worker load: ${ev.message || 'unknown'} (${ev.filename}:${ev.lineno})`));
      });
      this.worker.addEventListener('messageerror', () => {
        this.worker.removeEventListener('message', handler);
        reject(new Error('worker messageerror'));
      });
    });

    this.worker.addEventListener('message', ({ data }) => {
      switch (data?.type) {
        case 'frame':
          this.onFrame(data);
          break;
        case 'stats':
          this.onStats(data);
          break;
        case 'error':
          this.onError(data);
          break;
        // 'ready' is consumed by the ready promise above.
      }
    });

    // Tracks in-flight drain() resolvers so close() can flush them — once
    // the worker is terminated, 'drained' replies never arrive and the
    // promises would otherwise hang forever.
    this._closed         = false;
    this._pendingDrains  = new Set();

    this.worker.postMessage({
      type: 'init',
      wasmBase: absBase,
      variant, threadCount, reduceNL, output,
    });
  }

  setReduceNL(n) { this.worker.postMessage({ type: 'setReduceNL', value: n | 0 }); }

  pushPacket(bytes) {
    // Transfer when possible (caller-owned buffer), else copy.  The worker
    // re-wraps into a Uint8Array at the receive side.
    if (bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength) {
      this.worker.postMessage({ type: 'packet', bytes: bytes.buffer }, [bytes.buffer]);
    } else {
      // Sliced view of a larger buffer — copy, can't transfer a partial.
      const copy = new Uint8Array(bytes.byteLength);
      copy.set(bytes);
      this.worker.postMessage({ type: 'packet', bytes: copy.buffer }, [copy.buffer]);
    }
  }

  reset() { this.worker.postMessage({ type: 'reset' }); }

  // Wait for the worker to drain in-flight packets and emit any frames they
  // produce.  Resolves once the worker echoes 'drained' back.  postMessage
  // is FIFO, so all 'frame' messages posted before 'drained' have already
  // been delivered to onFrame by the time this promise resolves.
  //
  // No internal timeout: the drain has to wait as long as the decoder needs
  // to chew through the backlog (tens of seconds is normal for a multi-second
  // 1080p+ clip).  An earlier 5 s default silently truncated playback.
  // The Stop case is handled by close() below, which flushes any pending
  // drain promises before terminating the worker.
  drain() {
    return new Promise((resolve) => {
      if (this._closed) { resolve(); return; }
      let finish;
      const handler = (ev) => {
        if (ev.data?.type === 'drained') finish();
      };
      finish = () => {
        this.worker.removeEventListener('message', handler);
        this._pendingDrains.delete(finish);
        resolve();
      };
      this._pendingDrains.add(finish);
      this.worker.addEventListener('message', handler);
      this.worker.postMessage({ type: 'drain' });
    });
  }

  close() {
    if (this._closed) return;
    this._closed = true;
    // Flush any drain() promises waiting on a 'drained' that won't arrive
    // (terminate() kills the worker; 'drained' replies in flight are lost).
    // Snapshot the Set first because each finish() removes itself.
    for (const finish of [...this._pendingDrains]) finish();
    this.worker.postMessage({ type: 'close' });
    this.worker.terminate();
  }
}
