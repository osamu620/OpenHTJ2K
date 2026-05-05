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

    // Batched-packet ingest staging.  pushPacket() copies each packet into
    // _batchBuf and records its end-offset in _batchOffs; the batch is
    // flushed (one postMessage) when it hits BATCH_PACKETS or BATCH_BYTES,
    // when drain()/reset()/close() is called, or by a deferred macrotask
    // so a low-rate producer's trailing partial batch reaches the worker
    // promptly.  Reduces postMessage overhead ~Nx for a high-bitrate stream
    // where N == batch size.
    //
    // Deferred flush uses MessageChannel — like fastYield in rtp_demo, this
    // is the cheapest macrotask we can schedule.  It must be a macrotask
    // (not queueMicrotask): a tight `for await` loop hits a microtask
    // boundary on every `await iter.next()`, so a microtask flush would
    // fire between every packet push and destroy the batching.  Macrotasks
    // wait until the loop yields control to the event loop, which is when
    // we actually want a partial batch to drain.
    this._BATCH_PACKETS = 16;
    this._BATCH_BYTES   = 24 * 1024;     // 16 packets × ~1500 B headroom
    this._batchBuf      = new Uint8Array(this._BATCH_BYTES);
    this._batchOffs     = [];
    this._batchLen      = 0;
    this._flushScheduled = false;
    this._flushChan     = new MessageChannel();
    this._flushChan.port1.onmessage = () => {
      this._flushScheduled = false;
      if (this._batchOffs.length > 0) this._flushBatch();
    };

    // SharedArrayBuffer plane ring: eliminates per-frame allocation by
    // reusing a fixed set of slots for decoded plane data.
    const SAB_SLOTS     = 3;
    const SAB_PLANE_MAX = 3840 * 2160;
    const SAB_SLOT_SIZE = SAB_PLANE_MAX * 3;
    const SAB_FLAG_BYTES = SAB_SLOTS * 4;
    const SAB_TOTAL     = SAB_FLAG_BYTES + SAB_SLOTS * SAB_SLOT_SIZE;
    if (typeof SharedArrayBuffer !== 'undefined' && self.crossOriginIsolated) {
      this._sab      = new SharedArrayBuffer(SAB_TOTAL);
      this._sabFlags = new Int32Array(this._sab, 0, SAB_SLOTS);
      this._sabSlots = SAB_SLOTS;
      this._sabPlaneMax = SAB_PLANE_MAX;
      this._sabSlotSize = SAB_SLOT_SIZE;
      this._sabFlagBytes = SAB_FLAG_BYTES;
    } else {
      this._sab = null;
      this._sabFlags = null;
    }

    this.worker.postMessage({
      type: 'init',
      wasmBase: absBase,
      variant, threadCount, reduceNL, output,
      sab: this._sab,
    });
  }

  get sab()      { return this._sab; }
  get sabFlags() { return this._sabFlags; }
  get sabSlotSize()  { return this._sabSlotSize; }
  get sabPlaneMax()  { return this._sabPlaneMax; }
  get sabFlagBytes() { return this._sabFlagBytes; }

  setReduceNL(n) {
    this._flushBatch();
    this.worker.postMessage({ type: 'setReduceNL', value: n | 0 });
  }

  setSkipInterval(n) {
    this._flushBatch();
    this.worker.postMessage({ type: 'setSkipInterval', value: n | 0 });
  }

  pushPacket(bytes) {
    const len = bytes.byteLength;
    // Defensive: an oversized packet shouldn't reach here (rtp_demo caps at
    // 2048), but if it does, send it as its own batch rather than corrupting
    // the staging buffer.
    if (len > this._BATCH_BYTES) {
      this._flushBatch();
      const copy = new Uint8Array(len);
      copy.set(bytes);
      this.worker.postMessage(
        { type: 'packet_batch', bytes: copy.buffer, offsets: [len] },
        [copy.buffer]
      );
      return;
    }
    // Flush before this packet would overflow the staging buffer.
    if (this._batchLen + len > this._BATCH_BYTES ||
        this._batchOffs.length >= this._BATCH_PACKETS) {
      this._flushBatch();
    }
    this._batchBuf.set(bytes, this._batchLen);
    this._batchLen += len;
    this._batchOffs.push(this._batchLen);
    if (this._batchOffs.length >= this._BATCH_PACKETS) {
      this._flushBatch();
    } else if (!this._flushScheduled) {
      this._flushScheduled = true;
      this._flushChan.port2.postMessage(0);   // macrotask deferred flush
    }
  }

  _flushBatch() {
    if (this._batchOffs.length === 0) return;
    // Transfer the staging buffer as-is; the worker only reads up to the
    // last entry in `offsets`, so trailing bytes don't matter.  Allocate a
    // fresh staging buffer for the next batch (transferring detaches the
    // old one).  Avoiding a trim-copy saves ~BATCH_BYTES of memcpy per
    // flush — at sustained 30 000 packets/s that's tens of MB/s of memory
    // traffic we don't have to do.
    const buf     = this._batchBuf;
    const offsets = this._batchOffs;
    this._batchBuf  = new Uint8Array(this._BATCH_BYTES);
    this._batchOffs = [];
    this._batchLen  = 0;
    this.worker.postMessage(
      { type: 'packet_batch', bytes: buf.buffer, offsets },
      [buf.buffer]
    );
  }

  reset() {
    this._flushBatch();
    this.worker.postMessage({ type: 'reset' });
  }

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
      // Make sure any staged packets reach the worker before the 'drain'
      // sentinel — otherwise the worker would echo 'drained' while still
      // holding unprocessed packets in this client's batch buffer.
      this._flushBatch();
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
    // Drop any staged packets — they'd be processed by a worker we're about
    // to terminate.  Detach the flush MessageChannel so a pending notify
    // doesn't fire after we're closed.
    this._batchOffs = [];
    this._batchLen  = 0;
    this._flushChan.port1.onmessage = null;
    this._flushChan.port1.close();
    this._flushChan.port2.close();
    this.worker.postMessage({ type: 'close' });
    this.worker.terminate();
  }
}
