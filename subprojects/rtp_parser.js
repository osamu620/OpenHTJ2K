// .rtp file parser — "Spark fixture" format (see
//   source/apps/rtp_recv/tools/rtp_file_replay.py):
//     [0xFFFF marker (2B, BE)] [length (2B, BE)] [RTP packet of that length] ...
// Async generator that yields one packet at a time from a fetch() ReadableStream,
// without waiting for the whole file to download first.

const MARKER = 0xFFFF;

// Append-only growable buffer with a rolling read offset.  When the offset
// exceeds half the backing length we shift-compact so memory doesn't grow
// unboundedly on long streams.  Amortised O(N) — avoids the O(N²) trap of
// `new Uint8Array(old.length + chunk.length); dst.set(old); dst.set(chunk)`.
class RollingBuffer {
  constructor(initialCap = 1 << 20) {
    this._buf  = new Uint8Array(initialCap);
    this._head = 0;   // read offset
    this._tail = 0;   // write offset
  }
  get length() { return this._tail - this._head; }
  bytes(off = 0, len = undefined) {
    const end = len === undefined ? this._tail : this._head + off + len;
    return this._buf.subarray(this._head + off, end);
  }
  consume(n) {
    this._head += n;
    // Compact once we've consumed more than half of the backing array.
    if (this._head > (this._buf.length >> 1)) {
      this._buf.copyWithin(0, this._head, this._tail);
      this._tail -= this._head;
      this._head = 0;
    }
  }
  append(chunk) {
    const need = this._tail + chunk.length;
    if (need > this._buf.length) {
      let cap = this._buf.length;
      while (cap < need) cap *= 2;
      const grown = new Uint8Array(cap);
      grown.set(this._buf.subarray(0, this._tail));
      this._buf = grown;
    }
    this._buf.set(chunk, this._tail);
    this._tail += chunk.length;
  }
}

// Quick-parse the fields we need for JS-side pacing (the WASM side re-parses
// everything it needs authoritatively).  Assumes `pkt` is a view over the
// raw RTP datagram starting at byte 0.
function quickParseRtp(pkt) {
  return {
    seq      : (pkt[2] << 8) | pkt[3],
    timestamp: ((pkt[4] << 24) | (pkt[5] << 16) | (pkt[6] << 8) | pkt[7]) >>> 0,
    marker   : (pkt[1] & 0x80) !== 0,
  };
}

/**
 * Async generator over RTP packets in a .rtp file.
 * @param {string | Uint8Array | ReadableStream<Uint8Array>} source
 *     URL to fetch, a Uint8Array containing the file bytes, or a ReadableStream.
 * @yields {{bytes: Uint8Array, seq: number, timestamp: number, marker: boolean}}
 *     `bytes` is valid only until the next `next()` call — copy it if you need
 *     to retain the data.
 */
export async function* parseRtpStream(source, opts = {}) {
  const reader = await sourceToReader(source);
  const buf    = new RollingBuffer();
  let   eof    = false;
  // Optional callback invoked once per chunk received from the underlying
  // stream — lets the caller measure disk-read rate vs. packet-consumption rate.
  const onChunk = typeof opts.onChunk === 'function' ? opts.onChunk : null;

  async function fillAtLeast(nBytes) {
    while (buf.length < nBytes && !eof) {
      const { value, done } = await reader.read();
      if (done) { eof = true; break; }
      if (value && value.length) {
        buf.append(value);
        if (onChunk) onChunk(value.length);
      }
    }
  }

  try {
    while (true) {
      await fillAtLeast(4);
      if (buf.length < 4) return;          // clean EOF
      const hdr    = buf.bytes(0, 4);
      const marker = (hdr[0] << 8) | hdr[1];
      if (marker !== MARKER) {
        throw new Error(`parseRtpStream: bad marker 0x${marker.toString(16)} at stream offset`);
      }
      const length = (hdr[2] << 8) | hdr[3];
      await fillAtLeast(4 + length);
      if (buf.length < 4 + length) {
        throw new Error(`parseRtpStream: truncated packet (need ${length} bytes)`);
      }
      const pkt = buf.bytes(4, length);   // view, lifetime = until next consume()
      const q   = quickParseRtp(pkt);
      yield { bytes: pkt, ...q };
      buf.consume(4 + length);
    }
  } finally {
    // Called on normal completion, exception, or when the consumer calls
    // iterator.return() (for-await `break`).  Cancel the underlying stream
    // so the Blob's file reader / fetch network socket closes and nothing
    // leaks across play→stop→play cycles.
    try { await reader.cancel(); } catch (e) { /* ignore */ }
    try { reader.releaseLock(); }  catch (e) { /* ignore */ }
  }
}

async function sourceToReader(source) {
  if (typeof source === 'string') {
    const res = await fetch(source);
    if (!res.ok) throw new Error(`parseRtpStream: HTTP ${res.status} for ${source}`);
    return res.body.getReader();
  }
  if (source instanceof Uint8Array) {
    // Wrap in a one-shot ReadableStream so the iterator code path is unified.
    let emitted = false;
    return new ReadableStream({
      pull(controller) {
        if (emitted) { controller.close(); return; }
        controller.enqueue(source);
        emitted = true;
      },
    }).getReader();
  }
  if (source && typeof source.getReader === 'function') {
    return source.getReader();
  }
  throw new Error('parseRtpStream: source must be URL string, Uint8Array, or ReadableStream');
}
