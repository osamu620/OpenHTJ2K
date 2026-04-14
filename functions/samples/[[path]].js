// Pages Function that streams files out of the bound R2 bucket.
//
// Routes every request under /samples/* to the R2 object with the same key.
// Same-origin with the rest of htj2k-demo.pages.dev, so no CORS is needed
// — the demo's fetch() doesn't trigger a preflight and the page's existing
// COOP/COEP headers apply.
//
// Handles HTTP range requests so the demo's Blob.stream()-style consumer
// can apply backpressure without downloading the whole file up front.
//
// R2 binding required (set in the Pages project's Settings → Functions →
// R2 Bucket Bindings):
//   Variable name: SAMPLES_BUCKET
//   R2 bucket:     htj2k-samples
//
// Public access on the r2.dev subdomain is NOT required with this setup —
// the Function is bound directly to the bucket.  You can revoke r2.dev
// public access after this is deployed for tighter security.

export async function onRequestGet(context) {
  const { request, env, params } = context;

  // [[path]] matches one or more path segments; normalise to the R2 object key.
  const key = Array.isArray(params.path) ? params.path.join('/') : String(params.path);

  // Forward the client's Range header (if any) to R2 so we get a partial object.
  const range = parseRange(request.headers.get('range'));

  const object = await env.SAMPLES_BUCKET.get(key, range ? { range } : undefined);
  if (!object) {
    return new Response('Not Found: ' + key, { status: 404 });
  }

  const headers = new Headers();
  // Writes Content-Type, Content-Language, Content-Disposition, Content-Encoding,
  // Cache-Control, and Content-Length from the R2 object's httpMetadata.
  object.writeHttpMetadata(headers);
  headers.set('etag', object.httpEtag);
  headers.set('accept-ranges', 'bytes');
  // Aggressive cache hint — these .rtp files are immutable test clips.
  headers.set('cache-control', 'public, max-age=86400, immutable');

  let status = 200;
  if (object.range) {
    const { offset, length } = object.range;
    headers.set(
      'content-range',
      `bytes ${offset}-${offset + length - 1}/${object.size}`
    );
    status = 206;
  }

  return new Response(object.body, { status, headers });
}

// Minimal RFC 7233 Range parser.  Handles `bytes=NNN-MMM`, `bytes=NNN-`,
// and `bytes=-NNN` (suffix).  Multi-range requests aren't supported (R2 only
// accepts a single range per call; the browser's fetch() / Blob.stream()
// path always emits a single contiguous range so this is fine in practice).
function parseRange(header) {
  if (!header || !header.toLowerCase().startsWith('bytes=')) return null;
  const spec = header.slice(6).split(',')[0].trim();
  const dash = spec.indexOf('-');
  if (dash < 0) return null;
  const startStr = spec.slice(0, dash);
  const endStr   = spec.slice(dash + 1);
  if (!startStr && endStr) {
    // bytes=-N (last N bytes)
    const n = parseInt(endStr, 10);
    if (!Number.isFinite(n) || n <= 0) return null;
    return { suffix: n };
  }
  const start = parseInt(startStr, 10);
  if (!Number.isFinite(start) || start < 0) return null;
  if (!endStr) {
    return { offset: start };
  }
  const end = parseInt(endStr, 10);
  if (!Number.isFinite(end) || end < start) return null;
  return { offset: start, length: end - start + 1 };
}
