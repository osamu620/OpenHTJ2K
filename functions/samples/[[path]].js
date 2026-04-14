// Pages Function that proxies /samples/* to the R2 bucket's public r2.dev
// URL.  Same origin as the rest of htj2k-demo.pages.dev from the browser's
// perspective, so the demo's fetch() works without CORS complications.
//
// We use fetch() over HTTPS (not an R2 binding) specifically because Pages
// bindings have to be configured in the dashboard, and that UI has shifted
// enough across 2025-2026 that it's hard to find.  fetch() of r2.dev is
// slightly slower than a direct R2 binding (one extra Cloudflare internal
// hop) but avoids the dashboard step entirely, which is the simpler
// tradeoff for a public demo.
//
// Both the client→Pages and Pages→r2.dev legs stay on Cloudflare's edge,
// so the total path is still much faster than S3 direct from outside the
// US West Coast.
//
// Range requests: the client's Range header is forwarded to r2.dev verbatim,
// and r2.dev's 206 Partial Content response is passed back through.  This
// keeps Blob.stream() / fetch-body backpressure working end-to-end.

const R2_PUBLIC_URL =
  'https://pub-21f3cc3ea54a4a65b2d083c2139002d6.r2.dev';

export async function onRequestGet(context) {
  const { request, params } = context;

  const key = Array.isArray(params.path) ? params.path.join('/') : String(params.path);
  const upstream = `${R2_PUBLIC_URL}/${key}`;

  // Forward Range (if any).  Don't forward the browser's own headers
  // wholesale — r2.dev doesn't need Cookie/Accept-Language/etc., and
  // stripping them lets Cloudflare cache the upstream response more
  // aggressively across all visitors.
  const upstreamHeaders = new Headers();
  const range = request.headers.get('range');
  if (range) upstreamHeaders.set('range', range);

  const upstreamRes = await fetch(upstream, {
    method:  'GET',
    headers: upstreamHeaders,
    // The Cloudflare fetch() implementation honors these hints; defaults are fine.
    cf: { cacheEverything: true, cacheTtl: 86400 },
  });

  // Copy response headers, then overwrite with our CORS / cache / ranges.
  // We explicitly set Access-Control-Allow-Origin so same-origin works even
  // if the request is ever hit cross-origin (e.g. from a future subdomain
  // demo page).  The page's own COOP/COEP come from _headers.
  const headers = new Headers(upstreamRes.headers);
  headers.set('access-control-allow-origin', '*');
  headers.set('accept-ranges', 'bytes');
  headers.set('cache-control', 'public, max-age=86400, immutable');

  return new Response(upstreamRes.body, {
    status:     upstreamRes.status,   // preserves 200 / 206 / 404
    statusText: upstreamRes.statusText,
    headers,
  });
}
