// Service worker that enables Cross-Origin Isolation by injecting COOP/COEP
// headers.  Required for SharedArrayBuffer (WASM pthreads) on pages served
// from static hosting without server-side header control.
//
// Based on https://github.com/nicolo-ribaudo/coi-serviceworker (MIT).

"use strict";

self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));

self.addEventListener("fetch", (e) => {
  // only-if-cached + !same-origin causes a TypeError in some browsers
  if (e.request.cache === "only-if-cached" && e.request.mode !== "same-origin")
    return;

  e.respondWith(
    fetch(e.request)
      .then((res) => {
        // opaque responses (status 0) cannot have headers modified
        if (res.status === 0) return res;

        // 304 Not Modified responses MUST be passed through unchanged.
        // The browser pairs a 304 with its disk-cache entry to fill in
        // the body; constructing a fresh `new Response(res.body, …)` here
        // gives the browser an *empty* body it can't pair with the cache,
        // and any consumer (especially `new Worker(url)`) ends up loading
        // empty content and silently failing.  This is what broke the
        // multi-threaded WASM mode for pthread Worker spawns.
        if (res.status === 304) return res;

        // If the response already carries COOP/COEP (e.g. when the host
        // sets them via response headers), pass through unchanged — there
        // is no value in proxying through a fresh Response, and doing so
        // adds bug surface (see 304 case above).
        if (res.headers.get("Cross-Origin-Embedder-Policy") &&
            res.headers.get("Cross-Origin-Opener-Policy")) {
          return res;
        }

        const headers = new Headers(res.headers);
        headers.set("Cross-Origin-Embedder-Policy", "credentialless");
        headers.set("Cross-Origin-Opener-Policy", "same-origin");
        return new Response(res.body, {
          status: res.status,
          statusText: res.statusText,
          headers,
        });
      })
      .catch((err) => console.error("COI fetch error:", err))
  );
});
