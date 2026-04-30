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
