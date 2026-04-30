#!/usr/bin/env node
// Tiny static server with COOP/COEP headers for SharedArrayBuffer (multi-
// threaded WASM).  Serves web/perf/* + web/wt_viewer/* + web/build_wt/html/*
// + fixture files under ~/Documents/data/videos (via /fixtures/).
//
// Usage:
//   node web/perf/serve.mjs [port]                    # HTTP, default 8765
//   node web/perf/serve.mjs --bind                    # listen on all interfaces
//   node web/perf/serve.mjs --cert C --key K          # HTTPS (PEM paths)
//
// HTTPS rationale: WebTransport is only available on secure-context pages.
// `http://localhost` qualifies; `http://<LAN-IP>` does not.  Serving over
// HTTPS — even with a self-signed cert the user has to click through —
// unblocks cross-LAN browsers without requiring a Chrome flag.

import http  from 'http';
import http2 from 'http2';
import { createReadStream, statSync, readFileSync } from 'fs';
import { extname, join, normalize, resolve } from 'path';
import { fileURLToPath } from 'url';
import os from 'os';

const REPO     = resolve(fileURLToPath(import.meta.url), '..', '..', '..');
const PORT     = parseInt(process.argv.find(a => /^\d+$/.test(a)) || '8765', 10);
const BIND_ALL = process.argv.includes('--bind');

function flag(name) {
  const i = process.argv.indexOf('--' + name);
  return i >= 0 && i + 1 < process.argv.length ? process.argv[i + 1] : null;
}
const CERT_PATH = flag('cert');
const KEY_PATH  = flag('key');
if ((CERT_PATH && !KEY_PATH) || (!CERT_PATH && KEY_PATH)) {
  console.error('--cert and --key must both be provided');
  process.exit(2);
}

const ROUTES = {
  '/perf/':      join(REPO, 'web', 'perf'),
  '/wt_viewer/': join(REPO, 'web', 'wt_viewer'),
  '/shared/':    join(REPO, 'web', 'shared'),
  '/wasm/':      join(REPO, 'web', 'build_wt', 'html'),
  '/fixtures/':  resolve(os.homedir(), 'Documents', 'data', 'videos'),
};

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js':   'text/javascript',
  '.mjs':  'text/javascript',
  '.wasm': 'application/wasm',
  '.json': 'application/json',
  '.rtp':  'application/octet-stream',
};

const handler = (req, res) => {
  // Cross-origin isolation headers for SharedArrayBuffer.
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
  res.setHeader('Cross-Origin-Resource-Policy', 'same-origin');
  res.setHeader('Cache-Control', 'no-store');

  // POST /report — page posts a JSON result here so headless Chrome runs
  // can be captured in stdout for shell automation.
  if (req.method === 'POST' && req.url.split('?')[0] === '/report') {
    let body = '';
    req.setEncoding('utf-8');
    req.on('data', c => { body += c; });
    req.on('end', () => {
      console.log('REPORT', body);
      res.writeHead(204).end();
    });
    return;
  }

  let url = decodeURIComponent(req.url.split('?')[0]);
  if (url === '/' || url === '/perf' || url === '/perf/')   url = '/perf/index.html';
  if (url === '/wt_viewer' || url === '/wt_viewer/')        url = '/wt_viewer/index.html';

  let filePath = null;
  for (const [prefix, dir] of Object.entries(ROUTES)) {
    if (url.startsWith(prefix)) {
      filePath = normalize(join(dir, url.slice(prefix.length)));
      if (!filePath.startsWith(dir)) { res.writeHead(403).end('forbidden'); return; }
      break;
    }
  }
  if (!filePath) { res.writeHead(404).end('not found'); return; }

  let st;
  try { st = statSync(filePath); } catch { res.writeHead(404).end('not found'); return; }
  if (st.isDirectory()) { res.writeHead(404).end('directory'); return; }

  const range = req.headers.range;
  const ext   = extname(filePath).toLowerCase();
  const ct    = MIME[ext] || 'application/octet-stream';
  res.setHeader('Accept-Ranges', 'bytes');
  res.setHeader('Content-Type', ct);
  if (range) {
    const m = /bytes=(\d+)-(\d*)/.exec(range);
    if (m) {
      const start = parseInt(m[1], 10);
      const end   = m[2] ? parseInt(m[2], 10) : st.size - 1;
      res.writeHead(206, {
        'Content-Range':  `bytes ${start}-${end}/${st.size}`,
        'Content-Length': end - start + 1,
      });
      createReadStream(filePath, { start, end }).pipe(res);
      return;
    }
  }
  res.writeHead(200, { 'Content-Length': st.size });
  createReadStream(filePath).pipe(res);
};

let server, scheme;
if (CERT_PATH) {
  // HTTPS via HTTP/2.  Critical for our use case: when the wt_viewer's
  // shared decoder Web Worker spawns N pthread inner workers, each one
  // independently fetches libopen_htj2k_mt_simd.{js,worker.js} during its
  // bootstrap.  Chrome's HTTP/1.1 connection cap is 6 per host, so even a
  // small worker pool can exceed it and silently lose half the inner
  // fetches.  HTTP/2 multiplexes everything over one TLS connection,
  // eliminating the cap entirely.  `allowHTTP1: true` keeps backward
  // compatibility for anything that doesn't speak h2.
  server = http2.createSecureServer({
    cert: readFileSync(CERT_PATH),
    key:  readFileSync(KEY_PATH),
    allowHTTP1: true,
  }, handler);
  scheme = 'https';
} else {
  server = http.createServer(handler);
  scheme = 'http';
}

const host = BIND_ALL ? '0.0.0.0' : '127.0.0.1';
server.listen(PORT, host, () => {
  console.log(`${scheme}://${host === '0.0.0.0' ? 'localhost' : host}:${PORT}/perf/`);
});
