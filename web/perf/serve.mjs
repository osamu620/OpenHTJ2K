#!/usr/bin/env node
// Tiny static server with COOP/COEP headers for SharedArrayBuffer (multi-
// threaded WASM).  Serves /web/perf/* + /subprojects/build_wt/html/* +
// /fixtures/* (mapped to ~/Documents/data/videos).
//
// Usage:
//   node web/perf/serve.mjs [port]            # default 8765
//   node web/perf/serve.mjs --bind 0.0.0.0    # bind to all interfaces

import http from 'http';
import { createReadStream, statSync } from 'fs';
import { extname, join, normalize, resolve } from 'path';
import { fileURLToPath } from 'url';
import os from 'os';

const REPO     = resolve(fileURLToPath(import.meta.url), '..', '..', '..');
const PORT     = parseInt(process.argv.find(a => /^\d+$/.test(a)) || '8765', 10);
const BIND_ALL = process.argv.includes('--bind');

const ROUTES = {
  '/perf/':    join(REPO, 'web', 'perf'),
  '/viewer/':  join(REPO, 'web', 'viewer'),
  '/wasm/':    join(REPO, 'subprojects', 'build_wt', 'html'),
  '/fixtures/':resolve(os.homedir(), 'Documents', 'data', 'videos'),
};

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js':   'text/javascript',
  '.mjs':  'text/javascript',
  '.wasm': 'application/wasm',
  '.json': 'application/json',
  '.rtp':  'application/octet-stream',
};

const server = http.createServer((req, res) => {
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
  if (url === '/viewer' || url === '/viewer/')              url = '/viewer/index.html';

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
});

const host = BIND_ALL ? '0.0.0.0' : '127.0.0.1';
server.listen(PORT, host, () => {
  console.log(`http://${host === '0.0.0.0' ? 'localhost' : host}:${PORT}/perf/`);
});
