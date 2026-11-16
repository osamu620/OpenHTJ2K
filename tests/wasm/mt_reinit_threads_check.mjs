// Checks that a multi-threaded WASM decoder keeps using the shared thread pool after it is
// re-initialised with reset_decoder_with_bytes(), as the RTP demo, the WebTransport viewer
// and any other caller that decodes a sequence of frames with one decoder do.
//
// From 0.25.0 to 0.28.0 the wrapper re-initialised with num_threads = 1, which decodes on the
// calling thread only, so every frame after the first ran single-threaded: about half the
// speed, with identical output, which no conformance test could see.  This test counts the
// tasks that the pool's worker threads run (pool_worker_tasks()) around each decode instead
// of timing anything, so it is deterministic.
//
// Usage: node mt_reinit_threads_check.mjs <dir with libopen_htj2k_mt_simd.{js,wasm}> <codestream>
// Prints "all cases passed" and exits 0 on success.
import { readFileSync } from 'fs';
import { dirname, join, resolve } from 'path';
import { createRequire } from 'module';
import { pathToFileURL } from 'url';

const [dirArg, csPath] = process.argv.slice(2);
if (!dirArg || !csPath) {
  console.error('usage: node mt_reinit_threads_check.mjs <wasm dir> <codestream>');
  process.exit(1);
}

// The Emscripten module mixes `__dirname` with ES-module syntax, which Node will not import
// directly; evaluate it as a function instead (as web/wasm_bench.mjs does).  Newer Emscripten
// appends a pthread self-start tail after the export, older versions use a separate
// *.worker.js; the pthread workers load the original file either way.
function loadFactory(p) {
  let src = readFileSync(p, 'utf-8');
  src = src.replace(/import\.meta\.url/g, JSON.stringify(pathToFileURL(p).href));
  src = src.replace(/;?\s*var isPthread\s*=\s*globalThis[\s\S]*$/, '');
  src = src.replace(/export\s+default\s+Module\s*;?\s*$/, '');
  return new Function('require', '__filename', '__dirname', src + '\nreturn Module;')(
      createRequire(p), p, dirname(p));
}
const dir = resolve(dirArg);
const M = await loadFactory(join(dir, 'libopen_htj2k_mt_simd.js'))(
    { wasmBinary: readFileSync(join(dir, 'libopen_htj2k_mt_simd.wasm')) });
if (typeof M._pool_worker_tasks !== 'function') {
  console.log(`FAIL: ${dir} has no pool_worker_tasks() export: the WASM build is older than this ` +
              'test; rebuild it (cmake --build web/build)');
  process.exit(1);
}
const F = {
  create:    M.cwrap('create_decoder', 'number', ['number', 'number', 'number']),
  create_mt: M.cwrap('create_decoder_mt', 'number', ['number', 'number', 'number', 'number']),
  reset:     M.cwrap('reset_decoder_with_bytes', 'void', ['number', 'number', 'number', 'number']),
  parse:     M.cwrap('parse_j2c_data', 'void', ['number']),
  planar_u8: M.cwrap('invoke_decoder_planar_u8', 'void', ['number', 'number', 'number', 'number']),
  width:     M.cwrap('get_width', 'number', ['number', 'number']),
  height:    M.cwrap('get_height', 'number', ['number', 'number']),
  release:   M.cwrap('release_j2c_data', 'void', ['number']),
  tasks:     M.cwrap('pool_worker_tasks', 'number', []),
};

const cs = readFileSync(csPath);
const csPtr = M._malloc(cs.length + 16);  // 16 bytes of padding for SIMD reads
M.HEAPU8.set(cs, csPtr);
const planes = [0, 0, 0];

// Parses and decodes; returns the number of pool tasks the decode ran.
function decode(dec) {
  F.parse(dec);
  if (!planes[0]) {
    const n = F.width(dec, 0) * F.height(dec, 0);
    for (let c = 0; c < 3; c++) planes[c] = M._malloc(n);
  }
  const before = F.tasks();
  F.planar_u8(dec, planes[0], planes[1], planes[2]);
  return F.tasks() - before;
}

let failed = 0;
function check(what, ok, detail) {
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${what}: ${detail}`);
  if (!ok) failed++;
}

const THREADS = 4;
const mt = F.create_mt(csPtr, cs.length, 0, THREADS);
const first = decode(mt);
check(`create_decoder_mt(${THREADS}) uses the pool`, first > 0,
      `${first} pool tasks (0 means the codestream is too small to use the pool)`);
for (let frame = 2; frame <= 3; frame++) {
  F.reset(mt, csPtr, cs.length, 0);
  const n = decode(mt);
  check(`frame ${frame} after reset_decoder_with_bytes() uses the pool`, n > 0, `${n} pool tasks`);
}
F.release(mt);

const st = F.create(csPtr, cs.length, 0);
const single = decode(st);
check('create_decoder() decodes on the calling thread', single === 0, `${single} pool tasks`);
F.reset(st, csPtr, cs.length, 0);
const singleAgain = decode(st);
check('and still does after reset_decoder_with_bytes()', singleAgain === 0, `${singleAgain} pool tasks`);
F.release(st);

if (failed) {
  console.log(`${failed} case(s) failed`);
  process.exit(1);
}
console.log('all cases passed');
process.exit(0);
