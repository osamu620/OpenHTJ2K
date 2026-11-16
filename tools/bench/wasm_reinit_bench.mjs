// Times the WebAssembly decoder on the path the RTP demo and the WebTransport viewer use for
// every frame: one decoder, re-initialised with reset_decoder_with_bytes() and decoded with
// invoke_decoder_planar_u8().  Used by tools/bench/ab_speed.py; prints one JSON line.
//
// Usage: node wasm_reinit_bench.mjs <dir with libopen_htj2k_*.{js,wasm}> <mt_simd|simd>
//                                   <codestream> [frames=30] [threads=4]
// threads is ignored by the single-threaded "simd" variant.  The first 3 frames are not
// timed (decoder and thread start-up).
import { readFileSync } from 'fs';
import { dirname, join, resolve } from 'path';
import { createRequire } from 'module';
import { pathToFileURL } from 'url';
import { performance } from 'perf_hooks';

const [dirArg, variant, csPath, framesArg = '30', threadsArg = '4'] = process.argv.slice(2);
if (!dirArg || !['mt_simd', 'simd'].includes(variant) || !csPath) {
  console.error('usage: node wasm_reinit_bench.mjs <wasm dir> <mt_simd|simd> <codestream> [frames] [threads]');
  process.exit(1);
}
const frames = Math.max(4, +framesArg), threads = +threadsArg;

// Same loader as tests/wasm/mt_reinit_threads_check.mjs (the module mixes `__dirname` with
// ES-module syntax, so Node will not import it directly).
function loadFactory(p) {
  let src = readFileSync(p, 'utf-8');
  src = src.replace(/import\.meta\.url/g, JSON.stringify(pathToFileURL(p).href));
  src = src.replace(/;?\s*var isPthread\s*=\s*globalThis[\s\S]*$/, '');
  src = src.replace(/export\s+default\s+Module\s*;?\s*$/, '');
  return new Function('require', '__filename', '__dirname', src + '\nreturn Module;')(
      createRequire(p), p, dirname(p));
}
const dir = resolve(dirArg);
const name = `libopen_htj2k_${variant}`;
const M = await loadFactory(join(dir, `${name}.js`))({ wasmBinary: readFileSync(join(dir, `${name}.wasm`)) });
const mt = variant === 'mt_simd';
const F = {
  create:    M.cwrap('create_decoder', 'number', ['number', 'number', 'number']),
  create_mt: mt ? M.cwrap('create_decoder_mt', 'number', ['number', 'number', 'number', 'number']) : null,
  reset:     M.cwrap('reset_decoder_with_bytes', 'void', ['number', 'number', 'number', 'number']),
  parse:     M.cwrap('parse_j2c_data', 'void', ['number']),
  planar_u8: M.cwrap('invoke_decoder_planar_u8', 'void', ['number', 'number', 'number', 'number']),
  width:     M.cwrap('get_width', 'number', ['number', 'number']),
  height:    M.cwrap('get_height', 'number', ['number', 'number']),
  release:   M.cwrap('release_j2c_data', 'void', ['number']),
};

const cs = readFileSync(csPath);
const csPtr = M._malloc(cs.length + 16);
M.HEAPU8.set(cs, csPtr);
const dec = mt ? F.create_mt(csPtr, cs.length, 0, threads) : F.create(csPtr, cs.length, 0);
const planes = [];
const times = [];
for (let f = 0; f < frames; f++) {
  const t0 = performance.now();
  if (f > 0) F.reset(dec, csPtr, cs.length, 0);
  F.parse(dec);
  if (!planes.length) {
    const n = F.width(dec, 0) * F.height(dec, 0);
    for (let c = 0; c < 3; c++) planes.push(M._malloc(n));
  }
  F.planar_u8(dec, planes[0], planes[1], planes[2]);
  if (f >= 3) times.push(performance.now() - t0);
}
F.release(dec);
times.sort((a, b) => a - b);
console.log(JSON.stringify({ variant, threads: mt ? threads : 1, frames: times.length,
                             median_ms: times[times.length >> 1] }));
process.exit(0);
