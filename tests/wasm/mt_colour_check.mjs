// Checks that the multi-threaded WASM decoder gives the same output on 4 threads as on one,
// for codestreams with a multi-component (colour) transform: there the inverse transform of
// each strip runs as thread-pool tasks, which the single-threaded conformance tests (the
// Node.js decoder's default) never reach.  Both entry points that decode through
// invoke_line_based_stream() are compared: invoke_decoder_stream() (the Node.js decoder) and
// invoke_decoder_to_rgba() (the still-image demo).
//
// Usage: node mt_colour_check.mjs <dir with libopen_htj2k_mt_simd.{js,wasm}> <codestream>...
// Prints "all cases passed" and exits 0 on success.
import { readFileSync } from 'fs';
import { basename, dirname, join, resolve } from 'path';
import { createRequire } from 'module';
import { pathToFileURL } from 'url';

const [dirArg, ...files] = process.argv.slice(2);
if (!dirArg || files.length === 0) {
  console.error('usage: node mt_colour_check.mjs <wasm dir> <codestream>...');
  process.exit(1);
}

// The same loader as mt_reinit_threads_check.mjs.
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
const THREADS = 4;

// Decodes cs on `threads` threads (1: create_decoder(), on the calling thread) through
// entry ('stream' or 'rgba'); returns a copy of the output.
function decode(cs, threads, entry) {
  const src = M._malloc(cs.length + 16);  // 16 bytes of padding for SIMD reads
  M.HEAPU8.set(cs, src);
  const dec = threads > 1 ? M._create_decoder_mt(src, cs.length, 0, threads)
                          : M._create_decoder(src, cs.length, 0);
  M._parse_j2c_data(dec);
  const w = M._get_width(dec, 0), h = M._get_height(dec, 0), nc = M._get_num_components(dec);
  const depth = M._get_depth(dec, 0);
  let bytes, dst;
  if (entry === 'stream') {
    const bps = depth > 8 ? 2 : 1;
    bytes = w * h * nc * bps;
    dst = M._malloc(bytes);
    M._invoke_decoder_stream(dec, dst, (1 << depth) - 1, bps, 1);
  } else {
    bytes = w * h * 4;
    dst = M._malloc(bytes);
    M._invoke_decoder_to_rgba(dec, dst);
  }
  const out = M.HEAPU8.slice(dst, dst + bytes);
  M._release_j2c_data(dec);
  M._free(dst);
  M._free(src);
  return out;
}

let failed = 0;
for (const f of files) {
  const cs = readFileSync(f);
  for (const entry of ['stream', 'rgba']) {
    const one = decode(cs, 1, entry), many = decode(cs, THREADS, entry);
    const same = one.length === many.length && Buffer.compare(Buffer.from(one), Buffer.from(many)) === 0;
    console.log(`${same ? 'ok  ' : 'FAIL'} ${basename(f)} ${entry}: ${THREADS} threads ${same ? '=' : '!='} 1 thread`);
    if (!same) failed++;
  }
}
if (failed) {
  console.log(`${failed} case(s) failed`);
  process.exit(1);
}
console.log('all cases passed');
process.exit(0);
