#!/usr/bin/env node
// WASM decoder benchmark: loads a chosen variant once, decodes the same
// codestream N times, reports min/median/mean/p95 wall-clock + throughput.
//
// Usage:
//   node wasm_bench.mjs -i <input.j2c> [--variant scalar|simd|mt|mt_simd] \
//                       [--threads N] [--iters N] [--build-dir <path>] [--warmup N]
//
// Defaults: variant=simd, iters=20, warmup=3, threads=1.

import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { createRequire } from 'module';

function parseArgs() {
  const a = process.argv.slice(2);
  const o = {
    input: null, variant: 'simd', threads: 1, iters: 20, warmup: 3,
    buildDir: null, reduce: 0,
  };
  for (let i = 0; i < a.length; i++) {
    const v = a[i];
    if ((v === '-i' || v === '--input') && i + 1 < a.length) o.input = a[++i];
    else if (v === '--variant' && i + 1 < a.length) o.variant = a[++i];
    else if (v === '--threads' && i + 1 < a.length) o.threads = parseInt(a[++i], 10);
    else if (v === '--iters' && i + 1 < a.length) o.iters = parseInt(a[++i], 10);
    else if (v === '--warmup' && i + 1 < a.length) o.warmup = parseInt(a[++i], 10);
    else if (v === '--build-dir' && i + 1 < a.length) o.buildDir = a[++i];
    else if (v === '--reduce' && i + 1 < a.length) o.reduce = parseInt(a[++i], 10);
  }
  if (!o.input) { console.error('usage: -i <file> [--variant simd|mt_simd|scalar|mt] [--threads N] [--iters N] [--warmup N] [--build-dir dir]'); process.exit(1); }
  const validVariants = ['scalar', 'simd', 'mt', 'mt_simd'];
  if (!validVariants.includes(o.variant)) {
    console.error(`--variant must be one of ${validVariants.join(',')}`); process.exit(1);
  }
  return o;
}

const opts = parseArgs();
const __dir = dirname(fileURLToPath(import.meta.url));
const buildDir = opts.buildDir || join(__dir, '..', 'build_wasm_prof', 'html');

const VARIANT_NAME = {
  scalar:  'libopen_htj2k',
  simd:    'libopen_htj2k_simd',
  mt:      'libopen_htj2k_mt',
  mt_simd: 'libopen_htj2k_mt_simd',
};
const moduleName = VARIANT_NAME[opts.variant];
const jsPath   = join(buildDir, `${moduleName}.js`);
const wasmPath = join(buildDir, `${moduleName}.wasm`);

// Same ESM-to-Function shim as open_htj2k_dec.mjs.
function loadEmscriptenFactory(jsPath) {
  let src = readFileSync(jsPath, 'utf-8');
  const fileUrl = 'file://' + jsPath;
  src = src.replace(/import\.meta\.url/g, JSON.stringify(fileUrl));
  src = src.replace(/;var isPthread=[\s\S]*$/, '');
  src = src.replace(/export\s+default\s+Module\s*;?\s*$/, '');
  const require = createRequire(jsPath);
  const fn = new Function('require', '__filename', '__dirname', src + '\nreturn Module;');
  return fn(require, jsPath, dirname(jsPath));
}

const wasmBinary = readFileSync(wasmPath);
const createModule = loadEmscriptenFactory(jsPath);
const M = await createModule({ wasmBinary });

const j2cData = readFileSync(opts.input);
const isMt = opts.variant === 'mt' || opts.variant === 'mt_simd';

function runOne() {
  const t0 = performance.now();
  const inPtr = M._malloc(j2cData.length);
  M.HEAPU8.set(j2cData, inPtr);
  const dec = isMt
    ? M._create_decoder_mt(inPtr, j2cData.length, opts.reduce, opts.threads)
    : M._create_decoder(inPtr, j2cData.length, opts.reduce);
  M._free(inPtr);
  M._parse_j2c_data(dec);
  const tParse = performance.now();

  const W = M._get_width(dec, 0);
  const H = M._get_height(dec, 0);
  const C = M._get_num_components(dec);
  const depth = M._get_depth(dec, 0);
  const rFactor = 1 << opts.reduce;
  const Wd = Math.ceil(W / rFactor);
  const Hd = Math.ceil(H / rFactor);
  const maxval = Math.min((1 << depth) - 1, 65535);
  const bytesPerSmp = maxval > 255 ? 2 : 1;
  const nBytes = Wd * Hd * C * bytesPerSmp;

  const outPtr = M._malloc(nBytes);
  M._invoke_decoder_stream(dec, outPtr, maxval, bytesPerSmp, 1);
  const tDec = performance.now();

  M._release_j2c_data(dec);
  M._free(outPtr);

  return {
    W: Wd, H: Hd, C, depth,
    parseMs:  tParse - t0,
    decodeMs: tDec - tParse,
    totalMs:  tDec - t0,
  };
}

// Warmup
for (let i = 0; i < opts.warmup; i++) runOne();

// Measured iterations
const samples = [];
let dims = null;
for (let i = 0; i < opts.iters; i++) {
  const s = runOne();
  if (!dims) dims = { W: s.W, H: s.H, C: s.C, depth: s.depth };
  samples.push(s);
}

function stats(arr) {
  const sorted = [...arr].sort((a, b) => a - b);
  const n = sorted.length;
  const q = (p) => sorted[Math.min(n - 1, Math.max(0, Math.floor(p * (n - 1))))];
  const mean = arr.reduce((a, b) => a + b, 0) / n;
  return {
    min:  sorted[0],
    p50:  q(0.5),
    p95:  q(0.95),
    max:  sorted[n - 1],
    mean,
  };
}

const totalStats  = stats(samples.map(s => s.totalMs));
const parseStats  = stats(samples.map(s => s.parseMs));
const decodeStats = stats(samples.map(s => s.decodeMs));

const nSamples = dims.W * dims.H * dims.C;
const mspsMean = nSamples / (totalStats.mean * 1e3);  // Msamples/s
const fpsMean  = 1000 / totalStats.mean;

console.log(JSON.stringify({
  variant: opts.variant,
  threads: opts.threads,
  input:   opts.input,
  dims,
  iters:   opts.iters,
  total_ms:  totalStats,
  parse_ms:  parseStats,
  decode_ms: decodeStats,
  throughput_msamples_per_s_mean: Number(mspsMean.toFixed(2)),
  fps_mean: Number(fpsMean.toFixed(2)),
}, null, 2));
