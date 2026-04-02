#!/usr/bin/env node
// HTJ2K / JPEG 2000 decoder CLI — runs the WASM build of OpenHTJ2K in Node.js.
// Usage: node decode_cli.mjs -i <input.j2c|.j2k|.jph> -o <output.ppm|.pgm> [-r <reduce_NL>]
//
// Output format is auto-selected:
//   1-component → PGM (P5 binary), 3-component → PPM (P6 binary)
//   depth ≤ 8 → 8-bit samples (maxval 255), depth 9–16 → 16-bit big-endian samples

import { readFileSync, writeFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join, extname } from 'path';

// ── Argument parsing ─────────────────────────────────────────────────────────
function parseArgs() {
  const args = process.argv.slice(2);
  let input = null, output = null, reduce = 0;
  for (let i = 0; i < args.length; i++) {
    if ((args[i] === '-i' || args[i] === '--input') && i + 1 < args.length)
      input = args[++i];
    else if ((args[i] === '-o' || args[i] === '--output') && i + 1 < args.length)
      output = args[++i];
    else if ((args[i] === '-r' || args[i] === '--reduce') && i + 1 < args.length)
      reduce = parseInt(args[++i], 10);
    else if (args[i] === '-h' || args[i] === '--help') {
      console.log('Usage: node decode_cli.mjs -i <input.j2c> -o <output.ppm> [-r <reduce_NL>]');
      console.log('  -i, --input   Input J2C/J2K/JPH file');
      console.log('  -o, --output  Output PPM (RGB) or PGM (grayscale) file');
      console.log('  -r, --reduce  Resolution reduction (0 = full, 1 = half, ...)');
      process.exit(0);
    }
  }
  return { input, output, reduce };
}

const { input, output, reduce } = parseArgs();
if (!input || !output) {
  console.error('Error: -i and -o are required.');
  console.error('Usage: node decode_cli.mjs -i <input.j2c> -o <output.ppm> [-r <reduce_NL>]');
  process.exit(1);
}

// ── Load WASM module (prefer SIMD build) ─────────────────────────────────────
const __dir = dirname(fileURLToPath(import.meta.url));
const simdPath = join(__dir, 'build/html/libopen_htj2k_simd.js');
const scalarPath = join(__dir, 'build/html/libopen_htj2k.js');

let M;
try {
  const { default: createModule } = await import(simdPath);
  M = await createModule();
} catch {
  const { default: createModule } = await import(scalarPath);
  M = await createModule();
}

// ── Read input file ──────────────────────────────────────────────────────────
let j2cData;
try {
  j2cData = readFileSync(input);
} catch (e) {
  console.error(`Error reading input file: ${e.message}`);
  process.exit(1);
}

// ── Decode ───────────────────────────────────────────────────────────────────
const t0 = performance.now();

// Copy J2C bytes into WASM heap
const inPtr = M._malloc(j2cData.length);
if (!inPtr) { console.error('WASM malloc failed for input buffer'); process.exit(1); }
M.HEAPU8.set(j2cData, inPtr);

// create_decoder(data, size, reduce_NL)
const dec = M._create_decoder(inPtr, j2cData.length, reduce);
M._free(inPtr);
if (!dec) { console.error('create_decoder returned null'); process.exit(1); }

try {
  M._parse_j2c_data(dec);
} catch (e) {
  console.error(`Parse error: ${e}`);
  M._release_j2c_data(dec);
  process.exit(1);
}

const W = M._get_width(dec, 0);
const H = M._get_height(dec, 0);
const C = M._get_num_components(dec);
const depth = M._get_depth(dec, 0);

// get_width/get_height return full-resolution dimensions. When --reduce is
// used, the decoder outputs a downsampled image: ceil(dim / 2^reduce).
const rFactor = 1 << reduce;
const Wd = Math.ceil(W / rFactor);
const Hd = Math.ceil(H / rFactor);

const maxval      = Math.min((1 << depth) - 1, 65535);
const bytesPerSmp = maxval > 255 ? 2 : 1;
const magic       = C === 1 ? 'P5' : 'P6';
const header      = `${magic}\n${Wd} ${Hd}\n${maxval}\n`;
const headerBuf   = Buffer.from(header, 'ascii');

// Allocate packed output buffer (W × H × C × bytes_per_sample).
// invoke_decoder_stream() writes directly here, avoiding a separate 96MB int32 buffer.
const nSamples  = Wd * Hd * C;
const packedPtr = M._malloc(nSamples * bytesPerSmp);
if (!packedPtr) {
  console.error('WASM malloc failed for packed output buffer');
  M._release_j2c_data(dec);
  process.exit(1);
}

try {
  M._invoke_decoder_stream(dec, packedPtr, maxval, bytesPerSmp);
} catch (e) {
  console.error(`Decode error: ${e}`);
  M._free(packedPtr);
  M._release_j2c_data(dec);
  process.exit(1);
}
M._release_j2c_data(dec);

const t1 = performance.now();

// Snapshot packed bytes before freeing (copy out of WASM heap).
// Re-read HEAPU8: ALLOW_MEMORY_GROWTH may have replaced the backing buffer.
const pixelBuf = Buffer.from(
  M.HEAPU8.buffer.slice(packedPtr, packedPtr + nSamples * bytesPerSmp)
);
M._free(packedPtr);

try {
  writeFileSync(output, Buffer.concat([headerBuf, pixelBuf]));
} catch (e) {
  console.error(`Error writing output file: ${e.message}`);
  process.exit(1);
}

const elapsed = (t1 - t0).toFixed(1);
const fmt     = C === 1 ? 'grayscale' : 'RGB';
console.log(`Decoded ${Wd}×${Hd} ${fmt} ${depth}bpc in ${elapsed} ms → ${output}`);
