#!/usr/bin/env node
// HTJ2K / JPEG 2000 decoder CLI — runs the WASM build of OpenHTJ2K in Node.js.
// Usage: node open_htj2k_dec.mjs -i <input.j2c|.j2k|.jph> -o <output.ppm|.pgm|.pgx> [-r <reduce_NL>] [-ycbcr bt601|bt709]
//
// Output format is auto-selected:
//   1-component → PGM (P5 binary), 3-component → PPM (P6 binary)
//   depth ≤ 8 → 8-bit samples (maxval 255), depth 9–16 → 16-bit big-endian samples
//
// YCbCr→RGB conversion:
//   For .jph inputs whose colour specification box declares YCbCr (EnumCS=18),
//   BT.601 conversion is applied automatically when writing PPM output.
//   Use -ycbcr bt709 to override, or -ycbcr bt601 to force BT.601 explicitly.

import { readFileSync, writeFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join, extname } from 'path';
import { createRequire } from 'module';

// ── Argument parsing ─────────────────────────────────────────────────────────
function parseArgs() {
  const args = process.argv.slice(2);
  let input = null, output = null, reduce = 0, ycbcr = null;
  for (let i = 0; i < args.length; i++) {
    if ((args[i] === '-i' || args[i] === '--input') && i + 1 < args.length)
      input = args[++i];
    else if ((args[i] === '-o' || args[i] === '--output') && i + 1 < args.length)
      output = args[++i];
    else if ((args[i] === '-r' || args[i] === '--reduce') && i + 1 < args.length)
      reduce = parseInt(args[++i], 10);
    else if ((args[i] === '-ycbcr' || args[i] === '--ycbcr') && i + 1 < args.length) {
      ycbcr = args[++i];
      if (ycbcr !== 'bt601' && ycbcr !== 'bt709') {
        console.error(`Error: -ycbcr takes 'bt601' or 'bt709', got '${ycbcr}'.`);
        process.exit(1);
      }
    } else if (args[i] === '-h' || args[i] === '--help') {
      console.log('Usage: node open_htj2k_dec.mjs -i <input.j2c> -o <output.ppm> [-r <reduce_NL>] [-ycbcr bt601|bt709]');
      console.log('  -i, --input      Input J2C/J2K/JPH file');
      console.log('  -o, --output     Output PPM (RGB), PGM (grayscale), or PGX (per-component) file');
      console.log('  -r, --reduce     Resolution reduction (0 = full, 1 = half, ...)');
      console.log('  -ycbcr bt601|bt709  YCbCr→RGB conversion for PPM output (auto-detected from JPH)');
      process.exit(0);
    }
  }
  return { input, output, reduce, ycbcr };
}

const { input, output, reduce, ycbcr } = parseArgs();
if (!input || !output) {
  console.error('Error: -i and -o are required.');
  console.error('Usage: node open_htj2k_dec.mjs -i <input.j2c> -o <output.ppm> [-r <reduce_NL>] [-ycbcr bt601|bt709]');
  process.exit(1);
}

// ── YCbCr→RGB coefficient tables (fixed-point, scaled by 2^14) ───────────────
// Matches the native decoder in dec_utils.hpp.
// Full-range BT.601 (JFIF / JPEG):  R = Y + 1.402*Cr,  B = Y + 1.772*Cb
// Full-range BT.709 (HDTV):         R = Y + 1.5748*Cr, B = Y + 1.8556*Cb
const YCBCR_COEFFS = {
  bt601: { crToR: 22970, cbToG: 5639,  crToG: 11701, cbToB: 29032 },
  bt709: { crToR: 25802, cbToG: 3069,  crToG: 7668,  cbToB: 30394 },
};

// Convert in-place: buf holds interleaved [Y,Cb,Cr,...] per pixel,
// packed as 8-bit (bytesPerSmp=1) or 16-bit big-endian (bytesPerSmp=2).
function applyYcbcrToRgb(buf, nPixels, maxval, coeffs, bytesPerSmp) {
  const clamp = (v) => v < 0 ? 0 : v > maxval ? maxval : v;
  const center = (maxval + 1) >> 1;  // 128 for 8-bit, 32768 for 16-bit
  const { crToR, cbToG, crToG, cbToB } = coeffs;
  if (bytesPerSmp === 1) {
    for (let i = 0; i < nPixels; i++) {
      const off = i * 3;
      const Y  =  buf[off];
      const Cb =  buf[off + 1] - center;
      const Cr =  buf[off + 2] - center;
      buf[off]     = clamp(Y + ((crToR * Cr + 8192) >> 14));
      buf[off + 1] = clamp(Y - ((cbToG * Cb + crToG * Cr + 8192) >> 14));
      buf[off + 2] = clamp(Y + ((cbToB * Cb + 8192) >> 14));
    }
  } else {
    // 16-bit big-endian: read two bytes per sample, convert, write back.
    for (let i = 0; i < nPixels; i++) {
      const off = i * 6;
      const Y  =  (buf[off]     << 8) | buf[off + 1];
      const Cb = ((buf[off + 2] << 8) | buf[off + 3]) - center;
      const Cr = ((buf[off + 4] << 8) | buf[off + 5]) - center;
      const R  = clamp(Y + ((crToR * Cr + 8192) >> 14));
      const G  = clamp(Y - ((cbToG * Cb + crToG * Cr + 8192) >> 14));
      const B  = clamp(Y + ((cbToB * Cb + 8192) >> 14));
      buf[off]     = R >> 8;   buf[off + 1] = R & 0xff;
      buf[off + 2] = G >> 8;   buf[off + 3] = G & 0xff;
      buf[off + 4] = B >> 8;   buf[off + 5] = B & 0xff;
    }
  }
}

// ── Load WASM module (prefer SIMD build) ─────────────────────────────────────
const __dir = dirname(fileURLToPath(import.meta.url));
const simdPath = join(__dir, 'build/html/libopen_htj2k_simd.js');
const scalarPath = join(__dir, 'build/html/libopen_htj2k.js');

// Emscripten 3.1.x with EXPORT_ES6=1 generates JS that mixes ESM constructs
// (import.meta.url, export default) with CJS ones (__dirname, require).  Node.js
// v24+ refuses to load such a .js file because it cannot determine the module
// format.  We work around this by reading the file as text, stripping the ESM
// export, and evaluating it as a plain script — the IIFE assigns the factory to
// a local `Module` variable which we return from a wrapper Function.
//
// We also pass wasmBinary directly to bypass Emscripten's fetch()-based .wasm
// loading, which fails on Node.js v24+ (native fetch expects URLs, not file
// paths).
function loadEmscriptenFactory(jsPath) {
  let src = readFileSync(jsPath, 'utf-8');
  // Replace ESM-only constructs so the code can run inside new Function():
  //   import.meta.url  → the file:// URL of the JS file (used for locateFile)
  //   export default   → stripped (we return Module explicitly)
  const fileUrl = 'file://' + jsPath;
  src = src.replace(/import\.meta\.url/g, JSON.stringify(fileUrl));
  src = src.replace(/export\s+default\s+Module\s*;?\s*$/, '');
  const require = createRequire(jsPath);
  const fn = new Function('require', '__filename', '__dirname', src + '\nreturn Module;');
  return fn(require, jsPath, dirname(jsPath));
}

let M;
try {
  const wasmBinary = readFileSync(join(__dir, 'build/html/libopen_htj2k_simd.wasm'));
  const createModule = loadEmscriptenFactory(simdPath);
  M = await createModule({ wasmBinary });
} catch {
  const wasmBinary = readFileSync(join(__dir, 'build/html/libopen_htj2k.wasm'));
  const createModule = loadEmscriptenFactory(scalarPath);
  M = await createModule({ wasmBinary });
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

// Detect JPH colorspace (returns 0 for raw codestreams, 18 for YCbCr JPH).
const ENUMCS_YCBCR = 18;
const detectedCs = M._get_colorspace(dec);

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

// Per-component properties (needed for PGX output).
const compDepth = [], compSigned = [], compWidth = [], compHeight = [];
for (let c = 0; c < C; c++) {
  compDepth.push(M._get_depth(dec, c));
  compSigned.push(M._get_signed(dec, c));
  compWidth.push(M._get_width(dec, c));
  compHeight.push(M._get_height(dec, c));
}

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

// Determine whether to apply YCbCr→RGB conversion.
// Auto-enable for JPH files declaring YCbCr colorspace (EnumCS=18) + PPM output.
const wantPpm = output.toLowerCase().endsWith('.ppm');
let doYcbcr = false;
let ycbcrCoeffs = null;
if (C >= 3 && wantPpm) {
  if (ycbcr) {
    doYcbcr    = true;
    ycbcrCoeffs = YCBCR_COEFFS[ycbcr];
    console.log(`INFO: YCbCr→RGB conversion enabled (${ycbcr.toUpperCase()}).`);
  } else if (detectedCs === ENUMCS_YCBCR) {
    doYcbcr    = true;
    ycbcrCoeffs = YCBCR_COEFFS.bt601;
    console.log('INFO: JPH colorspace: YCbCr');
    console.log('INFO: YCbCr→RGB conversion auto-enabled (BT.601). Use -ycbcr bt709 to override.');
  } else if (detectedCs === 16) {
    console.log('INFO: JPH colorspace: sRGB');
  } else if (detectedCs === 17) {
    console.log('INFO: JPH colorspace: Grayscale');
  }
}

const isPGX = output.toLowerCase().endsWith('.pgx');

if (isPGX) {
  // PGX output: use invoke_decoder_planar() which returns per-component planar int32
  // buffers at native resolution (no clamping, no DC offset, handles subsampling).
  // Allocate per-component buffers and a pointer array.
  const compPtrs = [];
  for (let c = 0; c < C; c++) {
    const cW = Math.ceil(compWidth[c] / rFactor);
    const cH = Math.ceil(compHeight[c] / rFactor);
    compPtrs.push(M._malloc(cW * cH * 4));
  }
  // Write pointer array into WASM heap (C pointers, 4 bytes each in wasm32).
  const ptrArrayPtr = M._malloc(C * 4);
  const ptrView = new Uint32Array(M.HEAPU8.buffer, ptrArrayPtr, C);
  for (let c = 0; c < C; c++) ptrView[c] = compPtrs[c];

  try {
    M._invoke_decoder_planar(dec, ptrArrayPtr);
  } catch (e) {
    console.error(`Decode error: ${e}`);
    compPtrs.forEach(p => M._free(p));
    M._free(ptrArrayPtr);
    M._release_j2c_data(dec);
    process.exit(1);
  }
  M._release_j2c_data(dec);
  const t1 = performance.now();

  // PGX: one file per component at native resolution.
  // Format: "PG LM <sign> <depth> <width> <height>\n" + raw little-endian samples.
  const baseName = output.slice(0, -4);
  for (let c = 0; c < C; c++) {
    const fname   = `${baseName}_${String(c).padStart(2, '0')}.pgx`;
    const sign    = compSigned[c] ? '-' : '+';
    const cDepth  = compDepth[c];
    const cW      = Math.ceil(compWidth[c] / rFactor);
    const cH      = Math.ceil(compHeight[c] / rFactor);
    const hdr     = `PG LM ${sign} ${cDepth} ${cW} ${cH}\n`;
    const nPixels = cW * cH;
    const bps     = Math.ceil(cDepth / 8);  // bytes per sample: 1 or 2

    // Re-read heap view (may have been invalidated by memory growth).
    const heap32 = new Int32Array(M.HEAPU8.buffer);
    const baseIdx = compPtrs[c] >> 2;

    const compBuf = Buffer.alloc(nPixels * bps);
    for (let i = 0; i < nPixels; i++) {
      const val = heap32[baseIdx + i];
      if (bps === 1) {
        compBuf[i] = val & 0xFF;
      } else {
        compBuf.writeInt16LE(val, i * 2);
      }
    }
    writeFileSync(fname, Buffer.concat([Buffer.from(hdr, 'ascii'), compBuf]));
    M._free(compPtrs[c]);
  }
  M._free(ptrArrayPtr);
  const elapsed = (t1 - t0).toFixed(1);
  console.log(`Decoded ${Wd}×${Hd} ${C}comp ${depth}bpc in ${elapsed} ms → ${baseName}_*.pgx`);
} else {
  // PPM/PGM output: use streaming decoder with DC offset for signed components.
  const nSamples  = Wd * Hd * C;
  const packedPtr = M._malloc(nSamples * bytesPerSmp);
  if (!packedPtr) {
    console.error('WASM malloc failed for packed output buffer');
    M._release_j2c_data(dec);
    process.exit(1);
  }
  try {
    M._invoke_decoder_stream(dec, packedPtr, maxval, bytesPerSmp, 1);
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
  // PPM/PGM output path.
  // Apply YCbCr→RGB in-place on the packed pixel buffer (after WASM chroma upsampling).
  if (doYcbcr) {
    applyYcbcrToRgb(pixelBuf, Wd * Hd, maxval, ycbcrCoeffs, bytesPerSmp);
  }

  try {
    writeFileSync(output, Buffer.concat([headerBuf, pixelBuf]));
  } catch (e) {
    console.error(`Error writing output file: ${e.message}`);
    process.exit(1);
  }

  const elapsed = (t1 - t0).toFixed(1);
  const fmt     = C === 1 ? 'grayscale' : (doYcbcr ? 'YCbCr→RGB' : 'RGB');
  console.log(`Decoded ${Wd}×${Hd} ${fmt} ${depth}bpc in ${elapsed} ms → ${output}`);
}
