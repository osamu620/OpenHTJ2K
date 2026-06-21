// Headless correctness proof for the OSD HTJ2K tile plumbing.
//
// For every decodable OSD level it does an independent FULL-level decode
// (invoke_decoder_to_rgba -- no col/row range), then decodes a set of tiles
// through the SAME shared decode core the Web Worker uses (geometry ->
// per-reduce region-decoder pool -> decode_region_to_rgba) and asserts every
// tile is byte-exact against the corresponding window of the full decode.
//
// Tiles checked per level: all of them when the level has few tiles, otherwise a
// curated set that always includes the right edge, bottom edge and bottom-right
// corner (the clipped-dimension cases) plus interior tiles.
//
//   node web/osd/test_tile_grid.mjs [input.j2c]
//
// Exits 0 on PASS, non-zero on the first failing tile.

import { readFileSync } from 'fs';
import { createRequire } from 'module';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';
import { bindFns, loadCodestream, makeDecodeCore } from './htj2k_decode_core.mjs';
import { levelDims, numTiles } from './htj2k_geometry.mjs';

const __dir = dirname(fileURLToPath(import.meta.url));
const infile = process.argv[2] || `${process.env.HOME}/Downloads/heic0602a.j2c`;

// Emscripten EXPORT_ES6 module mixes ESM + CJS that Node can't load directly;
// read + strip + eval (same workaround as web/open_htj2k_dec.mjs / test_region_rgba.mjs).
function loadFactory(jsPath) {
  let src = readFileSync(jsPath, 'utf-8');
  src = src.replace(/import\.meta\.url/g, JSON.stringify('file://' + jsPath));
  src = src.replace(/;var isPthread=[\s\S]*$/, '');
  src = src.replace(/export\s+default\s+Module\s*;?\s*$/, '');
  const require = createRequire(jsPath);
  return new Function('require', '__filename', '__dirname', src + '\nreturn Module;')(
    require, jsPath, dirname(jsPath));
}

const jsPath = join(__dir, '../build/html/libopen_htj2k_simd.js');
const wasmBinary = readFileSync(join(__dir, '../build/html/libopen_htj2k_simd.wasm'));
const M = await loadFactory(jsPath)({ wasmBinary });
const F = bindFns(M);

const bytes = readFileSync(infile);
const { dataPtr, dataLen, pyramid, nc } = loadCodestream(M, F, bytes);
const core = makeDecodeCore({ M, F, dataPtr, dataLen, pyramid });

console.log(`fixture ${infile}`);
console.log(`  ${pyramid.fullW}x${pyramid.fullH}, nc=${nc}, maxReduce=${pyramid.maxReduce}, ` +
            `OSD levels ${pyramid.minLevel}..${pyramid.maxLevel}, tile=${pyramid.tileSize}`);

// Tiles to verify at a level with nx*ny tiles.  Always include the clipped
// cases (last column, last row, bottom-right corner) and the corners/center.
function pickTiles(nx, ny) {
  if (nx * ny <= 64) {
    const all = [];
    for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) all.push([x, y]);
    return all;
  }
  const cx = nx >> 1, cy = ny >> 1;
  const set = new Set([
    [0, 0], [nx - 1, 0], [0, ny - 1], [nx - 1, ny - 1], // corners (right/bottom = clipped)
    [cx, 0], [0, cy], [nx - 1, cy], [cx, ny - 1],        // edge midpoints
    [cx, cy], [cx + 1, cy], [cx, cy + 1],                // interior cluster
  ].map((p) => p.join(',')));
  return [...set].map((s) => s.split(',').map(Number));
}

let totalTiles = 0;
let totalBytes = 0;
const t0 = Date.now();

for (let level = pyramid.minLevel; level <= pyramid.maxLevel; level++) {
  const { reduce, levelW, levelH } = levelDims(level, pyramid);
  const { nx, ny } = numTiles(level, pyramid);

  // Independent reference: full decode of this level to RGBA.
  const refPtr = F.malloc(levelW * levelH * 4);
  const refDec = F.create_decoder(dataPtr, dataLen, reduce);
  F.parse(refDec);
  F.invoke_to_rgba(refDec, refPtr);
  F.release(refDec);

  const tiles = pickTiles(nx, ny);
  let levelMism = 0;
  for (const [tx, ty] of tiles) {
    const tile = core.decodeTileToHeap(level, tx, ty);
    if (!tile) { console.error(`  L${level} (${tx},${ty}) unexpectedly empty`); process.exit(2); }
    const { ptr, w, h, x0, y0 } = tile;
    // Re-grab views after the decode (ALLOW_MEMORY_GROWTH may have moved them).
    const ref = M.HEAPU8;
    const out = M.HEAPU8;
    for (let ly = 0; ly < h && levelMism === 0; ly++) {
      const refRow = refPtr + ((y0 + ly) * levelW + x0) * 4;
      const outRow = ptr + ly * w * 4;
      for (let k = 0; k < w * 4; k++) {
        if (ref[refRow + k] !== out[outRow + k]) {
          const px = x0 + ((k / 4) | 0), py = y0 + ly, ch = k & 3;
          console.error(`FAIL L${level} reduce=${reduce} tile(${tx},${ty}) ` +
            `px(${px},${py}) ch=${ch}: region=${out[outRow + k]} ref=${ref[refRow + k]}`);
          process.exit(3);
        }
      }
    }
    totalTiles++;
    totalBytes += w * h * 4;
  }
  F.free(refPtr);
  console.log(`  L${level} reduce=${reduce} ${levelW}x${levelH} grid ${nx}x${ny}: ` +
              `${tiles.length} tiles byte-exact`);
}

core.dispose();
console.log(`PASS: ${totalTiles} tiles across ${pyramid.maxLevel - pyramid.minLevel + 1} levels ` +
            `byte-exact vs full-level decode (${(totalBytes / 1e6).toFixed(1)} MP, ${Date.now() - t0} ms)`);
