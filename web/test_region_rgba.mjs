// Byte-exact test for decode_region_to_rgba (the OSD plugin primitive):
// a windowed region decode must equal the corresponding window of a full decode.
//   node web/test_region_rgba.mjs <input.j2c> [reduce] [win]
import { readFileSync } from "fs";
import { createRequire } from "module";
import { dirname, join } from "path";
import { fileURLToPath } from "url";

const __dir = dirname(fileURLToPath(import.meta.url));
const infile = process.argv[2] || `${process.env.HOME}/Downloads/heic0602a.j2c`;
const reduce = parseInt(process.argv[3] || "4", 10);
const win = parseInt(process.argv[4] || "256", 10);
const FULL_W = 15852, FULL_H = 12392;  // Carina fixture (full res)
const ceilDiv = (a, b) => Math.ceil(a / b);
const W = ceilDiv(FULL_W, 2 ** reduce), H = ceilDiv(FULL_H, 2 ** reduce);

// Emscripten EXPORT_ES6 module mixes ESM + CJS constructs that Node v24 can't
// load directly; read + strip + eval (same workaround as open_htj2k_dec.mjs).
function loadFactory(jsPath) {
  let src = readFileSync(jsPath, "utf-8");
  src = src.replace(/import\.meta\.url/g, JSON.stringify("file://" + jsPath));
  src = src.replace(/;var isPthread=[\s\S]*$/, "");
  src = src.replace(/export\s+default\s+Module\s*;?\s*$/, "");
  const require = createRequire(jsPath);
  return new Function("require", "__filename", "__dirname", src + "\nreturn Module;")(
    require, jsPath, dirname(jsPath));
}
const jsPath = join(__dir, "build/html/libopen_htj2k_simd.js");
const wasmBinary = readFileSync(join(__dir, "build/html/libopen_htj2k_simd.wasm"));
const M = await loadFactory(jsPath)({ wasmBinary });
const bytes = readFileSync(infile);
const dataPtr = M._malloc(bytes.length);
M.HEAPU8.set(bytes, dataPtr);

// Reference: full decode of the reduced level -> RGBA.
const dec = M.ccall("create_decoder", "number", ["number", "number", "number"],
  [dataPtr, bytes.length, reduce]);
if (!dec) { console.error("create_decoder failed"); process.exit(1); }
M.ccall("parse_j2c_data", null, ["number"], [dec]);
const fullPtr = M._malloc(W * H * 4);
M.ccall("invoke_decoder_to_rgba", null, ["number", "number"], [dec, fullPtr]);

// Region: centred win x win window via the new export.
const w = Math.min(win, W), h = Math.min(win, H);
const x0 = (W - w) >> 1, y0 = (H - h) >> 1;
const rdec = M.ccall("create_region_decoder", "number", [], []);
const tilePtr = M._malloc(w * h * 4);
const rc = M.ccall("decode_region_to_rgba", "number",
  ["number", "number", "number", "number", "number", "number", "number", "number", "number"],
  [rdec, dataPtr, bytes.length, reduce, x0, y0, w, h, tilePtr]);
if (rc !== 0) { console.error("decode_region_to_rgba rc=" + rc); process.exit(1); }

// Re-grab heap views AFTER all allocations/decodes (ALLOW_MEMORY_GROWTH may
// have reallocated the heap, detaching any view taken earlier).
const full = M.HEAPU8.subarray(fullPtr, fullPtr + W * H * 4);
const tile = M.HEAPU8.subarray(tilePtr, tilePtr + w * h * 4);

// Compare the tile against the window of the full decode.
let mism = 0, firstX = -1, firstY = -1, maxAbs = 0;
for (let ly = 0; ly < h; ly++) {
  for (let lx = 0; lx < w; lx++) {
    for (let c = 0; c < 4; c++) {
      const t = tile[(ly * w + lx) * 4 + c];
      const f = full[((y0 + ly) * W + (x0 + lx)) * 4 + c];
      if (t !== f) {
        if (mism === 0) { firstX = x0 + lx; firstY = y0 + ly; }
        maxAbs = Math.max(maxAbs, Math.abs(t - f));
        mism++;
      }
    }
  }
}
console.log(`level reduce=${reduce} (${W}x${H}), window ${w}x${h} @ (${x0},${y0})`);
if (mism === 0) {
  console.log(`PASS: region RGBA byte-exact vs full-decode window (${w * h} px)`);
  process.exit(0);
}
console.log(`FAIL: ${mism} byte mismatches (first @ ${firstX},${firstY}, maxAbs=${maxAbs})`);
process.exit(2);
