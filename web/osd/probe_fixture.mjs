// Quick probe: read full dims + max-safe reduce (NL) + components for a codestream.
//   node web/osd/probe_fixture.mjs <input.j2c>
import { readFileSync } from "fs";
import { createRequire } from "module";
import { dirname, join } from "path";
import { fileURLToPath } from "url";

const __dir = dirname(fileURLToPath(import.meta.url));
const infile = process.argv[2] || `${process.env.HOME}/Downloads/heic0602a.j2c`;

function loadFactory(jsPath) {
  let src = readFileSync(jsPath, "utf-8");
  src = src.replace(/import\.meta\.url/g, JSON.stringify("file://" + jsPath));
  src = src.replace(/;var isPthread=[\s\S]*$/, "");
  src = src.replace(/export\s+default\s+Module\s*;?\s*$/, "");
  const require = createRequire(jsPath);
  return new Function("require", "__filename", "__dirname", src + "\nreturn Module;")(
    require, jsPath, dirname(jsPath));
}
const jsPath = join(__dir, "../build/html/libopen_htj2k_simd.js");
const wasmBinary = readFileSync(join(__dir, "../build/html/libopen_htj2k_simd.wasm"));
const M = await loadFactory(jsPath)({ wasmBinary });
const bytes = readFileSync(infile);
const dataPtr = M._malloc(bytes.length);
M.HEAPU8.set(bytes, dataPtr);

const dec = M.ccall("create_decoder", "number", ["number", "number", "number"],
  [dataPtr, bytes.length, 0]);
M.ccall("parse_j2c_data", null, ["number"], [dec]);
const W = M.ccall("get_width", "number", ["number", "number"], [dec, 0]);
const H = M.ccall("get_height", "number", ["number", "number"], [dec, 0]);
const nc = M.ccall("get_num_components", "number", ["number"], [dec]);
const depth = M.ccall("get_depth", "number", ["number", "number"], [dec, 0]);
const cs = M.ccall("get_colorspace", "number", ["number"], [dec]);
const minDWT = M.ccall("get_minimum_DWT_levels", "number", ["number"], [dec]);
const maxR = M.ccall("get_max_safe_reduce_NL", "number", ["number"], [dec]);
console.log(JSON.stringify({ file: infile, W, H, nc, depth, colorspace: cs, minDWTlevels: minDWT, maxSafeReduce: maxR }, null, 2));
