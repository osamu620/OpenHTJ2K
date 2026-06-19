// run_wasm.cjs — Node.js runner for the WASM build of col_range_compare.
//
// Validates the WASM-SIMD sub-range horizontal IDWT kernels
// (idwt_1d_filtr_irrev97_planar_sr_wasm) at runtime, by driving the module's
// exported crc_validate() via ccall.  main() is not used: emscripten's
// main()-on-load path is unreliable under recent Node, so — exactly like
// web/open_htj2k_dec.mjs — we call an exported C function instead.
//
// Build the module (from the repo root, after an emcmake `web` build so that
// libopen_htj2k_simd_lib.a exists):
//
//   emcc -O3 -flto -msimd128 -mnontrapping-fptoint -mbulk-memory -std=c++17 \
//     -DOPENHTJ2K_ENABLE_WASM_SIMD \
//     -Isource/core/common -Isource/core/transform -Isource/core/codestream \
//     -Isource/core/coding -Isource/core/interface -Isource/core/jph -Isource/core/jpip \
//     tests/tools/col_range_compare/main_crcmp.cpp web/build/libopen_htj2k_simd_lib.a \
//     -sMODULARIZE=1 -sENVIRONMENT=node -sINVOKE_RUN=0 \
//     -sEXPORTED_FUNCTIONS=_crc_validate,_malloc,_free \
//     -sEXPORTED_RUNTIME_METHODS=ccall,HEAPU8 \
//     -sALLOW_MEMORY_GROWTH=1 -sSINGLE_FILE=1 -o /tmp/crc_wasm.js
//
// Run (exits 0 on bit-exact match / benign reuse-skip, 1 on mismatch):
//
//   node tests/tools/col_range_compare/run_wasm.cjs /tmp/crc_wasm.js <input.j2k> [-reuse] [-reduce N]
//
// -reuse drives the single-tile reuse path — the path that exercises the SIMD
// sub-range kernels (the WASM JPIP viewer's path).  Note: a few fixtures hit a
// pre-existing reuse-path codestream re-parse issue under WASM; use a fixture
// that decodes cleanly (e.g. a single-tile 9/7 lossy stream).
const { readFileSync } = require('fs');

const argv = process.argv.slice(2);
if (argv.length < 2) {
  console.error('usage: node run_wasm.cjs <module.js> <input.j2k> [-reuse] [-reduce N]');
  process.exit(2);
}
const modulePath = argv[0];
const rest = argv.slice(1);
const input = rest.find((a) => !a.startsWith('-'));
const reuse = rest.includes('-reuse') ? 1 : 0;
let reduce = 0;
const ri = rest.indexOf('-reduce');
if (ri >= 0 && ri + 1 < rest.length) reduce = parseInt(rest[ri + 1], 10);

const factory = require(modulePath);
(async () => {
  const m = await factory();
  const bytes = readFileSync(input);
  const ptr = m._malloc(bytes.length);
  m.HEAPU8.set(bytes, ptr);
  const rc = m.ccall('crc_validate', 'number',
    ['number', 'number', 'number', 'number'], [ptr, bytes.length, reuse, reduce]);
  m._free(ptr);
  console.log(`${rc === 0 ? 'PASS' : 'FAIL'} ${input}${reuse ? ' (reuse)' : ''}`);
  process.exit(rc | 0);
})();
