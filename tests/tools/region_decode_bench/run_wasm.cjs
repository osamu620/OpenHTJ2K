// run_wasm.cjs — Node.js runner for the WASM build of region_decode_bench.
//
// Drives the module's exported rdbench_region() via ccall over a sweep of
// reduce levels x square window sizes, and prints the same table shape as the
// native tool — so the WASM column can sit next to the native one in the M1
// (OpenSeadragon region-decode) comparison.  main() is not used: emscripten's
// main()-on-load path is unreliable under recent Node, so we call an exported C
// function instead (exactly like web/open_htj2k_dec.mjs).
//
// Build the module (from the repo root, after the web libs are built so that
// libopen_htj2k_simd_lib.a / libopen_htj2k_mt_simd_lib.a exist):
//
//   # single-thread + SIMD
//   emcc -O3 -flto -msimd128 -mnontrapping-fptoint -mbulk-memory -std=c++17 \
//     -DOPENHTJ2K_ENABLE_WASM_SIMD \
//     -Isource/core/common -Isource/core/transform -Isource/core/codestream \
//     -Isource/core/coding -Isource/core/interface -Isource/core/jph -Isource/core/jpip \
//     tests/tools/region_decode_bench/main_rdbench.cpp web/build/libopen_htj2k_simd_lib.a \
//     -sMODULARIZE=1 -sENVIRONMENT=node -sINVOKE_RUN=0 \
//     -sEXPORTED_FUNCTIONS=_rdbench_region,_malloc,_free \
//     -sEXPORTED_RUNTIME_METHODS=ccall,HEAPU8,HEAPF64 \
//     -sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=2GB -sSINGLE_FILE=1 \
//     -o /tmp/rdbench_st.js
//
//   # multi-thread + SIMD (note PTHREAD_POOL_SIZE; pass -threads <= pool size)
//   emcc ... -pthread -sUSE_PTHREADS=1 \
//     -sPTHREAD_POOL_SIZE='Math.min(require("os").cpus().length,16)' \
//     tests/tools/region_decode_bench/main_rdbench.cpp web/build/libopen_htj2k_mt_simd_lib.a \
//     ... -o /tmp/rdbench_mt.js   (build_wasm.sh does this for you)
//
// Run:
//   node tests/tools/region_decode_bench/run_wasm.cjs <module.js> <input.j2k> \
//        [-threads T] [-iter K] [-warmup W] [-win 256,512,1024] [-maxlevel L] [-csv]
const { readFileSync } = require('fs');
const os = require('os');

const argv = process.argv.slice(2);
if (argv.length < 2) {
  console.error(
    'usage: node run_wasm.cjs <module.js> <input.j2k> [-threads T] [-iter K] [-warmup W] [-win L1,L2,..] [-maxlevel L] [-csv]');
  process.exit(2);
}
const modulePath = argv[0];
const rest = argv.slice(1);
const input = rest.find((a) => !a.startsWith('-') && a !== modulePath);

function optInt(name, def) {
  const i = rest.indexOf(name);
  return i >= 0 && i + 1 < rest.length ? parseInt(rest[i + 1], 10) : def;
}
let threads = optInt('-threads', 1);
const iter = optInt('-iter', 21);
const warmup = optInt('-warmup', 3);
const maxlevel = optInt('-maxlevel', 5);
const csv = rest.includes('-csv');
let wins = [256, 512, 1024];
{
  const i = rest.indexOf('-win');
  if (i >= 0 && i + 1 < rest.length) wins = rest[i + 1].split(',').map((x) => parseInt(x, 10)).filter((x) => x > 0);
}
wins.sort((a, b) => a - b);

// HARD RULE: WASM num_threads must be <= PTHREAD_POOL_SIZE (pool sizes to
// min(hwc,16)); passing more forces a dynamic Worker spawn that hangs silently
// on multi-core.  Clamp here exactly like every other caller in this repo.
const POOL = Math.min(os.cpus().length, 16);
if (threads > POOL) {
  console.error(`# clamping -threads ${threads} -> ${POOL} (PTHREAD_POOL_SIZE)`);
  threads = POOL;
}

const factory = require(modulePath);
(async () => {
  const m = await factory();
  const bytes = readFileSync(input);
  const ptr = m._malloc(bytes.length);
  m.HEAPU8.set(bytes, ptr);
  const outPtr = m._malloc(6 * 8);  // six doubles

  if (csv) {
    console.log('level,level_w,level_h,window,win_w,win_h,win_px,threads,iters,med_ms,dec_ms,set_ms,min_ms,regions_s,mpix_s');
  } else {
    console.log(`# region_decode_bench (WASM)  file=${input}  (${bytes.length} bytes)`);
    console.log(`# module=${modulePath}  threads=${threads} (pool ${POOL})  iter=${iter}  warmup=${warmup}`);
    console.log(
      ['level', 'level_dims'.padEnd(13), 'window'.padEnd(11), 'win_px'.padEnd(9), 'iters'.padEnd(5),
       'med_ms'.padEnd(9), 'dec_ms'.padEnd(9), 'set_ms'.padEnd(9), 'min_ms'.padEnd(9),
       'regions/s'.padEnd(10), 'Mpix/s'].join('  '));
  }

  for (let li = 0; li <= maxlevel; li++) {
    let prevKey = '';
    for (const S of wins) {
      const rc = m.ccall('rdbench_region', 'number',
        ['number', 'number', 'number', 'number', 'number', 'number', 'number', 'number'],
        [ptr, bytes.length, li, S, threads, iter, warmup, outPtr]);
      if (rc !== 0) { console.error(`# level ${li} win ${S}: decode error (rc=${rc})`); continue; }
      const o = (k) => m.HEAPF64[(outPtr >> 3) + k];
      const medTot = o(0), medDec = o(1), medSet = o(2), minTot = o(3), W = o(4) | 0, H = o(5) | 0;
      const w = Math.min(S, W), h = Math.min(S, H);
      const key = `${w}x${h}`;
      if (key === prevKey) continue;  // window clamped to same dims as previous (small level)
      prevKey = key;
      const winpx = w * h;
      const rps = 1000 / medTot, mpixps = winpx / medTot / 1000;
      if (csv) {
        console.log([li, W, H, key, w, h, winpx, threads, iter,
          medTot.toFixed(4), medDec.toFixed(4), medSet.toFixed(4), minTot.toFixed(4),
          rps.toFixed(1), mpixps.toFixed(1)].join(','));
      } else {
        console.log([
          String(li).padEnd(5), `${W}x${H}`.padEnd(13), key.padEnd(11), String(winpx).padEnd(9),
          String(iter).padEnd(5), medTot.toFixed(4).padEnd(9), medDec.toFixed(4).padEnd(9),
          medSet.toFixed(4).padEnd(9), minTot.toFixed(4).padEnd(9), rps.toFixed(1).padEnd(10),
          mpixps.toFixed(1)].join('  '));
      }
    }
  }
  m._free(outPtr);
  m._free(ptr);
  process.exit(0);
})();
