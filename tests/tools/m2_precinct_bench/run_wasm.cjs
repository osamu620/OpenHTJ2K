// run_wasm.cjs — Node.js runner for the WASM build of m2_precinct_bench.
//
// Drives the module's exported m2bench_modes() via ccall over a sweep of reduce
// levels x square window sizes, and prints the same table shape as the native
// tool — so the WASM precinct-filter column (the deployment path) can sit next
// to the native one in the M2 (OpenSeadragon) comparison.  main() is not used
// (-sINVOKE_RUN=0); we call the exported C function instead.
//
// Build first (build_wasm.sh does this), then:
//   node tests/tools/m2_precinct_bench/run_wasm.cjs <module.js> <input.j2k> \
//        [-threads T] [-iter K] [-warmup W] [-win 256,512,1024] [-maxlevel L] [-noverify] [-csv]
const { readFileSync } = require('fs');
const os = require('os');

const argv = process.argv.slice(2);
if (argv.length < 2) {
  console.error(
    'usage: node run_wasm.cjs <module.js> <input.j2k> [-threads T] [-iter K] [-warmup W] [-win L1,L2,..] [-maxlevel L] [-noverify] [-csv]');
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
const iter = optInt('-iter', 15);
const warmup = optInt('-warmup', 3);
const maxlevel = optInt('-maxlevel', 5);
const csv = rest.includes('-csv');
const doVerify = rest.includes('-noverify') ? 0 : 1;
let wins = [256, 512, 1024];
{
  const i = rest.indexOf('-win');
  if (i >= 0 && i + 1 < rest.length) wins = rest[i + 1].split(',').map((x) => parseInt(x, 10)).filter((x) => x > 0);
}
wins.sort((a, b) => a - b);

// HARD RULE: WASM num_threads must be <= PTHREAD_POOL_SIZE (pool = min(hwc,16)).
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
  const outPtr = m._malloc(11 * 8);  // eleven doubles

  if (csv) {
    console.log('level,level_w,level_h,win_w,win_h,threads,kept,total,A_tot,A_dec,B_tot,B_dec,C_tot,C_dec,win_ms,verify');
  } else {
    console.log(`# m2_precinct_bench (WASM)  file=${input}  (${bytes.length} bytes)`);
    console.log(`# module=${modulePath}  threads=${threads} (pool ${POOL})  iter=${iter}  warmup=${warmup}`);
    console.log('# A=keep-all(M1)  B=drop-all(floor)  C=keep-window(M2); ms = median total');
    console.log(
      ['level', 'level_dims'.padEnd(13), 'window'.padEnd(11), 'kept/total'.padEnd(11),
       'A(M1)'.padEnd(7), 'B(flr)'.padEnd(7), 'C(M2)'.padEnd(7), 'A.dec'.padEnd(7),
       'B.dec'.padEnd(8), 'C.dec'.padEnd(8), 'verify'].join('  '));
  }

  let fails = 0;
  for (let li = 0; li <= maxlevel; li++) {
    let prevKey = '';
    for (const S of wins) {
      const rc = m.ccall('m2bench_modes', 'number',
        ['number', 'number', 'number', 'number', 'number', 'number', 'number', 'number', 'number'],
        [ptr, bytes.length, li, S, threads, iter, warmup, doVerify, outPtr]);
      if (rc !== 0) { console.error(`# level ${li} win ${S}: error (rc=${rc})`); continue; }
      const o = (k) => m.HEAPF64[(outPtr >> 3) + k];
      const At = o(0), Ad = o(1), Bt = o(2), Bd = o(3), Ct = o(4), Cd = o(5);
      const kept = o(6) | 0, total = o(7) | 0, W = o(8) | 0, H = o(9) | 0, verify = o(10) | 0;
      const w = Math.min(S, W), h = Math.min(S, H);
      const key = `${w}x${h}`;
      if (key === prevKey) continue;  // window clamped to same dims (small level)
      prevKey = key;
      if (doVerify && verify !== 1) fails++;
      const vtxt = doVerify ? (verify === 1 ? 'PASS' : 'FAIL') : '-';
      const kt = `${kept}/${total}`;
      if (csv) {
        console.log([li, W, H, w, h, threads, kept, total,
          At.toFixed(3), Ad.toFixed(3), Bt.toFixed(3), Bd.toFixed(3), Ct.toFixed(3), Cd.toFixed(3),
          (At - Ct).toFixed(3), vtxt].join(','));
      } else {
        console.log([
          String(li).padEnd(5), `${W}x${H}`.padEnd(13), key.padEnd(11), kt.padEnd(11),
          At.toFixed(2).padEnd(7), Bt.toFixed(2).padEnd(7), Ct.toFixed(2).padEnd(7),
          Ad.toFixed(2).padEnd(7), Bd.toFixed(2).padEnd(8), Cd.toFixed(2).padEnd(8), vtxt].join('  '));
      }
    }
  }
  if (doVerify) console.log(`# verify: ${fails ? 'FAIL' : 'PASS'} (${fails} failures)`);
  m._free(outPtr);
  m._free(ptr);
  process.exit(fails ? 2 : 0);
})();
