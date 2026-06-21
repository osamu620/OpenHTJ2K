#!/usr/bin/env bash
# build_wasm.sh — build the WASM modules for m2_precinct_bench (OSD M2 spike).
#
# Mirrors region_decode_bench/build_wasm.sh.  Produces single-thread+SIMD and
# (if present) multi-thread+SIMD modules that run_wasm.cjs drives via ccall.
# Requires the web core libs to exist first (web/build/libopen_htj2k_simd_lib.a
# and *_mt_simd_lib.a) — those carry the core decoder AND the JPIP geometry
# (precinct_index.cpp.o + view_window.cpp.o), so this just links the bench entry
# point.  Rebuild the web libs from current source if stale:
#   cmake --build web/build --target open_htj2k_simd_lib open_htj2k_mt_simd_lib
#
# Usage:  tests/tools/m2_precinct_bench/build_wasm.sh [out_dir]   (default /tmp)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

OUT="${1:-/tmp}"
SRC=tests/tools/m2_precinct_bench/main_m2bench.cpp
INCS=(-Isource/core/common -Isource/core/transform -Isource/core/codestream
      -Isource/core/coding -Isource/core/interface -Isource/core/jph -Isource/core/jpip)
COMMON=(-O3 -flto -msimd128 -mnontrapping-fptoint -mbulk-memory -std=c++17
        -DOPENHTJ2K_ENABLE_WASM_SIMD "${INCS[@]}" "$SRC"
        -sMODULARIZE=1 -sENVIRONMENT=node -sINVOKE_RUN=0
        -sEXPORTED_FUNCTIONS=_m2bench_modes,_malloc,_free
        -sEXPORTED_RUNTIME_METHODS=ccall,HEAPU8,HEAPF64
        -sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=2GB -sNO_DISABLE_EXCEPTION_CATCHING -sSINGLE_FILE=1)

ST_LIB=web/build/libopen_htj2k_simd_lib.a
MT_LIB=web/build/libopen_htj2k_mt_simd_lib.a
[[ -f "$ST_LIB" ]] || { echo "ERROR: $ST_LIB missing — build the web libs first" >&2; exit 1; }

echo "=== building st_simd -> $OUT/m2bench_st.js ==="
emcc "${COMMON[@]}" "$ST_LIB" -o "$OUT/m2bench_st.js"

# MT module is optional — the OSD deployment model is single-thread-per-tile +
# scale-out across web workers, so the ST module above is the one that matters.
# pthreads + MODULARIZE additionally requires -sEXPORT_NAME (emcc hard error);
# keep its failure non-fatal so a missing MT build never blocks the ST module.
if [[ -f "$MT_LIB" ]]; then
  echo "=== building mt_simd -> $OUT/m2bench_mt.js (optional) ==="
  emcc "${COMMON[@]/-sENVIRONMENT=node/-sENVIRONMENT=node,worker}" \
    -pthread -sUSE_PTHREADS=1 -sEXPORT_NAME=M2BenchMT \
    -sPTHREAD_POOL_SIZE='Math.min((typeof navigator!=="undefined"?navigator.hardwareConcurrency:require("os").cpus().length),16)' \
    "$MT_LIB" -o "$OUT/m2bench_mt.js" \
    || echo "(mt_simd build failed — ST is the deployment module; ignoring)"
else
  echo "(skip mt_simd: $MT_LIB missing)"
fi
echo "done."
