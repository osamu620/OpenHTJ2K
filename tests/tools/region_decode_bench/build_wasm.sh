#!/usr/bin/env bash
# build_wasm.sh — build the WASM modules for region_decode_bench.
#
# Produces single-thread+SIMD and multi-thread+SIMD modules that the
# run_wasm.cjs runner drives via ccall.  Requires the web core libs to be built
# first (web/build/libopen_htj2k_simd_lib.a and *_mt_simd_lib.a) — those are the
# core decoder compiled for Wasm; this just links the benchmark entry point.
#
# Usage:  tests/tools/region_decode_bench/build_wasm.sh [out_dir]   (default /tmp)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

OUT="${1:-/tmp}"
SRC=tests/tools/region_decode_bench/main_rdbench.cpp
INCS=(-Isource/core/common -Isource/core/transform -Isource/core/codestream
      -Isource/core/coding -Isource/core/interface -Isource/core/jph -Isource/core/jpip)
COMMON=(-O3 -flto -msimd128 -mnontrapping-fptoint -mbulk-memory -std=c++17
        -DOPENHTJ2K_ENABLE_WASM_SIMD "${INCS[@]}" "$SRC"
        -sMODULARIZE=1 -sENVIRONMENT=node -sINVOKE_RUN=0
        -sEXPORTED_FUNCTIONS=_rdbench_region,_malloc,_free
        -sEXPORTED_RUNTIME_METHODS=ccall,HEAPU8,HEAPF64
        -sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=2GB -sNO_DISABLE_EXCEPTION_CATCHING -sSINGLE_FILE=1)

ST_LIB=web/build/libopen_htj2k_simd_lib.a
MT_LIB=web/build/libopen_htj2k_mt_simd_lib.a
[[ -f "$ST_LIB" ]] || { echo "ERROR: $ST_LIB missing — build the web libs first" >&2; exit 1; }

echo "=== building st_simd -> $OUT/rdbench_st.js ==="
emcc "${COMMON[@]}" "$ST_LIB" -o "$OUT/rdbench_st.js"

if [[ -f "$MT_LIB" ]]; then
  echo "=== building mt_simd -> $OUT/rdbench_mt.js ==="
  # PTHREAD_POOL_SIZE pre-creates the worker pool; run_wasm.cjs clamps -threads
  # to <= this (min(hwc,16)).  pthreads + ENVIRONMENT must include 'worker'.
  emcc "${COMMON[@]/-sENVIRONMENT=node/-sENVIRONMENT=node,worker}" \
    -pthread -sUSE_PTHREADS=1 \
    -sPTHREAD_POOL_SIZE='Math.min((typeof navigator!=="undefined"?navigator.hardwareConcurrency:require("os").cpus().length),16)' \
    "$MT_LIB" -o "$OUT/rdbench_mt.js"
else
  echo "(skip mt_simd: $MT_LIB missing)"
fi
echo "done."
