#!/usr/bin/env bash
# Driver for region_decode_bench: sweeps thread counts as SEPARATE processes,
# because the decoder's ThreadPool is a first-call-wins process-global singleton
# (you cannot change the thread count within one process).
#
# Usage:
#   run_bench.sh <input.j2k|.jph> [bin] [threads_csv] [extra args...]
# e.g.
#   run_bench.sh ~/Downloads/heic0602a.j2c
#   run_bench.sh ~/Downloads/heic0602a.j2c build/bin/region_decode_bench "1,2,4,6" -full
set -euo pipefail

IN="${1:?usage: run_bench.sh <input> [bin] [threads_csv] [extra...]}"
BIN="${2:-build/bin/region_decode_bench}"
THREADS_CSV="${3:-1,4}"
shift || true; shift || true; shift || true
EXTRA=("$@")

if [[ ! -x "$BIN" ]]; then echo "ERROR: $BIN not found/executable" >&2; exit 1; fi
if [[ ! -f "$IN" ]]; then echo "ERROR: input $IN not found" >&2; exit 1; fi

IFS=',' read -r -a THREADS <<< "$THREADS_CSV"
for T in "${THREADS[@]}"; do
  echo "==================== threads=$T ===================="
  "$BIN" "$IN" -threads "$T" "${EXTRA[@]}"
  echo
done
