#!/bin/bash
# Runs every fuzz target that tests/fuzz/build.sh left in <directory> (libFuzzer builds) for
# <seconds> each, all at the same time, from its seed corpus and with its dictionary.  The
# limits are those of OSS-Fuzz: 25 s for one input, 2560 MB of memory.
# Prints one line per target and exits 1 if a target reported anything; the inputs it
# reported are in <directory>/run/<target>/artifacts, its output in .../log.txt.
# usage: run.sh <directory> [<seconds, default 60>]
set -u
out=${1:?usage: run.sh <directory> [<seconds>]}
seconds=${2:-60}
run=$out/run
export ASAN_OPTIONS=${ASAN_OPTIONS:-detect_leaks=1:detect_stack_use_after_return=1:allocator_may_return_null=1:handle_abort=1}
export UBSAN_OPTIONS=${UBSAN_OPTIONS:-print_stacktrace=1}

names=()
pids=()
for target in "$out"/*_fuzzer; do
  name=$(basename "$target")
  mkdir -p "$run/$name/corpus" "$run/$name/seeds" "$run/$name/artifacts"
  unzip -q -o "$out/${name}_seed_corpus.zip" -d "$run/$name/seeds"
  "$target" -max_total_time="$seconds" -timeout=25 -rss_limit_mb=2560 -max_len=65536 \
    -dict="$out/$name.dict" -artifact_prefix="$run/$name/artifacts/" -print_final_stats=1 \
    "$run/$name/corpus" "$run/$name/seeds" > "$run/$name/log.txt" 2>&1 &
  names+=("$name")
  pids+=($!)
done

failed=0
for i in "${!pids[@]}"; do
  name=${names[$i]}
  if wait "${pids[$i]}"; then
    result=ok
  else
    result=FAILED
    failed=1
  fi
  runs=$(sed -n 's/^stat::number_of_executed_units: *//p' "$run/$name/log.txt")
  echo "$name: $result, ${runs:-?} inputs run, $(find "$run/$name/corpus" -type f | wc -l) new in the corpus"
  if [ "$result" = FAILED ]; then
    grep -m 3 -E "ERROR: |SUMMARY: |runtime error: " "$run/$name/log.txt" | sed 's/^/    /'
    find "$run/$name/artifacts" -type f | sed 's/^/    /'
  fi
done
exit $failed
