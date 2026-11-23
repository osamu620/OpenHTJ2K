#!/bin/bash
# Builds the fuzz targets with their dictionaries and seed corpora, laid out the way OSS-Fuzz
# and ClusterFuzzLite take them:
#   $OUT/<name>_fuzzer                   the target
#   $OUT/<name>_fuzzer.dict              its dictionary
#   $OUT/<name>_fuzzer_seed_corpus.zip   its seeds
# The environment gives the compiler and its flags (CXX, CXXFLAGS), the fuzzing engine to link
# (LIB_FUZZING_ENGINE; libFuzzer when unset), WORK (the build tree) and OUT.
# docs/dev/fuzzing.md has the values for a run on your own machine;
# .github/workflows/fuzz.yml uses them too.
#
# OPENHTJ2K_FUZZ_CPU chooses the code the targets exercise on x86-64: generic (the SSE2 code,
# the default: the targets run on any x86-64 machine) or x86-64-v3 (the AVX2 code: the
# machines that run the targets must have AVX2).
set -eu
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
: "${WORK:?set WORK to a directory for the build tree}"
: "${OUT:?set OUT to a directory for the fuzz targets}"
cpu=${OPENHTJ2K_FUZZ_CPU:-generic}
targets="decode transcode jpip_client jpip_request jpip_server image_reader"
mkdir -p "$WORK" "$OUT"

# Without libtiff: the targets do not read TIFF files, and a system libtiff is not built
# with the sanitizer of the run (MemorySanitizer needs every library instrumented).
cmake -S "$root" -B "$WORK/build" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
  -DOPENHTJ2K_BUILD_APPS=OFF -DOPENHTJ2K_BUILD_TESTS=OFF -DOPENHTJ2K_BUILD_FUZZERS=ON \
  -DOPENHTJ2K_TARGET_CPU="$cpu" -DCMAKE_DISABLE_FIND_PACKAGE_TIFF=ON
cmake --build "$WORK/build" -j"$(nproc)"

for t in $targets; do
  cp "$WORK/build/bin/${t}_fuzzer" "$OUT/"
done

for t in decode transcode jpip_client jpip_server; do
  cp "$here/codestream.dict" "$OUT/${t}_fuzzer.dict"
done
cp "$here/jpip_request.dict" "$OUT/jpip_request_fuzzer.dict"
cp "$here/image_reader.dict" "$OUT/image_reader_fuzzer.dict"

"$here/make_seed_corpus.sh" "$WORK/seeds"
for t in $targets; do
  rm -f "$OUT/${t}_fuzzer_seed_corpus.zip"
  (cd "$WORK/seeds/${t}_fuzzer" && zip -q -r "$OUT/${t}_fuzzer_seed_corpus.zip" .)
done
