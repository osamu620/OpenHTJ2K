#!/bin/bash
# Makes the seed corpus of every fuzz target, under <output directory>/<name>_fuzzer/:
#   - the inputs of corpus/<name>_fuzzer, which the test suite replays;
#   - for the targets that read a codestream: the codestreams and JPH files of tests/data
#     and the conformance codestreams below 64 KB, each followed by zero option bytes (the
#     default options; see fuzz_common.hpp);
#   - for image_reader_fuzzer: the PGM, PPM and PGX files of tests/data below 8 KB, each
#     followed by the option byte of its extension.
# usage: make_seed_corpus.sh <output directory>
set -eu
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
out=${1:?usage: make_seed_corpus.sh <output directory>}

for t in decode transcode jpip_client jpip_request jpip_server image_reader; do
  mkdir -p "$out/${t}_fuzzer"
  cp "$here/corpus/${t}_fuzzer/"* "$out/${t}_fuzzer/"
done

# seed <file> <target> <option bytes, as printf escapes>
seed() {
  {
    cat "$1"
    printf "$3"
  } > "$out/$2_fuzzer/$(basename "$1")"
}

find "$root/tests/data" "$root/conformance_data" -type f -size -65536c \
  \( -name '*.j2c' -o -name '*.j2k' -o -name '*.jhc' -o -name '*.jph' -o -name '*.jp2' \) | sort |
  while read -r f; do
    seed "$f" decode '\0\0\0\0\0\0\0\0'
    seed "$f" transcode '\0\0\0\0'
    case "$f" in
      *.jph | *.jp2) ;;  # the JPIP server reads a raw codestream
      *) seed "$f" jpip_server '\4\0\4\0\0\0\377\377' ;;  # the whole image
    esac
  done

find "$root/tests/data" -type f -size -8192c \( -name '*.pgm' -o -name '*.ppm' -o -name '*.pgx' \) | sort |
  while read -r f; do
    case "$f" in
      *.pgm) seed "$f" image_reader '\0' ;;
      *.ppm) seed "$f" image_reader '\4' ;;
      *.pgx) seed "$f" image_reader '\10' ;;
    esac
  done

for t in decode transcode jpip_client jpip_request jpip_server image_reader; do
  echo "${t}_fuzzer: $(find "$out/${t}_fuzzer" -type f | wc -l) seeds"
done
