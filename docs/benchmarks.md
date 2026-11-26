# Benchmarks — coding speed against OpenJPEG

This page backs the tables in the top-level README: lossless encode and decode
time, and lossy decode time, of OpenHTJ2K with its Part 1 (EBCOT/MQ) and its
HT (Part 15) block coder, against OpenJPEG, single-threaded, on x86-64.
Everything here is produced by two scripts, `tools/bench/lossless_vs_openjpeg.py`
and `tools/bench/lossy_decode_vs_openjpeg.py`, so the numbers can be
regenerated on any machine and refreshed at each release.

Measured 2026-11-26 on OpenHTJ2K 0.30.0 against OpenJPEG v2.5.4. The lossless
numbers are within 1% of those measured on 0.23.0 (2026-11-01).

## Lossless coding

4096×2160 8-bit RGB frame (see *Fixtures*). Medians of 10 interleaved
repetitions, single-threaded (`-threads 1` / `-num_threads 1`), one core.

**The two decoders' timers do not cover the same work.** OpenHTJ2K's decoder
streams rows to the output file while it decodes, so its reported time
includes writing the 26 MB PPM; OpenJPEG's `decode time` excludes its PPM
writer, which costs it about 230 ms at this size (compare the two tables
below). The codec-time decode row is therefore conservative for OpenHTJ2K.
The encode timers are comparable: both include writing the codestream and
exclude reading the PPM.

Codec time as reported by each tool:

| Lossless, one thread | OpenJPEG 2.5.4 | OpenHTJ2K Part 1 | OpenHTJ2K HT |
|---|---:|---:|---:|
| Encode | 798 ms | 555 ms (1.4×) | 69 ms (11.6×) |
| Decode | 582 ms | 548 ms (1.1×) | 64 ms (9.1×) |
| Codestream | 5.437 MB | 5.437 MB | 6.275 MB |

Wall time of the whole CLI process (read the PPM, code, write the output):

| Lossless, one thread | OpenJPEG 2.5.4 | OpenHTJ2K Part 1 | OpenHTJ2K HT |
|---|---:|---:|---:|
| Encode | 802 ms | 558 ms (1.4×) | 71 ms (11.3×) |
| Decode | 811 ms | 551 ms (1.5×) | 66 ms (12.2×) |

`kodim23.ppm` (768×512 8-bit RGB, `conformance_data/references/`), the
in-repository fixture, codec time:

| Lossless, one thread | OpenJPEG 2.5.4 | OpenHTJ2K Part 1 | OpenHTJ2K HT |
|---|---:|---:|---:|
| Encode | 52 ms | 44 ms (1.2×) | 4 ms (13.7×) |
| Decode | 42 ms | 44 ms (1.0×) | 3 ms (13.3×) |
| Codestream | 0.418 MB | 0.418 MB | 0.460 MB |

Reading the numbers:

- The Part 1 codestreams of the two encoders have the same size (within
  0.02%): same transform, same quantization (none), same block coder, same
  default coding parameters, so the encode rows compare the same work.
- The Part 1 decode rows decode the same codestream, the one OpenJPEG wrote.
- The HT columns are the same images through the HT block coder; the
  codestream is 15% larger, which is the usual price of HT's lossless mode.
- The two decode tables differ for OpenJPEG because of the timer scopes
  described above.
- Run-to-run spread is small: minimum and median differ by less than 2% in
  every cell of the 4K tables, and by at most 1 ms on kodim23.

## Lossy decoding

The same 4096×2160 frame coded by OpenHTJ2K's encoder with the irreversible
9/7 transform at `Qfactor=90`, once with the Part 1 block coder and once with
the HT block coder (the same quantization), and decoded by both decoders.
Medians of 10 interleaved repetitions, single-threaded, one core. The timer
note above applies here too: OpenHTJ2K's codec time includes writing the PPM,
OpenJPEG's does not.

Codec time as reported by each tool:

| Lossy 9/7 decode, one thread | OpenJPEG 2.5.4 | OpenHTJ2K |
|---|---:|---:|
| Part 1 codestream (0.618 MB, 0.56 bpp) | 190 ms | 89 ms (2.1×) |
| HT codestream (0.691 MB, 0.62 bpp) | 139 ms | 30 ms (4.7×) |

Wall time of the whole CLI process (read the codestream, decode, write the
PPM):

| Lossy 9/7 decode, one thread | OpenJPEG 2.5.4 | OpenHTJ2K |
|---|---:|---:|
| Part 1 codestream | 418 ms | 92 ms (4.5×) |
| HT codestream | 367 ms | 32 ms (11.5×) |

`kodim23.ppm` at `Qfactor=90`, codec time:

| Lossy 9/7 decode, one thread | OpenJPEG 2.5.4 | OpenHTJ2K |
|---|---:|---:|
| Part 1 codestream (0.042 MB, 0.85 bpp) | 7 ms | 6 ms (1.2×) |
| HT codestream (0.045 MB, 0.91 bpp) | 4 ms | 2 ms (2.2×) |

Reading the numbers:

- Both decoders decode the same codestreams, and their images agree within
  one level per sample (rounding): PSNR against the source is 45.14 dB for both
  on the 4K frame and 40.82 dB on kodim23, for either block coder.
- Encoding is not compared for lossy coding: OpenJPEG and OpenHTJ2K would have
  to be matched in rate and quality first.
- Minimum and median differ by at most 1 ms in every cell.

## Method

- **Hardware and OS:** AMD Ryzen 9 9950X, Linux 7.0, every run pinned to one
  core with `taskset -c 5`; the machine was otherwise idle.
- **Compilers and flags:** GCC 14.2 for both codecs, CMake `Release`
  (`-O3 -DNDEBUG`) plus `-march=native` (OpenHTJ2K's `OPENHTJ2K_TARGET_CPU=native`,
  the default; OpenJPEG configured with `-DCMAKE_C_FLAGS=-march=native`).
  OpenJPEG v2.5.4 built from its tag with `-DBUILD_CODEC=ON -DBUILD_SHARED_LIBS=ON`.
- **Threads:** one. `opj_compress` / `opj_decompress -threads 1`,
  `open_htj2k_enc` / `open_htj2k_dec -num_threads 1`.
- **Coding parameters:** the defaults of both encoders, which coincide:
  reversible 5/3 transform, five decomposition levels, 64×64 code-blocks, no
  precincts, one layer, one tile, RCT for RGB, no coder mode switches.
  OpenHTJ2K's Part 1 coder is selected with `Cmodes=0`; its HT coder is the
  default.
- **Timing:** two numbers per run. The codec time is what each tool prints
  (`encode time` / `decode time` for OpenJPEG, `elapsed time` for OpenHTJ2K;
  see the note above on what the decoders' timers enclose). The wall time is
  the whole process, measured by the script, with outputs on `/dev/shm`. Ten
  timed repetitions after one warm-up, the six commands interleaved so that
  frequency or thermal drift affects every column alike; the tables show
  medians.
- **Checks:** the lossless script refuses to print a table unless every
  decoded image is byte-identical to the input and OpenJPEG decodes the Part 1
  codestream that OpenHTJ2K wrote. The lossy script refuses unless the two
  decoders' images of each codestream agree within one level per sample.
- **Lossy codestreams:** `open_htj2k_enc Creversible=no Qfactor=90`, with
  `Cmodes=0` for Part 1; otherwise the same defaults as above (ICT for RGB).

## Fixtures

- **4K frame:** a 4096×2160 frame of *Elephants Dream* (Blender Foundation,
  CC BY 2.5), stored as an 8-bit RGB PPM; derived from a 16-bit PPM of the
  frame by keeping the high byte of each sample.
  SHA-256 of the 8-bit file:
  `47e7062fabc4d6774872e1c2b246d3e6ed0c6e1fbdc29a98be24147f974f9199`.
  (OpenJPEG 2.5.4 does not read 16-bit PPM, so the 16-bit frame was not used.)
- **kodim23:** `conformance_data/references/kodim23.ppm`, in the repository,
  so that row can be reproduced from a clean checkout.

Any 8-bit RGB PPM works with the script; the ratios above were stable across
the two sizes.

## Reproducing

```bash
# OpenJPEG at a pinned tag, same compiler, same optimisation level
git clone --depth 1 --branch v2.5.4 https://github.com/uclouvain/openjpeg.git openjpeg
cmake -S openjpeg -B openjpeg/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS=-march=native \
      -DBUILD_CODEC=ON -DBUILD_SHARED_LIBS=ON -DBUILD_TESTING=OFF
cmake --build openjpeg/build -j

# OpenHTJ2K (native target is the default)
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j

python3 tools/bench/lossless_vs_openjpeg.py --ohtj2k-bin build/bin \
        --opj-bin openjpeg/build/bin --reps 10 --cpu 5 --opj-label "OpenJPEG 2.5.4" image.ppm
python3 tools/bench/lossy_decode_vs_openjpeg.py --ohtj2k-bin build/bin \
        --opj-bin openjpeg/build/bin --reps 10 --cpu 5 --opj-label "OpenJPEG 2.5.4" image.ppm
```

Each script prints its tables in Markdown. Multi-threaded and Apple Silicon
numbers are deliberately not on this page; the README tables are refreshed from
these scripts at each release.
