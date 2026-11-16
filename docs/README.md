# OpenHTJ2K documentation

## Using OpenHTJ2K

- [**building.md**](building.md): building and installing. Covers the
  requirements, the CMake options, the WebAssembly build and the
  prerequisites of the streaming tools.
- [**cli_encoder.md**](cli_encoder.md): `open_htj2k_enc`, with its inputs,
  options, rate control and examples.
- [**cli_decoder.md**](cli_decoder.md): `open_htj2k_dec`, with its outputs,
  options and examples.
- [**qfactor.md**](qfactor.md): the `Qfactor` quality setting.
  - How it sets the step sizes.
  - Flat weights for non-visual data (`Qweights=flat`).
  - The precision-scaled Qfactor for more than 8 bits (`Qscale=bits`).
  - The experimental CSF weighting.
  - `estimate_qfactor`, which recovers the Qfactor of a codestream.
- [**library.md**](library.md): the C++ API, with decode and encode
  examples, threads, errors and warnings.
- [**benchmarks.md**](benchmarks.md): lossless speed against OpenJPEG, and
  how it was measured.

Every tool also prints its options with `-h`.

### Streaming (experimental)

- [**cli_rtp_recv.md**](cli_rtp_recv.md): `open_htj2k_rtp_recv`, the live
  RTP receiver (RFC 9828). Covers its options, operation (kernel
  `rmem_max`), the hardware for 4K at 60 fps, and known issues.
- [**wt_viewer.md**](wt_viewer.md): viewing live RFC 9828 streams in the
  browser over WebTransport. Covers the `wt_bridge` relay, the page, its URL
  parameters and troubleshooting.
- [**jpip.md**](jpip.md): JPIP (ISO/IEC 15444-9). Covers
  `open_htj2k_jpip_server`, `open_htj2k_jpip_demo`, `open_htj2k_jpip_benchmark`
  and the foveation and gigapixel demos in the browser.

## Working on OpenHTJ2K

Start with [CONTRIBUTING.md](../CONTRIBUTING.md). Background for particular
areas:

- [CONTRIBUTING.md](../CONTRIBUTING.md#build-options-for-development): the
  build options for development (C++ standard, SIMD code paths, profiling,
  the installed-package checks), and
  [running the test suite](building.md#running-the-test-suite).
- [benchmarks.md](benchmarks.md#reproducing): regenerating the benchmark
  table with `tools/bench/lossless_vs_openjpeg.py`.
- [wasm_bench.md](wasm_bench.md): `web/wasm_bench.mjs`, the benchmark
  harness for the WebAssembly decoder variants. Covers its byte-exact
  plane-dump mode for regression checks and profiling recipes.
- [planar_idwt_porting.md](planar_idwt_porting.md): the structure and
  verification protocol of the planar horizontal IDWT kernels on each SIMD
  platform.
- [qfactor.md](qfactor.md#calibration-and-provenance): how the Qfactor
  tables were calibrated.
- [jpip.md](jpip.md#core-architecture): JPIP internals, including data bins,
  the cache model, the JPP-stream format and the WASM C API.

## Other references

- [CHANGELOG](../CHANGELOG): release history.
- [LICENSE](../LICENSE): BSD 3-Clause.
