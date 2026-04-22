# OpenHTJ2K documentation

The top-level [README](../README.md) gives the elevator pitch, the
one-command build, a short CLI quick-start, and a pointer to this
directory. Dig into the specific topics here.

## Build

- [**building.md**](building.md) — full CMake flag reference, native
  build, WebAssembly build + Node.js CLI decoder, experimental RTP
  receiver prerequisites.

## CLI applications

Every CLI prints its full option reference via `-h` at runtime; the
documents below are for offline browsing.

- [**cli_encoder.md**](cli_encoder.md) — `open_htj2k_enc` option
  reference and invocation examples.
- [**cli_decoder.md**](cli_decoder.md) — `open_htj2k_dec` option
  reference and invocation examples.
- [**cli_rtp_recv.md**](cli_rtp_recv.md) — `open_htj2k_rtp_recv`
  option reference, operational guide (kernel `rmem_max`), hardware
  requirements for 4K @ 60 fps, and known issues.

## JPIP (foveated streaming + gigapixel viewer)

- [**jpip.md**](jpip.md) — end-to-end JPIP (ISO/IEC 15444-9)
  reference: `open_htj2k_jpip_server`, `open_htj2k_jpip_demo`, and
  `open_htj2k_jpip_benchmark` CLIs; the in-browser foveation and
  pan-and-zoom viewer demos; core library architecture; wire-format
  details (JPP-stream, §C.9 cache model); WASM C API.

## Developer tools

- [**wasm_bench.md**](wasm_bench.md) — `subprojects/wasm_bench.mjs`
  iteration-loop benchmark harness for the WebAssembly decoder
  variants; byte-exact plane-dump mode for regression checks;
  profiling recipes.

## Other references

- [CHANGELOG](../CHANGELOG) — release history.
- [LICENSE](../LICENSE) — BSD 3-Clause.
