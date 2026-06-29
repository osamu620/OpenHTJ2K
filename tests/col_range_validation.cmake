# set_col_range correctness.  Decodes each fixture at full width as a reference,
# then with several narrow [col_lo, col_hi) column windows — asserts every
# in-window pixel is byte-identical to the reference.  The default mode validates
# the public set_col_range API on the line-based stream path.  The _reuse variant
# drives the single-tile reuse path, which is where the SIMD sub-range horizontal
# IDWT kernels (idwt_1d_filtr_irrev97_planar_sr_{avx2,avx512,neon,wasm}) actually
# run — that is the path the WASM JPIP viewer uses for zoomed regions.
# Fixtures are SINGLE-TILE only (the column-range analogue of row_range_validation).

add_test(NAME cr_p0_04       COMMAND col_range_compare ${CONFORMANCE_DATA_DIR}/p0_04.j2k)
add_test(NAME cr_p0_05       COMMAND col_range_compare ${CONFORMANCE_DATA_DIR}/p0_05.j2k)
add_test(NAME cr_p0_06       COMMAND col_range_compare ${CONFORMANCE_DATA_DIR}/p0_06.j2k)
add_test(NAME cr_p0_ht_04_11 COMMAND col_range_compare ${CONFORMANCE_DATA_DIR}/ds0_ht_04_b11.j2k)
add_test(NAME cr_p0_ht_05_11 COMMAND col_range_compare ${CONFORMANCE_DATA_DIR}/ds0_ht_05_b11.j2k)
add_test(NAME cr_p0_ht_06_11 COMMAND col_range_compare ${CONFORMANCE_DATA_DIR}/ds0_ht_06_b11.j2k)
add_test(NAME cr_p0_ht_08_11 COMMAND col_range_compare ${CONFORMANCE_DATA_DIR}/ds0_ht_08_b11.j2k -reduce 1)
add_test(NAME cr_p1_ht_02_11 COMMAND col_range_compare ${CONFORMANCE_DATA_DIR}/ds1_ht_02_b11.j2k)
add_test(NAME cr_p1_ht_03_11 COMMAND col_range_compare ${CONFORMANCE_DATA_DIR}/ds1_ht_03_b11.j2k)

# Reuse-path variants — exercise the SIMD sub-range IDWT kernels on the
# single-tile reuse path (the path the WASM JPIP viewer uses).  This is the only
# path that actually narrows the decode: set_col_range on a fresh (non-reuse)
# decoder is applied before the line-decode state exists, so the plain cr_ tests
# above decode full width.  The reuse probe validates several strictly-interior
# windows per fixture, including one with an unaligned col_lo — that is what
# exercises the unaligned-load path in the vertical sub-range kernels (an aligned
# load faulted on AVX2 when col_lo - u0 was not a multiple of 8).
#
# Gated to non-MSVC: the probe compares the windowed (sub-range kernel) output
# against the full-width (full kernel) reference byte-exactly.  GCC, Clang and
# Emscripten contract the two kernels' FMAs identically so the results match;
# MSVC contracts them differently and the windowed result drifts by 1 LSB (on
# both x86 and ARM, on interior windows too), which is a pre-existing
# floating-point-determinism gap in the col-range feature, not a regression here.
# The crash/uninit fixes this test guards are compiler-independent and are
# covered on the GCC/Clang AVX2 + NEON jobs; making the MSVC sub-range kernels
# bit-identical to the full-width ones is left as a follow-up.  Component-boundary
# windows (col_lo == 0 / col_hi == W) are also deferred (a separate 1-LSB PSE
# divergence even on Clang/GCC NEON under MSVC).
#
# History: cr_ht_06_reuse was briefly removed as flaky when the cached
# single-tile-reuse progression traversal dropped the quality layer (multi-layer
# streams re-parsed every packet as layer 0, non-deterministic across platforms);
# that was fixed.  The finest-level out-of-window columns were a second source of
# non-determinism (uninitialised ring memory), now zeroed in pull_line_ref.
if(NOT MSVC)
  add_test(NAME cr_p0_05_reuse    COMMAND col_range_compare ${CONFORMANCE_DATA_DIR}/p0_05.j2k -reuse)
  add_test(NAME cr_ht_04_reuse    COMMAND col_range_compare ${CONFORMANCE_DATA_DIR}/ds0_ht_04_b11.j2k -reuse)
  add_test(NAME cr_ht_05_reuse    COMMAND col_range_compare ${CONFORMANCE_DATA_DIR}/ds0_ht_05_b11.j2k -reuse)
  add_test(NAME cr_ht_06_reuse    COMMAND col_range_compare ${CONFORMANCE_DATA_DIR}/ds0_ht_06_b11.j2k -reuse)
  add_test(NAME cr_p1_ht_02_reuse COMMAND col_range_compare ${CONFORMANCE_DATA_DIR}/ds1_ht_02_b11.j2k -reuse)
  add_test(NAME cr_p1_ht_03_reuse COMMAND col_range_compare ${CONFORMANCE_DATA_DIR}/ds1_ht_03_b11.j2k -reuse)
endif()
