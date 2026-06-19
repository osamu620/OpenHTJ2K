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

# Reuse-path variant — exercises the SIMD sub-range planar IDWT kernel on a
# single-tile, reuse-eligible 9/7 HT fixture (ds0_ht_06 decodes cleanly through
# the reuse machinery; some other conformance fixtures hit a pre-existing
# reuse-path codestream re-parse limitation and are intentionally not used here).
add_test(NAME cr_ht_06_reuse COMMAND col_range_compare ${CONFORMANCE_DATA_DIR}/ds0_ht_06_b11.j2k -reuse)
