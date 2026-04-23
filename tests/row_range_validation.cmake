# set_row_range correctness.  Decodes each fixture at full canvas as a
# reference, then with several narrow [row_lo, row_hi) windows (bottom half,
# top half, non-aligned offset, narrow window) — asserts every in-window pixel
# is byte-identical to the reference.  Fixtures are SINGLE-TILE only; multi-
# tile row_range is out of scope for the initial landing (the general path in
# invoke_line_based_stream does not yet filter the per-tile-row accumulator by
# row_lo).

add_test(NAME rr_p0_04       COMMAND row_range_compare ${CONFORMANCE_DATA_DIR}/p0_04.j2k)
add_test(NAME rr_p0_05       COMMAND row_range_compare ${CONFORMANCE_DATA_DIR}/p0_05.j2k)
add_test(NAME rr_p0_06       COMMAND row_range_compare ${CONFORMANCE_DATA_DIR}/p0_06.j2k)
add_test(NAME rr_p0_ht_04_11 COMMAND row_range_compare ${CONFORMANCE_DATA_DIR}/ds0_ht_04_b11.j2k)
add_test(NAME rr_p0_ht_05_11 COMMAND row_range_compare ${CONFORMANCE_DATA_DIR}/ds0_ht_05_b11.j2k)
add_test(NAME rr_p0_ht_06_11 COMMAND row_range_compare ${CONFORMANCE_DATA_DIR}/ds0_ht_06_b11.j2k)
add_test(NAME rr_p0_ht_08_11 COMMAND row_range_compare ${CONFORMANCE_DATA_DIR}/ds0_ht_08_b11.j2k -reduce 1)
add_test(NAME rr_p1_ht_02_11 COMMAND row_range_compare ${CONFORMANCE_DATA_DIR}/ds1_ht_02_b11.j2k)
add_test(NAME rr_p1_ht_03_11 COMMAND row_range_compare ${CONFORMANCE_DATA_DIR}/ds1_ht_03_b11.j2k)
