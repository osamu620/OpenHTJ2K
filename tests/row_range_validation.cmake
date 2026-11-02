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

# Image origin.  set_row_range rows are relative to the first output row; the
# per-level IDWT window lives on the reference grid, so the decoder offsets it by
# the component origin (odd origins flip the row parity the lifting sees).  The
# kodim23 streams come from the encoder round-trip tests (Sorigin={1,0} and
# {7,0}); ds1_ht_01_b11 has origin y = 128 and a 2x1 subsampled component.
add_test(NAME rr_p1_ht_01_11 COMMAND row_range_compare ${CONFORMANCE_DATA_DIR}/ds1_ht_01_b11.j2k)
add_test(NAME rr_origin_y COMMAND row_range_compare kodim23_origin_y.j2c)
set_tests_properties(rr_origin_y PROPERTIES DEPENDS enc_origin_y)
add_test(NAME rr_origin_precincts COMMAND row_range_compare kodim23_origin_precincts.j2c)
set_tests_properties(rr_origin_precincts PROPERTIES DEPENDS enc_origin_precincts)
