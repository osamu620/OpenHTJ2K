# Conformance tests for the batch decode path (-batch flag).
# Mirrors decoder_conformance.cmake but drives open_htj2k_dec with -batch
# so that invoke() is exercised instead of the default invoke_line_based_stream().
# Output files are prefixed with "batch_" to avoid clashing with stream-path outputs.

# ── HT Profile 0 ─────────────────────────────────────────────────────────────
# Decoding
add_test(NAME batch_dec_p0_ht_01_11 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds0_ht_01_b11.j2k -o batch_ht_p0_01_b11.pgx -batch)
add_test(NAME batch_dec_p0_ht_02_11 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds0_ht_02_b11.j2k -o batch_ht_p0_02_b11.pgx -batch)
add_test(NAME batch_dec_p0_ht_03_11 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds0_ht_03_b11.j2k -o batch_ht_p0_03_b11.pgx -batch)
add_test(NAME batch_dec_p0_ht_03_14 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds0_ht_03_b14.j2k -o batch_ht_p0_03_b14.pgx -batch)
add_test(NAME batch_dec_p0_ht_04_11 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds0_ht_04_b11.j2k -o batch_ht_p0_04_b11.pgx -batch)
add_test(NAME batch_dec_p0_ht_04_12 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds0_ht_04_b12.j2k -o batch_ht_p0_04_b12.pgx -batch)
add_test(NAME batch_dec_p0_ht_05_11 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds0_ht_05_b11.j2k -o batch_ht_p0_05_b11.pgx -batch)
add_test(NAME batch_dec_p0_ht_05_12 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds0_ht_05_b12.j2k -o batch_ht_p0_05_b12.pgx -batch)
add_test(NAME batch_dec_p0_ht_06_11 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds0_ht_06_b11.j2k -o batch_ht_p0_06_b11.pgx -batch)
add_test(NAME batch_dec_p0_ht_06_15 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds0_ht_06_b15.j2k -o batch_ht_p0_06_b15.pgx -batch)
add_test(NAME batch_dec_p0_ht_06_18 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds0_ht_06_b18.j2k -o batch_ht_p0_06_b18.pgx -batch)
add_test(NAME batch_dec_p0_hm_06_11 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds0_hm_06_b11.j2k -o batch_hm_p0_06_b11.pgx -batch)
add_test(NAME batch_dec_p0_hm_06_18 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds0_hm_06_b18.j2k -o batch_hm_p0_06_b18.pgx -batch)
add_test(NAME batch_dec_p0_ht_07_11 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds0_ht_07_b11.j2k -o batch_ht_p0_07_b11.pgx -batch)
add_test(NAME batch_dec_p0_ht_07_15 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds0_ht_07_b15.j2k -o batch_ht_p0_07_b15.pgx -batch)
add_test(NAME batch_dec_p0_ht_07_16 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds0_ht_07_b16.j2k -o batch_ht_p0_07_b16.pgx -batch)
add_test(NAME batch_dec_p0_ht_08_11 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds0_ht_08_b11.j2k -o batch_ht_p0_08_b11.pgx -reduce 1 -batch)
add_test(NAME batch_dec_p0_ht_08_15 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds0_ht_08_b15.j2k -o batch_ht_p0_08_b15.pgx -reduce 1 -batch)
add_test(NAME batch_dec_p0_ht_08_16 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds0_ht_08_b16.j2k -o batch_ht_p0_08_b16.pgx -reduce 1 -batch)
add_test(NAME batch_dec_p0_ht_09_11 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds0_ht_09_b11.j2k -o batch_ht_p0_09_b11.pgx -batch)
add_test(NAME batch_dec_p0_ht_10_11 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds0_ht_10_b11.j2k -o batch_ht_p0_10_b11.pgx -batch)
add_test(NAME batch_dec_p0_ht_11_10 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds0_ht_11_b10.j2k -o batch_ht_p0_11_b10.pgx -batch)
add_test(NAME batch_dec_p0_ht_12_11 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds0_ht_12_b11.j2k -o batch_ht_p0_12_b11.pgx -batch)
add_test(NAME batch_dec_p0_ht_14_11 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds0_ht_14_b11.j2k -o batch_ht_p0_14_b11.pgx -batch)
add_test(NAME batch_dec_p0_ht_15_11 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds0_ht_15_b11.j2k -o batch_ht_p0_15_b11.pgx -batch)
add_test(NAME batch_dec_p0_ht_15_14 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds0_ht_15_b14.j2k -o batch_ht_p0_15_b14.pgx -batch)
add_test(NAME batch_dec_p0_hm_15_8  COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds0_hm_15_b8.j2k  -o batch_hm_p0_15_b8.pgx  -batch)
add_test(NAME batch_dec_p0_ht_16_11 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds0_ht_16_b11.j2k -o batch_ht_p0_16_b11.pgx -batch)

# PAE / MSE comparisons
# 1
add_test(NAME batch_comp_p0_ht_01_11 COMMAND imgcmp batch_ht_p0_01_b11_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_01-0.pgx 0 0)
set_tests_properties(batch_comp_p0_ht_01_11 PROPERTIES DEPENDS batch_dec_p0_ht_01_11)
# 2
add_test(NAME batch_comp_p0_ht_02_11 COMMAND imgcmp batch_ht_p0_02_b11_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_02-0.pgx 1 0.001)
set_tests_properties(batch_comp_p0_ht_02_11 PROPERTIES DEPENDS batch_dec_p0_ht_02_11)
# 3
add_test(NAME batch_comp_p0_ht_03_11 COMMAND imgcmp batch_ht_p0_03_b11_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_03-0.pgx 17 0.15)
add_test(NAME batch_comp_p0_ht_03_14 COMMAND imgcmp batch_ht_p0_03_b14_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_03-0.pgx 0 0)
set_tests_properties(batch_comp_p0_ht_03_11 PROPERTIES DEPENDS batch_dec_p0_ht_03_11)
set_tests_properties(batch_comp_p0_ht_03_14 PROPERTIES DEPENDS batch_dec_p0_ht_03_14)
# 4
add_test(NAME batch_comp_p0_ht_04_11r COMMAND imgcmp batch_ht_p0_04_b11_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_04-0.pgx 7 0.876)
add_test(NAME batch_comp_p0_ht_04_11g COMMAND imgcmp batch_ht_p0_04_b11_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_04-1.pgx 6 0.726)
add_test(NAME batch_comp_p0_ht_04_11b COMMAND imgcmp batch_ht_p0_04_b11_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_04-2.pgx 8 1.170)
add_test(NAME batch_comp_p0_ht_04_12r COMMAND imgcmp batch_ht_p0_04_b12_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_04-0.pgx 5 0.776)
add_test(NAME batch_comp_p0_ht_04_12g COMMAND imgcmp batch_ht_p0_04_b12_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_04-1.pgx 4 0.626)
add_test(NAME batch_comp_p0_ht_04_12b COMMAND imgcmp batch_ht_p0_04_b12_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_04-2.pgx 6 1.070)
set_tests_properties(batch_comp_p0_ht_04_11r batch_comp_p0_ht_04_11g batch_comp_p0_ht_04_11b PROPERTIES DEPENDS batch_dec_p0_ht_04_11)
set_tests_properties(batch_comp_p0_ht_04_12r batch_comp_p0_ht_04_12g batch_comp_p0_ht_04_12b PROPERTIES DEPENDS batch_dec_p0_ht_04_12)
# 5
add_test(NAME batch_comp_p0_ht_05_11a COMMAND imgcmp batch_ht_p0_05_b11_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_05-0.pgx 2 0.319)
add_test(NAME batch_comp_p0_ht_05_11b COMMAND imgcmp batch_ht_p0_05_b11_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_05-1.pgx 3 0.324)
add_test(NAME batch_comp_p0_ht_05_11c COMMAND imgcmp batch_ht_p0_05_b11_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_05-2.pgx 3 0.318)
add_test(NAME batch_comp_p0_ht_05_11d COMMAND imgcmp batch_ht_p0_05_b11_03.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_05-3.pgx 2 0.001)
add_test(NAME batch_comp_p0_ht_05_12a COMMAND imgcmp batch_ht_p0_05_b12_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_05-0.pgx 2 0.319)
add_test(NAME batch_comp_p0_ht_05_12b COMMAND imgcmp batch_ht_p0_05_b12_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_05-1.pgx 2 0.323)
add_test(NAME batch_comp_p0_ht_05_12c COMMAND imgcmp batch_ht_p0_05_b12_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_05-2.pgx 2 0.317)
add_test(NAME batch_comp_p0_ht_05_12d COMMAND imgcmp batch_ht_p0_05_b12_03.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_05-3.pgx 0 0)
set_tests_properties(batch_comp_p0_ht_05_11a batch_comp_p0_ht_05_11b batch_comp_p0_ht_05_11c batch_comp_p0_ht_05_11d PROPERTIES DEPENDS batch_dec_p0_ht_05_11)
set_tests_properties(batch_comp_p0_ht_05_12a batch_comp_p0_ht_05_12b batch_comp_p0_ht_05_12c batch_comp_p0_ht_05_12d PROPERTIES DEPENDS batch_dec_p0_ht_05_12)
# 6
add_test(NAME batch_comp_p0_ht_06_11a COMMAND imgcmp batch_ht_p0_06_b11_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_06-0.pgx 3135 86287)
add_test(NAME batch_comp_p0_ht_06_11b COMMAND imgcmp batch_ht_p0_06_b11_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_06-1.pgx 403 6124)
add_test(NAME batch_comp_p0_ht_06_11c COMMAND imgcmp batch_ht_p0_06_b11_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_06-2.pgx 378 3968)
add_test(NAME batch_comp_p0_ht_06_11d COMMAND imgcmp batch_ht_p0_06_b11_03.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_06-3.pgx 200 2000)
add_test(NAME batch_comp_p0_ht_06_15a COMMAND imgcmp batch_ht_p0_06_b15_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_06-0.pgx 635 11287)
add_test(NAME batch_comp_p0_ht_06_15b COMMAND imgcmp batch_ht_p0_06_b15_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_06-1.pgx 403 6124)
add_test(NAME batch_comp_p0_ht_06_15c COMMAND imgcmp batch_ht_p0_06_b15_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_06-2.pgx 378 3968)
add_test(NAME batch_comp_p0_ht_06_15d COMMAND imgcmp batch_ht_p0_06_b15_03.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_06-3.pgx 12 10.0)
add_test(NAME batch_comp_p0_ht_06_18a COMMAND imgcmp batch_ht_p0_06_b18_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_06-0.pgx 635 11287)
add_test(NAME batch_comp_p0_ht_06_18b COMMAND imgcmp batch_ht_p0_06_b18_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_06-1.pgx 403 6124)
add_test(NAME batch_comp_p0_ht_06_18c COMMAND imgcmp batch_ht_p0_06_b18_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_06-2.pgx 378 3968)
add_test(NAME batch_comp_p0_ht_06_18d COMMAND imgcmp batch_ht_p0_06_b18_03.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_06-3.pgx 0 0)
add_test(NAME batch_comp_p0_hm_06_11a COMMAND imgcmp batch_hm_p0_06_b11_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_06-0.pgx 635 11287)
add_test(NAME batch_comp_p0_hm_06_11b COMMAND imgcmp batch_hm_p0_06_b11_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_06-1.pgx 403 6124)
add_test(NAME batch_comp_p0_hm_06_11c COMMAND imgcmp batch_hm_p0_06_b11_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_06-2.pgx 378 3968)
add_test(NAME batch_comp_p0_hm_06_11d COMMAND imgcmp batch_hm_p0_06_b11_03.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_06-3.pgx 200 2000)
add_test(NAME batch_comp_p0_hm_06_18a COMMAND imgcmp batch_hm_p0_06_b18_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_06-0.pgx 635 11287)
add_test(NAME batch_comp_p0_hm_06_18b COMMAND imgcmp batch_hm_p0_06_b18_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_06-1.pgx 403 6124)
add_test(NAME batch_comp_p0_hm_06_18c COMMAND imgcmp batch_hm_p0_06_b18_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_06-2.pgx 378 3968)
add_test(NAME batch_comp_p0_hm_06_18d COMMAND imgcmp batch_hm_p0_06_b18_03.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_06-3.pgx 0 0)
set_tests_properties(batch_comp_p0_ht_06_11a batch_comp_p0_ht_06_11b batch_comp_p0_ht_06_11c batch_comp_p0_ht_06_11d PROPERTIES DEPENDS batch_dec_p0_ht_06_11)
set_tests_properties(batch_comp_p0_ht_06_15a batch_comp_p0_ht_06_15b batch_comp_p0_ht_06_15c batch_comp_p0_ht_06_15d PROPERTIES DEPENDS batch_dec_p0_ht_06_15)
set_tests_properties(batch_comp_p0_ht_06_18a batch_comp_p0_ht_06_18b batch_comp_p0_ht_06_18c batch_comp_p0_ht_06_18d PROPERTIES DEPENDS batch_dec_p0_ht_06_18)
set_tests_properties(batch_comp_p0_hm_06_11a batch_comp_p0_hm_06_11b batch_comp_p0_hm_06_11c batch_comp_p0_hm_06_11d PROPERTIES DEPENDS batch_dec_p0_hm_06_11)
set_tests_properties(batch_comp_p0_hm_06_18a batch_comp_p0_hm_06_18b batch_comp_p0_hm_06_18c batch_comp_p0_hm_06_18d PROPERTIES DEPENDS batch_dec_p0_hm_06_18)
# 7
add_test(NAME batch_comp_p0_ht_07_11r COMMAND imgcmp batch_ht_p0_07_b11_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_07-0.pgx 40 25.0)
add_test(NAME batch_comp_p0_ht_07_11g COMMAND imgcmp batch_ht_p0_07_b11_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_07-1.pgx 40 25.0)
add_test(NAME batch_comp_p0_ht_07_11b COMMAND imgcmp batch_ht_p0_07_b11_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_07-2.pgx 50 25.0)
add_test(NAME batch_comp_p0_ht_07_15r COMMAND imgcmp batch_ht_p0_07_b15_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_07-0.pgx 2 0.075)
add_test(NAME batch_comp_p0_ht_07_15g COMMAND imgcmp batch_ht_p0_07_b15_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_07-1.pgx 2 0.05)
add_test(NAME batch_comp_p0_ht_07_15b COMMAND imgcmp batch_ht_p0_07_b15_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_07-2.pgx 2 0.075)
add_test(NAME batch_comp_p0_ht_07_16r COMMAND imgcmp batch_ht_p0_07_b16_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_07-0.pgx 0 0)
add_test(NAME batch_comp_p0_ht_07_16g COMMAND imgcmp batch_ht_p0_07_b16_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_07-1.pgx 0 0)
add_test(NAME batch_comp_p0_ht_07_16b COMMAND imgcmp batch_ht_p0_07_b16_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_07-2.pgx 0 0)
set_tests_properties(batch_comp_p0_ht_07_11r batch_comp_p0_ht_07_11g batch_comp_p0_ht_07_11b PROPERTIES DEPENDS batch_dec_p0_ht_07_11)
set_tests_properties(batch_comp_p0_ht_07_15r batch_comp_p0_ht_07_15g batch_comp_p0_ht_07_15b PROPERTIES DEPENDS batch_dec_p0_ht_07_15)
set_tests_properties(batch_comp_p0_ht_07_16r batch_comp_p0_ht_07_16g batch_comp_p0_ht_07_16b PROPERTIES DEPENDS batch_dec_p0_ht_07_16)
# 8
add_test(NAME batch_comp_p0_ht_08_11r COMMAND imgcmp batch_ht_p0_08_b11_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_08-0.pgx 40 45.0)
add_test(NAME batch_comp_p0_ht_08_11g COMMAND imgcmp batch_ht_p0_08_b11_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_08-1.pgx 40 30.0)
add_test(NAME batch_comp_p0_ht_08_11b COMMAND imgcmp batch_ht_p0_08_b11_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_08-2.pgx 40 45.0)
add_test(NAME batch_comp_p0_ht_08_15r COMMAND imgcmp batch_ht_p0_08_b15_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_08-0.pgx 0 0)
add_test(NAME batch_comp_p0_ht_08_15g COMMAND imgcmp batch_ht_p0_08_b15_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_08-1.pgx 1 0.001)
add_test(NAME batch_comp_p0_ht_08_15b COMMAND imgcmp batch_ht_p0_08_b15_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_08-2.pgx 0 0)
add_test(NAME batch_comp_p0_ht_08_16r COMMAND imgcmp batch_ht_p0_08_b16_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_08-0.pgx 0 0)
add_test(NAME batch_comp_p0_ht_08_16g COMMAND imgcmp batch_ht_p0_08_b16_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_08-1.pgx 0 0)
add_test(NAME batch_comp_p0_ht_08_16b COMMAND imgcmp batch_ht_p0_08_b16_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_08-2.pgx 0 0)
set_tests_properties(batch_comp_p0_ht_08_11r batch_comp_p0_ht_08_11g batch_comp_p0_ht_08_11b PROPERTIES DEPENDS batch_dec_p0_ht_08_11)
set_tests_properties(batch_comp_p0_ht_08_15r batch_comp_p0_ht_08_15g batch_comp_p0_ht_08_15b PROPERTIES DEPENDS batch_dec_p0_ht_08_15)
set_tests_properties(batch_comp_p0_ht_08_16r batch_comp_p0_ht_08_16g batch_comp_p0_ht_08_16b PROPERTIES DEPENDS batch_dec_p0_ht_08_16)
# 9
add_test(NAME batch_comp_p0_ht_09_11 COMMAND imgcmp batch_ht_p0_09_b11_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_09-0.pgx 0 0)
set_tests_properties(batch_comp_p0_ht_09_11 PROPERTIES DEPENDS batch_dec_p0_ht_09_11)
# 10
add_test(NAME batch_comp_p0_ht_10_11r COMMAND imgcmp batch_ht_p0_10_b11_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_10-0.pgx 0 0)
add_test(NAME batch_comp_p0_ht_10_11g COMMAND imgcmp batch_ht_p0_10_b11_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_10-1.pgx 0 0)
add_test(NAME batch_comp_p0_ht_10_11b COMMAND imgcmp batch_ht_p0_10_b11_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_10-2.pgx 0 0)
set_tests_properties(batch_comp_p0_ht_10_11r batch_comp_p0_ht_10_11g batch_comp_p0_ht_10_11b PROPERTIES DEPENDS batch_dec_p0_ht_10_11)
# 11
add_test(NAME batch_comp_p0_ht_11_10 COMMAND imgcmp batch_ht_p0_11_b10_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_11-0.pgx 0 0)
set_tests_properties(batch_comp_p0_ht_11_10 PROPERTIES DEPENDS batch_dec_p0_ht_11_10)
# 12
add_test(NAME batch_comp_p0_ht_12_11 COMMAND imgcmp batch_ht_p0_12_b11_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_12-0.pgx 0 0)
set_tests_properties(batch_comp_p0_ht_12_11 PROPERTIES DEPENDS batch_dec_p0_ht_12_11)
# 14
add_test(NAME batch_comp_p0_ht_14_11r COMMAND imgcmp batch_ht_p0_14_b11_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_14-0.pgx 0 0)
add_test(NAME batch_comp_p0_ht_14_11g COMMAND imgcmp batch_ht_p0_14_b11_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_14-1.pgx 0 0)
add_test(NAME batch_comp_p0_ht_14_11b COMMAND imgcmp batch_ht_p0_14_b11_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_14-2.pgx 0 0)
set_tests_properties(batch_comp_p0_ht_14_11r batch_comp_p0_ht_14_11g batch_comp_p0_ht_14_11b PROPERTIES DEPENDS batch_dec_p0_ht_14_11)
# 15
add_test(NAME batch_comp_p0_ht_15_11 COMMAND imgcmp batch_ht_p0_15_b11_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_15-0.pgx 17 0.15)
add_test(NAME batch_comp_p0_ht_15_14 COMMAND imgcmp batch_ht_p0_15_b14_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_15-0.pgx 0 0)
add_test(NAME batch_comp_p0_hm_15_8  COMMAND imgcmp batch_hm_p0_15_b8_00.pgx  ${CONFORMANCE_DATA_DIR}/references/c1p0_15-0.pgx 0 0)
set_tests_properties(batch_comp_p0_ht_15_11 PROPERTIES DEPENDS batch_dec_p0_ht_15_11)
set_tests_properties(batch_comp_p0_ht_15_14 PROPERTIES DEPENDS batch_dec_p0_ht_15_14)
set_tests_properties(batch_comp_p0_hm_15_8  PROPERTIES DEPENDS batch_dec_p0_hm_15_8)
# 16
add_test(NAME batch_comp_p0_ht_16_11 COMMAND imgcmp batch_ht_p0_16_b11_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_16-0.pgx 0 0)
set_tests_properties(batch_comp_p0_ht_16_11 PROPERTIES DEPENDS batch_dec_p0_ht_16_11)

# ── HT Profile 1 ─────────────────────────────────────────────────────────────
# Decoding
add_test(NAME batch_dec_p1_ht_01_11 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds1_ht_01_b11.j2k -o batch_ht_p1_01_b11.pgx -batch)
add_test(NAME batch_dec_p1_ht_01_12 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds1_ht_01_b12.j2k -o batch_ht_p1_01_b12.pgx -batch)
add_test(NAME batch_dec_p1_ht_02_11 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds1_ht_02_b11.j2k -o batch_ht_p1_02_b11.pgx -batch)
add_test(NAME batch_dec_p1_ht_02_12 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds1_ht_02_b12.j2k -o batch_ht_p1_02_b12.pgx -batch)
add_test(NAME batch_dec_p1_ht_03_11 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds1_ht_03_b11.j2k -o batch_ht_p1_03_b11.pgx -batch)
add_test(NAME batch_dec_p1_ht_03_12 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds1_ht_03_b12.j2k -o batch_ht_p1_03_b12.pgx -batch)
add_test(NAME batch_dec_p1_ht_04_9  COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds1_ht_04_b9.j2k  -o batch_ht_p1_04_b9.pgx  -batch)
add_test(NAME batch_dec_p1_ht_05_11 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds1_ht_05_b11.j2k -o batch_ht_p1_05_b11.pgx -batch)
add_test(NAME batch_dec_p1_ht_06_11 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds1_ht_06_b11.j2k -o batch_ht_p1_06_b11.pgx -batch)
add_test(NAME batch_dec_p1_ht_07_11 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds1_ht_07_b11.j2k -o batch_ht_p1_07_b11.pgx -batch)

# PAE / MSE comparisons
# 1
add_test(NAME batch_comp_p1_ht_01_11 COMMAND imgcmp batch_ht_p1_01_b11_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_01-0.pgx 1 0.001)
add_test(NAME batch_comp_p1_ht_01_12 COMMAND imgcmp batch_ht_p1_01_b12_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_01-0.pgx 0 0)
set_tests_properties(batch_comp_p1_ht_01_11 PROPERTIES DEPENDS batch_dec_p1_ht_01_11)
set_tests_properties(batch_comp_p1_ht_01_12 PROPERTIES DEPENDS batch_dec_p1_ht_01_12)
# 2
add_test(NAME batch_comp_p1_ht_02_11r COMMAND imgcmp batch_ht_p1_02_b11_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_02-0.pgx 7 0.865)
add_test(NAME batch_comp_p1_ht_02_11g COMMAND imgcmp batch_ht_p1_02_b11_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_02-1.pgx 6 0.716)
add_test(NAME batch_comp_p1_ht_02_11b COMMAND imgcmp batch_ht_p1_02_b11_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_02-2.pgx 8 1.151)
add_test(NAME batch_comp_p1_ht_02_12r COMMAND imgcmp batch_ht_p1_02_b12_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_02-0.pgx 5 0.765)
add_test(NAME batch_comp_p1_ht_02_12g COMMAND imgcmp batch_ht_p1_02_b12_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_02-1.pgx 4 0.616)
add_test(NAME batch_comp_p1_ht_02_12b COMMAND imgcmp batch_ht_p1_02_b12_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_02-2.pgx 6 1.051)
set_tests_properties(batch_comp_p1_ht_02_11r batch_comp_p1_ht_02_11g batch_comp_p1_ht_02_11b PROPERTIES DEPENDS batch_dec_p1_ht_02_11)
set_tests_properties(batch_comp_p1_ht_02_12r batch_comp_p1_ht_02_12g batch_comp_p1_ht_02_12b PROPERTIES DEPENDS batch_dec_p1_ht_02_12)
# 3
add_test(NAME batch_comp_p1_ht_03_11a COMMAND imgcmp batch_ht_p1_03_b11_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_03-0.pgx 2 0.311)
add_test(NAME batch_comp_p1_ht_03_11b COMMAND imgcmp batch_ht_p1_03_b11_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_03-1.pgx 3 0.310)
add_test(NAME batch_comp_p1_ht_03_11c COMMAND imgcmp batch_ht_p1_03_b11_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_03-2.pgx 3 0.317)
add_test(NAME batch_comp_p1_ht_03_11d COMMAND imgcmp batch_ht_p1_03_b11_03.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_03-3.pgx 2 0.001)
add_test(NAME batch_comp_p1_ht_03_12a COMMAND imgcmp batch_ht_p1_03_b12_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_03-0.pgx 2 0.311)
add_test(NAME batch_comp_p1_ht_03_12b COMMAND imgcmp batch_ht_p1_03_b12_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_03-1.pgx 2 0.280)
add_test(NAME batch_comp_p1_ht_03_12c COMMAND imgcmp batch_ht_p1_03_b12_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_03-2.pgx 1 0.267)
add_test(NAME batch_comp_p1_ht_03_12d COMMAND imgcmp batch_ht_p1_03_b12_03.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_03-3.pgx 0 0)
set_tests_properties(batch_comp_p1_ht_03_11a batch_comp_p1_ht_03_11b batch_comp_p1_ht_03_11c batch_comp_p1_ht_03_11d PROPERTIES DEPENDS batch_dec_p1_ht_03_11)
set_tests_properties(batch_comp_p1_ht_03_12a batch_comp_p1_ht_03_12b batch_comp_p1_ht_03_12c batch_comp_p1_ht_03_12d PROPERTIES DEPENDS batch_dec_p1_ht_03_12)
set_tests_properties(batch_comp_p1_ht_03_12b PROPERTIES WILL_FAIL false)
set_tests_properties(batch_comp_p1_ht_03_12c PROPERTIES WILL_FAIL false)
# 4
add_test(NAME batch_comp_p1_ht_04_9 COMMAND imgcmp batch_ht_p1_04_b9_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_04-0.pgx 624 3080)
set_tests_properties(batch_comp_p1_ht_04_9 PROPERTIES DEPENDS batch_dec_p1_ht_04_9)
# 5
add_test(NAME batch_comp_p1_ht_05_11r COMMAND imgcmp batch_ht_p1_05_b11_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_05-0.pgx 40 8.458)
add_test(NAME batch_comp_p1_ht_05_11g COMMAND imgcmp batch_ht_p1_05_b11_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_05-1.pgx 40 9.716)
add_test(NAME batch_comp_p1_ht_05_11b COMMAND imgcmp batch_ht_p1_05_b11_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_05-2.pgx 40 10.154)
set_tests_properties(batch_comp_p1_ht_05_11r batch_comp_p1_ht_05_11g batch_comp_p1_ht_05_11b PROPERTIES DEPENDS batch_dec_p1_ht_05_11)
# 6
add_test(NAME batch_comp_p1_ht_06_11r COMMAND imgcmp batch_ht_p1_06_b11_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_06-0.pgx 2 0.600)
add_test(NAME batch_comp_p1_ht_06_11g COMMAND imgcmp batch_ht_p1_06_b11_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_06-1.pgx 2 0.600)
add_test(NAME batch_comp_p1_ht_06_11b COMMAND imgcmp batch_ht_p1_06_b11_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_06-2.pgx 2 0.600)
set_tests_properties(batch_comp_p1_ht_06_11r batch_comp_p1_ht_06_11g batch_comp_p1_ht_06_11b PROPERTIES DEPENDS batch_dec_p1_ht_06_11)
# 7
add_test(NAME batch_comp_p1_ht_07_11a COMMAND imgcmp batch_ht_p1_07_b11_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_07-0.pgx 0 0)
add_test(NAME batch_comp_p1_ht_07_11b COMMAND imgcmp batch_ht_p1_07_b11_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_07-1.pgx 0 0)
set_tests_properties(batch_comp_p1_ht_07_11a batch_comp_p1_ht_07_11b PROPERTIES DEPENDS batch_dec_p1_ht_07_11)

# ── HT HiFi ──────────────────────────────────────────────────────────────────
# Decoding
add_test(NAME batch_dec_HF_ht1_02 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/hifi_ht1_02.j2k -o batch_HF_ht1_02.pgx -batch)

# PAE / MSE comparisons
add_test(NAME batch_comp_HF_ht1_02_0 COMMAND imgcmp batch_HF_ht1_02_00.pgx ${CONFORMANCE_DATA_DIR}/references/hifi-0.pgx 47 82)
add_test(NAME batch_comp_HF_ht1_02_1 COMMAND imgcmp batch_HF_ht1_02_01.pgx ${CONFORMANCE_DATA_DIR}/references/hifi-1.pgx 36 65)
add_test(NAME batch_comp_HF_ht1_02_2 COMMAND imgcmp batch_HF_ht1_02_02.pgx ${CONFORMANCE_DATA_DIR}/references/hifi-2.pgx 42 86)
set_tests_properties(batch_comp_HF_ht1_02_0 batch_comp_HF_ht1_02_1 batch_comp_HF_ht1_02_2 PROPERTIES DEPENDS batch_dec_HF_ht1_02)

# ── Part 1 HiFi ──────────────────────────────────────────────────────────────
# Decoding
add_test(NAME batch_dec_HF_p1_02 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/hifi_p1_02.j2k -o batch_HF_p1_02.pgx -batch)

# PAE / MSE comparisons
add_test(NAME batch_comp_HF_p1_02_0 COMMAND imgcmp batch_HF_p1_02_00.pgx ${CONFORMANCE_DATA_DIR}/references/hifi-0.pgx 43 80)
add_test(NAME batch_comp_HF_p1_02_1 COMMAND imgcmp batch_HF_p1_02_01.pgx ${CONFORMANCE_DATA_DIR}/references/hifi-1.pgx 33 62)
add_test(NAME batch_comp_HF_p1_02_2 COMMAND imgcmp batch_HF_p1_02_02.pgx ${CONFORMANCE_DATA_DIR}/references/hifi-2.pgx 38 72)
set_tests_properties(batch_comp_HF_p1_02_0 batch_comp_HF_p1_02_1 batch_comp_HF_p1_02_2 PROPERTIES DEPENDS batch_dec_HF_p1_02)

# ── Part 1 Profile 0 ─────────────────────────────────────────────────────────
# Decoding
add_test(NAME batch_dec_p0_01 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/p0_01.j2k -o batch_p0_01.pgx -batch)
add_test(NAME batch_dec_p0_02 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/p0_02.j2k -o batch_p0_02.pgx -batch)
add_test(NAME batch_dec_p0_03 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/p0_03.j2k -o batch_p0_03.pgx -batch)
add_test(NAME batch_dec_p0_04 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/p0_04.j2k -o batch_p0_04.pgx -batch)
add_test(NAME batch_dec_p0_05 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/p0_05.j2k -o batch_p0_05.pgx -batch)
add_test(NAME batch_dec_p0_06 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/p0_06.j2k -o batch_p0_06.pgx -batch)
add_test(NAME batch_dec_p0_07 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/p0_07.j2k -o batch_p0_07.pgx -batch)
add_test(NAME batch_dec_p0_08 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/p0_08.j2k -o batch_p0_08.pgx -reduce 1 -batch)
add_test(NAME batch_dec_p0_09 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/p0_09.j2k -o batch_p0_09.pgx -batch)
add_test(NAME batch_dec_p0_10 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/p0_10.j2k -o batch_p0_10.pgx -batch)
add_test(NAME batch_dec_p0_11 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/p0_11.j2k -o batch_p0_11.pgx -batch)
add_test(NAME batch_dec_p0_12 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/p0_12.j2k -o batch_p0_12.pgx -batch)
add_test(NAME batch_dec_p0_14 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/p0_14.j2k -o batch_p0_14.pgx -batch)
add_test(NAME batch_dec_p0_15 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/p0_15.j2k -o batch_p0_15.pgx -batch)
add_test(NAME batch_dec_p0_16 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/p0_16.j2k -o batch_p0_16.pgx -batch)

# PAE / MSE comparisons
# 1
add_test(NAME batch_comp_p0_01 COMMAND imgcmp batch_p0_01_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_01-0.pgx 0 0)
set_tests_properties(batch_comp_p0_01 PROPERTIES DEPENDS batch_dec_p0_01)
# 2
add_test(NAME batch_comp_p0_02 COMMAND imgcmp batch_p0_02_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_02-0.pgx 0 0)
set_tests_properties(batch_comp_p0_02 PROPERTIES DEPENDS batch_dec_p0_02)
# 3
add_test(NAME batch_comp_p0_03 COMMAND imgcmp batch_p0_03_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_03-0.pgx 0 0)
set_tests_properties(batch_comp_p0_03 PROPERTIES DEPENDS batch_dec_p0_03)
# 4
add_test(NAME batch_comp_p0_04_r COMMAND imgcmp batch_p0_04_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_04-0.pgx 5 0.776)
add_test(NAME batch_comp_p0_04_g COMMAND imgcmp batch_p0_04_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_04-1.pgx 4 0.626)
add_test(NAME batch_comp_p0_04_b COMMAND imgcmp batch_p0_04_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_04-2.pgx 6 1.070)
set_tests_properties(batch_comp_p0_04_r batch_comp_p0_04_g batch_comp_p0_04_b PROPERTIES DEPENDS batch_dec_p0_04)
# 5
add_test(NAME batch_comp_p0_05_a COMMAND imgcmp batch_p0_05_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_05-0.pgx 2 0.319)
add_test(NAME batch_comp_p0_05_b COMMAND imgcmp batch_p0_05_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_05-1.pgx 2 0.323)
add_test(NAME batch_comp_p0_05_c COMMAND imgcmp batch_p0_05_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_05-2.pgx 2 0.317)
add_test(NAME batch_comp_p0_05_d COMMAND imgcmp batch_p0_05_03.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_05-3.pgx 0 0)
set_tests_properties(batch_comp_p0_05_a batch_comp_p0_05_b batch_comp_p0_05_c batch_comp_p0_05_d PROPERTIES DEPENDS batch_dec_p0_05)
# 6
add_test(NAME batch_comp_p0_06_a COMMAND imgcmp batch_p0_06_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_06-0.pgx 635 11287)
add_test(NAME batch_comp_p0_06_b COMMAND imgcmp batch_p0_06_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_06-1.pgx 403 6124)
add_test(NAME batch_comp_p0_06_c COMMAND imgcmp batch_p0_06_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_06-2.pgx 378 3968)
add_test(NAME batch_comp_p0_06_d COMMAND imgcmp batch_p0_06_03.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_06-3.pgx 0 0)
set_tests_properties(batch_comp_p0_06_a batch_comp_p0_06_b batch_comp_p0_06_c batch_comp_p0_06_d PROPERTIES DEPENDS batch_dec_p0_06)
# 7
add_test(NAME batch_comp_p0_07_r COMMAND imgcmp batch_p0_07_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_07-0.pgx 0 0)
add_test(NAME batch_comp_p0_07_g COMMAND imgcmp batch_p0_07_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_07-1.pgx 0 0)
add_test(NAME batch_comp_p0_07_b COMMAND imgcmp batch_p0_07_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_07-2.pgx 0 0)
set_tests_properties(batch_comp_p0_07_r batch_comp_p0_07_g batch_comp_p0_07_b PROPERTIES DEPENDS batch_dec_p0_07)
# 8
add_test(NAME batch_comp_p0_08_r COMMAND imgcmp batch_p0_08_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_08-0.pgx 0 0)
add_test(NAME batch_comp_p0_08_g COMMAND imgcmp batch_p0_08_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_08-1.pgx 0 0)
add_test(NAME batch_comp_p0_08_b COMMAND imgcmp batch_p0_08_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_08-2.pgx 0 0)
set_tests_properties(batch_comp_p0_08_r batch_comp_p0_08_g batch_comp_p0_08_b PROPERTIES DEPENDS batch_dec_p0_08)
# 9
add_test(NAME batch_comp_p0_09 COMMAND imgcmp batch_p0_09_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_09-0.pgx 0 0)
set_tests_properties(batch_comp_p0_09 PROPERTIES DEPENDS batch_dec_p0_09)
# 10
add_test(NAME batch_comp_p0_10_r COMMAND imgcmp batch_p0_10_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_10-0.pgx 0 0)
add_test(NAME batch_comp_p0_10_g COMMAND imgcmp batch_p0_10_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_10-1.pgx 0 0)
add_test(NAME batch_comp_p0_10_b COMMAND imgcmp batch_p0_10_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_10-2.pgx 0 0)
set_tests_properties(batch_comp_p0_10_r batch_comp_p0_10_g batch_comp_p0_10_b PROPERTIES DEPENDS batch_dec_p0_10)
# 11
add_test(NAME batch_comp_p0_11 COMMAND imgcmp batch_p0_11_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_11-0.pgx 0 0)
set_tests_properties(batch_comp_p0_11 PROPERTIES DEPENDS batch_dec_p0_11)
# 12
add_test(NAME batch_comp_p0_12 COMMAND imgcmp batch_p0_12_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_12-0.pgx 0 0)
set_tests_properties(batch_comp_p0_12 PROPERTIES DEPENDS batch_dec_p0_12)
# 14
add_test(NAME batch_comp_p0_14_r COMMAND imgcmp batch_p0_14_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_14-0.pgx 0 0)
add_test(NAME batch_comp_p0_14_g COMMAND imgcmp batch_p0_14_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_14-1.pgx 0 0)
add_test(NAME batch_comp_p0_14_b COMMAND imgcmp batch_p0_14_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_14-2.pgx 0 0)
set_tests_properties(batch_comp_p0_14_r batch_comp_p0_14_g batch_comp_p0_14_b PROPERTIES DEPENDS batch_dec_p0_14)
# 15
add_test(NAME batch_comp_p0_15 COMMAND imgcmp batch_p0_15_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_15-0.pgx 0 0)
set_tests_properties(batch_comp_p0_15 PROPERTIES DEPENDS batch_dec_p0_15)
# 16
add_test(NAME batch_comp_p0_16 COMMAND imgcmp batch_p0_16_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_16-0.pgx 0 0)
set_tests_properties(batch_comp_p0_16 PROPERTIES DEPENDS batch_dec_p0_16)

# ── Part 1 Profile 1 ─────────────────────────────────────────────────────────
# Decoding
add_test(NAME batch_dec_p1_01 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/p1_01.j2k -o batch_p1_01.pgx -batch)
add_test(NAME batch_dec_p1_02 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/p1_02.j2k -o batch_p1_02.pgx -batch)
add_test(NAME batch_dec_p1_03 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/p1_03.j2k -o batch_p1_03.pgx -batch)
add_test(NAME batch_dec_p1_04 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/p1_04.j2k -o batch_p1_04.pgx -batch)
add_test(NAME batch_dec_p1_05 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/p1_05.j2k -o batch_p1_05.pgx -batch)
add_test(NAME batch_dec_p1_06 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/p1_06.j2k -o batch_p1_06.pgx -batch)
add_test(NAME batch_dec_p1_07 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/p1_07.j2k -o batch_p1_07.pgx -batch)

# PAE / MSE comparisons
# 1
add_test(NAME batch_comp_p1_01 COMMAND imgcmp batch_p1_01_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_01-0.pgx 0 0)
set_tests_properties(batch_comp_p1_01 PROPERTIES DEPENDS batch_dec_p1_01)
# 2
add_test(NAME batch_comp_p1_02_r COMMAND imgcmp batch_p1_02_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_02-0.pgx 5 0.765)
add_test(NAME batch_comp_p1_02_g COMMAND imgcmp batch_p1_02_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_02-1.pgx 4 0.616)
add_test(NAME batch_comp_p1_02_b COMMAND imgcmp batch_p1_02_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_02-2.pgx 6 1.051)
set_tests_properties(batch_comp_p1_02_r batch_comp_p1_02_g batch_comp_p1_02_b PROPERTIES DEPENDS batch_dec_p1_02)
# 3
add_test(NAME batch_comp_p1_03_a COMMAND imgcmp batch_p1_03_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_03-0.pgx 2 0.311)
add_test(NAME batch_comp_p1_03_b COMMAND imgcmp batch_p1_03_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_03-1.pgx 2 0.280)
add_test(NAME batch_comp_p1_03_c COMMAND imgcmp batch_p1_03_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_03-2.pgx 1 0.267)
add_test(NAME batch_comp_p1_03_d COMMAND imgcmp batch_p1_03_03.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_03-3.pgx 0 0)
set_tests_properties(batch_comp_p1_03_a batch_comp_p1_03_b batch_comp_p1_03_c batch_comp_p1_03_d PROPERTIES DEPENDS batch_dec_p1_03)
set_tests_properties(batch_comp_p1_03_b PROPERTIES WILL_FAIL false)
set_tests_properties(batch_comp_p1_03_c PROPERTIES WILL_FAIL false)
# 4
add_test(NAME batch_comp_p1_04 COMMAND imgcmp batch_p1_04_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_04-0.pgx 624 3080)
set_tests_properties(batch_comp_p1_04 PROPERTIES DEPENDS batch_dec_p1_04)
# 5
add_test(NAME batch_comp_p1_05_r COMMAND imgcmp batch_p1_05_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_05-0.pgx 40 8.458)
add_test(NAME batch_comp_p1_05_g COMMAND imgcmp batch_p1_05_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_05-1.pgx 40 9.716)
add_test(NAME batch_comp_p1_05_b COMMAND imgcmp batch_p1_05_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_05-2.pgx 40 10.154)
set_tests_properties(batch_comp_p1_05_r batch_comp_p1_05_g batch_comp_p1_05_b PROPERTIES DEPENDS batch_dec_p1_05)
# 6
add_test(NAME batch_comp_p1_06_r COMMAND imgcmp batch_p1_06_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_06-0.pgx 2 0.600)
add_test(NAME batch_comp_p1_06_g COMMAND imgcmp batch_p1_06_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_06-1.pgx 2 0.600)
add_test(NAME batch_comp_p1_06_b COMMAND imgcmp batch_p1_06_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_06-2.pgx 2 0.600)
set_tests_properties(batch_comp_p1_06_r batch_comp_p1_06_g batch_comp_p1_06_b PROPERTIES DEPENDS batch_dec_p1_06)
# 7
add_test(NAME batch_comp_p1_07_a COMMAND imgcmp batch_p1_07_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_07-0.pgx 0 0)
add_test(NAME batch_comp_p1_07_b COMMAND imgcmp batch_p1_07_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_07-1.pgx 0 0)
set_tests_properties(batch_comp_p1_07_a batch_comp_p1_07_b PROPERTIES DEPENDS batch_dec_p1_07)
