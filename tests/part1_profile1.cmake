# Decoding
add_test(NAME  dec_p1_01 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/p1_01.j2k -o p1_01.pgx)
add_test(NAME  dec_p1_02 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/p1_02.j2k -o p1_02.pgx)
add_test(NAME  dec_p1_03 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/p1_03.j2k -o p1_03.pgx)
add_test(NAME  dec_p1_04 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/p1_04.j2k -o p1_04.pgx)
add_test(NAME  dec_p1_05 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/p1_05.j2k -o p1_05.pgx)
add_test(NAME  dec_p1_06 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/p1_06.j2k -o p1_06.pgx)
add_test(NAME  dec_p1_07 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/p1_07.j2k -o p1_07.pgx)

# calculate PAE and MSE
# 1
add_test(NAME comp_p1_01 COMMAND imgcmp p1_01_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_01-0.pgx 0 0)
set_tests_properties(comp_p1_01 PROPERTIES DEPENDS dec_p1_01)
# 2
add_test(NAME comp_p1_02_r COMMAND imgcmp p1_02_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_02-0.pgx 5 0.765)
add_test(NAME comp_p1_02_g COMMAND imgcmp p1_02_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_02-1.pgx 4 0.616)
add_test(NAME comp_p1_02_b COMMAND imgcmp p1_02_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_02-2.pgx 6 1.051)
set_tests_properties(comp_p1_02_r comp_p1_02_g comp_p1_02_b PROPERTIES DEPENDS dec_p1_02)
# 3
add_test(NAME comp_p1_03_a COMMAND imgcmp p1_03_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_03-0.pgx 2 0.311) # 0.300 in old spec
add_test(NAME comp_p1_03_b COMMAND imgcmp p1_03_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_03-1.pgx 2 0.280) # 0.210 in old spec
add_test(NAME comp_p1_03_c COMMAND imgcmp p1_03_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_03-2.pgx 1 0.267) # 0.200 in old spec
add_test(NAME comp_p1_03_d COMMAND imgcmp p1_03_03.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_03-3.pgx 0 0)
set_tests_properties(comp_p1_03_a comp_p1_03_b comp_p1_03_c comp_p1_03_d PROPERTIES DEPENDS dec_p1_03)
set_tests_properties(comp_p1_03_b PROPERTIES WILL_FAIL false)
set_tests_properties(comp_p1_03_c PROPERTIES WILL_FAIL false)
# 4
add_test(NAME comp_p1_04 COMMAND imgcmp p1_04_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_04-0.pgx 624 3080)
set_tests_properties(comp_p1_04 PROPERTIES DEPENDS dec_p1_04)
# 5
add_test(NAME comp_p1_05_r COMMAND imgcmp p1_05_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_05-0.pgx 40 8.458)
add_test(NAME comp_p1_05_g COMMAND imgcmp p1_05_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_05-1.pgx 40 9.716)
add_test(NAME comp_p1_05_b COMMAND imgcmp p1_05_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_05-2.pgx 40 10.154)
set_tests_properties(comp_p1_05_r comp_p1_05_g comp_p1_05_b PROPERTIES DEPENDS dec_p1_05)
# 6
add_test(NAME comp_p1_06_r COMMAND imgcmp p1_06_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_06-0.pgx 2 0.600)
add_test(NAME comp_p1_06_g COMMAND imgcmp p1_06_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_06-1.pgx 2 0.600)
add_test(NAME comp_p1_06_b COMMAND imgcmp p1_06_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_06-2.pgx 2 0.600)
set_tests_properties(comp_p1_06_r comp_p1_06_g comp_p1_06_b PROPERTIES DEPENDS dec_p1_06)
# 7
add_test(NAME comp_p1_07_a COMMAND imgcmp p1_07_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_07-0.pgx 0 0)
add_test(NAME comp_p1_07_b COMMAND imgcmp p1_07_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_07-1.pgx 0 0)
set_tests_properties(comp_p1_07_a comp_p1_07_b PROPERTIES DEPENDS dec_p1_07)
