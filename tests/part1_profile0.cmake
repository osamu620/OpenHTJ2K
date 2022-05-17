# Decoding
add_test(NAME  dec_p0_01 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/p0_01.j2k -o p0_01.pgx)
add_test(NAME  dec_p0_02 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/p0_02.j2k -o p0_02.pgx)
add_test(NAME  dec_p0_03 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/p0_03.j2k -o p0_03.pgx)
add_test(NAME  dec_p0_04 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/p0_04.j2k -o p0_04.pgx)
add_test(NAME  dec_p0_05 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/p0_05.j2k -o p0_05.pgx)
add_test(NAME  dec_p0_06 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/p0_06.j2k -o p0_06.pgx)
add_test(NAME  dec_p0_07 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/p0_07.j2k -o p0_07.pgx)
add_test(NAME  dec_p0_08 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/p0_08.j2k -o p0_08.pgx -reduce 1)
add_test(NAME  dec_p0_09 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/p0_09.j2k -o p0_09.pgx)
add_test(NAME  dec_p0_10 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/p0_10.j2k -o p0_10.pgx)
add_test(NAME  dec_p0_11 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/p0_11.j2k -o p0_11.pgx)
add_test(NAME  dec_p0_12 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/p0_12.j2k -o p0_12.pgx)
add_test(NAME  dec_p0_14 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/p0_14.j2k -o p0_14.pgx)
add_test(NAME  dec_p0_15 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/p0_15.j2k -o p0_15.pgx)
add_test(NAME  dec_p0_16 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/p0_16.j2k -o p0_16.pgx)

# calculate PAE and MSE
# 1
add_test(NAME comp_p0_01 COMMAND imgcmp p0_01_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_01-0.pgx 0 0)
set_tests_properties(comp_p0_01 PROPERTIES DEPENDS dec_p0_01)
# 2
add_test(NAME comp_p0_02 COMMAND imgcmp p0_02_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_02-0.pgx 0 0)
set_tests_properties(comp_p0_02 PROPERTIES DEPENDS dec_p0_02)
# 3
add_test(NAME comp_p0_03 COMMAND imgcmp p0_03_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_03-0.pgx 0 0)
set_tests_properties(comp_p0_03 PROPERTIES DEPENDS dec_p0_03)
# 4
add_test(NAME comp_p0_04_r COMMAND imgcmp p0_04_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_04-0.pgx 5 0.776)
add_test(NAME comp_p0_04_g COMMAND imgcmp p0_04_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_04-1.pgx 4 0.626)
add_test(NAME comp_p0_04_b COMMAND imgcmp p0_04_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_04-2.pgx 6 1.070)
set_tests_properties(comp_p0_04_r comp_p0_04_g comp_p0_04_b PROPERTIES DEPENDS dec_p0_04)
# 5
add_test(NAME comp_p0_05_a COMMAND imgcmp p0_05_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_05-0.pgx 2 0.319) # 0.302 in old spec
add_test(NAME comp_p0_05_b COMMAND imgcmp p0_05_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_05-1.pgx 2 0.323) # 0.307 in old spec
add_test(NAME comp_p0_05_c COMMAND imgcmp p0_05_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_05-2.pgx 2 0.317) # 0.269 in old spec
add_test(NAME comp_p0_05_d COMMAND imgcmp p0_05_03.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_05-3.pgx 0 0)
set_tests_properties(comp_p0_05_a comp_p0_05_b comp_p0_05_c comp_p0_05_d PROPERTIES DEPENDS dec_p0_05)
# 6
add_test(NAME comp_p0_06_a COMMAND imgcmp p0_06_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_06-0.pgx 635 11287)
add_test(NAME comp_p0_06_b COMMAND imgcmp p0_06_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_06-1.pgx 403 6124)
add_test(NAME comp_p0_06_c COMMAND imgcmp p0_06_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_06-2.pgx 378 3968)
add_test(NAME comp_p0_06_d COMMAND imgcmp p0_06_03.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_06-3.pgx 0 0)
set_tests_properties(comp_p0_06_a comp_p0_06_b comp_p0_06_c comp_p0_06_d PROPERTIES DEPENDS dec_p0_06)
# 7
add_test(NAME comp_p0_07_r COMMAND imgcmp p0_07_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_07-0.pgx 0 0)
add_test(NAME comp_p0_07_g COMMAND imgcmp p0_07_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_07-1.pgx 0 0)
add_test(NAME comp_p0_07_b COMMAND imgcmp p0_07_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_07-2.pgx 0 0)
set_tests_properties(comp_p0_07_r comp_p0_07_g comp_p0_07_b PROPERTIES DEPENDS dec_p0_07)
# 8
add_test(NAME comp_p0_08_r COMMAND imgcmp p0_08_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_08-0.pgx 0 0)
add_test(NAME comp_p0_08_g COMMAND imgcmp p0_08_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_08-1.pgx 0 0)
add_test(NAME comp_p0_08_b COMMAND imgcmp p0_08_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_08-2.pgx 0 0)
set_tests_properties(comp_p0_08_r comp_p0_08_g comp_p0_08_b PROPERTIES DEPENDS dec_p0_08)
# 9
add_test(NAME comp_p0_09 COMMAND imgcmp p0_09_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_09-0.pgx 0 0)
set_tests_properties(comp_p0_09 PROPERTIES DEPENDS dec_p0_09)
# 10
add_test(NAME comp_p0_10_r COMMAND imgcmp p0_10_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_10-0.pgx 0 0)
add_test(NAME comp_p0_10_g COMMAND imgcmp p0_10_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_10-1.pgx 0 0)
add_test(NAME comp_p0_10_b COMMAND imgcmp p0_10_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_10-2.pgx 0 0)
set_tests_properties(comp_p0_10_r comp_p0_10_g comp_p0_10_b PROPERTIES DEPENDS dec_p0_10)
# 11
add_test(NAME comp_p0_11 COMMAND imgcmp p0_11_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_11-0.pgx 0 0)
set_tests_properties(comp_p0_11 PROPERTIES DEPENDS dec_p0_11)
# 12
add_test(NAME comp_p0_12 COMMAND imgcmp p0_12_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_12-0.pgx 0 0)
set_tests_properties(comp_p0_12 PROPERTIES DEPENDS dec_p0_12)
# 14
add_test(NAME comp_p0_14_r COMMAND imgcmp p0_14_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_14-0.pgx 0 0)
add_test(NAME comp_p0_14_g COMMAND imgcmp p0_14_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_14-1.pgx 0 0)
add_test(NAME comp_p0_14_b COMMAND imgcmp p0_14_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_14-2.pgx 0 0)
set_tests_properties(comp_p0_14_r comp_p0_14_g comp_p0_14_b PROPERTIES DEPENDS dec_p0_14)
# 15
add_test(NAME comp_p0_15 COMMAND imgcmp p0_15_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_15-0.pgx 0 0)
set_tests_properties(comp_p0_15 PROPERTIES DEPENDS dec_p0_15)
# 16
add_test(NAME comp_p0_16 COMMAND imgcmp p0_16_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_16-0.pgx 0 0)
set_tests_properties(comp_p0_16 PROPERTIES DEPENDS dec_p0_16)
