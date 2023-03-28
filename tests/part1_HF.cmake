# Decoding
add_test(NAME dec_HF_p1_02 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/hifi_p1_02.j2k -o HF_p1_02.pgx)

# calculate PAE and MSE
# 1
add_test(NAME comp_HF_p1_02_0 COMMAND imgcmp HF_p1_02_00.pgx ${CONFORMANCE_DATA_DIR}/references/hifi-0.pgx 43 80)
add_test(NAME comp_HF_p1_02_1 COMMAND imgcmp HF_p1_02_01.pgx ${CONFORMANCE_DATA_DIR}/references/hifi-1.pgx 33 62)
add_test(NAME comp_HF_p1_02_2 COMMAND imgcmp HF_p1_02_02.pgx ${CONFORMANCE_DATA_DIR}/references/hifi-2.pgx 38 72)
set_tests_properties(comp_HF_p1_02_0 comp_HF_p1_02_1 comp_HF_p1_02_2 PROPERTIES DEPENDS dec_HF_p1_02)
