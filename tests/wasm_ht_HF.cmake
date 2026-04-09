# WASM Decoding — HT HiFi
add_test(NAME wasm_dec_HF_ht1_02 COMMAND ${NODE_EXECUTABLE} ${WASM_DEC_MJS} -i ${CONFORMANCE_DATA_DIR}/hifi_ht1_02.j2k -o wasm_HF_ht1_02.pgx)

# calculate PAE and MSE
# 1
add_test(NAME wasm_comp_HF_ht1_02_0 COMMAND imgcmp wasm_HF_ht1_02_00.pgx ${CONFORMANCE_DATA_DIR}/references/hifi-0.pgx 47 82)
add_test(NAME wasm_comp_HF_ht1_02_1 COMMAND imgcmp wasm_HF_ht1_02_01.pgx ${CONFORMANCE_DATA_DIR}/references/hifi-1.pgx 36 65)
add_test(NAME wasm_comp_HF_ht1_02_2 COMMAND imgcmp wasm_HF_ht1_02_02.pgx ${CONFORMANCE_DATA_DIR}/references/hifi-2.pgx 42 86)
set_tests_properties(wasm_comp_HF_ht1_02_0 wasm_comp_HF_ht1_02_1 wasm_comp_HF_ht1_02_2 PROPERTIES DEPENDS wasm_dec_HF_ht1_02)
