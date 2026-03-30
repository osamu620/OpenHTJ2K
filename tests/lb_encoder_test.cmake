# Line-based version of encoder_test.cmake.
# Each chain: encode with -line_based, decode with -lb, compare to original.
# Test names mirror encoder_test.cmake with lb_ prefix.
enable_testing()
set(CONFORMANCE_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/conformance_data/references")

add_test(NAME lb_enc_lossless
    COMMAND open_htj2k_enc -i ${CONFORMANCE_DATA_DIR}/kodim23.ppm
            -o lb_kodim23lossless.j2c Creversible=yes -line_based)
add_test(NAME lb_dec_lossless
    COMMAND open_htj2k_dec -i lb_kodim23lossless.j2c -o lb_kodim23lossless.ppm -lb)
set_tests_properties(lb_dec_lossless PROPERTIES DEPENDS lb_enc_lossless)
add_test(NAME lb_comp_lossless
    COMMAND imgcmp lb_kodim23lossless.ppm ${CONFORMANCE_DATA_DIR}/kodim23.ppm 0 0)
set_tests_properties(lb_comp_lossless PROPERTIES DEPENDS lb_dec_lossless)

add_test(NAME lb_enc_lossy
    COMMAND open_htj2k_enc -i ${CONFORMANCE_DATA_DIR}/kodim23.ppm
            -o lb_kodim23lossy.j2c Qfactor=90 -line_based)
add_test(NAME lb_dec_lossy
    COMMAND open_htj2k_dec -i lb_kodim23lossy.j2c -o lb_kodim23lossy.ppm -lb)
set_tests_properties(lb_dec_lossy PROPERTIES DEPENDS lb_enc_lossy)
add_test(NAME lb_comp_lossy
    COMMAND imgcmp lb_kodim23lossy.ppm ${CONFORMANCE_DATA_DIR}/kodim23.ppm 23 6)
set_tests_properties(lb_comp_lossy PROPERTIES DEPENDS lb_dec_lossy)

add_test(NAME lb_enc_lossless_odd
    COMMAND open_htj2k_enc -i ${CONFORMANCE_DATA_DIR}/kodim23odd.ppm
            -o lb_kodim23odd_lossless.j2c Creversible=yes -line_based)
add_test(NAME lb_dec_lossless_odd
    COMMAND open_htj2k_dec -i lb_kodim23odd_lossless.j2c
            -o lb_kodim23odd_lossless.ppm -lb)
set_tests_properties(lb_dec_lossless_odd PROPERTIES DEPENDS lb_enc_lossless_odd)
add_test(NAME lb_comp_lossless_odd
    COMMAND imgcmp lb_kodim23odd_lossless.ppm ${CONFORMANCE_DATA_DIR}/kodim23odd.ppm 0 0)
set_tests_properties(lb_comp_lossless_odd PROPERTIES DEPENDS lb_dec_lossless_odd)
