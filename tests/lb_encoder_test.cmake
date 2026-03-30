# Tests for encode and decode with line-based (-line_based / -lb) flags.
# Each chain: encode with -line_based, decode (optionally with -lb), compare.
enable_testing()
set(CONFORMANCE_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/conformance_data/references")

# ── Lossless, single tile ─────────────────────────────────────────────────────
# Encode with -line_based, decode without -lb, compare to original.
add_test(NAME enc_lb_lossless
    COMMAND open_htj2k_enc -i ${CONFORMANCE_DATA_DIR}/kodim23.ppm
            -o kodim23_lb_lossless.j2c Creversible=yes -line_based)
add_test(NAME dec_lb_lossless
    COMMAND open_htj2k_dec -i kodim23_lb_lossless.j2c -o kodim23_lb_lossless.ppm)
set_tests_properties(dec_lb_lossless PROPERTIES DEPENDS enc_lb_lossless)
add_test(NAME comp_lb_lossless
    COMMAND imgcmp kodim23_lb_lossless.ppm ${CONFORMANCE_DATA_DIR}/kodim23.ppm 0 0)
set_tests_properties(comp_lb_lossless PROPERTIES DEPENDS dec_lb_lossless)

# Encode with -line_based, decode with -lb (invoke_line_based_stream), compare.
add_test(NAME dec_lb_lossless_stream
    COMMAND open_htj2k_dec -i kodim23_lb_lossless.j2c
            -o kodim23_lb_lossless_stream.ppm -lb)
set_tests_properties(dec_lb_lossless_stream PROPERTIES DEPENDS enc_lb_lossless)
add_test(NAME comp_lb_lossless_stream
    COMMAND imgcmp kodim23_lb_lossless_stream.ppm ${CONFORMANCE_DATA_DIR}/kodim23.ppm 0 0)
set_tests_properties(comp_lb_lossless_stream PROPERTIES DEPENDS dec_lb_lossless_stream)

# ── Lossy, single tile ────────────────────────────────────────────────────────
add_test(NAME enc_lb_lossy
    COMMAND open_htj2k_enc -i ${CONFORMANCE_DATA_DIR}/kodim23.ppm
            -o kodim23_lb_lossy.j2c Qfactor=90 -line_based)
add_test(NAME dec_lb_lossy
    COMMAND open_htj2k_dec -i kodim23_lb_lossy.j2c -o kodim23_lb_lossy.ppm)
set_tests_properties(dec_lb_lossy PROPERTIES DEPENDS enc_lb_lossy)
add_test(NAME comp_lb_lossy
    COMMAND imgcmp kodim23_lb_lossy.ppm ${CONFORMANCE_DATA_DIR}/kodim23.ppm 23 6)
set_tests_properties(comp_lb_lossy PROPERTIES DEPENDS dec_lb_lossy)

# Decode lossy output with -lb flag; compare against original with same tolerance.
add_test(NAME dec_lb_lossy_stream
    COMMAND open_htj2k_dec -i kodim23_lb_lossy.j2c
            -o kodim23_lb_lossy_stream.ppm -lb)
set_tests_properties(dec_lb_lossy_stream PROPERTIES DEPENDS enc_lb_lossy)
add_test(NAME comp_lb_lossy_stream
    COMMAND imgcmp kodim23_lb_lossy_stream.ppm ${CONFORMANCE_DATA_DIR}/kodim23.ppm 23 6)
set_tests_properties(comp_lb_lossy_stream PROPERTIES DEPENDS dec_lb_lossy_stream)

# ── Odd image dimensions, lossless ───────────────────────────────────────────
add_test(NAME enc_lb_lossless_odd
    COMMAND open_htj2k_enc -i ${CONFORMANCE_DATA_DIR}/kodim23odd.ppm
            -o kodim23odd_lb_lossless.j2c Creversible=yes -line_based)
add_test(NAME dec_lb_lossless_odd
    COMMAND open_htj2k_dec -i kodim23odd_lb_lossless.j2c
            -o kodim23odd_lb_lossless.ppm)
set_tests_properties(dec_lb_lossless_odd PROPERTIES DEPENDS enc_lb_lossless_odd)
add_test(NAME comp_lb_lossless_odd
    COMMAND imgcmp kodim23odd_lb_lossless.ppm ${CONFORMANCE_DATA_DIR}/kodim23odd.ppm 0 0)
set_tests_properties(comp_lb_lossless_odd PROPERTIES DEPENDS dec_lb_lossless_odd)

# Decode with -lb (streaming) and compare to original.
add_test(NAME dec_lb_lossless_odd_stream
    COMMAND open_htj2k_dec -i kodim23odd_lb_lossless.j2c
            -o kodim23odd_lb_lossless_stream.ppm -lb)
set_tests_properties(dec_lb_lossless_odd_stream PROPERTIES DEPENDS enc_lb_lossless_odd)
add_test(NAME comp_lb_lossless_odd_stream
    COMMAND imgcmp kodim23odd_lb_lossless_stream.ppm
            ${CONFORMANCE_DATA_DIR}/kodim23odd.ppm 0 0)
set_tests_properties(comp_lb_lossless_odd_stream PROPERTIES DEPENDS dec_lb_lossless_odd_stream)
