# Conformance testing by Ctest
enable_testing()
set(ENCODER_REF_DIR "${CMAKE_CURRENT_SOURCE_DIR}/conformance_data/references")

add_test(NAME enc_lossless COMMAND open_htj2k_enc -i ${ENCODER_REF_DIR}/kodim23.ppm -o kodim23lossless.j2c Creversible=yes)
add_test(NAME dec_lossless COMMAND open_htj2k_dec -i kodim23lossless.j2c -o kodim23lossless.ppm)
set_tests_properties(dec_lossless PROPERTIES DEPENDS enc_lossless)
add_test(NAME comp_lossless COMMAND imgcmp kodim23lossless.ppm ${ENCODER_REF_DIR}/kodim23.ppm 0 0)
set_tests_properties(comp_lossless PROPERTIES DEPENDS dec_lossless)

add_test(NAME enc_lossy COMMAND open_htj2k_enc -i ${ENCODER_REF_DIR}/kodim23.ppm -o kodim23lossy.j2c Qfactor=90)
add_test(NAME dec_lossy COMMAND open_htj2k_dec -i kodim23lossy.j2c -o kodim23lossy.ppm)
set_tests_properties(dec_lossy PROPERTIES DEPENDS enc_lossy)
add_test(NAME comp_lossy COMMAND imgcmp kodim23lossy.ppm ${ENCODER_REF_DIR}/kodim23.ppm 23 6)
set_tests_properties(comp_lossy PROPERTIES DEPENDS dec_lossy)

add_test(NAME enc_lossless_odd COMMAND open_htj2k_enc -i ${ENCODER_REF_DIR}/kodim23odd.ppm -o kodim23odd_lossless.j2c Creversible=yes)
add_test(NAME dec_lossless_odd COMMAND open_htj2k_dec -i kodim23odd_lossless.j2c -o kodim23odd_lossless.ppm)
set_tests_properties(dec_lossless_odd PROPERTIES DEPENDS enc_lossless_odd)
add_test(NAME comp_lossless_odd COMMAND imgcmp kodim23odd_lossless.ppm ${ENCODER_REF_DIR}/kodim23odd.ppm 0 0)
set_tests_properties(comp_lossless_odd PROPERTIES DEPENDS dec_lossless_odd)

# Require the decoder-conformance cleanup fixture so these encoder tests are
# guaranteed to run AFTER cleanup_artifacts, never concurrently with it.
# Without this, ctest -j was free to schedule cleanup_artifacts (which globs
# *.ppm in the build dir and deletes everything matching) at any time, which
# under parallel load sometimes landed between dec_lossless writing
# kodim23lossless.ppm and comp_lossless reading it -- producing an
# intermittent "File kodim23lossless.ppm is not found" failure on
# comp_lossless / comp_lossless_odd that vanished on serial reruns.
#
# The `test_artifacts` fixture is declared by tests/decoder_conformance.cmake,
# which is included before this file from CMakeLists.txt, so the name is
# already visible here.
set_tests_properties(
  enc_lossless dec_lossless comp_lossless
  enc_lossy    dec_lossy    comp_lossy
  enc_lossless_odd dec_lossless_odd comp_lossless_odd
  PROPERTIES FIXTURES_REQUIRED test_artifacts)