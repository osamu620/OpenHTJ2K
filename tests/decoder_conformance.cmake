# Conformance testing by Ctest
enable_testing()
set(CONFORMANCE_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/conformance_data")

# Remove stale test artifacts (*.pgx, *.ppm, *.pgm, *.raw) before tests run.  Attached
# via the CTest `test_artifacts` fixture so it executes once before any test
# that depends on it.  The fixture is auto-attached to every other test by the
# loop at the end of CMakeLists.txt — that loop runs AFTER every tests/*.cmake
# include site, so it covers tests defined by sibling files (jpip_phase1,
# row_range_validation, encoder_test) too.
add_test(NAME cleanup_artifacts
  COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/cleanup_artifacts.cmake)
set_tests_properties(cleanup_artifacts PROPERTIES FIXTURES_SETUP test_artifacts)

## Conformance tests for HT
# PROFILE 0
include(${CMAKE_CURRENT_SOURCE_DIR}/tests/ht_profile0.cmake)
# PROFILE 1
include(${CMAKE_CURRENT_SOURCE_DIR}/tests/ht_profile1.cmake)
# HiFi
include(${CMAKE_CURRENT_SOURCE_DIR}/tests/ht_HF.cmake)
## Part 1 decoding tests
# PROFILE 0
include(${CMAKE_CURRENT_SOURCE_DIR}/tests/part1_profile0.cmake)
# PROFILE 1
include(${CMAKE_CURRENT_SOURCE_DIR}/tests/part1_profile1.cmake)
# HiFi
include(${CMAKE_CURRENT_SOURCE_DIR}/tests/part1_HF.cmake)
## Part 2 decoding tests
# DFS + ATK
include(${CMAKE_CURRENT_SOURCE_DIR}/tests/part2_dfs_atk.cmake)

## Security regressions — each test exercises an input that used to
## crash and must now be rejected cleanly (non-zero exit, no signal).
include(${CMAKE_CURRENT_SOURCE_DIR}/tests/security_regressions.cmake)

## WASM conformance tests (require Node.js + WASM build output)
find_program(NODE_EXECUTABLE NAMES node nodejs)
if(NODE_EXECUTABLE)
  set(WASM_DEC_MJS "${CMAKE_CURRENT_SOURCE_DIR}/web/open_htj2k_dec.mjs")
  set(WASM_BUILD_OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/web/build/html/libopen_htj2k.wasm")
  if(EXISTS ${WASM_DEC_MJS} AND EXISTS ${WASM_BUILD_OUTPUT})
    message(STATUS "Node.js found: ${NODE_EXECUTABLE} -- enabling WASM conformance tests")
    # PROFILE 0
    include(${CMAKE_CURRENT_SOURCE_DIR}/tests/wasm_ht_profile0.cmake)
    # PROFILE 1
    include(${CMAKE_CURRENT_SOURCE_DIR}/tests/wasm_ht_profile1.cmake)
    # HiFi
    include(${CMAKE_CURRENT_SOURCE_DIR}/tests/wasm_ht_HF.cmake)
    # A multi-threaded decoder keeps using the thread pool after reset_decoder_with_bytes()
    # (counts the pool's worker tasks; 0.25.0 to 0.28.0 decoded later frames on one thread)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/web/build/html/libopen_htj2k_mt_simd.wasm")
      add_test(NAME wasm_mt_reinit_keeps_threads
               COMMAND ${NODE_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/wasm/mt_reinit_threads_check.mjs
                       ${CMAKE_CURRENT_SOURCE_DIR}/web/build/html ${CONFORMANCE_DATA_DIR}/ATK_DFS_IRV.j2c)
      set_tests_properties(wasm_mt_reinit_keeps_threads PROPERTIES
                           PASS_REGULAR_EXPRESSION "all cases passed" TIMEOUT 120)
      # The same output on 4 threads as on 1 for codestreams with a colour transform, whose
      # inverse then runs as thread-pool tasks (the conformance tests above use one thread)
      add_test(NAME wasm_mt_colour_matches_one_thread
               COMMAND ${NODE_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/wasm/mt_colour_check.mjs
                       ${CMAKE_CURRENT_SOURCE_DIR}/web/build/html
                       ${CONFORMANCE_DATA_DIR}/4KuF_H0L1_arri_1bpp.j2c ${CONFORMANCE_DATA_DIR}/ds0_ht_04_b11.j2k
                       ${CONFORMANCE_DATA_DIR}/ds0_ht_10_b11.j2k ${CONFORMANCE_DATA_DIR}/p0_13.j2k
                       ${CONFORMANCE_DATA_DIR}/ds1_ht_06_b11.j2k ${CONFORMANCE_DATA_DIR}/u04_small.j2c
                       ${CONFORMANCE_DATA_DIR}/hifi_ht1_02.j2k ${CONFORMANCE_DATA_DIR}/p1_05.j2k
                       ${CONFORMANCE_DATA_DIR}/u10_part15_lossless.j2c)
      set_tests_properties(wasm_mt_colour_matches_one_thread PROPERTIES
                           PASS_REGULAR_EXPRESSION "all cases passed" TIMEOUT 300)
    endif()
  else()
    message(STATUS "WASM build output not found -- skipping WASM tests")
  endif()
else()
  message(STATUS "Node.js not found -- skipping WASM conformance tests")
endif()