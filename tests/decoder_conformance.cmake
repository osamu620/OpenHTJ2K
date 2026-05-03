# Conformance testing by Ctest
enable_testing()
set(CONFORMANCE_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/conformance_data")

# Remove stale test artifacts (*.pgx, *.ppm, *.pgm) before tests run.  Attached
# via the CTest `test_artifacts` fixture so it executes once before any test
# that depends on it.  The fixture is auto-attached to every other test by the
# loop at the end of CMakeLists.txt — that loop runs AFTER every tests/*.cmake
# include site, so it covers tests defined by sibling files (batch_validation,
# jpip_phase1, row_range_validation, encoder_test) too.
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
  else()
    message(STATUS "WASM build output not found -- skipping WASM tests")
  endif()
else()
  message(STATUS "Node.js not found -- skipping WASM conformance tests")
endif()