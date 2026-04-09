# Conformance testing by Ctest
enable_testing()
set(CONFORMANCE_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/conformance_data")

# Remove stale test artifacts (*.pgx, *.ppm, *.pgm, *.j2c) before tests run.
# Attached via CTest fixtures so it executes once before the first decode test.
add_test(NAME cleanup_artifacts
  COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/cleanup_artifacts.cmake)
set_tests_properties(cleanup_artifacts PROPERTIES FIXTURES_SETUP test_artifacts)

# Collect all test names added by included files so we can attach the cleanup
# fixture to them after all includes.  We snapshot the test list before and
# after to compute the delta.
get_property(_tests_before DIRECTORY PROPERTY TESTS)

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

## WASM conformance tests (require Node.js + WASM build output)
find_program(NODE_EXECUTABLE NAMES node nodejs)
if(NODE_EXECUTABLE)
  set(WASM_DEC_MJS "${CMAKE_CURRENT_SOURCE_DIR}/subprojects/open_htj2k_dec.mjs")
  set(WASM_BUILD_OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/subprojects/build/html/libopen_htj2k.wasm")
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

# Attach the cleanup fixture to every test added above.
get_property(_tests_after DIRECTORY PROPERTY TESTS)
foreach(_t IN LISTS _tests_after)
  if(NOT _t IN_LIST _tests_before AND NOT _t STREQUAL "cleanup_artifacts")
    set_tests_properties(${_t} PROPERTIES FIXTURES_REQUIRED test_artifacts)
  endif()
endforeach()