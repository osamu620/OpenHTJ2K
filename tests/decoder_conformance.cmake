# Conformance testing by Ctest
enable_testing()
set(CONFORMANCE_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/conformance_data")

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

## WASM conformance tests (require Node.js + WASM build)
find_program(NODE_EXECUTABLE NAMES node nodejs)
if(NODE_EXECUTABLE)
  set(WASM_DEC_MJS "${CMAKE_CURRENT_SOURCE_DIR}/subprojects/open_htj2k_dec.mjs")
  if(EXISTS ${WASM_DEC_MJS})
    message(STATUS "Node.js found: ${NODE_EXECUTABLE} -- enabling WASM conformance tests")
    # PROFILE 0
    include(${CMAKE_CURRENT_SOURCE_DIR}/tests/wasm_ht_profile0.cmake)
    # PROFILE 1
    include(${CMAKE_CURRENT_SOURCE_DIR}/tests/wasm_ht_profile1.cmake)
    # HiFi
    include(${CMAKE_CURRENT_SOURCE_DIR}/tests/wasm_ht_HF.cmake)
  else()
    message(STATUS "WASM decoder mjs not found -- skipping WASM tests")
  endif()
else()
  message(STATUS "Node.js not found -- skipping WASM conformance tests")
endif()