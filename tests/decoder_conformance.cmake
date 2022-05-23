# Conformance testing by Ctest
enable_testing()
set(CONFORMANCE_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/conformance_data")

## Conformance tests for HT
# PROFILE 0
include(${CMAKE_CURRENT_SOURCE_DIR}/tests/ht_profile0.cmake)
# PROFILE 1
include(${CMAKE_CURRENT_SOURCE_DIR}/tests/ht_profile1.cmake)
## Part 1 decoding tests
# PROFILE 0
include(${CMAKE_CURRENT_SOURCE_DIR}/tests/part1_profile0.cmake)
# PROFILE 1
include(${CMAKE_CURRENT_SOURCE_DIR}/tests/part1_profile1.cmake)

