# Line-based versions of the HT conformance decoder tests.
# Each test decodes the same conformance codestream with -lb and compares
# the output to the same reference PGX files as the non-lb tests.
enable_testing()
set(CONFORMANCE_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/conformance_data")

# PROFILE 0
include(${CMAKE_CURRENT_SOURCE_DIR}/tests/lb_ht_profile0.cmake)
# PROFILE 1
include(${CMAKE_CURRENT_SOURCE_DIR}/tests/lb_ht_profile1.cmake)
# HiFi
include(${CMAKE_CURRENT_SOURCE_DIR}/tests/lb_ht_HF.cmake)
