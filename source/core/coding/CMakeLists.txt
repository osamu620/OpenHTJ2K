cmake_policy(SET CMP0076 NEW)
target_sources(open_htj2k
    PRIVATE
    coding_units.cpp
    ht_block_decoding.cpp
    ht_block_decoding_neon.cpp
    ht_block_decoding_avx2.cpp
    ht_block_decoding_wasm.cpp
    ht_block_encoding.cpp
    ht_block_encoding_neon.cpp
    ht_block_encoding_avx2.cpp
    ht_block_encoding_wasm.cpp
    block_decoding.cpp
    mq_decoder.cpp
    subband_row_buf.cpp
)

# clang-18 miscompiles NEON bitstream helper functions when they are inlined.
# Disabling inlining for the affected files fixes the issue without changing
# the source; -fno-inline still respects always_inline attributes.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "19")
  set_source_files_properties(
    "${CMAKE_CURRENT_SOURCE_DIR}/coding_units.cpp"
    DIRECTORY "${CMAKE_SOURCE_DIR}"
    PROPERTIES COMPILE_OPTIONS "-fno-inline"
  )
endif()
