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