cmake_policy(SET CMP0076 NEW)
target_sources(open_htj2k
    PRIVATE
        encoder.cpp
        decoder.cpp
        transcoder.cpp
        openhtj2k_message.cpp
)