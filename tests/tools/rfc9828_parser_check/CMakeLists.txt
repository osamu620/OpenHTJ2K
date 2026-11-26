cmake_policy(SET CMP0076 NEW)
# The parser is the RTP receiver's own source file (it uses C++17); the receiver itself is
# built only with -DOPENHTJ2K_RTP=ON, as it needs a window system.
target_sources(rfc9828_parser_check
    PRIVATE
    main.cpp
    ${CMAKE_SOURCE_DIR}/source/apps/rtp_recv/rfc9828_parser.cpp
)
target_include_directories(rfc9828_parser_check PRIVATE ${CMAKE_SOURCE_DIR}/source/apps/rtp_recv)
set_target_properties(rfc9828_parser_check PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
