cmake_policy(SET CMP0076 NEW)
# The parser is the RTP receiver's own source file (it uses C++17); the receiver itself is
# built only with -DOPENHTJ2K_RTP=ON, as it needs a window system.  The path is relative to
# this directory: the WebAssembly build configures the project from web/, where
# CMAKE_SOURCE_DIR is not the repository root.
set(_rfc9828_dir ${CMAKE_CURRENT_LIST_DIR}/../../../source/apps/rtp_recv)
target_sources(rfc9828_parser_check
    PRIVATE
    main.cpp
    ${_rfc9828_dir}/rfc9828_parser.cpp
)
target_include_directories(rfc9828_parser_check PRIVATE ${_rfc9828_dir})
set_target_properties(rfc9828_parser_check PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
