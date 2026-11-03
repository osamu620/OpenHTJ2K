cmake_policy(SET CMP0076 NEW)
target_sources(jph_box_check
    PRIVATE
    main.cpp
)
# jph.hpp and the internal headers it includes.  The tool calls only the public API and
# jp2c_box_header(), which is inline, so it links against the shared library on every
# platform.
target_include_directories(jph_box_check PRIVATE
    ${CMAKE_SOURCE_DIR}/source/core/interface
    ${CMAKE_SOURCE_DIR}/source/core/jph
    ${CMAKE_SOURCE_DIR}/source/core/codestream
    ${CMAKE_SOURCE_DIR}/source/core/common
)
