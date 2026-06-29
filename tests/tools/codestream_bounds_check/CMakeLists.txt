cmake_policy(SET CMP0076 NEW)
target_sources(codestream_bounds_check
    PRIVATE
    main.cpp
)
target_include_directories(codestream_bounds_check PRIVATE
    ${CMAKE_SOURCE_DIR}/source/core/interface
)
