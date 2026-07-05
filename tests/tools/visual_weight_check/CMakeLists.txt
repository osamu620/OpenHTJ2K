cmake_policy(SET CMP0076 NEW)
target_sources(visual_weight_check
    PRIVATE
    main.cpp
)
target_include_directories(visual_weight_check PRIVATE
    ${CMAKE_SOURCE_DIR}/source/core/codestream
)
