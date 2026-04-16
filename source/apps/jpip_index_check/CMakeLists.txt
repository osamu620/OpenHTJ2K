cmake_policy(SET CMP0076 NEW)
target_sources(jpip_index_check
    PRIVATE
    main.cpp
)
target_include_directories(jpip_index_check PRIVATE ${CMAKE_SOURCE_DIR}/source/core/jpip)
