cmake_policy(SET CMP0076 NEW)
target_sources(jpip_precinct_check
    PRIVATE
    main.cpp
)
target_include_directories(jpip_precinct_check
    PRIVATE
    ${CMAKE_SOURCE_DIR}/source/core/jpip
    ${CMAKE_SOURCE_DIR}/source/core/interface
)
