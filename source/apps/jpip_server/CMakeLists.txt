cmake_policy(SET CMP0076 NEW)
target_sources(open_htj2k_jpip_server
    PRIVATE
    main.cpp
)
target_include_directories(open_htj2k_jpip_server
    PRIVATE
    ${CMAKE_SOURCE_DIR}/source/core/jpip
    ${CMAKE_SOURCE_DIR}/source/core/interface
)
