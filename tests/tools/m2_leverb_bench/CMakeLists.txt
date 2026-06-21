cmake_policy(SET CMP0076 NEW)
target_sources(m2_leverb_bench
    PRIVATE
    main_m2leverb.cpp
)
target_include_directories(m2_leverb_bench PRIVATE ${CMAKE_SOURCE_DIR}/source/core/jpip)
