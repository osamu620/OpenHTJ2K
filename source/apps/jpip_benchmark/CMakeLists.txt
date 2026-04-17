target_sources(open_htj2k_jpip_benchmark PRIVATE main_jpip_benchmark.cpp)
target_include_directories(open_htj2k_jpip_benchmark PRIVATE
    ${CMAKE_SOURCE_DIR}/source/core/interface
    ${CMAKE_SOURCE_DIR}/source/core/jpip)
