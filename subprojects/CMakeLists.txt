cmake_minimum_required(VERSION 3.13)

project (
    open_htj2k
    LANGUAGES CXX
)

message(STATUS "**** WASM BUILD")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/html)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/html)

set(WASM_FLAGS "\
    -03 -s WASM=1 -s EXPORT_ES6=1 -s MODULARIZE=1 -s ENVIRONMENT=web \
    -s EXPORTED_FUNCTIONS=[_free,_malloc] \
    -s EXPORTED_RUNTIME_METHODS=[ccall,cwrap,writeArrayToMemory,getValue,HEAP32] \
    -s NO_EXIT_RUNTIME=1 \
    -s ALLOW_MEMORY_GROWTH=1 \
    -sASSERTIONS \
    -sNO_DISABLE_EXCEPTION_CATCHING \
    "
)
add_subdirectory(".." open_htj2k EXCLUDE_FROM_ALL)
add_executable(libopen_htj2k "src/wrapper.cpp")
set_target_properties(libopen_htj2k PROPERTIES SUFFIX ".js" LINK_FLAGS ${WASM_FLAGS})
target_link_libraries(libopen_htj2k PRIVATE open_htj2k)

add_executable(libopen_htj2k_simd "src/wrapper.cpp")
target_compile_definitions(libopen_htj2k_simd PUBLIC "OPENHTJ2K_ENABLE_ARM_NEON")
set_target_properties(libopen_htj2k_simd PROPERTIES COMPILE_FLAGS "-msimd128 -mfpu=neon")
set_target_properties(libopen_htj2k_simd PROPERTIES SUFFIX ".js" LINK_FLAGS ${WASM_FLAGS})
target_link_libraries(libopen_htj2k_simd PRIVATE open_htj2k)
