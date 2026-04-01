cmake_minimum_required(VERSION 3.13)

project (
    open_htj2k
    LANGUAGES CXX
)

message(STATUS "**** WASM BUILD")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/html)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/html)

set(WASM_FLAGS "\
    -O3 -flto -s WASM=1 -s EXPORT_ES6=1 -s MODULARIZE=1 -s ENVIRONMENT=web \
    -s EXPORTED_FUNCTIONS=[_free,_malloc] \
    -s MALLOC=emmalloc \
    -s FILESYSTEM=0 \
    -s EXPORTED_RUNTIME_METHODS=[ccall,cwrap,writeArrayToMemory,getValue,HEAP32] \
    -s NO_EXIT_RUNTIME=1 \
    -s ALLOW_MEMORY_GROWTH=1 \
    -sNO_DISABLE_EXCEPTION_CATCHING \
    "
)
add_subdirectory(".." open_htj2k EXCLUDE_FROM_ALL)

# Scalar (non-SIMD) build — open_htj2k library compiled without any SIMD flags.
add_executable(libopen_htj2k "src/wrapper.cpp")
set_target_properties(libopen_htj2k PROPERTIES COMPILE_FLAGS "-O3 -flto")
set_target_properties(libopen_htj2k PROPERTIES SUFFIX ".js" LINK_FLAGS ${WASM_FLAGS})
target_link_libraries(libopen_htj2k PRIVATE open_htj2k)

# WASM-SIMD build — enable WASM-SIMD on the library and wrapper.
# NOTE: Because open_htj2k is compiled once and shared by both targets, libopen_htj2k
# also links to the SIMD-compiled library.  For a true scalar fallback, run cmake
# from a separate build directory without -DENABLE_WASM_SIMD or configure the parent
# CMakeLists.txt to omit the WASM_SIMD define.
# We use PRIVATE so the flag is applied to the library's own TUs without propagating
# to the (already-built) scalar wrapper.
target_compile_definitions(open_htj2k PRIVATE "OPENHTJ2K_ENABLE_WASM_SIMD")
target_compile_options(open_htj2k PRIVATE "-msimd128")

add_executable(libopen_htj2k_simd "src/wrapper.cpp")
set_target_properties(libopen_htj2k_simd PROPERTIES COMPILE_FLAGS "-O3 -flto -msimd128")
set_target_properties(libopen_htj2k_simd PROPERTIES SUFFIX ".js" LINK_FLAGS ${WASM_FLAGS})
target_link_libraries(libopen_htj2k_simd PRIVATE open_htj2k)
