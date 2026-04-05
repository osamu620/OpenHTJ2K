cmake_minimum_required(VERSION 3.13)

project (
    open_htj2k
    LANGUAGES CXX
)

message(STATUS "**** WASM BUILD")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/html)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/html)

set(WASM_FLAGS "\
    -O3 -flto -mnontrapping-fptoint -mbulk-memory -s WASM=1 -s EXPORT_ES6=1 -s MODULARIZE=1 -s ENVIRONMENT=web,node \
    -s EXPORTED_FUNCTIONS=[_free,_malloc] \
    -s MALLOC=emmalloc \
    -s FILESYSTEM=0 \
    -s EXPORTED_RUNTIME_METHODS=[ccall,cwrap,writeArrayToMemory,getValue,HEAP32,HEAPU8] \
    -s NO_EXIT_RUNTIME=1 \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s MAXIMUM_MEMORY=1GB \
    -sNO_DISABLE_EXCEPTION_CATCHING \
    "
)
# Pull in the open_htj2k library (compiled without SIMD flags → scalar build).
add_subdirectory(".." open_htj2k_dir EXCLUDE_FROM_ALL)

# Collect the library's sources, include dirs, and any extra compile flags so
# that we can recompile them a second time with WASM-SIMD enabled.
get_target_property(_SOURCES   open_htj2k SOURCES)
get_target_property(_INC_DIRS  open_htj2k INCLUDE_DIRECTORIES)
get_target_property(_COMP_DEFS open_htj2k COMPILE_DEFINITIONS)
get_target_property(_COMP_OPTS open_htj2k COMPILE_OPTIONS)
get_target_property(_IFACE_INC open_htj2k INTERFACE_INCLUDE_DIRECTORIES)

# ── Scalar (non-SIMD) wrapper ────────────────────────────────────────────────
# open_htj2k is already compiled without any SIMD flags; link it directly.
add_executable(libopen_htj2k "src/wrapper.cpp")
set_target_properties(libopen_htj2k PROPERTIES COMPILE_FLAGS "-O3 -flto -mnontrapping-fptoint -mbulk-memory")
set_target_properties(libopen_htj2k PROPERTIES SUFFIX ".js" LINK_FLAGS ${WASM_FLAGS})
target_link_libraries(libopen_htj2k PRIVATE open_htj2k)

# ── WASM-SIMD variant ────────────────────────────────────────────────────────
# Recompile all library sources with -msimd128 + OPENHTJ2K_ENABLE_WASM_SIMD.
# Using an OBJECT library keeps the two compilations independent.
add_library(open_htj2k_simd_objs OBJECT ${_SOURCES})
target_include_directories(open_htj2k_simd_objs PRIVATE ${_INC_DIRS})
if(_COMP_DEFS)
  target_compile_definitions(open_htj2k_simd_objs PRIVATE ${_COMP_DEFS})
endif()
if(_COMP_OPTS)
  target_compile_options(open_htj2k_simd_objs PRIVATE ${_COMP_OPTS})
endif()
target_compile_definitions(open_htj2k_simd_objs PRIVATE "OPENHTJ2K_ENABLE_WASM_SIMD")
target_compile_options(open_htj2k_simd_objs PRIVATE "-O3" "-flto" "-msimd128" "-mnontrapping-fptoint" "-mbulk-memory")

add_library(open_htj2k_simd_lib STATIC $<TARGET_OBJECTS:open_htj2k_simd_objs>)
if(_IFACE_INC)
  target_include_directories(open_htj2k_simd_lib INTERFACE ${_IFACE_INC})
endif()

add_executable(libopen_htj2k_simd "src/wrapper.cpp")
set_target_properties(libopen_htj2k_simd PROPERTIES COMPILE_FLAGS "-O3 -flto -msimd128 -mnontrapping-fptoint -mbulk-memory")
set_target_properties(libopen_htj2k_simd PROPERTIES SUFFIX ".js" LINK_FLAGS ${WASM_FLAGS})
target_link_libraries(libopen_htj2k_simd PRIVATE open_htj2k_simd_lib)
