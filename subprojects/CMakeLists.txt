cmake_minimum_required(VERSION 3.13)

project (
    open_htj2k
    LANGUAGES CXX
)

# rtp_reassembly sources use std::optional → require C++17 for the wrapper +
# RTP object libraries (the core open_htj2k library still auto-selects per its
# own CMakeLists).
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message(STATUS "**** WASM BUILD")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/html)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/html)

set(WASM_FLAGS_COMMON "\
    -O3 -flto -mnontrapping-fptoint -mbulk-memory -s WASM=1 -s EXPORT_ES6=1 -s MODULARIZE=1 -s ENVIRONMENT=web,node \
    -s EXPORTED_FUNCTIONS=[_free,_malloc] \
    -s MALLOC=dlmalloc \
    -s FILESYSTEM=0 \
    -s EXPORTED_RUNTIME_METHODS=[ccall,cwrap,writeArrayToMemory,getValue,HEAP32,HEAPU8,UTF8ToString] \
    -s NO_EXIT_RUNTIME=1 \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s MAXIMUM_MEMORY=2GB \
    -sNO_DISABLE_EXCEPTION_CATCHING \
    "
)

# Single-threaded link flags (original)
set(WASM_FLAGS "${WASM_FLAGS_COMMON}")

# Multi-threaded link flags — adds pthreads and SharedArrayBuffer support.
# PTHREAD_POOL_SIZE is set at link time; the JS runtime pre-creates this many
# Web Workers so that std::thread / ThreadPool do not block on worker creation.
set(WASM_MT_FLAGS "\
    ${WASM_FLAGS_COMMON} \
    -pthread \
    -s USE_PTHREADS=1 \
    -s PTHREAD_POOL_SIZE=navigator.hardwareConcurrency||4 \
    "
)
# Pull in the open_htj2k library (compiled without SIMD or pthread flags → scalar single-threaded build).
add_subdirectory(".." open_htj2k_dir EXCLUDE_FROM_ALL)

# Collect the library's sources, include dirs, and any extra compile flags so
# that we can recompile them a second time with WASM-SIMD enabled.
get_target_property(_SOURCES   open_htj2k SOURCES)
get_target_property(_INC_DIRS  open_htj2k INCLUDE_DIRECTORIES)
get_target_property(_COMP_DEFS open_htj2k COMPILE_DEFINITIONS)
get_target_property(_COMP_OPTS open_htj2k COMPILE_OPTIONS)
get_target_property(_IFACE_INC open_htj2k INTERFACE_INCLUDE_DIRECTORIES)

# ── RTP reassembly OBJECT libraries ──────────────────────────────────────────
# Pure-C++ frame_handler + RFC 9828 packet parser, ported as-is from the
# native open_htj2k_rtp_recv app.  No OS dependencies (no sockets, no threads).
#
# We build ONE OBJECT library per wrapper-flag set so the flags match the
# wrapper.cpp compile flags exactly — otherwise std::vector / std::deque
# layouts compiled with different flag sets can mismatch (seen as a crash
# inside pop_front() destructor), because libc++'s hardening macros differ
# between -msimd128 / -pthread builds.
set(_RTP_RECV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../source/apps/rtp_recv")
set(_RTP_SRCS
    ${_RTP_RECV_DIR}/rfc9828_parser.cpp
    ${_RTP_RECV_DIR}/frame_handler.cpp)

# Scalar (no SIMD, no pthreads)
add_library(rtp_reassembly_objs OBJECT ${_RTP_SRCS})
target_include_directories(rtp_reassembly_objs PUBLIC ${_RTP_RECV_DIR})
target_compile_options(rtp_reassembly_objs PRIVATE
    "-O3" "-flto" "-mnontrapping-fptoint" "-mbulk-memory")

# SIMD
add_library(rtp_reassembly_simd_objs OBJECT ${_RTP_SRCS})
target_include_directories(rtp_reassembly_simd_objs PUBLIC ${_RTP_RECV_DIR})
target_compile_options(rtp_reassembly_simd_objs PRIVATE
    "-O3" "-flto" "-msimd128" "-mnontrapping-fptoint" "-mbulk-memory")

# Pthreads (no SIMD)
add_library(rtp_reassembly_mt_objs OBJECT ${_RTP_SRCS})
target_include_directories(rtp_reassembly_mt_objs PUBLIC ${_RTP_RECV_DIR})
target_compile_options(rtp_reassembly_mt_objs PRIVATE
    "-O3" "-flto" "-mnontrapping-fptoint" "-mbulk-memory" "-pthread")

# Pthreads + SIMD
add_library(rtp_reassembly_mt_simd_objs OBJECT ${_RTP_SRCS})
target_include_directories(rtp_reassembly_mt_simd_objs PUBLIC ${_RTP_RECV_DIR})
target_compile_options(rtp_reassembly_mt_simd_objs PRIVATE
    "-O3" "-flto" "-msimd128" "-mnontrapping-fptoint" "-mbulk-memory" "-pthread")

# ── Scalar (non-SIMD) wrapper ────────────────────────────────────────────────
# open_htj2k is already compiled without any SIMD flags; link it directly.
add_executable(libopen_htj2k "src/wrapper.cpp" $<TARGET_OBJECTS:rtp_reassembly_objs>)
target_include_directories(libopen_htj2k PRIVATE ${_RTP_RECV_DIR})
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

add_executable(libopen_htj2k_simd "src/wrapper.cpp" $<TARGET_OBJECTS:rtp_reassembly_simd_objs>)
target_include_directories(libopen_htj2k_simd PRIVATE ${_RTP_RECV_DIR})
target_compile_definitions(libopen_htj2k_simd PRIVATE "OPENHTJ2K_ENABLE_WASM_SIMD")
set_target_properties(libopen_htj2k_simd PROPERTIES COMPILE_FLAGS "-O3 -flto -msimd128 -mnontrapping-fptoint -mbulk-memory")
set_target_properties(libopen_htj2k_simd PROPERTIES SUFFIX ".js" LINK_FLAGS ${WASM_FLAGS})
target_link_libraries(libopen_htj2k_simd PRIVATE open_htj2k_simd_lib)

# ── Multi-threaded variants (pthreads) ──────────────────────────────────────
# Recompile all library sources with OPENHTJ2K_THREAD + -pthread so that
# ThreadPool and std::thread are available via Emscripten's pthreads layer.
# Produces two additional targets:
#   libopen_htj2k_mt      — scalar + pthreads
#   libopen_htj2k_mt_simd — WASM-SIMD + pthreads

# ── Scalar + pthreads ───────────────────────────────────────────────────────
add_library(open_htj2k_mt_objs OBJECT ${_SOURCES})
target_include_directories(open_htj2k_mt_objs PRIVATE ${_INC_DIRS})
if(_COMP_DEFS)
  target_compile_definitions(open_htj2k_mt_objs PRIVATE ${_COMP_DEFS})
endif()
if(_COMP_OPTS)
  target_compile_options(open_htj2k_mt_objs PRIVATE ${_COMP_OPTS})
endif()
target_compile_definitions(open_htj2k_mt_objs PRIVATE "OPENHTJ2K_THREAD" "OPENHTJ2K_ENABLE_WASM_THREADS")
target_compile_options(open_htj2k_mt_objs PRIVATE "-O3" "-flto" "-mnontrapping-fptoint" "-mbulk-memory" "-pthread")

add_library(open_htj2k_mt_lib STATIC $<TARGET_OBJECTS:open_htj2k_mt_objs>)
if(_IFACE_INC)
  target_include_directories(open_htj2k_mt_lib INTERFACE ${_IFACE_INC})
endif()

add_executable(libopen_htj2k_mt "src/wrapper.cpp" $<TARGET_OBJECTS:rtp_reassembly_mt_objs>)
target_include_directories(libopen_htj2k_mt PRIVATE ${_RTP_RECV_DIR})
target_compile_definitions(libopen_htj2k_mt PRIVATE "OPENHTJ2K_THREAD")
set_target_properties(libopen_htj2k_mt PROPERTIES COMPILE_FLAGS "-O3 -flto -mnontrapping-fptoint -mbulk-memory -pthread")
set_target_properties(libopen_htj2k_mt PROPERTIES SUFFIX ".js" LINK_FLAGS ${WASM_MT_FLAGS})
target_link_libraries(libopen_htj2k_mt PRIVATE open_htj2k_mt_lib)

# ── WASM-SIMD + pthreads ────────────────────────────────────────────────────
add_library(open_htj2k_mt_simd_objs OBJECT ${_SOURCES})
target_include_directories(open_htj2k_mt_simd_objs PRIVATE ${_INC_DIRS})
if(_COMP_DEFS)
  target_compile_definitions(open_htj2k_mt_simd_objs PRIVATE ${_COMP_DEFS})
endif()
if(_COMP_OPTS)
  target_compile_options(open_htj2k_mt_simd_objs PRIVATE ${_COMP_OPTS})
endif()
target_compile_definitions(open_htj2k_mt_simd_objs PRIVATE "OPENHTJ2K_ENABLE_WASM_SIMD" "OPENHTJ2K_THREAD" "OPENHTJ2K_ENABLE_WASM_THREADS")
target_compile_options(open_htj2k_mt_simd_objs PRIVATE "-O3" "-flto" "-msimd128" "-mnontrapping-fptoint" "-mbulk-memory" "-pthread")

add_library(open_htj2k_mt_simd_lib STATIC $<TARGET_OBJECTS:open_htj2k_mt_simd_objs>)
if(_IFACE_INC)
  target_include_directories(open_htj2k_mt_simd_lib INTERFACE ${_IFACE_INC})
endif()

add_executable(libopen_htj2k_mt_simd "src/wrapper.cpp" $<TARGET_OBJECTS:rtp_reassembly_mt_simd_objs>)
target_include_directories(libopen_htj2k_mt_simd PRIVATE ${_RTP_RECV_DIR})
target_compile_definitions(libopen_htj2k_mt_simd PRIVATE "OPENHTJ2K_ENABLE_WASM_SIMD" "OPENHTJ2K_THREAD")
set_target_properties(libopen_htj2k_mt_simd PROPERTIES COMPILE_FLAGS "-O3 -flto -msimd128 -mnontrapping-fptoint -mbulk-memory -pthread")
set_target_properties(libopen_htj2k_mt_simd PROPERTIES SUFFIX ".js" LINK_FLAGS ${WASM_MT_FLAGS})
target_link_libraries(libopen_htj2k_mt_simd PRIVATE open_htj2k_mt_simd_lib)

# ── JPIP browser demo ────────────────────────────────────────────────────────
# Exposes the JPIP client-side pipeline (parse → reassemble → decode) via
# a thin C API that the JS side calls per frame.
set(JPIP_EXPORTED_FUNCTIONS "[_malloc,_free,_jpip_create_context,_jpip_get_canvas_width,_jpip_get_canvas_height,_jpip_get_num_components,_jpip_get_total_precincts,_jpip_set_reduce,_jpip_get_cache_model,_jpip_begin_frame,_jpip_reset_cache,_jpip_get_response_buffer,_jpip_add_response,_jpip_feed_stream_begin,_jpip_feed_stream,_jpip_feed_stream_end,_jpip_track_precincts_in_cache,_jpip_set_precinct_cache_budget,_jpip_get_precinct_cache_count,_jpip_end_frame,_jpip_end_frame_region,_jpip_destroy_context]")
set(JPIP_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/../source/core/jpip
    ${CMAKE_CURRENT_SOURCE_DIR}/../source/core/interface
    ${CMAKE_CURRENT_SOURCE_DIR}/../source/core/common)

# Single-threaded SIMD variant
set(JPIP_WASM_FLAGS "${WASM_FLAGS_COMMON} -s EXPORTED_FUNCTIONS=${JPIP_EXPORTED_FUNCTIONS}")
add_executable(libopen_htj2k_jpip "src/jpip_wrapper.cpp")
target_include_directories(libopen_htj2k_jpip PRIVATE ${JPIP_INCLUDE_DIRS})
set_target_properties(libopen_htj2k_jpip PROPERTIES
    COMPILE_FLAGS "-O3 -flto -msimd128 -mnontrapping-fptoint -mbulk-memory"
    SUFFIX ".js"
    LINK_FLAGS "${JPIP_WASM_FLAGS}")
target_link_libraries(libopen_htj2k_jpip PRIVATE open_htj2k_simd_lib)

# Multi-threaded SIMD variant (pthreads + WASM-SIMD)
set(JPIP_MT_WASM_FLAGS "${WASM_MT_FLAGS} -s EXPORTED_FUNCTIONS=${JPIP_EXPORTED_FUNCTIONS}")
add_executable(libopen_htj2k_jpip_mt_simd "src/jpip_wrapper.cpp")
target_include_directories(libopen_htj2k_jpip_mt_simd PRIVATE ${JPIP_INCLUDE_DIRS})
target_compile_definitions(libopen_htj2k_jpip_mt_simd PRIVATE "OPENHTJ2K_ENABLE_WASM_SIMD" "OPENHTJ2K_THREAD")
set_target_properties(libopen_htj2k_jpip_mt_simd PROPERTIES
    COMPILE_FLAGS "-O3 -flto -msimd128 -mnontrapping-fptoint -mbulk-memory -pthread"
    SUFFIX ".js"
    LINK_FLAGS "${JPIP_MT_WASM_FLAGS}")
target_link_libraries(libopen_htj2k_jpip_mt_simd PRIVATE open_htj2k_mt_simd_lib)
