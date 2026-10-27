cmake_minimum_required(VERSION 3.13)

# MSVC: choose the C runtime through CMAKE_MSVC_RUNTIME_LIBRARY (default /MD,
# /MDd for Debug), as package managers and consuming projects expect.  The
# compiler flags set below replace CMAKE_CXX_FLAGS_<CONFIG>, where the runtime
# flag lived under the old policy, leaving the runtime to the build tool (the
# static CRT for Ninja and other command-line generators).
if(POLICY CMP0091)
  cmake_policy(SET CMP0091 NEW)
endif()

project(
  open_htj2k
  LANGUAGES CXX
  DESCRIPTION "Open source implementation of ISO/IEC 15444-1 and -15")
if (NOT EMSCRIPTEN)
  option(BUILD_SHARED_LIBS "Shared Libraries" ON)
else()
  option(BUILD_SHARED_LIBS "Shared Libraries" OFF)
endif()
enable_language(CXX)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

############################################################
# Parse version file
# credit: https://stackoverflow.com/a/47084079 and OpenJPH project https://github.com/aous72/OpenJPH

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/source/core/common/open_htj2k_version.hpp" VERFILE)
if (NOT VERFILE)
    message(FATAL_ERROR "Failed to parse open_htj2k_version.hpp!")
endif()

string(REGEX MATCH "OPENHTJ2K_VERSION_MAJOR ([0-9]*)" _ ${VERFILE})
set(OPENHTJ2K_VERSION_MAJOR ${CMAKE_MATCH_1})
string(REGEX MATCH "OPENHTJ2K_VERSION_MINOR ([0-9]*)" _ ${VERFILE})
set(OPENHTJ2K_VERSION_MINOR ${CMAKE_MATCH_1})
string(REGEX MATCH "OPENHTJ2K_VERSION_PATCH ([a-z0-9]*)" _ ${VERFILE})
set(OPENHTJ2K_VERSION_PATCH ${CMAKE_MATCH_1})

set(PROJECT_VERSION "${OPENHTJ2K_VERSION_MAJOR}.${OPENHTJ2K_VERSION_MINOR}.${OPENHTJ2K_VERSION_PATCH}")

# Standard install layout (lib/, lib64/, include/, bin/) on every platform; the
# install() rules and the pkg-config file below rely on these variables.
include(GNUInstallDirs)
# pkg-config paths are written relative to ${prefix} so the installed tree can
# be relocated (pkg-config --define-variable=prefix=...).  Absolute
# CMAKE_INSTALL_<dir> values are used as given.
foreach(_dir LIBDIR INCLUDEDIR)
  if(IS_ABSOLUTE "${CMAKE_INSTALL_${_dir}}")
    set(PKG_CONFIG_${_dir} "${CMAKE_INSTALL_${_dir}}")
  else()
    set(PKG_CONFIG_${_dir} "\${prefix}/${CMAKE_INSTALL_${_dir}}")
  endif()
endforeach()
# Only needed when linking the static library (pkg-config --static).
set(PKG_CONFIG_REQUIRES_PRIVATE "")
set(PKG_CONFIG_LIBS_PRIVATE "")
# Check C++ version supported by compiler
include(CheckCXXCompilerFlag)
if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
  check_cxx_compiler_flag("/std:c++11" COMPILER_SUPPORTS_CXX11)
  check_cxx_compiler_flag("/std:c++14" COMPILER_SUPPORTS_CXX14)
  check_cxx_compiler_flag("/std:c++17" COMPILER_SUPPORTS_CXX17)
else()
  check_cxx_compiler_flag("-std=c++11" COMPILER_SUPPORTS_CXX11)
  check_cxx_compiler_flag("-std=c++14" COMPILER_SUPPORTS_CXX14)
  check_cxx_compiler_flag("-std=c++17" COMPILER_SUPPORTS_CXX17)
endif()

# Compiler settings: the newest of C++17 / 14 / 11 the compiler supports,
# unless the standard is chosen with -DCMAKE_CXX_STANDARD=11|14|17|...
if(NOT DEFINED CMAKE_CXX_STANDARD)
  if(COMPILER_SUPPORTS_CXX17)
    set(CMAKE_CXX_STANDARD 17)
  elseif(COMPILER_SUPPORTS_CXX14)
    set(CMAKE_CXX_STANDARD 14)
  else()
    set(CMAKE_CXX_STANDARD 11)
  endif()
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")

message(STATUS "${CMAKE_SYSTEM_PROCESSOR} ${CMAKE_CXX_COMPILER_ID}")

# Default BUILD_TYPE is "Release"
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
  message(STATUS "CMAKE_BUILD_TYPE is unspecified. Set it as Release.")
endif()

add_library(open_htj2k)
add_subdirectory(source/core/codestream)
add_subdirectory(source/core/coding)
add_subdirectory(source/core/transform)
add_subdirectory(source/core/interface)
add_subdirectory(source/core/jph)
add_subdirectory(source/core/jpip)
target_include_directories(
    open_htj2k
    PRIVATE source/core/common
    source/core/codestream
    source/core/coding
    source/core/transform
    # source/core/interface
    source/core/jph
    source/core/jpip
    # ${CMAKE_CURRENT_SOURCE_DIR}/source/thirdparty/highway
)
# In-tree code also sees source/core/common; installed consumers get only the
# public headers (see "Install" below).
target_include_directories(
    open_htj2k
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/source/core/interface>
              $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/source/core/common>
              $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/open_htj2k/interface>)
set_target_properties(
        open_htj2k PROPERTIES
        OUTPUT_NAME openhtj2k
        EXPORT_NAME openhtj2k
        VERSION ${PROJECT_VERSION}
        SOVERSION ${OPENHTJ2K_VERSION_MAJOR})
# Same name as the installed CMake package's target, for add_subdirectory().
add_library(openhtj2k::openhtj2k ALIAS open_htj2k)
set(PKG_CONFIG_CFLAGS_EXTRA "")
if(NOT BUILD_SHARED_LIBS)
  # Tells the public headers not to use __declspec(dllimport) (MSVC).
  target_compile_definitions(open_htj2k PUBLIC OHTJ2K_STATIC)
  set(PKG_CONFIG_CFLAGS_EXTRA " -DOHTJ2K_STATIC")
endif()
# target_link_libraries(open_htj2k PUBLIC ${CMAKE_THREAD_LIBS_INIT} PRIVATE hwy)

find_package(Threads)
find_package(TIFF)

if (EMSCRIPTEN)
  # No automatic SIMD here — web/CMakeLists.txt controls SIMD enablement
  # for the two WASM variants (libopen_htj2k = scalar, libopen_htj2k_simd = WASM-SIMD).
  # Threading: enabled when OPENHTJ2K_ENABLE_WASM_THREADS is set by the
  # web/ build (pthreads variant).
  if (OPENHTJ2K_ENABLE_WASM_THREADS)
    target_compile_definitions(open_htj2k PUBLIC "OPENHTJ2K_THREAD")
    target_compile_options(open_htj2k PUBLIC "-pthread")
    message(STATUS "OPENHTJ2K_THREAD is set (Emscripten pthreads)")
  endif()
endif()
if (NOT EMSCRIPTEN)
  # Check Thread support
  if (Threads_FOUND)
    message(STATUS "Thread library found")
    target_link_libraries(open_htj2k PUBLIC Threads::Threads)
    target_compile_definitions(open_htj2k PUBLIC "OPENHTJ2K_THREAD")
    message(STATUS "OPENHTJ2K_THREAD is set")
    set(PKG_CONFIG_LIBS_PRIVATE "${CMAKE_THREAD_LIBS_INIT}")
  endif()
  # Check TIFF support
  if (TIFF_FOUND)
    message(STATUS "libtiff found")
    message(STATUS "tiff include = ${TIFF_INCLUDE_DIR}")
    message(STATUS "tiff library = ${TIFF_LIBRARIES}")
    target_compile_definitions(open_htj2k PUBLIC "OPENHTJ2K_TIFF_SUPPORT")
    # The public headers do not include tiffio.h, so libtiff is a private
    # dependency: it matters to consumers only when linking the static library.
    target_link_libraries(open_htj2k PRIVATE TIFF::TIFF)
    message(STATUS "OPENHTJ2K_TIFF_SUPPORT is set ")
    # libtiff's pkg-config module is "libtiff-4".
    set(PKG_CONFIG_REQUIRES_PRIVATE "libtiff-4")
  endif()
  # JPIP TCP socket wrapper (source/core/jpip/tcp_socket.cpp) uses Winsock2.
  if(WIN32)
    target_link_libraries(open_htj2k PUBLIC ws2_32)
  endif()
endif()

# ── QUIC / HTTP/3 support (Phase 5) ─────────────────────────────────────────
# Requires MsQuic (QUIC transport) and nghttp3 (HTTP/3 framing).
# On macOS: brew install libmsquic libnghttp3
# On Linux: install from packages or build from source.
option(OPENHTJ2K_QUIC "Build HTTP/3 QUIC transport for JPIP (requires MsQuic + nghttp3)" OFF)
if(OPENHTJ2K_QUIC AND NOT EMSCRIPTEN)
  find_library(MSQUIC_LIBRARY NAMES msquic)
  find_path(MSQUIC_INCLUDE_DIR NAMES msquic.h)
  find_library(NGHTTP3_LIBRARY NAMES nghttp3)
  find_path(NGHTTP3_INCLUDE_DIR NAMES nghttp3/nghttp3.h)

  if(MSQUIC_LIBRARY AND MSQUIC_INCLUDE_DIR AND NGHTTP3_LIBRARY AND NGHTTP3_INCLUDE_DIR)
    message(STATUS "OPENHTJ2K_QUIC: MsQuic found: ${MSQUIC_LIBRARY}")
    message(STATUS "OPENHTJ2K_QUIC: nghttp3 found: ${NGHTTP3_LIBRARY}")
    target_compile_definitions(open_htj2k PUBLIC "OPENHTJ2K_ENABLE_QUIC")
    target_include_directories(open_htj2k PUBLIC ${MSQUIC_INCLUDE_DIR} ${NGHTTP3_INCLUDE_DIR})
    target_link_libraries(open_htj2k PUBLIC ${MSQUIC_LIBRARY} ${NGHTTP3_LIBRARY})
  else()
    message(WARNING "OPENHTJ2K_QUIC requested but libraries not found:")
    if(NOT MSQUIC_LIBRARY OR NOT MSQUIC_INCLUDE_DIR)
      message(WARNING "  MsQuic not found (brew install libmsquic)")
    endif()
    if(NOT NGHTTP3_LIBRARY OR NOT NGHTTP3_INCLUDE_DIR)
      message(WARNING "  nghttp3 not found (brew install libnghttp3)")
    endif()
    set(OPENHTJ2K_QUIC OFF)
  endif()
endif()

# Compiler optimization settings
if(NOT MSVC)
  set(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wconversion -Wsign-conversion -Wunused-parameter -Wformat=0 -fexceptions"
  )
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fcolor-diagnostics")
  endif ()
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-color=always")
  endif()
  if(NOT COMPILER_SUPPORTS_CXX17)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-attributes")
  endif()
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O0 -g")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS} -O3 -g -DNDEBUG")
else()
  set(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} /wd5051 /D \"_CRT_SECURE_NO_WARNINGS\""
  )
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} /Od /DDEBUG /ZI")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} /Ox")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS} /O2 /ZI")
endif ()
if(CMAKE_CXX_COMPILER_ID MATCHES "Intel$") # Intel Compiler Classic
  message(STATUS "Intel compiler")
  set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wconversion -Wsign-conversion -Wunused-parameter -Wformat=0 -fexceptions -fcolor-diagnostics")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -diag-disable=10441 -Wno-conversion -fno-omit-frame-pointer -mno-omit-leaf-frame-pointer")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O0 -p")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG -ipo")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS} -O3 -g -DNDEBUG")
endif()

if(NOT EMSCRIPTEN)
  # SIMD code paths are selected at compile time from the compiler's target
  # macros (__AVX2__, __AVX512F__, ...), so the flags below decide which CPUs
  # the resulting binaries run on.  Match on the target processor, not the
  # build host, so cross builds get the right flags.  The pointer-size check
  # keeps 32-bit builds on a 64-bit host (for example Visual Studio -A Win32,
  # where CMAKE_SYSTEM_PROCESSOR still names the host) on the scalar code.
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "^[xX]86_64$|^[aA][mM][dD]64$" AND CMAKE_SIZEOF_VOID_P EQUAL 8) # x86_64
    option(ENABLE_AVX2 "Enable the use of Intel AVX2 intrinsics" ON)
    set(OPENHTJ2K_TARGET_CPU "native" CACHE STRING
        "x86-64 code generation: native (build machine; binaries may not run elsewhere), x86-64-v3 (any AVX2 CPU), x86-64-v4 (any AVX-512 CPU), generic (no AVX2)")
    set_property(CACHE OPENHTJ2K_TARGET_CPU PROPERTY STRINGS native x86-64-v3 x86-64-v4 generic)
    if(NOT OPENHTJ2K_TARGET_CPU MATCHES "^(native|x86-64-v3|x86-64-v4|generic)$")
      message(FATAL_ERROR
              "OPENHTJ2K_TARGET_CPU must be native, x86-64-v3, x86-64-v4 or generic (got '${OPENHTJ2K_TARGET_CPU}')")
    endif()
    set(_ohtj2k_cpu "${OPENHTJ2K_TARGET_CPU}")
    if(NOT ENABLE_AVX2)
      # ENABLE_AVX2=OFF means no AVX2 (or wider) code at all, not just no
      # hand-written AVX2 kernels: -march=native would still let __AVX2__ code
      # and compiler auto-vectorization in.
      set(_ohtj2k_cpu "generic")
    endif()

    if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /D \"_CRT_SECURE_NO_WARNINGS\"")
      # MSVC has no "native"; AVX2 is its closest equivalent.
      if(_ohtj2k_cpu STREQUAL "x86-64-v4")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX512")
      elseif(NOT _ohtj2k_cpu STREQUAL "generic")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")
      endif()
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU|IntelLLVM|Intel")
      if(_ohtj2k_cpu STREQUAL "native")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mtune=native")
      elseif(_ohtj2k_cpu STREQUAL "generic")
        # Spell out the baseline: a compiler's default target can be higher
        # (the GCC 15 on the Ubuntu 26.04 CI runners emits AVX code without
        # -march).
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=x86-64 -mtune=generic")
      else()
        # Microarchitecture levels need GCC 11 / Clang 12; spell the feature
        # set out for older compilers.
        string(REPLACE "-" "_" _ohtj2k_level_var "OPENHTJ2K_CXX_HAS_MARCH_${_ohtj2k_cpu}")
        check_cxx_compiler_flag("-march=${_ohtj2k_cpu}" ${_ohtj2k_level_var})
        if(${_ohtj2k_level_var})
          set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=${_ohtj2k_cpu}")
        else()
          set(_ohtj2k_isa "-mavx2 -mfma -mbmi -mbmi2 -mlzcnt -mpopcnt -mf16c -mmovbe")
          if(_ohtj2k_cpu STREQUAL "x86-64-v4")
            string(APPEND _ohtj2k_isa " -mavx512f -mavx512bw -mavx512cd -mavx512dq -mavx512vl")
          endif()
          set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${_ohtj2k_isa}")
        endif()
      endif()
    endif()
    message(STATUS "x86-64 target CPU: ${_ohtj2k_cpu}")
  endif()

  if(CMAKE_SYSTEM_PROCESSOR MATCHES "^[aA][rR][mM]64$|^[aA][aA][rR][cC][hH]64$" AND CMAKE_SIZEOF_VOID_P EQUAL 8) # aarch64
    option(ENABLE_ARM_NEON "Enable the use of ARM NEON intrinsics" ON)
    if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
      set(CMAKE_CXX_FLAGS
          "${CMAKE_CXX_FLAGS} /EHsc /D \"_CRT_SECURE_NO_WARNINGS\"")
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      if(APPLE)
        message(STATUS "Added -mcpu=apple-m1")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=apple-m1 -mtune=native")
      else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a -mtune=cortex-a72")
      endif()
    else()
      set(CMAKE_CXX_FLAGS
          "${CMAKE_CXX_FLAGS} -Wno-unused-parameter -Wno-unused-variable -Wformat-overflow=0 -Wno-unused-but-set-variable"
      )
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a -mtune=cortex-a72")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flax-vector-conversions")
    endif()
  endif()
endif()
# add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/source/thirdparty/highway
# EXCLUDE_FROM_ALL)

# INSTALL related settings
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/source/pkg-config.pc.cmake"
  "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${PROJECT_NAME}.pc"
  @ONLY
)

if (NOT EMSCRIPTEN)
  if(ENABLE_ARM_NEON)
    message(STATUS "OPENHTJ2K_ENABLE_ARM_NEON is set")
    target_compile_definitions(open_htj2k PUBLIC "OPENHTJ2K_ENABLE_ARM_NEON")
  elseif(ENABLE_AVX2)
    message(STATUS "OPENHTJ2K_TRY_AVX2 is set")
    target_compile_definitions(open_htj2k PUBLIC "OPENHTJ2K_TRY_AVX2")
  endif()
endif()

# Command-line tools.  Turn off with -DOPENHTJ2K_BUILD_APPS=OFF to build only
# the library (the test suite needs the tools, so it is skipped as well).
option(OPENHTJ2K_BUILD_APPS "Build the command-line tools (decoder, encoder, JPIP server, ...)" ON)
option(OPENHTJ2K_BUILD_TESTS "Build the test tools and register the test suite (needs OPENHTJ2K_BUILD_APPS)" ON)
set(_ohtj2k_build_tests ${OPENHTJ2K_BUILD_TESTS})
if(OPENHTJ2K_BUILD_TESTS AND NOT OPENHTJ2K_BUILD_APPS)
  message(STATUS "OPENHTJ2K_BUILD_APPS is OFF: the test suite is not built")
  set(_ohtj2k_build_tests OFF)
endif()

if(OPENHTJ2K_BUILD_APPS)
  # decoder target
  add_executable(open_htj2k_dec)
  target_include_directories(open_htj2k_dec PUBLIC source/core/interface)
  add_subdirectory(source/apps/decoder)
  set_target_properties(
    open_htj2k_dec
    PROPERTIES OUTPUT_NAME
               $<IF:$<CONFIG:Debug>,open_htj2k_dec_dbg,open_htj2k_dec>)
  target_link_libraries(open_htj2k_dec PUBLIC open_htj2k)

  # Image-I/O utility (static; in-tree apps only — not part of libopen_htj2k ABI)
  add_library(openhtj2k_imgio STATIC)
  target_include_directories(openhtj2k_imgio PUBLIC source/apps/imgio)
  add_subdirectory(source/apps/imgio)
  if (TIFF_FOUND)
    target_compile_definitions(openhtj2k_imgio PRIVATE OPENHTJ2K_TIFF_SUPPORT)
    target_include_directories(openhtj2k_imgio PRIVATE ${TIFF_INCLUDE_DIR})
    target_link_libraries(openhtj2k_imgio PRIVATE ${TIFF_LIBRARIES})
  endif()

  # encoder target
  add_executable(open_htj2k_enc)
  target_include_directories(open_htj2k_enc PUBLIC source/core/interface)
  add_subdirectory(source/apps/encoder)
  set_target_properties(
    open_htj2k_enc
    PROPERTIES OUTPUT_NAME
               $<IF:$<CONFIG:Debug>,open_htj2k_enc_dbg,open_htj2k_enc>)
  target_link_libraries(open_htj2k_enc PUBLIC open_htj2k PRIVATE openhtj2k_imgio)

  # Estimates the Qfactor used at encode time from a J2C codestream's QCD/QCC.
  add_executable(estimate_qfactor)
  add_subdirectory(source/apps/estimate_qfactor)

  # JPIP Phase-3 standalone HTTP/1.1 server
  add_executable(open_htj2k_jpip_server)
  add_subdirectory(source/apps/jpip_server)
  target_link_libraries(open_htj2k_jpip_server PUBLIC open_htj2k)
  set_target_properties(
    open_htj2k_jpip_server
    PROPERTIES OUTPUT_NAME
               $<IF:$<CONFIG:Debug>,open_htj2k_jpip_server_dbg,open_htj2k_jpip_server>)

  # JPIP foveation benchmark (bandwidth + decode workload comparison)
  add_executable(open_htj2k_jpip_benchmark)
  add_subdirectory(source/apps/jpip_benchmark)
  target_link_libraries(open_htj2k_jpip_benchmark PUBLIC open_htj2k)
  set_target_properties(
    open_htj2k_jpip_benchmark
    PROPERTIES OUTPUT_NAME
               $<IF:$<CONFIG:Debug>,open_htj2k_jpip_benchmark_dbg,open_htj2k_jpip_benchmark>)
endif()

# RFC 9828 RTP receiver (experimental, requires GLFW)
option(OPENHTJ2K_RTP "Build experimental RFC 9828 RTP receiver (requires GLFW)" OFF)
if (OPENHTJ2K_RTP AND OPENHTJ2K_BUILD_APPS AND NOT EMSCRIPTEN)
  find_package(glfw3 QUIET)
  if (glfw3_FOUND)
    # On Apple: native Metal renderer (no OpenGL dependency).
    # Elsewhere: OpenGL 3.3 core renderer.
    if (APPLE)
      set(_RTP_RENDERER_FOUND TRUE)
    else()
      find_package(OpenGL QUIET)
      set(_RTP_RENDERER_FOUND ${OpenGL_FOUND})
    endif()
    if (_RTP_RENDERER_FOUND)
      message(STATUS "OPENHTJ2K_RTP: building open_htj2k_rtp_recv")
      add_executable(open_htj2k_rtp_recv)
      target_include_directories(open_htj2k_rtp_recv PUBLIC source/core/interface)
      # Offline decode profiler shares the same sub-CMakeLists for source lists.
      add_executable(open_htj2k_rtp_decode_profile)
      target_include_directories(open_htj2k_rtp_decode_profile PUBLIC source/core/interface)
      add_subdirectory(source/apps/rtp_recv)
      set_target_properties(
        open_htj2k_rtp_recv
        PROPERTIES OUTPUT_NAME
                   $<IF:$<CONFIG:Debug>,open_htj2k_rtp_recv_dbg,open_htj2k_rtp_recv>)
      set_target_properties(
        open_htj2k_rtp_decode_profile
        PROPERTIES OUTPUT_NAME
                   $<IF:$<CONFIG:Debug>,open_htj2k_rtp_decode_profile_dbg,open_htj2k_rtp_decode_profile>)
      if (APPLE)
        target_compile_definitions(open_htj2k_rtp_recv PRIVATE OPENHTJ2K_USE_METAL)
        target_link_libraries(open_htj2k_rtp_recv PUBLIC open_htj2k glfw
                              "-framework Metal" "-framework QuartzCore" "-framework Cocoa")
        message(STATUS "OPENHTJ2K_USE_METAL is set")
      elseif(WIN32)
        target_link_libraries(open_htj2k_rtp_recv PUBLIC open_htj2k glfw OpenGL::GL ws2_32)
      else()
        target_link_libraries(open_htj2k_rtp_recv PUBLIC open_htj2k glfw OpenGL::GL)
      endif()
      target_link_libraries(open_htj2k_rtp_decode_profile PUBLIC open_htj2k)

      # JPIP Phase-1 foveation demo — shares the rtp_recv renderer, so it
      # lives in the same glfw-gated block.
      message(STATUS "OPENHTJ2K_RTP: building open_htj2k_jpip_demo")
      add_executable(open_htj2k_jpip_demo)
      add_subdirectory(source/apps/jpip_demo)
      set_target_properties(
        open_htj2k_jpip_demo
        PROPERTIES OUTPUT_NAME
                   $<IF:$<CONFIG:Debug>,open_htj2k_jpip_demo_dbg,open_htj2k_jpip_demo>)
      if (APPLE)
        target_link_libraries(open_htj2k_jpip_demo PUBLIC open_htj2k glfw
                              "-framework Metal" "-framework QuartzCore" "-framework Cocoa")
      elseif(WIN32)
        target_link_libraries(open_htj2k_jpip_demo PUBLIC open_htj2k glfw OpenGL::GL)
      else()
        target_link_libraries(open_htj2k_jpip_demo PUBLIC open_htj2k glfw OpenGL::GL)
      endif()
    else()
      message(WARNING "OPENHTJ2K_RTP requested but glfw3/OpenGL not found; skipping rtp_recv target")
    endif()
  else()
    message(WARNING "OPENHTJ2K_RTP requested but glfw3 not found; skipping rtp_recv target")
  endif()
endif()

################################################################################################
# Install
################################################################################################
if(OPENHTJ2K_BUILD_APPS)
  install(TARGETS open_htj2k_dec open_htj2k_enc
    DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
install(TARGETS open_htj2k
  EXPORT openhtj2k-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# Public headers.  Consumers add <prefix>/include/open_htj2k/interface to their
# include path (see the pkg-config Cflags and the CMake package's target), so
# every header the public API includes must be installed there —
# tests/install_check.cmake builds a program against the installed tree to
# keep it that way.
install(DIRECTORY source/core/interface
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/open_htj2k
  FILES_MATCHING
  PATTERN "*.hpp")
install(FILES source/core/common/planar_output_desc.hpp  # used by decoder.hpp
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/open_htj2k/interface)

install(FILES "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${PROJECT_NAME}.pc"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)

# CMake package: find_package(openhtj2k CONFIG) gives the imported target
# openhtj2k::openhtj2k.
include(CMakePackageConfigHelpers)
set(OPENHTJ2K_CMAKE_PACKAGE_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/openhtj2k")
# Dependencies the installed target refers to: the thread library is linked
# publicly; libtiff is private, so only the static library needs it.
set(OPENHTJ2K_CONFIG_FIND_THREADS FALSE)
if(Threads_FOUND AND NOT EMSCRIPTEN)
  set(OPENHTJ2K_CONFIG_FIND_THREADS TRUE)
endif()
set(OPENHTJ2K_CONFIG_FIND_TIFF FALSE)
if(TIFF_FOUND AND NOT EMSCRIPTEN AND NOT BUILD_SHARED_LIBS)
  set(OPENHTJ2K_CONFIG_FIND_TIFF TRUE)
endif()
configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/source/openhtj2k-config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/openhtj2k-config.cmake"
  INSTALL_DESTINATION "${OPENHTJ2K_CMAKE_PACKAGE_DIR}")
# 0.x releases may change the API in any minor version.
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/openhtj2k-config-version.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMinorVersion)
install(EXPORT openhtj2k-targets
  NAMESPACE openhtj2k::
  DESTINATION "${OPENHTJ2K_CMAKE_PACKAGE_DIR}")
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/openhtj2k-config.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/openhtj2k-config-version.cmake"
  DESTINATION "${OPENHTJ2K_CMAKE_PACKAGE_DIR}")

################################################################################################
# Testing
################################################################################################
if(_ohtj2k_build_tests)
  include(${CMAKE_CURRENT_SOURCE_DIR}/tests/test_tools.cmake)

  # Snapshot the test list before any tests/*.cmake include so we can attach the
  # `test_artifacts` cleanup fixture to every test added below — without this,
  # tests added by sibling files (jpip_phase1, encoder_test,
  # row_range_validation) escaped the FIXTURES_REQUIRED loop that used to live
  # inside decoder_conformance.cmake, letting cleanup_artifacts race their
  # decoder output and produce intermittent "file X not found" failures under
  # `ctest -j`.
  get_property(_tests_before_conformance DIRECTORY PROPERTY TESTS)

  # conformance testing — declares the `test_artifacts` cleanup fixture and the
  # main HT/Part-1/Part-2/WASM test sets.
  include(${CMAKE_CURRENT_SOURCE_DIR}/tests/decoder_conformance.cmake)
  include(${CMAKE_CURRENT_SOURCE_DIR}/tests/encoder_test.cmake)
  include(${CMAKE_CURRENT_SOURCE_DIR}/tests/jpip_phase1.cmake)
  include(${CMAKE_CURRENT_SOURCE_DIR}/tests/row_range_validation.cmake)
  include(${CMAKE_CURRENT_SOURCE_DIR}/tests/col_range_validation.cmake)
  include(${CMAKE_CURRENT_SOURCE_DIR}/tests/concurrency.cmake)
  include(${CMAKE_CURRENT_SOURCE_DIR}/tests/install_check.cmake)

  # Auto-attach the cleanup fixture to every test added by the includes above.
  # Skip cleanup_artifacts itself (it IS the fixture), and skip tests that
  # already declared FIXTURES_REQUIRED explicitly (encoder_test does this so
  # its DEPENDS chain is preserved).
  get_property(_tests_after_conformance DIRECTORY PROPERTY TESTS)
  foreach(_t IN LISTS _tests_after_conformance)
    if(NOT _t IN_LIST _tests_before_conformance AND NOT _t STREQUAL "cleanup_artifacts")
      get_test_property(${_t} FIXTURES_REQUIRED _existing)
      if(NOT _existing OR NOT "test_artifacts" IN_LIST _existing)
        set_tests_properties(${_t} PROPERTIES FIXTURES_REQUIRED test_artifacts)
      endif()
    endif()
  endforeach()
endif()
