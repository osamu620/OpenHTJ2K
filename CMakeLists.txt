cmake_minimum_required(VERSION 3.13)

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

if(NOT MSVC)
  include(GNUInstallDirs)
  set(PKG_CONFIG_LIBDIR "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
  set(PKG_CONFIG_INCLUDEDIR "${CMAKE_INSTALL_PREFIX}/include/open_htj2k")
  set(PKG_CONFIG_LIBS "-L${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/open_htj2k")
  set(PKG_CONFIG_CFLAGS "-I${PKG_CONFIG_INCLUDEDIR}/interface")
endif()
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

# Compiler settings
if(COMPILER_SUPPORTS_CXX17)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
elseif(COMPILER_SUPPORTS_CXX14)
  set(CMAKE_CXX_STANDARD 14)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
else()
  set(CMAKE_CXX_STANDARD 11)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

message(STATUS "${CMAKE_HOST_SYSTEM_PROCESSOR} ${CMAKE_CXX_COMPILER_ID}")

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
target_include_directories(
    open_htj2k
    PRIVATE source/core/common
    source/core/codestream
    source/core/coding
    source/core/transform
    # source/core/interface
    source/core/jph
    # ${CMAKE_CURRENT_SOURCE_DIR}/source/thirdparty/highway
)
target_include_directories(open_htj2k INTERFACE source/core/interface)
set_target_properties(
        open_htj2k PROPERTIES
        OUTPUT_NAME $<IF:$<CONFIG:Debug>,open_htj2k_d,open_htj2k_R>
        VERSION ${PROJECT_VERSION}
        SOVERSION ${OPENHTJ2K_VERSION_MAJOR})
# target_link_libraries(open_htj2k PUBLIC ${CMAKE_THREAD_LIBS_INIT} PRIVATE hwy)

find_package(Threads)
find_package(TIFF)

if (EMSCRIPTEN)
  target_compile_definitions(open_htj2k PUBLIC "OPENHTJ2K_ENABLE_ARM_NEON")
  set(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} -msimd128 -mrelaxed-simd -mfpu=neon"
  )
endif()
if (NOT EMSCRIPTEN)
  # Check Thread support
  if (Threads_FOUND)
    message(STATUS "Thread library found")
    target_link_libraries(open_htj2k PUBLIC ${CMAKE_THREAD_LIBS_INIT})
    target_compile_definitions(open_htj2k PUBLIC "OPENHTJ2K_THREAD")
    message(STATUS "OPENHTJ2K_THREAD is set")
  endif()
  # Check TIFF support
  if (TIFF_FOUND)
    message(STATUS "libtiff found")
    message(STATUS "tiff include = ${TIFF_INCLUDE_DIR}")
    message(STATUS "tiff library = ${TIFF_LIBRARIES}")
    target_compile_definitions(open_htj2k PUBLIC "OPENHTJ2K_TIFF_SUPPORT")
    target_include_directories(open_htj2k PUBLIC ${TIFF_INCLUDE_DIR})
    target_link_libraries(open_htj2k PUBLIC ${TIFF_LIBRARIES})
    message(STATUS "OPENHTJ2K_TIFF_SUPPORT is set ")
    set(PKG_CONFIG_REQUIRES "libtiff")
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
  if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "^[xX]86_64$|^[aA][mM][dD]64$") # x86_64
    option(ENABLE_AVX2 "Enable the use of Intel AVX2 intrinsics" ON)
    if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
      set(CMAKE_CXX_FLAGS
          "${CMAKE_CXX_FLAGS} /arch:AVX2 /EHsc /D \"_CRT_SECURE_NO_WARNINGS\"")
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU|IntelLLVM|Intel")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mtune=native")
    endif()
  endif()

  if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "^[aA][rR][mM]64$|^[aA][aA][rR][cC][hH]64$") # aarch64
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

# decoder target
add_executable(open_htj2k_dec)
target_include_directories(open_htj2k_dec PUBLIC source/core/interface)
add_subdirectory(source/apps/decoder)
set_target_properties(
  open_htj2k_dec
  PROPERTIES OUTPUT_NAME
             $<IF:$<CONFIG:Debug>,open_htj2k_dec_dbg,open_htj2k_dec>)
target_link_libraries(open_htj2k_dec PUBLIC open_htj2k)

# encoder target
add_executable(open_htj2k_enc)
target_include_directories(open_htj2k_enc PUBLIC source/core/interface)
add_subdirectory(source/apps/encoder)
set_target_properties(
  open_htj2k_enc
  PROPERTIES OUTPUT_NAME
             $<IF:$<CONFIG:Debug>,open_htj2k_enc_dbg,open_htj2k_enc>)
target_link_libraries(open_htj2k_enc PUBLIC open_htj2k)

# imgcmp target
add_executable(imgcmp)
add_subdirectory(source/apps/imgcmp)
target_include_directories(imgcmp PUBLIC source/apps/imgcmp/)

################################################################################################
# Install
################################################################################################
install(TARGETS open_htj2k_dec
  DESTINATION bin)
install(TARGETS open_htj2k_enc
  DESTINATION bin)
install(TARGETS open_htj2k LIBRARY
  DESTINATION ${CMAKE_INSTALL_LIBDIR})

install (DIRECTORY source/core/interface
  DESTINATION include/open_htj2k
  FILES_MATCHING
  PATTERN "*.hpp")

install(FILES "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${PROJECT_NAME}.pc"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)

################################################################################################
# Testing
################################################################################################
# conformance testing
include(${CMAKE_CURRENT_SOURCE_DIR}/tests/decoder_conformance.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/tests/encoder_test.cmake)
