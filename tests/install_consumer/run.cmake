# Installs the build tree into a scratch prefix, then builds and runs
# consumer.cpp against it using only what an external project sees: the
# installed headers, the installed library and the installed package files.
# The consumer is built twice:
#   1. as the CMake project in this directory, via find_package(openhtj2k);
#   2. when PKG_CONFIG is set, directly with the compiler and
#      `pkg-config --cflags --libs open_htj2k`.
#
# Invoked by tests/install_check.cmake as `cmake -D... -P run.cmake` with:
#   BUILD_DIR     build tree to install
#   CONFIG        build configuration to install and build the consumer in
#   WORK_DIR      scratch directory (recreated on every run)
#   CONSUMER_DIR  this directory (consumer.cpp and the consumer CMake project)
#   INPUT         codestream passed to the consumer
#   CXX           C++ compiler
#   GENERATOR, GENERATOR_PLATFORM, GENERATOR_TOOLSET, MAKE_PROGRAM
#                 the build tree's CMake generator settings, reused for the
#                 consumer project
#   PKG_CONFIG    pkg-config executable, or empty to skip step 2
#   LIBDIR        CMAKE_INSTALL_LIBDIR (relative)
#   BINDIR        CMAKE_INSTALL_BINDIR (relative; Windows DLLs go there)
#   STATIC        true when the library is built static (adds --static)

foreach(_v BUILD_DIR CONFIG WORK_DIR CONSUMER_DIR INPUT CXX GENERATOR LIBDIR BINDIR)
  if(NOT DEFINED ${_v} OR "${${_v}}" STREQUAL "")
    message(FATAL_ERROR "install_check: ${_v} is not set")
  endif()
endforeach()

set(_prefix "${WORK_DIR}/prefix")
file(REMOVE_RECURSE "${WORK_DIR}")

execute_process(
  COMMAND "${CMAKE_COMMAND}" --install "${BUILD_DIR}" --prefix "${_prefix}" --config "${CONFIG}"
  RESULT_VARIABLE _rc OUTPUT_VARIABLE _out ERROR_VARIABLE _out)
if(_rc)
  message(FATAL_ERROR "cmake --install failed:\n${_out}")
endif()

# Windows finds DLLs on PATH; elsewhere the consumers carry an rpath.  (Script
# mode has no target platform, so test the host; this test does not run when
# cross compiling.)
set(_exe_suffix "")
if(CMAKE_HOST_WIN32)
  set(_exe_suffix ".exe")
  file(TO_NATIVE_PATH "${_prefix}/${BINDIR}" _dll_dir)
  set(ENV{PATH} "${_dll_dir};$ENV{PATH}")
endif()

function(run_consumer exe label)
  execute_process(
    COMMAND "${exe}" "${INPUT}"
    RESULT_VARIABLE _rc OUTPUT_VARIABLE _out ERROR_VARIABLE _out)
  message(STATUS "${label}: ${_out}")
  if(_rc)
    message(FATAL_ERROR "${label}: consumer built against the installed tree failed (exit ${_rc})")
  endif()
endfunction()

# ── 1. CMake package ─────────────────────────────────────────────────────────
set(_cm_bin "${WORK_DIR}/cmake_consumer")
set(_cm_args -G "${GENERATOR}")
if(GENERATOR_PLATFORM)
  list(APPEND _cm_args -A "${GENERATOR_PLATFORM}")
endif()
if(GENERATOR_TOOLSET)
  list(APPEND _cm_args -T "${GENERATOR_TOOLSET}")
endif()
if(NOT GENERATOR MATCHES "^Visual Studio|^Xcode")
  # The IDE generators pick their own compiler and build tool.
  list(APPEND _cm_args "-DCMAKE_CXX_COMPILER=${CXX}" "-DCMAKE_BUILD_TYPE=${CONFIG}")
  if(MAKE_PROGRAM)
    list(APPEND _cm_args "-DCMAKE_MAKE_PROGRAM=${MAKE_PROGRAM}")
  endif()
endif()
execute_process(
  COMMAND "${CMAKE_COMMAND}" ${_cm_args} "-DCMAKE_PREFIX_PATH=${_prefix}"
          -S "${CONSUMER_DIR}" -B "${_cm_bin}"
  RESULT_VARIABLE _rc OUTPUT_VARIABLE _out ERROR_VARIABLE _out)
if(_rc)
  message(FATAL_ERROR "configuring the find_package(openhtj2k) consumer failed:\n${_out}")
endif()
execute_process(
  COMMAND "${CMAKE_COMMAND}" --build "${_cm_bin}" --config "${CONFIG}"
  RESULT_VARIABLE _rc OUTPUT_VARIABLE _out ERROR_VARIABLE _out)
if(_rc)
  message(FATAL_ERROR "building the find_package(openhtj2k) consumer failed:\n${_out}")
endif()
run_consumer("${_cm_bin}/out/consumer${_exe_suffix}" "find_package")

# ── 2. pkg-config ────────────────────────────────────────────────────────────
if(NOT PKG_CONFIG)
  return()
endif()

# The installed .pc names the configured prefix.  Relocate it by rewriting
# only its prefix= line: every other path in the file must follow.
set(_pc "${_prefix}/${LIBDIR}/pkgconfig/open_htj2k.pc")
if(NOT EXISTS "${_pc}")
  message(FATAL_ERROR "open_htj2k.pc was not installed to ${_pc}")
endif()
file(READ "${_pc}" _pc_text)
string(REGEX REPLACE "(^|\n)prefix=[^\n]*" "\\1prefix=${_prefix}" _pc_text "${_pc_text}")
file(WRITE "${_pc}" "${_pc_text}")

set(ENV{PKG_CONFIG_PATH} "${_prefix}/${LIBDIR}/pkgconfig")
set(_pc_args "")
if(STATIC)
  list(APPEND _pc_args --static)
endif()
execute_process(
  COMMAND "${PKG_CONFIG}" ${_pc_args} --cflags --libs open_htj2k
  RESULT_VARIABLE _rc OUTPUT_VARIABLE _flags ERROR_VARIABLE _err
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(_rc)
  message(FATAL_ERROR "pkg-config could not resolve the installed open_htj2k.pc:\n${_err}")
endif()
message(STATUS "pkg-config flags: ${_flags}")
separate_arguments(_flags UNIX_COMMAND "${_flags}")

execute_process(
  COMMAND "${CXX}" -std=c++11 "${CONSUMER_DIR}/consumer.cpp" ${_flags} "-Wl,-rpath,${_prefix}/${LIBDIR}"
          -o "${WORK_DIR}/consumer"
  RESULT_VARIABLE _rc OUTPUT_VARIABLE _out ERROR_VARIABLE _out)
if(_rc)
  message(FATAL_ERROR "building against the installed tree failed:\n${_out}")
endif()
run_consumer("${WORK_DIR}/consumer" "pkg-config")
