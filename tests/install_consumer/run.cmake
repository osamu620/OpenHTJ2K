# Installs the build tree into a scratch prefix, then builds and runs
# consumer.cpp against it using only what an external project sees: the
# installed headers, the installed library and
# `pkg-config --cflags --libs open_htj2k`.
#
# Invoked by tests/install_check.cmake as `cmake -D... -P run.cmake` with:
#   BUILD_DIR     build tree to install
#   WORK_DIR      scratch directory (recreated on every run)
#   CONSUMER_SRC  consumer.cpp
#   INPUT         codestream passed to the consumer
#   CXX           C++ compiler
#   PKG_CONFIG    pkg-config executable
#   LIBDIR        CMAKE_INSTALL_LIBDIR (relative)
#   STATIC        true when the library is built static (adds --static)

foreach(_v BUILD_DIR WORK_DIR CONSUMER_SRC INPUT CXX PKG_CONFIG LIBDIR)
  if(NOT DEFINED ${_v})
    message(FATAL_ERROR "install_check: ${_v} is not set")
  endif()
endforeach()

set(_prefix "${WORK_DIR}/prefix")
file(REMOVE_RECURSE "${WORK_DIR}")

execute_process(
  COMMAND "${CMAKE_COMMAND}" --install "${BUILD_DIR}" --prefix "${_prefix}"
  RESULT_VARIABLE _rc OUTPUT_VARIABLE _out ERROR_VARIABLE _out)
if(_rc)
  message(FATAL_ERROR "cmake --install failed:\n${_out}")
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
  COMMAND "${CXX}" -std=c++11 "${CONSUMER_SRC}" ${_flags} "-Wl,-rpath,${_prefix}/${LIBDIR}"
          -o "${WORK_DIR}/consumer"
  RESULT_VARIABLE _rc OUTPUT_VARIABLE _out ERROR_VARIABLE _out)
if(_rc)
  message(FATAL_ERROR "building against the installed tree failed:\n${_out}")
endif()

execute_process(
  COMMAND "${WORK_DIR}/consumer" "${INPUT}"
  RESULT_VARIABLE _rc OUTPUT_VARIABLE _out ERROR_VARIABLE _out)
message(STATUS "${_out}")
if(_rc)
  message(FATAL_ERROR "consumer built against the installed tree failed (exit ${_rc})")
endif()
