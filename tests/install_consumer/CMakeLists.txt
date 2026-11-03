# A project that uses an installed OpenHTJ2K through its CMake package, as
# any external project would.  Configured and built by run.cmake with
# CMAKE_PREFIX_PATH pointing at the scratch install prefix.
cmake_minimum_required(VERSION 3.13)
# MSVC: no default /W3 in CMAKE_CXX_FLAGS to clash with the /W4 below.
if(POLICY CMP0092)
  cmake_policy(SET CMP0092 NEW)
endif()
project(openhtj2k_install_consumer LANGUAGES CXX)

find_package(openhtj2k CONFIG REQUIRED)

# Warnings are errors: a warning that the installed headers raise in a
# consumer's build fails the test.  (NO_SYSTEM_FROM_IMPORTED below: the
# include directories of an imported target are otherwise system directories,
# where compilers do not warn.)
if(MSVC)
  set(_consumer_warnings /W4 /WX)
else()
  set(_consumer_warnings -Wall -Wextra -Wpedantic -Wshadow -Wundef -Wconversion -Wsign-conversion
                         -Wold-style-cast -Wcast-qual -Wmissing-declarations -Werror)
endif()

add_executable(consumer consumer.cpp consumer_paths.cpp)
target_link_libraries(consumer PRIVATE openhtj2k::openhtj2k)
target_compile_options(consumer PRIVATE ${_consumer_warnings})
if(OPENHTJ2K_EXPECT_STATIC)
  target_compile_definitions(consumer PRIVATE OPENHTJ2K_EXPECT_STATIC)
endif()
set_target_properties(consumer PROPERTIES
  CXX_STANDARD 11
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  NO_SYSTEM_FROM_IMPORTED ON
  # One output directory for every configuration, so run.cmake can find it.
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<1:out>")

# Every installed public header must compile on its own (C++17 here; the
# consumer above is C++11): one translation unit per header, which includes it
# first by its long path, then again by its short name.
get_target_property(_ohtj2k_include_dirs openhtj2k::openhtj2k INTERFACE_INCLUDE_DIRECTORIES)
set(_ohtj2k_headers "")
foreach(_dir IN LISTS _ohtj2k_include_dirs)
  if(_dir MATCHES "/open_htj2k/interface$")
    file(GLOB _ohtj2k_headers RELATIVE "${_dir}" "${_dir}/*.hpp")
  endif()
endforeach()
if(NOT _ohtj2k_headers)
  message(FATAL_ERROR "no installed headers under <include>/open_htj2k/interface (${_ohtj2k_include_dirs})")
endif()
set(_header_tus "")
foreach(_header IN LISTS _ohtj2k_headers)
  string(MAKE_C_IDENTIFIER "${_header}" _id)
  set(_tu "${CMAKE_CURRENT_BINARY_DIR}/header_${_id}.cpp")
  file(WRITE "${_tu}" "#include <open_htj2k/interface/${_header}>\n#include <${_header}>\n")
  list(APPEND _header_tus "${_tu}")
endforeach()
message(STATUS "installed headers: ${_ohtj2k_headers}")
add_library(consumer_headers OBJECT ${_header_tus})
target_link_libraries(consumer_headers PRIVATE openhtj2k::openhtj2k)
target_compile_options(consumer_headers PRIVATE ${_consumer_warnings})
set_target_properties(consumer_headers PROPERTIES CXX_STANDARD 17 CXX_EXTENSIONS OFF NO_SYSTEM_FROM_IMPORTED ON)
