# A project that uses an installed OpenHTJ2K through its CMake package, as
# any external project would.  Configured and built by run.cmake with
# CMAKE_PREFIX_PATH pointing at the scratch install prefix.
cmake_minimum_required(VERSION 3.13)
project(openhtj2k_install_consumer LANGUAGES CXX)

find_package(openhtj2k CONFIG REQUIRED)

add_executable(consumer consumer.cpp)
target_link_libraries(consumer PRIVATE openhtj2k::openhtj2k)
set_target_properties(consumer PROPERTIES
  CXX_STANDARD 11
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  # One output directory for every configuration, so run.cmake can find it.
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<1:out>")
