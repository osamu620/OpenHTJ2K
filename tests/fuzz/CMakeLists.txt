# Fuzz targets (docs/dev/fuzzing.md).
#
# Each <name>_fuzzer.cpp defines LLVMFuzzerTestOneInput().  It is built
#   - with OPENHTJ2K_BUILD_FUZZERS=ON: against a fuzzing engine, $LIB_FUZZING_ENGINE when the
#     environment gives one (OSS-Fuzz, ClusterFuzzLite), libFuzzer otherwise;
#   - without it, as part of the test suite: with standalone_main.cpp, which runs the target
#     once on every file given, and the tests fuzz_replay_<name> run it on corpus/<name>_fuzzer.
set(_ohtj2k_fuzz_targets
    decode_fuzzer
    transcode_fuzzer
    jpip_client_fuzzer
    jpip_request_fuzzer
    jpip_server_fuzzer
    image_reader_fuzzer)

if(OPENHTJ2K_BUILD_FUZZERS)
  if(BUILD_SHARED_LIBS)
    # A fuzz target is copied to and run on other machines: it must not need libopenhtj2k.so.
    message(FATAL_ERROR "OPENHTJ2K_BUILD_FUZZERS needs the static library: add -DBUILD_SHARED_LIBS=OFF")
  endif()
  if(DEFINED ENV{LIB_FUZZING_ENGINE})
    # The caller's CXXFLAGS carry the instrumentation that goes with its engine.
    set(_ohtj2k_fuzz_engine "$ENV{LIB_FUZZING_ENGINE}")
  else()
    set(_ohtj2k_fuzz_engine "-fsanitize=fuzzer")
    target_compile_options(open_htj2k PRIVATE -fsanitize=fuzzer-no-link)
  endif()
  message(STATUS "Fuzz targets: built with the fuzzing engine ${_ohtj2k_fuzz_engine}")
endif()

foreach(_t IN LISTS _ohtj2k_fuzz_targets)
  add_executable(${_t} ${_t}.cpp)
  target_include_directories(${_t} PRIVATE
      ${PROJECT_SOURCE_DIR}/source/core/interface
      ${PROJECT_SOURCE_DIR}/source/core/jpip)
  target_link_libraries(${_t} PRIVATE open_htj2k openhtj2k_internal)
  if(OPENHTJ2K_BUILD_FUZZERS)
    if(NOT DEFINED ENV{LIB_FUZZING_ENGINE})
      target_compile_options(${_t} PRIVATE -fsanitize=fuzzer-no-link)
    endif()
    target_link_libraries(${_t} PRIVATE ${_ohtj2k_fuzz_engine})
  else()
    target_sources(${_t} PRIVATE standalone_main.cpp)
  endif()

  if(_ohtj2k_build_tests)
    file(GLOB _inputs CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/corpus/${_t}/*)
    string(REGEX REPLACE "_fuzzer$" "" _name ${_t})
    add_test(NAME fuzz_replay_${_name} COMMAND ${_t} ${_inputs})
    set_tests_properties(fuzz_replay_${_name} PROPERTIES TIMEOUT 300)
  endif()
endforeach()
