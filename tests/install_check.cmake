# The installed package must work on its own: an external project that only
# has the installed headers, the library and the package files has to be able
# to compile, link and run against it.  The in-tree targets cannot catch a
# missing header or a broken package file, because they see the whole source
# tree through open_htj2k's build-tree include directories.
#
# The test installs the build and builds tests/install_consumer twice: as a
# CMake project using find_package(openhtj2k CONFIG), and (GCC/Clang-style
# toolchains with pkg-config only) from the flags `pkg-config` reports.

if(NOT EMSCRIPTEN AND NOT CMAKE_CROSSCOMPILING AND NOT IS_ABSOLUTE "${CMAKE_INSTALL_LIBDIR}")
  set(_ohtj2k_consumer_pkg_config "")
  if(NOT WIN32)
    find_program(OPENHTJ2K_PKG_CONFIG NAMES pkg-config pkgconf)
    if(OPENHTJ2K_PKG_CONFIG)
      set(_ohtj2k_consumer_pkg_config "${OPENHTJ2K_PKG_CONFIG}")
    endif()
  endif()
  add_test(NAME install_consumer
           COMMAND ${CMAKE_COMMAND}
                   -DBUILD_DIR=${CMAKE_BINARY_DIR}
                   -DCONFIG=$<CONFIG>
                   -DWORK_DIR=${CMAKE_BINARY_DIR}/install_check
                   -DCONSUMER_DIR=${CMAKE_CURRENT_SOURCE_DIR}/tests/install_consumer
                   -DINPUT=${CONFORMANCE_DATA_DIR}/ds0_ht_02_b11.j2k
                   -DCXX=${CMAKE_CXX_COMPILER}
                   "-DGENERATOR=${CMAKE_GENERATOR}"
                   "-DGENERATOR_PLATFORM=${CMAKE_GENERATOR_PLATFORM}"
                   "-DGENERATOR_TOOLSET=${CMAKE_GENERATOR_TOOLSET}"
                   "-DMAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
                   "-DPKG_CONFIG=${_ohtj2k_consumer_pkg_config}"
                   -DLIBDIR=${CMAKE_INSTALL_LIBDIR}
                   -DBINDIR=${CMAKE_INSTALL_BINDIR}
                   -DSTATIC=$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>
                   -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/install_consumer/run.cmake)
endif()
