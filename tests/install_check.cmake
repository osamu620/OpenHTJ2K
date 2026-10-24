# The installed package must work on its own: an external project that only
# has the installed headers, the library and the pkg-config file has to be
# able to compile, link and run against it.  The in-tree targets cannot catch
# a missing header or a broken .pc file, because they see the whole source
# tree through open_htj2k's INTERFACE include directories.
#
# Needs pkg-config and a native, non-Windows toolchain (the consumer is built
# with GCC/Clang-style flags and run directly).

if(NOT WIN32 AND NOT EMSCRIPTEN AND NOT CMAKE_CROSSCOMPILING AND NOT IS_ABSOLUTE "${CMAKE_INSTALL_LIBDIR}")
  find_program(OPENHTJ2K_PKG_CONFIG NAMES pkg-config pkgconf)
  if(OPENHTJ2K_PKG_CONFIG)
    add_test(NAME install_consumer
             COMMAND ${CMAKE_COMMAND}
                     -DBUILD_DIR=${CMAKE_BINARY_DIR}
                     -DWORK_DIR=${CMAKE_BINARY_DIR}/install_check
                     -DCONSUMER_SRC=${CMAKE_CURRENT_SOURCE_DIR}/tests/install_consumer/consumer.cpp
                     -DINPUT=${CONFORMANCE_DATA_DIR}/ds0_ht_02_b11.j2k
                     -DCXX=${CMAKE_CXX_COMPILER}
                     -DPKG_CONFIG=${OPENHTJ2K_PKG_CONFIG}
                     -DLIBDIR=${CMAKE_INSTALL_LIBDIR}
                     -DSTATIC=$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>
                     -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/install_consumer/run.cmake)
  endif()
endif()
