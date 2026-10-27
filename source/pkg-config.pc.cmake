prefix=@CMAKE_INSTALL_PREFIX@
libdir=@PKG_CONFIG_LIBDIR@
includedir=@PKG_CONFIG_INCLUDEDIR@/open_htj2k

Name: @PROJECT_NAME@
Description: @PROJECT_DESCRIPTION@
Version: @PROJECT_VERSION@
Requires.private: @PKG_CONFIG_REQUIRES_PRIVATE@
Libs: -L${libdir} -lopenhtj2k
Libs.private: @PKG_CONFIG_LIBS_PRIVATE@
Cflags: -I${includedir}/interface@PKG_CONFIG_CFLAGS_EXTRA@
