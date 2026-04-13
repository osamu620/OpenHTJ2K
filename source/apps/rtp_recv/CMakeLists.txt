cmake_policy(SET CMP0076 NEW)
target_sources(open_htj2k_rtp_recv
    PRIVATE
        main_rtp_recv.cpp
        cli.cpp
        decode_helpers.cpp
        smoke_tests.cpp
        pipeline_single_threaded.cpp
        pipeline_multi_threaded.cpp
        rtp_socket.cpp
        rfc9828_parser.cpp
        frame_handler.cpp
)
if (APPLE)
  target_sources(open_htj2k_rtp_recv PRIVATE metal_renderer.mm)
  # shaders.metal is compiled at runtime from the embedded source string in
  # metal_renderer.mm.  The .metal file is kept as the authoritative source
  # for readability; a future step can add offline .metallib compilation
  # once the Metal Toolchain component is installed (xcodebuild
  # -downloadComponent MetalToolchain).
else()
  target_sources(open_htj2k_rtp_recv PRIVATE gl_loader.cpp gl_renderer.cpp)
endif()

# Offline decode profiler — feeds dumped .j2c codestreams through the same
# decoder path decode_thread_main uses, with per-stage timing instrumentation.
# Standalone executable; does not depend on glfw/OpenGL.
target_sources(open_htj2k_rtp_decode_profile
    PRIVATE
        tools/rtp_decode_profile.cpp
)
