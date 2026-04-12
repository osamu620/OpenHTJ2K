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
        gl_loader.cpp
        gl_renderer.cpp
)

# Offline decode profiler — feeds dumped .j2c codestreams through the same
# decoder path decode_thread_main uses, with per-stage timing instrumentation.
# Standalone executable; does not depend on glfw/OpenGL.
target_sources(open_htj2k_rtp_decode_profile
    PRIVATE
        tools/rtp_decode_profile.cpp
)
