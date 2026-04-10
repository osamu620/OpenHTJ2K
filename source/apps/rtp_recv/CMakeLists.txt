cmake_policy(SET CMP0076 NEW)
target_sources(open_htj2k_rtp_recv
    PRIVATE
        main_rtp_recv.cpp
        rtp_socket.cpp
        rfc9828_parser.cpp
        frame_handler.cpp
        gl_renderer.cpp
)
