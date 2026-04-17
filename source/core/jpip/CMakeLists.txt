cmake_policy(SET CMP0076 NEW)
target_sources(open_htj2k
    PRIVATE
    precinct_index.cpp
    view_window.cpp
    vbas.cpp
    jpp_message.cpp
    codestream_walker.cpp
    data_bin_emitter.cpp
    jpp_parser.cpp
    packet_locator.cpp
    codestream_assembler.cpp
    jpip_request.cpp
    jpip_response.cpp
    tcp_socket.cpp
    jpip_client.cpp
)
