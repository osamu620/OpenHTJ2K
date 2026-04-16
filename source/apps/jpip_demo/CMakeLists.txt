cmake_policy(SET CMP0076 NEW)

# Pull the renderer sources (and their GLFW/Metal/GL glue) from the rtp_recv
# app.  Keeping them in rtp_recv/ avoids duplicating the renderer backend
# while still letting the demo ship as its own executable.
set(_RTP_SRC ${CMAKE_SOURCE_DIR}/source/apps/rtp_recv)

target_sources(open_htj2k_jpip_demo
    PRIVATE
    main_jpip_demo.cpp
)

target_include_directories(open_htj2k_jpip_demo
    PRIVATE
    ${CMAKE_SOURCE_DIR}/source/core/interface
    ${CMAKE_SOURCE_DIR}/source/core/jpip
    ${_RTP_SRC}
)

if (APPLE)
    target_sources(open_htj2k_jpip_demo PRIVATE
        ${_RTP_SRC}/metal_renderer.mm
    )
    target_compile_definitions(open_htj2k_jpip_demo PRIVATE OPENHTJ2K_USE_METAL)
else()
    target_sources(open_htj2k_jpip_demo PRIVATE
        ${_RTP_SRC}/gl_renderer.cpp
        ${_RTP_SRC}/gl_loader.cpp
    )
endif()
