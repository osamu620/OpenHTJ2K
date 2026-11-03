# Test tools used by the CTest suite (included from the top-level
# CMakeLists.txt when OPENHTJ2K_BUILD_TESTS is ON; paths are relative to the
# top-level source directory).  Tools that link open_htj2k also link
# openhtj2k_internal: they include internal headers, which must see the
# library's configuration macros.

# imgcmp target
add_executable(imgcmp)
add_subdirectory(tests/tools/imgcmp)
target_include_directories(imgcmp PUBLIC tests/tools/imgcmp/)
target_link_libraries(imgcmp PRIVATE openhtj2k_imgio)

add_executable(row_range_compare)
add_subdirectory(tests/tools/row_range_compare)
target_include_directories(row_range_compare PUBLIC source/core/interface)
target_link_libraries(row_range_compare PUBLIC open_htj2k PRIVATE openhtj2k_internal)

add_executable(col_range_compare)
add_subdirectory(tests/tools/col_range_compare)
target_include_directories(col_range_compare PUBLIC source/core/interface)
target_link_libraries(col_range_compare PUBLIC open_htj2k PRIVATE openhtj2k_internal)

# stream_rows_check: the rows of every streaming entry point against invoke_line_based()
add_executable(stream_rows_check)
add_subdirectory(tests/tools/stream_rows_check)
target_include_directories(stream_rows_check PUBLIC source/core/interface)
target_link_libraries(stream_rows_check PUBLIC open_htj2k PRIVATE openhtj2k_internal)

# concurrent_instances: many decoder/encoder instances running at once on the
# shared worker pool (instance lifetime + per-thread encoder scratch)
if(Threads_FOUND AND NOT EMSCRIPTEN)
  add_executable(concurrent_instances)
  add_subdirectory(tests/tools/concurrent_instances)
  target_include_directories(concurrent_instances PUBLIC source/core/interface)
  target_link_libraries(concurrent_instances PUBLIC open_htj2k PRIVATE openhtj2k_internal Threads::Threads)
  # threadpool_check: the worker pool's task ring grows instead of failing, and
  # SingleThreadScope hides the pool from a one-thread instance
  add_executable(threadpool_check)
  add_subdirectory(tests/tools/threadpool_check)
  target_link_libraries(threadpool_check PUBLIC open_htj2k PRIVATE openhtj2k_internal Threads::Threads)
endif()

# message_handler_check: the library prints nothing by itself; warnings go to the message handler
# and errors are exceptions with a description
add_executable(message_handler_check)
add_subdirectory(tests/tools/message_handler_check)
target_link_libraries(message_handler_check PUBLIC open_htj2k)

# codestream_bounds_check: unit test for j2c_src_memory's 4 GiB allocation bound
add_executable(codestream_bounds_check)
add_subdirectory(tests/tools/codestream_bounds_check)
target_link_libraries(codestream_bounds_check PUBLIC open_htj2k PRIVATE openhtj2k_internal)

# jph_box_check: LBox / XLBox / LBox 0 in JPH and JP2 files, written and read
add_executable(jph_box_check)
add_subdirectory(tests/tools/jph_box_check)
target_link_libraries(jph_box_check PUBLIC open_htj2k PRIVATE openhtj2k_internal)

# visual_weight_check: dump + invariant self-check for the analytic per-subband
# visual (CSF) weighting (header-only, no library dependency)
add_executable(visual_weight_check)
add_subdirectory(tests/tools/visual_weight_check)

# encode_entry_points: buffered vs streaming encoder API over a matrix of
# component counts / MCT / transforms / DWT levels / bit depths
add_executable(encode_entry_points)
add_subdirectory(tests/tools/encode_entry_points)
target_include_directories(encode_entry_points PUBLIC source/core/interface)
target_link_libraries(encode_entry_points PUBLIC open_htj2k PRIVATE openhtj2k_internal)

# region_decode_bench: M1 region-decode latency/throughput benchmark (OSD eval)
add_executable(region_decode_bench)
add_subdirectory(tests/tools/region_decode_bench)
target_include_directories(region_decode_bench PUBLIC source/core/interface)
target_link_libraries(region_decode_bench PUBLIC open_htj2k PRIVATE openhtj2k_internal)

# JPIP Phase-1 index-check harness (used by tests/jpip_phase1.cmake)
add_executable(jpip_index_check)
add_subdirectory(tests/tools/jpip_index_check)
target_link_libraries(jpip_index_check PUBLIC open_htj2k PRIVATE openhtj2k_internal)

# JPIP Phase-1 partial-decode sanity harness (used by tests/jpip_phase1.cmake)
add_executable(jpip_partial_decode_check)
add_subdirectory(tests/tools/jpip_partial_decode_check)
target_include_directories(jpip_partial_decode_check PUBLIC source/core/interface)
target_link_libraries(jpip_partial_decode_check PUBLIC open_htj2k PRIVATE openhtj2k_internal)

# JPIP Phase-2 VBAS codec self-test (used by tests/jpip_phase1.cmake)
add_executable(jpip_vbas_check)
add_subdirectory(tests/tools/jpip_vbas_check)
target_link_libraries(jpip_vbas_check PUBLIC open_htj2k PRIVATE openhtj2k_internal)

# JPIP Phase-2 message header codec self-test (used by tests/jpip_phase1.cmake)
add_executable(jpip_message_check)
add_subdirectory(tests/tools/jpip_message_check)
target_link_libraries(jpip_message_check PUBLIC open_htj2k PRIVATE openhtj2k_internal)

# JPIP Phase-2 header data-bin emitter self-test (used by tests/jpip_phase1.cmake)
add_executable(jpip_emitter_check)
add_subdirectory(tests/tools/jpip_emitter_check)
target_link_libraries(jpip_emitter_check PUBLIC open_htj2k PRIVATE openhtj2k_internal)

# JPIP Phase-2 JPP-stream parser self-test (used by tests/jpip_phase1.cmake)
add_executable(jpip_parser_check)
add_subdirectory(tests/tools/jpip_parser_check)
target_link_libraries(jpip_parser_check PUBLIC open_htj2k PRIVATE openhtj2k_internal)

# JPIP Phase-3 HTTP request/response formatting self-test
add_executable(jpip_http_check)
add_subdirectory(tests/tools/jpip_http_check)
target_link_libraries(jpip_http_check PUBLIC open_htj2k PRIVATE openhtj2k_internal)

# JPIP Phase-3 TCP socket wrapper self-test
add_executable(jpip_tcp_check)
add_subdirectory(tests/tools/jpip_tcp_check)
target_link_libraries(jpip_tcp_check PUBLIC open_htj2k PRIVATE openhtj2k_internal)
if(WIN32)
  target_link_libraries(jpip_tcp_check PUBLIC ws2_32)
endif()

# JPIP session/channel management self-test (§B.2, §C.3, §C.9, §D.2)
add_executable(jpip_session_check)
add_subdirectory(tests/tools/jpip_session_check)
target_link_libraries(jpip_session_check PUBLIC open_htj2k PRIVATE openhtj2k_internal)

# JPIP issue #297 chunked-transfer round-trip self-test
add_executable(jpip_chunked_check)
add_subdirectory(tests/tools/jpip_chunked_check)
target_link_libraries(jpip_chunked_check PUBLIC open_htj2k PRIVATE openhtj2k_internal)
if(WIN32)
  target_link_libraries(jpip_chunked_check PUBLIC ws2_32)
endif()

# JPIP issue #297 StreamingJppParser split-at-every-offset self-test
add_executable(jpip_streaming_parser_check)
add_subdirectory(tests/tools/jpip_streaming_parser_check)
target_link_libraries(jpip_streaming_parser_check PUBLIC open_htj2k PRIVATE openhtj2k_internal)

# JPIP Phase-2 precinct data-bin + packet-locator self-test
add_executable(jpip_precinct_check)
add_subdirectory(tests/tools/jpip_precinct_check)
target_link_libraries(jpip_precinct_check PUBLIC open_htj2k PRIVATE openhtj2k_internal)

# JPIP Phase-2 full JPP-stream round-trip (emit → parse → reassemble → decode)
add_executable(jpip_assembler_check)
add_subdirectory(tests/tools/jpip_assembler_check)
target_link_libraries(jpip_assembler_check PUBLIC open_htj2k PRIVATE openhtj2k_internal)
