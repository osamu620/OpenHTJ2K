# The library writes nothing to stdout or stderr itself: warnings go to the handler installed with
# open_htj2k::set_message_handler() and errors are exceptions whose what() describes them.
# library_messages_silent fails on any output; library_messages_handler checks that the handler
# receives the encoder's warning.

set(_MSG_INPUT ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/security_marker_cod_levels.j2k)
add_test(NAME library_messages_silent COMMAND message_handler_check silent ${_MSG_INPUT})
set_tests_properties(library_messages_silent PROPERTIES FAIL_REGULAR_EXPRESSION "." TIMEOUT 30)
add_test(NAME library_messages_handler COMMAND message_handler_check handler ${_MSG_INPUT})
set_tests_properties(library_messages_handler PROPERTIES
    PASS_REGULAR_EXPRESSION "warnings=[1-9][0-9]* last=\"Cycc is set to 'no'.*error=\"COD number of decomposition levels 255 exceeds 32"
    TIMEOUT 30)

# A JPH or JP2 file of more than 4 components: the file writer said only "invalid color
# space specification"; the error now says what it takes.
set(_MSG_PGX ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/mixed_u8.pgx)
add_test(NAME library_messages_jph_components
         COMMAND open_htj2k_enc -i ${_MSG_PGX},${_MSG_PGX},${_MSG_PGX},${_MSG_PGX},${_MSG_PGX}
                 -o library_messages_5c.jph)
set_tests_properties(library_messages_jph_components PROPERTIES
    PASS_REGULAR_EXPRESSION "takes 1 to 4 components .* but the image has 5" TIMEOUT 30)
