# Rows of the streaming decoders for subsampled components (decoder.hpp,
# invoke_line_based_stream).  The fixtures in tests/data/stream_rows were made with
# this library's encoder (buffered invoke_line_based) from synthetic images; the
# reference PGX files are the source images, which every fixture decodes to exactly
# (also with two other decoders).  They cover what the streaming path got wrong:
#   yr_mix                vertical factors 1, 3, 4, 2, 5 in one tile (strips of 64 rows
#                         that are not a multiple of 3 lost rows of the factor-3 components)
#   tiles_origin          factors 1, 2, 3; image origin y = 1; 31x15 tiles in three columns
#   tile_column           factors 1, 2, 3, 5; origin x = 1; one column of 7-row tiles, so
#                         component rows span tile boundaries
#   y0_coarse             factors 3 and 6, origin (1, 1): component 0 is subsampled too
#   not_multiples         factors 2, 3, 1 (not multiples of component 0's): more output rows
#                         than component 0 has; 23x11 tiles
#   not_multiples_column  factors 3, 2 in 7-row tiles, irreversible 9/7
#   mct_extra             colour transform on components 0-2 (factor 2) and a fourth
#                         component with factor 1, in 9-row tiles
# Each one is decoded with the command-line decoder (invoke_line_based_stream) with 1 and
# 4 threads and compared with imgcmp, and checked with stream_rows_check, which compares
# every streaming entry point (also with row and column windows, the single-tile reuse
# path and the direct-to-planar path) with the buffered invoke_line_based().
set(_SR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/stream_rows")
set(_SR_CRASH_RE
    "AddressSanitizer|UndefinedBehaviorSanitizer|runtime error|Segmentation fault|SIGSEGV|SIGABRT|core dumped|terminate called")
foreach(_case "yr_mix;5" "tiles_origin;3" "tile_column;4" "y0_coarse;2" "not_multiples;3"
              "not_multiples_column;2" "mct_extra;4")
  list(GET _case 0 _name)
  list(GET _case 1 _nc)
  math(EXPR _last "${_nc} - 1")
  foreach(_t 1 4)
    add_test(NAME dec_stream_rows_${_name}_t${_t}
             COMMAND open_htj2k_dec -i ${_SR_DIR}/${_name}.j2c -o stream_rows_${_name}_t${_t}.pgx
                     -num_threads ${_t})
    foreach(_c RANGE ${_last})
      string(LENGTH "${_c}" _len)
      set(_cc "${_c}")
      if(_len EQUAL 1)
        set(_cc "0${_c}")
      endif()
      add_test(NAME comp_stream_rows_${_name}_t${_t}_c${_c}
               COMMAND imgcmp stream_rows_${_name}_t${_t}_${_cc}.pgx ${_SR_DIR}/${_name}_${_cc}.pgx 0 0)
      set_tests_properties(comp_stream_rows_${_name}_t${_t}_c${_c} PROPERTIES
                           DEPENDS dec_stream_rows_${_name}_t${_t})
    endforeach()
    add_test(NAME stream_rows_${_name}_t${_t} COMMAND stream_rows_check ${_SR_DIR}/${_name}.j2c -threads ${_t})
    set_tests_properties(stream_rows_${_name}_t${_t} PROPERTIES
                         PASS_REGULAR_EXPRESSION "PASS" FAIL_REGULAR_EXPRESSION "FAIL|${_SR_CRASH_RE}" TIMEOUT 60)
  endforeach()
  add_test(NAME stream_rows_${_name}_reduce1 COMMAND stream_rows_check ${_SR_DIR}/${_name}.j2c -reduce 1)
  set_tests_properties(stream_rows_${_name}_reduce1 PROPERTIES
                       PASS_REGULAR_EXPRESSION "PASS" FAIL_REGULAR_EXPRESSION "FAIL|${_SR_CRASH_RE}" TIMEOUT 60)
endforeach()

