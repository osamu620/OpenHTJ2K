# Valid codestreams whose marker segments the decoder once read against the standard.
#
# The fixtures in tests/data/marker_*.j2c were made from our encoder's codestreams of
# tests/data/dwt_levels_src.pgm (61 x 47, 8 bits) and tests/data/rgb16_full_range.ppm (67 x 37,
# 16 bits), coded with SOP and EPH so that each packet can be located and moved, then rewritten:
#
#   marker_ppm_tile_parts       2 x 2 tiles of 3 tile-parts each, interleaved (tile-part 0 of
#                               every tile, then tile-part 1, ...), packet headers in PPM markers
#                               that appear in decreasing Zppm.  One PPM record per tile-part,
#                               in codestream order (ISO/IEC 15444-1, A.7.4).
#   marker_ppt_tile_parts       the same with the Part 1 block coder and PPT markers, in
#                               decreasing Zppt; the first tile-part of each tile has no packet
#                               and no PPT marker (A.7.5).
#   marker_sop_optional         SOP marker segments allowed (Scod) but on every other packet
#                               only (A.8.1).
#
# Each decodes to its source image (the irreversible ones within a tolerance), as another
# decoder decodes it; the decoder of 0.24.0 failed on or misdecoded every one.

set(MARKER_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tests/data")
set(_MARKER_GRAY "${MARKER_DATA_DIR}/dwt_levels_src.pgm")
set(_MARKER_RGB "${MARKER_DATA_DIR}/rgb16_full_range.ppm")

# name, fixture, source image, PAE and MSE thresholds
function(_marker_decode_test name fixture source pae mse)
  get_filename_component(_ext "${source}" EXT)
  foreach(_t 1 4)
    if(_ext STREQUAL ".ppm")
      set(_out ${name}_t${_t}.ppm)
      set(_cmp ${_out})
    else()
      set(_out ${name}_t${_t}.pgx)
      set(_cmp ${name}_t${_t}_00.pgx)
    endif()
    add_test(NAME dec_${name}_t${_t}
             COMMAND open_htj2k_dec -i ${MARKER_DATA_DIR}/${fixture} -o ${_out} -num_threads ${_t})
    add_test(NAME comp_${name}_t${_t} COMMAND imgcmp ${_cmp} ${source} ${pae} ${mse})
    set_tests_properties(comp_${name}_t${_t} PROPERTIES DEPENDS dec_${name}_t${_t})
  endforeach()
endfunction()

_marker_decode_test(marker_ppm_tile_parts marker_ppm_tile_parts.j2c ${_MARKER_GRAY} 0 0)
_marker_decode_test(marker_ppt_tile_parts marker_ppt_tile_parts.j2c ${_MARKER_GRAY} 0 0)
_marker_decode_test(marker_sop_optional marker_sop_optional.j2c ${_MARKER_RGB} 0 0)
