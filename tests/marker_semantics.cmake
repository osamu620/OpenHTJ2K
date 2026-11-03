# Valid codestreams whose marker segments the decoder once read against the standard, and
# encoder output whose marker segments other decoders read.
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
#   marker_coc_tile_cod         a main-header COC for component 0 and a tile-part COD in tile 1:
#                               tile-part COD > main COC (A.6.1).
#   marker_qcc_tile_qcd         a main-header QCC for component 0 and a tile-part QCD in tile 1:
#                               tile-part QCD > main QCC (A.6.4).  Irreversible, so compared
#                               within a tolerance; the main QCC's step sizes give 14 times
#                               the squared error.
#   marker_coc_other_component  main COC and QCC for component 0, tile-part COC and QCC for
#                               component 1 only: component 0 keeps the main-header ones.
#   marker_poc_all_components   COD says RPCL; a main POC with CEpoc = 0 (all components, Table
#                               A.32) gives the LRCP order the packets are in.  Irreversible.
#   marker_poc_tile_parts       POC of tile-part 0: LRCP for layer 0; POC of tile-part 1: CPRL for
#                               layer 1 (A.6.6).  Irreversible.
#   marker_htdeclared           HT code-blocks in an HTDECLARED codestream (Ccap15 bits 15:14 =
#                               10): no tile-component mixes HT and Part 1 code-blocks (ISO/IEC
#                               15444-15, A.3.2), so the rules that tell the two kinds apart do
#                               not apply.  Decoded silently wrong before.
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
_marker_decode_test(marker_coc_tile_cod marker_coc_tile_cod.j2c ${_MARKER_GRAY} 0 0)
_marker_decode_test(marker_qcc_tile_qcd marker_qcc_tile_qcd.j2c ${_MARKER_GRAY} 24 20)
_marker_decode_test(marker_coc_other_component marker_coc_other_component.j2c ${_MARKER_RGB} 0 0)
_marker_decode_test(marker_poc_all_components marker_poc_all_components.j2c ${_MARKER_RGB} 65535 8000000)
_marker_decode_test(marker_poc_tile_parts marker_poc_tile_parts.j2c ${_MARKER_RGB} 65535 8000000)
_marker_decode_test(marker_htdeclared marker_htdeclared.j2c ${_MARKER_GRAY} 0 0)

# More than 10921 tiles: the TLM marker, 6 bytes per tile, takes more than one marker segment
# (Ltlm is 16 bits).  It wrapped, and no decoder could read the codestream.
add_test(NAME enc_tlm_many_tiles
         COMMAND open_htj2k_enc -i ${CMAKE_CURRENT_SOURCE_DIR}/conformance_data/references/kodim23.ppm
                 -o tlm_many_tiles.j2c "Stiles={6,6}" Clevels=1 Creversible=yes)
add_test(NAME dec_tlm_many_tiles COMMAND open_htj2k_dec -i tlm_many_tiles.j2c -o tlm_many_tiles.ppm)
set_tests_properties(dec_tlm_many_tiles PROPERTIES DEPENDS enc_tlm_many_tiles)
add_test(NAME comp_tlm_many_tiles
         COMMAND imgcmp tlm_many_tiles.ppm ${CMAKE_CURRENT_SOURCE_DIR}/conformance_data/references/kodim23.ppm 0 0)
set_tests_properties(comp_tlm_many_tiles PROPERTIES DEPENDS dec_tlm_many_tiles)

# The magnitude bound B in bits 0-4 of Ccap15 (CAP marker, ISO/IEC 15444-15, A.3.7 and 8.7.3):
# the cleanup-pass magnitude bit-planes of the coded HT code-blocks, less n_b - 1 for an
# irreversible sub-band at decomposition level n_b, rounded up to a value P can signal.
# Before, reversible streams with more than one guard bit and irreversible ones signalled too
# small a B (the guard bits were left out, and irreversible took the smallest exponent), and a
# bound of 28 to 30 came out as 27.  The CAP marker is FF50 0008 00020000 (Pcap15) followed by
# Ccap15.
function(_cap_magb_test name ccap15)
  add_test(NAME enc_cap_magb_${name} COMMAND open_htj2k_enc -i ${_MARKER_GRAY} -o cap_magb_${name}.j2c ${ARGN})
  add_test(NAME check_cap_magb_${name}
           COMMAND ${CMAKE_COMMAND} -DFILE=cap_magb_${name}.j2c "-DHEX=ff50000800020000${ccap15}"
                   -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_file_hex.cmake)
  set_tests_properties(check_cap_magb_${name} PROPERTIES DEPENDS enc_cap_magb_${name})
endfunction()
_cap_magb_test(rev_g4 0003 Creversible=yes Qguard=4)   # B = 11 (was 8)
_cap_magb_test(irv_tiny 0034 Qstep=0.000000001)        # B = 31 for 30 (was 27); bit 5: HTIRV
_cap_magb_test(irv_levels0 0021 Clevels=0)             # B = 9 (was 8): LL at n_b = 0
