# Codestreams cut short (a truncated file or download; decoder.hpp, is_truncated()).
# truncation_check decodes the first L bytes of a codestream, for each length L past the
# marker code of its first SOT marker segment (or each S-th), through every decoder entry
# point: each must decode without an error, all must give the same samples and report the
# cut with is_truncated(), and a codestream that lacks only its EOC marker must decode like
# the whole one.  Most cuts used to fail ("HT code-block decoding reported failure", "MQ
# segment length exceeds codeblock bytes"): a code-block whose data the cut packet held in
# part was decoded from that part, a code-block left with HT placeholder passes alone was
# refused, and a cut inside a later tile-part header, an SOP marker segment or an EPH marker
# was taken for a malformed one.
#   marker_ppm_tile_parts   packet headers in PPM markers; 2 x 2 tiles of 3 tile-parts
#   marker_ppt_tile_parts   the same in PPT markers, Part 1 block coder
#   marker_poc_tile_parts   POC markers in tile-part headers, EPH markers
#   marker_sop_optional     SOP marker segments on every other packet
#   ht_refine_tall_cblk     HT SigProp and MagRef passes
#   ds1_ht_01_b11           HT in 5 quality layers (placeholder passes), SOP and EPH
#   p0_09, p1_07            Part 1
# and, encoded here, Part 1 with the BYPASS and RESTART mode switches in 3 quality layers
# (several codeword segments per code-block), and HT in 4 tiles.
set(TRUNC_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tests/data")

foreach(_name marker_ppm_tile_parts marker_ppt_tile_parts)
  add_test(NAME truncation_${_name} COMMAND truncation_check ${TRUNC_DATA_DIR}/${_name}.j2c)
endforeach()
add_test(NAME truncation_marker_poc_tile_parts
         COMMAND truncation_check ${TRUNC_DATA_DIR}/marker_poc_tile_parts.j2c -step 2)
add_test(NAME truncation_marker_sop_optional
         COMMAND truncation_check ${TRUNC_DATA_DIR}/marker_sop_optional.j2c -step 9)
add_test(NAME truncation_ht_refine_tall_cblk
         COMMAND truncation_check ${TRUNC_DATA_DIR}/ht_refine_tall_cblk.j2k)
add_test(NAME truncation_ds1_ht_01_b11 COMMAND truncation_check ${CONFORMANCE_DATA_DIR}/ds1_ht_01_b11.j2k)
add_test(NAME truncation_ds1_ht_01_b11_t4
         COMMAND truncation_check ${CONFORMANCE_DATA_DIR}/ds1_ht_01_b11.j2k -threads 4)
foreach(_name p0_09 p1_07)
  add_test(NAME truncation_${_name} COMMAND truncation_check ${CONFORMANCE_DATA_DIR}/${_name}.j2k)
endforeach()

add_test(NAME enc_truncation_p1_layers
         COMMAND open_htj2k_enc -i ${TRUNC_DATA_DIR}/rgb16_full_range.ppm -o truncation_p1_layers.j2c
                 "Cmodes=BYPASS|RESTART" -rate 1,2,4)
add_test(NAME truncation_p1_layers COMMAND truncation_check truncation_p1_layers.j2c -step 3)
set_tests_properties(truncation_p1_layers PROPERTIES DEPENDS enc_truncation_p1_layers)
add_test(NAME enc_truncation_ht_tiles
         COMMAND open_htj2k_enc -i ${TRUNC_DATA_DIR}/rgb16_full_range.ppm -o truncation_ht_tiles.j2c
                 "Stiles={32,32}" Qfactor=90)
add_test(NAME truncation_ht_tiles COMMAND truncation_check truncation_ht_tiles.j2c -step 3)
set_tests_properties(truncation_ht_tiles PROPERTIES DEPENDS enc_truncation_ht_tiles)
