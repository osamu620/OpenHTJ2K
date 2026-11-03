# Single-tile reuse across frames whose coding parameters differ (decoder.hpp,
# enable_single_tile_reuse).  reuse_switch_check edits each codestream in memory into
# variants that decode to other samples (step sizes in the main-header QCD or QCC, a QCD
# or QCC added to the tile-part header, one more resolution reduction), decodes input,
# variant, input on one decoder with reuse enabled through the streaming and the direct
# entry points, and compares every frame with a fresh decoder's.  The reused structure
# used to be kept for such frames, decoding them with the previous frame's step sizes.
#   ds0_ht_04_b11  HT, three components, main-header QCCs
#   ds0_ht_06_b11  HT, four subsampled components, COC, QCCs and RGN
#   hifi_ht1_02    HT, three components, no QCC (the tile-part QCD variant)
#   p0_04          Part 1, main-header QCCs
#   hifi_p1_02     Part 1, no QCC
foreach(_name ds0_ht_04_b11 ds0_ht_06_b11 hifi_ht1_02 p0_04 hifi_p1_02)
  add_test(NAME reuse_switch_${_name} COMMAND reuse_switch_check ${CONFORMANCE_DATA_DIR}/${_name}.j2k)
endforeach()
add_test(NAME reuse_switch_ds0_ht_04_b11_t4
         COMMAND reuse_switch_check ${CONFORMANCE_DATA_DIR}/ds0_ht_04_b11.j2k -threads 4)
