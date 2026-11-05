# Conformance testing by Ctest
enable_testing()
set(ENCODER_REF_DIR "${CMAKE_CURRENT_SOURCE_DIR}/conformance_data/references")

add_test(NAME enc_lossless COMMAND open_htj2k_enc -i ${ENCODER_REF_DIR}/kodim23.ppm -o kodim23lossless.j2c Creversible=yes)
add_test(NAME dec_lossless COMMAND open_htj2k_dec -i kodim23lossless.j2c -o kodim23lossless.ppm)
set_tests_properties(dec_lossless PROPERTIES DEPENDS enc_lossless)
add_test(NAME comp_lossless COMMAND imgcmp kodim23lossless.ppm ${ENCODER_REF_DIR}/kodim23.ppm 0 0)
set_tests_properties(comp_lossless PROPERTIES DEPENDS dec_lossless)

add_test(NAME enc_lossy COMMAND open_htj2k_enc -i ${ENCODER_REF_DIR}/kodim23.ppm -o kodim23lossy.j2c Qfactor=90)
add_test(NAME dec_lossy COMMAND open_htj2k_dec -i kodim23lossy.j2c -o kodim23lossy.ppm)
set_tests_properties(dec_lossy PROPERTIES DEPENDS enc_lossy)
add_test(NAME comp_lossy COMMAND imgcmp kodim23lossy.ppm ${ENCODER_REF_DIR}/kodim23.ppm 23 6)
set_tests_properties(comp_lossy PROPERTIES DEPENDS dec_lossy)

add_test(NAME enc_lossless_odd COMMAND open_htj2k_enc -i ${ENCODER_REF_DIR}/kodim23odd.ppm -o kodim23odd_lossless.j2c Creversible=yes)
add_test(NAME dec_lossless_odd COMMAND open_htj2k_dec -i kodim23odd_lossless.j2c -o kodim23odd_lossless.ppm)
set_tests_properties(dec_lossless_odd PROPERTIES DEPENDS enc_lossless_odd)
add_test(NAME comp_lossless_odd COMMAND imgcmp kodim23odd_lossless.ppm ${ENCODER_REF_DIR}/kodim23odd.ppm 0 0)
set_tests_properties(comp_lossless_odd PROPERTIES DEPENDS dec_lossless_odd)

# ISO/IEC 15444-2 (Part 2) PRCL progression order. kodim23 is RGB with 5 DWT levels (3 components
# x 6 resolutions), so the PRCL resolution-then-component packet interleaving differs from every
# Part 1 order. The lossless round-trip must reproduce the input exactly (PAE=0), which proves the
# encoder write order and decoder read order agree and map packets to the correct subbands.
add_test(NAME enc_prcl COMMAND open_htj2k_enc -i ${ENCODER_REF_DIR}/kodim23.ppm -o kodim23prcl.j2c Creversible=yes Corder=PRCL)
add_test(NAME dec_prcl COMMAND open_htj2k_dec -i kodim23prcl.j2c -o kodim23prcl.ppm)
set_tests_properties(dec_prcl PROPERTIES DEPENDS enc_prcl)
add_test(NAME comp_prcl COMMAND imgcmp kodim23prcl.ppm ${ENCODER_REF_DIR}/kodim23.ppm 0 0)
set_tests_properties(comp_prcl PROPERTIES DEPENDS dec_prcl)

# Regression: explicit precincts that GROUP codeblocks (precinct >= codeblock).
# The streaming encoder used to build per-strip codeblock tables assuming a
# single precinct per subband and segfaulted on any multi-precinct config; it
# now stamps codeblocks on the full band-raster grid. Validate a lossless
# round-trip for a Part 1 order with several precinct sizes per level, each
# holding multiple codeblocks.
add_test(NAME enc_precincts COMMAND open_htj2k_enc -i ${ENCODER_REF_DIR}/kodim23.ppm -o kodim23prec.j2c Creversible=yes Corder=RPCL "Cprecincts={256,256}{128,128}")
add_test(NAME dec_precincts COMMAND open_htj2k_dec -i kodim23prec.j2c -o kodim23prec.ppm)
set_tests_properties(dec_precincts PROPERTIES DEPENDS enc_precincts)
add_test(NAME comp_precincts COMMAND imgcmp kodim23prec.ppm ${ENCODER_REF_DIR}/kodim23.ppm 0 0)
set_tests_properties(comp_precincts PROPERTIES DEPENDS dec_precincts)

# Small precincts are supported when paired with a code-block that fits inside
# them (precinct >= codeblock): many precincts, one codeblock each.
add_test(NAME enc_precincts_small COMMAND open_htj2k_enc -i ${ENCODER_REF_DIR}/kodim23.ppm -o kodim23precsm.j2c Creversible=yes Corder=PCRL "Cblk={32,32}" "Cprecincts={128,128}{64,64}")
add_test(NAME dec_precincts_small COMMAND open_htj2k_dec -i kodim23precsm.j2c -o kodim23precsm.ppm)
set_tests_properties(dec_precincts_small PROPERTIES DEPENDS enc_precincts_small)
add_test(NAME comp_precincts_small COMMAND imgcmp kodim23precsm.ppm ${ENCODER_REF_DIR}/kodim23.ppm 0 0)
set_tests_properties(comp_precincts_small PROPERTIES DEPENDS dec_precincts_small)

# Precinct SMALLER than the code-block: Rec. ITU-T T.800 B.7 clamps the
# code-block to the precinct-subband (ycb' = min(ycb, PPy-1)), giving the
# single-code-block-per-precinct layout used by the ISO/IEC 15444-18 profiles.
# The streaming encoder strips at the effective (clamped) code-block height, so
# this round-trips losslessly. Two precinct sizes per level exercise both the
# vertical-clamp (CBH/2) and finer-clamp levels.
add_test(NAME enc_precincts_clamped COMMAND open_htj2k_enc -i ${ENCODER_REF_DIR}/kodim23.ppm -o kodim23clamp.j2c Creversible=yes Corder=PCRL "Cprecincts={64,64}{32,32}")
add_test(NAME dec_precincts_clamped COMMAND open_htj2k_dec -i kodim23clamp.j2c -o kodim23clamp.ppm)
set_tests_properties(dec_precincts_clamped PROPERTIES DEPENDS enc_precincts_clamped)
add_test(NAME comp_precincts_clamped COMMAND imgcmp kodim23clamp.ppm ${ENCODER_REF_DIR}/kodim23.ppm 0 0)
set_tests_properties(comp_precincts_clamped PROPERTIES DEPENDS dec_precincts_clamped)

# PRCL with explicit precincts: now that multi-precinct encoding works, this
# exercises PRCL's multi-position packet ordering (the y/x examination loops and
# per-(component,resolution) precinct advancement) end to end.
add_test(NAME enc_prcl_prec COMMAND open_htj2k_enc -i ${ENCODER_REF_DIR}/kodim23.ppm -o kodim23prclprec.j2c Creversible=yes Corder=PRCL "Cprecincts={128,128}")
add_test(NAME dec_prcl_prec COMMAND open_htj2k_dec -i kodim23prclprec.j2c -o kodim23prclprec.ppm)
set_tests_properties(dec_prcl_prec PROPERTIES DEPENDS enc_prcl_prec)
add_test(NAME comp_prcl_prec COMMAND imgcmp kodim23prclprec.ppm ${ENCODER_REF_DIR}/kodim23.ppm 0 0)
set_tests_properties(comp_prcl_prec PROPERTIES DEPENDS dec_prcl_prec)

# ISO/IEC 15444-18 (Profiles) round-trip on a real Part 18 codestream.
# conformance_data/4KuF_H0L1_arri_1bpp.j2c is a 3968x2160 ARRI codestream in
# profile H0 / Level 1: PRCL progression (signalled via CAP Ccap2 bit-14),
# an ATK wavelet kernel, and one code-block per precinct. Decoding it exercises
# the Part 18 decode path (PRCL + ATK + clamped precincts). Re-encoding the
# decoded image losslessly with PRCL and a single-code-block-per-precinct layout
# ({64,64} precincts < 64x64 code-block) and decoding again must reproduce it
# exactly (PAE=0) -- exercising the PRCL progression order (Part 2) and the
# clamped streaming-encode path on real 4K Part 18 content.
set(P18_J2C ${CMAKE_CURRENT_SOURCE_DIR}/conformance_data/4KuF_H0L1_arri_1bpp.j2c)
add_test(NAME dec_p18_arri COMMAND open_htj2k_dec -i ${P18_J2C} -o arri_p18.ppm)
add_test(NAME enc_p18_rt COMMAND open_htj2k_enc -i arri_p18.ppm -o arri_p18_rt.j2c Creversible=yes Corder=PRCL "Cprecincts={64,64}")
set_tests_properties(enc_p18_rt PROPERTIES DEPENDS dec_p18_arri)
add_test(NAME dec_p18_rt COMMAND open_htj2k_dec -i arri_p18_rt.j2c -o arri_p18_rt.ppm)
set_tests_properties(dec_p18_rt PROPERTIES DEPENDS enc_p18_rt)
add_test(NAME comp_p18_rt COMMAND imgcmp arri_p18_rt.ppm arri_p18.ppm 0 0)
set_tests_properties(comp_p18_rt PROPERTIES DEPENDS dec_p18_rt)

# 16-bit samples >= 32768 must survive the decoder's PGM and PGX writers.
# tests/data/gray16_full_range.pgm is 67x37, 16-bit, with about half its samples
# >= 32768 (full-range ramps plus 0 and 65535); width 67 exercises both the
# vectorized row packing and its scalar tail.  A lossless round trip written as
# PGM (big-endian) and as PGX (little-endian) must reproduce it exactly.
set(GRAY16_PGM ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/gray16_full_range.pgm)
add_test(NAME enc_gray16_rt COMMAND open_htj2k_enc -i ${GRAY16_PGM} -o gray16_rt.j2c Creversible=yes)
add_test(NAME dec_gray16_rt_pgm COMMAND open_htj2k_dec -i gray16_rt.j2c -o gray16_rt.pgm)
set_tests_properties(dec_gray16_rt_pgm PROPERTIES DEPENDS enc_gray16_rt)
add_test(NAME comp_gray16_rt_pgm COMMAND imgcmp gray16_rt_00.pgm ${GRAY16_PGM} 0 0)
set_tests_properties(comp_gray16_rt_pgm PROPERTIES DEPENDS dec_gray16_rt_pgm)
add_test(NAME dec_gray16_rt_pgx COMMAND open_htj2k_dec -i gray16_rt.j2c -o gray16_rt.pgx)
set_tests_properties(dec_gray16_rt_pgx PROPERTIES DEPENDS enc_gray16_rt)
add_test(NAME comp_gray16_rt_pgx COMMAND imgcmp gray16_rt_00.pgx ${GRAY16_PGM} 0 0)
set_tests_properties(comp_gray16_rt_pgx PROPERTIES DEPENDS dec_gray16_rt_pgx)

# Components of different bit depth and sign: the decoder writes every PGX, PGM and RAW
# file with its own component's depth, sign and bytes per sample (it used component 0's
# bytes per sample, and for PGM its sign offset, for all of them).  tests/data/mixed_*.pgx
# are 67x45 planes (odd width: vectorized row packing plus scalar tail) holding the
# extremes of their range: unsigned 12-bit, signed 12-bit, unsigned 8-bit, signed 4-bit.
# mixed_us: unsigned 12, signed 4, unsigned 8; mixed_su: signed 12, unsigned 8, signed 4.
# Component 0 is the deepest here; the tests after this block put it first.
# A lossless round trip must reproduce every component as PGX.  As PGM, unsigned
# components equal their input and signed ones (offset by half their range) equal the
# PGM of the same plane encoded alone.  RAW files take 2 bytes per sample at 12 bits and
# 1 at 8 and 4 bits.  A PPM file has one maximum value for its three components: their
# depths differing is an error.
set(MIXED_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests/data)
foreach(_one s4 s12)
  add_test(NAME enc_mixed_${_one} COMMAND open_htj2k_enc -i ${MIXED_DIR}/mixed_${_one}.pgx -o mixed_${_one}.j2c
           Creversible=yes)
  add_test(NAME dec_mixed_${_one}_pgm COMMAND open_htj2k_dec -i mixed_${_one}.j2c -o mixed_${_one}.pgm)
  set_tests_properties(dec_mixed_${_one}_pgm PROPERTIES DEPENDS enc_mixed_${_one})
endforeach()
foreach(_case "us;u12;s4;u8" "su;s12;u8;s4")
  list(GET _case 0 _t)
  list(REMOVE_AT _case 0)
  set(_in "")
  foreach(_p IN LISTS _case)
    list(APPEND _in ${MIXED_DIR}/mixed_${_p}.pgx)
  endforeach()
  string(REPLACE ";" "," _in "${_in}")
  add_test(NAME enc_mixed_${_t} COMMAND open_htj2k_enc -i ${_in} -o mixed_${_t}.j2c Creversible=yes Cycc=no)
  foreach(_fmt pgx pgm raw)
    add_test(NAME dec_mixed_${_t}_${_fmt} COMMAND open_htj2k_dec -i mixed_${_t}.j2c -o mixed_${_t}.${_fmt})
    set_tests_properties(dec_mixed_${_t}_${_fmt} PROPERTIES DEPENDS enc_mixed_${_t})
  endforeach()
  set(_c 0)
  foreach(_p IN LISTS _case)
    add_test(NAME comp_mixed_${_t}_${_c}_pgx COMMAND imgcmp mixed_${_t}_0${_c}.pgx ${MIXED_DIR}/mixed_${_p}.pgx 0 0)
    set_tests_properties(comp_mixed_${_t}_${_c}_pgx PROPERTIES DEPENDS dec_mixed_${_t}_pgx)
    if(_p MATCHES "^s")
      add_test(NAME comp_mixed_${_t}_${_c}_pgm COMMAND imgcmp mixed_${_t}_0${_c}.pgm mixed_${_p}_00.pgm 0 0)
      set_tests_properties(comp_mixed_${_t}_${_c}_pgm PROPERTIES DEPENDS "dec_mixed_${_t}_pgm;dec_mixed_${_p}_pgm")
    else()
      add_test(NAME comp_mixed_${_t}_${_c}_pgm COMMAND imgcmp mixed_${_t}_0${_c}.pgm ${MIXED_DIR}/mixed_${_p}.pgx 0 0)
      set_tests_properties(comp_mixed_${_t}_${_c}_pgm PROPERTIES DEPENDS dec_mixed_${_t}_pgm)
    endif()
    if(_p MATCHES "12$")
      set(_bytes 6030)
    else()
      set(_bytes 3015)
    endif()
    add_test(NAME size_mixed_${_t}_${_c}_raw COMMAND ${CMAKE_COMMAND} -DFILE=mixed_${_t}_0${_c}.raw -DMAX=${_bytes}
             -DMIN=${_bytes} -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_file_size.cmake)
    set_tests_properties(size_mixed_${_t}_${_c}_raw PROPERTIES DEPENDS dec_mixed_${_t}_raw)
    math(EXPR _c "${_c} + 1")
  endforeach()
endforeach()
# At a reduced resolution too: the signed 4-bit component as decoded with the others and
# alone.
add_test(NAME dec_mixed_us_reduce COMMAND open_htj2k_dec -i mixed_us.j2c -o mixed_us_r1.pgx -reduce 1)
set_tests_properties(dec_mixed_us_reduce PROPERTIES DEPENDS enc_mixed_us)
add_test(NAME dec_mixed_s4_reduce COMMAND open_htj2k_dec -i mixed_s4.j2c -o mixed_s4_r1.pgx -reduce 1)
set_tests_properties(dec_mixed_s4_reduce PROPERTIES DEPENDS enc_mixed_s4)
add_test(NAME comp_mixed_us_reduce COMMAND imgcmp mixed_us_r1_01.pgx mixed_s4_r1_00.pgx 0 0)
set_tests_properties(comp_mixed_us_reduce PROPERTIES DEPENDS "dec_mixed_us_reduce;dec_mixed_s4_reduce")
add_test(NAME dec_mixed_su_ppm_rejected COMMAND open_htj2k_dec -i mixed_su.j2c -o mixed_su.ppm)
set_tests_properties(dec_mixed_su_ppm_rejected PROPERTIES DEPENDS enc_mixed_su
                     PASS_REGULAR_EXPRESSION "a PPM file needs three components of the same bit depth")

# Component 0 shallower than the others.  The QCD marker follows component 0's bit depth,
# so the encoder writes a QCC marker for every component of another depth (without it a
# deeper component lost its top bit-planes and a lossless encode was not lossless), and it
# turns the colour transform off when the three components differ in depth or sign (here
# without Cycc=no).  Lossless round trips with the HT and the Part 1 block coder.
foreach(_case "u8u12;u8;u12" "u8s12;u8;s12" "s4u12u8;s4;u12;u8" "u8u12s12;u8;u12;s12")
  list(GET _case 0 _t)
  list(REMOVE_AT _case 0)
  set(_in "")
  foreach(_p IN LISTS _case)
    list(APPEND _in ${MIXED_DIR}/mixed_${_p}.pgx)
  endforeach()
  string(REPLACE ";" "," _in "${_in}")
  foreach(_coder ht p1)
    if(_coder STREQUAL "p1")
      set(_modes Cmodes=0)
    else()
      set(_modes Cmodes=HT)
    endif()
    set(_n qcc_${_t}_${_coder})
    add_test(NAME enc_${_n} COMMAND open_htj2k_enc -i ${_in} -o ${_n}.j2c Creversible=yes ${_modes})
    add_test(NAME dec_${_n} COMMAND open_htj2k_dec -i ${_n}.j2c -o ${_n}.pgx)
    set_tests_properties(dec_${_n} PROPERTIES DEPENDS enc_${_n})
    set(_c 0)
    foreach(_p IN LISTS _case)
      add_test(NAME comp_${_n}_${_c} COMMAND imgcmp ${_n}_0${_c}.pgx ${MIXED_DIR}/mixed_${_p}.pgx 0 0)
      set_tests_properties(comp_${_n}_${_c} PROPERTIES DEPENDS dec_${_n})
      math(EXPR _c "${_c} + 1")
    endforeach()
  endforeach()
endforeach()

# Lossy decoding of more than 13-bit samples scales them up (downshift =
# FRACBITS - bit depth < 0) in every finalize path.  Decode a 16-bit Qfactor 90
# codestream through the command-line decoder and the row- and column-window
# APIs; the UBSan CI job checks those shifts.  The error bound is loose on
# purpose (measured: PAE 2458, MSE 3.7e5 on x86-64): it catches broken output,
# not floating-point differences between builds.
add_test(NAME enc_gray16_q90 COMMAND open_htj2k_enc -i ${GRAY16_PGM} -o gray16_q90.j2c Qfactor=90)
add_test(NAME dec_gray16_q90 COMMAND open_htj2k_dec -i gray16_q90.j2c -o gray16_q90.pgm)
set_tests_properties(dec_gray16_q90 PROPERTIES DEPENDS enc_gray16_q90)
add_test(NAME comp_gray16_q90 COMMAND imgcmp gray16_q90_00.pgm ${GRAY16_PGM} 4096 600000)
set_tests_properties(comp_gray16_q90 PROPERTIES DEPENDS dec_gray16_q90)
add_test(NAME rr_gray16_q90 COMMAND row_range_compare gray16_q90.j2c)
set_tests_properties(rr_gray16_q90 PROPERTIES DEPENDS enc_gray16_q90)
add_test(NAME cr_gray16_q90 COMMAND col_range_compare gray16_q90.j2c)
set_tests_properties(cr_gray16_q90 PROPERTIES DEPENDS enc_gray16_q90)

# Lossy precision above and below 13 bits.  The decoder rounds each float sample
# half up to the nearest output LSB, round(v * 2^-ds); it used to truncate toward
# zero to the 13-bit grid first, which left samples below mid-grey 1 LSB too
# bright (half of them at 12 bits) and capped output above 13 bits at 13-bit
# precision.  The encoder used to right-shift input above 13 bits in integer
# arithmetic before the transform, dropping its low bits as well.  A 12-bit
# round trip at a fine step is now exact (PAE 1, MSE 0.24 before); a 16-bit grey
# round trip has MSE 0.005 (PAE 15, MSE 56 before both fixes; PAE 8, MSE 19 with
# the decoder fix alone), and a 16-bit RGB round trip MSE 0.1 through the colour
# transform (0.005 without it), against PAE 8, MSE 20 before.  The bounds leave
# room for floating-point differences between builds.
set(GRAY12_PGM ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/gray12_full_range.pgm)
add_test(NAME enc_gray12_fine COMMAND open_htj2k_enc -i ${GRAY12_PGM} -o gray12_fine.j2c Qstep=0.0001)
add_test(NAME dec_gray12_fine COMMAND open_htj2k_dec -i gray12_fine.j2c -o gray12_fine.pgm)
set_tests_properties(dec_gray12_fine PROPERTIES DEPENDS enc_gray12_fine)
add_test(NAME comp_gray12_fine COMMAND imgcmp gray12_fine_00.pgm ${GRAY12_PGM} 1 0.05)
set_tests_properties(comp_gray12_fine PROPERTIES DEPENDS dec_gray12_fine)
add_test(NAME enc_gray16_fine COMMAND open_htj2k_enc -i ${GRAY16_PGM} -o gray16_fine.j2c Qstep=0.00001)
add_test(NAME dec_gray16_fine COMMAND open_htj2k_dec -i gray16_fine.j2c -o gray16_fine.pgm)
set_tests_properties(dec_gray16_fine PROPERTIES DEPENDS enc_gray16_fine)
add_test(NAME comp_gray16_fine COMMAND imgcmp gray16_fine_00.pgm ${GRAY16_PGM} 2 0.5)
set_tests_properties(comp_gray16_fine PROPERTIES DEPENDS dec_gray16_fine)
# Code-blocks 4 samples wide or high.  The encoder gives each code-block
# round_up(W, 8) x round_up(H, 8) samples of scratch, so a 4 x 1024 or 1024 x 4
# code-block takes 8192 samples, twice its area.  The scratch for resolutions
# coded outside the strip pipeline was sized at 4096 samples per code-block and
# overflowed on the heap (a crash).  Clevels=0 puts the whole image in such a
# resolution; each image is one full-size code-block.  Lossless round trips
# with both block coders.
foreach(_shape tall wide)
  if(_shape STREQUAL "tall")
    set(_cblk "Cblk={1024,4}")
  else()
    set(_cblk "Cblk={4,1024}")
  endif()
  set(_src ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/narrow_cblk_${_shape}.pgm)
  foreach(_coder HT 0)
    set(_t narrow_cblk_${_shape}_${_coder})
    add_test(NAME enc_${_t} COMMAND open_htj2k_enc -i ${_src} -o ${_t}.j2c Clevels=0 ${_cblk} Cmodes=${_coder} Creversible=yes)
    add_test(NAME dec_${_t} COMMAND open_htj2k_dec -i ${_t}.j2c -o ${_t}.pgm)
    set_tests_properties(dec_${_t} PROPERTIES DEPENDS enc_${_t})
    add_test(NAME comp_${_t} COMMAND imgcmp ${_t}_00.pgm ${_src} 0 0)
    set_tests_properties(comp_${_t} PROPERTIES DEPENDS dec_${_t})
  endforeach()
endforeach()

set(RGB16_PPM ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/rgb16_full_range.ppm)
add_test(NAME enc_rgb16_fine COMMAND open_htj2k_enc -i ${RGB16_PPM} -o rgb16_fine.j2c Qstep=0.00001)
add_test(NAME dec_rgb16_fine COMMAND open_htj2k_dec -i rgb16_fine.j2c -o rgb16_fine.ppm)
set_tests_properties(dec_rgb16_fine PROPERTIES DEPENDS enc_rgb16_fine)
add_test(NAME comp_rgb16_fine COMMAND imgcmp rgb16_fine.ppm ${RGB16_PPM} 2 1.0)
set_tests_properties(comp_rgb16_fine PROPERTIES DEPENDS dec_rgb16_fine)
add_test(NAME enc_rgb16_fine_noycc COMMAND open_htj2k_enc -i ${RGB16_PPM} -o rgb16_fine_noycc.j2c Qstep=0.00001 Cycc=no)
add_test(NAME dec_rgb16_fine_noycc COMMAND open_htj2k_dec -i rgb16_fine_noycc.j2c -o rgb16_fine_noycc.ppm)
set_tests_properties(dec_rgb16_fine_noycc PROPERTIES DEPENDS enc_rgb16_fine_noycc)
add_test(NAME comp_rgb16_fine_noycc COMMAND imgcmp rgb16_fine_noycc.ppm ${RGB16_PPM} 2 0.5)
set_tests_properties(comp_rgb16_fine_noycc PROPERTIES DEPENDS dec_rgb16_fine_noycc)

# Unknown marker segments must be skipped by their length (Lmar), not scanned
# byte-by-byte. kodim23_unknown_marker.j2c is a lossless kodim23 codestream with
# an unknown marker (0xFF6F, a 6-byte segment) injected into the main header;
# decoding it must skip that marker cleanly and reproduce kodim23 exactly (PAE=0).
add_test(NAME dec_unknown_marker COMMAND open_htj2k_dec -i ${CMAKE_CURRENT_SOURCE_DIR}/conformance_data/kodim23_unknown_marker.j2c -o kodim23_um.ppm)
add_test(NAME comp_unknown_marker COMMAND imgcmp kodim23_um.ppm ${ENCODER_REF_DIR}/kodim23.ppm 0 0)
set_tests_properties(comp_unknown_marker PROPERTIES DEPENDS dec_unknown_marker)

# --- Qfactor estimate round-trip (encoder <-> estimate_qfactor consistency) ----
# estimate_qfactor recomputes the QCD/QCC step formula from the SAME
# visual_weighting.hpp the encoder uses, so inverting a file encoded with a given
# (Qfactor, Qcsf, Qppd, Qzoom) must recover that Qfactor with ~0 residual. The
# checks are exit-code based (--expect-q / --max-residual) with no output
# parsing, so they are portable across platforms. The visual-weighting model is
# not signalled in the codestream, so each estimate is told the encode-time model.

# Legacy weighting (default path): reuse enc_lossy's Qfactor=90 output.
add_test(NAME qfest_legacy COMMAND estimate_qfactor kodim23lossy.j2c --expect-q 90 --max-residual 0.01)
set_tests_properties(qfest_legacy PROPERTIES DEPENDS enc_lossy)

# Pin the legacy Qfactor output to its historical bytes. qfest_legacy alone
# cannot catch a change to the legacy tables/gains -- the encoder and
# estimate_qfactor read the same shared code, so they stay in agreement at
# residual 0 even if both shift. This golden compare against a committed
# reference is what guards the "default Qfactor output is bit-identical"
# contract against future drift in visual_weighting.hpp.
#
# We pin only the QCD/QCC *marker* bytes (dumped by estimate_qfactor
# --dump-quant), NOT the whole codestream. Those marker bytes are a pure
# function of (Qfactor, bit-depth, the visual_weighting.hpp tables) and carry no
# sample data, so they are byte-identical on every platform. The entropy-coded
# payload is not: the lossy 9/7 DWT runs in float, and -march=native codegen
# (x86 FMA/AVX2 vs ARM NEON) makes those coefficients -- and thus the packet
# bytes -- diverge bit-for-bit across architectures. A full-file compare
# therefore failed on x86 CI while passing on ARM; the marker dump is portable.
add_test(NAME qfest_legacy_dump COMMAND estimate_qfactor kodim23lossy.j2c --dump-quant kodim23lossy.quant.txt)
set_tests_properties(qfest_legacy_dump PROPERTIES DEPENDS enc_lossy)
add_test(NAME qfest_legacy_golden COMMAND ${CMAKE_COMMAND} -E compare_files
         kodim23lossy.quant.txt ${CMAKE_CURRENT_SOURCE_DIR}/conformance_data/kodim23_q90_legacy.quant.txt)
set_tests_properties(qfest_legacy_golden PROPERTIES DEPENDS qfest_legacy_dump)

# Analytic Mannos-Sakrison (EXPERIMENTAL).
add_test(NAME enc_qf_mannos COMMAND open_htj2k_enc -i ${ENCODER_REF_DIR}/kodim23.ppm -o kodim23_qfmannos.j2c Qfactor=90 Qcsf=mannos)
add_test(NAME qfest_mannos COMMAND estimate_qfactor kodim23_qfmannos.j2c --csf mannos --expect-q 90 --max-residual 0.01)
set_tests_properties(qfest_mannos PROPERTIES DEPENDS enc_qf_mannos)

# Analytic Mannos-Sakrison with zoom (the viewing-condition axis).
add_test(NAME enc_qf_mannos_zoom COMMAND open_htj2k_enc -i ${ENCODER_REF_DIR}/kodim23.ppm -o kodim23_qfmz.j2c Qfactor=85 Qcsf=mannos Qzoom=2)
add_test(NAME qfest_mannos_zoom COMMAND estimate_qfactor kodim23_qfmz.j2c --csf mannos --zoom 2 --expect-q 85 --max-residual 0.01)
set_tests_properties(qfest_mannos_zoom PROPERTIES DEPENDS enc_qf_mannos_zoom)

# Analytic Daly with an explicit reference ppd.
add_test(NAME enc_qf_daly COMMAND open_htj2k_enc -i ${ENCODER_REF_DIR}/kodim23.ppm -o kodim23_qfdaly.j2c Qfactor=80 Qcsf=daly Qppd=39)
add_test(NAME qfest_daly COMMAND estimate_qfactor kodim23_qfdaly.j2c --csf daly --ppd 39 --expect-q 80 --max-residual 0.01)
set_tests_properties(qfest_daly PROPERTIES DEPENDS enc_qf_daly)

# Negative test: inverting the Mannos file under the (wrong) legacy assumption
# must exceed the residual ceiling. Assert on the printed "CHECK FAIL" rather
# than WILL_FAIL, so a missing/unreadable file (which exits non-zero WITHOUT
# reaching the residual check) fails the test instead of passing it spuriously.
add_test(NAME qfest_mismatch COMMAND estimate_qfactor kodim23_qfmannos.j2c --max-residual 0.01)
set_tests_properties(qfest_mismatch PROPERTIES DEPENDS enc_qf_mannos PASS_REGULAR_EXPRESSION "CHECK FAIL")

# --- Sub-sampling-aware analytic weighting (per-axis frequency mapping) --------
# visual_weight_check pins the mapping invariants directly on the shared header:
# (1,1) bit-identity with the pre-generalization output, the exact one-level
# shift at (2,2), the 4:2:2 HL>LH anisotropy, and the generic default role.
add_test(NAME qf_vw_invariants COMMAND visual_weight_check check)

# End-to-end 4:2:0: a 3-component input with half-size components 1/2 (any PGX
# trio with those shapes will do -- QCD/QCC bytes never depend on sample data).
# Sub-sampled input forces the MCT off, so the codestream cannot label channels;
# Qctype=Y,Cb,Cr selects the chroma CSF and the estimator must be told the same
# hints to invert exactly. The residual-0 round-trip proves encoder and
# estimator agree on the XRsiz/YRsiz-driven frequency mapping.
set(QF420_INPUT "${ENCODER_REF_DIR}/c1p0_05-0.pgx,${ENCODER_REF_DIR}/c1p0_05-2.pgx,${ENCODER_REF_DIR}/c1p0_05-3.pgx")
add_test(NAME enc_qf_420_ctype COMMAND open_htj2k_enc -i ${QF420_INPUT} -o kodim_qf420ct.j2c Qfactor=88 Qcsf=mannos Qctype=Y,Cb,Cr)
add_test(NAME qfest_420_ctype COMMAND estimate_qfactor kodim_qf420ct.j2c --csf mannos --ctype Y,Cb,Cr --expect-q 88 --max-residual 0.01)
set_tests_properties(qfest_420_ctype PROPERTIES DEPENDS enc_qf_420_ctype)

# Negative control: without the ctype hints the estimator assumes generic
# (luminance-CSF) chroma and must NOT match -- proof the chroma CSF actually
# reached the emitted QCC bytes (the path was dead code before Qctype existed).
add_test(NAME qfest_420_ctype_required COMMAND estimate_qfactor kodim_qf420ct.j2c --csf mannos --max-residual 0.01)
set_tests_properties(qfest_420_ctype_required PROPERTIES DEPENDS enc_qf_420_ctype PASS_REGULAR_EXPRESSION "CHECK FAIL")

# Reuse-luma A/B switch round-trip (luminance CSF shape + sub-sampling mapping).
add_test(NAME enc_qf_420_reuseluma COMMAND open_htj2k_enc -i ${QF420_INPUT} -o kodim_qf420rl.j2c Qfactor=88 Qcsf=mannos Qctype=Y,Cb,Cr Qchromacsf=luma)
add_test(NAME qfest_420_reuseluma COMMAND estimate_qfactor kodim_qf420rl.j2c --csf mannos --ctype Y,Cb,Cr --chroma-csf luma --expect-q 88 --max-residual 0.01)
set_tests_properties(qfest_420_reuseluma PROPERTIES DEPENDS enc_qf_420_reuseluma)

# End-to-end 4:2:2 on tiny synthetic fixtures (Y 64x64, Cb/Cr 32x64). Also a
# regression guard for the Cycc guard fix: 4:2:2 differs in size only
# horizontally, which used to leave the MCT enabled across differently-sized
# component buffers and corrupt the heap. The legacy encode exercises exactly
# that path (default Cycc=yes + Qfactor); the analytic pair round-trips the
# anisotropic (2,1) mapping through the emitted QCC bytes.
set(QF422_INPUT "${ENCODER_REF_DIR}/synth422_y.pgx,${ENCODER_REF_DIR}/synth422_cb.pgx,${ENCODER_REF_DIR}/synth422_cr.pgx")
add_test(NAME enc_qf_422_legacy COMMAND open_htj2k_enc -i ${QF422_INPUT} -o synth_qf422lg.j2c Qfactor=85)
add_test(NAME qfest_422_legacy COMMAND estimate_qfactor synth_qf422lg.j2c --expect-q 85 --max-residual 0.01)
set_tests_properties(qfest_422_legacy PROPERTIES DEPENDS enc_qf_422_legacy)
add_test(NAME enc_qf_422_ctype COMMAND open_htj2k_enc -i ${QF422_INPUT} -o synth_qf422ct.j2c Qfactor=85 Qcsf=mannos Qctype=Y,Cb,Cr)
add_test(NAME qfest_422_ctype COMMAND estimate_qfactor synth_qf422ct.j2c --csf mannos --ctype Y,Cb,Cr --expect-q 85 --max-residual 0.01)
set_tests_properties(qfest_422_ctype PROPERTIES DEPENDS enc_qf_422_ctype)

# The default (legacy) Qfactor weighting follows the JPEG Qfactor guideline,
# WG1N101553 v2. qfactor_guideline_check recomputes every step of a codestream
# from the guideline's own filters and Tables 2-4, sharing no code with the
# encoder, so a wrong table in visual_weighting.hpp fails here even though the
# estimate_qfactor round trips (which read the same header) still pass. Q = 30
# applies the full visual weights, Q = 85 two thirds of them; 6 levels reach the
# weight the tables give to levels past the fifth; enc_lossy (4:4:4) is the
# control.
add_test(NAME qfgl_444 COMMAND qfactor_guideline_check kodim23lossy.j2c 90)
set_tests_properties(qfgl_444 PROPERTIES DEPENDS enc_lossy)
add_test(NAME enc_qf_420_legacy COMMAND open_htj2k_enc -i ${QF420_INPUT} -o kodim_qf420lg.j2c Qfactor=85)
add_test(NAME qfgl_420 COMMAND qfactor_guideline_check kodim_qf420lg.j2c 85)
set_tests_properties(qfgl_420 PROPERTIES DEPENDS enc_qf_420_legacy)
add_test(NAME qfest_420_legacy COMMAND estimate_qfactor kodim_qf420lg.j2c --expect-q 85 --max-residual 0.01)
set_tests_properties(qfest_420_legacy PROPERTIES DEPENDS enc_qf_420_legacy)
add_test(NAME enc_qf_420_legacy_q30 COMMAND open_htj2k_enc -i ${QF420_INPUT} -o kodim_qf420lg30.j2c Qfactor=30)
add_test(NAME qfgl_420_q30 COMMAND qfactor_guideline_check kodim_qf420lg30.j2c 30)
set_tests_properties(qfgl_420_q30 PROPERTIES DEPENDS enc_qf_420_legacy_q30)
add_test(NAME enc_qf_420_legacy_l6 COMMAND open_htj2k_enc -i ${QF420_INPUT} -o kodim_qf420lg6.j2c Qfactor=50 Clevels=6)
add_test(NAME qfgl_420_l6 COMMAND qfactor_guideline_check kodim_qf420lg6.j2c 50)
set_tests_properties(qfgl_420_l6 PROPERTIES DEPENDS enc_qf_420_legacy_l6)
add_test(NAME qfgl_422 COMMAND qfactor_guideline_check synth_qf422lg.j2c 85)
set_tests_properties(qfgl_422 PROPERTIES DEPENDS enc_qf_422_legacy)
add_test(NAME enc_qf_422_legacy_q30 COMMAND open_htj2k_enc -i ${QF422_INPUT} -o synth_qf422lg30.j2c Qfactor=30)
add_test(NAME qfgl_422_q30 COMMAND qfactor_guideline_check synth_qf422lg30.j2c 30)
set_tests_properties(qfgl_422_q30 PROPERTIES DEPENDS enc_qf_422_legacy_q30)

# Flat Qfactor weights (Qweights=flat): every subband weight is 1 at every Qfactor, and
# without the MCT every component is independent (unit colour gain). The Qfactor still
# sets the reference step, so at Qfactor >= 97, where the guideline's weights are flat
# already, an RGB encode is byte-identical to the default. qfactor_guideline_check
# --weights flat recomputes every step independently of the encoder; estimate_qfactor
# --weights flat recovers the Qfactor, and without --weights it must not. The decoded
# error bounds are about 2.5x the measured ones (kodim23 Q50: PAE 25, MSE 9.5; 16-bit
# gray Q40: PAE 3848, MSE 1356050).
add_test(NAME enc_qw_444 COMMAND open_htj2k_enc -i ${ENCODER_REF_DIR}/kodim23.ppm -o kodim23_qw50.j2c Qfactor=50 Qweights=flat)
add_test(NAME qwgl_444 COMMAND qfactor_guideline_check kodim23_qw50.j2c 50 --weights flat)
set_tests_properties(qwgl_444 PROPERTIES DEPENDS enc_qw_444)
add_test(NAME qwest_444 COMMAND estimate_qfactor kodim23_qw50.j2c --weights flat --expect-q 50 --max-residual 0.01)
set_tests_properties(qwest_444 PROPERTIES DEPENDS enc_qw_444)
add_test(NAME qwest_444_mismatch COMMAND estimate_qfactor kodim23_qw50.j2c --expect-q 50 --max-residual 0.01)
set_tests_properties(qwest_444_mismatch PROPERTIES DEPENDS enc_qw_444 PASS_REGULAR_EXPRESSION "CHECK FAIL")
add_test(NAME dec_qw_444 COMMAND open_htj2k_dec -i kodim23_qw50.j2c -o kodim23_qw50.ppm)
set_tests_properties(dec_qw_444 PROPERTIES DEPENDS enc_qw_444)
add_test(NAME comp_qw_444 COMMAND imgcmp kodim23_qw50.ppm ${ENCODER_REF_DIR}/kodim23.ppm 63 24)
set_tests_properties(comp_qw_444 PROPERTIES DEPENDS dec_qw_444)
add_test(NAME qwgl_visual_file COMMAND qfactor_guideline_check kodim23lossy.j2c 90 --weights flat)
set_tests_properties(qwgl_visual_file PROPERTIES DEPENDS enc_lossy
                     PASS_REGULAR_EXPRESSION "[1-9][0-9]* of [0-9]+ steps differ")
add_test(NAME enc_qw_444_q98 COMMAND open_htj2k_enc -i ${ENCODER_REF_DIR}/kodim23.ppm -o kodim23_qw98.j2c Qfactor=98 Qweights=flat)
add_test(NAME enc_qw_444_q98_visual COMMAND open_htj2k_enc -i ${ENCODER_REF_DIR}/kodim23.ppm -o kodim23_qv98.j2c Qfactor=98)
add_test(NAME qw_q98_identical COMMAND ${CMAKE_COMMAND} -E compare_files kodim23_qw98.j2c kodim23_qv98.j2c)
set_tests_properties(qw_q98_identical PROPERTIES DEPENDS "enc_qw_444_q98;enc_qw_444_q98_visual")
add_test(NAME enc_qw_nomct COMMAND open_htj2k_enc -i ${ENCODER_REF_DIR}/kodim23.ppm -o kodim23_qwn60.j2c Qfactor=60 Qweights=flat Cycc=no)
add_test(NAME qwgl_nomct COMMAND qfactor_guideline_check kodim23_qwn60.j2c 60 --weights flat)
set_tests_properties(qwgl_nomct PROPERTIES DEPENDS enc_qw_nomct)
add_test(NAME qwest_nomct COMMAND estimate_qfactor kodim23_qwn60.j2c --weights flat --expect-q 60 --max-residual 0.01)
set_tests_properties(qwest_nomct PROPERTIES DEPENDS enc_qw_nomct)
add_test(NAME enc_qw_420 COMMAND open_htj2k_enc -i ${QF420_INPUT} -o kodim_qw420.j2c Qfactor=70 Qweights=flat)
add_test(NAME qwgl_420 COMMAND qfactor_guideline_check kodim_qw420.j2c 70 --weights flat)
set_tests_properties(qwgl_420 PROPERTIES DEPENDS enc_qw_420)
add_test(NAME qwest_420 COMMAND estimate_qfactor kodim_qw420.j2c --weights flat --expect-q 70 --max-residual 0.01)
set_tests_properties(qwest_420 PROPERTIES DEPENDS enc_qw_420)
add_test(NAME enc_qw_gray16 COMMAND open_htj2k_enc -i ${GRAY16_PGM} -o gray16_qw40.j2c Qfactor=40 Qweights=flat)
add_test(NAME qwgl_gray16 COMMAND qfactor_guideline_check gray16_qw40.j2c 40 --weights flat)
set_tests_properties(qwgl_gray16 PROPERTIES DEPENDS enc_qw_gray16)
add_test(NAME qwest_gray16 COMMAND estimate_qfactor gray16_qw40.j2c --weights flat --expect-q 40 --max-residual 0.01)
set_tests_properties(qwest_gray16 PROPERTIES DEPENDS enc_qw_gray16)
add_test(NAME dec_qw_gray16 COMMAND open_htj2k_dec -i gray16_qw40.j2c -o gray16_qw40.pgm)
set_tests_properties(dec_qw_gray16 PROPERTIES DEPENDS enc_qw_gray16)
add_test(NAME comp_qw_gray16 COMMAND imgcmp gray16_qw40_00.pgm ${GRAY16_PGM} 9620 3400000)
set_tests_properties(comp_qw_gray16 PROPERTIES DEPENDS dec_qw_gray16)
add_test(NAME enc_qw_unknown COMMAND open_htj2k_enc -i ${ENCODER_REF_DIR}/kodim23.ppm -o kodim23_qwx.j2c Qfactor=50 Qweights=perceptual)
set_tests_properties(enc_qw_unknown PROPERTIES PASS_REGULAR_EXPRESSION "unknown Qweights value")

# Require the decoder-conformance cleanup fixture so these encoder tests are
# guaranteed to run AFTER cleanup_artifacts, never concurrently with it.
# Without this, ctest -j was free to schedule cleanup_artifacts (which globs
# *.ppm in the build dir and deletes everything matching) at any time, which
# under parallel load sometimes landed between dec_lossless writing
# kodim23lossless.ppm and comp_lossless reading it -- producing an
# intermittent "File kodim23lossless.ppm is not found" failure on
# comp_lossless / comp_lossless_odd that vanished on serial reruns.
#
# The `test_artifacts` fixture is declared by tests/decoder_conformance.cmake,
# which is included before this file from CMakeLists.txt, so the name is
# already visible here.
set_tests_properties(
  enc_lossless dec_lossless comp_lossless
  enc_lossy    dec_lossy    comp_lossy
  enc_lossless_odd dec_lossless_odd comp_lossless_odd
  enc_prcl         dec_prcl         comp_prcl
  enc_precincts    dec_precincts    comp_precincts
  enc_precincts_small dec_precincts_small comp_precincts_small
  enc_precincts_clamped dec_precincts_clamped comp_precincts_clamped
  enc_prcl_prec    dec_prcl_prec    comp_prcl_prec
  dec_p18_arri     enc_p18_rt       dec_p18_rt       comp_p18_rt
  dec_unknown_marker comp_unknown_marker
  qfest_legacy        qfest_legacy_dump      qfest_legacy_golden
  enc_qf_mannos       qfest_mannos
  enc_qf_mannos_zoom  qfest_mannos_zoom
  enc_qf_daly         qfest_daly
  qfest_mismatch
  qf_vw_invariants
  enc_qf_420_ctype    qfest_420_ctype   qfest_420_ctype_required
  enc_qf_420_reuseluma qfest_420_reuseluma
  enc_qf_422_legacy   qfest_422_legacy
  enc_qf_422_ctype    qfest_422_ctype
  qfgl_444
  enc_qf_420_legacy   qfgl_420          qfest_420_legacy
  enc_qf_420_legacy_q30 qfgl_420_q30
  enc_qf_420_legacy_l6  qfgl_420_l6
  qfgl_422
  enc_qf_422_legacy_q30 qfgl_422_q30
  enc_qw_444 qwgl_444 qwest_444 qwest_444_mismatch dec_qw_444 comp_qw_444 qwgl_visual_file
  enc_qw_444_q98 enc_qw_444_q98_visual qw_q98_identical
  enc_qw_nomct qwgl_nomct qwest_nomct
  enc_qw_420 qwgl_420 qwest_420
  enc_qw_gray16 qwgl_gray16 qwest_gray16 dec_qw_gray16 comp_qw_gray16
  enc_qw_unknown
  PROPERTIES FIXTURES_REQUIRED test_artifacts)

# Library-API encodes: the buffered entry point (invoke_line_based), which the
# CLI never uses, must accept every image the streaming one does and produce
# the same codestream.  It used to crash for any component encoded without the
# MCT (grayscale, MCT off, or more than three components).
add_test(NAME enc_entry_points COMMAND encode_entry_points)

# ISO/IEC 15444-1 (Part 1) block coder, selected with Cmodes=0: EBCOT/MQ in its
# default mode, one codeword segment per code-block.  The lossless round-trips
# must be exact.  The lossy stream carries the same quantisation indices as the
# HT stream of the same Qfactor, so it is held to the same tolerance as enc_lossy.
add_test(NAME enc_part1_lossless COMMAND open_htj2k_enc -i ${ENCODER_REF_DIR}/kodim23.ppm -o kodim23part1.j2c Creversible=yes Cmodes=0)
add_test(NAME dec_part1_lossless COMMAND open_htj2k_dec -i kodim23part1.j2c -o kodim23part1.ppm)
set_tests_properties(dec_part1_lossless PROPERTIES DEPENDS enc_part1_lossless)
add_test(NAME comp_part1_lossless COMMAND imgcmp kodim23part1.ppm ${ENCODER_REF_DIR}/kodim23.ppm 0 0)
set_tests_properties(comp_part1_lossless PROPERTIES DEPENDS dec_part1_lossless)

add_test(NAME enc_part1_lossy COMMAND open_htj2k_enc -i ${ENCODER_REF_DIR}/kodim23.ppm -o kodim23part1lossy.j2c Qfactor=90 Cmodes=0)
add_test(NAME dec_part1_lossy COMMAND open_htj2k_dec -i kodim23part1lossy.j2c -o kodim23part1lossy.ppm)
set_tests_properties(dec_part1_lossy PROPERTIES DEPENDS enc_part1_lossy)
add_test(NAME comp_part1_lossy COMMAND imgcmp kodim23part1lossy.ppm ${ENCODER_REF_DIR}/kodim23.ppm 23 6)
set_tests_properties(comp_part1_lossy PROPERTIES DEPENDS dec_part1_lossy)

# Odd image size with 32x32 code-blocks: partial stripes and partial blocks.
add_test(NAME enc_part1_lossless_odd COMMAND open_htj2k_enc -i ${ENCODER_REF_DIR}/kodim23odd.ppm -o kodim23odd_part1.j2c Creversible=yes Cmodes=0 "Cblk={32,32}")
add_test(NAME dec_part1_lossless_odd COMMAND open_htj2k_dec -i kodim23odd_part1.j2c -o kodim23odd_part1.ppm)
set_tests_properties(dec_part1_lossless_odd PROPERTIES DEPENDS enc_part1_lossless_odd)
add_test(NAME comp_part1_lossless_odd COMMAND imgcmp kodim23odd_part1.ppm ${ENCODER_REF_DIR}/kodim23odd.ppm 0 0)
set_tests_properties(comp_part1_lossless_odd PROPERTIES DEPENDS dec_part1_lossless_odd)

# Several precincts per resolution in RPCL order: the single-segment pass-length
# signalling of Part 1 code-blocks across many packets.
add_test(NAME enc_part1_precincts COMMAND open_htj2k_enc -i ${ENCODER_REF_DIR}/kodim23.ppm -o kodim23part1prec.j2c Creversible=yes Cmodes=0 Corder=RPCL "Cprecincts={256,256}{128,128}")
add_test(NAME dec_part1_precincts COMMAND open_htj2k_dec -i kodim23part1prec.j2c -o kodim23part1prec.ppm)
set_tests_properties(dec_part1_precincts PROPERTIES DEPENDS enc_part1_precincts)
add_test(NAME comp_part1_precincts COMMAND imgcmp kodim23part1prec.ppm ${ENCODER_REF_DIR}/kodim23.ppm 0 0)
set_tests_properties(comp_part1_precincts PROPERTIES DEPENDS dec_part1_precincts)

# The JPH file format requires HT block coding: Part 1 output into .jph is refused.
add_test(NAME enc_part1_jph_rejected COMMAND open_htj2k_enc -i ${ENCODER_REF_DIR}/kodim23.ppm -o kodim23part1.jph Cmodes=0)
set_tests_properties(enc_part1_jph_rejected PROPERTIES WILL_FAIL TRUE)
# JPH with an alpha channel: 2 components (gray + alpha) and 4 (RGB + alpha).
# The colour specification box declares greyscale (EnumCS 17) or sRGB (16) for
# the colour channels, and a Channel Definition box (ISO/IEC 15444-1 I.5.3.6)
# declares the last component as opacity (Typ 1, Asoc 0) and the others as
# colours 1, 2, 3.  Both files must decode back to their input components.
# Two different 512x512 planes alternate, so a swapped component would show.
set(JPH_ALPHA_P0 "${ENCODER_REF_DIR}/c1p0_05-2.pgx")
set(JPH_ALPHA_P1 "${ENCODER_REF_DIR}/c1p0_05-3.pgx")

# Qfactor with an alpha channel: gray + alpha (2 components) and colour + alpha (4, the
# colour transform on components 0-2).  The alpha channel is quantized like a gray-scale
# plane, so it decodes with the same error as the second plane of the 2-component image.
# Decoded components are checked against a mean squared error bound (about twice the
# measured one).
set(QF_ALPHA_2_MSE 15 15)
set(QF_ALPHA_4_MSE 50 35 90 15)
foreach(NC IN ITEMS 2 4)
  if(NC EQUAL 2)
    set(_in "${JPH_ALPHA_P0},${JPH_ALPHA_P1}")
  else()
    set(_in "${JPH_ALPHA_P0},${JPH_ALPHA_P1},${JPH_ALPHA_P0},${JPH_ALPHA_P1}")
  endif()
  add_test(NAME enc_qf_alpha${NC} COMMAND open_htj2k_enc -i ${_in} -o qf_alpha${NC}.j2c Qfactor=90)
  add_test(NAME dec_qf_alpha${NC} COMMAND open_htj2k_dec -i qf_alpha${NC}.j2c -o qf_alpha${NC}.pgx)
  set_tests_properties(dec_qf_alpha${NC} PROPERTIES DEPENDS enc_qf_alpha${NC})
  math(EXPR _last "${NC} - 1")
  foreach(C RANGE ${_last})
    math(EXPR _odd "${C} % 2")
    list(GET QF_ALPHA_${NC}_MSE ${C} _mse)
    add_test(NAME comp_qf_alpha${NC}_${C} COMMAND imgcmp qf_alpha${NC}_0${C}.pgx ${JPH_ALPHA_P${_odd}} 255 ${_mse})
    set_tests_properties(comp_qf_alpha${NC}_${C} PROPERTIES DEPENDS dec_qf_alpha${NC})
  endforeach()
endforeach()
set(JPH_ALPHA_2_HEX "0000000f636f6c7201000000000011;00000016636465660002000000000001000100010000")
set(JPH_ALPHA_4_HEX "0000000f636f6c7201000000000010;00000022636465660004000000000001000100000002000200000003000300010000")
foreach(NC IN ITEMS 2 4)
  if(NC EQUAL 2)
    set(JPH_ALPHA_IN "${JPH_ALPHA_P0},${JPH_ALPHA_P1}")
  else()
    set(JPH_ALPHA_IN "${JPH_ALPHA_P0},${JPH_ALPHA_P1},${JPH_ALPHA_P0},${JPH_ALPHA_P1}")
  endif()
  add_test(NAME enc_jph_alpha${NC} COMMAND open_htj2k_enc -i ${JPH_ALPHA_IN} -o jph_alpha${NC}.jph Creversible=yes)
  add_test(NAME box_jph_alpha${NC} COMMAND ${CMAKE_COMMAND} -DFILE=jph_alpha${NC}.jph "-DHEX=${JPH_ALPHA_${NC}_HEX}"
           -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_file_hex.cmake)
  set_tests_properties(box_jph_alpha${NC} PROPERTIES DEPENDS enc_jph_alpha${NC})
  add_test(NAME dec_jph_alpha${NC} COMMAND open_htj2k_dec -i jph_alpha${NC}.jph -o jph_alpha${NC}.pgx)
  set_tests_properties(dec_jph_alpha${NC} PROPERTIES DEPENDS enc_jph_alpha${NC})
  set(JPH_ALPHA_TESTS enc_jph_alpha${NC} box_jph_alpha${NC} dec_jph_alpha${NC})
  math(EXPR LAST "${NC} - 1")
  foreach(C RANGE ${LAST})
    math(EXPR ODD "${C} % 2")
    add_test(NAME comp_jph_alpha${NC}_${C} COMMAND imgcmp jph_alpha${NC}_0${C}.pgx ${JPH_ALPHA_P${ODD}} 0 0)
    set_tests_properties(comp_jph_alpha${NC}_${C} PROPERTIES DEPENDS dec_jph_alpha${NC})
    list(APPEND JPH_ALPHA_TESTS comp_jph_alpha${NC}_${C})
  endforeach()
  set_tests_properties(${JPH_ALPHA_TESTS} PROPERTIES FIXTURES_REQUIRED test_artifacts)
endforeach()

# JP2 files (ISO/IEC 15444-1 Annex I) carry a Part 1 codestream: Cmodes=0, lossless,
# 1 to 4 components alternating the two 512x512 8-bit planes above.  Each file must
# hold the signature box, the file type box (brand 'jp2 ', MinV 0, compatibility
# 'jp2 ') and the JP2 header box: image header (512x512, NC components, 8-bit
# unsigned, C 7, UnkC 0, IPR 0), colour specification (enumerated: greyscale, 17, for
# 1 and 2 components, sRGB, 16, for 3 and 4) and, for 2 and 4 components, the
# channel definition box of the JPH files above.  The contiguous codestream box comes
# next (its type, then the SOC and SIZ markers).  Every component must decode exactly.
set(JP2_1_HEX "0000000c6a5020200d0a870a00000014667479706a703220000000006a7032200000002d6a703268000000166968647200000200000002000001070700000000000f636f6c7201000000000011")
set(JP2_2_HEX "0000000c6a5020200d0a870a00000014667479706a703220000000006a703220000000436a703268000000166968647200000200000002000002070700000000000f636f6c720100000000001100000016636465660002000000000001000100010000")
set(JP2_3_HEX "0000000c6a5020200d0a870a00000014667479706a703220000000006a7032200000002d6a703268000000166968647200000200000002000003070700000000000f636f6c7201000000000010")
set(JP2_4_HEX "0000000c6a5020200d0a870a00000014667479706a703220000000006a7032200000004f6a703268000000166968647200000200000002000004070700000000000f636f6c720100000000001000000022636465660004000000000001000100000002000200000003000300010000")
set(JP2_CODESTREAM_HEX "6a703263ff4fff51")
foreach(NC IN ITEMS 1 2 3 4)
  math(EXPR LAST "${NC} - 1")
  set(JP2_IN "")
  foreach(C RANGE ${LAST})
    math(EXPR ODD "${C} % 2")
    list(APPEND JP2_IN "${JPH_ALPHA_P${ODD}}")
  endforeach()
  string(REPLACE ";" "," JP2_IN "${JP2_IN}")
  add_test(NAME enc_jp2_${NC}comp COMMAND open_htj2k_enc -i ${JP2_IN} -o jp2_${NC}comp.jp2 Creversible=yes Cmodes=0)
  add_test(NAME box_jp2_${NC}comp COMMAND ${CMAKE_COMMAND} -DFILE=jp2_${NC}comp.jp2
           "-DHEX=${JP2_${NC}_HEX};${JP2_CODESTREAM_HEX}" -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_file_hex.cmake)
  set_tests_properties(box_jp2_${NC}comp PROPERTIES DEPENDS enc_jp2_${NC}comp)
  add_test(NAME dec_jp2_${NC}comp COMMAND open_htj2k_dec -i jp2_${NC}comp.jp2 -o jp2_${NC}comp.pgx)
  set_tests_properties(dec_jp2_${NC}comp PROPERTIES DEPENDS enc_jp2_${NC}comp)
  set(JP2_TESTS enc_jp2_${NC}comp box_jp2_${NC}comp dec_jp2_${NC}comp)
  foreach(C RANGE ${LAST})
    math(EXPR ODD "${C} % 2")
    add_test(NAME comp_jp2_${NC}comp_${C} COMMAND imgcmp jp2_${NC}comp_0${C}.pgx ${JPH_ALPHA_P${ODD}} 0 0)
    set_tests_properties(comp_jp2_${NC}comp_${C} PROPERTIES DEPENDS dec_jp2_${NC}comp)
    list(APPEND JP2_TESTS comp_jp2_${NC}comp_${C})
  endforeach()
  set_tests_properties(${JP2_TESTS} PROPERTIES FIXTURES_REQUIRED test_artifacts)
endforeach()
# -jph_color_space YCC declares sYCC (EnumCS 18) in a JP2 file too.
set(JP2_YCC_HEX "0000000c6a5020200d0a870a00000014667479706a703220000000006a7032200000002d6a703268000000166968647200000200000002000003070700000000000f636f6c7201000000000012")
add_test(NAME enc_jp2_ycc COMMAND open_htj2k_enc -i ${JPH_ALPHA_P0},${JPH_ALPHA_P1},${JPH_ALPHA_P0} -o jp2_ycc.jp2
         Creversible=yes Cmodes=0 Cycc=no -jph_color_space YCC)
add_test(NAME box_jp2_ycc COMMAND ${CMAKE_COMMAND} -DFILE=jp2_ycc.jp2 "-DHEX=${JP2_YCC_HEX};${JP2_CODESTREAM_HEX}"
         -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_file_hex.cmake)
set_tests_properties(box_jp2_ycc PROPERTIES DEPENDS enc_jp2_ycc)
set_tests_properties(enc_jp2_ycc box_jp2_ycc PROPERTIES FIXTURES_REQUIRED test_artifacts)
# A JP2 file carries only a Part 1 codestream: the default HT block coder (a JPH file's)
# and the Part 2 progression order PRCL are refused.
add_test(NAME enc_ht_jp2_rejected COMMAND open_htj2k_enc -i ${ENCODER_REF_DIR}/kodim23.ppm -o kodim23ht.jp2)
set_tests_properties(enc_ht_jp2_rejected PROPERTIES WILL_FAIL TRUE)
add_test(NAME enc_part1_prcl_jp2_rejected COMMAND open_htj2k_enc -i ${ENCODER_REF_DIR}/kodim23.ppm -o kodim23prcl.jp2 Cmodes=0 Corder=PRCL)
set_tests_properties(enc_part1_prcl_jp2_rejected PROPERTIES WILL_FAIL TRUE)
# Box lengths (ISO/IEC 15444-1 I.4) of JPH and JP2 files: the Contiguous Codestream box
# header for code lengths around the 32-bit LBox limit (LBox = 1 and a 64-bit XLBox above
# it), and the reader with jp2c boxes of LBox = 1 + XLBox and LBox = 0, every box with
# XLBox, and malformed XLBox values, which must be rejected.  See
# tests/tools/jph_box_check/main.cpp.
add_test(NAME jph_box_lengths COMMAND jph_box_check)
set_tests_properties(jph_box_lengths PROPERTIES PASS_REGULAR_EXPRESSION "all cases passed")

# Part 1 code-block mode switches.  Each lossless round-trip must be exact.
#   bypass:  raw SigProp/MagRef passes from the fifth bit-plane, so several
#            codeword segments per code-block (Table D.9).
#   restart: every pass terminated with the predictable (ERTERM) procedure.
#   all:     all six switches at once on the odd-size image with 32x32 blocks.
#   causal:  vertically causal contexts and segmentation symbols, 4x1024 blocks
#            (every stripe of the block is its last).
foreach(P1MODE IN ITEMS "bypass;BYPASS;kodim23;{64,64}"
                        "restart;RESTART|ERTERM;kodim23;{64,64}"
                        "all;BYPASS|RESET|RESTART|CAUSAL|ERTERM|SEGMARK;kodim23odd;{32,32}"
                        "causal;CAUSAL|SEGMARK;kodim23;{4,1024}")
  list(GET P1MODE 0 P1NAME)
  list(GET P1MODE 1 P1FLAGS)
  list(GET P1MODE 2 P1IMG)
  list(GET P1MODE 3 P1CBLK)
  add_test(NAME enc_part1_mode_${P1NAME} COMMAND open_htj2k_enc -i ${ENCODER_REF_DIR}/${P1IMG}.ppm -o part1_mode_${P1NAME}.j2c Creversible=yes "Cmodes=${P1FLAGS}" "Cblk=${P1CBLK}")
  add_test(NAME dec_part1_mode_${P1NAME} COMMAND open_htj2k_dec -i part1_mode_${P1NAME}.j2c -o part1_mode_${P1NAME}.ppm)
  set_tests_properties(dec_part1_mode_${P1NAME} PROPERTIES DEPENDS enc_part1_mode_${P1NAME})
  add_test(NAME comp_part1_mode_${P1NAME} COMMAND imgcmp part1_mode_${P1NAME}.ppm ${ENCODER_REF_DIR}/${P1IMG}.ppm 0 0)
  set_tests_properties(comp_part1_mode_${P1NAME} PROPERTIES DEPENDS dec_part1_mode_${P1NAME})
endforeach()

# Mode switches change how the quantisation indices are coded, not the indices:
# the lossy stream with all six switches is held to the enc_part1_lossy tolerance.
add_test(NAME enc_part1_mode_lossy COMMAND open_htj2k_enc -i ${ENCODER_REF_DIR}/kodim23.ppm -o part1_mode_lossy.j2c Qfactor=90 "Cmodes=BYPASS|RESET|RESTART|CAUSAL|ERTERM|SEGMARK")
add_test(NAME dec_part1_mode_lossy COMMAND open_htj2k_dec -i part1_mode_lossy.j2c -o part1_mode_lossy.ppm)
set_tests_properties(dec_part1_mode_lossy PROPERTIES DEPENDS enc_part1_mode_lossy)
add_test(NAME comp_part1_mode_lossy COMMAND imgcmp part1_mode_lossy.ppm ${ENCODER_REF_DIR}/kodim23.ppm 23 6)
set_tests_properties(comp_part1_mode_lossy PROPERTIES DEPENDS dec_part1_mode_lossy)

# HT code-blocks take the CAUSAL switch (ISO/IEC 15444-15 Table A.3) and no other.
add_test(NAME enc_ht_causal COMMAND open_htj2k_enc -i ${ENCODER_REF_DIR}/kodim23.ppm -o kodim23htcausal.j2c Creversible=yes "Cmodes=HT|CAUSAL")
add_test(NAME dec_ht_causal COMMAND open_htj2k_dec -i kodim23htcausal.j2c -o kodim23htcausal.ppm)
set_tests_properties(dec_ht_causal PROPERTIES DEPENDS enc_ht_causal)
add_test(NAME comp_ht_causal COMMAND imgcmp kodim23htcausal.ppm ${ENCODER_REF_DIR}/kodim23.ppm 0 0)
set_tests_properties(comp_ht_causal PROPERTIES DEPENDS dec_ht_causal)
add_test(NAME enc_ht_bypass_rejected COMMAND open_htj2k_enc -i ${ENCODER_REF_DIR}/kodim23.ppm -o kodim23htbypass.j2c "Cmodes=HT|BYPASS")
set_tests_properties(enc_ht_bypass_rejected PROPERTIES WILL_FAIL TRUE)
add_test(NAME enc_cmodes_unknown_rejected COMMAND open_htj2k_enc -i ${ENCODER_REF_DIR}/kodim23.ppm -o kodim23cmodes.j2c "Cmodes=BYPASS|TERMALL")
set_tests_properties(enc_cmodes_unknown_rejected PROPERTIES WILL_FAIL TRUE)

# HT SigProp and HT MagRef encoding (Cht_refine=N, experimental): each HT
# code-block is coded as one HT set, a cleanup pass N bit-planes above the LSB
# followed by the SigProp and MagRef passes of the plane below.  The planes under
# the refined one are not coded and SigProp only reaches samples next to a
# significant one, so the round trips are bounded rather than exact.  The bounds
# sit just above the errors measured on x86-64 (PAE/MSE): 3/0.043 for one plane
# reversible, 4/0.073 with 16x16 code-blocks on the odd-size image, 10/2.13 for
# three planes at a fine step with 32x32 code-blocks, and 55/12.9 for two planes
# at Qfactor 90.
add_test(NAME enc_ht_refine1 COMMAND open_htj2k_enc -i ${ENCODER_REF_DIR}/kodim23.ppm -o kodim23refine1.j2c Creversible=yes Cht_refine=1)
add_test(NAME dec_ht_refine1 COMMAND open_htj2k_dec -i kodim23refine1.j2c -o kodim23refine1.ppm)
set_tests_properties(dec_ht_refine1 PROPERTIES DEPENDS enc_ht_refine1)
add_test(NAME comp_ht_refine1 COMMAND imgcmp kodim23refine1.ppm ${ENCODER_REF_DIR}/kodim23.ppm 3 0.05)
set_tests_properties(comp_ht_refine1 PROPERTIES DEPENDS dec_ht_refine1)

add_test(NAME enc_ht_refine1_odd COMMAND open_htj2k_enc -i ${ENCODER_REF_DIR}/kodim23odd.ppm -o kodim23odd_refine1.j2c Creversible=yes Cht_refine=1 "Cblk={16,16}")
add_test(NAME dec_ht_refine1_odd COMMAND open_htj2k_dec -i kodim23odd_refine1.j2c -o kodim23odd_refine1.ppm)
set_tests_properties(dec_ht_refine1_odd PROPERTIES DEPENDS enc_ht_refine1_odd)
add_test(NAME comp_ht_refine1_odd COMMAND imgcmp kodim23odd_refine1.ppm ${ENCODER_REF_DIR}/kodim23odd.ppm 4 0.08)
set_tests_properties(comp_ht_refine1_odd PROPERTIES DEPENDS dec_ht_refine1_odd)

add_test(NAME enc_ht_refine3_odd COMMAND open_htj2k_enc -i ${ENCODER_REF_DIR}/kodim23odd.ppm -o kodim23odd_refine3.j2c Qstep=0.002 Cht_refine=3 "Cblk={32,32}")
add_test(NAME dec_ht_refine3_odd COMMAND open_htj2k_dec -i kodim23odd_refine3.j2c -o kodim23odd_refine3.ppm)
set_tests_properties(dec_ht_refine3_odd PROPERTIES DEPENDS enc_ht_refine3_odd)
add_test(NAME comp_ht_refine3_odd COMMAND imgcmp kodim23odd_refine3.ppm ${ENCODER_REF_DIR}/kodim23odd.ppm 12 2.5)
set_tests_properties(comp_ht_refine3_odd PROPERTIES DEPENDS dec_ht_refine3_odd)

add_test(NAME enc_ht_refine2_lossy COMMAND open_htj2k_enc -i ${ENCODER_REF_DIR}/kodim23.ppm -o kodim23refine2lossy.j2c Qfactor=90 Cht_refine=2)
add_test(NAME dec_ht_refine2_lossy COMMAND open_htj2k_dec -i kodim23refine2lossy.j2c -o kodim23refine2lossy.ppm)
set_tests_properties(dec_ht_refine2_lossy PROPERTIES DEPENDS enc_ht_refine2_lossy)
add_test(NAME comp_ht_refine2_lossy COMMAND imgcmp kodim23refine2lossy.ppm ${ENCODER_REF_DIR}/kodim23.ppm 60 14)
set_tests_properties(comp_ht_refine2_lossy PROPERTIES DEPENDS dec_ht_refine2_lossy)

# CAUSAL keeps SigProp from looking at the stripe below: measured 3/0.052.
add_test(NAME enc_ht_refine1_causal COMMAND open_htj2k_enc -i ${ENCODER_REF_DIR}/kodim23.ppm -o kodim23refine1causal.j2c Creversible=yes Cht_refine=1 "Cmodes=HT|CAUSAL")
add_test(NAME dec_ht_refine1_causal COMMAND open_htj2k_dec -i kodim23refine1causal.j2c -o kodim23refine1causal.ppm)
set_tests_properties(dec_ht_refine1_causal PROPERTIES DEPENDS enc_ht_refine1_causal)
add_test(NAME comp_ht_refine1_causal COMMAND imgcmp kodim23refine1causal.ppm ${ENCODER_REF_DIR}/kodim23.ppm 3 0.06)
set_tests_properties(comp_ht_refine1_causal PROPERTIES DEPENDS dec_ht_refine1_causal)

# Code-blocks 4 samples wide (256 stripes of four rows, each a single coding
# group) or 4 high (one stripe of 1024 columns, the widest a code-block can be).
# The images are dense noise, so every sample of the refined plane is next to a
# significant one and the round trip is exact.
foreach(_shape tall wide)
  if(_shape STREQUAL "tall")
    set(_cblk "Cblk={1024,4}")
  else()
    set(_cblk "Cblk={4,1024}")
  endif()
  set(_src ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/narrow_cblk_${_shape}.pgm)
  set(_t ht_refine1_${_shape})
  add_test(NAME enc_${_t} COMMAND open_htj2k_enc -i ${_src} -o ${_t}.j2c Clevels=0 ${_cblk} Creversible=yes Cht_refine=1)
  add_test(NAME dec_${_t} COMMAND open_htj2k_dec -i ${_t}.j2c -o ${_t}.pgm)
  set_tests_properties(dec_${_t} PROPERTIES DEPENDS enc_${_t})
  add_test(NAME comp_${_t} COMMAND imgcmp ${_t}_00.pgm ${_src} 0 0)
  set_tests_properties(comp_${_t} PROPERTIES DEPENDS dec_${_t})
endforeach()

# The refinement passes are coded on the encoder's worker threads: a single-
# threaded encode must give the same codestream.
add_test(NAME enc_ht_refine1_st COMMAND open_htj2k_enc -i ${ENCODER_REF_DIR}/kodim23.ppm -o kodim23refine1_st.j2c Creversible=yes Cht_refine=1 -num_threads 1)
add_test(NAME cmp_ht_refine1_st COMMAND ${CMAKE_COMMAND} -E compare_files kodim23refine1.j2c kodim23refine1_st.j2c)
set_tests_properties(cmp_ht_refine1_st PROPERTIES DEPENDS "enc_ht_refine1;enc_ht_refine1_st")

# Refinement passes exist only in the HT block coder: Cht_refine with Cmodes=0 is refused.
add_test(NAME enc_ht_refine_part1_rejected COMMAND open_htj2k_enc -i ${ENCODER_REF_DIR}/kodim23.ppm -o kodim23refine_part1.j2c Cmodes=0 Cht_refine=1)
set_tests_properties(enc_ht_refine_part1_rejected PROPERTIES WILL_FAIL TRUE)

# Tiles and image offsets that move subbands off the strip pipeline of the streaming
# encoder: subbands starting on an odd row (they used to crash), or between code-block rows
# (they used to be coded wrong, even lossless, and the decoder rejected such codestreams:
# its strips did not follow the code-block grid either).  Lossless round trips must be
# exact; the lossy ones measured 2/0.164 and 2/0.163.
foreach(_case "tiles96;Stiles={96,256};Creversible=yes;0;0" "tiles96_lossy;Stiles={96,256};Qstep=0.002;4;0.25"
              "tiles200;Stiles={200,300};Creversible=yes;0;0" "tiles200_lossy;Stiles={200,300};Qstep=0.002;4;0.25"
              "tiles136;Stiles={136,768};Creversible=yes;0;0"
              "origin_y;Sorigin={1,0};Creversible=yes;0;0"
              "origin_tiles;Sorigin={5,3} Stiles={100,77};Creversible=yes;0;0"
              "origin_x;Sorigin={0,1};Creversible=yes;0;0"
              "origin_tiles64;Sorigin={33,17} Stiles={64,64};Creversible=yes;0;0"
              "origin_precincts;Sorigin={7,0} Cprecincts={64,64};Creversible=yes;0;0")
  list(GET _case 0 _t)
  list(GET _case 1 _layout)
  list(GET _case 2 _mode)
  list(GET _case 3 _pae)
  list(GET _case 4 _mse)
  separate_arguments(_layout)
  add_test(NAME enc_${_t} COMMAND open_htj2k_enc -i ${ENCODER_REF_DIR}/kodim23.ppm -o kodim23_${_t}.j2c ${_layout} ${_mode})
  add_test(NAME dec_${_t} COMMAND open_htj2k_dec -i kodim23_${_t}.j2c -o kodim23_${_t}.ppm)
  set_tests_properties(dec_${_t} PROPERTIES DEPENDS enc_${_t})
  add_test(NAME comp_${_t} COMMAND imgcmp kodim23_${_t}.ppm ${ENCODER_REF_DIR}/kodim23.ppm ${_pae} ${_mse})
  set_tests_properties(comp_${_t} PROPERTIES DEPENDS dec_${_t})
endforeach()

# Rate control (EXPERIMENTAL): -rate cuts the HT code-blocks so that the codestream
# takes at most rate x image area / 8 bytes.  The search for the budget lands within a
# few bytes of the target; the size checks allow 0.5% below it.  Measured PAE/MSE:
# 18/4.38 at 1 bpp, 70/23.3 at 0.25 bpp, 41/11.1 at 0.5 bpp with 256x256 tiles and
# 32/6.93 at 0.7 bpp on the odd-size image; the thresholds leave room for the small
# floating-point differences between instruction sets.
set(_size_check ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_file_size.cmake)
foreach(_case "rate1;kodim23;1;49152;48906;25;5.3" "rate025;kodim23;0.25;12288;12226;95;28"
              "rate05_tiles;kodim23;0.5;24576;24453;70;13.5" "rate07_odd;kodim23odd;0.7;34204;34033;42;8.4")
  list(GET _case 0 _t)
  list(GET _case 1 _img)
  list(GET _case 2 _rate)
  list(GET _case 3 _max)
  list(GET _case 4 _min)
  list(GET _case 5 _pae)
  list(GET _case 6 _mse)
  set(_extra "")
  if(_t STREQUAL "rate05_tiles")
    set(_extra "Stiles={256,256}")
  endif()
  add_test(NAME enc_${_t} COMMAND open_htj2k_enc -i ${ENCODER_REF_DIR}/${_img}.ppm -o ${_t}.j2c -rate ${_rate} ${_extra})
  add_test(NAME size_${_t} COMMAND ${CMAKE_COMMAND} -DFILE=${_t}.j2c -DMAX=${_max} -DMIN=${_min} -P ${_size_check})
  set_tests_properties(size_${_t} PROPERTIES DEPENDS enc_${_t})
  add_test(NAME dec_${_t} COMMAND open_htj2k_dec -i ${_t}.j2c -o ${_t}.ppm)
  set_tests_properties(dec_${_t} PROPERTIES DEPENDS enc_${_t})
  add_test(NAME comp_${_t} COMMAND imgcmp ${_t}.ppm ${ENCODER_REF_DIR}/${_img}.ppm ${_pae} ${_mse})
  set_tests_properties(comp_${_t} PROPERTIES DEPENDS dec_${_t})
endforeach()

# The cut does not depend on the number of threads.
add_test(NAME enc_rate1_st COMMAND open_htj2k_enc -i ${ENCODER_REF_DIR}/kodim23.ppm -o rate1_st.j2c -rate 1 -num_threads 1)
add_test(NAME cmp_rate1_st COMMAND ${CMAKE_COMMAND} -E compare_files rate1.j2c rate1_st.j2c)
set_tests_properties(cmp_rate1_st PROPERTIES DEPENDS "enc_rate1;enc_rate1_st")

# A target size needs the irreversible wavelet and a base step size.
add_test(NAME enc_rate_reversible_rejected COMMAND open_htj2k_enc -i ${ENCODER_REF_DIR}/kodim23.ppm -o rate_rev.j2c -rate 1 Creversible=yes)
set_tests_properties(enc_rate_reversible_rejected PROPERTIES WILL_FAIL TRUE)
add_test(NAME enc_rate_qfactor_rejected COMMAND open_htj2k_enc -i ${ENCODER_REF_DIR}/kodim23.ppm -o rate_qf.j2c -rate 1 Qfactor=80)
set_tests_properties(enc_rate_qfactor_rejected PROPERTIES WILL_FAIL TRUE)
# Rate control with the Part 1 block coder: every coding pass is a truncation point, in any
# mode.  Measured PAE/MSE: 15/4.00 at 1 bpp, 59/20.2 at 0.25 bpp, 30/9.91 at 0.5 bpp with
# BYPASS|ERTERM and 256x256 tiles, 26/6.63 at 0.7 bpp with RESTART|CAUSAL on the odd-size
# image.
foreach(_case "p1rate1;kodim23;1;49152;48906;22;5;0" "p1rate025;kodim23;0.25;12288;12226;80;24;0"
              "p1rate05_bypass_tiles;kodim23;0.5;24576;24453;45;12;BYPASS|ERTERM"
              "p1rate07_odd;kodim23odd;0.7;34204;34033;38;8;RESTART|CAUSAL")
  list(GET _case 0 _t)
  list(GET _case 1 _img)
  list(GET _case 2 _rate)
  list(GET _case 3 _max)
  list(GET _case 4 _min)
  list(GET _case 5 _pae)
  list(GET _case 6 _mse)
  list(GET _case 7 _modes)
  set(_extra "")
  if(_t STREQUAL "p1rate05_bypass_tiles")
    set(_extra "Stiles={256,256}")
  endif()
  add_test(NAME enc_${_t} COMMAND open_htj2k_enc -i ${ENCODER_REF_DIR}/${_img}.ppm -o ${_t}.j2c -rate ${_rate} Cmodes=${_modes} ${_extra})
  add_test(NAME size_${_t} COMMAND ${CMAKE_COMMAND} -DFILE=${_t}.j2c -DMAX=${_max} -DMIN=${_min} -P ${_size_check})
  set_tests_properties(size_${_t} PROPERTIES DEPENDS enc_${_t})
  add_test(NAME dec_${_t} COMMAND open_htj2k_dec -i ${_t}.j2c -o ${_t}.ppm)
  set_tests_properties(dec_${_t} PROPERTIES DEPENDS enc_${_t})
  add_test(NAME comp_${_t} COMMAND imgcmp ${_t}.ppm ${ENCODER_REF_DIR}/${_img}.ppm ${_pae} ${_mse})
  set_tests_properties(comp_${_t} PROPERTIES DEPENDS dec_${_t})
endforeach()
add_test(NAME enc_p1rate1_st COMMAND open_htj2k_enc -i ${ENCODER_REF_DIR}/kodim23.ppm -o p1rate1_st.j2c -rate 1 Cmodes=0 -num_threads 1)
add_test(NAME cmp_p1rate1_st COMMAND ${CMAKE_COMMAND} -E compare_files p1rate1.j2c p1rate1_st.j2c)
set_tests_properties(cmp_p1rate1_st PROPERTIES DEPENDS "enc_p1rate1;enc_p1rate1_st")

# Quality layers (Part 1): one rate per layer, layer l and the layers below it within the
# l-th lowest.  The codestream takes at most the highest rate's bytes, and the encoder
# warns if a lower layer takes more than its own target, which fails the test.  Decodes
# that stop at a lower layer were checked against other decoders outside this suite.
# Measured PAE/MSE (all layers): 15/4.03 at 0.25,0.5,1 bpp; 11/2.64 at five rates up to
# 1.6 bpp with BYPASS|ERTERM and 256x256 tiles; 29/8.15 at 0.2,0.4,0.7 bpp with
# RESTART|CAUSAL, RPCL, precincts and SOP/EPH markers on the odd-size image.
foreach(_case "p1layers3;kodim23;0.25,0.5,1;49152;48906;22;5;0;"
              "p1layers5_bypass_tiles;kodim23;0.1,0.2,0.4,0.8,1.6;78643;78250;20;3.5;BYPASS|ERTERM;Stiles={256,256}"
              "p1layers3_odd_rpcl;kodim23odd;0.2,0.4,0.7;34204;34033;40;9.5;RESTART|CAUSAL;Corder=RPCL Cprecincts={128,128},{64,64} Cuse_sop=yes Cuse_eph=yes")
  list(GET _case 0 _t)
  list(GET _case 1 _img)
  list(GET _case 2 _rate)
  list(GET _case 3 _max)
  list(GET _case 4 _min)
  list(GET _case 5 _pae)
  list(GET _case 6 _mse)
  list(GET _case 7 _modes)
  list(GET _case 8 _extra)
  separate_arguments(_extra)
  add_test(NAME enc_${_t} COMMAND open_htj2k_enc -i ${ENCODER_REF_DIR}/${_img}.ppm -o ${_t}.j2c -rate ${_rate} Cmodes=${_modes} ${_extra})
  set_tests_properties(enc_${_t} PROPERTIES FAIL_REGULAR_EXPRESSION "WARNING")
  add_test(NAME size_${_t} COMMAND ${CMAKE_COMMAND} -DFILE=${_t}.j2c -DMAX=${_max} -DMIN=${_min} -P ${_size_check})
  set_tests_properties(size_${_t} PROPERTIES DEPENDS enc_${_t})
  add_test(NAME dec_${_t} COMMAND open_htj2k_dec -i ${_t}.j2c -o ${_t}.ppm)
  set_tests_properties(dec_${_t} PROPERTIES DEPENDS enc_${_t})
  add_test(NAME comp_${_t} COMMAND imgcmp ${_t}.ppm ${ENCODER_REF_DIR}/${_img}.ppm ${_pae} ${_mse})
  set_tests_properties(comp_${_t} PROPERTIES DEPENDS dec_${_t})
  # The cut does not depend on the number of threads.
  add_test(NAME enc_${_t}_st COMMAND open_htj2k_enc -i ${ENCODER_REF_DIR}/${_img}.ppm -o ${_t}_st.j2c -rate ${_rate} Cmodes=${_modes} ${_extra} -num_threads 1)
  add_test(NAME cmp_${_t}_st COMMAND ${CMAKE_COMMAND} -E compare_files ${_t}.j2c ${_t}_st.j2c)
  set_tests_properties(cmp_${_t}_st PROPERTIES DEPENDS "enc_${_t};enc_${_t}_st")
endforeach()

# Several quality layers need the Part 1 block coder: with HT, a list of rates is rejected.
add_test(NAME enc_rate_list_rejected COMMAND open_htj2k_enc -i ${ENCODER_REF_DIR}/kodim23.ppm -o rate_list.j2c -rate 1,0.5)
set_tests_properties(enc_rate_list_rejected PROPERTIES WILL_FAIL TRUE)

# Command-line values that are not numbers, out of range or too large for the reference grid
# end with an error message and a non-zero exit code; std::stoi and its relatives used to end
# the program with an uncaught exception (SIGABRT), and a huge thread count took all memory.
set(_enc_fail ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_enc_exit_failure.cmake)
set(_enc_src ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/dwt_levels_src.pgm)
foreach(_case "levels_not_a_number;Clevels=abc;takes an integer"
              "levels_too_large;Clevels=99999999999;takes an integer"
              "tiles_negative;Stiles={-1,-1};takes an integer"
              "origin_off_the_grid;Sorigin={0,4294967234};exceeds the reference grid"
              "precincts_not_a_number;Cprecincts={x,y};takes an integer"
              "qstep_not_a_number;Qstep=nan;shall be in the range"
              "qzoom_not_a_number;Qzoom=abc;shall be in the range"
              "threads_too_many;-num_threads;100000;takes an integer from 0 to 1024"
              "order_with_suffix;Corder=LRCPX;unknown progression order"
              "precincts_too_small;Cprecincts={1,1};precinct size exponents")
  list(GET _case 0 _t)
  list(REMOVE_AT _case 0)
  list(GET _case -1 _expect)
  list(REMOVE_AT _case -1)
  add_test(NAME enc_cli_${_t} COMMAND ${CMAKE_COMMAND} -DENC=$<TARGET_FILE:open_htj2k_enc>
           "-DARGS=-i;${_enc_src};-o;cli_${_t}.j2c;${_case}" "-DEXPECT=${_expect}" -P ${_enc_fail})
endforeach()
# The irreversible transform needs a guard bit (without one, coefficients can need more
# magnitude bit-planes than the codestream gives them).
add_test(NAME enc_cli_irreversible_guard0 COMMAND ${CMAKE_COMMAND} -DENC=$<TARGET_FILE:open_htj2k_enc>
         "-DARGS=-i;${_enc_src};-o;cli_guard0.j2c;Qguard=0" "-DEXPECT=at least 1 guard bit" -P ${_enc_fail})

# Several PGM files give one component each (only the first used to be read): a 12-bit and a
# 16-bit 67x37 image, coded losslessly as two components.
add_test(NAME enc_two_pgm COMMAND open_htj2k_enc -i ${GRAY12_PGM},${GRAY16_PGM} -o two_pgm.j2c Creversible=yes)
add_test(NAME dec_two_pgm COMMAND open_htj2k_dec -i two_pgm.j2c -o two_pgm.pgx)
set_tests_properties(dec_two_pgm PROPERTIES DEPENDS enc_two_pgm)
add_test(NAME comp_two_pgm_0 COMMAND imgcmp two_pgm_00.pgx ${GRAY12_PGM} 0 0)
add_test(NAME comp_two_pgm_1 COMMAND imgcmp two_pgm_01.pgx ${GRAY16_PGM} 0 0)
set_tests_properties(comp_two_pgm_0 comp_two_pgm_1 PROPERTIES DEPENDS dec_two_pgm)
