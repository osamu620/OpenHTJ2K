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
