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
  PROPERTIES FIXTURES_REQUIRED test_artifacts)
