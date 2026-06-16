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
  PROPERTIES FIXTURES_REQUIRED test_artifacts)