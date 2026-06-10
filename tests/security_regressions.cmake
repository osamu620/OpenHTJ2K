# Security regression tests.
#
# Each fixture lives under tests/data/ and exercises a code path that
# previously crashed on a malformed input.  A passing test means the
# fixed decoder rejects the input cleanly (non-zero exit, no signal,
# no sanitizer hit) — i.e. the bug does not re-emerge.
#
# Add new fixtures by:
#   1. dropping the file into tests/data/<slug>.j2k (small is better;
#      any extra bytes that don't trigger the bug should be trimmed)
#   2. appending an add_test() block here with WILL_FAIL TRUE plus a
#      FAIL_REGULAR_EXPRESSION that matches platform crash markers —
#      the test must distinguish "graceful reject" from "crashed".

set(SECURITY_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tests/data")

# Regex matching evidence of a crash rather than an orderly rejection.
# Covers: libc stack canary trips, ASan/UBSan fires, kernel SIGSEGV /
# SIGABRT text across Linux + macOS + Windows run-time formats.
set(_SEC_CRASH_RE
    "stack smashing detected|AddressSanitizer|UndefinedBehaviorSanitizer|Segmentation fault|SIGSEGV|SIGABRT|Abort trap|Aborted \\(core dumped\\)|core dumped|terminate called after throwing")

# CVE / GHSA pending — HT block decoding stack buffer overflow.
# Reported by IM JUN SEO (KISIA, @Jseanxx) and OH HAN GUEL
# (SANGMYUNG UNIVERSITY, @5asever40-a11y).
# SHA-256: da756e7aa7421e1207b92b7651996ed4b328edb820bd30cdb5f4af953d1eb063
#
# Assertion model: codeblock-level decode failures are logged as
# warnings and do NOT propagate to the decoder's top-level exit
# code, so we can't rely on WILL_FAIL.  Instead, require that the
# specific guard printf from the patched code path fires (proof the
# bounds check is in effect) and reject any crash-marker text.
add_test(NAME security_ht_segments_overflow
         COMMAND open_htj2k_dec
                 -i ${SECURITY_DATA_DIR}/security_ht_segments_overflow.j2k
                 -o security_ht_segments_overflow.pgx)
set_tests_properties(security_ht_segments_overflow PROPERTIES
    PASS_REGULAR_EXPRESSION "too many HT coding-pass segments"
    FAIL_REGULAR_EXPRESSION "${_SEC_CRASH_RE}"
    TIMEOUT 60)

# API contract — the decoder's file-path constructor used to call
# exit(EXIT_FAILURE) directly when the file was missing, terminating
# the host process from inside the shared library.  v0.15.2 makes it
# throw std::runtime_error instead.  This test runs open_htj2k_dec
# against a nonexistent path and asserts the CLI exits cleanly with
# the propagated "input file not found" message — no abort, no
# signal, no orphan process.
add_test(NAME api_decoder_throws_on_missing_file
         COMMAND open_htj2k_dec
                 -i ${SECURITY_DATA_DIR}/path/that/does/not/exist.j2k
                 -o api_decoder_missing_file.pgx)
set_tests_properties(api_decoder_throws_on_missing_file PROPERTIES
    PASS_REGULAR_EXPRESSION "input file not found"
    FAIL_REGULAR_EXPRESSION "${_SEC_CRASH_RE}"
    TIMEOUT 30)

# Valid-input regression (opposite polarity from the fixtures above: this
# input is a CONFORMING codestream that must DECODE, not be rejected).
# SP_dec::importSigPropBit used to validate the bit-stuffing position (the
# MSB of a byte following 0xFF) and threw when it was 1.  T.814 F.4 permits
# refinement-segment terminations that overlap the SigProp and MagRef
# byte-streams (termSPandMRPackers NOTE), so a tail byte read by the SigProp
# reader can legitimately carry a MagRef bit there (T.814 7.1.5 NOTE 2).
# On this fixture the old code aborted under multi-threaded decode and
# silently wrote a truncated image (with exit code 0) single-threaded.
# The imgcmp steps catch both regressions: a reintroduced throw aborts the
# decode test, and a truncated or corrupted plane fails the comparison.
add_test(NAME security_sigprop_refinement_overlap_mt
         COMMAND open_htj2k_dec
                 -i ${SECURITY_DATA_DIR}/sigprop_refinement_overlap.j2k
                 -o sigprop_refinement_overlap_mt.pgx
                 -num_threads 2)
set_tests_properties(security_sigprop_refinement_overlap_mt PROPERTIES
    FAIL_REGULAR_EXPRESSION "${_SEC_CRASH_RE}|importSigPropBit"
    TIMEOUT 60)

add_test(NAME security_sigprop_refinement_overlap
         COMMAND open_htj2k_dec
                 -i ${SECURITY_DATA_DIR}/sigprop_refinement_overlap.j2k
                 -o sigprop_refinement_overlap.pgx
                 -num_threads 1)
set_tests_properties(security_sigprop_refinement_overlap PROPERTIES
    FAIL_REGULAR_EXPRESSION "${_SEC_CRASH_RE}|importSigPropBit"
    TIMEOUT 60)
foreach(_comp 00 01 02)
  add_test(NAME comp_sigprop_refinement_overlap_${_comp}
           COMMAND imgcmp sigprop_refinement_overlap_${_comp}.pgx
                   ${SECURITY_DATA_DIR}/sigprop_refinement_overlap_ref_${_comp}.pgx 2 0.01)
  set_tests_properties(comp_sigprop_refinement_overlap_${_comp} PROPERTIES
      DEPENDS security_sigprop_refinement_overlap)
endforeach()
