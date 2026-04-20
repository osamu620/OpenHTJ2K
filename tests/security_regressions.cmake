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
    # On Linux + macOS this fixture decodes in <5 ms once the guard
    # short-circuits the bad codeblock.  Windows CI has been observed
    # to hang for >10 minutes on the same input — cause unclear,
    # probably an unrelated pathological path the malformed stream
    # reaches downstream of the fixed overflow.  Cap the test at 60 s
    # so CI fails fast; the separate investigation is tracked without
    # blocking the v0.15.1 advisory.
    TIMEOUT 60)
