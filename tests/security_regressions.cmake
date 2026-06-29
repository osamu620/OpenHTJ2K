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

# Threaded decode of malformed input must fail safely.
# The fixture is a synthetic 64x64 HT codestream with one byte corrupted in a
# code-block's MagSgn region.  In the line-based streaming path each code-block
# decode runs as a task on a thread-pool worker.  ThreadPool::worker() used to
# invoke tasks with no try/catch, so a throw from htj2k_decode either reached
# std::terminate() or escaped before the in-flight task counter was decremented,
# leaving the streaming driver spinning on its completion barrier forever (a
# hang).  The fix catches the throw in the task, records a sticky par_error and
# always decrements the counter, then re-throws on the driver thread after the
# barrier — matching the clean single-threaded error path.
#
# This regression is registered on x86 only.  Reaching the worker exception path
# the fix protects requires the corrupted code-block to actually throw, which it
# does on x86 (the AVX2 reader throws unconditionally on the over-read).  Other
# ISAs respond to the same bytes differently — the NEON reader clamps the
# over-read in a Release build (decode finishes with garbage), so the fixture
# would neither exercise the fix nor reproduce the failure there.  On x86 the
# patched decoder prints its worker-catch log line and exits cleanly; a reverted
# fix hangs (caught by TIMEOUT) or aborts (caught by the crash regex).
# Single-threaded decode was always clean, so the threaded run is what matters.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  add_test(NAME security_threaded_decode_abort_mt
           COMMAND open_htj2k_dec
                   -i ${SECURITY_DATA_DIR}/security_threaded_decode_abort.j2k
                   -o security_threaded_decode_abort_mt.pgx
                   -num_threads 4)
  set_tests_properties(security_threaded_decode_abort_mt PROPERTIES
      PASS_REGULAR_EXPRESSION "code-block decode task failed"
      FAIL_REGULAR_EXPRESSION "${_SEC_CRASH_RE}"
      TIMEOUT 60)

  # Multi-component (color) counterpart.  The grayscale case above is decoded
  # with a serial strip pull, so its code-block throw unwinds straight to the
  # interface catch.  With NC>1 the streaming driver pulls each component's
  # strip on a thread-pool worker (can_parallel_pull), so the throw must also be
  # caught at that strip-pull boundary — otherwise it escapes the worker and
  # reaches std::terminate().  This fixture is a 3-component HT stream with one
  # corrupted code-block byte; the patched driver prints the strip-pull
  # worker-catch line and exits cleanly, while a decoder missing that catch
  # aborts (crash regex) or hangs (TIMEOUT).  x86-only for the same reason as
  # the grayscale case (the over-read only throws on the AVX2 reader).
  add_test(NAME security_threaded_decode_abort_color_mt
           COMMAND open_htj2k_dec
                   -i ${SECURITY_DATA_DIR}/security_threaded_decode_abort_color.j2k
                   -o security_threaded_decode_abort_color_mt.pgx
                   -num_threads 4)
  set_tests_properties(security_threaded_decode_abort_color_mt PROPERTIES
      PASS_REGULAR_EXPRESSION "strip-pull decode task failed"
      FAIL_REGULAR_EXPRESSION "${_SEC_CRASH_RE}"
      TIMEOUT 60)
endif()

# Codestream allocation bound (j2c_src_memory::alloc_memory / borrow_memory).
# A codestream length that cannot be represented by the uint32_t buffer
# accounting once the 16-byte over-read pad is added must be rejected, not
# wrapped to a tiny allocation (heap overflow on the following copy).  This is
# pure integer-bound logic, so it runs on every platform with no large input
# (rejected lengths throw before any allocation).  The tool returns non-zero
# on any unguarded case.
add_test(NAME security_codestream_alloc_bound COMMAND codestream_bounds_check)
set_tests_properties(security_codestream_alloc_bound PROPERTIES
    PASS_REGULAR_EXPRESSION "all cases passed"
    FAIL_REGULAR_EXPRESSION "${_SEC_CRASH_RE}"
    TIMEOUT 30)
