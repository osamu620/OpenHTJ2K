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
# The fixture is a grab-bag of several malformed code-blocks, each of which
# trips a different htj2k_decode bounds/segment guard (the >4-segment stack
# write among them).  Those guards return false rather than throwing; the
# streaming decoder used to ignore that return and keep decoding, so a
# malformed block silently produced a garbage (or uninitialised) plane while
# the process exited 0.  The decoder now treats a false return like a thrown
# error and fails fast at the first rejected block: the serial path throws from
# decode_strip_core and the threaded path records par_error and re-throws at the
# strip barrier.  These are scalar prologue guards, so the propagation is
# platform-independent (not arch-gated).
#
# Assertion model: the CLI catches the decode exception and still exits 0, so
# WILL_FAIL is not usable; instead require the fail-fast propagation message
# (which appears only once the false return is honoured — a reverted fix prints
# neither) and reject any crash marker.  The FAIL regex is the load-bearing
# security assertion — the reported PoC must never stack-smash — and it holds
# because the block is now rejected instead of decoded.
add_test(NAME security_ht_segments_overflow
         COMMAND open_htj2k_dec
                 -i ${SECURITY_DATA_DIR}/security_ht_segments_overflow.j2k
                 -o security_ht_segments_overflow.pgx
                 -num_threads 1)
set_tests_properties(security_ht_segments_overflow PROPERTIES
    PASS_REGULAR_EXPRESSION "HT code-block decoding reported failure"
    FAIL_REGULAR_EXPRESSION "${_SEC_CRASH_RE}"
    TIMEOUT 60)

# Threaded counterpart: the same fixture under -num_threads 4 exercises the
# pool-worker path, where a code-block decode failure is recorded as par_error
# and re-thrown at the completion barrier (decode_strip_core / trigger_prefetch)
# rather than thrown inline.  A tiny image may still fall back to the serial
# decode, so the PASS regex accepts either the barrier message or the inline
# throw; either way a reverted fix prints neither and the test fails.
add_test(NAME security_ht_segments_overflow_mt
         COMMAND open_htj2k_dec
                 -i ${SECURITY_DATA_DIR}/security_ht_segments_overflow.j2k
                 -o security_ht_segments_overflow_mt.pgx
                 -num_threads 4)
set_tests_properties(security_ht_segments_overflow_mt PROPERTIES
    PASS_REGULAR_EXPRESSION "code-block decode task failed|HT code-block decoding reported failure"
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

# Marker-segment length underflow.  A variable-length marker whose declared
# length (Lmar) is smaller than its fixed fields makes a `for (i < Lmar - len)`
# loop (Lmar/len are uint16_t) wrap to a ~1.8e19 count; the marker readers
# (get_byte/get_word) then run off the end of the buffer.  The fixture is a
# codestream whose COM marker carries Lmar=3 (< its 4-byte fixed length); the
# patched reader rejects the over-long read instead of crashing/looping.
# Marker parsing is platform-independent scalar code, so the positive check is
# not arch-gated.  A reverted fix runs off the buffer (crash) or spins (TIMEOUT).
add_test(NAME security_marker_length_underflow
         COMMAND open_htj2k_dec
                 -i ${SECURITY_DATA_DIR}/security_marker_com_underflow.j2k
                 -o security_marker_com_underflow.pgx
                 -num_threads 1)
set_tests_properties(security_marker_length_underflow PROPERTIES
    PASS_REGULAR_EXPRESSION "past its declared length"
    FAIL_REGULAR_EXPRESSION "${_SEC_CRASH_RE}"
    TIMEOUT 30)

# COD decomposition-level count out of range.  SPcod[0] (NL) is capped at 32 by
# the spec, but was read unmasked (0-255); NL>=33 indexes SPcod[5+r] out of
# bounds and NL=255 makes setCODparams' `for (uint8_t r=0; r<=NL; r++)` loop
# never terminate (growing precinct_size without bound).  The fixture sets a
# COD's NL byte to 255; the patched parser rejects it at parse time.  A reverted
# fix loops/OOMs (TIMEOUT) or reads out of bounds (crash regex).
add_test(NAME security_marker_cod_levels
         COMMAND open_htj2k_dec
                 -i ${SECURITY_DATA_DIR}/security_marker_cod_levels.j2k
                 -o security_marker_cod_levels.pgx
                 -num_threads 1)
set_tests_properties(security_marker_cod_levels PROPERTIES
    PASS_REGULAR_EXPRESSION "decomposition levels 255 exceeds 32"
    FAIL_REGULAR_EXPRESSION "${_SEC_CRASH_RE}"
    TIMEOUT 30)

# Code-block length spanning a tile-part (buf_chain node) boundary.
# create_compressed_buffer clamps a code-block's borrowed byte count, but the
# bound used the cross-node total (get_remaining_bytes summed every later
# tile-part) while the borrow/copy hand back a contiguous span from the current
# node only.  On a tile split into multiple tile-parts an inflated packet-header
# length therefore slipped the clamp and the code-block read the following
# tile-part's SOT/SOD + body bytes as its own (in-bounds of the single
# contiguous codestream buffer, so not an out-of-bounds access, but wrong) and
# let the HT modDcup write 0xFF across the node boundary.  The fix bounds the
# length by the current tile-part's remaining bytes, matching the single-node
# invariant the borrow/copy asserts already document; malformed input now fails
# fast at the clamp on every platform instead of decoding cross-tile-part bytes.
#
# The fixture is a single-resolution (one-packet) stream whose tile is split so
# the lone code-block straddles the tile-part boundary.  The clamp is scalar, so
# its warning is platform-independent (not arch-gated), and the one packet ends
# right after the clamped block — the HT cleanup-length guard (present in every
# decoder variant) rejects the truncated block without entropy-decoding garbage,
# so there is no platform-divergent tail.  A reverted fix prints no warning
# (PASS regex absent) and, on other ISAs, decodes the cross-tile-part bytes.
add_test(NAME security_multitilepart_cblk_length
         COMMAND open_htj2k_dec
                 -i ${SECURITY_DATA_DIR}/security_multitilepart_cblk_length.j2k
                 -o security_multitilepart_cblk_length.pgm
                 -num_threads 1)
set_tests_properties(security_multitilepart_cblk_length PROPERTIES
    PASS_REGULAR_EXPRESSION "left in tile-part"
    FAIL_REGULAR_EXPRESSION "${_SEC_CRASH_RE}"
    TIMEOUT 30)

# Line-based streaming setup throwing mid-init must not crash the finalize guard.
# init_line_decode() builds each component's IDWT state: it allocates j2k_tcomp_line_dec,
# sets NL_active, then validates the per-level DWT directions and allocates the
# per-level state arrays (states/ctxs/hl/lh/hh).  decode_line_based_stream() guards
# the whole decode with a LineDecodeFinalizeGuard whose destructor runs
# finalize_line_decode() on every unwind, including when init_line_decode() throws.
# finalize_line_decode() drains/frees those per-level arrays for each index
# < NL_active -- so when init_line_decode() threw AFTER publishing NL_active but
# BEFORE the arrays existed (the DWT_VERT rejection, or a std::bad_alloc), the guard
# dereferenced still-null arrays and the process crashed (SIGSEGV) while unwinding.
# The fix publishes NL_active only after the arrays are allocated, so a throw before
# then leaves NL_active==0 and finalize touches nothing.
#
# The fixture is a 115-byte Part-2 codestream: a 32x32 single-component image whose
# COC marks the component as DFS-driven and whose DFS marker declares the sole
# decomposition level VERTICAL (HONLY/VONLY).  The line-based path does not support
# DWT_VERT and rejects it in init_line_decode().  The tile body is two empty packets
# (0x00), so packet parsing succeeds and the decode reaches the IDWT setup that
# throws.  This is platform-independent scalar marker/IDWT-setup logic (no entropy
# decoding), so it is not arch-gated and runs both single- and multi-threaded -- the
# guard's drain path is compiled in whenever threads are available, so both runs hit
# it.  The patched decoder prints the rejection and exits cleanly; a reverted fix
# crashes while unwinding (before the message is printed), failing the PASS regex and
# tripping the crash regex.
add_test(NAME security_line_decode_dfs_vert
         COMMAND open_htj2k_dec
                 -i ${SECURITY_DATA_DIR}/security_line_decode_dfs_vert.j2k
                 -o security_line_decode_dfs_vert.pgx
                 -num_threads 1)
set_tests_properties(security_line_decode_dfs_vert PROPERTIES
    PASS_REGULAR_EXPRESSION "does not support DWT_VERT"
    FAIL_REGULAR_EXPRESSION "${_SEC_CRASH_RE}"
    TIMEOUT 30)

add_test(NAME security_line_decode_dfs_vert_mt
         COMMAND open_htj2k_dec
                 -i ${SECURITY_DATA_DIR}/security_line_decode_dfs_vert.j2k
                 -o security_line_decode_dfs_vert_mt.pgx
                 -num_threads 4)
set_tests_properties(security_line_decode_dfs_vert_mt PROPERTIES
    PASS_REGULAR_EXPRESSION "does not support DWT_VERT"
    FAIL_REGULAR_EXPRESSION "${_SEC_CRASH_RE}"
    TIMEOUT 30)
