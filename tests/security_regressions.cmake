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
# Assertion model: WILL_FAIL would also pass a crash, so require the fail-fast
# propagation message instead (it appears only once the false return is honoured
# — a reverted fix prints neither) and reject any crash marker.  The FAIL regex
# is the load-bearing security assertion — the reported PoC must never
# stack-smash — and it holds because the block is now rejected instead of decoded.
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
# This regression is registered only when the AVX2 block-decoder is compiled
# in.  Reaching the worker exception path the fix protects requires the
# corrupted code-block to actually throw, which only the AVX2 reader does
# (unconditionally, on the over-read).  Every other reader responds to the
# same bytes differently — the NEON and scalar readers clamp the over-read in
# a Release build (decode finishes with garbage), so the fixture would
# neither exercise the fix nor reproduce the failure there.  That makes a
# processor check the wrong gate: an x86_64 build with -DENABLE_AVX2=OFF
# clamps like NEON and used to fail these tests.  Gate on the AVX2 reader
# actually being built instead (ENABLE_AVX2 plus __AVX2__ under the active
# compile flags, mirroring the OPENHTJ2K_TRY_AVX2 && __AVX2__ source gate).
# With the AVX2 reader, the patched decoder prints its worker-catch log line
# and exits cleanly; a reverted fix hangs (caught by TIMEOUT) or aborts
# (caught by the crash regex).  Single-threaded decode was always clean, so
# the threaded run is what matters.
if(ENABLE_AVX2)
  include(CheckCXXSourceCompiles)
  check_cxx_source_compiles("#ifndef __AVX2__\n#error \"AVX2 not enabled\"\n#endif\nint main() { return 0; }"
                            OPENHTJ2K_TESTS_HAVE_AVX2_READER)
endif()
if(OPENHTJ2K_TESTS_HAVE_AVX2_READER)
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
  # aborts (crash regex) or hangs (TIMEOUT).  AVX2-reader-only for the same
  # reason as the grayscale case (the over-read only throws on that reader).
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

# Codestream truncated inside a packet header (issue #462, reported by
# @rafaelhutter).  buf_chain::get_byte() pads a short codestream with 0x00 so
# partial decoding still works, but the packet-header tag trees leave their
# loop only on a 1 bit: on the padding, j2k_precinct_subband::parse_packet_header
# spun forever in the zero-bitplane loop (current_value counting up with nothing
# to stop it) and the decoder never returned -- a hang inside the library, on
# the caller's thread.  The reader now records that it went past the end
# (buf_chain::is_past_end) and the parser stops there, dropping the code-blocks
# whose signalling is missing so the tile decodes from the packets that did
# arrive.
#
# The fixture is the first 152 bytes of conformance_data/ds0_ht_09_b11.j2k --
# a cut that lands inside the first packet header.  Any such cut reproduces it;
# this is just the shortest.  Packet-header parsing is platform-independent
# scalar code (no entropy decoding), so the test is not arch-gated, and the
# parse runs on the calling thread either way -- the threaded run additionally
# covers the strip-pull path decoding a tile with dropped code-blocks.  The
# patched decoder prints the truncation warning and exits cleanly; a reverted
# fix never prints it and is caught by TIMEOUT.
add_test(NAME security_truncated_packet_header
         COMMAND open_htj2k_dec
                 -i ${SECURITY_DATA_DIR}/security_truncated_packet_header.j2k
                 -o security_truncated_packet_header.pgx
                 -num_threads 1)
set_tests_properties(security_truncated_packet_header PROPERTIES
    PASS_REGULAR_EXPRESSION "ends inside a packet header"
    FAIL_REGULAR_EXPRESSION "${_SEC_CRASH_RE}"
    TIMEOUT 30)

add_test(NAME security_truncated_packet_header_mt
         COMMAND open_htj2k_dec
                 -i ${SECURITY_DATA_DIR}/security_truncated_packet_header.j2k
                 -o security_truncated_packet_header_mt.pgx
                 -num_threads 4)
set_tests_properties(security_truncated_packet_header_mt PROPERTIES
    PASS_REGULAR_EXPRESSION "ends inside a packet header"
    FAIL_REGULAR_EXPRESSION "${_SEC_CRASH_RE}"
    TIMEOUT 30)

# HT refinement decoding of code-blocks 4 samples wide or high.  The AVX2, NEON
# and WASM decoders pack a code-block's significance into a fixed stack array
# before decoding its HT SigProp and HT MagRef passes.  The array was sized for
# 64 x 64 code-blocks, but a code-block may be up to 1024 samples high or wide
# (4096 samples in all): a 4 x 1024 (width x height) code-block needs 2056
# entries and a 1024 x 4 one 528, against 432, so decoding their refinement
# passes wrote past the array on the stack.  The codestreams are valid.
#
# Each fixture is a 4 x 1024 or 1024 x 4 8-bit grey image coded reversibly as a
# single code-block with no wavelet levels: an HT cleanup pass one bit-plane up,
# then HT SigProp and HT MagRef passes for the LSB, which here reach every
# sample, so the decode equals the source image.  A release build does not show
# the overflow (its output is still right); the UBSan CI job reports it.
foreach(_shape tall wide)
  add_test(NAME security_ht_refine_${_shape}_cblk
           COMMAND open_htj2k_dec
                   -i ${SECURITY_DATA_DIR}/ht_refine_${_shape}_cblk.j2k
                   -o ht_refine_${_shape}_cblk.pgm
                   -num_threads 1)
  set_tests_properties(security_ht_refine_${_shape}_cblk PROPERTIES
      FAIL_REGULAR_EXPRESSION "${_SEC_CRASH_RE}"
      TIMEOUT 30)
  add_test(NAME comp_ht_refine_${_shape}_cblk
           COMMAND imgcmp ht_refine_${_shape}_cblk_00.pgm
                   ${SECURITY_DATA_DIR}/narrow_cblk_${_shape}.pgm 0 0)
  set_tests_properties(comp_ht_refine_${_shape}_cblk PROPERTIES
      DEPENDS security_ht_refine_${_shape}_cblk)
endforeach()

# Main-header and marker validation.  Each fixture below is a malformed
# codestream (security_siz_origin_2p31 is a valid one this decoder does not
# support) that crashed, hung, read or wrote out of bounds, or tried to
# allocate gigabytes when decoded the way OpenCV does it (parse the main
# header, then decode row by row).  The decoder now rejects it with the message
# given.  The PASS regex requires that message (the CLI's exit code is not
# checked) and the FAIL regex rejects any crash marker.  Fixtures whose bug
# involved several tiles also run with 4 threads (test name suffix _mt).
function(_sec_rejects name fixture message)
  set(_threads 1)
  if("mt" IN_LIST ARGN)
    list(APPEND _threads 4)
  endif()
  foreach(_nt ${_threads})
    set(_name ${name})
    if(_nt EQUAL 4)
      set(_name ${name}_mt)
    endif()
    add_test(NAME ${_name}
             COMMAND open_htj2k_dec -i ${SECURITY_DATA_DIR}/${fixture} -o ${_name}.pgx -num_threads ${_nt})
    set_tests_properties(${_name} PROPERTIES
        PASS_REGULAR_EXPRESSION "${message}"
        FAIL_REGULAR_EXPRESSION "${_SEC_CRASH_RE}"
        TIMEOUT 30)
  endforeach()
endfunction()

# An empty input (memcpy from a null pointer) and codestreams without the SOC
# marker or without one of the SIZ, COD and QCD markers every main header needs
# (null pointer dereferences once the decoder asked for the image size or the
# coding style).
_sec_rejects(security_empty_input security_empty_input.j2k "is empty")
_sec_rejects(security_missing_soc security_missing_soc.jph "does not begin with an SOC marker")
_sec_rejects(security_missing_siz security_missing_siz.j2k "main header has no SIZ marker")
_sec_rejects(security_missing_cod security_missing_cod.jph "main header has no COD marker")
_sec_rejects(security_missing_qcd security_missing_qcd.jph "main header has no QCD marker")

# A COC marker too short for its coding-style fields (read past the marker).
_sec_rejects(security_coc_too_short security_coc_too_short.jph "COC marker length 8 is less than 9")

# QCD markers: an undefined quantization style; fewer step sizes than the
# decomposition levels need (read past the step-size array); quantization
# style 0, which has no mantissas, with the 9/7 transform (read from an empty
# mantissa array).
_sec_rejects(security_qcd_invalid_style security_qcd_invalid_style.jph "QCD quantization style 31 is invalid")
_sec_rejects(security_qcd_too_few_steps security_qcd_too_few_steps.j2k
             "needs 10 quantization step sizes for 3 decomposition levels, but its QCD/QCC marker has 7")
_sec_rejects(security_qcd_style0_irreversible security_qcd_style0_irreversible.jph
             "quantization style 0 \\(no quantization\\) with a wavelet transform other than the reversible 5/3")

# RGN markers: an undefined region-of-interest style; a shift of 79 (shifts of
# 32-bit samples by 32 or more); a shift that puts the component's 12 magnitude
# bit-planes past the 31 bits of a sample.
_sec_rejects(security_rgn_style security_rgn_style.jph "RGN region-of-interest style 1 is not supported")
_sec_rejects(security_rgn_shift_too_large security_rgn_shift_too_large.j2k
             "RGN region-of-interest shift 79 exceeds 31")
_sec_rejects(security_rgn_shift_over_bitplanes security_rgn_shift_over_bitplanes.j2k
             "region-of-interest shift 20 on top of 12 magnitude bit-planes")

# PPM markers with no complete packet-header record (an empty record chain was
# read), and with fewer records than tiles (read past the record list).
_sec_rejects(security_ppm_no_record security_ppm_no_record.jph "PPM markers hold no packet-header record")
_sec_rejects(security_ppm_missing_tile_record security_ppm_missing_tile_record.jph
             "PPM markers hold 1 packet-header records, none for tile-part 1" mt)

# SIZ markers breaking the tiling rules of ISO/IEC 15444-1, A.5.1: a zero tile
# size (division by zero), and tile origins after the image origin, which gave
# tiles that end before they start.  These crashed in many places: a 16 GB
# allocation, a hang, negative array indices, a wild memcpy, a read past a
# strip buffer, a multi-tile row scatter writing out of bounds, and rows
# narrower than the image handed to the row callback.
_sec_rejects(security_siz_tile_size_zero security_siz_tile_size_zero.jph "SIZ tile size XTsiz is zero")
_sec_rejects(security_siz_tile_origin security_siz_tile_origin.jph
             "SIZ tile origin XTOsiz=1 lies after the image origin XOsiz=0")
_sec_rejects(security_siz_huge_allocation security_siz_huge_allocation.j2c
             "SIZ tile origin XTOsiz=4284678145 lies after the image origin")
_sec_rejects(security_siz_cpu_hang security_siz_cpu_hang.jph
             "SIZ tile origin YTOsiz=187 lies after the image origin YOsiz=128")
_sec_rejects(security_siz_negative_coord security_siz_negative_coord.jph
             "SIZ tile origin XTOsiz=4294967295 lies after the image origin")
_sec_rejects(security_siz_wild_memcpy security_siz_wild_memcpy.jph
             "SIZ tile origin XTOsiz=4294770688 lies after the image origin")
_sec_rejects(security_siz_strip_overread security_siz_strip_overread.j2c
             "SIZ tile origin XTOsiz=4294967295 lies after the image origin")
_sec_rejects(security_siz_multitile_scatter security_siz_multitile_scatter.j2c
             "SIZ tile origin XTOsiz=4294443008 lies after the image origin" mt)

# A valid codestream whose image area lies past 2^31 on the reference grid
# (origin 2147483649).  The decoder holds sample coordinates in 32-bit signed
# integers, so it rejects reference grids larger than 2147483647 as not
# supported instead of crashing.
_sec_rejects(security_siz_origin_2p31 security_siz_origin_2p31.jph
             "Xsiz=2147483710 exceeds 2147483647: not supported by this decoder")

# A tile declaring 24321 layers of small precincts: 588 million packets in a
# 908-byte codestream, which made the decoder allocate 28 GB of packet state.
# Every packet takes at least one byte, so the decoder now rejects a tile with
# more packets than the codestream can hold before allocating for them.
_sec_rejects(security_packets_exceed_codestream security_packets_exceed_codestream.j2c
             "more than the codestream can hold" mt)

# The colour transform with components 1 and 2 subsampled by 2: the transform
# was sized by component 0 and wrote past the rows of the smaller components.
# A truncated tile-part followed by one or two tile-parts without a body: the packet
# reader stepped from the exhausted node onto an empty one and read through its null
# buffer (SIGSEGV).  Empty nodes are skipped now, and the truncation is reported.
_sec_rejects(security_empty_tile_part security_empty_tile_part.j2c "malformed input" mt)
_sec_rejects(security_two_empty_tile_parts security_two_empty_tile_parts.j2c "malformed input" mt)

_sec_rejects(security_mct_subsampled security_mct_subsampled.j2k
             "uses the colour transform, but component 1 is subsampled differently" mt)
# The fixtures below come from fuzzing the streaming decode (invoke_line_based_stream,
# the path image I/O libraries use) and run single- and multi-threaded.  Each used to
# corrupt memory or reach undefined behaviour; the patched decoder decodes the input or
# rejects it with the message in PASS_REGULAR_EXPRESSION (on the threaded runs a code-
# block failure may surface as the generic decode-task message).  Some of the undefined
# behaviour does not show in a release build, where the UBSan CI job is what catches a
# regression.
function(_sec_decode_test name input threads pass_re)
  add_test(NAME ${name}
           COMMAND open_htj2k_dec -i ${SECURITY_DATA_DIR}/${input} -o ${name}.pgx
                   -num_threads ${threads})
  if(pass_re)
    set_tests_properties(${name} PROPERTIES
        PASS_REGULAR_EXPRESSION "${pass_re}"
        FAIL_REGULAR_EXPRESSION "${_SEC_CRASH_RE}"
        TIMEOUT 60)
  else()
    # Must decode: exit 0 and no crash marker.
    set_tests_properties(${name} PROPERTIES FAIL_REGULAR_EXPRESSION "${_SEC_CRASH_RE}" TIMEOUT 60)
  endif()
endfunction()

# 31 and 32 DWT levels, the most the spec allows.  The per-resolution arrays of
# create_resolutions held 32 entries but are indexed by resolution 0..NL (the
# normalization shifts up to NL + 1), and the extents of resolutions and subbands
# divided by 2^(NL - r) with a 32-bit shift, which is undefined at 32.  dwt31_levels.j2c
# and dwt32_levels.j2c are valid HT codestreams of dwt_levels_src.pgm (61 x 47, 8 bits,
# reversible; open_htj2k_enc Clevels=31/32 Creversible=yes) and must decode to it exactly
# (a release build of the old decoder got the 32-level one wrong).  The fuzzed
# 32-level file must decode without a crash.
foreach(_nl 31 32)
  foreach(_t 1 4)
    _sec_decode_test(dec_dwt${_nl}_levels_t${_t} dwt${_nl}_levels.j2c ${_t} "")
    add_test(NAME comp_dwt${_nl}_levels_t${_t}
             COMMAND imgcmp dec_dwt${_nl}_levels_t${_t}_00.pgx ${SECURITY_DATA_DIR}/dwt_levels_src.pgm 0 0)
    set_tests_properties(comp_dwt${_nl}_levels_t${_t} PROPERTIES DEPENDS dec_dwt${_nl}_levels_t${_t})
  endforeach()
endforeach()
foreach(_t 1 4)
  _sec_decode_test(security_dwt32_levels_fuzzed_t${_t} security_dwt32_levels_fuzzed.jph ${_t} "")
endforeach()

# Tile-parts.  A codestream that ends before some tiles (a truncated download) left those
# tiles with no tile-part, and the decoder read the first of none; it now decodes them as
# empty, like the packets missing from a tile cut short.  A tile with 256 tile-parts (one
# more than the spec allows) wrapped an 8-bit count to 0 and must decode.  A tile whose
# tile-parts lack the first one (TPsot 0, which carries the tile header) was never set up
# and is now rejected.
foreach(_t 1 4)
  _sec_decode_test(security_tile_without_tile_parts_t${_t} security_tile_without_tile_parts.j2c ${_t}
                   "no data for tile 1 .* decoding it as empty")
  _sec_decode_test(security_256_tile_parts_t${_t} security_256_tile_parts.j2c ${_t} "")
  _sec_decode_test(security_first_tile_part_missing_t${_t} security_first_tile_part_missing.jph ${_t}
                   "not its first \\(TPsot 0\\)")
endforeach()

# An HT code-block holding only placeholder passes decodes to zeros, but nothing wrote
# them: its samples kept whatever the reused strip buffer held, so the image depended on
# the heap.  Decode it with the heap filled differently (glibc's MALLOC_PERTURB_; elsewhere
# both runs are alike) and with 1 and 4 threads; the outputs must be identical.
_sec_decode_test(security_empty_ht_block_t1 security_empty_ht_block.j2c 1 "")
_sec_decode_test(security_empty_ht_block_t4 security_empty_ht_block.j2c 4 "")
set_tests_properties(security_empty_ht_block_t1 PROPERTIES ENVIRONMENT "MALLOC_PERTURB_=0")
set_tests_properties(security_empty_ht_block_t4 PROPERTIES ENVIRONMENT "MALLOC_PERTURB_=170")
foreach(_comp 00 01 02)
  add_test(NAME cmp_security_empty_ht_block_${_comp}
           COMMAND ${CMAKE_COMMAND} -E compare_files security_empty_ht_block_t1_${_comp}.pgx
                   security_empty_ht_block_t4_${_comp}.pgx)
  set_tests_properties(cmp_security_empty_ht_block_${_comp} PROPERTIES
      DEPENDS "security_empty_ht_block_t1;security_empty_ht_block_t4")
endforeach()

# Bounds taken from the codestream before they are used as shift counts or array sizes:
# a Part 1 code-block whose zero bit-planes and coding passes run below bit 0 of the
# samples; a quantization exponent giving a subband more than 30 magnitude bit-planes
# (Part 1 and HT dequantization shift by 31 - M_b and M_b + 1); a code-block segment length
# signalled in more than 32 bits.
foreach(_t 1 4)
  _sec_decode_test(security_part1_missing_msbs_t${_t} security_part1_missing_msbs.jph ${_t}
                   "missing MSBs of .* coding passes|decode task failed")
  _sec_decode_test(security_mb_part1_dequant_t${_t} security_mb_part1_dequant.jph ${_t}
                   "has 31 magnitude bit-planes")
  _sec_decode_test(security_mb_ht_dequantize_t${_t} security_mb_ht_dequantize.j2c ${_t}
                   "has 31 magnitude bit-planes")
  _sec_decode_test(security_segment_length_bits_t${_t} security_segment_length_bits.j2c ${_t}
                   "segment length of 37 bits")
endforeach()

# Undefined behaviour without a release-build symptom: a quantization step whose integer
# scale for the ROI dequantization exceeds int32 (the conversion now saturates; the file
# decodes), and an MQ code register that a corrupt codeword shifted past 2^31 while it was
# a signed int32 (the decoder then rejects a later segment as too long).
foreach(_t 1 4)
  _sec_decode_test(security_dequant_scale_range_t${_t} security_dequant_scale_range.jph ${_t} "")
  _sec_decode_test(security_mq_code_register_t${_t} security_mq_code_register.jph ${_t}
                   "MQ segment length exceeds|decode task failed")
endforeach()

# The CLI exited 0 after a decode error, so scripts could not tell a failed decode from
# a good one.  It must now exit non-zero (not crash) and say why.
add_test(NAME security_dec_cli_exit_code
         COMMAND ${CMAKE_COMMAND} -DDEC=$<TARGET_FILE:open_htj2k_dec>
                 -DINPUT=${SECURITY_DATA_DIR}/security_segment_length_bits.j2c
                 -DOUTPUT=security_dec_cli_exit_code.pgx -DTHREADS=1
                 "-DEXPECT=segment length of 37 bits" "-DCRASH=${_SEC_CRASH_RE}"
                 -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_exit_failure.cmake)
set_tests_properties(security_dec_cli_exit_code PROPERTIES TIMEOUT 60)

# Memory sized by header values that the codestream's data does not back.  Each fixture is
# a valid codestream of about 100 bytes (50 KB for the one with 16384 components) that made
# the decoder allocate hundreds of megabytes to more than 8 GB.  The decoder now builds a
# precinct's code-blocks when its first packet that is not empty arrives, keeps no
# per-layer state for each code-block, sizes its line buffers by the subband, and rejects as
# not supported what would still take more than its budget (the larger of 256 MiB, 512
# bytes per codestream byte and 2 bytes per image sample).  The fixtures that must decode
# are run with the output discarded (an output name without an extension):
#  - 1024 x 1024, 4 x 4 code-blocks, 65535 layers (8.6 GB of per-layer state);
#  - 16384 x 16384, 4 x 4 code-blocks, empty precincts (5.7 GB of code-blocks);
#  - 1048576 x 2, 4 x 1024 code-blocks (line buffers 1024 rows high for 1-row subbands);
#  - a component without columns and 2^26 rows (a strip list sized by its height).
function(_sec_decode_discard name input threads)
  add_test(NAME ${name}
           COMMAND open_htj2k_dec -i ${SECURITY_DATA_DIR}/${input} -o ${name} -num_threads ${threads})
  set_tests_properties(${name} PROPERTIES FAIL_REGULAR_EXPRESSION "${_SEC_CRASH_RE}" TIMEOUT 60)
endfunction()
foreach(_t 1 4)
  _sec_decode_discard(security_layers_65535_t${_t} security_layers_65535.j2c ${_t})
  _sec_decode_discard(security_cblk_state_empty_precincts_t${_t} security_cblk_state_empty_precincts.j2c ${_t})
  _sec_decode_discard(security_ring_cblk_height_t${_t} security_ring_cblk_height.j2c ${_t})
  _sec_decode_discard(security_width0_component_t${_t} security_width0_component.j2c ${_t})
endforeach()
# Rejected before the memory is allocated: 4096 x 4096 with 4 x 4 code-blocks whose
# packets are not empty but carry no code-block (320 MB of code-blocks from 112 bytes);
# 1400 x 1400 with 2 x 2 precincts (660 thousand precincts, 1 GB); 16384 components in 64
# tiles (the tile-components of every tile, 670 MB).
_sec_rejects(security_cblk_state_budget security_cblk_state_budget.j2c
             "the code-blocks of a precinct would take .* not supported" mt)
_sec_rejects(security_precinct_state_budget security_precinct_state_budget.j2c
             "its precincts would take .* not supported" mt)
_sec_rejects(security_tile_components_budget security_tile_components_budget.j2c
             "its tile-components would take .* not supported" mt)

# A Part 2 DFS marker declaring decomposition levels without a transform (DWT_NO): such a
# level has no subbands, but its resolution counted 3, and setting up its precincts read
# them from a null array.  The fixture is the main header of ATK_DFS_REV.j2c with those
# levels and no packet data; it must decode.
foreach(_t 1 4)
  _sec_decode_discard(security_dfs_no_transform_t${_t} security_dfs_no_transform.j2c ${_t})
endforeach()

# Found by fuzzing the decoder the way image libraries call it:
#  - 65536 x 16 RGB with the colour transform and 4 x 1024 code-blocks (133 bytes): in a
#    build whose colour conversion runs on the thread pool (generic x86-64, SSE2), its
#    colour strip was 1024 rows high whatever the tile height, 860 MB with 4 threads;
#  - 65280 tiles of 1 x 1 (1322 bytes): every tile's state was kept until the end of the
#    decode, 1.5 GB;
#  - 1048576 x 2 RGB (112 bytes): its line buffers would take 516 MB, an eight-row IDWT
#    ring per level whatever the image height, so it is refused as not supported.
foreach(_t 1 4)
  _sec_decode_discard(security_color_strip_short_t${_t} security_color_strip_short.j2c ${_t})
  _sec_decode_discard(security_many_tiles_65280_t${_t} security_many_tiles_65280.j2c ${_t})
endforeach()
_sec_rejects(security_line_buffers_budget security_line_buffers_budget.j2c
             "its line-decoding buffers would take .* not supported" mt)

# Set-up work sized by header values: 9361 progression order changes in the main header,
# each replayed over the packets of every one of 65535 tiles (65 KB; 30 s), and 16384
# components in 65535 tiles without data (49 KB; minutes).  The decoder bounds the
# progression steps by the larger of 2^28 and 64 per codestream byte, and the
# tile-components set up by the larger of 2^20 and one per codestream byte.
_sec_rejects(security_poc_steps_budget security_poc_steps_budget.j2c
             "its packet progressions would take more than .* not supported" mt)
_sec_rejects(security_tile_components_count security_tile_components_count.j2c
             "sets up more than 1048576 tile-components .* not supported" mt)
