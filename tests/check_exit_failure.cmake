# Fails unless the decoder rejects INPUT: open_htj2k_dec must exit with a non-zero code (not
# a signal), print EXPECT, and print no crash marker (CRASH).  A plain add_test cannot check
# all three: PASS_REGULAR_EXPRESSION ignores the exit code, and WILL_FAIL would also pass a
# crash.
# Invoked via: cmake -DDEC=<exe> -DINPUT=<file> -DOUTPUT=<file> -DTHREADS=<n>
#                    -DEXPECT=<regex> -DCRASH=<regex> -P check_exit_failure.cmake
execute_process(COMMAND "${DEC}" -i "${INPUT}" -o "${OUTPUT}" -num_threads ${THREADS}
                RESULT_VARIABLE _rc OUTPUT_VARIABLE _out ERROR_VARIABLE _out)
message("${_out}")
if(NOT _rc MATCHES "^[0-9]+$")
  message(FATAL_ERROR "the decoder terminated abnormally: ${_rc}")
endif()
if(_rc EQUAL 0)
  message(FATAL_ERROR "the decoder exited 0 after rejecting the input")
endif()
if(NOT _out MATCHES "${EXPECT}")
  message(FATAL_ERROR "the output lacks \"${EXPECT}\"")
endif()
if(_out MATCHES "${CRASH}")
  message(FATAL_ERROR "the output shows a crash")
endif()
message(STATUS "exit code ${_rc}")
