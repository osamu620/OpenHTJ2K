# Fails unless the encoder rejects its arguments: open_htj2k_enc must exit with a non-zero code
# (not a signal, such as the abort of an uncaught exception) and print EXPECT.  WILL_FAIL alone
# would also pass a crash.
# Invoked via: cmake -DENC=<exe> "-DARGS=<arg>[;<arg>...]" -DEXPECT=<regex> -P check_enc_exit_failure.cmake
execute_process(COMMAND "${ENC}" ${ARGS} RESULT_VARIABLE _rc OUTPUT_VARIABLE _out ERROR_VARIABLE _out)
message("${_out}")
if(NOT _rc MATCHES "^[0-9]+$")
  message(FATAL_ERROR "the encoder terminated abnormally: ${_rc}")
endif()
if(_rc EQUAL 0)
  message(FATAL_ERROR "the encoder exited 0")
endif()
if(NOT _out MATCHES "${EXPECT}")
  message(FATAL_ERROR "the output lacks \"${EXPECT}\"")
endif()
message(STATUS "exit code ${_rc}")
