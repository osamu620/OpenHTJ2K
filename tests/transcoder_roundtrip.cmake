# Round-trip check of open_htj2k_transcode, run as a CTest test:
#
#   cmake -DDEC=<open_htj2k_dec> -DTRANS=<open_htj2k_transcode> -DWORK=<scratch dir>
#         [-DENC=<open_htj2k_enc> -DENC_IN=<image> "-DENC_ARGS=a@@b@@c" -DENC_EXT=<j2c|jp2>]
#         [-DINPUT=<Part 1 codestream or JP2 file>]
#         [-DTRANS_ARGS=a@@b] [-DOUT_EXT=<j2c|jph>] [-DMODE=identical|close|reject]
#         [-DIMGCMP=<imgcmp> -DPAE=<n> -DMSE=<x>] [-DPATTERN=<regex>]
#         -P transcoder_roundtrip.cmake
#
# The input is either INPUT, or made by the encoder from ENC_IN and ENC_ARGS (separated by @@).
#   identical  decode(input) and decode(transcode(input)) must be the same, file by file; the
#              transcode must also give the same bytes with 1 and with 2 threads
#   close      the same, but compared with imgcmp (peak absolute error PAE, mean squared error MSE)
#   reject     the transcode must fail, and its output must match PATTERN
cmake_minimum_required(VERSION 3.13)

if(NOT DEFINED MODE)
  set(MODE identical)
endif()
if(NOT DEFINED OUT_EXT)
  set(OUT_EXT j2c)
endif()
file(REMOVE_RECURSE "${WORK}")
file(MAKE_DIRECTORY "${WORK}")

function(run_ok what)
  execute_process(COMMAND ${ARGN} RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "${what} failed (exit ${rc}):\n${out}\n${err}")
  endif()
endfunction()

# The input.
if(DEFINED ENC_ARGS)
  if(NOT DEFINED ENC_EXT)
    set(ENC_EXT j2c)
  endif()
  string(REPLACE "@@" ";" _enc_args "${ENC_ARGS}")
  set(INPUT "${WORK}/input.${ENC_EXT}")
  run_ok("encoding the input" "${ENC}" -i "${ENC_IN}" -o "${INPUT}" ${_enc_args})
endif()

if(MODE STREQUAL "reject")
  set(_rtargs "")
  if(DEFINED TRANS_ARGS)
    string(REPLACE "@@" ";" _rtargs "${TRANS_ARGS}")
  endif()
  execute_process(COMMAND "${TRANS}" ${_rtargs} -i "${INPUT}" -o "${WORK}/out.${OUT_EXT}"
                  RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
  # The tool reports errors on stdout and exits with failure; a crash gives a signal text instead.
  if(rc EQUAL 0)
    message(FATAL_ERROR "the transcode of ${INPUT} was expected to fail, but succeeded")
  endif()
  if(NOT rc MATCHES "^[0-9]+$")
    message(FATAL_ERROR "the transcode of ${INPUT} ended with '${rc}' instead of a clean error")
  endif()
  if(DEFINED PATTERN AND NOT "${out}${err}" MATCHES "${PATTERN}")
    message(FATAL_ERROR "the error of the transcode did not match '${PATTERN}':\n${out}\n${err}")
  endif()
  return()
endif()

# Decode the input.
run_ok("decoding the input" "${DEC}" -i "${INPUT}" -o "${WORK}/ref.pgx")

set(_targs "")
if(DEFINED TRANS_ARGS)
  string(REPLACE "@@" ";" _targs "${TRANS_ARGS}")
endif()

# Transcode with one and with two threads: the output must not depend on the thread count.
run_ok("transcoding (1 thread)" "${TRANS}" ${_targs} -num_threads 1 -i "${INPUT}" -o "${WORK}/ht1.${OUT_EXT}")
run_ok("transcoding (2 threads)" "${TRANS}" ${_targs} -num_threads 2 -i "${INPUT}" -o "${WORK}/ht2.${OUT_EXT}")
execute_process(COMMAND ${CMAKE_COMMAND} -E compare_files "${WORK}/ht1.${OUT_EXT}" "${WORK}/ht2.${OUT_EXT}"
                RESULT_VARIABLE same)
if(NOT same EQUAL 0)
  message(FATAL_ERROR "the output depends on the number of threads")
endif()
# Reading the input where it is, or from a copy of it, gives the same output.
run_ok("transcoding (input copied)" "${TRANS}" ${_targs} -copy_input -num_threads 1 -i "${INPUT}" -o "${WORK}/htc.${OUT_EXT}")
execute_process(COMMAND ${CMAKE_COMMAND} -E compare_files "${WORK}/ht1.${OUT_EXT}" "${WORK}/htc.${OUT_EXT}"
                RESULT_VARIABLE same_c)
if(NOT same_c EQUAL 0)
  message(FATAL_ERROR "the output depends on whether the input is copied")
endif()

# Decode the output and compare, file by file (one file per component).
run_ok("decoding the transcoded file" "${DEC}" -i "${WORK}/ht1.${OUT_EXT}" -o "${WORK}/ht.pgx")
file(GLOB _refs RELATIVE "${WORK}" "${WORK}/ref*.pgx")
list(LENGTH _refs _n)
if(_n EQUAL 0)
  message(FATAL_ERROR "the decoder wrote no reference image")
endif()
foreach(_r IN LISTS _refs)
  string(REGEX REPLACE "^ref" "ht" _h "${_r}")
  if(NOT EXISTS "${WORK}/${_h}")
    message(FATAL_ERROR "the decoder wrote ${_r} but no ${_h}")
  endif()
  if(MODE STREQUAL "identical")
    execute_process(COMMAND ${CMAKE_COMMAND} -E compare_files "${WORK}/${_r}" "${WORK}/${_h}"
                    RESULT_VARIABLE rc)
    if(NOT rc EQUAL 0)
      message(FATAL_ERROR "${_r} and ${_h} differ")
    endif()
  else()
    run_ok("comparing ${_r} and ${_h}" "${IMGCMP}" "${WORK}/${_r}" "${WORK}/${_h}" ${PAE} ${MSE})
  endif()
endforeach()

# Keep the scratch directory only when the test fails (the failure paths above stop the script).
file(REMOVE_RECURSE "${WORK}")
