# Round trip of open_htj2k_transcode through HTJ2K and back, run as a CTest test:
#
#   cmake -DDEC=<open_htj2k_dec> -DTRANS=<open_htj2k_transcode> -DWORK=<scratch dir>
#         [-DENC=<open_htj2k_enc> -DENC_IN=<image> "-DENC_ARGS=a@@b@@c"] [-DINPUT=<Part 1 codestream>]
#         ["-DBACK_ARGS=a@@b"] [-DBYTES=1] -P transcoder_back.cmake
#
# The Part 1 input (INPUT, or made by the encoder from ENC_IN and ENC_ARGS, separated by @@) is transcoded
# into HTJ2K with -layers keep, and the result back into Part 1 with BACK_ARGS.  The decode of the result
# must be that of the input.  With BYTES=1 the result must be the input, byte for byte: the encoder cuts
# the layers of a Part 1 code-block as the transcoder does (the shortest prefix of the coded data for a
# cut that a later layer continues, a terminated last cut), so a transcode back with the input's
# code-block style gives the encoder's own codestream.  That checks every layer of every code-block,
# which the decoder (it decodes all layers) cannot.
cmake_minimum_required(VERSION 3.13)

file(REMOVE_RECURSE "${WORK}")
file(MAKE_DIRECTORY "${WORK}")

function(run_ok what)
  execute_process(COMMAND ${ARGN} RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "${what} failed (exit ${rc}):\n${out}\n${err}")
  endif()
endfunction()

if(DEFINED ENC_ARGS)
  string(REPLACE "@@" ";" _enc_args "${ENC_ARGS}")
  set(INPUT "${WORK}/input.j2c")
  run_ok("encoding the input" "${ENC}" -i "${ENC_IN}" -o "${INPUT}" ${_enc_args})
endif()
set(_bargs "")
if(DEFINED BACK_ARGS)
  string(REPLACE "@@" ";" _bargs "${BACK_ARGS}")
endif()

run_ok("transcoding into HTJ2K" "${TRANS}" -layers keep -i "${INPUT}" -o "${WORK}/ht.j2c")
run_ok("transcoding back into Part 1" "${TRANS}" ${_bargs} -i "${WORK}/ht.j2c" -o "${WORK}/back.j2c")

if(BYTES)
  execute_process(COMMAND ${CMAKE_COMMAND} -E compare_files "${INPUT}" "${WORK}/back.j2c" RESULT_VARIABLE same)
  if(NOT same EQUAL 0)
    message(FATAL_ERROR "the round trip did not give the input back")
  endif()
endif()

run_ok("decoding the input" "${DEC}" -i "${INPUT}" -o "${WORK}/ref.pgx")
run_ok("decoding the round trip" "${DEC}" -i "${WORK}/back.j2c" -o "${WORK}/back.pgx")
file(GLOB _refs RELATIVE "${WORK}" "${WORK}/ref*.pgx")
list(LENGTH _refs _n)
if(_n EQUAL 0)
  message(FATAL_ERROR "the decoder wrote no reference image")
endif()
foreach(_r IN LISTS _refs)
  string(REGEX REPLACE "^ref" "back" _b "${_r}")
  execute_process(COMMAND ${CMAKE_COMMAND} -E compare_files "${WORK}/${_r}" "${WORK}/${_b}" RESULT_VARIABLE rc)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "${_r} and ${_b} differ")
  endif()
endforeach()

file(REMOVE_RECURSE "${WORK}")
