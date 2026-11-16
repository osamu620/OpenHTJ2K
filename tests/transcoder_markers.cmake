# Transcode INPUT and fail unless the output holds each byte sequence of HEX (marker segments
# the transcoder must carry over).  Invoked via:
#   cmake -DTRANS=<open_htj2k_transcode> -DINPUT=<file> -DWORK=<dir> "-DHEX=<hex>[;<hex>...]" -P transcoder_markers.cmake
file(MAKE_DIRECTORY "${WORK}")
if(NOT DEFINED OUT_EXT)
  set(OUT_EXT j2c)
endif()
set(_targs "")
if(DEFINED TRANS_ARGS)
  string(REPLACE "@@" ";" _targs "${TRANS_ARGS}")
endif()
execute_process(COMMAND ${TRANS} ${_targs} -i ${INPUT} -o ${WORK}/out.${OUT_EXT} RESULT_VARIABLE rc OUTPUT_QUIET ERROR_QUIET)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "the transcode of ${INPUT} failed")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} -DFILE=${WORK}/out.${OUT_EXT} "-DHEX=${HEX}"
                -P ${CMAKE_CURRENT_LIST_DIR}/check_file_hex.cmake RESULT_VARIABLE rc2)
if(NOT rc2 EQUAL 0)
  message(FATAL_ERROR "a marker segment of the input is missing from the output")
endif()
