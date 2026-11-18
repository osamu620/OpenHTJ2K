# Tests of open_htj2k_transcode (Part 1 -> HTJ2K).  Included from the top-level CMakeLists.txt
# after decoder_conformance.cmake.  Every case runs tests/transcoder_roundtrip.cmake: decode the
# input, transcode it (1 and 2 threads, which must give the same bytes), decode the result and
# compare the images.
if(TARGET open_htj2k_transcode)
  set(_tc_driver ${CMAKE_CURRENT_SOURCE_DIR}/tests/transcoder_roundtrip.cmake)
  set(_tc_work   ${CMAKE_CURRENT_BINARY_DIR}/transcoder_tests)
  set(_tc_ref    ${CMAKE_CURRENT_SOURCE_DIR}/conformance_data/references)

  # tc_add(<name> [INPUT file | ENC_IN file ENC_ARGS a@@b [ENC_EXT jp2]] [OUT_EXT jph]
  #        [MODE identical|close|reject] [PAE n MSE x] [PATTERN regex])
  function(tc_add name)
    cmake_parse_arguments(A "" "INPUT;ENC_IN;ENC_ARGS;ENC_EXT;OUT_EXT;MODE;PAE;MSE;PATTERN;TRANS_ARGS" "" ${ARGN})
    set(_args -DDEC=$<TARGET_FILE:open_htj2k_dec> -DTRANS=$<TARGET_FILE:open_htj2k_transcode>
              -DWORK=${_tc_work}/${name})
    if(A_INPUT)
      list(APPEND _args -DINPUT=${A_INPUT})
    endif()
    if(A_ENC_ARGS)
      list(APPEND _args -DENC=$<TARGET_FILE:open_htj2k_enc> -DENC_IN=${A_ENC_IN} "-DENC_ARGS=${A_ENC_ARGS}")
      if(A_ENC_EXT)
        list(APPEND _args -DENC_EXT=${A_ENC_EXT})
      endif()
    endif()
    foreach(_k OUT_EXT MODE PAE MSE)
      if(A_${_k})
        list(APPEND _args -D${_k}=${A_${_k}})
      endif()
    endforeach()
    if(A_PATTERN)
      list(APPEND _args "-DPATTERN=${A_PATTERN}")
    endif()
    if(A_TRANS_ARGS)
      list(APPEND _args "-DTRANS_ARGS=${A_TRANS_ARGS}")
    endif()
    if(A_MODE STREQUAL "close")
      list(APPEND _args -DIMGCMP=$<TARGET_FILE:imgcmp>)
    endif()
    add_test(NAME tc_${name} COMMAND ${CMAKE_COMMAND} ${_args} -P ${_tc_driver})
    set_tests_properties(tc_${name} PROPERTIES LABELS transcoder)
  endfunction()

  # Part 1 conformance codestreams.  p0_03, p0_13 and p0_15 have RGN (region of interest), p0_06
  # a lossy one; p0_02 .. p1_03 have COC (some with another wavelet filter than the COD), p0_03,
  # p0_15 POC, p1_05 PPM and p1_04 tile-part header QCD segments.
  set(_tc_conf
      hifi_p1_02 p0_01 p0_02 p0_03 p0_04 p0_05 p0_07 p0_08 p0_09 p0_10 p0_11 p0_12 p0_13 p0_14
      p0_15 p0_16 p1_01 p1_02 p1_03 p1_04 p1_05 p1_06 p1_07)
  foreach(_c IN LISTS _tc_conf)
    tc_add(conf_${_c} INPUT ${CONFORMANCE_DATA_DIR}/${_c}.j2k)
  endforeach()
  # Lossy region of interest: the HT decoder places its mid-point differently for the region's
  # coefficients, so the result is close to the Part 1 decode, not identical (a few levels of 4095).
  tc_add(conf_p0_06 INPUT ${CONFORMANCE_DATA_DIR}/p0_06.j2k MODE close PAE 4 MSE 1)
  # A large lossless one.
  tc_add(conf_u10_part1_lossless INPUT ${CONFORMANCE_DATA_DIR}/u10_part1_lossless.j2c)
  # The colour space of a JPH file made from a raw codestream: sRGB (EnumCS 16) by default, sYCC (18)
  # on request; the options that make no sense are refused.
  foreach(_v "default:" "ycc:-jph_color_space@@YCC" "rgb:-jph_color_space@@RGB")
    string(REPLACE ":" ";" _p "${_v}")
    list(GET _p 0 _n)
    list(GET _p 1 _a)
    if(_n STREQUAL "ycc")
      set(_enum 00000012)
    else()
      set(_enum 00000010)
    endif()
    add_test(NAME tc_colour_${_n} COMMAND ${CMAKE_COMMAND} -DTRANS=$<TARGET_FILE:open_htj2k_transcode>
             -DINPUT=${CONFORMANCE_DATA_DIR}/p0_04.j2k -DWORK=${_tc_work}/colour_${_n} -DOUT_EXT=jph
             "-DTRANS_ARGS=${_a}" "-DHEX=636f6c72010000${_enum}"
             -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/transcoder_markers.cmake)
    set_tests_properties(tc_colour_${_n} PROPERTIES LABELS transcoder)
  endforeach()
  tc_add(colour_ycc_j2c_refused INPUT ${CONFORMANCE_DATA_DIR}/p0_04.j2k MODE reject TRANS_ARGS "-jph_color_space@@YCC"
         PATTERN "JPH output only")
  tc_add(colour_ycc_grey_refused INPUT ${CONFORMANCE_DATA_DIR}/p0_01.j2k MODE reject OUT_EXT jph
         TRANS_ARGS "-jph_color_space@@YCC" PATTERN "three or four components")
  tc_add(colour_jp2_refused ENC_IN ${CMAKE_CURRENT_SOURCE_DIR}/conformance_data/references/kodim23.ppm
         ENC_ARGS "Cmodes=0" ENC_EXT jp2 MODE reject OUT_EXT jph TRANS_ARGS "-jph_color_space@@YCC"
         PATTERN "own colour specification")
  # Marker segments carried over: CRG (p0_03, main header), a COM in a tile-part header (p1_04, tile 29).
  add_test(NAME tc_keep_crg COMMAND ${CMAKE_COMMAND} -DTRANS=$<TARGET_FILE:open_htj2k_transcode>
           -DINPUT=${CONFORMANCE_DATA_DIR}/p0_03.j2k -DWORK=${_tc_work}/keep_crg "-DHEX=ff630006ff907f2e"
           -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/transcoder_markers.cmake)
  add_test(NAME tc_keep_tile_com COMMAND ${CMAKE_COMMAND} -DTRANS=$<TARGET_FILE:open_htj2k_transcode>
           -DINPUT=${CONFORMANCE_DATA_DIR}/p1_04.j2k -DWORK=${_tc_work}/keep_tile_com
           "-DHEX=ff64ffff0000ff4fff51;ff64001a0001437265617465642062792041776172652c20496e632e"
           -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/transcoder_markers.cmake)
  # p0_07 has PLT marker segments in its tile-part headers; the output has them again, for its own packets.
  add_test(NAME tc_plt COMMAND ${CMAKE_COMMAND} -DTRANS=$<TARGET_FILE:open_htj2k_transcode>
           -DINPUT=${CONFORMANCE_DATA_DIR}/p0_07.j2k -DWORK=${_tc_work}/plt "-DHEX=ff58"
           -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/transcoder_markers.cmake)
  set_tests_properties(tc_keep_crg tc_keep_tile_com tc_plt PROPERTIES LABELS transcoder)
  # A fuzzer's 610 byte stream whose SIZ declares about 28000 tiles, nearly all without a tile-part.
  # They are left out of the output; the transcoder once allocated 4 GB for them.
  tc_add(fuzz_many_empty_tiles INPUT ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/transcoder_many_empty_tiles.j2c)

  # 65,280 tiles of one sample each (a flat image, so the stream is small).  The input tile's state
  # counts against the decoder's memory budget; the transcoder once did not give it back after
  # each tile, so this valid stream was refused at about tile 58,000 as "not supported".
  file(MAKE_DIRECTORY ${_tc_work})
  file(WRITE ${_tc_work}/flat_255x256.ppm "P6\n255 256\n127\n")
  # (string(REPEAT) needs CMake 3.15; the project supports 3.13.)
  set(_tc_flat "@")
  while(1)
    string(LENGTH "${_tc_flat}" _tc_len)
    if(_tc_len GREATER_EQUAL 195840)
      break()
    endif()
    string(APPEND _tc_flat "${_tc_flat}")
  endwhile()
  string(SUBSTRING "${_tc_flat}" 0 195840 _tc_flat)
  file(APPEND ${_tc_work}/flat_255x256.ppm "${_tc_flat}")
  tc_add(many_tiles_budget ENC_IN ${_tc_work}/flat_255x256.ppm
         ENC_ARGS "Cmodes=0@@Creversible=yes@@Clevels=10@@Stiles={1,1}")
  set_tests_properties(tc_many_tiles_budget PROPERTIES TIMEOUT 600)

  # Streams the OpenHTJ2K encoder makes with the Part 1 block coder: rates (blocks cut inside a
  # bit-plane), layers, every code-block style switch, Qfactor, lossless, 16 bit, block sizes,
  # levels, tiles, SOP/EPH and a precinct/progression change.
  set(_K ${_tc_ref}/kodim23.ppm)
  set(_W ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/rgb16_full_range.ppm)
  tc_add(gen_rate1       ENC_IN ${_K} ENC_ARGS "Cmodes=0@@-rate@@1")
  tc_add(gen_rate025     ENC_IN ${_K} ENC_ARGS "Cmodes=0@@-rate@@0.25")
  tc_add(gen_rate2       ENC_IN ${_K} ENC_ARGS "Cmodes=0@@-rate@@2")
  tc_add(gen_layers3     ENC_IN ${_K} ENC_ARGS "Cmodes=0@@-rate@@0.25,0.5,1")
  tc_add(gen_layers5     ENC_IN ${_K} ENC_ARGS "Cmodes=0@@-rate@@0.1,0.25,0.5,1,2")
  tc_add(gen_bypass      ENC_IN ${_K} ENC_ARGS "Cmodes=BYPASS@@-rate@@1")
  tc_add(gen_reset       ENC_IN ${_K} ENC_ARGS "Cmodes=RESET@@-rate@@1")
  tc_add(gen_restart     ENC_IN ${_K} ENC_ARGS "Cmodes=RESTART@@-rate@@1")
  tc_add(gen_causal      ENC_IN ${_K} ENC_ARGS "Cmodes=CAUSAL@@-rate@@1")
  tc_add(gen_erterm      ENC_IN ${_K} ENC_ARGS "Cmodes=ERTERM@@-rate@@1")
  tc_add(gen_segmark     ENC_IN ${_K} ENC_ARGS "Cmodes=SEGMARK@@-rate@@1")
  tc_add(gen_all_modes   ENC_IN ${_K} ENC_ARGS "Cmodes=BYPASS|RESET|RESTART|CAUSAL|ERTERM|SEGMARK@@-rate@@1")
  tc_add(gen_all_layers  ENC_IN ${_K} ENC_ARGS "Cmodes=BYPASS|RESET|RESTART|CAUSAL|ERTERM|SEGMARK@@-rate@@0.25,0.5,1")
  tc_add(gen_qfactor     ENC_IN ${_K} ENC_ARGS "Cmodes=0@@Qfactor=50")
  tc_add(gen_lossless    ENC_IN ${_K} ENC_ARGS "Cmodes=0@@Creversible=yes")
  tc_add(gen_16bit       ENC_IN ${_W} ENC_ARGS "Cmodes=0@@-rate@@2")
  tc_add(gen_cblk32      ENC_IN ${_K} ENC_ARGS "Cmodes=0@@Cblk={32,32}@@-rate@@1")
  tc_add(gen_levels2     ENC_IN ${_K} ENC_ARGS "Cmodes=0@@Clevels=2@@-rate@@1")
  tc_add(gen_tiles       ENC_IN ${_K} ENC_ARGS "Cmodes=0@@Stiles={256,256}@@-rate@@1")
  tc_add(gen_sop_eph     ENC_IN ${_K} ENC_ARGS "Cmodes=0@@Cuse_sop=yes@@Cuse_eph=yes@@-rate@@1")
  tc_add(gen_rpcl_prec   ENC_IN ${_K} ENC_ARGS "Cmodes=0@@Corder=RPCL@@Cprecincts={256,256}{128,128}@@-rate@@1")

  # JP2 input, JPH output.
  tc_add(file_jp2_to_codestream ENC_IN ${_K} ENC_ARGS "Cmodes=0@@-rate@@1" ENC_EXT jp2)
  tc_add(file_jp2_to_jph        ENC_IN ${_K} ENC_ARGS "Cmodes=0@@-rate@@1" ENC_EXT jp2 OUT_EXT jph)
  tc_add(file_raw_to_jph        ENC_IN ${_K} ENC_ARGS "Cmodes=0@@-rate@@1" OUT_EXT jph)

  # Inputs the transcoder must refuse, with an error and not a crash.
  tc_add(reject_ht_input   INPUT ${CONFORMANCE_DATA_DIR}/ds0_ht_01_b11.j2k MODE reject PATTERN "already uses the HT")
  tc_add(reject_part2      INPUT ${CONFORMANCE_DATA_DIR}/ATK_DFS_REV.j2c   MODE reject PATTERN "DFS/ATK")
  tc_add(reject_not_a_j2k  INPUT ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt MODE reject PATTERN "neither a JP2")
endif()
