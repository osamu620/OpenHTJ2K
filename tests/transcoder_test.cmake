# Tests of open_htj2k_transcode (Part 1 -> HTJ2K, and HTJ2K -> Part 1 for an HTJ2K input).  Included
# from the top-level CMakeLists.txt after decoder_conformance.cmake.  Most cases run
# tests/transcoder_roundtrip.cmake: decode the input, transcode it (1 and 2 threads, which must give the
# same bytes), decode the result and compare the images.
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
  # The header of the output: Rsiz 0x4000 whatever the Part 1 profile of the input (p0_01 has
  # Profile 0, p1_04 Profile 1); the CAP flags RGN (bit 12) for an RGN marker segment, in the main
  # header (p0_13) or a tile-part header (p0_03), and HETEROGENEOUS (bit 11) for coding parameters in
  # a tile-part header (p0_03 RGN, p1_04 QCD); the input's CAUSAL switch in the code-block style
  # (p1_02: 0x48).
  function(tc_header name input hex)
    add_test(NAME tc_${name} COMMAND ${CMAKE_COMMAND} -DTRANS=$<TARGET_FILE:open_htj2k_transcode>
             -DINPUT=${CONFORMANCE_DATA_DIR}/${input}.j2k -DWORK=${_tc_work}/${name} "-DHEX=${hex}"
             -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/transcoder_markers.cmake)
    set_tests_properties(tc_${name} PROPERTIES LABELS transcoder)
  endfunction()
  tc_header(header_profile0 p0_01 "ff5100294000;ff5000080002000000")
  tc_header(header_rgn_main p0_13 "ff5000080002000010")
  tc_header(header_rgn_tile p0_03 "ff5000080002000018")
  tc_header(header_profile1_het p1_04 "ff5100294000;ff5000080002000008")
  # The profile of the input in a CPF marker segment (Pcpf = Rsiz + 1), for a transcode back: p1_04 has
  # Profile 1 (Rsiz 2).
  tc_header(header_cpf p1_04 "ff5900040003")
  tc_header(header_causal p1_02 "ff52001301000001010604044800")
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

  # -layers keep: the layers of the input, kept with placeholder passes (WG1 N101237).  The decode must
  # be the Part 1 decode, and the output must not depend on the thread count.  The conformance streams
  # with more than one layer (p0_03 and p0_15 with RGN and POC, p0_07 with PLT, p0_08 with 30 layers,
  # p1_05 with PPM), and generated ones with layers and every code-block style switch.
  foreach(_c p0_02 p0_03 p0_05 p0_07 p0_08 p0_10 p0_15 p0_16 p1_01 p1_02 p1_03 p1_05)
    tc_add(layers_keep_${_c} INPUT ${CONFORMANCE_DATA_DIR}/${_c}.j2k TRANS_ARGS "-layers@@keep")
  endforeach()
  set(_L12 "0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.7,1,1.5,2,3")
  tc_add(layers_keep_gen3      ENC_IN ${_K} ENC_ARGS "Cmodes=0@@-rate@@0.25,0.5,1" TRANS_ARGS "-layers@@keep")
  tc_add(layers_keep_gen12     ENC_IN ${_K} ENC_ARGS "Cmodes=0@@-rate@@${_L12}" TRANS_ARGS "-layers@@keep")
  tc_add(layers_keep_all_modes ENC_IN ${_K} ENC_ARGS "Cmodes=BYPASS|RESET|RESTART|CAUSAL|ERTERM|SEGMARK@@-rate@@${_L12}"
         TRANS_ARGS "-layers@@keep")
  tc_add(layers_keep_tiles     ENC_IN ${_K} ENC_ARGS "Cmodes=0@@Stiles={64,64}@@-rate@@0.25,0.5,1,2" TRANS_ARGS "-layers@@keep")
  tc_add(layers_keep_rpcl_prec ENC_IN ${_K} ENC_ARGS "Cmodes=0@@Corder=RPCL@@Cprecincts={64,64},{32,32}@@-rate@@0.1,0.3,0.6,1.2"
         TRANS_ARGS "-layers@@keep")
  tc_add(layers_keep_sop_eph   ENC_IN ${_K} ENC_ARGS "Cmodes=0@@Corder=PCRL@@Cuse_sop=yes@@Cuse_eph=yes@@-rate@@0.2,0.4,0.8,1.6"
         TRANS_ARGS "-layers@@keep")
  # The COD of the output keeps the 6 layers of p0_02, with the HT code-block style.
  add_test(NAME tc_layers_keep_cod COMMAND ${CMAKE_COMMAND} -DTRANS=$<TARGET_FILE:open_htj2k_transcode>
           -DINPUT=${CONFORMANCE_DATA_DIR}/p0_02.j2k -DWORK=${_tc_work}/layers_keep_cod "-DTRANS_ARGS=-layers@@keep"
           "-DHEX=ff52000c06000006000304044000" -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/transcoder_markers.cmake)
  set_tests_properties(tc_layers_keep_cod PROPERTIES LABELS transcoder)
  # What keep refuses: another -partial mode, a lossy stream with a region of interest; and a bad value.
  tc_add(layers_keep_partial_refused INPUT ${CONFORMANCE_DATA_DIR}/p0_02.j2k MODE reject
         TRANS_ARGS "-layers@@keep@@-partial@@drop" PATTERN "partial refine")
  tc_add(layers_keep_lossy_roi_refused INPUT ${CONFORMANCE_DATA_DIR}/p0_06.j2k MODE reject
         TRANS_ARGS "-layers@@keep" PATTERN "region of interest")
  tc_add(layers_bad_value_refused INPUT ${CONFORMANCE_DATA_DIR}/p0_02.j2k MODE reject
         TRANS_ARGS "-layers@@all" PATTERN "merge or keep")

  # JP2 input, JPH output.
  tc_add(file_jp2_to_codestream ENC_IN ${_K} ENC_ARGS "Cmodes=0@@-rate@@1" ENC_EXT jp2)
  tc_add(file_jp2_to_jph        ENC_IN ${_K} ENC_ARGS "Cmodes=0@@-rate@@1" ENC_EXT jp2 OUT_EXT jph)
  tc_add(file_raw_to_jph        ENC_IN ${_K} ENC_ARGS "Cmodes=0@@-rate@@1" OUT_EXT jph)

  # Inputs the transcoder must refuse, with an error and not a crash.
  tc_add(reject_part2      INPUT ${CONFORMANCE_DATA_DIR}/ATK_DFS_REV.j2c   MODE reject PATTERN "DFS/ATK")
  tc_add(reject_not_a_j2k  INPUT ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt MODE reject PATTERN "neither a JP2")

  # The output tile is sized by the input's header.  It is an encoder tile, which builds every
  # code-block with its passes per layer (the decoder's tile builds the code-blocks its packets
  # have data for, without per-layer state), and it was not counted against the codestream's
  # memory budget:
  #  - 1024 x 1024 in 4 x 4 code-blocks with 65535 layers (112 bytes, HTJ2K): 11 GB of passes per
  #    layer; now refused as not supported, like a decode that would pass the budget;
  #  - 180 x 180 in 4 x 4 code-blocks with 32768 layers (97 bytes, HTJ2K), within the budget: the
  #    packet headers of the output took layers x layers steps per code-block (minutes).  Must
  #    transcode, to a codestream that decodes to the image of the input.
  tc_add(reject_output_tile_budget INPUT ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/security_layers_65535.j2c
         MODE reject PATTERN "output tile of the transcoder would take .* not supported")
  tc_add(many_layers_output INPUT ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/security_transcode_layers.j2c)
  set_tests_properties(tc_reject_output_tile_budget tc_many_layers_output PROPERTIES TIMEOUT 120)
  # An HT code-block of several layers that the block decoder refuses (113 bytes).  The
  # transcode ends with that error; the bytes the code-block owns used to be lost on the way
  # (64 bytes here, 4096 in larger blocks), which a build with LeakSanitizer reports.
  tc_add(reject_ht_block_error
         INPUT ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/security_transcode_ht_block_error.j2c
         MODE reject PATTERN "transcoding failed: Value of numbits = [0-9]+ is out of range")

  # HTJ2K -> Part 1.  The HTJ2K conformance codestreams, with one HT set per code-block (ds0_ht_*, ds1_ht_*)
  # and mixed HT and Part 1 code-blocks (ds0_hm_*): the Part 1 output must decode to the image of the
  # input.  Many have layers made with placeholder passes and more missing MSBs than the data have.
  set(_tc_ht_conf
      ds0_hm_06_b11 ds0_hm_06_b18 ds0_hm_15_b8 ds0_ht_01_b11 ds0_ht_02_b11 ds0_ht_02_b12 ds0_ht_03_b14
      ds0_ht_04_b11 ds0_ht_04_b12 ds0_ht_05_b11 ds0_ht_05_b12 ds0_ht_07_b11 ds0_ht_07_b15 ds0_ht_07_b16
      ds0_ht_08_b11 ds0_ht_08_b15 ds0_ht_08_b16 ds0_ht_09_b11 ds0_ht_10_b11 ds0_ht_11_b10 ds0_ht_12_b11
      ds0_ht_13_b11 ds0_ht_14_b11 ds0_ht_15_b14 ds0_ht_16_b11 ds1_ht_01_b11 ds1_ht_01_b12 ds1_ht_02_b11
      ds1_ht_02_b12 ds1_ht_03_b11 ds1_ht_03_b12 ds1_ht_04_b9 ds1_ht_05_b11 ds1_ht_06_b11 ds1_ht_07_b11)
  foreach(_c IN LISTS _tc_ht_conf)
    tc_add(to_part1_${_c} INPUT ${CONFORMANCE_DATA_DIR}/${_c}.j2k)
  endforeach()
  # Region of interest with code-blocks coded above their last bit-plane (lossy, or bounded by the
  # magnitude bound B): the HT and the Part 1 decoders of this library place the reconstruction of the
  # region's coefficients differently, so the decodes of the same indices are close, not identical.  For
  # ds0_ht_06 (from p0_06) both are within the conformance tolerance of p0_06.
  tc_add(to_part1_ds0_ht_03_b11 INPUT ${CONFORMANCE_DATA_DIR}/ds0_ht_03_b11.j2k MODE close PAE 1 MSE 0.01)
  tc_add(to_part1_ds0_ht_15_b11 INPUT ${CONFORMANCE_DATA_DIR}/ds0_ht_15_b11.j2k MODE close PAE 1 MSE 0.01)
  tc_add(to_part1_ds0_ht_06_b18 INPUT ${CONFORMANCE_DATA_DIR}/ds0_ht_06_b18.j2k MODE close PAE 635 MSE 11287)
  # HTJ2K made by the encoder: a JPH file, refinement passes (rate control), and one layer merged.
  tc_add(to_part1_file_jph  ENC_IN ${_K} ENC_ARGS "Qfactor=80" ENC_EXT jph)
  tc_add(to_part1_ht_rate   ENC_IN ${_K} ENC_ARGS "-rate@@1")
  tc_add(to_part1_merge     INPUT ${CONFORMANCE_DATA_DIR}/ds0_ht_08_b16.j2k TRANS_ARGS "-layers@@merge")
  tc_add(to_part1_modes     INPUT ${CONFORMANCE_DATA_DIR}/ds0_ht_02_b12.j2k TRANS_ARGS "-Cmodes@@BYPASS|RESTART|ERTERM")
  # The header of the output: Rsiz from the CPF marker segment of the input (ds0_ht_01: Pcpf 2, so
  # Profile 0), and no CAP or CPF marker segment (SIZ is followed by COD); the input's CAUSAL switch and
  # 19 layers (ds1_ht_02); one layer with merge; the switches -Cmodes gives.
  tc_header(to_part1_header_rsiz ds0_ht_01_b11
            "ff4fff510029000100000080000000800000000000000000000000800000008000000000000000000001070101ff52")
  tc_header(to_part1_header_causal ds1_ht_02_b12 "ff52001301000013010604040800")
  add_test(NAME tc_to_part1_header_merge COMMAND ${CMAKE_COMMAND} -DTRANS=$<TARGET_FILE:open_htj2k_transcode>
           -DINPUT=${CONFORMANCE_DATA_DIR}/ds0_ht_02_b12.j2k -DWORK=${_tc_work}/to_part1_header_merge
           "-DTRANS_ARGS=-layers@@merge" "-DHEX=ff5200100700000100030404000077888888"
           -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/transcoder_markers.cmake)
  add_test(NAME tc_to_part1_header_modes COMMAND ${CMAKE_COMMAND} -DTRANS=$<TARGET_FILE:open_htj2k_transcode>
           -DINPUT=${CONFORMANCE_DATA_DIR}/ds0_ht_02_b12.j2k -DWORK=${_tc_work}/to_part1_header_modes
           "-DTRANS_ARGS=-Cmodes@@BYPASS|ERTERM" "-DHEX=ff5200100700000600030404110077888888"
           -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/transcoder_markers.cmake)
  set_tests_properties(tc_to_part1_header_merge tc_to_part1_header_modes PROPERTIES LABELS transcoder)
  # The options of the other direction, and outputs that cannot hold a Part 1 codestream, are refused.
  tc_add(to_part1_jph_refused INPUT ${CONFORMANCE_DATA_DIR}/ds0_ht_01_b11.j2k MODE reject OUT_EXT jph
         PATTERN "raw Part 1 codestream")
  tc_add(to_part1_partial_refused INPUT ${CONFORMANCE_DATA_DIR}/ds0_ht_01_b11.j2k MODE reject
         TRANS_ARGS "-partial@@drop" PATTERN "for a Part 1 input")
  tc_add(to_part1_bad_cmodes_refused INPUT ${CONFORMANCE_DATA_DIR}/ds0_ht_01_b11.j2k MODE reject
         TRANS_ARGS "-Cmodes@@BYPASS|FAST" PATTERN "unknown -Cmodes flag")
  tc_add(cmodes_part1_input_refused INPUT ${CONFORMANCE_DATA_DIR}/p0_01.j2k MODE reject
         TRANS_ARGS "-Cmodes@@BYPASS" PATTERN "for an HTJ2K input")

  # Part 1 -> HTJ2K (-layers keep) -> Part 1: every code-block must come back with its layers.  For
  # codestreams of the encoder the result is the input, byte for byte (with the input's code-block
  # style); for the conformance ones, made by other encoders, the decode is that of the input.
  function(tc_back name)
    cmake_parse_arguments(A "BYTES" "INPUT;ENC_ARGS;BACK_ARGS" "" ${ARGN})
    set(_args -DDEC=$<TARGET_FILE:open_htj2k_dec> -DTRANS=$<TARGET_FILE:open_htj2k_transcode>
              -DWORK=${_tc_work}/back_${name})
    if(A_INPUT)
      list(APPEND _args -DINPUT=${A_INPUT})
    else()
      list(APPEND _args -DENC=$<TARGET_FILE:open_htj2k_enc> -DENC_IN=${_K} "-DENC_ARGS=${A_ENC_ARGS}")
    endif()
    if(A_BACK_ARGS)
      list(APPEND _args "-DBACK_ARGS=${A_BACK_ARGS}")
    endif()
    if(A_BYTES)
      list(APPEND _args -DBYTES=1)
    endif()
    add_test(NAME tc_back_${name} COMMAND ${CMAKE_COMMAND} ${_args}
             -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/transcoder_back.cmake)
    set_tests_properties(tc_back_${name} PROPERTIES LABELS transcoder)
  endfunction()
  set(_AM "BYPASS|RESET|RESTART|CAUSAL|ERTERM|SEGMARK")
  tc_back(gen_layers12   BYTES ENC_ARGS "Cmodes=0@@-rate@@${_L12}")
  tc_back(gen_all_modes  BYTES ENC_ARGS "Cmodes=${_AM}@@-rate@@${_L12}" BACK_ARGS "-Cmodes@@${_AM}")
  tc_back(gen_bypass_erterm BYTES ENC_ARGS "Cmodes=BYPASS|ERTERM@@-rate@@0.3,1,2.5,5" BACK_ARGS "-Cmodes@@BYPASS|ERTERM")
  tc_back(gen_restart    BYTES ENC_ARGS "Cmodes=RESTART@@-rate@@0.1,0.25,0.5,1,2" BACK_ARGS "-Cmodes@@RESTART")
  tc_back(gen_causal_pcrl BYTES ENC_ARGS "Cmodes=CAUSAL@@Corder=PCRL@@Cblk={16,16}@@-rate@@0.1,0.4,1.2,3")
  tc_back(gen_tiles_rpcl BYTES ENC_ARGS "Cmodes=0@@Stiles={128,160}@@Corder=RPCL@@Cprecincts={64,64}@@-rate@@0.2,0.6,1.5")
  tc_back(gen_sop_eph    BYTES ENC_ARGS "Cmodes=0@@Corder=PCRL@@Cuse_sop=yes@@Cuse_eph=yes@@-rate@@0.2,0.4,0.8,1.6")
  foreach(_c p0_02 p0_03 p0_08 p1_01 p1_03 p1_05)
    tc_back(conf_${_c} INPUT ${CONFORMANCE_DATA_DIR}/${_c}.j2k)
  endforeach()
  tc_back(conf_p1_02 INPUT ${CONFORMANCE_DATA_DIR}/p1_02.j2k BACK_ARGS "-Cmodes@@RESET|CAUSAL")
endif()
