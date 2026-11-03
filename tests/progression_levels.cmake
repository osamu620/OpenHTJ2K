# The position-driven progression orders (RPCL, PCRL, CPRL and the Part 2 PRCL) visit
# a precinct where the reference-grid position is a multiple of
# XRsiz * 2^(PPx + NL - r) (ISO/IEC 15444-1 B.12.1.3 to B.12.1.5).  With maximal
# precincts (PPx = 15) that multiple took a 32-bit shift by up to 47 and overflowed
# from 16 DWT levels on: with a sub-sampled component (XRsiz = 2) it wrapped to 0, and
# both the encoder and the decoder stopped on an integer division by zero; without
# sub-sampling the shift was undefined and the packets went in the wrong order.
#
# synth422_{y,cb,cr}.pgx are a 64 x 64 luma plane and two 32 x 64 chroma planes
# (XRsiz = 2).  Each case must round-trip losslessly; a release build of the old encoder
# crashed on all of them, and OpenJPEG 2.5.4 decodes the new codestreams of the Part 1
# cases to the same samples.
set(_PROG_REF_DIR "${CMAKE_CURRENT_SOURCE_DIR}/conformance_data/references")
set(_PROG_IN "${_PROG_REF_DIR}/synth422_y.pgx,${_PROG_REF_DIR}/synth422_cb.pgx,${_PROG_REF_DIR}/synth422_cr.pgx")
set(_PROG_PLANES synth422_y synth422_cb synth422_cr)

# name:levels:order:Cmodes
set(_PROG_CASES
    prog_nl16_rpcl:16:RPCL:HT
    prog_nl17_pcrl:17:PCRL:HT
    prog_nl32_cprl:32:CPRL:HT
    prog_nl32_prcl:32:PRCL:HT
    prog_nl17_rpcl_p1:17:RPCL:0
    prog_nl32_pcrl_p1:32:PCRL:0)
foreach(_entry IN LISTS _PROG_CASES)
  string(REPLACE ":" ";" _case "${_entry}")
  list(GET _case 0 _n)
  list(GET _case 1 _nl)
  list(GET _case 2 _order)
  list(GET _case 3 _mode)
  add_test(NAME enc_${_n}
           COMMAND open_htj2k_enc -i ${_PROG_IN} -o ${_n}.j2c Clevels=${_nl} Corder=${_order}
                   Creversible=yes Cycc=no Cmodes=${_mode})
  foreach(_t 1 4)
    add_test(NAME dec_${_n}_t${_t} COMMAND open_htj2k_dec -i ${_n}.j2c -o ${_n}_t${_t}.pgx -num_threads ${_t})
    set_tests_properties(dec_${_n}_t${_t} PROPERTIES DEPENDS enc_${_n})
    foreach(_c 0 1 2)
      list(GET _PROG_PLANES ${_c} _p)
      add_test(NAME comp_${_n}_t${_t}_${_c} COMMAND imgcmp ${_n}_t${_t}_0${_c}.pgx ${_PROG_REF_DIR}/${_p}.pgx 0 0)
      set_tests_properties(comp_${_n}_t${_t}_${_c} PROPERTIES DEPENDS dec_${_n}_t${_t})
    endforeach()
  endforeach()
endforeach()

# An image wider than one maximal precinct (32800 > 2^15 columns) has two precinct-grid
# positions per row.  At 17 levels the multiple for the lowest resolution is 2^32: the old
# encoder and decoder took it as 1, visited that resolution's one precinct again at column
# 32768 and stopped with "attempt to access precinct whose index is out of the valid
# range", also on such codestreams from other encoders.  The image (one gray 8-bit
# component, two rows of the digits and letters repeated) is written at configure time,
# into a subdirectory that cleanup_artifacts leaves alone.
set(_PROG_WIDE_PGX ${CMAKE_CURRENT_BINARY_DIR}/test_inputs/prog_wide_src.pgx)
set(_PROG_WIDE_ROW "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
foreach(_i RANGE 10)  # 62 * 2^11 > 2 * 32800 samples
  set(_PROG_WIDE_ROW "${_PROG_WIDE_ROW}${_PROG_WIDE_ROW}")
endforeach()
string(SUBSTRING "${_PROG_WIDE_ROW}" 0 65600 _PROG_WIDE_ROW)
file(WRITE ${_PROG_WIDE_PGX} "PG ML + 8 32800 2\n${_PROG_WIDE_ROW}")
foreach(_entry prog_wide_nl17_rpcl:17:RPCL:0 prog_wide_nl17_pcrl:17:PCRL:HT prog_wide_nl20_cprl:20:CPRL:0)
  string(REPLACE ":" ";" _case "${_entry}")
  list(GET _case 0 _n)
  list(GET _case 1 _nl)
  list(GET _case 2 _order)
  list(GET _case 3 _mode)
  add_test(NAME enc_${_n}
           COMMAND open_htj2k_enc -i ${_PROG_WIDE_PGX} -o ${_n}.j2c Clevels=${_nl} Corder=${_order}
                   Creversible=yes Cmodes=${_mode})
  foreach(_t 1 4)
    add_test(NAME dec_${_n}_t${_t} COMMAND open_htj2k_dec -i ${_n}.j2c -o ${_n}_t${_t}.pgx -num_threads ${_t})
    set_tests_properties(dec_${_n}_t${_t} PROPERTIES DEPENDS enc_${_n})
    add_test(NAME comp_${_n}_t${_t} COMMAND imgcmp ${_n}_t${_t}_00.pgx ${_PROG_WIDE_PGX} 0 0)
    set_tests_properties(comp_${_n}_t${_t} PROPERTIES DEPENDS dec_${_n}_t${_t})
  endforeach()
endforeach()

# The positions those orders examine along each axis of a tile were counted from 0, not
# from the tile: with the image origin at 2147418112 and 2 x 2 precincts every tile took
# a billion steps.  The 192 4 x 4 tiles of dwt_levels_src.pgm (61 x 47) took the old
# encoder 52 s and the old decoder 69 s; both now take milliseconds, with the same
# codestream.
set(_PROG_FAR_SRC ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/dwt_levels_src.pgm)
add_test(NAME enc_prog_far_origin
         COMMAND open_htj2k_enc -i ${_PROG_FAR_SRC} -o prog_far_origin.j2c "Sorigin={0,2147418112}"
                 "Stiles={4,4}" "Cprecincts={2,2}" Corder=RPCL Creversible=yes Clevels=1)
set_tests_properties(enc_prog_far_origin PROPERTIES TIMEOUT 20)
foreach(_t 1 4)
  add_test(NAME dec_prog_far_origin_t${_t}
           COMMAND open_htj2k_dec -i prog_far_origin.j2c -o prog_far_origin_t${_t}.pgm -num_threads ${_t})
  set_tests_properties(dec_prog_far_origin_t${_t} PROPERTIES DEPENDS enc_prog_far_origin TIMEOUT 20)
  add_test(NAME comp_prog_far_origin_t${_t} COMMAND imgcmp prog_far_origin_t${_t}_00.pgm ${_PROG_FAR_SRC} 0 0)
  set_tests_properties(comp_prog_far_origin_t${_t} PROPERTIES DEPENDS dec_prog_far_origin_t${_t})
endforeach()
