# The decoder only reads a borrowed codestream (init_borrow()): borrow_check decodes one
# buffer twice in a row and twice at once from two threads, and requires the same samples
# every time and an unchanged buffer.  The HT decoder used to replace the last two bytes of
# every cleanup segment in place, which broke all three.  HT codestreams of one tile and of
# 2 x 2, 15 x 15 and 16 x 16 tiles, and a Part 1 one (the MQ decoder only ever read); 1 and
# 4 threads (the strip prefetch reads the buffer from pool workers).  Needs threads.
if(NOT TARGET borrow_check)
  return()
endif()
foreach(_bc ds0_ht_07_b11 ds0_ht_09_b11 ds0_ht_12_b11 ds1_ht_05_b11 ds0_hm_15_b8 p0_07)
  foreach(_t 1 4)
    add_test(NAME borrow_${_bc}_t${_t}
             COMMAND borrow_check ${CONFORMANCE_DATA_DIR}/${_bc}.j2k -threads ${_t})
    set_tests_properties(borrow_${_bc}_t${_t} PROPERTIES PASS_REGULAR_EXPRESSION "PASS:" TIMEOUT 120)
  endforeach()
endforeach()
