# Many decoder and encoder instances running at the same time, each from its
# own application thread, all sharing the process-wide worker pool.  This is
# how an image I/O library embeds the codec.  Every decode / encode result must
# match a reference produced by one instance at a time.
#
# Regression for two bugs that only show up with concurrent instances:
#   * the first instance to be destroyed shut the shared pool down while the
#     others were still pushing work to it (crash / "Cannot schedule new task
#     after shutdown");
#   * pool workers reused a per-thread encoder scratch slot across encoders,
#     so two threads could write to the same codeblock allocator.
#
# concurrent_mixed additionally runs single-threaded instances next to pooled
# ones (a pool appearing while a single-threaded instance is mid-decode).

if(TARGET concurrent_instances)
  set(_CONC_INPUT ${CONFORMANCE_DATA_DIR}/4KuF_H0L1_arri_1bpp.j2c)
  add_test(NAME concurrent_instances
           COMMAND concurrent_instances ${_CONC_INPUT} -app 8 -iter 4 -threads 4)
  add_test(NAME concurrent_mixed
           COMMAND concurrent_instances ${_CONC_INPUT} -app 8 -iter 4 -threads 4 -mixed)
endif()
