// Part of the install_consumer test (see consumer.cpp): includes the installed
// headers only as <open_htj2k/interface/...>, the form that cannot collide
// with another project's decoder.hpp or encoder.hpp.
#include <open_htj2k/interface/decoder.hpp>
#include <open_htj2k/interface/encoder.hpp>

bool consumer_paths_ok();
bool consumer_paths_ok() {
  open_htj2k::PlanarOutputDesc desc = {};
  return desc.base == nullptr && open_htj2k::ENUMCS_GRAYSCALE == 17u;
}
