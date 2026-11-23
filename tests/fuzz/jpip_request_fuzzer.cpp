// Fuzz target: what a JPIP server reads from a client (source/core/jpip): the request line,
// the query string and its fields, the cache model, and the channel requests.
//
// Input = the text of a request: a request line ("GET /path?query HTTP/1.1"), or a query
// string alone.  No option bytes.
//
// The cache model is read within the limits of a small image, as a server reads it (a
// statement is a few bytes and can name 2^64 data-bins; see CacheModel::Limits).
#include <cstdint>
#include <exception>
#include <string>
#include "decoder.hpp"
#include "cache_model.hpp"
#include "channel_manager.hpp"
#include "jpip_request.hpp"
#include "fuzz_common.hpp"

using namespace open_htj2k;
using namespace open_htj2k::jpip;

extern "C" int LLVMFuzzerInitialize(int *, char ***) {
  set_message_handler(fuzz::quiet, nullptr);
  return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  if (size == 0) return 0;
  const std::string text(reinterpret_cast<const char *>(data), size);
  CacheModel::Limits image;
  image.precinct_ids    = 1u << 16;
  image.tile_header_ids = 1u << 10;
  image.metadata_ids    = 16;
  image.bins            = 1u << 16;
  try {
    std::string path, query;
    if (split_http_get_line(text, &path, &query)) {
      JpipRequest request;
      (void)parse_jpip_query(query, &request);
    }
    JpipRequest request;
    (void)parse_jpip_query(text, &request);
    CacheModel model = CacheModel::parse(request.model.empty() ? text : request.model, image);
    fuzz::use(model.size() + model.format().size());
    model.apply(text, image);
    ChannelManager channels;
    const std::string cid = channels.open();
    (void)channels.apply_model(cid, text, image);
    (void)channels.apply_model(text, text, image);
    (void)channels.close(text);
    fuzz::use(ChannelManager::negotiate_transport(text).size());
  } catch (std::exception &) {
  }
  return 0;
}
