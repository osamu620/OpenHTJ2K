// Fuzz target: what a JPIP client reads from a server (source/core/jpip): a JPP-stream ->
// data-bins -> the codestream index of the main header -> the reassembled codestream ->
// decode.
//
// Input = the bytes from the server, then 1 option byte (fuzz_common.hpp says why it comes
// last):
//   o0 & 3   0 a JPP-stream, parsed in one piece
//            1 a JPP-stream, fed to the streaming parser in pieces; the first piece is
//              1 + (o0 >> 2) bytes, each next one twice as long plus one
//            2, 3 an HTTP response (Content-Length or chunked) with a JPP-stream as its body
#include <algorithm>
#include <cstdint>
#include <exception>
#include <memory>
#include <vector>
#include "decoder.hpp"
#include "codestream_assembler.hpp"
#include "jpip_response.hpp"
#include "jpp_parser.hpp"
#include "precinct_index.hpp"
#include "fuzz_common.hpp"

using namespace open_htj2k;
using namespace open_htj2k::jpip;

namespace {
void reassemble_and_decode(const DataBinSet &set) {
  if (!set.contains(6, 0)) return;  // the main header data-bin
  std::unique_ptr<CodestreamIndex> idx = CodestreamIndex::build_from_main_header_bin(set.get(6, 0));
  if (!idx) return;
  std::vector<uint8_t> cs;
  if (reassemble_codestream_client(set, *idx, cs) == ReassembleStatus::Ok) {
    fuzz::decode_result(cs.data(), cs.size());
  }
}

void whole_stream(const uint8_t *p, size_t n) {
  DataBinSet set;
  (void)parse_jpp_stream(p, n, &set);
  reassemble_and_decode(set);
}
}  // namespace

extern "C" int LLVMFuzzerInitialize(int *, char ***) {
  set_message_handler(fuzz::quiet, nullptr);
  return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  if (size < 2) return 0;
  const size_t n   = size - 1;
  const uint8_t o0 = data[n];
  try {
    switch (o0 & 3) {
      case 0:
        whole_stream(data, n);
        break;
      case 1: {
        DataBinSet set;
        StreamingJppParser parser;
        size_t at = 0, step = 1u + (o0 >> 2);
        while (at < n) {
          const size_t k = std::min(step, n - at);
          if (!parser.feed(data + at, k, &set)) break;
          at += k;
          step = step * 2 + 1;
        }
        fuzz::use(static_cast<uint64_t>(parser.finish()) + parser.pending());
        reassemble_and_decode(set);
        break;
      }
      default: {
        size_t body_at        = 0;
        const size_t body_len = parse_http_content_length(data, n, &body_at);
        if (body_len != SIZE_MAX && body_at <= n) {
          whole_stream(data + body_at, std::min(body_len, n - body_at));
        }
        size_t chunks_at = 0;
        if (parse_http_chunked(data, n, &chunks_at) && chunks_at <= n) {
          std::vector<uint8_t> body;
          if (decode_chunked_body(data + chunks_at, n - chunks_at, &body)) {
            whole_stream(body.data(), body.size());
          }
        }
        std::vector<uint8_t> body;
        (void)decode_chunked_body(data, n, &body);
        break;
      }
    }
  } catch (std::exception &) {
  }
  return 0;
}
