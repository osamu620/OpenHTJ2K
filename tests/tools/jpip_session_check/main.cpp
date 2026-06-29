// jpip_session_check: ctest harness for JPIP session/channel support
// (ISO/IEC 15444-9 §B.2, §C.3, §C.9, §D.2.3, §D.2.4).
//
// Covers the interop fixes for session-mode clients:
//   - ChannelManager: transport negotiation (Table D.1 exact-token rule),
//     per-cid cache-model continuation, LRU eviction, cclose.
//   - Request parser: type preference lists + media-type form (§C.4),
//     qid (§C.3.5), cclose (§C.3.4), comps ranges (§C.4.6).
//   - CacheModel: ":bytes" partial-bin qualifier and "-" subtractive
//     statements (§C.9.2).
//   - Response formatter: JPIP-qid echo via extra headers (§D.2.4).
//
// Self-contained — runs every test internally, takes no input.
#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include "cache_model.hpp"
#include "channel_manager.hpp"
#include "data_bin_emitter.hpp"
#include "jpip_request.hpp"
#include "jpip_response.hpp"
#include "jpp_message.hpp"

using open_htj2k::jpip::BinWindow;
using open_htj2k::jpip::CacheModel;
using open_htj2k::jpip::ChannelManager;
using open_htj2k::jpip::emit_metadata_bin_zero;
using open_htj2k::jpip::MessageHeaderContext;
using open_htj2k::jpip::format_jpp_response;
using open_htj2k::jpip::format_jpp_response_headers_chunked;
using open_htj2k::jpip::JpipRequest;
using open_htj2k::jpip::kMsgClassMainHeader;
using open_htj2k::jpip::kMsgClassMetadata;
using open_htj2k::jpip::kMsgClassPrecinct;
using open_htj2k::jpip::parse_jpip_query;
using open_htj2k::jpip::RequestParseStatus;

namespace {

int failures = 0;

#define CHECK(cond, ...)                                            \
  do {                                                              \
    if (!(cond)) {                                                  \
      std::fprintf(stderr, "FAIL [%s:%d] %s — ", __FILE__, __LINE__, #cond); \
      std::fprintf(stderr, __VA_ARGS__);                            \
      std::fprintf(stderr, "\n");                                   \
      ++failures;                                                   \
    }                                                               \
  } while (0)

bool contains(const std::vector<uint8_t> &resp, const std::string &needle) {
  const std::string s(resp.begin(), resp.end());
  return s.find(needle) != std::string::npos;
}

}  // namespace

int main() {
  // ── Transport negotiation (§C.3.3 + Table D.1) ────────────────────────
  {
    CHECK(ChannelManager::negotiate_transport("http") == "http", "plain http");
    CHECK(ChannelManager::negotiate_transport("http-tcp,http") == "http", "list with http");
    CHECK(ChannelManager::negotiate_transport(" http ") == "http", "whitespace trim");
    // Table D.1: granted transport shall be one the client supplied —
    // "http-tcp" alone must NOT be answered with "http".
    CHECK(ChannelManager::negotiate_transport("http-tcp").empty(), "http-tcp alone");
    CHECK(ChannelManager::negotiate_transport("http-udp").empty(), "http-udp alone");
    CHECK(ChannelManager::negotiate_transport("https").empty(), "https unsupported");
    CHECK(ChannelManager::negotiate_transport("").empty(), "empty list");
  }

  // ── ChannelManager lifecycle ──────────────────────────────────────────
  {
    ChannelManager cm;
    const std::string cid = cm.open();
    CHECK(!cid.empty(), "cid issued");
    CacheModel snap;
    CHECK(cm.snapshot(cid, &snap), "snapshot known cid");
    CHECK(snap.size() == 0, "fresh channel model empty");
    CHECK(!cm.snapshot("bogus", &snap), "snapshot unknown cid");

    // Bins committed after delivery must be visible to the next request.
    // Partial deliveries record their resume offset; complete ones the flag.
    cm.commit(cid, {{kMsgClassMetadata, 0, 0, true},
                    {kMsgClassMainHeader, 0, 142, true},
                    {kMsgClassPrecinct, 7, 987, false}});
    CHECK(cm.snapshot(cid, &snap), "snapshot after commit");
    CHECK(snap.has(kMsgClassMetadata, 0), "metadata committed");
    CHECK(snap.has(kMsgClassMainHeader, 0), "main header committed");
    CHECK(!snap.has(kMsgClassPrecinct, 7), "partial precinct not complete");
    CHECK(snap.received_bytes(kMsgClassPrecinct, 7) == 987, "partial resume offset kept");
    CHECK(!snap.has(kMsgClassPrecinct, 8), "uncommitted bin absent");

    // Completing the partial bin on a later response upgrades it.
    cm.commit(cid, {{kMsgClassPrecinct, 7, 1500, true}});
    CHECK(cm.snapshot(cid, &snap), "snapshot after completion");
    CHECK(snap.has(kMsgClassPrecinct, 7), "precinct completed");

    // Client-side model updates fold into the channel (§C.9).
    CHECK(cm.apply_model(cid, "-P7"), "apply subtractive");
    CHECK(cm.snapshot(cid, &snap), "snapshot after apply");
    CHECK(!snap.has(kMsgClassPrecinct, 7), "subtractive unmarked precinct");

    CHECK(cm.close(cid), "close known cid");
    CHECK(!cm.close(cid), "double close");
    CHECK(!cm.snapshot(cid, &snap), "snapshot after close");
  }

  // ── LRU eviction at the channel cap ───────────────────────────────────
  {
    ChannelManager cm(2);
    const std::string a = cm.open();
    const std::string b = cm.open();
    CacheModel snap;
    CHECK(cm.snapshot(a, &snap), "touch a (now b is LRU)");
    const std::string c = cm.open();  // evicts b
    CHECK(cm.size() == 2, "cap respected");
    CHECK(cm.snapshot(a, &snap), "a survived");
    CHECK(cm.snapshot(c, &snap), "c present");
    CHECK(!cm.snapshot(b, &snap), "b evicted");
  }

  // ── type preference list + media-type form (§C.4) ─────────────────────
  {
    JpipRequest req;
    CHECK(parse_jpip_query("type=jpp-stream", &req) == RequestParseStatus::Ok, "short token");
    CHECK(req.type == "jpp-stream", "type stored");
    CHECK(parse_jpip_query("type=jpp-stream,jpt-stream", &req) == RequestParseStatus::Ok,
          "list with acceptable first");
    CHECK(parse_jpip_query("type=jpt-stream,jpp-stream", &req) == RequestParseStatus::Ok,
          "list with acceptable second");
    CHECK(req.type == "jpp-stream", "served type is jpp-stream");
    CHECK(parse_jpip_query("type=image/jpp-stream", &req) == RequestParseStatus::Ok,
          "media-type form");
    CHECK(parse_jpip_query("type=jpp-stream;ptype=ext", &req) == RequestParseStatus::Ok,
          "parameter suffix ignored");
    CHECK(parse_jpip_query("type=jpt-stream", &req) == RequestParseStatus::UnsupportedType,
          "jpt only rejected");
    CHECK(parse_jpip_query("type=raw", &req) == RequestParseStatus::UnsupportedType,
          "raw rejected");
  }

  // ── qid (§C.3.5) and cclose (§C.3.4) ──────────────────────────────────
  {
    JpipRequest req;
    CHECK(parse_jpip_query("qid=42&cid=JPH1", &req) == RequestParseStatus::Ok, "qid parse");
    CHECK(req.has_qid && req.qid == 42, "qid value");
    CHECK(req.has_cid && req.cid == "JPH1", "cid value");
    CHECK(parse_jpip_query("qid=x7", &req) == RequestParseStatus::MalformedField, "qid malformed");

    CHECK(parse_jpip_query("cclose=*&cid=JPH2", &req) == RequestParseStatus::Ok, "cclose star");
    CHECK(req.has_cclose && req.cclose == "*", "cclose star value");
    CHECK(parse_jpip_query("cclose=JPH3,JPH4", &req) == RequestParseStatus::Ok, "cclose list");
    CHECK(req.cclose == "JPH3,JPH4", "cclose list value");
  }

  // ── comps ranges (§C.4.6) ─────────────────────────────────────────────
  {
    JpipRequest req;
    CHECK(parse_jpip_query("comps=0-2", &req) == RequestParseStatus::Ok, "comps range");
    CHECK(req.view_window.comps.size() == 3 && req.view_window.comps[2] == 2, "range expanded");
    CHECK(parse_jpip_query("comps=0,3-5", &req) == RequestParseStatus::Ok, "mixed list");
    CHECK(req.view_window.comps.size() == 4 && req.view_window.comps[3] == 5, "mixed expanded");
    CHECK(parse_jpip_query("comps=2-1", &req) == RequestParseStatus::MalformedField,
          "descending range rejected");
    CHECK(parse_jpip_query("comps=x", &req) == RequestParseStatus::MalformedField,
          "non-numeric rejected");
    // A range wider than the 16384-component maximum must be rejected, not
    // expanded into a multi-billion-entry push loop (DoS).
    CHECK(parse_jpip_query("comps=0-4294967295", &req) == RequestParseStatus::MalformedField,
          "huge comps range rejected");
    CHECK(parse_jpip_query("comps=0-16383", &req) == RequestParseStatus::Ok,
          "full 16384-component range still accepted");
    CHECK(req.view_window.comps.size() == 16384, "full range expands to 16384, got %zu",
          req.view_window.comps.size());
    CHECK(parse_jpip_query("comps=0-16384", &req) == RequestParseStatus::MalformedField,
          "one past the component cap rejected");
  }

  // ── CacheModel: partial-bin qualifier + subtractive statements (§C.9.2) ─
  {
    CacheModel m = CacheModel::parse("Hm,P0-9,-P5");
    CHECK(m.has(kMsgClassMainHeader, 0), "Hm marked");
    CHECK(m.has(kMsgClassPrecinct, 4), "P4 marked");
    CHECK(!m.has(kMsgClassPrecinct, 5), "P5 unmarked by subtractive");

    // A partial holding is not complete (cannot satisfy a whole-bin skip)
    // but records its byte count so delivery resumes there (§C.9.2).
    m = CacheModel::parse("P5:1234");
    CHECK(!m.has(kMsgClassPrecinct, 5), "partial precinct not complete");
    CHECK(m.received_bytes(kMsgClassPrecinct, 5) == 1234, "partial precinct byte count");
    m = CacheModel::parse("Hm:20");
    CHECK(!m.has(kMsgClassMainHeader, 0), "partial main header not complete");
    CHECK(m.received_bytes(kMsgClassMainHeader, 0) == 20, "partial main header byte count");

    // Holdings only grow; completion wins over any byte count.
    m = CacheModel::parse("P5:1234");
    m.mark_partial(kMsgClassPrecinct, 5, 100);
    CHECK(m.received_bytes(kMsgClassPrecinct, 5) == 1234, "smaller partial is a no-op");
    m.mark_partial(kMsgClassPrecinct, 5, 2000);
    CHECK(m.received_bytes(kMsgClassPrecinct, 5) == 2000, "larger partial grows");
    m.mark(kMsgClassPrecinct, 5);
    CHECK(m.has(kMsgClassPrecinct, 5), "mark upgrades to complete");

    // format() round-trips partial holdings with the ":bytes" qualifier.
    m.clear();
    m.mark_partial(kMsgClassPrecinct, 3, 555);
    CacheModel rt = CacheModel::parse(m.format());
    CHECK(!rt.has(kMsgClassPrecinct, 3), "round-trip partial not complete");
    CHECK(rt.received_bytes(kMsgClassPrecinct, 3) == 555, "round-trip partial bytes");

    // apply() folds updates onto existing state in statement order.
    m = CacheModel::parse("P0-3");
    m.apply("-P1,P9");
    CHECK(!m.has(kMsgClassPrecinct, 1), "apply subtractive");
    CHECK(m.has(kMsgClassPrecinct, 9), "apply additive");
    CHECK(m.has(kMsgClassPrecinct, 0), "prior state kept");

    // A maximal range must not spin ~2^64 iterations: apply() stops at its
    // bin-operation budget (kMaxModelBinOps = 1<<22) and returns with a
    // bounded model.  Reaching this CHECK at all proves it terminated.
    CacheModel big;
    big.apply("P0-18446744073709551615");
    CHECK(big.size() == (static_cast<std::size_t>(1) << 22),
          "huge model range bounded to the budget, got %zu", big.size());
    CHECK(big.has(kMsgClassPrecinct, 0), "low end of the range still applied");
  }

  // ── BinWindow: budget-blocked vs complete (empty metadata bin) ────────
  {
    MessageHeaderContext ctx;
    std::vector<uint8_t> out;
    BinWindow blocked;
    blocked.budget = 1;  // smaller than any message header
    emit_metadata_bin_zero(ctx, out, &blocked);
    CHECK(out.empty(), "nothing appended under tiny budget");
    CHECK(blocked.budget_blocked && !blocked.complete, "tiny budget reports blocked");

    MessageHeaderContext ctx2;
    BinWindow open_window;  // default: no cap
    emit_metadata_bin_zero(ctx2, out, &open_window);
    CHECK(!out.empty(), "empty bin emits its is_last message");
    CHECK(open_window.complete && open_window.payload_sent == 0 && !open_window.budget_blocked,
          "empty bin completes with zero payload");
  }

  // ── JPIP-qid echo via extra headers (§D.2.4) ──────────────────────────
  {
    const uint8_t body[] = {0x00};
    auto resp = format_jpp_response(body, sizeof(body), "tid-x", "cid=JPH9,transport=http",
                                    "JPIP-qid: 7\r\nCache-Control: no-cache\r\n");
    CHECK(contains(resp, "JPIP-qid: 7\r\n"), "qid echoed");
    CHECK(contains(resp, "Cache-Control: no-cache\r\n"), "no-cache present");
    CHECK(contains(resp, "JPIP-cnew: cid=JPH9,transport=http\r\n"), "cnew present");

    auto hdrs = format_jpp_response_headers_chunked("tid-x", "", "JPIP-qid: 3\r\n");
    CHECK(contains(hdrs, "JPIP-qid: 3\r\n"), "qid echoed (chunked)");
  }

  if (failures) {
    std::fprintf(stderr, "%d check(s) FAILED\n", failures);
    return 1;
  }
  std::printf("jpip_session_check: all checks passed\n");
  return 0;
}
