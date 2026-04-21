// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.

#ifdef OPENHTJ2K_ENABLE_QUIC

#include "h3_server.hpp"

#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <msquic.h>
#include <nghttp3/nghttp3.h>

namespace open_htj2k {
namespace jpip {

// ── Per-connection state ────────────────────────────────────────────────────

struct H3ConnCtx {
  const QUIC_API_TABLE *q       = nullptr;
  HQUIC                 conn    = nullptr;
  nghttp3_conn         *h3conn  = nullptr;
  H3RequestHandler      handler;
  std::string           error;

  // Stream map: QUIC stream handle → stream ID (for nghttp3)
  std::unordered_map<HQUIC, int64_t> stream_ids;
  std::unordered_map<int64_t, HQUIC> id_to_stream;
  int64_t next_uni_id       = 0;  // server-initiated uni: 3, 7, 11, ...
  int64_t next_peer_bidi_id = 0;  // client-initiated bidi: 0, 4, 8, ...
  int64_t next_peer_uni_id  = 0;  // client-initiated uni: 2, 6, 10, ...

  // Per-request accumulator
  struct ReqState {
    std::string method;
    std::string path;
    std::string authority;
  };
  std::unordered_map<int64_t, ReqState> requests;
};

// ── Forward declarations ────────────────────────────────────────────────────

static QUIC_STATUS QUIC_API listener_cb(HQUIC, void *, QUIC_LISTENER_EVENT *);
static QUIC_STATUS QUIC_API connection_cb(HQUIC, void *, QUIC_CONNECTION_EVENT *);
static QUIC_STATUS QUIC_API stream_cb(HQUIC, void *, QUIC_STREAM_EVENT *);

// StreamSend context: both the QUIC_BUFFER and the data must remain valid
// until QUIC_STREAM_EVENT_SEND_COMPLETE fires.
struct SendCtx {
  QUIC_BUFFER qb;
  uint8_t     data[];
};

static void h3_flush_writes(H3ConnCtx *ctx) {
  for (;;) {
    nghttp3_vec vec[16];
    int         fin  = 0;
    int64_t     sid  = -1;
    nghttp3_ssize n = nghttp3_conn_writev_stream(ctx->h3conn, &sid, &fin, vec, 16);
    if (n < 0) break;
    if (sid < 0) break;
    if (n == 0 && !fin) break;

    auto it = ctx->id_to_stream.find(sid);
    if (it == ctx->id_to_stream.end()) break;
    HQUIC stream = it->second;

    size_t total = 0;
    for (nghttp3_ssize i = 0; i < n; ++i) total += vec[i].len;

    if (total > 0 || fin) {
      auto *sc = static_cast<SendCtx *>(std::malloc(sizeof(SendCtx) + total));
      sc->qb.Length = static_cast<uint32_t>(total);
      sc->qb.Buffer = sc->data;
      size_t off = 0;
      for (nghttp3_ssize i = 0; i < n; ++i) {
        std::memcpy(sc->data + off, vec[i].base, vec[i].len);
        off += vec[i].len;
      }
      QUIC_SEND_FLAGS flags = fin ? QUIC_SEND_FLAG_FIN : QUIC_SEND_FLAG_NONE;
      QUIC_STATUS s = ctx->q->StreamSend(stream, &sc->qb, 1, flags, sc);
      if (QUIC_FAILED(s)) {
        std::free(sc);
      }
      nghttp3_conn_add_write_offset(ctx->h3conn, sid, static_cast<int64_t>(total));
    }
    if (n == 0) break;
  }
}

// ── nghttp3 server-side open unidirectional stream helper ───────────────────

static int64_t open_uni_stream(H3ConnCtx *ctx) {
  HQUIC stream = nullptr;
  QUIC_STATUS s = ctx->q->StreamOpen(ctx->conn, QUIC_STREAM_OPEN_FLAG_UNIDIRECTIONAL,
                                     stream_cb, ctx, &stream);
  if (QUIC_FAILED(s)) return -1;
  s = ctx->q->StreamStart(stream, QUIC_STREAM_START_FLAG_IMMEDIATE);
  if (QUIC_FAILED(s)) { ctx->q->StreamClose(stream); return -1; }

  // Server-initiated unidirectional stream IDs: 0x03, 0x07, 0x0B, ...
  int64_t id = 0x03 + 4 * ctx->next_uni_id++;
  ctx->stream_ids[stream] = id;
  ctx->id_to_stream[id]   = stream;
  return id;
}

// ── nghttp3 callbacks ──���────────────────────────────────────────────────────

static int h3_recv_header(nghttp3_conn *, int64_t stream_id, int32_t token,
                          nghttp3_rcbuf *name, nghttp3_rcbuf *value, uint8_t, void *conn_data,
                          void *) {
  auto *ctx = static_cast<H3ConnCtx *>(conn_data);
  auto nv = nghttp3_rcbuf_get_buf(name);
  auto vv = nghttp3_rcbuf_get_buf(value);
  std::string n(reinterpret_cast<const char *>(nv.base), nv.len);
  std::string v(reinterpret_cast<const char *>(vv.base), vv.len);

  auto &req = ctx->requests[stream_id];
  if (n == ":method")    req.method = v;
  else if (n == ":path") req.path   = v;
  else if (n == ":authority") req.authority = v;
  return 0;
}

static int h3_end_headers(nghttp3_conn *, int64_t, int, void *, void *) {
  return 0;
}

static int h3_end_stream(nghttp3_conn *, int64_t stream_id, void *conn_data, void *) {
  auto *ctx = static_cast<H3ConnCtx *>(conn_data);
  auto it = ctx->requests.find(stream_id);
  if (it == ctx->requests.end()) return 0;

  H3Request req;
  req.stream_id = stream_id;
  req.method    = it->second.method;
  auto qpos = it->second.path.find('?');
  if (qpos != std::string::npos) {
    req.path  = it->second.path.substr(0, qpos);
    req.query = it->second.path.substr(qpos + 1);
  } else {
    req.path = it->second.path;
  }
  ctx->requests.erase(it);

  // The HTTP/3 handler still builds the full JPP-stream before the data
  // reader hands it to nghttp3 — progressive delivery over H3 would need
  // the handler to produce JPP messages incrementally and feed them to
  // `read_data` across multiple invocations.  That is a follow-up to
  // issue #297 once the HTTP/1.1 chunked path is proven out.  On the
  // wire the response still lands as one or more H3 DATA frames (nghttp3
  // fragments the body by QUIC flow-control window), so gigapixel views
  // are not penalised any worse than they already were.
  std::vector<uint8_t> body = ctx->handler(req);

  // Submit HTTP/3 response
  char content_length[32];
  std::snprintf(content_length, sizeof(content_length), "%zu", body.size());
  nghttp3_nv nva[] = {
    {(uint8_t *)":status", (uint8_t *)"200", 7, 3, NGHTTP3_NV_FLAG_NONE},
    {(uint8_t *)"content-type", (uint8_t *)"image/jpp-stream", 12, 16, NGHTTP3_NV_FLAG_NONE},
    {(uint8_t *)"access-control-allow-origin", (uint8_t *)"*", 27, 1, NGHTTP3_NV_FLAG_NONE},
    {(uint8_t *)"content-length", (uint8_t *)content_length, 14,
     std::strlen(content_length), NGHTTP3_NV_FLAG_NO_COPY_NAME | NGHTTP3_NV_FLAG_NO_COPY_VALUE},
  };

  // Store body for the data provider
  struct BodyCtx { std::vector<uint8_t> data; size_t offset = 0; };
  auto *bc = new BodyCtx{std::move(body), 0};

  nghttp3_data_reader dr;
  dr.read_data = [](nghttp3_conn *, int64_t, nghttp3_vec *vec, size_t veccnt,
                     uint32_t *pflags, void *conn_data, void *stream_data) -> nghttp3_ssize {
    (void)conn_data;
    auto *b = static_cast<BodyCtx *>(stream_data);
    if (veccnt == 0) return 0;
    size_t remain = b->data.size() - b->offset;
    if (remain == 0) {
      *pflags = NGHTTP3_DATA_FLAG_EOF;
      return 0;
    }
    vec[0].base = b->data.data() + b->offset;
    vec[0].len  = remain;
    b->offset  += remain;
    *pflags = NGHTTP3_DATA_FLAG_EOF;
    return 1;
  };

  nghttp3_conn_set_stream_user_data(ctx->h3conn, stream_id, bc);
  nghttp3_conn_submit_response(ctx->h3conn, stream_id, nva, 4, &dr);
  h3_flush_writes(ctx);
  return 0;
}

static int h3_acked_stream_data(nghttp3_conn *, int64_t, uint64_t, void *, void *stream_data) {
  // Body fully acknowledged — free the BodyCtx
  // Note: we keep the BodyCtx alive until stream close for simplicity
  (void)stream_data;
  return 0;
}

static int h3_stream_close(nghttp3_conn *, int64_t, uint64_t, void *, void *stream_data) {
  // Free per-stream body context
  if (stream_data) {
    struct BodyCtx { std::vector<uint8_t> data; size_t offset; };
    delete static_cast<BodyCtx *>(stream_data);
  }
  return 0;
}

static int h3_deferred_consume(nghttp3_conn *, int64_t stream_id, size_t consumed,
                               void *conn_data, void *) {
  auto *ctx = static_cast<H3ConnCtx *>(conn_data);
  auto it = ctx->id_to_stream.find(stream_id);
  if (it != ctx->id_to_stream.end()) {
    ctx->q->StreamReceiveComplete(it->second, consumed);
  }
  return 0;
}

// ── Setup nghttp3 on a new connection ───────────────────────────────────────

static bool setup_h3(H3ConnCtx *ctx) {
  nghttp3_callbacks cb = {};
  cb.recv_header       = h3_recv_header;
  cb.end_headers       = h3_end_headers;
  cb.end_stream        = h3_end_stream;
  cb.acked_stream_data = h3_acked_stream_data;
  cb.stream_close      = h3_stream_close;
  cb.deferred_consume  = h3_deferred_consume;

  nghttp3_settings settings;
  nghttp3_settings_default(&settings);

  int rv = nghttp3_conn_server_new(&ctx->h3conn, &cb, &settings, nghttp3_mem_default(), ctx);
  if (rv != 0) return false;

  int64_t ctrl = open_uni_stream(ctx);
  int64_t qenc = open_uni_stream(ctx);
  int64_t qdec = open_uni_stream(ctx);
  if (ctrl < 0 || qenc < 0 || qdec < 0) return false;

  rv = nghttp3_conn_bind_control_stream(ctx->h3conn, ctrl);
  if (rv != 0) return false;
  rv = nghttp3_conn_bind_qpack_streams(ctx->h3conn, qenc, qdec);
  if (rv != 0) return false;
  h3_flush_writes(ctx);
  return true;
}

// ── MsQuic stream callback ─────────────────────────────────────────────────

static QUIC_STATUS QUIC_API stream_cb(HQUIC stream, void *context, QUIC_STREAM_EVENT *event) {
  auto *ctx = static_cast<H3ConnCtx *>(context);
  switch (event->Type) {
    case QUIC_STREAM_EVENT_RECEIVE: {
      auto it = ctx->stream_ids.find(stream);
      if (it == ctx->stream_ids.end()) break;
      int64_t sid = it->second;
      for (uint32_t i = 0; i < event->RECEIVE.BufferCount; ++i) {
        nghttp3_conn_read_stream(ctx->h3conn, sid,
                                 event->RECEIVE.Buffers[i].Buffer,
                                 event->RECEIVE.Buffers[i].Length, 0);
      }
      if (event->RECEIVE.Flags & QUIC_RECEIVE_FLAG_FIN) {
        nghttp3_conn_read_stream(ctx->h3conn, sid, nullptr, 0, 1);
      }
      h3_flush_writes(ctx);
      break;
    }
    case QUIC_STREAM_EVENT_SEND_COMPLETE: {
      std::free(event->SEND_COMPLETE.ClientContext);
      break;
    }
    case QUIC_STREAM_EVENT_SHUTDOWN_COMPLETE: {
      auto it = ctx->stream_ids.find(stream);
      if (it != ctx->stream_ids.end()) {
        ctx->id_to_stream.erase(it->second);
        ctx->stream_ids.erase(it);
      }
      ctx->q->StreamClose(stream);
      break;
    }
    default:
      break;
  }
  return QUIC_STATUS_SUCCESS;
}

// ── MsQuic connection callback ──────────────────────────────────────────────

static QUIC_STATUS QUIC_API connection_cb(HQUIC conn, void *context,
                                          QUIC_CONNECTION_EVENT *event) {
  auto *ctx = static_cast<H3ConnCtx *>(context);
  switch (event->Type) {
    case QUIC_CONNECTION_EVENT_CONNECTED:
      setup_h3(ctx);
      break;
    case QUIC_CONNECTION_EVENT_PEER_STREAM_STARTED: {
      HQUIC stream = event->PEER_STREAM_STARTED.Stream;
      ctx->q->SetCallbackHandler(stream, (void *)stream_cb, ctx);
      int64_t sid;
      if (event->PEER_STREAM_STARTED.Flags & QUIC_STREAM_OPEN_FLAG_UNIDIRECTIONAL) {
        sid = 0x02 + 4 * ctx->next_peer_uni_id++;
      } else {
        sid = 4 * ctx->next_peer_bidi_id++;
      }
      ctx->stream_ids[stream]  = sid;
      ctx->id_to_stream[sid]   = stream;
      break;
    }
    case QUIC_CONNECTION_EVENT_SHUTDOWN_COMPLETE:
      if (ctx->h3conn) { nghttp3_conn_del(ctx->h3conn); ctx->h3conn = nullptr; }
      ctx->q->ConnectionClose(conn);
      delete ctx;
      break;
    default:
      break;
  }
  return QUIC_STATUS_SUCCESS;
}

// ── H3Server::Impl ──────────────────────────────────────────────────────────

struct H3Server::Impl {
  const QUIC_API_TABLE *q      = nullptr;
  HQUIC                 reg    = nullptr;
  HQUIC                 config = nullptr;
  HQUIC                 listener = nullptr;
  H3RequestHandler      handler;
  std::string           error;
};

static QUIC_STATUS QUIC_API listener_cb(HQUIC listener, void *context,
                                        QUIC_LISTENER_EVENT *event) {
  auto *impl = static_cast<H3Server::Impl *>(context);
  if (event->Type == QUIC_LISTENER_EVENT_NEW_CONNECTION) {
    auto *ctx     = new H3ConnCtx();
    ctx->q        = impl->q;
    ctx->conn     = event->NEW_CONNECTION.Connection;
    ctx->handler  = impl->handler;
    impl->q->SetCallbackHandler(event->NEW_CONNECTION.Connection,
                                (void *)connection_cb, ctx);
    return impl->q->ConnectionSetConfiguration(event->NEW_CONNECTION.Connection,
                                               impl->config);
  }
  return QUIC_STATUS_NOT_SUPPORTED;
}

// ── H3Server public API ─────────────────────────────────────────────────────

H3Server::H3Server() : impl_(new Impl()) {}

H3Server::~H3Server() {
  stop();
  delete impl_;
}

bool H3Server::start(uint16_t port, const TlsCertConfig &tls, H3RequestHandler handler) {
  impl_->handler = std::move(handler);

  if (QUIC_FAILED(MsQuicOpen2(&impl_->q))) {
    impl_->error = "MsQuicOpen2 failed";
    return false;
  }

  QUIC_REGISTRATION_CONFIG rc = {"jpip-h3-server", QUIC_EXECUTION_PROFILE_LOW_LATENCY};
  if (QUIC_FAILED(impl_->q->RegistrationOpen(&rc, &impl_->reg))) {
    impl_->error = "RegistrationOpen failed";
    return false;
  }

  const QUIC_BUFFER alpn = {2, (uint8_t *)"h3"};
  QUIC_SETTINGS settings = {};
  settings.IdleTimeoutMs = 30000;
  settings.IsSet.IdleTimeoutMs = TRUE;
  settings.PeerBidiStreamCount = 128;
  settings.IsSet.PeerBidiStreamCount = TRUE;
  settings.PeerUnidiStreamCount = 8;
  settings.IsSet.PeerUnidiStreamCount = TRUE;

  if (QUIC_FAILED(impl_->q->ConfigurationOpen(impl_->reg, &alpn, 1, &settings,
                                               sizeof(settings), nullptr, &impl_->config))) {
    impl_->error = "ConfigurationOpen failed";
    return false;
  }

  QUIC_CERTIFICATE_FILE cert_file = {};
  cert_file.CertificateFile = tls.cert_file.c_str();
  cert_file.PrivateKeyFile  = tls.key_file.c_str();
  QUIC_CREDENTIAL_CONFIG cred = {};
  cred.Type = QUIC_CREDENTIAL_TYPE_CERTIFICATE_FILE;
  cred.CertificateFile = &cert_file;

  if (QUIC_FAILED(impl_->q->ConfigurationLoadCredential(impl_->config, &cred))) {
    impl_->error = "ConfigurationLoadCredential failed";
    return false;
  }

  if (QUIC_FAILED(impl_->q->ListenerOpen(impl_->reg, listener_cb, impl_, &impl_->listener))) {
    impl_->error = "ListenerOpen failed";
    return false;
  }

  QUIC_ADDR addr = {};
  QuicAddrSetFamily(&addr, QUIC_ADDRESS_FAMILY_UNSPEC);
  QuicAddrSetPort(&addr, port);
  if (QUIC_FAILED(impl_->q->ListenerStart(impl_->listener, &alpn, 1, &addr))) {
    impl_->error = "ListenerStart failed";
    return false;
  }

  return true;
}

void H3Server::stop() {
  if (impl_->listener) { impl_->q->ListenerClose(impl_->listener); impl_->listener = nullptr; }
  if (impl_->config)   { impl_->q->ConfigurationClose(impl_->config); impl_->config = nullptr; }
  if (impl_->reg)      { impl_->q->RegistrationClose(impl_->reg); impl_->reg = nullptr; }
  if (impl_->q)        { MsQuicClose(impl_->q); impl_->q = nullptr; }
}

std::string H3Server::last_error() const { return impl_->error; }

}  // namespace jpip
}  // namespace open_htj2k

#endif  // OPENHTJ2K_ENABLE_QUIC
