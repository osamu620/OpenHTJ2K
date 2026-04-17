// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.

#ifdef OPENHTJ2K_ENABLE_QUIC

#include "h3_client.hpp"

#include <cstdio>
#include <cstring>
#include <condition_variable>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <msquic.h>
#include <nghttp3/nghttp3.h>

namespace open_htj2k {
namespace jpip {

// ── Per-connection state ────────────────────────────────────────────────────

struct ClientConn {
  const QUIC_API_TABLE *q      = nullptr;
  HQUIC                 conn   = nullptr;
  HQUIC                 config = nullptr;
  nghttp3_conn         *h3conn = nullptr;

  std::unordered_map<HQUIC, int64_t> stream_ids;
  std::unordered_map<int64_t, HQUIC> id_to_stream;
  int64_t next_uni_id = 0;

  // Synchronisation for blocking fetch()
  std::mutex              mu;
  std::condition_variable cv;
  bool connected   = false;
  bool conn_failed = false;

  // Per-request response accumulator
  struct RespState {
    std::vector<uint8_t> body;
    bool                 done = false;
  };
  std::unordered_map<int64_t, RespState> responses;
};

// ── Forward declarations ────────────────────────────────────────────────────

static QUIC_STATUS QUIC_API client_conn_cb(HQUIC, void *, QUIC_CONNECTION_EVENT *);
static QUIC_STATUS QUIC_API client_stream_cb(HQUIC, void *, QUIC_STREAM_EVENT *);

// ── nghttp3 flush writes ────────────────────────────────────────────────────

static void h3_client_flush(ClientConn *c) {
  for (;;) {
    nghttp3_vec vec[16];
    int         fin = 0;
    int64_t     sid = -1;
    nghttp3_ssize n = nghttp3_conn_writev_stream(c->h3conn, &sid, &fin, vec, 16);
    if (n < 0 || sid < 0) break;
    if (n == 0 && !fin) break;

    auto it = c->id_to_stream.find(sid);
    if (it == c->id_to_stream.end()) break;
    HQUIC stream = it->second;

    size_t total = 0;
    for (nghttp3_ssize i = 0; i < n; ++i) total += vec[i].len;

    if (total > 0 || fin) {
      auto *buf = new uint8_t[total];
      size_t off = 0;
      for (nghttp3_ssize i = 0; i < n; ++i) {
        std::memcpy(buf + off, vec[i].base, vec[i].len);
        off += vec[i].len;
      }
      QUIC_BUFFER qb;
      qb.Length = static_cast<uint32_t>(total);
      qb.Buffer = buf;
      QUIC_SEND_FLAGS flags = fin ? QUIC_SEND_FLAG_FIN : QUIC_SEND_FLAG_NONE;
      c->q->StreamSend(stream, &qb, 1, flags, buf);
      nghttp3_conn_add_write_offset(c->h3conn, sid, static_cast<int64_t>(total));
    }
    if (n == 0) break;
  }
}

// ── Open client unidirectional stream ───────────────────────────────────────

static int64_t client_open_uni(ClientConn *c) {
  HQUIC stream = nullptr;
  QUIC_STATUS s = c->q->StreamOpen(c->conn, QUIC_STREAM_OPEN_FLAG_UNIDIRECTIONAL,
                                   client_stream_cb, c, &stream);
  if (QUIC_FAILED(s)) return -1;
  s = c->q->StreamStart(stream, QUIC_STREAM_START_FLAG_NONE);
  if (QUIC_FAILED(s)) { c->q->StreamClose(stream); return -1; }

  // Client-initiated unidirectional: 0x02, 0x06, 0x0A, ...
  int64_t id = 0x02 + 4 * c->next_uni_id++;
  c->stream_ids[stream] = id;
  c->id_to_stream[id]   = stream;
  return id;
}

// ── nghttp3 client callbacks ────────────────────────────────────────────────

static int h3c_recv_data(nghttp3_conn *, int64_t stream_id, const uint8_t *data,
                         size_t datalen, void *conn_data, void *) {
  auto *c = static_cast<ClientConn *>(conn_data);
  std::lock_guard<std::mutex> lk(c->mu);
  auto &resp = c->responses[stream_id];
  resp.body.insert(resp.body.end(), data, data + datalen);
  return 0;
}

static int h3c_end_stream(nghttp3_conn *, int64_t stream_id, void *conn_data, void *) {
  auto *c = static_cast<ClientConn *>(conn_data);
  {
    std::lock_guard<std::mutex> lk(c->mu);
    c->responses[stream_id].done = true;
  }
  c->cv.notify_all();
  return 0;
}

static int h3c_deferred_consume(nghttp3_conn *, int64_t stream_id, size_t consumed,
                                void *conn_data, void *) {
  auto *c = static_cast<ClientConn *>(conn_data);
  auto it = c->id_to_stream.find(stream_id);
  if (it != c->id_to_stream.end()) {
    c->q->StreamReceiveComplete(it->second, consumed);
  }
  return 0;
}

// ── Setup nghttp3 client connection ─────────────────────────────────────────

static bool setup_h3_client(ClientConn *c) {
  nghttp3_callbacks cb = {};
  cb.recv_data        = h3c_recv_data;
  cb.end_stream       = h3c_end_stream;
  cb.deferred_consume = h3c_deferred_consume;

  nghttp3_settings settings;
  nghttp3_settings_default(&settings);

  int rv = nghttp3_conn_client_new(&c->h3conn, &cb, &settings, nghttp3_mem_default(), c);
  if (rv != 0) return false;

  int64_t ctrl = client_open_uni(c);
  int64_t qenc = client_open_uni(c);
  int64_t qdec = client_open_uni(c);
  if (ctrl < 0 || qenc < 0 || qdec < 0) return false;

  nghttp3_conn_bind_control_stream(c->h3conn, ctrl);
  nghttp3_conn_bind_qpack_streams(c->h3conn, qenc, qdec);
  h3_client_flush(c);
  return true;
}

// ── MsQuic client callbacks ─────────────────────────────────────────────────

static QUIC_STATUS QUIC_API client_stream_cb(HQUIC stream, void *context,
                                             QUIC_STREAM_EVENT *event) {
  auto *c = static_cast<ClientConn *>(context);
  switch (event->Type) {
    case QUIC_STREAM_EVENT_RECEIVE: {
      auto it = c->stream_ids.find(stream);
      if (it == c->stream_ids.end()) break;
      int64_t sid = it->second;
      for (uint32_t i = 0; i < event->RECEIVE.BufferCount; ++i) {
        nghttp3_conn_read_stream(c->h3conn, sid,
                                 event->RECEIVE.Buffers[i].Buffer,
                                 event->RECEIVE.Buffers[i].Length, 0);
      }
      if (event->RECEIVE.Flags & QUIC_RECEIVE_FLAG_FIN) {
        nghttp3_conn_read_stream(c->h3conn, sid, nullptr, 0, 1);
      }
      h3_client_flush(c);
      break;
    }
    case QUIC_STREAM_EVENT_SEND_COMPLETE: {
      delete[] static_cast<uint8_t *>(event->SEND_COMPLETE.ClientContext);
      break;
    }
    case QUIC_STREAM_EVENT_SHUTDOWN_COMPLETE: {
      auto it = c->stream_ids.find(stream);
      if (it != c->stream_ids.end()) {
        c->id_to_stream.erase(it->second);
        c->stream_ids.erase(it);
      }
      c->q->StreamClose(stream);
      break;
    }
    default:
      break;
  }
  return QUIC_STATUS_SUCCESS;
}

static QUIC_STATUS QUIC_API client_conn_cb(HQUIC conn, void *context,
                                           QUIC_CONNECTION_EVENT *event) {
  auto *c = static_cast<ClientConn *>(context);
  switch (event->Type) {
    case QUIC_CONNECTION_EVENT_CONNECTED: {
      std::lock_guard<std::mutex> lk(c->mu);
      c->connected = true;
      c->cv.notify_all();
      break;
    }
    case QUIC_CONNECTION_EVENT_SHUTDOWN_INITIATED_BY_TRANSPORT:
    case QUIC_CONNECTION_EVENT_SHUTDOWN_INITIATED_BY_PEER: {
      std::lock_guard<std::mutex> lk(c->mu);
      c->conn_failed = true;
      c->cv.notify_all();
      break;
    }
    case QUIC_CONNECTION_EVENT_PEER_STREAM_STARTED: {
      HQUIC stream = event->PEER_STREAM_STARTED.Stream;
      c->q->SetCallbackHandler(stream, (void *)client_stream_cb, c);
      QUIC_UINT62 sid = 0;
      uint32_t sz = sizeof(sid);
      c->q->GetParam(stream, QUIC_PARAM_STREAM_ID, &sz, &sid);
      c->stream_ids[stream]                        = static_cast<int64_t>(sid);
      c->id_to_stream[static_cast<int64_t>(sid)]   = stream;
      break;
    }
    case QUIC_CONNECTION_EVENT_SHUTDOWN_COMPLETE:
      break;
    default:
      break;
  }
  return QUIC_STATUS_SUCCESS;
}

// ── H3Client::Impl ──────────────────────────────────────────────────────────

struct H3Client::Impl {
  const QUIC_API_TABLE *q    = nullptr;
  HQUIC                 reg  = nullptr;
  HQUIC                 cfg  = nullptr;
  ClientConn           *conn = nullptr;
  std::string           host;
  std::string           error;
};

H3Client::H3Client() : impl_(new Impl()) {}

H3Client::~H3Client() {
  disconnect();
  delete impl_;
}

bool H3Client::connect(const std::string &host, uint16_t port, bool validate_cert) {
  impl_->host = host;

  if (QUIC_FAILED(MsQuicOpen2(&impl_->q))) {
    impl_->error = "MsQuicOpen2 failed";
    return false;
  }

  QUIC_REGISTRATION_CONFIG rc = {"jpip-h3-client", QUIC_EXECUTION_PROFILE_LOW_LATENCY};
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
                                               sizeof(settings), nullptr, &impl_->cfg))) {
    impl_->error = "ConfigurationOpen failed";
    return false;
  }

  QUIC_CREDENTIAL_CONFIG cred = {};
  cred.Type  = QUIC_CREDENTIAL_TYPE_NONE;
  cred.Flags = QUIC_CREDENTIAL_FLAG_CLIENT;
  if (!validate_cert) {
    cred.Flags = (QUIC_CREDENTIAL_FLAGS)(cred.Flags | QUIC_CREDENTIAL_FLAG_NO_CERTIFICATE_VALIDATION);
  }
  if (QUIC_FAILED(impl_->q->ConfigurationLoadCredential(impl_->cfg, &cred))) {
    impl_->error = "ConfigurationLoadCredential failed";
    return false;
  }

  auto *c = new ClientConn();
  c->q = impl_->q;
  impl_->conn = c;

  if (QUIC_FAILED(impl_->q->ConnectionOpen(impl_->reg, client_conn_cb, c, &c->conn))) {
    impl_->error = "ConnectionOpen failed";
    return false;
  }
  if (QUIC_FAILED(impl_->q->ConnectionStart(c->conn, impl_->cfg,
                                             QUIC_ADDRESS_FAMILY_UNSPEC,
                                             host.c_str(), port))) {
    impl_->error = "ConnectionStart failed";
    return false;
  }

  // Wait for connection to complete
  {
    std::unique_lock<std::mutex> lk(c->mu);
    c->cv.wait(lk, [c] { return c->connected || c->conn_failed; });
    if (c->conn_failed) { impl_->error = "QUIC handshake failed"; return false; }
  }

  if (!setup_h3_client(c)) {
    impl_->error = "HTTP/3 setup failed";
    return false;
  }

  return true;
}

void H3Client::disconnect() {
  if (impl_->conn) {
    if (impl_->conn->h3conn) {
      nghttp3_conn_del(impl_->conn->h3conn);
      impl_->conn->h3conn = nullptr;
    }
    if (impl_->conn->conn) {
      impl_->q->ConnectionShutdown(impl_->conn->conn, QUIC_CONNECTION_SHUTDOWN_FLAG_NONE, 0);
      impl_->q->ConnectionClose(impl_->conn->conn);
    }
    delete impl_->conn;
    impl_->conn = nullptr;
  }
  if (impl_->cfg) { impl_->q->ConfigurationClose(impl_->cfg); impl_->cfg = nullptr; }
  if (impl_->reg) { impl_->q->RegistrationClose(impl_->reg); impl_->reg = nullptr; }
  if (impl_->q)   { MsQuicClose(impl_->q); impl_->q = nullptr; }
}

std::vector<uint8_t> H3Client::fetch(const std::string &path_and_query) {
  auto *c = impl_->conn;
  if (!c || !c->h3conn) { impl_->error = "not connected"; return {}; }

  // Open a new bidirectional stream
  HQUIC stream = nullptr;
  if (QUIC_FAILED(impl_->q->StreamOpen(c->conn, QUIC_STREAM_OPEN_FLAG_NONE,
                                       client_stream_cb, c, &stream))) {
    impl_->error = "StreamOpen failed";
    return {};
  }
  if (QUIC_FAILED(impl_->q->StreamStart(stream, QUIC_STREAM_START_FLAG_NONE))) {
    impl_->error = "StreamStart failed";
    impl_->q->StreamClose(stream);
    return {};
  }

  // Get the QUIC-assigned stream ID
  QUIC_UINT62 qid = 0;
  uint32_t sz = sizeof(qid);
  impl_->q->GetParam(stream, QUIC_PARAM_STREAM_ID, &sz, &qid);
  int64_t sid = static_cast<int64_t>(qid);
  c->stream_ids[stream] = sid;
  c->id_to_stream[sid]  = stream;

  // Submit HTTP/3 request headers
  nghttp3_nv nva[] = {
    {(uint8_t *)":method", (uint8_t *)"GET", 7, 3, NGHTTP3_NV_FLAG_NONE},
    {(uint8_t *)":scheme", (uint8_t *)"https", 7, 5, NGHTTP3_NV_FLAG_NONE},
    {(uint8_t *)":authority", (uint8_t *)impl_->host.c_str(), 10,
     impl_->host.size(), NGHTTP3_NV_FLAG_NO_COPY_VALUE},
    {(uint8_t *)":path", (uint8_t *)path_and_query.c_str(), 5,
     path_and_query.size(), NGHTTP3_NV_FLAG_NO_COPY_VALUE},
  };

  {
    std::lock_guard<std::mutex> lk(c->mu);
    c->responses[sid] = {};
  }

  nghttp3_conn_submit_request(c->h3conn, sid, nva, 4, nullptr, nullptr);
  h3_client_flush(c);

  // Wait for response to complete
  {
    std::unique_lock<std::mutex> lk(c->mu);
    c->cv.wait(lk, [c, sid] {
      auto it = c->responses.find(sid);
      return it != c->responses.end() && it->second.done;
    });
  }

  std::lock_guard<std::mutex> lk(c->mu);
  auto it = c->responses.find(sid);
  std::vector<uint8_t> body = std::move(it->second.body);
  c->responses.erase(it);
  return body;
}

std::string H3Client::last_error() const { return impl_->error; }

}  // namespace jpip
}  // namespace open_htj2k

#endif  // OPENHTJ2K_ENABLE_QUIC
