// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.
//
// Cross-platform TCP socket wrapper for JPIP HTTP/1.1 transport.
// Modelled after source/apps/rtp_recv/rtp_socket.hpp but for TCP
// (stream-oriented, connection-based) instead of UDP (datagram).
//
// Provides two classes:
//
//   TcpListener  — server side: bind(), listen(), accept() → TcpStream.
//   TcpStream    — bidirectional byte stream: connect(), send_all(),
//                  recv_all(), recv_until().
//
// On Windows the caller must call tcp_wsa_init() once before creating
// any socket and tcp_wsa_cleanup() before exit (mirroring UdpSocket's
// pattern).  On POSIX these are no-ops.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#ifdef _WIN32
  #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
  #endif
  #include <winsock2.h>
  #include <ws2tcpip.h>
  using tcp_socket_t = SOCKET;
  constexpr tcp_socket_t kTcpInvalidSocket = INVALID_SOCKET;
#else
  using tcp_socket_t = int;
  constexpr tcp_socket_t kTcpInvalidSocket = -1;
#endif

#if defined(_MSC_VER) && !defined(OHTJ2K_STATIC)
  #define OPENHTJ2K_JPIP_EXPORT __declspec(dllexport)
#else
  #define OPENHTJ2K_JPIP_EXPORT
#endif

namespace open_htj2k {
namespace jpip {

OPENHTJ2K_JPIP_EXPORT bool tcp_wsa_init();
OPENHTJ2K_JPIP_EXPORT void tcp_wsa_cleanup();

class TcpStream;

class OPENHTJ2K_JPIP_EXPORT TcpListener {
 public:
  TcpListener() = default;
  ~TcpListener();
  TcpListener(const TcpListener &)            = delete;
  TcpListener &operator=(const TcpListener &) = delete;

  bool bind(uint16_t port, const std::string &host = "");
  bool listen(int backlog = 8);
  // Block until a client connects.  Returns a connected TcpStream.
  // On error the returned stream is not open (check is_open()).
  TcpStream accept();
  void close();
  bool is_open() const { return fd_ != kTcpInvalidSocket; }
  const std::string &last_error() const { return err_; }

 private:
  tcp_socket_t fd_ = kTcpInvalidSocket;
  std::string  err_;
};

class OPENHTJ2K_JPIP_EXPORT TcpStream {
 public:
  TcpStream() = default;
  explicit TcpStream(tcp_socket_t fd) : fd_(fd) {}
  ~TcpStream();
  TcpStream(const TcpStream &)            = delete;
  TcpStream &operator=(const TcpStream &) = delete;
  TcpStream(TcpStream &&o) noexcept : fd_(o.fd_), err_(std::move(o.err_)) { o.fd_ = kTcpInvalidSocket; }
  TcpStream &operator=(TcpStream &&o) noexcept;

  bool connect(const std::string &host, uint16_t port);
  // Send exactly `len` bytes.  Returns true on success.
  bool send_all(const uint8_t *buf, std::size_t len);
  bool send_all(const std::vector<uint8_t> &v) { return send_all(v.data(), v.size()); }
  // Receive exactly `len` bytes.  Returns true on success.
  bool recv_all(uint8_t *buf, std::size_t len);
  // Read until the buffer contains the 4-byte pattern "\r\n\r\n" (HTTP
  // header terminator) or `max_bytes` have been read.  Returns the total
  // bytes accumulated in `buf`.  Returns 0 on EOF / error.
  std::size_t recv_until_header_end(std::vector<uint8_t> &buf, std::size_t max_bytes = 65536);
  void close();
  bool is_open() const { return fd_ != kTcpInvalidSocket; }
  const std::string &last_error() const { return err_; }

 private:
  tcp_socket_t fd_ = kTcpInvalidSocket;
  std::string  err_;
};

}  // namespace jpip
}  // namespace open_htj2k
