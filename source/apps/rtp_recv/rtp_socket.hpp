// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.
//
// Licensed under the same BSD 3-Clause terms as the rest of OpenHTJ2K.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

// Platform socket abstraction: SOCKET on Windows, int on POSIX.
#ifdef _WIN32
  #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
  #endif
  #include <winsock2.h>
  #include <ws2tcpip.h>
#endif

namespace open_htj2k::rtp_recv {

#ifdef _WIN32
using socket_t = SOCKET;
constexpr socket_t kInvalidSocket = INVALID_SOCKET;
#else
using socket_t = int;
constexpr socket_t kInvalidSocket = -1;
#endif

// Minimal cross-platform UDP receive socket.  Binds to a host:port, reads
// datagrams into a caller-supplied buffer, returns the number of bytes
// received or one of the negative sentinels below.  No exceptions; error
// messages are captured via last_error() so the CLI layer decides how to
// report them.
//
// Lifetime: one instance owns one socket handle.  Move-only, close on
// destruction.
//
// On Windows, the caller must call wsa_init() once before creating any
// UdpSocket and wsa_cleanup() before exit.  On POSIX these are no-ops.
class UdpSocket {
 public:
  static constexpr int kAgain = -1;  // no datagram available (non-blocking)
  static constexpr int kError = -2;  // hard error, see last_error()

  // Winsock2 startup/shutdown (no-op on POSIX).
  static bool wsa_init();
  static void wsa_cleanup();

  UdpSocket() = default;
  ~UdpSocket();

  UdpSocket(const UdpSocket&)            = delete;
  UdpSocket& operator=(const UdpSocket&) = delete;
  UdpSocket(UdpSocket&&) noexcept;
  UdpSocket& operator=(UdpSocket&&) noexcept;

  // Bind to `host`:`port`.  host="" or "0.0.0.0" binds to the wildcard.
  // Returns true on success; on failure last_error() holds the reason.
  bool bind(const std::string& host, uint16_t port);

  // Put the socket in non-blocking mode (recv() returns kAgain when empty).
  // Must be called after bind().
  bool set_nonblocking();

  // Receive a single datagram into buf[0..buf_size).  Returns the datagram
  // length on success, kAgain if non-blocking and no data, or kError on a
  // fatal socket error.  Datagrams larger than buf_size are truncated and
  // the excess bytes are discarded (MSG_TRUNC is not used).
  //
  // Safe to call in a tight receive loop; no allocation.
  ptrdiff_t recv(void* buf, size_t buf_size);

  // Set SO_RCVBUF hint (best-effort).  Useful for 4K streams at high bitrate.
  // On Linux the kernel doubles the value and clamps to net.core.rmem_max;
  // on Windows the default limit is typically generous (8 MB+).
  // Callers should check last_granted_recv_buf() for the actual value.
  bool set_recv_buffer_size(int bytes);

  // Bytes the kernel actually granted on the most recent set_recv_buffer_size()
  // call.  Includes the kernel's internal x2 doubling.  Returns 0 if no
  // request has been made yet.
  int last_granted_recv_buf() const;

  // Block up to `timeout_ms` milliseconds waiting for the socket to become
  // readable.  Returns 1 on readable, 0 on timeout, -1 on error (last_error()
  // holds the reason).  Lets the main loop sleep briefly between socket polls
  // without pegging a CPU core, while still letting GLFW events fire every
  // few milliseconds.
  int wait_readable(int timeout_ms);

  bool is_open() const { return fd_ != kInvalidSocket; }
  const std::string& last_error() const { return last_error_; }

  void close();

 private:
  socket_t    fd_                     = kInvalidSocket;
  int         last_granted_recv_buf_  = 0;
  std::string last_error_;
};

}  // namespace open_htj2k::rtp_recv
