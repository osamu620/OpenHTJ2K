// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.
//
// Licensed under the same BSD 3-Clause terms as the rest of OpenHTJ2K.

#include "rtp_socket.hpp"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <utility>

namespace open_htj2k::rtp_recv {

namespace {
std::string errno_message(const char* what) {
  std::string s(what);
  s += ": ";
  s += std::strerror(errno);
  return s;
}
}  // namespace

UdpSocket::~UdpSocket() { close(); }

UdpSocket::UdpSocket(UdpSocket&& other) noexcept
    : fd_(other.fd_), last_error_(std::move(other.last_error_)) {
  other.fd_ = -1;
}

UdpSocket& UdpSocket::operator=(UdpSocket&& other) noexcept {
  if (this != &other) {
    close();
    fd_         = other.fd_;
    last_error_ = std::move(other.last_error_);
    other.fd_   = -1;
  }
  return *this;
}

void UdpSocket::close() {
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
}

bool UdpSocket::bind(const std::string& host, uint16_t port) {
  close();

  fd_ = ::socket(AF_INET, SOCK_DGRAM, 0);
  if (fd_ < 0) {
    last_error_ = errno_message("socket()");
    return false;
  }

  // SO_REUSEADDR so restarts don't hit EADDRINUSE during a cycling kdu_stream_send.
  int yes = 1;
  if (::setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes)) < 0) {
    last_error_ = errno_message("setsockopt(SO_REUSEADDR)");
    close();
    return false;
  }

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port   = htons(port);
  if (host.empty() || host == "0.0.0.0") {
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
  } else {
    if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
      last_error_ = "inet_pton(" + host + "): not a valid IPv4 address";
      close();
      return false;
    }
  }

  if (::bind(fd_, reinterpret_cast<const sockaddr*>(&addr), sizeof(addr)) < 0) {
    last_error_ = errno_message("bind()");
    close();
    return false;
  }

  return true;
}

bool UdpSocket::set_nonblocking() {
  if (fd_ < 0) {
    last_error_ = "set_nonblocking(): socket not open";
    return false;
  }
  int flags = ::fcntl(fd_, F_GETFL, 0);
  if (flags < 0) {
    last_error_ = errno_message("fcntl(F_GETFL)");
    return false;
  }
  if (::fcntl(fd_, F_SETFL, flags | O_NONBLOCK) < 0) {
    last_error_ = errno_message("fcntl(F_SETFL O_NONBLOCK)");
    return false;
  }
  return true;
}

bool UdpSocket::set_recv_buffer_size(int bytes) {
  if (fd_ < 0) {
    last_error_ = "set_recv_buffer_size(): socket not open";
    return false;
  }
  if (::setsockopt(fd_, SOL_SOCKET, SO_RCVBUF, &bytes, sizeof(bytes)) < 0) {
    // Best-effort; some kernels silently clamp without failing.
    last_error_ = errno_message("setsockopt(SO_RCVBUF)");
    return false;
  }
  // Read back the actual value the kernel applied (it doubles the request
  // internally and clamps to net.core.rmem_max, often silently).  Bytes
  // outside the granted window will be dropped from the kernel buffer
  // when the application falls behind, so the caller must know.
  int       got     = 0;
  socklen_t got_len = sizeof(got);
  if (::getsockopt(fd_, SOL_SOCKET, SO_RCVBUF, &got, &got_len) == 0) {
    last_granted_recv_buf_ = got;
  }
  return true;
}

int UdpSocket::last_granted_recv_buf() const { return last_granted_recv_buf_; }

int UdpSocket::wait_readable(int timeout_ms) {
  if (fd_ < 0) {
    last_error_ = "wait_readable(): socket not open";
    return -1;
  }
  pollfd pfd{};
  pfd.fd     = fd_;
  pfd.events = POLLIN;
  int rc     = ::poll(&pfd, 1, timeout_ms);
  if (rc < 0) {
    if (errno == EINTR) return 0;  // signal, treat as timeout
    last_error_ = errno_message("poll()");
    return -1;
  }
  return rc > 0 ? 1 : 0;
}

ptrdiff_t UdpSocket::recv(void* buf, size_t buf_size) {
  if (fd_ < 0) {
    last_error_ = "recv(): socket not open";
    return kError;
  }
  ssize_t n = ::recvfrom(fd_, buf, buf_size, 0, nullptr, nullptr);
  if (n < 0) {
    if (errno == EAGAIN || errno == EWOULDBLOCK) return kAgain;
    if (errno == EINTR) return kAgain;  // signal, treat as "try again"
    last_error_ = errno_message("recvfrom()");
    return kError;
  }
  return static_cast<ptrdiff_t>(n);
}

}  // namespace open_htj2k::rtp_recv
