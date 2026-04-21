// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.

#include "tcp_socket.hpp"

#include <cerrno>
#include <cstdio>
#include <cstring>

#ifndef _WIN32
  #include <arpa/inet.h>
  #include <netdb.h>
  #include <netinet/in.h>
  #include <sys/socket.h>
  #include <unistd.h>
#endif

namespace open_htj2k {
namespace jpip {

namespace {

#ifdef _WIN32
std::string sock_error_str() {
  int e = WSAGetLastError();
  char buf[256] = {};
  FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM, nullptr, static_cast<DWORD>(e), 0,
                 buf, sizeof(buf), nullptr);
  return std::string(buf);
}
inline void close_socket(tcp_socket_t fd) { closesocket(fd); }
#else
std::string sock_error_str() { return std::strerror(errno); }
inline void close_socket(tcp_socket_t fd) { ::close(fd); }
#endif

}  // namespace

bool tcp_wsa_init() {
#ifdef _WIN32
  WSADATA wd;
  return WSAStartup(MAKEWORD(2, 2), &wd) == 0;
#else
  return true;
#endif
}

void tcp_wsa_cleanup() {
#ifdef _WIN32
  WSACleanup();
#endif
}

// ── TcpListener ──────────────────────────────────────────────────────────

TcpListener::~TcpListener() { close(); }

bool TcpListener::bind(uint16_t port, const std::string &host) {
  close();
  fd_ = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  if (fd_ == kTcpInvalidSocket) { err_ = sock_error_str(); return false; }
  int opt = 1;
  setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR,
             reinterpret_cast<const char *>(&opt), sizeof(opt));
  struct sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port   = htons(port);
  if (host.empty() || host == "0.0.0.0") {
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
  } else {
    inet_pton(AF_INET, host.c_str(), &addr.sin_addr);
  }
  if (::bind(fd_, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr)) != 0) {
    err_ = sock_error_str();
    close();
    return false;
  }
  return true;
}

bool TcpListener::listen(int backlog) {
  if (fd_ == kTcpInvalidSocket) { err_ = "not bound"; return false; }
  if (::listen(fd_, backlog) != 0) { err_ = sock_error_str(); return false; }
  return true;
}

TcpStream TcpListener::accept() {
  if (fd_ == kTcpInvalidSocket) { err_ = "not bound"; return {}; }
  struct sockaddr_in client_addr{};
  socklen_t client_len = sizeof(client_addr);
  tcp_socket_t cfd = ::accept(fd_, reinterpret_cast<struct sockaddr *>(&client_addr),
                              &client_len);
  if (cfd == kTcpInvalidSocket) { err_ = sock_error_str(); return {}; }
  return TcpStream(cfd);
}

void TcpListener::close() {
  if (fd_ != kTcpInvalidSocket) { close_socket(fd_); fd_ = kTcpInvalidSocket; }
}

// ── TcpStream ────────────────────────────────────────────────────────────

TcpStream::~TcpStream() { close(); }

TcpStream &TcpStream::operator=(TcpStream &&o) noexcept {
  if (this != &o) {
    close();
    fd_  = o.fd_;
    err_ = std::move(o.err_);
    o.fd_ = kTcpInvalidSocket;
  }
  return *this;
}

bool TcpStream::connect(const std::string &host, uint16_t port) {
  close();
  struct addrinfo hints{}, *res = nullptr;
  hints.ai_family   = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  char port_str[8];
  std::snprintf(port_str, sizeof(port_str), "%u", port);
  if (getaddrinfo(host.c_str(), port_str, &hints, &res) != 0 || !res) {
    err_ = "getaddrinfo failed: " + sock_error_str();
    return false;
  }
  fd_ = ::socket(res->ai_family, res->ai_socktype, res->ai_protocol);
  if (fd_ == kTcpInvalidSocket) { err_ = sock_error_str(); freeaddrinfo(res); return false; }
  if (::connect(fd_, res->ai_addr, static_cast<socklen_t>(res->ai_addrlen)) != 0) {
    err_ = sock_error_str();
    close();
    freeaddrinfo(res);
    return false;
  }
  freeaddrinfo(res);
  return true;
}

bool TcpStream::send_all(const uint8_t *buf, std::size_t len) {
  std::size_t sent = 0;
  while (sent < len) {
    auto n = ::send(fd_, reinterpret_cast<const char *>(buf + sent),
                    static_cast<int>(len - sent), 0);
    if (n <= 0) { err_ = sock_error_str(); return false; }
    sent += static_cast<std::size_t>(n);
  }
  return true;
}

bool TcpStream::recv_all(uint8_t *buf, std::size_t len) {
  std::size_t got = 0;
  while (got < len) {
    auto n = ::recv(fd_, reinterpret_cast<char *>(buf + got),
                    static_cast<int>(len - got), 0);
    if (n <= 0) { err_ = (n == 0) ? "EOF" : sock_error_str(); return false; }
    got += static_cast<std::size_t>(n);
  }
  return true;
}

std::size_t TcpStream::recv_until_header_end(std::vector<uint8_t> &buf,
                                             std::size_t max_bytes) {
  buf.clear();
  uint8_t tmp[4096];
  while (buf.size() < max_bytes) {
    auto chunk = std::min<std::size_t>(sizeof(tmp), max_bytes - buf.size());
    auto n = ::recv(fd_, reinterpret_cast<char *>(tmp), static_cast<int>(chunk), 0);
    if (n <= 0) return buf.size();  // EOF or error
    buf.insert(buf.end(), tmp, tmp + n);
    // Check for "\r\n\r\n" in the last few bytes.
    if (buf.size() >= 4) {
      for (std::size_t i = (buf.size() > static_cast<std::size_t>(n) + 3)
                               ? (buf.size() - static_cast<std::size_t>(n) - 3)
                               : 0;
           i + 4 <= buf.size(); ++i) {
        if (buf[i] == '\r' && buf[i + 1] == '\n' && buf[i + 2] == '\r' && buf[i + 3] == '\n') {
          return buf.size();
        }
      }
    }
  }
  return buf.size();
}

std::size_t TcpStream::recv_some(uint8_t *buf, std::size_t len) {
  if (len == 0) return 0;
  auto n = ::recv(fd_, reinterpret_cast<char *>(buf), static_cast<int>(len), 0);
  if (n == 0) return 0;
  if (n < 0) { err_ = sock_error_str(); return SIZE_MAX; }
  return static_cast<std::size_t>(n);
}

std::size_t TcpStream::recv_to_eof(std::vector<uint8_t> &buf) {
  std::size_t total = 0;
  uint8_t tmp[16 * 1024];
  while (true) {
    std::size_t n = recv_some(tmp, sizeof(tmp));
    if (n == 0 || n == SIZE_MAX) return total;
    buf.insert(buf.end(), tmp, tmp + n);
    total += n;
  }
}

void TcpStream::close() {
  if (fd_ != kTcpInvalidSocket) { close_socket(fd_); fd_ = kTcpInvalidSocket; }
}

}  // namespace jpip
}  // namespace open_htj2k
