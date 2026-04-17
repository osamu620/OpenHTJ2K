// jpip_tcp_check: ctest harness for the TCP socket wrapper.
//
// Spawns a listener on a localhost port, connects a client, sends a
// payload, receives it on the server side, and echoes it back.
// Verifies byte-identical round-trip.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

#include "tcp_socket.hpp"

using open_htj2k::jpip::TcpListener;
using open_htj2k::jpip::TcpStream;
using open_htj2k::jpip::tcp_wsa_cleanup;
using open_htj2k::jpip::tcp_wsa_init;

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

}  // namespace

int main() {
  tcp_wsa_init();

  constexpr uint16_t kPort = 0;  // 0 = let OS pick an ephemeral port

  // Use a fixed port for simplicity (ephemeral port selection would
  // require getsockname).  Pick something unlikely to collide.
  constexpr uint16_t kTestPort = 19283;

  // ── Server thread: accept one connection, echo the payload back ────
  std::vector<uint8_t> server_got;
  bool server_ok = false;
  std::thread server_thread([&]() {
    TcpListener listener;
    if (!listener.bind(kTestPort, "127.0.0.1")) {
      std::fprintf(stderr, "server bind: %s\n", listener.last_error().c_str());
      return;
    }
    if (!listener.listen()) {
      std::fprintf(stderr, "server listen: %s\n", listener.last_error().c_str());
      return;
    }
    TcpStream conn = listener.accept();
    if (!conn.is_open()) {
      std::fprintf(stderr, "server accept: %s\n", listener.last_error().c_str());
      return;
    }
    // Read a 4-byte length prefix, then that many bytes.
    uint8_t len_buf[4];
    if (!conn.recv_all(len_buf, 4)) return;
    const uint32_t payload_len =
        (static_cast<uint32_t>(len_buf[0]) << 24) | (static_cast<uint32_t>(len_buf[1]) << 16) |
        (static_cast<uint32_t>(len_buf[2]) << 8)  |  static_cast<uint32_t>(len_buf[3]);
    server_got.resize(payload_len);
    if (!conn.recv_all(server_got.data(), payload_len)) return;
    // Echo back.
    if (!conn.send_all(server_got.data(), payload_len)) return;
    server_ok = true;
  });

  // Give the server thread a moment to bind+listen.
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  // ── Client: connect, send, receive echo ────────────────────────────
  {
    TcpStream client;
    CHECK(client.connect("127.0.0.1", kTestPort), "client connect: %s",
          client.last_error().c_str());
    if (failures) { server_thread.join(); tcp_wsa_cleanup(); return 1; }

    const std::vector<uint8_t> payload = {0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE};
    const uint32_t plen = static_cast<uint32_t>(payload.size());
    const uint8_t len_buf[4] = {
        static_cast<uint8_t>((plen >> 24) & 0xFF),
        static_cast<uint8_t>((plen >> 16) & 0xFF),
        static_cast<uint8_t>((plen >> 8) & 0xFF),
        static_cast<uint8_t>(plen & 0xFF),
    };
    CHECK(client.send_all(len_buf, 4), "send length");
    CHECK(client.send_all(payload.data(), payload.size()), "send payload");

    std::vector<uint8_t> echo(payload.size());
    CHECK(client.recv_all(echo.data(), echo.size()), "recv echo");
    CHECK(echo == payload, "echo mismatch");
  }

  server_thread.join();
  CHECK(server_ok, "server thread reported failure");
  CHECK(server_got.size() == 6, "server got %zu bytes", server_got.size());

  tcp_wsa_cleanup();

  if (failures == 0) {
    std::printf("OK tcp_check: loopback send/recv round-trip\n");
    return 0;
  }
  std::fprintf(stderr, "tcp_check: %d failures\n", failures);
  return 1;
}
