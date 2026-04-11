#!/usr/bin/env python3
# Minimal .rtp file replayer for the Spark fixture format:
#   [0xFFFF marker (2B)] [length BE (2B)] [RTP packet of that length] ...
# Sends each packet as UDP to (host, port) as fast as we can, with an
# optional per-packet sleep to approximate wire rate.

import argparse
import socket
import struct
import sys
import time


def iter_packets(path):
    with open(path, "rb") as f:
        data = f.read()
    i = 0
    n = len(data)
    while i + 4 <= n:
        marker, length = struct.unpack_from(">HH", data, i)
        if marker != 0xFFFF:
            raise RuntimeError(f"bad marker 0x{marker:04x} at offset {i}")
        i += 4
        if i + length > n:
            raise RuntimeError(f"truncated packet at {i}: need {length}, have {n - i}")
        yield data[i : i + length]
        i += length


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rtp_file")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=6000)
    ap.add_argument("--max-packets", type=int, default=0, help="0 = all")
    ap.add_argument("--rate-bps", type=float, default=0.0,
                    help="pace to this bits-per-second (payload), 0 = max rate")
    ap.add_argument("--start-delay", type=float, default=0.3,
                    help="seconds to wait before sending first packet")
    args = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8 * 1024 * 1024)
    except OSError:
        pass
    dst = (args.host, args.port)

    time.sleep(args.start_delay)

    sent = 0
    bytes_out = 0
    t0 = time.perf_counter()
    interval = 0.0
    for pkt in iter_packets(args.rtp_file):
        sock.sendto(pkt, dst)
        sent += 1
        bytes_out += len(pkt)
        if args.max_packets and sent >= args.max_packets:
            break
        if args.rate_bps > 0:
            # naive wire pacing via expected time vs. elapsed
            expected = bytes_out * 8 / args.rate_bps
            elapsed = time.perf_counter() - t0
            drift = expected - elapsed
            if drift > 0.0005:
                time.sleep(drift)

    dt = time.perf_counter() - t0
    mbps = (bytes_out * 8 / 1e6) / max(dt, 1e-9)
    print(f"sent {sent} packets, {bytes_out / (1024*1024):.1f} MiB "
          f"in {dt:.2f}s ({mbps:.1f} Mbps payload)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
