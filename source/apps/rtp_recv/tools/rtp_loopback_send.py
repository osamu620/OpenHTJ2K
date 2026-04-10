#!/usr/bin/env python3
# Copyright (c) 2026, Osamu Watanabe
# All rights reserved.
#
# Licensed under the same BSD 3-Clause terms as the rest of OpenHTJ2K.
#
# Loopback test helper for open_htj2k_rtp_recv.
#
# Wraps a single JPEG 2000 codestream (.j2k) in one RFC 9828 Main Packet and
# sends it as a single UDP datagram to the receiver.  Used to regression-test
# the depacketizer + decoder wiring without needing an external sender like
# kdu_stream_send.
#
# Usage:
#   python3 rtp_loopback_send.py [CODESTREAM] [DST_HOST] [DST_PORT]
#
# Defaults: the 8 KB HT conformance bitstream in conformance_data/, 127.0.0.1, 6000.
#
# The generated Main Packet has:
#   MH=3 (single Main), TP=0 (progressive), ORDH=4 (PCRL+resync),
#   P=0 (PTSTAMP invalid), XTRAC=0,
#   R=0 (no main-header reuse), S=0 (colorspace unspecified — receiver must
#   use its --colorspace CLI fallback), C=0 (no caching), RANGE=0.
#
# The RTP marker bit is set so the receiver emits the frame immediately.

import os
import socket
import struct
import sys

DEFAULT_CS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "..", "conformance_data", "ds0_ht_01_b11.j2k",
)


def main() -> int:
    cs_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CS
    host    = sys.argv[2] if len(sys.argv) > 2 else "127.0.0.1"
    port    = int(sys.argv[3]) if len(sys.argv) > 3 else 6000

    with open(cs_path, "rb") as f:
        cs = f.read()

    # 12-byte RTP fixed header.
    # Byte 0: V(2)=2 P(1)=0 X(1)=0 CC(4)=0            = 0x80
    # Byte 1: M(1)=1 PT(7)=96                           = 0xE0
    # Bytes 2-3: sequence = 1
    # Bytes 4-7: timestamp = 0
    # Bytes 8-11: SSRC = 0xCAFEBABE
    rtp_hdr = struct.pack(
        "!BBHII",
        0x80,
        0x80 | 96,
        1,
        0,
        0xCAFEBABE,
    )

    # 8-byte RFC 9828 Main Packet payload header.
    # Byte 0: MH(2)=3 TP(3)=0 ORDH(3)=4  -> (3<<6) | (0<<3) | 4 = 0xC4
    # Byte 1: P(1)=0 XTRAC(3)=0 PTSTAMP[11:8]=0        = 0x00
    # Byte 2: PTSTAMP[7:0]                              = 0x00
    # Byte 3: ESEQ(8)                                     = 0x00
    # Byte 4: R(1)=0 S(1)=0 C(1)=0 RSVD(4)=0 RANGE(1)=0 = 0x00
    # Byte 5: PRIMS = 0
    # Byte 6: TRANS = 0
    # Byte 7: MAT   = 0
    main_hdr = bytes([0xC4, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])

    datagram = rtp_hdr + main_hdr + cs
    print(f"sending {len(datagram)} bytes ({len(cs)} codestream) to {host}:{port}")

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.sendto(datagram, (host, port))
    finally:
        s.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
