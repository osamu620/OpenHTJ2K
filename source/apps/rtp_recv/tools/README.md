# rtp_recv development tools

Developer utilities for the RFC 9828 RTP receiver. None of these are shipped
in a release build; they're here to drive regression tests and performance
measurements during rtp_recv work.

## `rtp_loopback_send.py`

Wraps a single `.j2k` / `.j2c` codestream in one RFC 9828 Main packet and
sends it as a single UDP datagram. Used for day-to-day smoke testing of the
receiver without needing a real RTP sender.

```
python3 rtp_loopback_send.py [CODESTREAM] [DST_HOST] [DST_PORT]
```

Defaults to the 8 KB HT conformance bitstream in `conformance_data/`.

## `rtp_file_replay.py`

Replays a prerecorded `.rtp` capture to the receiver at a chosen payload
rate. The file format is the simple custom framing:

```
[2-byte 0xFFFF marker][2-byte BE length][RTP packet of that length] ...
```

which is **not** libpcap and **not** rtpdump — it's specific to the test
fixtures used for this project.

```
python3 rtp_file_replay.py <rtp_file> \
    [--host 127.0.0.1] [--port 6000] \
    [--rate-bps 900000000] [--start-delay 1.0] [--max-packets N]
```

`--rate-bps 0` sends as fast as possible; at gigabit+ rates the kernel UDP
buffer overflows well before the decoder is exercised, so pick a rate about
1.5x wire rate (e.g. ~1 Gbps for the Spark 4K 4:2:2 1.7 bpp fixture) to
exercise the decode thread as the bottleneck without losing packets.

## `rtp_decode_profile.cpp` → `open_htj2k_rtp_decode_profile`

Offline profiler: feeds a directory of `.j2c` codestreams through the same
decoder path `rtp_recv::decode_thread_main` uses (one long-lived decoder,
per-frame `init()` + `parse()` + `invoke_line_based_stream()` with a planar
int32→u8 shift callback). Reports per-stage timing.

Built alongside `open_htj2k_rtp_recv` when `-DOPENHTJ2K_RTP=ON`. Binary
lands at `build/bin/open_htj2k_rtp_decode_profile`.

```
open_htj2k_rtp_decode_profile <codestream_dir> [max_frames=200] [loops=3] [threads=2]
```

## End-to-end reproduction (the flow used for v4 baseline)

1. **Dump codestreams from a `.rtp` fixture.** Start the receiver in
   capture-only mode (no decode, no render) and fire the replayer in
   parallel:

   ```bash
   mkdir -p /tmp/spark_cs
   ./build/bin/open_htj2k_rtp_recv --port 6000 --bind 127.0.0.1 \
       --no-render --no-decode --frames 250 \
       --dump-codestream /tmp/spark_cs/f_%05d.j2c &
   python3 source/apps/rtp_recv/tools/rtp_file_replay.py \
       ~/path/to/Spark_4K_2997_422_1.7bpp.rtp \
       --host 127.0.0.1 --port 6000 --rate-bps 900000000 --start-delay 1.0
   ```

2. **Run the profiler.** Pick `max_frames` ≥ 100 and `loops` ≥ 3 so the
   mean/stddev stabilize:

   ```bash
   ./build/bin/open_htj2k_rtp_decode_profile /tmp/spark_cs 200 3 2
   ```

3. **Optional — perf callgraph.** Symbols are required, so build with
   `-DCMAKE_BUILD_TYPE=RelWithDebInfo` (or full Release plus `-g`):

   ```bash
   perf record -F 999 --call-graph dwarf \
       -o /tmp/rtp_decode.perfdata -- \
       ./build/bin/open_htj2k_rtp_decode_profile /tmp/spark_cs 200 3 2
   perf report -i /tmp/rtp_decode.perfdata --stdio --no-children --percent-limit 0.5
   ```
