# wt_viewer test fixtures

Small captured RFC 9828 RTP streams used by the CI smoke test
(`.github/workflows/wt_viewer.yml`) and by `scripts/e2e_smoke.sh`
when no fixture is passed on the command line.

The format is the same one that the existing `open_htj2k_rtp_recv`
fixtures use:

    [0xFFFF marker (2 B BE)] [length (2 B BE)] [RTP packet of that length] …

Each entry is one captured UDP datagram. Replay scripts (`udp_replay.mjs`)
walk the file and re-send the packets at a chosen pace.

## Files

- **`1080p2997_30frames.rtp`** — 1920×1080, 29.97 fps, 30 frames
  (≈ 1 s wall time at source rate). About 7 MB. Sized for CI: small
  enough to commit, long enough to verify end-to-end decode beyond
  the cold-start / first-frame edge cases.

Larger fixtures (4K @ 30 / 4K @ 60) live outside the repo. They're
useful for stress testing but aren't required for bitrot detection.

## Adding more fixtures

If you add a new `.rtp` file here, keep it under ~10 MB and prefer
30 frames or fewer of source content. CI clones the repo on every
run; large fixtures slow every contributor's checkout, every PR
build, and every `git clone --depth 1` mirror.
