#!/usr/bin/env python3
# Speed A/B check: times two builds of OpenHTJ2K on the same machine and fails when the second
# ("head") is slower than the first ("base") by more than a threshold.  Used by
# .github/workflows/perf-ab.yml on pull requests, and before a release against the previous
# one (run it on a quiet machine there; CI runners are noisy).
#
#   python3 tools/bench/ab_speed.py --base-bin BASE/build/bin --head-bin HEAD/build/bin \
#       [--base-wasm BASE/web/build/html --head-wasm HEAD/web/build/html] \
#       [--rounds 7] [--threshold 0.25]
#
# Cases:
#   native decode of conformance_data/4KuF_H0L1_arri_1bpp.j2c (3840x2160 HT) on 1 and 4 threads;
#   native decode of that image coded in 64 x 64 tiles (by the base encoder) on 4 threads;
#   native encode of that image (12-bit RGB) at Qfactor=90 and lossless, on 1 and 4 threads;
#   with --*-wasm: the WebAssembly decoder on the RTP demo's per-frame path (one decoder,
#   reset_decoder_with_bytes() every frame) for conformance_data/ATK_DFS_IRV.j2c (12-bit 4:2:2),
#   multi-threaded (4 threads) and single-threaded.
#
# Base and head runs alternate (ABBA order within each round) and each side's median is
# compared.  A case over the threshold is measured again with twice the rounds before it
# counts as a failure, so one noisy stretch does not fail the check.  Native times are the
# wall time of the whole CLI process; WASM times are the median decode time per frame that
# tools/bench/wasm_reinit_bench.mjs reports.  A summary table goes to stdout and, in GitHub
# Actions, to the job summary.
import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEC_FIXTURE = os.path.join(ROOT, "conformance_data", "4KuF_H0L1_arri_1bpp.j2c")
WASM_FIXTURE = os.path.join(ROOT, "conformance_data", "ATK_DFS_IRV.j2c")
WASM_BENCH = os.path.join(ROOT, "tools", "bench", "wasm_reinit_bench.mjs")


def run_native(cmd):
    t0 = time.perf_counter()
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    dt = time.perf_counter() - t0
    if r.returncode != 0:
        sys.exit(f"command failed ({r.returncode}): {' '.join(cmd)}\n{r.stderr.decode(errors='replace')}")
    return dt * 1000.0


def run_wasm(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"command failed ({r.returncode}): {' '.join(cmd)}\n{r.stdout}{r.stderr}")
    return float(json.loads(r.stdout.strip().splitlines()[-1])["median_ms"])


def cases(args, tmp, ppm):
    out = []
    for threads, iters in ((1, 3), (4, 10)):
        out.append((f"decode 4K HT, {threads} thread{'s' if threads > 1 else ''} (x{iters})", run_native,
                    lambda b, t=threads, i=iters: [os.path.join(b, "open_htj2k_dec"), "-i", DEC_FIXTURE,
                                                   "-o", os.path.join(tmp, "dec.ppm"), "-num_threads", str(t),
                                                   "-iter", str(i)]))
    out.append(("decode 4K in 64x64 tiles, 4 threads (x3)", run_native,
                lambda b: [os.path.join(b, "open_htj2k_dec"), "-i", os.path.join(tmp, "tiles64.j2c"),
                           "-o", os.path.join(tmp, "dec.ppm"), "-num_threads", "4", "-iter", "3"]))
    for label, opts in (("Qfactor=90", ["Qfactor=90"]), ("lossless", ["Creversible=yes"])):
        for threads in (1, 4):
            out.append((f"encode 4K 12-bit {label}, {threads} thread{'s' if threads > 1 else ''}", run_native,
                        lambda b, o=opts, t=threads: [os.path.join(b, "open_htj2k_enc"), "-i", ppm,
                                                      "-o", os.path.join(tmp, "enc.j2c"), *o,
                                                      "-num_threads", str(t)]))
    if args.base_wasm and args.head_wasm:
        out.append(("WASM 4:2:2 per-frame reset, mt_simd 4 threads (ms/frame)", run_wasm,
                    lambda d: ["node", WASM_BENCH, d, "mt_simd", WASM_FIXTURE, "30", "4"]))
        out.append(("WASM 4:2:2 per-frame reset, simd 1 thread (ms/frame)", run_wasm,
                    lambda d: ["node", WASM_BENCH, d, "simd", WASM_FIXTURE, "10"]))
    return out


def measure(fn, make, base, head, rounds):
    b, h = [], []
    fn(make(base)), fn(make(head))  # warm-up: page cache, CPU frequency
    for r in range(rounds):
        order = ((base, b), (head, h)) if r % 2 == 0 else ((head, h), (base, b))
        for side, acc in order:
            acc.append(fn(make(side)))
    return statistics.median(b), statistics.median(h)


def main():
    ap = argparse.ArgumentParser(description="Speed A/B check of two OpenHTJ2K builds (see the header)")
    ap.add_argument("--base-bin", required=True)
    ap.add_argument("--head-bin", required=True)
    ap.add_argument("--base-wasm")
    ap.add_argument("--head-wasm")
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--threshold", type=float, default=0.25,
                    help="fail when head/base - 1 exceeds this (default 0.25)")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        ppm = os.path.join(tmp, "input.ppm")
        run_native([os.path.join(args.base_bin, "open_htj2k_dec"), "-i", DEC_FIXTURE, "-o", ppm])
        run_native([os.path.join(args.base_bin, "open_htj2k_enc"), "-i", ppm, "-o", os.path.join(tmp, "tiles64.j2c"),
                    "Stiles={64,64}"])
        rows, failed = [], []
        for label, fn, make in cases(args, tmp, ppm):
            native = fn is run_native
            base, head = (args.base_bin, args.head_bin) if native else (args.base_wasm, args.head_wasm)
            mb, mh = measure(fn, make, base, head, args.rounds)
            ratio = mh / mb
            note = ""
            if ratio - 1.0 > args.threshold:
                mb, mh = measure(fn, make, base, head, 2 * args.rounds)
                ratio = mh / mb
                note = " (re-measured)"
            ok = ratio - 1.0 <= args.threshold
            if not ok:
                failed.append(label)
            rows.append((label, mb, mh, ratio, ("ok" if ok else "SLOWER") + note))
            print(f"{'ok    ' if ok else 'SLOWER'} {label}: base {mb:.1f} ms, head {mh:.1f} ms, "
                  f"head/base {ratio:.3f}{note}", flush=True)

    table = ["| Case | Base (ms) | Head (ms) | Head / base | |", "|---|---:|---:|---:|---|"]
    table += [f"| {l} | {b:.1f} | {h:.1f} | {r:.3f} | {s} |" for l, b, h, r, s in rows]
    summary = (f"### Speed A/B (threshold +{100 * args.threshold:.0f}%, median of {args.rounds} "
               f"alternating runs)\n\n" + "\n".join(table) + "\n")
    print("\n" + summary)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(summary)
    if failed:
        print(f"FAIL: head is more than {100 * args.threshold:.0f}% slower in: " + "; ".join(failed))
        return 1
    print("PASS: no case is more than "
          f"{100 * args.threshold:.0f}% slower")
    return 0


if __name__ == "__main__":
    sys.exit(main())
