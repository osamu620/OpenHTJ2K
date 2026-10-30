#!/usr/bin/env python3
# Lossless encode/decode wall-time comparison: OpenHTJ2K (Part 1 and HT) against
# OpenJPEG, single-threaded, interleaved repetitions.  Produces the table in
# docs/benchmarks.md; see that file for the method and the fixture.
#
#   python3 tools/bench/lossless_vs_openjpeg.py --ohtj2k-bin build/bin \
#       --opj-bin /path/to/openjpeg/build/bin image.ppm [--reps 10] [--cpu 5]
#
# Two times are recorded per run: the codec time each tool reports itself
# (OpenJPEG's "encode time" / "decode time", OpenHTJ2K's "elapsed time", both
# excluding the PPM file I/O) and the wall time of the whole CLI process (read
# the PPM, code, write the result).  Outputs go to --tmp (default /dev/shm when
# present).  Each decode is checked to reproduce the input exactly, so a wrong
# number cannot hide behind a fast one.
import argparse
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time

CODEC_TIME = re.compile(r"(?:encode time|decode time|elapsed time)[: ]+([0-9.]+)\s*\[?ms")


def read_ppm(path):
    data = open(path, "rb").read()
    toks, i = [], 0
    while len(toks) < 4:
        while data[i : i + 1].isspace():
            i += 1
        if data[i : i + 1] == b"#":
            while data[i : i + 1] not in (b"\n", b""):
                i += 1
            continue
        j = i
        while not data[j : j + 1].isspace():
            j += 1
        toks.append(data[i:j])
        i = j
    i += 1
    w, h, mv = int(toks[1]), int(toks[2]), int(toks[3])
    n = w * h * 3 * (2 if mv > 255 else 1)
    return (toks[0], w, h, mv), data[i : i + n]


def run(cmd, cpu):
    if cpu is not None and shutil.which("taskset"):
        cmd = ["taskset", "-c", str(cpu)] + cmd
    t0 = time.perf_counter()
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    t1 = time.perf_counter()
    if r.returncode != 0:
        sys.exit("command failed (%d): %s\n%s" % (r.returncode, " ".join(cmd), r.stdout[-2000:]))
    m = CODEC_TIME.search(r.stdout)
    if m is None:
        sys.exit("no codec time in the output of: " + " ".join(cmd))
    return float(m.group(1)), (t1 - t0) * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="8-bit or 16-bit RGB PPM (P6)")
    ap.add_argument("--ohtj2k-bin", required=True, help="directory with open_htj2k_enc / open_htj2k_dec")
    ap.add_argument("--opj-bin", required=True, help="directory with opj_compress / opj_decompress")
    ap.add_argument("--reps", type=int, default=10, help="timed repetitions per command (default 10)")
    ap.add_argument("--cpu", type=int, default=None, help="pin every run to this CPU (Linux taskset)")
    ap.add_argument("--tmp", default="/dev/shm" if os.path.isdir("/dev/shm") else None,
                    help="scratch directory for codestreams and decoded images")
    ap.add_argument("--opj-label", default="OpenJPEG", help="column label for OpenJPEG")
    args = ap.parse_args()

    tmp = args.tmp or os.getcwd()
    work = os.path.join(tmp, "lossless_vs_openjpeg_%d" % os.getpid())
    os.makedirs(work, exist_ok=True)
    enc = os.path.join(args.ohtj2k_bin, "open_htj2k_enc")
    dec = os.path.join(args.ohtj2k_bin, "open_htj2k_dec")
    opj_c = os.path.join(args.opj_bin, "opj_compress")
    opj_d = os.path.join(args.opj_bin, "opj_decompress")
    for exe in (enc, dec, opj_c, opj_d):
        if not os.access(exe, os.X_OK):
            sys.exit("not executable: " + exe)
    img = os.path.abspath(args.image)
    p = lambda name: os.path.join(work, name)

    # Lossless, one layer, one tile, 5 DWT levels, 64x64 code-blocks, RCT for RGB:
    # the defaults of both encoders.  OpenHTJ2K's Cmodes=0 selects the Part 1
    # (EBCOT/MQ) block coder; the default is the HT block coder.
    cmds = {
        "enc_opj": [opj_c, "-threads", "1", "-i", img, "-o", p("opj.j2c")],
        "enc_p1": [enc, "-i", img, "-o", p("oh_p1.j2c"), "Creversible=yes", "Cmodes=0", "-num_threads", "1"],
        "enc_ht": [enc, "-i", img, "-o", p("oh_ht.j2c"), "Creversible=yes", "-num_threads", "1"],
        # Both Part 1 decoders read the same codestream, the one OpenJPEG wrote.
        "dec_opj": [opj_d, "-threads", "1", "-i", p("opj.j2c"), "-o", p("opj_dec.ppm")],
        "dec_p1": [dec, "-i", p("opj.j2c"), "-o", p("oh_dec_p1.ppm"), "-num_threads", "1"],
        "dec_ht": [dec, "-i", p("oh_ht.j2c"), "-o", p("oh_dec_ht.ppm"), "-num_threads", "1"],
    }
    order = ["enc_opj", "enc_p1", "enc_ht", "dec_opj", "dec_p1", "dec_ht"]

    # Warm-up (untimed) run of everything, then verify every decode is lossless and
    # that OpenJPEG reads the Part 1 codestream OpenHTJ2K wrote.
    for k in order:
        run(cmds[k], args.cpu)
    hdr, src = read_ppm(img)
    for name in ("opj_dec.ppm", "oh_dec_p1.ppm", "oh_dec_ht.ppm"):
        h2, pix = read_ppm(p(name))
        if h2[1:] != hdr[1:] or pix != src:
            sys.exit("decode is not lossless: " + name)
    run([opj_d, "-threads", "1", "-i", p("oh_p1.j2c"), "-o", p("x.ppm")], args.cpu)
    h2, pix = read_ppm(p("x.ppm"))
    if pix != src:
        sys.exit("OpenJPEG does not reproduce the OpenHTJ2K Part 1 codestream")

    codec = {k: [] for k in order}
    wall = {k: [] for k in order}
    for _ in range(args.reps):
        for k in order:  # interleaved: thermal and frequency drift hit every column alike
            c, w_ = run(cmds[k], args.cpu)
            codec[k].append(c)
            wall[k].append(w_)
    med = {k: statistics.median(v) for k, v in codec.items()}
    mw = {k: statistics.median(v) for k, v in wall.items()}
    size = {k: os.path.getsize(p(f)) for k, f in (("opj", "opj.j2c"), ("p1", "oh_p1.j2c"), ("ht", "oh_ht.j2c"))}

    w, h = hdr[1], hdr[2]
    print("image: %s (%dx%d, %d-bit RGB), %d reps, medians"
          % (os.path.basename(img), w, h, 16 if hdr[3] > 255 else 8, args.reps))
    print("host: %s, %s" % (platform.platform(), platform.processor() or platform.machine()))
    print()
    print("Codec time as reported by each tool (file I/O excluded):")
    print()
    print("| Lossless, one thread | %s | OpenHTJ2K Part 1 | OpenHTJ2K HT |" % args.opj_label)
    print("|---|---:|---:|---:|")
    for row, a, b, c in (("Encode", "enc_opj", "enc_p1", "enc_ht"), ("Decode", "dec_opj", "dec_p1", "dec_ht")):
        print("| %s | %.0f ms | %.0f ms (%.1fx) | %.0f ms (%.1fx) |"
              % (row, med[a], med[b], med[a] / med[b], med[c], med[a] / med[c]))
    print("| Codestream | %.3f MB | %.3f MB | %.3f MB |" % (size["opj"] / 1e6, size["p1"] / 1e6, size["ht"] / 1e6))
    print()
    print("Wall time of the whole CLI process (PPM read, code, output write):")
    print()
    print("| Lossless, one thread | %s | OpenHTJ2K Part 1 | OpenHTJ2K HT |" % args.opj_label)
    print("|---|---:|---:|---:|")
    for row, a, b, c in (("Encode", "enc_opj", "enc_p1", "enc_ht"), ("Decode", "dec_opj", "dec_p1", "dec_ht")):
        print("| %s | %.0f ms | %.0f ms (%.1fx) | %.0f ms (%.1fx) |"
              % (row, mw[a], mw[b], mw[a] / mw[b], mw[c], mw[a] / mw[c]))
    print()
    print("codec min/median ms: " + ", ".join("%s %.0f/%.0f" % (k, min(codec[k]), med[k]) for k in order))
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
