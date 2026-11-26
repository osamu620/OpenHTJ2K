#!/usr/bin/env python3
# Lossy decode comparison: OpenHTJ2K against OpenJPEG, single-threaded, interleaved
# repetitions, on the same codestreams.  Produces the lossy tables in docs/benchmarks.md; see
# that file for the method and the fixture.
#
#   python3 tools/bench/lossy_decode_vs_openjpeg.py --ohtj2k-bin build/bin \
#       --opj-bin /path/to/openjpeg/build/bin image.ppm [--reps 10] [--cpu 5] [--qfactor 90]
#
# OpenHTJ2K's encoder codes the image once (untimed) with the irreversible 9/7 transform at a
# Qfactor, as a Part 1 (EBCOT/MQ) and as an HT codestream with the same quantization; both
# decoders then decode both.  Encoding is not compared: two encoders would have to be matched
# in rate and quality first.  As in lossless_vs_openjpeg.py, two times are recorded per run:
# the codec time each tool reports (OpenJPEG's "decode time", which excludes writing the PPM;
# OpenHTJ2K's "elapsed time", which includes it) and the wall time of the whole process.
# The two decoders' images of each codestream must agree within one level per sample, and
# their PSNR against the source is printed.
import argparse
import math
import os
import platform
import shutil
import statistics
import sys

sys.dont_write_bytecode = True  # importing the sibling script leaves no __pycache__ behind
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lossless_vs_openjpeg import read_ppm, run  # noqa: E402


def psnr(a, b):
    mse = sum((x - y) * (x - y) for x, y in zip(a, b)) / len(a)
    return float("inf") if mse == 0 else 10.0 * math.log10(255.0 * 255.0 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="8-bit RGB PPM (P6)")
    ap.add_argument("--ohtj2k-bin", required=True, help="directory with open_htj2k_enc / open_htj2k_dec")
    ap.add_argument("--opj-bin", required=True, help="directory with opj_decompress")
    ap.add_argument("--reps", type=int, default=10, help="timed repetitions per command (default 10)")
    ap.add_argument("--cpu", type=int, default=None, help="pin every run to this CPU (Linux taskset)")
    ap.add_argument("--qfactor", type=int, default=90, help="OpenHTJ2K Qfactor of the codestreams (default 90)")
    ap.add_argument("--tmp", default="/dev/shm" if os.path.isdir("/dev/shm") else None,
                    help="scratch directory for codestreams and decoded images")
    ap.add_argument("--opj-label", default="OpenJPEG", help="column label for OpenJPEG")
    args = ap.parse_args()

    tmp = args.tmp or os.getcwd()
    work = os.path.join(tmp, "lossy_decode_vs_openjpeg_%d" % os.getpid())
    os.makedirs(work, exist_ok=True)
    enc = os.path.join(args.ohtj2k_bin, "open_htj2k_enc")
    dec = os.path.join(args.ohtj2k_bin, "open_htj2k_dec")
    opj_d = os.path.join(args.opj_bin, "opj_decompress")
    for exe in (enc, dec, opj_d):
        if not os.access(exe, os.X_OK):
            sys.exit("not executable: " + exe)
    img = os.path.abspath(args.image)
    hdr, src = read_ppm(img)
    if hdr[3] > 255:
        sys.exit("an 8-bit PPM is needed (PSNR is computed for 8-bit samples)")
    p = lambda name: os.path.join(work, name)

    # Irreversible 9/7 + ICT, five DWT levels, 64x64 code-blocks, one layer, one tile; the
    # same Qfactor quantization for both block coders (Cmodes=0 selects Part 1).
    q = "Qfactor=%d" % args.qfactor
    run([enc, "-i", img, "-o", p("p1.j2c"), "Creversible=no", "Cmodes=0", q, "-num_threads", "1"], args.cpu)
    run([enc, "-i", img, "-o", p("ht.j2c"), "Creversible=no", q, "-num_threads", "1"], args.cpu)

    cmds = {
        "opj_p1": [opj_d, "-threads", "1", "-i", p("p1.j2c"), "-o", p("opj_p1.ppm")],
        "oh_p1": [dec, "-i", p("p1.j2c"), "-o", p("oh_p1.ppm"), "-num_threads", "1"],
        "opj_ht": [opj_d, "-threads", "1", "-i", p("ht.j2c"), "-o", p("opj_ht.ppm")],
        "oh_ht": [dec, "-i", p("ht.j2c"), "-o", p("oh_ht.ppm"), "-num_threads", "1"],
    }
    order = ["opj_p1", "oh_p1", "opj_ht", "oh_ht"]

    # Warm-up (untimed), then check that the two decoders agree on each codestream.
    for k in order:
        run(cmds[k], args.cpu)
    quality = {}
    for cs in ("p1", "ht"):
        _, a = read_ppm(p("opj_%s.ppm" % cs))
        _, b = read_ppm(p("oh_%s.ppm" % cs))
        if len(a) != len(src) or len(b) != len(src):
            sys.exit("decoded image has the wrong size: " + cs)
        worst = max(abs(x - y) for x, y in zip(a, b))
        if worst > 1:
            sys.exit("the decoders differ by %d levels on the %s codestream" % (worst, cs))
        quality[cs] = (psnr(a, src), psnr(b, src))

    codec = {k: [] for k in order}
    wall = {k: [] for k in order}
    for _ in range(args.reps):
        for k in order:  # interleaved: thermal and frequency drift hit every column alike
            c, w_ = run(cmds[k], args.cpu)
            codec[k].append(c)
            wall[k].append(w_)
    med = {k: statistics.median(v) for k, v in codec.items()}
    mw = {k: statistics.median(v) for k, v in wall.items()}
    w, h = hdr[1], hdr[2]
    size = {cs: os.path.getsize(p(cs + ".j2c")) for cs in ("p1", "ht")}
    bpp = {cs: size[cs] * 8.0 / (w * h) for cs in size}

    print("image: %s (%dx%d, 8-bit RGB), Qfactor %d, %d reps, medians"
          % (os.path.basename(img), w, h, args.qfactor, args.reps))
    print("host: %s, %s" % (platform.platform(), platform.processor() or platform.machine()))
    print()
    for title, m in (("Codec time as reported by each tool:", med),
                     ("Wall time of the whole CLI process (read the codestream, decode, write the PPM):", mw)):
        print(title)
        print()
        print("| Lossy 9/7 decode, one thread | %s | OpenHTJ2K |" % args.opj_label)
        print("|---|---:|---:|")
        for row, cs in (("Part 1 codestream", "p1"), ("HT codestream", "ht")):
            print("| %s | %.0f ms | %.0f ms (%.1fx) |"
                  % (row, m["opj_" + cs], m["oh_" + cs], m["opj_" + cs] / m["oh_" + cs]))
        print()
    for cs, name in (("p1", "Part 1"), ("ht", "HT")):
        print("%s codestream: %.3f MB (%.2f bpp), PSNR %.2f dB (%s) / %.2f dB (OpenHTJ2K)"
              % (name, size[cs] / 1e6, bpp[cs], quality[cs][0], args.opj_label, quality[cs][1]))
    print("codec min/median ms: " + ", ".join("%s %.0f/%.0f" % (k, min(codec[k]), med[k]) for k in order))
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
