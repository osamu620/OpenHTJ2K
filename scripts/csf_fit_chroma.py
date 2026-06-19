#!/usr/bin/env python3
"""Validate ONE analytic chroma CSF (per channel) + sampling factors reproduces
all three QCC tables (4:4:4 / 4:2:0 / 4:2:2) in j2kmarkers.cpp:1000-1023.

Key idea: 4:2:0 / 4:2:2 are NOT separate CSFs -- they are the SAME chroma CSF
sampled at frequencies shifted by the chroma subsampling factor (sx, sy).
"""
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = [f"{o}{l}" for l in range(1, 6) for o in ("HH", "LH", "HL")]

# sqrt-domain weights from the code (order per level: HH, LH, HL) ---------------
T444 = {
 "Cb": [0.0263,0.0863,0.0863,0.1362,0.2564,0.2564,0.3346,0.4691,0.4691,0.5444,0.6523,0.6523,0.7078,0.7797,0.7797],
 "Cr": [0.0773,0.1835,0.1835,0.2598,0.4130,0.4130,0.5040,0.6464,0.6464,0.7220,0.8254,0.8254,0.8769,0.9424,0.9424]}
T420 = {
 "Cb": [0.1362,0.2564,0.2564,0.3346,0.4691,0.4691,0.5444,0.6523,0.6523,0.7078,0.7797,0.7797,1.0,1.0,1.0],
 "Cr": [0.2598,0.4130,0.4130,0.5040,0.6464,0.6464,0.7220,0.8254,0.8254,0.8769,0.9424,0.9424,1.0,1.0,1.0]}
T422 = {
 "Cb": [0.0863,0.0863,0.2564,0.2564,0.2564,0.4691,0.4691,0.4691,0.6523,0.6523,0.6523,0.7797,0.7797,0.7797,1.0],
 "Cr": [0.1835,0.1835,0.4130,0.4130,0.4130,0.6464,0.6464,0.6464,0.8254,0.8254,0.8254,0.9424,0.9424,0.9424,1.0]}

PPD = 72.0          # same physical viewing condition as the luma anchor
F_N = PPD / 2.0

def chroma_csf(f, a, b):           # low-pass stretched exponential, =1 at DC
    return math.exp(-((a * f) ** b))

def predict(a, b, sx, sy):
    fNx, fNy = F_N / sx, F_N / sy
    w = []
    for lvl in range(1, 6):
        dx = fNx * 2.0 ** -lvl * math.sqrt(2.0)   # horizontal detail freq
        dy = fNy * 2.0 ** -lvl * math.sqrt(2.0)   # vertical detail freq
        f_hh = math.hypot(dx, dy)                 # HH = sqrt(dx^2+dy^2)
        w += [min(1.0, chroma_csf(f_hh, a, b)),   # HH
              min(1.0, chroma_csf(dy, a, b)),     # LH (vertical detail)
              min(1.0, chroma_csf(dx, a, b))]     # HL (horizontal detail)
    return w

def rms(p, t):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(p, t)) / len(t))

def maxerr(p, t):
    return max(abs(x - y) for x, y in zip(p, t))

# fit (a,b) on the 4:4:4 table only ; then VALIDATE on 4:2:0 and 4:2:2 ----------
fits = {}
for ch in ("Cb", "Cr"):
    best = None
    for ai in range(50, 1200):
        a = ai / 10000.0
        for bi in range(50, 160):
            b = bi / 100.0
            r = rms(predict(a, b, 1, 1), T444[ch])
            if best is None or r < best[0]:
                best = (r, a, b)
    fits[ch] = best
    print(f"{ch}: fit on 4:4:4 -> a={best[1]:.4f} b={best[2]:.3f}  RMS={best[0]:.4f}")

print("\nValidation with the SAME (a,b), only sampling factors change:")
print(f"  {'fmt':6s}{'sx,sy':7s}{'Cb RMS':>9s}{'Cb max':>8s}{'Cr RMS':>9s}{'Cr max':>8s}")
for fmt, sx, sy, tab in (("4:4:4",1,1,T444),("4:2:0",2,2,T420),("4:2:2",2,1,T422)):
    row = f"  {fmt:6s}{f'{sx},{sy}':7s}"
    for ch in ("Cb","Cr"):
        a,b = fits[ch][1], fits[ch][2]
        p = predict(a,b,sx,sy)
        row += f"{rms(p,tab[ch]):9.4f}{maxerr(p,tab[ch]):8.4f}"
    print(row)

# plot ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True, sharey=True)
for col,(fmt,sx,sy,tab) in enumerate((("4:4:4",1,1,T444),("4:2:0",2,2,T420),("4:2:2",2,1,T422))):
    for row,ch in enumerate(("Cb","Cr")):
        ax = axes[row][col]
        a,b = fits[ch][1], fits[ch][2]
        p = predict(a,b,sx,sy)
        x = range(15)
        ax.plot(x, tab[ch], "ko-", ms=5, label="table")
        ax.plot(x, p, "r^--", ms=5, label=f"analytic (RMS={rms(p,tab[ch]):.3f})")
        ax.set_title(f"{ch}  {fmt}")
        ax.grid(True, alpha=0.3); ax.set_ylim(-0.05,1.1)
        ax.legend(fontsize=7, loc="upper left")
        if col==0: ax.set_ylabel(f"{ch} weight")
        if row==1: ax.set_xticks(x); ax.set_xticklabels(LABELS, rotation=45, fontsize=7)
fig.suptitle("ONE chroma CSF per channel (fit on 4:4:4) reproduces 4:2:0 & 4:2:2 via (sx,sy)")
plt.tight_layout()
plt.savefig("csf_fit_chroma.png", dpi=120)
print("\nplot -> csf_fit_chroma.png")
