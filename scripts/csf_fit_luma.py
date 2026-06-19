#!/usr/bin/env python3
"""Verify analytic CSF models reproduce OpenHTJ2K's W_b_Y table (j2kmarkers.cpp:715).

(a) Mannos-Sakrison vs Daly, each fit by viewing geometry only.
(b) table-vs-fit plot -> csf_fit_luma.png
"""
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- table under test ----------------------------------------------------------
W_b_Y = [0.0901, 0.2758, 0.2758, 0.7018, 0.8378, 0.8378, 1.0000, 1.0000,
         1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000]
LABELS = [f"{o}{l}" for l in range(1, 6) for o in ("HH", "LH", "HL")]

# --- CSF models (f in cycles/degree) -------------------------------------------
def mannos(f):
    return 2.6 * (0.0192 + 0.114 * f) * math.exp(-((0.114 * f) ** 1.1))

def daly(f, L=100.0, eps=0.9):
    # Daly 1993 CSF core (light-adaptation form); low-freq/area term dropped
    # because we clamp below the peak anyway, so only the falling side matters.
    A = 0.801 * (1.0 + 0.7 / L) ** -0.2
    B = 0.3 * (1.0 + 100.0 / L) ** 0.15
    x = B * eps * f
    return A * eps * f * math.exp(-x) * math.sqrt(1.0 + 0.06 * math.exp(x))

def normalizer(model):
    grid = [i * 0.002 for i in range(1, 100000)]
    fpk = max(grid, key=model)
    return fpk, model(fpk)

# --- subband radial frequencies ------------------------------------------------
def band_freqs(ppd, hh_factor, convention):
    f_N = ppd / 2.0
    out = []
    for lvl in range(1, 6):
        if convention == "geomean":   # geo-mean of octave band [f_N/2^l, f_N/2^(l-1)]
            f_r = f_N * 2.0 ** -lvl * math.sqrt(2.0)
        elif convention == "upper":   # upper band edge
            f_r = f_N * 2.0 ** -(lvl - 1)
        else:                          # lower band edge
            f_r = f_N * 2.0 ** -lvl
        out += [f_r * hh_factor, f_r, f_r]   # HH, LH, HL
    return out

def predict(model, fpk, hpk, ppd, hh_factor, convention):
    w = []
    for f in band_freqs(ppd, hh_factor, convention):
        w.append(1.0 if f <= fpk else model(f) / hpk)   # flat below peak
    return w

def rms(pred):
    return math.sqrt(sum((p - a) ** 2 for p, a in zip(pred, W_b_Y)) / len(pred))

def maxerr(pred):
    return max(abs(p - a) for p, a in zip(pred, W_b_Y))

def fit(model, convention, fit_hh):
    fpk, hpk = normalizer(model)
    best = None
    hh_range = range(80, 201, 1) if fit_hh else [int(round(math.sqrt(2.0) * 100))]
    for p in range(800, 30000, 5):
        ppd = p / 100.0
        for h in hh_range:
            hh = h / 100.0
            r = rms(predict(model, fpk, hpk, ppd, hh, convention))
            if best is None or r < best[0]:
                best = (r, ppd, hh)
    r, ppd, hh = best
    pred = predict(model, fpk, hpk, ppd, hh, convention)
    return dict(rms=r, ppd=ppd, hh=hh, fpk=fpk, dist=ppd * 180.0 / math.pi,
                maxerr=maxerr(pred), pred=pred)

# --- (a) report ----------------------------------------------------------------
print("=== (a) Which CSF + viewing distance reproduces W_b_Y? ===\n")
hdr = f"{'model':16s}{'conv':9s}{'ppd':>8s}{'dist[px]':>10s}{'HHfac':>7s}{'RMS':>8s}{'maxErr':>8s}"
print(hdr); print("-" * len(hdr))
results = {}
for mname, model in (("Mannos-Sakrison", mannos), ("Daly(L=100)", daly)):
    for conv in ("geomean", "upper"):
        for fit_hh in (False, True):
            r = fit(model, conv, fit_hh)
            tag = f"{mname}{'+HHfit' if fit_hh else ''}"
            print(f"{tag:16s}{conv:9s}{r['ppd']:8.2f}{r['dist']:10.0f}"
                  f"{r['hh']:7.2f}{r['rms']:8.4f}{r['maxerr']:8.4f}")
            results[(mname, conv, fit_hh)] = r

print(f"\nMannos peak = {results[('Mannos-Sakrison','geomean',False)]['fpk']:.2f} cpd, "
      f"Daly peak = {results[('Daly(L=100)','geomean',False)]['fpk']:.2f} cpd")
print("Nominal target from Zeng et al. Table 2 caption: ~1700 px viewing distance.")

# representative per-band table: Daly, geomean, HH fixed at sqrt(2)
rD = results[("Daly(L=100)", "geomean", False)]
rM = results[("Mannos-Sakrison", "geomean", False)]
print(f"\nper-band (Daly geomean, ppd={rD['ppd']:.1f} ~= {rD['dist']:.0f}px):")
print(f"  {'band':5s}{'table':>9s}{'Daly':>9s}{'Mannos':>9s}")
for lab, a, d, m in zip(LABELS, W_b_Y, rD["pred"], rM["pred"]):
    print(f"  {lab:5s}{a:9.4f}{d:9.4f}{m:9.4f}")

# --- (b) plot ------------------------------------------------------------------
x = list(range(len(W_b_Y)))
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), height_ratios=[3, 1], sharex=True)

ax1.plot(x, W_b_Y, "ko-", lw=2, ms=7, label="W_b_Y table (Zeng Table 2)", zorder=5)
ax1.plot(x, rM["pred"], "s--", color="tab:blue", ms=6,
         label=f"Mannos fit  (ppd={rM['ppd']:.0f}≈{rM['dist']:.0f}px, RMS={rM['rms']:.3f})")
ax1.plot(x, rD["pred"], "^--", color="tab:red", ms=6,
         label=f"Daly fit  (ppd={rD['ppd']:.0f}≈{rD['dist']:.0f}px, RMS={rD['rms']:.3f})")
rDh = results[("Daly(L=100)", "geomean", True)]
ax1.plot(x, rDh["pred"], "v:", color="tab:green", ms=6,
         label=f"Daly+HHfit (HH={rDh['hh']:.2f}, RMS={rDh['rms']:.3f})")
ax1.set_ylabel("visual weight  W (sqrt-domain)")
ax1.set_title("Analytic CSF vs OpenHTJ2K W_b_Y (luminance, 5-level)")
ax1.legend(fontsize=8, loc="lower right")
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-0.05, 1.1)

ax2.axhline(0, color="k", lw=0.8)
ax2.plot(x, [p - a for p, a in zip(rM["pred"], W_b_Y)], "s--", color="tab:blue", ms=5)
ax2.plot(x, [p - a for p, a in zip(rD["pred"], W_b_Y)], "^--", color="tab:red", ms=5)
ax2.plot(x, [p - a for p, a in zip(rDh["pred"], W_b_Y)], "v:", color="tab:green", ms=5)
ax2.set_ylabel("fit − table")
ax2.set_xticks(x)
ax2.set_xticklabels(LABELS, rotation=45, fontsize=8)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("csf_fit_luma.png", dpi=130)
print("\nplot -> csf_fit_luma.png")
