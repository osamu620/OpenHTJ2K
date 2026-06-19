# Qfactor and visual weighting

`Qfactor` is a single 1–100 "quality" knob for lossy HTJ2K encoding, in the
spirit of the classic JPEG quality factor: a small, intuitive number that
scales the quantization step sizes, with a strong connection to visual quality
(100 = best). This document explains how `Qfactor` is turned into per-subband
quantization steps, and the **experimental** analytic visual-weighting
extension that lets the weighting follow an arbitrary viewing condition (zoom)
and colour space instead of a single hard-coded table.

- Stable / default behaviour: [The Qfactor → step-size formula](#the-qfactor--step-size-formula) and [Legacy visual weighting](#legacy-visual-weighting-default).
- Experimental: [Analytic visual weighting](#analytic-visual-weighting-experimental).
- Tooling: [Recovering the Qfactor](#recovering-the-qfactor-estimate_qfactor).

> **Status.** The default path (`Qcsf=legacy`) is the long-standing Qfactor
> behaviour and is **bit-identical** to previous releases. The analytic models
> (`Qcsf=mannos`, `Qcsf=daly`) and the `Qppd` / `Qzoom` knobs are **experimental**
> and opt-in; they only change the emitted `QCD`/`QCC` quantization markers, so
> any compliant decoder reads the result normally.

## Quick start

```bash
# Default Qfactor (legacy weighting) — quality 90
open_htj2k_enc -i input.ppm -o out.j2c Qfactor=90

# Experimental: analytic Mannos–Sakrison CSF weighting
open_htj2k_enc -i input.ppm -o out.j2c Qfactor=90 Qcsf=mannos

# Tune for a closer / magnified viewing condition (zoom-in flattens the
# weighting toward flat MSE-optimal quantization)
open_htj2k_enc -i input.ppm -o out.j2c Qfactor=90 Qcsf=mannos Qzoom=2
```

`Qfactor` requires lossy mode (`Creversible=no`, the default) and is designed
for a luma/chroma (YCbCr) decomposition: `Cycc` defaults to `yes`, and if you
disable it on 3-component input the encoder warns and treats the input as
already YCbCr. When `Qfactor` is present, `Qstep` is ignored.

## The Qfactor → step-size formula

For each subband the encoder writes a quantization step into the `QCD` (luma /
component 0) or `QCC` (chroma / components 1, 2) marker. The step for subband
`i` of component `c` is

```
              delta_Q · G_color[ref]
step(i, c) = ───────────────────────────────────
              G_basis(i) · W(i)^p · G_color[c]
```

| factor | meaning | source |
|---|---|---|
| `delta_Q` | the global perceptual scale set by Qfactor | `q_to_delta()` |
| `p` (`qfactor_power`) | how strongly visual weighting is applied (1 → full, 0 → flat) | `q_to_delta()` |
| `G_basis(i)` | L2 synthesis gain of the DWT basis for subband `i` (so equal step in transform domain ⇒ equal MSE in image domain) | DWT filter taps |
| `W(i)` | per-subband **visual weight** (the CSF) | `luma_/chroma_visual_weights()` |
| `G_color[c]` | per-component colour synthesis gain (maps component error to display-RGB error) | `color_gain()` |

`ref` is the luma/reference component. Everything lives in one header,
[`source/core/codestream/visual_weighting.hpp`](../source/core/codestream/visual_weighting.hpp),
which is the single source of truth shared by the `QCD`/`QCC` marker
construction and by `estimate_qfactor`.

### `delta_Q` and the quality curve

`q_to_delta(Qfactor, RI)` maps the 1–100 factor to `delta_Q` and the weighting
exponent `p`. The shape is the familiar libjpeg-style curve:

```
M_Q = (Q < 50) ? 50/Q : 2·(1 − Q/100)      # master multiplier
delta_Q = alpha_Q · M_Q + eps0             # eps0 = sqrt(1/2) / 2^RI  (sub-LSB floor; RI = component bit depth)
```

`alpha_Q` interpolates `0.04 → 0.10` and the visual-weighting exponent `p`
fades from **1** (full CSF weighting, `Q ≤ 65`) to **0** (flat / MSE-optimal,
`Q ≥ 97`). The near-lossless fade avoids differentially starving the
high-frequency bands when the overall distortion is already tiny. The detail of
this curve is described in the HTJ2K white paper
(<https://htj2k.com/wp-content/uploads/white-paper.pdf>).

## Legacy visual weighting (default)

With `Qcsf=legacy` (the default) the per-subband weights `W(i)` come from the
fixed contrast-sensitivity tables of Zeng, Daly & Lei, *"An overview of the
visual optimization tools in JPEG 2000"*, Signal Processing: Image
Communication 17 (2002) 85–104, Table 2 — luminance plus per-format chroma
(4:4:4 / 4:2:0 / 4:2:2). The colour gains `G_color` are the inverse-ICT column
norms `{√3, 1.80511, 1.57340}` (stored rounded to 4 dp as
`{1.7321, 1.8051, 1.5734}`). These tables target one specific viewing condition
(≈1700-pixel viewing distance) and assume a YCbCr (luma/chroma) decomposition.

This path is unchanged from previous releases and produces **byte-identical**
codestreams.

## Analytic visual weighting (experimental)

The analytic models replace the single table with a contrast sensitivity
function (CSF) evaluated at each subband's radial spatial frequency, so the
weighting tracks the viewing condition instead of being fixed.

### Models (`Qcsf`)

| `Qcsf` | model | notes |
|---|---|---|
| `legacy` | hard-coded Zeng Table 2 | default; bit-identical |
| `mannos` | Mannos–Sakrison CSF | reproduces the legacy luma table to RMS ≈ 0.029 at `Qppd=72` |
| `daly`   | Daly 1993 CSF (light-adaptation form) | rolls off lower; reproduces the legacy table near the paper's ~1700 px distance but with a looser shape |

### Viewing condition (`Qppd`, `Qzoom`)

The CSF is defined in cycles per **degree**; mapping it to a subband requires
the display geometry, expressed as pixels-per-degree (ppd):

- `Qppd` — reference pixels-per-degree at zoom 1.0 (default **72**, the value
  that makes `mannos` reproduce the legacy luma table).
- `Qzoom` — display magnification. The effective `ppd = Qppd / Qzoom`, so
  **zoom-in (`Qzoom > 1`) lowers the effective ppd**, shifts every subband to a
  lower cycles/degree, and flattens the weighting toward **1.0** — i.e. toward
  flat MSE-optimal quantization, the correct limit when the image is heavily
  magnified and the eye can resolve the finest detail.

A level-`ℓ` detail subband is assigned the geometric-mean radial frequency of
its octave band, `f_r(ℓ) = (ppd/2) · 2^−ℓ · √2`; the diagonal `HH` band sits a
factor `hh_factor` (default `√2`) higher. The luminance CSF is normalized to its
peak and held flat below the peak (so DC and low frequencies are never
down-weighted), matching how the original tables were built.

### Chroma weighting

Chrominance is low-pass (the eye resolves far less colour detail than luma), so
the analytic chroma weight is a stretched-exponential `exp(−(a·f)^b)`, with one
parameter pair per opponent channel calibrated at `Qppd=72`:

| channel | a | b |
|---|---|---|
| Cb (blue-yellow) | 0.1173 | 0.840 |
| Cr (red-green)   | 0.0699 | 1.050 |

Chroma subsampling is **not** a separate model: 4:2:0 / 4:2:2 are the same CSF
sampled at frequencies shifted by the horizontal / vertical subsampling factors
`(sx, sy)`. A 4:2:2 channel therefore gets different `LH` (vertical detail) and
`HL` (horizontal detail) weights for free.

### Colour transform and gains

`G_color[c]` is the L2 norm of column `c` of the inverse colour transform —
the factor by which quantization error in component `c` is amplified in
reconstructed RGB. The model in force is resolved per encode:

| transform | gains | component roles |
|---|---|---|
| `ict` (9/7 MCT, YCbCr) | inverse-ICT column norms | comp 0 = luma, 1/2 = chroma |
| `none` (no MCT) | all `1.0` | every component is luma-role (independent RGB) |

Legacy mode always assumes `ict` (reproducing historical behaviour); analytic
modes honour the MCT actually applied, so an undecorrelated RGB encode gets unit
gains and the **luminance** CSF on every channel (never the chroma roll-off).

## CLI reference

All three are encoder options and require `Qfactor`:

- `Qcsf=legacy|mannos|daly` — visual-weighting model. Default **legacy**
  (bit-identical). `mannos` / `daly` are experimental.
- `Qppd=Float` — reference pixels-per-degree at zoom 1.0. Default **72**.
- `Qzoom=Float` — display magnification; `> 1` is zoom-in. Default **1.0**.

When an analytic model is selected the encoder prints an `EXPERIMENTAL:` status
line; selecting one without `Qfactor` prints a warning and has no effect.

## C++ API

```cpp
open_htj2k::openhtj2k_encoder encoder(/* ... */, qfactor, /* ... */);
encoder.set_output_buffer(out);
// EXPERIMENTAL: model 0 = legacy (default), 1 = Mannos–Sakrison, 2 = Daly.
// ref_ppd / zoom ≤ 0 keep their defaults. Call before invoke_*.
encoder.set_visual_weighting(/*model=*/1, /*ref_ppd=*/72.0, /*zoom=*/2.0);
encoder.invoke_line_based();
```

The default-constructed weighting is `legacy`, so callers that never call
`set_visual_weighting()` (including the WASM and JPIP bindings) are unchanged.

## Recovering the Qfactor (`estimate_qfactor`)

The Qfactor is **not** signalled in the codestream — `estimate_qfactor` reverse
-engineers it by inverting the step-size formula above (using the same shared
header, so it tracks the encoder exactly). The visual-weighting model is also
not signalled, so for an analytic encode you must tell the tool which model /
viewing condition was used; a low residual confirms the guess.

```bash
# Legacy encode
estimate_qfactor out.j2c

# Analytic encode — pass the same model / viewing condition used at encode time
estimate_qfactor out.j2c --csf mannos --zoom 2

# Scriptable check mode (CI): assert the recovered Q and a residual ceiling
estimate_qfactor out.j2c --csf mannos --expect-q 90 --max-residual 0.01
```

Check-mode exit codes: `0` = OK / `CHECK PASS`, `2` = `CHECK FAIL` (Q or
residual violated), `3` = `CHECK SKIP` (nothing to evaluate, e.g. a lossless
stream), `1` = usage / parse error. A large residual means the assumed
weighting did not match — the file was likely produced by a different model,
viewing condition, or encoder.

## Calibration and provenance

The analytic constants (`ref_ppd = 72`, the chroma `a`/`b`) were fit to the
legacy tables. The fitting scripts are kept under
[`scripts/`](../scripts) and regenerate the comparison plots:

- `scripts/csf_fit_luma.py` — fits Mannos–Sakrison and Daly to the luminance
  table; this is where `ref_ppd ≈ 72` comes from (Mannos RMS ≈ 0.029). It also
  shows that `daly` reproduces the table near the paper's ~1700 px distance but
  with a looser shape, confirming that the reference ppd is a per-model
  calibration constant rather than a literal physical distance.
- `scripts/csf_fit_chroma.py` — fits the per-channel chroma CSF on 4:4:4
  (Cb RMS 0.010, Cr RMS 0.021) and shows the same parameters reproduce 4:2:0
  and 4:2:2 via the `(sx, sy)` frequency shift.

```bash
python3 scripts/csf_fit_luma.py     # writes csf_fit_luma.png
python3 scripts/csf_fit_chroma.py   # writes csf_fit_chroma.png
```

## Limitations

- **Experimental.** The analytic path only affects the `QCD`/`QCC` step sizes;
  it is opt-in and the legacy default is byte-identical.
- **Signal-independent.** This is fixed-frequency weighting (one CSF sample per
  octave band). It does not model visual masking or content adaptation, so
  "Q = 80 looks like Q = 80" holds only approximately and is image-dependent.
- **Colour target.** `G_color` maps to *linear RGB MSE*; it is not a full
  appearance model (no display EOTF/gamma, primaries, or chromatic adaptation).
- **Chroma anchor.** The chroma parameters are calibrated at `ref_ppd = 72`
  against the Mannos anchor; switching the luminance model to `daly` (which
  prefers a different anchor) would warrant re-calibrating them.
- `estimate_qfactor` inverts under the model you specify; it cannot detect which
  analytic model produced an unknown file (the residual only tells you whether
  your guess was right).

## References

- W. Zeng, S. Daly, S. Lei, *"An overview of the visual optimization tools in
  JPEG 2000"*, Signal Processing: Image Communication 17 (2002) 85–104.
- HTJ2K white paper (legacy Qfactor formula):
  <https://htj2k.com/wp-content/uploads/white-paper.pdf>
- Encoder CLI: [`cli_encoder.md`](cli_encoder.md).
