# Qfactor and visual weighting

`Qfactor` is a single 1–100 "quality" knob for lossy HTJ2K encoding, in the
spirit of the classic JPEG quality factor: a small, intuitive number that
scales the quantization step sizes, with a strong connection to visual quality
(100 = best). This document explains how `Qfactor` is turned into per-subband
quantization steps, and the **experimental** analytic visual-weighting
extension that lets the weighting follow an arbitrary viewing condition (zoom)
and colour space instead of a single hard-coded table.

- Stable / default behaviour: [The Qfactor → step-size formula](#the-qfactor--step-size-formula) and [Legacy visual weighting](#legacy-visual-weighting-default).
- Experimental: [Analytic visual weighting](#analytic-visual-weighting-experimental) and [Precision-scaled Qfactor](#precision-scaled-qfactor-experimental).
- Tooling: [Recovering the Qfactor](#recovering-the-qfactor-estimate_qfactor).

> **Status.** The default path (`Qcsf=legacy`) is the long-standing Qfactor
> behaviour and follows the JPEG Qfactor guideline (WG1N101553 v2). It is
> **bit-identical** to previous releases except for sub-sampled (4:2:0 / 4:2:2)
> chroma, whose weights were brought in line with the guideline after 0.25.0
> (see [Legacy visual weighting](#legacy-visual-weighting-default)). The analytic models
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

`Qfactor` takes 1 to 4 components: gray, gray + alpha, colour, and colour +
alpha. An alpha channel (component 1 of 2, or component 3 of 4) gets a `QCC`
marker of its own, built as for a gray-scale image of its bit depth: the
luminance weights, and no colour-transform gain (the colour transform covers
components 0-2 only).

## The Qfactor → step-size formula

For each subband the encoder writes a quantization step into the `QCD` (luma /
component 0) or `QCC` (chroma / components 1, 2; alpha, see above) marker. The step for subband
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
| `W(i)` | per-subband **visual weight** (the CSF); for the LL band (and levels past a 5-level table) the tail weight | `luma_/chroma_visual_weights()`, `chroma_tail_weight()` |
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

With `Qcsf=legacy` (the default) the per-subband weights `W(i)` are the nominal
visual weights of the JPEG Qfactor guideline, *"Controlling JPEG 2000 image
quality using a single parameter (Qfactor) v2.0"* (ISO/IEC JTC 1/SC 29/WG 1
N101553), Tables 2–4: luminance plus per-format chroma (4:4:4 / 4:2:0 / 4:2:2).
The luminance and 4:4:4 chrominance weights are those of Zeng, Daly & Lei,
*"An overview of the visual optimization tools in JPEG 2000"*, Signal
Processing: Image Communication 17 (2002) 85–104, Table 2. The colour gains
`G_color` are the inverse-ICT column norms `{√3, 1.80511, 1.57340}` (stored
rounded to 4 dp as `{1.7321, 1.8051, 1.5734}`). These tables target one specific
viewing condition (≈1700-pixel viewing distance) and assume a YCbCr
(luma/chroma) decomposition.

A sub-sampled chroma component is weighted as 4:4:4 chroma one level coarser
along each sub-sampled axis, times the gain of upsampling it for display:

| format | `W(i)` (sqrt-domain) | LL band and levels past the fifth |
|---|---|---|
| 4:4:4 | Table 2 | 1.0 |
| 4:2:0 | 2 × (4:4:4 weight one level up) | 2.0 |
| 4:2:2 | √2 × (HL: 4:4:4 LH one level up; LH, HH: 4:4:4 LH at the same level) | 1.4142 |

Like every weight, the LL / tail weight is raised to the power `p`, so it fades
to 1 at `Q ≥ 97`.

This path produces **byte-identical** codestreams to previous releases for gray,
gray + alpha and 4:4:4 colour input. Sub-sampled chroma used the guideline's
v1.0 weights (without the upsampling gain, and 1.0 for the LL band) up to
0.25.0, so its chroma steps below `Q = 97` are now finer: by up to 2× for 4:2:0
and √2× for 4:2:2. `qfactor_guideline_check` (built with the test suite) checks a
codestream's steps against the guideline independently of this header.

## Analytic visual weighting (experimental)

The analytic models replace the single table with a contrast sensitivity
function (CSF) evaluated at each subband's radial spatial frequency, so the
weighting tracks the viewing condition instead of being fixed.

### Models (`Qcsf`)

| `Qcsf` | model | notes |
|---|---|---|
| `legacy` | fixed tables of WG1N101553 v2 (Zeng Table 2 for luma and 4:4:4 chroma) | default |
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
`(sx, sy)` — taken per component from the SIZ `XRsiz`/`YRsiz` fields. A
component sub-sampled by `s` on an axis sits on a grid `s×` coarser, so that
axis's angular frequency divides by `s`. Two consequences fall out of the
mapping with no special-casing:

- **Isotropic 4:2:0 is exactly a one-level shift.** Halving both axes halves
  every subband's radial frequency, which is identical to moving one
  decomposition level coarser — the classic "weight chroma like luma one level
  down" rule. Do not add a manual level offset on top; the `(sx, sy)` scaling
  already subsumes it (this equivalence, times the display gain below, is
  asserted bit-for-bit by `visual_weight_check check`).
- **4:2:2 is anisotropic.** Only the horizontal axis is halved, so a 4:2:2
  channel gets different `LH` (vertical detail) and `HL` (horizontal detail)
  weights — something no scalar per-subband table can represent.
- **Display upsampling gain.** As in the legacy tables (WG1N101553 v2), every
  weight of a component sub-sampled by `(sx, sy)`, the LL band included, is
  multiplied by `√(sx·sy)`: 2 for 4:2:0, √2 for 4:2:2. It accounts for the
  error being spread over `sx·sy` display pixels when the component is
  upsampled, and applies whatever the component's role.

**Component roles need a hint when chroma is sub-sampled.** Sub-sampled
components rule out the built-in MCT (it needs identically-sized components),
and without an MCT the codestream does not label channels — the encoder cannot
know component 1 is `Cb`. Pass `Qctype=Y,Cb,Cr` to select the chroma CSF;
unhinted components default to `generic` (luminance CSF, still with the
`(sx, sy)` frequency shift). `Qchromacsf=luma` is an A/B switch that keeps the
mapping but swaps the chroma shape for the luminance CSF.

The `visual_weight_check` test tool (built with the test suite) dumps the
derived table for any configuration, e.g. for a 4:2:2 `Cb` channel:

```bash
visual_weight_check dump --ctype Cb --sx 2 --sy 1          # analytic
visual_weight_check dump --ctype Cb --legacy-format 422    # legacy table row
```

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
gains and the **luminance** CSF on every channel. The one exception is a
`Qctype` hint: a no-MCT component labelled `Cb`/`Cr` (pre-decorrelated,
typically sub-sampled YCbCr input) takes the chroma CSF while keeping unit
colour gain — there is no inverse transform amplifying its error, but its
content is still chrominance.

## Precision-scaled Qfactor (experimental)

The guideline's reference step is a fraction of the sample range, so a Qfactor
keeps about the same number of bits at every bit depth: 4.5 bits at
`Qfactor=50`, whether the samples have 8 or 16 bits. That suits content that is
looked at on a display. Users who keep 12- or 16-bit data for its precision
(medical, scientific, camera raw, depth maps) instead get most of the scale
unused: at 16 bits, `Qfactor=99` gives a reference step of about 132 LSBs and
`Qfactor=100` one of 0.71 LSB, with nothing in between.

`Qscale=bits` (EXPERIMENTAL, opt-in) makes every Qfactor remove the same
*fraction* of the precision at every bit depth. With `Delta_8(Q)` the
guideline's step at 8 bits, in LSBs, the bits it removes above the 0.71-LSB
floor of `Qfactor=100` are `log2(sqrt(2) * Delta_8(Q))`; at `RI` bits the
proposal removes `RI/8` times as many:

```
Delta_RI(Q) = (1/sqrt(2)) * (sqrt(2) * Delta_8(Q))^(RI/8)      # in LSBs of the RI-bit samples
```

- At 8 bits it is the guideline (the same bytes).
- `Qfactor=100` stays 0.71 LSB at every bit depth.
- Integer Qfactors cover the whole range of 16-bit samples; near 100, a step of
  1 changes the 16-bit step by a factor of 2 to 3.

The visual-weight exponent `p` follows the step relative to the sample range,
`c = (Delta_RI - 1/sqrt(2)) / 2^RI`, instead of the Qfactor:
`p = clamp(log(c / c_T1) / log(c_T0 / c_T1), 0, 1)` with
`c_T0 = alpha_T0 * M_T0 = 0.028` and `c_T1 = alpha_T1 * M_T1 = 0.006`. Between the
two thresholds the guideline's `log(alpha_Q * M_Q)` is linear in its exponent,
so at 8 bits this is the guideline's `p` for every Qfactor; at higher precision,
steps too fine to see get flat (MSE-optimal) weights.

Reference step in LSBs, with the bits kept in brackets:

| Qfactor | 8-bit (= guideline) | 12-bit guideline | 12-bit precision-scaled | 16-bit guideline | 16-bit precision-scaled |
|---:|---:|---:|---:|---:|---:|
| 25 | 21.2 (3.6) | 328 (3.6) | 116 (5.1) | 5244 (3.6) | 635 (6.7) |
| 50 | 10.9 (4.5) | 165 (4.6) | 43.1 (6.6) | 2622 (4.6) | 169 (8.6) |
| 75 | 6.5 (5.3) | 93.6 (5.5) | 19.8 (7.7) | 1487 (5.5) | 60.0 (10.1) |
| 90 | 4.0 (6.0) | 53.0 (6.3) | 9.4 (8.8) | 837 (6.3) | 22.3 (11.5) |
| 99 | 1.2 (7.7) | 8.9 (8.8) | 1.6 (11.3) | 132 (9.0) | 2.1 (14.9) |
| 100 | 0.71 (8.5) | 0.71 (12.5) | 0.71 (12.5) | 0.71 (16.5) | 0.71 (16.5) |

Measured on a 1920x1080 crop of a 16-bit RGB frame of *Elephants Dream*
(5 levels, 64x64 code-blocks, 9/7 + ICT, one layer):

| Qfactor | 16-bit guideline | 16-bit precision-scaled |
|---:|---:|---:|
| 10 | 0.05 bpp, 32.7 dB | 0.17 bpp, 36.2 dB |
| 50 | 0.24 bpp, 37.6 dB | 1.97 bpp, 66.6 dB |
| 90 | 0.75 bpp, 46.0 dB | 3.13 bpp, 84.3 dB |
| 99 | 2.14 bpp, 68.5 dB | 4.51 bpp, 101.5 dB |
| 100 | 5.25 bpp, 106.5 dB | 5.25 bpp, 106.5 dB |

PSNR and SSIM rise steadily with the Qfactor; at equal rate the two scales stay
close (PSNR within -0.9 to +1.6 dB, 1 - SSIM within a factor of 1.45), so the
option mainly changes which Qfactor gives which rate. Above 8 bits every
Qfactor gives finer steps and larger files than the guideline, also for 10- to
12-bit content meant for viewing, where the guideline's scale may be the better
choice. The option is not part of the guideline (WG1N101553 v2); it is offered
for discussion and may change.

## CLI reference

All of these are encoder options and require `Qfactor`:

- `Qcsf=legacy|mannos|daly` — visual-weighting model. Default **legacy**
  (the guideline tables). `mannos` / `daly` are experimental.
- `Qppd=Float` — reference pixels-per-degree at zoom 1.0. Default **72**.
- `Qzoom=Float` — display magnification; `> 1` is zoom-in. Default **1.0**.
- `Qctype=T,T,T` — per-component role hints (`Y|Cb|Cr|generic`) for analytic
  models on no-MCT (e.g. sub-sampled YCbCr) input. Default **generic**.
- `Qchromacsf=chroma|luma` — CSF shape for `Cb`/`Cr` components. Default
  **chroma** (the low-pass chroma CSF).
- `Qscale=guideline|bits` — how the Qfactor scales with the sample precision.
  Default **guideline**; `bits` is the experimental
  [precision-scaled Qfactor](#precision-scaled-qfactor-experimental), identical
  at 8 bits.

When an analytic model is selected the encoder prints an `EXPERIMENTAL:` status
line; selecting one without `Qfactor` prints a warning and has no effect.

## C++ API

```cpp
open_htj2k::openhtj2k_encoder encoder(/* ... */, qfactor, /* ... */);
encoder.set_output_buffer(out);
// EXPERIMENTAL: model 0 = legacy (default), 1 = Mannos–Sakrison, 2 = Daly.
// ref_ppd / zoom ≤ 0 keep their defaults. Call before invoke_*.
encoder.set_visual_weighting(/*model=*/1, /*ref_ppd=*/72.0, /*zoom=*/2.0);
// EXPERIMENTAL: component roles for no-MCT input (0 generic, 1 Y, 2 Cb, 3 Cr)
// and the chroma-CSF A/B switch — the Qctype= / Qchromacsf= equivalents.
encoder.set_component_types(1, 2, 3);
encoder.set_chroma_csf_reuse_luma(false);
// EXPERIMENTAL: 1 = precision-scaled Qfactor (Qscale=bits), 0 = guideline (default).
encoder.set_qfactor_scale(1);
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

# Sub-sampled YCbCr analytic encode — pass the same role hints too
estimate_qfactor out420.j2c --csf mannos --ctype Y,Cb,Cr

# Precision-scaled encode (Qscale=bits)
estimate_qfactor out16.j2c --scale bits

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
  (RMS 0.128 / 0.062 on weights up to 2.0) and 4:2:2 (0.055 / 0.045) via the
  `(sx, sy)` frequency shift and the display upsampling gain.

```bash
python3 scripts/csf_fit_luma.py     # writes csf_fit_luma.png
python3 scripts/csf_fit_chroma.py   # writes csf_fit_chroma.png
```

## Limitations

- **Experimental.** The analytic path only affects the `QCD`/`QCC` step sizes;
  it is opt-in and leaves the legacy default unchanged.
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

- ISO/IEC JTC 1/SC 29/WG 1 N101553, *"Controlling JPEG 2000 image quality using
  a single parameter (Qfactor) v2.0"*, guideline:
  <https://ds.jpeg.org/documents/jpeg2000/wg1n101553-112-COM-Controlling_JPEG_2000_quality_using_Qfactor_v2.pdf>
- W. Zeng, S. Daly, S. Lei, *"An overview of the visual optimization tools in
  JPEG 2000"*, Signal Processing: Image Communication 17 (2002) 85–104.
- HTJ2K white paper (legacy Qfactor formula):
  <https://htj2k.com/wp-content/uploads/white-paper.pdf>
- Encoder CLI: [`cli_encoder.md`](cli_encoder.md).
