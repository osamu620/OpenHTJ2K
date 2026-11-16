# Qfactor: how the step sizes are computed

The technical reference behind [qfactor.md](../qfactor.md): the formula that
turns a Qfactor into quantization step sizes, the visual-weighting tables of
the JPEG Qfactor guideline, the experimental analytic CSF models and how
their constants were calibrated. For using the Qfactor, see
[qfactor.md](../qfactor.md).

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
[`source/core/codestream/visual_weighting.hpp`](../../source/core/codestream/visual_weighting.hpp),
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
modes and [flat weights](../qfactor.md#flat-weights-qweightsflat) honour the MCT actually applied, so an undecorrelated RGB encode gets unit
gains and the **luminance** CSF on every channel. The one exception is a
`Qctype` hint: a no-MCT component labelled `Cb`/`Cr` (pre-decorrelated,
typically sub-sampled YCbCr input) takes the chroma CSF while keeping unit
colour gain — there is no inverse transform amplifying its error, but its
content is still chrominance.

## Calibration and provenance

The analytic constants (`ref_ppd = 72`, the chroma `a`/`b`) were fit to the
legacy tables. The fitting scripts are kept under
[`scripts/`](../../scripts) and regenerate the comparison plots:

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
