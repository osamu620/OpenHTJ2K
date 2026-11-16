# Qfactor and visual weighting

`Qfactor` is a single 1–100 quality setting for lossy encoding, in the spirit of
the JPEG quality factor: a higher number gives finer quantization steps, better
quality and larger files (100 = best). By default it follows the JPEG Qfactor
guideline (WG1N101553 v2), which weights the steps for viewing: coarser where
the eye sees less.

Which settings to use:

- Images people look at: `Qfactor` alone.
- Scientific or other non-visual data, where every sample value matters: add
  [`Qweights=flat`](#flat-weights-qweightsflat).
- Samples of more than 8 bits kept for their precision: add
  [`Qscale=bits`](#precision-scaled-qfactor-qscalebits).
- A different viewing condition (experimental):
  [analytic visual weighting](#analytic-visual-weighting-experimental).
- Finding the Qfactor of an existing file:
  [`estimate_qfactor`](#recovering-the-qfactor-estimate_qfactor).

> **Status.** The default (`Qcsf=legacy`) follows the guideline and gives the
> same codestreams as previous releases, except for sub-sampled (4:2:0 / 4:2:2)
> chroma, whose weights were brought in line with the guideline in 0.26.0. The
> analytic models (`Qcsf=mannos`, `Qcsf=daly`) and `Qppd` / `Qzoom` are
> **experimental** and opt-in. Every option only changes the step sizes in the
> `QCD`/`QCC` markers, so any compliant decoder reads the result.

## Quick start

```bash
# Default Qfactor (legacy weighting) — quality 90
open_htj2k_enc -i input.ppm -o out.j2c Qfactor=90

# Experimental: analytic Mannos–Sakrison CSF weighting
open_htj2k_enc -i input.ppm -o out.j2c Qfactor=90 Qcsf=mannos

# Tune for a closer / magnified viewing condition (zoom-in flattens the
# weighting toward flat MSE-optimal quantization)
open_htj2k_enc -i input.ppm -o out.j2c Qfactor=90 Qcsf=mannos Qzoom=2

# No visual weights (scientific or other non-visual data)
open_htj2k_enc -i input.pgm -o out.j2c Qfactor=50 Qweights=flat
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

## Flat weights (`Qweights=flat`)

The visual weights suit images that people look at, from about the viewing
distance the tables assume. They also make the high-frequency subbands much
coarser than the rest, which is wrong for scientific or other non-visual data,
where every sample value matters. `Qweights=flat` turns them off and keeps the
Qfactor as the quantization knob:

- Every subband weight is 1 at every Qfactor (`p = 0`), so each step only
  normalizes the subband's synthesis gain `G_b`
  ([the formula](dev/qfactor_design.md#the-qfactor--step-size-formula)): the steps are MSE-optimal, as
  the guideline's are from `Q = 97` up.
- The Qfactor sets the reference step `delta_Q` exactly as with visual weights.
  The LL step is unchanged and the detail steps are finer, so a Qfactor gives
  larger files and smaller errors than with visual weights.
- With the MCT, the colour-transform gains stay, so the error is spread evenly
  over the reconstructed R, G and B samples. Without the MCT (`Cycc=no`, or
  sub-sampled input), every component is independent data with unit gain. The
  default weighting instead assumes the YCbCr gains for components 1 and 2
  whether or not the MCT is applied.
- `Qcsf`, `Qppd`, `Qzoom`, `Qctype` and `Qchromacsf` have no effect.
- At `Q ≥ 97`, gray and RGB (with the MCT) input give the same codestream as
  the default.

This is not part of the JPEG Qfactor guideline. On a 1920×1080 crop of an
RGB frame of *Elephants Dream* (5 levels, 64×64 code-blocks, 9/7 + ICT):

| Qfactor | 8-bit, visual | 8-bit, flat | 16-bit, visual | 16-bit, flat |
|---:|---:|---:|---:|---:|
| 30 | 0.144 bpp, 35.6 dB | 0.221 bpp, 38.2 dB | 0.151 bpp, 35.8 dB | 0.226 bpp, 38.5 dB |
| 50 | 0.228 bpp, 37.2 dB | 0.352 bpp, 40.8 dB | 0.245 bpp, 37.6 dB | 0.373 bpp, 41.3 dB |
| 75 | 0.376 bpp, 40.0 dB | 0.557 bpp, 43.8 dB | 0.432 bpp, 41.1 dB | 0.680 bpp, 46.0 dB |
| 90 | 0.677 bpp, 44.9 dB | 0.865 bpp, 48.6 dB | 0.751 bpp, 46.0 dB | 0.945 bpp, 49.9 dB |

At equal rate, flat weights give 0.5–1.2 dB more PSNR and 1–12% lower
1 − SSIM. Both measure pixel fidelity, which flat weights maximize; neither
measures how the image looks at a viewing distance, which the visual weights
are for.

## Precision-scaled Qfactor (`Qscale=bits`)

The guideline's reference step is a fraction of the sample range, so a Qfactor
keeps about the same number of bits at every bit depth: 4.5 bits at
`Qfactor=50`, whether the samples have 8 or 16 bits. That suits content that is
looked at on a display. Users who keep 12- or 16-bit data for its precision
(medical, scientific, camera raw, depth maps) instead get most of the scale
unused: at 16 bits, `Qfactor=99` gives a reference step of about 132 LSBs and
`Qfactor=100` one of 0.71 LSB, with nothing in between.

`Qscale=bits` (opt-in) makes every Qfactor remove the same
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
steps too fine to see get flat (MSE-optimal) weights. The weights are flat from
`Qfactor=33` at 16 bits, 67 at 12 bits and 89 at 10 bits (the guideline's from 97).
With [`Qweights=flat`](#flat-weights-qweightsflat) as well, every subband weight is 1
at every Qfactor and only the step changes; below those values that gives different
steps, above them the same.

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

PSNR and SSIM rise steadily with the Qfactor. On this frame, at 12 and 16 bits,
the two scales stay close at equal rate (PSNR within -0.95 to +1.59 dB, 1 - SSIM
within a factor of 1.45, both taken at the measured points of either curve), so
the option mainly changes which Qfactor gives which rate. Above 8 bits every
Qfactor gives finer steps and larger files than the guideline, also for 10- to
12-bit content meant for viewing, where the guideline's scale may be the better
choice.

The precision-scaled Qfactor is an OpenHTJ2K extension, not part of the
guideline (WG1N101553 v2). Its mapping is fixed: a Qfactor gives the same steps
in later releases.

## Analytic visual weighting (experimental)

The default weights are the guideline's tables, made for one viewing condition
(about 1700 pixels of viewing distance). The analytic models compute the weights
from a contrast sensitivity function (CSF) instead, so they can follow other
viewing conditions:

- `Qcsf=mannos` (Mannos–Sakrison) or `Qcsf=daly` selects a model; `legacy`, the
  default, keeps the tables.
- `Qzoom` is the display magnification. Above 1 (zoomed in), the weights flatten
  toward MSE-optimal steps, because the eye then resolves the finest detail.
  `Qppd` sets the reference pixels per degree (default 72, where `mannos`
  reproduces the guideline's luminance table).
- For sub-sampled YCbCr input, which has no colour transform in the codestream,
  say which component is which with `Qctype=Y,Cb,Cr`. Without it, every
  component gets the luminance CSF.

The models, how a subband maps to a spatial frequency, the chroma weights and
the calibration of the constants are in
[dev/qfactor_design.md](dev/qfactor_design.md#analytic-visual-weighting-experimental).

## CLI reference

All of these are encoder options and require `Qfactor`:

- `Qweights=visual|flat` — visual weights (default **visual**) or none
  (`flat`: every subband weight 1, see [Flat weights](#flat-weights-qweightsflat)).
  With `flat` the options below have no effect.
- `Qcsf=legacy|mannos|daly` — visual-weighting model. Default **legacy**
  (the guideline tables). `mannos` / `daly` are experimental.
- `Qppd=Float` — reference pixels-per-degree at zoom 1.0. Default **72**.
- `Qzoom=Float` — display magnification; `> 1` is zoom-in. Default **1.0**.
- `Qctype=T,T,T` — per-component role hints (`Y|Cb|Cr|generic`) for analytic
  models on no-MCT (e.g. sub-sampled YCbCr) input. Default **generic**.
- `Qchromacsf=chroma|luma` — CSF shape for `Cb`/`Cr` components. Default
  **chroma** (the low-pass chroma CSF).
- `Qscale=guideline|bits` — how the Qfactor scales with the sample precision.
  Default **guideline**; `bits` is the
  [precision-scaled Qfactor](#precision-scaled-qfactor-qscalebits), identical at 8 bits.

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
// Qweights=flat equivalent: 0 = visual weights (default), 1 = flat.
encoder.set_qfactor_weights(1);
// Qscale=bits equivalent: 1 = precision-scaled Qfactor, 0 = guideline (default).
encoder.set_qfactor_scale(1);
encoder.invoke_line_based();
```

The default-constructed weighting is `legacy`, so callers that never call
`set_visual_weighting()` (including the WASM and JPIP bindings) are unchanged.

## Recovering the Qfactor (`estimate_qfactor`)

The Qfactor is **not** signalled in the codestream — `estimate_qfactor` reverse
-engineers it by inverting the
[step-size formula](dev/qfactor_design.md#the-qfactor--step-size-formula) (using
the same shared header, so it tracks the encoder exactly). The visual-weighting model is also
not signalled, so for an analytic encode you must tell the tool which model /
viewing condition was used; a low residual confirms the guess.

```bash
# Legacy encode
estimate_qfactor out.j2c

# Analytic encode — pass the same model / viewing condition used at encode time
estimate_qfactor out.j2c --csf mannos --zoom 2

# Sub-sampled YCbCr analytic encode — pass the same role hints too
estimate_qfactor out420.j2c --csf mannos --ctype Y,Cb,Cr

# Qweights=flat encode
estimate_qfactor out.j2c --weights flat

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
  your guess was right). The same holds for `--weights flat` and `--scale bits`.
- **Bit depth as declared.** Both scales use the bit depth in the SIZ marker,
  which the encoder takes from the input and never changes: a PGM/PPM with
  maxval 4095 is coded as 12-bit. Data with fewer significant bits than its
  input declares, such as 12-bit values in an image declared as 16-bit, gets the
  steps of the declared depth, with the guideline's scale as with `Qscale=bits`.

## How the steps are computed

[dev/qfactor_design.md](dev/qfactor_design.md) gives the formula from a Qfactor
to each subband's step, the guideline's visual-weighting tables (including the
sub-sampled chroma weights changed in 0.26.0), the analytic CSF models and how
their constants were calibrated.

## References

- ISO/IEC JTC 1/SC 29/WG 1 N101553, *"Controlling JPEG 2000 image quality using
  a single parameter (Qfactor) v2.0"*, guideline:
  <https://ds.jpeg.org/documents/jpeg2000/wg1n101553-112-COM-Controlling_JPEG_2000_quality_using_Qfactor_v2.pdf>
- W. Zeng, S. Daly, S. Lei, *"An overview of the visual optimization tools in
  JPEG 2000"*, Signal Processing: Image Communication 17 (2002) 85–104.
- HTJ2K white paper (legacy Qfactor formula):
  <https://htj2k.com/wp-content/uploads/white-paper.pdf>
- Encoder CLI: [`cli_encoder.md`](cli_encoder.md).
