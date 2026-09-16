# Science model

## Generative model

For dither *d*:

```
f_d(λ) = C_d(λ) + K_d ⊗ [ N(λ | z, θ_NLR) + B(λ | z, θ_BLR) ] + ε_d
```

* `C_d` local continuum (regularised B-spline; **not** an SSP decomposition —
  the red grism covers only 1.25–1.85 µm and adding a population model before
  residuals demand it is unjustified complexity);
* `N` narrow lines sharing one non-parametric velocity profile `P_NLR(v)` with
  non-negative amplitudes and smoothness regularisation — the central
  Chilingarian-inspired component;
* `B` broad components, allowed **only** on permitted transitions;
* `K_d` effective Euclid LSF for that dither.

At fixed `(z, σ_BLR, Δv_BLR)` the amplitudes and profile coefficients solve as a
constrained regularised linear problem; only the small non-linear space is
scanned. That separation is what makes the problem tractable at Euclid scale.

## Line-spread function

`LSF_SIG` from the product is the baseline effective kernel. SIR already applies
a virtual-slit transformation and a morphology-dependent LSF estimate, so the
released spectrum must **not** be convolved by the source profile a second time.

Measured over tile 102160339 (1000 objects): `LSF_SIG` percentiles
[1, 5, 25, 50, 75, 95, 99] = [12.3, 12.9, 13.3, 13.7, 15.6, 54.1, 74.1] Å,
i.e. R ≈ 465 at 1.5 µm for compact sources falling to R ≈ 90 for extended ones.
A line width is always an *intrinsic* width inferred after this kernel is in the
forward model. No universal minimum FWHM is adopted as truth.

## Sampling

Bins are 13.4 Å wide against a median LSF σ of 13.7 Å: about one σ per pixel.
Model line profiles are therefore integrated across each pixel
(`gaussian_pixel_integral`), not sampled at bin centres.

## Redshift hypotheses

SPE redshifts are hypotheses, not truth. Hypotheses come from high-quality SPE
solutions, PHZ probabilities, detected-peak line identifications, and a blind
line-family scan. The winning hypothesis records its origin.

## Wavelength coverage and branches

For 12500–18500 Å (vacuum rest wavelengths):

| feature | z range |
|---|---|
| Pa-β 12822 | 0.00–0.44 |
| He I 10833 | 0.15–0.71 |
| Hα 6564.6 | 0.90–1.82 |
| [S II] 6718 | 0.86–1.75 |
| Hβ 4862.7 | 1.57–2.81 |
| [O III] 5008 | 1.50–2.69 |
| [O II] 3727 | 2.35–3.96 |
| Mg II 2796 | 3.47–5.62 |

The classical [N II] BPT set is simultaneously visible only over about
1.57 < z < 1.81, so BPT can never be a compulsory selection gate. Three
branches stay separate: BLR (primary), NLR diagnostics where coverage allows,
and external evidence as validation metadata only.

## What a broad line can be and still be measured

A very broad Gaussian is degenerate with continuum curvature. The degeneracy is
measurable: project the whitened broad-line basis vector orthogonal to the
continuum span and see what fraction of its norm survives. Measured on the real
Q1 grid (448 fitted pixels):

| σ (FWHM) km/s | 6 knots | 12 knots | 25 knots | 50 knots |
|---|---|---|---|---|
| 300 (706) | 0.95 | 0.92 | 0.86 | 0.73 |
| 600 (1413) | 0.93 | 0.86 | 0.78 | 0.56 |
| 1200 (2826) | 0.86 | 0.74 | 0.58 | 0.23 |
| 2500 (5887) | 0.70 | 0.47 | 0.21 | 0.01 |
| 5000 (11774) | 0.42 | 0.13 | 0.01 | 0.00 |
| 8000 (18839) | 0.19 | 0.02 | 0.00 | 0.00 |

With a 12-knot continuum, widths beyond σ ≈ 2500 km/s (FWHM ≈ 5900 km/s) carry
almost no information independent of the continuum. A substantial part of the
classical BLR width range is therefore *intrinsically* degenerate with the
continuum in the Euclid red grism — a property of the data, not of the code.

Widths failing `min_broad_orthogonality` are excluded from the scan, and the
orthogonality of the best surviving width is recorded on every row as a
selection-function axis. The first pilot run without this guard had every
top-ranked candidate railed at the widest width in the grid: those were the
continuum model being repaired by a line, not detections.

## Line identification is degenerate

Hα at z = 1.2 falls at 14442 Å and Pa-β at z = 0.1264 at 14458 Å — 1.15 pixels
apart. Ranking hypotheses by broad-line gain alone puts the wrong
identification first. Redshift is therefore chosen using all the line evidence
(narrow + broad), while the broad gain remains the AGN statistic; both rankings
are refined so a BLR with no narrow lines is not discarded. Each row records how
much better the winner is than the best hypothesis of a *different* line system.

## Statistics

`Δχ²` between the no-BLR and BLR models is a statistic, not a significance: the
broad flux is bounded at zero and many hypotheses are searched. Detection
significance is calibrated empirically on null spectra and injection/recovery.
Anything not probabilistically calibrated is called a *score*.

Measured on 934 real Q1 spectra, the reported variances are too small by ~1.45×
and adjacent pixels are correlated at ρ₁ = +0.18, so a naive Δχ² overstates the
evidence by a factor of about 2 (bracketed 1.65–2.33). The per-spectrum
inflation η is carried on every screening row and
`delta_chi2_effective = Δχ² / η²` is reported alongside the raw statistic.
Neither is a significance.
