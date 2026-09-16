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

## Statistics

`Δχ²` between the no-BLR and BLR models is a statistic, not a significance: the
broad flux is bounded at zero and many hypotheses are searched. Detection
significance is calibrated empirically on null spectra and injection/recovery.
Anything not probabilistically calibrated is called a *score*.
