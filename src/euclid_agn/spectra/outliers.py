"""Robust rejection of outlier pixels that the archive mask leaves in.

Why
---
The six spectra with the largest Stage-1 identification statistics in the
blind-recovery experiment were all driven by pixels the SIR mask does not flag:
single-pixel spikes at +20 sigma, deep single-pixel dips, and unmasked edges.
A Gaussian matched filter awards Delta chi-squared ~ 400 to *any* template
that can place a line on a +20-sigma pixel, so those pixels decide the redshift
regardless of the real lines.  The same pixels explain the SPE "detections" with
FWHM 90-140 A that sit on flat spectra.

What counts as an outlier
-------------------------
Two things, both required:

* the pixel deviates from a robust local baseline (running median) by more
  than ``threshold`` robust sigma of the *scaled* residual, and
* the deviation is narrower than the instrument can make it - a real line at
  R ~ 450 has FWHM >= 2.4 pixels, so a feature confined to ``max_width`` = 2
  adjacent pixels is not a line.

Runs of three or more consecutive high pixels are left alone: they may be a
line, a broad feature, or a genuine problem, but they are for the fit and its
goodness-of-fit flag to judge, not for a pre-filter to erase.

Everything found is *recorded* (a count and a mask), never silently applied to
the archive arrays.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter


def robust_sigma(values: np.ndarray) -> float:
    """1.4826 * MAD, with a floor so an all-zero input cannot divide by zero."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    mad = np.median(np.abs(values - np.median(values)))
    return float(max(1.4826 * mad, 1e-300))


def run_lengths(flags: np.ndarray) -> np.ndarray:
    """Length of the run of consecutive True values each True pixel belongs to."""
    flags = np.asarray(flags, dtype=bool)
    out = np.zeros(flags.size, dtype=int)
    if not flags.any():
        return out
    padded = np.concatenate([[False], flags, [False]])
    edges = np.flatnonzero(np.diff(padded.astype(int)))
    for start, stop in zip(edges[::2], edges[1::2], strict=True):
        out[start:stop] = stop - start
    return out


def isolated_outliers(
    flux: np.ndarray,
    variance: np.ndarray,
    usable: np.ndarray,
    threshold: float = 5.0,
    max_width: int = 2,
    window: int = 15,
    lsf_sigma_pixels: float = 1.0,
    neighbour_fraction: float = 0.5,
) -> np.ndarray:
    """Boolean mask of spike-like outlier pixels among the usable ones.

    A pixel run is rejected only if it is *both* far from the local baseline
    and shaped unlike anything the instrument can produce.  The shape test is
    the one that matters: the first version rejected any run of at most
    ``max_width`` high pixels, and on real Q1 spectra that removed the emission
    lines it was meant to protect, because a line at S/N 10-20 puts only its
    top one or two pixels above 5 sigma.

    For a Gaussian line of width ``lsf_sigma_pixels`` (at least ~1 pixel in
    Q1), the pixels just outside a run of width ``w`` centred on the line lie
    at offset ``(w + 1) / 2`` pixels and carry a fraction
    ``exp(-offset^2 / 2 sigma^2)`` of the peak - 0.61 for a one-pixel run and
    0.32 for a two-pixel run at sigma = 1 px.  A run whose outer neighbours are
    both below ``neighbour_fraction`` times that expectation is a spike.

    Positive and negative deviations are both tested: a -20 sigma dip is not an
    emission line, but it steers the continuum fit and every statistic built on
    the residual.
    """
    flux = np.asarray(flux, dtype=np.float64)
    variance = np.asarray(variance, dtype=np.float64)
    usable = np.asarray(usable, dtype=bool)
    out = np.zeros(flux.size, dtype=bool)
    index = np.flatnonzero(usable)
    if index.size < window:
        return out
    values = flux[index]
    baseline = median_filter(values, size=window, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        normalised = (values - baseline) / np.sqrt(variance[index])
    normalised[~np.isfinite(normalised)] = 0.0
    sigma = robust_sigma(normalised)
    scaled = normalised / sigma
    high = np.abs(scaled) > threshold
    if not high.any():
        return out
    lengths = run_lengths(high)
    sigma_px = max(float(lsf_sigma_pixels), 0.5)
    padded = np.concatenate([[False], high, [False]])
    edges = np.flatnonzero(np.diff(padded.astype(int)))
    for start, stop in zip(edges[::2], edges[1::2], strict=True):
        width = stop - start
        if width > max_width:
            continue
        run = scaled[start:stop]
        peak = float(np.max(np.abs(run)))
        sign = np.sign(run[int(np.argmax(np.abs(run)))])
        offset = 0.5 * (width + 1)
        expected = float(np.exp(-0.5 * (offset / sigma_px) ** 2))
        floor = neighbour_fraction * expected * peak
        left = sign * scaled[start - 1] if start > 0 else 0.0
        right = sign * scaled[stop] if stop < scaled.size else 0.0
        if left < floor and right < floor:
            out[index[start:stop]] = True
    del lengths
    return out


def outlier_summary(outliers: np.ndarray, usable: np.ndarray) -> dict[str, float]:
    n_usable = int(np.count_nonzero(usable))
    n_out = int(np.count_nonzero(outliers))
    return {
        "n_outlier_pixels": float(n_out),
        "outlier_fraction": float(n_out) / n_usable if n_usable else float("nan"),
    }
