"""Per-pixel dither coherence: reject what the exposures disagree about.

Slitless spectra are contaminated by neighbours' orders and by detector
defects that differ between the four dithers, and the archive combination does
not always reject them: on a bright low-z galaxy (object 2712466074669129632)
the four dithers read 14, 15, 10 and 17 x 1e-17 in 15900-16800 A, and 0, 27,
12 and 22 in 17500-17800 A, while the combined pixels carry no mask bit.  A
template fit follows such structure and lands at the wrong redshift.

The statistic is, per pixel, the reduced chi-squared of the usable dither
values about their inverse-variance mean.  Under the archive noise model its
median is ~ eta^2 (the known noise inflation, ~2); pixels far above that are
incoherent.  ``threshold`` is a multiple of the object's own median so an
inflated variance scale does not flag everything.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CoherenceReport:
    reduced_chi2: np.ndarray  # per pixel; NaN where fewer than 2 usable dithers
    n_usable: np.ndarray  # usable dithers per pixel
    bad: np.ndarray  # pixels to reject
    median_reduced_chi2: float
    threshold_used: float

    @property
    def n_bad(self) -> int:
        return int(self.bad.sum())


def dither_coherence(observation, threshold: float = 5.0, min_dithers: int = 2, grow: int = 1,
                     absolute_floor: float = 8.0) -> CoherenceReport | None:
    """Flag pixels whose dithers disagree beyond noise.

    ``threshold`` multiplies the object's median reduced chi-squared, with an
    ``absolute_floor`` so a very coherent object is not over-flagged: with 3
    degrees of freedom (4 dithers) pure noise exceeds a reduced chi-squared of
    8 with probability 2.5e-5 per pixel, i.e. ~0.01 pixels per spectrum, while
    a floor of 4 would flag 0.7 % of clean pixels;
    ``grow`` dilates each flagged pixel by that many neighbours because
    contamination edges bleed into adjacent bins through the LSF.  Pixels the
    combined spectrum has but fewer than ``min_dithers`` usable dithers support
    are also rejected: nothing can vouch for them.
    """
    dithers = list(observation.dithers)
    if len(dithers) < 2:
        return None
    wavelength = observation.combined.wavelength
    n = wavelength.size
    flux = np.full((len(dithers), n), np.nan)
    ivar = np.zeros((len(dithers), n))
    for i, d in enumerate(dithers):
        if d.wavelength.size != n or not np.allclose(d.wavelength, wavelength):
            return None
        ok = d.usable()
        flux[i, ok] = d.flux[ok]
        ivar[i, ok] = 1.0 / d.variance[ok]
    n_usable = (ivar > 0).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nansum(np.where(ivar > 0, flux * ivar, 0.0), axis=0) / ivar.sum(axis=0)
        chi2 = np.nansum(np.where(ivar > 0, (flux - mean) ** 2 * ivar, 0.0), axis=0)
        reduced = np.where(n_usable >= 2, chi2 / np.maximum(n_usable - 1, 1), np.nan)
    median = float(np.nanmedian(reduced)) if np.isfinite(reduced).any() else float("nan")
    cut = max(threshold * median, absolute_floor) if np.isfinite(median) else absolute_floor
    bad = (reduced > cut) | (n_usable < min_dithers)
    bad &= observation.combined.usable()  # only pixels the fit would otherwise use
    if grow > 0 and bad.any():
        kernel = np.ones(2 * grow + 1, dtype=int)
        bad = np.convolve(bad.astype(int), kernel, mode="same") > 0
    return CoherenceReport(reduced, n_usable, bad, median, float(cut))


def apply_coherence_mask(observation, **kwargs):
    """The combined spectrum with incoherent pixels set NOT_USE (or unchanged if no dithers)."""
    report = dither_coherence(observation, **kwargs)
    if report is None or not report.bad.any():
        return observation.combined, report
    return observation.combined.with_extra_mask(report.bad, reason="coherence"), report


__all__ = ["CoherenceReport", "dither_coherence", "apply_coherence_mask"]
