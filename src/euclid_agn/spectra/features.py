"""Stage 0: does this spectrum contain a signal at all, and is it real?

Two questions that must be answered *before* any redshift is attempted, and
answered without reference to any line list or redshift:

1. **Is there a feature?**  The strongest single emission-line-shaped excess
   anywhere in the covered range, found by a matched filter at the object's
   LSF width on every pixel.  This is redshift-agnostic: the same number for
   H-alpha, Pa-beta or a cosmic ray.

2. **Is it real?**  In slitless spectroscopy an unrelated source's emission
   line can land on the extraction in one grism orientation and not another.
   VERIFIED on Q1: the strongest features in the most "confident" wrong
   redshifts were present in a single dither at 9-29 sigma and absent (< 2
   sigma) in the others.  A feature that appears in every evaluable dither is
   the object's; one that appears in one is a neighbour's.

Everything here is recorded on the screening row and drives quality bits.  It
is not used to *select* AGN candidates by host property - it is a data-quality
gate, and the gate's effect is part of the selection function.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from euclid_agn.fit.screen import ScreenSettings, matched_filter, prepare
from euclid_agn.numerics import blas_safe


@dataclass(frozen=True)
class FeatureReport:
    max_delta_chi2: float
    wavelength: float
    n_dithers_evaluable: int
    n_dithers_detected: int
    dither_excess_sigma: tuple[float, ...]

    min_dithers: int = 3

    @property
    def coherent(self) -> bool:
        """Detected in at least ``min_dithers`` dithers, capped at what is evaluable.

        The default of three, at 2.5 sigma per dither, is the knee of the
        purity-completeness front measured against DESI on 218 galaxies
        (`plots/gate_tradeoff.png`): 72 per cent purity among passers at 54 per
        cent completeness on objects with a detectable H-alpha.  Requiring
        three dithers dominates requiring two at every completeness.
        """
        if self.n_dithers_evaluable == 0:
            return False
        need = min(self.min_dithers, self.n_dithers_evaluable)
        return self.n_dithers_detected >= need

    def as_row(self) -> dict[str, float]:
        return {
            "feature_max_dchi2": self.max_delta_chi2,
            "feature_wavelength": self.wavelength,
            "feature_n_dithers_evaluable": float(self.n_dithers_evaluable),
            "feature_n_dithers_detected": float(self.n_dithers_detected),
            "feature_coherent": float(self.coherent),
        }


@blas_safe
def strongest_line_feature(projected, sigma_kms: float = 150.0) -> tuple[float, float]:
    """Best single-line matched-filter statistic over every pixel position.

    Returns ``(delta_chi2, wavelength)``; ``(0, nan)`` if nothing positive.
    """
    from euclid_agn.spectra.lsf import effective_sigma, sigma_kms_to_angstrom

    best, best_w = 0.0, float("nan")
    for centre in projected.wavelength[3:-3]:
        width = effective_sigma(sigma_kms_to_angstrom(sigma_kms, centre), projected.lsf_sigma)
        stat, amp = matched_filter(projected, projected.line_column(centre, width)[:, None])
        if stat > best and amp[0] > 0:
            best, best_w = float(stat), float(centre)
    return best, best_w


def dither_excess(dither, wavelength: float, core_pixels: int = 2, window_pixels: int = 12) -> float | None:
    """Peak excess over the local median at ``wavelength``, in local sigma.

    ``None`` when too few usable pixels surround the position to say anything.
    """
    usable = dither.usable()
    index = int(np.argmin(np.abs(dither.wavelength - wavelength)))
    core = slice(max(index - core_pixels, 0), index + core_pixels + 1)
    window = slice(max(index - window_pixels, 0), index + window_pixels + 1)
    if usable[core].sum() < 3 or usable[window].sum() < 10:
        return None
    baseline = float(np.median(dither.flux[window][usable[window]]))
    sigma = float(np.median(np.sqrt(dither.variance[window][usable[window]])))
    if not np.isfinite(sigma) or sigma <= 0:
        return None
    return float((np.max(dither.flux[core][usable[core]]) - baseline) / sigma)


def feature_report(
    observation,
    settings: ScreenSettings = ScreenSettings(),
    detection_sigma: float | None = None,
    min_dithers: int | None = None,
) -> FeatureReport:
    """Strongest feature of the combined spectrum and its presence per dither."""
    detection_sigma = settings.dither_sigma if detection_sigma is None else detection_sigma
    min_dithers = settings.min_coherent_dithers if min_dithers is None else min_dithers
    projected = prepare(observation.combined, settings)
    if projected is None:
        return FeatureReport(0.0, float("nan"), 0, 0, (), min_dithers)
    stat, wavelength = strongest_line_feature(projected)
    excesses = []
    if np.isfinite(wavelength):
        for dither in observation.dithers:
            excesses.append(dither_excess(dither, wavelength))
    evaluable = [e for e in excesses if e is not None]
    detected = sum(1 for e in evaluable if e > detection_sigma)
    return FeatureReport(
        max_delta_chi2=stat,
        wavelength=wavelength,
        n_dithers_evaluable=len(evaluable),
        n_dithers_detected=detected,
        dither_excess_sigma=tuple(float(e) if e is not None else float("nan") for e in excesses),
        min_dithers=min_dithers,
    )
