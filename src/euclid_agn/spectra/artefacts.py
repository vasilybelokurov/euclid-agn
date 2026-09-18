"""Detection of decontamination artefacts in slitless spectra.

Near a bright extended neighbour the SIR decontamination step can remove far
too much flux over the wavelength range where the neighbour's trace crosses
the object's own.  The result is a rectangular deficit in the middle of the
band with normal flux on both sides - nothing a stellar population or a galaxy
can produce, and enough to drive a template fit to a nonsense redshift.

Measured in Q1 around NGC 1527 (D25 = 4.5'): 51 % of spectra within 3' of the
galaxy show a deficit deeper than a factor of two, against 0 % at 3-6', 1 % at
6-10' and 0 % of 241 spectra in blank control fields.  The same radius is where
velocity fits rail at the search boundary, so this one artefact accounts for
the inner-region failures.

The test is deliberately shape-based and template-free: a smooth continuum is
compared with its own median level, and a contiguous depressed stretch bounded
by normal flux on both sides is flagged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter


@dataclass(frozen=True)
class TroughReport:
    """A contiguous flux deficit that no astrophysical continuum explains."""

    flagged: bool
    start_angstrom: float
    stop_angstrom: float
    depth: float  # median flux inside the trough / median outside it
    width_angstrom: float
    pixel_fraction: float

    def as_row(self, prefix: str = "trough_") -> dict:
        return {f"{prefix}flagged": self.flagged, f"{prefix}start": self.start_angstrom,
                f"{prefix}stop": self.stop_angstrom, f"{prefix}depth": self.depth,
                f"{prefix}width": self.width_angstrom, f"{prefix}pixel_fraction": self.pixel_fraction}


def continuum_trough(spectrum, depth: float = 0.5, min_width_angstrom: float = 500.0,
                     smooth_pixels: int = 21, edge_margin_pixels: int = 10) -> TroughReport:
    """Largest contiguous stretch where the smoothed flux falls below ``depth`` of its level.

    ``depth`` 0.5 means "at least a factor of two down".  The stretch must be
    at least ``min_width_angstrom`` wide and must not touch either end of the
    usable range (a deficit at the edge is an extraction roll-off, not this
    artefact).
    """
    ok = spectrum.usable()
    empty = TroughReport(False, float("nan"), float("nan"), float("nan"), 0.0, 0.0)
    if ok.sum() < 4 * smooth_pixels:
        return empty
    wavelength = spectrum.wavelength[ok]
    flux = median_filter(spectrum.flux[ok], size=smooth_pixels, mode="nearest")
    level = float(np.nanmedian(flux))
    if not np.isfinite(level) or level <= 0:
        return empty
    low = flux < depth * level
    if not low.any():
        return empty
    edges = np.flatnonzero(np.diff(np.concatenate([[0], low.astype(int), [0]])))
    best = empty
    for start, stop in zip(edges[::2], edges[1::2], strict=True):
        if start < edge_margin_pixels or stop > low.size - edge_margin_pixels:
            continue  # touches the edge: roll-off, not a trough
        width = float(wavelength[stop - 1] - wavelength[start])
        if width < min_width_angstrom or width <= best.width_angstrom:
            continue
        outside = np.ones(low.size, bool)
        outside[start:stop] = False
        ratio = float(np.nanmedian(flux[start:stop]) / np.nanmedian(flux[outside]))
        best = TroughReport(True, float(wavelength[start]), float(wavelength[stop - 1]), ratio, width,
                            float((stop - start) / low.size))
    return best


__all__ = ["TroughReport", "continuum_trough"]
