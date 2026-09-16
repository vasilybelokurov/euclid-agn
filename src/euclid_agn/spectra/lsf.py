"""Effective line-spread function and pixel-integrated convolution.

Two Euclid-specific facts drive this module.

1. In slitless spectroscopy the effective LSF depends on the projected extent of
   the source.  SIR already performs a "virtual slit" transformation and
   estimates a per-spectrum effective Gaussian LSF, exposed as ``LSF_SIG``
   (Angstrom).  We take that as the baseline kernel and do **not** re-convolve
   by the source morphology: that would double-count SIR's own correction.
   VERIFIED distribution over the 1000 objects of tile 102160339::

       LSF_SIG percentiles [1, 5, 25, 50, 75, 95, 99] =
           [12.3, 12.9, 13.3, 13.7, 15.6, 54.1, 74.1] Angstrom

   i.e. R = lambda / (2.355 sigma) ~ 465 at 1.5 micron for compact sources,
   with a long tail of extended sources reaching R ~ 90.

2. The stored bin width is 13.4 Angstrom, which is ~1.0 LSF sigma for compact
   sources.  The model is therefore **integrated across each pixel** rather
   than evaluated at bin centres; sampling at centres biases narrow-line
   amplitudes low by several per cent at this sampling.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erf

from euclid_agn.constants import C_KMS

SQRT2 = np.sqrt(2.0)
FWHM_OVER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


def sigma_kms_to_angstrom(sigma_kms: float, wavelength: float) -> float:
    """Velocity dispersion (km/s) -> wavelength dispersion (Angstrom) at ``wavelength``."""
    return float(sigma_kms) * float(wavelength) / C_KMS


def sigma_angstrom_to_kms(sigma_angstrom: float, wavelength: float) -> float:
    """Wavelength dispersion (Angstrom) -> velocity dispersion (km/s) at ``wavelength``."""
    return float(sigma_angstrom) * C_KMS / float(wavelength)


def resolving_power(lsf_sigma_angstrom: float, wavelength: float) -> float:
    """R = lambda / FWHM for a Gaussian LSF."""
    return float(wavelength) / (FWHM_OVER_SIGMA * float(lsf_sigma_angstrom))


def effective_sigma(intrinsic_sigma_angstrom: float, lsf_sigma_angstrom: float) -> float:
    """Observed Gaussian width of an intrinsically Gaussian line.

    Gaussians add in quadrature, so an intrinsic width is only ever recovered
    *after* the instrumental term is included in the forward model.
    """
    return float(np.hypot(intrinsic_sigma_angstrom, lsf_sigma_angstrom))


def pixel_edges(wavelength: np.ndarray, bin_width: float | None = None) -> np.ndarray:
    """Bin edges for a (possibly non-uniform) wavelength grid of bin centres."""
    wavelength = np.asarray(wavelength, dtype=np.float64)
    if wavelength.size == 0:
        return np.empty(0)
    if wavelength.size == 1:
        half = 0.5 * (bin_width if bin_width else 1.0)
        return np.array([wavelength[0] - half, wavelength[0] + half])
    mid = 0.5 * (wavelength[1:] + wavelength[:-1])
    first = wavelength[0] - (mid[0] - wavelength[0])
    last = wavelength[-1] + (wavelength[-1] - mid[-1])
    return np.concatenate([[first], mid, [last]])


def gaussian_pixel_integral(
    wavelength: np.ndarray,
    centre: float,
    sigma: float,
    bin_width: float | None = None,
) -> np.ndarray:
    """Mean flux density per pixel of a unit-flux Gaussian emission line.

    Returns the pixel-averaged value of a Gaussian of unit *integrated flux*,
    i.e. ``sum(result * pixel_width) == 1`` up to truncation at the grid edges.

    Parameters
    ----------
    wavelength : ndarray
        Bin centres, Angstrom.
    centre : float
        Observed line centre, Angstrom.
    sigma : float
        Total (intrinsic + LSF) Gaussian width, Angstrom.
    """
    edges = pixel_edges(wavelength, bin_width)
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    z = (edges - centre) / (SQRT2 * sigma)
    cdf = 0.5 * (1.0 + erf(z))
    widths = np.diff(edges)
    return np.diff(cdf) / widths


def gaussian_kernel(sigma_angstrom: float, bin_width: float, n_sigma: float = 5.0) -> np.ndarray:
    """Normalised, pixel-integrated Gaussian kernel on a uniform grid.

    Used to convolve a model sampled on the same uniform grid.  The kernel sums
    to one, so convolution conserves integrated flux (up to truncation).
    """
    if sigma_angstrom <= 0 or bin_width <= 0:
        raise ValueError("sigma and bin_width must be positive")
    half = max(1, int(np.ceil(n_sigma * sigma_angstrom / bin_width)))
    offsets = np.arange(-half, half + 1, dtype=np.float64) * bin_width
    edges = np.concatenate([offsets - 0.5 * bin_width, [offsets[-1] + 0.5 * bin_width]])
    cdf = 0.5 * (1.0 + erf(edges / (SQRT2 * sigma_angstrom)))
    kernel = np.diff(cdf)
    total = kernel.sum()
    if total <= 0:  # pragma: no cover - defensive
        raise ValueError("degenerate kernel")
    return kernel / total


def convolve_lsf(
    model: np.ndarray, sigma_angstrom: float, bin_width: float, n_sigma: float = 5.0
) -> np.ndarray:
    """Convolve a uniformly sampled model with the effective LSF.

    Edges are handled by reflection, which preserves a locally flat continuum
    instead of pulling it towards zero.
    """
    kernel = gaussian_kernel(sigma_angstrom, bin_width, n_sigma=n_sigma)
    pad = len(kernel) // 2
    if pad == 0:
        return np.asarray(model, dtype=np.float64).copy()
    padded = np.pad(np.asarray(model, dtype=np.float64), pad, mode="reflect")
    return np.convolve(padded, kernel, mode="valid")
