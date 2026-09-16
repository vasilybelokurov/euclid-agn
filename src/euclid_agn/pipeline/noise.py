"""Empirical audit of the Q1 noise model.

Q1 supplies a per-pixel variance but no covariance matrix, and the SIR pipeline
resamples dispersed 2D data onto a common 1D grid, which correlates
neighbouring pixels.  Two questions therefore have to be answered from the data
before any chi-squared difference is turned into a significance:

1. **Is the reported variance right?**  If the normalised residual
   ``r = (f - model) / sqrt(VAR)`` has standard deviation far from 1, every
   chi-squared in the pipeline is scaled by a constant nobody chose.

2. **Are neighbouring pixels independent?**  If ``r`` is autocorrelated, the
   effective number of independent pixels is smaller than the nominal one, and
   a naive Delta chi-squared overstates the evidence for any added component.

The measurement is deliberately model-light: a smooth B-spline continuum is
removed, pixels deviating strongly from it are clipped out so real emission
lines do not masquerade as noise, and what remains is characterised by its
standard deviation and autocorrelation.

The number to carry forward is the *noise-inflation factor*

.. math:: \\eta = \\sigma_r \\sqrt{1 + 2\\sum_{k\\ge1} \\rho_k}

which converts a nominal chi-squared into one that accounts for both a
mis-scaled variance and correlated pixels.  It is an estimate, not a
replacement for empirical calibration on null spectra; it says how far from
"independent pixels with correct variances" the data actually are.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from euclid_agn.constants import RGS_SCIENCE_WMAX_ANGSTROM, RGS_SCIENCE_WMIN_ANGSTROM
from euclid_agn.fit.linear import LinearProblem, solve
from euclid_agn.io.sir import open_sir_file
from euclid_agn.numerics import blas_safe
from euclid_agn.spectra.continuum import continuum_block

log = logging.getLogger(__name__)

#: Residuals beyond this many sigma are treated as signal, not noise.
CLIP_SIGMA = 4.0

#: Lags at which the autocorrelation is reported.
DEFAULT_LAGS: tuple[int, ...] = (1, 2, 3, 4, 5)


@dataclass(frozen=True)
class NoiseAudit:
    """Noise diagnostics for one spectrum."""

    object_id: int
    n_used: int
    residual_sigma: float
    autocorrelation: dict[int, float]
    inflation: float
    clipped_fraction: float
    metadata: dict = field(default_factory=dict)

    def as_row(self) -> dict[str, float | int]:
        row: dict[str, float | int] = {
            "object_id": self.object_id,
            "n_used": self.n_used,
            "residual_sigma": self.residual_sigma,
            "inflation": self.inflation,
            "clipped_fraction": self.clipped_fraction,
        }
        for lag, value in self.autocorrelation.items():
            row[f"acf_lag{lag}"] = value
        row.update(self.metadata)
        return row


@blas_safe
def autocorrelation(residual: np.ndarray, lags: tuple[int, ...] = DEFAULT_LAGS) -> dict[int, float]:
    """Sample autocorrelation of a zero-mean series at the requested lags.

    Pixels excluded by clipping or masking must already be removed *and* the
    series must be contiguous in wavelength, otherwise a lag of one pixel is
    not a lag of one pixel.  :func:`audit_spectrum` enforces this by working on
    the longest contiguous run of usable pixels.
    """
    residual = np.asarray(residual, dtype=np.float64)
    residual = residual - residual.mean()
    denominator = float(residual @ residual)
    if denominator <= 0:
        return dict.fromkeys(lags, float("nan"))
    return {
        lag: float(residual[lag:] @ residual[:-lag] / denominator) if lag < residual.size else float("nan")
        for lag in lags
    }


def inflation_factor(residual_sigma: float, acf: dict[int, float]) -> float:
    """Factor by which a nominal chi-squared understates the true scatter.

    ``sigma_r * sqrt(1 + 2 * sum_k rho_k)`` over the positive lags, with the sum
    truncated at the first non-positive lag so that noise in the tail of the
    autocorrelation does not deflate the estimate.
    """
    total = 0.0
    for lag in sorted(acf):
        value = acf[lag]
        if not np.isfinite(value) or value <= 0:
            break
        total += value
    return float(residual_sigma * np.sqrt(max(1.0 + 2.0 * total, 0.0)))


def longest_contiguous(indices: np.ndarray) -> np.ndarray:
    """Longest run of consecutive integers in a sorted index array."""
    if indices.size == 0:
        return indices
    breaks = np.flatnonzero(np.diff(indices) != 1)
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks + 1, [indices.size]])
    lengths = ends - starts
    best = int(np.argmax(lengths))
    return indices[starts[best] : ends[best]]


@blas_safe
def audit_spectrum(
    spectrum,
    object_id: int = -1,
    n_knots: int = 12,
    clip_sigma: float = CLIP_SIGMA,
    lags: tuple[int, ...] = DEFAULT_LAGS,
    wavelength_min: float = RGS_SCIENCE_WMIN_ANGSTROM,
    wavelength_max: float = RGS_SCIENCE_WMAX_ANGSTROM,
    min_pixels: int = 100,
) -> NoiseAudit | None:
    """Measure residual scatter and autocorrelation for one spectrum.

    Returns ``None`` when the spectrum has too few contiguous usable pixels in
    the science window to say anything.
    """
    usable = spectrum.usable()
    window = (spectrum.wavelength >= wavelength_min) & (spectrum.wavelength <= wavelength_max)
    indices = longest_contiguous(np.flatnonzero(usable & window))
    if indices.size < min_pixels:
        return None

    wavelength = spectrum.wavelength[indices]
    flux = spectrum.flux[indices]
    sigma = np.sqrt(spectrum.variance[indices])

    design, penalty, names = continuum_block(wavelength, n_knots=n_knots)
    solution = solve(
        LinearProblem(
            design=design,
            data=flux,
            variance=sigma**2,
            regularisation=penalty,
            regularisation_weight=0.1,
            names=names,
        )
    )
    residual = (flux - solution.model) / sigma

    # Clip emission lines and outliers, then re-take the longest clean run so
    # the lag structure stays meaningful.
    robust = 1.4826 * np.median(np.abs(residual - np.median(residual)))
    keep = np.abs(residual - np.median(residual)) < clip_sigma * max(robust, 1e-12)
    clipped_fraction = float(1.0 - keep.mean())
    clean_indices = longest_contiguous(np.flatnonzero(keep))
    if clean_indices.size < min_pixels:
        return None
    clean = residual[clean_indices]

    sigma_r = float(np.std(clean))
    acf = autocorrelation(clean, lags)
    return NoiseAudit(
        object_id=object_id,
        n_used=int(clean.size),
        residual_sigma=sigma_r,
        autocorrelation=acf,
        inflation=inflation_factor(sigma_r, acf),
        clipped_fraction=clipped_fraction,
        metadata={
            "lsf_sigma": float(spectrum.lsf_sigma),
            "median_snr": float(np.median(flux / sigma)),
        },
    )


def audit_file(
    path: str,
    filesystem=None,
    max_objects: int | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Run :func:`audit_spectrum` over every object in one SIR file."""
    rows = []
    with open_sir_file(path, filesystem=filesystem) as sir:
        groups = list(sir.groups().values())
        if max_objects is not None:
            groups = groups[:max_objects]
        for group in groups:
            combined = sir.read_combined(group)
            audit = audit_spectrum(combined, object_id=group.object_id, **kwargs)
            if audit is not None:
                row = audit.as_row()
                row["tile_id"] = sir.tile_id
                rows.append(row)
    return pd.DataFrame(rows)


def summarise_audit(table: pd.DataFrame, lags: tuple[int, ...] = DEFAULT_LAGS) -> dict[str, float]:
    """Headline numbers: is the variance right, and are pixels independent?"""
    if table.empty:
        return {"n_spectra": 0}
    out: dict[str, float] = {
        "n_spectra": int(len(table)),
        "residual_sigma_median": float(np.nanmedian(table["residual_sigma"])),
        "residual_sigma_p16": float(np.nanpercentile(table["residual_sigma"], 16)),
        "residual_sigma_p84": float(np.nanpercentile(table["residual_sigma"], 84)),
        "inflation_median": float(np.nanmedian(table["inflation"])),
        "inflation_p84": float(np.nanpercentile(table["inflation"], 84)),
        "clipped_fraction_median": float(np.nanmedian(table["clipped_fraction"])),
    }
    for lag in lags:
        column = f"acf_lag{lag}"
        if column in table:
            out[f"acf_lag{lag}_median"] = float(np.nanmedian(table[column]))
    return out


# --- method bias -----------------------------------------------------------
# Removing a continuum with p free parameters from n pixels absorbs part of the
# noise, so the audit under-reports the scatter and induces a small negative
# autocorrelation even on perfectly white data.  The bias is measured, not
# assumed, by running the identical procedure on simulated spectra whose noise
# is white and whose variances are exact.


@dataclass(frozen=True)
class MethodBias:
    """Audit statistics recovered from data known to be ideal."""

    n_knots: int
    residual_sigma: float
    acf_lag1: float
    inflation: float
    n_realisations: int


def measure_method_bias(
    n_knots: int = 25,
    n_realisations: int = 60,
    seed: int = 20260916,
    sigma: float = 1.0e-18,
    continuum: float = 1.0e-17,
    slope_per_micron: float = 0.5,
    **kwargs,
) -> MethodBias:
    """Run the audit on white noise with exact variances.

    Any departure of the returned values from ``sigma_r = 1``, ``rho_1 = 0`` is
    the method's own bias at that continuum flexibility and must be divided out
    before a statement is made about the real data.
    """
    from euclid_agn.spectra.types import Spectrum1D
    from euclid_agn.validation.simulator import sir_wavelength_grid

    wavelength = sir_wavelength_grid()
    span = wavelength[-1] - wavelength[0]
    sigmas, acfs, inflations = [], [], []
    for offset in range(n_realisations):
        rng = np.random.default_rng(seed + offset)
        level = continuum * (1.0 + slope_per_micron * (wavelength - wavelength[0]) / span)
        spectrum = Spectrum1D(
            wavelength=wavelength,
            flux=level + rng.normal(0.0, sigma, wavelength.size),
            variance=np.full(wavelength.size, sigma**2),
            mask=np.zeros(wavelength.size, dtype=int),
            quality=np.ones(wavelength.size),
            lsf_sigma=13.7,
            bin_width=13.4,
        )
        audit = audit_spectrum(spectrum, n_knots=n_knots, **kwargs)
        if audit is None:  # pragma: no cover - the grid is always long enough
            continue
        sigmas.append(audit.residual_sigma)
        acfs.append(audit.autocorrelation[1])
        inflations.append(audit.inflation)
    return MethodBias(
        n_knots=n_knots,
        residual_sigma=float(np.median(sigmas)),
        acf_lag1=float(np.median(acfs)),
        inflation=float(np.median(inflations)),
        n_realisations=len(sigmas),
    )


def corrected_summary(
    table: pd.DataFrame, bias: MethodBias, lags: tuple[int, ...] = DEFAULT_LAGS
) -> dict[str, float]:
    """Audit summary with the method bias divided out.

    ``residual_sigma_corrected`` is the measured scatter divided by the scatter
    the same procedure recovers from ideal data; ``acf_lag1_corrected``
    subtracts the bias in the same way.  The corrected inflation factor is what
    a naive chi-squared should be divided by, in the sense that
    ``chi2_effective = chi2_nominal / inflation**2``.
    """
    summary = summarise_audit(table, lags)
    if not summary.get("n_spectra"):
        return summary
    sigma = summary["residual_sigma_median"] / bias.residual_sigma
    rho1 = summary.get("acf_lag1_median", 0.0) - bias.acf_lag1
    acf = {1: rho1}
    for lag in lags[1:]:
        acf[lag] = summary.get(f"acf_lag{lag}_median", 0.0)
    summary.update(
        {
            "bias_n_knots": float(bias.n_knots),
            "bias_residual_sigma": bias.residual_sigma,
            "bias_acf_lag1": bias.acf_lag1,
            "residual_sigma_corrected": float(sigma),
            "acf_lag1_corrected": float(rho1),
            "inflation_corrected": inflation_factor(sigma, acf),
            "chi2_scale_corrected": float(inflation_factor(sigma, acf) ** 2),
        }
    )
    return summary
