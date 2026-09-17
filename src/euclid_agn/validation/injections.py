"""Injection/recovery into real spectra, and empirical nulls.

This is the experiment that turns the pipeline's statistic into a selection
function.  Nothing about AGN detection may be claimed before it has run.

Injection
---------
A broad Gaussian is added to a *real* combined spectrum at a *known* redshift -
one the pipeline has already recovered against DESI, so the identification
question is settled and only measurement is being tested.  The line is
LSF-convolved with the object's own ``LSF_SIG`` and pixel-integrated on the
object's own grid, then the M0/M1 fit is run at that redshift and the recovered
broad flux and Delta chi-squared recorded against what went in.  The variance
is left as reported: an injected line of the fluxes considered here adds
negligibly to the photon noise of a continuum at S/N 3 per pixel, and leaving
it fixed keeps the null and the injection on the same footing.

Nulls
-----
Two constructions, both on real spectra with **nothing injected**:

``off_redshift``   the M1 fit at a redshift displaced from the true one by more
                   than the line widths, where no permitted line falls on a
                   real feature;
``forbidden``      a broad component attached to a *forbidden* transition at the
                   true redshift.  The machinery fits it; the AGN model must
                   never read it as a BLR.

The Delta chi-squared distribution over these is the false-positive
distribution against which any threshold has to be set.  Every row carries the
object's LSF, S/N, usable fraction, noise inflation and the broad component's
orthogonality to the continuum, because completeness must be reported against
all of them, not as one flux curve.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from euclid_agn.constants import C_KMS
from euclid_agn.fit.screen import ScreenSettings, prepare, refine
from euclid_agn.models.broad import BroadComponent
from euclid_agn.models.line_catalog import BY_NAME, BY_SYSTEM
from euclid_agn.spectra.types import Spectrum1D

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Injection:
    """One injected broad line."""

    line_name: str
    flux: float  # erg s^-1 cm^-2
    sigma_kms: float
    velocity_kms: float = 0.0


def inject_broad_line(spectrum: Spectrum1D, z: float, injection: Injection) -> Spectrum1D:
    """Return a copy of ``spectrum`` with a broad line added at redshift ``z``.

    The profile uses the object's own LSF and pixel grid, so what is injected
    is exactly what the forward model would predict for that object.  Masked
    pixels receive the flux too; they are dropped by ``usable()`` as always.
    """
    component = BroadComponent(
        line_name=injection.line_name,
        z=z,
        sigma_kms=injection.sigma_kms,
        lsf_sigma=spectrum.lsf_sigma,
        velocity_kms=injection.velocity_kms,
        allow_forbidden=True,
    )
    profile = component.basis(spectrum.wavelength, spectrum.bin_width)
    metadata = dict(spectrum.metadata)
    metadata["injected"] = {
        "line": injection.line_name,
        "flux": injection.flux,
        "sigma_kms": injection.sigma_kms,
        "velocity_kms": injection.velocity_kms,
        "z": z,
    }
    return replace(spectrum, flux=spectrum.flux + injection.flux * profile, metadata=metadata)


def measurement_row(spectrum: Spectrum1D, z: float, system: str, sigma_kms: float,
                    settings: ScreenSettings, noise_inflation: float = 1.0,
                    forbidden_line: str | None = None) -> dict | None:
    """Run M0/M1 at a fixed redshift and flatten the result.

    With ``forbidden_line`` the broad family is replaced by that single
    forbidden transition (the null construction); the fit is otherwise
    identical.
    """
    fit = refine(spectrum, z, system, sigma_kms, settings=settings) if forbidden_line is None else _refine_forbidden(
        spectrum, z, system, sigma_kms, forbidden_line, settings
    )
    if fit is None:
        return None
    summary = fit.summary()
    projected = prepare(spectrum, settings)
    fluxes = fit.broad_fluxes
    errors = fit.broad_flux_errors
    primary = next(iter(fluxes), None)
    row = {
        "z": z,
        "system": system,
        "broad_line": primary or "",
        "broad_sigma_kms": sigma_kms,
        "delta_chi2": summary["delta_chi2"],
        "delta_chi2_effective": summary["delta_chi2"] / noise_inflation**2,
        "noise_inflation": noise_inflation,
        "recovered_flux": fluxes.get(primary, np.nan) if primary else np.nan,
        "recovered_flux_error": errors.get(primary, np.nan) if primary else np.nan,
        "chi2_m0": summary["chi2_m0"],
        "n_pixels": summary["n_pixels"],
        "lsf_sigma": spectrum.lsf_sigma,
        "n_outlier_pixels": projected.n_outliers if projected is not None else 0,
        "broad_resolution_ratio": summary.get("broad_resolution_ratio", np.nan),
    }
    if projected is not None and primary:
        from euclid_agn.fit.screen import broad_column, continuum_orthogonality

        column = broad_column(projected, primary, z, sigma_kms) if not forbidden_line else None
        row["broad_continuum_orthogonality"] = (
            continuum_orthogonality(projected, column) if column is not None else np.nan
        )
    return row


def _refine_forbidden(spectrum, z, system, sigma_kms, forbidden_line, settings):
    """M0/M1 with the broad component forced onto a forbidden line."""
    from euclid_agn.models.broad import BroadFamily
    from euclid_agn.models.forward import fit_hypothesis
    from euclid_agn.models.narrow import NarrowSystem, velocity_grid

    projected = prepare(spectrum, settings)
    if projected is None:
        return None
    lines = BY_SYSTEM[system].lines()
    visible = [ln.name for ln in lines if projected.wavelength[0] <= ln.rest * (1 + z) <= projected.wavelength[-1]]
    if forbidden_line not in visible:
        return None
    narrow = NarrowSystem(tuple(visible), z, projected.lsf_sigma,
                          velocity_grid(settings.narrow_velocity_half_width_kms, settings.narrow_velocity_step_kms))
    broad = BroadFamily((forbidden_line,), z, sigma_kms, projected.lsf_sigma, allow_forbidden=True)
    keep = np.isin(spectrum.wavelength, projected.wavelength)
    return fit_hypothesis(spectrum.wavelength[keep], spectrum.flux[keep], spectrum.variance[keep],
                          narrow, broad, n_knots=settings.n_knots, bin_width=spectrum.bin_width,
                          smoothness=settings.narrow_smoothness)


def flux_grid(log10_min: float = -17.0, log10_max: float = -15.0, n: int = 9) -> np.ndarray:
    return np.logspace(log10_min, log10_max, n)


def injection_recovery(
    targets: Iterable[tuple[int, Spectrum1D, float, str, float]],
    fluxes: Sequence[float],
    sigmas_kms: Sequence[float],
    settings: ScreenSettings,
    line_name: str = "Halpha",
) -> pd.DataFrame:
    """Inject a grid of broad lines into each target and measure recovery.

    ``targets`` yields ``(object_id, spectrum, z, system, noise_inflation)``.
    The unmodified spectrum is measured too (``injected_flux = 0``), which is
    the on-redshift null for that object.
    """
    rows = []
    for object_id, spectrum, z, system, inflation in targets:
        base = measurement_row(spectrum, z, system, float(sigmas_kms[0]), settings, inflation)
        if base is None:
            continue
        metrics = spectrum.quality_metrics()
        common = {"object_id": object_id, "usable_pixel_fraction": metrics["usable_pixel_fraction"],
                  "median_snr_per_pixel": metrics["median_snr_per_pixel"]}
        for sigma in sigmas_kms:
            null = measurement_row(spectrum, z, system, float(sigma), settings, inflation)
            if null is not None:
                rows.append({**common, **null, "injected_flux": 0.0, "injected_sigma_kms": float(sigma), "kind": "on_redshift_null"})
            for flux in fluxes:
                injected = inject_broad_line(spectrum, z, Injection(line_name, float(flux), float(sigma)))
                row = measurement_row(injected, z, system, float(sigma), settings, inflation)
                if row is None:
                    continue
                rows.append({**common, **row, "injected_flux": float(flux), "injected_sigma_kms": float(sigma), "kind": "injection"})
    return pd.DataFrame(rows)


def null_off_redshift(
    targets: Iterable[tuple[int, Spectrum1D, float, str, float]],
    sigmas_kms: Sequence[float],
    settings: ScreenSettings,
    offsets: Sequence[float] = (-0.25, -0.15, 0.15, 0.25),
) -> pd.DataFrame:
    """M1 at redshifts displaced from the true one, nothing injected.

    Offsets are in ``dz/(1+z)``; ±0.15 moves H-alpha by ~2000 A, well clear of
    its own profile and of the [N II]/[S II] complex.
    """
    rows = []
    for object_id, spectrum, z, system, inflation in targets:
        for offset in offsets:
            z_off = (1.0 + z) * (1.0 + offset) - 1.0
            for sigma in sigmas_kms:
                row = measurement_row(spectrum, z_off, system, float(sigma), settings, inflation)
                if row is None:
                    continue
                rows.append({"object_id": object_id, "true_z": z, "offset": offset, "kind": "off_redshift_null",
                             "injected_flux": 0.0, "injected_sigma_kms": float(sigma), **row})
    return pd.DataFrame(rows)


def null_forbidden(
    targets: Iterable[tuple[int, Spectrum1D, float, str, float]],
    sigmas_kms: Sequence[float],
    settings: ScreenSettings,
    forbidden_line: str = "NII6584",
) -> pd.DataFrame:
    """Broad component on a forbidden line at the true redshift, nothing injected."""
    rows = []
    for object_id, spectrum, z, system, inflation in targets:
        for sigma in sigmas_kms:
            row = measurement_row(spectrum, z, system, float(sigma), settings, inflation, forbidden_line=forbidden_line)
            if row is None:
                continue
            rows.append({"object_id": object_id, "kind": "forbidden_null", "forbidden_line": forbidden_line,
                         "injected_flux": 0.0, "injected_sigma_kms": float(sigma), **row})
    return pd.DataFrame(rows)


# --- metrics ------------------------------------------------------------------


def threshold_for_false_positive_rate(null_statistic: np.ndarray, rate: float) -> float:
    """Statistic value exceeded by a fraction ``rate`` of the null."""
    null_statistic = np.asarray(null_statistic, dtype=float)
    null_statistic = null_statistic[np.isfinite(null_statistic)]
    if null_statistic.size == 0:
        return float("nan")
    # "higher" so the returned threshold is never exceeded by more than ``rate``
    # of the null sample - conservative in the direction that matters.
    return float(np.quantile(null_statistic, 1.0 - rate, method="higher"))


def completeness_table(
    injections: pd.DataFrame,
    threshold: float,
    statistic: str = "delta_chi2_effective",
    by: Sequence[str] = ("injected_flux", "injected_sigma_kms"),
) -> pd.DataFrame:
    """Fraction of injections recovered above ``threshold``, by the given axes.

    Also reports the median recovered/injected flux ratio, which is the flux
    bias of the measurement conditional on detection.
    """
    table = injections[injections["kind"] == "injection"].copy()
    if table.empty:
        return pd.DataFrame()
    table["detected"] = table[statistic] > threshold
    table["flux_ratio"] = table["recovered_flux"] / table["injected_flux"]
    grouped = table.groupby(list(by), observed=True)
    out = grouped.agg(
        n=("detected", "size"),
        completeness=("detected", "mean"),
        flux_ratio_median=("flux_ratio", "median"),
    ).reset_index()
    detected_only = table[table["detected"]].groupby(list(by), observed=True)["flux_ratio"].median()
    out = out.merge(detected_only.rename("flux_ratio_if_detected").reset_index(), on=list(by), how="left")
    return out


def velocity_to_redshift_offset(velocity_kms: float) -> float:
    return velocity_kms / C_KMS


__all__ = [
    "Injection", "inject_broad_line", "measurement_row", "injection_recovery",
    "null_off_redshift", "null_forbidden", "threshold_for_false_positive_rate",
    "completeness_table", "flux_grid", "BY_NAME",
]
