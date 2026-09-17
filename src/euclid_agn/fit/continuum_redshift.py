"""Continuum (template) redshift scan - the GALAXY/QSO/STAR class fits.

This is the Redrock/AMAZED step our emission-line scan lacked: at every trial
redshift the spectrum is fitted by a linear combination of a template basis -
PCA components of the XSL stellar populations, or a quasar composite - and the
redshift is the chi-squared minimum.  No spline continuum is projected out
here: the templates *are* the continuum, so all the continuum shape information
(1.6 micron bump, CO bandheads, Paschen absorption) is used rather than
discarded.

A short polynomial in wavelength multiplies nothing and is added as extra free
columns instead: it absorbs residual flux-calibration tilt without being able
to mimic a template.  Coefficients are unconstrained (PCA components are not
positive).

Reliability follows Redrock: ``delta_chi2`` is the chi-squared difference
between the best minimum and the best minimum at a redshift more than
``separation_kms`` away.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from euclid_agn.constants import C_KMS
from euclid_agn.models.library import ContinuumBasis, Template, project_template
from euclid_agn.numerics import blas_safe


@dataclass(frozen=True)
class ContinuumScanResult:
    z: float
    chi2: float
    chi2_null: float  # polynomial-only fit
    delta_chi2_runner_up: float
    z_runner_up: float
    n_pixels: int
    n_parameters: int
    kind: str
    grid_z: np.ndarray
    grid_chi2: np.ndarray
    coefficients: np.ndarray

    @property
    def delta_chi2_null(self) -> float:
        """Improvement over having no template at all."""
        return self.chi2_null - self.chi2

    def as_row(self) -> dict:
        return {"cz_z": self.z, "cz_chi2": self.chi2, "cz_delta_chi2_null": self.delta_chi2_null,
                "cz_delta_chi2_runner_up": self.delta_chi2_runner_up, "cz_z_runner_up": self.z_runner_up,
                "cz_kind": self.kind, "cz_n_pixels": self.n_pixels}


def redshift_grid(z_min: float, z_max: float, step_kms: float) -> np.ndarray:
    n = int(np.ceil(np.log((1 + z_max) / (1 + z_min)) / (step_kms / C_KMS)))
    return (1 + z_min) * np.exp(np.arange(n + 1) * step_kms / C_KMS) - 1


def _polynomial_columns(projected, degree: int) -> np.ndarray:
    x = (projected.wavelength - projected.wavelength.mean()) / (projected.wavelength.ptp() / 2)
    return np.column_stack([x**k for k in range(degree + 1)]) if degree >= 0 else np.zeros((x.size, 0))


@blas_safe
def _chi2_at(projected, templates: list[Template], z: float, poly: np.ndarray, data_w: np.ndarray):
    columns = []
    for t in templates:
        c = project_template(t, projected, z)
        if c is None:
            return None, None
        columns.append(c)
    design = np.column_stack([*columns, poly]) * projected.weight[:, None]
    coefficients, *_ = np.linalg.lstsq(design, data_w, rcond=None)
    residual = data_w - design @ coefficients
    return float(residual @ residual), coefficients


@blas_safe
def continuum_redshift_scan(
    spectrum,
    projected,
    basis_templates: list[Template],
    z_min: float = 0.0,
    z_max: float = 1.0,
    step_kms: float = 300.0,
    poly_degree: int = 1,
    separation_kms: float = 3000.0,
    kind: str = "GALAXY",
) -> ContinuumScanResult | None:
    """Chi-squared of the template basis at every redshift on the grid.

    ``projected`` must come from :func:`euclid_agn.fit.screen.prepare`; only its
    kept pixels, weights and edges are used - its continuum projection is not.
    """
    if projected is None:
        return None
    keep = np.isin(spectrum.wavelength, projected.wavelength)
    data_w = spectrum.flux[keep] * projected.weight
    poly = _polynomial_columns(projected, poly_degree)
    grid = redshift_grid(z_min, z_max, step_kms)
    chi2 = np.full(grid.size, np.nan)
    coefficients = [None] * grid.size
    for i, z in enumerate(grid):
        c2, coef = _chi2_at(projected, basis_templates, float(z), poly, data_w)
        if c2 is not None:
            chi2[i] = c2; coefficients[i] = coef
    if not np.isfinite(chi2).any():
        return None
    # null: polynomial only
    design0 = poly * projected.weight[:, None]
    coef0, *_ = np.linalg.lstsq(design0, data_w, rcond=None) if poly.shape[1] else (np.zeros(0),)
    resid0 = data_w - (design0 @ coef0 if poly.shape[1] else 0.0)
    chi2_null = float(resid0 @ resid0)
    best = int(np.nanargmin(chi2))
    far = np.abs(C_KMS * (grid - grid[best]) / (1 + grid[best])) > separation_kms
    if np.isfinite(chi2[far]).any():
        runner = int(np.nanargmin(np.where(far, chi2, np.nan)))
        delta_runner, z_runner = float(chi2[runner] - chi2[best]), float(grid[runner])
    else:
        delta_runner, z_runner = float("inf"), float("nan")
    return ContinuumScanResult(
        z=float(grid[best]), chi2=float(chi2[best]), chi2_null=chi2_null,
        delta_chi2_runner_up=delta_runner, z_runner_up=z_runner, n_pixels=int(projected.wavelength.size),
        n_parameters=len(basis_templates) + poly.shape[1], kind=kind, grid_z=grid, grid_chi2=chi2,
        coefficients=coefficients[best],
    )


def refine_minimum(result: ContinuumScanResult) -> float:
    """Parabolic refinement of the chi-squared minimum on the grid."""
    i = int(np.nanargmin(result.grid_chi2))
    if i == 0 or i == result.grid_z.size - 1:
        return result.z
    y0, y1, y2 = result.grid_chi2[i - 1: i + 2]
    if not np.isfinite([y0, y1, y2]).all():
        return result.z
    denom = y0 - 2 * y1 + y2
    if denom <= 0:
        return result.z
    offset = 0.5 * (y0 - y2) / denom
    x = np.log(1 + result.grid_z[i - 1: i + 2])
    return float(np.exp(x[1] + offset * (x[2] - x[1])) - 1)
