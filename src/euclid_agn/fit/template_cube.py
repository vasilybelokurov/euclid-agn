"""Precomputed template projections on the common archive grid.

Every Q1 SIR combined spectrum lives on the same 531-pixel grid
(11900-19002 Angstrom, 13.4 Angstrom bins; verified across tiles), so the
expensive part of a continuum redshift scan - redshifting, LSF-smoothing and
pixel-integrating each template at each trial redshift - can be done once per
(template set, redshift grid, LSF) and reused for every object.  Per object
only the kept-pixel subset and a batch of small linear solves remain, which
turns 2 s per spectrum into a few milliseconds.

The LSF varies from object to object (median 13.7 Angstrom, tail to ~100).
Cubes are keyed on the LSF rounded to ``lsf_step`` Angstrom; at the 13.4
Angstrom pixel scale the difference between 13.7 and 15 Angstrom smoothing is
far below the noise, and objects with anomalously broad LSF get their own cube.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from scipy.optimize import nnls

from euclid_agn.constants import C_KMS
from euclid_agn.fit.continuum_redshift import ContinuumScanResult, redshift_grid
from euclid_agn.models.library import Template, project_template_on_edges
from euclid_agn.numerics import blas_safe
from euclid_agn.spectra.lsf import pixel_edges


@dataclass(frozen=True)
class TemplateCube:
    kind: str
    grid_z: np.ndarray  # (n_z,)
    wavelength: np.ndarray  # full archive grid (n_pix,)
    columns: np.ndarray  # (n_z, n_pix, n_templates); NaN where a template does not cover the grid
    lsf_sigma: float
    names: tuple[str, ...]
    metadata: dict = field(default_factory=dict)

    @property
    def n_templates(self) -> int:
        return self.columns.shape[2]

    def covers(self, keep: np.ndarray) -> np.ndarray:
        """Boolean per redshift: every kept pixel has a finite template value."""
        return np.isfinite(self.columns[:, keep, :]).all(axis=(1, 2))


def build_cube(
    templates: list[Template],
    grid_z: np.ndarray,
    wavelength: np.ndarray,
    bin_width: float,
    lsf_sigma: float,
    extra_sigma_kms: float = 0.0,
    kind: str | None = None,
) -> TemplateCube:
    """Project ``templates`` at every redshift in ``grid_z`` onto the full grid.

    A template that does not cover the whole grid at some redshift gets NaN
    there; :meth:`TemplateCube.covers` decides per object whether the kept
    pixels are covered, so partial coverage costs only the redshifts where the
    object's own kept range falls outside the template.
    """
    full_edges = pixel_edges(wavelength, bin_width)
    edges = np.column_stack([full_edges[:-1], full_edges[1:]])
    cube = np.full((grid_z.size, wavelength.size, len(templates)), np.nan)
    for i, z in enumerate(grid_z):
        for j, t in enumerate(templates):
            column = project_template_on_edges(t, edges, lsf_sigma, float(z), extra_sigma_kms)
            if column is None:
                # try the covered sub-range: mark uncovered pixels NaN
                observed = t.wavelength * (1.0 + z)
                inside = (edges[:, 0] >= observed[0]) & (edges[:, 1] <= observed[-1])
                if inside.sum() >= 2:
                    partial = project_template_on_edges(t, edges[inside], lsf_sigma, float(z), extra_sigma_kms)
                    if partial is not None:
                        cube[i, inside, j] = partial
                continue
            cube[i, :, j] = column
    return TemplateCube(kind or templates[0].kind, np.asarray(grid_z, float), np.asarray(wavelength, float), cube,
                        float(lsf_sigma), tuple(t.name for t in templates), {"extra_sigma_kms": extra_sigma_kms})


def polynomial_columns(wavelength: np.ndarray, degree: int, reference: np.ndarray | None = None) -> np.ndarray:
    """Legendre-like columns in a wavelength variable scaled to [-1, 1] over ``reference``."""
    ref = wavelength if reference is None else reference
    x = (wavelength - ref.mean()) / (np.ptp(ref) / 2)
    if degree < 0:
        return np.zeros((wavelength.size, 0))
    return np.column_stack([x**k for k in range(degree + 1)])


def prior_penalty(grid_z: np.ndarray, z_prior: float | None, sigma: float = 0.05, outlier_fraction: float = 0.13,
                  z_range: float = 5.0) -> np.ndarray:
    """-2 ln of a Gaussian-plus-uniform photometric-redshift prior, zero at its peak.

    Same mixture as :func:`euclid_agn.fit.screen.apply_redshift_prior`: the
    uniform component caps the penalty (~11 for the defaults) so a wrong
    photo-z cannot veto strong spectral evidence.
    """
    if z_prior is None or not np.isfinite(z_prior):
        return np.zeros_like(grid_z)
    f = min(max(outlier_fraction, 1e-6), 1.0 - 1e-6)
    pull = (grid_z - z_prior) / (1.0 + z_prior) / sigma
    gaussian = (1.0 - f) * np.exp(-0.5 * pull**2) / (sigma * np.sqrt(2.0 * np.pi))
    uniform = f / z_range
    peak = (1.0 - f) / (sigma * np.sqrt(2.0 * np.pi)) + uniform
    return -2.0 * np.log((gaussian + uniform) / peak)


def _best_and_runner(grid_z, chi2, separation_kms):
    best = int(np.nanargmin(chi2))
    far = np.abs(C_KMS * (grid_z - grid_z[best]) / (1 + grid_z[best])) > separation_kms
    if np.isfinite(chi2[far]).any():
        runner = int(np.nanargmin(np.where(far, chi2, np.nan)))
        return best, float(chi2[runner] - chi2[best]), float(grid_z[runner])
    return best, float("inf"), float("nan")


@blas_safe
def cube_scan(
    spectrum,
    projected,
    cube: TemplateCube,
    poly_degree: int = 1,
    nonnegative: bool = False,
    separation_kms: float = 3000.0,
    z_prior: float | None = None,
    prior_sigma: float = 0.05,
    prior_outlier_fraction: float = 0.13,
    spline_nuisance: bool = False,
) -> ContinuumScanResult | None:
    """Chi-squared of the cube's templates (+ polynomial) at every redshift.

    ``spline_nuisance``: instead of the polynomial, the spline continuum basis
    already in ``projected`` (``n_knots`` of the ScreenSettings used to prepare
    it) is projected out of both data and templates.  Only spectral *features*
    then contribute: the fit is insensitive to broad-band shape errors
    (flux calibration, extraction aperture of extended sources) at the cost of
    discarding the 1.6 micron bump's shape.

    Unconstrained: batched normal equations, one ``solve`` for all redshifts.
    ``nonnegative``: template coefficients >= 0 via NNLS (polynomial columns
    are sign-split so they stay free) - the archetype mode, in which a fit is
    a physical mixture rather than an arbitrary combination.

    ``chi2`` values are on the archive variance; a caller who knows the noise
    inflation should divide the Delta chi-squared values by it.
    """
    if projected is None:
        return None
    if spectrum.wavelength.size != cube.wavelength.size or not np.allclose(spectrum.wavelength, cube.wavelength):
        raise ValueError("spectrum is not on the cube's wavelength grid")
    keep = np.isin(spectrum.wavelength, projected.wavelength)
    weight = projected.weight
    data_w = spectrum.flux[keep] * weight
    if spline_nuisance:
        poly = np.zeros((weight.size, 0))
        data_w = data_w - projected.basis @ (projected.basis.T @ data_w)
    else:
        poly = polynomial_columns(projected.wavelength, poly_degree, reference=spectrum.wavelength) * weight[:, None]
    covered = cube.covers(keep)
    n_z = cube.grid_z.size
    chi2 = np.full(n_z, np.nan)
    coefficients = [None] * n_z
    if not covered.any():
        return None
    templates = cube.columns[covered][:, keep, :] * weight[None, :, None]  # (n_cov, n_kept, n_t)
    if spline_nuisance:
        b = projected.basis
        templates = templates - np.einsum("ik,zkj->zij", b, np.einsum("ik,zij->zkj", b, templates))
    n_cov = templates.shape[0]
    poly_b = np.broadcast_to(poly, (n_cov, *poly.shape))
    design = np.concatenate([templates, poly_b], axis=2)  # (n_cov, n_kept, k)
    # column scaling for conditioning (whitened polynomial columns are ~1e17
    # times larger than unit-RMS template columns): unit RMS per column, undone
    # on the coefficients afterwards.  The tiny ridge only guards exact rank loss.
    scale = np.sqrt(np.mean(design**2, axis=1, keepdims=True))
    scale = np.where(scale > 0, scale, 1.0)
    design = design / scale
    idx = np.flatnonzero(covered)
    if not nonnegative:
        # batched minimum-norm least squares (pinv tolerates the rank
        # deficiency a flat template + constant polynomial column produces);
        # chi-squared from explicit residuals, never by cancellation
        pinv = np.linalg.pinv(design, rcond=1e-10)
        coef = np.einsum("zki,i->zk", pinv, data_w)
        residual = data_w[None, :] - np.einsum("zik,zk->zi", design, coef)
        chi2[idx] = np.einsum("zi,zi->z", residual, residual)
        coef = coef / scale[:, 0, :]
        for j, i in enumerate(idx):
            coefficients[i] = coef[j]
    else:
        n_t = templates.shape[2]
        for j, i in enumerate(idx):
            a = np.concatenate([design[j], -design[j][:, n_t:]], axis=1)
            c, rnorm = nnls(a, data_w, maxiter=50 * a.shape[1])
            chi2[i] = rnorm**2
            c = np.concatenate([c[:n_t], c[n_t : n_t + poly.shape[1]] - c[n_t + poly.shape[1] :]]) / scale[j, 0, :]
            coefficients[i] = c
    # null: nuisance only
    if spline_nuisance:
        resid0 = data_w  # already orthogonal to the spline basis
    elif poly.shape[1]:
        c0, *_ = np.linalg.lstsq(poly, data_w, rcond=None)
        resid0 = data_w - poly @ c0
    else:
        resid0 = data_w
    n_nuisance = projected.basis.shape[1] if spline_nuisance else poly.shape[1]
    chi2_null = float(resid0 @ resid0)
    best, delta_runner, z_runner = _best_and_runner(cube.grid_z, chi2, separation_kms)
    result = ContinuumScanResult(
        z=float(cube.grid_z[best]), chi2=float(chi2[best]), chi2_null=chi2_null,
        delta_chi2_runner_up=delta_runner, z_runner_up=z_runner, n_pixels=int(keep.sum()),
        n_parameters=cube.n_templates + n_nuisance, kind=cube.kind, grid_z=cube.grid_z, grid_chi2=chi2,
        coefficients=coefficients[best],
    )
    if z_prior is not None and np.isfinite(z_prior):
        penalised = chi2 + prior_penalty(cube.grid_z, z_prior, prior_sigma, prior_outlier_fraction)
        pbest, pdelta, pz_runner = _best_and_runner(cube.grid_z, penalised, separation_kms)
        result = ContinuumScanResult(**{**result.__dict__, "z_prior": float(cube.grid_z[pbest]),
                                        "delta_chi2_runner_up_prior": pdelta})
    return result


def fit_at(spectrum, projected, cube: TemplateCube, z: float, **kwargs) -> ContinuumScanResult | None:
    """The scan restricted to the grid point nearest ``z`` (for diagnostics)."""
    i = int(np.argmin(np.abs(np.log1p(cube.grid_z) - np.log1p(z))))
    single = TemplateCube(cube.kind, cube.grid_z[i : i + 1], cube.wavelength, cube.columns[i : i + 1], cube.lsf_sigma,
                          cube.names, cube.metadata)
    return cube_scan(spectrum, projected, single, **kwargs)


@blas_safe
def model_flux(spectrum, projected, cube: TemplateCube, result: ContinuumScanResult, poly_degree: int = 1,
               spline_nuisance: bool = False) -> np.ndarray:
    """Best-fit model (flux units) on the kept pixels for ``result``.

    With ``spline_nuisance`` the returned model is the template part plus the
    spline component that best fits the residual, so it is comparable to the data.
    """
    keep = np.isin(spectrum.wavelength, projected.wavelength)
    i = int(np.argmin(np.abs(cube.grid_z - result.z)))
    templates = cube.columns[i][keep]  # (n_kept, n_t), flux units
    n_t = templates.shape[1]
    coef = np.asarray(result.coefficients)
    model = templates @ coef[:n_t]
    if spline_nuisance:
        w = projected.weight
        resid_w = (spectrum.flux[keep] - model) * w
        model = model + (projected.basis @ (projected.basis.T @ resid_w)) / w
    else:
        poly = polynomial_columns(projected.wavelength, poly_degree, reference=spectrum.wavelength)
        model = model + poly @ coef[n_t : n_t + poly.shape[1]]
    return model


class CubeStore:
    """Cubes keyed on (template set name, LSF bucket); built lazily."""

    def __init__(self, templates_by_kind: dict[str, list[Template]], grid_z: np.ndarray, wavelength: np.ndarray,
                 bin_width: float, lsf_step: float = 5.0, extra_sigma_kms: dict[str, float] | None = None):
        self.templates_by_kind = templates_by_kind
        self.grid_z = np.asarray(grid_z, float)
        self.wavelength = None if wavelength is None else np.asarray(wavelength, float)
        self.bin_width = float(bin_width)
        self.lsf_step = float(lsf_step)
        self.extra_sigma_kms = extra_sigma_kms or {}
        self._cubes: dict[tuple[str, float], TemplateCube] = {}

    def lsf_bucket(self, lsf_sigma: float) -> float:
        return float(max(self.lsf_step, np.round(lsf_sigma / self.lsf_step) * self.lsf_step))

    def get(self, kind: str, lsf_sigma: float) -> TemplateCube:
        key = (kind, self.lsf_bucket(lsf_sigma))
        if key not in self._cubes:
            if self.wavelength is None:
                raise ValueError("CubeStore.wavelength must be set before building cubes")
            self._cubes[key] = build_cube(self.templates_by_kind[kind], self.grid_z, self.wavelength, self.bin_width,
                                          key[1], self.extra_sigma_kms.get(kind, 0.0), kind=kind)
        return self._cubes[key]

    def __len__(self) -> int:
        return len(self._cubes)


__all__ = ["TemplateCube", "CubeStore", "build_cube", "cube_scan", "fit_at", "model_flux", "polynomial_columns", "prior_penalty",
           "redshift_grid"]
