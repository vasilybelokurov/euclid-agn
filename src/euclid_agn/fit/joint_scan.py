"""Joint continuum + emission-line redshift scan.

One chi-squared curve per object: at every trial redshift the design holds
the continuum templates from a :class:`~euclid_agn.fit.template_cube.TemplateCube`,
one fixed-ratio emission-line template per visible line system (narrow,
``narrow_sigma_kms``) and the nuisance polynomial.  A z = 0.3 galaxy is then
found from its continuum shape *and* Pa-beta/[S III]; a z = 1.5 galaxy from its
continuum and the H-alpha complex; a line-free continuum still gets a redshift.
This replaces running the emission-line matched filter and the continuum scan
separately and reconciling them afterwards.

Line amplitudes are constrained non-negative (emission, never absorption);
continuum coefficients follow ``nonnegative``; the polynomial is free.  At the
best redshift the continuum-only chi-squared is refitted so
``delta_chi2_lines`` - the evidence the lines add - is available to the AGN
stage and to the ``NO_LINE`` quality bit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import nnls

from euclid_agn.constants import C_KMS
from euclid_agn.fit.continuum_redshift import ContinuumScanResult
from euclid_agn.fit.template_cube import TemplateCube, _best_and_runner, polynomial_columns, prior_penalty
from euclid_agn.models.templates import TEMPLATES, EmissionTemplate
from euclid_agn.models.templates import template_column as line_template_column
from euclid_agn.numerics import blas_safe

#: One representative template per line system: enough to *find* the redshift;
#: the AGN stage refits ratios and widths afterwards.
DEFAULT_LINE_TEMPLATES: tuple[str, ...] = ("halpha_hii", "hbeta_hii", "pabeta_hii", "hei_paschen", "oii_neiii")


def line_templates(names=DEFAULT_LINE_TEMPLATES) -> list[EmissionTemplate]:
    by_name = {t.name: t for t in TEMPLATES}
    return [by_name[n] for n in names]


@dataclass(frozen=True)
class JointScanResult(ContinuumScanResult):
    delta_chi2_lines: float = float("nan")  # chi2(continuum only) - chi2(joint) at the best redshift
    line_amplitudes: dict | None = None  # template name -> reference-line flux (data units)
    line_snr: dict | None = None  # template name -> amplitude / its formal error
    n_line_templates: int = 0

    def as_row(self) -> dict:
        row = super().as_row()
        row.update({"cz_delta_chi2_lines": self.delta_chi2_lines, "cz_n_line_templates": self.n_line_templates})
        if self.line_snr:
            best = max(self.line_snr, key=self.line_snr.get)
            row["cz_best_line_template"] = best
            row["cz_best_line_snr"] = float(self.line_snr[best])
        return row


def _solve(design: np.ndarray, data: np.ndarray, n_free_tail: int, nonneg_columns: int):
    """NNLS on the first ``nonneg_columns`` columns, free on the rest (sign-split)."""
    if nonneg_columns == design.shape[1]:
        c, rnorm = nnls(design, data, maxiter=50 * design.shape[1])
        return c, rnorm**2
    free = design[:, nonneg_columns:]
    a = np.concatenate([design[:, :nonneg_columns], free, -free], axis=1)
    c, rnorm = nnls(a, data, maxiter=50 * a.shape[1])
    k = free.shape[1]
    coef = np.concatenate([c[:nonneg_columns], c[nonneg_columns : nonneg_columns + k] - c[nonneg_columns + k :]])
    return coef, rnorm**2


@blas_safe
def joint_scan(
    spectrum,
    projected,
    cube: TemplateCube,
    lines: list[EmissionTemplate] | None = None,
    poly_degree: int = 3,
    nonnegative: bool = True,
    narrow_sigma_kms: float = 200.0,
    edge_margin_pixels: float = 4.0,
    separation_kms: float = 3000.0,
    z_prior: float | None = None,
    prior_sigma: float = 0.05,
    prior_outlier_fraction: float = 0.13,
    multiplicative_degree: int = 0,
    multiplicative_iterations: int = 2,
) -> JointScanResult | None:
    """Chi-squared of continuum templates + line templates + polynomial at every redshift.

    ``multiplicative_degree`` > 0 multiplies the *continuum* mixture by
    ``1 + sum p_k L_k`` (lines are not multiplied: their fluxes are what we
    measure), solved by alternation as in :func:`cube_scan`.

    ``nonnegative`` applies to the continuum coefficients; line amplitudes are
    always >= 0 and the polynomial is always free.  With ``nonnegative=False``
    the continuum columns are sign-split too, so a single NNLS solves each
    redshift either way.
    """
    if projected is None:
        return None
    lines = line_templates() if lines is None else lines
    keep = np.isin(spectrum.wavelength, projected.wavelength)
    weight = projected.weight
    data_w = spectrum.flux[keep] * weight
    poly = polynomial_columns(projected.wavelength, poly_degree, reference=spectrum.wavelength) * weight[:, None]
    covered = cube.covers(keep)
    grid = cube.grid_z
    chi2 = np.full(grid.size, np.nan)
    coefficients = [None] * grid.size
    columns_at = [None] * grid.size
    edge = edge_margin_pixels * float(projected.bin_width)
    n_c = cube.n_templates
    mult = polynomial_columns(projected.wavelength, multiplicative_degree, reference=spectrum.wavelength)[:, 1:] if multiplicative_degree > 0 else None
    for i in np.flatnonzero(covered):
        z = float(grid[i])
        cont = cube.columns[i][keep] * weight[:, None]
        line_cols, names = [], []
        for t in lines:
            col, n_used = line_template_column(projected, t, z, narrow_sigma_kms, edge_margin=edge)
            if col is not None and n_used > 0 and np.any(col > 0):
                line_cols.append(col * weight)
                names.append(t.name)
        design = np.concatenate([cont, np.column_stack(line_cols) if line_cols else np.zeros((weight.size, 0)), poly], axis=1)
        scale = np.sqrt(np.mean(design**2, axis=0))
        scale = np.where(scale > 0, scale, 1.0)
        design = design / scale
        n_nonneg = (n_c if nonnegative else 0) + len(line_cols)
        if not nonnegative:
            # continuum columns free: sign-split them by moving them after the lines
            design = np.concatenate([design[:, n_c : n_c + len(line_cols)], design[:, :n_c], design[:, n_c + len(line_cols) :]], axis=1)
            coef, c2 = _solve(design, data_w, 0, len(line_cols))
            coef = np.concatenate([coef[len(line_cols) : len(line_cols) + n_c], coef[: len(line_cols)], coef[len(line_cols) + n_c :]])
            scale = np.concatenate([scale[n_c : n_c + len(line_cols)], scale[:n_c], scale[n_c + len(line_cols) :]])
            scale = np.concatenate([scale[len(line_cols) : len(line_cols) + n_c], scale[: len(line_cols)], scale[len(line_cols) + n_c :]])
        else:
            coef, c2 = _solve(design, data_w, 0, n_nonneg)
            if mult is not None and mult.shape[1]:
                # alternate: multiplicative correction of the continuum part, then refit
                base = design
                for _ in range(multiplicative_iterations):
                    mixture = base[:, :n_c] @ coef[:n_c]
                    if not np.any(mixture):
                        break
                    resid = data_w - design @ coef
                    pc, *_ = np.linalg.lstsq(mixture[:, None] * mult, resid, rcond=None)
                    factor = 1.0 + mult @ pc
                    design = np.concatenate([base[:, :n_c] * factor[:, None], base[:, n_c:]], axis=1)
                    coef, c2 = _solve(design, data_w, 0, n_nonneg)
        chi2[i] = c2
        coefficients[i] = coef / scale
        columns_at[i] = (names, line_cols)
    if not np.isfinite(chi2).any():
        return None
    # null: polynomial only
    c0, *_ = np.linalg.lstsq(poly, data_w, rcond=None) if poly.shape[1] else (np.zeros(0),)
    resid0 = data_w - (poly @ c0 if poly.shape[1] else 0.0)
    chi2_null = float(resid0 @ resid0)
    best, delta_runner, z_runner = _best_and_runner(grid, chi2, separation_kms)
    names, line_cols = columns_at[best]
    coef = coefficients[best]
    # continuum-only refit at the best redshift -> evidence added by the lines
    cont = cube.columns[best][keep] * weight[:, None]
    design_c = np.concatenate([cont, poly], axis=1)
    scale_c = np.sqrt(np.mean(design_c**2, axis=0)); scale_c = np.where(scale_c > 0, scale_c, 1.0)
    if nonnegative:
        _, chi2_cont = _solve(design_c / scale_c, data_w, 0, n_c)
    else:
        cc, *_ = np.linalg.lstsq(design_c / scale_c, data_w, rcond=None)
        r = data_w - (design_c / scale_c) @ cc
        chi2_cont = float(r @ r)
    # formal line errors from the joint design's normal matrix
    amplitudes, snr = {}, {}
    if line_cols:
        design = np.concatenate([cont, np.column_stack(line_cols), poly], axis=1)
        try:
            cov = np.linalg.pinv(design.T @ design, rcond=1e-12)
            for j, name in enumerate(names):
                k = n_c + j
                amplitudes[name] = float(coef[k])
                sigma = float(np.sqrt(max(cov[k, k], 0.0)))
                snr[name] = float(coef[k] / sigma) if sigma > 0 else float("nan")
        except np.linalg.LinAlgError:
            pass
    result = JointScanResult(
        z=float(grid[best]), chi2=float(chi2[best]), chi2_null=chi2_null, delta_chi2_runner_up=delta_runner,
        z_runner_up=z_runner, n_pixels=int(keep.sum()), n_parameters=n_c + len(line_cols) + poly.shape[1],
        kind=cube.kind, grid_z=grid, grid_chi2=chi2, coefficients=coef,
        delta_chi2_lines=float(chi2_cont - chi2[best]), line_amplitudes=amplitudes, line_snr=snr,
        n_line_templates=len(lines),
    )
    if z_prior is not None and np.isfinite(z_prior):
        penalised = chi2 + prior_penalty(grid, z_prior, prior_sigma, prior_outlier_fraction)
        pbest, pdelta, _ = _best_and_runner(grid, penalised, separation_kms)
        result = JointScanResult(**{**result.__dict__, "z_prior": float(grid[pbest]), "delta_chi2_runner_up_prior": pdelta})
    return result


__all__ = ["JointScanResult", "joint_scan", "line_templates", "DEFAULT_LINE_TEMPLATES"]
