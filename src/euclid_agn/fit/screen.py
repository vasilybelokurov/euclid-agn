"""Stage 1: cheap screening of every usable combined spectrum.

Cost is the design constraint.  Q1 holds 4.3 million extracted spectra, and a
blind redshift scan at the instrumental resolution puts thousands of hypotheses
on each one.  Running the full alternating narrow-line fit at every hypothesis
would cost seconds per object and is therefore impossible at survey scale.

So Stage 1 is itself two-tier:

**Tier A - matched filter.**  The continuum basis does not depend on redshift,
so it is whitened and orthonormalised *once* per spectrum.  Every line
hypothesis is then tested by projecting its basis orthogonal to the continuum
span and taking the closed-form optimum

.. math:: \\Delta\\chi^2 = \\frac{(b_\\perp^T y_\\perp)^2}{b_\\perp^T b_\\perp}
          \\quad\\text{for}\\quad b_\\perp^T y_\\perp > 0,

which costs a few dot products instead of a constrained solve.  Projecting out
the continuum is not an optimisation detail: a broad Gaussian overlaps the
continuum basis badly, and a matched filter that ignores that overlap
manufactures broad-line detections out of continuum mismatch.

**Tier B - full fit.**  Only the best few hypotheses get the real M0/M1
treatment from :mod:`euclid_agn.models.forward`, with the non-parametric narrow
profile and the non-negativity constraints.

Every number that leaves this module is a *statistic*.  The measured noise
inflation (:mod:`euclid_agn.pipeline.noise`) is carried alongside it so the
calibration stage can use it, but nothing here is converted into a significance.

Choosing which hypotheses to refine
-----------------------------------
Ranking by the broad-line gain alone gets the redshift wrong, and the failure is
not subtle.  VERIFIED on a synthetic object at z = 1.2 with a broad H-alpha:
H-alpha at z = 1.2 falls at 14442 A and Pa-beta at z = 0.1264 falls at 14458 A,
**1.15 pixels apart**.  On the broad gain alone the true redshift did not reach
the top five, beaten by a single-line Pa-beta identification of the same
feature.  Ranking by narrow + broad evidence put z = 1.2009 first, because the
H-alpha system contributes six lines and the Pa-beta system only one.

Redshift is therefore selected using *all* the line evidence, while the broad
gain remains the AGN statistic.  Both rankings are refined - the union of the
best by total evidence and the best by broad gain - so an object with a strong
BLR and no narrow lines is not thrown away by a rule designed for the opposite
case.  ``rank_alternatives`` records how much better the winner is than the best
hypothesis of a *different* line system, which is the honest measure of how
ambiguous an identification is.

Comparing line systems with different numbers of lines
------------------------------------------------------
The line systems do not have the same number of members, and every visible
member is a free non-negative amplitude.  Ranking hypotheses on raw Delta
chi-squared therefore hands the win to whichever system has the most lines, and
it does so overwhelmingly: in the first blind-recovery experiment on 659 real
spectra, 75 per cent of objects were assigned ``hbeta_oiii`` (6.6 visible lines
on average) and the agreement with Euclid's own SPE redshift was 2.9 per cent.

The size of the bias was measured rather than assumed.  Taking the best Delta
chi-squared *within each system* over a full blind scan of 150 real spectra -
most of which have no strong lines, so this is close to a null - the median of
that per-object maximum grows linearly with the number of free components ``k``::

    median max Delta chi2  =  7.0 + 6.6 k       (k = 1 ... 8)

The slope is far larger than the naive AIC penalty of 2 per parameter because
the statistic is a *maximum* over ~900 hypotheses, and extreme values of a
k-component statistic grow with k.  ``ScreenSettings.component_penalty``
defaults to that measured slope, and ranking uses

    delta_chi2_penalised = delta_chi2_total - component_penalty * n_components

Raw and penalised values are both recorded.


How wide a broad line can be and still be measurable
----------------------------------------------------
A very broad Gaussian is degenerate with continuum curvature, and the first
pilot run on real Q1 spectra showed the consequence: every top-ranked candidate
railed at the widest width in the scan grid, and the median spectrum "preferred"
a broad component.  Those are not detections, they are the continuum model
being repaired by a line.

The degeneracy is measurable.  :func:`continuum_orthogonality` returns the
fraction of a whitened broad-line basis vector that survives projection
orthogonal to the continuum span; a value near zero means the width carries
almost no independent information.  VERIFIED on the real Q1 grid (448 fitted
pixels):

======================  =====  =====  =====  =====
sigma (FWHM) km/s       k=6    k=12   k=25   k=50
======================  =====  =====  =====  =====
300 (706)               0.95   0.92   0.86   0.73
600 (1413)              0.93   0.86   0.78   0.56
1200 (2826)             0.86   0.74   0.58   0.23
2500 (5887)             0.70   0.47   0.21   0.01
5000 (11774)            0.42   0.13   0.01   0.00
8000 (18839)            0.19   0.02   0.00   0.00
======================  =====  =====  =====  =====

So with a 12-knot continuum, widths beyond about sigma = 2500 km/s
(FWHM ~ 5900 km/s) are not independently measurable, and much of the classical
BLR width range is intrinsically degenerate with the continuum in this grism.
Widths failing ``min_broad_orthogonality`` are excluded from the scan - testing
them only adds trials and manufactures the failure above - and the orthogonality
of the surviving best width is recorded on every row, because it is a selection
-function axis in its own right.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from euclid_agn.constants import (
    C_KMS,
    RGS_SCIENCE_WMAX_ANGSTROM,
    RGS_SCIENCE_WMIN_ANGSTROM,
)
from euclid_agn.fit.hypotheses import RedshiftHypothesis
from euclid_agn.models.broad import BroadFamily, sigma_grid
from euclid_agn.models.forward import HypothesisFit, fit_hypothesis
from euclid_agn.models.line_catalog import BY_NAME, BY_SYSTEM, LineSystem, visible_systems
from euclid_agn.models.narrow import NarrowSystem, velocity_grid
from euclid_agn.models.templates import template_column, templates_for
from euclid_agn.numerics import blas_safe
from euclid_agn.spectra.continuum import bspline_basis
from euclid_agn.spectra.outliers import isolated_outliers

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScreenSettings:
    """Everything that can change a Stage-1 number."""

    n_knots: int = 12
    narrow_sigma_kms: float = 200.0
    broad_sigma_kms: tuple[float, ...] = field(
        default_factory=lambda: tuple(sigma_grid(400.0, 6000.0, 6))
    )
    broad_velocity_kms: tuple[float, ...] = (0.0,)
    wavelength_min: float = RGS_SCIENCE_WMIN_ANGSTROM
    wavelength_max: float = RGS_SCIENCE_WMAX_ANGSTROM
    min_usable_pixels: int = 100
    n_refine: int = 3
    #: Minimum fraction of a broad-line basis vector that must survive
    #: projection orthogonal to the continuum for its width to be scanned.
    min_broad_orthogonality: float = 0.5
    #: Minimum fraction of a broad line's flux that must fall on covered
    #: pixels.  Guards against edge artefacts absorbed by a truncated wing.
    min_broad_containment: float = 0.8
    #: Penalty per free line amplitude when comparing hypotheses, in units of
    #: chi-squared.  The default is the measured slope of the null maximum
    #: against the number of components; see the module docstring.
    component_penalty: float = 6.6
    #: Photometric-redshift prior used in ranking: a mixture of a Gaussian of
    #: width ``phz_prior_sigma`` in dz/(1+z) and a uniform component carrying
    #: ``phz_outlier_fraction`` of the probability over ``phz_prior_range``.
    #: The penalty is -2 ln of that mixture relative to its peak, so it is
    #: naturally capped: with sigma 0.05, outlier fraction 0.13 and range 5 the
    #: cap is about 11 in Delta chi-squared units.  That is the honest weight
    #: of a photometric redshift - it settles moderate ambiguities and cannot
    #: overturn a strong data preference.  VERIFIED on 114 DESI galaxies at
    #: 0.9 < z < 1.8: PHZ |dz/(1+z)| median 0.026, 87 per cent within 0.1.
    phz_prior_sigma: float | None = 0.05
    phz_outlier_fraction: float = 0.13
    phz_prior_range: float = 5.0
    #: Lines (template or broad) whose centre lies within this many pixels of
    #: either end of the covered range are not tested: the outermost pixels of
    #: the science window carry edge artefacts that a line placed on them fits
    #: perfectly.
    edge_margin_pixels: float = 4.0
    #: Robust rejection of narrow outlier pixels the archive mask leaves in
    #: (see :mod:`euclid_agn.spectra.outliers`).  ``0`` disables it.
    outlier_threshold: float = 5.0
    outlier_max_width: int = 2
    #: Which statistic ranks hypotheses.  ``"template"`` uses the best
    #: fixed-ratio template (one free amplitude, see
    #: :mod:`euclid_agn.models.templates`) plus the broad gain;
    #: ``"penalised"`` uses the free-amplitude total minus the component
    #: penalty.  Measured on real Q1 spectra the free-amplitude statistic
    #: agrees with SPE 2.6% of the time; see the module docstring.
    rank_by: str = "template"
    #: Local redshift refinement: after the coarse scan, the best candidates
    #: are re-scanned on a fine grid of +/- ``local_half_width_kms`` in steps
    #: of ``local_step_kms``.  Two line systems can map the same observed
    #: features onto each other to within a pixel or two at the coarse step
    #: (H-beta/[O III] at z = 1.965 against H-alpha/[S II] at z = 1.2 differ by
    #: 1.6 pixels); the wavelength *ratios* discriminate them only once each
    #: is placed at its own best redshift.
    local_half_width_kms: float = 700.0
    local_step_kms: float = 60.0
    n_local: int = 6
    #: Optional per-system null offsets (median of the per-object maximum
    #: Delta chi-squared within each system, measured on line-free spectra).
    #: When given they replace the linear component penalty, which is a fit to
    #: the same numbers and loses the per-system detail.
    system_null_offsets: dict[str, float] | None = None
    #: Extent of the shared narrow-line velocity profile.  This is the
    #: model's *definition* of narrow: anything the profile can represent is
    #: narrow, anything wider must go to the broad component.  VERIFIED on real
    #: spectra with injected broad lines: at +/-1000 km/s the M0 profile widened
    #: to 450-650 km/s and absorbed a 700 km/s injection entirely (Delta chi2 =
    #: 0); at +/-400 it stayed at the LSF-limited ~265 km/s.
    narrow_velocity_half_width_kms: float = 400.0
    narrow_velocity_step_kms: float = 200.0
    narrow_smoothness: float = 1.0
    require_line_in_range: bool = True


@dataclass
class ProjectedSpectrum:
    """A spectrum whitened and projected orthogonal to the continuum basis.

    ``basis`` is an orthonormal basis of the whitened continuum span, computed
    once; ``residual`` is the whitened data with that span removed.  Projecting
    any trial line basis through :meth:`project` gives the part of it the
    continuum cannot absorb.
    """

    wavelength: np.ndarray
    weight: np.ndarray  # 1/sigma
    basis: np.ndarray  # (n_pixels, n_continuum) orthonormal
    residual: np.ndarray  # whitened, continuum-projected data
    bin_width: float
    lsf_sigma: float
    chi2_continuum: float
    edges: np.ndarray | None = None  # (n_kept, 2) true lower/upper edge of each kept pixel
    n_outliers: int = 0  # narrow outlier pixels rejected before fitting

    def __post_init__(self) -> None:
        if self.edges is None:
            # Fallback for callers that build a ProjectedSpectrum by hand:
            # assume a contiguous grid of the stated bin width.
            half = 0.5 * self.bin_width
            self.edges = np.column_stack([self.wavelength - half, self.wavelength + half])
        self.widths = self.edges[:, 1] - self.edges[:, 0]

    def line_column(self, centre: float, sigma_angstrom: float) -> np.ndarray:
        """Unit-flux Gaussian integrated over each *kept* pixel's true edges.

        The edges come from the full archive grid, so a masked gap between two
        kept pixels contributes nothing: a line falling in the gap gets no
        flux on the kept pixels and fails containment, instead of the gap's
        width being assigned to its neighbours.  VERIFIED failure before this
        fix: a broad column drawn as a straight line across a 60-pixel gap, on
        which a Delta chi-squared of 820 was built.
        """
        from scipy.special import erf

        lower = 0.5 * (1.0 + erf((self.edges[:, 0] - centre) / (np.sqrt(2.0) * sigma_angstrom)))
        upper = 0.5 * (1.0 + erf((self.edges[:, 1] - centre) / (np.sqrt(2.0) * sigma_angstrom)))
        return (upper - lower) / self.widths

    @blas_safe
    def project(self, columns: np.ndarray) -> np.ndarray:
        """Whiten trial columns and remove their continuum component."""
        whitened = np.asarray(columns, dtype=np.float64)
        if whitened.ndim == 1:
            whitened = whitened[:, None]
        whitened = whitened * self.weight[:, None]
        return whitened - self.basis @ (self.basis.T @ whitened)


@blas_safe
def prepare(spectrum, settings: ScreenSettings = ScreenSettings()) -> ProjectedSpectrum | None:
    """Whiten a spectrum, fit and project out the continuum.

    Returns ``None`` when there are too few usable pixels in the science window
    to screen the object at all - that is a parent-sample fact to be recorded,
    not an error.
    """
    usable = spectrum.usable()
    window = (spectrum.wavelength >= settings.wavelength_min) & (
        spectrum.wavelength <= settings.wavelength_max
    )
    keep = usable & window
    n_outliers = 0
    if settings.outlier_threshold > 0:
        outliers = isolated_outliers(
            spectrum.flux,
            spectrum.variance,
            keep,
            threshold=settings.outlier_threshold,
            max_width=settings.outlier_max_width,
            lsf_sigma_pixels=float(spectrum.lsf_sigma) / float(spectrum.bin_width),
        )
        n_outliers = int(np.count_nonzero(outliers))
        keep &= ~outliers
    if np.count_nonzero(keep) < settings.min_usable_pixels:
        return None

    wavelength = spectrum.wavelength[keep]
    flux = spectrum.flux[keep]
    weight = 1.0 / np.sqrt(spectrum.variance[keep])
    # True pixel edges from the full grid, then subset: gaps stay gaps.
    from euclid_agn.spectra.lsf import pixel_edges

    full_edges = pixel_edges(spectrum.wavelength, spectrum.bin_width)
    edges = np.column_stack([full_edges[:-1][keep], full_edges[1:][keep]])

    design, _ = bspline_basis(wavelength, n_knots=settings.n_knots)
    whitened = design * weight[:, None]
    basis, _ = np.linalg.qr(whitened)
    data = flux * weight
    residual = data - basis @ (basis.T @ data)
    return ProjectedSpectrum(
        wavelength=wavelength,
        weight=weight,
        basis=basis,
        residual=residual,
        bin_width=float(spectrum.bin_width),
        lsf_sigma=float(spectrum.lsf_sigma),
        chi2_continuum=float(residual @ residual),
        edges=edges,
        n_outliers=n_outliers,
    )


@blas_safe
def matched_filter(projected: ProjectedSpectrum, columns: np.ndarray) -> tuple[float, np.ndarray]:
    """Best non-negative joint amplitude fit of ``columns`` to the residual.

    Returns ``(delta_chi2, amplitudes)``.  For a single column this is the
    closed-form matched filter; for several it is a small normal-equation solve
    with a single non-negativity pass, which is enough at screening precision
    and orders of magnitude cheaper than a full bounded solve.
    """
    perpendicular = projected.project(columns)
    normal = perpendicular.T @ perpendicular
    rhs = perpendicular.T @ projected.residual
    try:
        amplitudes = np.linalg.solve(normal, rhs)
    except np.linalg.LinAlgError:
        return 0.0, np.zeros(perpendicular.shape[1])
    if np.any(amplitudes < 0):
        keep = amplitudes > 0
        amplitudes = np.zeros_like(amplitudes)
        if np.any(keep):
            sub = perpendicular[:, keep]
            try:
                values = np.linalg.solve(sub.T @ sub, sub.T @ projected.residual)
            except np.linalg.LinAlgError:
                return 0.0, amplitudes
            values = np.clip(values, 0.0, None)
            amplitudes[keep] = values
    model = perpendicular @ amplitudes
    statistic = float(2.0 * (model @ projected.residual) - model @ model)
    return max(statistic, 0.0), amplitudes


def narrow_columns(
    projected: ProjectedSpectrum, system: LineSystem, z: float, sigma_kms: float
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Fixed-width Gaussian columns for the narrow lines visible at ``z``."""
    from euclid_agn.spectra.lsf import effective_sigma, sigma_kms_to_angstrom

    columns, names = [], []
    for line in system.lines():
        centre = line.rest * (1.0 + z)
        if not (projected.wavelength[0] <= centre <= projected.wavelength[-1]):
            continue
        width = effective_sigma(sigma_kms_to_angstrom(sigma_kms, centre), projected.lsf_sigma)
        columns.append(projected.line_column(centre, width))
        names.append(line.name)
    if not columns:
        return np.zeros((projected.wavelength.size, 0)), ()
    return np.column_stack(columns), tuple(names)


@blas_safe
def continuum_orthogonality(projected: ProjectedSpectrum, column: np.ndarray) -> float:
    """Fraction of a trial column's norm that the continuum cannot absorb.

    1.0 means the component is completely independent of the continuum model;
    0.0 means the continuum can reproduce it exactly, so any amplitude fitted
    to it is measuring continuum mismatch rather than a line.
    """
    whitened = np.asarray(column, dtype=np.float64) * projected.weight
    norm = float(np.linalg.norm(whitened))
    if norm <= 0:
        return 0.0
    return float(np.linalg.norm(projected.project(column)[:, 0]) / norm)


def identifiable_sigmas(
    projected: ProjectedSpectrum,
    line_name: str,
    z: float,
    settings: ScreenSettings,
) -> list[tuple[float, float]]:
    """Broad widths whose profile is distinguishable from the continuum.

    Returns ``(sigma_kms, orthogonality, column)`` for the widths that pass
    ``settings.min_broad_orthogonality``; the zero-velocity column is returned
    so the caller does not build it a second time.
    """
    out = []
    for sigma in settings.broad_sigma_kms:
        column = broad_column(
            projected,
            line_name,
            z,
            sigma,
            min_containment=settings.min_broad_containment,
            edge_margin_pixels=settings.edge_margin_pixels,
        )
        if column is None:
            continue
        orthogonality = continuum_orthogonality(projected, column)
        if orthogonality >= settings.min_broad_orthogonality:
            out.append((float(sigma), orthogonality, column))
    return out


def broad_column(
    projected: ProjectedSpectrum,
    line_name: str,
    z: float,
    sigma_kms: float,
    velocity_kms: float = 0.0,
    min_containment: float = 0.0,
    edge_margin_pixels: float = 0.0,
) -> np.ndarray | None:
    """Single broad column, or ``None`` if too little of it is observed.

    ``min_containment`` is the fraction of the line's flux that must fall on
    covered pixels; see
    :meth:`euclid_agn.models.broad.BroadComponent.contained_fraction`.
    """
    from euclid_agn.models.broad import BroadComponent

    component = BroadComponent(
        line_name=line_name,
        z=z,
        sigma_kms=sigma_kms,
        lsf_sigma=projected.lsf_sigma,
        velocity_kms=velocity_kms,
    )
    if not component.in_range(projected.wavelength, n_sigma=1.0):
        return None
    margin = edge_margin_pixels * projected.bin_width
    if not (projected.wavelength[0] + margin <= component.centre <= projected.wavelength[-1] - margin):
        return None
    column = projected.line_column(component.centre, component.observed_sigma_angstrom)
    if min_containment > 0.0 and float(np.sum(column * projected.widths)) < min_containment:
        return None
    return column


def quick_scan(
    spectrum,
    hypotheses: list[RedshiftHypothesis],
    settings: ScreenSettings = ScreenSettings(),
) -> pd.DataFrame:
    """Tier A: matched-filter statistics for every hypothesis.

    One row per (hypothesis, line system).  ``delta_chi2_broad`` is the gain of
    the best broad component *over* the narrow-only model at the same redshift.
    """
    projected = prepare(spectrum, settings)
    if projected is None:
        return pd.DataFrame()

    rows = []
    for hypothesis in hypotheses:
        systems = (
            [BY_SYSTEM[hypothesis.system]]
            if hypothesis.system in BY_SYSTEM
            else list(
                visible_systems(
                    hypothesis.z,
                    projected.wavelength[0],
                    projected.wavelength[-1],
                    min_lines=1,
                )
            )
        )
        for system in systems:
            columns, names = narrow_columns(
                projected, system, hypothesis.z, settings.narrow_sigma_kms
            )
            if columns.shape[1] == 0 and settings.require_line_in_range:
                continue
            narrow_statistic, narrow_amplitudes = (
                matched_filter(projected, columns) if columns.shape[1] else (0.0, np.zeros(0))
            )
            template_statistic, template_name, template_lines = 0.0, "", 0
            template_best_column = None
            for template in templates_for(system):
                column, n_lines = template_column(
                    projected,
                    template,
                    hypothesis.z,
                    settings.narrow_sigma_kms,
                    min_containment=settings.min_broad_containment,
                    edge_margin=settings.edge_margin_pixels * projected.bin_width,
                )
                if column is None:
                    continue
                statistic, _ = matched_filter(projected, column[:, None])
                if statistic > template_statistic:
                    template_statistic, template_name, template_lines = (
                        statistic,
                        template.name,
                        n_lines,
                    )
                    template_best_column = column
            # Identification statistic: template and broad component fitted
            # *jointly*, so flux shared between them is counted once.  Summing
            # two separately optimised statistics let a doublet template and a
            # broad Gaussian both claim the same blob.
            identification = template_statistic

            best = {
                "delta_chi2_broad": 0.0,
                "sigma": np.nan,
                "velocity": np.nan,
                "flux": 0.0,
                "orthogonality": np.nan,
            }
            sigma_max = 0.0
            for line_name in system.broad_members:
                allowed = identifiable_sigmas(projected, line_name, hypothesis.z, settings)
                if allowed:
                    sigma_max = max(sigma_max, max(s for s, _, _ in allowed))
                for sigma, orthogonality, zero_velocity_column in allowed:
                    for velocity in settings.broad_velocity_kms:
                        column = (
                            zero_velocity_column
                            if velocity == 0.0
                            else broad_column(
                                projected,
                                line_name,
                                hypothesis.z,
                                sigma,
                                velocity,
                                min_containment=settings.min_broad_containment,
                                edge_margin_pixels=settings.edge_margin_pixels,
                            )
                        )
                        if column is None:
                            continue
                        joint = (
                            np.column_stack([columns, column])
                            if columns.shape[1]
                            else column[:, None]
                        )
                        statistic, amplitudes = matched_filter(projected, joint)
                        gain = statistic - narrow_statistic
                        joint_identification = (
                            np.column_stack([template_best_column, column])
                            if template_best_column is not None
                            else column[:, None]
                        )
                        identification = max(
                            identification, matched_filter(projected, joint_identification)[0]
                        )
                        if gain > best["delta_chi2_broad"]:
                            best = {
                                "delta_chi2_broad": gain,
                                "sigma": sigma,
                                "velocity": velocity,
                                "flux": float(amplitudes[-1]),
                                "line": line_name,
                                "orthogonality": orthogonality,
                            }
            rows.append(
                {
                    "z": hypothesis.z,
                    "origin": hypothesis.origin,
                    "system": system.name,
                    "n_narrow_lines": int(columns.shape[1]),
                    "delta_chi2_narrow": narrow_statistic,
                    "narrow_flux_max": float(np.max(narrow_amplitudes))
                    if narrow_amplitudes.size
                    else 0.0,
                    "delta_chi2_broad": best["delta_chi2_broad"],
                    "broad_sigma_kms": best["sigma"],
                    "broad_velocity_kms": best["velocity"],
                    "broad_flux": best["flux"],
                    "broad_line": best.get("line", ""),
                    "broad_continuum_orthogonality": best["orthogonality"],
                    "broad_sigma_max_identifiable_kms": sigma_max,
                    "n_components": int(columns.shape[1])
                    + (1 if best["delta_chi2_broad"] > 0 else 0),
                    "delta_chi2_template": template_statistic,
                    "delta_chi2_identification": identification,
                    "template": template_name,
                    "n_template_lines": template_lines,
                }
            )
    table = pd.DataFrame(rows)
    if not table.empty:
        table["delta_chi2_total"] = table["delta_chi2_narrow"] + table["delta_chi2_broad"]
        if settings.system_null_offsets:
            offsets = table["system"].map(settings.system_null_offsets)
            fallback = settings.component_penalty * table["n_components"]
            table["delta_chi2_penalised"] = table["delta_chi2_total"] - offsets.fillna(fallback)
        else:
            table["delta_chi2_penalised"] = (
                table["delta_chi2_total"] - settings.component_penalty * table["n_components"]
            )
        # delta_chi2_identification is the joint template + broad fit computed
        # above; it ranks hypotheses when rank_by is "template".  The
        # free-amplitude columns stay for measurement.
        table["rank_statistic"] = (
            table["delta_chi2_identification"]
            if settings.rank_by == "template"
            else table["delta_chi2_penalised"]
        )
    return table


def refine(
    spectrum,
    z: float,
    system_name: str,
    broad_sigma_kms: float,
    broad_velocity_kms: float = 0.0,
    settings: ScreenSettings = ScreenSettings(),
) -> HypothesisFit | None:
    """Tier B: the full M0/M1 fit at one hypothesis."""
    projected = prepare(spectrum, settings)
    if projected is None:
        return None
    system = BY_SYSTEM[system_name]
    visible = [
        line.name
        for line in system.lines()
        if projected.wavelength[0] <= line.rest * (1.0 + z) <= projected.wavelength[-1]
    ]
    if not visible:
        return None
    narrow = NarrowSystem(
        line_names=tuple(visible),
        z=z,
        lsf_sigma=projected.lsf_sigma,
        velocities=velocity_grid(
            settings.narrow_velocity_half_width_kms, settings.narrow_velocity_step_kms
        ),
    )
    broad_names = tuple(n for n in system.broad_members if n in visible)
    broad = (
        BroadFamily(
            line_names=broad_names,
            z=z,
            sigma_kms=broad_sigma_kms,
            lsf_sigma=projected.lsf_sigma,
            velocity_kms=broad_velocity_kms,
        )
        if broad_names
        else None
    )
    keep = np.isin(spectrum.wavelength, projected.wavelength)
    return fit_hypothesis(
        spectrum.wavelength[keep],
        spectrum.flux[keep],
        spectrum.variance[keep],
        narrow,
        broad,
        n_knots=settings.n_knots,
        bin_width=spectrum.bin_width,
        smoothness=settings.narrow_smoothness,
    )


def apply_redshift_prior(
    scan: pd.DataFrame, z_prior: float | None, settings: ScreenSettings
) -> pd.DataFrame:
    """Subtract a capped Gaussian photometric-redshift penalty from the ranking.

    ``rank_statistic`` keeps the data-only value; ``rank_statistic_prior`` is
    what ranking uses when a prior is available.  The winning origin, the
    prior redshift and the penalty applied are all recorded, so any candidate
    can be traced to whether the prior decided it.
    """
    if scan.empty:
        return scan
    scan = scan.copy()
    if z_prior is None or settings.phz_prior_sigma is None or not np.isfinite(z_prior):
        scan["prior_penalty"] = 0.0
        scan["rank_statistic_prior"] = scan["rank_statistic"]
        return scan
    sigma = settings.phz_prior_sigma
    f = min(max(settings.phz_outlier_fraction, 1e-6), 1.0 - 1e-6)
    pull = (scan["z"] - z_prior) / (1.0 + z_prior) / sigma
    gaussian = (1.0 - f) * np.exp(-0.5 * pull**2) / (sigma * np.sqrt(2.0 * np.pi))
    uniform = f / settings.phz_prior_range
    peak = (1.0 - f) / (sigma * np.sqrt(2.0 * np.pi)) + uniform
    scan["prior_penalty"] = -2.0 * np.log((gaussian + uniform) / peak)
    scan["rank_statistic_prior"] = scan["rank_statistic"] - scan["prior_penalty"]
    return scan


def local_redshift_grid(z: float, half_width_kms: float, step_kms: float) -> np.ndarray:
    """Fine redshift grid around ``z``, uniform in velocity."""
    n = int(np.floor(half_width_kms / step_kms))
    offsets = np.arange(-n, n + 1) * step_kms / C_KMS
    return (1.0 + z) * (1.0 + offsets) - 1.0


def refine_redshifts_locally(
    spectrum,
    scan: pd.DataFrame,
    settings: ScreenSettings,
    rank_column: str = "rank_statistic",
) -> pd.DataFrame:
    """Re-scan the best coarse candidates on a fine local grid.

    Takes the best ``settings.n_local`` rows of ``scan`` *by system* - one per
    system, so every competing identification gets the same chance - and
    returns the coarse scan with those rows replaced by their locally refined
    best.  ``origin`` is preserved so provenance survives the refinement.
    """
    if scan.empty or settings.n_local <= 0:
        return scan
    best_per_system = scan.loc[scan.groupby("system")[rank_column].idxmax()]
    top = best_per_system.nlargest(settings.n_local, rank_column)
    replacements = []
    for index, row in top.iterrows():
        grid = local_redshift_grid(
            float(row["z"]), settings.local_half_width_kms, settings.local_step_kms
        )
        hypotheses = [
            RedshiftHypothesis(float(z), str(row["origin"]), str(row["system"])) for z in grid
        ]
        local = quick_scan(spectrum, hypotheses, settings)
        if local.empty:
            continue
        best = local.loc[local[rank_column].idxmax()].copy()
        best["coarse_z"] = row["z"]
        best.name = index
        replacements.append(best)
    if not replacements:
        return scan
    refined = scan.copy()
    refined["coarse_z"] = refined["z"]
    for best in replacements:
        for column, value in best.items():
            refined.loc[best.name, column] = value
    return refined


def winning_line_edge_pixels(projected, system: str, template_name: str, z: float) -> float:
    """Distance, in pixels, from the nearest template line of the winner to the
    nearest end of the covered range.  Small values mean the identification
    rests on a line at the edge."""
    if projected is None:
        return float("nan")
    candidates = [t for t in templates_for(system) if t.name == template_name] or list(templates_for(system))
    if not candidates:
        return float("nan")
    lo, hi = float(projected.wavelength[0]), float(projected.wavelength[-1])
    distances = []
    for name in candidates[0].ratios:
        centre = BY_NAME[name].rest * (1.0 + z)
        if lo <= centre <= hi:
            distances.append(min(centre - lo, hi - centre) / projected.bin_width)
    return float(min(distances)) if distances else float("nan")


def rank_alternatives(scan: pd.DataFrame, winner) -> dict[str, float]:
    """How much better the winning hypothesis is than a different identification.

    ``delta_chi2_over_other_system`` compares the winner with the best
    hypothesis whose line system differs, and
    ``velocity_to_best_alternative_kms`` says how far away in velocity it sits.
    A small margin means the redshift is ambiguous and the candidate should not
    be reported as if it were settled.
    """
    others = scan[scan["system"] != winner["system"]]
    if others.empty:
        return {
            "delta_chi2_over_other_system": float("inf"),
            "velocity_to_best_alternative_kms": float("nan"),
            "best_alternative_system": "",
            "best_alternative_z": float("nan"),
        }
    best = others.loc[others["rank_statistic"].idxmax()]
    z_w, z_a = float(winner["z"]), float(best["z"])
    return {
        "delta_chi2_over_other_system": float(
            winner["rank_statistic"] - best["rank_statistic"]
        ),
        "velocity_to_best_alternative_kms": float(
            C_KMS * abs(z_w - z_a) / (1.0 + 0.5 * (z_w + z_a))
        ),
        "best_alternative_system": str(best["system"]),
        "best_alternative_z": z_a,
    }


def select_for_refinement(
    scan: pd.DataFrame, n_refine: int, rank_column: str = "rank_statistic"
) -> pd.DataFrame:
    """Union of the best hypotheses by total evidence and by broad gain.

    Total evidence picks the redshift; broad gain protects objects whose only
    signal is a BLR.
    """
    if scan.empty or n_refine <= 0:
        return scan.head(0)
    column = rank_column if rank_column in scan.columns else "rank_statistic"
    by_total = scan.nlargest(n_refine, column)
    by_broad = scan.nlargest(n_refine, "delta_chi2_broad")
    selected = pd.concat([by_total, by_broad]).drop_duplicates(subset=["z", "system"])
    selected = selected.assign(
        selected_by=[
            "total+broad"
            if (index in by_total.index and index in by_broad.index)
            else ("total" if index in by_total.index else "broad")
            for index in selected.index
        ]
    )
    return selected.sort_values(column, ascending=False)


def screen_spectrum(
    spectrum,
    hypotheses: list[RedshiftHypothesis],
    settings: ScreenSettings = ScreenSettings(),
    object_id: int = -1,
    noise_inflation: float = 1.0,
    context: dict | None = None,
    z_prior: float | None = None,
) -> pd.DataFrame:
    """Full Stage 1 for one spectrum: scan, then refine the best hypotheses.

    Returns one row per refined hypothesis, with the quick-scan statistic kept
    alongside the refined one so the two can be compared during calibration.
    """
    scan = quick_scan(spectrum, hypotheses, settings)
    if scan.empty:
        return pd.DataFrame()
    scan = scan.reset_index(drop=True)
    scan = refine_redshifts_locally(spectrum, scan, settings)
    scan = apply_redshift_prior(scan, z_prior, settings)
    projected_for_rows = prepare(spectrum, settings)
    selected = select_for_refinement(scan, settings.n_refine, rank_column="rank_statistic_prior")

    quality = spectrum.quality_metrics()
    rows = []
    for _, candidate in selected.iterrows():
        sigma = candidate["broad_sigma_kms"]
        fit = refine(
            spectrum,
            float(candidate["z"]),
            str(candidate["system"]),
            float(sigma) if np.isfinite(sigma) else settings.broad_sigma_kms[0],
            float(candidate["broad_velocity_kms"]) if np.isfinite(candidate["broad_velocity_kms"]) else 0.0,
            settings,
        )
        if fit is None:
            continue
        row: dict = {
            "object_id": object_id,
            "z_hypothesis_source": candidate["origin"],
            "system": candidate["system"],
            "selected_by": candidate["selected_by"],
            "n_narrow_lines": candidate["n_narrow_lines"],
            "quick_delta_chi2_narrow": candidate["delta_chi2_narrow"],
            "quick_delta_chi2_broad": candidate["delta_chi2_broad"],
            "quick_delta_chi2_total": candidate["delta_chi2_total"],
            "quick_delta_chi2_penalised": candidate["delta_chi2_penalised"],
            "quick_delta_chi2_identification": candidate["delta_chi2_identification"],
            "template": candidate["template"],
            "rank_by": settings.rank_by,
            "z_prior": z_prior if z_prior is not None else float("nan"),
            "prior_penalty": candidate.get("prior_penalty", 0.0),
            "n_components": candidate["n_components"],
            "noise_inflation": noise_inflation,
            "n_outlier_pixels": projected_for_rows.n_outliers if projected_for_rows else 0,
            "delta_chi2_effective": float(candidate["delta_chi2_broad"]) / noise_inflation**2,
            "broad_continuum_orthogonality": candidate["broad_continuum_orthogonality"],
            "broad_sigma_max_identifiable_kms": candidate["broad_sigma_max_identifiable_kms"],
        }
        row.update(rank_alternatives(scan, candidate))
        others = scan[scan["system"] != candidate["system"]]
        row["data_margin"] = (
            float(candidate["rank_statistic"] - others["rank_statistic"].max())
            if not others.empty
            else float("inf")
        )
        row["winning_line_edge_pixels"] = winning_line_edge_pixels(
            projected_for_rows, str(candidate["system"]), str(candidate["template"]), float(candidate["z"])
        )
        row.update(fit.summary())
        row["delta_chi2_refined_effective"] = row["delta_chi2"] / noise_inflation**2
        # A Delta chi2 from a model whose own reduced chi2 is far from one is
        # measuring model error, not a line. Recorded, never silently applied.
        denominator = row["n_pixels"] - row["dof_m0"]
        row["chi2_reduced_m0"] = (
            float(row["chi2_m0"] / denominator) if denominator > 0 else float("nan")
        )
        row["continuum_model_ok"] = bool(row["chi2_reduced_m0"] < 4.0)
        row.update({f"quality_{k}": v for k, v in quality.items()})
        if context:
            row.update(context)
        rows.append(row)
    from euclid_agn.fit.quality import add_zwarn

    return add_zwarn(pd.DataFrame(rows))


def null_offsets_from_table(table: pd.DataFrame, statistic: str = "median") -> dict[str, float]:
    """Per-system null offsets from a table of per-object maxima.

    ``table`` has one row per (object, system) with the best total statistic
    that system reached anywhere in a blind scan; see the null calibration in
    JOURNAL.md.  The median over objects is the offset.
    """
    grouped = table.groupby("system")["max_total"]
    return {k: float(v) for k, v in getattr(grouped, statistic)().items()}
