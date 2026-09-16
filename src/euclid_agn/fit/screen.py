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
from euclid_agn.models.line_catalog import BY_SYSTEM, LineSystem, visible_systems
from euclid_agn.models.narrow import NarrowSystem, velocity_grid
from euclid_agn.numerics import blas_safe
from euclid_agn.spectra.continuum import bspline_basis

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
    #: Optional per-system null offsets (median of the per-object maximum
    #: Delta chi-squared within each system, measured on line-free spectra).
    #: When given they replace the linear component penalty, which is a fit to
    #: the same numbers and loses the per-system detail.
    system_null_offsets: dict[str, float] | None = None
    narrow_velocity_half_width_kms: float = 1000.0
    narrow_velocity_step_kms: float = 250.0
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
    edges: np.ndarray | None = None  # pixel edges, computed once in prepare()

    def __post_init__(self) -> None:
        if self.edges is None:
            from euclid_agn.spectra.lsf import pixel_edges

            self.edges = pixel_edges(self.wavelength, self.bin_width)
        self.widths = np.diff(self.edges)

    def line_column(self, centre: float, sigma_angstrom: float) -> np.ndarray:
        """Unit-flux, pixel-integrated Gaussian on the cached pixel edges.

        Identical to :func:`euclid_agn.spectra.lsf.gaussian_pixel_integral`
        but without recomputing the edges for every one of the tens of
        thousands of columns a blind scan builds.
        """
        from scipy.special import erf

        z = (self.edges - centre) / (np.sqrt(2.0) * sigma_angstrom)
        return np.diff(0.5 * (1.0 + erf(z))) / self.widths

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
    if np.count_nonzero(keep) < settings.min_usable_pixels:
        return None

    wavelength = spectrum.wavelength[keep]
    flux = spectrum.flux[keep]
    weight = 1.0 / np.sqrt(spectrum.variance[keep])

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
            projected, line_name, z, sigma, min_containment=settings.min_broad_containment
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
    usable = spectrum.usable()
    window = (spectrum.wavelength >= settings.wavelength_min) & (
        spectrum.wavelength <= settings.wavelength_max
    )
    keep = usable & window
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
    best = others.loc[others["delta_chi2_penalised"].idxmax()]
    z_w, z_a = float(winner["z"]), float(best["z"])
    return {
        "delta_chi2_over_other_system": float(
            winner["delta_chi2_penalised"] - best["delta_chi2_penalised"]
        ),
        "velocity_to_best_alternative_kms": float(
            C_KMS * abs(z_w - z_a) / (1.0 + 0.5 * (z_w + z_a))
        ),
        "best_alternative_system": str(best["system"]),
        "best_alternative_z": z_a,
    }


def select_for_refinement(scan: pd.DataFrame, n_refine: int) -> pd.DataFrame:
    """Union of the best hypotheses by total evidence and by broad gain.

    Total evidence picks the redshift; broad gain protects objects whose only
    signal is a BLR.
    """
    if scan.empty or n_refine <= 0:
        return scan.head(0)
    by_total = scan.nlargest(n_refine, "delta_chi2_penalised")
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
    return selected.sort_values("delta_chi2_penalised", ascending=False)


def screen_spectrum(
    spectrum,
    hypotheses: list[RedshiftHypothesis],
    settings: ScreenSettings = ScreenSettings(),
    object_id: int = -1,
    noise_inflation: float = 1.0,
    context: dict | None = None,
) -> pd.DataFrame:
    """Full Stage 1 for one spectrum: scan, then refine the best hypotheses.

    Returns one row per refined hypothesis, with the quick-scan statistic kept
    alongside the refined one so the two can be compared during calibration.
    """
    scan = quick_scan(spectrum, hypotheses, settings)
    if scan.empty:
        return pd.DataFrame()
    scan = scan.reset_index(drop=True)
    selected = select_for_refinement(scan, settings.n_refine)

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
            "n_components": candidate["n_components"],
            "noise_inflation": noise_inflation,
            "delta_chi2_effective": float(candidate["delta_chi2_broad"]) / noise_inflation**2,
            "broad_continuum_orthogonality": candidate["broad_continuum_orthogonality"],
            "broad_sigma_max_identifiable_kms": candidate["broad_sigma_max_identifiable_kms"],
        }
        row.update(rank_alternatives(scan, candidate))
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
    return pd.DataFrame(rows)


def null_offsets_from_table(table: pd.DataFrame, statistic: str = "median") -> dict[str, float]:
    """Per-system null offsets from a table of per-object maxima.

    ``table`` has one row per (object, system) with the best total statistic
    that system reached anywhere in a blind scan; see the null calibration in
    JOURNAL.md.  The median over objects is the offset.
    """
    grouped = table.groupby("system")["max_total"]
    return {k: float(v) for k, v in getattr(grouped, statistic)().items()}
