"""Redshift hypotheses.

The Euclid SPE redshift is a *hypothesis*, not truth.  Fixing it would inherit
every bias of the galaxy/QSO template fit that produced it - and the AGN this
project is looking for are precisely the objects those templates classify
badly.  Hypotheses are therefore drawn from four independent origins:

``spe``     high-quality SPE galaxy or QSO solutions;
``phz``     photometric-redshift modes;
``peaks``   line identifications built from detected emission peaks;
``blind``   a uniform scan in velocity across the allowed redshift range.

Every hypothesis records where it came from, and the winning hypothesis of a
fit carries that label into the candidate catalogue, so it is always possible to
ask how many candidates would have been missed by trusting the catalogue
redshift.

The blind scan is what makes the search independent of prior classification; it
is also the expensive part, so its step is set by the instrumental resolution
rather than by taste.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from euclid_agn.constants import C_KMS, RGS_SCIENCE_WMAX_ANGSTROM, RGS_SCIENCE_WMIN_ANGSTROM
from euclid_agn.fit.linear import LinearProblem, solve
from euclid_agn.models.line_catalog import BY_NAME, SYSTEMS, LineSystem, visible_systems
from euclid_agn.numerics import blas_safe
from euclid_agn.spectra.continuum import continuum_block

log = logging.getLogger(__name__)

#: Ordering of origins when duplicates are merged: catalogue solutions win ties
#: because they carry external information, but they never exclude the others.
ORIGIN_PRIORITY: dict[str, int] = {
    "spe_galaxy": 0,
    "spe_qso": 1,
    "peaks": 2,
    "phz_mode_1": 3,
    "phz_median": 4,
    "phz_mode_2": 5,
    "blind": 6,
}


@dataclass(frozen=True)
class RedshiftHypothesis:
    """One redshift to test, and where it came from."""

    z: float
    origin: str
    system: str | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def priority(self) -> int:
        return ORIGIN_PRIORITY.get(self.origin, 99)

    def velocity_from(self, other: RedshiftHypothesis | float) -> float:
        """Velocity separation in km/s from another redshift."""
        z_other = other.z if isinstance(other, RedshiftHypothesis) else float(other)
        return C_KMS * abs(self.z - z_other) / (1.0 + 0.5 * (self.z + z_other))


def from_catalogue(row, min_probability: float = 0.0) -> list[RedshiftHypothesis]:
    """Hypotheses from SPE and PHZ columns of a manifest row.

    Missing columns are skipped silently: the manifest is allowed to be partial,
    and a source with no catalogue information still gets a blind scan.
    """
    out: list[RedshiftHypothesis] = []

    def value(name):
        try:
            v = row[name]
        except (KeyError, IndexError, TypeError):
            return None
        return None if v is None or (isinstance(v, float) and not np.isfinite(v)) else float(v)

    for column, origin in (("spe_gal_z", "spe_galaxy"), ("spe_qso_z", "spe_qso")):
        z = value(column)
        if z is None or z < 0:
            continue
        probability = value(column.replace("_z", "_z_prob"))
        if probability is not None and probability < min_probability:
            continue
        out.append(
            RedshiftHypothesis(
                z=z,
                origin=origin,
                metadata={"probability": probability} if probability is not None else {},
            )
        )
    for column, origin in (
        ("phz_mode_1", "phz_mode_1"),
        ("phz_median", "phz_median"),
        ("phz_mode_2", "phz_mode_2"),
    ):
        z = value(column)
        if z is not None and z >= 0:
            out.append(RedshiftHypothesis(z=z, origin=origin))
    return out


@blas_safe
def continuum_subtracted(
    wavelength: np.ndarray,
    flux: np.ndarray,
    variance: np.ndarray,
    n_knots: int = 12,
    smoothness: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Continuum-subtracted flux and its per-pixel signal-to-noise.

    Used only for peak finding.  The fit itself never works on
    continuum-subtracted data: the continuum is part of the generative model.
    """
    design, penalty, names = continuum_block(wavelength, n_knots=n_knots)
    solution = solve(
        LinearProblem(
            design=design,
            data=flux,
            variance=variance,
            regularisation=penalty,
            regularisation_weight=smoothness,
            names=names,
        )
    )
    residual = flux - solution.model
    return residual, residual / np.sqrt(variance)


def find_peaks(
    wavelength: np.ndarray,
    snr: np.ndarray,
    threshold: float = 4.0,
    min_separation_pixels: int = 2,
    max_peaks: int = 12,
) -> np.ndarray:
    """Indices of local signal-to-noise maxima above ``threshold``.

    Deliberately simple and permissive: a peak here is only a *suggestion* of a
    line, and a wrong suggestion costs one extra hypothesis, not a false
    detection.
    """
    snr = np.asarray(snr, dtype=np.float64)
    n = snr.size
    if n < 3:
        return np.empty(0, dtype=int)
    interior = np.arange(1, n - 1)
    is_max = (snr[interior] >= snr[interior - 1]) & (snr[interior] > snr[interior + 1])
    candidates = interior[is_max & (snr[interior] > threshold)]
    if candidates.size == 0:
        return candidates
    order = candidates[np.argsort(snr[candidates])[::-1]]
    kept: list[int] = []
    for index in order:
        if all(abs(index - other) >= min_separation_pixels for other in kept):
            kept.append(int(index))
        if len(kept) >= max_peaks:
            break
    return np.array(sorted(kept), dtype=int)


def from_peaks(
    wavelength: np.ndarray,
    flux: np.ndarray,
    variance: np.ndarray,
    threshold: float = 4.0,
    line_names: Sequence[str] | None = None,
    z_min: float = 0.0,
    z_max: float = 6.0,
    require_second_line: bool = True,
    n_knots: int = 12,
) -> list[RedshiftHypothesis]:
    """Identify detected peaks as known lines and return the implied redshifts.

    With ``require_second_line`` a hypothesis survives only if the line system
    it implies has at least one *other* member inside the covered range, which
    removes most single-noise-spike identifications at no cost to real systems.
    """
    residual, snr = continuum_subtracted(wavelength, flux, variance, n_knots=n_knots)
    peaks = find_peaks(wavelength, snr, threshold=threshold)
    pool = tuple(BY_NAME[n] for n in line_names) if line_names else tuple(BY_NAME.values())
    wmin, wmax = float(wavelength[0]), float(wavelength[-1])

    out: list[RedshiftHypothesis] = []
    for index in peaks:
        observed = float(wavelength[index])
        for line in pool:
            z = observed / line.rest - 1.0
            if not (z_min <= z <= z_max):
                continue
            systems = visible_systems(z, wmin, wmax, min_lines=2 if require_second_line else 1)
            systems = [s for s in systems if line.name in s.members]
            if require_second_line and not systems:
                continue
            out.append(
                RedshiftHypothesis(
                    z=z,
                    origin="peaks",
                    system=systems[0].name if systems else None,
                    metadata={
                        "peak_wavelength": observed,
                        "peak_snr": float(snr[index]),
                        "line": line.name,
                    },
                )
            )
    return out


def blind_grid(
    z_min: float = 0.0,
    z_max: float = 5.7,
    step_kms: float = 300.0,
    systems: Sequence[LineSystem] | None = None,
    wavelength_min: float = RGS_SCIENCE_WMIN_ANGSTROM,
    wavelength_max: float = RGS_SCIENCE_WMAX_ANGSTROM,
    require_broad: bool = True,
) -> list[RedshiftHypothesis]:
    """Uniform scan in velocity, restricted to redshifts that can be tested.

    A redshift at which no permitted line falls inside the covered range cannot
    support a BLR test, so scanning it buys nothing but trials.  The grid is
    uniform in ``ln(1+z)`` because a fixed velocity step is a fixed step in that
    variable.
    """
    if step_kms <= 0:
        raise ValueError("step_kms must be positive")
    systems = tuple(systems or SYSTEMS)
    step = step_kms / C_KMS
    n = int(np.ceil(np.log((1.0 + z_max) / (1.0 + z_min)) / step))
    grid = (1.0 + z_min) * np.exp(step * np.arange(n + 1)) - 1.0

    out: list[RedshiftHypothesis] = []
    for z in grid:
        found = visible_systems(
            float(z),
            wavelength_min,
            wavelength_max,
            min_lines=1,
            require_broad=require_broad,
        )
        found = [s for s in found if s.name in {x.name for x in systems}]
        if not found:
            continue
        out.append(RedshiftHypothesis(z=float(z), origin="blind", system=found[0].name))
    return out


def deduplicate(
    hypotheses: Iterable[RedshiftHypothesis], tolerance_kms: float = 200.0
) -> list[RedshiftHypothesis]:
    """Merge hypotheses closer than ``tolerance_kms``, keeping the best origin.

    "Best" means the lowest :data:`ORIGIN_PRIORITY`; the survivor records the
    origins it absorbed, so a blind-scan redshift that happens to coincide with
    the SPE solution is not silently reported as a blind discovery.
    """
    ordered = sorted(hypotheses, key=lambda h: (h.z, h.priority))
    out: list[RedshiftHypothesis] = []
    for hypothesis in ordered:
        if out and hypothesis.velocity_from(out[-1]) <= tolerance_kms:
            previous = out[-1]
            keeper, absorbed = (
                (previous, hypothesis)
                if previous.priority <= hypothesis.priority
                else (hypothesis, previous)
            )
            merged = dict(keeper.metadata)
            origins = set(merged.get("merged_origins", ()))
            origins.update({absorbed.origin, *absorbed.metadata.get("merged_origins", ())})
            merged["merged_origins"] = tuple(sorted(origins))
            out[-1] = RedshiftHypothesis(
                z=keeper.z,
                origin=keeper.origin,
                system=keeper.system or absorbed.system,
                metadata=merged,
            )
        else:
            out.append(hypothesis)
    return out


def generate(
    wavelength: np.ndarray | None = None,
    flux: np.ndarray | None = None,
    variance: np.ndarray | None = None,
    row=None,
    sources: Sequence[str] = ("spe", "phz", "peaks", "blind"),
    z_min: float = 0.0,
    z_max: float = 5.7,
    blind_step_kms: float = 300.0,
    peak_threshold: float = 4.0,
    tolerance_kms: float = 200.0,
    wavelength_min: float = RGS_SCIENCE_WMIN_ANGSTROM,
    wavelength_max: float = RGS_SCIENCE_WMAX_ANGSTROM,
) -> list[RedshiftHypothesis]:
    """All hypotheses for one spectrum, deduplicated and ordered by redshift."""
    out: list[RedshiftHypothesis] = []
    if row is not None and ("spe" in sources or "phz" in sources):
        for hypothesis in from_catalogue(row):
            family = hypothesis.origin.split("_")[0]
            if family in sources or hypothesis.origin.startswith("phz") and "phz" in sources:
                out.append(hypothesis)
    if "peaks" in sources and wavelength is not None:
        out.extend(
            from_peaks(
                wavelength, flux, variance, threshold=peak_threshold, z_min=z_min, z_max=z_max
            )
        )
    if "blind" in sources:
        out.extend(
            blind_grid(
                z_min=z_min,
                z_max=z_max,
                step_kms=blind_step_kms,
                wavelength_min=wavelength_min,
                wavelength_max=wavelength_max,
            )
        )
    out = [h for h in out if z_min <= h.z <= z_max]
    return deduplicate(out, tolerance_kms=tolerance_kms)
