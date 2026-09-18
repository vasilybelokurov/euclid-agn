"""Class-comparison redshift engine: GALAXY, QSO and STAR template scans.

This is the Redrock/AMAZED architecture on Euclid templates.  Each object
class has its own template set and redshift range; every class is scanned
with :func:`euclid_agn.fit.template_cube.cube_scan` and the object is assigned
the (class, z) with the smallest chi-squared.  Two margins are reported:

* ``delta_chi2_runner_up`` - within the winning class, best minus the best
  minimum more than ``separation_kms`` away (redshift reliability);
* ``delta_chi2_class`` - best minus the best of any *other* class (class
  reliability).  Redrock compares raw chi-squared across classes; we do the
  same and record the parameter counts so a penalised comparison can be made
  afterwards.

Cubes are built lazily per (class, LSF bucket) and shared across objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from euclid_agn.fit.continuum_redshift import ContinuumScanResult
from euclid_agn.fit.quality import ZWarn
from euclid_agn.fit.template_cube import CubeStore, cube_scan, redshift_grid
from euclid_agn.models.library import Template


@dataclass(frozen=True)
class ClassSpec:
    """Templates and scan settings for one object class."""

    kind: str
    templates: list[Template]
    z_min: float
    z_max: float
    step_kms: float = 300.0
    poly_degree: int = 1
    nonnegative: bool = False
    extra_sigma_kms: float = 0.0  # intrinsic broadening added to the LSF (e.g. galaxy velocity dispersion)
    use_prior: bool = True  # STAR ignores a photometric-redshift prior


@dataclass(frozen=True)
class Classification:
    best: ContinuumScanResult
    per_class: dict[str, ContinuumScanResult]
    delta_chi2_class: float
    runner_up_class: str | None
    zwarn: int

    @property
    def kind(self) -> str:
        return self.best.kind

    @property
    def z(self) -> float:
        return self.best.z

    def as_row(self) -> dict:
        row = {"class": self.kind, "z": self.z, "chi2": self.best.chi2, "n_parameters": self.best.n_parameters,
               "n_pixels": self.best.n_pixels, "delta_chi2_runner_up": self.best.delta_chi2_runner_up,
               "delta_chi2_class": self.delta_chi2_class, "runner_up_class": self.runner_up_class,
               "z_prior": self.best.z_prior, "zwarn": self.zwarn}
        for kind, res in self.per_class.items():
            row[f"chi2_{kind.lower()}"] = res.chi2
            row[f"z_{kind.lower()}"] = res.z
        return row


class RedshiftEngine:
    """Scan every class and pick the best (class, z)."""

    def __init__(self, specs: list[ClassSpec], wavelength: np.ndarray, bin_width: float, lsf_step: float = 5.0,
                 separation_kms: float = 3000.0, min_delta_chi2: float = 25.0, min_delta_chi2_class: float = 9.0):
        self.specs = {s.kind: s for s in specs}
        self.stores = {
            s.kind: CubeStore({s.kind: s.templates}, redshift_grid(s.z_min, s.z_max, s.step_kms), wavelength,
                              bin_width, lsf_step=lsf_step, extra_sigma_kms={s.kind: s.extra_sigma_kms})
            for s in specs
        }
        self.separation_kms = separation_kms
        self.min_delta_chi2 = min_delta_chi2
        self.min_delta_chi2_class = min_delta_chi2_class

    def scan_class(self, kind: str, spectrum, projected, z_prior: float | None = None,
                   lsf_override: float | None = None) -> ContinuumScanResult | None:
        spec = self.specs[kind]
        cube = self.stores[kind].get(kind, lsf_override if lsf_override is not None else spectrum.lsf_sigma)
        return cube_scan(spectrum, projected, cube, poly_degree=spec.poly_degree, nonnegative=spec.nonnegative,
                         separation_kms=self.separation_kms, z_prior=z_prior if spec.use_prior else None)

    def run_at_lsf(self, spectrum, projected, lsf_sigma: float, z_prior: float | None = None) -> Classification | None:
        """Classify using ``lsf_sigma`` for the template smoothing instead of the spectrum's header value."""
        return self.run(spectrum, projected, z_prior=z_prior, lsf_override=lsf_sigma)

    def run(self, spectrum, projected, z_prior: float | None = None, noise_inflation: float = 1.0,
            lsf_override: float | None = None) -> Classification | None:
        """Best class and redshift; ``noise_inflation`` divides the margins before thresholding."""
        per_class = {}
        for kind in self.specs:
            res = self.scan_class(kind, spectrum, projected, z_prior, lsf_override)
            if res is not None:
                per_class[kind] = res
        if not per_class:
            return None
        order = sorted(per_class, key=lambda k: per_class[k].chi2)
        best = per_class[order[0]]
        if len(order) > 1:
            runner_kind = order[1]
            delta_class = float(per_class[runner_kind].chi2 - best.chi2)
        else:
            runner_kind, delta_class = None, float("inf")
        zwarn = ZWarn.NONE
        if best.delta_chi2_runner_up / noise_inflation < self.min_delta_chi2:
            zwarn |= ZWarn.SMALL_DELTA_CHI2
        if delta_class / noise_inflation < self.min_delta_chi2_class:
            zwarn |= ZWarn.AMBIGUOUS_CLASS
        if best.n_pixels < 0.5 * spectrum.wavelength.size:
            zwarn |= ZWarn.LOW_USABLE_FRACTION
        if z_prior is None or not np.isfinite(z_prior):
            zwarn |= ZWarn.NO_PRIOR
        elif np.isfinite(best.z_prior) and abs(best.z_prior - best.z) > 1e-6:
            zwarn |= ZWarn.PRIOR_DECIDED
        return Classification(best, per_class, delta_class, runner_kind, int(zwarn))


def default_specs(galaxy_templates: list[Template], qso_templates: list[Template] | None = None,
                  star_templates: list[Template] | None = None, galaxy_nonnegative: bool = False,
                  star_nonnegative: bool = False, z_max_galaxy: float = 2.0, z_max_qso: float = 3.3) -> list[ClassSpec]:
    """Class specifications for Euclid red-grism spectra.

    QSO z_max 3.3 is where the Glikman composite's blue end (2762 A) leaves the
    grism; STAR is scanned over +-600 km/s so a star cannot borrow a galaxy's
    redshift.  Classes compared by raw chi-squared must be *equally
    constrained*: VERIFIED that an 8-component free-sign stellar PCA beat
    NNLS galaxy archetypes on 63 % of DESI-confirmed low-z galaxies and 73 %
    of z ~ 1 emission-line galaxies, because it can fit any smooth continuum.
    Use non-negative archetypes for STAR as for GALAXY.
    """
    specs = [ClassSpec("GALAXY", galaxy_templates, 0.0, z_max_galaxy, nonnegative=galaxy_nonnegative)]
    if qso_templates:
        # a single composite: its amplitude must be positive, or a negative quasar plus polynomial
        # becomes a free smooth-continuum model that wins on faint spectra (VERIFIED: 47 % of z ~ 1
        # DESI emission-line galaxies were called QSO with a free-sign amplitude)
        specs.append(ClassSpec("QSO", qso_templates, 0.0, z_max_qso, extra_sigma_kms=1500.0, nonnegative=True))
    if star_templates:
        specs.append(ClassSpec("STAR", star_templates, -0.002, 0.002, step_kms=100.0, use_prior=False,
                               nonnegative=star_nonnegative))
    return specs


__all__ = ["ClassSpec", "Classification", "RedshiftEngine", "default_specs"]
