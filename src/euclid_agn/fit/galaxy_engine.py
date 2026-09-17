"""GALAXY-class redshift: two models, one answer, chosen by evidence.

Measured on DESI truth (JOURNAL, session 11):

* faint emission-line galaxies (S/N ~ 3 per pixel, z ~ 1-1.8): a line scan
  against a projected-out spline continuum reaches 40 %; every continuum
  template added costs accuracy because no template set reproduces a faint
  young dusty continuum and the templates fit residual structure differently
  at each redshift;
* bright low-z galaxies (S/N 16-160, z < 0.5): the stellar continuum is the
  only information; 18 XSL archetypes x multiplicative cubic reach 46 %
  (Euclid SPE: 20 %) and lines are irrelevant.

So the class runs both:

  A. ``lines + spline``  - :func:`joint_scan` with ``cube=None`` and
     ``spline_nuisance=True``;
  B. ``archetypes + lines`` - :func:`joint_scan` with the archetype cube,
     multiplicative cubic, dither-scatter-rescaled variance;

and picks per object by *redshift evidence*: the model whose chi2(z) curve
has the larger best-minus-runner-up margin (on the same, dither-scatter-
rescaled variance) wins, with the continuum S/N as a guard at both ends
(below ``snr_switch`` the lines model is used; above ``snr_continuum`` the
continuum model).  Absolute fit quality (chi2 or BIC) must *not* be used: a
12-knot spline always fits a bright continuum better than physical
templates, yet that chi2 carries no redshift information - VERIFIED: a BIC
rule chose the spline model for 480 of 489 bright low-z galaxies and
collapsed the agreement with DESI from 46 % to 6 %.  Both models' redshifts
are kept; their agreement is a quality signal (``MODELS_DISAGREE``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from euclid_agn.constants import C_KMS
from euclid_agn.fit.joint_scan import JointScanResult, joint_scan
from euclid_agn.fit.quality import ZWarn
from euclid_agn.fit.screen import ScreenSettings, prepare
from euclid_agn.fit.template_cube import CubeStore
from euclid_agn.spectra.coherence import dither_variance_rescale


@dataclass(frozen=True)
class GalaxyRedshift:
    chosen: str  # "lines" | "continuum"
    result: JointScanResult
    lines: JointScanResult | None
    continuum: JointScanResult | None
    bic_lines: float
    bic_continuum: float
    zwarn: int
    snr: float = float("nan")  # median continuum S/N per pixel used for the switch

    @property
    def z(self) -> float:
        return self.result.z

    def as_row(self) -> dict:
        row = {"gal_model": self.chosen, "gal_z": self.result.z, "gal_z_prior": self.result.z_prior,
               "gal_delta_chi2_runner_up": self.result.delta_chi2_runner_up,
               "gal_delta_chi2_runner_up_prior": self.result.delta_chi2_runner_up_prior,
               "gal_bic_lines": self.bic_lines, "gal_bic_continuum": self.bic_continuum, "gal_zwarn": self.zwarn,
               "gal_z_lines": self.lines.z if self.lines else np.nan,
               "gal_z_continuum": self.continuum.z if self.continuum else np.nan,
               "gal_delta_chi2_lines": self.result.delta_chi2_lines, "gal_snr": self.snr}
        return row


def bic(result: JointScanResult) -> float:
    return float(result.chi2 + result.n_parameters * np.log(max(result.n_pixels, 2)))


class GalaxyEngine:
    """Run models A and B and choose."""

    def __init__(self, store: CubeStore, n_knots: int = 12, multiplicative_degree: int = 3,
                 min_continuum_snr: float = 0.0, snr_switch: float = 5.0, snr_continuum: float = 8.0,
                 separation_kms: float = 3000.0, agreement_kms: float = 3000.0):
        self.store = store
        self.n_knots = n_knots
        self.multiplicative_degree = multiplicative_degree
        self.min_continuum_snr = min_continuum_snr  # skip model B below this median S/N (0 = always run)
        # below this median continuum S/N per pixel the lines+spline model is used regardless of BIC:
        # measured on DESI truth, continuum templates only hurt at S/N ~ 3 (Halpha sample) and BIC
        # cannot see template inadequacy when both models reach the noise floor
        self.snr_switch = snr_switch
        self.snr_continuum = snr_continuum  # above this, the continuum model is used regardless of margins
        self.separation_kms = separation_kms
        self.agreement_kms = agreement_kms

    def run(self, observation, z_prior: float | None = None) -> GalaxyRedshift | None:
        # model B's variance (dither scatter) is used for both models so BIC compares like with like
        spectrum, _ = dither_variance_rescale(observation) if observation.dithers else (observation.combined, None)
        proj_a = prepare(spectrum, ScreenSettings(n_knots=self.n_knots, outlier_threshold=5.0))
        if proj_a is None:
            return None
        cube = self.store.get("GALAXY", spectrum.lsf_sigma)
        a = joint_scan(spectrum, proj_a, None, grid_z=cube.grid_z, spline_nuisance=True, z_prior=z_prior,
                       separation_kms=self.separation_kms)
        b = None
        keep = np.isin(spectrum.wavelength, proj_a.wavelength)
        snr = float(np.nanmedian(spectrum.flux[keep] * proj_a.weight))
        if snr >= self.min_continuum_snr:
            proj_b = prepare(spectrum, ScreenSettings(n_knots=1, outlier_threshold=5.0))
            if proj_b is not None:
                b = joint_scan(spectrum, proj_b, cube, poly_degree=0, nonnegative=True,
                               multiplicative_degree=self.multiplicative_degree, z_prior=z_prior,
                               separation_kms=self.separation_kms)
        if a is None and b is None:
            return None
        bic_a = bic(a) if a is not None else np.inf
        bic_b = bic(b) if b is not None else np.inf
        margin = lambda r: (r.delta_chi2_runner_up_prior if np.isfinite(r.delta_chi2_runner_up_prior) else r.delta_chi2_runner_up) if r is not None else -np.inf
        if b is None or (a is not None and snr < self.snr_switch):
            chosen, result = "lines", a
        elif a is None or snr > self.snr_continuum:
            chosen, result = "continuum", b
        else:
            chosen, result = ("continuum", b) if margin(b) > margin(a) else ("lines", a)
        zwarn = ZWarn.NONE
        if a is not None and b is not None:
            dv = C_KMS * abs(a.z - b.z) / (1 + min(a.z, b.z))
            if dv > self.agreement_kms:
                zwarn |= ZWarn.MODELS_DISAGREE
        if result.delta_chi2_runner_up < 25:
            zwarn |= ZWarn.SMALL_DELTA_CHI2
        if z_prior is None or not np.isfinite(z_prior):
            zwarn |= ZWarn.NO_PRIOR
        return GalaxyRedshift(chosen, result, a, b, bic_a, bic_b, int(zwarn), float(snr))


__all__ = ["GalaxyEngine", "GalaxyRedshift", "bic"]
