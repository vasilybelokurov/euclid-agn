"""Fit a Euclid NISP spectrum with rvspecfit (Koposov; PHOENIX templates).

Setup ``nisp_red`` is built by ``scripts/rvspecfit_nisp_setup.sh``; the
configuration file lives next to the templates
(``~/data/euclid/rvspecfit/config.yaml``).  This wrapper turns a
:class:`~euclid_agn.spectra.types.Spectrum1D` into rvspecfit's ``SpecData``,
runs the CCF first guess and the maximum-likelihood fit, and returns the
stellar parameters, velocity and chi-squared - the stellar chi-squared is what
the class comparison uses against GALAXY/QSO.

The velocity range is +-1500 km/s: at R ~ 450 the LSF is ~300 km/s per pixel
and the redshift of a star is zero, so anything beyond is not a star.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

DEFAULT_CONFIG = Path("~/data/euclid/rvspecfit/config.yaml").expanduser()
SETUP = "nisp_red"


@dataclass(frozen=True)
class StarFit:
    vel_kms: float
    vel_err_kms: float
    teff: float
    logg: float
    feh: float
    alpha: float
    chi2: float
    n_pixels: int
    npoly: int
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def reduced_chi2(self) -> float:
        return self.chi2 / max(self.n_pixels - 5 - self.npoly, 1)

    def as_row(self, prefix: str = "rvs_") -> dict:
        return {f"{prefix}vel": self.vel_kms, f"{prefix}vel_err": self.vel_err_kms, f"{prefix}teff": self.teff,
                f"{prefix}logg": self.logg, f"{prefix}feh": self.feh, f"{prefix}alpha": self.alpha,
                f"{prefix}chi2": self.chi2, f"{prefix}n_pixels": self.n_pixels}


@lru_cache(maxsize=1)
def load_config(path: str = str(DEFAULT_CONFIG)):
    from rvspecfit import utils

    return utils.read_config(path)


def spec_data(spectrum, wavelength_min: float = 11900.0, wavelength_max: float = 19000.0):
    """rvspecfit SpecData from a Spectrum1D (usable pixels only, in the science window)."""
    from rvspecfit import spec_fit

    keep = spectrum.usable() & (spectrum.wavelength >= wavelength_min) & (spectrum.wavelength <= wavelength_max)
    lam = np.asarray(spectrum.wavelength[keep], dtype=np.float64)
    flux = np.asarray(spectrum.flux[keep], dtype=np.float64)
    sigma = np.sqrt(np.asarray(spectrum.variance[keep], dtype=np.float64))
    # rvspecfit works in arbitrary flux units but its polynomial continuum is better conditioned near unity
    scale = float(np.nanmedian(np.abs(flux))) or 1.0
    return spec_fit.SpecData(SETUP, lam, flux / scale, sigma / scale), scale, int(keep.sum())


def fit_star(spectrum, npoly: int = 4, config_path: str = str(DEFAULT_CONFIG)) -> StarFit | None:
    """CCF first guess, then maximum-likelihood fit; ``None`` if too few pixels."""
    from rvspecfit import fitter_ccf, vel_fit

    config = load_config(config_path)
    data, scale, n = spec_data(spectrum)
    if n < 100:
        return None
    guess = fitter_ccf.fit([data], config)
    start = dict(guess["best_par"])
    start["vel"] = float(guess.get("best_vel", 0.0))
    result = vel_fit.process([data], start, config=config, options={"npoly": npoly})
    param = result.get("param", {})
    return StarFit(
        vel_kms=float(result.get("vel", np.nan)), vel_err_kms=float(result.get("vel_err", np.nan)),
        teff=float(param.get("teff", np.nan)), logg=float(param.get("logg", np.nan)),
        feh=float(param.get("feh", np.nan)), alpha=float(param.get("alpha", np.nan)),
        chi2=float(np.sum(result.get("chisq_array", [np.nan]))) if "chisq_array" in result else float(result.get("chisq", np.nan)),
        n_pixels=n, npoly=npoly, raw={k: v for k, v in result.items() if k not in ("yfit",)},
    )


__all__ = ["StarFit", "fit_star", "spec_data", "load_config"]
