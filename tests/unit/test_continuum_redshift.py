from pathlib import Path

import numpy as np
import pytest

from euclid_agn.fit.continuum_redshift import continuum_redshift_scan, redshift_grid, refine_minimum
from euclid_agn.fit.screen import ScreenSettings, prepare
from euclid_agn.models.library import DEFAULT_ROOT, Template, build_pca_basis, load_xsl_ssp_library, project_template
from euclid_agn.spectra.types import Spectrum1D
from euclid_agn.validation.simulator import sir_wavelength_grid

HAVE_XSL = any(Path(DEFAULT_ROOT).glob("xsl_ssp/**/*.fits"))
needs_xsl = pytest.mark.skipif(not HAVE_XSL, reason="XSL SSP library not downloaded")


def test_redshift_grid_is_uniform_in_velocity():
    g = redshift_grid(0.0, 1.0, 1000.0)
    steps = 299792.458 * np.diff(g) / (1 + g[:-1])
    assert np.allclose(steps, 1000.0, rtol=5e-3)  # exp(s)-1 vs s


def synthetic_galaxy(template: Template, z: float, snr: float, seed: int, lsf: float = 15.0) -> Spectrum1D:
    w = sir_wavelength_grid()
    n = w.size
    base = Spectrum1D(wavelength=w, flux=np.ones(n), variance=np.ones(n), mask=np.zeros(n, int), quality=np.ones(n), lsf_sigma=lsf, bin_width=13.4)
    projected = prepare(base, ScreenSettings(n_knots=1, outlier_threshold=0.0))
    column = project_template(template, projected, z)
    flux = np.zeros(n); flux[np.isin(w, projected.wavelength)] = column * 1e-17
    sigma = 1e-17 / snr
    rng = np.random.default_rng(seed)
    return Spectrum1D(wavelength=w, flux=flux + rng.normal(0, sigma, n), variance=np.full(n, sigma**2),
                      mask=np.zeros(n, int), quality=np.ones(n), lsf_sigma=lsf, bin_width=13.4)


@needs_xsl
def test_recovers_the_redshift_of_a_redshifted_ssp():
    lib = load_xsl_ssp_library(mh_min=-0.5, log_age_min=9.0)
    basis = build_pca_basis(lib, n_components=4)
    truth = [t for t in lib if t.metadata["log_age"] == 9.7 and abs(t.metadata["mh"]) < 0.05][0]
    for z_true, snr in ((0.12, 20.0), (0.35, 20.0), (0.6, 10.0)):
        spectrum = synthetic_galaxy(truth, z_true, snr, seed=int(z_true * 100))
        projected = prepare(spectrum, ScreenSettings(outlier_threshold=0.0))
        result = continuum_redshift_scan(spectrum, projected, basis.templates(), z_min=0.0, z_max=1.0, step_kms=300.0)
        assert result is not None
        z_hat = refine_minimum(result)
        assert abs(z_hat - z_true) / (1 + z_true) < 0.003, (z_true, z_hat)
        assert result.delta_chi2_runner_up > 9  # 3-sigma equivalent; z=0.6 at S/N 10 gives ~12
        assert result.delta_chi2_null > 25  # S/N 10 at z=0.6 gives ~40


@needs_xsl
def test_featureless_spectrum_gives_no_confident_redshift():
    lib = load_xsl_ssp_library(mh_min=-0.5, log_age_min=9.0)
    basis = build_pca_basis(lib, n_components=4)
    w = sir_wavelength_grid(); n = w.size; rng = np.random.default_rng(7)
    spectrum = Spectrum1D(wavelength=w, flux=1e-17 + rng.normal(0, 1e-18, n), variance=np.full(n, 1e-36), mask=np.zeros(n, int),
                          quality=np.ones(n), lsf_sigma=15.0, bin_width=13.4)
    result = continuum_redshift_scan(spectrum, prepare(spectrum, ScreenSettings(outlier_threshold=0.0)), basis.templates(), 0.0, 1.0, 600.0)
    assert result.delta_chi2_runner_up < 25
