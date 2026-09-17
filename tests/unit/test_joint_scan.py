import numpy as np
import pytest

from euclid_agn.fit.joint_scan import joint_scan, line_templates
from euclid_agn.fit.screen import ScreenSettings, prepare
from euclid_agn.fit.template_cube import build_cube, cube_scan, redshift_grid
from euclid_agn.models.library import Template
from euclid_agn.models.line_catalog import BY_NAME
from euclid_agn.spectra.types import Spectrum1D
from euclid_agn.validation.simulator import sir_wavelength_grid


def continuum_templates():
    w = np.exp(np.arange(np.log(3000.0), np.log(30000.0), 2e-4))
    a = (w / 15000.0) ** -0.7 * (1 + 0.5 * np.exp(-0.5 * ((w - 16000.0) / 900.0) ** 2))
    b = (w / 15000.0) ** -1.5
    return [Template("a", "GALAXY", w, a, np.ones(w.size, bool)), Template("b", "GALAXY", w, b, np.ones(w.size, bool))]


def observe(z, line_flux=0.0, snr=8.0, seed=1, lsf=14.0):
    w = sir_wavelength_grid(); n = w.size
    cube = build_cube(continuum_templates(), np.array([z]), w, 13.4, lsf)
    cont = 1e-17 * cube.columns[0, :, 0] / np.nanmean(cube.columns[0, :, 0])
    model = cont.copy()
    if line_flux:
        # the H-alpha complex at the halpha_hii ratios: a lone H-alpha *is* Pa-beta at another z
        for name, ratio in (("Halpha", 1.0), ("NII6584", 0.25), ("NII6548", 0.085), ("SII6716", 0.20), ("SII6731", 0.15)):
            centre = BY_NAME[name].rest * (1 + z)
            sigma = np.hypot(lsf, 200.0 / 299792.458 * centre)
            model += ratio * line_flux * np.exp(-0.5 * ((w - centre) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    noise = 1e-17 / snr
    rng = np.random.default_rng(seed)
    return Spectrum1D(wavelength=w, flux=model + rng.normal(0, noise, n), variance=np.full(n, noise**2),
                      mask=np.zeros(n, int), quality=np.ones(n), lsf_sigma=lsf, bin_width=13.4)


@pytest.fixture(scope="module")
def cube():
    return build_cube(continuum_templates(), redshift_grid(0.0, 2.0, 300.0), sir_wavelength_grid(), 13.4, 14.0)


def test_lines_break_a_continuum_degeneracy(cube):
    # z = 1.2: the 1.6 micron bump is out of the window, the continuum alone is a power law,
    # and H-alpha at 14442 A is also Pa-beta at z = 0.126: only the [N II]/[S II] companions
    # (20 %/15 % of H-alpha in the template) can break the degeneracy, so the line must be bright
    z = 1.2
    spectrum = observe(z, line_flux=1.2e-15, snr=8.0)
    projected = prepare(spectrum, ScreenSettings(n_knots=1, outlier_threshold=0.0))
    cont_only = cube_scan(spectrum, projected, cube, poly_degree=3, nonnegative=True)
    joint = joint_scan(spectrum, projected, cube, poly_degree=3)
    assert abs(joint.z - z) < 0.004
    assert joint.delta_chi2_lines > 50
    assert joint.line_snr["halpha_hii"] > 5
    assert joint.chi2 <= cont_only.chi2 + 1e-6  # lines can only help at the same redshift
    assert joint.as_row()["cz_best_line_template"] == "halpha_hii"


def test_line_free_spectrum_still_gets_a_continuum_redshift(cube):
    z = 0.1
    spectrum = observe(z, line_flux=0.0, snr=30.0)
    projected = prepare(spectrum, ScreenSettings(n_knots=1, outlier_threshold=0.0))
    joint = joint_scan(spectrum, projected, cube, poly_degree=1)
    assert abs(joint.z - z) < 0.004
    assert joint.delta_chi2_lines < 15  # no real lines: little evidence from the line columns
    assert all(v >= 0 for v in joint.line_amplitudes.values())


def test_free_continuum_mode_runs(cube):
    spectrum = observe(0.1, snr=30.0)
    projected = prepare(spectrum, ScreenSettings(n_knots=1, outlier_threshold=0.0))
    res = joint_scan(spectrum, projected, cube, poly_degree=1, nonnegative=False)
    assert abs(res.z - 0.1) < 0.004
    assert len(line_templates()) == 5
