import numpy as np
import pytest

from euclid_agn.fit.continuum_redshift import continuum_redshift_scan, redshift_grid
from euclid_agn.fit.screen import ScreenSettings, prepare
from euclid_agn.fit.template_cube import CubeStore, build_cube, cube_scan, prior_penalty
from euclid_agn.models.library import Template
from euclid_agn.spectra.types import Spectrum1D
from euclid_agn.validation.simulator import sir_wavelength_grid


def toy_templates():
    w = np.exp(np.arange(np.log(5000.0), np.log(30000.0), 2e-4))
    # a power law, not a constant: a constant would duplicate the polynomial's
    # first column and the toy problem would be exactly rank-deficient
    slope = (w / 15000.0) ** -1.5
    bump = np.exp(-0.5 * ((w - 16000.0) / 800.0) ** 2) - 0.3 * np.exp(-0.5 * ((w - 10900.0) / 300.0) ** 2)
    return [Template("slope", "GALAXY", w, slope, np.ones(w.size, bool)),
            Template("bump", "GALAXY", w, bump, np.ones(w.size, bool))]


def toy_spectrum(z=0.08, snr=30.0, seed=0, masked=()):
    w = sir_wavelength_grid(); n = w.size
    base = Spectrum1D(wavelength=w, flux=np.ones(n), variance=np.ones(n), mask=np.zeros(n, int), quality=np.ones(n), lsf_sigma=14.0, bin_width=13.4)
    projected = prepare(base, ScreenSettings(n_knots=1, outlier_threshold=0.0))
    cube = build_cube(toy_templates(), np.array([z]), w, 13.4, 14.0)
    model = 1e-17 * (cube.columns[0, :, 0] + 3.0 * cube.columns[0, :, 1])
    sigma = 1e-17 / snr
    rng = np.random.default_rng(seed)
    mask = np.zeros(n, int)
    for lo, hi in masked:
        mask[lo:hi] = 1
    return Spectrum1D(wavelength=w, flux=model + rng.normal(0, sigma, n), variance=np.full(n, sigma**2), mask=mask,
                      quality=np.ones(n), lsf_sigma=14.0, bin_width=13.4)


def test_cube_scan_matches_direct_scan_and_recovers_redshift():
    spectrum = toy_spectrum(z=0.08, masked=((200, 240),))
    projected = prepare(spectrum, ScreenSettings(n_knots=1, outlier_threshold=0.0))
    grid = redshift_grid(0.0, 0.3, 300.0)
    cube = build_cube(toy_templates(), grid, spectrum.wavelength, 13.4, spectrum.lsf_sigma)
    fast = cube_scan(spectrum, projected, cube, poly_degree=1)
    slow = continuum_redshift_scan(spectrum, projected, toy_templates(), z_min=0.0, z_max=0.3, step_kms=300.0, poly_degree=1)
    assert abs(fast.z - 0.08) < 0.003 and abs(slow.z - 0.08) < 0.003
    finite = np.isfinite(fast.grid_chi2) & np.isfinite(slow.grid_chi2)
    # 1e-5 relative: the LSF-smoothing reference wavelength differs (full grid vs kept pixels)
    assert np.allclose(fast.grid_chi2[finite], slow.grid_chi2[finite], rtol=1e-4)
    assert fast.delta_chi2_runner_up > 25


def test_nonnegative_mode_agrees_when_solution_is_positive():
    spectrum = toy_spectrum(z=0.12)
    projected = prepare(spectrum, ScreenSettings(n_knots=1, outlier_threshold=0.0))
    grid = redshift_grid(0.0, 0.3, 300.0)
    cube = build_cube(toy_templates(), grid, spectrum.wavelength, 13.4, spectrum.lsf_sigma)
    free = cube_scan(spectrum, projected, cube, poly_degree=1)
    nn = cube_scan(spectrum, projected, cube, poly_degree=1, nonnegative=True)
    assert abs(nn.z - 0.12) < 0.003
    assert nn.chi2 >= free.chi2 - 1e-6  # constraint can only raise chi2
    assert abs(nn.chi2 - free.chi2) < 1e-3 * free.chi2


def test_prior_shifts_choice_only_when_data_are_ambiguous():
    grid = np.linspace(0, 1, 101)
    pen = prior_penalty(grid, 0.5)
    assert pen[50] == pytest.approx(0.0, abs=1e-9)
    assert pen[0] > 10 and pen[0] < 12  # capped by the outlier component
    spectrum = toy_spectrum(z=0.08)
    projected = prepare(spectrum, ScreenSettings(n_knots=1, outlier_threshold=0.0))
    cube = build_cube(toy_templates(), redshift_grid(0.0, 0.3, 300.0), spectrum.wavelength, 13.4, 14.0)
    res = cube_scan(spectrum, projected, cube, z_prior=0.25)
    assert abs(res.z - 0.08) < 0.003 and abs(res.z_prior - 0.08) < 0.003  # strong data beat a wrong prior


def test_cube_store_buckets_lsf():
    w = sir_wavelength_grid()
    store = CubeStore({"GALAXY": toy_templates()}, np.array([0.0, 0.1]), w, 13.4, lsf_step=3.0)
    a = store.get("GALAXY", 13.7); b = store.get("GALAXY", 14.9); c = store.get("GALAXY", 20.0)
    assert a is b and a is not c and len(store) == 2
    assert np.isfinite(a.columns).all()


def test_partial_coverage_marks_nan_not_failure():
    w = sir_wavelength_grid()
    # covers 7000-13000 rest: at z=0 only the blue end of the grid, at z=0.5 all of it
    short = Template("short", "GALAXY", np.linspace(7000, 13000, 2000), np.ones(2000), np.ones(2000, bool))
    cube = build_cube([short], np.array([0.0, 0.5]), w, 13.4, 14.0)
    assert np.isfinite(cube.columns[0, :5, 0]).all() and np.isnan(cube.columns[0, -5:, 0]).all()
    assert np.isfinite(cube.columns[1]).all()
    keep = np.ones(w.size, bool)
    assert list(cube.covers(keep)) == [False, True]


def test_spline_nuisance_recovers_redshift_from_features_only():
    spectrum = toy_spectrum(z=0.08)
    projected = prepare(spectrum, ScreenSettings(n_knots=6, outlier_threshold=0.0))
    cube = build_cube(toy_templates(), redshift_grid(0.0, 0.3, 300.0), spectrum.wavelength, 13.4, 14.0)
    res = cube_scan(spectrum, projected, cube, spline_nuisance=True)
    # the toy's 800 A-wide bump is mostly absorbed by the spline; the narrow dip carries the redshift
    assert abs(res.z - 0.08) < 0.006
    assert res.n_parameters == 2 + projected.basis.shape[1]
    # the data are orthogonal to the spline, so the null chi2 is the projected chi2
    assert res.chi2_null == pytest.approx(projected.chi2_continuum, rel=1e-9)


def test_multiplicative_polynomial_corrects_a_tilted_spectrum():
    spectrum = toy_spectrum(z=0.08)
    w = spectrum.wavelength
    x = (w - w.mean()) / (np.ptp(w) / 2)
    tilted = Spectrum1D(wavelength=w, flux=spectrum.flux * (1 + 0.3 * x - 0.2 * x**2), variance=spectrum.variance,
                        mask=spectrum.mask, quality=spectrum.quality, lsf_sigma=14.0, bin_width=13.4)
    projected = prepare(tilted, ScreenSettings(n_knots=1, outlier_threshold=0.0))
    cube = build_cube(toy_templates(), redshift_grid(0.0, 0.3, 300.0), w, 13.4, 14.0)
    additive = cube_scan(tilted, projected, cube, poly_degree=0)
    mult = cube_scan(tilted, projected, cube, poly_degree=0, multiplicative_degree=2)
    # a quadratic multiplicative correction can shift the toy's 800 A-wide bump by a few
    # pixels, so the redshift is less sharply defined than with the additive fit
    assert abs(mult.z - 0.08) < 0.006
    assert mult.chi2 < 0.7 * additive.chi2  # the multiplicative correction absorbs the tilt (919 -> 491 in this toy)
    assert mult.chi2 < 1.3 * projected.wavelength.size  # and reaches the noise level
    # unconstrained mode takes the same path
    free = cube_scan(tilted, projected, cube, poly_degree=0, multiplicative_degree=2, nonnegative=False)
    assert abs(free.z - 0.08) < 0.006 and free.chi2 <= mult.chi2 + 1e-6
