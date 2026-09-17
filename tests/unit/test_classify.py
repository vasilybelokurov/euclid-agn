import numpy as np

from euclid_agn.fit.classify import ClassSpec, RedshiftEngine, default_specs
from euclid_agn.fit.quality import ZWarn
from euclid_agn.fit.screen import ScreenSettings, prepare
from euclid_agn.fit.template_cube import build_cube
from euclid_agn.models.library import Template
from euclid_agn.spectra.types import Spectrum1D
from euclid_agn.validation.simulator import sir_wavelength_grid


def rest_grid():
    return np.exp(np.arange(np.log(3000.0), np.log(30000.0), 2e-4))


def galaxy_templates():
    w = rest_grid()
    bump = (w / 15000.0) ** -0.5 * (1 + 0.6 * np.exp(-0.5 * ((w - 16000.0) / 900.0) ** 2))
    return [Template("gal", "GALAXY", w, bump, np.ones(w.size, bool))]


def star_templates():
    w = rest_grid()
    # steep power law with Pa-beta and Pa-gamma absorption so the velocity is constrained
    absorption = 1 - 0.4 * np.exp(-0.5 * ((w - 12821.6) / 25.0) ** 2) - 0.3 * np.exp(-0.5 * ((w - 10941.1) / 25.0) ** 2)
    return [Template("star", "STAR", w, (w / 15000.0) ** -3.0 * absorption, np.ones(w.size, bool))]


def qso_templates():
    w = rest_grid()
    pl = (w / 15000.0) ** -1.5 * (1 + 2.0 * np.exp(-0.5 * ((w - 6564.6) / 60.0) ** 2))
    return [Template("qso", "QSO", w, pl, np.ones(w.size, bool))]


def observe(templates, z, snr=40.0, seed=0, lsf=14.0):
    w = sir_wavelength_grid(); n = w.size
    cube = build_cube(templates, np.array([z]), w, 13.4, lsf)
    model = 1e-17 * cube.columns[0, :, 0] / np.nanmean(cube.columns[0, :, 0])
    sigma = 1e-17 / snr
    rng = np.random.default_rng(seed)
    return Spectrum1D(wavelength=w, flux=model + rng.normal(0, sigma, n), variance=np.full(n, sigma**2),
                      mask=np.zeros(n, int), quality=np.ones(n), lsf_sigma=lsf, bin_width=13.4)


def engine():
    specs = default_specs(galaxy_templates(), qso_templates(), star_templates(), z_max_galaxy=0.6, z_max_qso=2.0)
    return RedshiftEngine(specs, sir_wavelength_grid(), 13.4)


def test_engine_picks_class_and_redshift():
    eng = engine()
    settings = ScreenSettings(n_knots=1, outlier_threshold=0.0)
    gal = observe(galaxy_templates(), 0.11)
    res = eng.run(gal, prepare(gal, settings))
    assert res.kind == "GALAXY" and abs(res.z - 0.11) < 0.003
    assert res.delta_chi2_class > 9 and not (res.zwarn & ZWarn.AMBIGUOUS_CLASS)
    qso = observe(qso_templates(), 1.4)
    res = eng.run(qso, prepare(qso, settings))
    assert res.kind == "QSO" and abs(res.z - 1.4) < 0.005
    star = observe(star_templates(), 0.0)
    res = eng.run(star, prepare(star, settings))
    assert res.kind == "STAR" and abs(res.z) < 0.001
    assert set(res.per_class) == {"GALAXY", "QSO", "STAR"}
    row = res.as_row()
    assert row["class"] == "STAR" and "chi2_galaxy" in row


def test_prior_bits_and_star_ignores_prior():
    eng = engine()
    settings = ScreenSettings(n_knots=1, outlier_threshold=0.0)
    gal = observe(galaxy_templates(), 0.11)
    res = eng.run(gal, prepare(gal, settings))
    assert res.zwarn & ZWarn.NO_PRIOR
    res = eng.run(gal, prepare(gal, settings), z_prior=0.12)
    assert not (res.zwarn & ZWarn.NO_PRIOR) and not (res.zwarn & ZWarn.PRIOR_DECIDED)
    assert not np.isfinite(res.per_class["STAR"].z_prior)
