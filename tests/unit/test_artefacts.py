import numpy as np

from euclid_agn.spectra.artefacts import continuum_trough
from euclid_agn.spectra.types import Spectrum1D
from euclid_agn.validation.simulator import sir_wavelength_grid


def spectrum_with(flux):
    w = sir_wavelength_grid(); n = w.size
    return Spectrum1D(wavelength=w, flux=flux, variance=np.full(n, (0.01 * np.nanmedian(flux)) ** 2),
                      mask=np.zeros(n, int), quality=np.ones(n), lsf_sigma=14.0, bin_width=13.4)


def test_clean_continuum_is_not_flagged():
    w = sir_wavelength_grid()
    assert not continuum_trough(spectrum_with((w / 15000.0) ** -1.0)).flagged


def test_rectangular_deficit_is_flagged_with_its_extent():
    w = sir_wavelength_grid()
    flux = (w / 15000.0) ** -1.0
    hole = (w > 14300) & (w < 16300)
    flux = np.where(hole, flux * 0.25, flux)
    r = continuum_trough(spectrum_with(flux))
    assert r.flagged and r.depth < 0.5
    assert 14200 < r.start_angstrom < 14450 and 16150 < r.stop_angstrom < 16400
    assert r.width_angstrom > 1500 and 0.2 < r.pixel_fraction < 0.5


def test_edge_rolloff_is_not_a_trough():
    w = sir_wavelength_grid()
    flux = (w / 15000.0) ** -1.0
    flux = np.where(w > 18000, flux * 0.1, flux)   # extraction roll-off at the red end
    assert not continuum_trough(spectrum_with(flux)).flagged


def test_narrow_absorption_is_not_a_trough():
    w = sir_wavelength_grid()
    flux = (w / 15000.0) ** -1.0 * (1 - 0.8 * np.exp(-0.5 * ((w - 15000) / 60.0) ** 2))
    assert not continuum_trough(spectrum_with(flux)).flagged
