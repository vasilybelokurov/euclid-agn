"""The nuisance continuum must not eat the broad line it sits under."""

import numpy as np

from euclid_agn.models.broad import BroadComponent, BroadFamily
from euclid_agn.models.forward import continuum_for, fit_hypothesis
from euclid_agn.models.line_catalog import BY_NAME
from euclid_agn.models.narrow import NarrowSystem, velocity_grid
from euclid_agn.spectra.continuum import power_law_block
from euclid_agn.validation.simulator import sir_wavelength_grid


def test_power_law_block_reproduces_a_power_law():
    w = sir_wavelength_grid()
    design, penalty, names = power_law_block(w)
    truth = (w / np.median(w)) ** -1.4
    coefficients, *_ = np.linalg.lstsq(design, truth, rcond=None)
    assert np.allclose(design @ coefficients, truth, rtol=2e-3)
    assert penalty.shape[0] == 0 and len(names) == design.shape[1]


def test_continuum_for_accepts_a_name_or_a_ready_made_block():
    w = sir_wavelength_grid()
    d1, _, n1 = continuum_for(w, "spline", n_knots=6)
    d2, _, n2 = continuum_for(w, "power_law")
    assert d1.shape[1] > d2.shape[1] and n1[0].startswith("continuum") and n2[0].startswith("powerlaw")
    supplied = (np.ones((w.size, 1)), np.zeros((0, 1)), ("flat",))
    assert continuum_for(w, supplied)[2] == ("flat",)


def quasar_spectrum(broad_flux=3e-16, sigma_kms=3000.0, z=1.2, snr=60.0, seed=3):
    """Power-law continuum plus one broad H-alpha, on the SIR grid.

    z = 1.2 puts H-alpha at 14442 A, near the middle of the grism; at z = 0.35 it
    would sit at 8862 A, outside it, and the fitter correctly returns nothing.
    """
    w = sir_wavelength_grid()
    continuum = 2e-17 * (w / 15000.0) ** -1.3
    centre = BY_NAME["Halpha"].rest * (1 + z)
    sigma = np.hypot(14.0, sigma_kms / 299792.458 * centre)
    line = broad_flux * np.exp(-0.5 * ((w - centre) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    noise = np.median(continuum) / snr
    rng = np.random.default_rng(seed)
    return w, continuum + line + rng.normal(0, noise, w.size), np.full(w.size, noise**2), z


def test_spline_continuum_wrecks_broad_lines_and_the_power_law_does_not():
    """Measured bias of the nuisance continuum on a known broad line.

    A B-spline with the knot spacing we use for narrow-line work is not stiff
    on the scale of a broad line: depending on the knot count it either
    swallows the line or blows the flux up by an order of magnitude, and it
    halves the detection statistic for the widest lines.  The power law is
    accurate to ~10 % from 1500 to 10000 km/s.  Numbers measured 2026-09-19,
    ``outputs/broad_continuum_bias.parquet``.
    """
    truth = 3e-16
    for sigma_kms, spline_range, power_range in (
        (1500.0, (0.9, 1.15), (0.9, 1.10)),
        (3000.0, (1.1, 1.5), (0.95, 1.25)),
        (10000.0, (2.0, 40.0), (0.85, 1.20)),   # the spline is off by a factor of ~20 here
    ):
        w, flux, variance, z = quasar_spectrum(broad_flux=truth, sigma_kms=sigma_kms)
        narrow = NarrowSystem(("Halpha", "NII6584"), z=z, lsf_sigma=14.0, velocities=velocity_grid(600.0, 200.0))
        broad = BroadFamily(line_names=("Halpha",), z=z, sigma_kms=sigma_kms, lsf_sigma=14.0)
        spline = fit_hypothesis(w, flux, variance, narrow, broad, bin_width=13.4, continuum="spline", n_knots=12)
        power = fit_hypothesis(w, flux, variance, narrow, broad, bin_width=13.4, continuum="power_law")
        assert spline_range[0] < spline.broad_fluxes["Halpha"] / truth < spline_range[1]
        assert power_range[0] < power.broad_fluxes["Halpha"] / truth < power_range[1]
        # and the power law keeps more of the signal, which is what sets the detection limit
        assert power.delta_chi2 > spline.delta_chi2
