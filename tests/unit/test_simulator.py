import numpy as np
import pytest

from euclid_agn.constants import SIR_BINWIDTH_ANGSTROM
from euclid_agn.models.line_catalog import BY_NAME
from euclid_agn.validation.simulator import (
    LineTruth,
    SpectrumTruth,
    noiseless_spectrum,
    simulate_arrays,
    simulate_observation,
    sir_wavelength_grid,
)


def test_grid_matches_the_release():
    w = sir_wavelength_grid()
    assert w.size == 531
    assert w[0] == pytest.approx(11900.0)
    assert w[-1] == pytest.approx(19002.0, abs=0.5)


def test_injected_line_flux_is_recovered_by_direct_integration():
    truth = SpectrumTruth(
        z=1.2, continuum_flux=0.0, lines=(LineTruth("Halpha", 3.0e-16, 200.0),)
    )
    w, f = noiseless_spectrum(truth)
    assert np.sum(f) * SIR_BINWIDTH_ANGSTROM == pytest.approx(3.0e-16, rel=1e-3)


def test_line_lands_at_the_expected_wavelength():
    z = 1.2
    truth = SpectrumTruth(z=z, continuum_flux=0.0, lines=(LineTruth("Halpha", 1e-16, 150.0),))
    w, f = noiseless_spectrum(truth)
    assert w[np.argmax(f)] == pytest.approx(BY_NAME["Halpha"].rest * (1 + z), abs=SIR_BINWIDTH_ANGSTROM)


def test_broad_line_is_wider_than_narrow_line():
    common = dict(z=1.2, continuum_flux=0.0)
    narrow = noiseless_spectrum(
        SpectrumTruth(lines=(LineTruth("Halpha", 1e-16, 150.0),), **common)
    )[1]
    broad = noiseless_spectrum(
        SpectrumTruth(lines=(LineTruth("Halpha", 1e-16, 3000.0, broad=True),), **common)
    )[1]
    assert broad.max() < narrow.max()
    assert np.count_nonzero(broad > 0.1 * broad.max()) > np.count_nonzero(
        narrow > 0.1 * narrow.max()
    )


def test_simulation_is_deterministic_given_the_seed():
    truth = SpectrumTruth(seed=7, lines=(LineTruth("Halpha", 2e-16, 200.0),))
    a = simulate_arrays(truth)[1]
    b = simulate_arrays(truth)[1]
    assert np.array_equal(a, b)


def test_noise_realisation_is_consistent_with_the_reported_variance():
    truth = SpectrumTruth(seed=11, continuum_flux=1e-17, lines=())
    w, f, v, m, q = simulate_arrays(truth)
    pull = (f - 1e-17) / np.sqrt(v)
    assert np.std(pull) == pytest.approx(1.0, rel=0.15)


def test_masked_fraction_is_honoured():
    truth = SpectrumTruth(seed=3, masked_fraction=0.1)
    w, f, v, m, q = simulate_arrays(truth)
    assert np.count_nonzero(m & 1) == pytest.approx(0.1 * w.size, abs=1)


def test_combined_spectrum_is_less_noisy_than_a_single_dither():
    truth = SpectrumTruth(seed=5, n_dithers=4, lines=())
    observation = simulate_observation(truth)
    combined_sigma = np.median(np.sqrt(observation.combined.variance))
    dither_sigma = np.median(np.sqrt(observation.dithers[0].variance))
    assert combined_sigma == pytest.approx(dither_sigma / 2.0, rel=0.1)
