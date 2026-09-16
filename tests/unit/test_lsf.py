import numpy as np
import pytest

from euclid_agn.constants import SIR_BINWIDTH_ANGSTROM
from euclid_agn.spectra.lsf import (
    convolve_lsf,
    effective_sigma,
    gaussian_kernel,
    gaussian_pixel_integral,
    pixel_edges,
    resolving_power,
    sigma_angstrom_to_kms,
    sigma_kms_to_angstrom,
)


def test_velocity_wavelength_roundtrip():
    sigma_kms = 350.0
    lam = 15000.0
    back = sigma_angstrom_to_kms(sigma_kms_to_angstrom(sigma_kms, lam), lam)
    assert back == pytest.approx(sigma_kms)


def test_resolving_power_matches_the_q1_median():
    # Q1 median LSF_SIG = 13.7 A for compact sources.
    assert resolving_power(13.7, 15000.0) == pytest.approx(465.0, rel=0.02)


def test_effective_sigma_adds_in_quadrature():
    assert effective_sigma(3.0, 4.0) == pytest.approx(5.0)


def test_pixel_edges_are_monotonic_and_cover_the_grid():
    w = np.linspace(12000.0, 19000.0, 531)
    edges = pixel_edges(w)
    assert edges.size == w.size + 1
    assert np.all(np.diff(edges) > 0)
    assert edges[0] < w[0] < edges[1]


def test_pixel_integral_conserves_flux():
    w = 11900.0 + SIR_BINWIDTH_ANGSTROM * np.arange(531)
    profile = gaussian_pixel_integral(w, 15000.0, 20.0, SIR_BINWIDTH_ANGSTROM)
    total = np.sum(profile) * SIR_BINWIDTH_ANGSTROM
    assert total == pytest.approx(1.0, rel=1e-6)


def test_pixel_integral_differs_from_centre_sampling_at_q1_sampling():
    # sigma ~ 1 bin: sampling at bin centres is a measurable bias, which is why
    # the model integrates over pixels.
    w = 11900.0 + SIR_BINWIDTH_ANGSTROM * np.arange(531)
    centre = w[200] + 0.5 * SIR_BINWIDTH_ANGSTROM
    sigma = 13.7
    integrated = gaussian_pixel_integral(w, centre, sigma, SIR_BINWIDTH_ANGSTROM)
    sampled = np.exp(-0.5 * ((w - centre) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    peak_ratio = sampled.max() / integrated.max()
    assert peak_ratio > 1.001
    assert np.sum(sampled) * SIR_BINWIDTH_ANGSTROM == pytest.approx(1.0, rel=1e-3)


def test_kernel_is_normalised():
    kernel = gaussian_kernel(13.7, SIR_BINWIDTH_ANGSTROM)
    assert kernel.sum() == pytest.approx(1.0)
    assert kernel.size % 2 == 1


def test_convolution_preserves_a_flat_continuum():
    model = np.full(300, 2.5)
    out = convolve_lsf(model, 13.7, SIR_BINWIDTH_ANGSTROM)
    assert np.allclose(out, 2.5, rtol=1e-10)


def test_convolution_preserves_integrated_line_flux():
    model = np.zeros(401)
    model[200] = 1.0
    out = convolve_lsf(model, 30.0, SIR_BINWIDTH_ANGSTROM)
    assert out.sum() == pytest.approx(1.0, rel=1e-6)
    assert out.max() < model.max()


def test_convolution_rejects_bad_widths():
    with pytest.raises(ValueError):
        gaussian_kernel(0.0, 13.4)
    with pytest.raises(ValueError):
        gaussian_pixel_integral(np.arange(10.0), 5.0, -1.0)
