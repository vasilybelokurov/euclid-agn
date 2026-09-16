"""The noise audit must give the right answer on data with known properties."""

from __future__ import annotations

import numpy as np
import pytest

from euclid_agn.pipeline.noise import (
    audit_spectrum,
    autocorrelation,
    inflation_factor,
    longest_contiguous,
    summarise_audit,
)
from euclid_agn.spectra.types import Spectrum1D
from euclid_agn.validation.simulator import sir_wavelength_grid


def spectrum(flux, variance, mask=None, lsf=13.7):
    w = sir_wavelength_grid()
    n = w.size
    return Spectrum1D(
        wavelength=w,
        flux=flux,
        variance=variance,
        mask=np.zeros(n, dtype=int) if mask is None else mask,
        quality=np.ones(n),
        lsf_sigma=lsf,
        bin_width=13.4,
    )


def white(seed=0, sigma=1e-18, continuum=1e-17):
    rng = np.random.default_rng(seed)
    w = sir_wavelength_grid()
    flux = continuum + rng.normal(0.0, sigma, w.size)
    return spectrum(flux, np.full(w.size, sigma**2))


def correlated(seed=0, sigma=1e-18, continuum=1e-17, width=2.0):
    """Smoothed noise: neighbouring pixels are correlated by construction."""
    rng = np.random.default_rng(seed)
    w = sir_wavelength_grid()
    raw = rng.normal(0.0, 1.0, w.size + 40)
    kernel = np.exp(-0.5 * (np.arange(-20, 21) / width) ** 2)
    kernel /= np.sqrt(np.sum(kernel**2))  # preserve the per-pixel variance
    smooth = np.convolve(raw, kernel, mode="same")[: w.size]
    return spectrum(continuum + sigma * smooth, np.full(w.size, sigma**2))


def test_longest_contiguous_run():
    assert longest_contiguous(np.array([0, 1, 2, 7, 8])).tolist() == [0, 1, 2]
    assert longest_contiguous(np.array([0, 5, 6, 7, 8])).tolist() == [5, 6, 7, 8]
    assert longest_contiguous(np.array([], dtype=int)).size == 0


def test_autocorrelation_of_white_noise_is_zero():
    rng = np.random.default_rng(1)
    r = rng.normal(size=5000)
    acf = autocorrelation(r)
    assert all(abs(v) < 0.05 for v in acf.values())


def test_autocorrelation_detects_smoothed_noise():
    rng = np.random.default_rng(2)
    raw = rng.normal(size=5040)
    kernel = np.ones(3) / 3
    r = np.convolve(raw, kernel, mode="same")
    acf = autocorrelation(r)
    assert acf[1] > 0.5
    assert acf[2] > 0.1
    assert acf[1] > acf[2]


def test_inflation_factor_is_one_for_ideal_noise():
    assert inflation_factor(1.0, {1: 0.0, 2: 0.0}) == pytest.approx(1.0)
    assert inflation_factor(1.0, {1: -0.1, 2: 0.3}) == pytest.approx(1.0)


def test_inflation_factor_grows_with_correlation_and_scatter():
    assert inflation_factor(1.0, {1: 0.5, 2: 0.0}) == pytest.approx(np.sqrt(2.0))
    assert inflation_factor(2.0, {1: 0.0}) == pytest.approx(2.0)


def test_white_noise_spectrum_is_measured_as_ideal():
    audit = audit_spectrum(white(seed=3), object_id=1)
    assert audit is not None
    assert audit.residual_sigma == pytest.approx(1.0, rel=0.1)
    assert abs(audit.autocorrelation[1]) < 0.15
    assert audit.inflation == pytest.approx(1.0, rel=0.25)


def test_underestimated_variance_shows_up_as_excess_scatter():
    """Halving the reported variance must show as sigma_r ~ sqrt(2)."""
    s = white(seed=4)
    bad = Spectrum1D(
        wavelength=s.wavelength,
        flux=s.flux,
        variance=s.variance / 2.0,
        mask=s.mask,
        quality=s.quality,
        lsf_sigma=s.lsf_sigma,
        bin_width=s.bin_width,
    )
    audit = audit_spectrum(bad, object_id=2)
    assert audit.residual_sigma == pytest.approx(np.sqrt(2.0), rel=0.12)


def test_correlated_noise_is_detected():
    audit = audit_spectrum(correlated(seed=5, width=2.0), object_id=3)
    assert audit.autocorrelation[1] > 0.3
    assert audit.inflation > 1.3


def test_emission_lines_are_clipped_rather_than_counted_as_noise():
    from euclid_agn.validation.simulator import LineTruth, SpectrumTruth, simulate_arrays

    truth = SpectrumTruth(
        z=1.2,
        continuum_flux=1e-17,
        noise_flux=1e-18,
        seed=6,
        lines=(LineTruth("Halpha", 3e-15, 200.0),),
    )
    w, f, v, m, q = simulate_arrays(truth)
    audit = audit_spectrum(spectrum(f, v), object_id=4)
    assert audit.clipped_fraction > 0.0
    assert audit.residual_sigma == pytest.approx(1.0, rel=0.25)


def test_too_few_usable_pixels_returns_nothing():
    s = white(seed=7)
    mask = np.ones(s.n_pixels, dtype=int)
    mask[:20] = 0
    blocked = Spectrum1D(
        wavelength=s.wavelength,
        flux=s.flux,
        variance=s.variance,
        mask=mask,
        quality=s.quality,
        lsf_sigma=s.lsf_sigma,
        bin_width=s.bin_width,
    )
    assert audit_spectrum(blocked, object_id=5) is None


def test_summary_of_an_empty_table():
    import pandas as pd

    assert summarise_audit(pd.DataFrame())["n_spectra"] == 0


def test_method_bias_is_small_and_downward():
    """The audit under-reports scatter because the continuum absorbs noise."""
    from euclid_agn.pipeline.noise import measure_method_bias

    bias = measure_method_bias(n_knots=12, n_realisations=20)
    assert 0.9 < bias.residual_sigma < 1.0
    assert -0.15 < bias.acf_lag1 < 0.0


def test_method_bias_grows_with_continuum_flexibility():
    from euclid_agn.pipeline.noise import measure_method_bias

    stiff = measure_method_bias(n_knots=6, n_realisations=20)
    flexible = measure_method_bias(n_knots=50, n_realisations=20)
    assert flexible.residual_sigma < stiff.residual_sigma
    assert flexible.acf_lag1 < stiff.acf_lag1


def test_corrected_summary_removes_the_bias():
    import pandas as pd

    from euclid_agn.pipeline.noise import MethodBias, corrected_summary

    table = pd.DataFrame(
        {
            "object_id": [1, 2],
            "n_used": [400, 400],
            "residual_sigma": [1.2, 1.2],
            "inflation": [1.4, 1.4],
            "clipped_fraction": [0.0, 0.0],
            "acf_lag1": [0.18, 0.18],
            "acf_lag2": [0.0, 0.0],
        }
    )
    bias = MethodBias(n_knots=25, residual_sigma=0.96, acf_lag1=-0.08, inflation=0.96, n_realisations=20)
    summary = corrected_summary(table, bias)
    assert summary["residual_sigma_corrected"] == pytest.approx(1.2 / 0.96)
    assert summary["acf_lag1_corrected"] == pytest.approx(0.26)
    assert summary["chi2_scale_corrected"] > 1.5


def test_inflation_factorises_into_scale_and_correlation():
    audit = audit_spectrum(correlated(seed=8, width=2.0), object_id=9)
    assert audit.variance_scale == pytest.approx(audit.residual_sigma**2)
    assert audit.residual_sigma * audit.correlation_inflation == pytest.approx(audit.inflation)
    assert audit.correlation_inflation > 1.0


def test_rescaling_the_variance_by_the_audit_restores_unit_scatter():
    """Under-reported variance -> sigma_r ~ 1.41 -> rescaled spectrum -> sigma_r ~ 1."""
    s = white(seed=12)
    bad = Spectrum1D(
        wavelength=s.wavelength, flux=s.flux, variance=s.variance / 2.0, mask=s.mask,
        quality=s.quality, lsf_sigma=s.lsf_sigma, bin_width=s.bin_width,
    )
    first = audit_spectrum(bad, object_id=1)
    fixed = bad.with_variance_scale(first.variance_scale)
    second = audit_spectrum(fixed, object_id=1)
    assert second.residual_sigma == pytest.approx(1.0, rel=0.05)
