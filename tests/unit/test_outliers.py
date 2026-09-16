import numpy as np
import pytest

from euclid_agn.fit.screen import ScreenSettings, prepare
from euclid_agn.spectra.outliers import isolated_outliers, outlier_summary, robust_sigma, run_lengths
from euclid_agn.spectra.types import Spectrum1D
from euclid_agn.validation.simulator import LineTruth, SpectrumTruth, simulate_arrays


def spectrum(seed=0, lines=(), spikes=()):
    truth = SpectrumTruth(z=1.2, continuum_flux=1e-17, lines=tuple(lines), noise_flux=2e-19, seed=seed)
    w, f, v, m, q = simulate_arrays(truth)
    f = f.copy()
    for index, height in spikes:
        f[index] += height * np.sqrt(v[index])
    return Spectrum1D(wavelength=w, flux=f, variance=v, mask=m, quality=q, lsf_sigma=13.7, bin_width=13.4)


def test_run_lengths():
    assert run_lengths(np.array([0, 1, 1, 0, 1, 0, 1, 1, 1], bool)).tolist() == [0, 2, 2, 0, 1, 0, 3, 3, 3]
    assert run_lengths(np.zeros(4, bool)).tolist() == [0, 0, 0, 0]


def test_robust_sigma_of_gaussian_noise_is_about_one():
    assert robust_sigma(np.random.default_rng(0).normal(size=20000)) == pytest.approx(1.0, rel=0.03)


def test_single_pixel_spikes_are_caught_in_both_directions():
    s = spectrum(spikes=[(200, +20.0), (300, -15.0)])
    out = isolated_outliers(s.flux, s.variance, s.usable())
    assert out[200] and out[300]
    assert out.sum() <= 4  # nothing else at 5 sigma in white noise of 531 pixels


@pytest.mark.parametrize("flux_line", [3e-15, 6e-16, 3e-16, 1.5e-16])
def test_a_real_line_is_not_an_outlier_at_any_strength(flux_line):
    """Lines from S/N ~ 100 down to ~ 5 have neighbours at >= 0.6 of the peak.

    The first version of this filter removed real lines at S/N 10-20, whose
    top one or two pixels alone clear 5 sigma. This is the regression test.
    """
    s = spectrum(lines=[LineTruth("Halpha", flux_line, 150.0)])
    out = isolated_outliers(s.flux, s.variance, s.usable(), lsf_sigma_pixels=13.7 / 13.4)
    centre = int(np.argmax(s.flux))
    assert not out[centre - 2 : centre + 3].any()


def test_a_line_centred_between_pixels_survives():
    """Two nearly equal high pixels with lower flanks: a line, not a spike."""
    from euclid_agn.models.line_catalog import BY_NAME
    from euclid_agn.constants import C_KMS

    # shift the line by half a pixel via a velocity offset
    half_pixel_kms = C_KMS * 6.7 / (BY_NAME["Halpha"].rest * 2.2)
    s = spectrum(lines=[LineTruth("Halpha", 6e-16, 150.0, velocity_kms=half_pixel_kms)])
    out = isolated_outliers(s.flux, s.variance, s.usable(), lsf_sigma_pixels=13.7 / 13.4)
    centre = int(np.argmax(s.flux))
    assert not out[centre - 2 : centre + 3].any()


def test_two_pixel_spike_is_rejected_three_pixel_feature_is_kept():
    two = spectrum(spikes=[(250, 12.0), (251, 12.0)])
    three = spectrum(spikes=[(250, 12.0), (251, 12.0), (252, 12.0)])
    assert isolated_outliers(two.flux, two.variance, two.usable())[250:252].all()
    assert not isolated_outliers(three.flux, three.variance, three.usable())[250:253].any()


def test_a_spike_on_a_line_wing_is_still_a_spike_but_the_line_survives():
    s = spectrum(lines=[LineTruth("Halpha", 6e-16, 150.0)], spikes=[(150, 20.0)])
    out = isolated_outliers(s.flux, s.variance, s.usable(), lsf_sigma_pixels=13.7 / 13.4)
    assert out[150]
    centre = int(np.argmax(np.where(np.arange(s.n_pixels) == 150, -np.inf, s.flux)))
    assert not out[centre - 1 : centre + 2].any()


def test_white_noise_yields_few_false_outliers():
    counts = [isolated_outliers(spectrum(seed=k).flux, spectrum(seed=k).variance,
                                spectrum(seed=k).usable()).sum() for k in range(20)]
    assert np.mean(counts) < 0.3


def test_prepare_drops_the_outliers_and_counts_them():
    s = spectrum(spikes=[(200, 25.0), (320, -25.0)])
    with_ = prepare(s, ScreenSettings(outlier_threshold=5.0))
    without = prepare(s, ScreenSettings(outlier_threshold=0.0))
    assert with_.n_outliers == 2 and without.n_outliers == 0
    assert with_.wavelength.size == without.wavelength.size - 2
    assert s.wavelength[200] not in with_.wavelength


def test_outlier_summary():
    out = np.array([True, False, True, False]); usable = np.ones(4, bool)
    summary = outlier_summary(out, usable)
    assert summary["n_outlier_pixels"] == 2 and summary["outlier_fraction"] == 0.5
