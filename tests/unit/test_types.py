import numpy as np
import pytest

from euclid_agn.spectra.types import CombinedSpectrum, Spectrum1D


def make(n=20, **kwargs):
    base = dict(
        wavelength=np.linspace(12000.0, 12000.0 + 13.4 * (n - 1), n),
        flux=np.ones(n) * 1e-17,
        variance=np.ones(n) * 1e-36,
        mask=np.zeros(n, dtype=int),
        quality=np.ones(n),
        lsf_sigma=13.7,
        bin_width=13.4,
    )
    base.update(kwargs)
    return Spectrum1D(**base)


def test_arrays_are_read_only():
    spectrum = make()
    with pytest.raises(ValueError):
        spectrum.flux[0] = 1.0


def test_length_mismatch_is_rejected():
    with pytest.raises(ValueError):
        make(flux=np.ones(5))


def test_non_monotonic_wavelength_is_rejected():
    with pytest.raises(ValueError):
        make(wavelength=np.array([1.0, 3.0, 2.0] + [4.0] * 17))


def test_usable_excludes_masked_nonfinite_and_nonpositive_variance():
    n = 6
    spectrum = make(
        n=n,
        mask=np.array([0, 1, 0, 0, 0, 0]),
        flux=np.array([1.0, 1.0, np.nan, 1.0, 1.0, 1.0]) * 1e-17,
        variance=np.array([1.0, 1.0, 1.0, 0.0, -1.0, 1.0]) * 1e-36,
    )
    assert spectrum.usable().tolist() == [True, False, False, False, False, True]


def test_masked_pixels_are_never_interpolated():
    # The contract is that usable() drops pixels; no array is modified.
    spectrum = make(n=5, mask=np.array([0, 1, 0, 0, 0]))
    assert np.count_nonzero(~spectrum.usable()) == 1
    assert np.isfinite(spectrum.flux).all()


def test_quality_metrics_report_what_is_needed_for_the_selection_function():
    spectrum = make(n=10, mask=np.array([0, 1, 0, 0, 0, 0, 2, 0, 0, 0]))
    metrics = spectrum.quality_metrics()
    assert metrics["usable_pixel_fraction"] == pytest.approx(0.9)
    assert metrics["mask_frac_not_use"] == pytest.approx(0.1)
    assert metrics["mask_frac_low_snr"] == pytest.approx(0.1)
    assert metrics["lsf_sigma_angstrom"] == pytest.approx(13.7)
    assert np.isfinite(metrics["median_snr_per_pixel"])


def test_combined_spectrum_tracks_ndith():
    n = 8
    spectrum = CombinedSpectrum(
        wavelength=np.arange(n, dtype=float) * 13.4 + 12000.0,
        flux=np.ones(n),
        variance=np.ones(n),
        mask=np.zeros(n, dtype=int),
        quality=np.ones(n),
        lsf_sigma=13.7,
        bin_width=13.4,
        ndith=np.array([0, 1, 2, 4, 4, 4, 2, 0]),
    )
    metrics = spectrum.quality_metrics()
    assert metrics["ndith_max"] == 4
    with pytest.raises(ValueError):
        CombinedSpectrum(
            wavelength=np.arange(n, dtype=float),
            flux=np.ones(n),
            variance=np.ones(n),
            mask=np.zeros(n, dtype=int),
            quality=np.ones(n),
            lsf_sigma=13.7,
            bin_width=1.0,
            ndith=np.ones(3),
        )


def test_variance_scale_returns_a_derived_copy_and_records_the_factor():
    spectrum = make()
    scaled = spectrum.with_variance_scale(2.0)
    assert np.allclose(scaled.variance, 2.0 * spectrum.variance)
    assert np.allclose(spectrum.variance, 1e-36)  # the original is untouched
    assert scaled.metadata["variance_scale"] == pytest.approx(2.0)
    assert scaled.with_variance_scale(1.5).metadata["variance_scale"] == pytest.approx(3.0)
    with pytest.raises(ValueError):
        spectrum.with_variance_scale(0.0)
