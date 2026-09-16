"""Regression tests against a frozen extract of real Q1 data.

``tests/data/EUC_SIR_W-COMBSPEC_102160339_two_objects.fits`` (222 kB) holds two
complete object groups copied verbatim out of
``q1/SIR/102160339/EUC_SIR_W-COMBSPEC_102160339_2024-11-05T16:26:34.614296Z.fits``:

* object 2731173428682078045 - the IRSA cloud-access tutorial reference object,
  an extended source with ``LSF_SIG`` = 98.4 A;
* object 2734482961680140786 - a compact, high signal-to-noise source with
  ``LSF_SIG`` = 12.5 A.

Only the ``<k>_`` extension prefixes were renumbered; every header keyword and
data value is unchanged.  These tests are the guard against silently changing
the archive representation, and they need no network.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from euclid_agn.io.sir import open_sir_file
from euclid_agn.spectra.lsf import resolving_power

FIXTURE = Path(__file__).resolve().parents[1] / "data" / (
    "EUC_SIR_W-COMBSPEC_102160339_two_objects.fits"
)
EXTENDED = 2731173428682078045
COMPACT = 2734482961680140786


@pytest.fixture(scope="module")
def sir():
    with open_sir_file(FIXTURE) as handle:
        yield handle


def test_file_identifies_itself_as_a_sir_product(sir):
    assert sir.primary_header["FITS_DEF"] == "sir.combinedSpectra"
    assert sir.tile_id == 102160339
    assert sir.grism_combination == "RGS"


def test_both_objects_are_discovered(sir):
    assert sir.object_ids() == sorted([EXTENDED, COMPACT])


def test_reference_object_metadata(sir):
    observation = sir.read_observation(EXTENDED)
    assert observation.object_id == EXTENDED
    assert observation.source.ra == pytest.approx(273.117342895423, abs=1e-9)
    assert observation.source.dec == pytest.approx(68.2078045024582, abs=1e-9)
    assert observation.n_dithers == 4


def test_wavelength_grid_is_monotonic_and_physical(sir):
    combined = sir.read_observation(EXTENDED, with_dithers=False).combined
    assert combined.n_pixels == 531
    assert np.all(np.diff(combined.wavelength) > 0)
    assert combined.wavelength[0] == pytest.approx(11900.0)
    assert combined.wavelength[-1] == pytest.approx(19002.0, abs=1.0)
    assert combined.bin_width == pytest.approx(13.4)


def test_flux_is_finite_and_in_physical_units(sir):
    combined = sir.read_observation(COMPACT, with_dithers=False).combined
    ok = combined.usable()
    assert ok.sum() > 300
    assert np.all(np.isfinite(combined.flux[ok]))
    # erg/s/cm2/A for a H~20-22 source: 1e-19 to 1e-14.
    scale = np.median(np.abs(combined.flux[ok]))
    assert 1e-20 < scale < 1e-13


def test_fscale_convention_is_quadratic_in_variance(sir):
    """VAR_phys = VAR_raw * FSCALE**2, pinned against the raw file."""
    with fits.open(FIXTURE) as hdul:
        names = [h.name for h in hdul]
        index = names.index("1_COMBINED1D_SIGNAL")
        fscale = hdul[index].header["FSCALE"]
        raw_signal = np.asarray(hdul[index].data["SIGNAL"], dtype=float)
        raw_var = np.asarray(hdul[index].data["VAR"], dtype=float)
    assert fscale == pytest.approx(1e-16)
    combined = sir.read_observation(COMPACT, with_dithers=False).combined
    assert np.allclose(combined.flux, raw_signal * fscale)
    assert np.allclose(combined.variance, raw_var * fscale**2)


def test_signal_to_noise_survives_the_scaling(sir):
    """The empirical check that fixed the convention, run on real pixels.

    In the raw stored units the pixel-to-pixel scatter of SIGNAL matches
    sqrt(VAR) to within a factor of a few.  Any other FSCALE convention would
    move this ratio by 16 orders of magnitude.
    """
    combined = sir.read_observation(COMPACT, with_dithers=False).combined
    ok = combined.usable()
    residual = np.diff(combined.flux[ok]) / np.sqrt(2.0)
    robust = 1.4826 * np.median(np.abs(residual - np.median(residual)))
    expected = np.median(np.sqrt(combined.variance[ok]))
    assert 0.3 < robust / expected < 4.0


def test_lsf_reflects_source_extent(sir):
    extended = sir.read_observation(EXTENDED, with_dithers=False).combined
    compact = sir.read_observation(COMPACT, with_dithers=False).combined
    assert extended.lsf_sigma == pytest.approx(98.41279, abs=1e-3)
    assert compact.lsf_sigma == pytest.approx(12.51415, abs=1e-3)
    # Compact source resolution is consistent with the NISP requirement R > 380.
    assert resolving_power(compact.lsf_sigma, 15000.0) > 380.0
    # The extended source is degraded by almost an order of magnitude.
    assert resolving_power(extended.lsf_sigma, 15000.0) < 100.0


def test_dither_level_metadata_is_complete(sir):
    observation = sir.read_observation(EXTENDED)
    assert [d.dither_id for d in observation.dithers] == [0, 1, 2, 3]
    assert [d.pointing_id for d in observation.dithers] == [11889, 11890, 11891, 11892]
    assert [d.gwa_position for d in observation.dithers] == [
        "RGS000",
        "RGS180",
        "RGS000",
        "RGS180",
    ]
    assert all(d.detector_id == 32 for d in observation.dithers)
    assert all(d.exposure_time == pytest.approx(549.6422, abs=1e-3) for d in observation.dithers)


def test_contaminants_are_real_and_dither_dependent(sir):
    observation = sir.read_observation(EXTENDED)
    counts = [d.n_contaminants for d in observation.dithers]
    assert counts == [9, 9, 13, 11]
    metrics = observation.contamination_metrics()
    assert metrics["contaminants_total"] == 42
    assert metrics["contaminants_unique"] <= metrics["contaminants_total"]


def test_combined_exposure_time_exceeds_any_single_dither(sir):
    observation = sir.read_observation(EXTENDED)
    assert observation.combined.exposure_time == pytest.approx(2198.569, abs=1e-3)
    assert observation.combined.exposure_time > max(
        d.exposure_time for d in observation.dithers
    )


def test_ndith_column_is_present_and_bounded(sir):
    observation = sir.read_observation(EXTENDED)
    ndith = observation.combined.ndith
    assert ndith is not None
    assert ndith.min() >= 0
    assert ndith.max() == observation.n_dithers


def test_mask_bits_are_read_from_the_primary_header(sir):
    definition = sir.mask_definition
    assert definition.value("NOT_USE") == 1
    assert definition.value("LOW_SNR") == 2
    assert definition.value("ABS_FLUX") == 64


def test_quality_metrics_are_recorded_not_hidden(sir):
    metrics = sir.read_observation(COMPACT, with_dithers=False).combined.quality_metrics()
    assert 0.0 <= metrics["usable_pixel_fraction"] <= 1.0
    assert set(metrics) >= {
        "usable_pixel_fraction",
        "median_quality",
        "lsf_sigma_angstrom",
        "mask_frac_not_use",
        "mask_frac_low_snr",
        "median_snr_per_pixel",
    }


def test_usable_fraction_separates_the_two_objects(sir):
    """Masking, not flux, is what makes a Q1 spectrum fittable.

    The tutorial reference object is 98.3 per cent NOT_USE-masked; its nine
    surviving pixels sit at 18881-19002 A, outside the 12500-18500 A science
    window, and their apparent per-pixel S/N of 411 is an artefact of the red
    edge.  A pipeline that looked only at S/N would accept it.
    """
    extended = sir.read_observation(EXTENDED, with_dithers=False).combined
    compact = sir.read_observation(COMPACT, with_dithers=False).combined

    extended_metrics = extended.quality_metrics()
    assert extended_metrics["usable_pixel_fraction"] == pytest.approx(0.017, abs=0.005)
    assert extended_metrics["mask_frac_not_use"] == pytest.approx(0.983, abs=0.005)
    assert extended_metrics["median_quality"] == 0.0
    assert extended_metrics["wavelength_min_usable"] > 18800.0

    compact_metrics = compact.quality_metrics()
    assert compact_metrics["usable_pixel_fraction"] > 0.98
    assert compact_metrics["median_quality"] > 0.9


def test_low_snr_pixels_are_kept_by_default(sir):
    """Rejecting LOW_SNR would discard a tenth of this spectrum for nothing."""
    compact = sir.read_observation(COMPACT, with_dithers=False).combined
    default = compact.usable().sum()
    strict = compact.usable(reject=("NOT_USE", "LOW_SNR")).sum()
    assert default - strict > 40
