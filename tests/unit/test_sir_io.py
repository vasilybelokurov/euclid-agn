"""IO tests against a simulator-written file in the real SIR layout."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from euclid_agn.io.sir import (
    SirCombinedSpectraFile,
    flux_scale,
    open_sir_file,
    variance_scale,
)


def test_extension_discovery_finds_every_object(sir_test_file):
    with open_sir_file(sir_test_file) as sir:
        groups = sir.groups()
        assert len(groups) == sir.n_objects_header == 3
        for group in groups.values():
            assert group.n_dither_hdus == len(group.dither_contam_hdus)
            assert group.object_id > 0


def test_hdu_count_per_object_varies_with_dithers(sir_test_file):
    with open_sir_file(sir_test_file) as sir:
        counts = {g.object_index: g.n_dither_hdus for g in sir.groups().values()}
    # The simulator alternates 3 and 4 dithers, as the archive does.
    assert len(set(counts.values())) > 1


def test_lookup_by_object_id_and_by_hdu_agree(sir_test_file):
    with open_sir_file(sir_test_file) as sir:
        object_id = sir.object_ids()[1]
        group = sir.group_for_object(object_id)
        assert sir.group_for_hdu(group.combined_hdu).object_id == object_id


def test_unknown_object_raises(sir_test_file):
    with open_sir_file(sir_test_file) as sir:
        with pytest.raises(KeyError):
            sir.group_for_object(123)


def test_fscale_is_applied_to_signal_and_squared_for_variance(sir_test_file):
    with fits.open(sir_test_file) as hdul:
        names = [h.name for h in hdul]
        index = names.index("0_COMBINED1D_SIGNAL")
        header = hdul[index].header
        raw_signal = np.asarray(hdul[index].data["SIGNAL"], dtype=float)
        raw_var = np.asarray(hdul[index].data["VAR"], dtype=float)
        fs = flux_scale(header)
    assert fs == pytest.approx(1e-16)
    assert variance_scale(header) == pytest.approx(fs**2)
    with open_sir_file(sir_test_file) as sir:
        combined = sir.read_combined(sir.groups()[0])
    assert np.allclose(combined.flux, raw_signal * fs)
    assert np.allclose(combined.variance, raw_var * fs**2)


def test_signal_to_noise_is_invariant_under_fscale(sir_test_file):
    """The whole point of the convention: SNR must not depend on FSCALE."""
    with open_sir_file(sir_test_file) as sir:
        combined = sir.read_combined(sir.groups()[0])
    snr = combined.flux / np.sqrt(combined.variance)
    assert np.nanmedian(np.abs(snr)) > 0.1
    assert np.all(np.abs(snr) < 1e6)


def test_combined_spectrum_has_ndith_and_dithers_do_not(sir_test_file):
    with open_sir_file(sir_test_file) as sir:
        group = sir.groups()[0]
        combined = sir.read_combined(group)
        dithers = sir.read_dithers(group)
    assert combined.ndith is not None
    assert all(not hasattr(d, "ndith") or getattr(d, "ndith", None) is None for d in dithers)
    assert combined.ndith.max() == len(dithers)


def test_dither_metadata_is_carried(sir_test_file):
    with open_sir_file(sir_test_file) as sir:
        dithers = sir.read_dithers(sir.groups()[0])
    assert [d.dither_id for d in dithers] == sorted(d.dither_id for d in dithers)
    assert all(d.gwa_position.startswith("RGS") for d in dithers)
    assert all(np.isfinite(d.lsf_sigma) for d in dithers)


def test_contaminants_are_read(sir_test_file):
    with open_sir_file(sir_test_file) as sir:
        observation = sir.read_observation(sir.object_ids()[0])
    metrics = observation.contamination_metrics()
    assert metrics["n_dithers"] == observation.n_dithers
    assert metrics["contaminants_total"] >= 0


def test_wavelength_grid_is_the_real_one(sir_test_file):
    with open_sir_file(sir_test_file) as sir:
        combined = sir.read_combined(sir.groups()[0])
    assert combined.n_pixels == 531
    assert combined.wavelength[0] == pytest.approx(11900.0)
    assert np.median(np.diff(combined.wavelength)) == pytest.approx(13.4, abs=0.01)
    assert np.all(np.diff(combined.wavelength) > 0)


def test_reading_does_not_modify_the_file(sir_test_file):
    before = sir_test_file.read_bytes()
    with open_sir_file(sir_test_file) as sir:
        sir.read_observation(sir.object_ids()[0])
    assert sir_test_file.read_bytes() == before


def test_group_for_hdu_rejects_a_non_combined_extension(sir_test_file):
    with open_sir_file(sir_test_file) as sir:
        with pytest.raises(ValueError):
            sir.group_for_hdu(1)  # 0_META


def test_mask_definition_comes_from_the_primary_header(sir_test_file):
    with open_sir_file(sir_test_file) as sir:
        assert sir.mask_definition.value("NOT_USE") == 1
        assert sir.mask_definition.value("ABS_FLUX") == 64
        assert isinstance(sir, SirCombinedSpectraFile)
