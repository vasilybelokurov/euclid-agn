"""Opt-in end-to-end tests against the live IRSA service.

Run with::

    PYTHONPATH=src pytest tests/online -m online

They are excluded from the default run (``addopts = -m 'not online'``) so
ordinary CI needs no network.
"""

from __future__ import annotations

import numpy as np
import pytest

from euclid_agn.archive.irsa import IrsaQ1Backend

pytestmark = pytest.mark.online

REFERENCE_OBJECT_ID = 2731173428682078045
REFERENCE_TILE_ID = 102160339


@pytest.fixture(scope="module")
def backend():
    return IrsaQ1Backend()


def test_object_lookup_resolves_to_a_file_and_hdu(backend):
    locations = backend.query_spectrum_locations(object_ids=[REFERENCE_OBJECT_ID])
    assert len(locations) == 1
    location = locations[0]
    assert location.object_id == REFERENCE_OBJECT_ID
    assert location.tile_id == REFERENCE_TILE_ID
    assert location.hdu == 1644
    assert location.path.startswith("nasa-irsa-euclid-q1/q1/SIR/102160339/")
    assert location.path.endswith(".fits")


def test_lazy_s3_read_returns_the_expected_object(backend):
    location = backend.query_spectrum_locations(object_ids=[REFERENCE_OBJECT_ID])[0]
    observation = backend.open_observation(location)
    assert observation.object_id == REFERENCE_OBJECT_ID
    assert observation.source.ra == pytest.approx(273.1173, abs=1e-3)
    assert observation.source.dec == pytest.approx(68.2078, abs=1e-3)
    assert observation.n_dithers == 4

    combined = observation.combined
    assert combined.n_pixels == 531
    assert np.all(np.diff(combined.wavelength) > 0)
    assert combined.wavelength[0] == pytest.approx(11900.0)

    # This source is 98.3 per cent NOT_USE-masked: the tutorial object is a
    # large extended galaxy and its combined spectrum is unusable for fitting.
    # The pipeline must report that rather than fit nine red-edge pixels.
    ok = combined.usable()
    assert 0 < ok.sum() < 20
    assert np.all(np.isfinite(combined.flux[ok]))
    assert np.all(combined.variance[ok] > 0)
    metrics = combined.quality_metrics()
    assert metrics["usable_pixel_fraction"] < 0.05
    assert metrics["wavelength_min_usable"] > 18000.0


def test_manifest_query_returns_context_without_selecting_on_it(backend):
    frame = backend.query_sources(object_ids=[REFERENCE_OBJECT_ID])
    assert len(frame) == 1
    row = frame.iloc[0]
    assert int(row["object_id"]) == REFERENCE_OBJECT_ID
    assert int(row["tile_id"]) == REFERENCE_TILE_ID
    assert row["sir_s3_key"].startswith("nasa-irsa-euclid-q1/")
    for column in ("ra", "dec", "flux_h_2fwhm_aper", "release"):
        assert column in frame.columns


def test_field_resolution_partitions_q1(backend):
    counts = {name: len(backend.list_tiles(name)) for name in ("EDF-N", "LDN1641")}
    assert counts["EDF-N"] == 124
    assert counts["LDN1641"] == 8
    assert REFERENCE_TILE_ID in backend.list_tiles("EDF-N")


def test_association_table_reports_missing_spectra_as_null_paths(backend):
    frame = backend.query_association(tile_id=REFERENCE_TILE_ID, limit=50)
    assert len(frame) == 50
    assert frame["path"].notna().all()
    assert (frame["tileid"] == REFERENCE_TILE_ID).all()
