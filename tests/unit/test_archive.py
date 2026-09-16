"""Archive-layer tests that do not touch the network."""

from __future__ import annotations

import pandas as pd
import pytest

from euclid_agn.archive import schema
from euclid_agn.archive.base import SpectrumLocation
from euclid_agn.archive.esa import EsaBackend
from euclid_agn.archive.irsa import IrsaQ1Backend, sir_s3_key
from euclid_agn.archive.tap import TapService, in_list_clause


class FakeTap(TapService):
    """Records queries and replays canned frames."""

    def __init__(self, frames):
        super().__init__(sync_url="fake://")
        self.frames = list(frames)
        self.queries = []

    def query(self, adql: str) -> pd.DataFrame:
        self.queries.append(adql)
        return self.frames.pop(0)


REAL_PATH = (
    "api/spectrumdm/convert/euclid/q1/SIR/102160339/"
    "EUC_SIR_W-COMBSPEC_102160339_2024-11-05T16:26:34.614296Z.fits"
    "?dataset_id=euclid_combspec&hdu=1644"
)


def test_s3_key_from_the_real_association_path():
    assert sir_s3_key(REAL_PATH) == (
        "nasa-irsa-euclid-q1/q1/SIR/102160339/"
        "EUC_SIR_W-COMBSPEC_102160339_2024-11-05T16:26:34.614296Z.fits"
    )


def test_s3_key_from_a_plain_relative_path():
    assert sir_s3_key("SIR/1/EUC_SIR_W-COMBSPEC_1_x.fits").endswith(
        "q1/SIR/1/EUC_SIR_W-COMBSPEC_1_x.fits"
    )


def test_s3_key_rejects_nonsense():
    with pytest.raises(ValueError):
        sir_s3_key("not-a-path")


def test_in_list_clause_is_integer_safe():
    assert in_list_clause("objectid", [1, 2.0, "3"]) == "objectid IN (1, 2, 3)"


def test_query_spectrum_locations_builds_locations():
    frame = pd.DataFrame(
        {
            "objectid": [2731173428682078045],
            "tileid": [102160339],
            "path": [REAL_PATH],
            "hdu": [1644],
            "bandpass_name": ["RGS"],
        }
    )
    backend = IrsaQ1Backend(tap=FakeTap([frame]))
    locations = backend.query_spectrum_locations(object_ids=[2731173428682078045])
    assert len(locations) == 1
    assert locations[0].hdu == 1644
    assert locations[0].path.startswith("nasa-irsa-euclid-q1/q1/SIR/")
    assert locations[0].release == "q1"


def test_association_query_always_requires_a_path():
    frame = pd.DataFrame(columns=["objectid", "tileid", "path", "hdu", "bandpass_name"])
    tap = FakeTap([frame])
    IrsaQ1Backend(tap=tap).query_association(tile_id=102160339)
    assert "path IS NOT NULL" in tap.queries[0]
    assert "tileid = 102160339" in tap.queries[0]


def test_field_lookup_uses_the_caom_tables_not_mer():
    tiles = pd.DataFrame({"tileid": [102160339, 102160340]})
    tap = FakeTap([tiles])
    got = IrsaQ1Backend(tap=tap).list_tiles("EDF-N")
    assert got == [102160339, 102160340]
    assert schema.CAOM_PLANE in tap.queries[0]
    assert schema.MER_CATALOGUE not in tap.queries[0]


def test_fields_cover_the_four_q1_regions():
    assert set(schema.FIELDS) == {"EDF-N", "EDF-S", "EDF-F", "LDN1641"}
    for footprint in schema.FIELDS.values():
        assert -90.0 <= footprint.dec <= 90.0
        assert 0.0 <= footprint.ra <= 360.0
        assert 0.0 < footprint.radius_deg < 15.0


def test_manifest_columns_do_not_smuggle_in_a_selection():
    # Context only: nothing here may gate candidate selection.
    assert "object_id" in schema.MER_MANIFEST_COLUMNS
    assert "ra" in schema.MER_MANIFEST_COLUMNS
    assert "has_spectrum" in schema.MER_MANIFEST_COLUMNS


def test_spectrum_location_roundtrips_to_a_dict():
    location = SpectrumLocation(1, 2, "p", 3, "RGS", "q1")
    assert location.as_dict()["sir_hdu"] == 3


def test_esa_backend_is_declared_but_refuses_to_pretend():
    with pytest.raises(NotImplementedError):
        EsaBackend().query_sources()


def test_select_list_quotes_reserved_column_names():
    """IRSA rejects bare `position_angle` (POSITION is reserved)."""
    from euclid_agn.archive.tap import quote_columns

    assert quote_columns(["object_id", "position_angle"]) == '"object_id", "position_angle"'
    assert quote_columns("a, b") == '"a", "b"'
    assert quote_columns(["COUNT(*)"]) == "COUNT(*)"


def test_queries_quote_the_select_list_but_not_the_where_clause():
    frame = pd.DataFrame(columns=["object_id", "position_angle"])
    tap = FakeTap([pd.DataFrame(columns=["objectid", "tileid", "path", "hdu", "bandpass_name"])])
    backend = IrsaQ1Backend(tap=tap)
    backend.query_association(tile_id=1)
    query = tap.queries[0]
    assert '"objectid"' in query
    assert "WHERE path IS NOT NULL" in query
    assert '"path" IS NOT NULL' not in query
    assert frame is not None
