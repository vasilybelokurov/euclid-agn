import json

import pytest

from euclid_agn.config import Config
from euclid_agn.provenance import RunManifest, file_checksum, package_versions


def test_default_config_is_valid():
    config = Config()
    assert config.archive.backend == "irsa"
    assert config.selection.wavelength_min < config.selection.wavelength_max


def test_config_roundtrips_through_yaml(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(
        "name: t\narchive:\n  backend: irsa\nselection:\n  field: EDF-N\n  limit: 5\n"
    )
    config = Config.from_yaml(path)
    assert config.selection.field == "EDF-N"
    assert config.selection.limit == 5


def test_shipped_configs_parse():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "configs"
    for name in ("q1.yaml", "validation.yaml"):
        Config.from_yaml(root / name)


def test_checksum_changes_with_content_and_is_stable():
    a = Config()
    b = Config()
    assert a.checksum() == b.checksum()
    c = Config(selection={"limit": 7})
    assert c.checksum() != a.checksum()


def test_run_manifest_records_what_reproduction_needs(tmp_path):
    manifest = RunManifest.create("test", Config())
    path = manifest.write(tmp_path)
    payload = json.loads(path.read_text())
    for key in (
        "run_id",
        "timestamp_utc",
        "release",
        "archive_backend",
        "config_checksum",
        "seed",
        "model_version",
        "line_list_version",
        "packages",
    ):
        assert key in payload
    assert payload["packages"]["numpy"] != "missing"


def test_file_checksum_handles_missing_files(tmp_path):
    assert file_checksum(tmp_path / "nope") is None
    target = tmp_path / "a.txt"
    target.write_text("x")
    assert len(file_checksum(target)) == 64


def test_package_versions_cover_the_science_stack():
    versions = package_versions()
    for name in ("numpy", "scipy", "astropy"):
        assert versions[name] not in ("missing", None)
