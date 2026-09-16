"""Cache tests with a fake filesystem: no network, no ~/data writes."""

from __future__ import annotations

from pathlib import Path

import pytest

from euclid_agn.io.cache import DEFAULT_CACHE_ROOT, ArchiveCache, strip_bucket

REAL_KEY = (
    "nasa-irsa-euclid-q1/q1/SIR/102160339/"
    "EUC_SIR_W-COMBSPEC_102160339_2024-11-05T16:26:34.614296Z.fits"
)


class FakeFS:
    """Copies a source file, and records what it was asked for."""

    protocol = "s3"

    def __init__(self, payload: bytes = b"fits", fail: bool = False):
        self.payload = payload
        self.requests: list[str] = []
        self.fail = fail

    def get(self, key: str, destination: str) -> None:
        self.requests.append(key)
        if self.fail:
            raise OSError("network went away")
        Path(destination).write_bytes(self.payload)


def test_default_root_is_the_users_data_directory():
    assert DEFAULT_CACHE_ROOT == Path("~/data/euclid").expanduser()
    assert ArchiveCache().root == DEFAULT_CACHE_ROOT


def test_key_maps_to_the_archive_layout(tmp_path):
    cache = ArchiveCache(root=tmp_path)
    assert strip_bucket(REAL_KEY).startswith("q1/SIR/102160339/")
    assert cache.local_path(REAL_KEY) == tmp_path / strip_bucket(REAL_KEY)
    assert cache.local_path("s3://" + REAL_KEY) == cache.local_path(REAL_KEY)


def test_fetch_downloads_once_then_hits(tmp_path):
    cache = ArchiveCache(root=tmp_path)
    fs = FakeFS(b"payload")
    first = cache.fetch(REAL_KEY, fs)
    second = cache.fetch(REAL_KEY, fs)
    assert first == second
    assert first.read_bytes() == b"payload"
    assert len(fs.requests) == 1


def test_force_refetches(tmp_path):
    cache = ArchiveCache(root=tmp_path)
    fs = FakeFS()
    cache.fetch(REAL_KEY, fs)
    cache.fetch(REAL_KEY, fs, force=True)
    assert len(fs.requests) == 2


def test_failed_download_leaves_no_partial_file(tmp_path):
    cache = ArchiveCache(root=tmp_path)
    fs = FakeFS(fail=True)
    with pytest.raises(OSError):
        cache.fetch(REAL_KEY, fs)
    assert not cache.contains(REAL_KEY)
    assert list(cache.local_path(REAL_KEY).parent.glob("*")) == []


def test_disabled_cache_streams_instead(tmp_path):
    cache = ArchiveCache(root=tmp_path, enabled=False)
    fs = FakeFS()
    assert cache.fetch(REAL_KEY, fs) is None
    with cache.open(REAL_KEY, fs) as (target, filesystem):
        assert target == REAL_KEY
        assert filesystem is fs
    assert fs.requests == []


def test_open_yields_a_local_path_when_cached(tmp_path):
    cache = ArchiveCache(root=tmp_path)
    fs = FakeFS()
    with cache.open(REAL_KEY, fs) as (target, filesystem):
        assert filesystem is None
        assert Path(target).is_file()


def test_usage_and_budget(tmp_path):
    cache = ArchiveCache(root=tmp_path, max_bytes=4)
    cache.fetch(REAL_KEY, FakeFS(b"0123456789"))
    assert cache.usage_bytes() == 10
    assert cache.over_budget()
    assert cache.describe()["n_files"] == 1


def test_backend_uses_the_cache(tmp_path, sir_test_file):
    """The backend must open the cached file, not re-stream it."""
    from euclid_agn.archive.base import SpectrumLocation
    from euclid_agn.archive.irsa import IrsaQ1Backend

    payload = Path(sir_test_file).read_bytes()
    fs = FakeFS(payload)
    backend = IrsaQ1Backend(filesystem=fs, cache=ArchiveCache(root=tmp_path))
    location = SpectrumLocation(
        object_id=0, tile_id=999999999, path=REAL_KEY, hdu=2, release="q1"
    )
    first = backend.open_observation(location)
    second = backend.open_observation(location)
    assert first.object_id == second.object_id
    assert len(fs.requests) == 1
