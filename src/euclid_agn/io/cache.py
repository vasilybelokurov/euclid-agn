"""Local cache for archive files.

Downloaded Euclid data live outside the repository, under ``~/data/euclid`` by
default, mirroring the archive's own directory layout::

    ~/data/euclid/q1/SIR/102160339/EUC_SIR_W-COMBSPEC_102160339_<ts>.fits

Two reasons for a cache at all.  A Q1 tile file is ~112 MB and holds 1000
objects, so any run that touches more than a handful of objects in the same tile
pays for the file once instead of once per object.  And repeated development
runs stop hammering the archive.

Cached files are treated as immutable: they are byte-identical copies of the
archive product, written atomically through a temporary file so an interrupted
download can never be mistaken for a complete one.
"""

from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

#: Where downloaded Euclid data go unless a config says otherwise.
DEFAULT_CACHE_ROOT = Path("~/data/euclid").expanduser()

#: Bucket prefixes stripped from a key before it becomes a local path.
_BUCKET_PREFIXES = ("s3://", "nasa-irsa-euclid-q1/")


def strip_bucket(key: str) -> str:
    """Archive key -> path relative to the cache root."""
    text = key
    for prefix in _BUCKET_PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix) :]
    # A stripped "s3://" may leave the bucket name behind.
    if text.startswith("nasa-irsa-euclid-q1/"):
        text = text[len("nasa-irsa-euclid-q1/") :]
    return text.lstrip("/")


@dataclass
class ArchiveCache:
    """A content-addressed-by-path mirror of archive files on local disk.

    Parameters
    ----------
    root : Path
        Cache root; created on demand.
    enabled : bool
        When false, :meth:`fetch` returns ``None`` and callers stream from the
        archive instead.
    max_bytes : int or None
        Advisory ceiling.  Nothing is evicted automatically - deleting data the
        user downloaded is not the pipeline's decision - but
        :meth:`usage_bytes` and :meth:`over_budget` let a caller warn.
    """

    root: Path = DEFAULT_CACHE_ROOT
    enabled: bool = True
    max_bytes: int | None = None

    def __post_init__(self) -> None:
        self.root = Path(self.root).expanduser()

    def local_path(self, key: str) -> Path:
        """Where ``key`` would live locally, whether or not it is there."""
        return self.root / strip_bucket(key)

    def contains(self, key: str) -> bool:
        path = self.local_path(key)
        return path.is_file() and path.stat().st_size > 0

    def fetch(self, key: str, filesystem, force: bool = False) -> Path | None:
        """Ensure ``key`` is present locally and return its path.

        Returns ``None`` when the cache is disabled, so the caller can fall
        back to streaming.  The download is atomic: bytes land in a temporary
        file next to the target and are renamed only on success.
        """
        if not self.enabled:
            return None
        path = self.local_path(key)
        if self.contains(key) and not force:
            log.debug("cache hit %s", path)
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + f".part{os.getpid()}")
        log.info("downloading %s -> %s", key, path)
        try:
            filesystem.get(strip_key_for(filesystem, key), str(temporary))
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return path

    @contextmanager
    def open(self, key: str, filesystem) -> Iterator[tuple[str, object | None]]:
        """Yield ``(path_or_key, filesystem_or_None)`` for :func:`open_sir_file`.

        A cached file is opened locally with no filesystem object; otherwise the
        archive key and its filesystem are passed straight through so the reader
        streams byte ranges.
        """
        path = self.fetch(key, filesystem) if self.enabled else None
        if path is not None:
            yield str(path), None
        else:
            yield key, filesystem

    def usage_bytes(self) -> int:
        if not self.root.exists():
            return 0
        return sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())

    def over_budget(self) -> bool:
        return self.max_bytes is not None and self.usage_bytes() > self.max_bytes

    def describe(self) -> dict[str, object]:
        return {
            "root": str(self.root),
            "enabled": self.enabled,
            "n_files": sum(1 for p in self.root.rglob("*") if p.is_file())
            if self.root.exists()
            else 0,
            "usage_bytes": self.usage_bytes(),
        }


def strip_key_for(filesystem, key: str) -> str:
    """Key in the form the given filesystem expects (bucket included for s3fs)."""
    text = key.removeprefix("s3://")
    if not text.startswith("nasa-irsa-euclid-q1/") and getattr(filesystem, "protocol", "") in (
        "s3",
        ("s3", "s3a"),
    ):
        text = f"nasa-irsa-euclid-q1/{text.lstrip('/')}"
    return text
