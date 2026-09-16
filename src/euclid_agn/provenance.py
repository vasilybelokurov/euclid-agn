"""Run provenance.

A science result without a reproducible run manifest is not a pipeline product.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import euclid_agn

PACKAGES_RECORDED = ("numpy", "scipy", "astropy", "pandas", "pyarrow", "s3fs", "fsspec")

#: Bump when the spectral model changes in a way that alters fitted numbers.
MODEL_VERSION = "0.1.0"
#: Bump when the rest-wavelength list or line families change.
LINE_LIST_VERSION = "0.1.0"


def git_commit(repo: str | Path | None = None) -> str | None:
    """Current git SHA, or ``None`` when the tree is not a git repository."""
    repo = Path(repo) if repo else Path(__file__).resolve().parents[2]
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None
    return out.stdout.strip() or None


def git_is_dirty(repo: str | Path | None = None) -> bool | None:
    repo = Path(repo) if repo else Path(__file__).resolve().parents[2]
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None
    return bool(out.stdout.strip())


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": platform.python_version()}
    for name in PACKAGES_RECORDED:
        try:
            module = __import__(name)
        except ImportError:  # pragma: no cover
            versions[name] = "missing"
        else:
            versions[name] = getattr(module, "__version__", "unknown")
    return versions


def file_checksum(path: str | Path, chunk: int = 1 << 20) -> str | None:
    """SHA-256 of a file, or ``None`` if it does not exist."""
    path = Path(path)
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


@dataclass
class RunManifest:
    """Everything needed to reproduce one pipeline run."""

    run_id: str
    command: str
    timestamp_utc: str
    release: str
    archive_backend: str
    config_checksum: str
    config: dict[str, Any]
    seed: int
    euclid_agn_version: str = euclid_agn.__version__
    model_version: str = MODEL_VERSION
    line_list_version: str = LINE_LIST_VERSION
    git_commit: str | None = None
    git_dirty: bool | None = None
    packages: dict[str, str] = field(default_factory=package_versions)
    input_manifest: str | None = None
    input_manifest_checksum: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        command: str,
        config,
        *,
        input_manifest: str | Path | None = None,
        extra: dict[str, Any] | None = None,
    ) -> RunManifest:
        return cls(
            run_id=f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}",
            command=command,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            release=config.archive.release,
            archive_backend=config.archive.backend,
            config_checksum=config.checksum(),
            config=config.to_dict(),
            seed=config.seed,
            git_commit=git_commit(),
            git_dirty=git_is_dirty(),
            input_manifest=str(input_manifest) if input_manifest else None,
            input_manifest_checksum=file_checksum(input_manifest) if input_manifest else None,
            extra=extra or {},
        )

    def write(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "run_manifest.json"
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=2, default=str)
        return path
