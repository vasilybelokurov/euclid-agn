"""Build the source/spectrum manifest.

One row per source that has an extracted Q1 spectrum, carrying its archive
address plus catalogue context.  The manifest is the only thing later stages
read; raw FITS stay in the archive and are never copied wholesale.

Restartability: work is sharded by tile.  A tile whose shard already exists is
skipped unless ``overwrite`` is set, so an interrupted field-scale build
resumes without repeating TAP queries.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from euclid_agn.archive.base import ArchiveBackend
from euclid_agn.archive.irsa import IrsaQ1Backend
from euclid_agn.config import Config
from euclid_agn.provenance import RunManifest

log = logging.getLogger(__name__)


def make_backend(config: Config) -> ArchiveBackend:
    if config.archive.backend == "irsa":
        return IrsaQ1Backend(anon=config.archive.anon_s3, timeout=config.archive.tap_timeout_s)
    from euclid_agn.archive.esa import EsaBackend

    return EsaBackend(release=config.archive.release)


def build_manifest(
    config: Config,
    output: str | Path,
    *,
    backend: ArchiveBackend | None = None,
    field: str | None = None,
    tile_ids: Sequence[int] | None = None,
    object_ids: Sequence[int] | None = None,
    limit: int | None = None,
    shard_dir: str | Path | None = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Query the archive and write ``data/manifests/<name>.parquet``.

    Returns the assembled DataFrame.  A ``run_manifest.json`` is written next
    to the output file.
    """
    backend = backend or make_backend(config)
    output = Path(output)
    field = field or config.selection.field
    tile_ids = list(tile_ids or config.selection.tile_ids)
    object_ids = list(object_ids or config.selection.object_ids)
    limit = limit if limit is not None else config.selection.limit

    field_used: str | None = None
    if object_ids:
        frames = [backend.query_sources(object_ids=object_ids, limit=limit)]
    else:
        if not tile_ids:
            if field is None:
                raise ValueError("one of field, tile_ids or object_ids is required")
            tile_ids = backend.list_tiles(field)  # type: ignore[attr-defined]
            field_used = field
            log.info("field %s resolves to %d tiles", field, len(tile_ids))
        shard_dir = Path(shard_dir) if shard_dir else output.parent / f"{output.stem}_shards"
        shard_dir.mkdir(parents=True, exist_ok=True)
        frames = []
        for tile in tile_ids:
            shard = shard_dir / f"tile_{tile}.parquet"
            if shard.exists() and not overwrite:
                log.info("tile %s: reusing %s", tile, shard)
                frames.append(pd.read_parquet(shard))
                continue
            log.info("tile %s: querying archive", tile)
            df = backend.query_sources(tile_id=tile, limit=limit)
            df.to_parquet(shard, index=False)
            frames.append(df)

    manifest = (
        pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["object_id"])
    )
    if not manifest.empty:
        manifest = manifest.drop_duplicates(subset="object_id").reset_index(drop=True)
    if limit is not None:
        manifest = manifest.head(limit)

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(output, index=False)
    RunManifest.create(
        f"archive build-manifest -> {output}",
        config,
        extra={
            "field": field_used,
            "selected_by": (
                "object_ids" if object_ids else ("tile_ids" if not field_used else "field")
            ),
            "n_tiles": len(tile_ids),
            "n_sources": int(len(manifest)),
            "output": str(output),
        },
    ).write(output.parent)
    log.info("wrote %d sources to %s", len(manifest), output)
    return manifest


def manifest_summary(manifest: pd.DataFrame) -> dict[str, float | int]:
    """Counts used in the journal and the run report."""
    out: dict[str, float | int] = {"n_sources": int(len(manifest))}
    if "tile_id" in manifest:
        out["n_tiles"] = int(manifest["tile_id"].nunique())
    if "spe_class" in manifest:
        for value, count in manifest["spe_class"].value_counts(dropna=False).items():
            out[f"spe_class_{value}"] = int(count)
    if "spe_gal_z" in manifest:
        z = manifest["spe_gal_z"]
        out["n_spe_gal_z"] = int(z.notna().sum())
    return out
