"""``euclid-agn`` command-line interface.

Every command is restartable, deterministic given a seed and config,
batchable and non-interactive, and every bulk command can be pointed at a
single object so development never needs the whole archive.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

from euclid_agn.config import Config

app = typer.Typer(add_completion=False, help="Host-unbiased AGN search in Euclid NISP spectra")
archive_app = typer.Typer(help="Archive queries and manifest building")
spectra_app = typer.Typer(help="Inspect individual spectra")
app.add_typer(archive_app, name="archive")
app.add_typer(spectra_app, name="spectra")


def _load_config(path: Path | None) -> Config:
    return Config.from_yaml(path) if path else Config()


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )


def _object_ids(object_id: list[int] | None, object_id_file: Path | None) -> list[int]:
    ids = list(object_id or [])
    if object_id_file:
        ids += [
            int(line.strip())
            for line in Path(object_id_file).read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
    return ids


@archive_app.command("build-manifest")
def build_manifest(
    output: Path = typer.Option(..., help="Output parquet path"),
    config: Path = typer.Option(None, help="YAML config"),
    field: str = typer.Option(None, help="EDF-N, EDF-S, EDF-F or LDN1641"),
    tile: list[int] = typer.Option(None, help="Tile id (repeatable)"),
    object_id: list[int] = typer.Option(None, help="Object id (repeatable)"),
    object_id_file: Path = typer.Option(None, help="File with one object id per line"),
    limit: int = typer.Option(None, help="Stop after this many sources"),
    overwrite: bool = typer.Option(False, help="Re-query tiles that already have a shard"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Query the archive and write the source/spectrum manifest."""
    _setup_logging(verbose)
    from euclid_agn.pipeline.ingest import build_manifest as _build
    from euclid_agn.pipeline.ingest import manifest_summary

    cfg = _load_config(config)
    manifest = _build(
        cfg,
        output,
        field=field,
        tile_ids=tile or None,
        object_ids=_object_ids(object_id, object_id_file) or None,
        limit=limit,
        overwrite=overwrite,
    )
    typer.echo(json.dumps(manifest_summary(manifest), indent=2))


@archive_app.command("list-tiles")
def list_tiles(
    field: str = typer.Argument(..., help="EDF-N, EDF-S, EDF-F or LDN1641"),
    config: Path = typer.Option(None, help="YAML config"),
) -> None:
    """Print the tile ids inside a field."""
    from euclid_agn.pipeline.ingest import make_backend

    backend = make_backend(_load_config(config))
    for tile in backend.list_tiles(field):  # type: ignore[attr-defined]
        typer.echo(tile)


@spectra_app.command("inspect")
def inspect(
    object_id: int = typer.Option(..., help="Euclid MER object id"),
    config: Path = typer.Option(None, help="YAML config"),
    plot: Path = typer.Option(None, help="Write a diagnostic plot here"),
) -> None:
    """Fetch one spectrum and print its quality and contamination metrics."""
    from euclid_agn.pipeline.ingest import make_backend

    backend = make_backend(_load_config(config))
    locations = backend.query_spectrum_locations(object_ids=[object_id])
    if not locations:
        raise typer.Exit(code=1)
    obs = backend.open_observation(locations[0])  # type: ignore[attr-defined]
    payload = {
        "object_id": obs.object_id,
        "ra": obs.source.ra,
        "dec": obs.source.dec,
        "tile_id": obs.source.tile_id,
        "n_dithers": obs.n_dithers,
        "combined": obs.combined.quality_metrics(),
        "contamination": obs.contamination_metrics(),
        "dithers": [
            {
                "dither_id": d.dither_id,
                "pointing_id": d.pointing_id,
                "gwa_position": d.gwa_position,
                "n_contaminants": d.n_contaminants,
                "lsf_sigma_angstrom": d.lsf_sigma,
            }
            for d in obs.dithers
        ],
    }
    typer.echo(json.dumps(payload, indent=2, default=float))
    if plot:
        from euclid_agn.plotting.diagnostics import plot_observation

        plot_observation(obs, plot)
        typer.echo(f"wrote {plot}")


@app.command("version")
def version() -> None:
    """Print package and model versions."""
    from euclid_agn import __version__
    from euclid_agn.provenance import LINE_LIST_VERSION, MODEL_VERSION, git_commit

    typer.echo(
        json.dumps(
            {
                "euclid_agn": __version__,
                "model_version": MODEL_VERSION,
                "line_list_version": LINE_LIST_VERSION,
                "git_commit": git_commit(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    app()
