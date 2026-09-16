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


@archive_app.command("availability")
def availability(
    config: Path = typer.Option(None, help="YAML config"),
    field: list[str] = typer.Option(None, help="Field (repeatable); default all four"),
    sample_tile: list[int] = typer.Option(
        None, help="Tile id to open and measure pixel-level usability (repeatable)"
    ),
    max_objects: int = typer.Option(None, help="Objects per sampled tile"),
    output: Path = typer.Option(None, help="Write the per-object quality table here"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Report how many Q1 spectra exist and what fraction is usable."""
    _setup_logging(verbose)
    import pandas as pd

    from euclid_agn.pipeline.availability import (
        count_availability,
        spectrum_quality_table,
        summarise_quality,
    )
    from euclid_agn.pipeline.ingest import make_backend

    cfg = _load_config(config)
    backend = make_backend(cfg)
    payload: dict = {}
    if not sample_tile:
        counts = count_availability(backend, field or None)
        payload["catalogue"] = counts.to_dict(orient="records")
    tables = []
    for tile in sample_tile or []:
        locations = backend.query_spectrum_locations(tile_id=tile, limit=1)
        if not locations:
            continue
        path = backend.download_file(locations[0])
        tables.append(spectrum_quality_table(str(path), max_objects=max_objects))
    if tables:
        table = pd.concat(tables, ignore_index=True)
        payload["pixels"] = summarise_quality(
            table, cfg.selection.min_usable_pixel_fraction
        )
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            table.to_parquet(output, index=False)
            payload["table"] = str(output)
    typer.echo(json.dumps(payload, indent=2, default=float))


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


@spectra_app.command("noise-audit")
def noise_audit(
    path: list[Path] = typer.Option(None, help="Local SIR file (repeatable)"),
    tile: list[int] = typer.Option(None, help="Tile id to fetch and audit (repeatable)"),
    config: Path = typer.Option(None, help="YAML config"),
    n_knots: int = typer.Option(25, help="Continuum flexibility used to form residuals"),
    max_objects: int = typer.Option(None, help="Objects per file"),
    output: Path = typer.Option(None, help="Write the per-spectrum audit table here"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Measure residual scatter and pixel-to-pixel correlation in real spectra.

    Reports the factor by which a naive chi-squared, built from the reported
    variances and an assumption of independent pixels, is wrong.
    """
    _setup_logging(verbose)
    import pandas as pd

    from euclid_agn.pipeline.noise import audit_file, corrected_summary, measure_method_bias

    files = [str(p) for p in (path or [])]
    if tile:
        from euclid_agn.pipeline.ingest import make_backend

        backend = make_backend(_load_config(config))
        for tile_id in tile:
            locations = backend.query_spectrum_locations(tile_id=tile_id, limit=1)
            if locations:
                files.append(str(backend.download_file(locations[0])))
    if not files:
        typer.echo("nothing to audit: pass --path or --tile")
        raise typer.Exit(code=1)

    tables = [audit_file(f, n_knots=n_knots, max_objects=max_objects) for f in files]
    table = pd.concat([t for t in tables if not t.empty], ignore_index=True)
    bias = measure_method_bias(n_knots=n_knots)
    payload = corrected_summary(table, bias)
    payload["n_files"] = len(files)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(output, index=False)
        payload["table"] = str(output)
    typer.echo(json.dumps(payload, indent=2, default=float))


@app.command("screen")
def screen(
    manifest: Path = typer.Option(..., help="Source manifest parquet"),
    output: Path = typer.Option(..., help="Output parquet path"),
    config: Path = typer.Option(None, help="YAML config"),
    object_id: list[int] = typer.Option(None, help="Restrict to these object ids"),
    object_id_file: Path = typer.Option(None, help="File with one object id per line"),
    limit: int = typer.Option(None, help="Stop after this many sources"),
    overwrite: bool = typer.Option(False, help="Re-screen files that already have a shard"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Stage 1: screen every usable spectrum in a manifest for broad lines."""
    _setup_logging(verbose)
    from euclid_agn.pipeline.screening import screen_manifest

    cfg = _load_config(config)
    results = screen_manifest(
        cfg,
        manifest,
        output,
        object_ids=_object_ids(object_id, object_id_file) or None,
        limit=limit,
        overwrite=overwrite,
    )
    summary = {"n_rows": int(len(results))}
    if not results.empty:
        summary["n_objects"] = int(results["object_id"].nunique())
        summary["median_delta_chi2_effective"] = float(
            results["delta_chi2_refined_effective"].median()
        )
        summary["n_above_25"] = int((results["delta_chi2_refined_effective"] > 25).sum())
    typer.echo(json.dumps(summary, indent=2, default=float))


@app.command("plot")
def plot(
    screen: Path = typer.Option(..., help="Screening results parquet"),
    files: list[Path] = typer.Option(None, help="SIR files to read spectra from (repeatable)"),
    glob: str = typer.Option(None, help="Glob of SIR files, e.g. '~/data/euclid/q1/SIR/*/*.fits'"),
    directory: Path = typer.Option(Path("plots"), help="Output directory for PNGs"),
    n_candidates: int = typer.Option(8, help="How many candidates to draw"),
    no_dithers: bool = typer.Option(False, help="Skip the per-dither spectrum pages"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Write a PNG atlas of spectra and their fits."""
    _setup_logging(verbose)
    import glob as globmodule

    import pandas as pd

    from euclid_agn.plotting.atlas import build_atlas

    paths = [str(f) for f in (files or [])]
    if glob:
        paths += sorted(globmodule.glob(str(Path(glob).expanduser())))
    if not paths:
        typer.echo("no SIR files given: pass --files or --glob")
        raise typer.Exit(code=1)

    table = pd.read_parquet(screen)
    created = build_atlas(
        table,
        paths,
        directory=directory,
        n_candidates=n_candidates,
        with_dithers=not no_dithers,
    )
    typer.echo(
        json.dumps({k: len(v) for k, v in created.items()} | {"directory": str(directory)}, indent=2)
    )


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
