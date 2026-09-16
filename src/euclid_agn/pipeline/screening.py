"""Stage-1 production loop: manifest in, screening table out.

Work is grouped by archive file rather than by object.  A Q1 SIR file holds
hundreds of objects and costs one download; opening it once per object would
multiply the I/O by that factor for no gain.

The loop is restartable at file granularity: each file's results are written to
a shard, and a rerun reuses shards that already exist.  Nothing about the
science depends on the order in which files are processed.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from euclid_agn.config import Config
from euclid_agn.fit.hypotheses import RedshiftHypothesis, from_catalogue
from euclid_agn.fit.hypotheses import generate as generate_hypotheses
from euclid_agn.fit.screen import ScreenSettings, screen_spectrum
from euclid_agn.io.sir import open_sir_file
from euclid_agn.pipeline.ingest import make_backend
from euclid_agn.pipeline.noise import audit_spectrum
from euclid_agn.provenance import RunManifest

log = logging.getLogger(__name__)

#: Manifest columns copied into every screening row as host/data context.
CONTEXT_COLUMNS: tuple[str, ...] = (
    "ra",
    "dec",
    "tile_id",
    "release",
    "flux_h_2fwhm_aper",
    "point_like_prob",
    "extended_prob",
    "segmentation_area",
    "sersic_sersic_nir_radius",
    "sersic_sersic_nir_axis_ratio",
    "phz_mode_1",
    "spe_class",
    "spe_gal_z",
    "spe_qso_z",
    "spe_n_dith_med",
)


def settings_from_config(config: Config) -> ScreenSettings:
    """Translate the YAML configuration into screening settings."""
    from euclid_agn.models.broad import sigma_grid

    return ScreenSettings(
        n_knots=config.continuum.n_knots,
        broad_sigma_kms=tuple(
            sigma_grid(
                config.broad.sigma_min_kms, config.broad.sigma_max_kms, config.broad.sigma_grid_n
            )
        ),
        wavelength_min=config.selection.wavelength_min,
        wavelength_max=config.selection.wavelength_max,
        narrow_velocity_half_width_kms=config.narrow.velocity_half_width_kms,
        narrow_velocity_step_kms=config.narrow.velocity_step_kms,
        narrow_smoothness=config.narrow.smoothness_lambda,
    )


def hypotheses_for(row, config: Config, spectrum=None) -> list[RedshiftHypothesis]:
    """Redshift hypotheses for one source, honouring the configured origins."""
    sources = tuple(config.screening.redshift_sources)
    if spectrum is None:
        return from_catalogue(row)
    usable = spectrum.usable()
    return generate_hypotheses(
        spectrum.wavelength[usable],
        spectrum.flux[usable],
        spectrum.variance[usable],
        row=row,
        sources=sources,
        z_min=config.screening.blind_z_min,
        z_max=config.screening.blind_z_max,
        blind_step_kms=config.screening.blind_z_step_kms,
        wavelength_min=config.selection.wavelength_min,
        wavelength_max=config.selection.wavelength_max,
    )


def screen_file(
    path: str,
    manifest: pd.DataFrame,
    config: Config,
    settings: ScreenSettings | None = None,
    filesystem=None,
    rows_per_object: int = 1,
) -> pd.DataFrame:
    """Screen every manifest object that lives in one SIR file."""
    settings = settings or settings_from_config(config)
    by_id = {int(r["object_id"]): r for _, r in manifest.iterrows()}
    out: list[pd.DataFrame] = []
    with open_sir_file(path, filesystem=filesystem) as sir:
        for group in sir.groups().values():
            row = by_id.get(group.object_id)
            if row is None:
                continue
            combined = sir.read_combined(group)
            metrics = combined.quality_metrics()
            if metrics["usable_pixel_fraction"] < config.selection.min_usable_pixel_fraction:
                continue
            audit = audit_spectrum(combined, object_id=group.object_id)
            # The measured per-object noise scale enters the likelihood here:
            # the variance is rescaled so that chi-squared means what it says,
            # and only the pixel-correlation part of the inflation is left to
            # apply to Delta chi-squared afterwards.
            if audit is not None and config.screening.noise_scale_free:
                fitted = combined.with_variance_scale(max(audit.variance_scale, 1.0))
                inflation = max(audit.correlation_inflation, 1.0)
            else:
                fitted = combined
                inflation = max(audit.inflation, 1.0) if audit is not None else 1.0
            hypotheses = hypotheses_for(row, config, fitted)
            context = {
                column: row[column] for column in CONTEXT_COLUMNS if column in row.index
            }
            context["noise_residual_sigma"] = (
                audit.residual_sigma if audit is not None else float("nan")
            )
            context["noise_variance_scale"] = float(
                fitted.metadata.get("variance_scale", 1.0)
            )
            context["noise_acf_lag1"] = (
                audit.autocorrelation.get(1, float("nan")) if audit is not None else float("nan")
            )
            phz = row.get("phz_mode_1") if hasattr(row, "get") else None
            z_prior = float(phz) if phz is not None and np.isfinite(float(phz)) else None
            table = screen_spectrum(
                fitted,
                hypotheses,
                settings,
                object_id=group.object_id,
                noise_inflation=inflation,
                context=context,
                z_prior=z_prior,
            )
            if not table.empty:
                out.append(table.head(rows_per_object))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def screen_manifest(
    config: Config,
    manifest_path: str | Path,
    output: str | Path,
    *,
    backend=None,
    limit: int | None = None,
    object_ids: Sequence[int] | None = None,
    shard_dir: str | Path | None = None,
    overwrite: bool = False,
    rows_per_object: int = 1,
) -> pd.DataFrame:
    """Run Stage 1 over a manifest and write ``screen_results.parquet``."""
    backend = backend or make_backend(config)
    manifest = pd.read_parquet(manifest_path)
    if object_ids:
        manifest = manifest[manifest["object_id"].isin(list(object_ids))]
    if limit is not None:
        manifest = manifest.head(limit)
    if manifest.empty:
        raise ValueError("no sources selected from the manifest")

    output = Path(output)
    shard_dir = Path(shard_dir) if shard_dir else output.parent / f"{output.stem}_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    settings = settings_from_config(config)

    frames: list[pd.DataFrame] = []
    for key, chunk in manifest.groupby("sir_s3_key"):
        shard = shard_dir / f"{Path(str(key)).stem}.parquet"
        if shard.exists() and not overwrite:
            log.info("reusing %s", shard)
            frames.append(pd.read_parquet(shard))
            continue
        local = backend.cache.fetch(str(key), backend.filesystem)
        source = str(local) if local is not None else str(key)
        log.info("screening %d objects from %s", len(chunk), source)
        table = screen_file(
            source,
            chunk,
            config,
            settings=settings,
            filesystem=None if local is not None else backend.filesystem,
            rows_per_object=rows_per_object,
        )
        table.to_parquet(shard, index=False)
        frames.append(table)

    results = pd.concat([f for f in frames if not f.empty], ignore_index=True) if frames else pd.DataFrame()
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(output, index=False)
    RunManifest.create(
        f"screen -> {output}",
        config,
        input_manifest=manifest_path,
        extra={
            "n_sources": int(len(manifest)),
            "n_rows": int(len(results)),
            "n_files": int(manifest["sir_s3_key"].nunique()),
            "output": str(output),
        },
    ).write(output.parent)
    log.info("wrote %d rows to %s", len(results), output)
    return results
