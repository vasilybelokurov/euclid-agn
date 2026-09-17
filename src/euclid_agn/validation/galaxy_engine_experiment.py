"""Adaptive GALAXY engine against DESI truth on any sample file.

Usage::

    source ~/Work/venvs/.venv/bin/activate; export PYTHONPATH=src
    python -m euclid_agn.validation.galaxy_engine_experiment --sample outputs/desi_lowz_baseline.parquet --out outputs/galaxy_engine_lowz.parquet
    python -m euclid_agn.validation.galaxy_engine_experiment --sample outputs/desi_comparison_all_gapfix.parquet --z-max 2.0 --out outputs/galaxy_engine_halpha.parquet
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from euclid_agn.fit.galaxy_engine import GalaxyEngine
from euclid_agn.fit.template_cube import CubeStore, redshift_grid
from euclid_agn.models.library import load_xsl_ssp_library
from euclid_agn.validation.continuum_experiments import DEFAULT_CACHE, iter_spectra, load_sample

log = logging.getLogger(__name__)


def run(sample: pd.DataFrame, z_max: float = 2.0, step_kms: float = 300.0, archetype_step: int = 6,
        cache: Path = DEFAULT_CACHE) -> pd.DataFrame:
    lib = load_xsl_ssp_library(log_age_min=8.5, mh_min=-0.5)[::archetype_step]
    engine = None
    rows = []
    started = time.time()
    for row, spectrum, obs in iter_spectra(sample, cache, with_dithers=True):
        if engine is None:
            store = CubeStore({"GALAXY": lib}, redshift_grid(0.0, z_max, step_kms), spectrum.wavelength, spectrum.bin_width)
            engine = GalaxyEngine(store)
        res = engine.run(obs, z_prior=float(row.get("phz_median", np.nan)))
        if res is None:
            continue
        out = {"object_id": int(row["object_id"]), "desi_z": float(row["desi_z"]),
               "snr": float(row.get("median_snr_per_pixel", np.nan)), "phz": float(row.get("phz_median", np.nan))}
        out.update(res.as_row())
        for key in ("gal_z", "gal_z_prior", "gal_z_lines", "gal_z_continuum"):
            out[f"ok_{key[4:]}"] = bool(abs(out[key] - out["desi_z"]) / (1 + out["desi_z"]) < 0.01) if np.isfinite(out[key]) else False
        rows.append(out)
    log.info("%d objects in %.0f s", len(rows), time.time() - started)
    return pd.DataFrame(rows)


def summarise(t: pd.DataFrame) -> str:
    lines = [f"n={len(t)}  chosen: {t.gal_model.value_counts().to_dict()}",
             f"agreement |dz|/(1+z)<0.01 - engine: {t.ok_z.mean():.1%}, engine+prior: {t.ok_z_prior.mean():.1%}, lines-model: {t.ok_z_lines.mean():.1%}, continuum-model: {t.ok_z_continuum.mean():.1%}"]
    for m, g in t.groupby("gal_model"):
        lines.append(f"  chosen={m}: n={len(g)}, engine {g.ok_z.mean():.1%} (+prior {g.ok_z_prior.mean():.1%}), median S/N {g.snr.median():.1f}")
    agree = ~(t.gal_zwarn.astype(int) & 4096).astype(bool)
    lines.append(f"  models agree: {agree.mean():.0%} of objects; when they agree engine ok {t[agree].ok_z.mean():.1%}; when not {t[~agree].ok_z.mean():.1%}")
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--z-max", type=float, default=2.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sample = load_sample(args.sample)
    if args.limit:
        sample = sample.iloc[: args.limit]
    table = run(sample, z_max=args.z_max)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out, index=False)
    print(summarise(table))


if __name__ == "__main__":
    main()
