"""Class purity of the GALAXY/QSO/STAR engine on labelled sets.

Sets: Gaia-matched bright point sources (`outputs/star_truth.parquet`, H < 17.5,
point_like_prob > 0.99, labelled by SPE's class - itself uncertain), and the
DESI-confirmed galaxies of the low-z and H-alpha samples (label GALAXY).  The
engine's class and its class margin are compared with each label, so both the
engine and SPE can be judged where they disagree.

Usage::

    source ~/Work/venvs/.venv/bin/activate; export PYTHONPATH=src
    python -m euclid_agn.validation.class_experiment --out outputs/class_experiment.parquet
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from euclid_agn.fit.classify import RedshiftEngine, default_specs
from euclid_agn.fit.screen import ScreenSettings, prepare
from euclid_agn.models.library import (
    build_pca_basis,
    load_glikman_composite,
    load_phoenix_library,
    load_xsl_dr3_library,
    load_xsl_ssp_library,
)
from euclid_agn.spectra.coherence import dither_variance_rescale
from euclid_agn.validation.continuum_experiments import DEFAULT_CACHE, iter_spectra, load_sample, pca_templates

log = logging.getLogger(__name__)


def labelled_sets(max_stars: int = 800, max_lowz: int = 250, max_halpha: int = 250) -> pd.DataFrame:
    stars = pd.read_parquet("outputs/star_truth.parquet")
    stars = stars[(stars.H_mag < 17.5) & (stars.point_like_prob > 0.99)].head(max_stars)
    stars = stars.assign(set="point_sources", label=stars.spe_class.fillna("none").str.upper(), desi_z=np.nan)
    lowz = load_sample(Path("outputs/desi_lowz_baseline.parquet")).head(max_lowz).assign(set="desi_lowz", label="GALAXY")
    halpha = load_sample(Path("outputs/desi_comparison_all_gapfix.parquet")).head(max_halpha).assign(set="desi_halpha", label="GALAXY")
    cols = ["object_id", "set", "label", "desi_z"]
    frames = [stars[cols + ["H_mag", "spe_class"]], lowz[cols + [c for c in ("phz_median",) if c in lowz]], halpha[cols + [c for c in ("phz_median",) if c in halpha]]]
    return pd.concat(frames, ignore_index=True)


def build_engine(wavelength, bin_width, star_source: str = "xsl") -> RedshiftEngine:
    galaxy = load_xsl_ssp_library(log_age_min=8.5, mh_min=-0.5)[::6]
    qso = [load_glikman_composite()]
    if star_source == "xsl":
        stars = pca_templates(build_pca_basis(load_xsl_dr3_library(), n_components=8, wmin=6000, wmax=19500))
    elif star_source == "phoenix":
        # PCA of the gap-free PHOENIX grid (Teff 3000-12000, all log g, Z = 0 and -1): the STAR class
        # needs shape coverage, not stellar parameters, at R ~ 450
        lib = load_phoenix_library(mh_values=(0.0, -1.0), teff_step=2)
        stars = pca_templates(build_pca_basis(lib, n_components=8, wmin=6000, wmax=19500))
    else:
        raise ValueError(star_source)
    specs = default_specs(galaxy, qso, stars, galaxy_nonnegative=True, z_max_galaxy=2.0, z_max_qso=3.3)
    return RedshiftEngine(specs, wavelength, bin_width)


def run(sample: pd.DataFrame, cache: Path = DEFAULT_CACHE, star_source: str = "xsl") -> pd.DataFrame:
    engine = None
    rows = []
    started = time.time()
    for row, spectrum, obs in iter_spectra(sample, cache, with_dithers=True):
        if engine is None:
            engine = build_engine(spectrum.wavelength, spectrum.bin_width, star_source)
        rescaled, _ = dither_variance_rescale(obs) if obs.dithers else (spectrum, None)
        projected = prepare(rescaled, ScreenSettings(n_knots=1, outlier_threshold=5.0))
        if projected is None:
            continue
        res = engine.run(rescaled, projected, z_prior=float(row.get("phz_median", np.nan)))
        if res is None:
            continue
        out = {"object_id": int(row["object_id"]), "set": row["set"], "label": row["label"],
               "desi_z": float(row.get("desi_z", np.nan)), "H_mag": float(row.get("H_mag", np.nan)),
               "snr": float(np.nanmedian(rescaled.flux[np.isin(rescaled.wavelength, projected.wavelength)] * projected.weight))}
        out.update(res.as_row())
        rows.append(out)
    log.info("%d objects in %.0f s", len(rows), time.time() - started)
    return pd.DataFrame(rows)


def summarise(t: pd.DataFrame) -> str:
    lines = []
    for s, g in t.groupby("set"):
        lines.append(f"{s}: n={len(g)}  engine class: {g['class'].value_counts().to_dict()}")
        if s == "point_sources":
            ct = pd.crosstab(g.label, g["class"]); lines.append(ct.to_string())
        else:
            gal = g[g["class"] == "GALAXY"]
            ok = (np.abs(gal.z - gal.desi_z) / (1 + gal.desi_z) < 0.01).mean() if len(gal) else np.nan
            lines.append(f"   GALAXY fraction {(g['class']=='GALAXY').mean():.0%}; of those, z within 0.01: {ok:.0%}")
        strong = g[g.delta_chi2_class > 25]
        lines.append(f"   with class margin > 25: {len(strong)/len(g):.0%} of set; classes {strong['class'].value_counts().to_dict()}")
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=Path("outputs/class_experiment.parquet"))
    parser.add_argument("--star-source", default="xsl")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    table = run(labelled_sets(), star_source=args.star_source)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out, index=False)
    print(summarise(table))


if __name__ == "__main__":
    main()
