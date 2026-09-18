"""What our fitter measures for quasars of known redshift.

The sample is `validation.qso_truth`: DESI DR1 quasars (zwarn = 0) matched to
cached Euclid Q1 spectra.  Each is fitted with the three-class engine -
GALAXY archetypes, the Glikman quasar composite and PHOENIX stars, all
non-negative, with the template smoothing width fitted per object because the
archive ``LSF_SIG`` is unreliable (session 12).

Two questions are separable and both matter for the AGN search:

* **classification** - is a real quasar recognised as one, and at what S/N?
  (We already know the class over-claims on faint galaxies; this is the
  opposite direction.)
* **redshift** - given the class, is the redshift right?  The Glikman
  composite covers rest 2762-35201 A, so the blue end leaves the grism above
  z ~ 3.5; failures beyond that are expected, not informative.

Usage::

    python -m euclid_agn.validation.qso_experiment --out outputs/qso_fits.parquet
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from euclid_agn.constants import C_KMS
from euclid_agn.fit.classify import ClassSpec, RedshiftEngine
from euclid_agn.fit.screen import ScreenSettings, prepare
from euclid_agn.io.sir import open_sir_file
from euclid_agn.models.library import load_glikman_composite, load_phoenix_library, load_xsl_ssp_library
from euclid_agn.spectra.artefacts import continuum_trough
from euclid_agn.spectra.coherence import dither_variance_rescale

log = logging.getLogger(__name__)

LSF_GRID: tuple[float, ...] = (13.7, 17.0, 22.0, 30.0, 45.0, 65.0, 90.0)


def build_engine(wavelength, bin_width, z_max_qso: float = 3.5, z_max_galaxy: float = 2.0,
                 step_kms: float = 300.0, lsf_step: float = 5.0) -> RedshiftEngine:
    galaxy = load_xsl_ssp_library(log_age_min=8.5, mh_min=-0.5)[::18]
    stars = [t for t in load_phoenix_library(mh_values=(0.0,), teff_step=5) if t.metadata["logg"] in (2.0, 4.5)]
    specs = [
        ClassSpec("GALAXY", galaxy, 0.0, z_max_galaxy, step_kms=step_kms, poly_degree=0, nonnegative=True),
        ClassSpec("QSO", [load_glikman_composite()], 0.0, z_max_qso, step_kms=step_kms, poly_degree=0,
                  nonnegative=True, extra_sigma_kms=1500.0),
        ClassSpec("STAR", stars, -0.002, 0.002, step_kms=100.0, poly_degree=0, nonnegative=True, use_prior=False),
    ]
    return RedshiftEngine(specs, wavelength, bin_width, lsf_step=lsf_step)


def run(sample: pd.DataFrame, fit_lsf: bool = True) -> pd.DataFrame:
    engine = None
    rows = []
    started = time.time()
    for path, group in sample.groupby("file"):
        with open_sir_file(path) as f:
            for _, row in group.iterrows():
                try:
                    obs = f.read_observation(int(row.object_id), with_dithers=True)
                except KeyError:
                    continue
                spectrum, _ = dither_variance_rescale(obs) if obs.dithers else (obs.combined, None)
                projected = prepare(spectrum, ScreenSettings(n_knots=1, outlier_threshold=5.0))
                if projected is None:
                    continue
                if engine is None:
                    engine = build_engine(spectrum.wavelength, spectrum.bin_width)
                ok = spectrum.usable()
                snr = float(np.nanmedian(spectrum.flux[ok] / np.sqrt(spectrum.variance[ok]))) if ok.any() else np.nan
                widths = [w for w in LSF_GRID if w <= max(spectrum.lsf_sigma, LSF_GRID[0])] if fit_lsf else [spectrum.lsf_sigma]
                best, best_lsf = None, np.nan
                for width in widths:
                    res = engine.run_at_lsf(spectrum, projected, width)
                    if res is not None and (best is None or res.best.chi2 < best.best.chi2):
                        best, best_lsf = res, width
                if best is None:
                    continue
                trough = continuum_trough(spectrum)
                out = {"object_id": int(row.object_id), "desi_z": float(row.desi_z), "desi_zerr": float(row.desi_zerr),
                       "subtype": row.get("subtype"), "desi_deltachi2": float(row.get("deltachi2", np.nan)),
                       "sep_arcsec": float(row.sep_arcsec), "snr": snr, "lsf_header": float(spectrum.lsf_sigma),
                       "lsf_fitted": best_lsf, "trough": trough.flagged}
                out.update(best.as_row())
                out["dz"] = (out["z"] - out["desi_z"]) / (1 + out["desi_z"])
                out["ok"] = abs(out["dz"]) < 0.01
                zq = out.get("z_qso", np.nan)
                out["dz_qso"] = (zq - out["desi_z"]) / (1 + out["desi_z"]) if np.isfinite(zq) else np.nan
                out["ok_qso"] = abs(out["dz_qso"]) < 0.01 if np.isfinite(out["dz_qso"]) else False
                rows.append(out)
    log.info("%d quasars in %.0f s", len(rows), time.time() - started)
    return pd.DataFrame(rows)


def summarise(t: pd.DataFrame) -> str:
    lines = [f"n = {len(t)} DESI quasars with Euclid spectra",
             f"  engine class: {t['class'].value_counts().to_dict()}",
             f"  redshift right (|dz|/(1+z) < 0.01), best class: {t.ok.mean():.0%}; QSO-class redshift: {t.ok_qso.mean():.0%}"]
    for lo, hi in ((0, 0.9), (0.9, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 9)):
        s = t[(t.desi_z >= lo) & (t.desi_z < hi)]
        if len(s):
            lines.append(f"    z {lo}-{hi}: n={len(s):3d}, called QSO {(s['class']=='QSO').mean():.0%}, "
                         f"z right {s.ok.mean():.0%} (QSO-class z right {s.ok_qso.mean():.0%})")
    for lo, hi in ((0, 3), (3, 10), (10, 1e9)):
        s = t[(t.snr >= lo) & (t.snr < hi)]
        if len(s):
            lines.append(f"    S/N {lo}-{hi}: n={len(s):3d}, called QSO {(s['class']=='QSO').mean():.0%}, z right {s.ok.mean():.0%}")
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sample", type=Path, default=Path("outputs/qso_truth_matched.parquet"))
    parser.add_argument("--out", type=Path, default=Path("outputs/qso_fits.parquet"))
    parser.add_argument("--header-lsf", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sample = pd.read_parquet(args.sample)
    print(f"{len(sample)} matched quasars; z quantiles {sample.desi_z.quantile([.1,.5,.9]).round(2).tolist()}")
    t = run(sample, fit_lsf=not args.header_lsf)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    t.to_parquet(args.out, index=False)
    print(summarise(t))


if __name__ == "__main__":
    main()
