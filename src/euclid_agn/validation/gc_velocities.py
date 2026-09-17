"""Velocities of compact sources around nearby galaxies: the GC/UCD test.

Ground truth is the host's systemic velocity (HyperLEDA).  A genuine globular
cluster or ultra-compact dwarf belonging to the host must come out within a few
hundred km/s of it; a foreground Galactic star sits near 0 km/s and a
background galaxy far above.  With NISP's 300 km/s pixels, hosts at
1,000-2,500 km/s (D = 15-35 Mpc) are separated from Galactic stars by 4-8
pixels, so this is a real test of both the redshift engine and the star class
on old single-burst populations - the simplest SED there is.

Each candidate is fitted three ways: rvspecfit (PHOENIX stars, +-1500 km/s),
our 3-class engine (GALAXY/QSO/STAR chi-squared comparison), and the adaptive
GALAXY engine (the redshift we would publish).

Usage::

    source ~/Work/venvs/.venv/bin/activate; export PYTHONPATH=src
    python -m euclid_agn.validation.gc_velocities --min-snr 3 --out outputs/gc_velocities.parquet
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from euclid_agn.constants import C_KMS
from euclid_agn.fit.classify import RedshiftEngine, default_specs
from euclid_agn.fit.galaxy_engine import GalaxyEngine
from euclid_agn.fit.screen import ScreenSettings, prepare
from euclid_agn.fit.template_cube import CubeStore, redshift_grid
from euclid_agn.io.sir import open_sir_file
from euclid_agn.models.library import load_glikman_composite, load_phoenix_library, load_xsl_ssp_library
from euclid_agn.spectra.coherence import dither_variance_rescale

log = logging.getLogger(__name__)


def build_engines(wavelength, bin_width, z_max: float = 0.05, with_qso: bool = False):
    """Engines tuned for the local-universe test: galaxies scanned only to z = z_max.

    ``with_qso`` is off by default: measured on a random control field, the QSO
    class (one composite, free redshift to 3.3 over 1,100 trial values) wins for
    12 of 23 objects at S/N 3-10 while at S/N > 20 every object is correctly a
    star - it is the most flexible hypothesis and the least relevant one for a
    local-universe test.
    """
    galaxy = load_xsl_ssp_library(log_age_min=8.5, mh_min=-0.5)[::6]
    stars = [t for t in load_phoenix_library(mh_values=(0.0, -1.0), teff_step=5) if t.metadata["logg"] in (2.0, 4.5)]
    specs = default_specs(galaxy, [load_glikman_composite()] if with_qso else None, stars, galaxy_nonnegative=True,
                          star_nonnegative=True, z_max_galaxy=z_max, z_max_qso=3.3)
    classifier = RedshiftEngine(specs, wavelength, bin_width)
    store = CubeStore({"GALAXY": galaxy}, redshift_grid(0.0, z_max, 100.0), wavelength, bin_width)
    return classifier, GalaxyEngine(store)


#: rvspecfit configuration with +-3000 km/s (the default +-1500 does not reach NGC 2110 at 2312 km/s)
WIDE_CONFIG = str(Path("~/data/euclid/rvspecfit/config_wide.yaml").expanduser())


def run(candidates: pd.DataFrame, z_max: float = 0.05, with_rvspecfit: bool = True, with_qso: bool = False,
        rvs_config: str = WIDE_CONFIG) -> pd.DataFrame:
    from euclid_agn.external.rvspecfit_star import fit_star

    rows = []
    classifier = galaxy_engine = None
    started = time.time()
    for path, group in candidates.groupby("file"):
        with open_sir_file(path) as f:
            for _, row in group.iterrows():
                obs = f.read_observation(int(row.object_id), with_dithers=True)
                spectrum, _ = dither_variance_rescale(obs) if obs.dithers else (obs.combined, None)
                if classifier is None:
                    classifier, galaxy_engine = build_engines(spectrum.wavelength, spectrum.bin_width, z_max, with_qso)
                projected = prepare(spectrum, ScreenSettings(n_knots=1, outlier_threshold=5.0))
                if projected is None:
                    continue
                out = {k: row[k] for k in ("object_id", "host", "host_v", "host_d_mpc", "host_type", "sep_arcmin", "snr", "lsf_sigma", "ra", "dec")}
                cls = classifier.run(spectrum, projected)
                if cls is not None:
                    out.update(cls.as_row())
                    out["v_class_kms"] = C_KMS * cls.z
                gal = galaxy_engine.run(obs)
                if gal is not None:
                    out.update(gal.as_row())
                    out["v_galaxy_kms"] = C_KMS * gal.z
                if with_rvspecfit:
                    try:
                        sf = fit_star(spectrum, config_path=rvs_config)
                    except Exception as exc:  # noqa: BLE001
                        log.warning("rvspecfit failed on %s: %s", row.object_id, str(exc)[:60])
                        sf = None
                    if sf is not None:
                        out.update(sf.as_row())
                rows.append(out)
    log.info("%d objects in %.0f s", len(rows), time.time() - started)
    return pd.DataFrame(rows)


def summarise(t: pd.DataFrame, tolerance_kms: float = 500.0) -> str:
    lines = [f"n = {len(t)} compact sources around {t.host.nunique()} hosts"]
    for col, label in (("v_class_kms", "3-class engine z"), ("v_galaxy_kms", "GALAXY engine z"), ("rvs_vel", "rvspecfit v")):
        if col not in t:
            continue
        dv = t[col] - t.host_v
        near = (dv.abs() < tolerance_kms).sum()
        gal_like = (t[col].abs() > 500).sum()
        lines.append(f"  {label:18s}: within {tolerance_kms:.0f} km/s of the host: {near}/{len(t)}; |v| > 500 km/s (not a Galactic star): {gal_like}")
    if "class" in t:
        lines.append(f"  engine classes: {t['class'].value_counts().to_dict()}")
    hi = t[t.snr > 5]
    if len(hi):
        lines.append(f"  S/N > 5 subset (n={len(hi)}): median |v_galaxy - host| = {np.nanmedian((hi.v_galaxy_kms - hi.host_v).abs()):.0f} km/s"
                     if "v_galaxy_kms" in hi else "")
    return "\n".join(x for x in lines if x)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--candidates", type=Path, default=Path("outputs/gc_candidates.parquet"))
    parser.add_argument("--min-snr", type=float, default=3.0)
    parser.add_argument("--max-objects", type=int, default=400)
    parser.add_argument("--z-max", type=float, default=0.05)
    parser.add_argument("--with-qso", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("outputs/gc_velocities.parquet"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    c = pd.read_parquet(args.candidates)
    c = c[c.snr > args.min_snr].nlargest(args.max_objects, "snr")
    print(f"{len(c)} candidates with S/N > {args.min_snr}; S/N quantiles {c.snr.quantile([.5, .9]).round(1).tolist()}")
    t = run(c, z_max=args.z_max, with_qso=args.with_qso)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    t.to_parquet(args.out, index=False)
    print(summarise(t))


if __name__ == "__main__":
    main()
