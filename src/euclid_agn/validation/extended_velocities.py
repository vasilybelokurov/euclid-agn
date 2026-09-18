"""Redshifts of *resolved* sources, where rvspecfit cannot be used.

rvspecfit assumes the point-source LSF baked into its template library
(R = lambda/32.3 A).  Resolved sources in Euclid's slitless spectra carry a
much wider effective LSF - 20-120 A for the objects around NGC 1527, against
13.7 A for a point source - because the dispersed image is the object's own
light profile.  Our own engine builds a template cube per LSF bucket, so it is
the right tool for them.

Two passes per object:

* **wide**: archetype continuum over 0 < z < ``z_max_wide`` at 300 km/s, which
  separates local objects from the background field;
* **fine**: for objects the wide pass puts below ``z_local``, a joint
  continuum+emission-line fit on a 50 km/s grid, which is what a velocity of a
  few hundred km/s requires (300 km/s steps quantise 1176 km/s to +-150).

Both use the multiplicative continuum polynomial and the dither-scatter
variance, the configuration adopted in session 11.

Usage::

    python -m euclid_agn.validation.extended_velocities --candidates outputs/gc_candidates_ngc1527.parquet \
        --out outputs/extended_velocities_ngc1527.parquet
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from euclid_agn.constants import C_KMS
from euclid_agn.fit.joint_scan import joint_scan
from euclid_agn.fit.screen import ScreenSettings, prepare
from euclid_agn.fit.template_cube import CubeStore, cube_scan, redshift_grid
from euclid_agn.io.sir import open_sir_file
from euclid_agn.models.library import load_xsl_ssp_library
from euclid_agn.spectra.coherence import dither_variance_rescale

log = logging.getLogger(__name__)


#: Template smoothing widths tried per object.  The archive LSF_SIG over-smooths badly (session 12:
#: header 63-69 A where the data want 10-14 A, doubling chi2 and erasing every absorption feature),
#: so it is used only as an upper bound.  The floor is NISP's nominal point-source LSF.
LSF_GRID: tuple[float, ...] = (13.7, 17.0, 22.0, 30.0, 45.0, 65.0, 90.0)


def lsf_choices(header_lsf: float, fit_lsf: bool) -> list[float]:
    if not fit_lsf:
        return [float(header_lsf)]
    return [w for w in LSF_GRID if w <= max(header_lsf, LSF_GRID[0])] or [LSF_GRID[0]]


def run(candidates: pd.DataFrame, z_max_wide: float = 1.0, z_local: float = 0.03, step_wide: float = 300.0,
        step_fine: float = 50.0, lsf_step: float = 5.0, fit_lsf: bool = True) -> pd.DataFrame:
    """Wide then fine redshift for each candidate; one row per object."""
    coarse = load_xsl_ssp_library(log_age_min=8.5, mh_min=-0.5)[::18]
    rich = load_xsl_ssp_library(log_age_min=8.5, mh_min=-0.5)[::6]
    wide_store = fine_store = None
    rows = []
    started = time.time()
    for path, group in candidates.groupby("file"):
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
                if wide_store is None:
                    wide_store = CubeStore({"GALAXY": coarse}, redshift_grid(0.0, z_max_wide, step_wide),
                                           spectrum.wavelength, spectrum.bin_width, lsf_step=lsf_step)
                    fine_store = CubeStore({"GALAXY": rich}, redshift_grid(0.0, z_local, step_fine),
                                           spectrum.wavelength, spectrum.bin_width, lsf_step=lsf_step)
                out = {k: row[k] for k in ("object_id", "ra", "dec", "sep_arcmin", "snr", "lsf_sigma") if k in row}
                out.update({k: row[k] for k in ("host", "host_v", "host_d_mpc") if k in row})
                widths = lsf_choices(spectrum.lsf_sigma, fit_lsf)
                wide, lsf_used = None, np.nan
                for width in widths:
                    trial = cube_scan(spectrum, projected, wide_store.get("GALAXY", width),
                                      poly_degree=0, nonnegative=True, multiplicative_degree=3)
                    if trial is not None and (wide is None or trial.chi2 < wide.chi2):
                        wide, lsf_used = trial, width
                if wide is None:
                    continue
                out.update({"z_wide": wide.z, "dchi2_wide": wide.delta_chi2_runner_up, "chi2_wide": wide.chi2,
                            "n_pixels": wide.n_pixels, "lsf_header": float(spectrum.lsf_sigma), "lsf_fitted": lsf_used,
                            "chi2_reduced": wide.chi2 / max(wide.n_pixels - wide.n_parameters, 1)})
                if wide.z < z_local:
                    fine = joint_scan(spectrum, projected, fine_store.get("GALAXY", lsf_used),
                                      poly_degree=0, nonnegative=True, multiplicative_degree=3)
                    if fine is not None:
                        out.update({"z_fine": fine.z, "v_fine": C_KMS * fine.z, "dchi2_fine": fine.delta_chi2_runner_up,
                                    "dchi2_lines": fine.delta_chi2_lines})
                rows.append(out)
    log.info("%d objects in %.0f s", len(rows), time.time() - started)
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-lsf", type=float, default=20.0)
    parser.add_argument("--min-snr", type=float, default=10.0)
    parser.add_argument("--max-objects", type=int, default=400)
    parser.add_argument("--host", default=None)
    parser.add_argument("--host-v", type=float, default=np.nan)
    parser.add_argument("--header-lsf", action="store_true", help="use the archive LSF_SIG instead of fitting it")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    c = pd.read_parquet(args.candidates)
    c = c[(c.lsf_sigma >= args.min_lsf) & (c.snr > args.min_snr)].nlargest(args.max_objects, "snr")
    if args.host:
        c = c.assign(host=args.host, host_v=args.host_v)
    if "sep_arcmin" not in c:
        c = c.assign(sep_arcmin=np.nan)
    print(f"{len(c)} resolved candidates (LSF >= {args.min_lsf} A, S/N > {args.min_snr})")
    t = run(c, fit_lsf=not args.header_lsf)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    t.to_parquet(args.out, index=False)
    local = t[t.z_wide < 0.03]
    if "lsf_fitted" in t:
        print(f"LSF: header median {t.lsf_header.median():.0f} A -> fitted median {t.lsf_fitted.median():.0f} A; "
              f"median reduced chi2 {t.chi2_reduced.median():.2f}")
    print(f"wide scan: {len(t)} fitted; z < 0.03: {len(local)} ({len(local)/max(len(t),1):.0%}); "
          f"z quantiles {t.z_wide.quantile([.25,.5,.75]).round(3).tolist()}")
    if "v_fine" in t and t.v_fine.notna().any():
        v = t.v_fine.dropna()
        print(f"fine velocities (n={len(v)}): median {v.median():.0f}, |v|<300 {(v.abs()<300).mean():.0%}, "
              f"in 900-1600 {((v>900)&(v<1600)).sum()}")


if __name__ == "__main__":
    main()
