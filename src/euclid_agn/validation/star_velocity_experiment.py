"""How faint can Euclid NISP measure a stellar line-of-sight velocity?

Empirical answer from the data themselves: rvspecfit is run on the combined
spectrum *and on each of the four dithers separately* for the Gaia-matched
point sources; the dither-to-dither scatter of the velocity is the
single-exposure precision, the combined-spectrum formal error is what the
fitter believes, and their ratio calibrates the formal errors.  Magnitudes:
Euclid H and I_E (VIS) from MER, Gaia G from the ESA archive.

Usage::

    source ~/Work/venvs/.venv/bin/activate; export PYTHONPATH=src
    python -m euclid_agn.validation.star_velocity_experiment --out outputs/star_velocity.parquet
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from euclid_agn.archive import schema
from euclid_agn.archive.tap import TapService, in_list_clause, quote_columns
from euclid_agn.external.rvspecfit_star import fit_star
from euclid_agn.spectra.coherence import dither_variance_rescale
from euclid_agn.validation.continuum_experiments import DEFAULT_CACHE, iter_spectra

log = logging.getLogger(__name__)
GAIA_TAP = "https://gea.esac.esa.int/tap-server/tap/sync"


def photometry(object_ids, gaia_ids) -> pd.DataFrame:
    tap = TapService(schema.IRSA_TAP_SYNC)
    ids = [int(x) for x in object_ids]
    frames = []
    for i in range(0, len(ids), 400):
        frames.append(tap.query(f"SELECT {quote_columns(['object_id', 'flux_vis_2fwhm_aper', 'flux_h_2fwhm_aper'])} FROM {schema.MER_CATALOGUE} WHERE {in_list_clause('object_id', ids[i:i + 400])}"))
    mer = pd.concat(frames, ignore_index=True)
    mer["object_id"] = mer.object_id.astype(np.int64)
    mer["I_E"] = 23.9 - 2.5 * np.log10(mer.flux_vis_2fwhm_aper.where(mer.flux_vis_2fwhm_aper > 0))
    gaia = pd.DataFrame(columns=["gaia_id", "G"])
    try:
        g = TapService(GAIA_TAP)
        gids = [int(x) for x in gaia_ids if np.isfinite(x) and x > 0]
        parts = []
        for i in range(0, len(gids), 400):
            parts.append(g.query(f"SELECT source_id, phot_g_mean_mag, parallax, pmra, pmdec FROM gaiadr3.gaia_source WHERE {in_list_clause('source_id', gids[i:i + 400])}"))
        gaia = pd.concat(parts, ignore_index=True).rename(columns={"source_id": "gaia_id", "phot_g_mean_mag": "G"})
        gaia["gaia_id"] = gaia.gaia_id.astype(np.int64)
    except Exception as exc:  # noqa: BLE001
        log.warning("Gaia TAP failed: %s", exc)
    return mer, gaia


def run(stars: pd.DataFrame, cache: Path = DEFAULT_CACHE) -> pd.DataFrame:
    rows = []
    started = time.time()
    for row, spectrum, obs in iter_spectra(stars[["object_id"]].assign(desi_z=np.nan), cache, with_dithers=True):
        rescaled, _ = dither_variance_rescale(obs) if obs.dithers else (spectrum, None)
        out = {"object_id": int(row["object_id"]), "n_dithers": len(obs.dithers)}
        try:
            comb = fit_star(rescaled)
        except Exception:  # noqa: BLE001
            comb = None
        if comb is not None:
            out.update(comb.as_row("comb_"))
        vels, errs = [], []
        for d in obs.dithers:
            try:
                f = fit_star(d)
            except Exception:  # noqa: BLE001
                f = None
            if f is not None and np.isfinite(f.vel_kms):
                vels.append(f.vel_kms); errs.append(f.vel_err_kms)
        out["n_dither_fits"] = len(vels)
        if len(vels) >= 2:
            v = np.array(vels)
            out["dither_vel_std"] = float(np.std(v, ddof=1))
            out["dither_vel_mean"] = float(np.mean(v))
            out["dither_err_median"] = float(np.median(errs))
        rows.append(out)
    log.info("%d stars in %.0f s", len(rows), time.time() - started)
    return pd.DataFrame(rows)


def summarise(t: pd.DataFrame, mag: str = "G") -> str:
    lines = [f"n={len(t)} with >=2 dither fits: {(t.n_dither_fits >= 2).sum()}"]
    bins = [12, 14, 15, 16, 17, 18, 19, 20, 21]
    for lo, hi in zip(bins[:-1], bins[1:]):
        s = t[(t[mag] >= lo) & (t[mag] < hi) & (t.n_dither_fits >= 2)]
        if len(s) < 3:
            continue
        # single-dither precision = dither std; combined precision ~ std/sqrt(n); formal = fitter's error
        lines.append(f"  {mag} {lo}-{hi}: n={len(s):3d}  single-dither sigma_v (median) {s.dither_vel_std.median():6.0f} km/s"
                     f"  -> combined ~{(s.dither_vel_std / np.sqrt(s.n_dither_fits)).median():5.0f} km/s"
                     f"  | formal error on combined {s.comb_vel_err.median():5.0f} km/s"
                     f"  | |v_comb| median {s.comb_vel.abs().median():4.0f}  | H median {s.H_mag.median():.1f}")
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=Path("outputs/star_velocity.parquet"))
    parser.add_argument("--max-h", type=float, default=19.0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    stars = pd.read_parquet("outputs/star_truth.parquet")
    stars = stars[(stars.H_mag < args.max_h) & (stars.point_like_prob > 0.99)]
    if args.limit:
        stars = stars.head(args.limit)
    table = run(stars).merge(stars[["object_id", "H_mag", "gaia_id", "spe_class"]], on="object_id", how="left")
    mer, gaia = photometry(table.object_id, table.gaia_id)
    table = table.merge(mer[["object_id", "I_E"]], on="object_id", how="left")
    if len(gaia):
        table = table.merge(gaia, on="gaia_id", how="left")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out, index=False)
    for mag in ("G", "I_E", "H_mag"):
        if mag in table and table[mag].notna().sum() > 10:
            print(f"--- by {mag}"); print(summarise(table, mag))


if __name__ == "__main__":
    main()
