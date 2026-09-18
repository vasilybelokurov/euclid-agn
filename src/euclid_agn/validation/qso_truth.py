"""Known quasars with Euclid Q1 spectra: an independent test of the QSO class.

Truth is DESI DR1 (`desi_dr1.zpix` joined to `desi_dr1.photometry` for
positions, via WSDB): 6,432 quasars with ``zwarn = 0`` inside the EDF-N cone,
z = 0.05-5.82, of which 5,191 lie above z = 0.9.  DESI does not cover the
southern Euclid deep fields, so EDF-N is the whole game.

This is deliberately not Euclid's own SPE QSO list: the point is to judge our
classifier against an outside catalogue, including the objects SPE misses.

Positions for the cached Euclid spectra are read from each SIR file's META
HDUs (the MER catalogue is not needed, and IRSA's TAP service was unusable for
much of this work).

Usage::

    python -m euclid_agn.validation.qso_truth --out outputs/qso_truth_matched.parquet
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from euclid_agn.io.sir import open_sir_file

log = logging.getLogger(__name__)

DEFAULT_CACHE = Path("~/data/euclid/q1/SIR").expanduser()
POSITION_INDEX = Path("outputs/cache_object_positions.parquet")

#: Q1 field cones (name, ra, dec, radius/deg), as in archive.schema.FIELDS.
FIELDS = {"EDF-N": (269.73, 66.018, 6.0), "EDF-S": (61.241, -48.423, 6.0),
          "EDF-F": (52.932, -28.088, 5.0), "LDN1641": (85.4, -8.0, 4.0)}


def desi_spectra(field: str = "EDF-N", spectype: str = "QSO", zwarn_max: int = 0) -> pd.DataFrame:
    """DESI DR1 redshifts of one spectral type inside a Q1 field cone (WSDB)."""
    import sqlutilpy as sqlutil

    ra, dec, radius = FIELDS[field]
    query = f"""
        WITH p AS MATERIALIZED (
            SELECT targetid, ra, dec FROM desi_dr1.photometry
            WHERE q3c_radial_query(ra, dec, {ra}, {dec}, {radius})
        )
        SELECT p.targetid, p.ra, p.dec, z.z AS desi_z, z.zerr AS desi_zerr, z.spectype,
               z.subtype, z.deltachi2, z.zwarn
        FROM p JOIN desi_dr1.zpix z ON z.targetid = p.targetid
        WHERE z.zwarn <= {zwarn_max} AND z.spectype = '{spectype}'
    """
    cols = sqlutil.get(query, asDict=True)
    return pd.DataFrame(cols)


def cache_positions(cache: Path = DEFAULT_CACHE, index_path: Path = POSITION_INDEX) -> pd.DataFrame:
    """object_id, ra, dec and file for every object in the local SIR cache (cached to parquet)."""
    files = sorted(cache.glob("*/EUC_SIR_W-COMBSPEC_*.fits"))
    if index_path.exists():
        done = pd.read_parquet(index_path)
        if set(done["file"].unique()) == {str(f) for f in files}:
            return done
    rows = []
    for path in files:
        with open_sir_file(path) as f:
            for oid in f.object_ids():
                ctx = f.read_source_context(f.group_for_object(oid))
                rows.append({"object_id": int(oid), "ra": float(ctx.ra), "dec": float(ctx.dec), "file": str(path)})
        log.info("%s: %d objects", path.name, len(rows))
    out = pd.DataFrame(rows).drop_duplicates("object_id")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(index_path, index=False)
    return out


def match(truth: pd.DataFrame, positions: pd.DataFrame, radius_arcsec: float = 1.5) -> pd.DataFrame:
    """Nearest Euclid spectrum to each truth object within ``radius_arcsec``."""
    from scipy.spatial import cKDTree

    def unit(ra, dec):
        r, d = np.deg2rad(ra), np.deg2rad(dec)
        return np.column_stack([np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)])

    tree = cKDTree(unit(positions.ra.values, positions.dec.values))
    limit = 2 * np.sin(np.deg2rad(radius_arcsec / 3600) / 2)
    dist, idx = tree.query(unit(truth.ra.values, truth.dec.values), distance_upper_bound=limit)
    hit = np.isfinite(dist)
    out = truth[hit].reset_index(drop=True)
    near = positions.iloc[idx[hit]].reset_index(drop=True)
    out["object_id"] = near.object_id.values
    out["file"] = near["file"].values
    out["euclid_ra"] = near.ra.values
    out["euclid_dec"] = near.dec.values
    out["sep_arcsec"] = np.rad2deg(2 * np.arcsin(dist[hit] / 2)) * 3600
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--field", default="EDF-N")
    parser.add_argument("--spectype", default="QSO")
    parser.add_argument("--radius-arcsec", type=float, default=1.5)
    parser.add_argument("--out", type=Path, default=Path("outputs/qso_truth_matched.parquet"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    truth = desi_spectra(args.field, args.spectype)
    print(f"DESI DR1 {args.spectype} in {args.field} with zwarn=0: {len(truth)}; z quantiles "
          f"{truth.desi_z.quantile([.1,.5,.9]).round(2).tolist()}")
    positions = cache_positions()
    print(f"cached Euclid spectra with positions: {len(positions)}")
    matched = match(truth, positions, args.radius_arcsec)
    matched.to_parquet(args.out, index=False)
    print(f"matched within {args.radius_arcsec}\": {len(matched)}")
    if len(matched):
        print(f"  separation median {matched.sep_arcsec.median():.2f}\"; z range {matched.desi_z.min():.2f}-{matched.desi_z.max():.2f}")
        print("  redshift distribution:",
              {f"{lo}-{hi}": int(((matched.desi_z >= lo) & (matched.desi_z < hi)).sum())
               for lo, hi in ((0, 0.9), (0.9, 1.5), (1.5, 2.5), (2.5, 6))})


if __name__ == "__main__":
    main()
