"""A position index of every EDF-N object that has a Q1 spectrum.

Built once from MER, tile by tile, and cached to parquet.  With it, any truth
catalogue - DESI quasars, DESI galaxies, Gaia stars - can be matched to Euclid
spectra locally in seconds, and the matched spectra can then be read straight
from the public S3 bucket without downloading whole files.

Why per tile: a 3-degree cone on the 400-million-row MER catalogue is refused
or times out, while ``WHERE tileid = ...`` returns in ~30 s.  124 tiles cover
EDF-N, each holding ~1,250 objects with spectra, so the whole index is
~155,000 rows and about an hour of queries.  It is written incrementally so an
interrupted run resumes.

Usage::

    python -m euclid_agn.validation.edfn_index --out outputs/edfn_spectra_index.parquet
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from euclid_agn.archive import schema
from euclid_agn.archive.tap import TapService, quote_columns

log = logging.getLogger(__name__)
COLUMNS = ("object_id", "tileid", "ra", "dec", "flux_h_2fwhm_aper", "flux_vis_2fwhm_aper",
           "point_like_prob", "has_spectrum")


def edfn_tiles(tap: TapService, ra: float = 269.73, dec: float = 66.018, radius: float = 3.0) -> list[int]:
    adql = (f"SELECT DISTINCT t.tileid FROM {schema.CAOM_TILE_ASSOCIATION} AS t "
            f"JOIN {schema.CAOM_PLANE} AS p ON p.obsid = t.obsid "
            f"WHERE CONTAINS(p.pt, CIRCLE('ICRS', {ra}, {dec}, {radius})) = 1")
    return sorted(int(v) for v in tap.query(adql)["tileid"])


def build(out: Path, tiles: list[int] | None = None, tap: TapService | None = None) -> pd.DataFrame:
    tap = tap or TapService(schema.IRSA_TAP_SYNC, retries=4, timeout=900)
    done = pd.read_parquet(out) if out.exists() else pd.DataFrame(columns=list(COLUMNS))
    have = set(done.tileid.unique()) if len(done) else set()
    if tiles is None:
        cached = Path("outputs/edfn_tiles.parquet")
        tiles = (sorted(int(v) for v in pd.read_parquet(cached)["tileid"]) if cached.exists()
                 else edfn_tiles(tap))
    frames = [done]
    started = time.time()
    for i, tile in enumerate(t for t in tiles if t not in have):
        try:
            rows = tap.query(f"SELECT {quote_columns(COLUMNS)} FROM {schema.MER_CATALOGUE} "
                             f"WHERE tileid = {tile} AND has_spectrum = 1")
        except Exception as exc:  # noqa: BLE001 - IRSA drops connections under load; keep what we have
            log.warning("tile %s failed: %s", tile, str(exc)[:80])
            continue
        frames.append(rows)
        if i % 5 == 0:
            pd.concat(frames, ignore_index=True).to_parquet(out, index=False)
            log.info("%d tiles, %d rows, %.0f s", i + 1, sum(len(f) for f in frames), time.time() - started)
    index = pd.concat(frames, ignore_index=True).drop_duplicates("object_id")
    index["object_id"] = index.object_id.astype(np.int64)
    index["H_AB"] = 23.9 - 2.5 * np.log10(index.flux_h_2fwhm_aper.where(index.flux_h_2fwhm_aper > 0))
    index.to_parquet(out, index=False)
    return index


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=Path("outputs/edfn_spectra_index.parquet"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    index = build(args.out)
    print(f"{len(index):,} EDF-N objects with a Q1 spectrum across {index.tileid.nunique()} tiles")
    print(f"  H_AB quantiles {index.H_AB.quantile([.1, .5, .9]).round(2).tolist()}; "
          f"point-like (>0.9): {(index.point_like_prob > 0.9).mean():.0%}")


if __name__ == "__main__":
    main()
