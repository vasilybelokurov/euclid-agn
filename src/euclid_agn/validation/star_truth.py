"""A star reference set from Euclid's own catalogues, for the STAR class.

DESI stars are not in the local truth tables (the matched file holds DESI
*galaxies* only), but the MER catalogue carries a Gaia DR3 match and a
point-like probability, and SPE publishes its own star/galaxy/QSO
classification.  A Gaia-matched, point-like source with a NISP spectrum is a
star with very high probability at these magnitudes; SPE's ``spe_class`` is a
second, independent-pipeline opinion.  Both are recorded so the engine's
class purity can be measured against each.

Usage::

    source ~/Work/venvs/.venv/bin/activate; export PYTHONPATH=src
    python -m euclid_agn.validation.star_truth --out outputs/star_truth.parquet
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from euclid_agn.archive import schema
from euclid_agn.archive.tap import TapService, in_list_clause, quote_columns

log = logging.getLogger(__name__)


def cached_tiles(index_path: Path = Path("outputs/cache_object_index.parquet")) -> pd.DataFrame:
    return pd.read_parquet(index_path)


def query_point_sources(tap: TapService, tiles, min_point_like: float = 0.9, chunk: int = 4) -> pd.DataFrame:
    """MER point-like Gaia-matched sources; 4 tiles per query (20 timed out on IRSA)."""
    cols = ["object_id", "tileid", "point_like_prob", "point_like_flag", "gaia_id", "gaia_match_quality",
            "flux_h_2fwhm_aper", "mag_stargal_sep", "spurious_prob"]
    frames = []
    tiles = [int(t) for t in tiles]
    for i in range(0, len(tiles), chunk):
        where = f"{in_list_clause('tileid', tiles[i:i + chunk])} AND gaia_id IS NOT NULL AND point_like_prob > {min_point_like}"
        frames.append(tap.query(f"SELECT {quote_columns(cols)} FROM {schema.MER_CATALOGUE} WHERE {where}"))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)


def query_spe_classification(tap: TapService, object_ids, chunk: int = 500) -> pd.DataFrame:
    cols = ["object_id", "spe_class", "spe_star_prob", "spe_gal_prob", "spe_qso_prob"]
    ids = [int(x) for x in object_ids]
    frames = []
    for i in range(0, len(ids), chunk):
        frames.append(tap.query(f"SELECT {quote_columns(cols)} FROM {schema.SPE_CLASSIFICATION} WHERE {in_list_clause('object_id', ids[i:i + chunk])}"))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)


def build_star_truth(index_path: Path = Path("outputs/cache_object_index.parquet"), min_point_like: float = 0.9) -> pd.DataFrame:
    tap = TapService(schema.IRSA_TAP_SYNC)
    index = cached_tiles(index_path)
    stars = query_point_sources(tap, sorted(index.tile_id.unique()), min_point_like)
    stars["object_id"] = stars["object_id"].astype(np.int64)
    stars = stars.merge(index[["object_id", "file"]], on="object_id", how="inner")
    log.info("%d Gaia-matched point-like sources with cached spectra", len(stars))
    if len(stars):
        spe = query_spe_classification(tap, stars.object_id)
        spe["object_id"] = spe["object_id"].astype(np.int64)
        stars = stars.merge(spe, on="object_id", how="left")
    stars["H_mag"] = 23.9 - 2.5 * np.log10(stars["flux_h_2fwhm_aper"].where(stars["flux_h_2fwhm_aper"] > 0))
    return stars


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=Path("outputs/star_truth.parquet"))
    parser.add_argument("--min-point-like", type=float, default=0.9)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    stars = build_star_truth(min_point_like=args.min_point_like)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    stars.to_parquet(args.out, index=False)
    print(f"{len(stars)} stars; SPE class counts: {stars['spe_class'].value_counts(dropna=False).to_dict() if 'spe_class' in stars else 'n/a'}")
    print("H quantiles:", stars["H_mag"].quantile([0.1, 0.5, 0.9]).round(2).tolist())


if __name__ == "__main__":
    main()
