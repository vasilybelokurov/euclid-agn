"""How many Q1 redshifts can we add beyond the official SPE release?

Population: every cached SIR spectrum (a 0.4 % sample of Q1's 4,307,177) is
classified by MER photometry into extended (galaxy) and point-like bins of H.
Per-object rates come from the DESI-validated samples (our engine right and
SPE wrong / right, per H bin).  The product, scaled by 4,307,177 / N_cached,
is the Q1 yield.  Stellar velocities and parameters are not in the SPE
release at all, so every star with a usable RV is "new".

Usage::

    python -m euclid_agn.validation.q1_yield
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from euclid_agn.archive import schema
from euclid_agn.archive.tap import TapService, in_list_clause, quote_columns

log = logging.getLogger(__name__)
Q1_SPECTRA = 4_307_177
H_BINS = [10, 15, 16, 17, 18, 19, 20, 30]


def cached_population(index_path=Path("outputs/cache_object_index.parquet"), cache_path=Path("outputs/cached_population_mer.parquet")) -> pd.DataFrame:
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    idx = pd.read_parquet(index_path)
    tap = TapService(schema.IRSA_TAP_SYNC)
    cols = ["object_id", "point_like_prob", "extended_prob", "spurious_prob", "flux_h_2fwhm_aper", "flux_vis_2fwhm_aper", "gaia_id"]
    ids = [int(x) for x in idx.object_id]
    frames = []
    for i in range(0, len(ids), 400):
        frames.append(tap.query(f"SELECT {quote_columns(cols)} FROM {schema.MER_CATALOGUE} WHERE {in_list_clause('object_id', ids[i:i + 400])}"))
        log.info("%d / %d", min(i + 400, len(ids)), len(ids))
    t = pd.concat(frames, ignore_index=True)
    t["object_id"] = t.object_id.astype(np.int64)
    t["H"] = 23.9 - 2.5 * np.log10(t.flux_h_2fwhm_aper.where(t.flux_h_2fwhm_aper > 0))
    t["kind"] = np.where(t.point_like_prob > 0.9, "point", np.where(t.point_like_prob < 0.1, "extended", "ambiguous"))
    t.to_parquet(cache_path, index=False)
    return t


def galaxy_rates() -> pd.DataFrame:
    """Per-H-bin rates from the DESI low-z sample: ours right, SPE right, ours right & SPE wrong."""
    t = pd.read_parquet("outputs/continuum_vs_spe_by_hmag.parquet")
    g = t.groupby(pd.cut(t.H, H_BINS), observed=True).agg(n=("cont_ok", "size"), ours=("cont_ok", "mean"), spe=("spe_ok", "mean"),
                                                            ours_not_spe=("cont_ok", lambda s: (s & ~t.loc[s.index, "spe_ok"]).mean()),
                                                            spe_not_ours=("spe_ok", lambda s: (s & ~t.loc[s.index, "cont_ok"]).mean()))
    return g


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    pop = cached_population()
    scale = Q1_SPECTRA / len(pop)
    print(f"cached spectra with MER match: {len(pop)}; scale to Q1: x{scale:.0f}")
    print("kind fractions:", pop.kind.value_counts(normalize=True).round(3).to_dict())
    counts = pop.groupby([pd.cut(pop.H, H_BINS), "kind"], observed=True).size().unstack(fill_value=0)
    print("\ncached counts by H and kind:\n", counts.to_string())
    rates = galaxy_rates()
    print("\nDESI-validated per-object rates (bright z<0.9 galaxies):\n", rates.round(3).to_string())
    out = []
    for hb, row in rates.iterrows():
        n_ext = counts.loc[hb, "extended"] if hb in counts.index and "extended" in counts else 0
        out.append({"H": str(hb), "Q1_extended": int(n_ext * scale), "ours_right": int(n_ext * scale * row.ours),
                    "spe_right": int(n_ext * scale * row.spe), "ours_right_spe_wrong": int(n_ext * scale * row.ours_not_spe)})
    y = pd.DataFrame(out)
    print("\nQ1 yield estimate for extended sources (scaled):\n", y.to_string(index=False))
    print("totals: ours right", y.ours_right.sum(), "| SPE right", y.spe_right.sum(), "| ours right & SPE wrong", y.ours_right_spe_wrong.sum())
    stars = pop[pop.kind == "point"]
    for hmax, sig in ((17, "~50-80 km/s"), (18, "~100-150 km/s"), (19, "~200-300 km/s")):
        print(f"point sources H<{hmax}: cached {int((stars.H < hmax).sum())} -> Q1 ~{int((stars.H < hmax).sum() * scale):,} stars with RV {sig}")


if __name__ == "__main__":
    main()
