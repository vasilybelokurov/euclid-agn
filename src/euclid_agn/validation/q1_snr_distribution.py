"""S/N distribution of every Q1 spectrum, from SPE's ``spe_cont_snr``.

``spe_cont_snr`` (galaxy-candidates table, rank 0) is an *integrated*
continuum S/N: on 685 DESI objects it is 19.3x our median S/N per pixel
(16-84 %: 15.1-21.1; sqrt(531 pixels) = 23), so per-pixel S/N = spe_cont_snr / 19.3.
The histogram is built in unit per-pixel S/N bins by SPE class, in cntr chunks because a
single aggregate over 18.4 M rows times out on IRSA.

Usage::

    python -m euclid_agn.validation.q1_snr_distribution --out outputs/q1_snr_hist.parquet
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from euclid_agn.archive import schema
from euclid_agn.archive.tap import TapService

log = logging.getLogger(__name__)
SPE_SNR_PER_PIXEL_FACTOR = 19.3
CNTR_LO, CNTR_HI = 18_510_600, 36_920_504


def histogram(tap: TapService, lo: int, hi: int) -> pd.DataFrame:
    g, c = schema.SPE_GALAXY_CANDIDATES, schema.SPE_CLASSIFICATION
    q = (f"SELECT FLOOR(g.spe_cont_snr / 19.3) AS lb, c.spe_class AS spe_class, COUNT(*) AS n "
         f"FROM {g} AS g JOIN {c} AS c ON g.object_id = c.object_id "
         f"WHERE g.spe_rank = 0 AND g.spe_cont_snr > 0 AND g.cntr >= {lo} AND g.cntr < {hi} "
         f"GROUP BY FLOOR(g.spe_cont_snr / 19.3), c.spe_class")  # Oracle: no aliases in GROUP BY
    return tap.query(q)


def histogram_nojoin(tap: TapService, lo: int, hi: int) -> pd.DataFrame:
    g = schema.SPE_GALAXY_CANDIDATES
    q = (f"SELECT FLOOR(spe_cont_snr / 19.3) AS lb, COUNT(*) AS n FROM {g} "
         f"WHERE spe_rank = 0 AND spe_cont_snr > 0 AND cntr >= {lo} AND cntr < {hi} GROUP BY FLOOR(spe_cont_snr / 19.3)")
    t = tap.query(q); t["spe_class"] = "all"
    return t


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=Path("outputs/q1_snr_hist.parquet"))
    parser.add_argument("--chunks", type=int, default=20)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    tap = TapService(schema.IRSA_TAP_SYNC)
    edges = np.linspace(CNTR_LO, CNTR_HI + 1, args.chunks + 1).astype(int)
    frames = []
    use_join = True
    for lo, hi in zip(edges[:-1], edges[1:]):
        try:
            frames.append(histogram(tap, int(lo), int(hi)) if use_join else histogram_nojoin(tap, int(lo), int(hi)))
        except Exception as exc:  # noqa: BLE001
            log.warning("join query failed (%s); falling back to no-join histogram", str(exc)[:120])
            use_join = False
            frames.append(histogram_nojoin(tap, int(lo), int(hi)))
        log.info("chunk %d-%d done (%d rows so far)", lo, hi, sum(int(f.n.sum()) for f in frames))
    h = pd.concat(frames, ignore_index=True)
    h["lb"] = h.lb.astype(int)
    h = h.groupby(["lb", "spe_class"], as_index=False).n.sum()
    h["snr_pix_lo"] = h.lb.astype(float)  # linear per-pixel S/N bins of width 1 (Oracle has neither LOG10 nor alias GROUP BY)
    h["snr_total_lo"] = h.snr_pix_lo * SPE_SNR_PER_PIXEL_FACTOR
    h.to_parquet(args.out, index=False)
    total = int(h.n.sum())
    print(f"Q1 spectra with spe_cont_snr > 0: {total:,}")
    byc = h.groupby("spe_class").n.sum(); print("by SPE class:", byc.to_dict())
    for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 5), (5, 10), (10, 20), (20, 50), (50, 1e9)):
        sel = h[(h.snr_pix_lo >= lo) & (h.snr_pix_lo < hi)]
        print(f"  per-pixel S/N {lo}-{hi}: {int(sel.n.sum()):>9,}  ({sel.n.sum()/total:5.1%})")


if __name__ == "__main__":
    main()
