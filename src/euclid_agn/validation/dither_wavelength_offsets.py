"""Per-dither wavelength zero-point offsets, measured on bright stars without templates.

For each dither the continuum-normalised spectrum is shifted (in ln-wavelength,
i.e. velocity) against the continuum-normalised combined spectrum and the
chi-squared minimum located by parabolic refinement.  A star's dithers should
agree to the noise; a systematic spread means the dither wavelength solutions
disagree, which sets a floor on any radial velocity from the combined
spectrum (whose zero point is the average of the dithers').

Usage::

    python -m euclid_agn.validation.dither_wavelength_offsets --max-h 17 --out outputs/dither_offsets.parquet
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

from euclid_agn.constants import C_KMS
from euclid_agn.validation.continuum_experiments import DEFAULT_CACHE, iter_spectra

log = logging.getLogger(__name__)


def normalised(spectrum, window: int = 41):
    ok = spectrum.usable()
    flux = np.where(ok, spectrum.flux, np.nan)
    filled = pd.Series(flux).interpolate(limit_direction="both").to_numpy()
    cont = median_filter(filled, size=window, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        norm = np.where(ok & (cont > 0), flux / cont - 1.0, np.nan)
        err = np.where(ok & (cont > 0), np.sqrt(spectrum.variance) / cont, np.nan)
    return norm, err


def velocity_offset(dither, combined, v_grid=np.arange(-1500.0, 1501.0, 25.0)):
    """Velocity shift of ``dither`` relative to ``combined`` (km/s) and its chi2 curve."""
    d, de = normalised(dither)
    c, _ = normalised(combined)
    lam = combined.wavelength
    loglam = np.log(lam)
    ok_c = np.isfinite(c)
    chi2 = np.full(v_grid.size, np.nan)
    for i, v in enumerate(v_grid):
        shifted = np.interp(loglam + v / C_KMS, loglam[ok_c], c[ok_c], left=np.nan, right=np.nan)
        ok = np.isfinite(shifted) & np.isfinite(d) & np.isfinite(de) & (de > 0)
        if ok.sum() < 100:
            continue
        r = (d[ok] - shifted[ok]) / de[ok]
        chi2[i] = float(r @ r)
    if not np.isfinite(chi2).any():
        return np.nan, np.nan, chi2
    i = int(np.nanargmin(chi2))
    v = v_grid[i]
    if 0 < i < v_grid.size - 1 and np.isfinite(chi2[i - 1: i + 2]).all():
        y0, y1, y2 = chi2[i - 1: i + 2]
        den = y0 - 2 * y1 + y2
        if den > 0:
            v = v_grid[i] + 0.5 * (y0 - y2) / den * (v_grid[1] - v_grid[0])
    # curvature -> formal error (delta chi2 = 1)
    err = np.nan
    if 0 < i < v_grid.size - 1:
        den = chi2[i - 1] - 2 * chi2[i] + chi2[i + 1]
        if np.isfinite(den) and den > 0:
            err = (v_grid[1] - v_grid[0]) / np.sqrt(den)
    return float(v), float(err), chi2


def run(stars: pd.DataFrame, cache: Path = DEFAULT_CACHE) -> pd.DataFrame:
    rows = []
    for row, spectrum, obs in iter_spectra(stars[["object_id"]].assign(desi_z=np.nan), cache, with_dithers=True):
        for d in obs.dithers:
            v, err, _ = velocity_offset(d, spectrum)
            rows.append({"object_id": int(row["object_id"]), "pointing_id": int(d.pointing_id), "detector_id": int(d.detector_id),
                         "v_offset": v, "v_err": err, "usable": float(d.usable().mean()),
                         "snr": float(np.nanmedian(d.flux[d.usable()] / np.sqrt(d.variance[d.usable()])))})
    return pd.DataFrame(rows)


def summarise(t: pd.DataFrame) -> str:
    ok = t[np.isfinite(t.v_offset) & (t.v_err < 200)]
    lines = [f"{len(t)} dither spectra, {len(ok)} with a measurable offset (formal error < 200 km/s)",
             f"offset distribution: median {ok.v_offset.median():+.0f}, robust sigma {1.4826 * np.median(np.abs(ok.v_offset - ok.v_offset.median())):.0f} km/s, "
             f"|offset| quantiles 50/90 %: {ok.v_offset.abs().quantile([.5, .9]).round(0).tolist()}; median formal error {ok.v_err.median():.0f} km/s",
             f"1 NISP pixel = {C_KMS * 13.4 / 15500:.0f} km/s at 1.55 um"]
    per_star = ok.groupby("object_id").v_offset.agg(["std", "count"])
    per_star = per_star[per_star["count"] >= 3]
    lines.append(f"within-star dither scatter (>=3 dithers): median std {per_star['std'].median():.0f} km/s (n={len(per_star)} stars)")
    by_p = ok.groupby("pointing_id").v_offset.agg(["median", "std", "count"]).sort_values("count", ascending=False).head(8)
    lines.append("by pointing (median offset, scatter, n):\n" + by_p.round(0).to_string())
    by_d = ok.groupby("detector_id").v_offset.agg(["median", "std", "count"]).sort_values("count", ascending=False).head(8)
    lines.append("by detector:\n" + by_d.round(0).to_string())
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--max-h", type=float, default=17.0)
    parser.add_argument("--out", type=Path, default=Path("outputs/dither_offsets.parquet"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    stars = pd.read_parquet("outputs/star_truth.parquet")
    stars = stars[(stars.H_mag < args.max_h) & (stars.point_like_prob > 0.99)]
    table = run(stars)
    table.to_parquet(args.out, index=False)
    print(summarise(table))


if __name__ == "__main__":
    main()
